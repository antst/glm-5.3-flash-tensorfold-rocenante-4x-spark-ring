"""CPU-only machine-health snapshot and source identity contracts."""
import copy
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

runtime=load('snapshot_runtime',ROOT/'scripts/runtime.py')
fixtures=load('snapshot_fixtures',ROOT/'tests/test_prefill_adapter.py')
validator=fixtures.validator


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.proc=self.root/'proc';self.pid=os.getpid()
        (self.proc/str(self.pid)).mkdir(parents=True)
        (self.proc/'sys/kernel/random').mkdir(parents=True)
        self.boot=self.proc/'sys/kernel/random/boot_id';self.boot.write_text('boot-a\n')
        self.stat=self.proc/str(self.pid)/'stat'
        self.stat.write_text(str(self.pid)+' (name with ) parens) '+' '.join(['S']+['0']*18+['12345']+['0']*4))
        self.path=self.root/'health.json'
        self.files={f'family/{i}.py':hashlib.sha256(str(i).encode()).hexdigest() for i in range(439)}
        self.report={'tensorfold_source_sha256':self.files,'failure':None,'initialized':True}

    def write(self):
        with patch.dict(os.environ,{'SPARK_RING_ENGINE_MANIFEST_SHA256':'c'*64}):
            return runtime.write_controller_snapshot(self.report,self.path,self.proc)

    def read(self):
        return runtime.read_controller_snapshot(self.path,self.proc,self.pid)

    def test_complete_atomic_compact_identity_and_original_untouched(self):
        first=self.write();self.assertEqual(self.read(),first)
        self.assertEqual(self.path.stat().st_mode&0o777,0o600)
        self.assertEqual(first['service_start_ticks'],12345)
        self.assertEqual(first['host_boot_id'],'boot-a')
        compact=first['report']
        self.assertNotIn('tensorfold_source_sha256',compact)
        self.assertEqual(compact['tensorfold_source_file_count'],439)
        self.assertEqual(compact['tensorfold_source_digest_sha256'],runtime.source_map_digest(self.files))
        self.assertEqual(compact['engine_manifest_sha256'],'c'*64)
        self.assertIs(self.report['tensorfold_source_sha256'],self.files)
        self.assertLess(self.path.stat().st_size,1024)
        old=self.path.read_bytes();real_replace=os.replace;observed=[]
        def replace(src,dst):
            observed.append(self.path.read_bytes())
            self.assertEqual(Path(src).stat().st_mode&0o777,0o600)
            self.assertEqual(json.loads(Path(src).read_text())['report']['failure'],'new')
            real_replace(src,dst)
        self.report['failure']='new'
        with patch.object(runtime.os,'replace',side_effect=replace):self.write()
        self.assertEqual(observed,[old]);self.assertEqual(self.read()['report']['failure'],'new')

    def test_failed_replace_preserves_old_complete_file_and_cleans_temp(self):
        self.write();old=self.path.read_bytes()
        with patch.object(runtime.os,'replace',side_effect=OSError('injected replace failure')):
            with self.assertRaisesRegex(OSError,'injected'):self.write()
        self.assertEqual(self.path.read_bytes(),old)
        self.assertEqual(list(self.root.glob('health.json.*.tmp')),[])

    def test_boot_pid_start_and_mode_mismatch_refused(self):
        for field,value in (('host_boot_id','other'),('service_pid',self.pid+1),
                            ('service_start_ticks',12346),('unix_seconds',float('nan'))):
            with self.subTest(field=field):
                row=self.write();row[field]=value;self.path.write_text(json.dumps(row))
                with self.assertRaises(RuntimeError):self.read()
        self.write();os.chmod(self.path,0o644)
        with self.assertRaisesRegex(RuntimeError,'mode'):self.read()
        self.write();self.stat.unlink()
        with self.assertRaises(FileNotFoundError):self.read()

    def test_zombie_and_embedded_loader_syntax(self):
        self.write();self.stat.write_text(self.stat.read_text().replace(') S ',') Z '))
        with self.assertRaisesRegex(RuntimeError,'not live'):self.read()
        tree=ast.parse(inspect.getsource(validator.host_observation))
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=='inside' for t in n.targets))
        source=assignment.value.func.value.value.replace('PAYLOAD','{}')
        compile(source,'container-snapshot-loader','exec')
        self.assertLess(source.index("Candidate source changed:"),source.index('from runtime import'))
        self.assertIn('read_controller_snapshot()',source)

    def test_source_map_digest_canonical_and_byte_sensitive(self):
        self.assertEqual(runtime.source_map_digest({'b':'2','a':'1'}),runtime.source_map_digest({'a':'1','b':'2'}))
        other=dict(self.files);other['family/0.py']='0'*64
        self.assertNotEqual(runtime.source_map_digest(other),runtime.source_map_digest(self.files))

    def test_source_corruption_count_and_manifest_header_rejected(self):
        cfg,identity,report=fixtures.OverlayTests().controller_fixture()
        validator.check_controller(report,cfg,0,identity)
        for key,value in (('tensorfold_source_digest_sha256','0'*64),
                          ('tensorfold_source_file_count',439),('tensorfold_source_file_count',True),
                          ('engine_manifest_sha256','0'*64)):
            broken=dict(report);broken[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'identity mismatch'):
                validator.check_controller(broken,cfg,0,identity)
        identity=copy.deepcopy(identity);identity['engine_source']['files']['new/front.py']='0'*64
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
            validator.check_controller(report,cfg,0,identity)


if __name__=='__main__':unittest.main()
