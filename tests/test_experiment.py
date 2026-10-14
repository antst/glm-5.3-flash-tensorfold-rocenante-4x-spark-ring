"""CPU-only namespace, admission and qualification gates; never create GPU jobs."""
import copy
import json
from pathlib import Path
from config_fixtures import prepared_example
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import environment,ordered_hcas,validate_profile
from cluster import Operator
from experiment import (ExperimentOperator,prepare_config,rank_settings,RECEIPT_FLAGS,
                        VENDOR_MANIFEST_SHA256,vendor_identity,verify_boot_qualification,experiment_identity,NCCL_COMMIT,NCCL_PATCH_SHA256)


class ExperimentTests(unittest.TestCase):
    def config(self,mode='rocenante'):
        cfg=prepared_example(mode);cfg['shared_root']=str(ROOT)
        cfg['state_root']=str(ROOT/'state');cfg['library']='lib/nccl/libnccl.so.2'
        return cfg
    def test_offline_plan_command_never_calls_ssh_or_writes(self):
        cfg=self.config();op=ExperimentOperator(cfg)
        with patch('subprocess.run',side_effect=AssertionError('SSH forbidden')),patch.object(Path,'write_text',side_effect=AssertionError('Writes forbidden')):
            commands=[op.command(r,Path('/MISSING/libnccl.so.2'),'QUALIFY_FIRST') for r in range(4)]
        self.assertEqual(len(commands),4)
        for cache,args in commands:
            self.assertIn('/opt/spark-ring/scripts/runtime.py',args)
            self.assertTrue(str(cache).endswith('/rocenante'))
            self.assertTrue(any('dst=/opt/spark-ring/mesh-health.json,readonly' in v for v in args))
            self.assertIn('SIRCL_ENABLED=0',args)
    def test_shared_api_port_retains_bootstrap_cache_and_owner_isolation(self):
        cfg=self.config()
        self.assertEqual(cfg['api_port'],8888)
        argv=ExperimentOperator(cfg).command(0,Path('/unused/nccl'),'sha')[1]
        self.assertEqual(argv[argv.index('--port')+1],'8888')
        for field,value in (('bootstrap_port',29551),('cache_root','/var/tmp/spark-ring-original-cache'),('container_name','spark-ring-glm53-tp4')):
            bad=copy.deepcopy(cfg);bad[field]=value
            with self.subTest(field=field),patch('experiment.load_config',return_value=bad),self.assertRaises(ValueError):
                prepare_config('/unused')
    def test_ab_modes_have_identical_model_transport_settings(self):
        a=self.config('nccl');b=self.config('rocenante')
        for rank in range(4):
            self.assertEqual(environment(a,rank,'digest'),environment(b,rank,'digest'))
            self.assertEqual(rank_settings(a,rank),rank_settings(b,rank))
            self.assertEqual(ordered_hcas(a['nodes'][rank]),['rocep1s0f0','rocep1s0f1','roceP2p1s0f0','roceP2p1s0f1'])
    def test_current_prepared_vendor_has_genuine_framework_and_exact_hashes(self):
        identity=vendor_identity(self.config())
        self.assertEqual(identity['manifest_sha256'],VENDOR_MANIFEST_SHA256)
        self.assertGreaterEqual(identity['files'],50)
    def test_profile_tuning_is_bounded_and_changes_bound_environment(self):
        cfg=self.config()
        for channels in (4,8,16):
            cfg['profile']=validate_profile({**cfg['profile'],'nccl_channels':channels})
            e=environment(cfg,0,'digest')
            self.assertEqual(e['NCCL_MIN_NCHANNELS'],str(channels));self.assertEqual(e['NCCL_MAX_NCHANNELS'],str(channels))
        for channels in (0,3,32,True):
            with self.subTest(channels=channels),self.assertRaises(ValueError):validate_profile({'nccl_channels':channels})
    def test_retained_state_budget_is_optional_and_does_not_change_transport(self):
        from common import transport_environment
        cfg=self.config();cfg["profile"].pop("kept_bytes_gib",None);original=environment(cfg,0,"digest")
        self.assertNotIn("TF_GLM_KEPT_BYTES_GIB",original)
        cfg["profile"]=validate_profile({**cfg["profile"],"cache_entries":64,"kept_bytes_gib":4.0})
        changed=environment(cfg,0,"digest")
        self.assertEqual(changed["TF_GLM_CACHE_ENTRIES"],"64")
        self.assertEqual(changed["TF_GLM_KEPT_BYTES_GIB"],"4.0")
        self.assertEqual(transport_environment(original),transport_environment(changed))

    def test_retained_state_budget_rejects_invalid_values(self):
        for value in (-1,True,False,"4",float("inf"),float("-inf"),float("nan"),None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_profile({"kept_bytes_gib":value})
        for value in (0,4.0,8):
            self.assertEqual(validate_profile({"kept_bytes_gib":value})["kept_bytes_gib"],value)

    def test_reasoning_retention_is_optional_and_preserves_transport_on_all_ranks(self):
        from common import transport_environment
        cfg=self.config()
        for field in ("keep_reasoning_entries","keep_reasoning_mb"):
            cfg["profile"].pop(field,None)
        original=[environment(cfg,rank,"digest") for rank in range(4)]
        for env in original:
            self.assertNotIn("TF_GLM_KEEP_REASONING",env)
            self.assertNotIn("TF_GLM_KEEP_REASONING_MB",env)
        cfg["profile"]=validate_profile({**cfg["profile"],"keep_reasoning_entries":8192,"keep_reasoning_mb":128})
        for rank,before in enumerate(original):
            after=environment(cfg,rank,"digest")
            self.assertEqual(after["TF_GLM_KEEP_REASONING"],"8192")
            self.assertEqual(after["TF_GLM_KEEP_REASONING_MB"],"128")
            self.assertEqual(transport_environment(before),transport_environment(after))
            self.assertEqual({k:v for k,v in after.items() if k not in
                ("TF_GLM_KEEP_REASONING","TF_GLM_KEEP_REASONING_MB")},before)

    def test_reasoning_retention_rejects_ambiguous_or_negative_budgets(self):
        for field in ("keep_reasoning_entries","keep_reasoning_mb"):
            for value in (-1,True,False,"128",128.0,float("inf"),float("nan"),None):
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    validate_profile({field:value})
            for value in (0,128,8192):
                self.assertEqual(validate_profile({field:value})[field],value)

    def test_native_geometry_defaults_and_choices_are_exact_and_bound_to_rank_settings(self):
        cfg=self.config();cfg['experiment'].pop('native_blocks')
        with patch('experiment.load_config',return_value=cfg):normalized=prepare_config('/unused')
        self.assertEqual(normalized['experiment']['native_blocks'],8)
        commands=[]
        for blocks in (4,8,16):
            candidate=copy.deepcopy(normalized);candidate['experiment']['native_blocks']=blocks
            with patch('experiment.load_config',return_value=candidate):checked=prepare_config('/unused')
            for rank in range(4):
                settings=rank_settings(checked,rank)
                self.assertEqual(settings['native_blocks'],blocks);self.assertEqual(settings['native_threads'],512)
            with patch('subprocess.run',side_effect=AssertionError('SSH forbidden')):
                commands.append(ExperimentOperator(checked).command(0,Path('/unused/nccl'),'sha')[1])
        self.assertEqual(len({json.dumps(command) for command in commands}),3)
    def test_native_block_setting_is_in_qualification_identity(self):
        cfg=self.config();identities=[]
        receipt={'source_commit':NCCL_COMMIT,'patch_sha256':NCCL_PATCH_SHA256,'nccl_get_version':23203,'native_cpu_test_passed':True}
        with patch('experiment.sha',return_value='sha'),patch.object(Path,'read_text',return_value=json.dumps(receipt)),\
             patch('experiment.make_plan',return_value={'complete':True,'plan_id':'plan','upstream_planner_sha256':'planner'}),\
             patch('experiment.marker_identity',return_value=('/unused','marker')),patch('experiment.vendor_identity',return_value={'exact':'vendor'}),\
             patch('experiment.source_hashes',return_value={'exact':'source'}),patch('subprocess.run',side_effect=AssertionError('SSH forbidden')):
            for blocks in (4,8,16):
                cfg['experiment']['native_blocks']=blocks;identity=experiment_identity(cfg,'sha')
                self.assertEqual(identity['settings']['native_blocks'],blocks);identities.append(json.dumps(identity,sort_keys=True))
        self.assertEqual(len(set(identities)),3)
    def test_native_geometry_rejects_unbounded_or_noninteger_values(self):
        for key,values in (('native_blocks',(0,3,32,True,'16',16.0)),('native_threads',(511,1024,True,'512',512.0))):
            for value in values:
                cfg=self.config();cfg['experiment'][key]=value
                with self.subTest(key=key,value=value),patch('experiment.load_config',return_value=cfg),self.assertRaises(ValueError):
                    prepare_config('/unused')
    def test_custom_receipt_rejects_missing_proofs_identity_oracle_and_mode_mismatches(self):
        op=ExperimentOperator(self.config());identity={'fresh':'exact'}
        good={**{flag:True for flag in RECEIPT_FLAGS},'identity':identity,'baseline_receipt_sha256':'baseline-hash',
              'modes_qualified':['nccl','rocenante'],'network_plan_id':'plan'}
        identity['network_plan_id']='plan'
        mutations=[]
        for flag in RECEIPT_FLAGS:
            bad=copy.deepcopy(good);bad[flag]=False;mutations.append((flag,bad))
        for field,value in (('identity',{'fresh':'old'}),('baseline_receipt_sha256','old-baseline'),
                            ('modes_qualified',['nccl']),('network_plan_id','old-topology')):
            bad=copy.deepcopy(good);bad[field]=value;mutations.append((field,bad))
        with patch.object(Operator,'qualification',return_value={'qualified':True}),patch('experiment.experiment_identity',return_value=identity),patch('experiment.sha',return_value='baseline-hash'):
            with patch.object(Path,'read_text',return_value=json.dumps(good)):
                self.assertTrue(op.qualification(Path('/baseline'),'sha')['prepared']['qualified'])
            for label,bad in mutations:
                with self.subTest(label=label),patch.object(Path,'read_text',return_value=json.dumps(bad)),self.assertRaises(RuntimeError):
                    op.qualification(Path('/baseline'),'sha')
    def test_runtime_guard_failure_reports_fault_then_exits_without_fallback(self):
        import runtime
        class Stop:
            def wait(self,seconds):return False
        class Controller:
            def check_health(self):raise AssertionError('native fault')
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return b''
        calls=[]
        def urlopen(request,timeout):
            calls.append(request)
            if request.get_method()=='POST':return Response()
            raise RuntimeError('marker/guardian disappeared')
        health={'address':'192.0.2.11','port':29661,'token':'test-only','plan_id':'plan','rank':0}
        with patch('runtime.urllib.request.urlopen',side_effect=urlopen),patch('runtime.os._exit',side_effect=SystemExit(70)),patch('builtins.print'):
            with self.assertRaises(SystemExit) as result:runtime.health_watch(Stop(),Controller(),health)
        self.assertEqual(result.exception.code,70);self.assertEqual([r.get_method() for r in calls],['GET','POST'])
    def test_reboot_rejects_previously_qualified_baseline_or_prepared_receipt(self):
        boots={'spark'+str(rank):'00000000-0000-0000-0000-'+str(rank).zfill(12) for rank in range(1,5)}
        guard={'hosts':{name:{} for name in boots},'network_boot_ids':boots}
        good={'baseline':{'qualified':True,'network_boot_ids':dict(boots)},
              'prepared':{'qualified':True,'network_boot_ids':dict(boots)}}
        with patch('subprocess.run',side_effect=AssertionError('No host operations in admission test')):
            verify_boot_qualification(good,guard)
            for mode in ('baseline','prepared'):
                stale=copy.deepcopy(good);stale[mode]['network_boot_ids']['spark3']='00000000-0000-0000-0000-000000000099'
                with self.subTest(mode=mode),self.assertRaisesRegex(RuntimeError,'previous host boot'):
                    verify_boot_qualification(stale,guard)
            missing=copy.deepcopy(good);del missing['prepared']['network_boot_ids']
            with self.assertRaisesRegex(RuntimeError,'previous host boot'):verify_boot_qualification(missing,guard)


if __name__=='__main__':unittest.main()
