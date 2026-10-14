import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from config_fixtures import prepared_example
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import engine_binding as e
from common import load_config,environment
from experiment import ExperimentOperator,prepare_config
class EngineBindingTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
  self.r=Path(self.t.name);self.engine=self.r/'engine';(self.engine/'tensorfold/cuda').mkdir(parents=True)
  (self.engine/'tensorfold/__init__.py').write_text('')
  (self.engine/'tensorfold/cuda/comm.py').write_text('original')
  self.files={str(p.relative_to(self.engine/'tensorfold')):e.sha(p) for p in (self.engine/'tensorfold').rglob('*') if p.is_file()}
  self.m=self.r/'manifest.json';self.m.write_text(json.dumps({'schema':'glm-prefill-engine-source/v1','source_id':'test','files':self.files}))
  self.cfg={'engine_overlay_root':str(self.engine),'engine_manifest':str(self.m),'engine_manifest_sha256':e.sha(self.m),'engine_flags':{'TF_GLM_HC_EXCHANGE':'p2p','TF_GLM_PROMPT_PARTIALS':'fp32'}}
 def test_complete_overlay_and_actual_import_source(self):
  self.assertEqual(e.identity(self.cfg,verify=True)['files'],self.files)
  env=e.environment(self.cfg);env['SPARK_RING_ENGINE_SOURCE_ROOT']=str(self.engine/'tensorfold');env['SPARK_RING_ENGINE_MANIFEST']=str(self.m)
  with patch.dict(os.environ,env):
   self.assertEqual(e.runtime_verify(self.engine/'tensorfold'),self.files)
   with self.assertRaises(RuntimeError):e.runtime_verify(self.r)
 def test_added_deleted_changed_source_refuses(self):
  p=self.engine/'tensorfold/cuda/comm.py';p.write_text('changed')
  with self.assertRaises(RuntimeError):e.identity(self.cfg,verify=True)
  p.write_text('original');other=self.engine/'tensorfold/extra.py';other.write_text('extra')
  with self.assertRaises(RuntimeError):e.identity(self.cfg,verify=True)
  other.unlink();p.unlink()
  with self.assertRaises(RuntimeError):e.identity(self.cfg,verify=True)
 def test_manifest_change_and_symlink_refuse(self):
  self.m.write_text(self.m.read_text()+' ')
  with self.assertRaises(RuntimeError):e.identity(self.cfg,verify=True)
  self.cfg['engine_manifest_sha256']=e.sha(self.m)
  p=self.engine/'tensorfold/link';p.symlink_to(self.r)
  with self.assertRaises(RuntimeError):e.identity(self.cfg,verify=True)
 def test_flags_fail_closed(self):
  for values in ({'TF_GLM_PROMPT_PARTIALS':'bf16'},{'TF_GLM_MSA':'1'},{'NCCL_ALGO':'Tree'},{'TF_GLM_PREFILL_ROWS':4096},[],{'TF_GLM_HC_EXCHANGE':'other'}):
   with self.subTest(values=values),self.assertRaises(ValueError):e.flags({'engine_flags':values})
 def test_unsafe_paths_and_ambiguous_digest_refuse(self):
  for name in ('../x','/x','a/../../x','a\\x','c:x'):
   v={'schema':'glm-prefill-engine-source/v1','source_id':'x','files':{name:'a'*64}};self.m.write_text(json.dumps(v))
   with self.subTest(name=name),self.assertRaises(ValueError):e.manifest(self.m,e.sha(self.m))
 def test_command_uses_declared_overlay_and_runtime_without_host_calls(self):
  cfg=prepared_example(overlay=self.cfg);cfg['runtime_source_root']=str(self.r/'runtime');cfg['shared_root']='/shared/spark-ring/tensorfold-rocenante-recipe'
  # Inject explicit source hashes to keep this test offline and independent of full qualification fixtures.
  with patch('experiment.source_hashes',return_value={'runtime.py':'a'*64}):
   cache,argv=ExperimentOperator(cfg).command(0,Path('/p/libnccl.so.2'),'b'*64)
  self.assertIn('type=bind,src='+str(self.engine)+',dst=/opt/glm-prefill/engine,readonly',argv)
  self.assertIn('type=bind,src='+str(self.r/'runtime')+',dst=/opt/spark-ring/scripts,readonly',argv)
  self.assertIn('TF_GLM_HC_EXCHANGE=p2p',argv)
  self.assertIn('PYTHONPATH=/opt/glm-prefill/engine',argv)
if __name__=='__main__':unittest.main()
