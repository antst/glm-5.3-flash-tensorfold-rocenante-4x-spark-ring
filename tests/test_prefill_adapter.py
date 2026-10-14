"""CPU-only overlay and eager-P2P ordering contracts; no GPU qualification."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from config_fixtures import prepared_example
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

helpers = load('prefill_adapter_helpers', ROOT / 'tests/test_rocenante_adapter.py')
adapter = helpers.adapter
validator = load('prefill_model_validator', ROOT / 'scripts/validate-model.py')


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.contract = helpers.ContractTests()
        self.contract.setUp()
        self.addCleanup(self.contract.tearDown)
        self.ctl, self.comm = self.contract.controller, self.contract.comm

    def test_native_exchange_graph_transfers_event_ownership(self):
        source = helpers.Tensor(bytearray(range(16)))
        output = helpers.Tensor(bytearray(64))
        self.ctl.all_gather(self.comm, source, output)
        main = helpers.FakeCuda.stream
        side = helpers.Stream('p2p')
        helpers.FakeCuda.stream = side
        submitted = []
        def exchange(comm, sends, recvs):
            self.assertIs(comm, self.comm)
            self.assertEqual(side.waits, [main])
            submitted.append((sends, recvs))
        self.ctl.nccl_exchange(self.comm, 'exchange_all', exchange, {'peer': source}, {'peer': output})
        self.assertEqual(self.ctl._last_stream, side)
        graph = helpers.Graph()
        replay_stream = helpers.Stream('replay')
        helpers.FakeCuda.stream = replay_stream
        self.ctl._graphs[graph] = True
        self.ctl._replay(graph)
        self.assertEqual(replay_stream.waits, [side])
        self.assertEqual(len(submitted), 1)
        self.assertEqual(self.ctl.report()['eager_p2p_calls'], {'exchange_all': 1})

    def test_capture_and_foreign_identity_refused_before_submission_without_poison(self):
        calls = []
        fn = lambda *args: calls.append(args)
        with self.assertRaisesRegex(RuntimeError, 'preinitialized'):
            self.ctl.nccl_exchange(object(), 'exchange', fn)
        for active, capturing in ((object(), False), (None, True)):
            self.ctl._active_graph = active
            helpers.FakeCuda.capturing = capturing
            with self.assertRaisesRegex(RuntimeError, 'eager-only'):
                self.ctl.nccl_exchange(self.comm, 'exchange', fn)
        self.ctl._active_graph = None
        helpers.FakeCuda.capturing = False
        self.assertEqual(calls, [])
        self.assertIsNone(self.ctl._failure)

    def test_submission_and_async_health_failure_poison_without_retry(self):
        calls = []
        def broken(*args):
            calls.append(args)
            raise RuntimeError('send failed')
        with self.assertRaisesRegex(RuntimeError, 'send failed'):
            self.ctl.nccl_exchange(self.comm, 'exchange_all', broken)
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            self.ctl.nccl_exchange(self.comm, 'exchange_all', broken)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.ctl.report()['eager_p2p_calls'], {})

    def test_health_failure_before_exchange_never_submits(self):
        self.ctl._runtime.fail = True
        with self.assertRaisesRegex(RuntimeError, 'native stopped'):
            self.ctl.nccl_exchange(self.comm, 'exchange', lambda *a: self.fail('submitted'))
        self.assertIsNotNone(self.ctl._failure)

    def test_concurrent_submissions_hold_one_lock_through_mark(self):
        entered, release, other_started, other_entered = (threading.Event() for _ in range(4))
        errors = []
        def first(*args):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test release timeout')
        def run(fn):
            try:
                self.ctl.nccl_exchange(self.comm, 'exchange_all', fn)
            except BaseException as exc:
                errors.append(exc)
        t1 = threading.Thread(target=run, args=(first,))
        def second_thread():
            other_started.set()
            run(lambda *args: other_entered.set())
        t2 = threading.Thread(target=second_thread)
        t1.start()
        self.assertTrue(entered.wait(1))
        t2.start()
        self.assertTrue(other_started.wait(1))
        self.assertFalse(other_entered.wait(0.05))
        release.set()
        t1.join(2); t2.join(2)
        self.assertFalse(t1.is_alive() or t2.is_alive())
        self.assertFalse(errors)
        self.assertTrue(other_entered.is_set())


class RingIntegrationTests(unittest.TestCase):
    def make(self, flag='1', exchange='p2p', original=None):
        byte_helpers = load('ring_adapter_bytes', ROOT / 'tests/test_ring_exchange.py')
        torch = byte_helpers.Torch()
        torch.stream = helpers.Stream('relay-main')
        torch.cuda.Event = helpers.Event
        with patch.dict(os.environ, {'TF_GLM_HC_RING_RELAY': flag, 'TF_GLM_HC_EXCHANGE': exchange}):
            ctl = adapter.Controller(helpers.settings(), mode='nccl', vendor_root='/unused',
                vendor_manifest_sha256='a'*64, source_hashes={}, torch_module=torch,
                original_gather=lambda *a: None, original_exchange_all=original)
        return ctl, torch, byte_helpers

    def test_flags_require_exact_values_p2p_and_saved_original(self):
        for flag in ('', 'true', '2', ' 1'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'exactly'):
                self.make(flag, original=lambda *a: None)
        with self.assertRaisesRegex(ValueError, 'requires TF_GLM_HC_EXCHANGE'):
            self.make(exchange='gather', original=lambda *a: None)
        with self.assertRaisesRegex(ValueError, 'saved original'):
            self.make()
        ctl, _, _ = self.make('0')
        self.assertFalse(ctl.relay_identity()['enabled'])

    def test_saved_original_two_adjacent_groups_persistent_source_bound_relay(self):
        calls = []
        def original(comm, sends, recvs):calls.append((comm, list(sends), list(recvs)))
        ctl, torch, b = self.make(original=original)
        comm = types.SimpleNamespace(rank=0, world=4)
        ctl.initialize(comm)
        relay = ctl._ring_relay
        for _ in range(2):
            ctl.nccl_exchange(comm, 'exchange_all', lambda *a: self.fail('recursive/direct wrapper'),
                              *b.inputs(0, [(8, 'torch.float32')], device='cuda:0'))
        self.assertIs(ctl._ring_relay, relay)
        self.assertEqual(calls, [(comm, [1, 3], [1, 3])] * 4)
        state = ctl.report()['ring_relay']
        self.assertEqual((state['calls'], state['rounds']), (2, 4))
        self.assertEqual(state['source_sha256'], hashlib.sha256((ROOT / 'scripts/ring_exchange.py').read_bytes()).hexdigest())
        self.assertEqual((state['max_scratch_bytes'], state['max_growth_bytes']), (64 << 20, 128 << 20))
        self.assertTrue(all(t.recorded for t in torch.allocated))
        with self.assertRaisesRegex(ValueError, 'diagonal'):
            ctl.nccl_exchange(comm, 'exchange', lambda *a: self.fail('diagonal'), [], [], 2)
        adjacent = []
        ctl.nccl_exchange(comm, 'exchange', lambda *a: adjacent.append(a), [], [], 1)
        self.assertEqual(len(adjacent), 1)
        torch.capturing = True
        with self.assertRaisesRegex(RuntimeError, 'eager-only'):
            ctl.nccl_exchange(comm, 'exchange_all', original, {}, {})
        self.assertIsNone(ctl._failure)

    def test_helper_change_after_construction_poisoned_before_submission(self):
        ctl, _, _ = self.make(original=lambda *a: self.fail('submitted'))
        ctl._relay_sha256 = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'source differs'):
            ctl.initialize(types.SimpleNamespace(rank=0, world=4))
        self.assertIsNotNone(ctl._failure)


class OverlayTests(unittest.TestCase):
    def test_full_manifest_drift_additions_and_wrong_import_root_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'tensorfold'
            (root / 'cuda').mkdir(parents=True)
            (root / 'cuda/comm.py').write_text('# comm\n')
            (root / 'front.py').write_text('# full overlay helper\n')
            files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in root.rglob('*') if p.is_file()}
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text(json.dumps({'schema': 'glm-prefill-engine-source/v1', 'source_id': 'unit', 'files': files}))
            env = {'SPARK_RING_ENGINE_MANIFEST': str(manifest),
                   'SPARK_RING_ENGINE_MANIFEST_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   'SPARK_RING_ENGINE_SOURCE_ROOT': str(root)}
            comm = types.ModuleType('unit_comm'); comm.__file__ = str(root / 'cuda/comm.py')
            with patch.dict(os.environ, env):
                self.assertEqual(adapter.verify_tensorfold(comm), files)
                (root / 'front.py').write_text('# drift\n')
                with self.assertRaisesRegex(RuntimeError, 'Engine source changed'):
                    adapter.verify_tensorfold(comm)
                (root / 'front.py').write_text('# full overlay helper\n')
                (root / 'extra.py').write_text('# unbound\n')
                with self.assertRaisesRegex(RuntimeError, 'Engine source changed'):
                    adapter.verify_tensorfold(comm)
                (root / 'extra.py').unlink()
                comm.__file__ = str(Path(directory) / 'outside/cuda/comm.py')
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    adapter.verify_tensorfold(comm)

    def test_legacy_binding_still_rejects_wrong_sources(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(adapter, 'sha256', return_value='f' * 64):
            comm = types.ModuleType('legacy_comm'); comm.__file__ = '/unused/tensorfold/cuda/comm.py'
            with self.assertRaisesRegex(RuntimeError, 'pinned source differs'):
                adapter.verify_tensorfold(comm)

    def controller_fixture(self):
        cfg = prepared_example('rocenante')
        files = {'cuda/comm.py': 'a' * 64, 'new/front.py': 'b' * 64}
        identity = {'engine_source': {'files': files, 'manifest_sha256': 'c' * 64,
                     'engine_flags': {'TF_GLM_HC_EXCHANGE': 'p2p'}},
                    'vendor': {'manifest_sha256': 'd' * 64},
                    'sources': {'rocenante_adapter.py': 'e' * 64}}
        report = dict(mode='rocenante', rank=0, world=4, initialized=True, closed=False, failure=None,
                      native_active=True, hca_names=validator.ordered_hcas(cfg['nodes'][0]),
                      vendor_manifest_sha256='d' * 64, adapter_sha256='e' * 64,
                      tensorfold_source_digest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                      tensorfold_source_file_count=len(files),engine_manifest_sha256='c'*64,
                      max_gather_bytes=cfg['experiment']['max_gather_bytes'],
                      min_gather_bytes=cfg['experiment']['min_gather_bytes'], native_threads=512,
                      native_blocks=cfg['experiment']['native_blocks'], runtime_native_threads=512,
                      runtime_native_blocks=cfg['experiment']['native_blocks'], path_counters=[])
        return cfg, identity, report

    def test_validator_requires_complete_controller_manifest(self):
        cfg, identity, report = self.controller_fixture()
        validator.check_controller(report, cfg, 0, identity)
        report['tensorfold_source_sha256'] = dict(adapter.SOURCE_HASHES)
        del report['tensorfold_source_digest_sha256']
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            validator.check_controller(report, cfg, 0, identity)

    def test_observation_uses_runtime_counter_root_and_identity_manifest(self):
        cfg, identity, report = self.controller_fixture()
        cfg['runtime_source_root'] = '/isolated/runtime/scripts'
        captured = []
        def remote(rank, code, timeout):
            tree = ast.parse(code)
            # Last line prints JSON(host_observation(payload)); inspect, never execute SSH code.
            payload = ast.literal_eval(tree.body[-1].value.args[0].args[0].args[0])
            captured.append(payload)
            return json.dumps({'controller':{'unix_seconds':1,'report':report},
                               'runtime_log':'SPARK_RING_ROCENANTE_ACTIVE {broken2026-10-14T'} )
        op = types.SimpleNamespace(config=cfg, command=lambda *a: (None, ['docker', '-e', 'TF_GLM_HC_EXCHANGE=p2p', validator.IMAGE, 'serve']), remote_python=remote)
        validator.observation(op, 0, 'unused', 'f' * 64, identity)
        self.assertEqual(captured[0]['counter_script'], '/isolated/runtime/scripts/capture-function-counters.py')
        self.assertEqual(captured[0]['engine_sources'], identity['engine_source']['files'])
        self.assertEqual(captured[0]['engine_overlay']['engine_flags'], {'TF_GLM_HC_EXCHANGE': 'p2p'})


class InstallTests(unittest.TestCase):
    def test_installed_optional_wrappers_preserve_arguments_and_old_communicators(self):
        for supported in (False, True):
            with self.subTest(supported=supported):
                calls = []
                class Comm:
                    def __init__(self, *args): pass
                    def all_gather(self, *args): pass
                if supported:
                    def exchange(self, *args, **kwargs):
                        calls.append(('exchange', args, kwargs))
                        return 'pair'
                    def exchange_all(self, *args, **kwargs):
                        calls.append(('exchange_all', args, kwargs))
                        return 'all'
                    Comm.exchange, Comm.exchange_all = exchange, exchange_all
                class Graph:
                    def capture_begin(self, *args): pass
                    def capture_end(self, *args): pass
                    def replay(self): pass
                fake = types.ModuleType('torch')
                fake.cuda = types.SimpleNamespace(CUDAGraph=Graph, is_current_stream_capturing=lambda: False,
                                                  current_stream=lambda: 'main')
                comm_module = types.ModuleType('fake_comm'); comm_module.NCCL = Comm
                with patch.object(adapter, '_INSTALLED', None), patch.object(sys, 'path', sys.path[:]), \
                     patch.object(adapter, 'verify_vendor', return_value=(Path('/unused'), {})), \
                     patch.object(adapter, 'verify_tensorfold', return_value={}), \
                     patch.object(adapter.importlib, 'import_module', return_value=comm_module), \
                     patch.dict(sys.modules, {'torch': fake}), patch.dict(os.environ, {'TF_GLM_COMM': 'nccl'}):
                    ctl = adapter.install(helpers.settings(), '/unused', 'a' * 64)
                    comm = object.__new__(Comm); ctl._comm = comm
                    if supported:
                        self.assertEqual(comm.exchange(['send'], ['recv'], 1), 'pair')
                        self.assertEqual(comm.exchange_all({1: ['a']}, recvs={1: ['b']}), 'all')
                        self.assertEqual(calls, [('exchange', (['send'], ['recv'], 1), {}),
                                                 ('exchange_all', ({1: ['a']},), {'recvs': {1: ['b']}})])
                        self.assertEqual(ctl._p2p_calls, {'exchange': 1, 'exchange_all': 1})
                    else:
                        self.assertFalse(hasattr(Comm, 'exchange_all'))


if __name__ == '__main__':
    unittest.main()
