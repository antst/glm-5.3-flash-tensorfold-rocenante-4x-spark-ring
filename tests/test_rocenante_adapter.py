# SPDX-License-Identifier: MIT
"""Offline adapter-contract tests; these do not qualify CUDA/RDMA hardware.

The dispatch tests use a byte-backed tensor/stream test double. A separate test
uses actual CPU Torch storage when installed to check dtype views and aliases.
No mocked preparation framework is shipped or used by the adapter at runtime.
"""
from contextlib import nullcontext
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rocenante_adapter_under_test", ROOT / "scripts/rocenante_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def settings(**updates):
    values = dict(rank=0, world=4, hca_names=["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"],
                  peer_hosts=["r0", "r1", "r2", "r3"], store_prefix="unit-run")
    values.update(updates)
    return values


SIZES = {"torch.float32": 4, "torch.bfloat16": 2, "torch.int32": 4,
         "torch.int64": 8, "torch.uint8": 1, "torch.float16": 2}


class Tensor:
    """Byte storage; casting a view never performs floating point arithmetic."""
    def __init__(self, storage, dtype="torch.float32", offset=0, nbytes=None, *, cuda=True, contiguous=True):
        self.storage, self.dtype, self.offset = storage, dtype, offset
        self.nbytes = len(storage) - offset if nbytes is None else nbytes
        self.is_cuda, self._contiguous, self.device = cuda, contiguous, "cuda:0"

    def numel(self): return self.nbytes // SIZES[self.dtype]
    def element_size(self): return SIZES[self.dtype]
    def data_ptr(self): return id(self.storage) * 32 + self.offset
    def is_contiguous(self): return self._contiguous
    def reshape(self, *shape): return self
    def view(self, dtype): return Tensor(self.storage, dtype, self.offset, self.nbytes)
    def __getitem__(self, item):
        start, end, step = item.indices(self.numel())
        assert step == 1
        size = self.element_size()
        return Tensor(self.storage, self.dtype, self.offset + start * size, (end - start) * size)

    def raw(self): return bytes(self.storage[self.offset:self.offset + self.nbytes])

    def copy_(self, other):
        def run():
            raw = other.raw()  # Snapshot before writes, including overlapping slices.
            assert len(raw) == self.nbytes
            self.storage[self.offset:self.offset + self.nbytes] = raw
        if FakeCuda.capturing:
            FakeCuda.graph.operations.append(run)
        else:
            run()
        return self


class Stream:
    def __init__(self, name): self.name, self.waits = name, []
    def wait_event(self, event): self.waits.append(event.last)


class Event:
    def __init__(self): self.last = None
    def record(self, stream): self.last = stream


class FakeCuda:
    stream = Stream("default")
    capturing = False
    graph = None
    @classmethod
    def current_stream(cls): return cls.stream
    @classmethod
    def is_current_stream_capturing(cls): return cls.capturing
    @staticmethod
    def synchronize(*args): pass


TORCH = types.SimpleNamespace(cuda=FakeCuda, uint8="torch.uint8")


class Graph:
    def __init__(self): self.operations = []
    def replay(self):
        for operation in self.operations: operation()


class Native:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.gid_indices = [3, 3, 3, 3]
        self._peer_hca_map = [(), (0, 2), (0, 3), (1, 3)]
        self._last_stream = None

    def check_health(self):
        if self.fail: raise RuntimeError("native stopped")

    def all_gather(self, send, *, dim, out, plan):
        self.check_health()
        self.calls.append((send, out, dim, plan))
        def run():
            shard = send.raw()
            # Independent oracle: copy raw shard once per rank, xor rank markers.
            out.storage[out.offset:out.offset + out.nbytes] = b"".join(
                bytes(value ^ rank for value in shard) for rank in range(4))
        if FakeCuda.capturing: FakeCuda.graph.operations.append(run)
        else: run()

    def benchmark_counters(self): return []


class ContractTests(unittest.TestCase):
    def setUp(self):
        FakeCuda.stream = Stream("default")
        FakeCuda.capturing, FakeCuda.graph = False, None
        self.oracle_calls = []
        self.comm = types.SimpleNamespace(rank=0, world=4)
        self.controller = adapter.Controller(settings(), mode="rocenante", vendor_root="/unused",
            vendor_manifest_sha256="a" * 64, source_hashes={}, torch_module=TORCH,
            original_gather=lambda *args: self.oracle_calls.append(args))
        self.controller._comm = self.comm
        self.controller._runtime, self.controller._plan = Native(), object()
        self.controller._scratch = Tensor(bytearray(adapter.MAX_GATHER_BYTES), "torch.uint8")
        self.controller._event = Event()
        self.controller._graph_originals = {"replay": Graph.replay}
        # This test-only context guard does not implement any prepared API.
        guard = types.ModuleType("b12x._lib.runtime_control")
        guard.kernel_resolution_guard = lambda label: nullcontext()
        self.modules = patch.dict(sys.modules, {"b12x._lib.runtime_control": guard})
        self.modules.start()

    def tearDown(self): self.modules.stop()

    def gather(self, send, recv):
        self.controller.all_gather(self.comm, send, recv)

    def test_exact_float_and_integer_payloads(self):
        # Includes NaN payloads, signed zeros, subnormal bits, and arbitrary ints.
        raw = bytes.fromhex("0100c07f0000008001000000ffffffff") * 2
        for dtype in ("torch.float32", "torch.bfloat16", "torch.int32", "torch.int64"):
            with self.subTest(dtype=dtype):
                send, recv = Tensor(bytearray(raw), dtype), Tensor(bytearray(4 * len(raw)), dtype)
                self.gather(send, recv)
                expected = b"".join(bytes(v ^ rank for v in raw) for rank in range(4))
                self.assertEqual(recv.raw(), expected)
                self.assertEqual(send.raw(), raw)
        self.assertEqual(self.controller._eager_calls, 4)
        self.assertFalse(self.oracle_calls)

    def test_aliasing_rank_slot_and_arbitrary_overlap(self):
        for offset in (0, 16, 32, 48, 4):
            storage = bytearray(80)
            raw = bytes(range(16))
            storage[offset:offset + 16] = raw
            send, recv = Tensor(storage, offset=offset, nbytes=16), Tensor(storage, nbytes=64)
            self.gather(send, recv)
            self.assertEqual(recv.raw(), b"".join(bytes(v ^ r for v in raw) for r in range(4)))
        self.assertEqual(self.controller._alias_calls, 5)

    def test_capacity_and_unsupported_fallback_before_native(self):
        samples = [
            (Tensor(bytearray(16), "torch.float16"), "dtype"),
            (Tensor(bytearray(16), cuda=False), "device"),
            (Tensor(bytearray(16), contiguous=False), "noncontiguous"),
            (Tensor(bytearray()), "empty"),
            (Tensor(bytearray(adapter.MAX_GATHER_BYTES + 16)), "capacity"),
        ]
        for send, reason in samples:
            recv = Tensor(bytearray(send.nbytes * 4), send.dtype)
            self.gather(send, recv)
            self.assertEqual(self.controller._fallbacks[reason], 1)
        self.assertFalse(self.controller._runtime.calls)
        self.assertEqual(len(self.oracle_calls), len(samples))

    def test_receive_validation_does_not_enqueue(self):
        with self.assertRaises(ValueError):
            self.gather(Tensor(bytearray(16)), Tensor(bytearray(32)))
        self.assertFalse(self.oracle_calls)
        self.assertFalse(self.controller._runtime.calls)

    def test_native_failure_never_falls_back_and_poison_is_sticky(self):
        self.controller._runtime.fail = True
        with self.assertRaisesRegex(RuntimeError, "native stopped"):
            self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.controller._runtime.fail = False
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.assertFalse(self.oracle_calls)

    def test_capture_requires_preinitialized_managed_graph(self):
        FakeCuda.capturing = True
        with self.assertRaisesRegex(RuntimeError, "managed"):
            self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.assertFalse(self.controller._runtime.calls)

    def test_multiple_captures_replays_then_eager_stream_ordering(self):
        graphs, tensors = [], []
        for index in range(2):
            graph = Graph()
            storage = bytearray(64)
            send, recv = Tensor(storage, offset=16, nbytes=16), Tensor(storage)
            self.controller._capture_begin(graph)
            FakeCuda.capturing, FakeCuda.graph = True, graph
            self.gather(send, recv)
            FakeCuda.capturing, FakeCuda.graph = False, None
            self.controller._capture_end(graph)
            graphs.append(graph)
            tensors.append((send, recv))
            # Earlier graph may replay BEFORE the next graph is captured.
            send.storage[16:32] = bytes([index + 7]) * 16
            self.controller._replay(graph)
        for repeat in range(4):
            for graph, (send, recv) in zip(graphs, tensors):
                raw = bytes((i + repeat) & 255 for i in range(16))
                send.storage[16:32] = raw
                self.controller.wrap_graph(graph).replay()
                self.assertEqual(recv.raw(), b"".join(bytes(v ^ r for v in raw) for r in range(4)))
        previous = FakeCuda.stream
        FakeCuda.stream = Stream("other")
        self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.assertEqual(FakeCuda.stream.waits, [previous])
        self.assertEqual(self.controller._guarded_replays, 10)
        self.assertEqual(self.controller._captured_calls, 2)

    def test_nccl_only_fallback_graph_is_guarded_and_can_be_wrapped(self):
        graph = Graph()
        self.controller._capture_begin(graph)
        FakeCuda.capturing, FakeCuda.graph = True, graph
        send = Tensor(bytearray(adapter.MAX_GATHER_BYTES + 16))
        recv = Tensor(bytearray(send.nbytes * 4))
        self.gather(send, recv)
        FakeCuda.capturing, FakeCuda.graph = False, None
        self.controller._capture_end(graph)
        self.assertFalse(self.controller._graphs[graph])
        self.assertFalse(self.controller._runtime.calls)
        self.controller.wrap_graph(graph).replay()
        previous = FakeCuda.stream
        FakeCuda.stream = Stream("native-after-fallback")
        self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.assertEqual(FakeCuda.stream.waits, [previous])
        self.assertEqual(self.controller._guarded_replays, 1)

    def test_direct_nccl_oracle_graph_registers_without_native_nodes(self):
        graph = Graph()
        self.controller._capture_begin(graph)
        FakeCuda.capturing, FakeCuda.graph = True, graph
        self.controller.nccl_all_gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        FakeCuda.capturing, FakeCuda.graph = False, None
        self.controller._capture_end(graph)
        self.assertFalse(self.controller._graphs[graph])
        self.controller.wrap_graph(graph).replay()
        self.assertEqual(self.controller._guarded_replays, 1)
        self.assertEqual(self.controller._captured_calls, 0)
        self.assertFalse(self.controller._runtime.calls)

    def test_capture_stream_change_refused_before_native(self):
        graph = Graph()
        self.controller._capture_begin(graph)
        FakeCuda.capturing, FakeCuda.graph = True, graph
        FakeCuda.stream = Stream("fork")
        with self.assertRaisesRegex(RuntimeError, "single capture stream"):
            self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.assertFalse(self.controller._runtime.calls)
        FakeCuda.capturing = False
        self.controller._capture_end(graph)

    def test_live_graph_blocks_teardown(self):
        graph = Graph()
        self.controller._graphs[graph] = True
        with self.assertRaisesRegex(RuntimeError, "destroy all"):
            self.controller.close(graphs_released=True)
        del graph
        gc.collect()
        self.assertEqual(len(self.controller._graphs), 0)

    def test_concurrent_eager_call_during_capture_refused_before_enqueue(self):
        graph = Graph()
        self.controller._capture_begin(graph)
        # A second CPU thread has a noncapturing thread-local stream.
        with self.assertRaisesRegex(RuntimeError, "eager submission"):
            self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        with self.assertRaisesRegex(RuntimeError, "eager NCCL"):
            self.controller.nccl_all_gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.assertFalse(self.controller._runtime.calls)
        self.assertFalse(self.oracle_calls)
        self.controller._capture_end(graph)

    def test_nccl_mode_no_native_launch(self):
        self.controller.mode = "nccl"
        self.gather(Tensor(bytearray(16)), Tensor(bytearray(64)))
        self.assertEqual(self.controller._fallbacks["mode_nccl"], 1)
        self.assertFalse(self.controller._runtime.calls)

    def test_tensorfold_end_request_health_hook_detects_async_failure(self):
        self.controller.comm_check(self.comm)
        self.controller._runtime.fail = True
        with self.assertRaisesRegex(RuntimeError, "native stopped"):
            self.controller.comm_check(self.comm)
        self.controller._runtime.fail = False
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            self.controller.comm_check(self.comm)
        with self.assertRaisesRegex(RuntimeError, "preinitialized"):
            self.controller.comm_check(object())


class OfflineValidationTests(unittest.TestCase):
    def test_legacy_tensorfold_roce_rejected_before_import_or_vendor_access(self):
        with patch.dict(os.environ, {"TF_GLM_COMM": "roce"}):
            with self.assertRaisesRegex(ValueError, "TF_GLM_COMM must be nccl"):
                adapter.install(settings(), "/never-accessed", "a" * 64)

    def test_actual_pinned_image_install_health_and_second_init_are_cpu_only(self):
        try:
            import torch
            from tensorfold.cuda import comm as comm_module
        except ImportError:
            self.skipTest("requires actual pinned Mia image; no CUDA or NCCL construction")
        vendor = ROOT / "third_party/prepared-b12x"
        if not vendor.is_dir():
            self.skipTest("pinned vendor source is not mounted")
        cls, graphs = comm_module.NCCL, torch.cuda.CUDAGraph
        originals = {"init": cls.__init__, "gather": cls.all_gather,
                     "check": getattr(cls, "check", None),
                     "begin": graphs.capture_begin, "end": graphs.capture_end,
                     "replay": graphs.replay}
        old_path = sys.path[:]
        initialized = torch.cuda.is_initialized()
        controller = None
        try:
            controller = adapter.install(settings(), vendor,
                adapter.sha256(vendor / "source-manifest.json"), mode="rocenante")
            # Bypass the constructor explicitly: we test the Python hooks only.
            comm = object.__new__(cls)
            controller._comm = comm
            comm.check()
            self.assertIsNot(graphs.replay, originals["replay"])
            with self.assertRaisesRegex(RuntimeError, "multiple NCCL communicators"):
                cls.__init__(object.__new__(cls), 0, 4, "never-contacted.invalid", 1)
            self.assertEqual(torch.cuda.is_initialized(), initialized)
            self.assertFalse(controller.report()["native_active"])
        finally:
            cls.__init__, cls.all_gather = originals["init"], originals["gather"]
            if originals["check"] is None:
                if hasattr(cls, "check"): delattr(cls, "check")
            else:
                cls.check = originals["check"]
            graphs.capture_begin, graphs.capture_end, graphs.replay = (
                originals["begin"], originals["end"], originals["replay"])
            sys.path[:] = old_path
            adapter._INSTALLED = None
            if controller is not None: controller.close()

    def test_settings_reject_bad_capacity_topology_and_prefix(self):
        for kwargs in (dict(rank=4), dict(world=2), dict(max_gather_bytes=32 << 20),
                       dict(max_gather_bytes=True), dict(min_gather_bytes=0),
                       dict(hca_names=["a"] * 4), dict(peer_hosts=["r"] * 4),
                       dict(store_prefix=""), dict(device="cuda:1")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                adapter.validate_settings(settings(**kwargs))

    def test_native_launch_settings_are_bounded_and_threads_fixed(self):
        default = adapter.validate_settings(settings())
        self.assertEqual((default['native_threads'], default['native_blocks']), (512, 8))
        for blocks in (4, 8, 16):
            normalized = adapter.validate_settings(settings(native_blocks=blocks))
            self.assertEqual((normalized['native_threads'], normalized['native_blocks']), (512, blocks))
        for blocks in (True, 8.0, '8', None, 0, 2, 32):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                adapter.validate_settings(settings(native_blocks=blocks))
        for threads in (True, 512.0, '512', None, 256, 1024):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                adapter.validate_settings(settings(native_threads=threads))

    def test_native_constructor_kwargs_and_runtime_geometry(self):
        class Factory:
            def __init__(self, *, threads=512, blocks=8):
                self._threads, self._blocks = threads, blocks
        for blocks in (4, 8, 16):
            requested = settings(native_blocks=blocks)
            kwargs = adapter.native_launch_kwargs(Factory, requested)
            self.assertEqual(kwargs, {'threads': 512, 'blocks': blocks})
            adapter.verify_native_launch(Factory(**kwargs), requested)
        # A constructor accepting but ignoring an option must not qualify.
        with self.assertRaisesRegex(RuntimeError, 'geometry mismatch'):
            adapter.verify_native_launch(Factory(blocks=8), settings(native_blocks=16))
        for bad in (types.SimpleNamespace(_threads=512),
                    types.SimpleNamespace(_threads=512.0, _blocks=8),
                    types.SimpleNamespace(_threads=256, _blocks=8)):
            with self.assertRaisesRegex(RuntimeError, 'geometry mismatch'):
                adapter.verify_native_launch(bad, settings())

    def test_native_factory_abi_mismatch_fails_before_construction(self):
        called = []
        class MissingBlocks:
            def __init__(self, *, threads=512): called.append(self)
        class CatchAll:
            def __init__(self, **kwargs): called.append(self)
        class Positional:
            def __init__(self, threads=512, blocks=8): called.append(self)
        for factory in (MissingBlocks, CatchAll, Positional):
            with self.subTest(factory=factory), self.assertRaisesRegex(RuntimeError, 'constructor'):
                adapter.native_launch_kwargs(factory, settings())
        self.assertEqual(called, [])

    def test_pinned_native_grid_and_prepared_setup_bind_launch_geometry(self):
        vendor = ROOT / 'third_party/prepared-b12x/b12x/comm/roce'
        tree = ast.parse((vendor / 'roce_oneshot.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_grid_blocks']
        self.assertEqual(len(functions), 1)
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'pinned-grid', 'exec'), namespace)
        grid = namespace['_grid_blocks']
        # Actual pinned formula, independent expected boundary values. Tiny
        # raw gathers pad to one 16-byte pack; 8 -> 16 differs only above128KiB.
        cases = ((16, (1, 1, 1)), (16384, (1, 1, 1)), (65536, (4, 4, 4)),
                 (131072, (4, 8, 8)), (131088, (4, 8, 16)),
                 (262144, (4, 8, 16)), (1048576, (4, 8, 16)),
                 (16777216, (4, 8, 16)))
        for nbytes, expected in cases:
            with self.subTest(nbytes=nbytes):
                self.assertEqual(tuple(grid(nbytes // 16, 512, limit) for limit in (4, 8, 16)), expected)
        constructor = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                           and n.name == 'RoceOneshotAllReduce')
        init = next(n for n in constructor.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        self.assertTrue({'threads', 'blocks'}.issubset({n.arg for n in init.args.kwonlyargs}))
        prepared = ast.parse((vendor / '_preparation.py').read_text())
        setup = next(n for n in prepared.body if isinstance(n, ast.FunctionDef) and n.name == '_runtime_setup')
        namespace = {'FrozenMapping': dict}
        exec(compile(ast.Module(body=[setup], type_ignores=[]), 'pinned-setup', 'exec'), namespace)
        runtime = types.SimpleNamespace(_threads=512, _blocks=16, _layout=types.SimpleNamespace(slots=2, flag_stride=16),
            hca_names=['a','b','c','d'], _opposite_paths=2, _peer_hca_map=[(),(0,2),(0,3),(1,3)],
            max_size=2097152, max_gather_bytes=16777216, gid_indices=[3]*4, spin_limit=1)
        bound = namespace['_runtime_setup'](runtime)
        self.assertEqual((bound['threads'], bound['max_blocks']), (512, 16))

    def test_host_report_records_requested_and_actual_native_geometry(self):
        controller = adapter.Controller(settings(native_blocks=16), mode="rocenante", vendor_root="/unused",
            vendor_manifest_sha256="a" * 64, source_hashes={}, torch_module=TORCH,
            original_gather=lambda *args: None)
        controller._runtime = Native()
        controller._runtime._threads, controller._runtime._blocks = 512, 16
        report = controller.report()
        self.assertEqual((report['native_threads'], report['native_blocks']), (512, 16))
        self.assertEqual((report['runtime_native_threads'], report['runtime_native_blocks']), (512, 16))

    def test_selection_cache_never_changes_warm_or_cold_policy(self):
        cache = adapter._NoSelectionCache()
        self.assertIsNone(cache.get("rank-local-key"))
        cache.save("rank-local-key", {"selected": "one-rank-only"})
        self.assertIsNone(cache.get("rank-local-key"))

    def test_vendor_inventory_rejects_drift_additions_and_path_escape(self):
        required = ["b12x/preparation/session.py", "b12x/preparation/types.py",
                    "b12x/_lib/compile_plan.py", "b12x/_lib/program_cache.py",
                    "b12x/comm/roce/roce_oneshot.py", "b12x/comm/roce/_preparation.py",
                    "b12x/comm/roce/_allgather_cute.py", "b12x/comm/roce/_roce_proxy.c"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in required:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"pinned source")
            manifest = dict(schema="tensorfold-rocenante-vendor/v1",
                            files={name: adapter.sha256(root / name) for name in required})
            path = root / "source-manifest.json"
            path.write_text(json.dumps(manifest))
            digest = adapter.sha256(path)
            adapter.verify_vendor(root, digest)
            (root / required[0]).write_text("drift")
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                adapter.verify_vendor(root, digest)
            (root / required[0]).write_bytes(b"pinned source")
            (root / "b12x/extra.py").write_text("unbound")
            with self.assertRaisesRegex(RuntimeError, "unbound"):
                adapter.verify_vendor(root, digest)
            manifest["files"]["../outside.py"] = "a" * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                adapter.verify_vendor(root, adapter.sha256(path))

    def test_torch_cpu_dtype_views_preserve_special_bits_and_alias(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch absent on management host; run in pinned no-GPU image")
        raw = bytes.fromhex("0100c07f0000008001000000ffffffff")
        for dtype in (torch.float32, torch.bfloat16, torch.int32, torch.int64):
            storage = torch.tensor(list(raw * 4), dtype=torch.uint8)
            source = storage[16:32].view(dtype)
            snapshot = source.reshape(-1).view(torch.uint8).clone()
            for rank in range(4):
                storage[rank * 16:(rank + 1) * 16].copy_(snapshot)
            self.assertEqual(bytes(storage.tolist()), raw * 4)


if __name__ == "__main__":
    unittest.main()
