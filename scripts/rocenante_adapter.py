# SPDX-License-Identifier: MIT
"""TensorFold's rank-major gathers through the genuine prepared RoCEnante API.

This module has no import-time Torch, CUDA, RDMA, or distributed side effects.
``install`` source-checks the pinned engine/vendor, then hooks NCCL construction
so Gloo, proxy construction, compilation, warmup, and scratch allocation finish
before TensorFold captures any graph. Reduction and TensorFold's rank-order sum
are unchanged. Geometry outside admission goes to the saved NCCL implementation;
an error after entering the native path permanently poisons this controller.

The native runtime owns one sequence/channel. Managed graph replays and eager
gathers share a CUDA ordering event; concurrent CPU submissions are serialized.
Capture uses one stream per graph. Every TensorFold gather graph is guarded,
including NCCL-only fallback graphs; unrelated graphs are untouched. Managed
graphs must be destroyed before controller teardown.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import weakref

MAX_GATHER_BYTES = 16 << 20
NATIVE_THREADS = 512
NATIVE_BLOCKS = (4, 8, 16)
SOURCE_HASHES = {
    "cuda/comm.py": "92f1be63fc4bf1ba77675eb74a13716d07e39bc74839579a82ee47b34bf2f986",
    "families/glm5_next/cuda/verify.py": "e0d450b84fe6b4e285ae0bb5942e92aa86762a554e29aae8e38d53c447b1e13f",
    "families/glm5_next/cuda/graphs.py": "43e31efc653c61d3e0c0a6fef4bc78afaa057582d16c7a8613986d3a73a4465d",
    "families/glm5_next/cuda/multi.py": "3f90da926a0bcb4de124dcd4416b2251c2f96723c05abb17568368314a2b9981",
}
_DTYPES = {"torch.float32", "torch.bfloat16", "torch.int32", "torch.int64"}
_INSTALLED = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_vendor(root, expected_manifest_sha256):
    """Validate every manifest-bound source before allowing a vendor import."""
    root = Path(root).resolve(strict=True)
    manifest_path = root / "source-manifest.json"
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest_sha256)):
        raise ValueError("expected vendor manifest SHA256 is required")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("vendor manifest SHA256 differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "tensorfold-rocenante-vendor/v1":
        raise ValueError("unsupported vendor manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("empty vendor source inventory")
    for name, digest in files.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or ":" in name or not re.fullmatch(r"[0-9a-f]{64}", str(digest))):
            raise ValueError("invalid vendor source inventory entry")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError("vendor source is missing or outside its root: " + name)
        if sha256(path) != digest:
            raise RuntimeError("vendor source SHA256 differs: " + name)
    required = {
        "b12x/preparation/session.py", "b12x/preparation/types.py",
        "b12x/_lib/compile_plan.py", "b12x/_lib/program_cache.py",
        "b12x/comm/roce/roce_oneshot.py", "b12x/comm/roce/_preparation.py",
        "b12x/comm/roce/_allgather_cute.py", "b12x/comm/roce/_roce_proxy.c",
    }
    if not required.issubset(files):
        raise ValueError("vendor manifest omits required prepared runtime sources")
    # A manifest must bind every importable package source, including additions.
    actual = {p.relative_to(root).as_posix() for p in (root / "b12x").rglob("*")
              if p.is_file() and p.suffix in {".py", ".c", ".cpp", ".h"}}
    if not actual.issubset(files):
        raise RuntimeError("unbound vendor package source: " + ", ".join(sorted(actual - set(files))))
    return root, manifest


def verify_tensorfold(comm_module):
    root = Path(inspect.getfile(comm_module)).resolve().parents[1]
    from engine_binding import runtime_verify
    overlay = runtime_verify(root)
    if overlay is not None:
        return overlay
    observed = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        observed[relative] = sha256(path)
        if observed[relative] != expected:
            raise RuntimeError("TensorFold pinned source differs: " + relative)
    return observed


def validate_settings(settings):
    settings = dict(settings)
    rank, world = settings.get("rank"), settings.get("world", 4)
    if type(rank) is not int or rank not in range(4) or type(world) is not int or world != 4:
        raise ValueError("prepared TensorFold adapter requires rank 0..3, world 4")
    maximum = settings.get("max_gather_bytes", MAX_GATHER_BYTES)
    minimum = settings.get("min_gather_bytes", 1)
    if type(maximum) is not int or not 16 <= maximum <= MAX_GATHER_BYTES or maximum % 16:
        raise ValueError("max_gather_bytes must be a 16-byte multiple no larger than 16MiB")
    if type(minimum) is not int or not 1 <= minimum <= maximum:
        raise ValueError("min_gather_bytes must be between 1 and max_gather_bytes")
    blocks = settings.get("native_blocks", 8)
    threads = settings.get("native_threads", NATIVE_THREADS)
    if type(blocks) is not int or blocks not in NATIVE_BLOCKS:
        raise ValueError("native_blocks must be an integer from 4, 8, 16")
    if type(threads) is not int or threads != NATIVE_THREADS:
        raise ValueError("native_threads is fixed at 512")
    hcas = settings.get("hca_names")
    if (not isinstance(hcas, (list, tuple)) or len(hcas) != 4
            or any(not isinstance(h, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", h) for h in hcas)
            or len(set(hcas)) != 4):
        raise ValueError("four distinct ordered HCA names are required")
    peers = settings.get("peer_hosts")
    if (not isinstance(peers, (list, tuple)) or len(peers) != world
            or any(not isinstance(p, str) or not p for p in peers) or len(set(peers)) != world):
        raise ValueError("four ordered unique peer_hosts are required")
    prefix = settings.get("store_prefix")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_./-]{1,240}", prefix):
        raise ValueError("an explicit run-specific store_prefix is required")
    timeout = settings.get("timeout_seconds", 600)
    workers = settings.get("compile_workers", 0)
    if type(timeout) is not int or not 10 <= timeout <= 3600:
        raise ValueError("timeout_seconds must be an integer from 10 to 3600")
    if type(workers) is not int or not 0 <= workers <= 8:
        raise ValueError("compile_workers must be an integer from 0 to 8")
    if settings.get("device", "cuda:0") != "cuda:0":
        raise ValueError("one GB10 GPU per rank requires device cuda:0")
    settings.update(rank=rank, world=world, hca_names=list(hcas), peer_hosts=list(peers),
                    max_gather_bytes=maximum, min_gather_bytes=minimum,
                    native_blocks=blocks, native_threads=threads,
                    timeout_seconds=timeout, compile_workers=workers, device="cuda:0")
    return settings


def native_launch_kwargs(factory, settings):
    """Require the pinned explicit constructor ABI before opening native QPs."""
    normalized = validate_settings(settings)
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError("cannot verify native constructor launch parameters") from error
    if any(name not in parameters or parameters[name].kind != inspect.Parameter.KEYWORD_ONLY
           for name in ("threads", "blocks")):
        raise RuntimeError("native constructor must explicitly accept keyword-only threads and blocks")
    return {"threads": normalized["native_threads"], "blocks": normalized["native_blocks"]}


def verify_native_launch(runtime, settings):
    """Fail closed if a factory ignored requested geometry, before preparation."""
    normalized = validate_settings(settings)
    for attribute, expected in (("_threads", normalized["native_threads"]),
                                ("_blocks", normalized["native_blocks"])):
        actual = getattr(runtime, attribute, None)
        if type(actual) is not int or actual != expected:
            raise RuntimeError("native runtime launch geometry mismatch: " + attribute)


def byte_overlap(send, recv):
    start, end = send.data_ptr(), send.data_ptr() + send.numel() * send.element_size()
    rstart, rend = recv.data_ptr(), recv.data_ptr() + recv.numel() * recv.element_size()
    return start < rend and rstart < end


def admission(send, recv, *, world, minimum, maximum):
    """Pure metadata admission; every rank must use the same collective geometry."""
    if recv.numel() != send.numel() * world or send.dtype != recv.dtype:
        raise ValueError("all_gather: recv must hold world x send of the same dtype")
    if str(send.dtype) not in _DTYPES:
        return "dtype"
    if not send.is_cuda or not recv.is_cuda or send.device != recv.device:
        return "device"
    if not send.is_contiguous() or not recv.is_contiguous():
        return "noncontiguous"
    nbytes = send.numel() * send.element_size()
    if nbytes == 0:
        return "empty"
    if nbytes < minimum:
        return "below_minimum"
    if nbytes > maximum:
        return "capacity"
    return None


class _NoSelectionCache:
    """Only selection caching is disabled; compiled-program caching is intact.

    The pinned session otherwise consults rank-local saved selections even with
    autotune=False. All ranks must prime the same fixed RDMA plan on every boot.
    This object is a cache policy for the genuine PreparationSession, not a
    replacement Plan, prepared state, session, kernel, or native transport.
    """
    def get(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None


class _GraphProxy:
    def __init__(self, controller, graph):
        self._controller, self._graph = controller, graph

    def replay(self):
        return self._controller._replay(self._graph)

    def __getattr__(self, name):
        return getattr(self._graph, name)


class Controller:
    def __init__(self, settings, *, mode, vendor_root, vendor_manifest_sha256,
                 source_hashes, torch_module, original_gather, original_exchange_all=None):
        self.settings = validate_settings(settings)
        relay = os.environ.get("TF_GLM_HC_RING_RELAY", "0")
        if relay not in ("0", "1"):
            raise ValueError("TF_GLM_HC_RING_RELAY must be exactly 0 or 1")
        self._relay_enabled = relay == "1"
        if self._relay_enabled and os.environ.get("TF_GLM_HC_EXCHANGE") != "p2p":
            raise ValueError("TF_GLM_HC_RING_RELAY=1 requires TF_GLM_HC_EXCHANGE=p2p")
        if self._relay_enabled and not callable(original_exchange_all):
            raise ValueError("ring relay requires the saved original NCCL.exchange_all")
        self._original_exchange_all = original_exchange_all
        self._ring_relay = None
        self._relay_path = Path(__file__).with_name("ring_exchange.py")
        self._relay_sha256 = sha256(self._relay_path) if self._relay_enabled else None
        if mode not in {"rocenante", "nccl"}:
            raise ValueError("mode must be rocenante or nccl")
        self.mode, self.vendor_root = mode, Path(vendor_root)
        self.vendor_manifest_sha256, self.source_hashes = vendor_manifest_sha256, source_hashes
        self.torch, self._original_gather = torch_module, original_gather
        self._lock = threading.RLock()
        self._comm = self._runtime = self._plan = self._session = self._result = None
        self._scratch = self._group = self._event = None
        self._owns_group = False
        self._last_stream = self._active_graph = self._capture_stream = None
        self._graphs = weakref.WeakKeyDictionary()
        self._graph_originals = {}
        self._captures = self._guarded_replays = self._eager_calls = self._eager_bytes = 0
        self._captured_calls = self._captured_bytes = self._alias_calls = 0
        self._fallbacks, self._geometries = Counter(), Counter()
        self._failure = None
        self._closed = False
        self._p2p_calls = Counter()

    def _check(self):
        if self._closed:
            raise RuntimeError("RoCEnante controller is closed")
        if self._failure is not None:
            raise RuntimeError("RoCEnante controller is poisoned: " + self._failure)

    def _poison(self, exc):
        self._failure = f"{type(exc).__name__}: {exc}"

    def initialize(self, comm):
        """Called after original NCCL initialization and before any model graph."""
        with self._lock:
            self._check()
            if self._comm is not None:
                if comm is not self._comm:
                    raise RuntimeError("one controller cannot serve multiple NCCL communicators")
                return self
            if (comm.rank, comm.world) != (self.settings["rank"], self.settings["world"]):
                raise ValueError("TensorFold communicator differs from adapter rank/world")
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("RoCEnante initialization must finish before CUDA capture")
            self._comm = comm
            if self._relay_enabled:
                try:
                    module = importlib.import_module("ring_exchange")
                    if (Path(inspect.getfile(module)).resolve() != self._relay_path.resolve()
                            or sha256(self._relay_path) != self._relay_sha256):
                        raise RuntimeError("ring relay helper source differs")
                    self._ring_relay = module.RingExchange(comm, self._original_exchange_all, self.torch,
                        max_scratch_bytes=64 << 20, max_growth_bytes=128 << 20)
                except BaseException as exc:
                    self._poison(exc)
                    raise
            if self.mode == "nccl":
                if self._relay_enabled:
                    self._event = self.torch.cuda.Event()
                    self._mark()
                return self
            try:
                import torch.distributed as dist
                from b12x.comm import roce
                from b12x.comm.roce import _preparation
                from b12x.preparation import PreparationSession, PreparedCall
                launch = native_launch_kwargs(roce.AllReduce, self.settings)
                if dist.is_initialized():
                    raise RuntimeError("adapter requires ownership of its Gloo default process group")
                store = dist.PrefixStore(self.settings["store_prefix"] + "/rocenante/", comm.store)
                dist.init_process_group("gloo", store=store, rank=comm.rank, world_size=comm.world,
                                        timeout=timedelta(seconds=self.settings["timeout_seconds"]))
                self._owns_group = True
                self._group = dist.group.WORLD
                policy = {
                    "mode": self.mode, "world": comm.world,
                    "min_gather_bytes": self.settings["min_gather_bytes"],
                    "max_gather_bytes": self.settings["max_gather_bytes"],
                    "native_threads": self.settings["native_threads"],
                    "native_blocks": self.settings["native_blocks"],
                    "peer_hosts": self.settings["peer_hosts"],
                    "vendor_manifest_sha256": self.vendor_manifest_sha256,
                    "tensorfold_source_sha256": self.source_hashes,
                    "ring_relay": self.relay_identity(),
                    "selection_cache": "disabled",
                }
                policies = [None] * comm.world
                dist.all_gather_object(policies, policy, group=self._group)
                if any(peer != policy for peer in policies):
                    raise RuntimeError("all ranks must use identical gather admission and source policy")
                device = self.torch.device(self.settings["device"])
                # Constructor honors hca_names; from_exchange_group ignores those kwargs.
                self._runtime = roce.AllReduce(exchange_group=self._group, device=device,
                    max_size=2 << 20, max_gather_bytes=self.settings["max_gather_bytes"],
                    hca_names=self.settings["hca_names"], **launch)
                verify_native_launch(self._runtime, self.settings)
                self._scratch = self.torch.empty(self.settings["max_gather_bytes"],
                                                dtype=self.torch.uint8, device=device)
                self._event = self.torch.cuda.Event()
                query = roce.query_from_runtime(self._runtime, surface="AllReduce.all_gather",
                    call={"dtypes": ("float32",)}, topology="roce_rdma",
                    peer_hosts=tuple(self.settings["peer_hosts"]))
                self._plan = roce.plan(query, runtime=self._runtime)
                seed = self.torch.zeros(4, dtype=self.torch.float32, device=device)

                def prepare(state):
                    reduce_call = _preparation.prepared_call(state, inp=seed)
                    gather_call = _preparation.prepared_gather_call(state, inp=seed)
                    return PreparedCall(run=lambda: (reduce_call.run(), gather_call.run()))

                self._session = PreparationSession(device=device, autotune=False,
                                                  compile_workers=self.settings["compile_workers"])
                self._session._cache = _NoSelectionCache()
                dist.barrier(group=self._group)
                self._result = self._session.prepare((self._plan.request(
                    name="tensorfold-raw-gather", prepare_call=prepare),))
                self.torch.cuda.synchronize(device)
                dist.barrier(group=self._group)
                self._runtime.check_health()
                self._mark()
            except BaseException as exc:
                self._poison(exc)
                raise
            return self

    def _order(self):
        current = self.torch.cuda.current_stream()
        if self._last_stream is not None and current != self._last_stream:
            current.wait_event(self._event)

    def _mark(self):
        if self._event is not None:
            current = self.torch.cuda.current_stream()
            self._event.record(current)
            self._last_stream = current

    def nccl_all_gather(self, send, recv, comm=None):
        """Saved NCCL oracle; no native dispatch and no native counter increment."""
        with self._lock:
            self._check()
            comm = comm or self._comm
            if comm is None:
                raise RuntimeError("controller is not initialized")
            capturing = self.torch.cuda.is_current_stream_capturing()
            if self._active_graph is not None and not capturing:
                raise RuntimeError("eager NCCL submission during native graph capture is unsupported")
            if capturing and self.mode == "rocenante" and self._active_graph is not None:
                if self.torch.cuda.current_stream() != self._capture_stream:
                    raise RuntimeError("all collective calls in a graph must use its single capture stream")
                # Also register direct oracle captures, which bypass all_gather.
                if self._active_graph not in self._graphs:
                    self._graphs[self._active_graph] = False
            if not capturing:
                self._order()
            try:
                self._original_gather(comm, send, recv)
                if not capturing:
                    self._mark()
            except BaseException as exc:
                self._poison(exc)
                raise

    def nccl_exchange(self, comm, operation, original, *args, **kwargs):
        """Eager NCCL P2P shares the native/graph ordering event and lock.

        No event is inserted into a capture: the prefill exchanges are eager.
        Refused identity/capture calls do not poison an otherwise healthy runtime;
        an error once ordering/submission starts does, with no transport retry.
        """
        with self._lock:
            self._check()
            if comm is not self._comm:
                raise RuntimeError("P2P communicator was not preinitialized")
            if self._active_graph is not None or self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("NCCL P2P is eager-only; submission during capture is unsupported")
            if self._relay_enabled:
                if self._ring_relay is None:
                    raise RuntimeError("ring relay is not initialized")
                if operation == "exchange":
                    peer = kwargs.get("peer", args[2] if len(args) > 2 else None)
                    rank = self.settings["rank"]
                    if type(peer) is not int or peer not in ((rank - 1) % 4, (rank + 1) % 4):
                        raise ValueError("ring relay forbids direct diagonal paired exchange")
                elif operation != "exchange_all":
                    raise ValueError("unknown ring relay operation")
            self.check_health()
            try:
                self._order()
                result = (self._ring_relay.exchange_all(*args, **kwargs)
                          if self._relay_enabled and operation == "exchange_all"
                          else original(comm, *args, **kwargs))
                self._mark()
                self._p2p_calls[operation] += 1
                self.check_health()
                return result
            except BaseException as exc:
                self._poison(exc)
                raise

    def all_gather(self, comm, send, recv):
        with self._lock:
            self.check_health()
            if comm is not self._comm:
                raise RuntimeError("gather communicator was not preinitialized")
            reason = admission(send, recv, world=self.settings["world"],
                minimum=self.settings["min_gather_bytes"], maximum=self.settings["max_gather_bytes"])
            if self.mode == "nccl":
                reason = "mode_nccl"
            if reason:
                self._fallbacks[reason] += 1
                return self.nccl_all_gather(send, recv, comm)
            if self._runtime is None or self._plan is None:
                raise RuntimeError("prepared native runtime is not initialized")
            capturing = self.torch.cuda.is_current_stream_capturing()
            if self._active_graph is not None and not capturing:
                raise RuntimeError("eager submission during native graph capture is unsupported")
            if capturing:
                if self._active_graph is None:
                    raise RuntimeError("native gather needs a managed CUDAGraph capture")
                current = self.torch.cuda.current_stream()
                if current != self._capture_stream:
                    raise RuntimeError("all native calls in a graph must use its single capture stream")
            else:
                self._order()
            nbytes = send.numel() * send.element_size()
            self._geometries[(str(send.dtype), nbytes)] += 1
            try:
                # Pack raw bytes before ANY output write, including arbitrary overlap.
                flat = send.reshape(-1)
                if byte_overlap(send, recv):
                    self._alias_calls += 1
                    staged = self._scratch[:nbytes]
                    staged.copy_(flat.view(self.torch.uint8))
                    flat = staged.view(send.dtype)
                from b12x._lib.runtime_control import kernel_resolution_guard
                with kernel_resolution_guard("TensorFold prepared raw RoCEnante gather"):
                    self._runtime.all_gather(flat, dim=0, out=recv.reshape(-1), plan=self._plan)
                if capturing:
                    self._captured_calls += 1
                    self._captured_bytes += nbytes
                    self._graphs[self._active_graph] = True
                else:
                    self._eager_calls += 1
                    self._eager_bytes += nbytes
                    self._mark()
                    self._runtime.check_health()
            except BaseException as exc:
                self._poison(exc)
                raise

    def _capture_begin(self, graph):
        with self._lock:
            self._check()
            if self.mode == "nccl":
                return
            if self._comm is None or self._runtime is None:
                raise RuntimeError("native initialization must precede graph capture")
            if self._active_graph is not None:
                raise RuntimeError("nested or concurrent native graph captures are unsupported")
            self._order()  # Before CUDA capture begins, never an outside event inside it.
            self._active_graph = graph
            self._capture_stream = self.torch.cuda.current_stream()
            self._runtime._last_stream = None
            self._captures += 1

    def _capture_end(self, graph):
        with self._lock:
            if self.mode == "nccl":
                return
            if self._active_graph is not graph:
                raise RuntimeError("CUDA graph capture ownership differs")
            self._active_graph = self._capture_stream = None
            self._runtime._last_stream = None

    def wrap_graph(self, graph):
        """Explicit graph handle for probes; runtime captures are auto-registered."""
        if self.mode == "rocenante" and graph not in self._graphs:
            raise RuntimeError("graph contains no registered TensorFold gather")
        return _GraphProxy(self, graph)

    def _replay(self, graph):
        with self._lock:
            self._check()
            replay = self._graph_originals["replay"]
            if graph not in self._graphs:
                return replay(graph)
            if self.torch.cuda.is_current_stream_capturing() or self._active_graph is not None:
                raise RuntimeError("managed graph replay during capture is unsupported")
            try:
                self._runtime.check_health()
                self._order()
                result = replay(graph)
                self._mark()
                self._guarded_replays += 1
                self._runtime.check_health()
                return result
            except BaseException as exc:
                self._poison(exc)
                raise

    @contextmanager
    def capture_scope(self):
        """Compatibility scope for probes; individual graph hooks own ordering."""
        self._check()
        if self._comm is None:
            raise RuntimeError("initialize the communicator before capture_scope")
        yield self

    def check_health(self):
        with self._lock:
            self._check()
            if self._runtime is not None:
                try:
                    self._runtime.check_health()
                except BaseException as exc:
                    self._poison(exc)
                    raise

    def comm_check(self, comm):
        """TensorFold's optional end-of-request ``comm.check()`` hook."""
        if comm is not self._comm:
            raise RuntimeError("health check communicator was not preinitialized")
        self.check_health()

    def relay_identity(self):
        return {"enabled": self._relay_enabled, "source_sha256": self._relay_sha256,
                "route": "adjacent-two-round" if self._relay_enabled else "direct",
                "max_scratch_bytes": 64 << 20, "max_growth_bytes": 128 << 20}

    def report(self, *, detailed=False):
        """Ordinary reports read host proxy state only; detailed stats may sync GPU."""
        with self._lock:
            if not self._closed and self._failure is None:
                self.check_health()
            runtime = self._runtime
            data = {"schema": "tensorfold-rocenante-adapter/v1", "mode": self.mode,
                "initialized": self._comm is not None, "native_active": runtime is not None,
                "rank": self.settings["rank"], "world": self.settings["world"],
                "hca_names": self.settings["hca_names"], "peer_hosts": self.settings["peer_hosts"],
                "vendor_manifest_sha256": self.vendor_manifest_sha256,
                "adapter_sha256": sha256(__file__),
                "tensorfold_source_sha256": self.source_hashes,
                "ring_relay": {**(self._ring_relay.report() if self._ring_relay is not None else {}),
                               **self.relay_identity()},
                "max_gather_bytes": self.settings["max_gather_bytes"],
                "min_gather_bytes": self.settings["min_gather_bytes"],
                "native_threads": self.settings["native_threads"],
                "native_blocks": self.settings["native_blocks"],
                "runtime_native_threads": getattr(runtime, "_threads", None),
                "runtime_native_blocks": getattr(runtime, "_blocks", None),
                "selection_cache": "disabled", "compiled_program_cache": "retained",
                "capture_count": self._captures, "live_managed_graphs": len(self._graphs),
                "live_native_graphs": sum(bool(native) for native in self._graphs.values()),
                "captured_native_calls": self._captured_calls, "captured_native_bytes": self._captured_bytes,
                "eager_native_calls": self._eager_calls, "eager_native_bytes": self._eager_bytes,
                "alias_staging_calls": self._alias_calls, "guarded_replays": self._guarded_replays,
                "eager_p2p_calls": dict(self._p2p_calls),
                "fallback_reasons": dict(self._fallbacks), "failure": self._failure,
                "closed": self._closed,
                "geometry_cache_entries": [{"dtype": d, "bytes": n, "calls": c}
                    for (d, n), c in sorted(self._geometries.items())]}
            if runtime is not None:
                data["gid_indices"] = list(runtime.gid_indices)
                data["peer_hca_map"] = [list(v) for v in runtime._peer_hca_map]
                data["path_counters"] = runtime.benchmark_counters()
                if detailed:
                    data["native_stats"] = runtime.stats()
            return data

    def close(self, *, graphs_released=False):
        """Never release staging/proxy state while a graph can still reference it."""
        with self._lock:
            if self._closed:
                return
            if self._graphs or self._active_graph is not None:
                raise RuntimeError("destroy all native CUDA graphs before closing the controller")
            if self._runtime is not None and not graphs_released:
                raise RuntimeError("close requires explicit graphs_released=True after destroying graphs")
            if self._runtime is not None:
                self.torch.cuda.synchronize()
            # Keep runtime alive while genuine prepared resources are released.
            if self._result is not None:
                self._result.close()
            if self._session is not None:
                self._session.close()
            if self._runtime is not None:
                self._runtime.close()
            if self._owns_group:
                import torch.distributed as dist
                dist.destroy_process_group(self._group)
            self._closed = True
            self._scratch = self._runtime = self._plan = self._result = self._session = None


def install(settings, vendor_root, expected_vendor_manifest_sha256, mode="rocenante"):
    """Source-bind and install exactly once; no CUDA/NIC work until NCCL init."""
    global _INSTALLED
    if _INSTALLED is not None:
        raise RuntimeError("RoCEnante adapter is already installed")
    if os.environ.get("TF_GLM_COMM", "nccl") != "nccl":
        raise ValueError("TF_GLM_COMM must be nccl; this adapter owns the prepared RoCEnante dispatch")
    normalized = validate_settings(settings)
    if mode not in {"rocenante", "nccl"}:
        raise ValueError("mode must be rocenante or nccl")
    root, _ = verify_vendor(vendor_root, expected_vendor_manifest_sha256)
    for name, module in tuple(sys.modules.items()):
        if name == "b12x" or name.startswith("b12x."):
            file = getattr(module, "__file__", None)
            if file is None or not Path(file).resolve().is_relative_to(root):
                raise RuntimeError("conflicting B12X module already imported: " + name)
    sys.path.insert(0, str(root))
    comm_module = importlib.import_module("tensorfold.cuda.comm")
    observed = verify_tensorfold(comm_module)
    import torch
    cls = comm_module.NCCL
    original_init, original_gather = cls.__init__, cls.all_gather
    controller = Controller(normalized, mode=mode, vendor_root=root,
        vendor_manifest_sha256=expected_vendor_manifest_sha256, source_hashes=observed,
        torch_module=torch, original_gather=original_gather,
        original_exchange_all=getattr(cls, "exchange_all", None))

    def initialize(comm, *args, **kwargs):
        if controller._comm is not None and comm is not controller._comm:
            raise RuntimeError("one controller cannot serve multiple NCCL communicators")
        original_init(comm, *args, **kwargs)
        controller.initialize(comm)

    def gather(comm, send, recv):
        return controller.all_gather(comm, send, recv)

    def check(comm):
        return controller.comm_check(comm)

    graph_cls = torch.cuda.CUDAGraph
    controller._graph_originals = {name: getattr(graph_cls, name)
                                   for name in ("capture_begin", "capture_end", "replay")}

    def capture_begin(graph, *args, **kwargs):
        controller._capture_begin(graph)
        try:
            return controller._graph_originals["capture_begin"](graph, *args, **kwargs)
        except BaseException as exc:
            controller._active_graph = controller._capture_stream = None
            controller._poison(exc)
            raise

    def capture_end(graph, *args, **kwargs):
        try:
            result = controller._graph_originals["capture_end"](graph, *args, **kwargs)
            controller._capture_end(graph)
            return result
        except BaseException as exc:
            controller._active_graph = controller._capture_stream = None
            controller._poison(exc)
            raise

    cls.__init__, cls.all_gather, cls.check = initialize, gather, check
    # Older/fake communicators need not expose these optional eager operations.
    def wrap_exchange(name, original):
        def exchange(comm, *args, **kwargs):
            return controller.nccl_exchange(comm, name, original, *args, **kwargs)
        return exchange
    for name in ("exchange", "exchange_all"):
        original = getattr(cls, name, None)
        if original is not None:
            setattr(cls, name, wrap_exchange(name, original))
    if mode == "rocenante":
        graph_cls.capture_begin, graph_cls.capture_end = capture_begin, capture_end
        graph_cls.replay = lambda graph: controller._replay(graph)
    _INSTALLED = controller
    return controller
