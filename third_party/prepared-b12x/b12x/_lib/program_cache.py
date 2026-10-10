"""Argument-key memoization with explicit production-program reclamation."""
from __future__ import annotations

import functools
import gc
import weakref
from collections import namedtuple
from threading import RLock
from contextlib import nullcontext

from .compile_plan import (
    CompiledCuTeProgram, _RESIDENT_PROGRAMS, evict_unretained_triton, program_keys,
    retained_program_keys,
)

_CACHES = weakref.WeakSet()
_MAPPING_CACHES = []


def register_program_cache(cache, *, mirrors=(), lock=None):
    if not any(existing is cache for existing, _, _ in _MAPPING_CACHES):
        _MAPPING_CACHES.append((cache, tuple(mirrors), lock))
_CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")


class program_cache:
    def __init__(self, function):
        functools.update_wrapper(self, function)
        self._function = function
        self._values = {}
        self._lock = RLock()
        self._hits = self._misses = 0
        _CACHES.add(self)

    def __call__(self, *args, **kwargs):
        key = functools._make_key(args, kwargs, typed=False)
        with self._lock:
            if key in self._values:
                self._hits += 1
                return self._values[key]
            self._misses += 1
        value = self._function(*args, **kwargs)
        with self._lock:
            self._values[key] = value
        return value

    def cache_info(self):
        with self._lock:
            return _CacheInfo(self._hits, self._misses, None, len(self._values))

    def cache_clear(self):
        with self._lock:
            self._values.clear()
            self._hits = self._misses = 0

    def evict_unretained(self, keep):
        with self._lock:
            obsolete = [key for key, value in self._values.items()
                        if not frozenset(program_keys(value)) <= keep]
            for key in obsolete:
                del self._values[key]
            return len(obsolete)


def evict_unretained(keep):
    keep = frozenset(keep) | retained_program_keys()
    removed = sum(cache.evict_unretained(keep) for cache in tuple(_CACHES))
    for cache, mirrors, lock in _MAPPING_CACHES:
        with lock if lock is not None else nullcontext():
            obsolete = [key for key, value in cache.items()
                        if not frozenset(program_keys(value)) <= keep]
            for key in obsolete:
                del cache[key]
                for mirror in mirrors:
                    mirror.pop(key, None)
            removed += len(obsolete)
    return removed + evict_unretained_triton(keep)


def _check(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA call failed: {error}")
    return values[0] if len(values) == 1 else values


def stack_limit_bytes():
    """The per-thread local memory (stack) limit of the current CUDA device."""
    from cuda.bindings import runtime
    return int(_check(runtime.cudaDeviceGetLimit(runtime.cudaLimit.cudaLimitStackSize)))


def _local_memory_bytes(executable):
    """The largest static local memory footprint among an executable's loaded kernels."""
    from cuda.bindings import driver, runtime
    from triton.compiler.compiler import CompiledKernel
    attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    if isinstance(executable, CompiledKernel):
        if executable.function is None:
            return 0
        return int(_check(driver.cuFuncGetAttribute(attribute, driver.CUfunction(executable.function))))
    if isinstance(executable, CompiledCuTeProgram):
        executable = executable._executable
    module = getattr(executable, "jit_module", None)
    if module is None:
        return 0
    device = driver.CUdevice(_check(runtime.cudaGetDevice()))
    kernels = [entry.kernel for entry in getattr(module, "cuda_modules", ())]
    for library in getattr(module, "cuda_library", ()):
        count = _check(runtime.cudaLibraryGetKernelCount(library))
        kernels.extend(_check(runtime.cudaLibraryEnumerateKernels(count, library)))
    return max((
        int(_check(driver.cuKernelGetAttribute(attribute, driver.CUkernel(int(kernel)), device)))
        for kernel in kernels
    ), default=0)


def reclaim_device_memory(keep, *, stack_limit):
    """Return the device memory that losing candidates leave behind after eviction.

    An evicted CuTe executable unloads its CUDA library when it is collected,
    but the DSL's executor objects sit in reference cycles, so the collection
    runs here rather than at an arbitrary later allocation. The CUDA driver
    also grows the context's per-thread local memory (stack) to the largest
    static requirement of any kernel launched so far and never shrinks it on
    its own, so one losing candidate that spills registers keeps hundreds of
    MiB allocated. With `stack_limit`, the limit before the preparation, the
    stack is reset and set to the larger of that limit and the retained
    kernels' largest local size: the state a preparation that launches only
    the retained programs reaches. `stack_limit` is None on a host device,
    where only the collection runs.
    """
    gc.collect()
    if stack_limit is None:
        return
    from cuda.bindings import runtime
    limit = stack_limit
    for program in keep:
        executable = _RESIDENT_PROGRAMS.get(program)
        if executable is not None:
            limit = max(limit, _local_memory_bytes(executable))
    for value in (0, limit):
        _check(runtime.cudaDeviceSetLimit(runtime.cudaLimit.cudaLimitStackSize, value))
