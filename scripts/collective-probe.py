#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Value-check TensorFold's actual NCCL all-gather API on four ring ranks."""
import ctypes
import hashlib
import json
import os
import importlib.util
import subprocess
from pathlib import Path
import time

import torch
from tensorfold.cuda.comm import NCCL


class DlInfo(ctypes.Structure):
    _fields_ = [("fname", ctypes.c_char_p), ("base", ctypes.c_void_p), ("sname", ctypes.c_char_p), ("saddr", ctypes.c_void_p)]


def resolved_library():
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    libdl.dladdr.restype = ctypes.c_int
    info = DlInfo()
    if not libdl.dladdr(ctypes.cast(ctypes.CDLL(None).ncclAllGather, ctypes.c_void_p), ctypes.byref(info)):
        raise RuntimeError("Cannot resolve ncclAllGather")
    return str(Path(info.fname.decode()).resolve())


def counter_snapshot():
    return json.loads(subprocess.check_output(["python3", "/validation/capture-function-counters.py"], text=True))


def counter_delta(before, after, send_bytes, repeats):
    gate = max(65536, send_bytes * repeats // 4)
    rows = {}
    for hca, prior in before["devices"].items():
        current = after["devices"][hca]
        if prior["pci_address"] != current["pci_address"] or prior["netdev"] != current["netdev"]:
            raise RuntimeError(f"HCA identity changed: {hca}")
        deltas = {key: current["function_rdma"][key] - value for key, value in prior["function_rdma"].items()}
        errors = {key: current["hw_requests_and_errors"].get(key, 0) - value for key, value in prior["hw_requests_and_errors"].items() if key not in ("rx_write_requests", "rx_read_requests", "rx_atomic_requests")}
        if any(value < 0 for value in deltas.values()) or any(value < 0 for value in errors.values()):
            raise RuntimeError(f"Counter reset: {hca}")
        rows[hca] = {"netdev": prior["netdev"], "pci_address": prior["pci_address"], "deltas": deltas, "errors": errors, "minimum_payload_bytes": gate, "payload_passed": max(deltas["rx_vport_rdma_unicast_bytes"], deltas["tx_vport_rdma_unicast_bytes"]) >= gate, "errors_passed": all(value == 0 for value in errors.values())}
    return rows


def main():
    rank = int(os.environ["RANK"])
    required = json.loads(os.environ["SPARK_RING_REQUIRED_HCAS"])
    if len(required) != 4 or len(set(required)) != 4:
        raise RuntimeError("Exactly four configured HCA functions are required")
    path = Path(os.environ["TF_NCCL_LIB"]).resolve()
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != os.environ["EXPECTED_NCCL_SHA256"]:
        raise RuntimeError("Library SHA256 mismatch")
    if resolved_library() != str(path):
        raise RuntimeError("Global ncclAllGather symbol resolves to another library")
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libnccl.so" in line})
    if any(Path(other).resolve() != path for other in mapped):
        raise RuntimeError(f"Multiple NCCL libraries mapped: {mapped}")
    torch.cuda.set_device(0)
    spec = importlib.util.spec_from_file_location('spark_rocenante_probe', Path(__file__).with_name('rocenante-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    from rocenante_adapter import install
    controller = install(json.loads(os.environ['SPARK_RING_ROCENANTE_SETTINGS']),
                         os.environ['SPARK_RING_VENDOR_ROOT'], os.environ['SPARK_RING_VENDOR_MANIFEST_SHA256'], mode='nccl')
    comm = NCCL(rank, 4, os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]))
    warm = torch.full((262144,), rank, dtype=torch.int32, device='cuda')
    warm_out = torch.empty(warm.numel() * 4, dtype=warm.dtype, device='cuda')
    comm.all_gather(warm, warm_out)
    torch.cuda.synchronize()
    diagnostic = probe.diagnostic_handshake({'rank': rank, 'pid': os.getpid(), 'required_hcas': required,
                                             'library_sha256': actual_hash})
    del warm, warm_out
    before_start = counter_snapshot()
    if set(before_start["devices"]) != set(required):
        raise RuntimeError("Function counter capture does not cover exactly the configured four HCAs")
    results = []
    report = {"rank": rank, "world": 4, "required_hcas": required, "library": str(path), "library_sha256": actual_hash, "mapped_nccl": mapped, "resolved_nccl": resolved_library(), "correctness_passed": False, "cuda_graph_passed": False, "payload_all_hcas_passed": False, "results": results, "qp_diagnostics": diagnostic, "adapter_control": controller.report()}
    report_path = Path(f"/validation/rank-{rank}.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    sizes = [16384, 65536, 1048576, 16777216, 67108864]
    def fill(source, size, epoch):
        raw = torch.frombuffer(bytearray(probe.raw_payload(size, rank, epoch)), dtype=torch.uint8).to('cuda')
        source.view(torch.uint8).copy_(raw)
    def check(output, size, epoch, label):
        expected = torch.frombuffer(bytearray(b''.join(probe.raw_payload(size, peer, epoch) for peer in range(4))), dtype=torch.uint8).to('cuda')
        if not torch.equal(output.view(torch.uint8), expected):
            raise RuntimeError('Full-byte NCCL gather differs: ' + label)
    for dtype in [torch.float32, torch.bfloat16, torch.int32, torch.int64]:
        for size in sizes:
            count = size // torch.empty((), dtype=dtype).element_size()
            for inplace in [False, True]:
                output = torch.empty(4 * count, device="cuda", dtype=dtype)
                source = output[rank * count:(rank + 1) * count] if inplace else torch.empty(count, device="cuda", dtype=dtype)
                fill(source, size, 0)
                for _ in range(3):
                    comm.all_gather(source, output)
                torch.cuda.synchronize()
                label = f"{dtype}/{size}/{inplace}"
                comm.ready("before/" + label, every=10, timeout=120)
                repeats = 32 if size >= 1048576 else 100
                before = counter_snapshot()
                comm.ready("counter-start/" + label, every=10, timeout=120)
                start = time.perf_counter()
                for _ in range(repeats):
                    comm.all_gather(source, output)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000 / repeats
                check(output, size, 0, label)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    comm.all_gather(source, output)
                fill(source, size, 1)
                graph.replay()
                torch.cuda.synchronize()
                check(output, size, 1, label + '/graph')
                comm.ready("after/" + label, every=10, timeout=120)
                after = counter_snapshot()
                traffic = counter_delta(before, after, size, repeats)
                row = {"dtype": str(dtype), "send_bytes": size, "inplace": inplace, "repeats": repeats, "milliseconds": elapsed, "correct": True, "cuda_graph_correct": True, "payload_all_hcas_passed": all(x["payload_passed"] and x["errors_passed"] for x in traffic.values()), "function_traffic": traffic, "counter_before": before, "counter_after": after}
                results.append(row)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k not in ("counter_before", "counter_after")}), flush=True)
                graph.reset()
                del graph, source, output
    comm.ready("probe-complete", every=10, timeout=120)
    report.update(correctness_passed=True, cuda_graph_passed=True, payload_all_hcas_passed=all(row["payload_all_hcas_passed"] for row in results))
    Path(f"/validation/rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    import gc
    gc.collect()
    controller.close(graphs_released=True)
    print(f"PASS rank={rank} tests={len(results)} all_hca_payload={report['payload_all_hcas_passed']}", flush=True)


if __name__ == "__main__":
    main()
