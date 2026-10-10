#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Full-byte GPU/RDMA qualification, with no import-time CUDA or Torch.

The unchanged NCCL oracle runs outside native traffic measurement windows.
Timing includes Python enqueue and final synchronization, not model tokens/s.
"""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import time

DTYPES = ('torch.float32', 'torch.bfloat16', 'torch.int32', 'torch.int64')
NATIVE_SIZES = (8, 32, 896, 7168, 16384, 32768, 65536, 131072,
                262144, 524288, 1048576, 4194304, 16777216)
FALLBACK_SIZE = 33554432
CASES = {(dtype, size, inplace) for dtype in DTYPES
         for size in (*NATIVE_SIZES, FALLBACK_SIZE) for inplace in (False, True)}
ERROR_FIELDS = {'out_of_buffer', 'req_cqe_error', 'resp_cqe_error',
                'req_transport_retries_exceeded', 'req_rnr_retries_exceeded',
                'packet_seq_err', 'local_ack_timeout_err'}


def case_sizes(settings):
    return tuple(sorted(set((*NATIVE_SIZES, FALLBACK_SIZE,
                             (settings['min_gather_bytes'] + 15) // 16 * 16,
                             settings['max_gather_bytes']))))


def raw_payload(size, rank, epoch):
    """All BF16 patterns and explicit FP32 NaNs, signed zero, and subnormals."""
    if size < 0 or rank not in range(4) or epoch < 0:
        raise ValueError('Invalid raw-payload geometry')
    special = struct.pack('<16I', 0, 0x80000000, 0x7f800000, 0xff800000,
                          0x7fc00001, 0x7f800001, 0xffc01234, 0xff800001,
                          1, 0x80000001, 0x007fffff, 0x00800000,
                          0x3f800000, 0xbf800000, 0x7fffffff, 0xffffffff)
    block = special + struct.pack('<65536H', *range(65536))
    mask = (rank * 59 + epoch * 23) % 256
    block = block.translate(bytes(value ^ mask for value in range(256)))
    return (block * ((size + len(block) - 1) // len(block)))[:size]


def path_delta(before, after, rank, hcas, *, require_both=True):
    """Require payload on both origin QPs to each of three peers."""
    keys = lambda row: (row['peer_rank'], row['path_index'])
    expected = {(peer, path) for peer in range(4) if peer != rank for path in range(2)}
    old, new = {keys(row): row for row in before}, {keys(row): row for row in after}
    if set(old) != expected or set(new) != expected or len(before) != 6 or len(after) != 6:
        raise RuntimeError('Native origin-QP inventory is not three peers times two paths')
    rows = []
    for key in sorted(expected):
        prior, current = old[key], new[key]
        fields = ('device', 'qp_number', 'remote_qp_number', 'local_hca_index', 'remote_hca_index', 'physical_hops')
        if any(prior.get(field) != current.get(field) for field in fields):
            raise RuntimeError('Native QP identity changed during measurement')
        delta = {field: current[field] - prior[field] for field in
                 ('payload_writes', 'payload_bytes', 'physical_hop_payload_bytes',
                  'flag_writes', 'send_completions', 'completion_errors')}
        if any(value < 0 for value in delta.values()):
            raise RuntimeError('Native QP counter reset')
        passed = delta['payload_bytes'] > 0 and delta['payload_writes'] > 0 and delta['completion_errors'] == 0
        rows.append({**{field: current[field] for field in ('peer_rank', 'path_index', *fields)},
                     'deltas': delta, 'payload_passed': passed})
    enough_payload = (all(row['payload_passed'] for row in rows) if require_both else
                      all(any(row['peer_rank'] == peer and row['payload_passed'] for row in rows)
                          for peer in range(4) if peer != rank))
    if ({row['device'] for row in rows} != set(hcas) or not enough_payload
        or any(row['deltas']['completion_errors'] for row in rows)):
        raise RuntimeError('Both native paths and all four HCA functions must carry payload')
    return rows


def function_delta(before, after, size, repeats, *, require_payload=True):
    gate = max(1, size * repeats // 16)
    if set(before['devices']) != set(after['devices']):
        raise RuntimeError('Function counter inventory changed')
    rows = {}
    for hca, old in before['devices'].items():
        new = after['devices'][hca]
        if (old['pci_address'], old['netdev']) != (new['pci_address'], new['netdev']):
            raise RuntimeError('HCA function identity changed')
        delta = {key: new['function_rdma'][key] - value for key, value in old['function_rdma'].items()}
        errors = {key: new['hw_requests_and_errors'].get(key, 0) - value
                  for key, value in old['hw_requests_and_errors'].items()
                  if key not in ('rx_write_requests', 'rx_read_requests', 'rx_atomic_requests')}
        passed = (ERROR_FIELDS.issubset(errors) and all(value == 0 for value in errors.values())
                  and all(value >= 0 for value in delta.values())
                  and delta.get('tx_vport_rdma_unicast_bytes', -1) >= gate
                  and delta.get('rx_vport_rdma_unicast_bytes', -1) >= gate)
        rows[hca] = {'deltas': delta, 'errors': errors, 'minimum_payload_bytes': gate,
                     'errors_passed': ERROR_FIELDS.issubset(errors) and all(value == 0 for value in errors.values()),
                     'netdev': old['netdev'], 'pci_address': old['pci_address'], 'payload_passed': passed}
    if (not all(row['errors_passed'] and all(value >= 0 for value in row['deltas'].values()) for row in rows.values())
        or (require_payload and not all(row['payload_passed'] for row in rows.values()))):
        raise RuntimeError('Native-only window lacks bidirectional function payload or has errors')
    return rows


def native_healthy(data):
    stats = data.get('native_stats', {})
    return (data.get('native_active') is True and data.get('failure') is None
            and stats.get('error_seq') == 0 and stats.get('abort_seq') == 0
            and stats.get('wait_stopped') is False
            and stats.get('ctrl_seq') == stats.get('completed_seq')
            and len(data.get('path_counters', [])) == 6
            and all(row.get('completion_errors') == 0 for row in data['path_counters']))


def diagnostic_handshake(data, timeout=600):
    ready = Path('/validation/probe-ready.json')
    temp = ready.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, ready)
    deadline = time.monotonic() + timeout
    ack = Path('/validation/diagnostics-ack.json')
    while not ack.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError('Host QP ownership diagnostic did not acknowledge probe')
        time.sleep(0.1)
    result = json.loads(ack.read_text())
    if result.get('passed') is not True:
        raise RuntimeError('Host QP ownership diagnostic failed')
    return result


def main():
    import torch
    from tensorfold.cuda.comm import NCCL
    from rocenante_adapter import install
    spec = importlib.util.spec_from_file_location('spark_collective_helpers', Path(__file__).with_name('collective-probe.py'))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    rank = int(os.environ['RANK'])
    required = json.loads(os.environ['SPARK_RING_REQUIRED_HCAS'])
    path = Path(os.environ['TF_NCCL_LIB']).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libnccl.so' in line})
    if digest != os.environ['EXPECTED_NCCL_SHA256'] or helpers.resolved_library() != str(path) or set(mapped) != {str(path)}:
        raise RuntimeError('Selected and actual NCCL library identity differ')
    torch.cuda.set_device(0)
    settings = json.loads(os.environ['SPARK_RING_ROCENANTE_SETTINGS'])
    cache_root = Path('/cache')
    cache_before = {str(p.relative_to(cache_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in cache_root.rglob('*') if p.is_file()}
    ctl = install(settings, os.environ['SPARK_RING_VENDOR_ROOT'], os.environ['SPARK_RING_VENDOR_MANIFEST_SHA256'])
    comm = NCCL(rank, 4, os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']))
    from b12x._lib.compiler import compile_cache_info
    cache_after_initialization = compile_cache_info()
    warm = torch.full((262144,), rank, device='cuda', dtype=torch.int32)
    warm_out = torch.empty(warm.numel() * 4, device='cuda', dtype=warm.dtype)
    ctl.nccl_all_gather(warm, warm_out, comm)
    torch.cuda.synchronize()
    ready = ctl.report(detailed=True)
    diagnostic = diagnostic_handshake({'rank': rank, 'pid': os.getpid(), 'native': ready,
                                       'required_hcas': required, 'library_sha256': digest})
    del warm, warm_out
    report = {'schema': 'tensorfold-rocenante-probe/v1', 'rank': rank, 'world': 4,
              'required_hcas': required, 'library': str(path), 'library_sha256': digest,
              'mapped_nccl': mapped, 'resolved_nccl': helpers.resolved_library(),
              'byte_exact_passed': False, 'cuda_graph_passed': False, 'mixed_size_passed': False,
              'inplace_passed': False, 'payload_all_hcas_passed': False, 'native_health_passed': False,
              'qp_diagnostics': diagnostic, 'cache_layout': os.environ.get('SPARK_RING_CACHE_LAYOUT', 'symmetric'),
              'cache_before': cache_before,
              'compile_cache_after_initialization': cache_after_initialization,
              'results': [], 'extra_alias_results': [], 'mixed_results': [],
              'timing_scope': 'Python enqueue plus final GPU synchronization; oracle outside native counter windows'}
    report_path = Path(f'/validation/rank-{rank}.json')
    def save():report_path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    def source_bytes(source, size, epoch):
        raw = torch.frombuffer(bytearray(raw_payload(size, rank, epoch)), dtype=torch.uint8).to('cuda')
        source.view(torch.uint8).copy_(raw)
    def expected_bytes(size, epoch):
        return torch.frombuffer(bytearray(b''.join(raw_payload(size, peer, epoch) for peer in range(4))),
                                dtype=torch.uint8).to('cuda')
    def check(output, size, epoch, label):
        torch.cuda.synchronize()
        ctl.check_health()
        if not torch.equal(output.view(torch.uint8), expected_bytes(size, epoch)):
            raise RuntimeError('Full-byte output differs: ' + label)
    def timing(call, repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):call()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / repeats
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name.split('.')[1])
        element = torch.empty((), dtype=dtype).element_size()
        for size in case_sizes(settings):
            count = size // element
            for inplace in (False, True):
                output = torch.empty(4 * count, device='cuda', dtype=dtype)
                source = output[rank * count:(rank + 1) * count] if inplace else torch.empty(count, device='cuda', dtype=dtype)
                oracle = torch.empty_like(output)
                source_bytes(source, size, 0)
                label = f'{dtype_name}/{size}/{inplace}'
                repeats = 16 if size >= 1048576 else 128
                oracle_ms = timing(lambda: ctl.nccl_all_gather(source, oracle, comm), repeats)
                check(oracle, size, 0, label + '/NCCL')
                oracle_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(oracle_graph):ctl.nccl_all_gather(source, oracle, comm)
                oracle_graph_ms = timing(oracle_graph.replay, repeats)
                check(oracle, size, 0, label + '/NCCL-graph')
                oracle_graph.reset()
                del oracle_graph
                gc.collect()
                comm.ready('before/' + label, every=10, timeout=180)
                before = helpers.counter_snapshot()
                native_before = ctl.report(detailed=True)
                comm.ready('window/' + label, every=10, timeout=180)
                eager_ms = timing(lambda: comm.all_gather(source, output), repeats)
                check(output, size, 0, label + '/eager')
                if not torch.equal(output.view(torch.uint8), oracle.view(torch.uint8)):
                    raise RuntimeError('Independent NCCL oracle differs')
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):comm.all_gather(source, output)
                graph_ms = timing(graph.replay, repeats)
                check(output, size, 0, label + '/graph-timing')
                for epoch in (1, 2, 3):
                    source_bytes(source, size, epoch)
                    graph.replay()
                    check(output, size, epoch, label + '/changing-graph')
                comm.ready('after/' + label, every=10, timeout=180)
                after = helpers.counter_snapshot()
                native_after = ctl.report(detailed=True)
                admitted = settings['min_gather_bytes'] <= size <= settings['max_gather_bytes']
                row = {'dtype': dtype_name, 'send_bytes': size, 'inplace': inplace, 'repeats': repeats,
                       'native_admitted': admitted, 'correct': True, 'cuda_graph_correct': True,
                       'oracle_ms': oracle_ms, 'oracle_graph_ms': oracle_graph_ms,
                       'eager_ms': eager_ms, 'graph_ms': graph_ms,
                       'counter_before': before, 'counter_after': after,
                       'native_before': native_before, 'native_after': native_after}
                if admitted:
                    # One padded16-byte pack has no second payload stripe.
                    # Tiny collectives still check every byte and all error counters.
                    payload_required = size > 16
                    row['payload_gate_required'] = payload_required
                    row['function_traffic'] = function_delta(before, after, size, repeats * 2, require_payload=payload_required)
                    row['native_path_traffic'] = path_delta(native_before['path_counters'], native_after['path_counters'], rank, required, require_both=payload_required)
                    row['payload_all_hcas_passed'] = True if payload_required else None
                    if not payload_required:row['payload_gate_exception'] = 'one padded16-byte pack; path1 may carry flags only'
                else:
                    reason = 'below_minimum' if size < settings['min_gather_bytes'] else 'capacity'
                    row['fallback_reason'] = reason
                    row['fallback_reason_verified'] = native_after['fallback_reasons'].get(reason, 0) > native_before['fallback_reasons'].get(reason, 0)
                    if not row['fallback_reason_verified']:raise RuntimeError('Capacity fallback was not observed')
                if not native_healthy(native_after):raise RuntimeError('Native health or sequence completion failed')
                report['results'].append(row)
                save()
                print(json.dumps({k: v for k, v in row.items() if k not in ('counter_before', 'counter_after', 'native_before', 'native_after')}), flush=True)
                graph.reset()
                del graph, source, output, oracle
                gc.collect()
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name.split('.')[1])
        size = min(settings['max_gather_bytes'], max(65536, (settings['min_gather_bytes'] + 15) // 16 * 16))
        count = size // torch.empty((), dtype=dtype).element_size()
        output = torch.empty(count * 4 + 1, dtype=dtype, device='cuda')[1:]
        source = output[1:count + 1]
        source_bytes(source, size, 4)
        comm.all_gather(source, output)
        check(output, size, 4, dtype_name + '/arbitrary-overlap')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):comm.all_gather(source, output)
        for epoch in (5, 6, 7):
            source_bytes(source, size, epoch)
            graph.replay()
            check(output, size, epoch, dtype_name + '/overlap-graph')
        report['extra_alias_results'].append({'dtype': dtype_name, 'send_bytes': size, 'native_admitted': True,
                                            'correct': True, 'cuda_graph_correct': True})
        graph.reset()
        del graph, source, output
        gc.collect()
        save()
    retained = []
    mixed_sizes = [16384, 65536, 1048576]
    if not any(settings['min_gather_bytes'] <= size <= settings['max_gather_bytes'] for size in mixed_sizes):
        mixed_sizes[-1] = settings['max_gather_bytes']
    for size in mixed_sizes:
        source = torch.empty(size // 4, dtype=torch.int32, device='cuda')
        output = torch.empty(source.numel() * 4, dtype=source.dtype, device='cuda')
        source_bytes(source, size, 8)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):comm.all_gather(source, output)
        retained.append((size, source, output, graph))
    stream = torch.cuda.Stream()
    for epoch in (9, 10, 11):
        for index, (size, source, output, graph) in enumerate(retained):
            source_bytes(source, size, epoch)
            torch.cuda.synchronize()
            if index % 2:
                with torch.cuda.stream(stream):graph.replay()
            else:graph.replay()
            check(output, size, epoch, 'mixed-graph')
            source_bytes(source, size, epoch + 20)
            with torch.cuda.stream(stream):comm.all_gather(source, output)
            check(output, size, epoch + 20, 'mixed-eager')
            ctl.nccl_all_gather(source, output, comm)
            check(output, size, epoch + 20, 'mixed-NCCL')
            report['mixed_results'].append({'size': size, 'epoch': epoch, 'correct': True,
                                             'graph_eager_nccl_ordering_passed': True})
    for size, source, output, graph in retained:graph.reset()
    del retained, size, source, output, graph, stream
    gc.collect()
    torch.cuda.synchronize()
    ctl.check_health()
    final = ctl.report(detailed=True)
    report['compile_cache_final'] = compile_cache_info()
    from b12x.comm.roce import _proxy
    proxy = _proxy.load()
    proxy_path = Path(proxy._name).resolve()
    report.update(byte_exact_passed=True, cuda_graph_passed=True, mixed_size_passed=True,
                  inplace_passed=True, payload_all_hcas_passed=True, native_health_passed=native_healthy(final),
                  native_final=final, native_proxy={'path': str(proxy_path), 'sha256': hashlib.sha256(proxy_path.read_bytes()).hexdigest(),
                  'abi_version': int(proxy.roce_abi_version()), 'blob_bytes': int(proxy.roce_blob_bytes())})
    if not report['native_health_passed'] or report['native_proxy']['abi_version'] != 6 or report['native_proxy']['blob_bytes'] != 456:
        raise RuntimeError('Native final health/ABI failed')
    save()
    comm.ready('probe-complete', every=10, timeout=180)
    ctl.close(graphs_released=True)
    report['teardown_passed'] = True
    save()
    print(f'PASS rank={rank} full_byte_cases={len(report["results"])}', flush=True)


if __name__ == '__main__':main()
