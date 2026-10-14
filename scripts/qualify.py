#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plan or explicitly qualify four-rank NCCL232 or prepared RoCEnante gathers.

This creates only owned, uniquely labelled test containers and task artifact
folders. No images/packages are installed and no host networking is configured.
All sixteen function counters, changing-data CUDA graphs, actual data values,
selected library/image/transport identity and unchanged host configuration must
pass before the canonical serving receipt becomes qualified.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import ipaddress
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid

BASE_COMMIT = '12df1a11afad322be5a204a2db890161cbf8131d'
PATCH_SHA256 = 'a0deca2f21b8cad56bc68a5f5468936b7a6b894db160cfa1d62a6a3d59358836'
PATCHED_TREE = 'e4ae48e878d15a5ba52d91a18c6b5e79b50a7db0'
CONTAINER_LIBRARY = '/opt/spark-ring/lib/libnccl.so.2'
CASES = {(dtype, size, inplace) for dtype in ('torch.float32', 'torch.bfloat16', 'torch.int32', 'torch.int64')
         for size in (16384, 65536, 1048576, 16777216, 67108864) for inplace in (False, True)}
ERROR_FIELDS = {'out_of_buffer', 'req_cqe_error', 'resp_cqe_error', 'req_transport_retries_exceeded',
                'req_rnr_retries_exceeded', 'packet_seq_err', 'local_ack_timeout_err'}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def flatten_qps(rows):
    """iproute2 versions expose both flat and nested RDMA resource objects."""
    output = []
    for row in rows:
        if 'qp' in row:
            output.extend({**{k: v for k, v in row.items() if k != 'qp'}, **q} for q in row['qp'])
        else:
            output.append(row)
    return output


def qp_hca(row):
    return str(row.get('ifname', row.get('dev', row.get('link', '')))).split('/')[0]


def qp_ownership(rows, hcas, owned_pids, marker_pids, native_paths=()):
    rows = flatten_qps(rows)
    selected = [r for r in rows if qp_hca(r) in hcas]
    owned = [r for r in selected if r.get('pid') in owned_pids]
    foreign = [r for r in selected if r.get('pid') not in set(owned_pids) | set(marker_pids)
               and not (r.get('pid') is None and r.get('comm') == 'ib_core' and r.get('type') == 'GSI')]
    pairs = {(qp_hca(r), int(r['lqpn'])) for r in owned if 'lqpn' in r}
    native_pairs = {(r['device'], r['qp_number']) for r in native_paths}
    if foreign or {qp_hca(r) for r in owned} != set(hcas) or not native_pairs.issubset(pairs):
        raise RuntimeError('Unowned QPs, missing HCA coverage, or native QP/PID identity mismatch')
    return {'passed': True, 'owned_pids': sorted(owned_pids), 'allowed_marker_pids': sorted(marker_pids),
            'owned_qps': owned, 'native_origin_qp_pairs': sorted(native_pairs), 'required_hcas': hcas}


def hardware_mesh_delta(before, after):
    """Use the network owner's exact eight-rule hardware-counter guard."""
    from network import hardware_payload_delta
    return hardware_payload_delta(before, after)


def worker(payload, action):
    """Root worker delivered over a literal sudo -s heredoc; no remote install."""
    if os.geteuid() != 0:
        raise RuntimeError('Qualification host actions require a sudo -n -s shell')
    node, rank = payload['node'], payload['rank']
    recipe, run_id = payload['recipe_id'], payload['run_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', recipe) or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise RuntimeError('Unsafe qualification identity')
    case = Path('/var/tmp/spark-ring-qualification') / recipe / run_id
    name = payload['container_name']
    labels = {'org.spark-ring.recipe': recipe, 'org.spark-ring.qualification': run_id}

    def command(args, check=True, timeout=120):
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if check and p.returncode:
            raise RuntimeError(f'{shlex.join(args)} failed: {p.stderr.strip()}')
        return p

    def event(action_name, **extra):
        if case.is_dir():
            with (case / 'changes.jsonl').open('a') as f:
                f.write(json.dumps({'utc': utc(), 'action': action_name, **extra}) + '\n')
            os.chmod(case / 'changes.jsonl', 0o600)

    def mutate(args, timeout=120):
        event('command-start', command=shlex.join(args))
        p = command(args, check=False, timeout=timeout)
        event('command-finish', command=shlex.join(args), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        if p.returncode:
            raise RuntimeError(f'{shlex.join(args)} failed: {p.stderr.strip()}')
        return p

    def normalized_routes(family):
        rows = json.loads(command(['ip', '-j', family, 'route', 'show', 'table', 'all']).stdout)
        return sorted(json.dumps({k: v for k, v in row.items() if k not in ('expires', 'lastuse', 'used', 'cache')}, sort_keys=True)
                      for row in rows)

    def snapshot():
        files = {}
        for base in ('/etc/netplan', '/etc/NetworkManager/system-connections', '/run/NetworkManager/system-connections'):
            for p in Path(base).rglob('*'):
                if p.is_file() and not p.is_symlink():
                    files[str(p)] = {'sha256': sha(p), 'mode': p.stat().st_mode & 0o777}
        addresses = json.loads(command(['ip', '-j', 'address', 'show']).stdout)
        addresses = [{'interface': p['ifname'], 'mac': p.get('address'), 'mtu': p.get('mtu'),
                      'up': 'UP' in p.get('flags', []),
                      'addresses': sorted(json.dumps({k: v for k, v in a.items() if k not in ('valid_life_time', 'preferred_life_time')}, sort_keys=True)
                                          for a in p.get('addr_info', []))} for p in addresses]
        controls = [Path('/proc/sys/net/ipv4/ip_forward')]
        controls += list(Path('/proc/sys/net/ipv4/conf').glob('*/rp_filter'))
        controls += list(Path('/proc/sys/net/ipv4/conf').glob('*/forwarding'))
        controls += list(Path('/proc/sys/net/ipv6/conf').glob('*/forwarding'))
        firewall = {}
        for tool in ('iptables-save', 'ip6tables-save'):
            text = '\n'.join(re.sub(r'\[\d+:\d+\]', '[COUNTERS]', line)
                             for line in command([tool]).stdout.splitlines() if not line.startswith('#'))
            # Record only a comparison hash; do not publish private firewall policy.
            firewall[tool + '_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        return {'files': files, 'addresses': sorted(addresses, key=lambda p: p['interface']),
                'routes4': normalized_routes('-4'), 'routes6': normalized_routes('-6'),
                'firewall': firewall, 'sysctls': {str(p): p.read_text().strip() for p in controls}}

    def existing_container():
        p = command(['docker', 'container', 'inspect', name], check=False)
        if p.returncode:
            # A daemon/connectivity failure is not evidence that a container is absent.
            if not command(['docker', 'info', '--format', '{{.ServerVersion}}'], check=False).returncode:
                return None
            raise RuntimeError('Cannot inspect Docker daemon')
        row = json.loads(p.stdout)[0]
        if any(row['Config'].get('Labels', {}).get(k) != v for k, v in labels.items()):
            raise RuntimeError('Container name belongs to another owner; refusing to act')
        return row

    def own_case():
        if case.is_symlink() or not case.is_dir():
            raise RuntimeError('Qualification task directory is absent or a symlink')
        owner = json.loads((case / 'owner.json').read_text())
        if owner != {'recipe_id': recipe, 'run_id': run_id, 'rank': rank}:
            raise RuntimeError('Qualification directory belongs to another invocation')

    if action == 'preflight':
        if case.exists() or case.is_symlink():
            raise RuntimeError('Qualification task directory already exists')
        if existing_container() is not None:
            raise RuntimeError('Qualification container already exists')
        if command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).stdout.strip():
            raise RuntimeError('Active GPU compute processes; stop owned serving workloads before qualification')
        image = json.loads(command(['docker', 'image', 'inspect', payload['image']]).stdout)[0]
        if image.get('Architecture') != 'arm64':
            raise RuntimeError('Pinned image must already be available for ARM64; this script never pulls')
        if payload['image'] not in image.get('RepoDigests', []):
            raise RuntimeError('Image digest does not match the pinned image')
        if sha(payload['library']) != payload['library_sha256']:
            raise RuntimeError('Shared library differs from operator SHA256')
        for filename, digest in payload['source_sha256'].items():
            if sha(Path(payload['script_directory']) / filename) != digest:
                raise RuntimeError('Shared qualification source differs from operator source')
        import sys
        sys.path.insert(0,payload['script_directory'])
        import engine_binding
        engine_binding.identity(payload.get('engine_config', {}), verify=True)
        if not Path('/usr/sbin/ethtool').is_file() or not Path('/lib/aarch64-linux-gnu/libmnl.so.0').is_file():
            raise RuntimeError('Host ethtool/libmnl bind dependencies are missing; no packages are installed here')
        interfaces = {p['ifname']: p for p in json.loads(command(['ip', '-j', 'address', 'show']).stdout)}
        management = interfaces.get(node['management_interface'], {})
        if node['management_address'] not in [a['local'] for a in management.get('addr_info', []) if a['family'] == 'inet']:
            raise RuntimeError('Management address/interface is absent')
        roots, gids, ports = {}, {}, []
        address_range = ipaddress.IPv4Network(payload['fabric_address_range'])
        for f in node['fabric']:
            d = Path('/sys/class/infiniband') / f['hca']
            associated = sorted(p.name for p in (d / 'device/net').iterdir())
            if associated != [f['interface']]:
                raise RuntimeError('HCA/netdev association differs: ' + f['hca'])
            p = interfaces.get(f['interface'], {})
            assigned = [a['local'] for a in p.get('addr_info', []) if a['family'] == 'inet']
            if assigned != [f['address']] or p.get('mtu') != payload['mtu'] or 'UP' not in p.get('flags', []):
                raise RuntimeError('Fabric address/MTU/UP mismatch: ' + f['interface'])
            state = (d / 'ports/1/state').read_text().strip()
            if 'ACTIVE' not in state:
                raise RuntimeError('RDMA port not ACTIVE: ' + f['hca'])
            pci = (d / 'device').resolve().name
            domain = pci.split(':')[0]
            roots.setdefault(f['rail'], set()).add(domain)
            selected = []
            for path in sorted((d / 'ports/1/gids').iterdir(), key=lambda p: int(p.name)):
                # Unallocated mlx5 slots are sysfs files that can return EINVAL.
                try:
                    index, value = path.name, ipaddress.IPv6Address(path.read_text().strip())
                    typ = (d / 'ports/1/gid_attrs/types' / index).read_text().strip()
                    dev = (d / 'ports/1/gid_attrs/ndevs' / index).read_text().strip()
                except (OSError, ValueError):
                    continue
                if typ == 'RoCE v2' and value.ipv4_mapped and value.ipv4_mapped in address_range:
                    if dev != f['interface'] or str(value.ipv4_mapped) != f['address']:
                        raise RuntimeError('Unexpected IPv4 candidate in fabric range: ' + f['hca'])
                    selected.append(int(index))
            if selected != [3]:
                raise RuntimeError('Exactly one IPv4 RoCE-v2 GID at current prepared index3 required: ' + f['hca'])
            gids[f['hca']] = selected[0]
            ports.append({'hca': f['hca'], 'interface': f['interface'], 'pci_address': pci, 'gid_index': selected[0]})
        if set(roots) != {0, 1} or any(len(domains) != 1 for domains in roots.values()) or roots[0] == roots[1]:
            raise RuntimeError('Configured rails must identify two separate PCI domains')
        qp = json.loads(command(['rdma', '-j', 'resource', 'show', 'qp']).stdout)
        selected_hcas = set(gids)
        if any(qp_hca(q) in selected_hcas and q.get('pid') not in payload['allowed_marker_pids']
               and not (q.get('pid') is None and q.get('comm') == 'ib_core' and q.get('type') == 'GSI')
               for q in flatten_qps(qp)):
            raise RuntimeError('Another userspace QP is active on the selected fabric functions')
        with socket.socket() as test:
            test.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test.bind((node['management_address'], payload['master_port']))
        env = dict(os.environ, SPARK_RING_REQUIRED_HCAS=json.dumps(list(gids)))
        counts = subprocess.run(['python3', str(Path(payload['script_directory']) / 'capture-function-counters.py')],
                                env=env, text=True, capture_output=True, check=True, timeout=60)
        counter = json.loads(counts.stdout)
        return {'utc': utc(), 'rank': rank, 'ports': ports, 'image_id': image['Id'],
                'library_sha256': payload['library_sha256'], 'counter_devices': sorted(counter['devices']), 'snapshot': snapshot()}
    if action == 'stage':
        for parent in (case.parent.parent, case.parent):
            if parent.is_symlink():
                raise RuntimeError('Refusing symlink qualification parent')
            parent.mkdir(mode=0o755, exist_ok=True)
        if case.exists() or case.is_symlink():
            raise RuntimeError('Refusing existing qualification run directory')
        case.mkdir(mode=0o755)
        (case / 'owner.json').write_text(json.dumps({'recipe_id': recipe, 'run_id': run_id, 'rank': rank}) + '\n')
        os.chmod(case / 'owner.json', 0o600)
        event('created-task-directory', path=str(case), rollback='Remove only this invocation directory after preserving evidence')
        for filename, digest in payload['source_sha256'].items():
            source = Path(payload['script_directory']) / filename
            if sha(source) != digest:
                raise RuntimeError('Shared qualification source changed after preflight')
            data = source.read_bytes()
            (case / filename).write_bytes(data)
            os.chmod(case / filename, 0o644)
            event('copied-source', path=str(case / filename), sha256=digest)
        (case / 'cache').mkdir(mode=0o755)
        event('created-private-compile-cache', path=str(case / 'cache'))
        if payload.get('cache_seed_run_id') is not None:
            seed_id = payload['cache_seed_run_id']
            if rank != 0 or not re.fullmatch(r'[A-Za-z0-9_-]+', seed_id):
                raise RuntimeError('Only rank0 may retain this invocation first-pass cache')
            seed = case.parent / seed_id
            if json.loads((seed / 'owner.json').read_text()) != {'recipe_id': recipe, 'run_id': seed_id, 'rank': rank}:
                raise RuntimeError('Asymmetric cache seed is not owned by this invocation')
            if any(p.is_symlink() for p in (seed / 'cache').rglob('*')):
                raise RuntimeError('Refusing symlink cache seed')
            shutil.copytree(seed / 'cache', case / 'cache', dirs_exist_ok=True)
            event('copied-owned-retained-cache', source=str(seed / 'cache'), destination=str(case / 'cache'))
        return {'utc': utc(), 'rank': rank, 'directory': str(case)}
    own_case()
    if action == 'create':
        if sha(payload['library']) != payload['library_sha256']:
            raise RuntimeError('Library changed after preflight')
        if existing_container() is not None:
            raise RuntimeError('Owned qualification container already exists')
        args = ['docker', 'create', '--name', name, '--gpus', 'all', '--ipc=host', '--network', 'host',
                '--device', '/dev/infiniband', '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
                '--mount', 'type=bind,src=/usr/sbin/ethtool,dst=/usr/sbin/ethtool,readonly',
                '--mount', 'type=bind,src=/lib/aarch64-linux-gnu/libmnl.so.0,dst=/lib/aarch64-linux-gnu/libmnl.so.0,readonly',
                '--mount', f'type=bind,src={case},dst=/validation',
                '--mount', f'type=bind,src={case / "cache"},dst=/cache',
                '--mount', f'type=bind,src={payload["vendor_root"]},dst=/opt/spark-ring/vendor,readonly',
                '--mount', f"type=bind,src={payload['library']},dst={CONTAINER_LIBRARY},readonly"]
        args += payload.get('engine_mounts', [])
        for key, value in labels.items():
            args += ['--label', key + '=' + value]
        for key, value in payload['environment'].items():
            args += ['--env', key + '=' + value]
        args += ['--entrypoint', 'python3', payload['image'], '/validation/' + payload['probe']]
        p = mutate(args)
        return {'utc': utc(), 'rank': rank, 'container_id': p.stdout.strip(), 'container_name': name}
    if action == 'start':
        if existing_container() is None:
            raise RuntimeError('Owned qualification container is absent')
        p = mutate(['docker', 'start', name])
        return {'utc': utc(), 'rank': rank, 'started': p.stdout.strip()}
    if action == 'diagnose':
        deadline = time.monotonic() + min(payload['timeout'], 900)
        ready_path = case / 'probe-ready.json'
        while not ready_path.is_file():
            container = existing_container()
            if container is None or (container['State'].get('Status') != 'created' and not container['State'].get('Running')):
                log = command(['docker', 'logs', name], check=False)
                (case / 'container.log').write_text(log.stdout + log.stderr)
                raise RuntimeError('Probe exited before QP proof: ' + (log.stdout + log.stderr)[-4000:])
            if time.monotonic() > deadline:
                raise RuntimeError('Probe initialization/QP proof timed out')
            time.sleep(0.2)
        ready = json.loads(ready_path.read_text())
        container = existing_container()
        host_pid = int(container['State']['Pid'])
        owned_pids = {int(p.name) for p in (Path('/proc') / str(host_pid) / 'task').iterdir()}
        rows = json.loads(command(['rdma', '-j', 'resource', 'show', 'qp']).stdout)
        native_paths = ready.get('native', {}).get('path_counters', [])
        proof = qp_ownership(rows, json.loads(payload['environment']['SPARK_RING_REQUIRED_HCAS']),
                             owned_pids, payload['allowed_marker_pids'], native_paths)
        if payload['probe'] == 'rocenante-probe.py' and len(native_paths) != 6:
            raise RuntimeError('Prepared origin-QP proof requires exactly six live native paths')
        proof.update(container_id=container['Id'], host_pid=host_pid, probe_ready=ready)
        (case / 'qp-diagnostics.json').write_text(json.dumps(proof, indent=2) + '\n')
        ack = case / 'diagnostics-ack.json'
        temp = ack.with_suffix('.tmp')
        temp.write_text(json.dumps({'passed': True}) + '\n')
        os.replace(temp, ack)
        event('captured-owned-qp-diagnostics', path=str(case / 'qp-diagnostics.json'))
        return proof
    if action == 'finish':
        if existing_container() is None:
            raise RuntimeError('Owned qualification container is absent')
        exit_code = command(['docker', 'wait', name], timeout=payload['timeout']).stdout.strip()
        log = command(['docker', 'logs', name], check=False, timeout=120)
        log_text = log.stdout + log.stderr
        (case / 'container.log').write_text(log_text)
        event('collected-container-output', exit_code=exit_code, log_path=str(case / 'container.log'))
        report_path = case / f'rank-{rank}.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        after = snapshot()
        (case / 'os-after.json').write_text(json.dumps(after, indent=2) + '\n')
        event('saved-read-only-preservation-snapshot', path=str(case / 'os-after.json'))
        return {'utc': utc(), 'rank': rank, 'exit_code': exit_code, 'report': report,
                'log': log_text, 'snapshot': after, 'local_directory': str(case)}
    if action == 'cleanup':
        container = existing_container()
        if container is not None:
            mutate(['docker', 'rm', '-f', name])
        return {'utc': utc(), 'rank': rank, 'owned_container_removed': True, 'audit_directory_retained': str(case)}
    raise RuntimeError('Unknown qualification worker action')


def validate_report(report, rank, hcas, digest):
    if not isinstance(report, dict) or report.get('rank') != rank or report.get('world') != 4:
        return False
    if report.get('library_sha256') != digest or set(report.get('required_hcas', [])) != set(hcas):
        return False
    if report.get('library') != CONTAINER_LIBRARY or report.get('resolved_nccl') != CONTAINER_LIBRARY:
        return False
    if set(report.get('mapped_nccl', [])) != {CONTAINER_LIBRARY}:
        return False
    if not all(report.get(k) is True for k in ('correctness_passed', 'cuda_graph_passed', 'payload_all_hcas_passed')):
        return False
    rows = report.get('results', [])
    if len(rows) != 40 or {(r.get('dtype'), r.get('send_bytes'), r.get('inplace')) for r in rows} != CASES:
        return False
    for row in rows:
        if not all(row.get(k) is True for k in ('correct', 'cuda_graph_correct', 'payload_all_hcas_passed')):
            return False
        traffic = row.get('function_traffic', {})
        if set(traffic) != set(hcas):
            return False
        repeats = 32 if row['send_bytes'] >= 1048576 else 100
        gate = max(65536, row['send_bytes'] * repeats // 4)
        if row.get('repeats') != repeats:
            return False
        for data in traffic.values():
            delta = data.get('deltas', {})
            errors = data.get('errors', {})
            if data.get('minimum_payload_bytes') != gate or data.get('payload_passed') is not True or data.get('errors_passed') is not True:
                return False
            if max(delta.get('rx_vport_rdma_unicast_bytes', -1), delta.get('tx_vport_rdma_unicast_bytes', -1)) < gate:
                return False
            if not ERROR_FIELDS.issubset(errors) or any(v != 0 for v in errors.values()) or any(v < 0 for v in delta.values()):
                return False
    return True


def rocenante_probe_module():
    spec = importlib.util.spec_from_file_location('spark_rocenante_probe', Path(__file__).with_name('rocenante-probe.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_geometry_matches(native, settings):
    """Require requested and actual immutable native geometry, without coercion."""
    blocks = settings.get('native_blocks', 8)
    threads = settings.get('native_threads', 512)
    if (not isinstance(native, dict) or type(blocks) is not int or blocks not in (4, 8, 16)
        or type(threads) is not int or threads != 512):
        return False
    expected = {'native_threads': 512, 'native_blocks': blocks,
                'runtime_native_threads': 512, 'runtime_native_blocks': blocks}
    return all(type(native.get(field)) is int and native[field] == value
               for field, value in expected.items())


def validate_ring_state(state, rank, source_sha):
    """Receipt metadata, not a substitute for the byte-exact physical probe."""
    try:
        integers = ('rank', 'world', 'calls', 'rounds', 'scratch_bytes', 'peak_growth_bytes',
                    'max_scratch_bytes', 'max_growth_bytes')
        if not all(type(state[k]) is int and state[k] >= 0 for k in integers):
            return False
        banks = state['scratch_by_dtype']
        if not isinstance(banks, dict):return False
        total = 0
        widths = {'torch.float32': 4, 'torch.bfloat16': 2, 'torch.int32': 4, 'torch.int64': 8}
        for dtype, bank in banks.items():
            if (dtype not in widths or type(bank['buffers']) is not int or bank['buffers'] != 2
                    or type(bank['elements_each']) is not int or bank['elements_each'] < 0
                    or type(bank['bytes']) is not int
                    or bank['bytes'] != 2 * bank['elements_each'] * widths[dtype]):return False
            total += bank['bytes']
        return (state['enabled'] is True and state['source_sha256'] == source_sha
                and state['route'] == 'adjacent-two-round' and state['rank'] == rank and state['world'] == 4
                and state['neighbors'] == sorted(((rank-1)%4, (rank+1)%4))
                and state['rounds'] == 2 * state['calls']
                and state['max_scratch_bytes'] == 64 << 20 and state['max_growth_bytes'] == 128 << 20
                and total == state['scratch_bytes'] <= 64 << 20
                and state['scratch_bytes'] <= state['peak_growth_bytes'] <= 128 << 20)
    except (KeyError, TypeError, ValueError):return False


def validate_mixed_p2p(report, *, relay_sha=None):
    try:
        mixed=report['mixed_p2p'];rows=mixed['rows'];before=mixed['before'];after=mixed['after']
        valid = (report.get('mixed_p2p_passed') is True and mixed.get('passed') is True
                and len(rows)==3 and [r['epoch'] for r in rows]==[41,42,43]
                and all(r['correct'] is True and type(r['send_bytes_per_tensor']) is int and r['send_bytes_per_tensor']>=65536 for r in rows)
                and after['eager_p2p_calls'].get('exchange_all',0)-before['eager_p2p_calls'].get('exchange_all',0)==3
                and after['eager_native_calls']-before['eager_native_calls']==6
                and after['guarded_replays']-before['guarded_replays']==3)
        if relay_sha is None:return valid
        edges = report['ring_relay_edges']; eb = edges['before']; ea = edges['after']
        states = [x['ring_relay'] for x in (before, after, eb, ea, report['native_final'])]
        return (valid and all(validate_ring_state(s, report['rank'], relay_sha) for s in states)
                and states[1]['calls'] - states[0]['calls'] == 3
                and states[2]['calls'] == states[1]['calls']
                and states[3]['calls'] - states[2]['calls'] == 2
                and states[4]['calls'] == states[3]['calls']
                and report.get('ring_relay_edges_passed') is True and edges.get('passed') is True
                and edges['rows'] == [{'elements': 1, 'dtype': 'float32', 'correct': True},
                                      {'elements': 7, 'dtype': 'bfloat16', 'correct': True}])
    except (KeyError,TypeError,ValueError):
        return False


def validate_rocenante_report(report, rank, hcas, digest, settings):
    probe = rocenante_probe_module()
    if not isinstance(report, dict) or report.get('rank') != rank or report.get('world') != 4:
        return False
    if (report.get('library_sha256') != digest or set(report.get('required_hcas', [])) != set(hcas)
        or report.get('library') != CONTAINER_LIBRARY or report.get('resolved_nccl') != CONTAINER_LIBRARY
        or set(report.get('mapped_nccl', [])) != {CONTAINER_LIBRARY}):
        return False
    flags = ('byte_exact_passed', 'cuda_graph_passed', 'mixed_size_passed', 'inplace_passed',
             'payload_all_hcas_passed', 'native_health_passed', 'teardown_passed')
    if not all(report.get(flag) is True for flag in flags):return False
    expected = {(dtype, size, inplace) for dtype in probe.DTYPES for size in probe.case_sizes(settings) for inplace in (False, True)}
    rows = report.get('results', [])
    if len(rows) != len(expected) or {(r.get('dtype'), r.get('send_bytes'), r.get('inplace')) for r in rows} != expected:return False
    admitted_count = 0
    payload_case_count = 0
    for row in rows:
        if row.get('correct') is not True or row.get('cuda_graph_correct') is not True:return False
        if any(type(row.get(field)) not in (float, int) or row[field] <= 0 for field in ('oracle_ms', 'oracle_graph_ms', 'eager_ms', 'graph_ms')):return False
        admitted = settings['min_gather_bytes'] <= row['send_bytes'] <= settings['max_gather_bytes']
        if row.get('native_admitted') is not admitted:return False
        if admitted:
            admitted_count += 1
            payload_required = row['send_bytes'] > 16
            payload_case_count += int(payload_required)
            try:
                traffic = probe.function_delta(row['counter_before'], row['counter_after'], row['send_bytes'], row['repeats'] * 2, require_payload=payload_required)
                paths = probe.path_delta(row['native_before']['path_counters'], row['native_after']['path_counters'], rank, hcas, require_both=payload_required)
            except (KeyError, ValueError, RuntimeError, TypeError):return False
            if (row.get('function_traffic') != traffic or row.get('native_path_traffic') != paths
                or row.get('payload_gate_required') is not payload_required
                or (payload_required and row.get('payload_all_hcas_passed') is not True)):return False
        else:
            reason = 'below_minimum' if row['send_bytes'] < settings['min_gather_bytes'] else 'capacity'
            if row.get('fallback_reason') != reason or row.get('fallback_reason_verified') is not True:return False
        native_after = row.get('native_after', {})
        if not native_geometry_matches(native_after, settings) or not probe.native_healthy(native_after):return False
    aliases, mixed = report.get('extra_alias_results', []), report.get('mixed_results', [])
    if len(aliases) != 4 or {r.get('dtype') for r in aliases} != set(probe.DTYPES) or not all(r.get('correct') is True and r.get('cuda_graph_correct') is True for r in aliases):return False
    if len(mixed) != 9 or not all(r.get('correct') is True and r.get('graph_eager_nccl_ordering_passed') is True for r in mixed):return False
    native = report.get('native_final', {})
    if (not admitted_count or not payload_case_count or not native_geometry_matches(native, settings)
        or not probe.native_healthy(native) or native.get('selection_cache') != 'disabled'
        or native.get('live_native_graphs') != 0):return False
    if (native.get('adapter_sha256') != sha(Path(__file__).with_name('rocenante_adapter.py'))
        or native.get('vendor_manifest_sha256') != settings['vendor_manifest_sha256']
        or native.get('gid_indices') != [3, 3, 3, 3] or set(native.get('hca_names', [])) != set(hcas)):
        return False
    proxy = report.get('native_proxy', {})
    return (proxy.get('abi_version') == 6 and proxy.get('blob_bytes') == 456
            and bool(re.fullmatch(r'[0-9a-f]{64}', str(proxy.get('sha256', '')))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('plan', 'run'))
    parser.add_argument('--config', help='explicit local configuration JSON')
    parser.add_argument('--plan', action='store_true', help='offline plan alias')
    parser.add_argument('--master-port', type=int, help='qualification TCP port (default bootstrap_port+20)')
    parser.add_argument('--timeout', type=int, default=1800, help='maximum seconds per rank; default 1800')
    parser.add_argument('--transport', choices=('nccl', 'rocenante'), default='nccl',
                        help='fresh NCCL232 control or prepared native qualification (two cache layouts)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--payload', help=argparse.SUPPRESS)
    parser.add_argument('--host-action', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(base64.b64decode(args.payload, validate=True)), args.host_action))); return
    if args.plan:
        if args.action not in (None, 'plan'):
            parser.error('--plan cannot be combined with run')
        args.action = 'plan'
    if not args.action or not args.config:
        parser.error('explicit plan/run action and --config are required')
    sys.dont_write_bytecode = True
    from common import IMAGE, environment, library_path, load_config, ssh_args, state_dir, topology_sha256, transport_environment
    from experiment import prepare_config, rank_settings, vendor_path, experiment_identity
    cfg = prepare_config(args.config, args.transport)
    port = args.master_port if args.master_port is not None else cfg['bootstrap_port'] + 20
    if not 1 <= port <= 65535 or port in (cfg['bootstrap_port'], cfg['api_port']) or not 120 <= args.timeout <= 7200:
        parser.error('Use a distinct valid qualification port and timeout 120..7200 seconds')
    lib = library_path(cfg)
    if args.action == 'plan':
        probe = rocenante_probe_module()
        print(json.dumps({'action': 'plan', 'image': IMAGE, 'library': str(lib),
                          'transport': args.transport, 'vendor': str(vendor_path(cfg)),
                          'hosts': [n['name'] for n in cfg['nodes']], 'topology_sha256': topology_sha256(cfg),
                          'master_port': port, 'tests_per_rank': len(CASES) if args.transport == 'nccl' else len(probe.case_sizes(cfg['experiment'])) * 8,
                          'cache_layouts': ['symmetric', 'asymmetric'] if args.transport == 'rocenante' else ['symmetric'],
                          'required_hcas_by_host': {n['name']: [f['hca'] for f in n['fabric']] for n in cfg['nodes']},
                          'guards': ['all-host idle/preflight before staging or GPU start', 'full-byte independent rank-major expected output and NCCL oracle',
                                     'changing-data CUDA graph replay', 'all16 function-specific payload/error counters',
                                     'all4 RouteFinal HCAs and owned live QPs per rank', 'host routes/files/firewall/forwarding unchanged',
                                     'RoC: six native origin-QPs, all8 hardware diagonal redirect counters, asymmetric filesystem caches',
                                     'fresh library/image/topology/transport receipt before serving'],
                          'remote_artifacts': '/var/tmp/spark-ring-qualification/<recipe_id>/<run_id>',
                          'receipt': str(state_dir(cfg) / ('collective-qualification.json' if args.transport == 'nccl' else 'rocenante-qualification.json'))}, indent=2)); return
    examples = [ipaddress.IPv4Network(s) for s in ('192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24')]
    if any(any(ipaddress.IPv4Address(n['management_address']) in subnet for subnet in examples) for n in cfg['nodes']):
        parser.error('Replace documentation-only management addresses before any host action')
    digest = sha(lib)
    build = json.loads((lib.parent / 'build-receipt.json').read_text())
    expected_build = {'base_commit': BASE_COMMIT, 'patch_sha256': PATCH_SHA256,
                      'patched_tree': PATCHED_TREE, 'library_sha256': digest, 'nccl_get_version': 23203,
                      'native_cpu_test_passed': True, 'native_cpu_library_load_passed': True}
    if any(build.get(k) != v for k, v in expected_build.items()):
        raise RuntimeError('Pinned native build receipt does not match this library; rebuild before qualification')
    loader = build.get('image_loader_proof', {})
    if any(loader.get(k) != v for k, v in {'passed': True, 'image_digest': IMAGE,
                                          'library_sha256': digest, 'nccl_get_version': 23203,
                                          'context_created': False, 'communicator_created': False}.items()):
        raise RuntimeError('Exact pinned image CPU loader proof is missing')
    from network import live_guard
    mesh_before = live_guard(cfg)
    identity = experiment_identity(cfg, digest)
    baseline_path = state_dir(cfg) / 'collective-qualification.json'
    baseline_digest = None
    if args.transport == 'rocenante':
        from cluster import Operator
        baseline = Operator(cfg).qualification(baseline_path, digest)
        if baseline.get('network_boot_ids') != mesh_before['network_boot_ids']:
            raise RuntimeError('NCCL oracle receipt predates the current host boots; rerun it first')
        baseline_digest = sha(baseline_path)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    state_dir(cfg).mkdir(parents=True, exist_ok=True)
    qualification_lock = (state_dir(cfg) / '.qualification.lock').open('a')
    try:
        fcntl.flock(qualification_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another qualification invocation holds this state directory lock')
    case = state_dir(cfg) / 'qualification-runs' / run_id
    case.mkdir(parents=True, exist_ok=False)
    os.chmod(case, 0o700)
    canonical = state_dir(cfg) / ('collective-qualification.json' if args.transport == 'nccl' else 'rocenante-qualification.json')
    name = 'spark-ring-qualify-' + run_id.lower()
    source = Path(__file__).read_text()
    from engine_binding import mounts as engine_mounts, identity as engine_identity
    engine_identity(cfg, verify=True)
    script_directory = Path(cfg.get('runtime_source_root', str(Path(cfg['shared_root']) / 'scripts')))
    source_hashes = {filename: sha(script_directory / filename) for filename in
                     ('collective-probe.py', 'rocenante-probe.py', 'rocenante_adapter.py', 'capture-function-counters.py', 'engine_binding.py', 'ring_exchange.py')}
    relay_sha = (source_hashes['ring_exchange.py'] if
                 (identity.get('engine_source') or {}).get('engine_flags', {}).get('TF_GLM_HC_RING_RELAY') == '1' else None)
    environments, hcas = {}, {}
    payloads = []
    for rank, node in enumerate(cfg['nodes']):
        host_hcas = [f['hca'] for f in node['fabric']]
        env = environment(cfg, rank, digest)
        environments[node['name']] = env
        hcas[node['name']] = host_hcas
        test_env = {**env, 'RANK': str(rank), 'MASTER_PORT': str(port), 'EXPECTED_NCCL_SHA256': digest,
                    'SPARK_RING_REQUIRED_HCAS': json.dumps(host_hcas), 'PYTHONUNBUFFERED': '1',
                    'PYTHONDONTWRITEBYTECODE': '1', 'SPARK_RING_ROCENANTE_MODE': args.transport,
                    'SPARK_RING_ROCENANTE_SETTINGS': json.dumps(rank_settings(cfg, rank, run_id)),
                    'SPARK_RING_VENDOR_ROOT': '/opt/spark-ring/vendor',
                    'SPARK_RING_VENDOR_MANIFEST_SHA256': cfg['experiment']['vendor_manifest_sha256'],
                    'XDG_CACHE_HOME': '/cache', 'B12X_ROCE_CACHE_DIR': '/cache/roce',
                    'B12X_COMPILE_CACHE_DIR': '/cache/b12x',
                    'CUTE_DSL_CACHE_DIR': '/cache/cute', 'SPARK_RING_CACHE_LAYOUT': 'symmetric'}
        payloads.append({'node': node, 'rank': rank, 'recipe_id': cfg['recipe_id'], 'run_id': run_id,
                         'container_name': name, 'image': IMAGE, 'library': str(lib), 'library_sha256': digest,
                         'source_sha256': source_hashes, 'script_directory': str(script_directory),
                         'engine_mounts': engine_mounts(cfg),
                         'engine_config': {k:cfg[k] for k in ('engine_overlay_root','engine_manifest','engine_manifest_sha256','engine_flags') if k in cfg},
                         'fabric_address_range': cfg['fabric_address_range'], 'mtu': cfg['mtu'],
                         'master_port': port, 'timeout': args.timeout, 'environment': test_env,
                         'probe': 'collective-probe.py' if args.transport == 'nccl' else 'rocenante-probe.py',
                         'vendor_root': str(vendor_path(cfg)),
                         'allowed_marker_pids': mesh_before['allowed_marker_pids_by_host'][node['name']]})
    lock = threading.Lock()
    def event(action, **extra):
        with lock:
            with (case / 'events.jsonl').open('a') as f:
                f.write(json.dumps({'utc': utc(), 'action': action, **extra}) + '\n')
    def atomic(path, value):
        temp = path.with_name(path.name + '.' + run_id + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        os.chmod(temp, 0o600)
        os.replace(temp, path)
        event('wrote-receipt', path=str(path), qualified=value.get('qualified'))
    summary = {'schema': 'spark-ring-tensorfold-qualification/v2', 'generated_utc': utc(), 'run_id': run_id,
               'qualified': False, 'correctness_passed': False, 'payload_all_hcas_passed': False, 'cuda_graph_passed': False,
               'host_configuration_unchanged': False, 'qp_diagnostics_all_hcas_passed': False,
               'hosts': [n['name'] for n in cfg['nodes']], 'required_hcas_by_host': hcas,
               'transport_env_by_host': {name: transport_environment(env) for name, env in environments.items()},
               'topology_sha256': topology_sha256(cfg), 'library_sha256': digest, 'image_digest': IMAGE,
               'master_port': port, 'source_sha256': source_hashes, 'run_directory': str(case)}
    summary.update(transport=args.transport, identity=identity, network_plan_id=mesh_before['plan_id'],
                   network_boot_ids=mesh_before['network_boot_ids'],
                   hardware_mesh_passed=False, hardware_mesh_payload_passed=False, baseline_receipt_sha256=baseline_digest,
                   byte_exact_passed=False, mixed_size_passed=False, inplace_passed=False,
                   native_health_passed=False, asymmetric_cache_passed=False, modes_qualified=[])
    (case / 'mesh-before.json').write_text(json.dumps(mesh_before, indent=2) + '\n')
    event('created-operator-run-directory', path=str(case), network_configuration_mutations=False)
    event('held-exclusive-qualification-lock', path=str(state_dir(cfg) / '.qualification.lock'))
    (case / 'selected-build-receipt.json').write_text(json.dumps(build, indent=2) + '\n')
    event('archived-selected-build-receipt', path=str(case / 'selected-build-receipt.json'))
    if canonical.exists():
        (case / 'previous-canonical.json').write_bytes(canonical.read_bytes())
        event('archived-previous-qualification', path=str(case / 'previous-canonical.json'))
    atomic(canonical, summary)  # A failed attempt must never leave a previous successful receipt current.
    event('recorded-selected-environment', environments=environments)
    def remote(payload, action):
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        prefix = 'import sys\nsys.argv=' + repr(['qualify.py', '--worker', '--payload', encoded, '--host-action', action]) + '\n'
        script = "sudo -n -s <<'SPARK_RING_QUAL_ROOT'\nset -eu\npython3 - <<'SPARK_RING_QUAL_PY'\n" + prefix + source + "\nSPARK_RING_QUAL_PY\nSPARK_RING_QUAL_ROOT"
        node = payload['node']
        event('ssh-start', node=node['name'], host_action=action, exact_script=script)
        try:
            p = subprocess.run(ssh_args(node, cfg) + [script], text=True, capture_output=True,
                               timeout=args.timeout + 150 if action in ('finish', 'diagnose') else 180)
            event('ssh-finish', node=node['name'], host_action=action, returncode=p.returncode,
                  stdout=p.stdout, stderr=p.stderr)
        except Exception as exc:
            event('ssh-error', node=node['name'], host_action=action, error=str(exc)); raise
        if p.returncode:
            raise RuntimeError(f"{node['name']} {action}: {p.stderr.strip()}")
        return json.loads(p.stdout)
    def all_hosts(action):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(remote, p, action) for p in payloads]
            results, errors = [], []
            for rank, future in enumerate(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(f"rank{rank}: {exc}")
            if errors:
                raise RuntimeError('; '.join(errors))
        return results
    error = None
    staged = False
    try:
        before = all_hosts('preflight')
        (case / 'preflight.json').write_text(json.dumps(before, indent=2) + '\n')
        event('all-host-preflight-passed', path=str(case / 'preflight.json'))
        staged = True  # Include cleanup after partial staging/creation failures.
        all_hosts('stage')
        all_hosts('create')
        all_hosts('start')
        diagnostics = all_hosts('diagnose')
        (case / 'qp-diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        finished = all_hosts('finish')
        correctness = True
        preserve = True
        diag_ok = True
        final_qp_hcas = {}
        for rank, result in enumerate(finished):
            node = cfg['nodes'][rank]
            report = result['report']
            (case / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
            (case / f'rank-{rank}.log').write_text(result['log'])
            (case / f'os-after-rank-{rank}.json').write_text(json.dumps(result['snapshot'], indent=2) + '\n')
            valid = (validate_report(report, rank, hcas[node['name']], digest) if args.transport == 'nccl'
                     else validate_rocenante_report(report, rank, hcas[node['name']], digest, cfg['experiment']))
            if args.transport=='rocenante' and (identity.get('engine_source') or {}).get('engine_flags',{}).get('TF_GLM_HC_EXCHANGE')=='p2p':
                valid &= validate_mixed_p2p(report, relay_sha=relay_sha)
            correctness &= result['exit_code'] == '0' and valid
            preserve &= result['snapshot'] == before[rank]['snapshot']
            diag = sorted(set(re.findall(r'RouteFinal[^\n]* hca=([^\s]+)', result['log'])))
            final_qp_hcas[node['name']] = diag
            diag_ok &= set(diag) == set(hcas[node['name']]) and diagnostics[rank]['passed'] is True
        summary.update(correctness_passed=correctness, cuda_graph_passed=correctness,
                       payload_all_hcas_passed=correctness, host_configuration_unchanged=preserve,
                       qp_diagnostics_all_hcas_passed=diag_ok,
                       qualified=bool(correctness and preserve and diag_ok), final_qp_hcas=final_qp_hcas,
                       container_exit_codes={cfg['nodes'][r]['name']: x['exit_code'] for r, x in enumerate(finished)})
        if args.transport == 'rocenante' and correctness and preserve and diag_ok:
            # New processes: only rank0 inherits this attempt's actual cache bytes.
            # Selection caching remains disabled in the genuine framework.
            all_hosts('cleanup')
            symmetric_payloads = payloads
            first_id = run_id
            asym_id = run_id + '-asymmetric'
            payloads = json.loads(json.dumps(symmetric_payloads))
            for payload in payloads:
                payload['run_id'] = asym_id
                payload['container_name'] = 'spark-ring-qualify-' + asym_id.lower()
                payload['cache_seed_run_id'] = first_id if payload['rank'] == 0 else None
                payload['environment']['SPARK_RING_CACHE_LAYOUT'] = 'asymmetric'
                settings = rank_settings(cfg, payload['rank'], asym_id)
                payload['environment']['SPARK_RING_ROCENANTE_SETTINGS'] = json.dumps(settings)
            asym_before = all_hosts('preflight')
            all_hosts('stage'); all_hosts('create'); all_hosts('start')
            asym_diagnostics = all_hosts('diagnose')
            asym_finished = all_hosts('finish')
            (case / 'asymmetric-qp-diagnostics.json').write_text(json.dumps(asym_diagnostics, indent=2) + '\n')
            cache_inventories = {}
            asym_ok = True
            for rank, result in enumerate(asym_finished):
                report = result['report']; node = cfg['nodes'][rank]
                (case / f'asymmetric-rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
                (case / f'asymmetric-rank-{rank}.log').write_text(result['log'])
                asym_ok &= (result['exit_code'] == '0' and validate_rocenante_report(report, rank, hcas[node['name']], digest, cfg['experiment'])
                            and result['snapshot'] == asym_before[rank]['snapshot'] and asym_diagnostics[rank]['passed'] is True)
                if (identity.get('engine_source') or {}).get('engine_flags',{}).get('TF_GLM_HC_EXCHANGE')=='p2p':asym_ok &= validate_mixed_p2p(report, relay_sha=relay_sha)
                cache_inventories[node['name']] = report.get('cache_before', {}) if isinstance(report, dict) else {}
            asym_ok &= bool(cache_inventories[cfg['nodes'][0]['name']]) and not any(cache_inventories[n['name']] for n in cfg['nodes'][1:])
            summary.update(asymmetric_cache_passed=bool(asym_ok), asymmetric_cache_inventories=cache_inventories,
                           cache_scope='new processes; rank0 retained filesystem proxy/compiler artifacts, other ranks empty; in-memory program caches are fresh; selection cache disabled')
            correctness &= asym_ok
        mesh_after = live_guard(cfg)
        (case / 'mesh-after.json').write_text(json.dumps(mesh_after, indent=2) + '\n')
        mesh_ok = (mesh_after['plan_id'] == mesh_before['plan_id']
                   and mesh_after['network_boot_ids'] == mesh_before['network_boot_ids']
                   and experiment_identity(cfg, digest) == identity)
        if args.transport == 'rocenante':
            summary['hardware_mesh_traffic'] = hardware_mesh_delta(mesh_before, mesh_after)
            summary['hardware_mesh_payload_passed'] = True
        summary.update(hardware_mesh_passed=mesh_ok, qualified=bool(correctness and preserve and diag_ok and mesh_ok))
        if args.transport == 'rocenante':
            if (identity.get('engine_source') or {}).get('engine_flags',{}).get('TF_GLM_HC_EXCHANGE')=='p2p':summary['mixed_p2p_passed']=bool(correctness)
            if relay_sha is not None:summary['ring_relay_passed']=bool(correctness)
            summary.update(byte_exact_passed=bool(correctness), mixed_size_passed=bool(correctness),
                           inplace_passed=bool(correctness), native_health_passed=bool(correctness),
                           modes_qualified=['nccl', 'rocenante'] if correctness else [])
    except Exception as exc:
        error = str(exc)
        summary['qualified'] = False
        summary['failure'] = error
        event('qualification-failed', error=error)
    finally:
        if staged:
            # Each exact name is removed only after matching both ownership labels.
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(remote, payload, 'cleanup') for payload in payloads]
                cleanup_errors = []
                for rank, future in enumerate(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        cleanup_errors.append(f"rank{rank}: {exc}")
                if cleanup_errors:
                    summary['cleanup_errors'] = cleanup_errors
                    summary['qualified'] = False
                    event('cleanup-failed', errors=cleanup_errors)
        summary['finished_utc'] = utc()
        atomic(case / 'qualification.json', summary)
        atomic(canonical, summary)
    print(json.dumps(summary, indent=2))
    if not summary['qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
