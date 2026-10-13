#!/usr/bin/env python3
"""Read-only host/cache telemetry; never submits inference or controls model services.

Usage: cache_monitor.py --config /path/to/config.json {start,status,stop,sample}
Run start/stop/sample under the deployment's coordinated action journal. Runtime
writes are confined to the monitor state directory. No credentials/prompts logged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
GIB = 1 << 30
INTERVAL = 30
RETENTION = 7 * 86400
LOG_BYTES = 64 << 20
SEGMENT_BYTES = 4 << 20
SCRIPT = Path(__file__).resolve()
GLM_MIA107_DIGEST = 'sha256:14b5072e8e9918f1080153f57321415a387bc7d78cf9c9b44a7aeae2d4399ccf'
MEMORY_FIELDS = ('MemAvailable', 'MemFree', 'SwapFree', 'SwapTotal')
COUNTER_FIELDS = ('pgmajfault', 'pswpin', 'pswpout')
HEALTH_FIELDS = ('context_length', 'pool_tokens', 'pool_free_tokens', 'kept_bytes',
                 'kept_bytes_cap', 'kept_prompts', 'requests_running', 'requests_total')
STREAM_FIELDS = ('max', 'decoding', 'filling', 'paused')
SERVICE_COUNTERS = ('requests_total', 'prompt_tokens_total', 'completion_tokens_total',
                    'prefill_seconds_total', 'decode_seconds_total', 'cached_tokens_total',
                    'rounds_total', 'drafted_total', 'accepted_total')
REMOTE_CODE = """import json,pathlib
m={k:int(v.split()[0])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)] if k in {'MemAvailable','MemFree','SwapFree','SwapTotal'}}
v={k:int(n) for k,n in map(str.split,pathlib.Path('/proc/vmstat').read_text().splitlines()) if k in {'pgmajfault','pswpin','pswpout'}}
print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'memory':m,'vmstat':v}))
"""
HTTP_CODE = """import sys,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(sys.argv[1],timeout=6) as r:
 data=r.read(65537)
 if len(data)>65536:raise ValueError('Health response exceeds64KiB')
 sys.stdout.buffer.write(data)
"""


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def optional_json(path):
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(data, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')
    os.replace(tmp, path)


def number(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + ' must be a nonnegative integer')
    return value


def load_config(path):
    raw = Path(path).read_bytes()
    cfg = json.loads(raw)
    nodes = cfg['nodes']
    if not isinstance(nodes, list) or len(nodes) != 4 or len({n['name'] for n in nodes}) != 4:
        raise ValueError('Monitor requires four uniquely named configured hosts')
    for n in nodes:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', n['name']):
            raise ValueError('Invalid host name')
        ipaddress.ip_address(n['management_address'])
        if n['ssh'].split('@')[-1] != n['management_address']:
            raise ValueError('SSH destination must match the configured management address')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+', n['ssh']):
            raise ValueError('Configured SSH destination must be user@host')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', n['host_key_alias']):
            raise ValueError('Invalid SSH host-key alias')
    port = number(cfg['api_port'], 'api_port')
    if not 1 <= port <= 65535:
        raise ValueError('Invalid API port')
    options = cfg.get('ssh_options', {})
    for key in ('identity_file', 'known_hosts_file'):
        if options.get(key):
            target = Path(options[key]).expanduser()
            options[key] = str((Path(path).resolve().parent / target).resolve())
    for key in ('shared_root', 'state_root'):
        if cfg.get(key):
            cfg[key] = str((Path(path).resolve().parent / Path(cfg[key]).expanduser()).resolve())
    return cfg, hashlib.sha256(raw).hexdigest()


def ssh_command(node, cfg):
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            '-o', 'ServerAliveInterval=3', '-o', 'ServerAliveCountMax=1',
            '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none',
            '-o', 'UpdateHostKeys=no', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'HostKeyAlias=' + node['host_key_alias']]
    options = cfg.get('ssh_options', {})
    if options.get('known_hosts_file'):
        args += ['-o', 'UserKnownHostsFile=' + options['known_hosts_file']]
    if options.get('identity_file'):
        args += ['-i', options['identity_file']]
    return args + [node['ssh'], "python3 -B - <<'CACHE_MONITOR_READ_ONLY'\n" + REMOTE_CODE + '\nCACHE_MONITOR_READ_ONLY']


def command_json(argv):
    # Parent deadline also bounds a slow/trickling HTTP peer or hung SSH session.
    done = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    if done.returncode:
        # Do not retain remote output: it could contain credentials or other text.
        raise RuntimeError('Read-only collector exited ' + str(done.returncode))
    if len(done.stdout) > 65536:
        raise ValueError('Collector response exceeds64KiB')
    return json.loads(done.stdout)


def clean_health(raw):
    if raw.get('ok') is not True or raw.get('backend') != 'tensorfold':
        raise ValueError('Health endpoint is not ready TensorFold')
    out = {k: number(raw.get(k), 'health.' + k) for k in HEALTH_FIELDS}
    out['streams'] = {k: number(raw.get('streams', {}).get(k), 'streams.' + k) for k in STREAM_FIELDS}
    if out['pool_free_tokens'] > out['pool_tokens']:
        raise ValueError('Free pool rows exceed total')
    out['pool_used_tokens'] = out['pool_tokens'] - out['pool_free_tokens']
    counters = {}
    for k in SERVICE_COUNTERS:
        v = raw.get(k)
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError('Missing or invalid service counter: ' + k)
        counters[k] = v
    out['counters'] = counters
    out['kept_mix'] = {k: number(raw.get('kept_mix', {}).get(k), 'kept_mix.' + k)
                       for k in ('own', 'shared', 'chats')}
    return out


def clean_host(raw):
    import uuid
    boot = raw['boot_id']
    if str(uuid.UUID(boot)) != boot:
        raise ValueError('Invalid host boot identity')
    out = {'boot_id': boot}
    out['memory'] = {k: number(raw['memory'].get(k), 'memory.' + k) for k in MEMORY_FIELDS}
    out['vmstat'] = {k: number(raw['vmstat'].get(k), 'vmstat.' + k) for k in COUNTER_FIELDS}
    if out['memory']['SwapFree'] > out['memory']['SwapTotal']:
        raise ValueError('SwapFree exceeds SwapTotal')
    return out


def attempt(fn):
    try:
        return {'ok': True, 'observed_utc': utc(), 'data': fn()}
    except Exception as exc:
        # Type and bounded local diagnostic only; no response bodies or prompts.
        message = str(exc) if isinstance(exc, (ValueError, KeyError, RuntimeError)) else type(exc).__name__
        return {'ok': False, 'observed_utc': utc(), 'error': {'type': type(exc).__name__, 'message': message[:240]}}


def collect(cfg, config_sha):
    host = cfg['nodes'][0]['management_address']
    authority = '[' + host + ']' if ':' in host else host
    url = 'http://' + authority + ':' + str(cfg['api_port']) + '/health'
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as pool:
        health = pool.submit(attempt, lambda: clean_health(command_json([sys.executable, '-B', '-c', HTTP_CODE, url])))
        jobs = {n['name']: pool.submit(attempt, lambda node=n: clean_host(command_json(ssh_command(node, cfg))))
                for n in cfg['nodes']}
        hosts = {name: future.result() for name, future in jobs.items()}
        h = health.result()
    return {'schema': 'tensorfold-cache-monitor/v1', 'utc': utc(), 'config_sha256': config_sha,
            'duration_seconds': round(time.monotonic() - started, 3), 'health': h, 'hosts': hosts,
            'complete': h['ok'] and all(v['ok'] for v in hosts.values())}


def p8_estimate(sample, cfg):
    h = sample['health']
    p = cfg.get('profile', {})
    supported = (h['ok'] and len(cfg['nodes']) == 4 and p.get('parallel') == 6 and
                 p.get('multi_window') == 64 and p.get('context') == 1048576 and
                 cfg.get('image', '').endswith('@' + GLM_MIA107_DIGEST) and
                 h['data']['streams']['max'] == 6 and h['data']['context_length'] == 1048576)
    out = {'status': 'unqualified_estimate' if supported else 'unsupported_profile',
           'eight_lanes_qualified': False, 'decision': 'hardware_qualification_required',
           'model_scope': 'Mia107 GLM5.3Flash TP4 W64 only; estimated geometry6780 bytes/row',
           'additional_p8_fixed_buffer_bytes': None,
           'note': 'Host MemAvailable is shared OS memory, not a CUDA/process allocation guarantee. '
                   'The12GiB runtime and16GiB startup comparisons are sampled policy margins, not enforced reserves.'}
    if not supported:
        return out
    target = 8 * 1048576 + 530432
    extra = max(0, target - h['data']['pool_tokens']) * 6780
    available = [v['data']['memory']['MemAvailable'] for v in sample['hosts'].values() if v['ok']]
    minimum = min(available) if len(available) == 4 else None
    out.update(target_pool_rows=target, retained_spare_rows=530432,
               estimated_additional_arena_and_row_scratch_bytes=extra,
               minimum_current_mem_available_bytes=minimum,
               estimated_remaining_before_unknown_p8_buffers_bytes=None if minimum is None else minimum - extra,
               estimated_margin_to_12gib_before_unknown_buffers_bytes=None if minimum is None else minimum - extra - 12 * GIB,
               estimated_margin_to_16gib_before_unknown_buffers_bytes=None if minimum is None else minimum - extra - 16 * GIB)
    return out


def update_summary(previous, sample, cfg):
    if previous is None or previous.get('config_sha256') != sample['config_sha256']:
        previous = {'schema': 'tensorfold-cache-monitor-summary/v1', 'config_sha256': sample['config_sha256'],
                    'since_utc': sample['utc'], 'samples': 0, 'complete_samples': 0, 'failed_samples': 0,
                    'health_errors': 0, 'hosts': {}, 'health_watermarks': {}}
    s = previous
    s['samples'] += 1
    s['complete_samples' if sample['complete'] else 'failed_samples'] += 1
    s['last_sample_utc'] = sample['utc']
    current_counters = sample['health'].get('data', {}).get('counters')
    old_counters = s.get('last_service_counters')
    host_boots = {n: v['data']['boot_id'] for n, v in sample['hosts'].items() if v['ok']}
    continuous = (sample['complete'] and s.get('previous_complete') is True and
                  host_boots == s.get('last_host_boots') and old_counters is not None)
    monotone = continuous and all(current_counters[k] >= old_counters[k] for k in SERVICE_COUNTERS)
    interval = {'valid': False, 'reason': 'first_sample_or_collection_gap',
                'cached_prompt_fraction': None, 'completion_tokens_per_second': None}
    if continuous and not monotone:
        interval['reason'] = 'service_counter_reset'
        s['service_counter_resets'] = s.get('service_counter_resets', 0) + 1
    elif s.get('last_host_boots') is not None and host_boots != s['last_host_boots']:
        interval['reason'] = 'host_boot_changed_or_unavailable'
    elif monotone:
        elapsed = (dt.datetime.fromisoformat(sample['utc']) - dt.datetime.fromisoformat(s['last_service_utc'])).total_seconds()
        if elapsed > 0:
            delta = {k: current_counters[k] - old_counters[k] for k in SERVICE_COUNTERS}
            interval = {'valid': True, 'seconds': elapsed, 'deltas': delta,
                        'cached_prompt_fraction': delta['cached_tokens_total'] / delta['prompt_tokens_total']
                         if delta['requests_total'] > 0 and delta['prompt_tokens_total'] > 0 else None,
                        'completion_tokens_per_second': delta['completion_tokens_total'] / elapsed,
                        'note': 'Prompt/cache/prefill counters fold on finish including cancellation; completion tokens include live replies. Ongoing cold prefill may not yet appear. '
                                'Resets are detected by counter decreases/host boots; unexposed service restarts may be missed.'}
            if interval['cached_prompt_fraction'] is not None and interval['cached_prompt_fraction'] > 1:
                interval = {'valid': False, 'reason': 'inconsistent_service_counters',
                            'cached_prompt_fraction': None, 'completion_tokens_per_second': None}
    s['service_interval'] = interval
    s['previous_complete'] = sample['complete']
    s['last_host_boots'] = host_boots
    if current_counters is not None:
        s['last_service_counters'] = current_counters
        s['last_service_utc'] = sample['utc']
    if sample['health']['ok']:
        h = sample['health']['data']
        s['last_good_health_utc'] = sample['health']['observed_utc']
        for k in ('pool_free_tokens', 'pool_used_tokens', 'kept_bytes', 'kept_prompts', 'requests_running'):
            v = h[k]
            old = s['health_watermarks'].setdefault(k, {'min': v, 'max': v})
            old['min'], old['max'] = min(old['min'], v), max(old['max'], v)
        active = sum(h['streams'][k] for k in ('decoding', 'filling', 'paused'))
        s['max_admitted_lanes'] = max(s.get('max_admitted_lanes', 0), active)
    else:
        s['health_errors'] += 1
    for name, item in sample['hosts'].items():
        state = s['hosts'].setdefault(name, {'good_samples': 0, 'errors': 0, 'counter_resets': 0,
                                          'vmstat_delta_totals': {k: 0 for k in COUNTER_FIELDS}})
        if not item['ok']:
            state['errors'] += 1
            continue
        state['good_samples'] += 1
        data = item['data']
        mem = data['memory']
        for k in MEMORY_FIELDS:
            field = k + '_minimum_sampled_bytes'
            state[field] = min(state.get(field, mem[k]), mem[k])
        old = state.get('last_counters')
        same_boot = state.get('boot_id') == data['boot_id']
        monotone = old is not None and all(data['vmstat'][k] >= old[k] for k in COUNTER_FIELDS)
        delta = {k: data['vmstat'][k] - old[k] for k in COUNTER_FIELDS} if same_boot and monotone else None
        if old is not None and delta is None:
            state['counter_resets'] += 1
        if delta is not None:
            for k in COUNTER_FIELDS:
                state['vmstat_delta_totals'][k] += delta[k]
        state.update(last_counters=data['vmstat'], last_counter_delta=delta, boot_id=data['boot_id'],
                     last_good_utc=item['observed_utc'], latest_mem_available_bytes=mem['MemAvailable'])
    s['p8_estimate'] = p8_estimate(sample, cfg)
    s['scope'] = 'Lifetime sampled watermarks for this config; logs limited to7days/64MiB. Errors are not zero readings. Host-wide counters include other workloads.'
    return s


class Store:
    def __init__(self, directory, *, max_bytes=LOG_BYTES, segment_bytes=SEGMENT_BYTES, age=RETENTION):
        self.path = Path(directory)
        self.max_bytes, self.segment_bytes, self.age = max_bytes, segment_bytes, age

    def prune(self, now):
        files = sorted(self.path.glob('samples-*.jsonl'), key=lambda p: p.name)
        for p in files[:]:
            if now - int(p.stem.split('-')[1]) / 1e9 > self.age:
                p.unlink()
                files.remove(p)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.max_bytes:
                break
            total -= p.stat().st_size
            p.unlink()

    def append(self, sample, cfg, *, now=None):
        now = time.time() if now is None else now
        line = (json.dumps(sample, sort_keys=True, allow_nan=False) + '\n').encode()
        if len(line) > min(self.max_bytes, 65536):
            raise ValueError('Telemetry sample exceeds bounded record size')
        self.prune(now)
        files = sorted(self.path.glob('samples-*.jsonl'))
        last = files[-1] if files else None
        if last is None or last.stat().st_size + len(line) > self.segment_bytes or now - int(last.stem.split('-')[1]) / 1e9 > 3600:
            last = self.path / ('samples-' + str(int(now * 1e9)) + '.jsonl')
        with last.open('ab') as out:
            out.write(line)
        self.prune(now)
        atomic_json(self.path / 'latest.json', sample)
        path = self.path / 'summary.json'
        old = read_json(path) if path.exists() else None
        atomic_json(path, update_summary(old, sample, cfg))


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def proc_identity(pid):
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text()
    # comm can contain whitespace and parentheses; remaining fields begin at3.
    rest = stat[stat.rfind(')') + 2:].split()
    if rest[0] == 'Z':
        raise ProcessLookupError('Monitor process exited')
    argv = (p / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    return {'pid': pid, 'boot_id': boot_id(), 'start_ticks': int(rest[19]),
            'argv': [os.fsdecode(v) for v in argv]}


def expected_argv(config, state):
    return [sys.executable, '-B', str(SCRIPT), '--config', str(config), '--state-dir', str(state), 'run']


def identity_matches(record, expected, inspect=proc_identity):
    try:
        if type(record.get('pid')) is not int or record['pid'] <= 1 or record.get('argv') != expected:
            return False
        current = inspect(record['pid'])
        return all(record.get(k) == current[k] for k in ('pid', 'boot_id', 'start_ticks', 'argv'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def safe_signal(record, expected):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support required for race-safe stop')
    try:
        fd = os.pidfd_open(record['pid'])
    except ProcessLookupError:
        return False
    try:
        if not identity_matches(record, expected):
            raise RuntimeError('Monitor identity differs; refusing to signal process')
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(fd)


def flock(path):
    handle = Path(path).open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def status(state, expected):
    path = state / 'pid.json'
    record = optional_json(path)
    running = record is not None and identity_matches(record, expected)
    latest = read_json(state / 'latest.json') if (state / 'latest.json').exists() else None
    age = None if latest is None else max(0, (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(latest['utc'])).total_seconds())
    return {'running': running, 'stale_or_foreign_pid_record': record is not None and not running,
            'state_directory': str(state), 'last_sample_age_seconds': age,
            'telemetry_stale': age is None or age > 2 * INTERVAL,
            'last_monitor_error': read_json(state / 'monitor-error.json') if (state / 'monitor-error.json').exists() else None,
            'latest': latest, 'summary': read_json(state / 'summary.json') if (state / 'summary.json').exists() else None}


def run(config, state, cfg, config_sha):
    held = flock(state / 'daemon.lock')
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    identity = proc_identity(os.getpid())
    atomic_json(state / 'pid.json', identity)
    store = Store(state)
    try:
        while not stop.is_set():
            started = time.monotonic()
            # Never silently apply a different profile/config after daemon start.
            if hashlib.sha256(config.read_bytes()).hexdigest() != config_sha:
                raise RuntimeError('Configuration changed; restart this monitor after reviewing the new profile')
            store.append(collect(cfg, config_sha), cfg)
            stop.wait(max(0, INTERVAL - (time.monotonic() - started)))
    except Exception as exc:
        atomic_json(state / 'monitor-error.json', {'utc': utc(), 'type': type(exc).__name__, 'message': str(exc)[:240]})
        raise
    finally:
        if (state / 'pid.json').exists() and read_json(state / 'pid.json') == identity:
            (state / 'pid.json').unlink()
        held.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('action', choices=('start', 'stop', 'status', 'sample', 'run'))
    args = parser.parse_args(argv)
    config = args.config.resolve(strict=True)
    cfg, sha = load_config(config)
    state = (args.state_dir or Path(cfg.get('state_root', str(Path(cfg['shared_root']) / 'state'))) / 'cache-monitor').resolve()
    expected = expected_argv(config, state)
    if args.action == 'status':
        print(json.dumps(status(state, expected), indent=2, sort_keys=True))
        return 0
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.action == 'run':
        run(config, state, cfg, sha)
        return 0
    with flock(state / 'control.lock'):
        existing = status(state, expected)
        if args.action == 'start':
            if existing['running']:
                print(json.dumps({'already_running': True}))
                return 0
            # A daemon may be starting with no PID receipt yet: fail closed.
            with flock(state / 'daemon.lock'):
                pass
            child = subprocess.Popen(expected, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError('Monitor failed at startup; inspect monitor-error.json')
                if status(state, expected)['running']:
                    print(json.dumps({'started': True, 'pid': child.pid, 'state_directory': str(state)}))
                    return 0
                time.sleep(0.1)
            raise RuntimeError('Monitor start receipt timed out; inspect status before retrying')
        if args.action == 'stop':
            path = state / 'pid.json'
            record = optional_json(path)
            if record is None:
                print(json.dumps({'already_stopped': True}))
                return 0
            if not identity_matches(record, expected):
                # Stale/foreign records must never become a signal target.
                print(json.dumps({'already_stopped': True, 'stale_or_foreign_pid_record': True, 'processes_signalled': False}))
                return 0
            safe_signal(record, expected)
            deadline = time.monotonic() + 20
            while identity_matches(record, expected) and time.monotonic() < deadline:
                time.sleep(0.1)
            if identity_matches(record, expected):
                raise RuntimeError('Monitor has not exited; no escalation signal sent')
            print(json.dumps({'stopped': True}))
            return 0
        with flock(state / 'daemon.lock'):
            sample = collect(cfg, sha)
            Store(state).append(sample, cfg)
            print(json.dumps(sample, indent=2))
            return 0 if sample['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
