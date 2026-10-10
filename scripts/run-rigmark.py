#!/usr/bin/env python3
"""Run the clean pinned Alex Ellis RigMark protocol against TensorFold."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

REVISION = 'd8353e93b274e8d880ab14a5df2a55c87d7bee16'
COMPARISON_ID = '2026-09-12-nccl-patch-ab-v1'


def health(url):
    with urllib.request.urlopen(url.rstrip('/') + '/health', timeout=15) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--model', default='GLM-5.3-Flash-EXL3')
    p.add_argument('--output', type=Path, required=True, help='new output directory')
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--skip-prefill', action='store_true')
    args = p.parse_args()
    source = args.source.resolve(strict=True)
    args.output = args.output.resolve()
    git = ['git', '-c', 'safe.directory=' + str(source), '-C', str(source)]
    revision = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(git + ['status', '--porcelain'], text=True).strip()
    if revision != REVISION or status:
        raise RuntimeError('RigMark source must be the clean pinned revision ' + REVISION)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.metadata.read_text())
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    before = health(args.url)
    if before.get('requests_running') or before.get('busy'):
        raise RuntimeError('Endpoint is busy; run against an idle service')
    command = [sys.executable, str(source / 'rigmark'), 'run', '--base-url', args.url,
               '--model', args.model, '--label', 'glm53-exl3-tensorfold-tp4-low',
               '--comparison-id', COMPARISON_ID, '--metadata', str(args.output / 'metadata.json'),
               '--seed', '20260912', '--runs', '5', '--decode-tokens', '4096',
               '--prefill-depths', '8192,32768,65536', '--prefill-runs', '3',
               '--concurrency', '1,2,4', '--concurrency-runs', '3',
               '--concurrency-tokens', '256', '--concurrency-workload', 'code',
               '--extra-body', '{"chat_template_kwargs":{"reasoning_effort":"low"}}',
               '--output', str(args.output / 'result.json')]
    if args.skip_prefill:
        command.append('--skip-prefill')
    started = datetime.now(timezone.utc).isoformat()
    (args.output / 'operator-command.json').write_text(json.dumps(command, indent=2) + '\n')
    # NFS may map ownership to nobody. Trust only this checkout for the child
    # process so upstream RigMark can record its Git identity, without changing
    # the operator's global Git configuration.
    env = os.environ.copy()
    slot = int(env.get('GIT_CONFIG_COUNT', '0'))
    env['GIT_CONFIG_KEY_' + str(slot)] = 'safe.directory'
    env['GIT_CONFIG_VALUE_' + str(slot)] = str(source)
    env['GIT_CONFIG_COUNT'] = str(slot + 1)
    env['GIT_OPTIONAL_LOCKS'] = '0'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    process = subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (args.output / 'run.log').open('w') as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
    rc = process.wait()
    after = health(args.url)
    result_path = args.output / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    decode = result.get('decode', {})
    prefill = result.get('prefill', {})
    concurrency = result.get('concurrency', {})
    measured = sum(len(v['runs']) for v in decode.values())
    measured += sum(len(v[arm]['runs']) for v in prefill.values() for arm in ('cold', 'warm_replay'))
    measured += sum(len(round_['streams']) for v in concurrency.values() for round_ in v['rounds'])
    delta = after.get('requests_total', 0) - before.get('requests_total', 0)
    gates = [v['completion_gate'] for v in decode.values()]
    audit = {
        'started': started, 'finished': datetime.now(timezone.utc).isoformat(),
        'return_code': rc, 'benchmark_revision': revision, 'source_clean': True,
        'requests_before': before.get('requests_total'), 'requests_after': after.get('requests_total'),
        'requests_delta': delta, 'receipt_requests': measured,
        'external_requests_detected': delta != measured,
        'active_after': after.get('requests_running'),
        'decode_gates_passed': len(gates) == 3 and all(g['passed'] == g['total'] == 5 for g in gates),
        'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest() if result_path.exists() else None,
        'scope': 'Client protocol matches Alex Sept12. Engine, quantisation, scheduler and context differ. No server configuration changes.',
    }
    (args.output / 'run-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
