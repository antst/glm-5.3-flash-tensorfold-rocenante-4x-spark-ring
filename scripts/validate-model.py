#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the owned candidate model and its traffic on a dedicated idle API.

Adapts the pinned recipe's greedy serial/drafted/concurrent correctness cases.
Host observations are read-only. Inference and new private result files are the
only intended changes. This checks selected answers and bit-identical visible
outputs, not arbitrary model quality or an actual million-token input.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
sys.dont_write_bytecode = True

from common import IMAGE, ordered_hcas, state_dir, transport_environment
from experiment import (ExperimentOperator, experiment_identity, prepare_config,
                        sha, verify_boot_qualification)
from network import hardware_payload_delta, live_guard
from rocenante_adapter import SOURCE_HASHES

MODEL = 'GLM-5.3-Flash-EXL3'
CASES = (
    ('code', 'Write a Python function named fibonacci(n) that returns a list of the first n Fibonacci numbers, beginning with 0 and 1. Reply only with Python code.', 128),
    ('fact', 'What is the capital of France? Reply with the city name only.', 64),
    ('arithmetic', 'What is 19 multiplied by 23? Reply with only the integer.', 32),
)
# Exact pinned, unmodified engine and drafter sources, in addition to the
# adapter's comm/verify/graphs/multi source contract. No engine patch is applied.
ENGINE_HASHES = {
    'families/glm5_next/cuda/engine.py': 'f986b4c005aa99652b125f88e13f7ce33af30c55a7a1fdfe1f97f3aadc4d9361',
    'families/glm5_next/cuda/dflash2.py': '0dcea19b7b9cc4c4b51436b2efcfcce134f90729a911ac1db2f15cea7a257475',
    'families/glm5_next/cuda/dflash2_multi.py': 'ccfbcf59c3f2fb3b8f8d5e2ebe46b332578ca51f2680ef46b41d7ef7eedfff99',
    'families/glm5_next/cuda/forward.py': 'efb7a4d425f9e6dc0ac19c7ab007feb0113cdd800cfb8198902f8a46789772a9',
    'families/glm5_next/cuda/hcsplit.py': 'fb5c8e31820859b621e7da8d9aa7edc296b87c56d1a4f09adf43bc8647c52232',
}
ERROR_FIELDS = {'out_of_buffer', 'req_cqe_error', 'resp_cqe_error',
                'req_transport_retries_exceeded', 'req_rnr_retries_exceeded',
                'packet_seq_err', 'local_ack_timeout_err'}
FUNCTION_FIELDS = {'rx_vport_rdma_unicast_bytes', 'tx_vport_rdma_unicast_bytes',
                   'rx_vport_rdma_unicast_packets', 'tx_vport_rdma_unicast_packets'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def request(endpoint, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(endpoint + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
            require(response.status == 200 and isinstance(payload, dict), 'Unexpected HTTP response')
    except urllib.error.HTTPError as error:
        raise RuntimeError('HTTP ' + str(error.code) + ': ' + error.read(8192).decode(errors='replace')) from error
    require(not payload.get('error'), 'API error: ' + str(payload.get('error')))
    return {'utc': utc(), 'seconds': time.monotonic() - started,
            'body': body, 'response': payload}


def visible(result):
    response = result['response']
    require(isinstance(response.get('choices'), list) and len(response['choices']) == 1,
            'Expected exactly one completion choice')
    choice = response['choices'][0]
    require(choice.get('finish_reason') == 'stop', 'Completion was truncated or did not finish normally')
    message = choice.get('message')
    require(isinstance(message, dict) and isinstance(message.get('content'), str), 'Missing textual answer')
    usage = response.get('usage', {})
    require(all(type(usage.get(key)) is int and usage[key] > 0 for key in ('prompt_tokens', 'completion_tokens')),
            'Missing positive integer token usage')
    require(not message.get('reasoning_content') and not message.get('tool_calls') and not message.get('refusal'),
            'Expected an answer with thinking disabled, without tools or refusal')
    require(usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0) == 0,
            'Thinking-disable override was not honored')
    return {key: message.get(key) for key in ('content', 'reasoning_content', 'tool_calls', 'refusal')}


def visible_bytes(result):
    return json.dumps(visible(result), sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def semantic(label, result):
    content = visible(result)['content'].strip()
    if label == 'fact':
        return re.sub('[^a-z]', '', content.lower()) == 'paris'
    if label == 'arithmetic':
        return content == '437'
    if label != 'code':
        raise ValueError('Unknown truth case')
    match = re.search(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
    try:
        tree = ast.parse(match.group(1) if match else content)
    except SyntaxError:
        return False
    # Parse only: generated code is never executed on the operator host.
    return any(isinstance(node, ast.FunctionDef) and node.name == 'fibonacci'
               and [a.arg for a in node.args.args] == ['n'] for node in tree.body)


def check_idle(health, parallel):
    require(health.get('ok') is True and health.get('busy') is False and health.get('requests_running') == 0,
            'Dedicated API must be healthy and idle')
    require(type(health.get('requests_total')) is int and health['requests_total'] >= 0,
            'Reliable requests_total counter is required')
    streams = health.get('streams', {})
    require(streams.get('max') == parallel and all(streams.get(k, 0) == 0 for k in
            ('decoding', 'prefilling', 'filling', 'paused')), 'Live scheduler/profile mismatch or competing traffic')


def parse_active(log):
    records = []
    for line in log.splitlines():
        if 'SPARK_RING_NATIVE_FATAL ' in line or 'SPARK_RING_ROCENANTE_EXIT ' in line:
            raise RuntimeError('Candidate logged a fatal fault or exited')
        tag = 'SPARK_RING_ROCENANTE_ACTIVE '
        if tag in line:
            stamp, text = line.split(tag, 1)
            timestamp = datetime.fromisoformat(stamp.strip().split()[0].replace('Z', '+00:00')).timestamp()
            records.append({'unix_seconds': timestamp, 'report': json.loads(text)})
    require(bool(records), 'No host-only runtime controller health report yet')
    return records[-1]


def host_observation(payload):
    """Delivered over the operator's strict SSH/sudo heredoc; no host writes."""
    import hashlib, importlib.util, json, os, pathlib, subprocess
    def run(args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=120)
    name = payload['container']
    container = json.loads(run(['docker', 'inspect', name]))[0]
    image = json.loads(run(['docker', 'image', 'inspect', payload['image']]))[0]
    if not container['State']['Running'] or container['State']['Pid'] <= 0:
        raise RuntimeError('Owned candidate container is not running')
    env = dict(value.split('=', 1) for value in container['Config']['Env'] if '=' in value)
    for key, wanted in payload['labels'].items():
        if container['Config'].get('Labels', {}).get(key) != wanted:
            raise RuntimeError('Candidate ownership/library label mismatch: ' + key)
    if (container['Config']['Image'] != payload['image'] or container['Image'] != image['Id'] or
            image['Architecture'] != 'arm64' or container['Config']['Cmd'] != payload['command']):
        raise RuntimeError('Candidate pinned image or rank/profile command mismatch')
    if any(env.get(key) != value for key, value in payload['env'].items()):
        raise RuntimeError('Candidate environment differs from the qualified rank command')
    if any(env.get(key) != value for key, value in payload['transport_env'].items()):
        raise RuntimeError('Candidate transport environment differs')
    inside = '''import hashlib,importlib.util,json,os,pathlib
data=PAYLOAD
library=pathlib.Path(os.environ['TF_NCCL_LIB']).resolve(strict=True)
maps=sorted({line.split(maxsplit=5)[-1].strip() for line in pathlib.Path('/proc/1/maps').read_text().splitlines() if '/libnccl' in line and len(line.split(maxsplit=5))==6})
if not maps or any(pathlib.Path(p).resolve()!=library for p in maps):raise RuntimeError('Unexpected live NCCL mapping')
digest=hashlib.sha256(library.read_bytes()).hexdigest()
if digest!=data['library_sha256']:raise RuntimeError('Live NCCL SHA256 differs')
spec=importlib.util.find_spec('tensorfold');root=pathlib.Path(next(iter(spec.submodule_search_locations)))
observed={}
for name,wanted in data['engine_sources'].items():
 actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
 if actual!=wanted:raise RuntimeError('Pinned TensorFold source changed: '+name)
 observed[name]=actual
for name,wanted in data['sources'].items():
 if hashlib.sha256((pathlib.Path('/opt/spark-ring/scripts')/name).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Candidate source changed: '+name)
vendor=pathlib.Path('/opt/spark-ring/vendor');manifest=vendor/'source-manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=data['vendor_manifest_sha256']:raise RuntimeError('Prepared vendor manifest changed')
for name,wanted in json.loads(manifest.read_text())['files'].items():
 if hashlib.sha256((vendor/name).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Prepared vendor source changed: '+name)
print(json.dumps({'library':str(library),'library_sha256':digest,'mapped_nccl':maps,'engine_sources':observed,'sources_verified':True,'vendor_verified':True}))
'''.replace('PAYLOAD', repr(payload))
    loader = json.loads(run(['docker', 'exec', name, 'python3', '-B', '-c', inside]))
    snapshot_env = dict(os.environ, SPARK_RING_REQUIRED_HCAS=json.dumps(payload['hcas']), PYTHONDONTWRITEBYTECODE='1')
    snapshot = json.loads(subprocess.check_output(['python3', '-B', payload['counter_script']],
                                                env=snapshot_env, text=True, timeout=120))
    if snapshot.get('live_qps_capture_returncode') != 0:
        raise RuntimeError('Live QP observation unavailable')
    rows = []
    for row in snapshot['live_qps']:
        rows.extend([{**{k: v for k, v in row.items() if k != 'qp'}, **q} for q in row['qp']]
                    if 'qp' in row else [row])
    owners = {}
    for tid in {q.get('pid') for q in rows if q.get('pid')}:
        status = pathlib.Path('/proc') / str(tid) / 'status'
        if status.exists():
            fields = dict(line.split(':', 1) for line in status.read_text().splitlines() if ':' in line)
            owners[str(tid)] = {key: fields[key].strip() for key in ('Name', 'Pid', 'Tgid', 'NSpid') if key in fields}
    log = run(['docker', 'logs', '--timestamps', '--tail', '3000', name])
    return {'container_id': container['Id'], 'pid': container['State']['Pid'],
            'image_spec': container['Config']['Image'], 'image_id': container['Image'],
            'labels': container['Config'].get('Labels', {}), 'command': container['Config']['Cmd'],
            'environment': env, 'loader': loader, 'snapshot': snapshot,
            'qp_owners': owners, 'qps': rows, 'runtime_log': log}


def observation(op, rank, library, digest, identity):
    cfg = op.config
    _, command = op.command(rank, library, digest)
    env = dict(command[i + 1].split('=', 1) for i, item in enumerate(command) if item == '-e')
    payload = {'container': cfg['container_name'], 'image': IMAGE,
               'labels': {'spark-ring.recipe': cfg['recipe_id'], 'spark-ring.root': cfg['shared_root'],
                          'spark-ring.nccl-sha256': digest},
               'command': command[command.index(IMAGE) + 1:], 'env': env,
               'transport_env': transport_environment(env), 'library_sha256': digest,
               'engine_sources': dict(SOURCE_HASHES, **ENGINE_HASHES), 'sources': identity['sources'],
               'vendor_manifest_sha256': identity['vendor']['manifest_sha256'],
               'hcas': ordered_hcas(cfg['nodes'][rank]),
               'counter_script': str(Path(cfg['shared_root']) / 'scripts/capture-function-counters.py')}
    code = inspect.getsource(host_observation) + '\nprint(__import__("json").dumps(host_observation(' + repr(payload) + ')))\n'
    record = json.loads(op.remote_python(rank, code, timeout=180))
    record['controller'] = parse_active(record['runtime_log'])
    check_controller(record['controller']['report'], cfg, rank, identity)
    return record


def check_controller(report, cfg, rank, identity):
    require(report.get('mode') == cfg['experiment']['mode'] and report.get('rank') == rank and
            report.get('world') == 4 and report.get('initialized') is True and
            report.get('closed') is False and report.get('failure') is None,
            'Live controller mode/rank/initialization/health mismatch')
    require(report.get('native_active') is (cfg['experiment']['mode'] == 'rocenante') and
            report.get('hca_names') == ordered_hcas(cfg['nodes'][rank]) and
            report.get('vendor_manifest_sha256') == identity['vendor']['manifest_sha256'] and
            report.get('adapter_sha256') == identity['sources']['rocenante_adapter.py'] and
            report.get('tensorfold_source_sha256') == SOURCE_HASHES and
            report.get('max_gather_bytes') == cfg['experiment']['max_gather_bytes'] and
            report.get('min_gather_bytes') == cfg['experiment']['min_gather_bytes'],
            'Live controller native/source/vendor/admission identity mismatch')
    blocks = cfg['experiment'].get('native_blocks', 8)
    require(type(report.get('native_threads')) is int and report['native_threads'] == 512 and
            type(report.get('native_blocks')) is int and report['native_blocks'] == blocks,
            'Live controller requested launch geometry differs from fixed512/configured blocks')
    require('runtime_native_threads' in report and 'runtime_native_blocks' in report,
            'Live controller actual launch geometry fields are missing')
    if cfg['experiment']['mode'] == 'rocenante':
        require(type(report['runtime_native_threads']) is int and report['runtime_native_threads'] == 512 and
                type(report['runtime_native_blocks']) is int and report['runtime_native_blocks'] == blocks,
                'Live native runtime launch geometry differs from requested geometry')
    else:
        require(report['runtime_native_threads'] is None and report['runtime_native_blocks'] is None,
                'NCCL control must not have native runtime launch geometry')
    require(all(row.get('completion_errors') == 0 for row in report.get('path_counters', [])),
            'Native completion errors')


def fresh_observations(op, library, digest, identity, after_time, timeout):
    def one(rank):
        deadline = time.monotonic() + timeout
        while True:
            record = observation(op, rank, library, digest, identity)
            if record['controller']['unix_seconds'] > after_time:
                return record
            require(time.monotonic() < deadline, 'Timed out waiting for a fresh host-only controller report')
            time.sleep(2)
    return op.parallel(one)


def qp_proof(record, hcas, marker_pids):
    def hca(row):
        return str(row.get('ifname', row.get('dev', row.get('link', '')))).split('/')[0]
    selected = [q for q in record['qps'] if hca(q) in hcas]
    tgid = lambda row: record['qp_owners'].get(str(row.get('pid')), {}).get('Tgid')
    owned = [q for q in selected if tgid(q) == str(record['pid']) and q.get('type') == 'RC' and q.get('state') == 'RTS']
    foreign = [q for q in selected if tgid(q) != str(record['pid']) and q.get('pid') not in marker_pids
               and not (q.get('pid') is None and q.get('type') == 'GSI' and q.get('comm') == 'ib_core')]
    require(not foreign and {hca(q) for q in owned} == set(hcas), 'Foreign QPs or missing model-owned RC/RTS HCA coverage')
    native = record['controller']['report'].get('path_counters', [])
    require({(row['device'], row['qp_number']) for row in native}.issubset(
            {(hca(q), int(q['lqpn'])) for q in owned}), 'Native origin QPs do not belong to this model PID')
    return {'passed': True, 'owned_rc_rts_qps': owned, 'native_origin_paths': native,
            'required_hca_qp_counts': {name: sum(hca(q) == name for q in owned) for name in hcas}}


def function_deltas(before, after, minimum_bytes):
    old, new = before['devices'], after['devices']
    require(set(old) == set(new) and len(old) == 4, 'Four-function counter identity changed')
    result = {}
    for hca in old:
        a, b = old[hca], new[hca]
        require((a['netdev'], a['pci_address']) == (b['netdev'], b['pci_address']), 'Fabric function mapping changed')
        require(FUNCTION_FIELDS.issubset(a['function_rdma']) and FUNCTION_FIELDS.issubset(b['function_rdma']) and
                ERROR_FIELDS.issubset(a['hw_requests_and_errors']) and ERROR_FIELDS.issubset(b['hw_requests_and_errors']),
                'Mandatory function counters are missing')
        payload = {key: b['function_rdma'][key] - a['function_rdma'][key] for key in FUNCTION_FIELDS}
        errors = {key: b['hw_requests_and_errors'][key] - a['hw_requests_and_errors'][key] for key in ERROR_FIELDS}
        require(all(value >= 0 for value in payload.values()) and all(value == 0 for value in errors.values()) and
                all(payload[key] >= minimum_bytes for key in ('rx_vport_rdma_unicast_bytes', 'tx_vport_rdma_unicast_bytes')) and
                all(payload[key] > 0 for key in ('rx_vport_rdma_unicast_packets', 'tx_vport_rdma_unicast_packets')),
                'Missing bidirectional function payload, reset, or transport error: ' + hca)
        result[hca] = {'payload': payload, 'errors': errors, 'minimum_bytes': minimum_bytes, 'passed': True}
    return result


def native_deltas(before, after, rank, hcas):
    key = lambda row: (row['peer_rank'], row['path_index'])
    old, new = {key(r): r for r in before}, {key(r): r for r in after}
    expected = {(peer, path) for peer in range(4) if peer != rank for path in range(2)}
    require(len(before) == len(after) == 6 and set(old) == set(new) == expected, 'Expected six native origin paths')
    results = []
    for path in sorted(expected):
        a, b = old[path], new[path]
        require(all(a.get(k) == b.get(k) for k in ('device', 'qp_number', 'remote_qp_number',
                'local_hca_index', 'remote_hca_index', 'physical_hops')), 'Native QP identity changed')
        delta = {field: b[field] - a[field] for field in ('payload_writes', 'payload_bytes',
                 'physical_hop_payload_bytes', 'flag_writes', 'send_completions', 'completion_errors')}
        require(all(v >= 0 for v in delta.values()) and delta['completion_errors'] == 0 and
                delta['payload_bytes'] > 0 and delta['payload_writes'] > 0, 'Native path lacks model payload or has errors')
        results.append({'peer_rank': path[0], 'path_index': path[1], 'device': b['device'],
                        'qp_number': b['qp_number'], 'deltas': delta})
    require({r['device'] for r in results} == set(hcas), 'Native payload must use all four HCA functions')
    return results


def tc_observation_deltas(before, after):
    """Retain hardware-rule deltas also for a direct-neighbor NCCL control."""
    require(before['plan_id'] == after['plan_id'] and before['network_boot_ids'] == after['network_boot_ids'],
            'Mesh plan/boots changed during validation')
    require(set(before['hosts']) == set(after['hosts']) and len(before['hosts']) == 4, 'Missing mesh host')
    result = []
    for host in sorted(before['hosts']):
        old = before['hosts'][host]['rules']
        new = after['hosts'][host]['rules']
        require(len(old) == len(new) == 2, 'Expected two hardware diagonal rules per host')
        for index, (a, b) in enumerate(zip(old, new)):
            require(a['rule_identity'] == b['rule_identity'] and all(row.get(key) is True for row in (a, b)
                    for key in ('in_hw', 'skip_sw')), 'Hardware TC identity/placement changed')
            require(len(a['actions_stats']) == len(b['actions_stats']) == 3, 'Unexpected TC action statistics')
            deltas = []
            for prior, current in zip(a['actions_stats'], b['actions_stats']):
                require(all(row.get(key, 0) == 0 for row in (prior, current)
                        for key in ('sw_packets', 'sw_bytes', 'drops')), 'Software transit or TC drops')
                fields = ('packets', 'bytes', 'hw_packets', 'hw_bytes', 'sw_packets', 'sw_bytes', 'drops')
                delta = {key: current[key] - prior[key] for key in fields if key in prior and key in current}
                require(all(type(v) is int and v >= 0 for v in delta.values()), 'TC counter reset/invalid statistics')
                deltas.append(delta)
            result.append({'host': host, 'rule_index': index, 'rule_identity': b['rule_identity'],
                           'actions_deltas': deltas, 'hardware_only': True})
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.chmod(path, 0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('nccl', 'rocenante'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='NEW directory beneath the configured ignored state directory')
    parser.add_argument('--timeout', type=int, default=600, help='HTTP timeout seconds')
    parser.add_argument('--report-timeout', type=int, default=90, help='Wait for the runtime’s 60-second host-only report')
    parser.add_argument('--minimum-bytes', type=int, default=1048576, help='Minimum RX and TX payload on each of the 16 functions')
    parser.add_argument('--plan', action='store_true', help='Offline plan: no SSH, API, GPU, or output writes')
    args = parser.parse_args()
    cfg = prepare_config(args.config, args.mode)
    output = args.output.resolve()
    require(output.is_relative_to(state_dir(cfg).resolve()) and output != state_dir(cfg).resolve() and not output.exists(),
            'Output must be a NEW private directory beneath configured state/')
    require(args.timeout > 0 and args.report_timeout >= 65 and args.minimum_bytes > 0, 'Invalid timeouts/payload threshold')
    parallel = cfg['profile']['parallel']
    require(parallel == 8, 'This candidate qualification explicitly exercises the eight-lane profile')
    endpoint = 'http://' + cfg['nodes'][0]['management_address'] + ':' + str(cfg['api_port'])
    expected_count = len(CASES) * (parallel + 2)
    if args.plan:
        print(json.dumps({'offline': True, 'endpoint': endpoint, 'mode': args.mode, 'profile': cfg['profile'],
            'output': str(output), 'chat_requests': expected_count, 'concurrency': parallel,
            'cases': [label for label, _, _ in CASES], 'host_actions': 'read-only binding/counters/QPs/mesh check',
            'receipts_required': ['state/collective-qualification.json', 'state/rocenante-qualification.json'],
            'native_hardware_payload_required': args.mode == 'rocenante'}, indent=2))
        return 0
    require(not any(ipaddress.ip_address(n['management_address']) in ipaddress.ip_network('192.0.2.0/24')
                    for n in cfg['nodes']), 'Configure real management addresses before host/API actions')
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    report = {'schema': 'tensorfold-rocenante-model-qualification/v1', 'started': utc(),
              'mode': args.mode, 'endpoint': endpoint, 'qualified': False, 'passed': False,
              'validator_sha256': sha(__file__), 'profile': cfg['profile'], 'cases': [],
              'expected_requests': expected_count, 'completed_requests': 0,
              'scope': 'Pinned TP4/P8 text model: greedy draft/serial/concurrent equality, two factual answers and Python AST shape, current identity and traffic. No generated code execution, vision, or full 1M input test.',
              'host_configuration_changes': False}
    destination = output / 'model-qualification.json'
    try:
        op = ExperimentOperator(cfg)
        library, digest = op.library_identity()
        qualification = op.qualification(state_dir(cfg) / 'collective-qualification.json', digest)
        identity = experiment_identity(cfg, digest)
        report['identity'] = identity
        report['transport_receipt_sha256'] = {name: sha(state_dir(cfg) / filename) for name, filename in
            (('nccl', 'collective-qualification.json'), ('rocenante', 'rocenante-qualification.json'))}
        models = request(endpoint, '/v1/models', timeout=args.timeout)['response']
        require(any(row.get('id') == MODEL for row in models.get('data', [])), 'Wrong served model')
        report['models_response'] = models
        tokenized = request(endpoint, '/tokenize', {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Ready.'}],
            'chat_template_kwargs': {'enable_thinking': False}}, args.timeout)['response']
        require(tokenized.get('max_model_len') == cfg['profile']['context'], 'Actual API context differs from configured context')
        report['context_advertisement'] = tokenized
        before = fresh_observations(op, library, digest, identity, time.time(), args.report_timeout)
        mesh_before = live_guard(cfg)
        verify_boot_qualification(qualification, mesh_before)
        report['before'] = before
        report['mesh_before'] = mesh_before
        report['qp_before'] = {n['name']: qp_proof(before[n['name']], ordered_hcas(n),
            mesh_before['allowed_marker_pids_by_host'][n['name']]) for n in cfg['nodes']}
        health_before = request(endpoint, '/health', timeout=args.timeout)['response']
        check_idle(health_before, parallel)
        report['health_before'] = health_before
        save(destination, report)
        for label, prompt, budget in CASES:
            body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                    'top_p': 1, 'seed': 42, 'max_tokens': budget, 'stream': False, 'reasoning_effort': 'none',
                    'chat_template_kwargs': {'enable_thinking': False}, 'draft': False}
            serial = request(endpoint, '/v1/chat/completions', body, args.timeout)
            report['completed_requests'] += 1
            drafted_body = dict(body, draft=True)
            drafted = request(endpoint, '/v1/chat/completions', drafted_body, args.timeout)
            report['completed_requests'] += 1
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                concurrent = list(pool.map(lambda _: request(endpoint, '/v1/chat/completions', drafted_body, args.timeout), range(parallel)))
            report['completed_requests'] += len(concurrent)
            expected = visible_bytes(serial)
            equal = [visible_bytes(r) == expected for r in [drafted, *concurrent]]
            semantic_ok = all(semantic(label, r) for r in [serial, drafted, *concurrent])
            row = {'case': label, 'serial': serial, 'drafted': drafted, 'concurrent': concurrent,
                   'visible_output_sha256': hashlib.sha256(expected).hexdigest(), 'byte_exact_matches': equal,
                   'semantic_passed': semantic_ok, 'passed': all(equal) and semantic_ok}
            report['cases'].append(row)
            save(destination, report)
            print(json.dumps({'case': label, 'passed': row['passed']}), flush=True)
        after = fresh_observations(op, library, digest, identity, time.time(), args.report_timeout)
        mesh_after = live_guard(cfg)
        report.update(after=after, mesh_after=mesh_after)
        verify_boot_qualification(qualification, mesh_after)
        health_after = request(endpoint, '/health', timeout=args.timeout)['response']
        report['health_after'] = health_after
        check_idle(health_after, parallel)
        require(health_after['requests_total'] - health_before['requests_total'] == expected_count == report['completed_requests'],
                'API request delta differs from this run: external traffic, reset, or missing requests')
        report.update(after=after, mesh_after=mesh_after, health_after=health_after,
                      request_delta=health_after['requests_total'] - health_before['requests_total'],
                      exclusive_api_requests_passed=True)
        report['function_payload'] = {}
        report['qp_after'] = {}
        report['native_payload'] = {}
        report['hardware_rule_deltas'] = tc_observation_deltas(mesh_before, mesh_after)
        for rank, node in enumerate(cfg['nodes']):
            name = node['name']
            require((before[name]['container_id'], before[name]['pid'], before[name]['image_id']) ==
                    (after[name]['container_id'], after[name]['pid'], after[name]['image_id']), 'Owned candidate process changed during validation')
            report['function_payload'][name] = function_deltas(before[name]['snapshot'], after[name]['snapshot'], args.minimum_bytes)
            report['qp_after'][name] = qp_proof(after[name], ordered_hcas(node), mesh_after['allowed_marker_pids_by_host'][name])
            if args.mode == 'rocenante':
                report['native_payload'][name] = native_deltas(before[name]['controller']['report']['path_counters'],
                    after[name]['controller']['report']['path_counters'], rank, ordered_hcas(node))
        if args.mode == 'rocenante':
            report['hardware_mesh_payload'] = hardware_payload_delta(mesh_before, mesh_after)
        else:
            # NCCL Ring uses direct neighbors; positive diagonal transit is a
            # native all-peer proof and is deliberately not required here.
            require(mesh_before['plan_id'] == mesh_after['plan_id'] and
                    mesh_before['network_boot_ids'] == mesh_after['network_boot_ids'], 'Mesh plan/boots changed')
            report['hardware_mesh_payload'] = {'required': False, 'scope': 'NCCL control; exact hardware-only TC attestations retained before/after'}
        require(experiment_identity(cfg, digest) == identity, 'Source/config/library/vendor identity changed during validation')
        require(all(row['passed'] for row in report['cases']), 'Model output equality or semantic truth check failed')
        report.update(qualified=True, passed=True, live_binding_passed=True, payload_all_hcas_passed=True,
                      qp_diagnostics_all_hcas_passed=True, host_configuration_unchanged=True,
                      native_payload_passed=args.mode == 'rocenante')
        return 0
    except Exception as error:
        report['failure'] = type(error).__name__ + ': ' + str(error)
        print(json.dumps({'passed': False, 'failure': report['failure'], 'receipt': str(destination)}), file=sys.stderr, flush=True)
        return 1
    finally:
        report['finished'] = utc()
        save(destination, report)
        save(output / 'changes.json', {'utc': report['finished'], 'created_directory': str(output),
            'created_files': ['model-qualification.json', 'changes.json'],
            'actions': 'Read-only SSH Docker/process/source/counter/QP/mesh observations, HTTP metadata and authorized correctness inference',
            'completed_chat_requests': report['completed_requests'], 'host_configuration_changes': False})
        print(json.dumps({'qualified': report['qualified'], 'receipt': str(destination)}), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
