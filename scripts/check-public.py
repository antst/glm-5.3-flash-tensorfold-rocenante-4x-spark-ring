#!/usr/bin/env python3
"""Offline checks for the separate RoCEnante recipe's portable sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {'.git', '__pycache__', 'local', 'state', 'logs', 'lib', 'models', 'cache', 'build', 'rigmark-source', 'node_modules', 'runs'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    files = {}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts) or not path.is_file():
            continue
        if path.name in ('release-manifest.json', 'SYSTEM-BUILD-CHANGES.txt') or path.name.startswith('config.local') or path.name in ('config.json', 'metadata.local.json'):
            continue
        if path.is_symlink():
            raise RuntimeError('Unexpected portable-source symlink: ' + str(relative))
        data = path.read_bytes()
        source = data.decode('utf-8')
        # Upstream sources can contain synthetic MACs and example home paths;
        # their identities and notices are verified separately, without editing.
        patterns = [r'-----BEGIN [A-Z ]*PRIVATE KEY-----', r'\bssh-(?:ed25519|rsa)\s+[A-Za-z0-9+/]{50,}={0,3}']
        if 'third_party' not in relative.parts:
            patterns += [r'192\.168\.[0-9]+\.[0-9]+', r'/home/[A-Za-z0-9_.-]+', r'\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b']
        if any(re.search(pattern, source) for pattern in patterns):
            raise RuntimeError('Private/site-specific material: ' + str(relative))
        if path.suffix == '.py':
            ast.parse(source, filename=str(relative))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        elif path.suffix == '.json':
            json.loads(source)
        files[str(relative)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    required = ('README.md', 'LICENSE', 'CREDITS.md', 'provenance.json', 'config.example.json',
                'scripts/network.py', 'scripts/experiment.py', 'scripts/rocenante_adapter.py',
                'scripts/runtime.py', 'scripts/build-nccl.sh', 'scripts/qualify.py',
                'scripts/rocenante-probe.py', 'docs/design.md', 'docs/network.md',
                'docs/benchmark-method.md', 'docs/system-changes.md')
    missing = [name for name in required if name not in files]
    if missing:
        raise RuntimeError('Missing maintained source: ' + ', '.join(missing))
    pins = json.loads((ROOT / 'provenance.json').read_text())
    if pins['sircl_enabled'] is not False or pins['nccl']['version'] != '2.32.3':
        raise RuntimeError('Transport identity differs from the selected current profile')
    vendor = ROOT / 'third_party/prepared-b12x'
    vendor_manifest = vendor / 'source-manifest.json'
    expected_vendor = '19b6189f2856655b6919bdcfcb4974bcf9dc7b71882f179b814de36c3615bf89'
    if hashlib.sha256(vendor_manifest.read_bytes()).hexdigest() != expected_vendor:
        raise RuntimeError('Prepared framework source inventory changed')
    bound = json.loads(vendor_manifest.read_text())['files']
    actual_vendor = {str(p.relative_to(vendor)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in vendor.rglob('*') if p.is_file() and p != vendor_manifest
                     and '__pycache__' not in p.parts}
    if bound != actual_vendor:
        raise RuntimeError('Prepared framework sources differ from their exact inventory')
    planner = ROOT / 'third_party/sparkring/fabric.py'
    if hashlib.sha256(planner.read_bytes()).hexdigest() != pins['network']['source_planner_sha256']:
        raise RuntimeError('Mesh planner changed')
    nccl = json.loads((ROOT / 'nccl-sources.json').read_text())
    patch = ROOT / nccl['patch']['path']
    if (nccl['commit'] != pins['nccl']['source_commit'] or
            hashlib.sha256(patch.read_bytes()).hexdigest() != pins['nccl']['patch_sha256']):
        raise RuntimeError('NCCL source/patch identity changed')
    receipt = {'schema': 'tensorfold-rocenante-source-files/v1', 'files': files}
    manifest = ROOT / 'release-manifest.json'
    if args.write_manifest:
        manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    elif json.loads(manifest.read_text()) != receipt:
        raise RuntimeError('Source manifest differs; review changes before regeneration')
    print(json.dumps({'passed': True, 'files': len(files), 'manifest_written': args.write_manifest}))


if __name__ == '__main__':
    main()
