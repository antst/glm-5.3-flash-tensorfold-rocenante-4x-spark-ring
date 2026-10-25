#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run cache regressions against an explicit built overlay; no model or GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path,
                        help='built directory containing tensorfold/ and source-manifest.json')
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    import build
    _, wanted, template = build.assets('4096-cache')
    build.verify_tree(root / 'tensorfold', wanted)
    if (root / 'source-manifest.json').read_bytes() != template.read_bytes():
        raise RuntimeError('cache tests require the exact current manifest')
    tests = Path(__file__).resolve().parent / 'cache-tests'
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TENSORFOLD_CACHE_TEST_ROOT=str(root))
    checks = [
        ['-m', 'unittest', 'discover', '-s', str(tests), '-v'],
        [str(tests / 'kept_reasoning_check.py'), '--source-root', str(root)],
        [str(tests / 'test_queued_cancellation.py'), '--source-root', str(root)],
        [str(tests / 'chat_continued_check.py')],
        [str(tests / 'fork_point_check.py')],
    ]
    for check in checks:
        print(json.dumps({'check': check}), flush=True)
        result = subprocess.run([sys.executable, '-B', *check], env=env, timeout=300)
        if result.returncode:
            return result.returncode
    print('ALL_CACHE_REGRESSIONS_PASS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
