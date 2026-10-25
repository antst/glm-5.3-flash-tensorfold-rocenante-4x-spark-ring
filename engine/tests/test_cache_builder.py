# SPDX-License-Identifier: Apache-2.0
"""Cache-chain packaging and failure handling, without TensorFold/Docker imports."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ENGINE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cache_engine_builder', ENGINE / 'build.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
FINAL = '7cfaeb1b24790d31e9bc1508c0b4eef2528fdf0c1157c0530f1e2d820117c31a'


class CacheAssets(unittest.TestCase):
    def test_current_default_plan_is_offline_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(builder.subprocess, 'run', side_effect=AssertionError('plan ran a process')):
            output = Path(temp) / 'absent'
            with contextlib.redirect_stdout(io.StringIO()) as text:
                self.assertEqual(builder.main(['--output', str(output)]), 0)
            result = json.loads(text.getvalue())
            self.assertEqual(result['manifest'], '4096-cache')
            self.assertEqual(result['source_files'], 446)
            self.assertEqual(result['source_manifest_sha256'], FINAL)
            self.assertFalse(output.exists())

    def test_new_layers_preserve_every_native_kernel_file(self):
        _, old, _ = builder.assets('4096-nvme')
        _, current, template = builder.assets('4096-cache')
        for name in old:
            if Path(name).suffix in ('.cu', '.cuh', '.cpp', '.h'):
                self.assertEqual(old[name], current[name], name)
        self.assertEqual((ENGINE / 'source-manifest.json').read_bytes(), template.read_bytes())

    def test_either_new_patch_tamper_is_rejected_before_output_creation(self):
        for name in ('mia107-cache-foundation.patch', 'mia107-cache-backports.patch'):
            with self.subTest(patch=name), tempfile.TemporaryDirectory() as temp:
                copied = Path(temp) / 'assets'
                shutil.copytree(ENGINE, copied)
                with (copied / 'patches' / name).open('ab') as out:
                    out.write(b'changed')
                output = Path(temp) / 'output'
                with patch.object(builder, 'ASSETS', copied), self.assertRaisesRegex(RuntimeError, 'asset changed'):
                    builder.build(output)
                self.assertFalse(output.exists())


@unittest.skipUnless(os.environ.get('PREFILL_TEST_ORIGINAL'), 'set PREFILL_TEST_ORIGINAL to the original439-file package')
class CacheEquivalence(unittest.TestCase):
    def setUp(self):
        self.original = Path(os.environ['PREFILL_TEST_ORIGINAL'])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'output'

    def test_full_chain_reproduces_deployed_tree_and_manifest(self):
        _, wanted, template = builder.assets('4096-cache')
        with patch.object(builder, 'extract_image', side_effect=AssertionError('offline build called Docker')):
            result = builder.build(self.output, source_package=self.original)
        self.assertTrue(result['passed'])
        self.assertEqual([x['output_file_count'] for x in result['patch_steps']], [441, 443, 446, 446])
        self.assertEqual(builder.inventory(self.output / 'tensorfold'), wanted)
        self.assertEqual((self.output / 'source-manifest.json').read_bytes(), template.read_bytes())
        self.assertEqual(builder.sha(template), FINAL)
        builder.verify_tree(self.original, builder.assets('2048')[0])

    def test_failed_last_patch_never_publishes_an_incomplete_engine(self):
        real = builder.subprocess.run
        def run(command, **kwargs):
            if command[-1].endswith('mia107-cache-backports.patch'):
                return builder.subprocess.CompletedProcess(command, 1, 'injected patch refusal', '')
            return real(command, **kwargs)
        with patch.object(builder.subprocess, 'run', side_effect=run), self.assertRaisesRegex(RuntimeError, 'patch failed'):
            builder.build(self.output, source_package=self.original)
        self.assertFalse((self.output / 'tensorfold').exists())
        result = json.loads((self.output / 'build-receipt.json').read_text())
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['patch_steps']), 3)

    def test_unrelated_source_drift_after_final_patch_is_rejected(self):
        real = builder.subprocess.run
        def run(command, **kwargs):
            result = real(command, **kwargs)
            if command[-1].endswith('mia107-cache-backports.patch'):
                work = Path(command[command.index('-d') + 1])
                with (work / 'tensorfold/cli.py').open('ab') as out:
                    out.write(b'\n# unexpected source drift\n')
            return result
        with patch.object(builder.subprocess, 'run', side_effect=run), self.assertRaisesRegex(RuntimeError, 'hash/cardinality'):
            builder.build(self.output, source_package=self.original)
        self.assertFalse((self.output / 'tensorfold').exists())


if __name__ == '__main__':
    unittest.main()
