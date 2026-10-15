# SPDX-License-Identifier: Apache-2.0
"""NVMe-v3 packaging equivalence; stdlib/GNU patch only, never imports TensorFold."""
from __future__ import annotations
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
spec = importlib.util.spec_from_file_location("nvme_engine_builder", ENGINE / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
FINAL = "9c83663da0f875931e5e721ca42e4177a3a83e2d3f0a72abe935a1a5364f1f7b"


class NVMeAssets(unittest.TestCase):
    def test_nvme_exact_manifest_and_distinct_chain(self):
        before, final, template = builder.assets("4096-nvme")
        self.assertEqual(len(before), 439)
        self.assertEqual(len(final), 443)
        self.assertEqual(builder.sha(template), FINAL)
        self.assertEqual(json.loads(template.read_text())["source_id"], "mia107-tp4-prefill-nvme-tier-v3")
        prefill = builder.assets("4096")[1]
        self.assertEqual({p for p in prefill.keys() | final.keys() if prefill.get(p) != final.get(p)}, builder.NVME_MODIFIED)
        self.assertEqual(set(final) - set(prefill), {"cuda/spill_blocks.py", "cuda/tier_protocol.py"})

    def test_default_plan_is_current_without_process_or_creation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(builder.subprocess, "run", side_effect=AssertionError("plan ran process")):
            output = Path(temp) / "absent"
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(builder.main(["--output", str(output)]), 0)
            plan = json.loads(log.getvalue())
            self.assertEqual(plan["manifest"], "4096-nvme")
            self.assertEqual(plan["source_files"], 443)
            self.assertEqual(plan["source_manifest_sha256"], FINAL)
            self.assertFalse(output.exists())

    def test_default_manifest_asset_is_exact_copy(self):
        self.assertEqual((ENGINE / "source-manifest.json").read_bytes(), (ENGINE / "manifests/source-4096-nvme.json").read_bytes())

    def test_nvme_patch_tamper_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "assets"
            shutil.copytree(ENGINE, copied)
            with (copied / "patches/mia107-nvme-v3.patch").open("ab") as stream:
                stream.write(b"changed")
            output = Path(temp) / "output"
            with patch.object(builder, "ASSETS", copied), self.assertRaisesRegex(RuntimeError, "asset changed"):
                builder.build(output)
            self.assertFalse(output.exists())

    def test_full_asset_inventory_rejects_unexpected_file(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "assets"
            shutil.copytree(ENGINE, copied)
            (copied / "extra-input.py").write_text("not packaged")
            with patch.object(builder, "ASSETS", copied), self.assertRaisesRegex(RuntimeError, "asset inventory"):
                builder.assets("4096-nvme")

    def test_license_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "assets"
            shutil.copytree(ENGINE, copied)
            (copied / "licenses/TensorFold-LICENSE").write_text("changed")
            with patch.object(builder, "ASSETS", copied), self.assertRaisesRegex(RuntimeError, "asset inventory"):
                builder.assets("4096-nvme")


@unittest.skipUnless(os.environ.get("PREFILL_TEST_ORIGINAL"), "set PREFILL_TEST_ORIGINAL to immutable439-file Mia107 package")
class NVMeEquivalence(unittest.TestCase):
    def setUp(self):
        self.original = Path(os.environ["PREFILL_TEST_ORIGINAL"])
        self.before, self.after, _ = builder.assets("4096-nvme")
        builder.verify_tree(self.original, self.before)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_default_full439_to443_exact_bytes_and_both_steps(self):
        output = self.root / "nvme-default"
        with patch.object(builder, "extract_image", side_effect=AssertionError("offline build called Docker")):
            receipt = builder.build(output, source_package=self.original)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["manifest_variant"], "4096-nvme")
        self.assertEqual(receipt["source_manifest_sha256"], FINAL)
        self.assertEqual([p["output_file_count"] for p in receipt["patch_steps"]], [441, 443])
        self.assertEqual(builder.inventory(output / "tensorfold"), self.after)
        self.assertEqual(builder.inventory(self.original), self.before)
        self.assertFalse(receipt["gpu_work"])
        self.assertEqual((output / "source-manifest.json").read_bytes(), (ENGINE / "manifests/source-4096-nvme.json").read_bytes())

    def test_second_patch_failure_retains_failure_without_publication(self):
        real = builder.subprocess.run
        def run(command, **kwargs):
            if command[-1].endswith("mia107-nvme-v3.patch"):
                return builder.subprocess.CompletedProcess(command, 1, "controlled second-patch failure", "")
            return real(command, **kwargs)
        output = self.root / "second-failed"
        with patch.object(builder.subprocess, "run", side_effect=run), self.assertRaisesRegex(RuntimeError, "nvme-v3.patch"):
            builder.build(output, source_package=self.original)
        self.assertFalse((output / "tensorfold").exists())
        receipt = json.loads((output / "build-receipt.json").read_text())
        self.assertFalse(receipt["passed"])
        self.assertEqual(len(receipt["patch_steps"]), 1)
        self.assertEqual(builder.inventory(self.original), self.before)

    def test_unchanged_file_drift_after_patch_is_rejected_by_full_tree_check(self):
        real = builder.subprocess.run
        def run(command, **kwargs):
            result = real(command, **kwargs)
            if command[-1].endswith("mia107-nvme-v3.patch"):
                work = Path(command[command.index("-d") + 1])
                with (work / "tensorfold/cli.py").open("ab") as stream:
                    stream.write(b"\n# unexpected unchanged-file mutation\n")
            return result
        output = self.root / "drift-failed"
        with patch.object(builder.subprocess, "run", side_effect=run), self.assertRaisesRegex(RuntimeError, "hash/cardinality"):
            builder.build(output, source_package=self.original)
        self.assertFalse((output / "tensorfold").exists())
        self.assertFalse(json.loads((output / "build-receipt.json").read_text())["passed"])

    def test_assets_rechecked_before_publication(self):
        copied = self.root / "assets"
        shutil.copytree(ENGINE, copied)
        real = builder.subprocess.run
        def run(command, **kwargs):
            result = real(command, **kwargs)
            if command[-1].endswith("mia107-nvme-v3.patch"):
                with (copied / "README.md").open("ab") as stream:
                    stream.write(b"\nchanged during build\n")
            return result
        output = self.root / "asset-race"
        with patch.object(builder, "ASSETS", copied), patch.object(builder.subprocess, "run", side_effect=run), self.assertRaisesRegex(RuntimeError, "asset inventory"):
            builder.build(output, source_package=self.original)
        self.assertFalse((output / "tensorfold").exists())


if __name__ == "__main__":
    unittest.main()
