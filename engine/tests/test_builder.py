# SPDX-License-Identifier: Apache-2.0
"""Offline source reconstruction/rejection tests; no Docker/Torch/SSH/GPU calls."""
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
spec = importlib.util.spec_from_file_location("prefill_engine_builder", ENGINE / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PackageTests(unittest.TestCase):
    def test_assets_and_variants_are_consistent(self):
        a = builder.assets("2048")
        b = builder.assets("4096")
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertEqual(json.loads(a[2].read_text()), json.loads(b[2].read_text()))
        self.assertNotEqual(a[2].read_bytes(), b[2].read_bytes())
        self.assertEqual(len(a[0]), 439)
        self.assertEqual(len(a[1]), 441)

    def test_plan_does_not_create_output_or_invoke_processes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(builder.subprocess, "run", side_effect=AssertionError("plan ran process")):
            out = Path(temp) / "planned"
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(builder.main(["--output", str(out), "--manifest", "2048"]), 0)
            self.assertEqual(json.loads(log.getvalue())["action"], "offline-plan")
            self.assertFalse(out.exists())

    def test_unsafe_manifest_names_and_hashes(self):
        for name in ["../evil", "/absolute", "./odd", "a/../odd", "a\\evil", "", ".", "a:b"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.validate_names({name: "0" * 64})
        with self.assertRaises(ValueError):
            builder.validate_names({"file.py": "invalid"})

    def test_existing_output_rejected_without_process(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(builder.subprocess, "run", side_effect=AssertionError("ran process")):
            with self.assertRaisesRegex(ValueError, "existing output"):
                builder.build(Path(temp))

    def test_modified_packaged_patch_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "assets"
            shutil.copytree(ENGINE, copied)
            with (copied / "patches/mia107-prefill.patch").open("ab") as stream:
                stream.write(b"tampered")
            out = Path(temp) / "output"
            with patch.object(builder, "ASSETS", copied), self.assertRaisesRegex(RuntimeError, "asset changed"):
                builder.build(out)
            self.assertFalse(out.exists())

    def test_image_extraction_command_has_no_gpu_network_or_imports(self):
        class Result:
            returncode = 0
            stdout = "extracted"
            stderr = ""
        with patch.object(builder.subprocess, "run", return_value=Result()) as run:
            builder.extract_image(Path("/tmp/explicit-output"))
        command = run.call_args.args[0]
        self.assertIn("--pull=never", command)
        self.assertIn("--read-only", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[command.index("--entrypoint") + 1], "python3")
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("import torch", builder.EXTRACT)
        self.assertNotIn("import tensorfold", builder.EXTRACT)
        self.assertEqual(command[-4], builder.IMAGE)


@unittest.skipUnless(os.environ.get("PREFILL_TEST_ORIGINAL"), "set PREFILL_TEST_ORIGINAL to immutable439-file Mia107 package")
class SnapshotEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.original = Path(os.environ["PREFILL_TEST_ORIGINAL"])
        self.before, self.after, _ = builder.assets("4096")
        builder.verify_tree(self.original, self.before)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_full439_to441_byte_equivalence_both_runtime_manifests(self):
        for variant in ["2048", "4096"]:
            with self.subTest(variant=variant):
                out = self.root / ("overlay-" + variant)
                with patch.object(builder, "extract_image", side_effect=AssertionError("offline build used Docker")):
                    receipt = builder.build(out, variant=variant, source_package=self.original)
                self.assertTrue(receipt["passed"])
                self.assertEqual(builder.inventory(out / "tensorfold"), self.after)
                self.assertEqual((out / "source-manifest.json").read_bytes(), (ENGINE / "manifests" / ("source-" + variant + ".json")).read_bytes())
                self.assertEqual(builder.inventory(self.original), self.before)
                parsed = json.loads((out / "source-manifest.json").read_text())
                self.assertEqual(parsed["schema"], "glm-prefill-engine-source/v1")
                self.assertEqual(parsed["files"], builder.inventory(out / "tensorfold"))

    def altered_copy(self):
        copy = self.root / "original-copy"
        shutil.copytree(self.original, copy)
        return copy

    def assert_preimage_rejected(self, copy):
        out = self.root / "failed-output"
        with patch.object(builder, "extract_image", side_effect=AssertionError("preimage refusal used Docker")):
            with self.assertRaises(RuntimeError):
                builder.build(out, source_package=copy)
        self.assertFalse(out.exists())

    def test_same_cardinality_modified_source_rejected(self):
        copy = self.altered_copy()
        with (copy / "cli.py").open("ab") as stream:
            stream.write(b"\n# unexpected change\n")
        self.assert_preimage_rejected(copy)

    def test_missing_source_rejected(self):
        copy = self.altered_copy()
        (copy / "cli.py").unlink()
        self.assert_preimage_rejected(copy)

    def test_extra_source_rejected(self):
        copy = self.altered_copy()
        (copy / "extra.py").write_text("extra")
        self.assert_preimage_rejected(copy)

    def test_symlink_source_rejected(self):
        copy = self.altered_copy()
        (copy / "cli.py").unlink()
        (copy / "cli.py").symlink_to(self.original / "cli.py")
        self.assert_preimage_rejected(copy)

    def test_patch_failure_leaves_failed_receipt_without_published_package(self):
        class Failed:
            returncode = 1
            stdout = "controlled failure"
            stderr = ""
        out = self.root / "failed-patch"
        with patch.object(builder.subprocess, "run", return_value=Failed()), self.assertRaisesRegex(RuntimeError, "patch failed"):
            builder.build(out, source_package=self.original)
        self.assertFalse((out / "tensorfold").exists())
        self.assertFalse(json.loads((out / "build-receipt.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
