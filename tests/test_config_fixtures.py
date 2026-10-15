"""CPU tests prove fixture separation without weakening declared-overlay admission."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from config_fixtures import ENGINE_FIELDS, example, loaded_example
import engine_binding


class ConfigFixtureTests(unittest.TestCase):
    def test_transport_fixture_removes_only_engine_fields_and_optional_tier(self):
        original = json.loads((ROOT / "config.example.json").read_text())
        self.assertTrue(all(field in original for field in ENGINE_FIELDS))
        stripped = example()
        self.assertEqual(stripped, {key:value for key,value in original.items() if key not in (*ENGINE_FIELDS, "tier")})
        with patch("subprocess.run", side_effect=AssertionError("host action forbidden")):
            parsed = loaded_example()
        self.assertNotIn("engine_overlay_root", parsed)

    def test_declared_unbuilt_overlay_is_still_rejected_by_real_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            overlay = {"engine_overlay_root": str(root / "missing-engine"),
                       "engine_manifest": str(root / "missing-manifest.json"),
                       "engine_manifest_sha256": "a" *64,
                       "engine_flags": {"TF_GLM_PROMPT_PARTIALS":"fp32"}}
            with patch("subprocess.run", side_effect=AssertionError("host action forbidden")), self.assertRaises(FileNotFoundError):
                loaded_example(overlay=overlay)

    def test_declared_bound_overlay_uses_actual_manifest_and_source_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);package = root / "engine/tensorfold"
            package.mkdir(parents=True);source = package / "__init__.py";source.write_text("# fixture\n")
            files = {"__init__.py": hashlib.sha256(source.read_bytes()).hexdigest()}
            manifest = root / "source-manifest.json"
            manifest.write_text(json.dumps({"schema":"glm-prefill-engine-source/v1","source_id":"fixture","files":files}))
            overlay = {"engine_overlay_root":str(package.parent),"engine_manifest":str(manifest),
                       "engine_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
                       "engine_flags":{"TF_GLM_PROMPT_PARTIALS":"fp32"}}
            with patch("subprocess.run", side_effect=AssertionError("host action forbidden")):
                cfg = loaded_example(overlay=overlay)
                self.assertEqual(engine_binding.identity(cfg, verify=True)["files"], files)
                source.write_text("# drift\n")
                with self.assertRaises(RuntimeError):
                    engine_binding.identity(cfg, verify=True)


if __name__ == "__main__":
    unittest.main()
