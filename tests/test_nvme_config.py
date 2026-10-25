"""CPU-only checks for the portable NVMe example and actual launch/stop policy."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from config_fixtures import config_file, example
from common import load_config
from cluster import Operator
from experiment import prepare_config, stop_owned

MANIFEST = "7cfaeb1b24790d31e9bc1508c0b4eef2528fdf0c1157c0530f1e2d820117c31a"


class NvmeConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        package = self.base / "engine/tensorfold"
        package.mkdir(parents=True)
        source = package / "__init__.py"
        source.write_text("# bounded CPU configuration fixture\n")
        manifest = self.base / "engine/source-manifest.json"
        manifest.write_text(json.dumps({"schema": "glm-prefill-engine-source/v1", "source_id": "fixture",
                                       "files": {"__init__.py": hashlib.sha256(source.read_bytes()).hexdigest()}}))
        self.overlay = {"engine_overlay_root": str(package.parent), "engine_manifest": str(manifest),
                        "engine_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "engine_flags": {"TF_GLM_POOL_TOKENS": "6098944"}}

    def parsed(self, mutate=None):
        value = example(overlay=self.overlay, tier=True)
        if mutate:
            mutate(value)
        path = self.base / "config.json"
        path.write_text(json.dumps(value))
        with patch("subprocess.run", side_effect=AssertionError("host operation forbidden")):
            return prepare_config(path)

    def test_portable_example_binds_final_manifest_pool_and_disk_budgets(self):
        value = json.loads((ROOT / "config.example.json").read_text())
        self.assertEqual(value["engine_manifest_sha256"], MANIFEST)
        self.assertEqual(value["engine_flags"]["TF_GLM_POOL_TOKENS"], "6098944")
        self.assertTrue(value["engine_overlay_root"].endswith("/local/engine-nvme"))
        self.assertEqual(value["tier"], {"root": "/var/tmp/spark-ring-nvme-tier-shared-history-v3", "gib": 256,
                                        "min_free_gib": 64, "min_tokens": 8192, "turns": 4,
                                        "host_mib": 512, "read_host_mib": 256,
                                        "stage_mib": 256, "read_blocks": 16})

    def test_actual_loader_rejects_bad_tier_caps_paths_and_missing_pool(self):
        for key, value in (("root", "/shared/cache"), ("gib", 257), ("gib", True),
                           ("min_free_gib", 32), ("host_mib", 0), ("read_host_mib", 512),
                           ("stage_mib", 1024), ("read_blocks", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parsed(lambda cfg, k=key, v=value: cfg["tier"].__setitem__(k, v))
        with self.assertRaises(ValueError):
            self.parsed(lambda cfg: cfg["engine_flags"].pop("TF_GLM_POOL_TOKENS"))

    def test_actual_commands_use_local_rw_store_and_all_rank_memory_limits(self):
        cfg = self.parsed()
        with patch("subprocess.run", side_effect=AssertionError("host operation forbidden")), \
                patch.object(Path, "write_text", side_effect=AssertionError("command generation must be read-only")):
            commands = [Operator(cfg).command(rank, Path("/unused/libnccl.so.2"), "TEST")[1] for rank in range(4)]
        for argv in commands:
            self.assertIn("type=bind,src=/var/tmp/spark-ring-nvme-tier-shared-history-v3,dst=/snapshots", argv)
            for setting in ("TF_GLM_POOL_TOKENS=6098944", "TF_GLM_SPILL_HOST_MIB=512",
                            "TF_GLM_SPILL_READ_HOST_MIB=256", "TF_GLM_SPILL_STAGE_MIB=256",
                            "TF_GLM_SPILL_READ_BLOCKS=16", "TF_GLM_SPILL_TURNS=4",
                            "TF_GLM_SPILL_LOAD_TIMEOUT_S=300"):
                self.assertEqual(argv.count(setting), 1)
            for flag, wanted in (("--snapshot-dir", "/snapshots"), ("--spill-gib", "256"),
                                 ("--spill-min-tokens", "8192"), ("--spill-min-free-gib", "64")):
                self.assertEqual(argv[argv.index(flag) + 1], wanted)

    def test_legacy_transport_fixture_does_not_enable_tier_implicitly(self):
        value = example()
        self.assertNotIn("tier", value)
        self.assertNotIn("engine_flags", value)
        with self.assertRaises(ValueError):
            example(tier=True)

    def test_tier_stop_finishes_rank_zero_before_stopping_followers(self):
        class FakeOperator:
            config = {"tier": {"root": "/var/tmp/spark-ring-nvme-tier-shared-history-v3"}}
            calls = []
            def stop_one(self, rank):
                self.calls.append(("stop", rank))
                return {"rank": rank}
            def parallel(self, action):
                self.calls.append(("parallel", None))
                return {rank: action(rank) for rank in range(4)}
        op = FakeOperator()
        result = stop_owned(op)
        self.assertEqual(op.calls, [("stop", 0), ("parallel", None),
                                    ("stop", 1), ("stop", 2), ("stop", 3)])
        self.assertEqual(result, {rank: {"rank": rank} for rank in range(4)})


if __name__ == "__main__":
    unittest.main()
