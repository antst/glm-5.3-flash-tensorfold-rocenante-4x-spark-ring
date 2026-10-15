#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the frozen Mia107 prefill and NVMe-v3 source overlays. Default --plan is offline."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

ASSETS = Path(__file__).resolve().parent
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14b5072e8e9918f1080153f57321415a387bc7d78cf9c9b44a7aeae2d4399ccf"
EXPECTED = {
    "source-before.json": "b4731db28fb34089b6f7be1a1cb8cdf1433f2f2a93704bd9b1d0153e6e0c1c83",
    "source-after.json": "ef0ff873b6af918d77db0395fc80dd1d12ac30e0278426a58dc6b6852417d5a4",
    "patches/mia107-prefill.patch": "dd1e8848d19c8a084a713bd7220a6e16bcaa1235f91d5f961105ba1652ec95c4",
    "manifests/source-2048.json": "3abe7f60184aaaaed1f97eca49ae4fdfe7ba08a0267a722b25d7e7470aadd958",
    "manifests/source-4096.json": "75c0035ad3ab4ac4950a9250af8f1ac6ea4ae5aacfb92d84bd3aacbbaf099666",
    "source-after-nvme.json": "f31e0d200755083a42d6290323ee8682b37e24d9091ae4e4b9782c4701f11ee0",
    "patches/mia107-nvme-v3.patch": "05c4a44926878b18d2e0514904a04db18f9784bcfc2265eaeae3454c47e01f94",
    "manifests/source-4096-nvme.json": "9c83663da0f875931e5e721ca42e4177a3a83e2d3f0a72abe935a1a5364f1f7b",
    "source-manifest.json": "9c83663da0f875931e5e721ca42e4177a3a83e2d3f0a72abe935a1a5364f1f7b",
}
MODIFIED = {"families/glm5_next/cuda/" + name for name in
            ("engine.py", "forward.py", "hcsplit.py", "latent.py", "sparse.py", "own_front.py", "microbatch.py")}

NVME_MODIFIED = {"cuda/spill_blocks.py", "cuda/tier_protocol.py", "server/metrics.py",
                 "families/glm5_next/cuda/engine.py", "families/glm5_next/cuda/multi.py"}
VARIANTS = ("2048", "4096", "4096-nvme")
DEFAULT_VARIANT = "4096-nvme"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_names(files):
    if not isinstance(files, dict) or not files:
        raise ValueError("empty source manifest")
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (not isinstance(name, str) or path.is_absolute() or name in ("", ".")
                or ".." in path.parts or "\\" in name or ":" in name
                or str(path) != name or not re.fullmatch("[0-9a-f]{64}", str(digest))):
            raise ValueError("unsafe source manifest entry")


def assets(variant):
    if variant not in VARIANTS:
        raise ValueError("manifest variant must be2048,4096 or4096-nvme")
    for name, wanted in EXPECTED.items():
        path = ASSETS / name
        if path.is_symlink() or sha(path) != wanted:
            raise RuntimeError("packaged source asset changed: " + name)
    # Inventory includes documentation, licenses, tests and builder itself. Critical
    # source/patch hashes above remain pinned independently of this self inventory.
    package = json.loads((ASSETS / "package-manifest.json").read_text())
    if package.get("schema") != "glm-prefill-nvme-public-engine-package/v2":
        raise ValueError("wrong engine package inventory schema")
    validate_names(package.get("files"))
    actual_package = inventory(ASSETS)
    actual_package.pop("package-manifest.json", None)
    if actual_package != package["files"]:
        raise RuntimeError("packaged asset inventory/hash mismatch")
    before = json.loads((ASSETS / "source-before.json").read_text())
    prefill = json.loads((ASSETS / "source-after.json").read_text())["files"]
    nvme_document = json.loads((ASSETS / "source-after-nvme.json").read_text())
    nvme = nvme_document["files"]
    for names in (before, prefill, nvme):
        validate_names(names)
    if len(before) != 439 or len(prefill) != 441 or len(nvme) != 443:
        raise ValueError("wrong source cardinality")
    changed = lambda a, b: {name for name in set(a) | set(b) if a.get(name) != b.get(name)}
    if changed(before, prefill) != MODIFIED:
        raise ValueError("prefill port no longer modifies exactly the seven frozen files")
    if changed(prefill, nvme) != NVME_MODIFIED:
        raise ValueError("NVMe port no longer modifies exactly the five frozen files")
    if (nvme_document.get("file_count") != 443
            or nvme_document.get("base_manifest_sha256") != EXPECTED["manifests/source-4096.json"]
            or nvme_document.get("source_manifest_sha256") != EXPECTED["manifests/source-4096-nvme.json"]):
        raise ValueError("NVMe source chain differs from frozen inventories")
    after = nvme if variant == "4096-nvme" else prefill
    template = ASSETS / "manifests" / ("source-" + variant + ".json")
    value = json.loads(template.read_text())
    source_id = "mia107-tp4-prefill-nvme-tier-v3" if variant == "4096-nvme" else "mia107-tp4-prefill-port-v1"
    if (value.get("schema") != "glm-prefill-engine-source/v1"
            or value.get("source_id") != source_id or value.get("files") != after):
        raise ValueError("runtime manifest differs from frozen output tree")
    return before, after, template


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source package must be a regular directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("source symlink refused: " + str(path.relative_to(root)))
        if path.is_file():
            files[str(path.relative_to(root))] = sha(path)
        elif not path.is_dir():
            raise RuntimeError("non-regular source entry")
    return files


def verify_tree(root, wanted):
    actual = inventory(root)
    if actual != wanted:
        changed = sorted(name for name in set(actual) | set(wanted) if actual.get(name) != wanted.get(name))
        raise RuntimeError("source hash/cardinality mismatch: " + ", ".join(changed[:12]))
    return actual


# This extraction program uses only stdlib metadata/filesystem calls. It never imports TensorFold/Torch/CUDA.
EXTRACT = r"""
import importlib.metadata, json, pathlib, shutil
package = pathlib.Path(importlib.metadata.distribution('tensorfold').locate_file('tensorfold'))
if package.is_symlink() or not package.is_dir():
    raise RuntimeError('installed TensorFold package path is not a regular directory')
out = pathlib.Path('/extract/tensorfold')
out.mkdir()
for path in package.rglob('*'):
    relative = path.relative_to(package)
    if '__pycache__' in relative.parts or path.suffix == '.pyc':
        continue
    if path.is_symlink():
        raise RuntimeError('installed source symlink refused')
    target = out / relative
    if path.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    elif path.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    else:
        raise RuntimeError('installed non-regular source entry refused')
print(json.dumps({'extracted': True, 'torch_imported': False, 'cuda_imported': False}))
"""


def extract_image(work):
    # --pull=never makes disk-changing image download a separate explicit operator command.
    command = ["docker", "run", "--rm", "--pull=never", "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user",
               str(os.getuid()) + ":" + str(os.getgid()), "--tmpfs", "/tmp:rw,nosuid,nodev",
               "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python3", "--mount",
               "type=bind,src=" + str(work) + ",dst=/extract", IMAGE, "-B", "-c", EXTRACT]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("CPU-only pinned-image extraction failed: " + result.stderr.strip())
    return {"image": IMAGE, "gpu_devices_requested": False, "network": "none",
            "tensorfold_or_torch_imported": False, "extraction_stdout": result.stdout.strip()}


def build(output, *, variant=DEFAULT_VARIANT, source_package=None):
    before, after, template = assets(variant)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("refusing existing output directory")
    source = Path(source_package).absolute() if source_package is not None else None
    if source is not None:
        if output == source or output.is_relative_to(source):
            raise ValueError("output cannot be inside source package")
        verify_tree(source, before)
    if shutil.which("patch") is None:
        raise RuntimeError("GNU patch is required")
    # Validate every packaged input before creating the new output directory.
    output.mkdir(parents=True)
    receipt = {"schema": "glm-prefill-nvme-engine-build/v2", "passed": False,
               "base_image": IMAGE, "manifest_variant": variant,
               "source_manifest_sha256": sha(template), "input_sha256": dict(EXPECTED),
               "before_file_count": len(before), "after_file_count": len(after),
               "intermediate_file_count": 441, "patch_steps": [],
               "source_mode": "offline-source-package" if source is not None else "cpu-only-pinned-image",
               "gpu_work": False, "performance_qualified": False}
    try:
        with tempfile.TemporaryDirectory(prefix=".engine-build-", dir=output) as temp:
            work = Path(temp)
            if source is None:
                receipt["extraction"] = extract_image(work)
            else:
                shutil.copytree(source, work / "tensorfold")
            verify_tree(work / "tensorfold", before)
            prefill = json.loads((ASSETS / "source-after.json").read_text())["files"]
            steps = [("mia107-prefill.patch", prefill)]
            if variant == "4096-nvme":
                steps.append(("mia107-nvme-v3.patch", after))
            logs = []
            for patch_name, wanted in steps:
                proc = subprocess.run(["patch", "--batch", "--forward", "--fuzz=0",
                                       "--no-backup-if-mismatch", "--reject-file=/dev/null", "-p1",
                                       "-d", str(work), "-i", str(ASSETS / "patches" / patch_name)],
                                      capture_output=True, text=True, timeout=60)
                log = proc.stdout + proc.stderr
                logs.append(patch_name + "\n" + log)
                (output / "patch.log").write_text("\n".join(logs))
                (output / (patch_name + ".log")).write_text(log)
                if proc.returncode:
                    raise RuntimeError("frozen source patch failed: " + patch_name)
                verify_tree(work / "tensorfold", wanted)
                receipt["patch_steps"].append({"patch": patch_name,
                    "sha256": EXPECTED["patches/" + patch_name],
                    "output_file_count": len(wanted), "all_source_hashes_verified": True})
            verify_tree(work / "tensorfold", after)
            if source is not None:
                verify_tree(source, before)
            assets(variant)
            shutil.copyfile(template, output / "source-manifest.json")
            if sha(output / "source-manifest.json") != EXPECTED["manifests/source-" + variant + ".json"]:
                raise RuntimeError("output manifest changed during build")
            shutil.move(str(work / "tensorfold"), str(output / "tensorfold"))
            verify_tree(output / "tensorfold", after)
            receipt["passed"] = True
            receipt["all_source_hashes_verified"] = True
    except BaseException as exc:
        receipt["failure"] = str(exc)
        raise
    finally:
        (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="NEW private overlay directory")
    parser.add_argument("--manifest", choices=VARIANTS, default=DEFAULT_VARIANT,
                        help="4096-nvme (default) adds NVMe-v3;2048/4096 retain the original prefill tree")
    parser.add_argument("--source-package", type=Path,
                        help="offline439-file preimage package; omit to extract the pinned local image")
    parser.add_argument("--run", action="store_true", help="actually create overlay; omitted means offline plan")
    parser.add_argument("--plan", action="store_true", help="explicit offline plan (default)")
    args = parser.parse_args(argv)
    if args.plan and args.run:
        parser.error("choose --plan or --run")
    before, after, template = assets(args.manifest)
    if not args.run:
        print(json.dumps({"action": "offline-plan", "image": IMAGE, "manifest": args.manifest,
                          "source_manifest_sha256": sha(template), "source_files": len(after),
                          "preimage_files": len(before), "output": str(args.output),
                          "source_mode": "offline-source-package" if args.source_package else "cpu-only-image",
                          "GPU_or_SSH_or_network_actions": False}, indent=2))
        return 0
    print(json.dumps(build(args.output, variant=args.manifest, source_package=args.source_package), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
