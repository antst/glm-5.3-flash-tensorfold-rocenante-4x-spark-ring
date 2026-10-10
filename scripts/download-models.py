#!/usr/bin/env python3
"""Download pinned snapshots to local storage; no SSH or GPU work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import subprocess
import urllib.request

from common import MODEL_PINS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, help="local model directory; e.g. /models in a download container")
    parser.add_argument("--plan", action="store_true", help="show immutable pins/outputs without network or writes")
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.output_root or Path(config["model_root"])
    if not root.is_absolute():
        parser.error("--output-root must be absolute local storage")
    shared = Path(config["shared_root"]).resolve()
    resolved_root = root.resolve()
    if resolved_root == shared or shared in resolved_root.parents:
        parser.error("download/HF locks must stay on local storage outside shared_root")
    specs = [(repo, revision, name) for name, (repo, revision) in MODEL_PINS.items()]
    print("DFlash2 uses CC BY-NC-ND 4.0; this profile requires non-commercial use or a separate commercial license.")
    if args.plan:
        print(json.dumps({"snapshots": specs, "output_root": str(root),
                          "image_includes_weights": False, "network_calls": False}, indent=2))
        return
    ancestor = resolved_root
    while not ancestor.exists():
        ancestor = ancestor.parent
    filesystem = subprocess.check_output(["stat", "-f", "-c", "%T", str(ancestor)], text=True).strip().lower()
    if filesystem.startswith("nfs") or filesystem in ("cifs", "smb2", "smb3"):
        parser.error("download/HF caches require local storage, not " + filesystem)
    # Set cache locations before importing HF, whose constants are cached at import.
    os.environ["HF_HOME"] = str(root / ".hf-cache")
    os.environ["HF_HUB_CACHE"] = str(root / ".hf-cache" / "hub")
    os.environ["HF_XET_CACHE"] = str(root / ".hf-cache" / "xet")
    # This dependency is in the pinned image; help/plan need only the stdlib.
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def event(action, **details):
        with lock:
            with (root / "download-events.jsonl").open("a") as output:
                output.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "action": action,
                                         **details}, sort_keys=True) + "\n")

    event("download_root_created", directory=str(root), hf_home=os.environ["HF_HOME"])

    def download(spec):
        repo, revision, name = spec
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=90) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise RuntimeError("Model API revision mismatch")
        for entry in metadata["siblings"]:
            part = Path(entry["rfilename"])
            if part.is_absolute() or ".." in part.parts:
                raise RuntimeError("Unsafe model metadata file path")
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        (root / (name + "-source.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        event("snapshot_download_begin", repo=repo, revision=revision, directory=str(target))
        print(f"DOWNLOADING {repo}@{revision} into {target}", flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), max_workers=6)
        files = []
        for entry in metadata["siblings"]:
            filename = entry["rfilename"]
            path = target / filename
            size = entry.get("size", entry.get("lfs", {}).get("size"))
            if not path.is_file() or (size is not None and path.stat().st_size != size):
                raise RuntimeError(f"Missing or incorrect file: {path}")
            record = {"path": filename, "size": path.stat().st_size,
                      "source_lfs_sha256": entry.get("lfs", {}).get("sha256")}
            if path.stat().st_size < 64 * 1024 * 1024:
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(record)
        for index_path in target.glob("*.safetensors.index.json"):
            for shard in set(json.loads(index_path.read_text()).get("weight_map", {}).values()):
                part = Path(shard)
                if part.is_absolute() or ".." in part.parts or not (target / part).is_file():
                    raise RuntimeError("Unsafe/missing weight shard")
        receipt = {"repo": repo, "revision": revision, "directory": name, "complete": True,
                   "total_bytes": sum(item["size"] for item in files), "files": files,
                   "verification": "Pinned HF snapshot/Xet integrity; full API file list/sizes; local SHA256 for files smaller than64MiB",
                   "timestamp": datetime.now(timezone.utc).isoformat()}
        receipt_path = root / (name + "-receipt.json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        event("snapshot_download_complete", repo=repo, revision=revision, receipt=str(receipt_path),
              total_bytes=receipt["total_bytes"], files=len(files))
        print(f"COMPLETE {name}: {receipt['total_bytes']} bytes in {len(files)} files", flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(executor.map(download, specs))
    output = root / "download-complete.json"
    output.write_text(json.dumps(receipts, indent=2) + "\n")
    event("download_complete", receipt=str(output))


if __name__ == "__main__":
    main()
