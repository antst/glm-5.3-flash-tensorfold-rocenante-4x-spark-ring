# Reproducible Mia107 prefill and cache source overlays

The current default, `4096-cache`, reconstructs the exact deployed **446-file**
overlay with manifest SHA256
`7cfaeb1b24790d31e9bc1508c0b4eef2528fdf0c1157c0530f1e2d820117c31a`.
The verified chain is original image (439 files), prefill (441), NVMe-v3 (443),
deployed cache foundation (446), then four upstream cache backports (446).
Every intermediate and final file is hash-checked. The foundation includes the
already-deployed allocator reclamation, SSE handling, shared token histories and
two-hour recency journal; the last patch contains only the four cache fixes.
See `cache-provenance.json` for changed paths, credits and exact patch identities.

The explicit `2048`, `4096` and `4096-nvme` variants retain their historical
source trees and byte-for-byte original manifest templates. No model weights,
CUDA runtime binaries, network changes or service lifecycle actions are included.

```sh
python3 -B engine/build.py --output local/engine-nvme --plan
python3 -B engine/build.py --output local/engine-nvme --manifest 4096-cache --run
```

The second command uses the already-present pinned image through the CPU-only
extraction path described below. For a separately retained exact original439-file
package, supply `--source-package /absolute/original/tensorfold`; this performs
all four patches and checks entirely offline without Docker. The builder never
pulls an image or imports TensorFold/Torch. Python 3.10+ and GNU `patch` are required.
The complete package asset inventory includes this documentation, notices,
licenses, builder and tests; critical patch/manifest hashes are also pinned in
`build.py`. Builds reject altered/missing/extra source files, symlinks, asset drift
and existing outputs. Each patch and source-check result is retained in
`build-receipt.json` and per-patch logs. A failed receipt does not qualify an overlay.

The historical NVMe-v3 layer is a locally corrected derivative of MiaAI-Lab PR124 rather than a raw PR
application. It stores bounded blocks of cache rows plus the snapshot's fixed
state and token IDs on each rank's local disk; restores require a common committed
computation generation across all four ranks. It uses the new
`tensorfold-cuda-spill-blocks-generation-2` namespace and control protocol 3, so raw
PR snapshots are not accepted as equivalent. Bounded staging, all-rank admission
and release backpressure remain necessary. The engine source does not enable or
size NVMe storage by itself; use separately qualified runtime flags, local disk
paths and memory budgets. RAM capacity, full-context accuracy, restart and serving
performance require their own model/hardware evidence. See `nvme-provenance.json`.
The exact historical NVMe manifest bytes are retained, including its original
qualification note; builder success is a source-equivalence result only.

The current cache layer moves repeated token histories and large checkpoint
metadata out of the RAM index. RAM and disk hits update live LRU immediately;
changed recency timestamps are coalesced into a transaction every **7,200 seconds**
and flushed on clean shutdown. Cache hits do not rewrite KV blocks. An abrupt
exit can lose unflushed recency, while already durable checkpoints remain intact.
The storage format is `tensorfold-cuda-spill-blocks-generation-3-shared-history`.
Use a separate host-local root for this format; old generation-2 snapshots are
not automatically migrated. The exact compatibility alias in the current
manifest preserves checkpoints from the matching shared-history releases only;
it does not authorize arbitrary source, layout or model substitutions.

The four backports prevent repeated disk reads on `NoRoom` retries, retain chat
identity through recoverable continuations, protect fork points as shared
prefixes, and restore reasoning in one pass over a tool history. They are adapted
from Mia main at `bf5f94523a8251029cedae86bf61f8ce37fac218`; no KV sharding is enabled.
The [sanitized validation summary](../results/cache-backports-20261025.json)
records the actual scope: 117 CPU regression checks, 300 reasoning histories,
both native transports and a disk/RAM replay with 15,936 cached tokens. This does
not rerun the historical full-capacity or throughput benchmarks.

To rerun the portable cache regressions in a CPU environment with the pinned
image's Torch/NumPy/safetensors dependencies, using a built overlay:

```sh
python3 -B engine/check_cache.py --source-root local/engine-nvme
```

The runner hides CUDA devices and loads no model. On a GPU host, use a container
without GPU devices or model mounts. Run the builder's stdlib equivalence tests
separately; they require no Torch installation.

To verify all three full builds and rejection tests with an immutable439-file
preimage:

```sh
PREFILL_TEST_ORIGINAL=/absolute/original/tensorfold python3 -B -m unittest discover -s engine/tests -v
```

# Retained prefill overlay details

This directory reconstructs the exact seven-file port of selected MiaAI-Lab TP4
optimizations onto the pinned Mia107 image. It distributes a source patch and
hash manifests, not a replacement runtime image or model weights. The existing
Torch, CUDA, TensorFold, DFlash and native-transport dependencies stay in the
pinned image. Runtime adapter, fabric and model qualification remain separate.

The port includes own-row DSA front/index selection (0102), two-half prefill
microbatching (0104), the default four-warp sparse-onepass launch for at most 16
local heads (only that part of 0101), a Q4 shared-expert decode side stream (0105),
and adapted arithmetic/cache identity (0107). It preserves Mia107's conditional
CUDA scoring/onepass paths and lazy Triton fallback, prelaunch/fill code,
FP32 rank-order partial sums, and existing grouped multi-prompt handling.
Optional MSA, BF16 partials, the upstream neighbor/native proxy relays and
network/driver patches are excluded. See `provenance.json` and `NOTICE`.

## Build the source without GPU work

From the public recipe root, first review the offline plan:

```sh
python3 -B engine/build.py --output local/engine-prefill --manifest 4096 --plan
```

Use `--manifest 2048` if the recipe selects that exact qualified template.
Both variants reconstruct identical 441 source files and use the same source_id;
their JSON documents are equal but key ordering differs, so their raw hashes
must be preserved. The emitted hashes are:

| Template | SHA256 of emitted source-manifest.json |
| --- | --- |
| 2048 |3abe7f60184aaaaed1f97eca49ae4fdfe7ba08a0267a722b25d7e7470aadd958|
| 4096 |75c0035ad3ab4ac4950a9250af8f1ac6ea4ae5aacfb92d84bd3aacbbaf099666|

On an ARM64 Docker host with the shared recipe mount, pull the exact image
separately if it is not already present:

```sh
docker pull ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14b5072e8e9918f1080153f57321415a387bc7d78cf9c9b44a7aeae2d4399ccf
python3 -B engine/build.py --output local/engine-prefill --manifest 4096 --run
```

The builder requires Python 3.10+ and GNU `patch`. It creates a new output only,
extracts the installed package with stdlib metadata/file operations, and runs a
short-lived non-root Docker container with network disabled, a read-only root
filesystem, no GPU request, and CUDA/NVIDIA devices hidden. It imports neither
TensorFold nor Torch. Image pulling is not automatic. It rejects altered assets,
symlinks, missing/extra source files or any hash mismatch. It verifies all 439
original and 441 patched files before publishing `tensorfold/`, then copies the
selected manifest byte for byte. `build-receipt.json` and `patch.log` retain the
build result. A failed build is not a usable overlay; retain its receipt and use
a new output directory for the next attempt.

The Docker extraction branch is provided for reproduction. The builder's
package-equivalence tests use the retained original source offline; they do not
claim a new hardware/image extraction run, CUDA qualification or speedup.

## Bind the overlay explicitly

Set these existing recipe configuration fields:

- `engine_overlay_root`: absolute shared path of the newly built output directory.
- `engine_manifest`: absolute path to its `source-manifest.json`.
- `engine_manifest_sha256`: the emitted exact template SHA256 above.
- `engine_flags`: supported explicit flags for the intended qualified profile.

The deployed example uses the separately qualified adjacent-ring relay. The
following gather/relay0 fragment is an alternative engine-only example, rather
than a reproduction of the selected deployment:

```json
{
  "TF_GLM_HC_EXCHANGE": "gather",
  "TF_GLM_PROMPT_OWN_FRONT": "1",
  "TF_GLM_PROMPT_MICROBATCH": "1",
  "TF_GLM_PREFILL_ROWS": "4096",
  "TF_GLM_SIDE": "1",
  "TF_GLM_PROMPT_PARTIALS": "fp32",
  "TF_GLM_MSA": "0",
  "TF_GLM_ONEPASS_SMALL_WARPS": "4",
  "TF_GLM_HC_RING_RELAY": "0"
}
```

This is a flag example, not authorization to change the current deployed profile.
Keep existing `TF_GLM_HC_SPLIT=1` and `TF_GLM_PREFILL_OVERLAP=2` transport settings.
SIDE defaults to 0 and must be enabled explicitly. At 2048 rows, the default 2048-row
minimum for each microbatch half keeps microbatching inactive;4096 admits two 2048
halves. Do not add a custom microbatch-minimum flag unless the recipe's flag
registry and new qualification explicitly permit it.

`TF_GLM_MULTI_PREFILL=1` is compatible: solo chunks use the new paths; grouped
multi-prompt chunks with an active mixer keep the original path. Changing source,
flags, prefill chunking, or transport requires new source-bound transport and
model receipts before serving. Do not reuse prior numerical/cache identities.

The historical manifest provenance reports failed direct diagonal NCCL p2p
qualification. A separately qualified runtime neighbor-ring relay is outside
this engine patch; it may select its own `p2p`/relay flags and matching receipts.
Never interpret local Gloo p2p math tests as physical NCCL p2p qualification.

## Memory and correctness scope

Own-front's lazy pools/token/count arrays need 10256 bytes per allocated row,
about 40.06MiB at 4096 rows. Four cached microbatch layouts plus the parent can
retain about 200MiB per world4 prompt buffer. The old estimator omits these arrays.
The upstream source estimate for doubled prompt row buffers is about 0.9GiB/rank,
plus own-front arrays; it is not a measured peak. FP32 gather scratch remains
allocated. Preserve pool/reserve/retained-state gates and measure actual memory.
Same-chunk comparisons do not establish cross-chunk KDA/recurrent-state identity.
No full 1M prompt, eight full 1M-context, vision or reboot claim follows from this
source reconstruction.

## Offline builder tests

Run self-contained tests without Docker, model imports or network requests:

```sh
python3 -B -m unittest discover -s engine/tests -v
```

For the full 439-to-441 equivalence and drift tests, supply a separately extracted,
immutable original package containing exactly the pinned 439 files:

```sh
PREFILL_TEST_ORIGINAL=/absolute/original/tensorfold python3 -B -m unittest discover -s engine/tests -v
```

This checks both output trees and both exact runtime-manifest serializations,
rejects changed/missing/extra/symlink sources, checks no-process offline plans,
and retains failure receipts without publishing a package. GNU `patch` is needed
for the full-equivalence case. The original snapshot is never modified.
