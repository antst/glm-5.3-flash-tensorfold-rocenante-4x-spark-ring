# TensorFold with prepared RoCEnante and patched NCCL on four Sparks

This separate recipe ports the communication approach of SparkRing's current
GLM profile to the pinned Mia TensorFold image. It retains GLM-5.3-Flash EXL3,
DFlash2, TensorFold's raw rank-major gather and FP32 summation, and the existing
four directly cabled Sparks. It selects prepared RoCEnante for eligible gathers
and patched NCCL 2.32.3 for the remaining communication. SIRCL is disabled.

**Historical transport and prefill releases are hardware/model-qualified and benchmarked.** The historical P8 configuration passed four-rank GPU collective qualification and
full-model correctness checks, including eight concurrent streams,
all 16 HCA functions and hardware diagonal forwarding. Initial matched
generation measurements are retained. The generation-priority profile uses a
4 MiB gather ceiling, 64 KiB forwarding window, 16 KiB chunks, four NCCL channels
and 512 threads with an eight-block cap. In the retained Mia96 matched SparkDash runs,
native median generation rates improved by 10–12% at C1 and 5–6% at C4;
C8 changes were modest, 0–3%. Cold 64K prefill was about 0.6% slower than the
local NCCL control. These are measured point estimates, with ranges retained.
The published [phase-one recipe](https://github.com/antst/glm-5.3-flash-tensorfold-4x-spark-ring/tree/v0.1.0)
remains the independent baseline and recovery path.

**v0.1.5 NVMe profile — qualified and serving on API 8888.**
The example selects a host-local persistent cache tier with a 6,098,944-row RAM
ceiling, P8 admission and the unchanged 1M per-request context/39.5625 GiB cache
request. Eight independent 1,046,528-token histories passed all 24 fresh/replay
requests, full cached-token counts and exact 32-token continuation checks.
The 11 disk-hit replays had a **3.375 s** median HTTP wall time, versus **376.392 s**
for the eight fresh controls. A separate 30-request P8 semantic gate and both
full-profile transports passed. The 64K pressure and four-worker clean-restart
checks also passed. Production readiness, all four live source bindings and
fresh matching monitoring were verified on **2026-10-15 at 09:21:35 UTC**.
See the [sanitized qualification and production result](results/nvme-tier-p8-20261015.json)
and [NVMe configuration, measurements and limits](docs/nvme-tier.md).
These are retention/correctness measurements; no new throughput benchmark is claimed.

Release **v0.1.3** retains **TensorFold 0.6.0 with 107 Mia patches** and qualified a long-context profile: six lanes, 1,048,576 total context tokens, 40 GiB extra cache, a 16 GiB reserve, 64 retained prompt snapshots and a 4 GiB retained-state budget per rank. Reasoning retention is bounded at 8,192 tool-call replies and 128 Mi characters. **Six-lane cache qualification passed:** six independent 1,046,528-token histories retained their prefixes across all four growth/replay waves, with exact 32-token replay consistency. A separate 24-request P6 semantic check passed before production returned to API8888. These are capacity and correctness results; generation benchmarks remain historical P8. See [cache capacity, validation and limits](docs/cache-retention.md). The [Mia107 image-upgrade results](docs/engine-upgrade.md) and published P8 performance tables retain their original profiles; they are not P6 measurements. Historical v0.1.1 remains Mia96 and v0.1.2 remains Mia107/P8.

The 2026-10-14 trim used **39.5625 GiB per rank** at P6, 1M total context
and the existing retention budgets. The current P8 prefill update retains that
cache request and changes admission/source settings as documented separately. The
2026-10-14 trim passed startup-capacity, six short-history replay and 24-request
semantic checks before production returned to API8888. Rank zero reported
**6,752,256 token-pool rows**, including **460,800 spare rows** beyond six full
context windows. The six near-1M history test above remains the historical
**40 GiB** qualification; it was not rerun for this trim. See the
[trim verification and limits](docs/cache-retention.md#cache-budget-trim-2026-10-14)
and its [sanitized result](results/cache-budget-trim-20261014.json).

See [transport design](docs/design.md), [network changes](docs/network.md),
[native builds and SSH configuration](docs/build.md),
[validation and tuning](docs/benchmark-method.md), and
[changes and recovery](docs/system-changes.md).
See [cache/memory monitoring and eight-lane sizing](docs/monitoring.md).
See the [sanitized six-lane qualification result](results/six-lane-cache-qualification.json) and [portable monitoring source](scripts/cache-monitor.py).
See the [complete performance comparison](docs/performance.md) and
[eight-configuration tuning results](docs/tuning.md).

The historical v0.1.4 prefill update used **P8 admission** with the same 1M per-request
context and **39.5625 GiB extra cache per rank**. It ports selected Mia TP4
compute optimizations into a fully hash-bound source overlay and retains the
prepared RoCEnante/NCCL transport. Its measurements used the actual pinned
SparkDash collectors at **C1/C2/C4/C8**, with three 400-token decode jobs per
workload and fresh cold-prefill prompts through 256K. See the
[prefill port](docs/prefill-port.md) and its measured comparison below the
historical tables. In that RAM-only profile, eight admitted requests shared the
existing arena; eight independent 1M histories were not qualified. The v0.1.5
NVMe retention result above is a separate source/profile qualification.
The selected **4096-row** setting was restored to port 8888 in v0.1.4, before the NVMe tier. Cold 256K prefill
measured **2828.06 tokens/s**, with 92.84s TTFT, versus 1711.57 tokens/s and 153.30s
on the original local baseline. See the [complete measured comparison](docs/sparkdash-native.md)
and [production verification](results/production-prefill-p8-20261014.json).

## What is being ported

The source is pinned to SparkRing commit
`6d27dd10a9f97da21278ae504ea6235f8634f303`. Its recommended GLM profile
uses prepared RoCEnante and NCCL, whereas its older research mesh profile also
uses SIRCL. This recipe uses the current prepared gather kernels and proxy.
The current compute overlay contains the selected TensorFold GLM prefill port;
the EXL3 target and DFlash2 checkpoints retain their pinned identities.

RoCEnante's generic gather transports FP32, BF16 and integer bit patterns
without the vocabulary packing needed by the superseded SIRCL prototype.
Compilation and shared buffers are prepared before CUDA graph capture. The
selected admission ceiling is 4 MiB; unsupported and larger calls use NCCL.
Admission and tuning apply to complete tensor messages, not Ethernet packets.

The physical network stays a ring. Opposite-peer RDMA packets use a two-hop
path forwarded by an intermediate ConnectX-7 ASIC. Two paths per peer use the
Socket Direct functions on both PCIe roots. These paths share the existing
cables' capacity; no extra physical bandwidth is created.

## Isolation

Use an identical shared absolute path on all four hosts, explicit `USER@IP`
SSH destinations and existing trusted host keys. Real site inputs belong in
ignored `local/`; raw host inventories, backups and measurements belong in
ignored `state/`. Each host's local model copies are read-only inputs reused by
both recipes. Native
libraries and the added preparation framework have their own pinned identities.

The experimental owner/container, bootstrap port 29561 and cache/state
directories remain separate from phase one. Both alternative recipes use the
same rank-zero API port **8888**, keeping clients on one stable URL. Stop both
stacks with their own launchers before switching, then start only the selected
recipe. The two full model copies cannot run concurrently on the same GPUs.
Fabric changes and GPU checks require the baseline to be stopped. No port
redirect or automatic handoff is needed.

The source manifest excludes private inputs, builds, model weights, image
layers and native binaries. See [provenance.json](provenance.json) for pins.
The operator runs from the shared checkout on the management host. SSH executes
host-local work on the four Sparks. Build compilation runs on an ARM64 Spark.

## Release checkout

The published prefill baseline is [v0.1.4](https://github.com/antst/glm-5.3-flash-tensorfold-rocenante-4x-spark-ring/releases/tag/v0.1.4).
The v0.1.5 NVMe profile passed private qualification and production readiness.
On the management host, clone the tagged recipe into a shared path mounted at
the same absolute location on all four Sparks:

```sh
git clone --branch v0.1.5 https://github.com/antst/glm-5.3-flash-tensorfold-rocenante-4x-spark-ring.git /shared/spark-ring-tensorfold-rocenante
cd /shared/spark-ring-tensorfold-rocenante
```

The release includes pinned source, the source manifest and sanitized benchmark
results. Build the native libraries using [the build guide](docs/build.md), then
adapt the private configuration and qualify your own cluster before serving.

## Prepare

Copy `config.example.json` to ignored `local/config.json`. Set the actual shared
path, explicit `admin@IP` destinations, management interfaces and trusted host
key aliases. List the nodes in physical f0-next/f1-previous ring order. Keep both
Socket Direct functions on each cable in the fabric inventory. Existing model
snapshots can be reused from phase one. Use this checkout's exact Mia image pin; the Mia107 release has a different digest from the published Mia96 recipe. Set
`api_port` to `8888`; retain this recipe's own container, bootstrap port and local
compiler/HF caches. The example selects P8 admission, 39.5625 GiB extra cache, an explicit 6,098,944-row pool and the bounded local NVMe tier described in [the tier guide](docs/nvme-tier.md). Check actual startup pool capacity and host MemAvailable on your cluster; selected memory budgets do not guarantee capacity on every installation. Image generation and other substantial GPU workloads must be stopped for this qualification.

Build the declared engine overlay on an ARM64 Docker host first, following
[the CPU-only engine builder](engine/README.md). Keep the output at the same
shared absolute path on all ranks, and set `engine_overlay_root`,
`engine_manifest`, its SHA256 and the selected flags in `local/config.json`.
Use builder variant `4096-nvme` and a new `local/engine-nvme` output. The explicit example requires this overlay; an unbuilt or changed manifest is
refused before launch. Review the builder plan before executing it.

Check sources and perform an offline model launch plan:

```sh
python3 -B scripts/check-public.py
python3 -B -m unittest discover -s tests
python3 -B scripts/experiment.py plan --config local/config.json
```

Collect the read-only fabric inventory and inspect the exact additions:

```sh
python3 -B scripts/network.py inventory --config local/config.json --output state/network-inventory.json
python3 -B scripts/network.py plan --config local/config.json --inventory state/network-inventory.json
```

Use `bash scripts/build-nccl.sh --help` for the native builder. On an ARM64 Spark,
it accepts a clean pinned NCCL checkout, the pinned archive or explicit `--fetch`.
Use a new host-local `--build-dir`, at most two jobs, the existing CUDA toolkit,
the pinned image through `--loader-image`, and a new shared `--publish lib/nccl`
directory. The exact marker source identified in `nccl-sources.json` is supplied
with `--marker-source`; `--publish-mesh lib/mesh` retains its binary and receipt.
The builder never executes the marker or initializes CUDA. Existing output
directories are refused; keep completed receipts instead of overwriting them.

## Qualify and serve

Stop the phase-one service with its own launcher before these commands. They
require idle GPUs and fabric QPs and retain root-only before-state backups.

```sh
python3 -B scripts/network.py prepare --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/network.py inventory --config local/config.json --output state/network-inventory.json
python3 -B scripts/network.py apply --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/network.py check --config local/config.json --inventory state/network-inventory.json
```

The NCCL control uses `--mode nccl` with the same library, model and compute
settings. Both modes retain `TF_GLM_COMM=nccl`: the adapter dispatches eligible
gathers, while TensorFold's old built-in RoCE wrapper stays inactive. Changing
the bound transport settings, sources, topology or library requires qualification
again. Check progress from this shared checkout:

```sh
python3 -B scripts/experiment.py status --config local/config.json
python3 -B scripts/experiment.py logs --config local/config.json
```

The selected recipe serves on the rank-zero management IP at port 8888:

```sh
python3 -B scripts/chat.py --url http://192.0.2.11:8888/v1
```

Replace that documentation IP with the configured rank-zero address. The same
URL works when switching to the exclusive original recipe. See the client's
`--help` for a one-shot prompt and generation settings.

Qualify inference on a private endpoint before serving clients. Copy the complete
configuration to `local/qualification.json` and set only `api_port` to **18899**;
keep the selected source, owner, profile, fabric and transport settings identical.
Run one stack at a time. Port relocation preserves the qualification identity,
while the live worker command and environment are checked again at startup.

```sh
python3 -B scripts/qualify.py run --config local/qualification.json --transport nccl
python3 -B scripts/qualify.py run --config local/qualification.json --transport rocenante
python3 -B scripts/experiment.py start --config local/qualification.json --mode rocenante
python3 -B scripts/validate-model.py --config local/qualification.json --mode rocenante --output state/model-validation
python3 -B scripts/experiment.py stop --config local/qualification.json
python3 -B scripts/experiment.py start --config local/config.json --mode rocenante
```

After restoration to **8888**, verify readiness with GET-only health/models checks
and the all-rank source/worker/monitor receipts. Keep inference tests and performance
measurements on the isolated endpoint so normal clients cannot affect them.

This checks selected answers, serial/speculative/concurrent output equality,
live deployment bindings and payload/error counters. The configured 1,048,576
token context is advertised by the API. The historical [long-context cache test](docs/cache-retention.md) qualified six independent 1,046,528-token prompts with 32-token completions and exact replay at 40 GiB extra cache; the historical P6 trim received the shorter checks described there. Context includes prompt and output together; a full 1,048,576-token prompt plus a reply is not valid. Vision, branching histories and real-agent reasoning restoration need their own tests.

## Stop and recover

For the tiered profile, use the owned stop command below so rank zero can finish
the bounded all-rank flush while followers remain alive. Preserve the host-local
snapshot stores across a clean restart; see [NVMe recovery](docs/nvme-tier.md).


```sh
python3 -B scripts/experiment.py stop --config local/config.json
python3 -B scripts/network.py rollback --config local/config.json --inventory state/network-inventory.json --restore-driver
```

Rollback removes only journaled resources, stops owned helpers and restores
the recorded NIC parameters with the stack idle. Check and restart phase one
using its own launcher. Keep both recipes' model logs and change journals.
This recipe installs no automatic model-start service; see
[reboot and recovery details](docs/system-changes.md) before restarting hosts.
