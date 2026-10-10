# TensorFold with prepared RoCEnante and patched NCCL on four Sparks

This separate recipe ports the communication approach of SparkRing's current
GLM profile to the pinned Mia TensorFold image. It retains GLM-5.3-Flash EXL3,
DFlash2, TensorFold's raw rank-major gather and FP32 summation, and the existing
four directly cabled Sparks. It selects prepared RoCEnante for eligible gathers
and patched NCCL 2.32.3 for the remaining communication. SIRCL is disabled.

**Developed, hardware/model-qualified and benchmarked.** Four-rank GPU collective qualification and
full-model correctness checks have passed, including eight concurrent streams,
all 16 HCA functions and hardware diagonal forwarding. Initial matched
generation measurements are retained. The generation-priority profile uses a
4 MiB gather ceiling, 64 KiB forwarding window, 16 KiB chunks, four NCCL channels
and 512 threads with an eight-block cap. In the retained Mia96 matched SparkDash runs,
native median generation rates improved by 10–12% at C1 and 5–6% at C4;
C8 changes were modest, 0–3%. Cold 64K prefill was about 0.6% slower than the
local NCCL control. These are measured point estimates, with ranges retained.
The published [phase-one recipe](https://github.com/antst/glm-5.3-flash-tensorfold-4x-spark-ring/tree/v0.1.0)
remains the independent baseline and recovery path.

Release **v0.1.2** uses **TensorFold 0.6.0 with 107 Mia patches**. Fresh image-upgrade inference and benchmark checks passed; see [Mia107 results and component updates](docs/engine-upgrade.md). The published v0.1.1 release retains its Mia96 image and historical measurements.

See [transport design](docs/design.md), [network changes](docs/network.md),
[native builds and SSH configuration](docs/build.md),
[validation and tuning](docs/benchmark-method.md), and
[changes and recovery](docs/system-changes.md).
See the [complete performance comparison](docs/performance.md) and
[eight-configuration tuning results](docs/tuning.md).

## What is being ported

The source is pinned to SparkRing commit
`6d27dd10a9f97da21278ae504ea6235f8634f303`. Its recommended GLM profile
uses prepared RoCEnante and NCCL, whereas its older research mesh profile also
uses SIRCL. This recipe uses the current prepared gather kernels and proxy.
It does not replace TensorFold's compute kernels or weights with vLLM/NVFP4.

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

The current source release is [v0.1.2](https://github.com/antst/glm-5.3-flash-tensorfold-rocenante-4x-spark-ring/releases/tag/v0.1.2). The v0.1.0 and Mia96 v0.1.1 releases remain historical. On the management host,
clone it into a shared path mounted at the same absolute location on all four
Sparks:

```sh
git clone --branch v0.1.2 https://github.com/antst/glm-5.3-flash-tensorfold-rocenante-4x-spark-ring.git /shared/spark-ring-tensorfold-rocenante
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
compiler/HF caches.

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
python3 -B scripts/qualify.py run --config local/config.json --transport nccl
python3 -B scripts/qualify.py run --config local/config.json --transport rocenante
python3 -B scripts/experiment.py start --config local/config.json --mode rocenante
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

Validate the running eight-lane model and retain a new private receipt:

```sh
python3 -B scripts/validate-model.py --config local/config.json --mode rocenante --output state/model-validation
```

This checks selected answers, serial/speculative/concurrent output equality,
live deployment bindings and payload/error counters. The configured 1,048,576
token context is advertised by the API; a full-length prompt and vision input
need their own tests.

## Stop and recover

```sh
python3 -B scripts/experiment.py stop --config local/config.json
python3 -B scripts/network.py rollback --config local/config.json --inventory state/network-inventory.json --restore-driver
```

Rollback removes only journaled resources, stops owned helpers and restores
the recorded NIC parameters with the stack idle. Check and restart phase one
using its own launcher. Keep both recipes' model logs and change journals.
This recipe installs no automatic model-start service; see
[reboot and recovery details](docs/system-changes.md) before restarting hosts.
