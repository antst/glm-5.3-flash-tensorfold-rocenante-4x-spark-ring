# Build the pinned native artifacts

Compile on an ARM64 Spark using existing tools. The x86 management host runs
the operator and offline plans; it cannot perform this native build. Keep the
build directory on host-local storage and the published artifacts in the shared
checkout mounted at the same absolute path on all four hosts.

The host needs Python 3, Git, GCC/G++, make, binutils (`readelf`), `findmnt`, an
existing CUDA toolkit with `nvcc` and `cuobjdump`, and libibverbs/libmlx5 headers
and libraries for the marker. The loader check needs Docker access and the
exact ARM64 Mia image below. The documented source download also uses `curl`
and `sha256sum`. The tested build used CUDA 13.0.88, GCC 13.3 and Git 2.43;
source archives must match the pinned Git archive format and hash. The builder
installs no packages, drivers or global libraries. Stop if a prerequisite is
missing rather than changing the host implicitly.

On the chosen Spark, enter the passwordless privileged shell and use the
configured shared checkout. The path below is the portable example; replace it
with your actual `shared_root`. Pulling the digest-pinned image is an explicit
image-store change; omit the pull if that exact image is already present.

```sh
sudo -s
cd /shared/spark-ring-tensorfold-rocenante
mkdir -p local
curl --fail --location \
  --output local/mlx5_rdma_tx_rewrite_probe.c \
  https://raw.githubusercontent.com/FujitsuPolycom/sparkring/6d27dd10a9f97da21278ae504ea6235f8634f303/spark_transport/fabric/cx7_hairpin_diagonal/native/mlx5_rdma_tx_rewrite_probe.c
printf '%s  %s\n' \
  '8684a6961b8e86aa474fa2310ff71e4cdf219a63a72ceb5593b2f95e54812792' \
  'local/mlx5_rdma_tx_rewrite_probe.c' | sha256sum --check
SPARK_RING_IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:bc34d7d63f978cf601f42863b284bc95a567c50c10e9adb0866a635be568bf5f
docker pull "$SPARK_RING_IMAGE"
docker image inspect "$SPARK_RING_IMAGE"
SPARK_RING_BUILD_DIR=/var/tmp/spark-ring-rocenante-nccl-$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/build-nccl.sh --fetch \
  --build-dir "$SPARK_RING_BUILD_DIR" \
  --jobs 2 --cuda-root /usr/local/cuda-13.0 \
  --loader-image "$SPARK_RING_IMAGE" \
  --marker-source "$PWD/local/mlx5_rdma_tx_rewrite_probe.c" \
  --publish "$PWD/lib/nccl" --publish-mesh "$PWD/lib/mesh"
```

Select the installed toolkit path if it differs. `--fetch` downloads only the
pinned NCCL commit, then verifies the source archive, original tree and
cumulative patch before compilation. It builds for SM121 with at most two
jobs. A clean checkout of the exact commit can instead be supplied with
`--source`; an exact pinned Git tar archive can be supplied with `--archive`.
GitHub codeload archives are not interchangeable with the pinned Git tar.
Inspect the manifest and logs if an archive check fails; do not bypass it.

Compilation writes only the new host-local build directory and requested shared
publication directories. The image pull reuses the same digest-pinned Mia image
as the original deployment; it is not a serving-image or compute-engine upgrade.
The isolated prepared framework is already pinned in this checkout. The helper
does not run a package installer, change system CUDA/GCC, tune GPU clocks/power,
reload a NIC or apply the mesh. Network preparation, GPU qualification and model
startup are separate recorded operator actions from the management host.

The local build directory and both publish directories must be new. A repeated
build refuses existing output directories. Preserve completed receipts and
choose new output locations for another build; review the corresponding config
paths instead of overwriting an artifact already bound to qualification.
The default publication creates:

- `lib/nccl/libnccl.so.2`, its versioned target and `build-receipt.json`;
- `lib/mesh/mlx5-rdma-tx-marker` and its `build-receipt.json`;
- source, compiler, CPU ABI, image-loader and action receipts/logs.

The builder never executes the marker, initializes CUDA, creates a communicator
or opens RDMA endpoints. Its temporary loader container has no GPU, IB device
or network and is removed afterward. CPU loading and native compilation are
prerequisites; they do not replace the fresh four-rank NCCL/RoCEnante hardware
qualification and running-model checks in the [main recipe](../README.md).
Keep the build actions and exact artifact hashes in the private system ledger.

# SSH trust and private configuration

Copy the example to ignored `local/config.json` and adapt both `shared_root`
and the absolute `library` path, or use a library path relative to the shared
root. Site addresses, trusted-key files and identities stay outside the public
export. Model snapshots and compiler/HF caches stay on local storage outside
the shared checkout.

The example's `ssh_options.known_hosts_file: "known_hosts"` is resolved relative
to the config file directory: `local/config.json` selects `local/known_hosts`.
Supply an existing, already verified absolute trust-store path, or remove that
option to use the existing SSH default trust store. A dedicated ignored file
can contain previously verified keys; the recipe does not discover or trust
new keys. Entries must match each node's `host_key_alias`.

SSH uses strict host verification, does not update trust files, bypasses the
agent and cannot prompt. Use an existing noninteractive file identity, either
a default SSH identity or explicit `ssh_options.identity_file`. A relative
identity path also resolves against the config directory. Keys are not copied
to the Sparks by this recipe. Remote operator actions use `sudo -n -s`; do not
disable host checks or add automatic key acceptance to make a plan execute.

After compiling, return to the management host and run plans, network
preparation, qualification and startup from the shared checkout. Set the current
API port to **8888** for both alternative recipes. Stop each stack with its own
launcher before switching or doing GPU/network maintenance; start only one
recipe. The rank-zero client URL stays stable, without a redirect. Follow
[recovery](system-changes.md) for ownership-safe rollback and post-reboot restart.
Existing model/image inputs can be reused, while both recipes retain separate
libraries, local compiler/HF caches, containers, bootstrap ports and qualification
receipts.
