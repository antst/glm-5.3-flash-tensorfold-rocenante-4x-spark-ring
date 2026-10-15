# System changes and recovery

## v0.1.5 local NVMe cache update

**Full-profile private qualification and production restoration PASSED.**
The completed eight-near-1M, full P8 semantic, both transport, private 64K
pressure/replay and clean four-worker restart gates
are documented in [the NVMe guide](nvme-tier.md) and the
[sanitized qualification and production result](../results/nvme-tier-p8-20261015.json).
Historical v0.1.3/v0.1.4
measurements and their source/profile bindings below are unchanged.

The approved maintenance sequence stopped the owned production GLM and read-only
monitor, preserved its exact config/source/qualification/control backups, and
cleared its in-memory cache. It created an isolated 443-file engine overlay and
source-bound runtime/controller configuration, local state/compiler caches and
per-rank host-local snapshot stores. The selected profile retains P8, 1M context,
39.5625 GiB extra cache, the 16 GiB reserve and prior snapshot/reasoning budgets,
and adds an explicit 6,098,944-row pool ceiling plus bounded NVMe settings.
The public example contains portable values; private inventories, worker IDs,
boot IDs, host keys, paths and raw test fixtures remain outside the export.

Private qualification created and removed owned workers on an isolated API.
It recorded fresh NCCL and prepared-RoCEnante proofs, ran short model checks,
wrote synthetic snapshots and exercised RAM eviction/disk restore. A deliberate
clean restart recreated all four workers with unchanged config/source and the
same local stores; the restart phase began with empty RAM and submitted no fresh
controls. Owned shutdown is ordered: rank zero completes the bounded all-rank
flush while followers are alive, then followers stop. Verify the truthful flush
outcome independently of docker-stop success. The ordinary launcher archives
pre-stop logs; final flush evidence needs a separate shutdown capture/receipt.
Startup/shutdown receipts,
point-manifest identities and failure attempts are retained in the private
ledger. The full eight-near-1M suite passed all 24 fresh/replay requests,
followed by 30 P8 semantic requests. Both full-profile transports passed.
Private rank-zero/follower flushes completed before shutdown. Production was
restored to API 8888 with P8 admission, 1M context, the 6,098,944-row pool and
control protocol 3. Actual source/config/worker bindings on all four ranks and
fresh matching monitoring passed at 2026-10-15 09:21:35 UTC. The final readiness
check submitted no production inference. No new throughput claim is made.

This update changes source/config/controller selection, owned worker lifecycle,
local NVMe snapshot contents, cache/state/log files and source-bound receipts.
It reuses the pinned model, DFlash2, image and NCCL/RoCEnante artifacts. It does
not require network apply on an unchanged healthy mesh and does not change
routes, forwarding, firewall, NIC/driver settings or host packages. No automatic
boot service is installed. A monitor restart must bind the final selected config.

### Portable upgrade and recovery

Archive the prior config, source revision, both qualification receipts and
control/startup/monitor bindings before stopping. Build a new 4096-nvme output
with the [CPU builder](../engine/README.md); keep the old overlay untouched.
Update the complete selected profile, exact manifest/flags and tier block in
ignored local config while preserving site inventory, owner, SSH trust,
model/library paths and ports. Verify each host's snapshot path is on its
intended local NVMe filesystem with enough space above the 64 GiB floor.
Do not put snapshots in the shared checkout or delete compatible stores during
an ordinary restart.

Review the offline launch plan, freshly qualify both transports against the
changed source/profile, then run model/cache/restart checks on a private API
with all-host memory/paging evidence. Reuse healthy owned mesh state after its
read-only check; do not apply routes or forwarding merely to enable persistence.
Only after all gates pass should the coordinator restore the same profile to
8888, verify every live worker/source/environment and restart monitoring with
the new config hash. This sequence passed for the recorded deployment; each
installation needs its own bound receipts.

For rollback, stop through the selected owned launcher, preserve new snapshots
and logs, and restore the exact previous source/config/control/receipts together.
Changing source or arithmetic can invalidate the disk compatibility namespace;
never rewrite identities to force use of old snapshots. A RAM-only rollback
may need cold prefill. After a host reboot or changed binding, old qualification
is invalid and both source-bound transport modes must be requalified. A clean
process-restart result is not an arbitrary power-loss guarantee.


Preparation creates an isolated source/config tree, pinned dependency sources,
native builds, logs and receipts. The superseded SIRCL prototype is retained
for audit and has never been deployed. Original TensorFold image, model
snapshots, published phase-one sources and original library stay independent.

The initial development and retained benchmark runs used port **18888** for the
candidate while the independent original deployment used 8888. Release v0.1.1
converges both alternative recipes on the same rank-zero API port **8888**;
18888 is a historical test endpoint. Benchmark records remain unchanged.

Only one recipe runs at a time: the same four GPUs cannot hold both full model
copies. Stop both stacks through their own launchers before switching, then
start the selected recipe. Returning to the original also removes the candidate's
owned mesh with the recovery steps below. Owner/container identities, bootstrap
ports and local cache/state paths stay separate; sharing the API port does not
merge ownership. Clients keep the same rank-zero URL. No automatic handoff or
port redirect is installed or needed.

The user authorized the mesh routes and hardware forwarding required by the
current RoCEnante profile, followed by development, testing and performance
tuning. A stopped-stack maintenance phase is needed before applying rules,
changing hairpin queues, creating GPU collectives or serving the candidate.

Exact site-specific paths, commands, timestamps, before/after values, native
binary identities, temporary containers, failures and cleanup belong in the
operator's complete task ledgers and ignored audit state. Portable sources
must contain no real management addresses, private keys or network backups.

The recorded system mutations comprise the separate shared recipe and published
native artifacts; host-local build/source directories, model snapshots and
compiler/HF caches; the pinned Docker image if it must be pulled; temporary
owned CPU/qualification containers; and the candidate's four owned model
containers. The network additions total eight endpoint routes, eight TC rules,
up to eight owned `clsact` qdiscs, eight native source markers and four health
guardians. Preparation journals the sixteen fabric PCI function queue values
and any hardware TC offload changes. Inventories, operation logs, receipts and
private root-only before-state archives are retained for every action.

The recipe uses existing system packages, drivers and CUDA tools. The published v0.1.1 deployment retained the original Mia96 image. The v0.1.2 Mia107 update changes its pinned serving image, reviewed engine source guards and bundled Transformers; weights, quantization and transport settings stay unchanged. Exact upgrade checks are in [the image-upgrade report](engine-upgrade.md). Patched NCCL and the isolated
prepared communication framework are separate additions; they do not replace
the original library or install globally. Build details and explicit image-store
effects are in [the build guide](build.md).

Network changes are scoped to the reviewed fabric plan. Existing addresses,
NetworkManager profiles, management/default routes, firewall and forwarding
settings are preserved. Hairpin driver-init parameters, owned endpoint routes,
TC rules/qdiscs and native marker processes need explicit ownership and rollback.
The tested fabric uses queue size 8192, retained from the initial 128 KiB
forwarding window. The selected 64 KiB window runs on that same prepared fabric;
the transport tuning trials do not reload the NICs. Preparation uses root only
for the dedicated fabric PCI driver-init parameters, function reloads and any
required hardware TC offload enablement, retaining original values and reload
proof. No firmware or driver replacement is implied by this recipe. No OS
forwarding, NAT or firewall policy change is part of the plan; TC transit must
be hardware-only and qualify with positive traffic on all eight rules.

Stop model workers before removing markers or hardware rules. Stop only owned
candidate containers, retain logs/receipts, then remove only the network
resources recorded as owned. Rollback first collects a fresh snapshot and
checks every host idle. It excludes only full route identities attested as
actually removed from current-boot owned journal records; it compares all
remaining routes, management defaults, profiles, sysctls and firewall policy.
The saved inventory still binds the physical plan. Foreign routes at planned
destinations are preserved, and failed or ambiguous cleanup fails closed.

Use `rollback --restore-driver` to restore prior hairpin and offload values with
the stack idle. Without that option, rollback retains journaled driver values
for a native restart. Verify the connected fabric before restarting either
stack. Same-boot stopped-stack rollback and reapplication have passed on all
four hosts, including owned-route removal and preservation checks. A partial
host reboot requires recreating all four selected model ranks together.
Physical reboot recovery is covered by CPU mocks; an actual host reboot has
not been tested.

## Cache budget trim: 2026-10-14

The [sanitized trim result](../results/cache-budget-trim-20261014.json) records
the selected profile and the bounded validation scope.

Approved maintenance reduced the extra cache request from 40 to
**39.5625 GiB per rank**, retaining P6 and the other long-context settings.
Startup reported 6,752,256 rank-zero pool rows with 460,800 rows beyond six
full context windows. Six short-history fresh/replay pairs and the 24-request
P6 semantic check passed before API8888 was restored. The original v0.1.3
40 GiB six-near-1M qualification and performance records remain historical;
that long-context suite and generation benchmarks were not rerun.

The recorded mutations include backing up the coherent prior config,
qualification receipts and local control guard; waiting for production to
be idle; stopping and recreating only the four owned model containers;
changing the cache budget and matching local control acceptance guard;
writing fresh profile-bound RoCEnante qualification, runtime compiler/cache
artifacts, startup/capacity/replay/semantic evidence and audit receipts;
stopping the private test workers and recreating production on API8888; and
restarting the existing read-only monitor with the new configuration identity.
Both model restarts cleared in-memory KV. Clients must rebuild prefixes after
a restart; restoring configuration cannot restore old cached conversations.

The image, model and draft weights, quantization/math code, transport settings,
healthy mesh, routes, forwarding rules, host packages and drivers were
unchanged. Exact commands, before/after hashes, runtime identities and private
receipts are retained in the operator audit state.

For another installation, arrange an approved maintenance window, stop owned
workers with the old coherent config, and preserve that config and its bound
receipts before changing only profile.cache_gib to 39.5625. Keep the existing
healthy mesh; this trim requires no network apply or rollback. Obtain fresh
qualification wherever the profile/source binding changes, verify at least
6,750,208 actual pool rows and the documented all-host memory floors, then run
short replay and semantic checks before admitting ordinary clients. Do not
relabel historical receipts or treat this startup floor as a new full
long-context stress test.

For same-boot recovery, stop the trimmed owned workers, retain their logs and
evidence, restore the saved coherent config/control guard and qualification
set, check the retained mesh, and recreate the prior workers together.
Refresh qualification if source, library, profile or boot bindings differ.
Rebind/restart the monitor to the restored config and retain the trim evidence.
In-memory KV starts empty on recovery too.

## Upgrade v0.1.2 to v0.1.3 with the healthy mesh retained

The published v0.1.3 example selected P6, 1M total context, 40 GiB extra cache, a 16 GiB
reserve, 64 snapshots, a 4 GiB retained-state budget and reasoning retention of
8,192 replies / 128 Mi characters. See [cache limits](cache-retention.md).
The six-lane qualification and 72 private cache/semantic requests passed;
review the published report and qualify the same profile on your own hosts.

Run on the management host at your existing shared checkout. Arrange a
maintenance window and stop GLM using **old source and old configuration before
switching**. Stop substantial competing workloads with their own controls.
Archive local edits; do not force a checkout.

~~~sh
cd /shared/spark-ring-tensorfold-rocenante
python3 -B scripts/experiment.py stop --config local/config.json
git status --short --untracked-files=no
SPARK_RING_CACHE_BACKUP="state/before-v0.1.3-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$SPARK_RING_CACHE_BACKUP"
cp -a local/config.json state/collective-qualification.json state/rocenante-qualification.json "$SPARK_RING_CACHE_BACKUP/"
git rev-parse HEAD > "$SPARK_RING_CACHE_BACKUP/source-commit.txt"
git fetch origin tag v0.1.3
git checkout --detach v0.1.3
~~~

Ensure tracked files are clean or edits archived before fetching/checking out.
Retain the backup path. Copy only the new profile into your ignored config.
Preserve addresses, shared/model/cache roots, owner, image/library, ports,
model pins, fabric inventory and experiment transport settings:

~~~sh
python3 -B - <<'PY'
import json, pathlib
p = pathlib.Path('local/config.json')
cfg = json.loads(p.read_text())
cfg['profile'] = json.loads(pathlib.Path('config.example.json').read_text())['profile']
p.write_text(json.dumps(cfg, indent=2) + '\n')
PY
python3 -B scripts/check-public.py
python3 -B scripts/experiment.py plan --config local/config.json
python3 -B scripts/network.py check --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/qualify.py run --config local/config.json --transport nccl
python3 -B scripts/qualify.py run --config local/config.json --transport rocenante
~~~

Changed launcher source/profile require **fresh source-bound qualifications for
both transports**. Back up historical receipts; never relabel them. Retaining a
healthy mesh needs no network apply, rollback, NIC reload, route additions or
forwarding change. Image pulls, native builds and downloads are unnecessary
when exact v0.1.2 pins and loader/build proofs remain valid. Changed image,
library, network plan or host boots require the corresponding qualification.

Start, verify actual allocations/memory, and run a new short semantic check:

~~~sh
python3 -B scripts/experiment.py start --config local/config.json --mode rocenante
python3 -B scripts/experiment.py status --config local/config.json
python3 -B scripts/validate-model.py --config local/config.json --mode rocenante --output state/model-validation-p6-NEW
~~~

Use an ignored maintenance config with a private API port for the long-context
independent-history, memory-floor, replay and final-wave admission checks
[described here](cache-retention.md). Return to 8888 after checks pass.
Budgets do not guarantee six context slots on hosts with other free memory or
workloads. Keep all test logs and receipts private.

For same-boot recovery, stop new owned workers with new source/config, preserve
logs/receipts, select the saved source-commit revision and restore the saved
config plus both qualification files together. Check the retained mesh and
start the prior recipe. Never mix old receipts and new source/profile. Changed
boots require fresh qualification. Restarting clears in-memory KV; restoring
configuration does not restore cached conversations.

## Mia107 image update

The qualified v0.1.2 update preserves the existing owner/root and healthy installed mesh. It pulls the new image on all four hosts, stops/recreates the owned workers, creates a new compiler/cache namespace and selects a copied NCCL artifact with a fresh image-loader receipt. The old image, cache, original binary/build receipt and exact prior config/source/qualification backups remain available. It adds no routes, rules, NIC reloads or OS forwarding changes.

Archive the prior coherent checkout/config/receipts before selecting a new image. Requalify both transports and validate the model; old source-bound receipts cannot authorize new workers. Restore the prior source/config/library receipt and its preserved qualification files together to return to Mia96, then start only that owner. Site-specific exact rollback commands and before/after hashes are retained in the operator audit ledgers. See [Mia107 validation and benchmarks](engine-upgrade.md).

## Upgrade v0.1.1 to v0.1.2 with the current mesh retained

Run these steps from the management host, using your configured shared root. Stop the old workers **before switching source or private config**, with the v0.1.1 checkout and its existing config. Confirm any other recipe is stopped through its own launcher. Keep the owner, root, inventory, model snapshots, bootstrap/guardian ports and transport/profile settings unchanged. Preserve local edits rather than forcing a checkout.

```sh
cd /shared/spark-ring-tensorfold-rocenante
python3 -B scripts/experiment.py stop --config local/config.json
git status --short --untracked-files=no
SPARK_RING_UPGRADE_BACKUP="state/before-v0.1.2-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$SPARK_RING_UPGRADE_BACKUP"
cp -a local/config.json state/collective-qualification.json state/rocenante-qualification.json "$SPARK_RING_UPGRADE_BACKUP/"
git rev-parse HEAD > "$SPARK_RING_UPGRADE_BACKUP/source-commit.txt"
git fetch origin tag v0.1.2
git checkout --detach v0.1.2
```

Select this checkout’s image and a **new** NCCL artifact directory. The short script below changes only the ignored config’s image/library fields and pulls the exact image through the configured admin SSH identities on all four hosts:

```sh
python3 -B - <<'PY'
import json, pathlib, shlex, sys
sys.path.insert(0, 'scripts')
from common import IMAGE, load_config
from cluster import Operator
p = pathlib.Path('local/config.json')
cfg = json.loads(p.read_text())
cfg['image'] = IMAGE
cfg['library'] = 'lib/nccl-mia107/libnccl.so.2'
p.write_text(json.dumps(cfg, indent=2) + '\n')
op = Operator(load_config(p))
print(json.dumps(op.parallel(lambda rank: op.remote(rank, shlex.join(['docker', 'pull', IMAGE]), timeout=1800))))
PY
```

On one ARM64 Spark, compile the pinned NCCL source into a new host-local directory and publish into that new shared library directory. For this existing healthy mesh, retain the existing marker artifact and omit marker build/publication:

```sh
sudo -s
cd /shared/spark-ring-tensorfold-rocenante
SPARK_RING_IMAGE=$(python3 -B -c 'import json; print(json.load(open("config.example.json"))["image"])')
SPARK_RING_BUILD_DIR="/var/tmp/spark-ring-rocenante-mia107-nccl-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/build-nccl.sh --fetch --build-dir "$SPARK_RING_BUILD_DIR" \
  --jobs 2 --cuda-root /usr/local/cuda-13.0 \
  --loader-image "$SPARK_RING_IMAGE" --publish "$PWD/lib/nccl-mia107"
```

Use your existing toolkit path if it differs; the helper installs no packages and does not initialize CUDA. Existing publish paths are refused. A rebuilt library must receive its own exact loader/build receipt and fresh collective qualification; its binary identity need not equal the historical local binary. The tested deployment instead reused an identical binary with a separately recorded derived image-loader receipt. Do not substitute the old-image loader proof or hand-edit qualification to authorize a new image.

Return to the management host. Check the retained healthy mesh; this image upgrade requires no network apply, rollback, new routes or NIC reload. Run fresh qualifications and start/validate the selected workers:

```sh
python3 -B scripts/network.py check --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/qualify.py run --config local/config.json --transport nccl
python3 -B scripts/qualify.py run --config local/config.json --transport rocenante
python3 -B scripts/experiment.py start --config local/config.json --mode rocenante
python3 -B scripts/validate-model.py --config local/config.json --mode rocenante --output state/model-validation-mia107-NEW
```

The first start builds image-specific kernels; retain the old image, caches and library. Each command must pass before the next dependent step. To return to Mia96 on the same boot, stop the new workers, select the preserved prior source revision, restore the archived config and both qualification receipts together, then check the retained mesh and start that prior recipe. A changed boot invalidates old qualification; follow the reboot procedure below instead. Preserve new logs and receipts for the audit.

## Upgrade v0.1.0 to v0.1.1 without changing the mesh

For an API-only upgrade, retain the existing mesh and inventory. Before editing
any configuration or switching revisions, use the existing **v0.1.0** checkout
and its **18888** configuration to stop the owned workers:

```sh
python3 -B scripts/experiment.py stop --config local/config.json
```

Ensure the original recipe is also stopped through its own launcher. Preserve
the prior private configuration, qualification receipts and logs in a new
ignored archive before replacing any canonical receipt. Confirm the tracked
checkout is clean; retain any local changes separately rather than resetting
or discarding them. Then fetch and select the release:

```sh
git status --short --untracked-files=no
git fetch origin tag v0.1.1
git checkout --detach v0.1.1
```

Set only the ignored local configuration's top-level `api_port` to **8888**.
Keep the retained mesh/inventory, bootstrap **29561**, guardian **29661**,
cache paths, vendor sources, libraries, model snapshots and performance knobs
unchanged. This API-only upgrade needs no network rollback, preparation or
application. Check the retained inventory, then rerun both qualifications:
v0.1.1's two-line API-port guard change alters the bound `experiment.py`
source hash, so the earlier source-bound receipts cannot authorize new workers.
Start RoCEnante only after both qualifications pass, and validate it into a new
output directory:

```sh
python3 -B scripts/network.py check --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/qualify.py run --config local/config.json --transport nccl
python3 -B scripts/qualify.py run --config local/config.json --transport rocenante
python3 -B scripts/experiment.py start --config local/config.json --mode rocenante
python3 -B scripts/validate-model.py --config local/config.json --mode rocenante --output state/model-validation-api8888-NEW
```

The validator requires 30 new model requests and bound traffic/identity proof;
preserve its new output alongside the earlier logs and receipts. The two recipe
owners remain exclusive. The occupied-port preflight protects **8888** and
fails rather than stopping another owner's listener. The release's migration
checks are recorded in [api-port-8888-validation.json](../results/api-port-8888-validation.json);
the retained v0.1.0 performance measurements were not rerun.

The mesh is deliberately volatile. After a reboot, run the following from the
shared checkout on the management host, with all model workers stopped. If the
phase-one stack is running, stop it with its own launcher first. These commands
assume the default `state` directory; follow the configured state path if it
differs. Archive the previous canonical inventory before each replacement:

```sh
cd /shared/spark-ring-tensorfold-rocenante
python3 -B scripts/experiment.py stop --config local/config.json
python3 -B scripts/network.py rollback --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/network.py inventory --config local/config.json --output state/network-inventory.json
python3 -B scripts/network.py prepare --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/network.py inventory --config local/config.json --output state/network-inventory.json
python3 -B scripts/network.py apply --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/network.py check --config local/config.json --inventory state/network-inventory.json
python3 -B scripts/qualify.py run --config local/config.json --transport nccl
python3 -B scripts/qualify.py run --config local/config.json --transport rocenante
python3 -B scripts/experiment.py start --config local/config.json --mode rocenante
```

Replace the portable directory example with the configured shared checkout,
available at the same absolute path on all four Sparks. Run these commands on
the management host, not independently inside each Spark. The operator uses
the configured strict SSH identities and `sudo -n -s` for remote actions. After
a full reboot, first restore ordinary management SSH reachability and the shared
mount; the recipe does not configure either service during recovery.

Ownership journals bind helper PIDs to their boot ID and start time. Explicit
rollback retires prior-boot volatile records without signaling reused PIDs or
deleting resources created in the new boot. Original backups and retirement
history remain available. Refresh inventory after preparation because a NIC
reload can change NetworkManager's automatic connected-route metrics. Only
that field on an exact configured connected fabric route is admitted, with
successful same-PCI `driver_reinit + 1` evidence. Each reboot invalidates both
hardware qualification receipts; bound source, library, profile, environment
or native geometry changes also require matching fresh qualifications. No
systemd model or mesh autostart service is installed.

Stop on any failed command and inspect `state/network-observed-<action>.json`,
the action logs and root-only ownership journals. These observations are
explicitly unqualified. An initial driver reload can remove an inactive
NetworkManager-generated profile in `/run`; the strict file guard still
rejects that change. Identify the exact inactive profile, archive its bytes and
hashes, and verify active/persistent profiles, management routes and other
preserved state before collecting a new reviewed inventory and retrying. Do
not exclude arbitrary `/run` files or ignore unexplained guard failures.

To return to the independent baseline, stop the candidate and remove the owned
mesh with driver restoration. These common steps run from this checkout:

```sh
python3 -B scripts/experiment.py stop --config local/config.json
python3 -B scripts/network.py rollback --config local/config.json --inventory state/network-inventory.json --restore-driver
python3 -B scripts/network.py inventory --config local/config.json --output state/network-after-original-recovery.json
```

Choose the baseline launcher that belongs to your deployment. The historical
installed original has `tensorfold/cluster.py` and takes no `--config` argument:

```sh
python3 -B /path/to/original-recipe/tensorfold/cluster.py check --context 1048576 --parallel 8 --multi-window 64 --cache-entries 32 --max-tokens 32768 --cache-gib 24 --port 8888
python3 -B /path/to/original-recipe/tensorfold/cluster.py start --context 1048576 --parallel 8 --multi-window 64 --cache-entries 32 --max-tokens 32768 --cache-gib 24 --port 8888
python3 -B /path/to/original-recipe/tensorfold/cluster.py status
```

The published [phase-one v0.1.0 checkout](https://github.com/antst/glm-5.3-flash-tensorfold-4x-spark-ring/tree/v0.1.0)
instead has `scripts/cluster.py` and requires its own `--config`. Its config must
select the baseline owner, original NCCL 2.30.7 library, port 8888 and bootstrap
port 29551, with the baseline's own qualified topology/environment. Do not pass
the candidate's config or qualification receipts. Set the baseline profile to
the explicit values shown here, then check and start from that checkout:

```sh
python3 -B /path/to/phase-one/scripts/cluster.py check --config /path/to/phase-one/local/config.json --context 1048576 --parallel 8 --multi-window 64 --cache-entries 32 --max-tokens 32768 --cache-gib 24 --reserve-gib 20 --port 8888
python3 -B /path/to/phase-one/scripts/cluster.py start --config /path/to/phase-one/local/config.json --context 1048576 --parallel 8 --multi-window 64 --cache-entries 32 --max-tokens 32768 --cache-gib 24 --reserve-gib 20 --port 8888
python3 -B /path/to/phase-one/scripts/cluster.py status --config /path/to/phase-one/local/config.json
```

Replace each generic path with that deployment's checkout. Both variants retain
the original 1M context limit, eight lanes, 64 verify rows and 24 GiB extra cache;
the historical installed launcher keeps its default 20 GiB reserve. All eight
lanes share the cache pool. A full 1M input
and eight simultaneous 1M requests have not been tested. Check fresh container
identities and semantic/long-context behavior after restoration: historical
model-success receipts and the candidate's NCCL 2.32.3/RoCEnante proof do not
certify newly started original NCCL 2.30.7 workers.

Clients use the selected rank-zero management address at port **8888** for either
exclusive recipe. Stop the candidate with its own launcher and the original
with its own launcher before switching; the examples above preserve each
recipe's bootstrap, container, library and cache ownership. Start only the
chosen stack, verify it, and keep the client URL unchanged. No port redirect
is needed. The retained 18888 measurements describe the initial test endpoint,
not the current default.

For a lightweight final observation between workloads, use the metadata and
network checks below from the management host. Choose new receipt filenames;
keep them private and leave the canonical plan inventory untouched:

```sh
python3 -B scripts/experiment.py status --config local/config.json > state/final-owned-status-NEW.json
python3 -B scripts/network.py check --config local/config.json --inventory state/network-inventory.json > state/final-mesh-check-NEW.json
python3 -B scripts/network.py inventory --config local/config.json --output state/final-readonly-network-inventory-NEW.json
```

`status` reads container identity, state and ownership labels. The mesh check
reads current boot IDs, exact routes/TC placement, helper PID identity and
guardian health. Inventory reads network/GPU-process facts and configuration
hashes. These commands generate only ignored local observations/events; they
do not perform inference or change host configuration. They are not a new
qualification or a temperature/throughput measurement. Avoid observer queries
during timed benchmarks, and use fresh validation for restored model workers.


## 2026-10-14 P8 prefill study

The study stopped the owned GLM workers and read-only monitor under the user's
approved maintenance window. It tested original-Mia107, ported 2048 and ported 4096
profiles on private port 18897, preserving raw failed attempts, exact ownership,
source identities and before/after receipts. Each completed arm passed 30 model
semantic requests and 115 actual-collector benchmark requests. Two 4096 attempts
stopped at the host-state gate because OS-managed temporary IPv6 addresses rotated
or expired; a fresh qualification in a stable lifetime window passed. These
were not payload failures, and no address lifetime or network setting was changed.

The selected update installs a verified 441-file TensorFold source overlay,
adds the bounded adjacent-ring exchange and atomic controller-health identity,
selects 4096-row solo prefill with two 2048-row microbatches and the decode side
stream, and enables P8 admission. The 1M per-request context, 39.5625 GiB extra pool,
64 retained snapshots, 4 GiB state cap, reasoning budgets, model/image/draft and
native-library pins stay documented independently. Network routes, NIC/driver
settings, forwarding policy and firewall are unchanged by this study. Image
services remain removed.

Production restoration relocates the identical qualified profile to 8888, checks
all ranks' image/source/command/environment/boot/PID bindings and kernel/paging
state without submitting inference, and starts a newly identified monitor with
a fresh matching configuration sample. The management-host start/stop wrappers
continue to use their no-parameter entry points; the installed control verifies
the selected P8 engine manifest and flags before launch. No autostart unit is added.

Complete site-specific command, mutation, backup and worker receipts stay private.
The release includes sanitized measurements, source patch/manifests, builder,
transport/runtime helpers, tests and the byte-exact upstream collector closure.
Eight admitted requests share the finite pool; eight independent 1M histories
and co-resident Qwen performance need separate qualification.
