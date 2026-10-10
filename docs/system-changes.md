# System changes and recovery

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

The recipe uses existing system packages, drivers and CUDA tools. Its serving
image, TensorFold compute engine, weights, quantization and model arithmetic
remain pinned to the original Mia deployment. Patched NCCL and the isolated
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
