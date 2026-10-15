# Cache and memory monitoring

The read-only monitor samples TensorFold health and all four hosts every 30 seconds.
It records aggregate pool occupancy, saved-state tensors, active lanes, request counters,
cached-prompt fractions and observed completion throughput. Host data include
MemAvailable/MemFree, swap capacity and paging counters. It logs no prompt text
or per-conversation token histories and submits no inference.

From the management host, using your configured shared checkout:

~~~sh
python3 -B scripts/cache-monitor.py --config local/config.json start
python3 -B scripts/cache-monitor.py --config local/config.json status
python3 -B scripts/cache-monitor.py --config local/config.json stop
~~~

Default output is the configured state directory's cache-monitor subdirectory.
Use --state-dir /absolute/path/cache-monitor before the action for another location.
Start/stop/sample create local metadata and logs; retain their commands and results
in your deployment change journal. Status is read-only. Existing start does not
replace a healthy monitor. Stop signals only its verified owned monitor process,
using Linux pidfd plus exact arguments, boot and process-start identities.

latest.json holds the latest collection; summary.json holds sampled memory minima,
cache/stream watermarks, counter deltas and collection errors. samples-*.jsonl retain
bounded history: at most seven days and 64 MiB. Summary watermarks cover the lifetime
of the unchanged configuration, rather than only the rotated log window.
An offline API or failed host collection is an explicit error, never a zero reading.
Status reports data freshness and the owning process. A changed configuration
stops monitoring rather than silently mixing profiles; review it and restart the monitor.

Completion throughput is observed activity over a sampling interval, including idle
time; it is not a benchmark. Prompt/cache/prefill counters update on finished or
cancelled requests. Large ongoing cold prefills may not yet appear. Restart or
collection gaps invalidate counter comparisons where detected. Host-wide memory
and paging include other workloads, and sampled minima can miss short transients.

## NVMe tier observations

For control protocol 3, health samples also retain rank-zero entries/blocks,
pending writes/generations, stored/free disk bytes, bounded write/index/read
host-memory fields, small-I/O scratch size and global committed/failed-generation
counters. The global counters describe ordered all-rank acknowledgment; local
disk bytes/free space describe rank zero only. This monitor does not collect
disk usage from every other rank. Keep separate all-rank storage/worker evidence
for qualification. Old control protocols are rejected.

A rising pending count alone does not prove loss; compare settled global
acknowledgments, failures, actual request cached-token counts and restore
results. Raw /health exposes further I/O counters. Persisted compatible points
may survive a clean worker restart while RAM state and current-boot counters
reset. Counter resets, config changes and collection gaps remain explicit
boundaries. Read-only monitoring does not flush or repair snapshots.

## Assessing eight full context windows

The current service admits eight requests. Monitor status changes no serving
settings and does not qualify eight independent full-context histories. For the exact Mia107 GLM-5.3-Flash TP4/W64 profile it provides an
unqualified sizing estimate for eight independent 1M allocations plus 530,432 spare
rows. The estimate uses 6,780 bytes per additional row and excludes unknown extra
P8 fixed-buffer costs. It is disabled for an unmatched image/profile. Its admission count is reported
separately from this unqualified all-RAM estimate. The monitor does not infer
qualification from its estimate or automatically incorporate the separate NVMe
retention proof; that proof is documented independently.

With the qualified 6,098,944-row arena, the same all-RAM target needs 2,820,096
additional rows: approximately 17.81 GiB per rank before additional overhead.
The historical 6,752,256-row P6 arena needed about 13.68 GiB and the
6,821,888-row P6 arena about 13.24 GiB. Compare that cost
against sampled available memory during realistic long-context traffic, including
retained-state growth and other host workloads. The 12 GiB runtime and 16 GiB startup
comparison margins are policy checks, not enforced OS reservations.

Eight concurrent requests sharing a finite arena are different from eight full
contexts simultaneously resident in RAM. The [NVMe tier](nvme-tier.md) instead
targets persistent long histories between turns; its sequential eight-near-1M
private qualification passed with 2,105 all-host samples per rank and a lowest
sampled MemAvailable of 20.839 GiB. The same profile passed production
readiness on API 8888 with a fresh matching monitor; see the
[sanitized result](../results/nvme-tier-p8-20261015.json).
This result does not qualify eight simultaneous full-context allocations.
Current free host RAM does not expand a
preallocated pool. A P8 trial must explicitly select and freshly qualify its
source/profile, actual pool allocation, all-host memory and cache replay behavior;
these monitoring estimates cannot replace that trial.

No package, model, GPU worker, route, forwarding policy or boot unit is changed by
the monitor. It runs on the management host and uses existing strict SSH identities.
It continues collecting host data if GLM stops. After a management-host reboot,
start it explicitly or from your site's GLM startup wrapper; no autostart unit is
installed by this portable recipe.

The [six-lane cache qualification](cache-retention.md) records historical pool
and memory limits. Full-profile private NVMe qualification and production
readiness passed. The v0.1.4 P8
short-prompt/cold-prefill measurements were made without the tier and remain
separate evidence.
