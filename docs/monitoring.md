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

## Assessing eight full context windows

The current service admits eight requests. Monitor status changes no serving
settings and does not qualify eight independent full-context histories. For the exact Mia107 GLM-5.3-Flash TP4/W64 profile it provides an
unqualified sizing estimate for eight independent 1M allocations plus 530,432 spare
rows. The estimate uses 6,780 bytes per additional row and excludes unknown extra
P8 fixed-buffer costs. It is disabled for an unmatched image/profile. Its admission count is reported
separately from its unqualified full-history estimate.

With the current 6,752,256-row arena, this target needs 2,166,784 additional
rows: approximately 13.68 GiB per rank before additional overhead. The
historical 6,821,888-row P6 arena required about 13.24 GiB. Compare that cost
against sampled available memory during realistic long-context traffic, including
retained-state growth and other host workloads. The 12 GiB runtime and 16 GiB startup
comparison margins are policy checks, not enforced OS reservations.

Eight concurrent requests sharing a finite arena are different from eight fully
retained independent 1M histories. Current free host RAM does not expand a
preallocated pool. A P8 trial must explicitly select and freshly qualify its
source/profile, actual pool allocation, all-host memory and cache replay behavior;
these monitoring estimates cannot replace that trial.

No package, model, GPU worker, route, forwarding policy or boot unit is changed by
the monitor. It runs on the management host and uses existing strict SSH identities.
It continues collecting host data if GLM stops. After a management-host reboot,
start it explicitly or from your site's GLM startup wrapper; no autostart unit is
installed by this portable recipe.

The [six-lane cache qualification](cache-retention.md) records the measured pool and memory limits. Eight independent full-context histories remain unqualified. The new P8
short-prompt and cold-prefill measurements are separate evidence.
