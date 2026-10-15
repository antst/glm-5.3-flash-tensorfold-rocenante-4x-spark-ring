# Local NVMe cache tier on four Sparks

The v0.1.5 update adds a persistent local cache tier to the Mia107
TensorFold prefill overlay. **Full-profile private qualification: PASSED.**
Eight near-1M retained histories, full-profile semantics and both transports
passed, in addition to the 64K pressure and controlled process-restart gates.
**Production readiness: PASSED on API 8888.** The same P8/1M profile, actual
four-worker source bindings and fresh matching monitoring were verified on
2026-10-15 at 09:21:35 UTC. See the
[sanitized qualification and production result](../results/nvme-tier-p8-20261015.json).
This document makes no new tokens-per-second or real-agent timeout claim.

The v0.1.3 cache tables and v0.1.4 SparkDash tables remain historical evidence.
The v0.1.4 compute benchmarks were measured **without this NVMe tier**. Its
prefill optimizations remain in the new source; their old measurements do not
measure the combined serving profile.

## Selected qualified configuration

| Setting | Selected value |
|---|---|
| Tensor-parallel ranks / admission lanes | 4 / 8 |
| Per-request context, prompt plus output | 1,048,576 tokens |
| Requested extra cache / startup reserve | 39.5625 / 16 GiB per rank |
| Explicit RAM token-pool ceiling | 6,098,944 rows |
| Retained RAM snapshots / tensor-state cap | 64 / 4 GiB per rank |
| Reasoning retention | 8,192 tool-call replies / 128 Mi characters |
| Host-local store root | /var/tmp/spark-ring-nvme-tier-v2 |
| Local store quota / free-space floor | 256 / 64 GiB per rank |
| Minimum persisted prompt / retained turns | 8,192 tokens / 4 per conversation |
| Write/index host budget / small-state read budget | 512 / 256 MiB per rank |
| Write staging target / read blocks | 256 MiB / 16 per rank |
| Restore deadline | 300 seconds |
| Control protocol / snapshot format | 3 / tensorfold-cuda-spill-blocks-generation-2 |

The quota is a ceiling, not preallocation or a guarantee of free capacity. The
64 GiB floor must remain available on each local filesystem; insufficient space
can refuse persistence. The staging settings bound separate buffers and pending
work; they are not a bound on total process memory. The 4 GiB tensor-state cap
also excludes the main token pool, model weights and transient working buffers.
Measure actual peak memory on all four hosts.

The explicit pool is smaller than the historical RAM-only pool. The gross
pool/row-scratch difference is not a net free-memory gain: the tier adds write,
read, metadata and I/O buffers. Six million pool rows do not fit eight full
1M allocations at once. Eight mixed-length active requests may share the RAM
arena; eight independent long histories are retained on disk between turns.
RAM admission and disk retention are different capacities.

## Source and TP4 coordination

Build the `4096-nvme` overlay with the [portable engine builder](../engine/README.md).
Its complete 443-file manifest is
`9c83663da0f875931e5e721ca42e4177a3a83e2d3f0a72abe935a1a5364f1f7b`.
The source includes the original 4096-row prefill port plus a corrected derivative
of [Mia PR124](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/124)
at pinned head `f32226b3cc5f443117d6d2411b1f2433c52f33a6`.
It is not an unmodified upstream patch.

Each rank saves its own rows, recurrent/draft state and token identities to its
own NVMe filesystem. There is no cross-rank shared-disk KV service. A shared
immutable computation generation identifies one coherent TP4 snapshot.
Restore must select a committed generation common to all four ranks; the
same tokens alone do not establish identical state.

The engine coordinates persistence status, release, restore reservations,
read readiness, cancellation and fallback through ordered rank operations.
Background disk threads do not enter collectives. Row holds protect cache
content until copying is safe, and bounded staging can delay admission rather
than overwrite data still being saved. Restore reserves the fixed-state budget
before GPU copies. Slow or temporarily busy reads retain healthy disk points;
missing/corrupt data is handled separately. The coordinated stop path keeps
followers alive while rank zero performs the bounded all-rank flush.

Protocol 3 describes the control sequence. Generation-2 describes the disk
format and compatibility namespace. Raw PR snapshots and older incompatible
namespaces are not accepted as equivalent. A source/model/arithmetic change
can intentionally make an old namespace unusable; retaining files does not
override that identity check.

## What retention means

The tier persists eligible saved checkpoints. It is not a transcript archive
or an infinite cache, and it has no promise to keep every intermediate turn.
Count, byte, disk quota, per-conversation retention, shared-prefix policy and
free-space pressure can retire checkpoints. Saving/restoring requires exact
token-prefix identity, including the selected template and any reconstructed
reasoning text. A changed prefix can still require prefill.

A local write acknowledgment is not the engine's all-rank durable acknowledgment.
Wait until pending generations settle and the global committed counter advances
before treating a newly saved generation as durable. During a clean process
restart, RAM and current-boot counters clear while compatible disk manifests can
survive. Recovery must establish the common all-rank generation again.
The measured clean-restart gate does not establish arbitrary power-loss behavior.

## Completed bounded hardware evidence

The following gates used a private P8 API with a 131,072-token context and a
262,144-row RAM pool. Eight independently tokenized synthetic histories each had
65,536 prompt tokens and a 32-token greedy drafted completion. Their combined
524,288 prompt tokens exceeded the RAM pool.

| Gate | Requests | Observed disk-hit replays | Replay wall median |
|---|---:|---:|---:|
| Fresh controls plus two replay rotations | 24 | 13 of 16 replays | 0.533 s, disk-hit subset |
| Same gate, all RAM and disk replays combined | Included above | 16 replays total | 0.519 s |
| All-four-worker clean restart, replay only | 8 | 8 of 8 | 0.534 s |

Disk-hit replay wall ranges were 0.490–0.585 seconds before restart and
0.490–0.594 seconds after restart. These are HTTP request wall times including
the 32-token continuation, not pure disk-copy latency or a generation benchmark.
Every exact replay reported all 65,536 prompt tokens cached and matched its
fresh control's visible continuation and generated token IDs. Every history
had an actual disk restore. Persistence settled with all-rank acknowledgments.

The restart phase began with empty RAM cache and zero inference requests.
It submitted zero fresh controls: it reconstructed the same private synthetic
fixtures and replayed them from disk. All four worker identities changed and
the persisted point-manifest set was preserved. These are controlled
process-restart results, not a reboot, fault-injection or full-capacity claim.

## Completed full-profile private qualification

The final private profile used P8 admission, a 1,048,576-token context and the
6,098,944-row RAM pool. Eight independent prompts each contained **1,046,528
tokens** and requested **32 greedy drafted completion tokens**. All eight fresh
controls were uncached, followed by two exact replay rotations: **24 exclusive
requests**. The retained prompt total was **8,372,224 tokens**, above the RAM pool.

All 16 replays reported the complete 1,046,528-token prompt cached and matched
the corresponding fresh visible continuation and generated token IDs. Every
history was restored from disk at least once; 11 of the 16 replay requests
observed a disk load and five were RAM hits. All-rank durability settled after
each request without new persistence or restore failures.

| Request subset | Samples | HTTP wall median, s | HTTP wall range, s |
|---|---:|---:|---:|
| Fresh near-1M controls | 8 | 376.392 | 374.445–377.157 |
| All exact replays, RAM and disk | 16 | 3.262 | 1.885–3.827 |
| Disk-hit exact replays only | 11 | 3.375 | 3.127–3.827 |

These wall times include the HTTP request and 32-token continuation. They are
not pure disk-transfer latency, TTFT measurements or decode throughput. The
subsequent durability polling is outside the HTTP wall timer. The run completed
within its two-hour request deadline.

Each rank had **2,105 independent memory/paging samples** during the long test:

| Rank | Minimum sampled MemAvailable, GiB | Host swap-in/out counter delta | Maximum GLM / cgroup swap bytes |
|---|---:|---|---|
| 0 | 20.839 | 0 / 0 | 0 / 0 |
| 1 | 24.414 | 0 / 0 | 0 / 0 |
| 2 | 23.835 | 0 / 0 | 0 / 0 |
| 3 | 24.191 | 0 / 0 | 0 / 0 |

No monitored kernel diagnostic event or target OOM kill was observed. These are
sampled observations, not an OS reservation or proof that every short transient
and other-process victim was excluded. Host paging counters are systemwide;
the selected GLM process and cgroup were monitored separately.

Both full-profile NCCL and prepared-RoCEnante qualifications passed. A separate
**30-request** model gate passed code, factual and arithmetic cases with equal
serial/drafted/P8 concurrent outputs and payload proof on all 16 HCA functions.
Private workers then completed the coordinated all-rank flush and stopped.

The complete private qualification binds source, profile and raw evidence with
SHA256 `752c38407c72618b97e1e403209e21388ecc273d04a874f973158a0758aced2d`.
The near-1M API result SHA256 is
`0ad81677faefc56d576e87b40e40788f422cb7ee8efd7d2bc5a49938de534808`.
Private paths, token fixtures, worker/boot identities and credentials are not
exported. The [sanitized combined result](../results/nvme-tier-p8-20261015.json)
records the private checks and final production readiness.

**Production status: verified ready on API 8888** with P8 admission, 1M context,
6,098,944 pool rows and control protocol 3. The final readiness check submitted
no production inference; model correctness was checked on the private API
before restoration. The private pass qualifies sequential retention and
recovery for this bounded synthetic workload. Eight simultaneous full 1M
contexts still do not fit the RAM pool. Real-agent branching/edited reasoning,
long-generation quality, arbitrary power loss, co-resident workloads and a DSH
timeout guarantee remain outside these tests.

## Build, qualify, operate and recover

Copy the full portable example into ignored local configuration; preserve your
own inventory, trust store, paths, owner and model/native-library pins. Build a
new source output, using `4096-nvme`, and bind its exact manifest and flags.
The `tier.root` path is host-local and must never be the shared checkout.
Give it sufficient local disk headroom; no model download or package upgrade
is required when the existing pins are already present.

Review the offline plan before host operations:

```sh
python3 -B engine/build.py --output local/engine-nvme --manifest 4096-nvme --plan
python3 -B scripts/experiment.py plan --config local/config.json
```

A model API qualification must use a private port, leaving production clients
out of request accounting. Qualify both transport modes against the new
source/profile before serving; unchanged healthy mesh state does not require
applying the network plan again. Keep all-rank memory/paging monitoring active.
See [cache retention](cache-retention.md), [monitoring](monitoring.md) and
[system changes/recovery](system-changes.md).

Always stop through this checkout's owned launcher so rank-zero flush runs
while followers remain alive. Check the truthful flush outcome; a successful
container stop alone does not prove every pending generation became durable.
The launcher's ordinary archive captures pre-stop logs, so retain a separate
shutdown log stream/receipt when qualification needs the final flush output.
Keep the local store through a clean restart.
Do not manually delete snapshots or substitute an old source/qualification
receipt to force compatibility. For rollback, preserve the new store and logs,
stop the owned stack, restore the exact previous config/source/receipts together,
and requalify after a boot or other invalidating change. RAM-only rollback can
cold-prefill; incompatible snapshots are intentionally not used.
