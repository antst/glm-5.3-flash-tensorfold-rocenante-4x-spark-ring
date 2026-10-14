# Long-context retained cache profile

Release v0.1.3 qualified a six-lane Mia107 configuration for retaining six large,
independent conversations. Cache qualification passed on 2026-10-13T19:58:15.370151+00:00: **48
private growth/replay requests**, followed by **24 private P6 semantic requests**.
The [sanitized result](../results/six-lane-cache-qualification.json) binds the
selected profile and source to these 72 exclusive requests. Production returned
to API8888 with an empty cache. No P6 throughput improvement is claimed.

The new P8 compute overlay retains the same cache request and budgets, but the
historical six-near-1M replay test does not qualify that workload on the new
source/profile. See the [current measured prefill study](sparkdash-native.md).

## Historical selected configurations

| Setting | v0.1.2 published profile | v0.1.3 full qualification | Historical P6 trim |
|---|---:|---:|---:|
| Tensor-parallel ranks | 4 | 4 | 4 |
| Concurrent decode lanes | 8 | 6 | 6 |
| Context tokens, prompt plus completion | 1,048,576 | 1,048,576 | 1,048,576 |
| Extra cache request per rank, GiB | 24 | 40 | 39.5625 |
| Startup reserve per rank, GiB | 20 | 16 | 16 |
| Retained prompt snapshots | 32 | 64 | 64 |
| Retained state budget per rank, GiB | Engine default | 4 | 4 |
| Retained tool-call replies | Engine default: 1,024 | 8,192 | 8,192 |
| Retained reasoning character budget | Engine default: 32 Mi characters | 128 Mi characters | 128 Mi characters |
| Verify window | 64 | 64 | 64 |

The configuration keys cache_entries, kept_bytes_gib, keep_reasoning_entries
and keep_reasoning_mb emit TF_GLM_CACHE_ENTRIES, TF_GLM_KEPT_BYTES_GIB,
TF_GLM_KEEP_REASONING and TF_GLM_KEEP_REASONING_MB on every rank. The last
setting counts characters in this pinned engine; it is not a precise RAM
limit. Omitted optional fields preserve existing engine behavior. The module's
fallback profile remains historical. The current example selects P8 admission
with a prefill source overlay; use the complete recorded historical configuration
when reproducing the P6 result.

For that historical P6 trim, image, model and draft weights, compute code, network plan, prepared
RoCEnante/NCCL settings and API8888 retain their previous pins. This adjusts
configuration and launcher validation; it does not rewrite the KV subsystem.

## Cache budget trim: 2026-10-14

The historical P6 trim requested **39.5625 GiB extra cache per rank**, a reduction
of 0.4375 GiB from the released 40 GiB profile. Six lanes, 1,048,576 total
context tokens, the 16 GiB reserve and all snapshot/reasoning budgets are
unchanged. This reduces the fixed pool request without changing model math.

The approved maintenance and production restoration passed; see the
[sanitized trim result](../results/cache-budget-trim-20261014.json). The private
rank-zero API reported **6,752,256 pool rows**, down from 6,821,888. Six full
context windows require 6,291,456 rows, leaving **460,800 spare rows**. This
exceeds the 6,750,208-row acceptance floor by 2,048 rows. All four ranks started
with the new cache environment and completed collective serving; peer arena
sizes were not directly read through an allocator API.

On an isolated API, six distinct **1,024-token** histories produced 32-token
completions and were replayed exactly: **12 requests**. Every replay reported
all 1,024 prompt tokens cached and matched its original 32 generated token IDs.
The separate P6 semantic validator passed **24 requests**, including
serial/draft/concurrent output equality and native/all-16-HCA payload checks.
Fresh profile-bound RoCEnante qualification passed, production was restored to
API8888 with empty KV, and monitoring was restarted against the new config.

These are startup and short replay/correctness checks. The six near-1M
growth/replay suite was **not rerun** at 39.5625 GiB. Its measured table and
sanitized result below remain the 40 GiB evidence; they do not measure the
trimmed profile. No new throughput benchmark or real-agent timeout guarantee
follows from this change. Branches, overlapping continuations and reasoning
restoration retain the limitations below. The spare rows do not guarantee room
for another large branch. On another cluster, check actual startup pool and
all-host memory rather than assuming the requested budget yields this pool.

## What expires and what occupies memory

The pinned engine has no time-based TTL for prompt snapshots. Count,
saved-state byte budget and pressure on its preallocated token pool can evict
them. A snapshot includes recurrent and draft state outside the pool; the
4 GiB cap bounds retained tensors, not total process memory. A saved prompt
boundary, its underlying token extent and separate reasoning-text metadata
must all remain reusable.

Host MemAvailable does not prove the token pool has room: the pool is fixed at
startup. Prefix reuse also depends on exact token identity. A changed template,
omitted reasoning text, branch or tool history can cause a miss without
exhausting RAM. Mia107 restores earlier tool-call reasoning omitted by clients;
evicting this metadata can change the reconstructed prompt despite retained KV.

Linear growth normally extends an existing extent. A branch or overlapping
continuation can require another large extent. Six context slots need spare
pool rows for admission and temporary state, rather than exactly six times the
context. Cancelling cold prefill does not guarantee all processed tokens become
reusable checkpoints. Restarting clears the in-memory caches.

Larger budgets address capacity and metadata-retention limits. The reported
agent slowdown at roughly 45–47% context has not been traced to one demonstrated
cause. These budgets alone do not prove every real-agent cache miss is fixed.

## Qualification method and required evidence

Use a dedicated API with all four ranks on the selected source/image/profile.
Keep ordinary clients and image-generation workers away. Check idle health,
current-boot source-bound NCCL and RoCEnante receipts, actual startup allocation
on every rank, and memory on all four hosts. API8888 remains production; use a
separate private test port during maintenance.

Acceptance gates: at least **6,750,208 token-pool rows**, at least **16 GiB
MemAvailable per host at startup**, and at least **12 GiB per host during the
test**. These gates are not a guarantee on every system. The pool gate includes
448 Ki rows beyond six full context windows. Record actual pool size rather
than reporting its startup estimate. On runtime memory-floor failure, cancel
this test's own HTTP clients and verify that requests retire.

Build six independent raw token-ID histories using valid ordinary non-special
tokens from the pinned tokenizer. Their first token must differ, before the
first 64-token checkpoint. Every longer prompt must keep its earlier prompt as
an exact prefix. Raw token input avoids templates; it does not test reasoning
reconstruction.

Use progressive prompt lengths **1,024; 262,144; 524,288; 1,046,528**. At each
length submit six concurrent growth requests, then six exact replays: **48
inference requests total**. Use draft=true, temperature=0, max_tokens=32,
ignore_eos=true and return_token_ids=true. Compare the complete 32-token output
sequence for each original/replay pair. Verify exact request-counter accounting
and no foreign traffic.

Lengths are aligned to the 64-token grid. Growth must report
usage.prompt_tokens_details.cached_tokens at least equal to its previous prompt
length; exact replay must report the complete prompt length cached. One output
token would test only the saved head; 32 also exercise restored recurrent and
speculative decode.

Observe six live engine admissions and six running requests during the final,
largest growth wave. Concurrent client futures alone do not establish this.
After each wave check six inferred cached chats and allocator occupied rows
at least six times the prompt length. Snapshot-row sums can overcount shared
extents. Extent IDs are not exposed by the engine, so this is proxy evidence
for independent allocation with an explicit limit.

Retain timings, usage, token-ID hashes, bounded health/memory samples, runtime
identities and receipt hashes. Keep prompt contents, raw logs, management
addresses, private host inventory, authorizations and state paths out of public
artifacts.

## Measured qualification: v0.1.3 at 40 GiB

The arena has **6,821,888 rows per rank**. Six full-size 1M allocations use
6,291,456 rows, leaving **530,432 spare rows**. During the final growth wave,
six active engine admissions and six running requests with at least six full
allocations were observed in **164 samples**. This establishes
simultaneous admission/allocation, not six simultaneous decoding lanes.
Every rank's startup capture retained a 1,048,576-token context and 1,048,592
base cache slots.

| Prompt tokens per history | Growth cached tokens, min–max | Growth wall seconds, min–max | Growth first visible text seconds, min–max | Exact replay wall seconds, min–max | Replay first visible text seconds, min–max |
|---|---:|---:|---:|---:|---:|
| 1,024 | 0–0 | 1.77–4.62 | 1.34–4.39 | 0.54–0.65 | 0.19–0.20 |
| 262,144 | 1,024–1,024 | 149.63–891.26 | 149.24–891.16 | 1.23–1.23 | 0.85–0.90 |
| 524,288 | 262,144–262,144 | 172.19–1023.20 | 171.64–1023.08 | 2.08–2.09 | 1.64–1.71 |
| 1,046,528 | 524,288–524,288 | 408.44–2442.30 | 407.93–2442.17 | 3.67–3.73 | 3.09–3.25 |

Growth reused at least the preceding aligned prompt. Every exact replay
reused its complete prompt and returned the same 32 generated token IDs as its
original. Wall times describe six concurrently submitted requests and include
queued prefill work; cold or extended-context prefill can take much longer than
warm exact replay. Increase client deadlines to cover cold/queued prefill after
a restart. A longer deadline does not solve lost cache or prefix mismatch.

The repetitive diagnostic input is favorable to speculative prediction.
These timings are not ordinary generation rates or an independent cold
reference for output quality. The short semantic suite separately passed
code shape, factual and arithmetic answers, serial/draft/concurrent visible
output equality, and all 16 HCA/native payload checks. Its 24 requests and the
48 cache requests each passed exclusive endpoint-counter accounting.

| Host | Minimum sampled MemAvailable, GiB |
|---|---:|
| spark1 | 19.3018 |
| spark4 | 21.6648 |
| spark3 | 21.5542 |
| spark2 | 21.5589 |

Memory was sampled approximately every two seconds during qualification,
subject to collector overhead. The acceptance floors were 16 GiB at startup
and 12 GiB while running. Sampled minima cannot prove short-lived peaks.
Fresh NCCL and prepared RoCEnante checks passed, followed by restoration of
the selected six-lane service to API8888. Image serving remained removed;
the model/draft/image pins and retained healthy mesh did not change.

The released offline suite contains 126 tests: 124 passed and two requiring
the pinned image/Torch environment were skipped on the management host.
The read-only monitor contributes 29 CPU tests; see [monitoring](monitoring.md)
and its [source](../scripts/cache-monitor.py). Status reads saved local
metadata and does not submit inference:

~~~sh
python3 -B scripts/cache-monitor.py --config local/config.json status
~~~

This qualifies synthetic linear raw-token capacity/correctness, not inference
throughput or realistic agent quality. Branches, overlaps, cancelled-prefill
recovery, stable logical chat IDs, metadata eviction, client reasoning
restoration, vision, mixed image generation and cross-worker KV transfer remain
untested. The largest prompt leaves 2,048 tokens for completion and engine
slack; a full 1M prompt plus a reply exceeds the context.

[Performance tables](performance.md) and [Mia107 upgrade results](engine-upgrade.md)
retain their historical P8 settings. No P6 tokens-per-second value follows from
those results.
