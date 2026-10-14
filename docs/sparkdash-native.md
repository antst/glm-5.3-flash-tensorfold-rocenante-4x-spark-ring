# Actual SparkDash protocol: P8 baseline and prompt-path candidates

The original-Mia107 baseline, optimized 2048-row candidate and optimized
4096-row candidate each completed **115 private inference requests** using
byte-exact SparkDash collectors: **345 requests** across three complete arms.
All three passed their operator gates and cleaned their four owned workers.
All **90 corresponding measured decode responses** matched decoded text and
usage across the three arms. The **4096-row profile was selected and restored
to production port 8888 at 2026-10-14 18:40:22 UTC**, with P8 admission and a fresh
monitor. Its separate GET-only deployment/identity gate passed; see the
[sanitized deployment receipt](../results/production-prefill-p8-20261014.json).

Read the sanitized [baseline](../results/sparkdash-native-baseline-20261014.json),
[2048-row candidate](../results/sparkdash-native-optimized-2048-20261014.json),
[4096-row candidate](../results/sparkdash-native-optimized-4096-20261014.json) and
[three-way comparison](../results/sparkdash-native-comparison-20261014.json).
The [portable bundle](../benchmarks/sparkdash-native/README.md) includes the
original collector source closure, exact locked dependency, recording client,
comparison helper, licenses/provenance and 28 CPU checks.

## Protocol and provenance

The collector pin is fab8dadff951d1c8cc1450a657805d8e3ea5c9f6, also the latest
SparkDash commit through 2026-10-09 23:59:59 UTC. The local study explicitly
selected prose/code, C1/C2/C4/C8, **400 measured output tokens**, temperature 0,
top_p 1 and thinking off. It ran **three complete jobs per type**, each with the
native 32-token unforced warmup: 96 decode requests. Prefill used the native
512/4096 warmups, three tiny calibration requests and 3/3/3/2/2/1 fresh-salt
samples at 8K/16K/32K/64K/128K/256K: 19 requests. Exact endpoint-counter deltas
matched 115 in each arm. Source/hash/receipt bindings are retained in the JSON.

Native decode rates sum post-first-token tokens over the concurrent
first-to-last nonempty-delta window. The decode tables report the median of
three native wave rates, with the complete range. Prefill displays the native
median-rate sample; with two samples it selects the lower middle, rather than
an arithmetic median. Its rate excludes cached prompt tokens and applies the
native bounded TTFT overhead correction. **All selected cold prefill rows
reported zero cached prompt tokens.** UUID salts cause small differences in
actual token counts, which are retained.

Mia's [TP4 ring table](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/d22003070575c2774de0f010062aa57bae06f3c7/README.md#4-sparks-experimental)
reports measurements dated 2026-10-09. Its source pin and README/config/changelog
hashes are recorded in the baseline JSON. These are published point references,
not local reruns. Exact historical UI flags, output caps, repeats and per-boot
clock state are unavailable. The engine/transport differ from this local study,
so the Mia comparison does not isolate any one implementation change.

## Aggregate decode, tokens per second

| Workload | Clients | Mia TP4 published | Local baseline median (range) | 2048-row median (range) | 4096-row median (range) | 4096 vs baseline | 4096 vs 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prose | C1 | 95.2 | 101.23 (101.21–101.70) | 102.22 (102.13–102.55) | 101.99 (101.64–102.80) | +0.75% | -0.23% |
| Prose | C2 | 133.2 | 136.39 (134.02–136.58) | 136.66 (134.87–137.34) | 134.56 (134.55–135.07) | -1.34% | -1.54% |
| Prose | C4 | 163.3 | 169.60 (169.06–169.67) | 172.60 (167.73–172.98) | 170.08 (169.22–172.73) | +0.28% | -1.46% |
| Prose | C8 | 222.4 | 224.20 (224.08–230.53) | 229.00 (224.86–232.06) | 230.94 (225.54–231.56) | +3.01% | +0.85% |
| Code | C1 | 138.6 | 149.01 (148.95–149.30) | 150.01 (149.71–150.34) | 150.62 (150.07–150.73) | +1.08% | +0.41% |
| Code | C2 | 185.1 | 190.56 (189.54–196.73) | 191.66 (191.44–196.90) | 190.80 (190.52–196.51) | +0.13% | -0.45% |
| Code | C4 | 224.5 | 232.90 (229.14–234.13) | 229.31 (219.55–234.24) | 229.53 (221.95–229.54) | -1.45% | +0.10% |
| Code | C8 | 286.0 | 294.17 (293.77–304.40) | 304.47 (297.65–305.61) | 295.01 (294.77–304.60) | +0.29% | -3.11% |

The decode changes are small and several three-wave ranges overlap. Keep the
observed decreases: the 2048-row C4 code median is 1.54% below baseline; the
4096-row C2 prose and C4 code medians are 1.34% and 1.45% below baseline.
Relative to 2048 rows, the 4096-row C8 code median is 3.11% lower. These samples
do not establish a universal decode improvement or a repeatable regression.

## Cold prefill, tokens per second and TTFT

| Target | Mia published TPS / TTFT s | Local baseline TPS / TTFT s | 2048-row TPS / TTFT s | 4096-row TPS / TTFT s | 4096 vs baseline | 4096 vs 2048 |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 2568.8 / 3.33 | 1846.64 / 4.58 | 2526.07 / 3.38 | 2954.51 / 2.91 | +59.99% | +16.96% |
| 16K | 2574.4 / 6.51 | 1795.15 / 9.27 | 2554.80 / 6.55 | 2970.20 / 5.66 | +65.46% | +16.26% |
| 32K | 2583.3 / 12.83 | 1810.90 / 18.24 | 2559.44 / 12.94 | 2963.96 / 11.20 | +63.67% | +15.81% |
| 64K | 2568.0 / 25.66 | 1747.72 / 37.64 | 2309.28 / 28.52 | 2649.64 / 24.88 | +51.61% | +14.74% |
| 128K | 2530.7 / 51.94 | 1733.92 / 75.73 | 2444.67 / 53.75 | 2805.46 / 46.86 | +61.80% | +14.76% |
| 256K | 2462.7 / 106.59 | 1711.57 / 153.30 | 2448.79 / 107.19 | 2828.06 / 92.84 | +65.23% | +15.49% |

The 4096-row cold-prefill rates are **51.6–65.5% above the local original
baseline** and **14.7–17.0% above the optimized 2048-row arm** across these sizes.
At 256K, throughput rises from 1711.57 to **2828.06 tokens/s** and TTFT falls
from 153.30 to **92.84 seconds**; the intermediate 2048-row arm records
2448.79 tokens/s and 107.19 seconds. This refers to the pinned filler workload
and native measurement protocol. It is not a warm replay or a claim about
every agent/tool prompt.

## Source switches and memory tradeoff

All three arms use TP4/P8, 1,048,576 total context tokens, the same pinned Mia107
image/weights, FP8 KV, q4 dense weights, DFlash2, prepared RoCEnante and patched
NCCL. The study retains 39.5625 GiB extra cache, a 16 GiB reserve, 64 snapshots,
a 4 GiB retained-state cap and the larger reasoning-metadata budgets.
All APIs reported **6,752,256 fixed pool rows**. P8 is an admission/decode-lane
setting; this pool is **not eight full 1M context slots**.

| Switch | Original baseline | Optimized 2048-row | Optimized 4096-row |
|---|---:|---:|---:|
| Prompt rows | 2048 | 2048 | 4096 |
| Prompt own-row front | 0 | 1 | 1 |
| Prompt microbatch flag | 0 | 1 | 1 |
| Prompt partials | FP32 | FP32 | FP32 |
| Sparse-attention MSA path | 0 | 0 | 0 |
| Decode side stream | 0 | 1 | 1 |
| Small-prompt warp choice | 8 | 4 | 4 |
| Hyper-connection exchange | gather | p2p | p2p |
| Hyper-connection ring relay | 0 | 1 | 1 |

Several source switches change together relative to the original baseline;
the benchmark does not identify their individual contributions. The two
optimized arms differ in their declared prompt-row setting. These are declared
settings, not proof that every dynamic prompt shape takes the same subpath.
FP32 partials remain selected, and the MSA alternative remains off.

| Host label | Baseline minimum MemAvailable GiB | 2048-row minimum GiB | 4096-row minimum GiB |
|---|---:|---:|---:|
| spark1 | 21.2789 | 19.1949 | 18.6460 |
| spark2 | 22.8256 | 19.2722 | 17.1978 |
| spark3 | 22.9399 | 19.7215 | 18.0585 |
| spark4 | 22.9864 | 19.6273 | 18.0651 |

The baseline has **520 samples per host**, the 2048-row arm **608** and the
4096-row arm **577**. Sampled minimum free memory is lower in the optimized
arms, showing a footprint tradeoff. The durations and workload dynamics
differ; this is not a direct measurement of extra allocated buffer bytes.
Observed swap-in/out deltas, worker/cgroup swap, OOM and checked kernel
diagnostics are zero on every host in all three arms. Private all-rank sample
and diagnostic receipt hashes are retained without raw logs, addresses or
process identities. Sampling cannot prove shorter peaks absent.

Each candidate JSON includes request-body and decoded-content SHA256 values
for all 90 matched measured decode responses, with 400 completion tokens each.
Generated text and raw token data stay private. This establishes the tested
short-prompt decode consistency; it does not establish long-agent quality,
eight simultaneous near 1M caches, image coexistence or a DSH timeout SLA.

## Reproduce the protocol

Arrange an owner-approved maintenance window and qualified idle private
workers. Keep model/image/draft, memory profile, clocks/power and other workloads
controlled and record live identities separately. Run from the management
client on the same network for both arms. The client changes no workers,
configuration or network and rejects production port 8888.

~~~sh
cd benchmarks/sparkdash-native
npm ci --ignore-scripts --no-audit --no-fund --cache ./npm-cache
node --test test-protocol.mjs
node run-sparkdash.mjs --url "$PRIVATE_API" --suite all --plan
node run-sparkdash.mjs --url "$PRIVATE_API" --suite all \
  --decode-types prose,code --concurrency 1,2,4,8 \
  --max-tokens 400 --repeats 3 \
  --output "$PWD/runs/baseline-NEW" --metadata "$BASELINE_RUNTIME_JSON"
~~~

Use the same arguments and a new output for the candidate, then compare:

~~~sh
node compare-sparkdash.mjs --baseline "$PWD/runs/baseline-NEW" \
  --candidate "$PWD/runs/candidate-NEW" --output "$PWD/runs/comparison-NEW.json"
~~~

Raw run areas and local dependencies are
gitignored; publish only a sanitized summary. Optional supplemental exact-body
prefill replay is separately labeled and never changes the cold rows.
A full source/default/protocol discussion and differences from the older
Python derivative are in the bundle README. Pending candidates receive no
numbers or success claim before their complete arm, correctness and accounting
gates pass.
