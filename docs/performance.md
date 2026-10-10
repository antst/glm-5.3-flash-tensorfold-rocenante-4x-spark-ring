# Performance

The tables below retain the historical Mia96 transport/recipe comparison. Fresh Mia96 versus Mia107 image-upgrade results are documented separately in [the upgrade report](engine-upgrade.md).

RoCEnante improves lower-concurrency generation in this tested profile: aggregate
decode medians are **9.7–12.1% higher at C1** and **4.6–6.3% higher at C4** than
the matched NCCL 2.32.3 control. At C8, gains are modest—about 0–3%
(**0.21–3.25%** from the unrounded medians)—with inconclusive overlapping code
and counting ranges. Cold-prefill throughput medians are slightly lower than the local
control; this result establishes no cold-prefill speedup.

The selected profile uses a **4 MiB gather ceiling, 64 KiB forwarding window,
16 KiB chunks, four NCCL channels, 512 threads and an eight-block cap**.
See the [finite tuning study](tuning.md) for the tested alternatives and tradeoffs.
Exact samples, ranges, source/settings pins and receipt hashes are in
[final-benchmarks.json](../results/final-benchmarks.json).

These retained performance runs used the source qualified for **v0.1.0**,
with the experimental API on port **18888**. **v0.1.1** moves API admission
and the default endpoint to **8888**; compute and transport settings are
unchanged. The port migration is validated separately with fresh source-bound
GPU qualification and 30-request model checks. The benchmark suites were not
rerun for v0.1.1, and all measurements and raw-source hashes below retain
their original run scope.

Four Sparks, pinned Mia TensorFold, P8 (eight decode lanes), 1M configured context. C1/C4/C8 count concurrent clients; C1 still uses the P8 profile.

Both final arms passed 30 semantic/equality requests, 120 SparkDash requests and the full 54-request RigMark suite. Source, configuration, image, patched NCCL library and settings match; only transport mode differs. The JSON retains exact receipt precision, samples and hashes.

The **original recipe** uses patched NCCL 2.30.7 on the initial ring. Its fresh
120-request SparkDash rerun was measured before the hardware-mesh changes;
[matched-phase-one-baseline.json](../results/matched-phase-one-baseline.json)
retains its C1/C4/C8 medians and ranges. The **matched NCCL 2.32.3 control**
below uses the new recipe, fabric and settings with native dispatch disabled.
It is a separate control, not the original recipe. Original-versus-native
comparisons include the NCCL upgrade and recipe/fabric changes; the matched
2.32.3 comparison isolates dispatch within the new setup.

## Difference from the matched NCCL control

Percent differences use the unrounded JSON medians. Both profiles retain P8; C1/C4/C8 denote clients. Positive values indicate higher throughput.

| Workload | Clients | Decode difference | E2E difference |
| --- | ---: | ---: | ---: |
| Prose | C1 | +11.50% | +11.20% |
| Prose | C4 | +6.32% | +5.96% |
| Prose | C8 | +3.25% | +2.25% |
| Code | C1 | +12.13% | +11.52% |
| Code | C4 | +5.09% | +4.76% |
| Code | C8 | +0.21% | +1.10% |
| Counting | C1 | +9.70% | +9.47% |
| Counting | C4 | +4.56% | +4.08% |
| Counting | C8 | +0.67% | +1.32% |

The C8 code and counting wave ranges overlap between modes. Prose has a modest +3.25% decode median difference; C8 E2E differences are +1.10–2.25%. These three-wave results support workload-specific observations, not statistical significance or a universal speedup. The finite parameter screens use a different output length and repeat count.

## Final SparkDash matrix: NCCL 2.32.3 control and native

400 output tokens, three waves per cell, greedy sampler, thinking disabled, DFlash retained. Values are tokens/s: median (minimum–maximum). Aggregate decode uses usage tokens minus one per stream divided by the span from earliest first visible content to latest last visible content. E2E includes prefill and queueing.

| Workload | Clients | NCCL 2.32.3 control decode | Native decode | NCCL 2.32.3 control E2E | Native E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | C1 | 88.01 (87.83–88.27) | 98.14 (97.86–98.14) | 86.19 (85.89–86.28) | 95.84 (95.53–95.84) |
| Prose | C4 | 156.91 (156.59–157.34) | 166.82 (165.95–169.76) | 155.16 (155.05–155.33) | 164.41 (164.17–166.41) |
| Prose | C8 | 216.03 (215.59–217.05) | 223.04 (219.55–223.98) | 214.66 (214.20–215.00) | 219.49 (217.53–222.51) |
| Code | C1 | 130.07 (128.89–134.38) | 145.84 (145.75–146.39) | 124.42 (123.06–128.21) | 138.75 (138.52–139.33) |
| Code | C4 | 216.56 (211.64–217.98) | 227.59 (223.61–229.66) | 209.54 (207.82–211.78) | 219.52 (218.26–221.60) |
| Code | C8 | 291.66 (288.91–293.79) | 292.26 (290.11–300.85) | 283.68 (282.99–283.80) | 286.81 (286.05–290.76) |
| Counting | C1 | 166.32 (166.21–167.20) | 182.46 (182.01–182.73) | 159.79 (159.57–160.88) | 174.93 (174.70–175.08) |
| Counting | C4 | 369.32 (368.84–371.60) | 386.16 (384.03–397.09) | 359.92 (359.72–360.69) | 374.59 (373.55–379.02) |
| Counting | C8 | 491.81 (490.18–495.05) | 495.10 (493.40–497.30) | 481.02 (479.60–487.10) | 487.35 (485.53–489.58) |

## Mia and our original/final P8 recipes

| Workload | Clients | Mia TP2/P8 | Mia TP3/P8 | Original TP4/P8, NCCL 2.30.7 decode | Final TP4/P8 native decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | C1 | — | 65.7 | 87.98 (87.55–88.43) | 98.14 (97.86–98.14) |
| Prose | C4 | 103.2 | 121.8 | 157.76 (157.01–157.93) | 166.82 (165.95–169.76) |
| Prose | C8 | 130.8 | 166.0 | 215.85 (215.62–220.46) | 223.04 (219.55–223.98) |
| Code | C1 | — | 100.4 | 135.46 (135.45–136.05) | 145.84 (145.75–146.39) |
| Code | C4 | 126.7 | 165.3 | 219.47 (218.70–226.16) | 227.59 (223.61–229.66) |
| Code | C8 | 167.0 | 211.5 | 291.45 (290.64–292.07) | 292.26 (290.11–300.85) |
| Counting | C1 | — | — | 169.10 (168.92–169.88) | 182.46 (182.01–182.73) |
| Counting | C4 | — | — | 374.88 (374.66–374.98) | 386.16 (384.03–397.09) |
| Counting | C8 | — | — | 498.30 (496.26–537.49) | 495.10 (493.40–497.30) |

Values are aggregate decode tokens/s, median (minimum–maximum) for our runs.
A dash means unpublished: Mia TP2/P8 C1 is unavailable for both prose/code workloads;
Mia TP3/P8 supplies the C1 points. No P8 counting reference is published.
TP2/P4 values belong to another lane profile and are not substitutes.

[Mia TP2/P8](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/7d42f905388a8d7fa91e2dc9458e0a4a68001540/README.md#L49-L61), [TP3/P8](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/7d42f905388a8d7fa91e2dc9458e0a4a68001540/README.md#L432-L445). Published point values are not local TP2/TP3 reruns. Their original output caps, raw flags and repeats are unavailable; 400 tokens is the retained SparkDash harness default. TP2/P4 has a stated 2200 MHz cap; P8/TP3 per-boot clocks are unverified. Mia smoothing defaults to 400 ms; our smoothing is off. This comparison does not isolate tensor-parallel scaling.

## Full RigMark

Exact pinned RigMark protocol, seed and client settings; 15 official completed-output gates, 18 cold/replay prefill requests and 21 C1/C2/C4 short-code responses. Decode includes separately emitted reasoning where present. Concurrent headline rates are E2E. Warm replay measures prefix reuse, not fresh-input processing. Code/prose completion gates establish non-truncated output; they do not prove arbitrary generated-code correctness.

| Metric, tokens/s | Alex TP4 v0.0.1 | Original phase one, NCCL 2.30.7 | Matched NCCL 2.32.3 control | Final native |
| --- | ---: | ---: | ---: | ---: |
| Completed code decode | 71.326 | 111.723 (106.840–115.079) | 111.01 (106.44–114.12) | 122.20 (117.44–125.13) |
| Completed prose decode | 31.236 | 62.945 (60.386–63.532) | 62.96 (60.14–63.11) | 71.23 (67.98–71.52) |
| Completed structured decode | 109.2 | 157.059 (156.996–157.218) | 157.54 (157.23–157.74) | 172.14 (171.62–172.41) |
| 8192 cold prefill | 1978.119 | 1798.647 (1798.448–1802.576) | 1827.84 (1440.60–1831.98) | 1824.95 (1817.01–1824.99) |
| 32768 cold prefill | 1971.556 | 1836.370 (1831.133–1842.939) | 1856.83 (1854.94–1858.70) | 1844.85 (1844.77–1846.11) |
| 65536 cold prefill | 1964.652 | 1816.426 (1816.192–1817.239) | 1835.44 (1833.53–1835.77) | 1823.91 (1822.50–1824.81) |
| 65536 cached replay | 36074.876 | 983831.985 (948395.126–995292.045) | 1090975.68 (944758.39–1149613.21) | 1003537.25 (988745.06–1035700.18) |
| C1 short-code aggregate E2E | 43.023 | 82.748 (77.535–85.969) | 83.46 (77.80–84.33) | 92.69 (86.50–95.67) |
| C2 short-code aggregate E2E | 72.456 | 118.949 (114.014–121.843) | 115.78 (114.12–118.46) | 123.64 (121.45–127.17) |
| C4 short-code aggregate E2E | 110.814 | 162.275 (157.384–165.935) | 161.43 (144.33–166.88) | 161.07 (160.22–169.52) |

The original NCCL 2.30.7 column is the separately published full RigMark run,
not a new RigMark rerun performed beside the final arms. Its medians/ranges
come from the [phase-one raw receipt](https://github.com/antst/glm-5.3-flash-tensorfold-4x-spark-ring/blob/v0.1.0/results/rigmark-result.json).
That raw receipt retained unknown Git identity; the separate
[phase-one source proof](https://github.com/antst/glm-5.3-flash-tensorfold-4x-spark-ring/blob/v0.1.0/results/rigmark-source-proof.json)
verifies the clean pinned RigMark checkout. All recipe columns retain their
own run timing and source scope; Alex remains the exact v0.0.1 published arm.

Against the matched local control, native cold-prefill median throughput is
0.16% lower at 8K, 0.65% lower at 32K and 0.63% lower at 64K.
The repeated final cold-prefill results do not show a speedup. The earlier
old-source 16 MiB screen's 8K gap was not reproduced by its current-source
16 MiB repeat; it cannot be attributed to the cutoff alone. Cached replay
rates are a separate prefix-reuse measurement. Neither healthy traffic
counters nor these timings establish a kernel, clock or collective cause.

[Alex's exact v0.0.1 receipt](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless/blob/e2d0839a92f118a0a874abcfc1fe8012ee69978e/data/rigmark/glm53-nccl-v0.0.1-tp4-20260912.json). Alex used vLLM/Libert NVFP4/Marlin, FP8 KV, static DFlash2 k7, 262144 context and six sequences. This recipe uses Mia EXL3/TensorFold, FP8 latent KV, Q4 dense, adaptive/copy DFlash and 1M/P8. It is a full-stack recipe comparison; local matched native versus NCCL arms provide the transport comparison.

## Historical phase one and screening

[Published phase one](https://github.com/antst/glm-5.3-flash-tensorfold-4x-spark-ring/blob/v0.1.0/docs/benchmark-comparison.md) used NCCL 2.30.7. Its P8/C8 decode medians were 215.71 prose, 290.75 code and 537.49 counting tokens/s; these are separate historical measurements, not this final control. Its cold 64K rate was 1816.426 versus Alex 1964.652 tokens/s, with cause unresolved.
The fresh original SparkDash rerun used in the Mia comparison above measured
498.2960967729492 tokens/s for C8 counting, not the historical 537.4867239431126
median. Its highest wave was 537.4946492822849 tokens/s. The historical median,
the fresh median and the fresh maximum are distinct observations; none replaces
the matched NCCL 2.32.3 control or the final native result.

Tuning screens used 256 output tokens and two C1/C8 waves, plus one cold/replay pair at 8K/64K. They select candidates and must not be mixed with final 400-token/three-wave results or the full RigMark suite.

All benchmark windows passed bidirectional payload and zero mandatory error deltas on 16 HCA functions. Native windows additionally passed positive ASIC payload on all eight hardware-only diagonal rules with zero observed software transit/drops. GPU temperature/clock readings are boundary samples, not continuous telemetry. The 1,048,576-token context limit is configured; these performance runs tested
raw prompts through 65,536 tokens. They do not establish full 1M-input performance
or capacity for eight simultaneous 1M inputs. No vision inference or physical
reboot was tested; reboot recovery remains unqualified.
