# Transport tuning

Selected generation profile: **4 MiB gather ceiling, 64 KiB forwarding window, 16 KiB chunks, four NCCL channels, 512 threads and an eight-block cap**. The engine remains TP4/P8 with a 1M configured context and shared verification budget 64.

Eight current-source configurations passed fresh NCCL/native GPU qualification, all 30 model correctness requests and all 61 screening requests. Both native cache layouts, all 16 HCA functions, all eight ASIC diagonal rules and zero new mandatory error deltas passed. The earlier historical anchor is kept in a separate source group.

Each generation cell has two 256-token waves; each prefill depth has one cold/immediate-replay pair. C8 is eight concurrent clients in the P8 service. Values below are median tokens/s; these short screens guide selection and do not establish statistical significance. The repeated 400-token final benchmarks are separate.

| Ceiling MiB | Window/chunk KiB | NCCL channels | Block cap | C8 prose | C8 code | C8 counting | C8 geometric mean | Cold 64K TTFT s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 128/32 | 4 | 8 | 222.72 | 283.64 | 506.09 | 317.39 | 35.709 |
| 1 | 128/32 | 4 | 8 | 225.34 | 294.96 | 511.49 | 323.95 | 35.020 |
| 1 | 64/16 | 4 | 8 | 225.85 | 289.33 | 536.49 | 327.28 | 34.951 |
| 4 | 128/32 | 4 | 8 | 222.89 | 283.86 | 541.54 | 324.79 | 35.735 |
| 4 | 128/32 | 4 | 16 | 226.58 | 290.35 | 540.95 | 328.93 | 35.661 |
| 4 | 128/32 | 8 | 8 | 226.07 | 290.78 | 524.13 | 325.40 | 35.976 |
| 4 | 128/64 | 4 | 8 | 222.83 | 290.95 | 558.74 | 330.88 | 35.596 |
| 4 | 64/16 | 4 | 8 | 230.75 | 298.29 | 554.03 | 336.59 | 35.897 |

The equal-weight C8 geometric mean is a selection heuristic across these three generation workloads. It is not a universal maximum. The selected 4 MiB/smaller-window candidate leads this heuristic at 336.59. The 1 MiB/same-window alternative scores 327.28 and has 2.63% lower observed cold-64K TTFT (34.951 versus 35.897 seconds); it is a reasonable prefill-priority alternative. C1 differences are small and mixed. Neither candidate dominates all objectives, and overlapping wave ranges prevent a significance claim.

Larger chunks, eight NCCL channels and 16 GPU blocks did not provide a clear generation advantage. No clock, power, model, quantization, image or compute-kernel changes were made for this search. Hairpin queue size stayed 8192; no NIC reload was performed between transport trials. Other lane counts, verification budgets, posting windows and block/channel combinations have not been tuned for this new transport.

The earlier historical 16 MiB anchor had an 8K cold TTFT of 5.752 seconds. The current-source 16 MiB repeat measured 4.499 seconds, close to the current 4 MiB baseline at 4.485 seconds. The earlier apparent cutoff gain was not reproduced; its cause remains unproven. Cold uses a unique prompt prefix, while immediate replay measures prefix-cache reuse.

Exact samples, ranges, settings, source groups and receipt hashes are retained in [final-benchmarks.json](../results/final-benchmarks.json). Network counters prove payload on both PCIe roots and hardware forwarding, not peak utilization of both 200 Gb/s cables. Full 1M input, eight simultaneous 1M inputs, vision and physical reboot remain untested.

Validated private screen-summary SHA256: `205a7b8620fb7417393565244367353d21545abc432873563a3c21517e1b47a5`.
