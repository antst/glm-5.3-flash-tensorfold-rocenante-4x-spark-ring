# Transport and arithmetic

TensorFold's communicator gathers four contiguous rank shards into a rank-major
output on the current CUDA stream. Its hyper-connection code sums FP32 partials
in rank order. Some metadata can also travel as integer bits in FP32 storage.
The port first replaces the gather implementation while keeping that existing
arithmetic and metadata representation.

SparkRing's prepared RoCEnante gather copies raw words. Eligible calls must
agree on world/rank, dtype, byte count, layout, device, capacity and backend on
all four ranks. The first qualified native admission ceiling was 16 MiB;
the selected generation-priority profile uses 4 MiB after testing smaller
ceilings. Admission uses the complete
per-rank send message, with four rank shards in the receive buffer. Messages
below the minimum, above the ceiling, or outside supported geometry use NCCL
before any native enqueue. NCCL retains pair exchange and the other operations.

The pinned model has width 4096, 45 main layers and five DFlash2 layers. Its
important gather sizes are:

| Caller | Per-rank send bytes | Eight-lane profile |
| --- | --- | --- |
| Main target verification FP32 HC partial | 16 KiB × target rows | Up to 64 rows together: 1 MiB |
| DFlash2 FP32 HC partial | 128 KiB × active draft streams | Up to eight streams: 1 MiB |
| Prefill HC split FP32 partial | 16 KiB × prompt rows | 2048-row example: 32 MiB |
| Prefill HC split BF16 owned share | 2 KiB × prompt rows | Same 2048-row example: 4 MiB |

Main verification has two principal HC gathers per layer, and drafting has two
per drafter layer. Candidate/token/control gathers are smaller. Four-byte
startup and scheduler words, and small padded tails, are raw copies rather
than floating-point conversions. In world 4 HC split, the FP32 exchange gathers
each rank's whole partial before owned rows are reconstructed; the later BF16
gather has a different size. The actual prefill row count can vary with scheduling
and chunk grouping. The 2048-row example is a size calculation, not a measured
claim that every prompt is processed in chunks of that size.

A 1 MiB ceiling admits the largest P8 target/draft partials while leaving larger
prefill messages on NCCL. A 4 MiB ceiling can also admit the BF16 share in the
2048-row example. Its 32 MiB FP32 partial remains on NCCL even at the supported
16 MiB ceiling. Select cutoffs from measured workload results and retain their
qualification identity; the largest supported ceiling is not a speed claim.

## Native launch geometry

`experiment.native_blocks` accepts only integer 4, 8 or 16, with default 8.
Threads are fixed at 512. The adapter calls the genuine pinned
`roce.AllReduce(..., threads=512, blocks=native_blocks)` constructor. It checks
that the constructor exposes those explicit keyword-only parameters before
initialization, and verifies the resulting `_threads` and `_blocks` before
building the prepared query. The requested settings are exchanged across all
ranks; the prepared framework also binds `threads` and `max_blocks` into its
immutable runtime setup. A constructor mismatch or ignored option fails closed.
There is no vendored kernel modification or alternate preparation framework.

The block setting caps a size-dependent grid; it does not force every gather
to use that many blocks. With 512 threads, the pinned grid aims for two 16-byte
packs per thread, rounds the required block count up to a power of two, then
applies the cap. At a cap of 8 or 16, 16/64/128 KiB messages use 1/4/8 blocks.
Increasing the cap from 8 to 16 changes messages above 128 KiB. Small solo target
verification can therefore use the same grid in both configurations; C8 and
prefill are useful workloads for measuring the larger cap. Extra blocks can
also add synchronization cost, so a benefit requires measurement.

Host-only controller reports retain requested and actual launch geometry.
The model and collective validators require exact integer 512/configured blocks
for RoCEnante. NCCL control reports retain requested settings but explicitly
show no actual native runtime. Configuration, report, prepared setup and source
identities must match a fresh qualification for each candidate.

Prepared launchers and scratch must exist before graph capture. The port binds
TensorFold source hashes, transport/proxy/framework hashes, library identity
and the real peer/HCA/GID mapping. It preserves in-place gathers and stable
buffer/stream lifetimes. Errors after native enqueue are fatal; retrying a native
failure through NCCL could diverge rank order or leave GPU waits unresolved.

The pinned B12X preparation framework is an added dependency, isolated from the
original image. Its collective initialization uses a CPU Gloo group and the
TensorFold rendezvous store. Rank-local selection-cache hits must not let one
rank skip a preparation collective; compiled-program caching can be retained.
Cold and asymmetric-cache initialization require separate qualification.

The dependency package contains unchanged, inventoried source from B12X commit
`a83336581a3a907076e60797df69ab66df5a2ff1` and the selected prepared transport.
Its preparation session differs from one historical upstream image preimage.
The adapter consequently disables that session's rank-local selection cache
and prepares the same distributed seed on every rank. It retains compiled
program caching. This policy changes the adapter, not vendored source files.

One TensorFold communicator is shared by the main model, DFlash and concurrent
serving. The adapter installs before the engine constructs that communicator.
It prepares and synchronizes native work before weights and model graphs are
created. CUDA graph replay and eager native operations share an event ordering
policy across streams; native gather capture uses one parent stream per graph.
The existing weight-prefetch side stream and prefill overlap remain enabled.
Unsupported capture usage is rejected before enqueue. A native error poisons
the controller and reports a fault to the four-host mesh health guard.
Neither eager nor graph failures switch transports after native work begins.
That historical launch-geometry tuning retained TensorFold's rank-major layout,
FP32 summation, EXL3/Q4 compute, HC split, draft policy and graph ordering. The
current source-overlay prefill changes are documented separately in
[prefill-port.md](prefill-port.md).

The historical Mia96 image passes imports of all nine prepared components and the
native GNU11 proxy ABI check without initializing CUDA or creating RDMA
endpoints. CPU adapter tests also pass in that image. These checks establish
dependency compatibility at import time; GPU compilation and model operation
are separately qualified.
The bounded-block implementation passed the genuine-image89-test CPU suite.
Earlier collective/model success belongs to its recorded source version;
CPU success does not reuse those receipts for changed adapter settings.

The current prepared backend expects direct connections to all three peers,
including the opposite rank. It cannot run unchanged on neighbor-only IP
reachability. The [mesh plan](network.md) provides the opposite-peer paths.

The [benchmark method](benchmark-method.md) distinguishes the original NCCL
baseline, matched NCCL 2.32.3 control, initial native 16 MiB results, and subsequent
bounded-block candidates. Native 1M context is configured; a full 1M input, vision
inference, and recovery across an actual host reboot remain untested.

Source contracts:

- [Current GLM profile](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/profiles/glm53-flash-nvfp4-spark-tp4/config.json).
- [Prepared transport](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/integrations/vllm/rocenante_prepared/README.md).
- [Prepared gather API](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/integrations/vllm/rocenante_prepared/roce/roce_oneshot.py).
- [Raw gather kernel](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/integrations/vllm/rocenante_prepared/roce/_allgather_cute.py).

## Mia107 compatibility

The later [Mia107 image update](engine-upgrade.md) retains the prepared transport, math and graph hooks. Three guarded engine sources remain byte-identical; the multi-stream scheduler guard changes to the reviewed image bytes. DFlash prelaunch uses the existing current-stream/candidate event ordering. CPU checks pass 90/90, Mia’s GPU prelaunch test passes, and fresh collectives, 30 model requests and both full benchmark suites pass. CUDA indexer-score kernels may fall back to Triton after a logged build failure; active-path claims require runtime log evidence.
