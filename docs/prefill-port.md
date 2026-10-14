# Mia TP4 prefill port on the hardware-forwarded ring

The 2026-10-14 study keeps the pinned Mia107 image, GLM EXL3 weights,
DFlash2 checkpoint, prepared RoCEnante gathers and patched NCCL 2.32.3.
It ports selected compute changes from Mia's TP4 branch at
`d22003070575c2774de0f010062aa57bae06f3c7` into an immutable TensorFold
source overlay. The original image remains an input and recovery option.

The overlay reduces duplicated solo-prefill work: each rank prepares its own
rows in the DSA front, and a 4096-row solo prompt chunk can run as two aligned
2048-row microbatches. A separate decode optimization overlaps the Q4 shared
expert with routed experts on another CUDA stream.
The small 16-head attention case uses four warps instead of eight. Mixed
prefill with active decoding retains the existing path. FP32 partials and
source-rank summation order are retained; MSA and BF16 partial compression
are disabled.

## Communication

Prepared RoCEnante still handles eligible raw gathers up to 4 MiB, with
ConnectX hardware forwarding for opposite peers. Larger gathers use NCCL.
The new HC split needs destination-specific exchanges, which the prepared
gather API does not expose. Direct opposite-peer NCCL connects failed on this
ring, so the adapter provides a bounded two-round exchange through the two
adjacent neighbors.

In the first round a rank sends both direct-neighbor messages and routes the
opposite message's two halves through those neighbors. The second round
forwards those halves to their destination. Receive slots retain their
original source-rank order. This sends four peer-payload equivalents per rank
instead of three for direct all-to-all, and it leaves the existing NIC rules,
endpoint routes and OS forwarding policy untouched. It is distinct from the
ASIC-forwarded RoCEnante gather path.

The helper retains two scratch buffers per supported dtype, with a 64 MiB
retained-storage cap and a 128 MiB old-plus-new growth cap. CUDA allocator
reserves and NCCL workspace are separate from these buffer limits. Scratch
reuse obeys the controller's stream ordering, and graph capture of this eager
exchange is refused. Graph replay of the existing gather path remains tested.

## Source and deployment binding

The engine builder verifies every file of the pinned source before applying
the exact patch and verifies the complete resulting source manifest. The
launcher mounts the overlay read-only, sets its import path explicitly, and
checks the actual imported files on every rank. Engine flags, runtime helper
hashes, model and image pins, transport geometry and topology are bound into
the qualification identity. Changing them requires new qualification.

Controller health is written atomically into each worker's cache directory.
Validation reads this file and checks its live process and boot identity plus
the independently verified engine source digest. Docker log lines are useful
for diagnostics, but are not used as an atomic identity document.

## Validation scope

CPU checks and nine bounded GPU port tests cover own-row and microbatch
arithmetic, decode shared-expert eager/capture equality, and the four-versus-
eight-warp attention case. Four-host qualification additionally exercises the
relay on several CUDA streams, odd and even payload lengths, native gathers
and graph replay, with exact received bytes.

Full-model trials use P8 on a private endpoint, serial/drafted/concurrent
semantic checks, and the actual pinned SparkDash collectors at C1/C2/C4/C8.
Cold prefill measurements use freshly salted prompts through 256K. See the
separate measured result and protocol documentation for the selected chunk
size, performance and memory. These tests do not establish eight independent
1M histories, vision correctness or real-agent quality at the context limit.

## Qwen reuse

The source changes are specific to GLM's DSA/KDA family and HC split. Qwen's
hybrid GDN/full-attention engine needs its own implementation and numerical,
memory and transport qualification. The source binding, controller ordering,
bounded relay and measurement workflow are reusable components; their GLM
results are not Qwen performance results.
