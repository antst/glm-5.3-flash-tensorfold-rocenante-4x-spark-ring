# Correctness before performance tuning

No candidate speedup is established by source inspection, compilation or CPU
tests. Keep native/model performance receipts separate and report the exact
image, libraries, transport sources, network plan and serving settings.

Qualification proceeds through source/import and native ABI checks, hardware
mesh payload checks, byte-exact FP32/BF16/integer gathers against NCCL, aliased
inputs, mixed-size eager/graph replay, all-function payload/error counters,
native health and asymmetric preparation-cache/restart checks. Full model
correctness is an additional gate after collective equality.
Collective and full-model receipts are separate. The model validator binds the
actual four containers, pinned image, live NCCL mappings, engine/drafter sources,
per-host environment, profile and native launch geometry. It runs 30 requests:
three cases, each serial without drafts, solo with drafts, and eight concurrent
with drafts. The facts and code shape must pass, and every visible drafted or
concurrent answer must match its serial answer byte for byte. Generated code is
parsed, not executed. Exact API request deltas reject competing inference.

Before/after traffic proof requires bidirectional RDMA traffic with zero error
deltas on all 16 HCA functions and live model-owned RC/RTS QPs on every function.
RoCEnante additionally requires payload on all 24 native origin-QP paths and
all 8 hardware-only diagonal TC rules. Host-only runtime reports establish
health without GPU synchronization. Positive diagonal native transit is not
required for the direct-neighbor NCCL control.

## Source versions and proof scope

The original phase-one baseline is the published NCCL 2.30.7 recipe at commit
`d6f455248bfc9c83187d89e000ec8fbdf7c19546`. This recipe uses the same
digest-pinned Mia image and unmodified TensorFold engine/drafter, with NCCL 2.32.3
source commit `12df1a11afad322be5a204a2db890161cbf8131d` and its recorded patch.

The first matched NCCL 2.32.3 control and initial native 16 MiB qualification used
adapter SHA256
`287599371bc246e1c188a64b4e6c25b2cdbfe8b561848c04e5e2a6020f6cbd75`
and qualifier SHA256
`16339d3cd69bc4c1571a86f65213bfa569bcad1e30147eacaa3ba5c39ef6fe93`.
They predate explicit bounded-block/report auditing and later network rollback
hardening. Their 30-request model receipts are historical evidence for that
source/configuration, not qualification of a later candidate.

The bounded-block adapter is SHA256
`ebbd0c83359448c617dbcc9e1b34d8d2e8529504165fbf8d95de631f647927b0`;
the model validator with explicit requested/actual geometry auditing is SHA256
`659e82d7e297f046182d74d81ea500979a7425ba412d76db7c194c434f9857a5`.
The genuine Mia 89-test CPU suite passed for the updated source. Each transport
variant still needs fresh matching hardware collective and full-model proofs.
Use [provenance.json](../provenance.json) and the complete source/configuration
identity in each result, rather than treating success for an earlier source
version as success for the current deployment.

## Controls

Retain the original phase-one service as a control. Also compare NCCL 2.32.3
alone with RoCEnante plus the same NCCL build so library upgrades are separated
from hybrid dispatch. Keep image, TensorFold/weights/draft revisions, prompts,
sampler flags, memory pool, lanes and cache policy identical in that matched
comparison. Record CPU affinity, clocks/temperature, draft acceptance and target
steps so a throughput change is not incorrectly assigned to transport.

Use alternating arms when practical and repeated waves for selected settings.
The current tuning screen and final protocols have different scopes:

| Protocol | Generation | Prefill | Inference requests |
| --- | --- | --- | --- |
| Transport screen | SparkDash prose/code/counting, 256 output tokens, two waves at C1/C8, plus 32-token workload warmups | Pinned RigMark raw-token 8192/65536 depths, one cold and immediate identical replay each, 8 output tokens | 57 generation + 4 prefill = 61 |
| Final SparkDash | Same three workloads, 400 output tokens, three waves at C1/C4/C8, plus 32-token workload warmups | Repeated chat prefixes; not a cold-prefill test | 117 measured + 3 warmups = 120 |
| Full pinned RigMark | Three decode workloads × five runs, 4096-token allowance and completed-output gates; code concurrency C1/C2/C4 × three waves, 256-token allowance | Raw-token 8192/32768/65536 depths, three cold/replay pairs each, 8 output tokens | 15 decode + 18 prefill + 21 concurrent = 54 |

The screen retains the executed SparkDash sampler/body: temperature 0, top_p 1,
thinking off, `ignore_eos:true`, no stop strings, and exact fixed output usage.
`min_tokens` is retained for protocol fidelity but is not implemented by this
TensorFold server. The EOS override and final token count enforce the length.
The final SparkDash command uses
`--tokens 400 --repeats 3 --concurrency 1 4 8`; changing those flags changes its
workload. C2 is available in the client but is not part of this final matrix.

The prefill screen calls the unchanged pinned RigMark prefill implementation,
with exact raw token depths and the same token IDs for the immediate warm
replay. Distinct screen nonces keep the cold inputs fresh across candidates.
Chat thinking-effort settings and the chat sampler seed do not apply to that
raw-token prefill route. This reduced screen ranks candidates; one cold/replay
pair is insufficient for a final repeated-prefill performance claim.

The full RigMark wrapper uses clean upstream commit
`d8353e93b274e8d880ab14a5df2a55c87d7bee16`, seed 20260912, and the retained
low-effort chat body. Its 4096/256-token allowances and completed-output gates
differ from SparkDash's forced fixed length. Keep the two clients' results
separate. A full 54-request run is final validation; a prefill-only subset or
`--skip-prefill` run must be labelled with its actual reduced scope.

Report SparkDash decode from final token usage and the first/last nonempty
content arrival span, not SSE event counts. Per stream it is `(tokens-1)/span`;
concurrent aggregate decode uses the sum of `(tokens-1)` over the shared earliest
first-content/latest last-content span. End-to-end throughput includes prefill
and uses total tokens over the shared request interval. Server decode timing is
separate supporting evidence. RigMark effective prefill is exact input tokens
divided by time to first token; cold input and cached replay remain separate.
C16 and different verify budgets require their own memory/context comparison.

## Tuning

Start with the current two-path peer map and safe forwarded posting window.
Measure message latency over the real TensorFold gather sizes before selecting
the RoCEnante minimum/maximum admission cutoff. Sweep NCCL channels and relevant
native chunk/window settings with substantial payload on both PCIe roots.
Compare hairpin queue configurations only with matching safe posting windows.
The bounded GPU launch grid is another candidate: `native_blocks` 4/8/16,
default 8, with 512 threads fixed. The genuine constructor and actual runtime
geometry are checked before preparation and bound in the prepared query and
all-rank policy. An 8-to-16 cap change affects send messages above 128 KiB; smaller
messages keep the same grid. A blocks16 speedup is a hypothesis to measure,
not an assumed consequence of admitting more blocks.

The P8 target verify window produces FP32 partials of 16 KiB per row up to
64 rows, or 1 MiB. DFlash2 partials are 128 KiB per active stream, up to 1 MiB.
In a 2048-row HC split prefill, the FP32 partial is 32 MiB and its BF16 owned share
is 4 MiB. These complete-message sizes motivate admission-ceiling candidates
of 1/4/16 MiB; actual scheduler chunk sizes are workload-dependent. Changing
cutoffs, blocks, forwarded window/chunk size or NCCL channels changes the bound
identity and requires fresh qualification before serving or timing a candidate.

After choosing transport parameters, evaluate concurrency and verify budget
without silently reducing context or changing model arithmetic. Keep settings
that pass correctness and improve the workloads being recommended. Retain
regressions and inconclusive runs; report the best measured configuration and
the tested search range rather than claiming a universal maximum.
Preserve raw outputs, failed/inconclusive trials, warmup counts, exact request
deltas and counter windows in new private receipts. Context 1M and eight lanes
do not establish a full 1M-input test, eight simultaneous 1M inputs, vision
correctness, or reboot recovery. Those remain separate untested scopes.

## Reproduce the final benchmarks

Run from the shared recipe checkout against an idle endpoint at the common
rank-zero API port **8888**. Use the same selected config for the matched NCCL
and native modes; stop and start with `experiment.py` when switching those modes.
When switching between the original and this recipe, stop both with their own
launchers and follow [recovery](system-changes.md) before starting the selected
stack. The client URL stays stable; owner/container, bootstrap and local cache
separation remains. Qualify both transports after any bound configuration change.
Run the model validator before timing each newly started arm.

Retained final benchmark receipts were measured on the initial 18888 test
endpoint. The current reproduction commands below use 8888; historical metrics
and endpoint records are not rewritten.

The maintained clients use Python's standard library. Fetch a clean copy of
the pinned RigMark source into ignored state, and prepare its required appliance
metadata. Replace every `CHANGE ME` entry with the actual hardware, topology,
immutable model/image/engine revisions, quantisation, cache format, scheduler
and competing workloads. Set `context_limit` to the actual served limit.
Metadata describes the observed deployment; copying the example alone is
insufficient. Record boundary temperatures/clocks and payload/error counters
alongside each run.

```sh
git clone https://github.com/alexellis/rigmark.git state/rigmark-source
git -c safe.directory="$PWD/state/rigmark-source" -C state/rigmark-source checkout --detach d8353e93b274e8d880ab14a5df2a55c87d7bee16
git -c safe.directory="$PWD/state/rigmark-source" -C state/rigmark-source status --porcelain
cp state/rigmark-source/metadata.example.json local/rigmark-metadata.json
```

After completing that private metadata file, use the rank-zero management
origin URL, without `/v1`. Replace the documentation address below. Every
output directory must be new:

```sh
python3 -B scripts/validate-model.py --config local/config.json --mode rocenante --output state/final-native-model
python3 -B scripts/run-sparkdash.py --url http://192.0.2.11:8888 --output state/final-native-sparkdash --tokens 400 --repeats 3 --concurrency 1 4 8 --workloads prose code structured --serving-lanes 8 --context-window 1048576 --metadata local/rigmark-metadata.json
python3 -B scripts/run-rigmark.py --source state/rigmark-source --url http://192.0.2.11:8888 --output state/final-native-rigmark --metadata local/rigmark-metadata.json
```

Repeat with the identical profile in `nccl` mode and separate metadata/output
receipts. SparkDash must report `passed:true` and exactly 120 accounted
inference requests. Full RigMark must retain a successful `run-audit.json`,
exactly 54 accounted requests, all 15 decode completion gates, no competing
requests and a clean pinned source identity. Its wrapper's exit code alone is
insufficient: inspect the audit and completion gates. The commands above create
model and client receipts; the model validator's counter window covers only
its own 30 requests. Release measurements additionally retain separate private
before/after captures and qualified traffic assessments for each benchmark.
To make the same network claim, retain counters, exact hardware-rule identity
and GPU telemetry around each timing window. Those assessments must establish
both PCIe roots/all 16 functions and, in native mode, all eight hardware-only
diagonal rules. The final report keeps the repeated benchmarks separate from
the shorter parameter screens.
