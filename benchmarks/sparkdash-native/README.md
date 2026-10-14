# SparkDash actual-collector benchmark bundle

This runner calls the **actual, byte-exact SparkDash DecodeBenchManager and
PrefillBenchManager**, not a rewritten rate calculator. It is prepared for
fresh matched GLM baseline/candidate workers configured with eight lanes.
The original baseline and both optimized candidates completed 115 requests
each. The three-way results and memory tradeoff are summarized in
[the study report](../../docs/sparkdash-native.md). All 90 corresponding measured
decode responses match across the arms. Production restoration/selection is
a separate operator gate.

The historical pin fab8dadff951d1c8cc1450a657805d8e3ea5c9f6 is also GitHub's
latest commit through 2026-10-09 23:59:59 UTC. Its commit date is
2026-10-09 19:38:14 UTC. Source URLs, hashes, the query and provenance limits are
in source-pin.json. Six previously archived files match byte-for-byte.
The isolated installed undici 8.9.0 package matches the upstream lock integrity;
Node 26.11.1 is present. Original LICENSE/LICENSE-MIT/NOTICE are retained.

Source references:
[DecodeBench](https://github.com/MiaAI-Lab/sparkDash/blob/fab8dadff951d1c8cc1450a657805d8e3ea5c9f6/server/collectors/DecodeBench.js),
[PrefillBench](https://github.com/MiaAI-Lab/sparkDash/blob/fab8dadff951d1c8cc1450a657805d8e3ea5c9f6/server/collectors/PrefillBench.js),
[streaming helper](https://github.com/MiaAI-Lab/sparkDash/blob/fab8dadff951d1c8cc1450a657805d8e3ea5c9f6/server/collectors/LlmStreaming.js).

## Run only after the operator prepares authorized private workers

The runner accepts an explicit private IP or localhost and port. Production
port 8888 is rejected. It reads APIhealth/models and submits benchmark inference;
it never changes serving configuration, services, hosts, model files or network.
Use the same reviewed runtime profile, clocks/power/isolation and these settings
for both variants. The caller records live image, model/draft, source, library,
container, profile, pool and resource identities separately. A supplied
--metadata JSON is archived/hash-bound, not independently attested by this CLI.

From the recipe root, enter this bundle and install only its locked dependency
into this directory. No model or service is installed by these commands.

~~~sh
cd benchmarks/sparkdash-native
npm ci --ignore-scripts --no-audit --no-fund --cache ./npm-cache
node --test test-protocol.mjs
~~~

Then, from this directory, inspect an offline plan. Replace PRIVATE_API with
the actual dedicated endpoint. --plan performs no HTTP and writes no files.

~~~sh
PRIVATE_API=http://192.0.2.11:18898
node run-sparkdash.mjs --url "$PRIVATE_API" --suite all --plan
~~~

192.0.2.11 is a documentation placeholder: replace it with the real private
address; the parser intentionally rejects public/documentation IP addresses.
Use a NEW output beneath this directory's runs/ subtree:

~~~sh
node run-sparkdash.mjs --url "$PRIVATE_API" --suite decode \
  --decode-types prose,code --concurrency 1,2,4,8 \
  --max-tokens 400 --repeats 3 \
  --output ./runs/baseline-decode-NEW \
  --metadata /path/to/operator-baseline-runtime.json

node run-sparkdash.mjs --url "$PRIVATE_API" --suite prefill \
  --prefill-sizes 8192,16384,32768,65536,131072,262144 \
  --output ./runs/baseline-prefill-NEW \
  --metadata /path/to/operator-baseline-runtime.json
~~~

Repeat exactly these settings against the candidate, with different NEW output
paths and its own runtime metadata. --suite all runs decode then prefill in one
receipt. --repeats 3 runs three complete native decode jobs per selected type,
each including the upstream warmup; it does not assert Mia used three repeats.
Default one repeat is one UI-equivalent job per type. --debug is available but
off by default, matching the upstream manager default. Hardware polling is not
supplied. SIGINT/SIGTERM cancels this client's current benchmark HTTP requests;
the operator must confirm retirement before maintenance or switching workers.

~~~sh
node compare-sparkdash.mjs --baseline /path/to/baseline-decode-NEW \
  --candidate /path/to/candidate-decode-NEW \
  --output ./runs/decode-comparison-NEW.json
~~~

Comparison requires passed exclusive-counter receipts, identical source pin
and settings. It reports native aggregate decode TPS and native prefill rates,
actual prompt/cached tokens and timing, plus decode text/usage consistency.
A text mismatch is recorded for correctness review; it does not silently erase
the measured data or claim that runtime correctness is qualified.

## Exact pinned protocol

| Item | Actual pinned collector |
|---|---|
| Decode workloads | Exact shared prose/code prompts; optional structured/JSON |
| Source UI default | Structured; this study explicitly selects prose/code |
| Concurrent code | Different Python task per stream; C1 is binary_search |
| Concurrent prose | Shared prompt with exact stream i/N suffix; C1 has no suffix |
| Warmup | One per job, max 32; no min_tokens/ignore_eos/stop forcing; code uses warmup_noop |
| Measured decode | Default max 400, min 400, ignore_eos true, stop []; temperature 0, top_p 1 |
| Thinking off | enable_thinking=false, thinking=false, thinking_mode=disabled in chat_template_kwargs |
| C1/C2/C4/C8 | Sequential waves; native Promise starts, no added thread barrier |
| Stream token count | Final usage preferred; upstream otherwise estimates chunks |
| Per-stream decode | (completion_tokens−1)/(last−first nonempty streamed answer/reasoning delta) |
| Aggregate decode | Sum post-first tokens/(latest last−earliest first), rounded by native collector |
| Decode timeout | 360s per request and per wave |
| Prefill warmups | Target 512, then 4096; output cap 8, no fill forcing |
| Calibration | Three fresh UUID-salted 16-token-target requests; median TTFT |
| Prefill samples | Fresh salt per sample, deterministic filler; actual usage tokens retained |
| Prefill target lengths | Study selects 8K/16K/32K/64K/128K/256K; source UI defaults 4K/8K/16K/32K |
| Repeats per size | 3 through 32K; 2 through 128K; 1 above |
| Prefill rate | Backend timings when available; otherwise uncached prompt tokens/(TTFT−calibrated overhead), correction capped at 50% |
| Median row | Lower middle rate sample for even counts, with that sample's tokens/timing |
| Prefill timeout | min(45min, max(90s, 60s + 8ms × target tokens)) |
| Warm replay | No exact-body replay in upstream PrefillBench |

Optional --replay-prefill replays the exact last cold sample body once per
selected size **after** the native cold sweep. It is explicitly supplemental,
records reused tokens and timing, and never changes the cold rows. Prefixes can
be evicted before this follow-up; reuse is observed rather than assumed.

For prose+code, one repeat and all four concurrency levels: 32 decode requests.
The six selected prefill sizes: 19 requests including warmups/calibration.
Combined: 51. Supplemental replay adds 6. Three native decode repeats produce 96
decode requests. Authoritative usage, complete DONE/EOF, zero reasoning, fixed
measured decode caps and exact endpoint request-count deltas are additional
receipt gates; missing usage, retries or partial rows fail the matched run.

## Difference from the existing Python run-sparkdash.py

| Existing portable derivative | This runner |
|---|---|
| Reimplements SSE/aggregation in Python | Calls original collector and streaming helper |
| Forces32-token warmup, reasoning_effort=none | Preserves unforced native warmup and its exact template flags |
| Explicit simultaneous thread barrier | Preserves native asynchronous Promise starts |
| One warmup then multiple sweeps | A full native job/warmup for each explicit repeat |
| Python unrounded aggregate and median | Preserves original rounded wave metrics; reports across-job median separately |
| Strict first/last content timing | Preserves upstream answer/reasoning delta timing; receipt rejects emitted reasoning |
| No native prefill collector | Preserves native calibration, fresh samples, repeat selection and correction |
| Same default400 alone does not prove historical identity | Exact source pinned; historical Mia UI flags/repeats still unknown |

Capture uses an import-resolution hook only for the helper's undici import.
Source files are unchanged. It taps the original response reader in memory and
flushes raw request JSON, response SSE bytes, usage, timing and SHA receipts
after each job, avoiding disk writes during measured waves. The capture adds
small client work and is used equally for both variants. Raw artifacts are
private and include synthetic benchmark prompts/model outputs; do not publish
these receipts without sanitizing them.

## Offline verification and changes

~~~sh
node --test test-protocol.mjs
node --check run-sparkdash.mjs
~~~

CPU checks substitute in-memory Responses for both HTTP transports. They run
actual managers and the full CLI, exercising split UTF8/CRLF, authoritative
usage, request accounting, source/default flags, all four concurrency levels,
prefill calibration/repeats and failure gates. They make no real HTTP requests.
changes.jsonl records wrapper output mutations in the ignored run area; upstream
manager checkpoints are retained in each run's own manager-state directory.
Raw run artifacts, node_modules, npm-cache and CPU fixture artifacts are ignored.
Keep them private. npm dependencies/cache remain local to this bundle.
The staged port changes only the private client's help-path examples; collector,
prompt and metric source bytes are unchanged. The public reports retain the
actual private runner hashes that produced the measured results.
