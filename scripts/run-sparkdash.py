#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Prompt/metric protocol derived from sparkDash, Copyright 2026 Mia's AI Lab.
# Original licences/notices: third_party/sparkdash/. This is a modified client.
"""Run the executed SparkDash comparison protocol; changes only new output files.

This portable derivative has offline tests, not a fresh hardware benchmark.
Use a dedicated idle endpoint without competing traffic. Published Mia results
did not retain their exact output cap or raw harness configuration.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
EXECUTED_RUNNER_SHA256 = "5ccd6fcd355a64964e356c707f98c2526d67179a418cf1b9eff78f81da6caa60"
SPARKDASH_REVISION = "fab8dadff951d1c8cc1450a657805d8e3ea5c9f6"
MIA_REVISION = "7d42f905388a8d7fa91e2dc9458e0a4a68001540"
CODE_TAIL = "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code."
CODE_TASKS = [name + "\n" + spec + "\n" + CODE_TAIL for name, spec in [
    ("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
    ("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
    ("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
    ("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
    ("ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."),
    ("dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."),
    ("edit_distance", "def edit_distance(a, b) -> int: Levenshtein distance between two strings."),
    ("semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b."),
]]
CODE_WARMUP = "warmup_noop\ndef warmup_noop(x): return x unchanged.\n" + CODE_TAIL
PROMPTS = {
    "prose": "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.",
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "code": CODE_TASKS[0],
}
MIA_TP2 = {"prose": {1: 60.4, 2: 79.2, 4: 108.8},
           "structured": {1: 114.7, 2: 147.6, 4: 227.9}, "code": {}}
MIA_TP2_PARALLEL8 = {"prose": {4: 103.2, 8: 130.8}, "code": {4: 126.7, 8: 167.0}}
MIA_TP3_PARALLEL8 = {"prose": {1: 65.7, 2: 93.8, 4: 121.8, 8: 166.0},
                     "code": {1: 100.4, 2: 129.5, 4: 165.3, 8: 211.5}}


def utc():
    return datetime.now(timezone.utc).isoformat()


def payload(model, prompt, tokens):
    # Preserve the actual executed client's payload, including its warmup policy.
    # TensorFold ignores min_tokens; ignore_eos/max_tokens/stop and final usage
    # establish the requested fixed length without assuming support for that key.
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_p": 1, "max_tokens": tokens,
            "min_tokens": tokens, "ignore_eos": True, "stop": [],
            "stream": True, "stream_options": {"include_usage": True},
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def read_stream(response, tokens, started, clock=time.monotonic):
    """Use usage tokens and first/last nonempty CONTENT arrivals, not SSE counts."""
    first = last = None
    content, reasoning = [], []
    usage = timing = finish_reason = None
    done = False
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError(str(event["error"]))
        now = clock()
        if event.get("usage") is not None:
            usage = event["usage"]
        if event.get("tensorfold") is not None:
            timing = event["tensorfold"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                first = now if first is None else first
                last = now
                content.append(piece)
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    ended = clock()
    if not done or not isinstance(usage, dict) or first is None or last <= first:
        raise RuntimeError("incomplete stream or missing token count/timing")
    if reasoning or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
        raise RuntimeError("thinking-off comparison unexpectedly emitted reasoning")
    count = usage.get("completion_tokens")
    if type(count) is not int or count != tokens:
        raise RuntimeError(f"fixed-length comparison expected {tokens} output tokens, got {count!r}")
    if ended <= started:
        raise RuntimeError("nonpositive request duration")
    result = {"started_monotonic": started, "ended_monotonic": ended,
              "first_content_monotonic": first, "last_content_monotonic": last,
              "completion_tokens": count, "usage": usage, "tensorfold": timing,
              "finish_reason": finish_reason, "content": "".join(content),
              "ttft_seconds": first - started, "request_seconds": ended - started,
              "sparkdash_decode_tokens_per_second": (count - 1) / (last - first),
              "end_to_end_tokens_per_second": count / (ended - started),
              "content_events": len(content)}
    if isinstance(timing, dict) and isinstance(timing.get("decode_s"), (int, float)) and timing["decode_s"] > 0:
        result["server_decode_tokens_per_second"] = count / timing["decode_s"]
    return result


def request(args, prompt, tokens, barrier=None):
    query = urllib.request.Request(args.url + "/v1/chat/completions",
                                   data=json.dumps(payload(args.model, prompt, tokens)).encode(),
                                   headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait(timeout=args.timeout)
    started = time.monotonic()
    with urllib.request.urlopen(query, timeout=args.timeout) as response:
        if not response.headers.get("Content-Type", "").startswith("text/event-stream"):
            raise RuntimeError("server did not return SSE")
        result = read_stream(response, tokens, started)
    return {"prompt": prompt, **result}


def phase_metrics(requests):
    decode_span = max(r["last_content_monotonic"] for r in requests) - min(r["first_content_monotonic"] for r in requests)
    request_span = max(r["ended_monotonic"] for r in requests) - min(r["started_monotonic"] for r in requests)
    if decode_span <= 0 or request_span <= 0:
        raise RuntimeError("nonpositive concurrent timing span")
    return {"completion_tokens": sum(r["completion_tokens"] for r in requests),
            "sparkdash_aggregate_decode_tokens_per_second": sum(r["completion_tokens"] - 1 for r in requests) / decode_span,
            "aggregate_end_to_end_tokens_per_second": sum(r["completion_tokens"] for r in requests) / request_span,
            "ttft_median_seconds": statistics.median(r["ttft_seconds"] for r in requests)}


def validate_health(health, args):
    """Known live fields must match; record-only permits missing evidence only."""
    streams = health.get("streams") if isinstance(health.get("streams"), dict) else {}
    idle_fields = {key: health[key] for key in ("busy", "requests_running", "requests_waiting") if key in health}
    idle_fields.update({"streams." + key: streams[key] for key in ("decoding", "prefilling", "filling", "paused", "waiting") if key in streams})
    if any(value for value in idle_fields.values()):
        raise RuntimeError("Endpoint is busy; reserve an idle service without competing traffic")
    profile = {"context_length": health.get("context_length"), "streams_max": streams.get("max", health.get("streams_max"))}
    for key, expected in (("context_length", args.context_window), ("streams_max", args.serving_lanes)):
        if profile[key] is not None and profile[key] != expected:
            raise RuntimeError(f"Live /health {key}={profile[key]!r} differs from recorded {expected}")
    verified = bool(idle_fields) and all(value is not None for value in profile.values())
    if args.health_mode == "required" and not verified:
        raise RuntimeError("Live health must prove idle state, context_length and streams.max; use explicit --health-mode record-only for another API")
    return {"idle_observed": bool(idle_fields), "profile_verified": all(value is not None for value in profile.values()),
            "profile": profile, "health_mode": args.health_mode}


def fetch_health(args):
    try:
        with urllib.request.urlopen(args.url + "/health", timeout=args.timeout) as response:
            health = json.load(response)
        if not isinstance(health, dict):
            raise RuntimeError("/health did not return an object")
    except (urllib.error.URLError, OSError, ValueError) as error:
        if args.health_mode != "record-only":
            raise
        return {"available": False, "error": str(error), "idle_observed": False, "profile_verified": False}
    return {"available": True, "snapshot": health, **validate_health(health, args)}


def request_audit(before, after, expected):
    a, b = before.get("snapshot", {}).get("requests_total"), after.get("snapshot", {}).get("requests_total")
    counted = type(a) is int and type(b) is int and 0 <= a <= b
    delta = b - a if counted else None
    return {"expected_inference_requests": expected, "requests_before": a, "requests_after": b,
            "requests_delta": delta, "request_count_verified": counted and delta == expected,
            "external_requests_or_counter_change_detected": delta != expected if counted else None,
            "limitations": "Counters cover warmups and measured requests, not /health or /v1/models. Matching counts cannot prove absence of all other resource contention."}


def read_receipts(args):
    documents, identities = {}, {}
    for name in ("runtime", "qualification", "metadata"):
        path = getattr(args, name)
        if path is None:
            continue
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError(name + " receipt must be a JSON object")
        documents[name] = raw
        identities[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        if name == "qualification" and document.get("qualified") is not True:
            raise RuntimeError("Provided collective qualification is not qualified")
        if name == "runtime":
            hosts = document.get("hosts")
            if document.get("passed") is not True or not isinstance(hosts, dict) or len(hosts) != 4:
                raise RuntimeError("Runtime receipt must contain four passed ranks")
            if not all(r.get("context") == args.context_window and r.get("parallel") == args.serving_lanes for r in hosts.values()):
                raise RuntimeError("runtime receipt does not match recorded context/decode lanes")
            identities[name]["container_ids"] = {h: r.get("container_id") for h, r in hosts.items()}
            identities[name]["binding_scope"] = "Provided receipt hash/profile/container IDs retained; this HTTP-only client does not independently verify live container or library identity."
    return documents, identities


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="explicit HTTP API origin, optionally ending in /v1")
    parser.add_argument("--output", required=True, type=Path, help="NEW private output directory; use ignored state/ inside this repository")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    parser.add_argument("--workloads", nargs="+", choices=tuple(PROMPTS), default=["prose", "code", "structured"])
    parser.add_argument("--serving-lanes", type=int, default=8, help="record and check existing server lanes; does not change the server")
    parser.add_argument("--context-window", type=int, default=1048576, help="record and check existing context; does not change the server")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--runtime", type=Path, help="optional separately captured four-rank runtime receipt")
    parser.add_argument("--qualification", type=Path, help="optional explicit collective qualification receipt for provenance")
    parser.add_argument("--metadata", type=Path, help="optional private JSON describing clocks, power, pool, image/weights and competing traffic")
    parser.add_argument("--health-mode", choices=("required", "record-only"), default="required", help="record-only explicitly permits missing live profile/idle evidence; known mismatches still fail")
    parser.add_argument("--plan", action="store_true", help="show requests/settings offline; no network calls or file writes")
    args = parser.parse_args(argv)
    if args.tokens < 64 or args.repeats < 1 or args.timeout < 1 or args.serving_lanes < 1 or args.context_window <= args.tokens:
        parser.error("tokens >=64, positive repeats/timeout/lanes, and context-window > tokens required")
    endpoint = urllib.parse.urlsplit(args.url.rstrip("/"))
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in ("", "/v1"):
        parser.error("--url must be an explicit HTTP(S) origin, optionally /v1, without credentials/query/fragment")
    args.url = urllib.parse.urlunsplit((endpoint.scheme, endpoint.netloc, "", "", ""))
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("--output must be a NEW directory; existing receipts are never overwritten")
    root = Path(__file__).resolve().parent.parent
    if args.output == root or root in args.output.parents:
        state = root / "state"
        if args.output == state or state not in args.output.parents:
            parser.error("Private benchmark URLs/results inside this repository must be under ignored state/<new-run>")
    args.workloads = list(dict.fromkeys(args.workloads))
    args.concurrency = list(dict.fromkeys(args.concurrency))
    return args


def summarize(args, phases):
    rows = []
    for kind in args.workloads:
        for concurrency in args.concurrency:
            matching = [p for p in phases if p["kind"] == kind and p["concurrency"] == concurrency]
            speeds = [p["sparkdash_aggregate_decode_tokens_per_second"] for p in matching]
            speed = statistics.median(speeds)
            table = MIA_TP2_PARALLEL8 if args.serving_lanes == 8 else MIA_TP2 if args.serving_lanes == 4 else {}
            baseline = table.get(kind, {}).get(concurrency)
            tp3 = MIA_TP3_PARALLEL8.get(kind, {}).get(concurrency) if args.serving_lanes == 8 else None
            rows.append({"kind": kind, "concurrency": concurrency,
                         "sparkdash_aggregate_decode_tokens_per_second_median": speed,
                         "decode_range": [min(speeds), max(speeds)],
                         "aggregate_end_to_end_tokens_per_second_median": statistics.median(p["aggregate_end_to_end_tokens_per_second"] for p in matching),
                         "ttft_median_seconds": statistics.median(r["ttft_seconds"] for p in matching for r in p["requests"]),
                         "mia_published_tp2_decode_tokens_per_second": baseline,
                         "mia_baseline_serving_lanes": args.serving_lanes if baseline else None,
                         "mia_published_tp3_decode_tokens_per_second": tp3,
                         "percent_difference_from_published_tp3": 100 * (speed / tp3 - 1) if tp3 else None,
                         "ratio_to_published_tp2": speed / baseline if baseline else None,
                         "percent_difference_from_published_tp2": 100 * (speed / baseline - 1) if baseline else None})
    return rows


def main(argv=None):
    args = parse_args(argv)
    expected = len(args.workloads) * (1 + args.repeats * sum(args.concurrency))
    documents, identities = read_receipts(args)
    settings = {"url": args.url, "model": args.model, "tokens": args.tokens, "repeats": args.repeats,
                "workloads": args.workloads, "concurrency": args.concurrency,
                "configured_context_window": args.context_window, "configured_serving_lanes": args.serving_lanes,
                "health_mode": args.health_mode, "expected_inference_requests": expected,
                "warmup_tokens": 32, "provided_receipts": identities}
    if args.plan:
        print(json.dumps({"offline_plan": True, "network_calls": False, "file_writes": False,
                          "output": str(args.output), "settings": settings, "prompts": PROMPTS,
                          "code_task_prompts": CODE_TASKS, "payload_example": payload(args.model, PROMPTS[args.workloads[0]], args.tokens)}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve().read_bytes()
    (args.output / "runner.py").write_bytes(runner)
    for name, raw in documents.items():
        (args.output / (name + "-provided.json")).write_bytes(raw)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    started, warmups, phases = utc(), [], []
    before = after = None
    try:
        before = fetch_health(args)
        (args.output / "health-before.json").write_text(json.dumps(before, indent=2) + "\n")
        with urllib.request.urlopen(args.url + "/v1/models", timeout=args.timeout) as response:
            models = json.load(response)
        if not any(row.get("id") == args.model for row in models.get("data", [])):
            raise RuntimeError("Requested model is not served")
        for kind in args.workloads:
            prompt = PROMPTS[kind]
            warmup = {"kind": kind, **request(args, CODE_WARMUP if kind == "code" else prompt, 32)}
            warmups.append(warmup)
            (args.output / (kind + "-warmup.json")).write_text(json.dumps(warmup, indent=2) + "\n")
            for repetition in range(args.repeats):
                for concurrency in args.concurrency:
                    barrier = threading.Barrier(concurrency)
                    def generate(index):
                        suffix = f" (stream {index + 1}/{concurrency})" if concurrency > 1 else ""
                        measured = CODE_TASKS[index] if kind == "code" else prompt + suffix
                        return request(args, measured, args.tokens, barrier)
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        requests = list(pool.map(generate, range(concurrency)))
                    phase = {"kind": kind, "repetition": repetition, "concurrency": concurrency,
                             **phase_metrics(requests), "requests": requests}
                    phases.append(phase)
                    (args.output / f"{kind}-repeat{repetition + 1}-concurrency{concurrency}.json").write_text(json.dumps(phase, indent=2) + "\n")
                    print(json.dumps({key: value for key, value in phase.items() if key != "requests"}), flush=True)
        after = fetch_health(args)
        (args.output / "health-after.json").write_text(json.dumps(after, indent=2) + "\n")
        audit = request_audit(before, after, expected)
        (args.output / "run-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        if audit["external_requests_or_counter_change_detected"]:
            raise RuntimeError("Request counter delta differs from this client's request count; refuse uncontended comparison")
        rows = summarize(args, phases)
        report = {"schema": "spark-ring-sparkdash-comparison/v2", "passed": True,
                  "started": started, "finished": utc(), **settings, "prompts": PROMPTS,
                  "warmups": warmups, "code_task_prompts": CODE_TASKS if "code" in args.workloads else None,
                  "runner_sha256": hashlib.sha256(runner).hexdigest(),
                  "source": {"sparkdash_revision": SPARKDASH_REVISION,
                             "executed_private_client_sha256": EXECUTED_RUNNER_SHA256,
                             "scope": "Portable derivative of the executed client. Prompts/payload/warmups/formulas retained; new health/output/provenance guards tested offline only."},
                  "health_before": before, "health_after": after, "request_audit": audit,
                  "metrics": {"decode": "SparkDash: sum(completion_tokens-1) / (latest last content timestamp-earliest first content timestamp)",
                              "end_to_end": "sum(completion_tokens) / full concurrent client request wall time",
                              "ttft": "request start to first nonempty content event; repeated prompts may reuse prefixes"},
                  "mia_baseline_url": "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/" + MIA_REVISION + "/README.md#performance",
                  "comparison_limits": ["Published TP2/TP3 results are reported baselines, not local reruns.",
                                        "Published tables do not preserve exact raw settings/output caps; 400 tokens is the reviewed SparkDash default.",
                                        "This client applies fixed-length flags to its 32-token warmups, preserving the executed client; upstream SparkDash warmups do not force filling.",
                                        "Changing tokens or model changes the workload; published ratios do not establish a matched workload.",
                                        "Mia explicitly reports 2200MHz for the TP2/P4 benchmark; TP2/P8 and TP3 per-boot clocks are unverified. This client does not change or query clocks/power.",
                                        "DFlash2 draft acceptance varies by generated text; full content/usage retained. Counting is not Python code.",
                                        "Exact prompt repeats/warmups preserve prefix caching; this is not a cold-prefill benchmark.",
                                        "Client content timestamps are approximate decode timing for drafted bursts; final usage supplies token counts and server timings are retained separately.",
                                        "Optional provided runtime/qualification documents are archived provenance, not independently verified live library/container identity.",
                                        "Use a dedicated idle endpoint without competing traffic; health/request counters do not detect all other GPU or network contention.",
                                        "Record-only health mode permits unverified context/lanes/idle state when fields are unavailable."],
                  "rows": rows, "phases": phases}
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": True, "output": str(args.output), "rows": rows, "request_audit": audit}, indent=2), flush=True)
        return 0
    except BaseException as error:
        failure = {"passed": False, "started": started, "finished": utc(), "error": str(error),
                   "completed_warmups": len(warmups), "completed_measured_requests": sum(len(p["requests"]) for p in phases),
                   "health_before": before, "health_after": after}
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
