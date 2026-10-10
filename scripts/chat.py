#!/usr/bin/env python3
"""Stream a terminal chat with the Spark cluster; Python standard library only."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from common import load_config


def events(response):
    """Read SSE data events, including multiline data fields."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def generate(args, messages):
    payload = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
               "temperature": args.temperature, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": args.thinking}}
    if not args.thinking:
        payload["reasoning_effort"] = "none"
    base = args.url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    request = urllib.request.Request(base + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    parts = []
    usage = None
    finished = False
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        if not response.headers.get("Content-Type", "").startswith("text/event-stream"):
            raise RuntimeError("server did not return a streaming response")
        for data in events(response):
            if data == "[DONE]":
                finished = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    print(text, end="", flush=True)
                    parts.append(text)
                thought = delta.get("reasoning_content") or ""
                if thought and args.thinking:
                    print(thought, end="", file=sys.stderr, flush=True)
    if not finished:
        raise RuntimeError("connection closed before the reply completed")
    print(flush=True)
    elapsed = time.monotonic() - started
    if usage:
        tokens = usage.get("completion_tokens", 0)
        print(f"[{tokens} output tokens; {elapsed:.2f}s; {tokens / elapsed:.1f} tokens/s including prefill]",
              file=sys.stderr)
    return "".join(parts)


def show_error(error):
    if isinstance(error, urllib.error.HTTPError):
        detail = error.read(8192).decode("utf-8", errors="replace")
        print(f"\nHTTP {error.code}: {detail}", file=sys.stderr)
    else:
        print(f"\nRequest failed: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="reviewed cluster JSON to select its management API")
    parser.add_argument("--url", help="explicit API endpoint; overrides --config")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--prompt", help="send one prompt and exit; otherwise open an interactive chat")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--thinking", action="store_true", help="enable reasoning; reasoning text goes to stderr")
    args = parser.parse_args()
    if not args.url:
        if not args.config:
            parser.error("explicitly supply --config or --url")
        cfg = load_config(args.config)
        args.url = f"http://{cfg['nodes'][0]['management_address']}:{cfg['api_port']}/v1"
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    errors = (urllib.error.URLError, OSError, ValueError, RuntimeError)
    if args.prompt is not None:
        try:
            generate(args, [{"role": "user", "content": args.prompt}])
        except errors as error:
            show_error(error)
            return 1
        return 0
    try:
        import readline  # Optional arrow-key editing; no history file is written.
    except ImportError:
        pass
    print(f"Connected target: {args.url}\n/reset starts a new chat; /quit exits. Ctrl-D also exits.",
          file=sys.stderr)
    messages = []
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return 0
        if prompt in ("/reset", "/clear"):
            messages.clear()
            print("Chat reset.", file=sys.stderr)
            continue
        messages.append({"role": "user", "content": prompt})
        print("glm> ", end="", flush=True)
        try:
            answer = generate(args, messages)
        except errors as error:
            messages.pop()
            show_error(error)
            continue
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n", file=sys.stderr)
        raise SystemExit(130)
