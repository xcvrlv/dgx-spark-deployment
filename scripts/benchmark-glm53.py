#!/usr/bin/env python3
"""Measure TP4 prefill TTFT and steady-state streaming decode throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


def read_prometheus_metrics(url: str, timeout: float) -> dict[str, float]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            lines = response.read().decode(errors="replace").splitlines()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return {}

    metrics: dict[str, float] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        if "spec_decode" not in name:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in after.items():
        result[name] = value - before.get(name, 0.0) if name.endswith("_total") else value
    accepted = next(
        (value for name, value in result.items() if "accepted_tokens_total" in name),
        None,
    )
    drafted = next(
        (value for name, value in result.items() if "draft_tokens_total" in name),
        None,
    )
    if accepted is not None and drafted:
        result["calculated_draft_acceptance_rate"] = accepted / drafted
    return result


def post_stream(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    usage: dict[str, int] = {}

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    emitted = delta.get("content") or delta.get("reasoning_content")
                    if emitted:
                        now = time.perf_counter()
                        first_token_at = first_token_at or now
                        last_token_at = now
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {body}") from error

    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without emitting a token")
    if not usage:
        raise RuntimeError("stream omitted token usage; stream_options.include_usage is required")

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = max((last_token_at or first_token_at) - first_token_at, 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_token_at - started,
        "request_seconds": finished - started,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": (
            prompt_tokens / (first_token_at - started) if prompt_tokens else 0.0
        ),
        "decode_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds
            if completion_tokens > 1 and decode_seconds > 0
            else 0.0
        ),
    }


def make_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def make_prefill_prompt(word_count: int) -> str:
    words = "alpha beta gamma delta epsilon zeta eta theta "
    repeated = (words * ((word_count // 8) + 1)).split()[:word_count]
    return (
        f"Unique benchmark nonce {uuid.uuid4()}. Read the following material and "
        "answer only with OK.\n\n" + " ".join(repeated)
    )


def run_case(
    url: str,
    model: str,
    prompt_factory: Callable[[], str],
    max_tokens: int,
    warmups: int,
    runs: int,
    timeout: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in range(warmups + runs):
        sample = post_stream(
            url,
            make_payload(model, prompt_factory(), max_tokens),
            timeout,
        )
        if index >= warmups:
            samples.append(sample)
    return samples


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = (
        "prompt_tokens",
        "completion_tokens",
        "ttft_seconds",
        "request_seconds",
        "decode_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    )
    return {
        "median": {
            key: statistics.median(float(sample[key]) for sample in samples)
            for key in numeric_keys
        },
        "samples": samples,
    }


def print_summary(results: dict[str, Any], output: Path | None) -> None:
    prefill = results["cases"]["prefill"]["median"]
    generic = results["cases"]["decode_generic"]["median"]
    structured = results["cases"]["decode_structured"]["median"]
    print("GLM-5.3 TP4 benchmark")
    print(
        "  prefill:    "
        f"{prefill['prefill_tokens_per_second']:.1f} tok/s approximate, "
        f"TTFT {prefill['ttft_seconds']:.3f}s, "
        f"{prefill['prompt_tokens']:.0f} prompt tokens"
    )
    print(f"  decode:     {generic['decode_tokens_per_second']:.1f} tok/s generic")
    print(
        "  structured: "
        f"{structured['decode_tokens_per_second']:.1f} tok/s "
        "(MTP acceptance-sensitive)"
    )
    acceptance = results["spec_decode_metrics"].get(
        "calculated_draft_acceptance_rate"
    )
    if acceptance is not None:
        print(f"  MTP accept:  {acceptance:.1%} of drafted tokens")
    if output:
        print(f"  result:     {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--prefill-words", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--profile", default="unspecified")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--mtp-tokens", type=int)
    parser.add_argument("--compilation-config", default="")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("runs", "prefill_words", "decode_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    api_root = args.base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    metrics_url = f"{api_root}/metrics"
    metrics_before = read_prometheus_metrics(metrics_url, min(args.timeout, 10.0))
    generic_prompt = (
        "Write a self-contained technical explanation of virtual memory in "
        "continuous prose. Do not use headings, lists, or repeated phrases."
    )
    structured_prompt = (
        "Write the integers from 1 through 400 in order, one integer per line. "
        "Do not add any other text."
    )
    cases = {
        "prefill": summarize(
            run_case(
                endpoint,
                args.model,
                lambda: make_prefill_prompt(args.prefill_words),
                1,
                args.warmups,
                args.runs,
                args.timeout,
            )
        ),
        "decode_generic": summarize(
            run_case(
                endpoint,
                args.model,
                lambda: generic_prompt,
                args.decode_tokens,
                args.warmups,
                args.runs,
                args.timeout,
            )
        ),
        "decode_structured": summarize(
            run_case(
                endpoint,
                args.model,
                lambda: structured_prompt,
                args.decode_tokens,
                args.warmups,
                args.runs,
                args.timeout,
            )
        ),
    }
    metrics_after = read_prometheus_metrics(metrics_url, min(args.timeout, 10.0))
    results = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "settings": {
            "runs": args.runs,
            "warmups": args.warmups,
            "prefill_words": args.prefill_words,
            "decode_tokens": args.decode_tokens,
            "deployment_profile": args.profile,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "mtp_speculative_tokens": args.mtp_tokens,
            "compilation_config": args.compilation_config,
        },
        "spec_decode_metrics": metric_delta(metrics_before, metrics_after),
        "cases": cases,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print_summary(results, args.output)


if __name__ == "__main__":
    main()
