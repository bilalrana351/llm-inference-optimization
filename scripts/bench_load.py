"""Drive a vLLM OpenAI-compatible server with controlled online load.

The client sends the same fixed-seed mixed-length workload under two arrival
processes:

  poisson  exponential gaps, normalized to the requested mean arrival rate
  bursty   groups of eight simultaneous requests, with the same mean rate

Every request uses streaming completions so TTFT and TPOT are measured at the
client. A background sampler records vLLM's Prometheus queue, running-request,
KV-usage, and preemption metrics when the installed server exposes them.

Start one vLLM server configuration, then run this script once. Reuse the same
arguments after each server restart and change only --server-config plus the
three server metadata flags. The first configuration should pass --overwrite;
later configurations append to the same CSVs.

Example:
    python scripts/bench_load.py --server-config baseline \
        --max-num-seqs 64 --gpu-mem-util 0.9 --chunked-prefill \
        --rates 0.5,1,2,3,4,6 --requests 48 --repeats 2 --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

import aiohttp
from transformers import AutoTokenizer


PROMPT_LENGTHS = (128, 512, 2048, 4096)
PROMPT_WEIGHTS = (0.30, 0.35, 0.25, 0.10)
OUTPUT_LENGTHS = (32, 64, 128, 256)
OUTPUT_WEIGHTS = (0.25, 0.35, 0.25, 0.15)

METRIC_NAMES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_usage": ("vllm:kv_cache_usage_perc",),
    "preemptions": (
        "vllm:num_preemptions_total",
        "vllm:num_preemptions",
    ),
}


def percentile(values: Iterable[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    rank = q / 100 * (len(clean) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[lower]
    return clean[lower] + (clean[upper] - clean[lower]) * (rank - lower)


def mean_or_nan(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else math.nan


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def write_rows(path: str, rows: list[dict], overwrite: bool) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if overwrite and os.path.exists(path):
        os.remove(path)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum metric series by base name, ignoring labels and comments."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def metric_value(metrics: dict[str, float], logical_name: str) -> float:
    for name in METRIC_NAMES[logical_name]:
        if name in metrics:
            return metrics[name]
    return math.nan


def build_arrival_schedule(
    pattern: str,
    target_rate: float,
    count: int,
    seed: int,
    burst_size: int,
) -> list[float]:
    if count < 2:
        return [0.0] * count
    duration = (count - 1) / target_rate
    if pattern == "poisson":
        rng = random.Random(seed)
        gaps = [rng.expovariate(1.0) for _ in range(count - 1)]
        cumulative = [0.0]
        for gap in gaps:
            cumulative.append(cumulative[-1] + gap)
        scale = duration / cumulative[-1]
        return [value * scale for value in cumulative]
    if pattern == "bursty":
        group_count = math.ceil(count / burst_size)
        if group_count == 1:
            return [0.0] * count
        group_gap = duration / (group_count - 1)
        return [(index // burst_size) * group_gap for index in range(count)]
    raise ValueError(f"unknown arrival pattern: {pattern}")


def choose_lengths(count: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    prompts = rng.choices(PROMPT_LENGTHS, weights=PROMPT_WEIGHTS, k=count)
    outputs = rng.choices(OUTPUT_LENGTHS, weights=OUTPUT_WEIGHTS, k=count)
    return list(zip(prompts, outputs))


def build_prompt_ids(tokenizer, target_tokens: int, request_id: int) -> list[int]:
    seed = "Measured systems turn assumptions into evidence. "
    token_ids = tokenizer(seed, add_special_tokens=False).input_ids
    if not token_ids:
        raise RuntimeError("tokenizer returned an empty seed")
    offset = request_id % len(token_ids)
    rotated = token_ids[offset:] + token_ids[:offset]
    repetitions = target_tokens // len(rotated) + 1
    return (rotated * repetitions)[:target_tokens]


@dataclass
class RequestResult:
    server_config: str
    arrival_pattern: str
    target_rate_rps: float
    repeat: int
    request_id: int
    prompt_tokens: int
    requested_output_tokens: int
    completed_output_tokens: int
    scheduled_offset_s: float
    submitted_offset_s: float
    arrival_lag_ms: float
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    http_status: int
    finish_reason: str
    success: bool
    slo_pass: bool
    error: str


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt_ids: list[int],
    output_tokens: int,
    scheduled_offset: float,
    run_start: float,
    config: str,
    pattern: str,
    rate: float,
    repeat: int,
    request_id: int,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
) -> RequestResult:
    await asyncio.sleep(max(0.0, run_start + scheduled_offset - time.perf_counter()))
    submitted = time.perf_counter()
    first_token: float | None = None
    finished = submitted
    status = 0
    finish_reason = ""
    error = ""
    success = False

    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    try:
        async with session.post(endpoint, json=payload) as response:
            status = response.status
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:300]}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                now = time.perf_counter()
                if first_token is None and choice.get("finish_reason") is None:
                    first_token = now
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
            finished = time.perf_counter()
            success = first_token is not None and finish_reason == "length"
    except Exception as exc:
        finished = time.perf_counter()
        error = str(exc).replace("\n", " ")[:500]

    if first_token is None:
        ttft_ms = math.nan
        tpot_ms = math.nan
    else:
        ttft_ms = (first_token - submitted) * 1000
        denominator = max(output_tokens - 1, 1)
        tpot_ms = (finished - first_token) * 1000 / denominator
    e2e_ms = (finished - submitted) * 1000
    slo_pass = bool(
        success and ttft_ms <= slo_ttft_ms and tpot_ms <= slo_tpot_ms
    )
    return RequestResult(
        server_config=config,
        arrival_pattern=pattern,
        target_rate_rps=rate,
        repeat=repeat,
        request_id=request_id,
        prompt_tokens=len(prompt_ids),
        requested_output_tokens=output_tokens,
        completed_output_tokens=output_tokens if success else 0,
        scheduled_offset_s=scheduled_offset,
        submitted_offset_s=submitted - run_start,
        arrival_lag_ms=(submitted - (run_start + scheduled_offset)) * 1000,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        http_status=status,
        finish_reason=finish_reason,
        success=success,
        slo_pass=slo_pass,
        error=error,
    )


async def fetch_metrics(
    session: aiohttp.ClientSession, metrics_url: str
) -> dict[str, float]:
    try:
        async with session.get(metrics_url) as response:
            if response.status != 200:
                return {}
            return parse_prometheus(await response.text())
    except Exception:
        return {}


async def sample_metrics(
    session: aiohttp.ClientSession,
    metrics_url: str,
    interval_s: float,
    stop: asyncio.Event,
    run_start: float,
) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    while not stop.is_set():
        metrics = await fetch_metrics(session, metrics_url)
        samples.append({
            "sample_offset_s": time.perf_counter() - run_start,
            "running": metric_value(metrics, "running"),
            "waiting": metric_value(metrics, "waiting"),
            "kv_usage": metric_value(metrics, "kv_usage"),
            "preemptions": metric_value(metrics, "preemptions"),
        })
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
    return samples


async def wait_for_server(session: aiohttp.ClientSession, base_url: str) -> None:
    for _ in range(120):
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(f"server did not become healthy: {base_url}")


async def run_one(
    session: aiohttp.ClientSession,
    args,
    tokenizer,
    pattern: str,
    rate: float,
    repeat: int,
) -> tuple[list[RequestResult], dict, list[dict]]:
    lengths = choose_lengths(args.requests, args.seed + repeat)
    schedule = build_arrival_schedule(
        pattern,
        rate,
        args.requests,
        args.seed + 10_000 + repeat,
        args.burst_size,
    )
    prompts = [
        build_prompt_ids(tokenizer, prompt_tokens, request_id + repeat * args.requests)
        for request_id, (prompt_tokens, _) in enumerate(lengths)
    ]
    before = await fetch_metrics(session, f"{args.base_url}/metrics")
    run_start = time.perf_counter()
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_metrics(
        session,
        f"{args.base_url}/metrics",
        args.metrics_interval,
        stop,
        run_start,
    ))
    tasks = [
        asyncio.create_task(send_request(
            session=session,
            endpoint=f"{args.base_url}/v1/completions",
            model=args.model,
            prompt_ids=prompts[request_id],
            output_tokens=lengths[request_id][1],
            scheduled_offset=schedule[request_id],
            run_start=run_start,
            config=args.server_config,
            pattern=pattern,
            rate=rate,
            repeat=repeat,
            request_id=request_id,
            slo_ttft_ms=args.slo_ttft_ms,
            slo_tpot_ms=args.slo_tpot_ms,
        ))
        for request_id in range(args.requests)
    ]
    request_results = await asyncio.gather(*tasks)
    stop.set()
    metric_samples = await sampler
    after = await fetch_metrics(session, f"{args.base_url}/metrics")
    run_wall_s = max(result.submitted_offset_s + result.e2e_ms / 1000
                     for result in request_results)
    successes = [result for result in request_results if result.success]
    good = [result for result in successes if result.slo_pass]
    submitted = sorted(result.submitted_offset_s for result in request_results)
    realized_rate = (
        (len(submitted) - 1) / (submitted[-1] - submitted[0])
        if len(submitted) > 1 and submitted[-1] > submitted[0]
        else math.nan
    )
    waiting = [sample["waiting"] for sample in metric_samples]
    running = [sample["running"] for sample in metric_samples]
    kv_usage = [sample["kv_usage"] for sample in metric_samples]
    before_preemptions = metric_value(before, "preemptions")
    after_preemptions = metric_value(after, "preemptions")
    preemption_delta = (
        after_preemptions - before_preemptions
        if math.isfinite(before_preemptions) and math.isfinite(after_preemptions)
        else math.nan
    )
    output_total = sum(result.completed_output_tokens for result in successes)
    good_output_total = sum(result.completed_output_tokens for result in good)
    row = {
        "server_config": args.server_config,
        "arrival_pattern": pattern,
        "target_rate_rps": rate,
        "repeat": repeat,
        "request_count": args.requests,
        "success_count": len(successes),
        "error_count": args.requests - len(successes),
        "mean_prompt_tokens": statistics.fmean(result.prompt_tokens for result in request_results),
        "mean_output_tokens": statistics.fmean(result.requested_output_tokens for result in request_results),
        "arrival_window_s": schedule[-1] - schedule[0],
        "run_wall_s": run_wall_s,
        "realized_submit_rate_rps": realized_rate,
        "achieved_request_rate_rps": len(successes) / run_wall_s,
        "output_throughput_tok_s": output_total / run_wall_s,
        "ttft_p50_ms": percentile((result.ttft_ms for result in successes), 50),
        "ttft_p95_ms": percentile((result.ttft_ms for result in successes), 95),
        "tpot_p50_ms": percentile((result.tpot_ms for result in successes), 50),
        "tpot_p95_ms": percentile((result.tpot_ms for result in successes), 95),
        "e2e_p50_ms": percentile((result.e2e_ms for result in successes), 50),
        "e2e_p95_ms": percentile((result.e2e_ms for result in successes), 95),
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_tpot_ms": args.slo_tpot_ms,
        "slo_pass_count": len(good),
        "slo_attainment_pct": 100 * len(good) / args.requests,
        "goodput_request_rps": len(good) / run_wall_s,
        "goodput_output_tok_s": good_output_total / run_wall_s,
        "client_lag_p95_ms": percentile((result.arrival_lag_ms for result in request_results), 95),
        "queue_mean": mean_or_nan(waiting),
        "queue_p95": percentile(waiting, 95),
        "queue_max": max((value for value in waiting if math.isfinite(value)), default=math.nan),
        "running_max": max((value for value in running if math.isfinite(value)), default=math.nan),
        "kv_usage_max": max((value for value in kv_usage if math.isfinite(value)), default=math.nan),
        "preemptions": preemption_delta,
        "max_num_seqs": args.max_num_seqs,
        "gpu_mem_util": args.gpu_mem_util,
        "chunked_prefill": args.chunked_prefill,
        "model": args.model,
        "gpu_name": gpu_name(),
        "vllm_version": package_version("vllm"),
        "torch_version": package_version("torch"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metric_rows = [{
        "server_config": args.server_config,
        "arrival_pattern": pattern,
        "target_rate_rps": rate,
        "repeat": repeat,
        **sample,
    } for sample in metric_samples]
    return request_results, row, metric_rows


async def warmup(session: aiohttp.ClientSession, args, tokenizer) -> None:
    prompt = build_prompt_ids(tokenizer, 128, -1)
    start = time.perf_counter()
    result = await send_request(
        session,
        f"{args.base_url}/v1/completions",
        args.model,
        prompt,
        8,
        0.0,
        start,
        args.server_config,
        "warmup",
        0.0,
        -1,
        -1,
        args.slo_ttft_ms,
        args.slo_tpot_ms,
    )
    if not result.success:
        raise RuntimeError(f"warmup failed: {result.error or result.finish_reason}")


async def async_main(args) -> None:
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=900)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await wait_for_server(session, args.base_url)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        await warmup(session, args, tokenizer)
        print(f"server healthy, warmup passed: {args.server_config}")

        request_rows: list[dict] = []
        run_rows: list[dict] = []
        metric_rows: list[dict] = []
        for pattern in args.patterns:
            for rate in args.rates:
                for repeat in range(args.repeats):
                    print(
                        f"{args.server_config}: {pattern}, {rate:g} req/s, "
                        f"repeat {repeat + 1}/{args.repeats}",
                        flush=True,
                    )
                    requests, run, metrics = await run_one(
                        session, args, tokenizer, pattern, rate, repeat
                    )
                    request_rows.extend(asdict(result) for result in requests)
                    run_rows.append(run)
                    metric_rows.extend(metrics)
                    print(
                        f"  TTFT p95={run['ttft_p95_ms']:.1f} ms, "
                        f"TPOT p95={run['tpot_p95_ms']:.1f} ms, "
                        f"goodput={run['goodput_request_rps']:.2f} req/s, "
                        f"SLO={run['slo_attainment_pct']:.1f}%, "
                        f"queue max={run['queue_max']:.0f}",
                        flush=True,
                    )
                    await asyncio.sleep(args.cooldown)

    write_rows(args.request_csv, request_rows, args.overwrite)
    write_rows(args.run_csv, run_rows, args.overwrite)
    write_rows(args.metrics_csv, metric_rows, args.overwrite)
    print(f"wrote {args.request_csv} ({len(request_rows)} rows)")
    print(f"wrote {args.run_csv} ({len(run_rows)} rows)")
    print(f"wrote {args.metrics_csv} ({len(metric_rows)} rows)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--server-config", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--gpu-mem-util", type=float, required=True)
    chunked = parser.add_mutually_exclusive_group(required=True)
    chunked.add_argument("--chunked-prefill", action="store_true", dest="chunked_prefill")
    chunked.add_argument("--no-chunked-prefill", action="store_false", dest="chunked_prefill")
    parser.add_argument("--patterns", default="poisson,bursty")
    parser.add_argument("--rates", default="0.5,1,2,3,4,6")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=100.0)
    parser.add_argument("--metrics-interval", type=float, default=0.1)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--request-csv", default="results/load_requests.csv")
    parser.add_argument("--run-csv", default="results/load_runs.csv")
    parser.add_argument("--metrics-csv", default="results/load_metrics.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.patterns = [item.strip() for item in args.patterns.split(",") if item.strip()]
    args.rates = [float(item) for item in args.rates.split(",") if item.strip()]
    if not set(args.patterns).issubset({"poisson", "bursty"}):
        parser.error("patterns must be poisson and/or bursty")
    if min(args.rates, default=0) <= 0:
        parser.error("every rate must be positive")
    if args.requests < 2 or args.repeats < 1:
        parser.error("requests must be >= 2 and repeats must be >= 1")
    return args


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
