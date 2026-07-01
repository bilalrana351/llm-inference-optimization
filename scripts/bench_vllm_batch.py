"""Task 6 (Week 3): the vLLM batching throughput sweep.

Everything measured before this is batch 1, which is latency, not throughput.
This script sends a growing number of concurrent sequences to vLLM and measures
how aggregate throughput and per-sequence latency move as the batch grows, to
locate three regions:

  - memory-bound "free lunch": throughput climbs roughly linearly, latency flat.
  - compute-bound: throughput growth slows, per-token latency starts climbing.
  - KV-cache wall: throughput flattens hard and requests queue (the scheduler
    runs a subset per step and defers the rest).

How vLLM's v1 engine shapes this measurement (see docs/batching-results.md):

  1. Concurrency is one llm.generate([N prompts]) call. vLLM's continuous
     batching schedules the list internally, so we do NOT spawn threads or use
     the async engine. Passing N prompts at once IS the batch.
  2. The KV pool is reserved at startup (gpu_memory_utilization), so device VRAM
     is flat through the whole sweep. The KV wall shows up as queuing, not as
     rising bytes. You cannot "watch memory fill up" the way the HF OOM run did.
  3. On a 1.5B model with a realistic 0.9 pool, decode goes compute-bound well
     before the KV pool runs out (hundreds of sequences fit). So we run the sweep
     twice: --tag realistic (gpu_mem_util 0.9, wide sweep) and --tag constrained
     (a deliberately small pool) so the KV wall bites at a low, cheap batch size.

Prefill and decode are split the same way baseline_vllm.py does it: time a run
with max_tokens=1 (prefill + first token = TTFT) and a run with the full
max_tokens, and take the difference as decode. This keeps the "never blend
prefill into decode" rule and avoids depending on vLLM's internal per-request
metrics, which moved between the v0 and v1 engines.

Latency is reported two ways. The robust primary is TPOT (time per output token
= decode wall / decode steps), which is a well-defined per-sequence latency while
all N sequences run in one wave and becomes a queuing signal past the wall. We
additionally attempt per-request p50/p95 end-to-end latency from
RequestOutput.metrics; that is degenerate (all near-equal) in the realistic run
and only becomes interesting in the constrained run where requests actually wait.
If the metrics are not populated on this vLLM build, those columns are NaN.

Usage (from Environment B, the vLLM venv):
    # realistic pool, wide sweep
    python scripts/bench_vllm_batch.py --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 --new-tokens 256 --gpu-mem-util 0.9 \
        --sweep 1,2,4,8,16,32,48,64,96,128,192,256 --repeats 2 --tag realistic

    # constrained pool, force the KV wall
    python scripts/bench_vllm_batch.py --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 --new-tokens 256 --gpu-mem-util 0.4 \
        --sweep 1,2,4,8,16,32,48,64 --repeats 2 --tag constrained
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from bench_common import device_used_mib, print_env


def build_prompt_ids(tokenizer, target_tokens: int) -> list[int]:
    """Make a token-id list of exactly target_tokens tokens.

    Identical seed-and-tile scheme to baseline_hf/baseline_vllm, so this sweep
    feeds the same prompt every prior run saw. We pass token ids straight to vLLM
    to skip its tokenizer and guarantee the length matches to the token.
    """
    seed = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(seed).input_ids
    reps = (target_tokens // len(ids)) + 1
    return (ids * reps)[:target_tokens]


def get_kv_cache_info(llm) -> tuple[int | None, int | None]:
    """Best-effort read of (num_gpu_blocks, block_size) from a constructed LLM.

    vLLM profiles the KV pool at startup and the block count is what sets the
    analytical wall. The attribute path has moved across engine versions, and on
    the v1 engine the profiling runs in the EngineCore child process, so the
    parent may or may not carry the resolved count. We try the known paths and
    return (None, None) if none work, in which case the wall marker is derived
    empirically from where queuing begins instead. Add a path here when a run on
    the box shows which one this vLLM build populates.
    """
    candidates = [
        lambda: (llm.llm_engine.cache_config.num_gpu_blocks,
                 llm.llm_engine.cache_config.block_size),
        lambda: (llm.llm_engine.model_config.get_num_gpu_blocks(),  # older API
                 llm.llm_engine.cache_config.block_size),
        lambda: (llm.llm_engine.vllm_config.cache_config.num_gpu_blocks,
                 llm.llm_engine.vllm_config.cache_config.block_size),
    ]
    for get in candidates:
        try:
            blocks, block_size = get()
            if blocks:
                return int(blocks), int(block_size)
        except Exception:
            continue
    return None, None


def request_latencies(outputs) -> list[float]:
    """Per-request end-to-end latency (finished - arrival) in seconds, if present.

    RequestOutput.metrics is populated only when the engine tracks it, and the
    field names shifted between v0 and v1. We read defensively and return an empty
    list if the metrics are absent, which the caller turns into NaN percentiles.
    """
    lats: list[float] = []
    for out in outputs:
        m = getattr(out, "metrics", None)
        if m is None:
            return []
        arrival = getattr(m, "arrival_time", None)
        finished = getattr(m, "finished_time", None) or getattr(m, "last_token_time", None)
        if arrival is None or finished is None:
            return []
        lats.append(float(finished) - float(arrival))
    return lats


def timed_generate(llm, prompt_token_ids: list[int], batch: int, max_tokens: int):
    """Run one blocking generate over `batch` identical prompts.

    Returns (wall_seconds, outputs). generate() blocks until every sequence in the
    batch finishes, so a plain perf_counter around it is correct (the call has
    already drained the GPU). temperature 0 is greedy; ignore_eos forces exactly
    max_tokens per sequence so the token counts are exact and reproducible.
    """
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    prompts = [{"prompt_token_ids": prompt_token_ids} for _ in range(batch)]
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
    return time.perf_counter() - start, outputs


@dataclass
class BatchSweepResult:
    """One measured (batch size, repeat) point. Written as a single CSV row."""

    pool_config: str        # "realistic" | "constrained"
    gpu_mem_util: float
    max_model_len: int
    num_gpu_blocks: int     # -1 when vLLM did not expose it
    kv_capacity_seqs: int   # analytical sequences that fit; -1 when unknown
    batch_size: int
    out_tokens_total: int
    total_s: float
    ttft_s: float
    decode_s: float
    throughput_out_tok_s: float      # end-to-end: N*new_tokens / total_s
    decode_throughput_tok_s: float   # decode-only: N*(new_tokens-1) / decode_s
    tpot_ms: float                   # per-token latency: decode_s / decode_steps
    p50_latency_s: float             # per-request end-to-end; NaN if unavailable
    p95_latency_s: float
    queued: bool                     # batch exceeded analytical KV capacity
    device_used_mib: float
    repeat_idx: int

    model: str = ""
    gpu_name: str = field(default_factory=lambda: _gpu_name())
    vllm_version: str = field(default_factory=lambda: _vllm_version())
    torch_version: str = field(default_factory=lambda: torch.__version__)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _vllm_version() -> str:
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "unknown"


def write_result(result: BatchSweepResult, csv_path: str) -> None:
    """Append a result row, writing the header if the file is new.

    A local copy of bench_common.write_result specialized to this dataclass so the
    sweep's wider schema does not have to be forced through BenchResult.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = asdict(result)
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (q in [0, 100]) by linear interpolation, or NaN."""
    if not values:
        return math.nan
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="KV context cap. Defaults to prompt+new+16 so the reserved pool is "
        "sized to the workload, not to the card.",
    )
    parser.add_argument(
        "--sweep",
        default="1,2,4,8,16,32,48,64,96,128,192,256",
        help="Comma-separated concurrent-sequence counts to measure.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--tag",
        choices=["realistic", "constrained"],
        default="realistic",
        help="realistic = 0.9 pool, wide sweep; constrained = small pool that "
        "forces the KV wall at a low batch size.",
    )
    parser.add_argument("--csv", default="results/vllm_batch_sweep.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Run this on the box, in the vLLM venv.")

    print_env()
    batch_sizes = [int(x) for x in args.sweep.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = build_prompt_ids(tokenizer, args.prompt_tokens)
    actual_prompt_tokens = len(prompt_token_ids)
    seq_len = actual_prompt_tokens + args.new_tokens
    max_model_len = args.max_model_len or (seq_len + 16)

    print(f"\nloading {args.model} in fp16 through vLLM ({args.tag} pool) ...")
    print(f"gpu_memory_utilization={args.gpu_mem_util}, max_model_len={max_model_len}")
    # One LLM for the whole sweep: the KV pool is the controlled variable, so we
    # size it once and vary only the number of concurrent sequences below.
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_model_len,
        enable_prefix_caching=False,  # each sequence must hold its own KV blocks
        enforce_eager=False,          # CUDA graphs on; part of vLLM's decode win
    )

    post_init_mib = device_used_mib()
    num_gpu_blocks, block_size = get_kv_cache_info(llm)
    if num_gpu_blocks:
        # Sequences that fit = total KV token slots / tokens each sequence holds.
        kv_capacity_seqs = (num_gpu_blocks * block_size) // seq_len
        print(f"KV pool: {num_gpu_blocks} blocks x {block_size} tok/block = "
              f"{num_gpu_blocks * block_size} token slots")
        print(f"analytical KV wall: ~{kv_capacity_seqs} concurrent sequences "
              f"of {seq_len} tokens")
    else:
        kv_capacity_seqs = None
        print("KV block count not exposed by this vLLM build; the wall will be "
              "read empirically from where throughput flattens / queuing begins.")
    print(f"device used after init: {post_init_mib:.0f} MiB (flat for the sweep, "
          f"pool is pre-reserved)")

    # --- warmup (discarded): first call pays CUDA graph capture and autotuning ---
    print("warmup (discarded) ...")
    timed_generate(llm, prompt_token_ids, batch=2, max_tokens=8)

    decode_steps = max(args.new_tokens - 1, 0)
    print(f"\nsweep {args.tag}: batches {batch_sizes}, {args.repeats} repeats each")
    print("-" * 72)

    for n in batch_sizes:
        for r in range(args.repeats):
            # prefill-batch timing (TTFT), then the full run; decode is the gap.
            ttft_s, _ = timed_generate(llm, prompt_token_ids, batch=n, max_tokens=1)
            total_s, outputs = timed_generate(
                llm, prompt_token_ids, batch=n, max_tokens=args.new_tokens
            )
            decode_s = max(total_s - ttft_s, 1e-9)

            # sanity: ignore_eos must give exactly n * new_tokens output tokens.
            out_tokens_total = sum(len(o.outputs[0].token_ids) for o in outputs)
            expected = n * args.new_tokens
            if out_tokens_total != expected:
                print(f"  WARN batch={n}: produced {out_tokens_total} tokens, "
                      f"expected {expected} (a sequence stopped early)")

            throughput_out = out_tokens_total / total_s
            decode_throughput = (n * decode_steps) / decode_s if decode_steps else 0.0
            tpot_ms = (decode_s / decode_steps * 1000) if decode_steps else 0.0

            lats = request_latencies(outputs)
            p50 = percentile(lats, 50)
            p95 = percentile(lats, 95)

            queued = bool(kv_capacity_seqs is not None and n > kv_capacity_seqs)

            result = BatchSweepResult(
                pool_config=args.tag,
                gpu_mem_util=args.gpu_mem_util,
                max_model_len=max_model_len,
                num_gpu_blocks=num_gpu_blocks if num_gpu_blocks else -1,
                kv_capacity_seqs=kv_capacity_seqs if kv_capacity_seqs is not None else -1,
                batch_size=n,
                out_tokens_total=out_tokens_total,
                total_s=total_s,
                ttft_s=ttft_s,
                decode_s=decode_s,
                throughput_out_tok_s=throughput_out,
                decode_throughput_tok_s=decode_throughput,
                tpot_ms=tpot_ms,
                p50_latency_s=p50,
                p95_latency_s=p95,
                queued=queued,
                device_used_mib=max(post_init_mib, device_used_mib()),
                repeat_idx=r,
                model=args.model,
            )
            write_result(result, args.csv)

            flag = "  [queued, past KV wall]" if queued else ""
            print(f"  batch={n:4d} r{r}  out_thru={throughput_out:8.1f} tok/s  "
                  f"decode_thru={decode_throughput:8.1f} tok/s  "
                  f"tpot={tpot_ms:6.2f} ms  ttft={ttft_s * 1000:6.1f} ms{flag}")

    print("-" * 72)
    _report_sanity(args, batch_sizes)
    print(f"rows appended -> {args.csv}")


def _report_sanity(args, batch_sizes: list[int]) -> None:
    """Cross-check the batch-1 decode rate against the vLLM baseline, and flag
    thermal drift by the spread across repeats at each batch size."""
    if 1 not in batch_sizes:
        return
    baseline_path = "results/baseline_vllm.csv"
    if not os.path.exists(baseline_path):
        return
    try:
        with open(baseline_path) as f:
            rows = list(csv.DictReader(f))
        vals = [float(r["decode_tokens_per_sec"]) for r in rows if r.get("decode_tokens_per_sec")]
        if vals:
            base = statistics.median(vals)
            print(f"sanity: baseline_vllm.csv batch-1 decode ~{base:.1f} tok/s; "
                  f"compare the batch=1 decode_thru rows above (should be within "
                  f"a few percent).")
    except Exception:
        pass


if __name__ == "__main__":
    main()
