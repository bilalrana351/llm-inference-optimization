"""Task 2: the vLLM baseline (the optimized engine).

Same model, same prompt, same new-token count, same batch size 1, same greedy
decoding as baseline_hf.py. The only thing that changes is the engine, which is
the whole point of the comparison: the gap between this and the HF control is
what the rest of the roadmap explains (PagedAttention, fused kernels, CUDA
graphs, Python out of the decode loop).

Two things make a fair prefill-vs-decode split harder here than on the HF path:

  1. vLLM's generate() is one blocking call that does prefill and decode
     internally, so we cannot wrap a timer around each phase the way the manual
     HF loop does. Instead we time two runs on the identical prompt: one with
     max_tokens=1 (prefill plus the first sampled token, which is the TTFT), and
     one with max_tokens=new_tokens (the whole thing). decode_time is the
     difference. This is version-independent: it does not depend on vLLM's
     internal RequestMetrics, which moved around between the v0 and v1 engines.

  2. Prefix caching would let the second run reuse the first run's prefill and
     wreck the subtraction, so we disable it (enable_prefix_caching=False).

The first token is produced by the prefill step in both engines, so decode
counts new_tokens-1 tokens in both, keeping the rate definition identical.

VRAM is not directly comparable to HF and we do not pretend it is. vLLM
pre-allocates a large KV-cache pool at construction (governed by
gpu_memory_utilization), so its memory figure is "weights plus the reserved
pool", not "weights plus what generation organically grew". The row's note
records this so the comparison table is read honestly.

Usage (from Environment B, the vLLM venv):
    python scripts/bench_vllm.py \
        --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 \
        --new-tokens 256 \
        --gpu-mem-util 0.9
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from bench_common import (
    BenchResult,
    device_used_mib,
    print_env,
    write_result,
)


def build_prompt_ids(tokenizer, target_tokens: int) -> list[int]:
    """Make a token-id list of exactly target_tokens tokens.

    Identical seed-and-tile scheme to baseline_hf.build_prompt_ids, so vLLM sees
    the exact same prompt tokens the HF control saw. We pass token ids (not a
    string) straight to vLLM to skip its tokenizer and guarantee the lengths
    match to the token.
    """
    seed = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(seed).input_ids
    reps = (target_tokens // len(ids)) + 1
    return (ids * reps)[:target_tokens]


def timed_generate(llm, prompt_token_ids: list[int], max_tokens: int) -> float:
    """Run one blocking generate over the given prompt, return wall seconds.

    generate() blocks until every sequence finishes, so a plain perf_counter
    around it is correct: no torch.cuda.synchronize needed, the call has already
    drained the GPU before it returns. greedy (temperature 0), ignore_eos so the
    model emits exactly max_tokens and never stops early, which is what makes the
    token counts reproducible and comparable to the fixed-length HF loop.
    """
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    start = time.perf_counter()
    llm.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling_params=sampling,
        use_tqdm=False,
    )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--csv", default="results/baseline_vllm.csv")
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.9,
        help="Fraction of VRAM vLLM may reserve for weights plus the KV pool.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="KV context cap. Defaults to prompt+new+small buffer to keep the "
        "reserved pool honest rather than letting vLLM size it to the card.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Phase 1 measures GPU inference; run this on the box.")

    print_env()

    # Same prompt tokens as the HF baseline. We use the HF tokenizer to build the
    # ids so the two engines are fed byte-identical input.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = build_prompt_ids(tokenizer, args.prompt_tokens)
    actual_prompt_tokens = len(prompt_token_ids)

    max_model_len = args.max_model_len or (actual_prompt_tokens + args.new_tokens + 16)

    print(f"\nloading {args.model} in fp16 through vLLM ...")
    print(f"gpu_memory_utilization={args.gpu_mem_util}, max_model_len={max_model_len}")
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_model_len,
        enable_prefix_caching=False,  # else run 2 reuses run 1's prefill
        enforce_eager=False,          # keep CUDA graphs on; they are part of the win
    )

    # After construction this is weights plus the pre-reserved KV pool plus the
    # captured CUDA-graph buffers, not just weights. Read via NVML, not
    # torch.cuda: the v1 engine allocates all of it in the EngineCore child
    # process, which the parent's torch allocator cannot see (it would report 0).
    post_init_mib = device_used_mib()
    print(f"device used after init (weights + KV pool + graphs): {post_init_mib:.0f} MiB")

    # --- warmup (discarded): first call pays CUDA graph capture and autotuning ---
    print("warmup (discarded) ...")
    timed_generate(llm, prompt_token_ids, max_tokens=8)

    # --- measured: prefill-only run, then the full run; decode is the gap ---
    # No torch peak-reset here: the v1 engine runs in a child process, so the
    # parent's torch peak counters never move. We read device-used via NVML
    # instead. vLLM reserves the whole pool at init, so used memory stays flat
    # through generation; we still take the max of init and post-run to be safe.
    print(f"measured: prompt={actual_prompt_tokens} tok, new={args.new_tokens} tok")

    prefill_s = timed_generate(llm, prompt_token_ids, max_tokens=1)
    total_s = timed_generate(llm, prompt_token_ids, max_tokens=args.new_tokens)
    peak_used_mib = max(post_init_mib, device_used_mib())

    decode_s = max(total_s - prefill_s, 1e-9)
    decode_count = max(args.new_tokens - 1, 0)
    decode_tps = decode_count / decode_s if decode_s > 0 else 0.0
    prefill_tps = actual_prompt_tokens / prefill_s if prefill_s > 0 else 0.0

    result = BenchResult(
        engine="vllm",
        model=args.model,
        dtype="float16",
        batch_size=1,
        prompt_tokens=actual_prompt_tokens,
        new_tokens=decode_count,
        prefill_seconds=prefill_s,
        decode_seconds=decode_s,
        decode_tokens_per_sec=decode_tps,
        prefill_tokens_per_sec=prefill_tps,
        weights_vram_mib=post_init_mib,
        # NVML reports one device-level "used" figure, not torch's
        # allocated/reserved split, so both peak fields carry that single number.
        peak_allocated_mib=peak_used_mib,
        peak_reserved_mib=peak_used_mib,
        note=(
            f"NVML device-used (weights+reserved KV pool+CUDA graphs) @ "
            f"gpu_mem_util={args.gpu_mem_util}, max_model_len={max_model_len}; "
            f"not comparable to HF organic growth. Per-component split "
            f"(weights / KV pool / graphs) is in the vLLM init log."
        ),
    )
    write_result(result, args.csv)

    # --- report ---
    print("\n" + "-" * 60)
    print(f"prefill        {prefill_s * 1000:8.1f} ms   ({prefill_tps:7.1f} tok/s over prompt)")
    print(f"decode         {decode_tps:8.1f} tok/s ({decode_count} tokens in {decode_s:.3f} s)")
    print(f"device used    {post_init_mib:8.0f} MiB  (weights + KV pool + graphs, via NVML)")
    print(f"peak device    {peak_used_mib:8.0f} MiB")
    print("-" * 60)
    print("note: vLLM VRAM is the reserved pool, not HF-style organic growth.")
    print("      per-component breakdown is in the vLLM init log lines.")
    print(f"row appended -> {args.csv}")


if __name__ == "__main__":
    main()
