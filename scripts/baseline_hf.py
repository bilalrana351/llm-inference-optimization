"""Task 1: the HuggingFace transformers baseline (the control).

The slow, readable path that every other number is measured against. It loads a
small model in fp16, runs a warmup, then a measured run that records prefill
latency, decode tokens/sec, and peak VRAM, and appends one row to results/.

We drive generation by hand (prefill forward, then a greedy decode loop over the
KV cache) instead of model.generate(), for two reasons:
  1. It splits prefill from decode exactly, with a synchronize and timer around
     each, which .generate() will not give us.
  2. It is the same loop the OOM sweep reuses, so the control and the headline
     experiment measure the identical code path.

Usage:
    python scripts/baseline_hf.py \
        --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 \
        --new-tokens 256
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_common import (
    BenchResult,
    CudaTimer,
    VramSnapshot,
    bytes_to_mib,
    print_env,
    reset_peak_vram,
    write_result,
)


def build_prompt_ids(tokenizer, target_tokens: int, device: str) -> torch.Tensor:
    """Make an input of exactly target_tokens tokens.

    We do not care about the prompt's meaning here, only its length, since the
    measurement is about shapes and memory, not output quality. We tile a short
    seed and trim to the exact length so prefill cost is reproducible.
    """
    seed = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(seed, return_tensors="pt").input_ids[0]
    reps = (target_tokens // ids.numel()) + 1
    ids = ids.repeat(reps)[:target_tokens]
    return ids.unsqueeze(0).to(device)


@torch.inference_mode()
def generate_manual(model, input_ids, new_tokens, device):
    """Prefill once, then greedy-decode new_tokens, timing the two phases apart.

    Returns (prefill_seconds, decode_seconds, generated_count). The first token
    is produced by the prefill pass itself, so the decode loop runs new_tokens-1
    times and the decode rate counts new_tokens-1 generated tokens.
    """
    # --- prefill: one parallel forward over the whole prompt ---
    with CudaTimer(device) as prefill_t:
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # --- decode: sequential, one token at a time, feeding the KV cache ---
    decode_steps = max(new_tokens - 1, 0)
    with CudaTimer(device) as decode_t:
        for _ in range(decode_steps):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return prefill_t.seconds, decode_t.seconds, decode_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--csv", default="results/baseline_hf.csv")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Phase 1 measures GPU inference; run this on the T4.")

    print_env()
    device = args.device
    dev_idx = torch.device(device).index or 0

    # --- load weights in fp16 ---
    print(f"\nloading {args.model} in fp16 ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    # VRAM occupied by weights alone, before any KV cache exists.
    torch.cuda.synchronize(dev_idx)
    weights_vram = bytes_to_mib(torch.cuda.memory_allocated(dev_idx))
    print(f"weights resident: {weights_vram:.0f} MiB")

    input_ids = build_prompt_ids(tokenizer, args.prompt_tokens, device)
    actual_prompt_tokens = input_ids.shape[1]

    # --- warmup (discarded): pays for CUDA init and kernel autotuning ---
    print("warmup (discarded) ...")
    generate_manual(model, input_ids, new_tokens=8, device=dev_idx)

    # --- measured run ---
    print(f"measured run: prompt={actual_prompt_tokens} tok, new={args.new_tokens} tok")
    reset_peak_vram(dev_idx)
    prefill_s, decode_s, decode_count = generate_manual(
        model, input_ids, args.new_tokens, device=dev_idx
    )
    vram = VramSnapshot.capture(dev_idx)

    decode_tps = decode_count / decode_s if decode_s > 0 else 0.0
    prefill_tps = actual_prompt_tokens / prefill_s if prefill_s > 0 else 0.0

    result = BenchResult(
        engine="hf",
        model=args.model,
        dtype="float16",
        batch_size=1,
        prompt_tokens=actual_prompt_tokens,
        new_tokens=decode_count,
        prefill_seconds=prefill_s,
        decode_seconds=decode_s,
        decode_tokens_per_sec=decode_tps,
        prefill_tokens_per_sec=prefill_tps,
        weights_vram_mib=weights_vram,
        peak_allocated_mib=vram.peak_allocated_mib,
        peak_reserved_mib=vram.peak_reserved_mib,
    )
    write_result(result, args.csv)

    # --- report ---
    print("\n" + "-" * 60)
    print(f"prefill        {prefill_s * 1000:8.1f} ms   ({prefill_tps:7.1f} tok/s over prompt)")
    print(f"decode         {decode_tps:8.1f} tok/s ({decode_count} tokens in {decode_s:.3f} s)")
    print(f"weights VRAM   {weights_vram:8.0f} MiB")
    print(f"peak allocated {vram.peak_allocated_mib:8.0f} MiB")
    print(f"peak reserved  {vram.peak_reserved_mib:8.0f} MiB")
    print("-" * 60)
    print(f"row appended -> {args.csv}")


if __name__ == "__main__":
    main()
