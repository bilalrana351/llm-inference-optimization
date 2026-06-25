"""Localize where the resident-weights VRAM is going.

The HF baseline reported 4737 MiB of weights for Qwen2.5-1.5B, but the fp16
floor is only ~2944 MiB (1.544e9 params x 2 bytes). That ~1.8 GB gap is too
large to be buffers or rounding, so something did not land in fp16, or there is
a second copy of a big tensor (an untied lm_head).

This script loads the model the same way baseline_hf.py does, then breaks the
live GPU allocation down by (param/buffer, dtype). A big float32 row pins an
incomplete cast; a False on the tied-embeddings check pins an untied head.

Usage:
    python test/probe_weights_vram.py --model Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Run this on the GPU box.")

    dev_idx = torch.device(args.device).index or 0

    # Same load path as baseline_hf.py, so the number matches the one we are
    # trying to explain.
    print(f"loading {args.model} in fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    torch.cuda.synchronize(dev_idx)

    # Live bytes broken down by (kind, dtype). The sum of these should track
    # memory_allocated almost exactly, since nothing else is on the card yet.
    by_kind_dtype: dict[tuple[str, torch.dtype], int] = defaultdict(int)
    for _, p in model.named_parameters():
        by_kind_dtype[("param", p.dtype)] += p.numel() * p.element_size()
    for _, b in model.named_buffers():
        by_kind_dtype[("buf", b.dtype)] += b.numel() * b.element_size()

    print("\nlive bytes by (kind, dtype):")
    for key, nbytes in sorted(by_kind_dtype.items(), key=lambda kv: -kv[1]):
        kind, dtype = key
        print(f"  {kind:6s} {str(dtype):16s} {nbytes / 2**20:9.1f} MiB")

    # The breakdown above pins the cost on buffers, not weights. Name the big
    # ones so we can see exactly which tensor (most likely a precomputed mask
    # sized to max_position_embeddings) is eating the budget.
    named_bufs = [
        (name, b.numel() * b.element_size(), tuple(b.shape), b.dtype)
        for name, b in model.named_buffers()
    ]
    print("\nlargest buffers by size:")
    for name, nbytes, shape, dtype in sorted(named_bufs, key=lambda x: -x[1])[:10]:
        print(f"  {nbytes / 2**20:9.1f} MiB  {str(dtype):14s} {shape}  {name}")

    total_params = sum(p.numel() for p in model.parameters())
    allocated_mib = torch.cuda.memory_allocated(dev_idx) / 2**20

    # The two structural questions: is the head tied, and did the cast take.
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    head_is_embed = head is not None and head.weight is embed

    print("\nsummary:")
    print(f"  total params           {total_params / 1e9:.3f} B")
    print(f"  config.tie_word_embeddings  {model.config.tie_word_embeddings}")
    print(f"  lm_head shares embed weight {head_is_embed}")
    print(f"  embed dtype            {embed.dtype}")
    print(f"  memory_allocated       {allocated_mib:.0f} MiB")
    print(f"  fp16 floor (params x2) {total_params * 2 / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
