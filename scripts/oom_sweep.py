"""Task 3: the KV-cache OOM experiment (the headline).

Push the context length up until the card dies, and watch where it dies. This is
the plot the whole repo is named after: measured peak VRAM vs context length,
with the Phase 0 analytical KV-cache prediction overlaid, and a marker at the
context length where CUDA throws out-of-memory.

Why this runs on the HF path, not vLLM. HF grows the KV cache organically: every
decoded token appends ~28 KB and torch reports exactly what is live, so measured
VRAM tracks the analytical line and the crash is a real "the cache filled the
card" event. vLLM would ruin the experiment, it reserves the whole KV pool at
startup (PagedAttention), so its memory is flat from token 1 and it never OOMs
this way, it just refuses new sequences. That contrast is the point we write up,
not the thing we measure here.

Why we grow by DECODE, not by a long prompt. Both build the identical KV cache
(28 KB/token), but the peak at a given context length depends on how you got
there. A long prompt is prefilled in one parallel pass, which materializes
activations for every position at once: a big transient bump on top of the KV
cache that makes the card OOM early and the measured line ride above the
prediction. Decode touches one position per step, so its transient is a tiny
constant. Peak = weights + KV + (almost nothing), which is the cleanest possible
match to the analytical line and the most predictable place to read the cliff.
The cost is time: reaching tens of thousands of tokens is tens of thousands of
sequential steps. We accept that for a clean curve.

The efficient shape for a decode sweep is ONE continuous generation, not N
independent runs. A single decode loop grows the cache monotonically, so the live
memory at each checkpoint already is the peak at that context length. We pay for
the longest run once instead of regenerating from zero at every step, snapshot at
each checkpoint, and catch the OOM in the same loop.

Usage (Environment A, the HF venv with torch 2.12):
    python scripts/oom_sweep.py \
        --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 32 \
        --checkpoint-every 1000 \
        --max-tokens 200000
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
    kv_cache_bytes,
    print_env,
    reset_peak_vram,
    write_result,
)


def build_prompt_ids(tokenizer, target_tokens: int, device: str) -> torch.Tensor:
    """A short, length-exact prompt. Same seed-and-tile scheme as the baseline.

    We keep this deliberately small (the default is 32 tokens): the prompt only
    has to seed the KV cache, the decode loop does all the growing. A small prompt
    keeps the one prefill pass cheap so the transient prefill bump never competes
    with the KV cache for the cliff.
    """
    seed = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(seed, return_tensors="pt").input_ids[0]
    reps = (target_tokens // ids.numel()) + 1
    ids = ids.repeat(reps)[:target_tokens]
    return ids.unsqueeze(0).to(device)


def kv_config(model) -> tuple[int, int, int]:
    """(num_layers, num_kv_heads, head_dim) for the analytical KV formula.

    num_key_value_heads (not num_attention_heads) is the GQA count that actually
    sizes the cache: Qwen2.5-1.5B has 12 query heads sharing 2 KV heads, so using
    the query count would inflate the prediction 6x. head_dim falls out of
    hidden_size / num_attention_heads when the config does not state it directly.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return num_layers, num_kv_heads, head_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Record a row (VRAM + analytical KV + decode rate) every this many "
        "generated tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200000,
        help="Safety cap so the loop terminates even if the card somehow never "
        "OOMs. We expect to hit the wall well before this.",
    )
    parser.add_argument("--csv", default="results/oom_sweep.csv")
    parser.add_argument("--plot", default="results/oom_curve.png")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Task 3 OOMs a real GPU; run this on the box.")

    print_env()
    device = args.device
    dev_idx = torch.device(device).index or 0

    # --- load weights in fp16, the flat memory offset every curve sits on ---
    print(f"\nloading {args.model} in fp16 ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    torch.cuda.synchronize(dev_idx)
    weights_vram = bytes_to_mib(torch.cuda.memory_allocated(dev_idx))
    num_layers, num_kv_heads, head_dim = kv_config(model)
    print(f"weights resident: {weights_vram:.0f} MiB")
    print(
        f"kv config: layers={num_layers}, kv_heads={num_kv_heads}, "
        f"head_dim={head_dim} -> "
        f"{kv_cache_bytes(num_layers, num_kv_heads, head_dim, 1) / 1024:.1f} KB/token"
    )

    input_ids = build_prompt_ids(tokenizer, args.prompt_tokens, device)
    prompt_tokens = input_ids.shape[1]

    # --- warmup (discarded): pays CUDA init and kernel autotuning ---
    print("warmup (discarded) ...")
    with torch.inference_mode():
        warm = model(input_ids=input_ids, use_cache=True)
        _ = model(
            input_ids=warm.logits[:, -1, :].argmax(dim=-1, keepdim=True),
            past_key_values=warm.past_key_values,
            use_cache=True,
        )
    del warm
    torch.cuda.empty_cache()

    # rows kept in memory so we can plot at the end without re-reading the CSV
    rows: list[dict] = []

    def record(generated: int, seg_tps: float, oom: bool, note: str) -> None:
        seq_len = prompt_tokens + generated
        kv_mib = bytes_to_mib(
            kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len)
        )
        vram = VramSnapshot.capture(dev_idx)
        result = BenchResult(
            engine="hf",
            model=args.model,
            dtype="float16",
            batch_size=1,
            prompt_tokens=prompt_tokens,
            new_tokens=generated,
            prefill_seconds=0.0,
            decode_seconds=0.0,
            decode_tokens_per_sec=seg_tps,
            prefill_tokens_per_sec=0.0,
            weights_vram_mib=weights_vram,
            peak_allocated_mib=vram.peak_allocated_mib,
            peak_reserved_mib=vram.peak_reserved_mib,
            oom=oom,
            note=note,
        )
        write_result(result, args.csv)
        rows.append(
            {
                "seq_len": seq_len,
                "kv_mib": kv_mib,
                "predicted_mib": weights_vram + kv_mib,
                "measured_alloc_mib": vram.peak_allocated_mib,
                "measured_reserved_mib": vram.peak_reserved_mib,
                "decode_tps": seg_tps,
                "oom": oom,
            }
        )
        print(
            f"  ctx {seq_len:>7} tok | "
            f"predicted {weights_vram + kv_mib:8.0f} | "
            f"measured alloc {vram.peak_allocated_mib:8.0f} / "
            f"reserved {vram.peak_reserved_mib:8.0f} MiB | "
            f"{seg_tps:6.1f} tok/s"
            + ("  <-- OOM" if oom else "")
        )

    # --- single continuous generation, snapshot at every checkpoint ---
    print(
        f"\nsweep: prompt={prompt_tokens} tok, checkpoint every "
        f"{args.checkpoint_every}, cap {args.max_tokens}\n"
    )
    reset_peak_vram(dev_idx)

    oom_seq = None
    try:
        with torch.inference_mode():
            # prefill produces the first token and seeds the cache
            out = model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = 1

            seg_timer = CudaTimer(dev_idx)
            seg_timer.__enter__()
            seg_start_count = generated

            while generated < args.max_tokens:
                out = model(
                    input_ids=next_token, past_key_values=past, use_cache=True
                )
                past = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated += 1

                if generated % args.checkpoint_every == 0:
                    seg_timer.__exit__()
                    seg_tokens = generated - seg_start_count
                    seg_tps = seg_tokens / seg_timer.seconds if seg_timer.seconds else 0.0
                    record(generated, seg_tps, oom=False, note="checkpoint")
                    seg_timer = CudaTimer(dev_idx)
                    seg_timer.__enter__()
                    seg_start_count = generated

    except torch.cuda.OutOfMemoryError:
        # The cache finally filled the card. Record the context length we reached.
        oom_seq = prompt_tokens + generated
        torch.cuda.empty_cache()
        record(
            generated,
            seg_tps=0.0,
            oom=True,
            note=f"CUDA OOM while decoding token {generated}",
        )
        print(f"\nOOM at ~{oom_seq} tokens of context. This is the cliff.")

    if oom_seq is None:
        print(
            f"\nReached the {args.max_tokens}-token cap without OOM. Raise "
            "--max-tokens to find the real cliff."
        )

    _maybe_plot(rows, oom_seq, weights_vram, args)
    print(f"\nrows appended -> {args.csv}")


def _maybe_plot(rows, oom_seq, weights_vram, args) -> None:
    """Plot measured VRAM vs context with the analytical KV line overlaid.

    Guarded by the matplotlib import: the CSV is the real artifact and is always
    written above, so a box without matplotlib still produces the data and the
    plot can be regenerated later from the CSV.
    """
    finite = [r for r in rows if not r["oom"]]
    if not finite:
        print("no finite checkpoints recorded; skipping plot")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV written, plot skipped")
        return

    seq = [r["seq_len"] for r in finite]
    predicted = [r["predicted_mib"] for r in finite]
    measured = [r["measured_reserved_mib"] for r in finite]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seq, predicted, "--", label="predicted: weights + analytical KV", color="tab:blue")
    ax.plot(seq, measured, "-o", label="measured peak reserved", color="tab:red", markersize=3)
    ax.axhline(12288, color="gray", ls=":", lw=1, label="12 GB card")
    if oom_seq is not None:
        ax.axvline(oom_seq, color="black", ls="-.", lw=1, label=f"OOM ~{oom_seq} tok")

    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("GPU memory (MiB)")
    ax.set_title(
        "KV cache hits the wall: Qwen2.5-1.5B fp16, RTX 3060 12GB, HF transformers\n"
        "grown by decode (1 token/step), so peak = weights + KV + tiny constant"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"plot saved -> {args.plot}")


if __name__ == "__main__":
    main()
