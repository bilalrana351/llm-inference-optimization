"""Week 6 Part A: capture a PyTorch profiler trace of the HF batch-1 decode step.

The blog claims from reasoning that batch-1 decode on the HF path is CPU-launch-
bound: hundreds of tiny kernels per token, each preceded by Python dispatch, and
a GPU that finishes its few microseconds of work and then sits idle waiting for
the next command packet. This script produces the trace that either supports that
claim or kills it.

Two passes over the same decode window, on purpose:

  Pass 1 (clean): no profiler anywhere. Times exactly the steps that pass 2 will
    profile, starting from the same prefill, so the sequence length matches. This
    is the honest wall time per step, call it T.

  Pass 2 (profiled): the same steps under torch.profiler. Gives the GPU kernel
    intervals, call the union of them B.

The published idle-gap number is T - B, one term from each pass. That split
matters. CUPTI does not change how long a kernel runs, so B survives the profiler
intact. Wall time does not survive it, so wall time comes from the clean pass and
the profiled pass's own wall time is only ever printed as a diagnostic.

Deliberately off: with_stack, record_shapes, with_modules. Stack unwinding costs
CPU time per op, and per-op CPU time is the exact quantity under test, so turning
it on would inflate the gaps in the direction of the hypothesis. It would also
tax HF (hundreds of dispatches per step) far harder than vLLM under CUDA graphs
(one), which biases the comparison too. If you later want to attribute a gap to a
specific Python frame, do a separate run with --with-stack and use only the
attribution, never the durations, and say so in the doc.

Usage:
    python scripts/profile_decode.py \
        --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 \
        --trace results/trace_hf_decode.json.gz

Then:
    python scripts/analyze_trace.py --trace hf=results/trace_hf_decode.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_hf import build_prompt_ids
from bench_common import print_env

STEP_MARKER = "decode_step"


@torch.inference_mode()
def prefill(model, input_ids):
    """One parallel forward over the prompt. Returns (past_key_values, first_token).

    Kept outside every timed and profiled region: prefill is a different regime
    (long kernels, saturated launch queue) and mixing it into the decode trace
    would swamp the thing we are looking at.
    """
    out = model(input_ids=input_ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].argmax(dim=-1, keepdim=True)


@torch.inference_mode()
def one_step(model, next_token, past):
    """A single greedy decode step. Identical body to baseline_hf.generate_manual."""
    out = model(input_ids=next_token, past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].argmax(dim=-1, keepdim=True)


@torch.inference_mode()
def clean_window(model, input_ids, skip, active, device):
    """Pass 1. Advance `skip` steps untimed, then time `active` steps with no profiler.

    The skip exists so the timed steps sit at the same sequence length the
    profiled ones will, and so the allocator and any lazily loaded module have
    settled. Returns seconds for the whole active block.
    """
    past, next_token = prefill(model, input_ids)
    for _ in range(skip):
        past, next_token = one_step(model, next_token, past)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(active):
        past, next_token = one_step(model, next_token, past)
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0


@torch.inference_mode()
def profiled_window(model, input_ids, wait, warmup, active, device, with_stack):
    """Pass 2. The same window under the profiler. Returns (prof, wall_seconds).

    The schedule's wait and warmup phases still execute the model, they just are
    not recorded, which is what makes the recorded steps land at the same
    sequence length as pass 1's timed steps (skip == wait + warmup).

    record_function marks each step so analyze_trace.py can segment the trace
    without guessing. Annotations only appear during active steps, so their count
    is also a check that the schedule did what we asked.
    """
    past, next_token = prefill(model, input_ids)

    sched = schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
        with_modules=False,
    ) as prof:
        for _ in range(wait + warmup + active):
            with record_function(STEP_MARKER):
                past, next_token = one_step(model, next_token, past)
            prof.step()
    torch.cuda.synchronize(device)
    return prof, time.perf_counter() - t0


def export_trace(prof, path: str) -> None:
    """Write the chrome trace, gzipping if the path asks for it.

    Traces are committed to results/ and a few active steps of batch-1 decode is
    several MB of JSON. Perfetto and chrome://tracing both open .gz directly, so
    there is no reason to store it raw.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".gz"):
        raw = path[:-3]
        prof.export_chrome_trace(raw)
        with open(raw, "rb") as src, gzip.open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(raw)
    else:
        prof.export_chrome_trace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--trace", default="results/trace_hf_decode.json.gz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--wait",
        type=int,
        default=10,
        help="Decode steps executed but not recorded, before warmup. Raising this "
        "profiles further into the sequence, at a longer KV cache.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--active",
        type=int,
        default=5,
        help="Decode steps actually recorded. Five is plenty: the step is a loop "
        "body, and every extra one multiplies trace size for no new information.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Diagnostic only. Adds per-op Python stack unwinding, which inflates "
        "exactly the CPU dispatch cost being measured. Use the attribution, "
        "discard every duration, and label the run in docs/profiling.md.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This trace only means anything on the 3060.")

    print_env()
    device = args.device
    dev_idx = torch.device(device).index or 0
    skip = args.wait + args.warmup

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"\nloading {args.model} in fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    input_ids = build_prompt_ids(tokenizer, args.prompt_tokens, device)
    prompt_tokens = input_ids.shape[1]
    seq_len_at_active = prompt_tokens + skip

    # Discarded warmup: CUDA context, autotuning, lazy module loading. Without
    # this the first profiled run pays for one-time costs that are not decode.
    print("warmup (discarded) ...")
    clean_window(model, input_ids, skip=2, active=2, device=dev_idx)

    print(f"\npass 1 (clean, no profiler): {args.active} steps at seq_len ~{seq_len_at_active}")
    clean_s = clean_window(model, input_ids, skip=skip, active=args.active, device=dev_idx)
    clean_step_ms = (clean_s / args.active) * 1000.0

    print(f"pass 2 (profiled): wait={args.wait} warmup={args.warmup} active={args.active}")
    if args.with_stack:
        print("  WARNING: --with-stack is on. Durations from this run are not publishable.")
    prof, prof_s = profiled_window(
        model, input_ids, args.wait, args.warmup, args.active, dev_idx, args.with_stack
    )
    prof_step_ms = (prof_s / (skip + args.active)) * 1000.0

    export_trace(prof, args.trace)

    print("\n" + "-" * 68)
    print(f"model              {args.model} fp16, batch 1")
    print(f"prompt tokens      {prompt_tokens}")
    print(f"profiled seq_len   ~{seq_len_at_active}")
    print(f"active steps       {args.active}")
    print(f"clean step time    {clean_step_ms:8.3f} ms   <- use this as T")
    print(f"profiled step time {prof_step_ms:8.3f} ms   (diagnostic only)")
    ratio = prof_step_ms / clean_step_ms if clean_step_ms > 0 else 0.0
    print(f"profiler inflation {ratio:8.2f}x")
    if ratio > 2.0:
        print("  NOTE: heavy inflation. GPU busy time is still trustworthy, but do not")
        print("        use the profiled wall time anywhere in the published gap number.")
    print("-" * 68)
    print(f"trace written -> {args.trace}")
    print("\nnext:")
    print(f"  python scripts/analyze_trace.py \\")
    print(f"      --trace hf={args.trace} \\")
    print(f"      --clean-step-ms hf={clean_step_ms:.4f}")


if __name__ == "__main__":
    main()
