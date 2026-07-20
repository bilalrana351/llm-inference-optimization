"""Week 6 Part A: capture a profiler trace of the vLLM decode step.

The counterpart to profile_decode.py. Same model, same prompt length, same fp16,
same batch size 1, so the only thing that changes is the engine.

Why this cannot just wrap torch.profiler around generate(): the v1 engine runs
the model in a separate EngineCore child process, so a profiler context opened in
the parent records an empty trace. bench_common.device_used_mib already documents
the same split for VRAM. vLLM's own hook handles it: set VLLM_TORCH_PROFILER_DIR
before the engine starts, then call llm.start_profile() and llm.stop_profile(),
and the workers write their traces to that directory. This script sets the
variable itself, before importing vllm, because setting it afterwards is silently
too late and produces no trace at all.

Run this twice:

    --enforce-eager off (default)  CUDA graphs on, the shipping configuration
    --enforce-eager                same engine, same kernels, graphs off

HF against vLLM confounds three things at once: CUDA graphs, better attention
kernels, and a leaner Python path. vLLM-eager against vLLM-graphs isolates the
graph term by itself, which is the specific claim blog/draft.md makes. The second
run costs twenty minutes and is what turns the comparison into an experiment.

Like the HF script, two passes: a clean unprofiled pass for the wall time per
step (T) and a profiled pass for the kernel intervals (B). The published gap is
T - B, each term taken from the run where it is not distorted.

Usage (Environment B, the vLLM venv):
    python scripts/profile_vllm.py --label vllm-graph
    python scripts/profile_vllm.py --label vllm-eager --enforce-eager
"""

from __future__ import annotations

import argparse
import os
import shutil
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--clean-tokens",
        type=int,
        default=256,
        help="Tokens for the unprofiled timing pass. Matches bench_vllm.py so the "
        "step time is directly checkable against results/baseline_vllm.csv.",
    )
    parser.add_argument(
        "--profile-tokens",
        type=int,
        default=12,
        help="Tokens for the profiled pass. Keep it small: the step is a loop "
        "body, and vLLM traces grow fast.",
    )
    parser.add_argument("--label", default="vllm-graph")
    parser.add_argument("--trace-dir", default="results/vllm_trace")
    parser.add_argument(
        "--copy-to",
        default="",
        help="Copy the produced trace here, e.g. results/trace_vllm_graph.json.gz. "
        "Defaults to results/trace_<label>.json.gz.",
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs. This is the control that isolates the graph "
        "term from vLLM's other advantages.",
    )
    return parser.parse_args()


def newest_traces(trace_dir: str, since: float, timeout: float = 180.0) -> list[str]:
    """Wait for the worker processes to flush their traces, then return them.

    stop_profile() returns before the child has finished writing, and a vLLM
    trace is tens of MB, so polling on file count alone races. We wait for sizes
    to stop changing as well.
    """
    deadline = time.time() + timeout
    last_sizes: dict[str, int] = {}
    stable_for = 0.0
    while time.time() < deadline:
        found = {
            os.path.join(trace_dir, f): os.path.getsize(os.path.join(trace_dir, f))
            for f in os.listdir(trace_dir)
            if os.path.getmtime(os.path.join(trace_dir, f)) >= since
        }
        if found and found == last_sizes:
            stable_for += 1.5
            if stable_for >= 4.5:
                return sorted(found)
        else:
            stable_for = 0.0
        last_sizes = found
        time.sleep(1.5)
    return sorted(last_sizes)


def main() -> None:
    args = parse_args()

    # Must happen before vllm is imported: the workers read this at startup, and
    # setting it later fails silently with an empty trace directory.
    trace_dir = os.path.abspath(args.trace_dir)
    os.makedirs(trace_dir, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = trace_dir

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    from bench_common import device_used_mib, print_env
    from bench_vllm import build_prompt_ids, timed_generate

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This trace only means anything on the 3060.")

    print_env()
    print(f"VLLM_TORCH_PROFILER_DIR = {trace_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = build_prompt_ids(tokenizer, args.prompt_tokens)
    prompt_tokens = len(prompt_token_ids)
    max_model_len = args.max_model_len or (prompt_tokens + args.clean_tokens + 16)

    mode = "eager (CUDA graphs OFF)" if args.enforce_eager else "CUDA graphs ON"
    print(f"\nloading {args.model} in fp16 through vLLM, {mode} ...")
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_model_len,
        # Without this the second run reuses the first run's prefill and the
        # prefill/decode subtraction below is meaningless.
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
    )
    print(f"device used after init: {device_used_mib():.0f} MiB")

    # Discarded. With graphs on, the first generate pays capture; either way it
    # pays autotuning and allocator warmup.
    print("warmup (discarded) ...")
    timed_generate(llm, prompt_token_ids, max_tokens=8)

    # --- pass 1: clean, no profiler ---
    # Same two-run subtraction as bench_vllm.py, because generate() is one
    # blocking call and will not hand us a prefill/decode split.
    print(f"\npass 1 (clean, no profiler): {args.clean_tokens} tokens")
    prefill_s = timed_generate(llm, prompt_token_ids, max_tokens=1)
    total_s = timed_generate(llm, prompt_token_ids, max_tokens=args.clean_tokens)
    decode_s = max(total_s - prefill_s, 1e-9)
    decode_count = max(args.clean_tokens - 1, 0)
    clean_step_ms = (decode_s / decode_count) * 1000.0
    decode_tps = decode_count / decode_s

    # --- pass 2: profiled ---
    print(f"pass 2 (profiled): {args.profile_tokens} tokens")
    started = time.time()
    llm.start_profile()
    prof_s = timed_generate(llm, prompt_token_ids, max_tokens=args.profile_tokens)
    llm.stop_profile()

    traces = newest_traces(trace_dir, since=started)
    if not traces:
        raise SystemExit(
            f"no trace appeared in {trace_dir}. Check that VLLM_TORCH_PROFILER_DIR "
            "was set before the engine started, which this script does, and that "
            "the vLLM version supports start_profile/stop_profile."
        )

    dest = args.copy_to or f"results/trace_{args.label}.json.gz"
    # One file per worker. Single GPU means one worker, so the largest file is
    # the model's trace; anything else is the parent process with nothing in it.
    biggest = max(traces, key=os.path.getsize)
    if len(traces) > 1:
        print(f"\n{len(traces)} trace files written, taking the largest:")
        for t in traces:
            print(f"  {os.path.getsize(t) / 1e6:8.1f} MB  {t}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copyfile(biggest, dest)

    print("\n" + "-" * 68)
    print(f"model              {args.model} fp16, batch 1")
    print(f"mode               {mode}")
    print(f"prompt tokens      {prompt_tokens}")
    print(f"clean decode       {decode_tps:8.1f} tok/s over {decode_count} tokens")
    print(f"clean step time    {clean_step_ms:8.3f} ms   <- use this as T")
    print(f"profiled pass      {prof_s:8.3f} s over {args.profile_tokens} tokens")
    print("-" * 68)
    print(f"trace written -> {dest}")
    print("\nnext:")
    if args.enforce_eager:
        # Eager has no annotations (the model runs in the child process, out of
        # reach of record_function) and no cudaGraphLaunch, so it needs a
        # once-per-forward-pass kernel as the step boundary. Prefill is the first
        # such pass, hence --trim-first 1.
        print(f"  python scripts/analyze_trace.py --trace {args.label}={dest} --list-kernels")
        print("  # pick a kernel whose count equals the number of forward passes, then:")
        print(f"  python scripts/analyze_trace.py \\")
        print(f"      --trace {args.label}={dest} \\")
        print(f"      --segment kernel --marker-kernel <name> --trim-first 1 \\")
        print(f"      --clean-step-ms {args.label}={clean_step_ms:.4f}")
    else:
        # With graphs on, each cudaGraphLaunch is a decode step boundary, and
        # prefill is not graph-captured so it falls outside every window and
        # drops out on its own.
        print(f"  python scripts/analyze_trace.py \\")
        print(f"      --trace {args.label}={dest} \\")
        print(f"      --segment graph \\")
        print(f"      --clean-step-ms {args.label}={clean_step_ms:.4f}")


if __name__ == "__main__":
    main()
