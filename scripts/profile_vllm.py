"""Week 6 Part A: capture a profiler trace of the vLLM decode step.

The counterpart to profile_decode.py. Same model, same prompt length, same fp16,
same batch size 1, so the only thing that changes is the engine.

Why this cannot just wrap torch.profiler around generate(): the v1 engine runs
the model in a separate EngineCore child process, so a profiler context opened in
the parent records an empty trace. bench_common.device_used_mib already documents
the same split for VRAM. vLLM's own hook handles it, but how you reach that hook
is version dependent. Up to roughly 0.11 it was the VLLM_TORCH_PROFILER_DIR
environment variable, read at worker startup. By 0.25 that variable is gone (the
engine logs it as unknown and start_profile() then raises "Profiling is not
enabled"), replaced by a profiler_config on the engine. enable_profiler() below
tries the config first and falls back to the environment variable, so the script
works either way and says which path it took.

Three configurations, because this vLLM does two separate things to the decode
step and lumping them together would misattribute the win:

    default          torch.compile (inductor) plus FULL_AND_PIECEWISE CUDA graphs
    --no-cudagraph   compilation kept, CUDA graphs off
    --enforce-eager  neither: no compile, no graphs

enforce_eager alone is a confounded control. It disables compilation as well as
graph capture, so eager-against-default measures fusion plus graphs together.
The middle run is the one that splits them: default against --no-cudagraph is the
CUDA-graph term on its own, and --no-cudagraph against --enforce-eager is the
inductor fusion term. That decomposition is the thing blog/draft.md currently
asserts without evidence.

Note that inductor fusion means vLLM's GPU kernel count per step should come in
well below the HF baseline's, not merely its launch count. A CUDA graph on its
own would leave the kernel count untouched.

Like the HF script, two passes: a clean unprofiled pass for the wall time per
step (T) and a profiled pass for the kernel intervals (B). The published gap is
T - B, each term taken from the run where it is not distorted.

Usage (Environment B, the vLLM venv):
    python scripts/profile_vllm.py --label vllm-graph
    python scripts/profile_vllm.py --label vllm-compile --no-cudagraph
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
        "--no-cudagraph",
        action="store_true",
        help="Disable CUDA graphs but keep torch.compile. This is the control "
        "that isolates the graph term from inductor fusion.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable compilation AND CUDA graphs. Confounds the two, so use it "
        "as the floor of the three-way comparison, not as the graph control.",
    )
    parser.add_argument(
        "--cudagraph-only",
        action="store_true",
        help="Full CUDA graphs with compilation off. The fourth cell of the "
        "design, and not guaranteed to be supported: verify from the init log "
        "that vLLM did not silently downgrade it.",
    )
    return parser.parse_args()


def enable_profiler(trace_dir: str) -> dict:
    """Turn on vLLM's torch profiler, returning kwargs for the LLM constructor.

    Two APIs exist in the wild. Newer engines (0.25 and up) take a profiler_config
    on the engine, exposed on the CLI as --profiler-config.profiler=torch and
    --profiler-config.torch_profiler_dir=DIR. Older ones read the environment
    variable VLLM_TORCH_PROFILER_DIR at worker startup, which the newer engines
    log as an unknown variable and otherwise ignore.

    Prefer the config, because a silently ignored environment variable does not
    fail until start_profile() raises, which is after the model has loaded and
    the clean pass has run. Set the variable too: it is harmless on engines that
    do not know it, and it is the whole mechanism on engines that do.
    """
    os.environ["VLLM_TORCH_PROFILER_DIR"] = trace_dir

    cfg_cls = None
    for module, name in (("vllm.config", "ProfilerConfig"),
                         ("vllm.config.profiler", "ProfilerConfig")):
        try:
            cfg_cls = getattr(__import__(module, fromlist=[name]), name)
            break
        except (ImportError, AttributeError):
            continue

    if cfg_cls is None:
        print("profiler: no ProfilerConfig in this vLLM, using VLLM_TORCH_PROFILER_DIR")
        return {}

    fields = _config_fields(cfg_cls)
    kwargs = {}
    if "profiler" in fields:
        kwargs["profiler"] = "torch"
    for candidate in ("torch_profiler_dir", "profiler_dir", "dir"):
        if candidate in fields:
            kwargs[candidate] = trace_dir
            break
    else:
        raise SystemExit(
            f"ProfilerConfig has no recognisable output-directory field. Fields: "
            f"{sorted(fields)}. Add the right one to enable_profiler()."
        )
    print(f"profiler: ProfilerConfig({kwargs})")
    return {"profiler_config": cfg_cls(**kwargs)}


def _config_fields(cls) -> set[str]:
    """Field names of a dataclass or a pydantic model, whichever this is."""
    import dataclasses

    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(getattr(cls, "model_fields", {}) or {})


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

    # On engines that use the environment variable, the workers read it at
    # startup, so it has to be set before vllm is imported. Setting it later
    # fails silently and produces an empty trace directory.
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
    print(f"profiler output dir = {trace_dir}")
    profiler_kwargs = enable_profiler(trace_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = build_prompt_ids(tokenizer, args.prompt_tokens)
    prompt_tokens = len(prompt_token_ids)
    max_model_len = args.max_model_len or (prompt_tokens + args.clean_tokens + 16)

    if args.enforce_eager:
        mode = "eager: no compile, no CUDA graphs"
        extra = {"enforce_eager": True}
    elif args.cudagraph_only:
        # CompilationMode.NONE is 0 (the init log prints the enum, VLLM_COMPILE
        # is 3). Piecewise graphs cannot exist here, since it is the compile pass
        # that splits the graph for them, but FULL capture is just stream capture
        # of whatever ops run and does not inherently need inductor.
        #
        # vLLM has historically warned and downgraded rather than failed on
        # unsupported combinations, so check the printed init line: if
        # cudagraph_mode is not FULL, this run is not the cell you asked for and
        # its numbers must not go in the table.
        mode = "CUDA graphs FULL, compilation OFF (verify in the init log)"
        extra = {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL"}}
    elif args.no_cudagraph:
        # Keep torch.compile, drop only graph capture. This is the control that
        # separates the CUDA-graph term from inductor's kernel fusion; setting
        # enforce_eager instead would remove both at once.
        mode = "compiled, CUDA graphs OFF"
        extra = {"compilation_config": {"cudagraph_mode": "NONE"}}
    else:
        mode = "compiled, CUDA graphs ON (shipping default)"
        extra = {}

    print(f"\nloading {args.model} in fp16 through vLLM, {mode} ...")
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_model_len,
        # Without this the second run reuses the first run's prefill and the
        # prefill/decode subtraction below is meaningless.
        enable_prefix_caching=False,
        **extra,
        **profiler_kwargs,
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
    # Printed here, not only in the final report: pass 2 is the fragile half
    # (profiler APIs move between vLLM versions) and there is no reason for its
    # failure to throw away a good measurement that already succeeded.
    print(f"  clean decode {decode_tps:.1f} tok/s, step {clean_step_ms:.3f} ms  <- T")

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
