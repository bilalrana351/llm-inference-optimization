"""Phase 3 study 1: energy per token through vLLM, across batch sizes.

The batching sweep showed throughput climbing with batch until the compute
ceiling; this script asks what that does to energy. Reading the same weights
once and advancing N sequences should amortize energy the way it amortizes
bandwidth, so joules per generated token should fall with batch until the
compute region, giving the energy curve its own knee.

Method mirrors bench_vllm.py exactly: prefill and decode are split by timing
(and here also integrating energy over) two runs on identical prompts, one
with max_tokens=1 and one with the full budget. Prefix caching is off so the
second run cannot reuse the first run's prefill. The PowerSampler runs in the
parent process, which is fine because NVML reads device-level board power: the
EngineCore child's work is visible to it.

Energy accounting per batch size:
  - prefill run:  E_p over its wall window
  - full run:     E_f over its wall window
  - decode energy = E_f - E_p, decode seconds = wall_f - wall_p
  - joules per generated token = decode energy / (N x (new_tokens - 1))
  - end-to-end joules per output token = E_f / (N x new_tokens), the number a
    serving bill actually sees

Usage (Environment B, the vLLM venv):
    python scripts/bench_energy_vllm.py --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 --new-tokens 512 \
        --sweep 1,2,4,8,16,32,64 --repeats 2
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from bench_common import (
    EnergyResult,
    PowerSampler,
    device_used_mib,
    print_env,
    write_energy_result,
)
from bench_vllm import build_prompt_ids


def timed_generate(llm, prompts, max_tokens: int):
    """One blocking generate; returns (t_start, t_end) perf_counter marks.

    generate() blocks until every sequence finishes, so wall marks around the
    call bound all device work. Greedy, ignore_eos, exactly max_tokens out per
    sequence.
    """
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
    return t0, time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=512)
    parser.add_argument("--sweep", default="1,2,4,8,16,32,64",
                        help="comma-separated batch sizes")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--idle-seconds", type=float, default=8.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0,
                        help="unmeasured quiet time before the idle window, so "
                        "post-warmup boosted clocks decay first")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--csv", default="results/energy_vllm.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This study measures GPU energy; run it on the box.")

    print_env()
    batches = [int(b) for b in args.sweep.split(",") if b.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = build_prompt_ids(tokenizer, args.prompt_tokens)
    actual_prompt_tokens = len(prompt_token_ids)
    max_model_len = actual_prompt_tokens + args.new_tokens + 16

    print(f"\nloading {args.model} in fp16 through vLLM ...")
    print(f"gpu_memory_utilization={args.gpu_mem_util}, max_model_len={max_model_len}")
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        enforce_eager=False,
    )
    post_init_mib = device_used_mib()
    print(f"device used after init: {post_init_mib:.0f} MiB")

    print("warmup (discarded) ...")
    timed_generate(llm, [{"prompt_token_ids": prompt_token_ids}], max_tokens=8)

    sampler = PowerSampler()
    sampler.start()

    # Counter primary, integration cross-check: NVML power updates only every
    # ~500 ms on this card (measured), so integration alone would need very
    # long windows. Clocks stay boosted after warmup, hence the settle period.
    print(f"idle floor: settling {args.settle_seconds:.0f}s, "
          f"then sampling {args.idle_seconds:.0f}s of quiet GPU ...")
    time.sleep(args.settle_seconds)
    idle_e0 = sampler.energy_mj()
    idle_t0 = time.perf_counter()
    time.sleep(args.idle_seconds)
    idle_t1 = time.perf_counter()
    idle_e1 = sampler.energy_mj()
    if idle_e0 is not None and idle_e1 is not None:
        idle_watts = (idle_e1 - idle_e0) / 1000.0 / (idle_t1 - idle_t0)
    else:
        idle_watts = sampler.mean_watts(idle_t0, idle_t1)
    print(f"idle floor: {idle_watts:.1f} W "
          f"({'counter' if sampler.energy_counter_supported else 'integration'})")

    for batch in batches:
        prompts = [{"prompt_token_ids": list(prompt_token_ids)} for _ in range(batch)]
        for rep in range(args.repeats):
            print(f"batch {batch:4d} run {rep + 1}/{args.repeats} ...", flush=True)
            e0 = sampler.energy_mj()
            p0, p1 = timed_generate(llm, prompts, max_tokens=1)
            e1 = sampler.energy_mj()
            f0, f1 = timed_generate(llm, prompts, max_tokens=args.new_tokens)
            e2 = sampler.energy_mj()

            prefill_s = p1 - p0
            full_s = f1 - f0
            decode_s = max(full_s - prefill_s, 1e-9)
            integrated_full_j = sampler.joules(f0, f1)

            counter_ok = e0 is not None and e1 is not None and e2 is not None
            if counter_ok:
                # Counter deltas over exactly the two generate calls. Note the
                # e1 to e2 delta also covers the tiny gap between the calls,
                # which is idle-priced and short; the integrated cross-check
                # bounds the error.
                method = "counter"
                prefill_j = (e1 - e0) / 1000.0
                full_j = (e2 - e1) / 1000.0
                counter_total_j = full_j
            else:
                method = "integration"
                prefill_j = sampler.joules(p0, p1)
                full_j = integrated_full_j
                counter_total_j = -1.0
            decode_j = max(full_j - prefill_j, 0.0)

            decode_count = batch * max(args.new_tokens - 1, 0)
            output_count = batch * args.new_tokens
            decode_tps = decode_count / decode_s
            jpt_gross = decode_j / decode_count if decode_count else 0.0
            jpt_net = (
                (decode_j - idle_watts * decode_s) / decode_count if decode_count else 0.0
            )
            e2e_jpt = full_j / output_count if output_count else 0.0

            result = EnergyResult(
                engine="vllm",
                model=args.model,
                dtype="float16",
                batch_size=batch,
                prompt_tokens=actual_prompt_tokens,
                new_tokens=decode_count,
                prefill_seconds=prefill_s,
                decode_seconds=decode_s,
                decode_tokens_per_sec=decode_tps,
                idle_watts=idle_watts,
                mean_watts_decode=decode_j / decode_s if decode_s > 0 else 0.0,
                prefill_joules=prefill_j,
                decode_joules=decode_j,
                total_joules=full_j,
                joules_per_token_gross=jpt_gross,
                joules_per_token_net=jpt_net,
                energy_method=method,
                integrated_total_joules=integrated_full_j,
                counter_total_joules=counter_total_j,
                energy_counter_supported=sampler.energy_counter_supported,
                power_samples=len(sampler.samples),
                power_update_interval_ms=sampler.observed_update_interval_ms(),
                device_used_mib=max(post_init_mib, device_used_mib()),
                driver_version=sampler.driver_version,
                note=(
                    f"end_to_end_joules_per_output_token={e2e_jpt:.4f}; "
                    f"gpu_mem_util={args.gpu_mem_util}; decode split by two-run "
                    f"subtraction (see bench_vllm.py)"
                ),
            )
            write_energy_result(result, args.csv)
            print(f"  {decode_tps:8.1f} tok/s | {result.mean_watts_decode:5.1f} W mean "
                  f"| {jpt_gross * 1000:7.1f} mJ/tok gross "
                  f"| {e2e_jpt * 1000:7.1f} mJ/tok end-to-end")

    sampler.stop()
    print(f"\nidle floor {idle_watts:.1f} W, power update interval "
          f"{sampler.observed_update_interval_ms():.0f} ms, "
          f"{len(sampler.samples)} samples")
    print(f"rows appended -> {args.csv}")


if __name__ == "__main__":
    main()
