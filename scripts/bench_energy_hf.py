"""Phase 3 study 1: energy per token on the HuggingFace path (fp16 and NF4).

Everyone reports tokens per second; almost nobody reports joules per token, and
the two do not have to move together. This script reruns the repo's manual
prefill/decode loop (the same one baseline_hf.py and the OOM sweep use) with the
PowerSampler from bench_common recording board power throughout, and writes one
CSV row per measured run.

What is measured, per run:
  - an idle-floor window: the GPU synced and quiet for a few seconds, with the
    model resident, which is the honest baseline power for "net" numbers
  - prefill energy and decode energy, integrated over exactly the same
    sync-bounded windows the timers use
  - the driver's own energy counter across the same windows, when the card
    supports it, as an independent check on the integration

NF4 is the interesting cell: it moves a quarter of the weight bytes but
dequantizes every layer every step and runs slower. Whether that nets out to
more or fewer joules per token than fp16 is exactly the kind of question that
has to be measured rather than reasoned about.

The default --new-tokens is 512, higher than the timing scripts use, because
the energy integral needs a window long enough to span many NVML power updates
(the sampler reports its observed update interval so the row carries its own
evidence).

Usage (Environment A):
    python scripts/bench_energy_hf.py --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 --new-tokens 512 --repeats 3
    python scripts/bench_energy_hf.py --model Qwen/Qwen2.5-1.5B \
        --prompt-tokens 512 --new-tokens 512 --repeats 3 --quant nf4
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_hf import build_prompt_ids
from bench_common import (
    EnergyResult,
    PowerSampler,
    bytes_to_mib,
    device_used_mib,
    print_env,
    write_energy_result,
)


@torch.inference_mode()
def generate_with_marks(model, input_ids, new_tokens: int, device: int, sampler=None):
    """The manual prefill/decode loop, returning sync-bounded marks.

    Same loop as baseline_hf.generate_manual, but instead of CudaTimer objects
    it returns the perf_counter marks (t0, t1, t2) around prefill and decode,
    plus the driver energy-counter readings (e0, e1, e2) taken at the same
    marks when a sampler is given. Each counter read is one NVML call after a
    sync that already happened, tens of microseconds against half-second
    windows, so it does not perturb what is being measured.
    """
    def counter():
        return sampler.energy_mj() if sampler is not None else None

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    e0 = counter()
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    e1 = counter()

    decode_steps = max(new_tokens - 1, 0)
    for _ in range(decode_steps):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    e2 = counter()
    return t0, t1, t2, e0, e1, e2, decode_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=8.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0,
                        help="unmeasured quiet time before the idle window, so "
                        "post-warmup boosted clocks decay first")
    parser.add_argument("--csv", default="results/energy_hf.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quant", choices=["none", "nf4"], default="none")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This study measures GPU energy; run it on the box.")

    print_env()
    device = args.device
    dev_idx = torch.device(device).index or 0

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.quant == "nf4":
        from transformers import BitsAndBytesConfig

        print(f"\nloading {args.model} in 4-bit NF4 (fp16 compute) ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map={"": dev_idx}
        )
        dtype_label = "nf4"
    else:
        print(f"\nloading {args.model} in fp16 ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16
        ).to(device)
        dtype_label = "float16"
    model.eval()

    torch.cuda.synchronize(dev_idx)
    weights_vram = bytes_to_mib(torch.cuda.memory_allocated(dev_idx))
    print(f"weights resident: {weights_vram:.0f} MiB")

    input_ids = build_prompt_ids(tokenizer, args.prompt_tokens, device)
    actual_prompt_tokens = input_ids.shape[1]

    print("warmup (discarded) ...")
    generate_with_marks(model, input_ids, new_tokens=8, device=dev_idx)

    sampler = PowerSampler(device=dev_idx)
    sampler.start()

    # Idle floor: model resident, GPU quiet. Clocks stay boosted for a while
    # after work stops, so an unmeasured settle period comes first. Measured
    # once, applied to every repeat. The counter is the primary instrument
    # here too; the first smoke run showed NVML power updating only every
    # ~500 ms on this card, so integration needs long windows to mean much.
    print(f"idle floor: settling {args.settle_seconds:.0f}s, "
          f"then sampling {args.idle_seconds:.0f}s of quiet GPU ...")
    torch.cuda.synchronize(dev_idx)
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

    for rep in range(args.repeats):
        print(f"measured run {rep + 1}/{args.repeats}: "
              f"prompt={actual_prompt_tokens} tok, new={args.new_tokens} tok")
        t0, t1, t2, e0, e1, e2, decode_count = generate_with_marks(
            model, input_ids, args.new_tokens, device=dev_idx, sampler=sampler
        )
        prefill_s = t1 - t0
        decode_s = t2 - t1
        integrated_total_j = sampler.joules(t0, t2)
        counter_ok = e0 is not None and e1 is not None and e2 is not None
        if counter_ok:
            method = "counter"
            prefill_j = (e1 - e0) / 1000.0
            decode_j = (e2 - e1) / 1000.0
            total_j = (e2 - e0) / 1000.0
            counter_total_j = total_j
        else:
            method = "integration"
            prefill_j = sampler.joules(t0, t1)
            decode_j = sampler.joules(t1, t2)
            total_j = integrated_total_j
            counter_total_j = -1.0

        decode_tps = decode_count / decode_s if decode_s > 0 else 0.0
        jpt_gross = decode_j / decode_count if decode_count else 0.0
        jpt_net = (
            (decode_j - idle_watts * decode_s) / decode_count if decode_count else 0.0
        )

        result = EnergyResult(
            engine="hf",
            model=args.model,
            dtype=dtype_label,
            batch_size=1,
            prompt_tokens=actual_prompt_tokens,
            new_tokens=decode_count,
            prefill_seconds=prefill_s,
            decode_seconds=decode_s,
            decode_tokens_per_sec=decode_tps,
            idle_watts=idle_watts,
            mean_watts_decode=decode_j / decode_s if decode_s > 0 else 0.0,
            prefill_joules=prefill_j,
            decode_joules=decode_j,
            total_joules=total_j,
            joules_per_token_gross=jpt_gross,
            joules_per_token_net=jpt_net,
            energy_method=method,
            integrated_total_joules=integrated_total_j,
            counter_total_joules=counter_total_j,
            energy_counter_supported=sampler.energy_counter_supported,
            power_samples=len(sampler.samples),
            power_update_interval_ms=sampler.observed_update_interval_ms(),
            device_used_mib=device_used_mib(dev_idx),
            driver_version=sampler.driver_version,
            note=f"weights_vram_mib={weights_vram:.0f}",
        )
        write_energy_result(result, args.csv)

        print(f"  decode {decode_tps:7.1f} tok/s | "
              f"{result.mean_watts_decode:5.1f} W mean | "
              f"{jpt_gross * 1000:7.1f} mJ/tok gross, {jpt_net * 1000:7.1f} mJ/tok net "
              f"[{method}] | cross-check integrated {integrated_total_j:.1f} J "
              f"vs counter {counter_total_j:.1f} J")

    sampler.stop()
    print(f"\nidle floor {idle_watts:.1f} W, power update interval "
          f"{sampler.observed_update_interval_ms():.0f} ms, "
          f"{len(sampler.samples)} samples")
    print(f"rows appended -> {args.csv}")


if __name__ == "__main__":
    main()
