# Energy per token: measured numbers and what they mean

Phase 3 study 1. Everyone reports tokens per second; this study reports joules
per token for the same configurations the repo has already characterized for
speed and memory, and the two rankings do not agree. The raw rows are in
`results/energy_hf.csv` and `results/energy_vllm.csv`, the figures are
`results/energy_per_token.png` and `results/energy_pareto.png`, and this file
is the reading of them.

Three findings, one instrument lesson:

1. **Batching is the energy lever: 40x.** vLLM's energy per generated token
   falls from 1552 mJ at batch 1 to 39 mJ at batch 128, while board power
   barely moves (149 to 162 W). The card burns roughly 150 W whether it is
   advancing 1 sequence or 128, so almost all of the throughput gain converts
   directly into energy savings.
2. **NF4 quantization loses on gross energy and wins on net.** Both statements
   are true at once. Which one matters is a utilization question: sleepable
   or fully busy GPUs favor fp16, an always-on box with idle gaps can favor
   NF4, and at low QPS the 38 W floor dwarfs the difference. Do not claim
   4-bit saves energy. The split is worked in the serving-minute section.
3. **vLLM's batch-1 energy win is smaller than its speed win.** 1.26x fewer
   joules per token against 1.6x more tokens per second, because keeping the
   memory pipe full costs power (149 W against HF's 116 W).

## Setup

- GPU: RTX 3060 12GB, Ampere sm_86, driver 595.71.05. A third rented box, so
  it was calibrated first per the repo rule: `measure_bandwidth.py` reports
  341.4 GB/s achievable read bandwidth and 26.2 fp16 tensor TFLOP/s. Numbers
  in this file are not comparable to the 291.5 GB/s profiling box, and the
  batch-1 baselines here confirm it (HF fp16 decodes at 59.5 tok/s on this
  box against 24.6 there; HF is launch-bound, so a faster host CPU moves it a
  lot).
- Model: Qwen/Qwen2.5-1.5B, the repo's standard subject. 512-token prompt,
  512 new tokens, greedy. The longer output (512 against the timing scripts'
  256) is deliberate: energy windows must span many instrument updates.
- Environment A (HF fp16 and NF4): torch 2.11.0+cu128, transformers 4.46.3,
  bitsandbytes. Environment B: vLLM 0.28.0 with torch 2.13.0+cu130.
- Scripts: `bench_energy_hf.py` (3 repeats per dtype), `bench_energy_vllm.py`
  (batch sweep 1 to 128, 2 repeats per point). Medians reported.

## The instrument, and why the energy counter is primary

`bench_common.PowerSampler` records two independent energy measurements:

- **The driver's energy counter** (`nvmlDeviceGetTotalEnergyConsumption`), a
  monotonic millijoule counter read at the same sync-bounded marks the timers
  use. Supported on this card. Exact.
- **Power integration**: board power sampled every 20 ms and trapezoid
  integrated. NVML turned out to refresh power only every **~503 ms** on this
  card, so integration is blind inside any window shorter than a few seconds.

The first smoke run made the failure mode concrete: on a 0.5-second decode
window, integration reported 2.2x less energy than the counter, and the
"idle" floor sampled right after warmup came out above the decode power
because boosted clocks had not decayed. The fixes are in the scripts: the
counter is the primary instrument, integration is recorded as a cross-check,
and the idle floor gets an unmeasured settle period before an 8-second
counter-timed window.

On the real runs (5 to 16 second windows) the two instruments agree within
0.5% on 20 of 22 rows. The two exceptions are the first run after an idle
period in each engine (counter 7.5% and 13.9% above integration), a
clock-transient effect the medians absorb.

The idle floor with the model resident and clocks settled is 37.7 to 39.3 W
across the three run groups. "Gross" numbers below charge the whole board
power to the tokens; "net" subtracts the idle floor times the window, which
is the incremental cost of the work itself.

## HF at batch 1: quantization does not save energy, except when it does

| | fp16 | NF4 | ratio |
| --- | --- | --- | --- |
| decode | 59.5 tok/s | 26.8 tok/s | 0.45x |
| board power during decode | 116.2 W | 63.6 W | 0.55x |
| energy per token, gross | 1960 mJ | 2374 mJ | **1.21x worse** |
| energy per token, net of idle | 1337 mJ | 905 mJ | **1.48x better** |

NF4 moves about a quarter of the weight bytes and the power draw shows it:
the incremental power above idle is 24 W against fp16's 78 W. But it decodes
2.2x slower (the repo's standing result: bitsandbytes dequantizes to fp16
every layer every step), so the run occupies the card 2.2x longer and the
idle floor is paid 2.2x more times.

Gross and net are two bills for the same run, not a fake number and a real
one. Gross is `board_watts x decode_seconds / tokens`. Net subtracts
`idle_watts x decode_seconds` first. The 38 W idle floor is measured once
per process (3 s unmeasured so boost clocks die, then 8 s of quiet with the
model still in VRAM) and is **not** inserted into generate. Prefill then
decode run back to back, with a `cuda.synchronize()` between them only so
the odometer stamp is honest.

**Why NF4 is cooler if memory clocks can stay the same.** Decode is a
streaming read of the weights. DRAM energy tracks **bytes actually moved**,
not the GDDR clock pin. The memory clock is a speed limit. fp16 toggles
that bus for ~3 GB per token; NF4 for about a quarter of that. Fewer row
activations and I/O toggles means the memory PHY and the voltage regulators
draw less, so the board sits at 64 W instead of 116 W even if the clock
frequency has not been set lower. We do not have a component split (NVML is
whole-board), but the 78 W to 24 W drop in incremental power is the size
you would expect from a 4x cut in HBM traffic on a memory-bound GEMV.

Prefill barely matters in either case: 11.1 J for the whole 512-token prompt,
about 22 mJ per prompt token, roughly 90x cheaper per token than decode.
Decode is where the energy lives, for the same memory-bound reason it is
where the time lives: one weight read serves 512 prompt tokens in prefill
and only one token in decode. Batching is that split again.

## What that means for a serving minute

The naive reading is "4-bit saves energy" or "4-bit wastes energy." Both
are wrong as slogans. Which precision wins is a utilization and billing
question. The arithmetic below uses the measured HF medians (59.5 vs 26.8
tok/s, 116 W vs 64 W, +78 W vs +24 W above a 38 W floor, 1960 vs 2374 mJ
gross, 1337 vs 905 mJ net) and is derived, not a live load test.

**Low traffic, GPU stays plugged in.** Twenty decode tokens in a 60 s
window. Both jobs finish in under a second; the rest of the minute is idle.

```
total = 38 W x 60 s  +  extra_watts x work_seconds
fp16:  2280 J + 78 x 0.34 s = 2306 J
NF4:   2280 J + 24 x 0.75 s = 2298 J
```

NF4 wins by 8 J on a 2300 J bill. Idle is ~99% of the minute. Quantization
is not the lever; turning the card off is.

A larger job that still fits: 952 tokens (16 s of fp16, 35.5 s of NF4),
GPU still on for the whole minute:

```
fp16:  2280 + 78 x 16   = 3528 J
NF4:   2280 + 24 x 35.5 = 3132 J
```

Same idle rent, cheaper DRAM work: NF4 wins on the minute's total joules.
This is the net ranking. It applies when uptime is fixed and idle gaps are
billed to the box anyway.

**Low traffic, GPU can sleep after the job.** Charge only busy time (gross):

```
fp16:  116 W x 16 s   = 1856 J
NF4:    64 W x 35.5 s = 2272 J
```

fp16 wins. It finishes sooner, so you pay 38 W for fewer seconds. This is
the dedicated / scale-to-zero ranking, and it is why Phase 1 already treated
NF4 as not a batch-1 speed lever.

**High traffic, GPU never idle.** Sixty seconds of continuous decode:

```
fp16:  116 W x 60 s = 6960 J for ~3570 tokens  ->  1960 mJ/tok
NF4:    64 W x 60 s = 3840 J for ~1608 tokens  ->  2374 mJ/tok
```

Per token, fp16 wins. NF4's board is cooler but it emits half the tokens.
Holding fp16's QPS takes ~2.2 NF4 GPUs: `2.2 x 64 W ≈ 141 W` against one
fp16 GPU at 116 W. More cool cards cost more power than one fast one.

The mapping people reach for (empty GPU → quantize, busy GPU → fp16) is
backwards on the busy side and overstated on the empty side:

| traffic | what the GPU does | right bill | energy winner |
| --- | --- | --- | --- |
| few requests, stays on between them | 38 W most of the time | net | NF4, often by a tiny amount vs the idle floor |
| few requests, can power down after | only busy seconds | gross | fp16 |
| many requests, always busy | no idle time | gross per token, or watts x GPUs for a QPS target | fp16 |

**The actual energy lever in this repo is still batching, not NF4 vs fp16.**
vLLM at batch 1 is 1552 mJ/tok; at batch 128 it is 39 mJ/tok, a 40x, because
board power stays near 150 W and many tokens share one weight read. That
swamps the 1.2x gross / 1.5x net quantization split. Quantization is the
place where gross and net **disagree**. Batching is the place where they
fall together.

Do not claim 4-bit saves energy. Claim that joules per token depend on
utilization and on who pays the 38 W. An empty GPU is a sleep problem. A
full GPU is a batching problem, and a faster precision can beat a cooler
one once you count cards.

## vLLM across batch: the 40x

| batch | decode tok/s | board W | mJ/tok gross | mJ/tok net | mJ/tok end to end |
| --- | --- | --- | --- | --- | --- |
| 1 | 96.0 | 149.0 | 1552 | 1149 | 1555 |
| 2 | 180.9 | 141.8 | 784 | 570 | 795 |
| 4 | 351.3 | 143.5 | 408 | 298 | 424 |
| 8 | 672.9 | 150.1 | 223 | 166 | 241 |
| 16 | 1243.0 | 156.9 | 126 | 95 | 144 |
| 32 | 2079.8 | 161.1 | 78 | 59 | 96 |
| 64 | 3554.4 | 165.7 | 47 | 36 | 66 |
| 128 | 4136.6 | 162.1 | 39 | 30 | 57 |

Board power is nearly flat, 142 to 166 W across a 43x throughput range. That
is the entire mechanism: a decode step reads the same weights whether it
advances 1 sequence or 128, so the DRAM traffic that costs the power is
almost fixed per step, and every added sequence divides the same joules
across more tokens. Energy per token therefore falls at almost exactly the
rate throughput rises, until the compute ceiling: from batch 64 to 128
throughput grows only 1.16x and the energy curve flattens with it, the same
knee `docs/batching-results.md` located in the throughput curve.

The end-to-end column (total energy over all output tokens, prefill included)
diverges from the decode-only column as batch grows, because the prefill of
batch x 512 prompt tokens is compute-heavy work that stops being negligible:
at batch 128 it is 18 of the 57 mJ/tok. A serving bill sees the end-to-end
number.

Engine against engine at batch 1: vLLM 1552 against HF 1960 mJ/tok gross,
1.26x, smaller than its 1.6x speed advantage on this box, because vLLM holds
the card at 149 W where launch-bound HF lets it rest at 116 W between
kernels. Removing idle time buys throughput at slightly worse than
energy-parity. The batching column is where the energy story actually is.

## Limitations

- Board-level power only. NVML reports whole-board watts; nothing here
  separates DRAM from SMs. The NF4 power drop is consistent with reduced
  memory traffic but is not a component-level measurement.
- One card, one model. The 40x is this card's idle-to-ceiling ratio times the
  batching range; a datacenter card with different idle behavior and a bigger
  model that saturates compute earlier will bend both curves.
- The idle floor is "model resident, clocks settled", not a wall-plug or
  whole-server figure. CPU and host energy are out of scope entirely.
- The vLLM decode split inherits the two-run subtraction from
  `bench_vllm.py`; the counter delta over the full run also prices the small
  gap between the counter read and the run start (microseconds at idle).
- fp16 to NF4 quality is out of scope here, as in Phase 1: this is an
  execution-cost study on fixed weights.
- The serving-minute arithmetic in the NF4 section is derived from the
  batch-1 HF medians, not from a live arrival process. Study 4 (load) is
  the place that would measure it under Poisson and bursty traffic.

## Reproducing

```bash
# Environment A
python scripts/bench_energy_hf.py --prompt-tokens 512 --new-tokens 512 --repeats 3
python scripts/bench_energy_hf.py --prompt-tokens 512 --new-tokens 512 --repeats 3 --quant nf4

# Environment B
python scripts/bench_energy_vllm.py --prompt-tokens 512 --new-tokens 512 \
    --sweep 1,2,4,8,16,32,64,128 --repeats 2

python scripts/plot_energy.py
```

Artifacts: `results/energy_hf.csv`, `results/energy_vllm.csv`,
`results/energy_per_token.png`, `results/energy_pareto.png`.
