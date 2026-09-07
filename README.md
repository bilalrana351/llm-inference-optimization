# llm-inference-optimization

This repository contains measured studies of how LLM inference behaves on a
12 GB GPU. Each study includes a clear question, runnable code, raw data,
plots, and a detailed report. Every comparison uses one calibrated RTX 3060,
so results from different machines are never mixed in the same table.

For simpler explanations, read the
[blog](https://www.bilalrana.com/blog/).

## Featured studies

### 1. Do quantization and sparsity make inference faster?

Smaller weights were not automatically faster. NF4 helped at batch 1 but became
slower at larger batches because it used a different kernel path. INT8 helped
only when the matrix was large enough. A custom CUDA 2:4 sparse kernel showed
that speed depends on both the representation and the kernel.

[Read the blog](https://www.bilalrana.com/blog/does-compression-make-llm-inference-faster/)
or [see the code, data, and full report](docs/formats-results.md).

### 2. How much energy does LLM inference use?

NF4 drew less power at batch 1 but also ran for longer. Batching made the larger
difference, reducing gross energy from 1,552 mJ per token at batch 1 to 39 mJ
per token at batch 128.

[Read the blog](https://www.bilalrana.com/blog/lower-power-higher-energy/)
or [see the code, data, and full report](docs/energy-results.md).

### 3. How much KV cache can be pruned before quality degrades?

Removing one-third of the cache changed perplexity from 11.213 to 11.219. A
bounded recent window also kept memory use and decode time almost flat as the
total context became longer.

[Read the blog](https://www.bilalrana.com/blog/how-much-kv-cache-can-you-prune/)
or [see the code, data, and full report](docs/eviction-results.md).

### 4. What happens when an LLM server receives too much traffic?

First-token latency failed before the time between later tokens. Fixed bursts
of eight requests missed the latency target even at a low average request rate,
showing that throughput alone does not describe the user experience.

[Read the blog](https://www.bilalrana.com/blog/what-breaks-first-under-load/)
or [see the code, data, and full report](docs/load-results.md).

## Earlier studies

| Question | Measured answer | Full report |
| --- | --- | --- |
| How large is the HuggingFace to vLLM gap? | vLLM produced about 4 times more decode tokens per second at batch 1. Prefill performance was similar. | [Baseline and vLLM](docs/baseline-vllm-results.md) |
| Where does the speed difference come from? | CUDA graphs removed most of the time spent waiting between kernel launches. Eager HuggingFace launched 1,198 kernels per token, while graph replay used 17 CPU launches. | [Profiling](docs/profiling.md) |
| Where does the KV cache run out of memory? | The run failed at 123,565 context tokens. Cache growth, buffer copying, and allocator fragmentation explain why the measured memory cost was above the analytical estimate. | [OOM study](docs/oom-results.md) |
| What does NF4 quantization provide at batch 1? | Model weights fell from 2,945 MiB to 1,099 MiB, but decode became slower. NF4 saved memory rather than execution time in this case. | [HuggingFace baseline](docs/baseline-hf-results.md) |
| Where does batching stop helping? | Throughput first scaled almost linearly, then reached the compute limit, and finally reached the KV-cache limit where requests began waiting. | [Batching](docs/batching-results.md) |
| Can a simple hardware model predict kernel time? | A bytes-per-output model described all eight decode matrix multiplications across a 590 times size range and showed which projections used bandwidth poorly. | [Kernel analysis](docs/gate-phase2.md) |

Two rented RTX 3060 machines with the same listed specifications differed by
1.22 times in device execution time. For that reason, every comparison stays
on one machine and uses that machine's measured bandwidth.

## What is here

```
scripts/
  bench_common.py            shared measurement harness (timing, VRAM, NVML, logging, KV math)
  baseline_hf.py             HuggingFace transformers baseline, fp16 and NF4 (the control)
  bench_vllm.py              vLLM, same model and workload
  oom_sweep.py               the KV-cache OOM experiment
  bench_vllm_batch.py        vLLM batching throughput/latency sweep
  plot_batch_sweep.py        plots for the batching sweep
  analyze_vram_deviations.py per-step VRAM delta reader for the OOM CSV
  profile_decode.py          PyTorch-profiler trace of HF decode steps
  profile_vllm.py            vLLM decode traces, the compile x CUDA-graphs 2x2
  analyze_trace.py           reduces traces to kernels, launches, and idle per step
  dump_step_kernels.py       one decode step, kernel by kernel
  roofline_step.py           every decode matmul against the bandwidth ceiling
  measure_bandwidth.py       the card's achievable-bandwidth microbenchmark
  bench_energy_hf.py         HF fp16 and NF4 energy per token
  bench_energy_vllm.py       vLLM energy against batch size
  bench_formats.py           dense, INT8, NF4, and 2:4 on decode-step shapes
  formats_cuda_kernel.cu     handwritten 2:4 sparse fp16 GEMV
  bench_eviction.py          KV eviction quality, memory, and decode-speed sweep
  plot_eviction.py           quality and systems plots for KV eviction
  bench_load.py              async Poisson and bursty vLLM load generator
  run_load_study.py          restarts and measures four server configurations
  plot_load.py               goodput, latency-knee, and scheduler plots
results/                     raw CSV logs, traces, and plots
docs/                        one writeup per study, plans, and the compute record
blog/draft.md                writeup, grown alongside the measurements
```

See `docs/phase1.md` for the original Phase 1 plan, `docs/vastai.md` for the
compute record, and `CLAUDE.md` for persistent context.

## What is next

The four core studies in [docs/phase3.md](docs/phase3.md) are complete. The
remaining stretch study is model splitting across processes. Same shape as
everything above: one question, one headline number, one doc, and focused
plots.

## Hardware

The studies run on rented Vast.ai boxes, not Colab. The full per-box record is
in `docs/vastai.md`.

- GPU: one NVIDIA RTX 3060, 12GB (12,288 MiB), Ampere, compute capability 8.6
  (sm_86) for every study.
- Each comparison table is pinned to one calibrated box. Driver, CUDA, torch,
  host CPU, and measured bandwidth are recorded with that study because
  nominally identical rentals did not perform identically.
- Recent Vast base containers use `/workspace` without a mounted volume. The
  repo is local-first: scripts are synced in and results are copied back before
  the instance is recycled or destroyed.

The small card is intentional: at 12GB the KV cache hits the wall at a realistic
context length, which is the headline experiment, not something to avoid.

## How to run

Two environments on purpose, so the vLLM install cannot disturb the baseline
torch.

### Environment A: HuggingFace baseline (Task 1)

Uses the preinstalled torch 2.12. Just add the baseline pins:

```bash
export HF_HOME=/workspace/hf-cache          # cache weights on persistent storage
pip install -r requirements.txt             # transformers, accelerate, bitsandbytes, matplotlib, pandas
python scripts/baseline_hf.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256
```

The same script runs the 4-bit NF4 comparison through the identical prefill/decode
loop, so the only change is how the weights are stored:

```bash
python scripts/baseline_hf.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256 --quant nf4
```

The lesson is that 4-bit saves memory but does not save time at batch 1: weights
drop from ~2945 to ~1099 MiB, but decode slows (bitsandbytes dequantizes NF4 to
fp16 every layer every step, and decode was already memory-bound). The freed VRAM
is the win, and it moves the OOM cliff out.

The OOM sweep (Task 3) also runs in Environment A, because HF grows the cache
organically so the crash is a real memory event:

```bash
python scripts/oom_sweep.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 32 \
    --checkpoint-every 1000 --csv results/oom_sweep.csv --plot results/oom_curve.png
```

The full sweep is a single continuous decode and takes about six hours of wall
time on the 3060.

### Environment B: vLLM (Task 2)

vLLM is binary-tied to a specific torch build, so a naive `pip install vllm` into
Environment A would clobber the box's torch 2.12 with vLLM's bundled 2.11. Install
it into a fresh, isolated venv and let the wheel bring its own torch for CUDA
13.0:

```bash
python -m venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate
# uv is the recommended installer; it resolves the CUDA 13.0 wheel cleanly
uv pip install vllm --torch-backend=cu130
python scripts/bench_vllm.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256
```

The batching throughput sweep (Week 3) also runs in Environment B. It sends a
growing number of concurrent sequences to vLLM and measures throughput and
latency at each batch size. Run it twice: a realistic pool to find where compute
saturates, and a deliberately constrained pool to force the KV-cache wall at a low
batch size. Then plot:

```bash
# realistic pool: throughput climbs then flattens from the compute ceiling
python scripts/bench_vllm_batch.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 \
    --new-tokens 256 --gpu-mem-util 0.9 --sweep 1,2,4,8,16,32,48,64,96,128,192,256 \
    --repeats 2 --tag realistic
# constrained pool: small KV pool so the wall bites early and requests queue
python scripts/bench_vllm_batch.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 \
    --new-tokens 256 --gpu-mem-util 0.4 --sweep 1,2,4,8,16,32,48,64 \
    --repeats 2 --tag constrained
python scripts/plot_batch_sweep.py
```

Throughput is total output tokens across the batch divided by wall time, never
blended with the batch-1 decode rate. `ignore_eos` forces exactly the requested
output length per sequence, so total output equals batch x output length.

### Profiling runs

The decode-step traces behind `docs/profiling.md` reproduce with:

```bash
# HF trace (Environment A)
python scripts/profile_decode.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512

# the four vLLM cells of the compile x graphs ablation (Environment B)
python scripts/profile_vllm.py --label vllm-graph
python scripts/profile_vllm.py --label vllm-compile   --no-cudagraph
python scripts/profile_vllm.py --label vllm-eager     --enforce-eager
python scripts/profile_vllm.py --label vllm-graphonly --cudagraph-only
```

Analysis commands, including how traces reduce to kernels, launches, and idle
time per step, are at the end of `docs/profiling.md`.

## Constraints

- fp16 only, never bf16. The 3060 supports bf16, but we target fp16 for
  comparability with the later NUST HPC V100/T4 runs (which do not support bf16).
  It does not change the KV-cache OOM math (fp16 and bf16 are both 2 bytes).
- Single GPU, 12GB. The small GPU is intentional: the point is to hit the
  KV-cache wall at a realistic context length.
- Small model (Qwen2.5-1.5B, ~3GB of fp16 weights), leaving roughly 9GB for the
  cache to grow into.
- One box per comparison table. Two rented 3060s with identical listings
  measured 1.22x apart in device busy time, so numbers from different boxes are
  never mixed in one table.

## Working dependency versions

The vLLM install is the friction point of this phase. The combinations that
actually ran on the 3060:

- GPU: RTX 3060 12GB (Ampere, sm_86), CUDA 13.0.
- Environment A (baseline + OOM): torch 2.12.0+cu130 (preinstalled),
  transformers 4.46.3, accelerate 0.34.2. Pins in `requirements.txt`.
- Environment B (vLLM): vLLM 0.23.0 (v1 engine) with torch 2.11.0+cu130, in the
  separate `/workspace/vllm-env`. The profiling runs used vLLM 0.25.1 with
  torch 2.12.0+cu130; see `docs/profiling.md` for that stack.
- The serving-under-load study ran on a CUDA 12.8 driver with vLLM 0.10.2,
  torch 2.8.0+cu128, transformers 4.53.2, and NumPy 2.2.6. See
  `docs/load-results.md` for the exact server and workload settings.

The transformers pin matters: 4.46+ moved rotary embeddings to a single
model-level module, dropping a fixed ~1.79 GB of duplicated cos/sin buffers that
4.44.x kept across all 28 layers. See the comment in `requirements.txt`.
