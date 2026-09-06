# llm-inference-optimization

A lab notebook of measured studies on LLM inference performance, all on a
single 12GB RTX 3060: how big the gap between a naive and an optimized engine
really is and where it comes from kernel by kernel, what the KV cache costs
until the card dies, what quantization buys and does not buy, and where
batching stops paying.

Every study follows the same discipline: one question, a reproducible script,
raw CSVs and plots committed alongside the code, and a writeup that separates
method from raw numbers from interpretation. The goal is a measured artifact,
not code that ran once.

## Results at a glance

| question | measured answer | writeup |
| --- | --- | --- |
| How big is the HuggingFace-to-vLLM gap? | ~4x decode throughput at batch 1 (18 to 23 tok/s against ~80), ~1.1x prefill. Decode memory-bandwidth utilization rises from 15 to 20% to ~68%. | [docs/baseline-vllm-results.md](docs/baseline-vllm-results.md) |
| Where does the gap come from? | 90% is removed CPU-launch idle, 9% faster kernels, measured by a torch.compile x CUDA-graphs ablation. Eager HF launches 1,198 kernels per token and idles the GPU 62.3% of each step; graphs replay the step with 17 CPU launches and near-zero idle. | [docs/profiling.md](docs/profiling.md) |
| Where is the KV-cache memory wall on 12 GB? | OOM at 123,565 tokens of context. Measured growth is 60 KB/token against the 28 KB/token analytical line: the 2x is copy-on-grow reallocation from `torch.cat`, and allocator fragmentation ends the run with 1.3 GiB still free but unusable. | [docs/oom-results.md](docs/oom-results.md) |
| What does 4-bit NF4 quantization buy? | Memory, not speed, at batch 1: weights drop 2945 to 1099 MiB, decode gets slower because dequantization runs every layer every step. The freed VRAM moves the OOM cliff out. | [docs/baseline-hf-results.md](docs/baseline-hf-results.md) |
| Where does batching stop paying? | Three measured regions: near-linear memory-bound scaling, the compute ceiling, then the KV-cache wall where requests queue and latency diverges. | [docs/batching-results.md](docs/batching-results.md) |
| Can first principles predict real kernels? | One bytes-per-output division predicts all eight matmul shapes in a decode step across a 590x size range, from a measured 291.5 GB/s bandwidth ceiling. MLP projections run at 96 to 97% of achievable bandwidth, grouped-query attention projections at 33%. | [docs/gate-phase2.md](docs/gate-phase2.md) |
| What does a token cost in energy? | Batching is the energy lever: 1552 mJ/token at batch 1 falls to 39 mJ at batch 128 (40x) because board power stays near 150 W regardless. NF4 is 1.2x worse on gross energy and 1.5x better net of idle: sleepable or busy GPUs favor fp16, an always-on idle box can favor NF4, and at low QPS the 38 W floor dwarfs both. | [docs/energy-results.md](docs/energy-results.md) |
| Do compressed tensor formats turn saved bytes into speed? | Only with a matching kernel and batch regime. Across the real decode matmuls, NF4 is 1.75x faster at M=1 but 2.09x slower at M=8. INT8 is unsupported below M=17, then wins 1.55x to 1.77x. Library 2:4 is 2.88x slower at M=1 despite 0.5625x weight storage; a handwritten CUDA GEMV recovers a 1.06x aggregate win. | [docs/formats-results.md](docs/formats-results.md) |
| How much KV cache can be evicted before quality and speed move? | A 4,096-token sink-and-window cache removed one-third of a 6,144-token WikiText context with perplexity effectively unchanged (11.213 to 11.219). At 120k context, a 512-token window held KV memory at 14 MiB and decode at 29.17 tok/s versus 3,281.6 MiB and 4.26 tok/s full cache, a 6.84x speedup. The simple attention-score policy was worse than the recent window at every budget. | [docs/eviction-results.md](docs/eviction-results.md) |
| What breaks first when requests arrive under load? | First-token latency, not decode latency. Poisson traffic meets a 1 s TTFT and 100 ms TPOT SLO through 2 requests/s, then misses at 3. Fixed bursts of eight miss TTFT even at 0.5 requests/s mean load. Memory 0.5 and chunked prefill do not move the knee; `max_num_seqs=16` protects TPOT under overload by making TTFT and the waiting queue worse. | [docs/load-results.md](docs/load-results.md) |

Two methodology findings worth stealing: two "identical" rented 3060s differed
by a uniform 1.22x in device busy time, so every comparison table here is
pinned to one box, and the advertised 360 GB/s is a spec while the measured
streaming-read ceiling is 291.5 GB/s, which is the honest denominator for any
bandwidth-utilization claim.

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

The lesson is that 4-bit is a memory lever, not a speed lever at batch 1: weights
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
