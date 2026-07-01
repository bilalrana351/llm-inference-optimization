# vLLM batching throughput sweep: method and results

The last Phase 1 experiment. Everything before this (HF baseline, vLLM baseline,
OOM curve) is batch 1, which is latency. Batch 1 was called the floor of vLLM's
advantage; this experiment measures the throughput that continuous batching buys
on top of it, and finds where the GPU stops giving it back. The harness is
`scripts/bench_vllm_batch.py`, the plots come from `scripts/plot_batch_sweep.py`,
and the raw rows are in `results/vllm_batch_sweep.csv`.

## What is being measured

Send a growing number of concurrent sequences to vLLM at once and, at each batch
size, record aggregate throughput and per-sequence latency. The goal is to locate
three regions:

- **Memory-bound (the free lunch):** at batch 1 the decode GEMV reads the whole
  weight matrix to advance one sequence one token, and the tensor cores sit nearly
  idle. Adding sequences reuses that same weight read across more rows of a GEMM,
  so throughput climbs almost linearly while per-token latency barely moves. This
  is the region where batching is free.
- **Compute-bound:** once the batched GEMM is wide enough to keep the tensor cores
  busy, the weight read is fully amortized and each added sequence costs real
  compute. Throughput growth slows and per-token latency starts climbing.
- **KV-cache wall:** when the concurrent sequences need more KV blocks than the
  reserved pool holds, vLLM cannot run them all at once. It schedules a subset and
  queues the rest, so submitted-batch throughput flattens hard and per-request
  latency diverges (queued requests wait for running ones to finish).

## How the vLLM v1 engine shapes the method

Three facts change how this is measured versus the HF experiments:

1. **Concurrency is one `generate` call with N prompts.** vLLM's continuous
   batching schedules the list internally, so the batch is just the length of the
   prompt list. No threads, no async engine.
2. **The KV pool is reserved at startup**, sized by `gpu_memory_utilization`. So
   device VRAM is flat for the entire sweep and the KV wall does *not* show up as
   rising memory the way the HF OOM run did. It shows up as queuing. To see the
   pool fill you read block occupancy and the queue, not `nvidia-smi`.
3. **On a 1.5B model with a realistic 0.9 pool, compute saturates before the KV
   pool does.** Hundreds of short sequences fit, so the throughput curve bends from
   the compute ceiling long before the wall. To make the wall observable cheaply we
   run the sweep a second time with a deliberately small pool, the same logic that
   made the small GPU the right choice for the OOM run.

## Setup

- GPU: RTX 3060 12GB, Ampere sm_86. Peak fp16 ~51 TFLOPS, bandwidth ~360 GB/s.
- Model: Qwen/Qwen2.5-1.5B, fp16, 28 layers, 12 query / 2 KV heads (GQA),
  head_dim 128. Same model and prompt-building scheme as every prior run.
- Stack: vLLM 0.23.0 (v1 engine), torch 2.11.0+cu130, in `/workspace/vllm-env`.
- Workload: 512-token prompt, 256 new tokens, greedy, `ignore_eos` so every
  sequence emits exactly 256 tokens (makes total output = batch x 256 exact).
- Two pool configs, both writing to `results/vllm_batch_sweep.csv`:
  - `realistic`: `gpu_memory_utilization=0.9`, wide sweep to 256.
  - `constrained`: small pool tuned so the analytical KV wall lands near batch
    32 to 64, short sweep.

### Metric definitions

- **Throughput (headline):** total output tokens across the batch / wall time for
  the whole batch. End-to-end (prefill included), which is what a serving system
  actually delivers.
- **Decode throughput:** `N x (new_tokens-1) / decode_wall`, prefill excluded, to
  keep the repo's "never blend prefill into decode" rule alongside the headline.
- **TPOT (per-token latency):** `decode_wall / (new_tokens-1)`. While all N
  sequences run in one wave they decode in lockstep, so this is a well-defined
  per-sequence latency; past the wall it inflates and becomes a queuing signal.
- **TTFT:** wall of the `max_tokens=1` run, prefill plus the first token.
- **p50 / p95 per-request latency:** attempted from `RequestOutput.metrics`. Near
  degenerate in the realistic run (identical requests finish together), meaningful
  in the constrained run where queuing spreads them out. NaN if this vLLM build
  does not populate the metrics.

Prefill and decode are split the same way `baseline_vllm.py` does it (time a
`max_tokens=1` run and a full run, subtract), so this sweep does not depend on
vLLM's internal per-request timing, which moved between v0 and v1.

## Measured

TODO: run on the box and fill this in. Both commands are in the README. Expected
shape, to be confirmed or corrected against the numbers:

| region | realistic pool | what the plot shows |
| --- | --- | --- |
| memory-bound | batch 1 to ~___ | throughput ~linear, TPOT ~flat |
| compute-bound | ~___ to ~___ | throughput bends, TPOT climbs |
| KV wall | ~___ (analytical ___) | throughput flattens, queued=True |

Batch-1 sanity: the `batch=1` decode throughput here must match the vLLM baseline
(~79.9 tok/s in `baseline-vllm-results.md`) within a few percent; if it does not,
something about the setup changed between experiments.

Analytical KV wall: `num_gpu_blocks x block_size / (prompt+new)` sequences, read
from the vLLM init log and cross-checked against `bench_common.kv_cache_bytes`.
Reported next to the measured flattening point, the same measured-vs-analytical
discipline as the OOM curve.

Plots: `results/vllm_batch_throughput.png`, `results/vllm_batch_latency.png`,
`results/vllm_batch_tradeoff.png`.
