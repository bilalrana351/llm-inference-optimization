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

## Measured: the realistic pool

Median of 2 repeats, `gpu_memory_utilization=0.9`, `max_model_len=784`,
`num_gpu_blocks=14836`, analytical KV capacity ~309 sequences. Raw rows in
`results/vllm_batch_sweep.csv`.

| batch | decode tok/s | TPOT ms | end-to-end tok/s | scaling eff |
| --- | --- | --- | --- | --- |
| 1 | 88.6 | 11.3 | 85.7 | 100% |
| 2 | 139 | 14.4 | 132 | 78% |
| 4 | 269 | 14.9 | 246 | 76% |
| 8 | 501 | 16.0 | 427 | 71% |
| 16 | 931 | 17.2 | 708 | 66% |
| 32 | 1499 | 21.4 | 997 | 53% |
| 48 | 2064 | 23.3 | 1220 | 49% |
| 64 | **2696** | 23.7 | 1416 | 47% |
| 96 | 2608 | 36.8 | 1393 | 31% |
| 128 | 2879 | 44.5 | 1466 | 25% |
| 192 | 2657 | 72.3 | 1407 | 16% |
| 256 | 2967 | 86.3 | 1490 | 13% |

`scaling eff` = `(decode tok/s / 88.6) / batch`: how much of ideal linear scaling
survives. Decode throughput peaks in practice around batch 64 (2696 tok/s) and
barely moves after (2967 at 4x the batch). TPOT is roughly flat to 64 (11 to 24
ms) then climbs steeply (37, 44, 72, 86 ms). So the sweet spot is ~64: past it you
buy latency for no throughput.

Batch-1 sanity: this sweep's batch-1 decode is 88.6 tok/s vs the vLLM baseline's
~79.9 tok/s (`baseline-vllm-results.md`), about 11% higher rather than a clean
match. Worth reconciling on the next run (the baseline was median-of-3 at a
slightly different point in the session); the shape of the sweep does not depend
on it, but the discrepancy should not be swept under the rug.

Constrained-pool run (forces the KV wall at low batch) and the plots are still
pending: `results/vllm_batch_throughput.png`, `vllm_batch_latency.png`,
`vllm_batch_tradeoff.png`.

## The three-part memory model

The naive model of a transformer forward pass is two terms: compute, and loading
the weights. That is wrong in a way that matters here. There are three sources of
memory traffic, not one:

1. **Weights.** Read once per forward pass and *shared* across every token and
   sequence in the batch. A fixed cost: ~3.09 GB for this model, batch-independent.
2. **KV cache.** Read every decode step, scales with *batch x context*, and is
   *not* shared across sequences. ~18 MB per sequence at this context. This is the
   term the two-part model ignores, and it is the one that ends the batching free
   lunch.
3. **Activations.** The token vectors flowing through the layers, scale with
   batch x hidden, usually small and often resident in L2/SRAM. Minor for decode.

FLOPs come from the weight matmuls (proportional to tokens processed) plus
attention (proportional to tokens x context, the C-squared term discussed under
prefill below). The one-line lesson of this whole experiment: **KV-cache traffic
is a batch-scaling memory cost that the compute-plus-weights model leaves out, and
it is what sets the batching sweet spot.**

## Why throughput climbs to 64, then plateaus

Model one decode step as a fixed cost plus a per-sequence cost:

```
step_time(batch) = T_fixed + c1 x batch
```

- `T_fixed` ~ 11 ms is the shared weight read (3.09 GB streamed through the ALUs),
  paid whether there is 1 sequence or 64.
- `c1` ~ 0.2 ms/seq is the per-sequence marginal cost: that sequence's own KV read
  plus its slice of the GEMM plus overhead.

Throughput is tokens per step over time per step:

```
throughput(batch) = batch / (T_fixed + c1 x batch)
```

- **Small batch:** `T_fixed` dominates, so throughput ~ batch / T_fixed, rising
  almost linearly. This is the free lunch: the fixed weight read is being
  amortized over more tokens. Batch 1 pays the full 11 ms to make a single token,
  the purest form of memory-bound decode waste; batch 64 pays ~24 ms to make 64.
- **Large batch:** `c1 x batch` dominates, so throughput approaches `1 / c1`, a
  constant. That is the plateau, and it is arithmetic, not a wall: once each step's
  time is proportional to the tokens in it, tokens per second is pinned. Throughput
  ever rose only because there was a fixed cost to spread; once the weight read is
  thin next to the per-sequence cost, there is nothing left to amortize and you are
  just doing N independent token-computes back to back.

Fitting the data: `T_fixed` ~ 11 ms, `c1` ~ 0.20 ms/seq at low batch rising to
~0.33 ms/seq at batch 256 (context lengthens and paged attention gets less
efficient), so `1 / c1` ~ 3000 tok/s, exactly where decode throughput parks. The
knee is where the batch-linear cost catches the fixed cost: `0.19 x batch ~ 11` ->
batch ~ 58, matching the measured ~64.

## Memory-bound, but not bandwidth-saturated

At the plateau the binding constraint is memory traffic, *not* compute, but the
memory bus is *not* maxed either:

- Decode FLOPs/token ~ 2 x params ~ 3 GFLOP. At 2967 tok/s that is **~8-9 TFLOPS,
  about 16% of the 3060's ~51 TFLOPS fp16 peak.** Compute is nowhere near the
  limit.
- Bytes/step at batch 256 ~ 3.1 GB weights + ~4.6 GB KV ~ 7.7 GB in 86 ms ~
  **~90 GB/s, about 25% of the 360 GB/s peak.** Bandwidth is not saturated.

So the honest label is "memory-*traffic*-limited at sub-peak efficiency," not
"bandwidth-saturated." The cost that scales with batch and pins throughput is the
KV read, which is memory in nature, so "decode is memory-bound" holds as a
first-order heuristic. But the reason neither peak is reached is *operation shape*:
the decode GEMMs are skinny (M = batch = 64 to 256), which is occupancy-bound and
tops out around ~8-9 TFLOPS, and the paged KV reads run at a fraction of peak
bandwidth because the access pattern is scattered. The plateau is the achievable
ceiling of *these specific kernels*, not the card's spec sheet.

## Prefill is batching over positions

Prefill and batched decode are the same mechanism: amortizing the fixed weight
read over many tokens. Decode amortizes over sequences; prefill amortizes over the
512 positions of one sequence, processed in a single parallel pass.

That one fact explains why prefill behaves so differently:

- **The weight GEMMs have M = 512 rows** (versus decode's M = batch). Big dense
  GEMMs fill the tensor cores, so prefill reaches ~68% of compute peak and is
  genuinely compute-bound. Prefill runs ~4100 tok/s against batch-1 decode's ~88:
  same weights read, 512 tokens produced from it instead of 1.
- **The C-squared in attention is compute, not memory.** Attention FLOPs are
  O(P-squared x d) because every token attends to every earlier one, but
  FlashAttention (vLLM's `FLASH_ATTN` backend) never writes the P x P score matrix
  to HBM: it computes it in on-chip SRAM tiles and discards it. So attention
  *memory* traffic is O(P x d), linear in context; only the *compute* is
  quadratic. The O(P-squared) memory blowup of naive attention is exactly what
  FlashAttention removes.
- **At P = 512 the C-squared compute is still minor.** Per token the weight
  matmuls are ~3 GFLOP; prefill attention is ~0.09 GFLOP. The FFN and projection
  GEMMs dominate. The quadratic term only overtakes them at much longer prompts
  (thousands of tokens), where attention compute finally exceeds the FFN.

Same card, same weights: prefill's M = 512 GEMMs reach compute-bound, decode's
M = 64 GEMMs stay skinny at ~16% of compute. Operation shape is the whole story.

## Roofline: theoretical vs measured optimal batch

The arithmetic intensity of batched decode is ~ batch size (FLOPs = 2 x params x
batch, weight bytes = 2 x params read once, so intensity ~ batch). The roofline
ridge, where memory-bound meets compute-bound, is at intensity = FLOPS / bandwidth:

```
B* = 51 TFLOPS / 360 GB/s ~ 142    (fp32-accumulate tensor peak)
   = 100 TFLOPS / 360 GB/s ~ 280   (fp16-accumulate peak)
```

So the naive roofline says the ideal batch is ~140 to 280. **Measured knee ~64**,
2x to 4x lower. They diverge because the roofline assumes you reach peak FLOPS,
which skinny decode GEMMs do not (they cap at ~8-9 TFLOPS), and because the
weights-only intensity model ignores the KV traffic that makes the practical
crossover a memory-vs-memory event (weight read vs KV read) rather than a compute
event. The gap between the idealized ridge and the measured knee is itself a
result: it is the cost of imperfect kernels and unshared KV traffic.

## Cross-GPU prediction (RTX 3090 hypothesis, to test)

The 3090 has ~936 GB/s bandwidth (2.6x the 3060's 360) and ~71 TFLOPS fp16 tensor
(1.4x the 3060's ~51). Because both `T_fixed` (weight read) and `c1` (KV read)
scale as 1 / bandwidth, the falsifiable predictions are:

- **Throughput up ~2 to 2.6x at every batch**, plateau near ~6000-7000 tok/s.
  Throughput tracks bandwidth, and the 3090 has much more of it.
- **Latency down ~2.6x**, especially at low batch, because the fixed weight read
  drops from ~8.6 ms to ~3.3 ms.
- **Optimal batch stays in the same ballpark (~64), possibly slightly lower, not
  higher.** This is the counterintuitive one. The compute-roofline ridge is
  actually *lower* on the 3090 (`71 / 936 ~ 76` vs `51 / 360 ~ 142`) because its
  bandwidth grew more than its compute, so relative to itself it is more
  memory-rich and goes compute-bound sooner. And since the measured knee (64) is
  already below the compute ridge, the knee is set by the weight-read-vs-KV-read
  crossover, which is `weight_bytes / KV_bytes_per_sequence`: a property of the
  *model*, not the GPU, and roughly bandwidth-invariant. If the 3090's knee jumps
  to ~150, this model is wrong and we learn something. If it stays ~64 with ~2.5x
  higher throughput, the model holds.

Other cards worth the same sweep: T4 (65 TFLOPS / 320 GB/s, ridge ~200) and V100
(125 TFLOPS / 900 GB/s, ridge ~140) on the NUST HPC cluster, since they are
already the roadmap hardware and bracket the 3060 on both axes.
