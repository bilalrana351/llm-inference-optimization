# vLLM baseline: measured numbers and what they mean

Task 2 results and their interpretation, on the same Vast.ai RTX 3060 as the HF
control. The whole point of this task is the gap against `baseline-hf-results.md`:
identical model, prompt, token count, batch, and decoding, only the engine
changed. The raw rows are in `results/baseline_vllm.csv`; this file is the
reading of them. Read alongside the HF baseline doc, since every number here is
relative to that one.

## Setup

- GPU: RTX 3060 12GB, Ampere, sm_86. Peak fp16 ~51 TFLOPS, memory bandwidth
  ~360 GB/s. Same card as the HF baseline.
- Model: Qwen/Qwen2.5-1.5B, fp16, 28 layers, 12 query heads / 2 KV heads (GQA),
  head_dim 128. The checkpoint ships bf16; vLLM cast it to fp16 on load as asked.
- Stack: vLLM 0.23.0 (v1 engine), torch 2.11.0+cu130 in the separate vLLM venv.
  FlashAttention 2 backend (`FLASH_ATTN`), confirming the old sm_75 fallback
  worry does not apply to this Ampere card.
- Workload: 512-token prompt, 256 new tokens, batch 1, greedy. Same prompt token
  ids as the HF run, built from the same seed-and-tile scheme and passed straight
  to vLLM so both engines see byte-identical input.
- Prefill and decode are split by timing two runs on the identical prompt: one
  with `max_tokens=1` (prefill plus the first sampled token, the TTFT) and one
  with `max_tokens=256`. decode time is the difference. This avoids depending on
  vLLM's internal RequestMetrics, which moved between the v0 and v1 engines.
  Prefix caching is disabled so the second run cannot reuse the first run's
  prefill, and `ignore_eos` forces exactly 256 tokens so the counts match the
  fixed-length HF loop.

## Measured (median of 3 runs)

| metric | vLLM | HF baseline | gap |
| --- | --- | --- | --- |
| prefill | ~124 ms (4114 tok/s over prompt) | ~137 ms (3740 tok/s) | ~1.1x |
| decode | ~79.9 tok/s (255 tokens) | ~18-23 tok/s | ~3.5-4x |
| decode MBU | ~68% | ~15-20% | |
| device VRAM | 11237 MiB (reserved, see below) | 3133 MiB peak alloc (organic) | not comparable |

The three runs were nearly identical: decode 79.889 / 79.903 / 79.908 tok/s, a
spread under 0.03%, against the HF baseline's ~30% run-to-run spread. CUDA graphs
make each decode step a deterministic replay, so the timing stops wandering with
clock and thermal state. Prefill was also stable at ~124 ms.

## The comparison is the result

### Prefill barely moved, and that is the expected answer

Both engines hand the prefill GEMMs to the same cuBLAS tensor-core kernels, so
there is no algorithm to beat. vLLM's ~10% edge (124 ms vs 137 ms, ~25% of the
compute peak vs ~23%) is fused kernels and CUDA graphs trimming launch overhead,
not a faster matmul. Prefill is compute-bound and already near the cuBLAS
ceiling, so this is where vLLM has the least to offer, and the table shows it.

### Decode is where the whole win lives: ~15-20% MBU to ~68% MBU

Run the same memory-bandwidth-utilization math the HF doc used:

```
vLLM decode: 3.191 s / 255 tokens = 12.5 ms/token
bytes moved per token = 3.09 GB weights (the same weights HF reads)
achieved bandwidth = 3.09 / 0.0125 = 247 GB/s
vs the 360 GB/s ceiling -> ~68% MBU
```

HF decode ran at ~15-20% MBU. vLLM is ~68%. That ~5-7x headroom the HF baseline
identified is now mostly cashed in. The mechanism is exactly the one the baseline
predicted: the gap was never FLOPs (decode arithmetic intensity is ~1-2
FLOP/byte, nothing can move it off the memory wall), it was memory traffic and
Python overhead. In HF, every decode token re-enters Python and launches hundreds
of per-layer kernels, and at batch 1 the GPU finishes each microscopic kernel
faster than Python can launch the next, so it stalls and the memory pipe runs
half-empty. vLLM captures the whole step into a CUDA graph (record once, replay
with one launch) and fuses kernels, so the GPU streams weights back to back with
no Python between tokens. Same card, same 360 GB/s ceiling, same weights to read;
vLLM just keeps the pipe full.

This single number, ~4x decode throughput at batch 1, is the core deliverable of
Task 2.

## VRAM: a reserved pool, read via NVML, not comparable to HF

The 11237 MiB figure is device-used memory, and it is not the HF-style "what
generation organically grew" number. Two things to know:

- vLLM pre-allocates the KV cache as one slab at construction, sized to
  `gpu_memory_utilization` (0.9 here). From the init log, the slab breaks down as
  weights 3.02 GiB, reserved KV pool 6.34 GiB, CUDA-graph buffers 0.41 GiB. The
  device-used total (11237 MiB) also includes the CUDA context and activation
  headroom on top of vLLM's own budget. The honest comparison against HF is
  decode tok/s and prefill latency, never this VRAM figure.
- The reading comes from NVML (`device_used_mib` in `bench_common.py`), not
  `torch.cuda`. The vLLM v1 engine runs the model in a separate EngineCore child
  process, so `torch.cuda.memory_allocated()` in the parent reads 0: the weights,
  the pool, and the graph buffers all live in the child. NVML reports the
  device's actual used memory, which does include the child's allocations. The
  first vLLM run recorded 0 MiB before this was fixed; that row was discarded.

The reserved pool held a `GPU KV cache size: 237,376 tokens` (6.34 GiB / 28 KB
per token), enough for ~302 concurrent 784-token sequences. That concurrency is
irrelevant at batch 1, but it is the headroom continuous batching would use, and
it is the same paged pool that will keep vLLM alive in the Task 3 OOM sweep while
plain HF fragments and crashes early.

## What carries into the next tasks

- The HF-vs-vLLM headline is set: ~1.1x prefill, ~4x decode, MBU ~15-20% to
  ~68%. This is the comparison table the blog draft needs.
- The ~4x is the batch-1 floor of vLLM's advantage, not the ceiling. It is
  almost entirely CUDA graphs plus fusion (getting decode to the bandwidth
  ceiling). The larger serving wins come from continuous batching, which a
  batch-1 benchmark deliberately does not show.
- PagedAttention contributes almost nothing at batch 1. Its value shows up in
  Task 3: the 237k-token paged pool is what the OOM experiment will contrast
  against HF's organic growth into `12 GB - weights`.
