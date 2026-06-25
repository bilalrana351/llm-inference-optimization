# HF baseline: measured numbers and what they mean

Task 1 results and their interpretation, on the Vast.ai RTX 3060. This is the
control every later number is measured against. The raw rows are in
`results/baseline_hf.csv`; this file is the reading of them.

## Setup

- GPU: RTX 3060 12GB, Ampere, sm_86. Peak fp16 ~51 TFLOPS (tensor core, FP32
  accumulate, dense). Memory bandwidth ~360 GB/s (192-bit GDDR6, 15 Gbps).
- Model: Qwen/Qwen2.5-1.5B, fp16, 1.544e9 params, tied embeddings, 28 layers,
  12 query heads / 2 KV heads (GQA), head_dim 128.
- Stack: torch 2.12.0+cu130, transformers 4.46.3.
- Workload: 512-token prompt, 256 new tokens, batch 1, greedy. Manual prefill +
  decode loop (not `.generate()`) so the two phases are timed apart.

## Measured (corrected run, transformers 4.46.3)

| metric | value |
| --- | --- |
| prefill | ~137 ms (3740 tok/s over the 512-token prompt) |
| decode | ~18-23 tok/s (255 tokens) |
| weights VRAM | 2945 MiB |
| peak allocated | 3133 MiB |
| peak reserved | 3360 MiB |

Weights are 2945 MiB, one MiB off the `1.544e9 x 2 bytes = 2944 MiB` fp16 floor.
The earlier 4737 MiB reading was a transformers 4.44.2 RoPE buffer artifact, since
fixed; see `notes-cuda-memory-and-timing.md` for that story. KV cache plus
activations for this 768-token run add only ~190 MiB on top of weights.

## The roofline this sits on

Two ceilings: 51 TFLOPS of compute and 360 GB/s of bandwidth. They cross at an
arithmetic intensity of `51e12 / 360e9 = 142 FLOP/byte`. Prefill lives left of
that ridge (compute-bound), decode lives far right of it (memory-bound). The two
phases are not slow and fast versions of the same thing; they hit different
walls.

### Prefill is compute-bound, and HF reaches ~23% of the compute ceiling

Forward-pass FLOPs are about `2 x params x tokens` (the linear layers):

```
2 x 1.544e9 x 512 = 1.58e12 FLOP
floor at 51 TFLOPS = 1.58e12 / 51e12 = 31 ms
measured           = 137 ms  ->  11.5 TFLOPS effective, ~23% of peak
```

So the compute floor is ~31 ms and HF takes ~137 ms, ~4.5x off. The 31 ms is
theoretical peak and not reachable at this shape: the prefill GEMMs are
M=512, K=1536, only moderately sized, so realistic tensor-core efficiency is
50-70% of peak, maybe 50-60 ms for a good kernel. The rest of the gap to 137 ms
is Python dispatch and per-layer kernel-launch overhead, exactly what vLLM's
fused kernels and CUDA graphs cut into.

Caveat for the OOM sweep: `2 x params x tokens` counts only the linear layers and
ignores attention, which is fine at 512 context (attention is ~3% on top) but
grows as `seq^2` while the linear term grows as `seq`. Past a few thousand tokens
the linear-only estimate undercounts.

### Decode is memory-bound, and HF reaches ~15-20% of the bandwidth ceiling

Each decode step reads the full weight set from HBM to produce one token, so the
bandwidth floor is `weights / bandwidth`:

```
weights read per token = 2945 MiB = 3.09 GB
floor at 360 GB/s      = 3.09 / 360 = 8.6 ms/token  (~116 tok/s ideal)
measured 18 tok/s      = 55.6 ms/token  ->  55 GB/s achieved, ~15% MBU
measured 23 tok/s      = 43.1 ms/token  ->  72 GB/s achieved, ~20% MBU
```

Note this corrects an earlier 23% MBU estimate that wrongly used the inflated
4737 MiB weight figure as the bytes-moved denominator. With the real 3.09 GB, HF
decode is running at only ~15-20% of memory bandwidth. That ~5-7x headroom is the
heart of the baseline-vs-vLLM gap, and it is a memory-traffic and Python-overhead
problem, not a FLOPs problem: decode arithmetic intensity is ~1-2 FLOP/byte, so
no kernel can move it off the memory wall, only closer to it.

### The asymmetry, in one line

Prefill 3740 tok/s vs decode ~20 tok/s is a ~190x throughput gap on the same
model and card, because prefill amortizes one weight read across 512 tokens of
parallel work while decode pays a full weight read per single token.

## Measurement-quality notes

- Decode tok/s varied run to run (18.0 then 23.2 under identical config), ~30%
  spread, likely clock and thermal state. For the blog, report a median over
  several runs, not a single number.
- Prefill was stable (~137 ms both runs).
- Weights, peak allocated, and peak reserved were identical across runs, as they
  should be: they are shape-determined, not timing-dependent.

## What carries into the next tasks

- Clean VRAM budget: 2945 MiB weights, ~9.05 GB free on the 12 GB card for KV and
  activations. The OOM sweep can trust `12 GB - weights` with no mystery constant.
- Decode ~15-20% MBU is the control. vLLM decode tok/s over this is the headline
  comparison.
- Prefill ~23% of compute peak is the other control, and the gap there is
  launch/overhead, which is what CUDA graphs target.
