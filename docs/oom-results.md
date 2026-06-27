# The KV-cache OOM experiment: measured numbers and what they mean

Task 3, the headline. We pushed the context length up one token at a time on the
HuggingFace path until the RTX 3060 ran out of memory, logging VRAM and decode
rate at every 1000-token checkpoint. The raw rows are in `results/oom_sweep.csv`,
the plot is `results/oom_curve.png`, and this file is the reading of them. The
result corrected our own prediction, which is noted honestly below.

## Setup

- GPU: RTX 3060 12GB (12,288 MiB total), Ampere, sm_86.
- Model: Qwen/Qwen2.5-1.5B, fp16, 28 layers, 12 query heads / 2 KV heads (GQA),
  head_dim 128. Weights resident 2945 MiB, the same clean fp16 floor as the
  baseline.
- Analytical KV per token (Phase 0 formula): `2 (K and V) x 28 layers x 2 KV
  heads x 128 head_dim x 2 bytes = 28,672 bytes = 28 KB/token`. The 2 KV heads,
  not the 12 query heads, are the GQA count that sizes the cache.
- Workload: a tiny 32-token prompt, then grow the context purely by decode, one
  token per step. Decode growth (not a long prompt) is deliberate: each decode
  step touches one position, so its transient activation is a small constant and
  the peak at any context length is weights + cache + a little, the cleanest
  match to the analytical line. The cost is wall time, the full sweep took about
  six hours of sequential decode.
- Why HF and not vLLM: HF grows the cache organically so the curve and the crash
  are real memory events. vLLM reserves its whole KV pool at startup, so it has
  no curve and never OOMs this way, it refuses new sequences instead. That
  contrast is the point, see `baseline-vllm-results.md`.

## What happened

The context grew from 32 to **123,565 tokens**, then `torch.cuda.OutOfMemoryError`
fired and the harness recorded the cliff and exited cleanly (the final row has
`oom=True`). The curve is smooth and monotonic. The instrument worked.

| quantity | value |
| --- | --- |
| OOM context length | 123,565 tokens |
| weights resident | 2945 MiB |
| measured peak allocated at OOM | 10,437 MiB |
| measured peak reserved at OOM | 11,764 MiB |
| card total | 12,288 MiB |
| analytical-KV slope (prediction) | 28 KB/token |
| measured peak-allocated slope | ~60 KB/token |

## The headline: measured memory is ~2x the analytical KV prediction

The measured line climbs at about **60 KB/token, roughly 2.15x the analytical 28
KB/token**, and sits well above the prediction the whole way. The gap is not
overhead or noise, it is a specific mechanism.

HuggingFace's decode loop reallocates the entire KV cache on every single token.
Each step runs `torch.cat([old_cache, new_token_kv])`, which cannot grow a tensor
in place: it allocates a fresh contiguous block for `n+1` tokens and copies the
old `n` in, so for the duration of the copy **both** the old and the new cache are
live. Peak allocated is therefore about `weights + 2 x KV`, not `weights + KV`.
The numbers confirm it at 120k tokens:

```
analytical KV       = 28,672 x 120,000  = 3.28 GiB
weights + 2 x KV    = 2945 + 2 x 3281   = 9507 MiB
measured allocated  =                     10,226 MiB   (the extra ~700 MiB is
                                                        attention scratch + context)
```

and the slope agrees: `d/dn (weights + 2 KV) = 2 x 28 KB = 56 KB/token`, against a
measured ~60 KB/token (the few extra KB is attention working memory that itself
grows with sequence length). So the measured curve is the `2 x KV` reallocation
model, not the textbook `1 x KV` size. This doubling is exactly what
PagedAttention removes: vLLM appends new tokens into pre-allocated fixed blocks,
no `cat`, no second copy. The vLLM run fit 237,376 tokens of real KV in 6.34 GiB;
the HF path effectively pays for each token twice.

## The reserved plateau and the actual OOM trigger

Peak reserved flattens at **11,732 MiB** from about 43k tokens onward and stays
pinned there, while allocated keeps climbing underneath it. The flat top is
PyTorch's caching allocator maxing out what it can take from the 12,288 MiB card,
the ~550 MiB above it is the CUDA context.

The crash itself is a fragmentation event, not a "free bytes hit zero" event. At
OOM, allocated was 10,437 MiB but reserved was 11,764 MiB: about **1.3 GiB sat
free inside the pool yet could not be handed out**, because the next `cat` needed
a single contiguous `2 x KV`-sized slab and the free space was scattered in gaps
between the cache blocks already parked. The parking-lot picture: enough empty
spaces in total for the bus, but no unbroken stretch long enough to fit it, so the
allocation fails. This is why the wall arrives when a token needs `2 x KV`
contiguous, earlier than raw free memory would suggest.

## Correcting our own prediction

Before the run we guessed "naive math says ~280k tokens, the card dies near 30k."
The card actually reached **~123k**. The mechanism we named was right, the numbers
were pessimistic. The naive 280k headroom is roughly halved by the `2 x KV`
reallocation (280k to ~140k), and fragmentation eats the last stretch (~140k to
123k). Keeping the wrong guess next to the measured answer is the honest version
of the story and a better one: the gap between prediction and reality has two
named, quantified causes.

## Decode slows as context grows: 30 to 3 tok/s

The per-segment decode rate falls from ~30 tok/s at the start to ~3 tok/s near the
cliff, about 10x. Decode is memory-bound, so the rate tracks bytes moved per step,
and three of those costs grow with context:

1. Weights: ~3 GB, fixed every step. This alone sets the early flat ~30 tok/s,
   when the cache is too small to matter.
2. Reading the KV cache for attention: the new query attends to all past keys and
   values, so the whole `28 KB x n` cache is read each step. Linear in context.
3. The `cat` copy: every step rewrites the entire cache, reading `n` and writing
   `n+1` tokens, about `2 x 28 KB x n` of traffic. The largest context-scaling
   term, and the same `cat` that drives the OOM.

Past ~10k tokens, terms 2 and 3 overtake the fixed weight read, and from there
each step moves more bytes the longer the context, so tok/s decays roughly as
1/context. At 120k the cache (~3.3 GB) is larger than the weights and the per-step
`cat` alone moves ~6 GB, so a step costs ~10x its starting traffic, which is the
30 to 3 tok/s seen. The same reallocation is both the memory wall and the speed
decay.

## What carries into the next phases

- The single artifact is `results/oom_curve.png`: measured peak reserved against
  context, with the analytical KV line, the 12 GB ceiling, and the OOM marker.
- Two HF-specific costs, both fixed by paging the cache, account for the whole
  gap between theory and measurement: the `2 x KV` copy-on-grow from `torch.cat`,
  and fragmentation of the ever-reallocated blocks. PagedAttention is the direct
  answer to both, which is why this experiment is the natural lead-in to the
  paged-cache phase.
- The decode slowdown with context is the same `cat` traffic, and is the
  motivation for in-place, block-structured KV storage rather than a monolithic
  growing tensor.
