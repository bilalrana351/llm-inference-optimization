# Watching the KV cache hit the wall

Phase 1 of an LLM inference optimization roadmap. Draft, written as the
measurements come in.

## The setup

One small model (Qwen2.5-1.5B, fp16), one GPU (a single RTX 3060, 12GB, Ampere
sm_86), run two ways: plain HuggingFace `transformers` as a readable control,
then vLLM as the optimized engine. Same model, same prompt (512 tokens), same 256
new tokens, greedy decoding, batch size 1. The only thing that changes between
runs is the engine, so the gap between them is attributable to the engine and
nothing else.

Everything is timed by one shared harness with a few non-negotiable rules:
prefill and decode are measured separately, the GPU is synchronized before every
timer read, a warmup run is discarded, and VRAM is logged with the peak captured.
The small card is on purpose: at 12GB the KV cache hits the wall at a realistic
context length, which is the headline experiment, not something to engineer
around.

## HuggingFace vs vLLM

| metric | HF transformers | vLLM | gap |
| --- | --- | --- | --- |
| prefill latency | ~137 ms (3740 tok/s over the prompt) | ~124 ms (4114 tok/s) | ~1.1x |
| decode tokens/sec | ~18-23 tok/s | ~79.9 tok/s | ~3.5-4x |
| decode bandwidth use | ~15-20% of 360 GB/s | ~68% | |
| VRAM | 3133 MiB peak, grown organically | 11237 MiB, reserved up front | not comparable |

Two numbers carry the whole comparison. Prefill barely moves, because both
engines hand the prompt's matrix multiplies to the same cuBLAS tensor-core
kernels, and you cannot beat cuBLAS on a raw GEMM. vLLM's ~10% edge there is
trimmed launch overhead, not a faster matmul. Decode is the opposite story: vLLM
is roughly 4x faster, and the bandwidth-use row says why. HF decode runs at only
~15-20% of the card's memory bandwidth, while vLLM reaches ~68%. The headroom was
never compute; it was that decode is memory-bound and HF was leaving most of the
memory pipe idle. The next section is why.

The VRAM column is the one trap. HF's figure is what generation organically grew
into; vLLM's 11237 MiB is a pool it reserves at startup (weights 3.0 GiB, KV pool
6.3 GiB, CUDA-graph buffers 0.4 GiB, plus context and headroom), sized to 90% of
the card. The two are not the same measurement, so the fair comparison is decode
rate and prefill latency, not peak bytes.

## The OOM curve

The headline experiment: a 32-token prompt, then grow the context one decoded
token at a time and log VRAM every 1000 tokens until the card dies. Decode growth
is deliberate, each step touches one position so its transient is a small
constant, and the peak at any length is weights plus cache plus a little, the
cleanest match to the Phase 0 KV-cache formula (28 KB/token for this model). It
ran on HuggingFace, not vLLM, because HF grows the cache organically so the curve
and the crash are real memory events; vLLM reserves its whole pool at startup and
would have no curve to plot.

The card reached 123,565 tokens of context, then threw `CUDA out of memory`. Two
things in the plot tell the story. First, the measured line climbs at ~60
KB/token, more than double the analytical 28 KB/token, and rides well above the
prediction the whole way. That gap is not overhead, it is a mechanism:
HuggingFace reallocates the entire KV cache every single decode step. Each token
does `torch.cat([old_cache, new_kv])`, which cannot grow a tensor in place, it
allocates a fresh block for n+1 tokens and copies the old n in, so for that copy
both caches are live and peak memory is weights + 2x KV, not weights + KV. The
numbers land on it: at 120k tokens, weights + 2x KV predicts 9507 MiB and the
card held 10,226. The textbook size was right; the naive size was just being paid
twice.

Second, the crash is a fragmentation event, not a "free bytes hit zero" event. At
OOM, 10,437 MiB was allocated but 11,764 was reserved, so ~1.3 GiB sat free inside
the pool yet could not be handed out: the next `cat` needed one contiguous
2x-KV-sized slab and the free space was scattered in gaps between cache blocks
already parked. Enough empty parking spaces in total for the bus, no single
stretch long enough to fit it. So the wall arrives when a token needs a big
contiguous block, earlier than raw free memory suggests.

Worth stating plainly: my pre-run guess was that the card would die near 30k
tokens. It reached 123k. The mechanism I named was right and the arithmetic was
pessimistic, the naive headroom is roughly halved by the 2x reallocation and the
last stretch is eaten by fragmentation. Keeping the wrong guess next to the
measured answer is the honest version, and the better one, because the gap between
prediction and reality has two named, quantified causes.

The same `torch.cat` also explains why decode slows from ~30 tok/s to ~3 tok/s as
context grows. Decode is memory-bound, and rewriting the whole cache every step is
O(context) memory traffic, so at 120k a single step moves ~6 GB just to copy the
cache, on top of the weight read. The reallocation is both the memory wall and the
speed decay. Both are exactly what paging the cache into fixed blocks removes,
which is where the next phase goes.

## Why the two engines differ

The gap is a scheduling problem, not an arithmetic one. The GPU runs kernels; the
CPU, driving Python, launches them. In HuggingFace's decode loop every new token
re-enters Python and walks all 28 layers, firing hundreds of separate kernels,
each one preceded by Python interpreting the code and dispatching the op. At batch
1 the GPU work per kernel is a few microseconds, often less than the time Python
needs to launch the next one, so the GPU finishes and stalls, waiting on the CPU.
That is what ~15-20% bandwidth use means: the memory pipe is half-empty because
the bottleneck is the interpreter, not the hardware.

vLLM takes the CPU out of that loop. It captures the whole decode step into a CUDA
graph, a recording of every kernel launch in order, and replays it with a single
call, and it fuses many of those kernels into fewer, bigger ones. Now one launch
hands the GPU the entire step and it streams weights back to back with no Python
in between, which is the jump to ~68% bandwidth use. Prefill cannot benefit the
same way: it is one long parallel pass that already keeps the GPU busy, so there
is no idle gap for graphs to close, which is exactly why prefill barely moved.

None of this is PagedAttention, which is a separate win and invisible at batch 1.
PagedAttention manages the KV cache like operating-system virtual memory, paging
it into fixed blocks so it never fragments and many sequences pack into one
reserved pool. That buys high-batch throughput and long context, not single-stream
decode speed. Its payoff shows up in the next section, where the same paged pool
is what lets vLLM keep going while plain transformers fragments and hits the wall.
The throughput here, ~4x at batch 1, is the floor of vLLM's advantage, not the
ceiling: continuous batching, which this benchmark deliberately does not exercise,
is where the larger serving wins live. Those are what the later phases build.

## What the PagedAttention reading names

Skimming the vLLM docs and the PagedAttention paper, the useful part was that it
puts names on exactly what the OOM run measured. The pre-paging approach gave each
sequence one contiguous KV buffer sized to the longest length it might reach, and
the paper splits the resulting waste three ways: reservation (space held for tokens
not yet generated), internal fragmentation (over-allocation beyond even the max,
the unused tail of the buffer), and external fragmentation (the contiguous holes
left by alloc and free, which is the crash I watched, OOM with free bytes still in
the pool because no single gap was big enough). PagedAttention slices the cache
into fixed `block_size`-token blocks and maps each sequence's logical positions to
physical blocks through a block table, the direct analogue of an OS page table.
Blocks need not be contiguous with each other, only the KV inside one block is, so
external fragmentation goes away (any freed block fits any future need) and internal
fragmentation shrinks to at most the last partial block. The same indirection
enables prefix sharing: two sequences with an identical prefix point their leading
blocks at the same physical frame, and copy only when they diverge (copy-on-write),
the same trick the OS uses for shared pages across processes. None of this is in
the batch-1 numbers above, it is the machinery the high-batch and long-context
wins are built on, which is where the next phase goes.
