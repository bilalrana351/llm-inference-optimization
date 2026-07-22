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
| decode tokens/sec | 18.0-28.3 tok/s (six runs) | ~79.9 tok/s | ~2.8-4.4x |
| decode bandwidth use | 16-24% of 360 GB/s | ~68% | |
| VRAM | 3133 MiB peak, grown organically | 11237 MiB, reserved up front | not comparable |

Two numbers carry the whole comparison. Prefill barely moves, because both
engines hand the prompt's matrix multiplies to the same cuBLAS tensor-core
kernels, and you cannot beat cuBLAS on a raw GEMM. vLLM's ~10% edge there is
trimmed launch overhead, not a faster matmul. Decode is the opposite story: vLLM
is roughly 3-4x faster, and the bandwidth-use row says why. HF decode runs at only
16-24% of the card's memory bandwidth, while vLLM reaches ~68%. The headroom was
never compute; it was that decode is memory-bound and HF was leaving most of the
memory pipe idle. The next section is why.

(The HF spread is wide because the first two runs, on 2026-06-25, came in at 18.0
and 23.2 tok/s while the four later ones cluster tightly at 28.0 to 28.3. Take
~28 tok/s as the settled figure and the wider range as a reminder that a rented
box is not a fixed quantity, a point the profiling section returns to.)

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

![Measured VRAM vs context length on the RTX 3060: the measured line climbs at
~60 KB/token, rides well above the 28 KB/token analytical prediction, and the card
throws CUDA out of memory at 123,565 tokens.](../results/oom_curve.png)

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

## The 4-bit lever: memory, not speed

If the wall is memory, the obvious lever is to make the weights smaller. Loading
the same model in 4-bit (bitsandbytes NF4, fp16 compute) and running it through the
identical prefill/decode loop isolates exactly one change: how the weights are
stored. Same 512-token prompt, same 255 decoded tokens, batch 1.

| metric | fp16 | NF4 4-bit |
| --- | --- | --- |
| prefill | ~122 ms (4205 tok/s over the prompt) | ~140 ms (3655 tok/s) |
| decode tokens/sec | ~28.1 | ~12.4 |
| weights VRAM | 2945 MiB | 1099 MiB |

The asymmetry mirrors the HF-vs-vLLM split, and for the same reason. Prefill takes
only a ~13% hit, but decode is ~2.3x *slower*. The instinct is that 4-bit should
make decode faster, since decode is memory-bound and 4-bit weights are a quarter of
the bytes. It does not, because bitsandbytes does not run a native 4-bit matmul. It
reads the 4-bit weights, dequantizes them back to fp16, and runs the ordinary fp16
GEMM, so the matmul reads fp16-width data either way and there is no bandwidth
saving where it would count. The dequant is strictly extra work. In prefill that
work is amortized over 512 positions and hidden behind the compute, so NF4 nearly
keeps up. In decode, batch 1 and arithmetic intensity ~1, there is nothing to
amortize over and no compute to hide behind, so the dequant pass lands directly on
per-token latency and more than doubles it.

So NF4 is a memory lever, not a speed lever at this batch size. The payoff is the
weights row: 2945 to 1099 MiB, about 1.8 GiB freed. That ties straight back to the
OOM curve. At the observed ~60 KB/token growth rate, 1.8 GiB of headroom is on the
order of 30k more tokens of context before the same wall, bought by giving up decode
throughput. (The paired lesson to keep honest: a native 4-bit kernel that stayed in
low precision through the matmul, like the int4 kernels vLLM can use, would cut the
matmul's memory traffic and could flip decode the other way. The dequant-to-fp16
path is why bitsandbytes does not.)

## Why the two engines differ

The gap is a scheduling problem, not an arithmetic one. The GPU runs kernels; the
CPU, driving Python, launches them. In HuggingFace's decode loop every new token
re-enters Python and walks all 28 layers, firing hundreds of separate kernels,
each one preceded by Python interpreting the code and dispatching the op. At batch
1 the GPU work per kernel is a few microseconds, often less than the time Python
needs to launch the next one, so the GPU finishes and stalls, waiting on the CPU.

That paragraph was an argument. Here is the trace behind it. Every configuration
below is one RTX 3060 running the same model, prompt, and batch size, profiled
with the PyTorch profiler, and the full method is in `docs/profiling.md`.

| run | kernels | CPU launches | GPU busy | step | idle |
| --- | --- | --- | --- | --- | --- |
| HF | 1198 | 1198 | 15.31 ms | 40.65 ms | 62.3% |
| vLLM, no compile, no graphs | 356 | 356 | 13.02 ms | 25.22 ms | 48.4% |
| vLLM, compile only | 354 | 354 | 12.90 ms | 22.41 ms | 42.4% |
| vLLM, graphs only | 385 | 17 | 12.77 ms | 12.57 ms | ~0% |
| vLLM, both (shipping default) | 383 | 17 | 12.71 ms | 12.56 ms | ~0% |

![Device occupancy during the first 1500 microseconds of one batch-1 decode step,
one row per configuration. Without CUDA graphs the row is a sparse comb of a few
microseconds of work separated by long white gaps; with graphs it is a solid bar.
Both rows run the same kernels.](../results/decode_timeline.png)

HuggingFace fires 1198 kernels per token and the CPU launches every one of them
individually. The device works for 15.31 ms of a 40.65 ms step and idles for the
other 25.34 ms. That is the white space in the figure, and it is 62% of every
token spent waiting on Python.

The per-launch cost is not the problem; it is 21 microseconds, against an average
kernel that runs 13. The problem is paying it 1198 times. Batch-1 decode kernels
are matrix-vector products, a few microseconds each, so the dispatch chain above
a kernel costs more than the kernel.

vLLM attacks this twice, and the two attacks are worth separating because the
obvious story gets them backwards.

**Its kernels are fused by hand, before any compiler runs.** Comparing the two
engines with compilation and graphs off on both sides, kernel count falls from
1198 to 356. That 3.4x is vLLM's own CUDA kernels: fused RMSNorm, fused rotary
embedding, fused SiLU-and-multiply, paged FlashAttention. Device time falls too,
15.31 to 13.02 ms, but only by 15%. Fusion is mostly buying fewer launches, not
less memory traffic, because at batch 1 the activations it keeps in registers are
about 0.15% of the bytes moved. The weights are the other 99.85% and every engine
has to read all of them.

Notice that vLLM's cost per launch is *higher* than HF's, 34 microseconds against
21, because each vLLM step also runs scheduling, block-table management, slot
mapping, and sampling that a hand-written loop does not have. It still wins the
total, 12.20 ms of idle against 25.34 ms, because it pays more, far fewer times.

**CUDA graphs remove the CPU from the loop.** A graph is a recording of the
launches in order, instantiated once and replayed with one call per replay, so
the CPU stops arbitrating between kernels. Launches per step fall from 356 to 17,
and the idle gap goes to zero. This is where the win is.

What a graph does *not* do is change the work. The same kernels run, from the same
resident cubins, in the same order: with graphs off, 29 GEMV, 28 CUTLASS, 28
FlashAttention split-KV, 56 RMSNorm; with graphs on, exactly the same counts, plus
29 small parameter copies that the graph itself needs. Device busy time is flat
across all four vLLM rows, 12.71 to 13.02 ms, a 2.4% spread. Graphs convert idle
time into nothing at all and leave arithmetic untouched.

The clean test of that: turning graphs on should collapse the step to precisely
the device work already there. Predicted from the graphs-off busy time, 13.02 and
12.90 ms; measured with graphs on, 12.57 and 12.56 ms. Within 3.5% and 2.6%.

**And the two are not additive.** `torch.compile` is worth 2.81 ms per step with
graphs off, and 0.01 ms with them on. Once graphs have taken the CPU off the
critical path, making the CPU's per-launch work cheaper buys nothing. Both
optimizations attack the same term, so their gains overlap almost completely.
This is also why the ablation was worth running: comparing only the shipping
default against plain eager would have credited CUDA graphs with a fusion win, or
fusion with a graph win, with no way to tell which.

Adding it up, of the 28.09 ms per token that separates the two engines, 25.34 ms
(90%) is device idle removed and 2.60 ms (9%) is faster device work. The jump to
~68% bandwidth use is almost entirely the GPU no longer waiting.

Prefill cannot benefit the same way: it is one long parallel pass that already
keeps the GPU busy, so there is no idle gap for graphs to close, which is exactly
why prefill barely moved.

None of this is PagedAttention, which is a separate win and invisible at batch 1.
PagedAttention manages the KV cache like operating-system virtual memory, paging
it into fixed blocks so it never fragments and many sequences pack into one
reserved pool. That buys high-batch throughput and long context, not single-stream
decode speed. Its payoff shows up in the next section, where the same paged pool
is what lets vLLM keep going while plain transformers fragments and hits the wall.
The throughput here, ~4x at batch 1, is the floor of vLLM's advantage, not the
ceiling: continuous batching, which the batching sweep below measures directly, is
where the larger serving wins live. Those are what the later phases build.

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
wins are built on, which the next section starts to measure.

## Batching: the throughput the floor was hiding

Batch 1 wastes the GPU. The decode step reads the entire weight matrix to advance
one sequence by one token, arithmetic intensity ~1, so the tensor cores sit nearly
idle while the memory pipe does all the work. The obvious lever is to make that
same weight read serve more than one sequence: send many at once and let each
decode step advance all of them together. The method is one `generate` call with N
identical sequences, swept over growing N, on the same 512-token prompt and 256
output tokens as every run above.

The measured decode throughput climbs steeply and then simply stops:

| batch | 1 | 8 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| decode tok/s | 89 | 501 | 1499 | 2696 | 2879 | 2967 |
| per-token latency (ms) | 11 | 16 | 21 | 24 | 44 | 86 |

Throughput rises ~30x from batch 1 to 64, then gains almost nothing to batch 256
while latency climbs 3.6x. The sweet spot is ~64. The shape falls out of one
equation. A decode step costs a fixed part plus a per-sequence part:
`step_time = T_fixed + c1 x batch`, where `T_fixed` ~ 11 ms is the shared weight
read (paid once no matter the batch) and `c1` ~ 0.2 ms is what each added sequence
costs (its own KV read, its slice of the matmul). Throughput is
`batch / (T_fixed + c1 x batch)`. While the fixed weight read dominates, throughput
rises almost linearly, the free lunch of spreading one weight read over more
tokens. Once `c1 x batch` dominates, throughput flattens at `1 / c1`, because a
step whose time is proportional to the tokens in it has a pinned tokens-per-second.
Throughput ever rose only because there was a fixed cost to amortize; past ~64
there is nothing left to spread.

The surprise is what the plateau is *not*. At batch 256 the card is doing ~9
TFLOPS (about 16% of its ~51 TFLOPS peak) and moving ~90 GB/s (about 25% of its
360 GB/s). Neither roofline is saturated, yet throughput will not rise. The reason
is operation shape: batched decode is a skinny matmul (M = batch = 64 to 256),
which is occupancy-bound and tops out far below tensor-core peak, while the paged
KV reads run at a fraction of bandwidth. The plateau is the ceiling of these
particular kernels, not of the GPU. Which is exactly why prefill is different:
prefill processes 512 positions at once, so its matmuls are M = 512, fat and dense,
and reach ~68% of compute. Prefill is just batching over positions instead of over
sequences, and the fatter GEMM is why it hits the compute the decode plateau
cannot. The full decomposition and the roofline (theoretical ideal batch ~140 vs
measured ~64) are in `docs/batching-results.md`.

The knee is a property of the model, not the GPU, and a second card proves it. Run
the same sweep on an RTX 3090 (2.6x the bandwidth, 1.4x the compute) and every
throughput number scales up, but the knee does not move: latency stays flat through
batch 64 on both cards, then breaks upward at 96 on both. Decode throughput goes
from ~89 to ~210 tok/s at batch 1 and from ~2970 to ~10,100 at the plateau, so the
3090 is ~2.4x to 3.4x faster, yet the sweet spot is still ~64. That is the
weight-read-versus-KV-read crossover: it depends on the model's weight size and
per-sequence KV, which are identical on both cards. The plateau *height* scaled a
bit more than bandwidth alone (3.4x vs 2.6x) because the 3090's extra compute also
lifts the skinny-GEMM ceiling, but the plateau *location* is fixed by the model.

| batch | 3060 tok/s | 3090 tok/s | ratio | 3060 TPOT (ms) | 3090 TPOT (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 89 | 210 | 2.4x | 11.3 | 4.8 |
| 8 | 501 | 1393 | 2.8x | 16.0 | 5.7 |
| 32 | 1499 | 4845 | 3.2x | 21.4 | 6.6 |
| 64 | 2696 | 8127 | 3.0x | 23.7 | 7.9 |
| 96 | 2608 | 8583 | 3.3x | 36.8 | 11.2 |
| 256 | 2967 | 10112 | 3.4x | 86.3 | 25.3 |

The overlay makes the two claims one picture: the lines separate vertically (the
3090 delivers more throughput at every batch) but bend at the same place (the knee
does not slide right with the faster card). The 3090's line keeps climbing past 64
where the 3060's has gone flat, which is the extra compute lifting the plateau, not
the knee moving.

![Decode throughput vs batch size for three configs on one axis: 3090 realistic,
3060 realistic, and 3060 constrained. The two realistic lines both bend near batch
64 but the 3090 sits far above the 3060 and keeps rising to ~10,100 tok/s at batch
256 while the 3060 plateaus near ~2900. The constrained 3060 line pins at ~1000
tok/s once it crosses its KV wall at ~25 sequences.](../results/vllm_batch_throughput.png)

The other ceiling is the KV cache itself, and it looks nothing like the OOM run.
vLLM reserves its whole KV pool at startup, so device VRAM is flat the entire
sweep. The limit shows up not as memory climbing but as vLLM queuing sequences it
cannot fit and running them in later waves. Shrinking the pool on the 3060 so it
holds only 25 sequences makes this bite early: past batch 25 the `queued` flag
flips, decode throughput pins at ~1000 tok/s (a third of the unconstrained ~2700)
no matter how many more sequences you submit, and latency climbs as the queued
requests wait. So there are two different plateaus. One is a compute-efficiency
ceiling where every sequence still runs; the other is a KV-admission ceiling where
the surplus queues. The first is about how fast the kernels go, the second about how
many sequences fit, and telling them apart is the whole reason for running the pool
constrained as well as full.
