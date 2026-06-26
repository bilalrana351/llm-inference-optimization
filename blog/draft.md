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

<!-- The headline. Push context length up in steps, log peak VRAM at each step,
catch the CUDA OOM, and plot measured VRAM against the analytical KV-cache
prediction. Written the moment results/oom_curve.png exists. -->

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
