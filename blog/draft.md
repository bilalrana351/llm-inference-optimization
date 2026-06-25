# Watching the KV cache hit the wall

Phase 1 of an LLM inference optimization roadmap. Draft, written as the
measurements come in.

## The setup

One small model (Qwen2.5-1.5B, fp16), one GPU (a single Colab T4, 16GB), run two
ways: plain HuggingFace `transformers` as a readable control, then vLLM as the
optimized engine. Same model, same prompt, same `max_new_tokens`, greedy
decoding, batch size 1. The only thing that changes between runs is the engine,
so the gap between them is attributable to the engine and nothing else.

Everything is timed by one shared harness with a few non-negotiable rules:
prefill and decode are measured separately, `torch.cuda.synchronize()` is called
before every timer read, a warmup run is discarded, and VRAM is tracked as both
allocated and reserved with the peak reset before each measured run.

## HuggingFace vs vLLM

<!-- TODO: fill in once both runs exist on the T4. -->

| metric | HF transformers | vLLM |
| --- | --- | --- |
| decode tokens/sec | _tbd_ | _tbd_ |
| prefill latency | _tbd_ | _tbd_ |
| peak VRAM | _tbd_ | _tbd_ (note: vLLM pre-allocates its KV pool) |

## The OOM curve

<!-- The headline. Push context length up in steps, log peak VRAM at each step,
catch the CUDA OOM, and plot measured VRAM against the analytical KV-cache
prediction. Written the moment results/oom_curve.png exists. -->

## Why the two engines differ

<!-- One paragraph on where the gap comes from: PagedAttention, continuous
batching, Python out of the decode hot loop. This is what the later phases
build. The wins are memory traffic and fusion, not faster FLOPs. -->
