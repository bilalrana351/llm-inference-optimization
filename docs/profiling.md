# Profiling the decode step: where the HF-to-vLLM gap actually comes from

Phase 1 measured that vLLM decodes about 3x faster than HuggingFace `transformers`
at batch 1, and the blog explained the gap by reasoning: HF re-enters Python for
every kernel, the GPU finishes each tiny kernel before the CPU can hand it the
next, and vLLM removes the CPU from the loop with CUDA graphs. That was an
argument, not a measurement. This document is the measurement.

The headline: of the 28.09 ms per token that vLLM saves, **25.34 ms (90%) is
device idle time removed and 2.60 ms (9%) is faster device work.** The gap is a
scheduling problem, and now there is a number behind that sentence.

A second result was not predicted. `torch.compile` saves 2.81 ms per step when
CUDA graphs are off and 0.01 ms when they are on. The two optimizations are not
additive, because they attack the same term.

## What is being measured

Per decode step, three quantities:

| quantity | meaning |
| --- | --- |
| GPU kernels | how many kernels actually execute on the device |
| CPU launches | how many launch calls the CPU makes to cause them |
| idle gap | time inside the step with no device work running |

The pair that carries the CUDA-graph claim is the second and third. A graph does
not reduce the number of kernels. The same kernels run, from the same resident
cubins, in the same order. What collapses is the CPU side, from one launch per
kernel to one graph launch for the whole step, and the idle gaps close as a
consequence.

## Method

### Two passes, and why the gap is a subtraction across them

Every run is measured twice.

- **Clean pass.** No profiler anywhere. Times the decode steps and yields the wall
  time per step, `T`.
- **Profiled pass.** The same steps under `torch.profiler`. Yields the device
  kernel intervals, whose union is the busy time `B`.

The reported gap is `T - B`. Each term comes from the run where it is
trustworthy. CUPTI does not change what a kernel computes, so `B` survives
profiling largely intact, but wall time does not: the HF profiled step ran 1.73x
slower than the clean one. Using the profiled pass's own wall time would
therefore inflate the gap in exactly the direction the hypothesis predicts, which
is how this analysis manufactures its own conclusion.

Both passes cover the same sequence positions. A decode step gets slower as the
KV cache grows, so timing a 255-token window and subtracting the busy time of 11
steps near the start of the sequence would compare two different workloads.

### Stack capture is off, deliberately

`with_stack`, `record_shapes`, and `with_modules` are all disabled. Stack
unwinding costs CPU time per operation, and per-operation CPU time is the exact
quantity under test. Worse, its cost scales with the number of CPU-side
dispatches, so it would tax HF (1198 dispatches per step) far harder than vLLM
under graphs (17), inflating the very difference being reported.

### Segmenting a trace into steps

Three mechanisms, because the engines mark themselves differently.

- **HF** uses `record_function` annotations, which `scripts/profile_decode.py`
  wraps around each step.
- **vLLM** runs the model in a separate EngineCore child process, so annotations
  from the parent never reach it. All four vLLM runs segment on
  `_prepare_pos_seq_lens_kernel`, which fires exactly once per forward pass and
  keeps its name under inductor. Prefill is the first such pass and is dropped.
- Device events map to steps by **correlation id**, not by timestamp. A kernel
  runs after the launch that issued it, sometimes well after, so bucketing
  kernels by their own timestamps misattributes work at every step boundary.
  Under 0.3% of events needed the timestamp fallback.

Segmenting the graph runs on `cudaGraphLaunch` was tried and is wrong here. With
`cudagraph_mode: FULL_AND_PIECEWISE` the compiled engine replays 40 graphs across
12 forward passes, not one per step, so it reports roughly a third of a step as a
whole step. Using one marker for all four runs also keeps the methodology uniform
across the comparison.

### The resolution floor

`T` and `B` come from separate runs, so their difference carries a systematic
error of roughly 2%: CUPTI slightly inflates measured kernel durations, and the
two passes need not sit at identical clocks. This is irrelevant when the gap is
milliseconds and decisive when it is near zero, where it can drive the difference
below zero. Both CUDA-graph rows land at -0.20 and -0.15 ms. Those are not
negative idle times. They are zero, measured imprecisely, and they are reported
as "at the floor" rather than quoted as negative numbers.

### Hardware, and a caveat about "identical" boxes

One RTX 3060, 12GB, sm_86, fp16, batch 1, Qwen2.5-1.5B, 512-token prompt.
vLLM 0.25.1, torch 2.12.0+cu130.

The runs span two rented Vast.ai boxes with the same GPU model and the same
12,488,343,552 bytes of VRAM. They are not the same speed. The four vLLM runs
were repeated on both, and device busy time was uniformly higher on the second:

| run | box A busy | box B busy | ratio |
| --- | --- | --- | --- |
| eager | 10.61 | 13.02 | 1.227 |
| compile | 10.54 | 12.90 | 1.224 |
| graphs only | 10.55 | 12.77 | 1.210 |
| both | 10.47 | 12.71 | 1.214 |

Mean 1.219, spread 1.4%. Identical work taking uniformly 22% longer, with a
spread that tight, is a property of the machine and not noise. The HF trace was
re-run on box B before being compared against anything: predicted 15.6 ms busy
from the scaling, measured 15.31 ms.

**The cause is not memory clock, and the obvious explanation was wrong.** The
Vast.ai listing for box B advertises 287 GB/s, and 192 bits at 12 Gbps is exactly
288, which made a roughly 20% memory downclock look like a clean explanation:
360/287 is 1.25 against a measured 1.22. Measuring the card directly killed it.
NVML reports 7501 MHz over 192 bits, so 360 GB/s theoretical, stock clocks. The
listing's 287 turns out to be an achievable figure, within 1.5% of the 291.5 GB/s
this repo measures with a streaming read, not a theoretical peak.

So the 1.22x is real, reproducible, and still unexplained. Remaining candidates,
none of them tested: core clock or power cap (box B's host is an older Xeon
E5-2680 v4 platform), thermal state, a co-tenant on the card, or the software
stack, since HF kernel *selection* also differed between the boxes (different
`gemv2T_kernel_val` template parameters, different FlashAttention split-KV
traits, 280 elementwise calls replaced by 5 fused ones). That last one is the
most interesting candidate and the easiest to check, by pinning versions and
re-running.

The practical lesson stands regardless: a "verified" listing with the same GPU
name and the same VRAM does not pin performance, and Vast.ai's bandwidth column
is a measured number, not a spec.

**Every number in the results below is from box B.** On Vast.ai, "same GPU model"
does not mean "same performance", and mixing boxes inside one comparison table
silently invalidates it.

## Results

| run | compile | graphs | kernels | CPU launches | busy ms | step ms | idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HF | n/a | n/a | 1198 | 1198 | 15.31 | 40.65 | 62.3% |
| vllm-eager | no | no | 356 | 356 | 13.02 | 25.22 | 48.4% |
| vllm-compile | yes | no | 354 | 354 | 12.90 | 22.41 | 42.4% |
| vllm-graphonly | no | yes | 385 | 17 | 12.77 | 12.57 | ~0% |
| vllm-graph | yes | yes | 383 | 17 | 12.71 | 12.56 | ~0% |

Reading the same table as throughput and memory bandwidth. Each decode step reads
all 3.088 GB of fp16 weights plus about 15 MB of KV cache at sequence length 525,
so 3.103 GB moves per token.

Two denominators, because they say different things.
`scripts/measure_bandwidth.py` reports both for this box:

- **Theoretical, 360.0 GB/s.** Computed from what NVML says the card is actually
  set to, 7501 MHz doubled for GDDR6 over a 192-bit bus, not from the model name.
  This card is at stock clocks.
- **Achievable, 291.5 GB/s**, measured by a large streaming read. That is 81% of
  theoretical, which is the normal range. No kernel can exceed it, so it is the
  honest ceiling to measure an engine against.

| run | tok/s | GB/s over the step | % achievable | GB/s while busy | % achievable |
| --- | --- | --- | --- | --- | --- |
| HF | 24.6 | 76.3 | 26% | 202.7 | 70% |
| vllm-eager | 39.7 | 123.1 | 42% | 238.3 | 82% |
| vllm-compile | 44.6 | 138.5 | 48% | 240.6 | 83% |
| vllm-graphonly | 79.6 | 246.9 | 85% | 243.0 | 83% |
| vllm-graph | 79.6 | 247.1 | 85% | 244.2 | 84% |

Against theoretical instead, the "over the step" column reads 21%, 34%, 39%, 69%,
69%, which is the framing the blog draft used.

The last two columns are the point. While the device is actually working, every
engine sits at 70% to 84% of achievable bandwidth. The spread in the throughput
column is almost entirely the spread in how much of the step the device spends
working at all.

The other point is the one the ceiling makes visible. vLLM's decode step runs at
**85% of this card's achievable read bandwidth**, so it is close to the hardware
limit and there is not much left to win at batch 1. The remaining headroom is not
in scheduling, it is in arithmetic intensity: batching amortizes the same weight
read across more tokens, which is what `docs/batching-results.md` measures. That
is the natural next lever, and it is a different lever from anything in this
document.

### The 2x2, in step time (ms)

```
                graphs OFF   graphs ON   graphs save
compile OFF          25.22       12.57         12.65
compile ON           22.41       12.56          9.85
compile saves         2.81        0.01
```

### Two predictions, stated before the runs and confirmed

If CUDA graphs remove idle time and nothing else, then turning graphs on should
collapse the step to exactly the device work that was already there.

| prediction | predicted | measured | error |
| --- | --- | --- | --- |
| graphs-only step == eager busy | 13.02 | 12.57 | 3.5% |
| graphs step == compile busy | 12.90 | 12.56 | 2.6% |

Both hold. Busy time is also flat across all four vLLM configurations (12.71 to
13.02 ms, a 2.4% spread), so neither lever meaningfully changes device work.

## What the numbers mean

### 1. HF is launch-bound, and it is the launch count that does it

1198 kernels per decode step, 1198 CPU launches, one per kernel, no fusion and no
graphs. The device works for 15.31 ms out of a 40.65 ms step and idles for the
other 25.34 ms. That is 62.3% of every token spent waiting on Python.

The per-launch cost is not large. It is 21.1 microseconds of idle per launch,
against an average kernel that runs 12.8 microseconds. The problem is that this
is paid 1198 times per token. Batch-1 decode kernels are matrix-vector products,
a few microseconds each, so the dispatch chain above each kernel costs more than
the kernel it dispatches.

This also corrects a claim in the blog draft. HF is at 21% of peak bandwidth over
the whole step, not the 15-20% previously asserted, and it reaches 56% while
actually running.

### 2. vLLM's advantage before any compilation or graphs is the kernel count

HF against vLLM eager is the cleanest comparison of the two engines' kernels,
with compilation and graphs off on both sides. Kernel count falls 1198 to 356, a
3.4x reduction, and device time falls 15.31 to 13.02 ms.

That 3.4x is vLLM's hand-written CUDA kernels: fused RMSNorm, fused rotary
embedding, fused SiLU-and-multiply, paged FlashAttention. The fusion that matters
is already in the engine's kernels before any compiler runs.

Per-launch CPU cost moves the other way. vLLM eager pays 34.3 microseconds per
launch against HF's 21.1, because each vLLM step also runs scheduling, block
table management, slot mapping, and sampling that the hand-written HF loop does
not have. vLLM still wins the total, 12.20 ms of idle against 25.34 ms, because
it pays a higher cost far fewer times. Fewer, more expensive launches beat many
cheap ones.

### 3. torch.compile reduces the cost per launch, not the number of launches

This one contradicted the initial expectation. Inductor changes the kernel count
by two, 356 to 354, and busy time by 0.9%. Yet it cuts 2.81 ms off the step.

Two kernels cannot be worth 2.81 ms. The saving is visible in the per-launch
cost, which drops from 34.3 to 26.9 microseconds, a 22% reduction. Compilation is
removing Python and dispatcher work sitting above each launch, not removing
launches.

A useful consequence: idle gap is not a single global constant times launch
count. Cost per launch varies by engine and by configuration (21.1, 34.3, 26.9
microseconds), so fitting one line through gap against launch count across
engines would be wrong.

### 4. The two optimizations are not additive, and that is the most useful result

`torch.compile` saves 2.81 ms with graphs off and 0.01 ms with graphs on. Once
graphs have removed the CPU from the decode loop, making the CPU's per-launch
work cheaper is worth nothing, because that work is no longer on the critical
path.

This is what makes the ablation worth running rather than reasoning about. The
default configuration is 22.41 to 12.56 ms faster than compile-alone and 12.57 to
12.56 faster than graphs-alone, so measuring only default against eager would
have credited CUDA graphs with a fusion win, or the reverse, depending on which
comparison happened to be made. The single-lever cells are what separate them.

### 5. Where the 3.24x actually comes from

24.6 to 79.6 tok/s, a 3.24x speedup, 28.09 ms saved per token:

| contribution | ms | share |
| --- | --- | --- |
| device idle removed | 25.34 | 90% |
| faster device work | 2.60 | 9% |

Both mechanisms are real, and one is an order of magnitude larger than the other.

## An anomaly worth reporting: graph replay measures slightly faster kernels

Busy time falls from 13.02 ms (eager) to 12.77 ms (graphs only), about 1.4%. CUDA
graphs should not make kernels faster, so this was investigated rather than
rounded away.

Per-kernel comparison between the two runs shows every compute kernel present in
identical counts: 29 of the large GEMV, 28 CUTLASS, 28 FlashAttention split-KV,
56 RMSNorm, 28 reshape-and-cache, and so on. The 29 extra device events in the
graph run are all `memcpy32_post`, totalling 47 microseconds, which are the
graph's parameter copies. No compute kernel appears or disappears.

What changed is per-instance duration, and it splits cleanly:

| kernel | per-instance eager | graphs only | change |
| --- | --- | --- | --- |
| GEMV (largest) | 238.59 us | 236.62 us | -0.8% |
| CUTLASS wmma | 113.59 | 112.29 | -1.1% |
| flash split-KV | 16.21 | 14.92 | -7.9% |
| rotary embedding | 4.07 | 3.69 | -9.2% |
| RMSNorm | 3.90 | 3.51 | -10.1% |
| reshape-and-cache | 3.84 | 3.23 | -15.7% |

Kernels captured inside the graph, which run 28 or more times per step and are
therefore well sampled, are uniformly faster, and the smaller the kernel the
larger the relative gain. Kernels that run once per step and sit outside the
graph (sampling, block-table gather, slot mapping) show no such shift and move in
both directions.

The likely mechanism is back-to-back execution. In eager mode the device idles
for tens of microseconds between kernels, so each kernel restarts from a colder
state. Under graph replay the kernels run without gaps, keeping caches warm and
clocks boosted, which is worth proportionally more to a 3.8 microsecond kernel
than to a 238 microsecond one.

The effect is real in the sense that the in-graph and out-of-graph split is
consistent, but it is 1.4% of busy time, at the edge of what this method
resolves, and it does not affect any conclusion above. It is recorded here rather
than smoothed over.

## Limitations

- **Batch 1 only.** Activations are about 0.15% of the bytes moved at batch 1,
  which is why compiler fusion cannot help device time here. That changes as
  batch grows and activation traffic scales while weight traffic does not, so
  none of the fusion conclusions should be carried to a served workload without
  re-measuring. See `docs/batching-results.md`.
- **Short window.** 11 to 12 decode steps near sequence position 512 to 525. The
  idle fraction has not been checked deep into a long generation, where a larger
  KV cache makes attention kernels longer while the launch count stays fixed, so
  the idle fraction should fall.
- **Async scheduling is on.** vLLM 0.25.1 overlaps CPU scheduling with GPU
  execution, which is a third mechanism reducing idle gaps and is not isolated by
  this 2x2.
- **HF uses the legacy cache path.** The run emits the tuple-of-tuples
  `past_key_values` deprecation warning, so some of its per-launch CPU cost is
  cache-format conversion. It is consistent with every other HF number in this
  repo, but "how much of HF's gap is the legacy cache path" is untested.
- **The two-box split.** Absolute times here are box B. The 1.22x difference
  between two nominally identical 3060s is a reminder that cross-run comparisons
  need the hardware pinned.

## Reproducing

```bash
# HF baseline trace
python scripts/profile_decode.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512

# the four vLLM cells (Environment B)
python scripts/profile_vllm.py --label vllm-graph
python scripts/profile_vllm.py --label vllm-compile   --no-cudagraph
python scripts/profile_vllm.py --label vllm-eager     --enforce-eager
python scripts/profile_vllm.py --label vllm-graphonly --cudagraph-only

# analysis. Each profile script prints its own clean step time to pass back in.
rm -f results/profile_summary.csv
python scripts/analyze_trace.py \
    --trace hf=results/trace_hf_decode.json.gz --clean-step-ms hf=40.6543
python scripts/analyze_trace.py \
    --trace vllm-eager=results/trace_vllm-eager.json.gz \
    --trace vllm-graphonly=results/trace_vllm-graphonly.json.gz \
    --trace vllm-compile=results/trace_vllm-compile.json.gz \
    --trace vllm-graph=results/trace_vllm-graph.json.gz \
    --segment kernel --marker-kernel _prepare_pos_seq_lens_kernel --trim-first 1 \
    --clean-step-ms vllm-eager=25.22 --clean-step-ms vllm-graphonly=12.57 \
    --clean-step-ms vllm-compile=22.41 --clean-step-ms vllm-graph=12.56 \
    --figure results/decode_timeline.png
```

Artifacts: `results/trace_*.json.gz` (raw traces),
`results/profile_summary.csv` (the table above),
`results/decode_timeline.png` (device occupancy, first 1500 microseconds of one
step, one row per configuration).

Note that `--cudagraph-only` is not a supported combination on every vLLM build.
Confirm from the engine's init log that `cudagraph_mode` really is `FULL` and
compilation mode really is `0` before trusting that row: vLLM tends to warn and
downgrade rather than fail.
