# Phase 2 gate: diagnosing a kernel from first principles

The Phase 2 gate is two questions, and the deliverable is answering them in
numbers rather than adjectives.

1. Take a kernel you did not write, from a source you can read, and account for
   its memory traffic, its coalescing, and its occupancy.
2. Do the same for a kernel out of your own decode trace, where nobody has
   handed you the source or the answer.

**Part 2 is written. Part 1 is a scaffold with the numbers not filled in.**
Passing on someone else's kernel is the bar and passing on your own decode path
is the proof, so the file stays visibly incomplete rather than quietly reordered.

The two parts are a matched pair, and the difference between them is the point:

| leg | Part 1, tiled matmul | Part 2, cuBLAS GEMV |
| --- | --- | --- |
| source | readable | closed |
| bytes and intensity | derivable exactly | derived exactly |
| coalescing | derivable exactly from the indexing | inferred from 96% of achieved bandwidth |
| occupancy | derivable exactly from the tile size | open, needs `ncu` |

Part 1 is the control. It establishes that all three legs can be done when
nothing is hidden, which is what licenses the inference in Part 2's coalescing
section and the flagged gap in its occupancy section. Without Part 1 a reader
cannot tell whether those two legs are soft because cuBLAS is closed or soft
because the analysis was not done.

## Part 1: naive against tiled matmul, from GPU MODE lecture 5

**Status: not written.** Everything below is a form to fill in. Every `TODO` is
a number that has to be derived, and any leg that still needs hand-waving when
the rest is done is the leg to go back and fix rather than write around.

Notation for this part: square matmul `C = A * B` with `A`, `B`, `C` all
`N x N` in fp32, one output element per thread, tile size `T` (so a block is
`T x T` threads and shared memory holds a `T x T` tile of `A` and of `B`).

### 1.1 Bytes per output element, and arithmetic intensity

State each answer symbolically first, then substitute numbers. A formula that
reduces to the right number is a derivation; a number on its own is a memory.

| quantity | naive | tiled, tile size `T` |
| --- | --- | --- |
| global reads of A per output element | TODO | TODO |
| global reads of B per output element | TODO | TODO |
| total global bytes per output element | TODO | TODO |
| FLOPs per output element | TODO | TODO |
| arithmetic intensity, FLOP/byte | TODO | TODO |
| ratio, tiled over naive | | TODO |

Then the two questions that turn the algebra into a claim about this card:

- Where does each land relative to the sm_86 ridge point, `peak FLOP/s` over
  `291.5 GB/s` achievable? TODO.
- What value of `T` would be needed to move the kernel from memory-bound to
  compute-bound, and is that `T` reachable given the limits in 1.3? TODO.

Note that the ridge point depends on a peak FLOP/s figure this repo has not
measured. Same caveat as Part 2, and the same fix: measure it rather than quote
it.

### 1.2 Coalescing, at the granularity of one warp and one instruction

For each of the four loads below, answer the same three things: what all 32
lanes of a single warp touch when that one instruction issues, how many 32-byte
sectors that becomes, and therefore whether it coalesces.

| load | what one warp touches | sectors per request | coalesced? |
| --- | --- | --- | --- |
| naive, A | TODO | TODO | TODO |
| naive, B | TODO | TODO | TODO |
| tiled, A tile into shared | TODO | TODO | TODO |
| tiled, B tile into shared | TODO | TODO | TODO |

Two things this table must not skip.

**Name the index mapping.** "Naive matmul is uncoalesced" is only true for one
of the two ways of assigning `threadIdx.x`. If `threadIdx.x` maps to the column
of C then one of the two loads is contiguous across the warp and the other is a
single broadcast address; swap the mapping and the same source becomes a 32-way
scatter. State which mapping you are analysing, then state what the other one
would do. TODO.

**Shared memory is a separate question from global memory.** Once the tiles are
resident, the reads out of shared memory have their own access pattern, and the
failure mode there is bank conflicts rather than uncoalesced sectors. For your
tile size and layout: how many of the 32 banks does one warp hit, and is there a
conflict? TODO. If yes, what padding removes it? TODO.

### 1.3 Occupancy on sm_86

Fill the hardware limits first, from the device rather than from memory:

| sm_86 limit | value | source |
| --- | --- | --- |
| SMs on this 3060 | TODO | `multi_processor_count` |
| max threads per SM | TODO | `max_threads_per_multi_processor` |
| max warps per SM | TODO | threads per SM over 32 |
| max blocks (thread blocks) per SM | TODO | CUDA C Programming Guide, table of compute capabilities |
| max threads per block | TODO | `max_threads_per_block` |
| shared memory per SM | TODO | `shared_memory_per_multiprocessor` |
| max shared memory per block | TODO | `shared_memory_per_block` |
| 32-bit registers per SM | TODO | `regs_per_multiprocessor` |

Most of those come straight off the device:

```python
import torch
p = torch.cuda.get_device_properties(0)
for k in ("name", "multi_processor_count", "max_threads_per_multi_processor",
          "shared_memory_per_multiprocessor", "shared_memory_per_block",
          "regs_per_multiprocessor", "warp_size", "major", "minor"):
    print(f"{k:36s} {getattr(p, k, 'n/a')}")
```

Do not fill these from memory. A spec sheet number already sent one section of
`docs/profiling.md` down a wrong explanation, and `scripts/measure_bandwidth.py`
exists because of it.

Then the occupancy itself, for at least two tile sizes so the tension in 1.1 is
visible rather than asserted:

| | `T = 16` | `T = 32` |
| --- | --- | --- |
| threads per block | TODO | TODO |
| warps per block | TODO | TODO |
| shared memory per block (both tiles, fp32) | TODO | TODO |
| blocks per SM limited by threads | TODO | TODO |
| blocks per SM limited by shared memory | TODO | TODO |
| blocks per SM limited by the block cap | TODO | TODO |
| blocks per SM limited by registers, at R registers per thread | TODO | TODO |
| **binding limit** | TODO | TODO |
| resident warps, and occupancy as a percentage | TODO | TODO |
| arithmetic intensity from 1.1 | TODO | TODO |

The conclusion this table exists to support: larger tiles raise arithmetic
intensity linearly in `T` while also raising threads and shared memory per
block, which can cut residency. State which tile size you would pick for this
card and why, in terms of which of the two effects dominates. TODO.

Registers per thread is the one quantity here that is a property of the compiled
kernel rather than of the tile size, so either state it as a parameter `R` and
show at which `R` it becomes the binding limit, or get it from
`nvcc -Xptxas -v` on the lecture 5 source. The second is better and costs one
command.

## Part 2: the GEMV out of the HF decode trace

Part 2's tables are regenerated by:

```bash
python scripts/roofline_step.py --trace results/trace_hf_decode.json.gz \
    --summary results/profile_summary.csv --csv results/roofline_step.csv --strict
```

Numbers, not prose, are the artifact: `results/roofline_step.csv`. The kernel
listing they are derived from is `results/hf_decode_step_kernels.txt`, produced
by `scripts/dump_step_kernels.py`.

### The kernel, and why this one

`gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 4, 4,
false, false, cublasGemvParams<...>>`, from cuBLAS. It is **46.2% of the device
time in one HuggingFace decode step**, which makes it the only kernel in the
trace whose analysis changes the story.

It is picked by cuBLAS, not by PyTorch and not by HuggingFace. `nn.Linear` calls
`aten::mm`, which calls cuBLAS, which looks at the operand shapes, sees that the
second operand is a single vector, and dispatches a GEMV routine rather than a
GEMM. `2T` is the transposed variant: `nn.Linear` stores its weight as
`(out_features, in_features)` and the product needs it the other way around.

**Identifying which projections it serves is done structurally, not by name.**
The kernel symbol does not say `gate_proj`. Three independent facts pin it:

- The 1198 kernels of a decode step are a 42-kernel block repeated 28 times,
  once per decoder layer, at a 100% periodic match, plus a 12-kernel head and an
  11-kernel tail. Inside a block the matmuls appear in architecture order.
- Qwen2 puts a bias on q, k and v only, so those three arrive as `aten::addmm`
  and the other four as `aten::mm`. The trace agrees.
- Every projection's measured duration has to be at least its own bytes divided
  by achievable bandwidth. That is a hard floor, and `--strict` fails the run if
  any assignment violates it.

The kernel serves `gate_proj` and `up_proj` in every layer (56 instances), and
`lm_head` once (1 instance). A different instantiation of the same template,
differing only in a bool, serves `q_proj`.

**The 57 instances are not alike, and the average lies about all of them.** 56
run at about 98 us and one runs at 1573.93 us. The 124 us mean describes
neither. This was caught by predicting `lm_head` at 1601 us from its size and
then looking for a kernel that took that long, which is the argument for
predicting first and aggregating second.

### Bytes per output element, and arithmetic intensity

For `y = Wx` with `W` of shape `(n_out, n_in)` in fp16 and `x` a single vector:

| quantity | per output element | whole kernel, gate_proj |
| --- | --- | --- |
| weight bytes read | `2 * n_in` = 3072 B | 27.53 MB |
| activation bytes read | `2 * n_in / n_out`, amortized, 0.34 B | 3.07 KB |
| bytes written | 2 B | 17.9 KB |
| FLOPs | `2 * n_in` = 3072 | 27.5 MFLOP |

**Arithmetic intensity is 1.0 FLOP per byte.** Two FLOPs (one multiply, one add)
per two bytes of weight. Every weight element is read from DRAM, used once, and
never touched again. There is no reuse to exploit, so no tiling, no shared
memory strategy, and no better kernel can change the byte count. This is the
structural difference from prefill, where the same weight byte serves all B
tokens in the batch and intensity is `1.0 * B`.

The activation vector is negligible: 1536 elements, 3 KB, read once per kernel
and resident in cache throughout. At batch 1 the model reads 3.087 GB of weights
per token and about 15 MB of KV cache, so weight traffic is 99.5% of the bytes.

Against this card's roofline, 1.0 FLOP/byte sits far on the memory side. With
291.5 GB/s measured achievable and a nominal 12.7 TFLOP/s of non-tensor-core
fp16 throughput, the ridge point is near 44 FLOP/byte, so a batch-1 GEMV is
roughly 44x away from being compute-bound. **The FLOP/s figure is a spec sheet
number, not one this repo has measured.** The 360 GB/s bandwidth assumption was
also a spec sheet number until `scripts/measure_bandwidth.py` contradicted the
story built on it, so the ridge point should be treated as provisional until a
matching FLOP/s probe exists.

### Predicted against measured, every matmul in the step

Predicted is weight bytes divided by 291.5 GB/s, the achievable read bandwidth
measured on this box. Achievable, not the 360 GB/s theoretical: no kernel can
beat a streaming read, so theoretical peak would understate every row.

| projection | shape | MB | predicted | measured | achieved GB/s | % of ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| q_proj | 1536x1536 | 4.72 | 16.2 us | 20.16 us | 234.1 | 80% |
| k_proj | 1536x256 | 0.79 | 2.7 us | 8.16 us | 96.4 | 33% |
| v_proj | 1536x256 | 0.79 | 2.7 us | 7.90 us | 99.5 | 34% |
| o_proj | 1536x1536 | 4.72 | 16.2 us | 21.52 us | 219.3 | 75% |
| gate_proj | 1536x8960 | 27.53 | 94.4 us | 97.84 us | 281.3 | 97% |
| up_proj | 1536x8960 | 27.53 | 94.4 us | 98.13 us | 280.5 | 96% |
| down_proj | 8960x1536 | 27.53 | 94.4 us | 113.49 us | 242.5 | 83% |
| lm_head | 1536x151936 | 466.75 | 1601.2 us | 1573.93 us | 296.5 | 102% |

One division predicts eight kernels spanning a 590x range in size, and the two
largest land within 4% and 2%. That is the memory-bound claim in falsifiable
form: had the ratios come out scattered, the claim would be dead.

`lm_head` at 102% is above the ceiling. It is not beyond the hardware, it is at
it: the ceiling and the duration each carry roughly 2% of measurement error, and
a 467 MB contiguous read is the most favourable access pattern the memory
system will ever see, so it can edge past a microbenchmark that was not tuned
for this exact size. Reporting it as 102% rather than clamping it to 100% is the
honest form.

#### Why the deviations, in order of size

**k_proj and v_proj, 33%, and this is an occupancy result.** These read 0.79 MB
and should take 2.7 us. They take 8 us. Two reasons, both about having too
little work.

Saturating DRAM needs many memory requests in flight simultaneously so that
latency overlaps. That requires enough concurrent warps, which requires enough
blocks. Grouped-query attention gives Qwen2.5-1.5B 2 KV heads against 12 query
heads, so `k_proj` and `v_proj` produce only 256 outputs each. There is not
enough output to spread across the chip.

Reading the template parameters as a 128-thread block covering 16 output
columns, the grid sizes follow:

| projection | outputs | blocks at 16/block | blocks per SM (28 SMs) |
| --- | --- | --- | --- |
| k_proj, v_proj | 256 | 16 | **0.6, so 12 SMs get nothing** |
| q_proj, o_proj | 1536 | 96 | 3.4 |
| gate_proj, up_proj | 8960 | 560 | 20.0 |
| lm_head | 151936 | 9496 | 339.1 |

That single column predicts the whole shape of the results table. Fewer blocks
than SMs means the card is idle by construction, and 16 blocks over 28 SMs
cannot exceed 57% utilisation no matter how good the kernel is. It also predicts
that the fix is not a better kernel but a different decomposition, splitting the
reduction dimension to manufacture parallelism.

**This interpretation of `128, 16` is a hypothesis, not a read fact.** cuBLAS is
closed source. What makes it worth stating is that it is testable in one command
on the box: `ncu --metrics launch__grid_size,launch__block_size` reports the
real grid, and if `k_proj` does not launch 16 blocks the explanation above is
wrong and needs replacing. Recorded as open, below.

**down_proj, 83%, and the shape is the whole reason.** It reads exactly the same
27.53 MB as `gate_proj` and takes 15.7 us longer, and it is not even the same
kernel: the trace shows `cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>`
followed by a second kernel, `splitKreduce_kernel`, 3.52 us, 28 times per step.

`gate_proj` is 1536 in and 8960 out: short dot products, many of them, plenty of
independent work. `down_proj` is 8960 in and 1536 out: long dot products, few of
them. With only 1536 outputs, cuBLAS cannot fill the device by parallelising
over outputs alone, so it splits each 8960-long reduction into chunks, computes
partial sums concurrently, and runs a second kernel to add them. Split-K buys
parallelism by paying extra memory traffic, since the partial sums are written
to DRAM and read back.

Identical bytes, 16 us apart, entirely because of which dimension was long. It
is the cleanest demonstration in the trace that on a memory-bound workload the
decomposition, not the arithmetic, sets the time.

**q_proj and o_proj, 75 to 80%.** The same effect as k and v, milder. 96 blocks
over 28 SMs is 3.4 waves, enough to fill the machine but with a ragged tail.

**gate_proj and up_proj, 96 to 97%.** Large enough that per-launch costs vanish,
with 560 blocks giving 20 full waves. A long contiguous streaming read, which is
the access pattern GPUs are built for.

### Coalescing: what one warp touches

`nn.Linear` stores its weight row-major as `(out_features, in_features)`, so the
`n_in` coefficients belonging to a single output are contiguous in memory. Two
ways to assign that to threads:

- **One thread per output.** Each thread walks its own row. At any instruction
  the 32 lanes of a warp are touching 32 addresses `2 * n_in` bytes apart, which
  for `gate_proj` is 3072 bytes. Every lane lands in a different 32-byte sector,
  so one instruction becomes 32 memory transactions and effective bandwidth
  collapses to a fraction of peak.
- **A group of threads per output, cooperating.** Lanes read consecutive
  elements of the same row, so a warp's 32 lanes cover 64 contiguous bytes in
  fp16 and the request coalesces into two 32-byte sectors. The partial products
  then need a reduction across lanes, done with warp shuffles.

**The measurement settles which one cuBLAS uses.** `gate_proj` achieves 281.3
GB/s, 96% of the 291.5 GB/s a pure streaming read achieves on this card. The
first scheme cannot produce that number: a 32-way scatter would land in the tens
of GB/s. So the weight loads coalesce, and the kernel is reading the weight
matrix as close to sequentially as the hardware allows.

Note that the argument runs from bandwidth back to access pattern. It is sound
as far as it goes, because 96% of achievable leaves no room for a scattered
access pattern, but it does not distinguish the second scheme from other
coalescing layouts, and it does not measure sector efficiency directly. The
direct evidence is `ncu --metrics
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`,
where 2.0 sectors per request would confirm the fp16 pair-wise coalescing above
and 32.0 would refute the whole paragraph.

The activation vector is the other operand and it does not need coalescing at
all. All 1536 elements, 3 KB, are read by every block, so after the first block
touches them they sit in L2 and are effectively broadcast.

### Occupancy on sm_86

Three quantities are needed: shared memory per block, registers per thread, and
the resulting blocks per SM. **Two of the three cannot be obtained from a
Chrome trace, and this section is where the analysis is currently incomplete.**

What the trace does give:

- Block size 128 threads, if the template parameter reads as claimed, so 4 warps
  per block against the sm_86 limit of 48 warps and 16 blocks resident per SM.
- Grid sizes as tabulated above, which is the part that actually explains the
  results, because for `k_proj` the limiter is not occupancy per SM but having
  fewer blocks in the entire grid than the GPU has SMs.

What it does not give: registers per thread and shared memory per block are
properties of the compiled cubin, not of the timeline. For a closed-source
cuBLAS kernel the honest answers come from `ncu`:

```bash
ncu --metrics \
  launch__grid_size,launch__block_size,launch__registers_per_thread,\
launch__shared_mem_per_block_static,sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name-base mangled --kernel-name regex:gemv2T \
  python scripts/baseline_hf.py --model Qwen/Qwen2.5-1.5B --new-tokens 4
```

Writing an occupancy calculation without those two numbers would be exactly the
hand-waving the gate is meant to catch, so it is recorded as open rather than
filled in with plausible values.

There is also a reason to expect occupancy to be the wrong question for this
kernel. `gate_proj` runs at 96% of achievable bandwidth. Whatever its occupancy
is, it is already sufficient, since the memory system is saturated and raising
warp count cannot make DRAM faster. Occupancy matters here only in the negative
case, `k_proj`, where the grid is too small to occupy the machine at all. That
asymmetry is the useful conclusion, and it does not depend on the two missing
numbers.

## What this says about the engine gap

Adding the floor up over the whole step: 3.087 GB of weights per token at 291.5
GB/s is **10.59 ms, a hard ceiling of 94.4 tokens per second** for this model on
this card at batch 1. Nothing, in any engine, in any language, beats it without
changing the batch size or the weight precision.

| | ms per token | tok/s | what separates it from the row above |
| --- | --- | --- | --- |
| hardware floor | 10.59 | 94.4 | nothing, this is bytes over bandwidth |
| vLLM, CUDA graphs | 12.56 | 79.6 | 1.97 ms of imperfect kernels and attention |
| HF device busy | 15.31 | 65.3 | 2.75 ms more, small kernels and bad shapes |
| HF measured step | 40.65 | 24.6 | 25.34 ms of a GPU waiting on Python |

Three conclusions the gate work produced that reasoning alone had not.

**1. No kernel in the HF decode path is meaningfully slow.** The two largest
matmuls run at 96 to 97% of the memory ceiling and `lm_head` runs at the ceiling
itself. 78% of device busy time is in 225 matmul kernels that are collectively
within about 13% of optimal. The 3.24x engine gap is not a kernel quality
problem.

**2. What HF loses on the device, it loses on shapes and on small kernels, not
on arithmetic.** 3.34 ms of the 15.31 ms busy time is spread across 973 kernels
that do nearly nothing, which is what vLLM's hand-written fusions attack. The
remaining 1.37 ms of matmul time above the floor is the 33% efficiency of the
GQA projections plus the split-K penalty on `down_proj`.

**3. vLLM has captured 84% of what the memory system can physically deliver,**
so scheduling is a nearly closed frontier at batch 1. The remaining headroom is
in arithmetic intensity, and the only lever on 1.0 FLOP per byte is serving more
than one token per weight read. That is batching, measured in
`docs/batching-results.md`, and it is a different mechanism from anything in
`docs/profiling.md`.

## Open, and what would close it

**0. Write Part 1.** It needs no GPU and it is the critical path: Phase 2 closes
when this file and `docs/profiling.md` both stand on numbers, and half of this
file is currently a form. The `T = 16` against `T = 32` occupancy table is the
part most likely to change what the section concludes, since the two tile sizes
pull arithmetic intensity and residency in opposite directions.

The remaining items need the GPU, and all are quick once a box is running.

1. **Confirm the grid geometry.** `ncu --metrics launch__grid_size,launch__block_size`
   on the `gemv2T` kernel. The prediction is 16 blocks of 128 threads for
   `k_proj` and 560 for `gate_proj`. If `k_proj` launches many more blocks than
   16, the occupancy explanation for the 33% row is wrong and has to be
   rewritten around whatever the real limiter is.
2. **Confirm coalescing directly.** Sectors per global load request on the same
   kernel. 2.0 confirms the cooperative layout inferred above, 32.0 refutes it.
3. **Measure this card's FLOP/s** the way `measure_bandwidth.py` measures its
   bandwidth, so the ridge point stops being a spec sheet number. The 360 GB/s
   episode is the precedent for not trusting one.

Until 1 and 2 are run, the coalescing argument stands on achieved bandwidth
(strong but indirect) and the occupancy argument stands on a reading of a
template parameter (a hypothesis with a matching prediction). Both are labelled
as such above.
