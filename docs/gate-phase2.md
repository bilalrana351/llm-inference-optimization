# Phase 2 gate: diagnosing a kernel from first principles

The Phase 2 gate is two questions, and the deliverable is answering them in
numbers rather than adjectives.

1. Take a kernel you did not write, from a source you can read, and account for
   its memory traffic, its coalescing, and its occupancy.
2. Do the same for a kernel out of your own decode trace, where nobody has
   handed you the source or the answer.

**Both parts are written.** Passing on someone else's kernel is the bar and
passing on your own decode path is the proof. What remains open in each is
stated at the end, and it is measurement of constants rather than missing
analysis.

The two parts are a matched pair, and the difference between them is the point:

| leg | Part 1, tiled matmul | Part 2, cuBLAS GEMV |
| --- | --- | --- |
| source | readable | closed |
| bytes and intensity | derived exactly, symbolically in `N` and `T` | derived exactly |
| coalescing | derived exactly from the indexing, both mappings | inferred from 96% of achieved bandwidth |
| occupancy | derived exactly, parametric in registers per thread | open, needs `ncu` |

Part 1 is the control. All three legs come out exactly when the source is
readable, which is what licenses the inference in Part 2's coalescing section
and the flagged gap in its occupancy section: those two are soft because cuBLAS
is closed, not because the analysis cannot be done.

The two parts also converge on the same mechanism from opposite directions.
Part 1 finds a tiled matmul that cannot reach compute-bound at any legal tile
size, because one output element per thread caps arithmetic intensity at `T/4`
and `T` is capped at 32. Part 2 finds a decode step pinned at 1.0 FLOP per byte
because a batch-1 GEMV reuses nothing. In both cases the fix is the same shape:
make each loaded byte serve more outputs, by register tiling in one case and by
batching in the other.

## Part 1: naive against tiled matmul, from GPU MODE lecture 5

**Status: written.** Numbers below are derived symbolically first, then
substituted. Three quantities are flagged for measurement rather than quoted:
peak FLOP/s (for the ridge point), registers per thread (from
`nvcc -Xptxas -v`), and the device properties (from the torch snippet in 1.3).
Everything else is exact from the indexing.

Notation: square matmul `C = A * B` with `A`, `B`, `C` all `N x N` in fp32, so
bytes per element `b = 4`. One output element per thread. Tile size `T`, so a
block is `T x T` threads and shared memory holds a `T x T` tile of `A` and of
`B`. The kernels are PMPP Fig 3.11 (naive) and the lecture 5 tiled kernel
(`TILE_WIDTH 16`), with `threadIdx.x` mapped to the column of `C` and
`threadIdx.y` to the row.

### 1.1 Bytes per output element, and arithmetic intensity

Naive: one output element is the dot product of a full row of `A` (`N` elements)
and a full column of `B` (`N` elements), each element read from global memory
once per output. Tiled: the thread loads one `A` element and one `B` element per
phase, over `N/T` phases, and every load is reused `T` times inside the block.

| quantity | naive | tiled, tile size `T` |
| --- | --- | --- |
| global reads of A per output element | `N` | `N/T` |
| global reads of B per output element | `N` | `N/T` |
| total global bytes per output element | `2Nb = 8N` | `2Nb/T = 8N/T` |
| FLOPs per output element | `2N` | `2N` |
| arithmetic intensity, FLOP/byte | `2N / 8N = 1/4 = 0.25` | `2N / (8N/T) = T/4` |
| ratio, tiled over naive | | `T` |

`N` cancels in both intensities, which is the point: intensity is a property of
the access pattern, not the problem size. Tiling multiplies intensity by exactly
`T`, no more and no less, because it turns `T` global reads into one. For
`T = 16`, tiled intensity is `4.0` FLOP/byte and the ratio over naive is `16`.

Then the two questions that turn the algebra into a claim about this card:

- **Where does each land relative to the sm_86 ridge point?** Ridge point is
  `peak FLOP/s / 291.5 GB/s`. Using the nominal fp32 figure of ~12.74 TFLOP/s
  for the 3060, the ridge is near **44 FLOP/byte**. Naive at `0.25` sits ~175x
  below it; tiled `T = 16` at `4.0` sits ~11x below; tiled `T = 32` at `8.0`
  sits ~5.5x below. All three are firmly memory-bound. **The FLOP/s figure is a
  spec sheet number this repo has not measured**, so the ridge is provisional
  until a FLOP/s probe exists (same fix as Part 2 and open item 3).
- **What `T` would move the kernel to compute-bound, and is it reachable?**
  Need `T/4 >= 44`, so `T >= 176`. But a block is `T x T` threads and the sm_86
  block cap is 1024 threads, so `T <= 32`. **Unreachable.** The one-element-per-
  thread tiled design cannot reach compute-bound on this card at any legal tile
  size; the most it can reach is `T = 32`, intensity `8`, still ~5.5x memory-
  bound. Raising intensity further needs register tiling (each thread computes
  several output elements, so each shared-memory read serves more FLOPs), which
  raises intensity without raising `T`. That is the real lever, and this kernel
  does not pull it.

### 1.2 Coalescing, at the granularity of one warp and one instruction

Index mapping analysed: `threadIdx.x -> Col`, `threadIdx.y -> Row`, block
`16 x 16`. Threads linearise as `tx + 16*ty`, so **warp 0 is `ty in {0,1}`,
`tx in 0..15`: one warp is two block-rows of sixteen.** That single fact drives
every row below.

| load | what one warp touches | sectors per request | coalesced? |
| --- | --- | --- | --- |
| naive, A = `M[Row*N + k]` | address has no `tx`; only `ty` varies, so **2 distinct addresses** `N*4` B apart, each read by 16 lanes | 2 sectors, 8 of 64 B useful | no, broadcast |
| naive, B = `N[k*N + Col]` | address has no `ty`; `tx` 0..15 gives **16 consecutive fp32** = 64 B, each read by the 2 `ty` lanes | 2 sectors, 64 B useful | yes |
| tiled, A tile = `M[Row*N + ph*T + tx]` | per `ty`, `tx` 0..15 is 16 consecutive fp32; the warp is 2 such runs, 64 B each, `N*4` B apart | 4 sectors (2 per block-row) | yes, per row |
| tiled, B tile = `N[(ph*T+ty)*N + Col]` | per `ty`, `tx` 0..15 is 16 consecutive fp32; 2 runs of 64 B | 4 sectors (2 per block-row) | yes, per row |

One clarification on the first row, so the "no" is not over-read. A 2-way
broadcast is not the expensive failure mode: 2 sectors serving 32 lanes is the
minimum traffic those lanes could possibly generate, and it is cheaper per lane
than the coalesced row below it. What actually costs the naive kernel on the `A`
side is not the shape of one warp's request but that the same row of `A` is
re-read from global memory by every block along the output row, which is exactly
the `N` against `N/T` difference in 1.1. Bad sector efficiency and bad reuse are
different defects, and only the second one is why tiling wins.

**Name the index mapping, and what the swap would do.** The analysis above is
for `threadIdx.x -> Col`. Under that mapping naive `B` is contiguous across the
warp and naive `A` is a broadcast. **Swap to `threadIdx.x -> Row`** and the two
trade places: `A = M[Row*N + k]` now has `Row` varying with `tx`, so the 16
lanes land on 16 addresses `N*4` B apart, a 16-way scatter, 16 sectors, 64 of
512 B useful (4 useful bytes per 32 B sector); and `B = N[k*N + Col]` now has
`Col` fixed across `tx` and varying only with `ty`, giving 2 distinct addresses
per warp, which is the same broadcast shape that `A` had before the swap. So the
same source is either a coalesced load or a 16-way scatter depending only on
which axis `threadIdx.x` names. This is why "naive matmul is uncoalesced" is
only half a statement: exactly one of the two loads is bad, and which one is a
choice.

**Shared memory is a separate question.** Once the tiles are resident, the
compute loop reads `Mds[ty][k]` and `Nds[k][tx]` out of shared memory, where the
failure mode is bank conflicts, not sectors. sm_86 has 32 banks of 4 B.

- `Mds[ty][k]`: `k` is the loop index, same for all lanes; only `ty` varies, so
  2 distinct addresses read by 16 lanes each. Broadcast, no conflict.
- `Nds[k][tx]`: `tx` 0..15 maps to 16 distinct banks, each read by the 2 `ty`
  lanes. Broadcast within each bank, no conflict.

So for `T = 16` in this row-major layout with this access pattern, **there is no
bank conflict**, and no padding is needed. A `T = 32` warp is one full row
(`tx` 0..31 -> banks 0..31), so `Nds[k][tx]` is still 32 distinct banks and
`Mds[ty][k]` is still a broadcast: also conflict-free. Padding to `T+1` would
only be needed if a load or the compute walked a *column* of a tile (stride `T`
across the warp), which this kernel never does.

### 1.3 Occupancy on sm_86

Fill the hardware limits from the device, not from memory:

| sm_86 limit | value | source |
| --- | --- | --- |
| SMs on this 3060 | 28 | `multi_processor_count` |
| max threads per SM | 1536 | `max_threads_per_multi_processor` |
| max warps per SM | 48 | threads per SM over 32 |
| max blocks per SM | 16 | CUDA C Programming Guide, compute-capability table |
| max threads per block | 1024 | `max_threads_per_block` |
| shared memory per SM | 100 KB (102400 B) | `shared_memory_per_multiprocessor` |
| max shared memory per block | 99 KB (101376 B) | `shared_memory_per_block` |
| 32-bit registers per SM | 65536 | `regs_per_multiprocessor` |

```python
import torch
p = torch.cuda.get_device_properties(0)
for k in ("name", "multi_processor_count", "max_threads_per_multi_processor",
          "shared_memory_per_multiprocessor", "shared_memory_per_block",
          "regs_per_multiprocessor", "warp_size", "major", "minor"):
    print(f"{k:36s} {getattr(p, k, 'n/a')}")
```

Values above are the published sm_86 and 3060 figures and must be confirmed by
running that snippet before this section is final, for the same reason
`docs/profiling.md` names: a spec sheet number already sent one section down a
wrong path.

Occupancy for the two tile sizes:

| | `T = 16` | `T = 32` |
| --- | --- | --- |
| threads per block | 256 | 1024 |
| warps per block | 8 | 32 |
| shared memory per block (both tiles, fp32) | `2*16*16*4` = 2 KB | `2*32*32*4` = 8 KB |
| blocks per SM limited by threads | `1536/256` = 6 | `1536/1024` = 1 |
| blocks per SM limited by shared memory | `102400/2048` = 50 | `102400/8192` = 12 |
| blocks per SM limited by the block cap | 16 | 16 |
| blocks per SM limited by registers, at `R` reg/thread | `65536/(256*R)` | `65536/(1024*R)` |
| **binding limit** | **6 (threads), if `R <= 42`** | **1 (threads), if `R <= 64`** |
| resident warps, and occupancy | `6*8` = 48 warps, **100%** | `1*32` = 32 warps, **66.7%** |
| arithmetic intensity from 1.1 | 4.0 FLOP/byte | 8.0 FLOP/byte |

Registers per thread `R` is a property of the compiled kernel, not the tile
size, so it enters as a parameter. **Get it from `nvcc -Xptxas -v` on the
lecture 5 source** (one command, and the better answer). The crossover: at
`T = 16` registers become the binding limit only for `R >= 43`
(`65536/(256*43) = 5.9 < 6`); at `T = 32`, threads already cap residency at 1
block, so registers cannot bind unless `R > 64`. For the PMPP tiled kernel `R`
is typically in the high 20s to mid 30s, which leaves threads as the limiter in
both columns, but confirm rather than assume. Note also that sm_86 splits the
register file across 4 sub-partitions and rounds register allocation per warp,
so the true blocks-per-SM can round down below the per-SM division; the per-SM
figure is the standard answer, state which you used.

**The conclusion this table exists to support.** Larger tiles raise intensity
linearly in `T` (4 to 8) while raising threads and shared memory per block.
The tension is sharp here:

- `T = 32` halves global traffic per output (`8N/T`, `T` doubled), which on a
  memory-bound kernel directly cuts runtime, and it doubles intensity. That is
  the win.
- But 1024-thread blocks pack badly into 1536 thread slots: only 1 block fits,
  wasting 512 slots, and occupancy falls to 66.7%. That is the textbook
  "performance cliff" from a block size that does not divide the thread-slot
  pool.

Which wins depends on whether 32 resident warps is enough to keep DRAM
saturated. If it is, the halved traffic dominates and `T = 32` is the better
pick despite lower occupancy, because on a memory-bound kernel bytes set the
time and 66.7% occupancy that still saturates bandwidth costs nothing. If 32
warps cannot hide the latency, `T = 16` at 100% wins. **This is the one call in
Part 1 that a hand calculation cannot settle**, so the honest form is: pick
`T = 32` on the traffic argument, and verify with
`ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
that DRAM throughput stays at ceiling; fall back to `T = 16` if it does not.

Part 2 is the reason to take that caveat seriously rather than as boilerplate.
The `gemv2T` kernel there reaches 96% of achievable bandwidth, and whatever its
occupancy turns out to be, it is already sufficient. Occupancy mattered in that
section only where the grid was too small to fill the machine at all. The same
asymmetry is the expectation here: 66.7% is very likely enough.

**A separate caveat on measuring this at all.** The grid for a small `N` is
tiny: at `N = 64` the grid is `4 x 4` = 16 blocks against 28 SMs, so most SMs
get one block or none and per-SM occupancy is a ceiling you never reach. That is
the identical failure that puts `k_proj` at 33% of the ceiling in Part 2, from
the same cause. Benchmark at `N` in the thousands so the grid has many waves;
keep `N = 64` only for the hand calculation.

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

Both parts are analytically complete. Every item below is a constant to measure
or a prediction to confirm, all of them need the GPU, and all are quick once a
box is running. Nothing here requires more reasoning, which is the difference
between an open item and an unwritten section.

**Affecting Part 2's conclusions:**

1. **Confirm the grid geometry.** `ncu --metrics launch__grid_size,launch__block_size`
   on the `gemv2T` kernel. The prediction is 16 blocks of 128 threads for
   `k_proj` and 560 for `gate_proj`. If `k_proj` launches many more blocks than
   16, the occupancy explanation for the 33% row is wrong and has to be
   rewritten around whatever the real limiter is.
2. **Confirm coalescing directly.** Sectors per global load request on the same
   kernel. 2.0 confirms the cooperative layout inferred above, 32.0 refutes it.

**Affecting Part 1's conclusions:**

3. **Registers per thread for the tiled kernel**, from `nvcc -Xptxas -v` on the
   lecture 5 source. The 1.3 table is parametric in `R` precisely so this does
   not block the analysis, but it does move one number: at `T = 16` any
   `R >= 43` drops residency from 6 blocks to 5, so 100% occupancy becomes 83%.
   The `T = 32` column is insensitive to `R` below 65.
4. **Confirm the sm_86 device properties**, using the torch snippet in 1.3.
   Eight constants currently taken from published tables.
5. **Settle the `T = 16` against `T = 32` choice**, which 1.3 states as the one
   call a hand calculation cannot make. Run both and compare
   `sm__warps_active.avg.pct_of_peak_sustained_active` against
   `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`. If `T = 32` holds
   DRAM at the ceiling on 32 resident warps, the halved traffic wins and the
   occupancy deficit is free.

**Affecting both:**

6. **Measure this card's FLOP/s** the way `measure_bandwidth.py` measures its
   bandwidth. Both parts place their kernels against a ridge point of 44
   FLOP/byte that rests on a 12.74 TFLOP/s spec figure, and the 360 GB/s episode
   is the standing precedent for not trusting one. This is the last unmeasured
   hardware constant the repo depends on.

Until 1 and 2 are run, Part 2's coalescing argument stands on achieved bandwidth
(strong but indirect) and its occupancy argument stands on a reading of a
template parameter (a hypothesis with a matching prediction). Until 3 through 6
are run, Part 1's arithmetic and coalescing are exact but its occupancy
percentages and both ridge distances carry unconfirmed constants. Every one of
those is labelled at the point it is used.
