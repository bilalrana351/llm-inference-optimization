# Notes: async CUDA execution, timing, and the memory allocator

Background notes worked out while building the Phase 1 measurement harness. These
are the mechanics behind two harness rules (synchronize before every timer read,
track allocated and reserved VRAM separately) and behind why the OOM sweep
crashes where it does. Written as a reference, not a tutorial.

## Why a CudaTimer instead of plain time.time()

Two things are bundled in the timer, and only one is about the clock.

The minor part: `perf_counter` over `time.time`. `perf_counter` is monotonic and
higher resolution. `time.time` can jump backward on NTP adjustments and is
coarser. Real, but not the reason the timer exists.

The actual point: the `torch.cuda.synchronize()` calls. CUDA kernel launches are
asynchronous. Calling `model(input_ids)` returns almost immediately: it has
enqueued kernels onto the CUDA stream and handed control back to the CPU, but the
GPU has not finished (often not started) the work. So this measures the wrong
thing:

```python
t0 = time.time()
out = model(input_ids)   # returns once kernels are queued, not done
t1 = time.time()         # measured launch/dispatch overhead, not GPU compute
```

On a decode step that reports microseconds of launch overhead in place of
milliseconds of real compute, so the decode rate comes out wildly too fast and
completely fake.

`torch.cuda.synchronize()` blocks the CPU until every queued kernel has actually
finished. Correct pattern:

```python
torch.cuda.synchronize(); t0 = perf_counter()
out = model(input_ids)
torch.cuda.synchronize(); t1 = perf_counter()   # t1 - t0 is real GPU time
```

The leading sync matters too: it drains work still in flight from the previous
step (warmup, prefill) so it does not leak into this measurement's start. The
harness encodes both syncs once so you cannot forget one; the first time you do,
the number is silently garbage. This is the single most common way inference
benchmarks lie.

For fine-grained per-kernel timing, CUDA events (`torch.cuda.Event` with
`elapsed_time`) timestamp on the stream without stalling the CPU. For whole-phase
wall time (prefill, decode), sync + `perf_counter` is the right tool.

## How `out` is valid before anything is computed

`out` is not the computed numbers. It is a handle to GPU memory where the numbers
will eventually land. When `model(input_ids)` is called:

1. PyTorch allocates the output buffer on the GPU immediately. It knows the shape
   and dtype from the op, so it reserves the memory now without computing
   anything. That allocation is what `out` points to.
2. It enqueues the kernels that will fill that buffer.
3. It returns `out`: a Python object (living in CPU RAM) wrapping a pointer into
   that as-yet-unwritten GPU memory, plus shape/stride/dtype/device metadata. All
   the metadata is correct immediately; only the contents are pending.

This is safe because everything on one stream runs in issue order. If the next
line is another GPU op reading `out`, that kernel is enqueued after the kernels
that write `out`, so the data is there by the time it runs. You never observe a
half-written tensor from GPU code. The ordering guarantee does the
synchronization implicitly, which is why a whole `generate()` loop can queue
thousands of kernels without a manual sync.

Contents only have to materialize when the CPU wants to read them, because the
CPU is not on the stream. That is when an implicit sync fires:

```python
out = model(input_ids)             # returns instantly, buffer not yet written
val = out.logits[0, -1, 0].item()  # .item() forces a sync: CPU blocks until written
```

`.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `print(tensor)` all cross the
GPU->CPU boundary and block until the relevant kernels finish. Benchmarking trap:
an accidental `.item()` inside a timing loop inserts a sync every iteration,
serializes the pipeline, and changes what you measure.

## Ordering: it is about streams, not data dependencies

The clean statement:

- Same stream: kernels always execute in enqueue order (FIFO), data dependency or
  not. The stream does not reorder independent work. No blocking needed.
- CPU reading GPU data: blocks until the producing kernels finish. This is the
  only place the "wait on a data dependency" intuition applies, and it is
  specifically the CPU<->GPU boundary.
- Different streams: no ordering guarantee relative to each other; they can
  overlap or reorder. Imposing order across streams needs a CUDA event one stream
  waits on.

What looks like "order can change" in single-stream inference is really the CPU
running ahead of the GPU: the GPU's execution order is fixed, the CPU's position
relative to it is what drifts. The decode loop is all one stream, so kernels are
correctly ordered for free, and the only sync inserted is the deliberate one
before reading the clock.

## Where tensors live: VRAM, not RAM

`out` and every intermediate in the forward pass live in VRAM. Weights are in
VRAM, activations are allocated in VRAM, logits land in VRAM. The Python object
is a tiny handle in CPU RAM holding metadata and a pointer; the numbers are on
the other side of the PCIe bus. `out.device` says `cuda:0`.

```python
y = out.logits.softmax(-1)     # new kernel, result also in VRAM, CPU sees nothing
z = out.logits.argmax(-1)      # still VRAM, still a handle
val = z.item()                 # crosses PCIe: GPU->RAM copy, CPU blocks, Python float out
cpu_logits = out.logits.cpu()  # explicit full-tensor copy VRAM->RAM
```

Any op that keeps the result a tensor keeps it in VRAM and stays asynchronous.
Data moves to RAM only on an explicit pull (`.cpu()`, `.numpy()`, `.item()`,
`.tolist()`, `print()`), and that copy is over PCIe, far slower than VRAM
bandwidth. Hence: keep the whole decode loop on-GPU, copy out only the final
token ids, never the full logits each step.

## Why VRAM grows during generation

Each forward pass allocates for two purposes with very different lifetimes:

- Activations: the intermediates (Q/K/V projections, attention scores, MLP
  hidden states, logits). Transient. Freed once the step finishes and nothing
  references them. They cause a temporary per-step bump, not growth. In decode
  they are small (one token at a time).
- KV cache: appends one token's worth of keys and values per layer every step,
  and stays resident because every future step attends to it. This is the
  `past_key_values` threaded through the loop. It grows monotonically: step 500
  holds 500 tokens, step 5000 holds 5000. This is the term in the Phase 0 formula
  and the thing that wins the race to fill 16GB.

So rebinding `out` each iteration frees the previous step's activations (refcount
to zero), but the cache is held by a separate live reference and never freed, so
it just gets bigger. One visible variable hides one thing being recycled and
another piling up.

## The caching allocator: allocated vs reserved

PyTorch does not call `cudaMalloc`/`cudaFree` (the slow, synchronizing driver
calls) per tensor. It runs a caching allocator in front of the driver, and it
allocates lazily and grows on demand (it does not grab one giant chunk at start;
that is vLLM's `gpu_memory_utilization`, a separate layer).

- `memory_allocated` = bytes currently handed out to live tensors. The real
  usage: weights + KV cache + this step's activations.
- `memory_reserved` = bytes PyTorch has taken from the driver into its pool,
  handed out or not. Always >= allocated.

When a tensor's refcount hits zero, "freed" means the block returns to PyTorch's
pool marked reusable, not back to the OS. `reserved` stays the same, `allocated`
drops, and the next similar-sized allocation reuses the block with no driver
call. From the OS's view the process still owns that VRAM. This is why the
harness tracks both, and why `peak_reserved` is the honest "how close to the
wall" number: the OS and driver overhead have to fit around reserved.

Size policy, roughly: two pools (small for requests up to ~1MB, large above),
so tiny tensors do not fragment space meant for big ones. On a trip to the
driver it rounds up and grabs a segment (small ~2MB, large rounded to a ~20MB
granularity, very large rounded to the request), carves the tensor out, and
keeps the remainder as a free block. Rounding up means the next similar request
is served from the pool with zero driver calls. Segment size is decided per
driver trip by the triggering request, not once up front; `reserved` is the sum
of segments.

## What "weights resident" actually includes: the RoPE buffer trap

`memory_allocated` right after load, before any forward pass, should equal the
weight floor and nothing else. For an fp16 model that floor is exactly
`params x 2 bytes`. Qwen2.5-1.5B is 1.544e9 params, so the floor is 2944 MiB. If
the measured number is far above that, something non-weight is resident, and it
is worth chasing before it silently eats the KV budget.

On the first baseline run, weights reported 4737 MiB, ~1.8 GB over floor. It was
not the weights and not an fp16-cast leak. Breaking the live allocation down by
(param/buffer, dtype) put the whole excess in fp16 buffers, and naming the
buffers found it: legacy per-layer RoPE caches.

- transformers 4.44.2 used the old rotary embedding that registers
  `cos_cached`/`sin_cached` as persistent buffers, one pair per attention layer,
  each sized to `max_position_embeddings` x `head_dim`. For Qwen2.5 that is
  131072 x 128 in fp16 = 32 MiB per table, two tables per layer, 28 layers:
  `28 x 2 x 131072 x 128 x 2 bytes = 1792 MiB`. That matched the excess exactly.

This is a bad space-time trade, and the contrast with the KV cache is the point.
The cos/sin table depends only on (position, frequency), never on token data, so
recomputing it is a couple of `sin`/`cos` calls over the current positions,
nearly free. The KV cache depends on the actual tokens and recomputing it means
redoing O(n^2) attention over the prefix, so it must be stored. The legacy code
cached the cheap, recomputable thing, sized to the full 128K context regardless
of the sequence actually run, and duplicated it across all 28 layers.

transformers 4.46+ hoists rotary to a single model-level module that computes
cos/sin on the fly for only the positions present, storing nothing but the tiny
`inv_freq` vector (64 fp32 entries per layer, ~7 KB total). After bumping the pin
to 4.46.3, resident weights dropped to 2945 MiB, one MiB off the floor.

Two takeaways for the harness:

1. Sanity-check resident weights against `params x dtype_bytes` before trusting
   any VRAM budget. A library version can silently park over a GB of buffers in
   what you are calling "weights." `test/probe_weights_vram.py` does this
   breakdown by (kind, dtype) and lists the largest named buffers.
2. The buffer never touched decode speed. Decode indexes one row of the cos/sin
   table per step, it does not stream all 1.79 GB, so tokens/sec was unaffected.
   The cost was purely a fixed KV-headroom tax: ~1.8 GB / 28 KiB-per-token is
   about 64k tokens of context the OOM sweep would have lost to a library wart.

## What OOM actually is

When the pool cannot satisfy a request from existing free blocks, it asks the
driver for more via `cudaMalloc`. If the driver has no free VRAM, that call fails
and PyTorch raises `CUDA out of memory`. The crash is one specific allocation
request the pool could not serve and the driver could not back.

Consequences for the sweep:

1. Request-sized, not gradual. You do not die when the cache "reaches" 16GB. You
   die at the step where the next allocation needs more than is free and that one
   request fails. Clean crash point, not a slow asymptote. The error reports the
   failed request size and the free amount.
2. Fragmentation can kill before "full". A tensor needs one contiguous span of
   VRAM addresses: kernels index a flat buffer with `base + i*stride`, and there
   is no page-table indirection making scattered VRAM look contiguous to a
   kernel. So a request needs one free block big enough, not N scattered bytes.
   You can have 7MB free total but no single gap >= 5MB and a 5MB request OOMs
   anyway.

## Why inference provokes fragmentation specifically

- Two lifetimes interleaved in one address space: the long-lived, only-growing KV
  cache and the short-lived, every-step-churning activations. Freed activation
  gaps are odd-sized holes scattered between cache blocks. Late in a long
  generation you can be near capacity with plenty of total free bytes but no
  contiguous block large enough for the next activation: holes of the wrong
  shape.
- Variable sequence lengths: a pool carved up by 512-token requests has
  512-shaped holes; a later 2048-token request needs a bigger contiguous block
  that may not exist. In a batched server, requests start and finish out of
  order, so blocks free in an order unrelated to allocation order, the textbook
  fragmentation recipe.
- The cache is the dominant growing tenant, so it keeps forcing new segments from
  the driver; once the driver is out, any request not servable from a hole OOMs.

This is exactly what PagedAttention exists to kill. vLLM stops storing each
sequence's KV cache as one contiguous buffer. It chops the cache into fixed-size
blocks (pages) with a block table mapping logical token positions to physical
blocks, the OS's RAM paging trick rebuilt for VRAM in software. Once every block
is the same size and any free block can serve any sequence, the "right bytes,
wrong shape" failure disappears and memory packs far closer to full before
OOMing. A big part of the baseline-vs-vLLM gap this phase measures.

## Consequences for the OOM sweep

- The measured crash point will sit somewhat below the analytical KV prediction.
  Allocator overhead, fragmentation, activation scratch, and the CUDA context eat
  the difference. That gap is a result to report in the blog, not an error to
  engineer away: it is the difference between idealized KV math and a real
  allocator on real hardware.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` changes the segment strategy
  to fragment less. Leave it off for the baseline so the sweep measures the
  honest, default-allocator wall.
- Running sweep steps in one process: after a step OOMs, drop references and call
  `torch.cuda.empty_cache()` (which does return PyTorch's free pool to the OS) so
  step N's reserved memory does not poison step N+1's measurement. To be handled
  when building `oom_sweep.py`.
