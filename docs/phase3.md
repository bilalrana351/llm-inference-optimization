# Phase 3: from one gap study to a small lab

Phases 1 and 2 answered one question with instruments built for it: why is the
optimized engine faster than the naive one on this card, and by exactly how much,
split by mechanism. Phase 3 turns the same instruments on four questions a
serving system actually faces. Each study is deliberately small, has one headline
number, and lands as the same artifact shape the repo already uses: one doc, one
plot, raw CSVs, and a runnable script.

The studies are ordered by instrument cost: each one reuses the harness and the
methodology discipline of the previous ones (prefill split from decode, sync
before timers, warmup discarded, medians over repeats, hardware pinned to one
box per comparison table).

## Study 1: energy per token

**The question.** Everyone reports tokens per second. Almost nobody reports
joules per token, and the two do not have to move together: a slower
configuration at lower power can win on energy. What does each configuration in
this repo actually cost in energy?

**Method.** Add NVML power polling to `bench_common.py`: a background thread
samples `nvmlDeviceGetPowerUsage` (total board power, milliwatts) every 20 to 50
ms and integrates to joules over exactly the measured window, prefill and decode
integrated separately. Measure the idle floor separately and report both gross
and net (floor-subtracted) energy. Validate the sampling first: confirm the NVML
update rate on this card by sampling a known constant load, and make runs long
enough that integration error is under a few percent.

**Grid.** The configurations the repo has already characterized for speed and
memory, now characterized for energy:

- HF fp16, HF NF4, vLLM fp16, all at batch 1 (the latency column).
- vLLM fp16 at batch 1, 2, 4, 8, 16, 32, 64 (the throughput column, reusing the
  `bench_vllm_batch.py` workload).

**What makes it interesting.** NF4 moves fewer weight bytes but dequantizes
every layer every step and runs slower. Whether that nets out to more or less
energy per token than fp16 is a real question this hardware can answer. And
batching should amortize energy the way it amortizes bandwidth, so joules per
token should fall with batch until the compute region, another curve with a knee
in it.

**Deliverables.** `results/energy_*.csv`, an energy-against-throughput Pareto
plot, `docs/energy-results.md`. Estimated 8 to 12 hours.

## Study 2: tensor formats against the hardware

**The question.** A representation that compresses well does not have to run
fast. The repo already has one measured case: NF4 quarters the weight bytes and
still loses to fp16 on decode speed at batch 1. Generalize that into a
format-by-format study: which compressed formats convert saved bytes into saved
time on this card, and can a bytes-per-output model predict the winners?

**Method.** Benchmark the same GEMM shapes the roofline study extracted from the
real decode step (`results/roofline_step.csv`, all eight shapes), across formats:

- dense fp16 (cuBLAS, the control)
- INT8 (`torch._int_mm`)
- NF4 (bitsandbytes `matmul_4bit`, already measured end to end)
- 2:4 semi-structured sparse fp16 (`torch.sparse.to_sparse_semi_structured`,
  cuSPARSELt under the hood; sm_86 has the sparse tensor cores for it)

Sweep the batch dimension M in {1, 8, 32, 128} per shape, because format wins
are batch-dependent: at M = 1 everything is bandwidth-bound and byte counts
should rule, at larger M compute and dequant cost enter.

**The model.** For each format, derive bytes moved per output element including
metadata (2:4 stores 2 of every 4 values plus index bits, about 0.56x dense;
INT8 is 0.5x plus scales; NF4 is 0.25x plus absmax blocks), predict runtime from
the measured bandwidth ceiling on the study box (341.4 GB/s for the completed
run), then measure. The prediction errors are the result: where the model
holds, bytes explain everything; where it breaks, the
kernel or the dequant path is the story. Write at least one comparison kernel in
CUDA rather than calling a library, so the analysis in `docs/gate-phase2.md`
gets an implementation leg.

**Scope note.** This is a kernel-level study. Model quality under pruning or
quantization is out of scope here; the formats are evaluated as storage and
execution formats on fixed weights.

**Deliverables.** `scripts/bench_formats.py`, per-format predicted-against-
measured tables, `docs/formats-results.md`. Estimated 15 to 20 hours.

## Study 3: KV-cache eviction

**Status.** Complete. The measured results and reproducible method are in
`docs/eviction-results.md`; raw and derived CSVs plus both plots are in
`results/`.

**The question.** The OOM study measured what happens when the cache keeps
everything: 60 KB per token until the card dies at 123k. Not every cached token
earns its keep. How much of the cache can be evicted before quality degrades,
and what do memory and speed win back?

**Method.** Implement two eviction policies on the HF path via the transformers
`Cache` API (the pinned 4.46.3 ships `SinkCache` as a starting point):

- sliding window plus attention sinks (keep the first few tokens and a recent
  window)
- an attention-score variant (keep the top-budget tokens by cumulative attention
  mass, which requires `attn_implementation="eager"` to expose scores; use it
  for the quality curve and the cheap window variant for throughput numbers)

**Metrics.** Perplexity on long WikiText slices against cache budget (full, 4k,
2k, 1k, 512 tokens); VRAM slope against the measured 60 KB/token baseline;
decode tok/s against context length, checked against the 30-to-3 decay the OOM
study measured, since a bounded cache should flatten that curve.

**Deliverables.** `scripts/bench_eviction.py`, quality-against-budget and
speed-against-context plots, `docs/eviction-results.md`. Estimated 15 to 20
hours.

## Study 4: serving under load

**Status.** Complete. The measured results and reproducible method are in
`docs/load-results.md`; raw and derived CSVs plus both plots are in `results/`.

**The question.** Every number so far is offline: hand the engine a batch, wait.
A served system faces arrivals. What breaks first when a bursty workload hits a
memory-constrained server, and which scheduling knob moves the breaking point?

**Method.** Run vLLM as a server and drive it with an async client generating
Poisson and bursty arrivals, prompt and output lengths sampled from a fixed-seed
realistic distribution (mixed short and long requests, not the uniform 512/256
of the offline runs). Sweep arrival rate to find the knee. Then move the knobs
vLLM exposes (`max_num_seqs`, `gpu_memory_utilization` at 0.9 against 0.5,
chunked prefill on and off) and measure how the knee moves.

**Metrics.** TTFT p50 and p95, per-token latency, throughput, preemption and
queue depth from the engine's own counters, and goodput under an explicit SLO
(for example TTFT under 1 s and per-token latency under 100 ms), which is the
number a serving paper would lead with.

**Deliverables.** `scripts/bench_load.py`, knee plots per knob,
`docs/load-results.md`. Estimated 12 to 15 hours.

## Stretch study: model splitting across processes

Only if the four above land. Split Qwen2.5-1.5B at layer k into two processes
that pass hidden states over localhost. The per-token payload is tiny (hidden
1536 x 2 bytes = 3 KB) but it lands on the critical path of every decode step,
so the question is at what latency per hop pipeline splitting stops making
sense. Measure end-to-end tok/s against split point, with and without injected
network delay, against a transfer-cost model. Estimated 10 hours.

## Exit condition

Phase 3 closes study by study, not all at once: a study is done when its doc,
plot, and CSVs are committed and the README results table has its row. The
studies are independent, so a stalled one does not block the others. Order of
execution is the order above, chosen so the cheapest instrument extension lands
first.
