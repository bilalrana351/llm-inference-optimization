# Week 2: baseline, vLLM, and the KV-cache OOM experiment

Working plan for this phase. Hand this to Claude Code as the task breakdown. Persistent context and constraints live in `CLAUDE.md`; this file is the ordered set of things to build and the precise definitions to build them against.

## The goal in one paragraph

Run a small model two ways on a single Colab T4: first with plain HuggingFace `transformers` (the slow, readable control), then through vLLM (the optimized engine). Measure both with the same harness. Then deliberately push the KV cache until it OOMs, plot VRAM against context length, and compare the measured crash point to the Phase 0 analytical prediction. The deliverable is a public, reproducible repo plus a blog draft, not just code that ran once.

## Suggested repo structure

```
llm-inference-optimization/
  README.md              # phase goal + how to reproduce
  requirements.txt       # pinned versions (vLLM, torch, transformers)
  CLAUDE.md              # persistent context (already added)
  PHASE1.md              # this file
  scripts/
    bench_common.py      # shared measurement harness (timing, VRAM, logging)
    baseline_hf.py       # task 1
    bench_vllm.py        # task 2
    oom_sweep.py         # task 3
  results/
    *.csv                # raw logs
    *.png                # plots (incl. the headline OOM curve)
  blog/
    draft.md             # write as you measure
  colab/
    run.ipynb            # thin notebook: git pull + run a script on the T4
```

## Shared measurement definitions (get these right first)

These are the rules every script follows. Most wrong inference numbers come from breaking one of these.

- Prefill vs decode are measured separately. Prefill is the single forward pass over the whole prompt (parallel, compute-bound). Decode is generating new tokens one at a time (sequential, memory-bound). Report them as separate numbers, never blended.
- Decode tokens/sec is the headline number. It is (new tokens generated) / (decode wall time), and it must exclude prompt tokens and exclude prefill time.
- Always `torch.cuda.synchronize()` immediately before reading any timer. GPU kernels launch asynchronously, so without a sync you measure launch time, not execution time.
- Run a warmup generation before any timed run and discard it. The first call pays for CUDA init and kernel setup.
- VRAM logging tracks both `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`, and reports the peak via `torch.cuda.max_memory_allocated()`. Reset peak stats before each measured run with `torch.cuda.reset_peak_memory_stats()`.
- fp16 only (`dtype=torch.float16` / `dtype="float16"`). Never bf16 on the T4.
- Inference mode: `model.eval()` and `torch.inference_mode()` for the HF path.
- Fairness: identical model, identical prompt, identical `max_new_tokens`, greedy decoding (temperature 0), and batch size 1 across HF and vLLM. A comparison is only valid if the only thing that changed is the engine.

## Task 0: scaffold the repo

- [ ] Create the structure above. README states the phase goal and a one-command reproduce path.
- [ ] `requirements.txt` with pinned versions. The vLLM / torch / CUDA version match is the main friction point of this phase, so pin deliberately and note the working combination in the README.
- [ ] Start `blog/draft.md` now with a title and an empty "the OOM curve" section. Writing as you measure is the point.

## Task 1: HuggingFace baseline (the control)

- [ ] Load Qwen2.5-1.5B (or TinyLlama) in fp16 on the T4.
- [ ] Build `bench_common.py`: the timing + VRAM + logging harness described above, written once and reused.
- [ ] `baseline_hf.py`: warmup, then a measured run that records prefill latency, decode tokens/sec, and peak VRAM, and writes a row to `results/`.
- [ ] Confirm the numbers are stable and sane across a couple of runs. This is the control, so it has to be boringly solid before vLLM is touched.

Gotchas: count only generated tokens in the decode rate; synchronize before every timer read; the model weights are roughly 3GB in fp16, so log how much VRAM is weights vs how much grows during generation.

## Task 2: vLLM serving the same model

- [ ] Install vLLM. Expect this to eat real time. Pin a version compatible with Colab's CUDA and torch. The T4 is compute capability 7.5: vLLM runs on it, but some optimized attention backends target newer GPUs, so it may fall back to a default backend. Note whatever combination works in the README.
- [ ] Use vLLM's offline `LLM` class (in-process), not the HTTP server, for a clean single-process benchmark. Set `dtype="float16"` and a sensible `gpu_memory_utilization`.
- [ ] `bench_vllm.py`: same model, same prompt, same `max_new_tokens`, same batch size 1, measured with the same harness logic.
- [ ] Produce a comparison table (HF vs vLLM): decode tokens/sec, prefill latency, peak VRAM.

Gotcha: vLLM pre-allocates a large KV-cache pool up front (controlled by `gpu_memory_utilization`), so its raw VRAM number is not directly comparable to HF's. Note what that number actually represents rather than comparing it naively.

## Task 3: the KV-cache OOM experiment (the headline)

- [ ] Run this on the HF baseline path, since it is the transparent one and easiest to control.
- [ ] `oom_sweep.py`: scale total sequence length up in steps. At each step, run a generation, record peak VRAM and decode tokens/sec.
- [ ] Catch the CUDA out-of-memory error, record the context length at which it fails, and stop cleanly.
- [ ] Compute the analytical KV-cache size at each step from the Phase 0 formula: `2 (K and V) x num_layers x num_kv_heads x head_dim x seq_len x dtype_bytes x batch`. 
- [ ] Plot measured peak VRAM vs context length, with the analytical KV-cache prediction overlaid. The match between predicted and measured (and where they diverge) is the result.
- [ ] Decide and document whether you grow the sequence via a long prompt (prefill) or long generation (decode). State the choice in the plot caption.

Output: `results/oom_curve.png` plus a short paragraph in the blog draft explaining the mechanism, written the moment the plot exists.

## Task 4: reading (skim, do not deep-dive)

- [ ] Skim the vLLM docs: quickstart plus the conceptual page on how PagedAttention manages the KV cache.
- [ ] Read one good PagedAttention explainer (the original vLLM blog post or paper summary).
- [ ] Drop two or three takeaways into the blog draft. Just enough that the serving vocabulary stops being unfamiliar, which lands naturally once you have watched the cache OOM yourself.

## Task 5: blog draft and repo polish

- [ ] Blog draft has: the setup, the HF-vs-vLLM table, the OOM curve with the predicted-vs-measured comparison, and a paragraph on why the gap between the two engines exists (the thing later phases will build).
- [ ] README explains the working dependency versions and how to reproduce on a fresh T4.

## Definition of done for Phase 1

- A reusable measurement harness committed.
- HF baseline numbers committed (decode tokens/sec, prefill latency, peak VRAM).
- vLLM run committed with the comparison table.
- The OOM curve committed, with the analytical prediction overlaid and the crash point identified.
- Blog draft holding the curve and the mechanism paragraph.
- A fresh clone runs on a T4 by following the README.

## Order and the one trap

Build in this order: harness, then HF baseline (make it boringly solid), then vLLM (budget for install friction), then the OOM sweep (which reuses the harness), then reading, then writing. The trap, called out in the week plan, is feeling like you understand it and skipping the writeup. The published measurement is the deliverable, so the blog draft grows alongside the code, not at the end.