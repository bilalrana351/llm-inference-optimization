# Week 6: pay back the artifact debt, then close the Phase 2 gate

Budget for the LLM inference track this week is about 30 hours. Weeks 4 and 5 delivered all of the input work and none of the output work: CS149 5 and 6, GPU MODE 2, 3, 4, 5 and 8, and the PMPP 4 to 6 reference pass are all done, but the repo has not gained a script, a doc, or a plot since 2026-07-03. The Phase 2 gate is still open because the gate was defined as writing and nothing is written.

So this is not a new week of material. This is the missing half of the last two weeks, done properly. No new lectures. No Phase 3. No kernel writing.

Before anything else, spend one hour checking the Vast.ai box for uncommitted work. If the traces or gate notes already exist there, this week is much shorter than it looks.

## Part A: the profiling pass, about 13 hours

This is Week 4 Part C, unchanged and now due.

- [x] **Trace the HF decode step (about 4 hours).** Run `baseline_hf.py` under the PyTorch profiler on the 3060 and capture a decode-step trace. Add the runner as `scripts/profile_decode.py` so it reproduces, and commit the raw trace to `results/`. The blog claims from reasoning that batch-1 decode is CPU-launch-bound: many small kernels and an idle GPU between them. The launch gaps should be visible.
  Landed: `scripts/profile_decode.py`, `results/trace_hf_decode.json.gz`. Gaps are visible and they dominate: 1198 kernels per step, 1198 CPU launches, 62.3% of the step is device idle.

- [x] **Trace the vLLM decode step (about 3 hours).** Same profiler, same model, same prompt and output lengths, run in Environment B. Show the single captured-graph replay against the HF kernel storm.
  Landed: `scripts/profile_vllm.py` plus four traces, not one. vLLM 0.25.1 applies torch.compile and CUDA graphs together and `enforce_eager` disables both, so a single vLLM run would have credited one optimization with the other's win. Ran the full 2x2 over compilation and graph capture instead.

- [x] **Reduce both traces to one number each (about 2 hours).** Kernel count per decode step and idle gap time per decode step, HF against vLLM. One table, one figure. This is the measured version of the CUDA-graph claim, and it is the part a reader remembers.
  Landed: `scripts/analyze_trace.py`, `results/profile_summary.csv`, `results/decode_timeline.png`. Three numbers rather than two, because CPU launch count and GPU kernel count separate under graphs and that separation is the mechanism.

- [x] **Commit `docs/profiling.md` (about 4 hours).** Same discipline as `baseline-hf-results.md` and `batching-results.md`: method, raw numbers, then what the numbers mean. Link the figure into `blog/draft.md` where the claim currently sits unsupported.
  Landed: `docs/profiling.md`, and `blog/draft.md` "Why the two engines differ" rewritten against the trace with `decode_timeline.png` linked in. Two claims there were wrong and are now fixed: bandwidth use was 16-24%, not 15-20%, and the paragraph credited CUDA graphs with kernel fusion that graphs demonstrably do not do (kernel counts are identical with graphs on and off).

**Part A is closed.** The measurement changed the argument rather than confirming it: the CUDA-graph win is real and is 90% of the gap, but the fusion the draft attributed to graphs is vLLM's hand-written kernels, and `torch.compile` turns out to be worth 2.81 ms per step without graphs and 0.01 ms with them. Repairing `results/baseline_hf.csv` along the way (header and first data row were joined by a missing newline) recovered the slowest fp16 run, which had been silently absent from every reader.

## Part B: the gate, in writing, about 10 hours

This is Week 5 Part B, unchanged and now due. The gate is the deliverable, not a feeling of readiness.

- [ ] **Take the gate on a kernel you did not write (about 5 hours).** Naive against tiled matmul from GPU MODE lecture 5 is the cleanest choice. Write it in `docs/gate-phase2.md` with numbers, not adjectives:
  - bytes read from global memory per output element, naive against tiled, and the arithmetic intensity that follows from each
  - which loads coalesce and which do not, and why, at the level of what a single warp touches in one instruction
  - the occupancy math for your chosen tile size on sm_86: shared memory per block, registers per thread, resulting blocks per SM
  If any of these three needs hand-waving, that is the part to go back and fix, not to write around.

- [ ] **Take the gate on your own trace (about 3 hours).** Pick one kernel out of the HF decode trace from Part A and diagnose it the same way. Passing on someone else's kernel is the bar. Passing on your own decode path is the proof, and it is also the thing that makes the profiling doc worth reading.

- [ ] **GPU MODE lecture 9, reduction, only if the gate feels shaky (about 2 hours).** Control divergence, memory divergence, and thread coarsening. This is the one lecture allowed this week, and only as a repair, not as new ground. If the gate lands clean, skip it.

## Part C: the admin gate, about 4 hours

- [ ] **Book the IELTS date (about 2 hours).** Not research it, book it. Supervisor outreach is September, HAT is mid-October, and the Canada and Finland and Switzerland deadlines cluster December to January. IELTS is the only item on the list where a slipped week cannot be recovered later.

- [ ] **One IELTS Writing task under timed conditions (about 2 hours).** Task 2, forty minutes, no edits after. Writing is the band that usually holds people under 7.5, and one timed sample now tells you whether this needs weekly hours or almost none.

## Part D: the rule change, about 3 hours

- [ ] **No checkbox without a commit (about 1 hour).** Add this line to `CLAUDE.md`. A lecture counts as done when notes are in the repo. A trace counts as done when the trace is committed. This single rule is what would have caught weeks 4 and 5 on the day it happened instead of two weeks later.

- [ ] **Backfill lecture notes for weeks 4 and 5 (about 2 hours).** Short, one page total, into `docs/`. Not for the reader, for the commit graph and for you. Right now the repo shows fifteen silent days and no evidence that twenty hours of Phase 2 material went in.

## Exit condition

Phase 2 closes when `docs/profiling.md` and `docs/gate-phase2.md` are both committed and both stand on numbers. If that happens this week, you are back on the nominal mid-August schedule and Phase 3 opens on time. If only one lands, next week finishes it and Phase 3 still opens mid-August. Either way, kernels do not start early.