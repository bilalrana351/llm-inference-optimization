# Week 5: consolidate, then clear the Phase 2 gate

Budget for the LLM inference track this week is about 30 to 32 hours, all technical. This is the gate week: one consolidation lecture, close any remaining reference gaps, then the gate itself, read a kernel you did not write and explain why it is fast or slow in coalescing, shared-vs-global, and occupancy terms. If it lands clean, Phase 2 is done and you are about two weeks ahead of the mid-August nominal.

## Part A: consolidate, about 6 to 8 hours

- [ ] **GPU MODE lecture 8, CUDA Performance Checklist (about 3 hours).** The best single consolidation of everything Phase 2 tests: coalescing, occupancy, memory-vs-compute bound, and the usual ways a kernel leaves performance on the table. This is the checklist you will run in your head during Phase 3.

- [ ] **GPU MODE lecture 9, parallel reduction, if time (about 2 hours).** Control divergence, memory divergence, minimizing global-memory access, and thread coarsening, shown on a reduction. Optional but high-value, these are the exact failure modes the gate asks you to name.

- [ ] **Close the PMPP chapter 4 to 6 gaps (about 2 hours).** Anything from last week's reference reading that is still fuzzy: the occupancy math, the tiling walk-through, the coalescing rules. Do not start new chapters.

## Part B: the gate, about 8 to 10 hours

- [ ] **Take the gate in writing (about 4 hours).** Take a kernel you did not write, the naive vs tiled matmul from GPU MODE lecture 5 is the cleanest choice, or a coalesced vs uncoalesced copy. Write out why one is faster in coalescing, shared-vs-global, and occupancy terms. Commit it. If you can do this without hand-waving, the gate is real.

- [ ] **Read a second kernel from your own traces (about 2 hours).** Go back to the profiling artifact from last week and diagnose one more kernel from your own decode trace. Passing the gate on someone else's kernel is the bar, passing it on your own code is the proof.

- [ ] **Do not rush the gate to unlock kernels.** The gate is not a checkbox that unlocks Phase 3, it is the debugging loop you will live inside for all of Phase 3. A slow kernel gives you a bare number and nothing else, and only this skill turns that number into a fix. A shaky gate makes Phase 3 slower and more painful, not faster. Spend the time to make it solid.

## Part C: what to do with the lead, about 6 to 8 hours

- [ ] **If the gate lands clean.** You are about two to three weeks ahead of the mid-August nominal. Do not open Phase 3 in a rush to spend the lead, kernels are the highest-signal artifact and deserve a clean start. Put the surplus into two places: deepen the profiling artifact (more kernels read, tighter writeup), and move the IELTS booking and Writing practice forward. IELTS is the gate that actually decides Fall 2027, not finishing the roadmap early.

- [ ] **If the gate is shaky.** Next week is a clean buffer: re-watch GPU MODE 4 and 5, redo the tiling and coalescing reasoning, and re-attempt. This is the one step whose whole value is not being rushed.

- [ ] **Still no kernel writing.** Phase 3 opens on its own schedule, mid-August, whether or not the gate cleared early.