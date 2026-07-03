# Week 4: the core of Phase 2 (the GPU itself), and prove the gate on your own repo

Budget for the LLM inference track this week is about 30 to 32 hours, all technical. Phase 1 is closed and Phase 2 is started, so this week is the core of Phase 2: the execution-model reading that explains the numbers you already measured, plus a profiling pass that turns the "why the two engines differ" claim in the blog from reasoning into a measured trace. The gate (read a kernel and explain why it is fast or slow) is the end-of-July target, and this week is what makes it reachable.

## Part A: finish the CS149 execution-model core, about 8 to 10 hours

- [ ] **CS149 Lecture 5, Performance Optimization I (about 4 hours).** Work distribution and scheduling: good work distribution while keeping overhead low, and work-stealing schedulers. This is the first half of the "why a kernel is slow" answer, uneven work and idle units.

- [ ] **CS149 Lecture 6, Performance Optimization II (about 4 hours).** Locality, communication, and contention: arithmetic intensity, pipelining, and avoiding contention. Arithmetic intensity here is the same quantity you computed in the batching roofline, so the lecture just puts a name on what you already measured. Light notes on both.

## Part B: GPU MODE plus PMPP, the gate vocabulary, about 10 to 12 hours

- [ ] **GPU MODE lectures 2 and 3 (about 3 hours).** Lecture 2 recaps PMPP chapters 1 to 3 with the CUDA C basics (heterogeneous computing, data parallelism, thread organization, memory). Lecture 3 is the same ideas written from PyTorch, so kernels compile and run from Python. Watch these fast, they are foundation.

- [ ] **GPU MODE lecture 4, Compute and Memory Basics (about 3 hours).** Compute architecture, memory management, and the first real optimizations: kernel fusion, tiling, and occupancy. This is a gate lecture, so slow down here.

- [ ] **GPU MODE lecture 5, matmul with shared memory and tiling (about 3 hours).** Optimizing matrix multiplication with shared memory and tiling. This is where coalescing and shared-vs-global click, and it is the single most gate-relevant hour of the week.

- [ ] **PMPP as reference, not cover to cover (about 2 hours).** Keep it open next to the lectures: chapter 4 for occupancy, chapter 5 for tiling and shared memory, chapter 6 for coalescing and control divergence. Read 5 and 6 with intent, skim the rest.

## Part C: profile your own repo, the slack move, about 8 to 10 hours

- [ ] **Trace the HF decode step (about 4 hours).** Run the HuggingFace baseline under the PyTorch profiler or Nsight Systems on the 3060 and capture a decode-step trace. In the blog you asserted, from reasoning, that batch-1 decode is CPU-launch-bound: hundreds of tiny kernels, the GPU idle between them, 15 to 20 percent bandwidth. Now show it, the launch gaps should be visible in the trace.

- [ ] **Trace the vLLM decode step (about 2 hours).** Do the same for vLLM and show the single captured-graph replay against the HF kernel storm. This is the measured version of the CUDA-graph claim.

- [ ] **Read one real kernel and diagnose it (about 2 hours).** Pick one kernel out of your own trace and explain why it is fast or slow in coalescing, shared-vs-global, and occupancy terms. This is the Phase 2 gate in miniature, run on your own code.

- [ ] **Commit a short profiling section (about 2 hours).** Add it to the repo, same discipline as the rest, so the "why the two engines differ" claim is measured, not reasoned. This is the repo's whole ethic: a measured artifact, not code that ran once.

- [ ] **No kernel writing this week.** Triton and CUDA kernels are Phase 3, mid-August. Profiling and reading kernels is Phase 2 and stays in bounds. No jumping ahead into FlashAttention or the literature reading either, that is Phase 4.