# Week 3: close out the real Phase 1, then begin Phase 2 (the GPU itself)

Budget for the LLM inference track this week is about 30 to 32 hours. Roughly 18 to 20 of those close out Phase 1 (the two experiments the repo version silently dropped), and the remaining 10 to 12 start Phase 2. The pairing is deliberate: the quantization and batching runs raise exactly the questions the CS149 execution-model lectures answer, so the measurements and the reading reinforce each other in the same week.

## Part A: finish the real Phase 1 (the two dropped experiments), about 18 to 20 hours

- [x] **Quantization, FP16 vs 4-bit (about 7 hours).** Load Qwen2.5-1.5B in 4-bit (bitsandbytes NF4 is the simplest path on the Ampere 3060) and run it through your existing harness against the fp16 baseline you already have. Log weights VRAM, peak VRAM, prefill latency, and decode tokens/sec for both. The honest lesson is that 4-bit is a memory lever, not a speed lever at batch 1: weights drop from about 3 GB to roughly 1 GB, but decode can stay flat or even slow down, because bitsandbytes dequantizes to fp16 on the fly and decode was already memory-bound. The clean win is the freed VRAM, which ties straight back to the OOM run: 4-bit weights leave more room for the KV cache, so the crash point moves out. Show that connection.

- [ ] **Batching throughput sweep in vLLM (about 9 hours).** Everything so far is batch 1, which is latency, not throughput. Feed vLLM a growing number of concurrent sequences and, at each batch size, measure aggregate throughput (total output tokens across all sequences divided by wall time) and per-request latency. Expect throughput to climb as continuous batching keeps the GPU fed, then flatten as the KV pool or compute saturates, while latency rises the whole way. Plot throughput vs batch size and the throughput vs latency tradeoff. This is the "why batching matters" lesson and the floor-vs-ceiling point the blog draft already gestures at. Nail the throughput definition before starting, total tokens across the batch over wall time, never blended with the batch-1 decode rate.

- [ ] **Repo and blog polish (about 3 hours).** Embed the OOM curve inline in `draft.md` (the one lingering gap from last week, the image is committed but not shown in the post). Add the two new sections (quantization, batching) as you measure, same discipline as the rest. Update the README with the 4-bit and batching reproduce steps and the working bitsandbytes version pin.

## Part B: begin Phase 2, the GPU itself, about 10 to 12 hours

- [ ] **Stanford CS149 first block (about 8 hours).** Watch the opening lectures on the execution model: the modern multi-core processor (ILP, SIMD, hardware multithreading), the parallel programming abstractions, and the GPU architecture and memory hierarchy. Take light notes. This is the why behind the numbers you just measured: it explains why decode is memory-bound and why batch 1 leaves the GPU idle.

- [ ] **GPU MODE lecture 1 (about 2 hours).** Watch lecture 1 to anchor the practical profiling and PyTorch-to-kernel framing. Notes only, no kernel writing.

- [ ] **No kernel writing this week.** Triton and CUDA kernels are Phase 3. The Phase 2 gate (read a kernel and explain why it is fast or slow, in terms of coalescing, shared vs global memory, and occupancy) is the mid-August target, not this week. Week 3 only builds the mental model that the gate will test.