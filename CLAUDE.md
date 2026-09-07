# CLAUDE.md

This file contains working instructions for agents editing this repository.

## What this repository contains

This is a public collection of measured LLM inference studies. Each study has
one question, runnable scripts, raw data, plots, and a report explaining the
method, results, and limitations.

The four featured studies cover:

- quantization and sparse tensor formats
- GPU energy per generated token
- KV-cache pruning
- LLM serving under steady and bursty traffic

Earlier work covers HuggingFace against vLLM, KV-cache memory growth, batching,
decode profiling, CUDA graphs, and kernel-level hardware analysis.

## Measurement rules

- Use fp16, not bf16, so results stay comparable across the target GPUs.
- Keep every comparison table on one calibrated GPU.
- Never combine measurements from different rented machines in one comparison.
- Separate prefill from decode.
- Synchronize the GPU before timing work.
- Discard warmup runs and report medians over repeated measurements.
- Keep code, raw data, plots, and the report together.
- Do not publish a numerical claim unless the supporting data is committed.
- Explain limitations and unexpected results directly.

## Hardware and environments

The completed studies used rented RTX 3060 12 GB GPUs. Machine details,
software versions, and measured bandwidth are recorded in `docs/vastai.md` and
the report for each study.

The rented machines may use a non-persistent `/workspace` directory. Treat the
local repository as the source of truth. Copy results back before stopping a
machine.

Keep the HuggingFace and vLLM environments separate because vLLM is tied to a
specific PyTorch and CUDA build. Follow the versions in the relevant study
report instead of assuming the newest packages will work.

## Project status

- Phase 1 is complete: HuggingFace and vLLM baselines, KV-cache OOM, and
  batching.
- Phase 2 is complete except for two optional Nsight Compute measurements in
  `docs/gate-phase2.md` that require access to GPU performance counters.
- Phase 3 is complete: tensor formats, energy, KV-cache pruning, and serving
  under load.
- Model splitting across processes remains an optional stretch study.

## Writing style

- Use simple, direct language.
- Explain the cause after reporting a result.
- Define uncommon terms when they first appear.
- Do not use em dashes. Use commas, colons, periods, or parentheses.
