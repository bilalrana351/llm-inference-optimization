# Environment and plan changes (Vast.ai RTX 3060 box)

Read alongside CLAUDE.md and phase1.md. This file records the actual compute we landed on and what it changes versus the original Colab-based plan. Where this file and CLAUDE.md disagree on hardware, this file wins.

## What changed: compute and workflow

- We are no longer using Colab. Phase 1 runs on a rented Vast.ai box: one RTX 3060 12GB, on-demand, about $0.045/hr.
- Workflow is now "Claude Code on the box," not local-first. Claude Code runs over SSH directly on the rented machine, so it edits, runs on the GPU, reads tracebacks, and commits all in one place. No git push/pull loop between a laptop and the runner.
- Persistent working directory is `/workspace`. Storage is billed while the instance is stopped, so the repo and the weights cache live in `/workspace`, not in `/root` or `/tmp`. Destroy the instance when done for the week.

## Confirmed environment (verified on the box)

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB (12GB), idle and clean (no other processes).
- Architecture: Ampere, compute capability 8.6 (sm_86).
- Driver 580.126.09, CUDA 13.0. This is an NVIDIA NGC PyTorch image.
- torch 2.12.0+cu130 preinstalled, with torchvision 0.27 and torchcodec 0.12. `torch.cuda.is_available()` is True, device is the 3060, `bf16` supported is True.
- NOT preinstalled: transformers, vllm, accelerate. These are ours to add.
- Network from the box: HuggingFace and GitHub both return HTTP 200, so weight downloads and git push both work.

## What this changes in phase1.md

1. The vLLM attention-backend worry is resolved in our favor. phase1.md warned that a Turing T4 (sm_75) might force vLLM into a fallback backend. On Ampere sm_86 that does not apply: FlashAttention and vLLM's good kernels support this card. Delete that worry.

2. The friction point moves, it does not disappear. The new risk is the very fresh CUDA stack. The box has torch 2.12.0+cu130, which is newer than the torch that vLLM currently bundles (2.11+cu130). vLLM compiles its own CUDA kernels and is binary-tied to a specific torch build, so a naive `pip install vllm` into the existing environment would pull vLLM's torch 2.11 and clobber the box's torch 2.12.

   Rule: install vLLM in a separate fresh virtual environment and let the wheel bring its own torch. Do not try to make vLLM reuse the system torch 2.12 (that path means building vLLM from source). CUDA 13.0 vLLM wheels exist, so a fresh-env install should not require a source build.

3. fp16 still stands, for a refined reason. This card supports bf16, unlike the T4 and V100. We still target fp16, both for comparability with the later NUST HPC V100/T4 runs and because it does not change the KV-cache OOM math (fp16 and bf16 are both 2 bytes per element).

4. OOM math on 12GB. A 1.5B model in fp16 is about 3GB of weights, leaving roughly 9GB for the KV cache to grow into. Expect the OOM cliff at a sensible context length, slightly sooner than on a 16GB card, which makes the sweep faster and cheaper.

## Concrete next steps (two environments, on purpose)

Environment A, the HF baseline. Use the existing preinstalled-torch environment. Only add:

```
pip install transformers accelerate
export HF_HOME=/workspace/hf-cache   # cache weights on persistent storage
```

Build and validate the baseline harness here (Task 1 in phase1.md). This path has no version drama, so it should be a clean first win.

Environment B, the vLLM run. Create a fresh, isolated environment so it cannot disturb Environment A:

```
python -m venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate
# install vLLM and let it pull its own matching torch for CUDA 13.0
# (uv is the recommended installer; uv pip install vllm --torch-backend=cu130)
pip install vllm
```

Run the vLLM benchmark (Task 2) from Environment B. Same model, same prompt, same max tokens, batch size 1, so the comparison against Environment A stays fair.

## Box C (2026-09-01, the energy and tensor-format studies)

Phase 3 study 1 ran on a third rented 3060, and the workflow changed with it:
the box is driven over SSH from the laptop (scripts written locally, synced
with rsync, results pulled back) rather than Claude Code living on the box.

- GPU: RTX 3060 12288 MiB, sm_86, driver 595.71.05. Vast base image (not
  NGC), CUDA 12.8 toolkit, system python via `/venv/main`, `/workspace` on
  the container overlay (no volume, nothing survives a recycle).
- Calibration (`measure_bandwidth.py`): 341.4 GB/s achievable read, 26.2 fp16
  tensor TFLOP/s. Compare the profiling box's 291.5 GB/s: three boxes, three
  speeds, same listing name. One box per comparison table, always.
- Environment A: torch 2.11.0+cu128 plus the pinned requirements. Environment
  B: vLLM 0.28.0 with torch 2.13.0+cu130 (the current vLLM wheel is built
  against CUDA 13, so `--torch-backend=cu128` fails at import with a missing
  libcudart.so.13; cu130 is the working install).
- Tensor-format build tools: CUDA 12.8 `nvcc` and Ninja 1.11.1. The handwritten
  2:4 GEMV was compiled for sm_86 through PyTorch's C++ extension loader.
- HF batch-1 fp16 decodes at 59.5 tok/s here against 24.6 on the profiling
  box: HF is launch-bound, so the host CPU moves it far more than the GPU
  name suggests. The engine gap itself is host-dependent.
- `ncu` is present but GPU performance counters are blocked in this
  containerized instance (ERR_NVGPUCTRPERM). The two open gate measurements
  in docs/gate-phase2.md need a VM-based instance instead.

## Box D (2026-09-03, the KV-cache eviction study)

Study 3 ran on a fourth rented 3060, again driven over SSH from the laptop.
This was a Vast container without a mounted volume, so `/workspace` was
ephemeral and the completed artifacts were copied back immediately after the
run.

- GPU: RTX 3060 12288 MiB, sm_86, driver 580.173.02, CUDA 12.8 toolkit.
- Environment A: torch 2.11.0+cu128, transformers 4.46.3, datasets 3.1.0,
  and the remaining pinned requirements in `/venv/main`.
- Calibration: 340.9 GB/s achievable streaming read (94.7% of the 360 GB/s
  theoretical rate), 7.42 fp32 SIMT TFLOP/s, 13.10 TF32 TFLOP/s, and 26.15
  fp16 tensor TFLOP/s.
- Quality workload: WikiText-2 test, 4,096 unscored warmup predictions and
  2,048 scored predictions. The full-cache perplexity was 11.213. A
  4,096-token sink-and-window cache used 33% less live KV memory and measured
  11.219 perplexity.
- Systems workload: exact synthetic KV shapes through real SDPA decode. At a
  120,000-token starting context, full cache used 3,281.6 MiB and decoded at
  4.26 tokens/sec. A 512-token window used 14 MiB and decoded at 29.17
  tokens/sec, a 6.84x speedup.

The environment began empty, and the official torch wheel host was unusually
slow on a single connection. The wheel was downloaded in verified byte ranges,
reassembled, and checked against its official SHA-256 before installation. This
was an installation workaround only and does not affect the benchmark method.

## Box E (2026-09-03, the serving-under-load study)

Study 4 ran on a fifth rented 3060, driven over SSH from the laptop. Like Box
D, `/workspace` is part of the container overlay rather than a mounted volume.
Stop and start preserve the container, but recycle or destroy erase it. All
completed results were copied back immediately.

- GPU: RTX 3060 12288 MiB, sm_86, driver 570.133.20, CUDA 12.8 toolkit. The
  card exposes a 150 W power limit and a 7501 MHz maximum memory clock.
- Calibration: 351.1 GB/s achievable streaming read against a 360.0 GB/s
  clock-derived peak, 7.12 fp32 SIMT TFLOP/s, 12.83 TF32 TFLOP/s, and 25.67
  fp16 tensor TFLOP/s. The 3 GiB read buffer is far larger than L2, although
  the 97.5% read-to-theoretical ratio is unusually high and is reported as
  measured rather than assumed.
- Environment A: `/venv/main` with torch 2.11.0+cu128 for calibration and the
  plotting packages. Environment B: vLLM 0.10.2 with torch 2.8.0+cu128,
  transformers 4.53.2, and NumPy 2.2.6 in `/workspace/vllm-env`.
- The current default vLLM wheel targets CUDA 13 and cannot run against this
  consumer card's CUDA 12.8 driver. vLLM 0.10.2 supplies CUDA 12.8 binaries.
  Its unconstrained dependencies now resolve to Transformers 5.x and NumPy
  2.5, both too new for this release, so the two compatibility pins above are
  part of the reproducible environment.
- The four-config load sweep completed 4,608 requests with no errors and no
  preemptions. Poisson SLO capacity was 2 requests/s for every configuration;
  fixed bursts of eight missed the TTFT SLO at every tested mean rate. Full
  method and results are in `docs/load-results.md`.

## One tooling note

Claude Code needs Node, which the NGC image likely does not include. If `node --version` fails, install a current Node first, then `npm install -g @anthropic-ai/claude-code`. Set git identity (`git config --global user.name` / `user.email`) so commits from the box are attributed correctly.
