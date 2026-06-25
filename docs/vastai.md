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

## One tooling note

Claude Code needs Node, which the NGC image likely does not include. If `node --version` fails, install a current Node first, then `npm install -g @anthropic-ai/claude-code`. Set git identity (`git config --global user.name` / `user.email`) so commits from the box are attributed correctly.
