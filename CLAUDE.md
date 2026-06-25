# CLAUDE.md

Persistent project context for Claude Code. Read at the start of every session.

## What this repo is

Phase 1 of an LLM inference optimization learning roadmap. The goal is a public, reproducible artifact (this repo plus an accompanying blog post) that demonstrates measured, first-principles understanding of how LLM inference behaves on real hardware. This artifact feeds two longer-term goals: funded MS applications and open-source contributions to LLM serving engines.

Phase 1 scope, concretely:

- Establish a baseline: run a small model with plain HuggingFace `transformers` `.generate()` on a single GPU. This is the slow, readable control. Everything else is measured against it.
- Run the same model through vLLM on the same GPU. The gap between the two is the entire point of the phase.
- Headline experiment: deliberately OOM the KV cache. Push context length up in steps, log VRAM at each step, plot the curve, find the crash point. This turns the Phase 0 KV-cache math into a measured plot.
- Capture as you measure: commit plots and logs and update the blog draft in the same pass. The published measurement is the deliverable, not the running of it.

For the complete details of Phase 1 (full scope, methodology, and deliverables), see `docs/phase1.md`.

## Who I am (so you calibrate, do not over-explain)

Final-year CS student at NUST Islamabad, founding engineer at an AI startup. Phase 0 is done and I have the theory cold: transformer inference arithmetic, KV-cache growth math, prefill vs decode, why decode is memory-bound, CUDA GEMM tiling (Simon Boehm's SGEMM article, Horace He's "Making Deep Learning Go Brrrr"). Assume that level. Skip introductory explanations of attention, the KV cache, or what fp16 is. When something is new, explain it at the level of someone who knows the systems theory but is running this exact stack for the first time.

## Current compute (Vast.ai RTX 3060 box)

We are no longer on Colab. Phase 1 runs on a rented Vast.ai box, see `docs/vastai.md` for the full record. Where this and any older Colab references disagree on hardware, the Vast.ai box wins.

- GPU: one NVIDIA RTX 3060, 12GB, on-demand, about $0.045/hr. Ampere, compute capability 8.6 (sm_86).
- Stack: NVIDIA NGC PyTorch image, driver 580.126.09, CUDA 13.0, torch 2.12.0+cu130 preinstalled. transformers, vLLM, accelerate are NOT preinstalled. HuggingFace and GitHub are both reachable from the box.
- Persistent working directory is `/workspace`. Storage is billed while stopped, so the repo and the weights cache (`HF_HOME=/workspace/hf-cache`) live in `/workspace`, never `/root` or `/tmp`. Destroy the instance when done for the week.
- Two environments on purpose. Environment A (HF baseline) uses the preinstalled torch 2.12, just add transformers + accelerate. Environment B (vLLM) is a fresh venv at `/workspace/vllm-env` where vLLM brings its own matching torch for CUDA 13.0 (uv recommended: `uv pip install vllm --torch-backend=cu130`). Do NOT `pip install vllm` into Environment A: it would clobber torch 2.12 with vLLM's bundled 2.11.
- The sm_86 card resolves the old T4 attention-backend worry: FlashAttention and vLLM's good kernels support this card, so no fallback backend.

## Hard technical constraints (these change your decisions)

- Target fp16. NOT bf16. The 3060 supports bf16, but we still target fp16 for comparability with the later NUST HPC V100/T4 runs (which do not support bf16), and it does not change the KV-cache OOM math (fp16 and bf16 are both 2 bytes per element). Never default to bf16.
- Single GPU, the one RTX 3060, 12GB. If a multi-GPU runtime ever appears, pin to one device so the measurement stays clean single-GPU.
- Small model: Qwen2.5-1.5B or TinyLlama. Weights in fp16 are roughly 3GB, which on 12GB leaves roughly 9GB for the KV cache to grow into.
- The small GPU is intentional for the OOM experiment. We WANT to hit the KV-cache wall at a realistic context length, so do not suggest a bigger GPU to "avoid" the crash. The crash is the experiment. On 12GB the cliff comes slightly sooner than on a 16GB card, which makes the sweep faster and cheaper.
- The measurement harness must separate prefill from decode, and must log both tokens/sec and VRAM (track allocated and reserved separately). Get this boringly correct on the transformers baseline before trusting any vLLM number.

## Workflow

- Claude Code runs on the box. Claude Code is over SSH directly on the rented Vast.ai machine, so it edits, runs on the GPU, reads tracebacks, and commits all in one place. No laptop-to-runner git push/pull loop.
- Claude Code needs Node, which the NGC image may not include. If `node --version` fails, install a current Node, then `npm install -g @anthropic-ai/claude-code`. Set git identity on the box so commits are attributed correctly.
- The repo is the artifact: keep it clean, reproducible, and documented. Pin dependency versions (the vLLM install is the main friction point of this phase, now the fresh CUDA 13.0 / torch 2.12 stack rather than a Turing backend issue). Prefer runnable scripts over notebook-only code so results reproduce.
- Commit experiment outputs (plots, CSV logs) alongside the code that produced them.

## Where this is heading (later phases)

- Phase 0 (done): the theory.
- Phase 1 (this repo): baseline, vLLM, and the OOM experiment on real hardware.
- Later phases: build and explain the optimizations that account for the baseline-vs-vLLM gap. PagedAttention (vLLM's KV-cache paging), continuous batching, kernel fusion, CUDA graphs, and getting Python out of the decode hot loop. The wins come from memory traffic and fusion, not faster FLOPs: you cannot beat cuBLAS on a raw GEMM.
- OSS target: SGLang specifically. It has under half of vLLM's contributor count, so visibility per PR is higher, and it is backed by xAI, Oracle, LinkedIn, and Cursor. A measured, hardware-grounded contribution (for example ROCm / MI300X enablement, which AMD staffs in the open and is less crowded) is the kind of artifact that the people doing hiring actually see.
- Hardware path later: the NUST HPC cluster for larger runs (1x V100 on compute1, 2x T4 on compute3 and compute4, SLURM scheduler). On that cluster: download weights on the login node first because compute nodes may be firewalled, use non-root conda or venv installs into the home directory, never run jobs on the master node, and request `--gres=gpu:t4:1` or `--gres=gpu:v100:1` as appropriate.

## The eventual goal: MS applications

This artifact is meant to be a credible research signal for funded MS admissions, targeting a Fall 2027 start. Target regions: Canada (SFU, UBC, McGill, with Waterloo and UofT as reaches), Germany, the UK (needs a Commonwealth or Gates scholarship), Finland, and Switzerland. Application gates in progress: IELTS (target 7.5 Academic), possibly GRE (Quant 165+), and the HAT for the Commonwealth route. Hold the repo and writing to a standard a research admissions committee would respect: clear methodology, honest measurement, reproducible runs.

## Style

- No em dashes. Anywhere. In code comments, docstrings, README, commit messages, and the blog draft. Use commas, colons, periods, or parentheses instead. Regular hyphens in compound words (real-time, fp16-only) are fine.
- Writing should be simple, direct, and concrete. Lead with the measurement or the mechanism, not with throat-clearing.