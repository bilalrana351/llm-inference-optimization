# CLAUDE.md

Persistent project context for Claude Code. Read at the start of every session.

## Private local context

When `.local-context/bilal-masters-context.local.md` exists, read it at the start
of sessions involving MS preparation, portfolio positioning, outreach, Mitacs,
CVs, SOPs, recommendation letters, scholarships, or research strategy. The file
is local-only and Git-ignored. Never commit it, publish it, or quote private
details externally without Bilal's explicit approval. For current technical
project status, inspect the repository and Git history because dated roadmap
notes in the private context may lag the implemented work.

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

We are no longer on Colab. The studies run on rented Vast.ai RTX 3060 boxes;
see `docs/vastai.md` for the per-box record. Every comparison table stays on
one calibrated box.

- Current box E: one NVIDIA RTX 3060 12GB, Ampere sm_86, driver 570.133.20,
  CUDA 12.8. Its measured achievable read bandwidth is 351.1 GB/s and fp16
  tensor throughput is 25.67 TFLOP/s.
- `/workspace` is the container overlay with no mounted volume. It is
  non-persistent, so sync scripts in and copy results out before recycle or
  destroy.
- Environment A is `/venv/main`: torch 2.11.0+cu128 plus the analysis packages.
  Environment B is `/workspace/vllm-env`: vLLM 0.10.2, torch 2.8.0+cu128,
  transformers 4.53.2, and NumPy 2.2.6. Model caches live under
  `/workspace/hf-cache` only while this box exists.
- Keep Environment B separate from Environment A because vLLM is binary-tied
  to its own torch build.
- The sm_86 card supports the required CUDA and attention kernels, so the old
  T4 fallback concern does not apply.

## Hard technical constraints (these change your decisions)

- Target fp16. NOT bf16. The 3060 supports bf16, but we still target fp16 for comparability with the later NUST HPC V100/T4 runs (which do not support bf16), and it does not change the KV-cache OOM math (fp16 and bf16 are both 2 bytes per element). Never default to bf16.
- Single GPU, the one RTX 3060, 12GB. If a multi-GPU runtime ever appears, pin to one device so the measurement stays clean single-GPU.
- Small model: Qwen2.5-1.5B or TinyLlama. Weights in fp16 are roughly 3GB, which on 12GB leaves roughly 9GB for the KV cache to grow into.
- The small GPU is intentional for the OOM experiment. We WANT to hit the KV-cache wall at a realistic context length, so do not suggest a bigger GPU to "avoid" the crash. The crash is the experiment. On 12GB the cliff comes slightly sooner than on a 16GB card, which makes the sweep faster and cheaper.
- The measurement harness must separate prefill from decode, and must log both tokens/sec and VRAM (track allocated and reserved separately). Get this boringly correct on the transformers baseline before trusting any vLLM number.

## Workflow

- Work locally, sync only the required scripts to `/workspace`, run them over
  SSH, and copy raw results and plots back immediately. The rented container is
  a runner, not the source of truth.
- Read `/etc/vast-agents-guide.md` before operating a fresh Vast box. It records
  the container's persistence and privilege rules.
- The repo is the artifact: keep it clean, reproducible, and documented. Pin dependency versions (the vLLM install is the main friction point of this phase, now the fresh CUDA 13.0 / torch 2.12 stack rather than a Turing backend issue). Prefer runnable scripts over notebook-only code so results reproduce.
- Commit experiment outputs (plots, CSV logs) alongside the code that produced them.

## Where this is heading (later phases)

- Phase 0 (done): the theory.
- Phase 1 (done): baseline, vLLM, the OOM experiment, and the batching sweep on real hardware.
- Phase 2 (done, except two open ncu measurements listed at the end of `docs/gate-phase2.md`): profiling the decode step (`docs/profiling.md`) and the kernel-diagnosis gate (`docs/gate-phase2.md`). The measured answer to the engine gap: 90% is CPU-launch idle removed by CUDA graphs, 9% is faster device work from vLLM's hand-written fused kernels, and torch.compile is worth nothing once graphs are on.
- Phase 3 (current): energy per token, compressed tensor formats, KV-cache
  eviction, and serving under load are complete. Only the optional model
  splitting stretch study remains, as defined in `docs/phase3.md`.
- OSS target: vLLM. This replaces the earlier SGLang plan because the strongest
  Mitacs and Canadian supervisor fits work on vLLM serving, and vLLM has wider
  name recognition. The concrete goal is one real merged contribution in KV-cache
  management, scheduling, or quantization.
- Hardware path later: the NUST HPC cluster for larger runs (1x V100 on compute1, 2x T4 on compute3 and compute4, SLURM scheduler). On that cluster: download weights on the login node first because compute nodes may be firewalled, use non-root conda or venv installs into the home directory, never run jobs on the master node, and request `--gres=gpu:t4:1` or `--gres=gpu:v100:1` as appropriate.

## The eventual goal: MS applications

This artifact is meant to be a credible research signal for funded MS admissions, targeting a Fall 2027 start. Target regions: Canada (SFU, UBC, McGill, with Waterloo and UofT as reaches), Germany, the UK (needs a Commonwealth or Gates scholarship), Finland, and Switzerland. Application gates in progress: IELTS (target 7.5 Academic), possibly GRE (Quant 165+), and the HAT for the Commonwealth route. Hold the repo and writing to a standard a research admissions committee would respect: clear methodology, honest measurement, reproducible runs.

## Style

- No em dashes. Anywhere. In code comments, docstrings, README, commit messages, and the blog draft. Use commas, colons, periods, or parentheses instead. Regular hyphens in compound words (real-time, fp16-only) are fine.
- Writing should be simple, direct, and concrete. Lead with the measurement or the mechanism, not with throat-clearing.
