# llm-inference-optimization

Phase 1 of an LLM inference optimization roadmap: run one small model two ways on
a single GPU, measure both with the same harness, then deliberately OOM the KV
cache and plot the crash against the analytical prediction.

The goal is a reproducible, measured artifact, not code that ran once. See
`docs/phase1.md` for the full plan, `docs/vastai.md` for the compute record, and
`CLAUDE.md` for persistent context.

## What is here

```
scripts/
  bench_common.py            shared measurement harness (timing, VRAM, logging, KV math)
  baseline_hf.py             Task 1: HuggingFace transformers baseline (the control)
  bench_vllm.py              Task 2: vLLM, same model and workload
  oom_sweep.py               Task 3: the KV-cache OOM experiment
  analyze_vram_deviations.py per-step VRAM delta reader for the OOM CSV
results/                     raw CSV logs and plots
docs/                        per-task writeups (baseline, vLLM, OOM) and the compute record
blog/draft.md                writeup, grown alongside the measurements
```

## Hardware

Phase 1 runs on a rented Vast.ai box, not Colab. The full record is in
`docs/vastai.md`.

- GPU: one NVIDIA RTX 3060, 12GB (12,288 MiB), Ampere, compute capability 8.6
  (sm_86). On-demand, about $0.045/hr.
- Image: NVIDIA NGC PyTorch, driver 580.126.09, CUDA 13.0, torch 2.12.0+cu130
  preinstalled. transformers, vLLM, and accelerate are not preinstalled.
- Persistent working directory is `/workspace`. The repo and the weights cache
  (`HF_HOME=/workspace/hf-cache`) live there, never `/root` or `/tmp`, because
  storage is billed while the instance is stopped. Destroy the instance when done.

The small card is intentional: at 12GB the KV cache hits the wall at a realistic
context length, which is the headline experiment, not something to avoid.

## How to run

Two environments on purpose, so the vLLM install cannot disturb the baseline
torch.

### Environment A: HuggingFace baseline (Task 1)

Uses the preinstalled torch 2.12. Just add the baseline pins:

```bash
export HF_HOME=/workspace/hf-cache          # cache weights on persistent storage
pip install -r requirements.txt             # transformers, accelerate, matplotlib, pandas
python scripts/baseline_hf.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256
```

The OOM sweep (Task 3) also runs in Environment A, because HF grows the cache
organically so the crash is a real memory event:

```bash
python scripts/oom_sweep.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 32 \
    --checkpoint-every 1000 --csv results/oom_sweep.csv --plot results/oom_curve.png
```

The full sweep is a single continuous decode and takes about six hours of wall
time on the 3060.

### Environment B: vLLM (Task 2)

vLLM is binary-tied to a specific torch build, so a naive `pip install vllm` into
Environment A would clobber the box's torch 2.12 with vLLM's bundled 2.11. Install
it into a fresh, isolated venv and let the wheel bring its own torch for CUDA
13.0:

```bash
python -m venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate
# uv is the recommended installer; it resolves the CUDA 13.0 wheel cleanly
uv pip install vllm --torch-backend=cu130
python scripts/bench_vllm.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256
```

## Constraints

- fp16 only, never bf16. The 3060 supports bf16, but we target fp16 for
  comparability with the later NUST HPC V100/T4 runs (which do not support bf16).
  It does not change the KV-cache OOM math (fp16 and bf16 are both 2 bytes).
- Single GPU, 12GB. The small GPU is intentional: the point is to hit the
  KV-cache wall at a realistic context length.
- Small model (Qwen2.5-1.5B, ~3GB of fp16 weights), leaving roughly 9GB for the
  cache to grow into.

## Working dependency versions

The vLLM install is the friction point of this phase. The combinations that
actually ran on the 3060:

- GPU: RTX 3060 12GB (Ampere, sm_86), driver 580.126.09, CUDA 13.0.
- Environment A (baseline + OOM): torch 2.12.0+cu130 (preinstalled),
  transformers 4.46.3, accelerate 0.34.2. Pins in `requirements.txt`.
- Environment B (vLLM): vLLM 0.23.0 (v1 engine) with torch 2.11.0+cu130, in the
  separate `/workspace/vllm-env`.

The transformers pin matters: 4.46+ moved rotary embeddings to a single
model-level module, dropping a fixed ~1.79 GB of duplicated cos/sin buffers that
4.44.x kept across all 28 layers. See the comment in `requirements.txt`.
