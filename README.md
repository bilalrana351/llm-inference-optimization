# llm-inference-optimization

Phase 1 of an LLM inference optimization roadmap: run one small model two ways on
a single GPU, measure both with the same harness, then deliberately OOM the KV
cache and plot the crash against the analytical prediction.

The goal is a reproducible, measured artifact, not code that ran once. See
`docs/phase1.md` for the full plan and `CLAUDE.md` for persistent context.

## What is here

```
scripts/
  bench_common.py   shared measurement harness (timing, VRAM, logging, KV math)
  baseline_hf.py    Task 1: HuggingFace transformers baseline (the control)
  bench_vllm.py     Task 2: vLLM, same model            (not built yet)
  oom_sweep.py      Task 3: the KV-cache OOM experiment  (not built yet)
results/            raw CSV logs and plots
blog/draft.md       writeup, grown alongside the measurements
colab/run.ipynb     thin notebook: git pull + run a script on the T4
```

## How to run (Colab T4)

The laptop is for development; the T4 on Colab is the GPU executor. The loop is:
push here, then open `colab/run.ipynb` on a T4 runtime, which pulls this repo and
runs a script. Paste tracebacks back into development to debug.

Quick version, in a fresh T4 Colab cell:

```python
!git clone https://github.com/bilalrana351/llm-inference-optimization.git
%cd llm-inference-optimization
!pip install -q -r requirements.txt
!python scripts/baseline_hf.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256
```

The notebook in `colab/run.ipynb` wraps this so you can re-pull and switch which
script runs without retyping anything.

## Constraints

- fp16 only. The T4 (compute capability 7.5) does not support bf16.
- Single GPU, 16GB. The small GPU is intentional: the point is to hit the
  KV-cache wall at a realistic context length.
- Small model (Qwen2.5-1.5B, ~3GB of fp16 weights), leaving headroom for the
  cache to grow into.

## Working dependency versions

The baseline pins live in `requirements.txt`. The vLLM install is the friction
point of this phase; the combination that actually works on the T4 will be
recorded here once Task 2 runs:

- GPU: Tesla T4 (compute 7.5)
- torch / CUDA: _Colab runtime default, recorded after first run_
- vLLM: _tbd_
