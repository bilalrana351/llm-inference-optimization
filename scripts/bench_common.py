"""Shared measurement harness for Phase 1.

Every script imports from here so the timing, VRAM, and logging rules are
written once and applied identically across the HuggingFace baseline, the vLLM
run, and the OOM sweep. Most wrong inference numbers come from breaking one of
the rules encoded below, so this file is the single source of truth for them.

The rules (see docs/phase1.md "Shared measurement definitions"):

- Prefill and decode are timed separately and never blended. Prefill is the one
  parallel forward pass over the whole prompt. Decode is the sequential
  one-token-at-a-time loop.
- Decode tokens/sec is the headline number: generated tokens / decode wall time.
  It excludes the prompt and excludes prefill time.
- torch.cuda.synchronize() is called immediately before every timer read,
  because CUDA kernels launch asynchronously.
- A warmup run happens before any timed run and is discarded.
- VRAM tracks allocated and reserved separately, plus the peak. Peak stats are
  reset before each measured run.
- fp16 only. Never bf16 on the T4.
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import torch


# ---------------------------------------------------------------------------
# VRAM accounting
# ---------------------------------------------------------------------------

def bytes_to_mib(n: int) -> float:
    """Bytes to mebibytes, the unit nvidia-smi and torch both think in."""
    return n / (1024 ** 2)


def reset_peak_vram(device: int = 0) -> None:
    """Reset the peak-memory counters before a measured run.

    Without this, max_memory_allocated() reports the peak since process start,
    which includes the warmup and every prior step of a sweep.
    """
    torch.cuda.reset_peak_memory_stats(device)


@dataclass
class VramSnapshot:
    """A point-in-time view of GPU memory, all in MiB.

    allocated/reserved are the live values at the moment of the snapshot.
    peak_allocated/peak_reserved are the maxima since the last reset, which is
    what you actually want for "how close did we get to OOM".
    """

    allocated_mib: float
    reserved_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float

    @classmethod
    def capture(cls, device: int = 0) -> "VramSnapshot":
        return cls(
            allocated_mib=bytes_to_mib(torch.cuda.memory_allocated(device)),
            reserved_mib=bytes_to_mib(torch.cuda.memory_reserved(device)),
            peak_allocated_mib=bytes_to_mib(torch.cuda.max_memory_allocated(device)),
            peak_reserved_mib=bytes_to_mib(torch.cuda.max_memory_reserved(device)),
        )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class CudaTimer:
    """A context manager that times GPU work correctly.

    Synchronizes before reading both the start and stop clock, so the elapsed
    value reflects kernel execution, not kernel launch.

        with CudaTimer() as t:
            run_some_gpu_work()
        print(t.seconds)
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.seconds: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "CudaTimer":
        torch.cuda.synchronize(self.device)
        self._start = _now()
        return self

    def __exit__(self, *exc) -> None:
        torch.cuda.synchronize(self.device)
        self.seconds = _now() - self._start


def _now() -> float:
    # perf_counter is monotonic and high resolution; time.time() is neither.
    import time

    return time.perf_counter()


# ---------------------------------------------------------------------------
# Result record + logging
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """One measured run. Written as a single CSV row.

    Keep this flat (no nested objects) so the CSV stays trivially loadable in
    pandas for plotting.
    """

    engine: str            # "hf" | "vllm"
    model: str
    dtype: str
    batch_size: int
    prompt_tokens: int
    new_tokens: int        # tokens actually generated (the decode count)

    prefill_seconds: float
    decode_seconds: float
    decode_tokens_per_sec: float
    prefill_tokens_per_sec: float

    weights_vram_mib: float        # VRAM after load, before any generation
    peak_allocated_mib: float
    peak_reserved_mib: float

    oom: bool = False              # did this configuration OOM
    note: str = ""

    gpu_name: str = field(default_factory=lambda: _gpu_name())
    torch_version: str = field(default_factory=lambda: torch.__version__)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def write_result(result: BenchResult, csv_path: str) -> None:
    """Append a result row, writing the header if the file is new."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = asdict(result)
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Analytical KV-cache size (Phase 0 formula, reused by the OOM sweep)
# ---------------------------------------------------------------------------

def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,   # fp16
    batch: int = 1,
) -> int:
    """KV-cache size in bytes.

    2 (K and V) x layers x kv_heads x head_dim x seq_len x dtype_bytes x batch.

    num_kv_heads (not num_attention_heads) is the right count under grouped-query
    attention, where several query heads share one KV head. Qwen2.5-1.5B uses
    GQA, so getting this wrong inflates the prediction.
    """
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes * batch


# ---------------------------------------------------------------------------
# Environment banner (printed at the top of every run for reproducibility)
# ---------------------------------------------------------------------------

def print_env() -> None:
    print("=" * 60)
    print(f"python      {platform.python_version()}")
    print(f"torch       {torch.__version__}")
    print(f"cuda avail  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu         {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"capability  {cap[0]}.{cap[1]}")
        print(f"cuda        {torch.version.cuda}")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0:
            print(f"vram total  {smi.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("=" * 60)
