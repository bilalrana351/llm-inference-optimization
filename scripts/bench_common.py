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


def device_used_mib(device: int = 0) -> float:
    """GPU memory used by every process on the device, in MiB, read via NVML.

    The vLLM path needs this. vLLM's v1 engine runs the model in a separate
    child process (EngineCore), so torch.cuda.memory_allocated() called from the
    parent reads 0: the weights, the reserved KV pool, and the CUDA-graph buffers
    all live in the child. NVML reports the device's actual used memory, which
    does include the child's allocations, so it is the honest number for vLLM.

    Falls back to nvidia-smi if pynvml is not importable. Returns 0.0 if neither
    works, so a measurement failure is visible as a zero rather than a crash.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        pynvml.nvmlShutdown()
        return bytes_to_mib(used)
    except Exception:
        pass
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(device)],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0:
            # nvidia-smi already reports MiB, so no conversion.
            return float(smi.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


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
# GPU power and energy (the energy study, Phase 3 study 1)
# ---------------------------------------------------------------------------

class PowerSampler:
    """Samples GPU board power on a background thread and integrates energy.

    Two independent measurements, recorded side by side so each can check the
    other:

      1. Power integration. nvmlDeviceGetPowerUsage (total board power, in
         milliwatts) is sampled every interval_s on a daemon thread, and
         joules(t0, t1) trapezoid-integrates the samples over a window given in
         time.perf_counter() seconds, the same clock every timer in this repo
         uses.
      2. The driver's own energy counter. nvmlDeviceGetTotalEnergyConsumption
         is a monotonically increasing millijoule counter maintained by the
         driver itself. It is supported on some cards and driver builds, not
         all; energy_mj() returns None when unsupported. When it exists, the
         difference of two reads is the exact energy between them, with none of
         the sampling error of method 1.

    NVML refreshes board power at its own internal rate, often slower than the
    sampling interval, so consecutive samples repeat values.
    observed_update_interval_ms() reports the median time between value
    changes; a run should be long enough to span at least ~50 such updates or
    its integral is built on too few real observations.

    Usage:
        sampler = PowerSampler()
        sampler.start()
        e0 = sampler.energy_mj()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ... gpu work ...
        torch.cuda.synchronize(); t1 = time.perf_counter()
        e1 = sampler.energy_mj()
        sampler.stop()
        joules_integrated = sampler.joules(t0, t1)
        joules_counter = (e1 - e0) / 1000.0 if e0 is not None else None
    """

    def __init__(self, device: int = 0, interval_s: float = 0.02):
        import threading

        self.device = device
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []  # (perf_counter s, watts)
        self._stop_event = threading.Event()
        self._thread: object | None = None
        self._nvml = None
        self._handle = None
        self.energy_counter_supported = False
        self.driver_version = ""

    def start(self) -> None:
        import threading

        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        try:
            self.driver_version = str(pynvml.nvmlSystemGetDriverVersion())
        except Exception:
            self.driver_version = ""
        self.energy_counter_supported = self.energy_mj() is not None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import time

        while not self._stop_event.is_set():
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append((time.perf_counter(), mw / 1000.0))
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self._handle = None

    def energy_mj(self) -> float | None:
        """Driver energy counter in millijoules, or None if unsupported."""
        if self._nvml is None or self._handle is None:
            return None
        try:
            return float(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None

    def _window(self, t0: float, t1: float) -> list[tuple[float, float]]:
        return [(t, w) for (t, w) in self.samples if t0 <= t <= t1]

    def joules(self, t0: float, t1: float) -> float:
        """Trapezoid-integrated energy over [t0, t1], in joules.

        Samples inside the window are integrated pairwise; the stubs between t0
        and the first sample, and between the last sample and t1, are filled as
        rectangles at the nearest sample's power. If no sample landed inside
        the window (a window shorter than the sampling interval), the nearest
        sample overall is used as a constant, which is the honest fallback for
        a window the instrument cannot resolve.
        """
        if t1 <= t0:
            return 0.0
        inside = self._window(t0, t1)
        if not inside:
            if not self.samples:
                return 0.0
            nearest = min(self.samples, key=lambda s: min(abs(s[0] - t0), abs(s[0] - t1)))
            return nearest[1] * (t1 - t0)
        total = inside[0][1] * (inside[0][0] - t0)
        for (ta, wa), (tb, wb) in zip(inside, inside[1:]):
            total += 0.5 * (wa + wb) * (tb - ta)
        total += inside[-1][1] * (t1 - inside[-1][0])
        return total

    def mean_watts(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        return self.joules(t0, t1) / (t1 - t0)

    def observed_update_interval_ms(self) -> float:
        """Median milliseconds between changes of the reported power value.

        This is the instrument's real resolution, as opposed to the sampling
        interval. Windows should span many of these.
        """
        changes = []
        last_t, last_w = None, None
        for t, w in self.samples:
            if last_w is not None and w != last_w:
                changes.append(t - last_t)
                last_t = t
            elif last_w is None:
                last_t = t
            last_w = w
        if not changes:
            return 0.0
        changes.sort()
        return 1000.0 * changes[len(changes) // 2]


@dataclass
class EnergyResult:
    """One measured energy run. Written as a single CSV row.

    joules_per_token_gross is decode energy over generated tokens as the wall
    sees it. joules_per_token_net subtracts the idle floor (idle_watts times
    decode seconds) so configurations with very different runtimes can be
    compared on the work itself. Counter fields are -1.0 when the driver does
    not expose the energy counter.
    """

    engine: str
    model: str
    dtype: str
    batch_size: int
    prompt_tokens: int
    new_tokens: int

    prefill_seconds: float
    decode_seconds: float
    decode_tokens_per_sec: float

    idle_watts: float
    mean_watts_decode: float
    prefill_joules: float
    decode_joules: float
    total_joules: float
    joules_per_token_gross: float
    joules_per_token_net: float

    # Which instrument produced the primary joules above. The driver's energy
    # counter is exact and preferred; power integration is the fallback and is
    # always recorded as a cross-check. On the first smoke run the two
    # disagreed 2.2x on a sub-second window because NVML power updates only
    # every ~500 ms on this card, which is why the counter is primary.
    energy_method: str = "integration"
    integrated_total_joules: float = 0.0
    counter_total_joules: float = -1.0
    energy_counter_supported: bool = False

    power_samples: int = 0
    power_update_interval_ms: float = 0.0

    device_used_mib: float = 0.0
    oom: bool = False
    note: str = ""

    gpu_name: str = field(default_factory=lambda: _gpu_name())
    driver_version: str = ""
    torch_version: str = field(default_factory=lambda: torch.__version__)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def write_energy_result(result: EnergyResult, csv_path: str) -> None:
    """Append an energy row, writing the header if the file is new."""
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
