"""Week 6 Part A: reduce a decode trace to the three numbers that carry the claim.

Same parser for HF and for vLLM, on purpose. The comparison is only honest if
"idle gap" means the identical thing on both sides, so both engines go through
this file rather than through two ad-hoc notebooks.

The three numbers, per decode step:

  GPU kernels    how many kernels actually execute on the device
  CPU launches   how many launch calls the CPU makes to cause them
  idle gap       device time inside the step with no device work running

The pair that carries the CUDA-graph claim is the second and third. A graph does
not reduce the kernel count: the same kernels run, from the same resident cubins,
in the same order. What collapses is the CPU side, from one launch per kernel to
one cudaGraphLaunch for the whole step, and the gaps close as a consequence. If
you expected vLLM's GPU kernel count to drop to one, the trace will look wrong
until you internalise that.

Busy time is the UNION of device intervals, not the sum. Summing double counts
anything concurrent and can hand you a busy time larger than wall time, which is
the standard way this analysis quietly goes wrong.

The published gap uses --clean-step-ms, the unprofiled wall time from
profile_decode.py pass 1. Kernel durations survive the profiler (CUPTI does not
slow a kernel down), wall time does not, so each term comes from the run where it
is trustworthy. Without --clean-step-ms the script still reports a gap, computed
against the profiled span, and labels it as biased.

Usage:
    python scripts/analyze_trace.py \
        --trace hf=results/trace_hf_decode.json.gz \
        --trace vllm-eager=results/trace_vllm_eager.json.gz \
        --trace vllm-graph=results/trace_vllm_graph.json.gz \
        --clean-step-ms hf=41.2 \
        --figure results/decode_timeline.png
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass

# Device-side work. Memcpy and memset occupy the device just as a kernel does, so
# they count toward busy time, but only real kernels count toward kernel count.
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
KERNEL_CATS = {"kernel"}
# The CPU-side CUDA API. Torch labels these "cuda_runtime"; some builds emit
# "cuda_driver" for the same calls.
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}

LAUNCH_PREFIXES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchCooperativeKernel")
GRAPH_PREFIXES = ("cudaGraphLaunch", "cuGraphLaunch")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_events(path: str) -> list[dict]:
    """Read a chrome trace and return its complete ("X") events.

    ts and dur are microseconds throughout this file, because that is what torch
    writes and converting early only creates rounding questions later.
    """
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]


def is_launch(e: dict) -> bool:
    return e.get("cat") in RUNTIME_CATS and str(e.get("name", "")).startswith(LAUNCH_PREFIXES)


def is_graph_launch(e: dict) -> bool:
    return e.get("cat") in RUNTIME_CATS and str(e.get("name", "")).startswith(GRAPH_PREFIXES)


# ---------------------------------------------------------------------------
# Segmentation: which events belong to which decode step
# ---------------------------------------------------------------------------

def kernel_marker_starts(events: list[dict], needle: str) -> list[float]:
    """CPU-timeline start of each forward pass, found via a once-per-pass kernel.

    For vLLM in eager mode there is nothing to segment on: no record_function
    annotations (the model runs in the EngineCore child, out of reach) and no
    cudaGraphLaunch. But any kernel that fires exactly once per forward pass, an
    embedding lookup for instance, is itself a step boundary. Use --list-kernels
    to find one.

    The boundary has to land on the CPU timeline, because it is also used to
    bucket runtime events and count launches, and the two timelines are not the
    same clock domain in a useful way here. So we take the marker kernel's
    correlation id and use the timestamp of the runtime call that issued it,
    which is the launch that began the step.
    """
    corr_to_launch_ts = {
        e["args"]["correlation"]: e["ts"]
        for e in events
        if e.get("cat") in RUNTIME_CATS and "correlation" in e.get("args", {})
    }
    out = []
    for e in events:
        if e.get("cat") not in KERNEL_CATS or needle not in str(e.get("name", "")):
            continue
        ts = corr_to_launch_ts.get(e.get("args", {}).get("correlation"))
        out.append(ts if ts is not None else e["ts"])
    return sorted(out)


def list_kernels(events: list[dict], label: str) -> None:
    """Print kernel names by occurrence count, rarest first.

    A kernel that appears once per forward pass shows up with a count equal to
    the number of passes in the trace, so the low-count end of this list is where
    the segmentation markers are.
    """
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        if e.get("cat") in KERNEL_CATS:
            counts[str(e.get("name", ""))] += 1
    print(f"\n[{label}] kernel names by count (rarest first, candidates for --marker-kernel):")
    for name, n in sorted(counts.items(), key=lambda kv: kv[1])[:25]:
        print(f"  {n:6d}  {name[:100]}")


def step_starts(
    events: list[dict], mode: str, marker: str, steps: int, marker_kernel: str
) -> tuple[list[float], str]:
    """Return the CPU-timeline start of each decode step, plus the mode used.

    Four strategies, because the engines mark themselves differently:

      annotation  HF. profile_decode.py wraps each step in record_function, and
                  those annotations only exist while the profiler is active, so
                  they segment the trace exactly.
      graph       vLLM with CUDA graphs. One cudaGraphLaunch per decode step is
                  itself the step boundary, which is a nice confirmation that the
                  mechanism is doing what it claims.
      kernel      vLLM eager. Segment on a once-per-forward-pass kernel named by
                  --marker-kernel. Prefill becomes step 0, so --trim-first 1
                  drops it.
      uniform     Last resort. Needs --steps and splits the span evenly, which is
                  only valid if the trace holds nothing but equal decode steps.
                  A vLLM trace starts with prefill, so uniform will be wrong there.
    """
    if mode == "kernel" or (mode == "auto" and marker_kernel):
        if not marker_kernel:
            raise SystemExit("--segment kernel needs --marker-kernel; try --list-kernels")
        ks = kernel_marker_starts(events, marker_kernel)
        if ks:
            return ks, "kernel"
        raise SystemExit(f"no kernel matching {marker_kernel!r}; try --list-kernels")

    if mode in ("auto", "annotation"):
        ann = sorted(
            (e for e in events if e.get("cat") == "user_annotation" and e.get("name") == marker),
            key=lambda e: e["ts"],
        )
        if ann:
            return [e["ts"] for e in ann], "annotation"
        if mode == "annotation":
            raise SystemExit(f"no '{marker}' annotations in trace; pass --segment graph or uniform")

    if mode in ("auto", "graph"):
        gl = sorted((e for e in events if is_graph_launch(e)), key=lambda e: e["ts"])
        if gl:
            return [e["ts"] for e in gl], "graph"
        if mode == "graph":
            raise SystemExit("no cudaGraphLaunch in trace; was the engine run with enforce_eager?")

    if steps <= 0:
        raise SystemExit("trace has no step markers; pass --steps N to split it uniformly")
    dev = [e for e in events if e.get("cat") in DEVICE_CATS]
    if not dev:
        raise SystemExit("trace contains no device events")
    lo = min(e["ts"] for e in dev)
    hi = max(e["ts"] + e["dur"] for e in dev)
    width = (hi - lo) / steps
    return [lo + i * width for i in range(steps)], "uniform"


def bucket(starts: list[float], ts: float) -> int | None:
    """Index of the step whose window contains ts, or None if before the first."""
    i = bisect.bisect_right(starts, ts) - 1
    return i if i >= 0 else None


def assign_device_events(
    events: list[dict], starts: list[float]
) -> tuple[dict[int, list[dict]], float]:
    """Map device events to steps, by correlation id where possible.

    A kernel runs after the launch that caused it, sometimes well after, so
    bucketing kernels by their own timestamp against CPU-side step boundaries
    misattributes work near every boundary. Correlation ids link each device
    event back to the runtime call that issued it, which is exact.

    Under graph replay every kernel in the step correlates to the single
    cudaGraphLaunch, so this stays correct there too.

    Falls back to timestamp bucketing if the trace has no usable correlation ids
    (some builds omit them), and returns the fraction that needed the fallback so
    the caller can print it rather than hide it.
    """
    corr_to_step: dict[int, int] = {}
    for e in events:
        if e.get("cat") not in RUNTIME_CATS:
            continue
        corr = e.get("args", {}).get("correlation")
        if corr is None:
            continue
        s = bucket(starts, e["ts"])
        if s is not None:
            corr_to_step[corr] = s

    by_step: dict[int, list[dict]] = defaultdict(list)
    fell_back = 0
    total = 0
    for e in events:
        if e.get("cat") not in DEVICE_CATS:
            continue
        total += 1
        s = corr_to_step.get(e.get("args", {}).get("correlation"))
        if s is None:
            s = bucket(starts, e["ts"])
            fell_back += 1
        if s is not None:
            by_step[s].append(e)

    return by_step, (fell_back / total if total else 0.0)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def merged_span(intervals: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Union duration, first start, last end, over possibly overlapping intervals."""
    if not intervals:
        return 0.0, 0.0, 0.0
    ordered = sorted(intervals)
    busy = 0.0
    last_end = ordered[0][1]
    cur_lo, cur_hi = ordered[0]
    for lo, hi in ordered[1:]:
        last_end = max(last_end, hi)
        if lo > cur_hi:
            busy += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    busy += cur_hi - cur_lo
    return busy, ordered[0][0], last_end


@dataclass
class TraceSummary:
    """One engine's decode step, reduced. All times in milliseconds."""

    label: str
    segment_mode: str
    steps: int
    gpu_kernels_per_step: float
    cpu_launches_per_step: float
    graph_launches_per_step: float
    gpu_busy_ms: float          # union of device intervals, per step
    trace_span_ms: float        # first device start to last device end, per step
    trace_gap_ms: float         # span - busy, inflated by the profiler
    clean_step_ms: float        # unprofiled wall time per step, 0 if not given
    clean_gap_ms: float         # clean_step_ms - gpu_busy_ms, the publishable one
    clean_idle_frac: float
    correlation_fallback_frac: float


def summarize(
    label: str, events: list[dict], mode: str, marker: str, steps_hint: int,
    trim_first: int, clean_step_ms: float, marker_kernel: str,
) -> tuple[TraceSummary, dict[int, list[dict]]]:
    starts, used_mode = step_starts(events, mode, marker, steps_hint, marker_kernel)
    by_step, fallback_frac = assign_device_events(events, starts)

    keep = [i for i in range(len(starts)) if i >= trim_first and by_step.get(i)]
    if not keep:
        raise SystemExit(f"[{label}] no steps left after --trim-first {trim_first}")

    launches = defaultdict(int)
    graphs = defaultdict(int)
    for e in events:
        s = bucket(starts, e["ts"])
        if s is None:
            continue
        if is_launch(e):
            launches[s] += 1
        elif is_graph_launch(e):
            graphs[s] += 1

    n_kern = busy = span = 0.0
    for i in keep:
        evs = by_step[i]
        n_kern += sum(1 for e in evs if e.get("cat") in KERNEL_CATS)
        b, lo, hi = merged_span([(e["ts"], e["ts"] + e["dur"]) for e in evs])
        busy += b
        span += hi - lo

    n = len(keep)
    busy_ms = busy / n / 1000.0
    span_ms = span / n / 1000.0
    clean_gap = (clean_step_ms - busy_ms) if clean_step_ms > 0 else 0.0

    return TraceSummary(
        label=label,
        segment_mode=used_mode,
        steps=n,
        gpu_kernels_per_step=n_kern / n,
        cpu_launches_per_step=sum(launches[i] for i in keep) / n,
        graph_launches_per_step=sum(graphs[i] for i in keep) / n,
        gpu_busy_ms=busy_ms,
        trace_span_ms=span_ms,
        trace_gap_ms=span_ms - busy_ms,
        clean_step_ms=clean_step_ms,
        clean_gap_ms=clean_gap,
        clean_idle_frac=(clean_gap / clean_step_ms) if clean_step_ms > 0 else 0.0,
        correlation_fallback_frac=fallback_frac,
    ), {i: by_step[i] for i in keep}


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def timeline_figure(
    panels: list[tuple[str, list[dict], TraceSummary]], path: str, window_us: float
) -> None:
    """A zoomed slice of one decode step per engine, device intervals as bars.

    The slice is not optional. A full HF step is tens of milliseconds and its
    features are single-digit microseconds, so at any printable width one pixel
    covers several kernels and the row renders as a solid block no matter how
    idle the device actually was. Plotting a window from the start of the step
    is the only way the gaps are visible at all.

    HF should read as a dashed comb, every tooth a few microseconds of work with
    white space after it. vLLM under graphs should read as a near-solid bar. That
    contrast is the figure the blog is missing.

    The caption carries the whole-step numbers so the slice is never mistaken for
    the measurement: the bars are an illustration, the table is the result.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 1.15 * len(panels) + 1.3), sharex=True, squeeze=False
    )
    axes = [a[0] for a in axes]

    for ax, (label, evs, s) in zip(axes, panels):
        origin = min(e["ts"] for e in evs)
        # Floor the drawn width at well under a pixel's worth of time, so a
        # sub-microsecond kernel is still visible without inflating anything that
        # would otherwise be distinguishable. Cosmetic only: every number in the
        # caption and the table comes from raw durations.
        floor = window_us / 1500.0
        bars = [
            (e["ts"] - origin, max(e["dur"], floor))
            for e in evs
            if e["ts"] - origin < window_us
        ]
        ax.broken_barh(bars, (0.1, 0.5), facecolors="#2b6cb0")
        idle = (s.clean_idle_frac * 100) if s.clean_step_ms > 0 else (
            s.trace_gap_ms / s.trace_span_ms * 100 if s.trace_span_ms else 0.0
        )
        step_ms = s.clean_step_ms if s.clean_step_ms > 0 else s.trace_span_ms
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_yticks([])
        ax.set_ylim(0, 1.05)
        ax.text(
            0.995, 0.97,
            f"whole step: {s.gpu_kernels_per_step:.0f} kernels, "
            f"{s.cpu_launches_per_step:.0f} CPU launches, "
            f"{step_ms:.1f} ms, {idle:.0f}% idle",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#4a5568",
        )

    axes[0].set_xlim(0, window_us)
    axes[-1].set_xlabel(f"microseconds from the start of one decode step (first {window_us:.0f} us)")
    axes[0].set_title("Device occupancy during one batch-1 decode step", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"figure written -> {path}")


# ---------------------------------------------------------------------------

def parse_pairs(items: list[str] | None, what: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--{what} expects LABEL=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument(
        "--clean-step-ms", action="append", metavar="LABEL=MS",
        help="Unprofiled wall time per decode step, from profile_decode.py pass 1. "
        "Without it the gap is computed from the profiled span and is biased.",
    )
    parser.add_argument("--segment", choices=["auto", "annotation", "graph", "kernel", "uniform"],
                        default="auto")
    parser.add_argument("--marker", default="decode_step")
    parser.add_argument(
        "--marker-kernel", default="",
        help="Substring of a kernel that fires exactly once per forward pass. "
        "Segments traces that have neither annotations nor graph launches, which "
        "is the vLLM eager case. Find one with --list-kernels.",
    )
    parser.add_argument(
        "--list-kernels", action="store_true",
        help="Print kernel names by count and exit, to pick a --marker-kernel.",
    )
    parser.add_argument("--steps", type=int, default=0,
                        help="Required for uniform segmentation, ignored otherwise.")
    parser.add_argument("--trim-first", type=int, default=0,
                        help="Drop leading steps. Use it to cut prefill out of a vLLM trace.")
    parser.add_argument("--csv", default="results/profile_summary.csv")
    parser.add_argument("--figure", default="")
    parser.add_argument(
        "--figure-window-us", type=float, default=1500.0,
        help="Width of the zoomed slice in the figure. A whole step is far too "
        "wide to resolve individual kernels; the table carries the real numbers.",
    )
    args = parser.parse_args()

    traces = parse_pairs(args.trace, "trace")
    clean = {k: float(v) for k, v in parse_pairs(args.clean_step_ms, "clean-step-ms").items()}

    summaries: list[TraceSummary] = []
    panels: list[tuple[str, list[dict], TraceSummary]] = []

    for label, path in traces.items():
        events = load_events(path)
        if args.list_kernels:
            list_kernels(events, label)
            continue
        s, kept = summarize(
            label, events, args.segment, args.marker, args.steps,
            args.trim_first, clean.get(label, 0.0), args.marker_kernel,
        )
        summaries.append(s)
        # Median kept step for the figure: avoids the first, which can still
        # carry allocator noise, and the last, which can be clipped.
        idx = sorted(kept)[len(kept) // 2]
        panels.append((label, kept[idx], s))

    if args.list_kernels:
        return

    header = f"{'engine':<14}{'kernels':>9}{'launches':>10}{'graphs':>8}{'busy ms':>10}{'gap ms':>9}{'idle %':>8}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        gap = s.clean_gap_ms if s.clean_step_ms > 0 else s.trace_gap_ms
        idle = s.clean_idle_frac * 100 if s.clean_step_ms > 0 else (
            s.trace_gap_ms / s.trace_span_ms * 100 if s.trace_span_ms else 0.0
        )
        print(
            f"{s.label:<14}{s.gpu_kernels_per_step:>9.0f}{s.cpu_launches_per_step:>10.0f}"
            f"{s.graph_launches_per_step:>8.0f}{s.gpu_busy_ms:>10.3f}{gap:>9.3f}{idle:>8.1f}"
        )
    print("-" * len(header))

    for s in summaries:
        notes = []
        if s.clean_step_ms <= 0:
            notes.append("gap from profiled span, inflated by profiler overhead")
        if s.segment_mode == "uniform":
            notes.append("uniform segmentation, per-step values are averages")
        if s.correlation_fallback_frac > 0.02:
            notes.append(
                f"{s.correlation_fallback_frac * 100:.0f}% of device events lacked "
                "correlation ids and were bucketed by timestamp"
            )
        for n in notes:
            print(f"  [{s.label}] {n}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    is_new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        if is_new:
            writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    print(f"\nrows appended -> {args.csv}")

    if args.figure:
        timeline_figure(panels, args.figure, args.figure_window_us)


if __name__ == "__main__":
    main()
