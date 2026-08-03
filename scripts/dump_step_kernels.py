"""Week 6 Part B: print one complete decode step, kernel by kernel.

`analyze_trace.py` reduces a step to three numbers. Those numbers are only
believable if the thing behind them is inspectable, so this script does the
opposite job: it takes one decode step out of a trace and prints every device
event in it, in order, with the aten op that launched it and the idle gap in
front of it. 1198 kernels is an assertion until you can scroll the list.

Three things it shows that the summary cannot.

1. The repeating block. A decoder layer is a fixed kernel sequence repeated once
   per layer, so the 1198 kernels are really about 42 kernels seen 28 times plus
   a head and a tail. The collapsed view is how you read the step as a model
   rather than as a list.

2. Where the time is. 46% of this step is one GEMV template, and the per-type
   table separates instance count from time so a kernel that runs 348 times and
   costs nothing does not get confused with one that runs 57 times and costs
   half the step.

3. Why the gaps are the CPU's fault, not the queue's. For each gap the script
   checks whether the CPU had even issued the next launch yet when the device
   went idle. If it had not, the device was starved. That is the direct evidence
   for "launch bound", as opposed to inferring it from a gap total.

On the clock domains: CPU (cuda_runtime) and device (kernel) timestamps come
from different sources, aligned by CUPTI to the same wall clock with a residual
skew of order a microsecond. The starvation test compares one against the other,
so it is only trustworthy because the gaps here are ~21 us, twenty times the
skew. Do not carry the same test to a trace whose gaps are sub-microsecond.

Attribution is to the enclosing aten op, not to the module. profile_decode.py
runs with `with_modules` off (stack capture would tax the CPU side, which is the
quantity under test), so nothing here can say "layers.3.mlp.gate_proj". The
layer index in the collapsed view comes from counting block repeats, and shapes
have to be reasoned about from the model config rather than read off the trace.

Usage:
    python scripts/dump_step_kernels.py --trace results/trace_hf_decode.json.gz
    python scripts/dump_step_kernels.py --trace results/trace_hf_decode.json.gz \
        --out results/hf_decode_step_kernels.txt
    # a vLLM trace has no annotations, so segment on a once-per-pass kernel
    python scripts/dump_step_kernels.py --trace results/trace_vllm-graph.json.gz \
        --segment kernel --marker-kernel _prepare_pos_seq_lens_kernel --step 3
"""

from __future__ import annotations

import argparse
import bisect
import re
import sys
from collections import Counter, defaultdict

from analyze_trace import (
    DEVICE_CATS,
    KERNEL_CATS,
    RUNTIME_CATS,
    bucket,
    is_graph_launch,
    is_launch,
    load_events,
    merged_span,
    step_starts,
)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

# Template wrappers that name nothing. cutlass::Kernel2<the_real_name> and
# internal::gemvx::kernel are both dispatch shells, so printing the last
# qualified segment alone would label half the table "kernel".
GENERIC_SEGMENTS = {"kernel", "Kernel", "Kernel2", "Kernel3", "kernel_impl", "impl", "type"}


def split_top_level(s: str, sep: str) -> list[str]:
    """Split on `sep` only where angle and paren nesting depth is zero."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        elif c == sep and depth == 0:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return out


def short_kernel_name(name: str) -> str:
    """Collapse a C++ kernel symbol to the identifier a human would quote.

    Kernel names in a torch trace are full template instantiations, hundreds of
    characters of nested type arguments. Three things have to survive the
    shortening or the table lies:

    - The distinguishing template parameters. gemv2T_kernel_val<...,4,4,false,
      false,...> and <...,4,4,false,true,...> are a 98 us kernel and a 20 us
      kernel, so the bools matter as much as the tile numbers and dropping them
      would merge two different rows into one.
    - The real name behind a dispatch shell. cutlass::Kernel2 carries its
      identity in its template argument, not in its name.
    - The anonymous namespace, which has to be removed FIRST. It contains
      parentheses at nesting depth zero, so a scan for the argument list stops
      inside it and truncates the symbol to nothing.
    """
    n = str(name).replace("(anonymous namespace)::", "")

    # Drop a return type: "void X", "std::enable_if<!(false), void>::type X".
    # The split has to ignore spaces inside <> and (), which is why enable_if
    # needs a nesting-aware split rather than a str.split.
    n = split_top_level(n, " ")[-1].strip()

    # Drop the function argument list, keep template arguments.
    depth = 0
    for i, c in enumerate(n):
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "(" and depth == 0:
            n = n[:i]
            break

    qualified, targs = n, ""
    lt = n.find("<")
    if lt != -1:
        qualified, inner = n[:lt], n[lt + 1 : n.rfind(">")]
        args = [a.strip() for a in split_top_level(inner, ",")]

        # A single identifier template argument IS the kernel name, as in
        # cutlass::Kernel2<cutlass_80_wmma_tensorop_...>.
        if len(args) == 1 and re.fullmatch(r"[A-Za-z_][\w]*", args[0]) and len(args[0]) > 12:
            return args[0][:56]

        # Scalar parameters at depth zero: the tile shape, the unroll factors,
        # and the bool flags. Prefer these, because they belong to this kernel
        # rather than to some nested functor type.
        scalars = [a for a in args if re.fullmatch(r"\d+u?|true|false", a)]
        # Flash traits nest their tile inside Flash_fwd_kernel_traits<...>, so
        # with nothing useful at depth zero, fall back to any numbers found.
        if sum(1 for a in scalars if a[0].isdigit()) < 2:
            scalars = re.findall(r"(?<![\w:])(\d+|true|false)(?![\w:])", inner)
        if scalars:
            targs = "<" + ",".join(scalars[:6]) + ">"

    segs = [s for s in qualified.split("::") if s]
    if not segs:
        return (qualified + targs)[:56] or str(name)[:56]
    # A generic last segment needs its parent to mean anything: gemvx::kernel,
    # not kernel.
    short = segs[-1]
    if short in GENERIC_SEGMENTS and len(segs) > 1:
        short = f"{segs[-2]}::{short}"
    return (short + targs)[:56]


def op_of_launch(launch: dict, ops_by_tid: dict[int, list[dict]]) -> str:
    """Innermost aten op enclosing a launch call, plus the outer op if useful.

    A launch sits inside a nest of cpu_op events on the same thread. The
    innermost one names the actual operation (aten::mm), and the outermost
    interesting one names the intent (aten::linear). Both are worth printing,
    because "aten::mul inside aten::linear" and a bare "aten::mul" are different
    facts about the model.
    """
    ops = ops_by_tid.get(launch["tid"], [])
    if not ops:
        return ""
    lo, hi = launch["ts"], launch["ts"] + launch["dur"]
    starts = [o["ts"] for o in ops]
    # Candidates start at or before the launch. The list is sorted by ts, so
    # walk back from the insertion point rather than scanning 23k events.
    i = bisect.bisect_right(starts, lo)
    enclosing = [o for o in ops[max(0, i - 64) : i] if o["ts"] + o["dur"] >= hi]
    if not enclosing:
        return ""
    inner = min(enclosing, key=lambda o: o["dur"])["name"]
    outer = max(enclosing, key=lambda o: o["dur"])["name"]
    inner = inner.replace("aten::", "")
    outer = outer.replace("aten::", "")
    return f"{outer}/{inner}" if outer != inner else inner


# ---------------------------------------------------------------------------
# One step, assembled
# ---------------------------------------------------------------------------

class Row:
    """One device event, with everything needed to read it in context."""

    __slots__ = ("idx", "launch_ts", "gpu_ts", "dur", "gap", "wait", "starved", "op", "kernel", "raw")

    def __init__(self, idx, launch_ts, gpu_ts, dur, gap, wait, starved, op, kernel, raw):
        self.idx = idx
        self.launch_ts = launch_ts
        self.gpu_ts = gpu_ts
        self.dur = dur
        self.gap = gap
        self.wait = wait
        self.starved = starved
        self.op = op
        self.kernel = kernel
        self.raw = raw


def build_step(events: list[dict], starts: list[float], want: int) -> tuple[list[Row], dict]:
    """Collect step `want` as an ordered list of rows, plus step-level facts."""
    ops_by_tid: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("cat") == "cpu_op":
            ops_by_tid[e["tid"]].append(e)
    for lst in ops_by_tid.values():
        lst.sort(key=lambda o: o["ts"])

    corr_to_launch: dict[int, dict] = {}
    for e in events:
        if e.get("cat") in RUNTIME_CATS and "correlation" in e.get("args", {}):
            corr_to_launch[e["args"]["correlation"]] = e

    # Device events belong to the step whose window contains their LAUNCH, not
    # their own timestamp. A kernel can run well after the launch that issued
    # it, so timestamp bucketing misattributes work at every step boundary.
    picked: list[tuple[float, dict, dict | None]] = []
    fell_back = 0
    for e in events:
        if e.get("cat") not in DEVICE_CATS:
            continue
        launch = corr_to_launch.get(e.get("args", {}).get("correlation"))
        s = bucket(starts, launch["ts"]) if launch is not None else bucket(starts, e["ts"])
        if launch is None:
            fell_back += 1
        if s == want:
            picked.append((launch["ts"] if launch is not None else e["ts"], e, launch))

    if not picked:
        raise SystemExit(f"step {want} holds no device events")

    # Order by device time: this is the timeline the GPU actually saw, which is
    # what the gaps are gaps in. Launch order is printed alongside and is nearly
    # always the same order on a single stream.
    picked.sort(key=lambda t: t[1]["ts"])
    # ONE origin for both the CPU and the device column, the step's start on the
    # CPU timeline. Torch aligns CUPTI device timestamps into the CPU clock
    # domain, so the two are directly comparable and the launch-to-execution
    # distance is readable straight off the two columns. Giving each column its
    # own origin (device events relative to the first device event) makes a
    # kernel that queued for 1.2 ms look like it ran 1.2 ms before its own
    # launch, which reads as a broken trace rather than as a deep queue.
    t0 = starts[want]

    rows: list[Row] = []
    running_end = picked[0][1]["ts"]
    for i, (lts, e, launch) in enumerate(picked):
        gap = e["ts"] - running_end
        # Starved: the device went idle and the CPU had not yet issued the
        # launch for the next kernel. The gap is then CPU latency, not queueing.
        starved = bool(gap > 0 and launch is not None and launch["ts"] > running_end)
        rows.append(
            Row(
                idx=i,
                launch_ts=lts - t0,
                gpu_ts=e["ts"] - t0,
                dur=e["dur"],
                gap=max(gap, 0.0),
                # From the launch call RETURNING to the kernel starting. Measured
                # from the call's end, not its start, because the submission
                # happens inside the call and the call itself costs 14 to 17 us
                # here. Timing from the start would report that per-launch CPU
                # cost as if it were queue latency, which is the opposite
                # conclusion. Slightly negative for the 0.3% of kernels that
                # begin before their launch call has returned.
                wait=(e["ts"] - (launch["ts"] + launch["dur"])) if launch is not None
                else float("nan"),
                starved=starved,
                op=op_of_launch(launch, ops_by_tid) if launch is not None else "",
                kernel=short_kernel_name(e["name"]),
                raw=str(e["name"]),
            )
        )
        running_end = max(running_end, e["ts"] + e["dur"])

    busy, lo, hi = merged_span([(e["ts"], e["ts"] + e["dur"]) for _, e, _ in picked])
    launches = sum(
        1 for e in events if is_launch(e) and bucket(starts, e["ts"]) == want
    )
    graph_launches = sum(
        1 for e in events if is_graph_launch(e) and bucket(starts, e["ts"]) == want
    )
    ann = [
        e for e in events
        if e.get("cat") == "user_annotation" and abs(e["ts"] - starts[want]) < 1.0
    ]

    facts = {
        "n_device": len(picked),
        "n_kernels": sum(1 for _, e, _ in picked if e.get("cat") in KERNEL_CATS),
        "n_launches": launches,
        "n_graph_launches": graph_launches,
        "busy_us": busy,
        "span_us": hi - lo,
        "annotated_us": ann[0]["dur"] if ann else 0.0,
        "corr_fallback": fell_back,
    }
    return rows, facts


# ---------------------------------------------------------------------------
# The repeating block
# ---------------------------------------------------------------------------

def find_period(tags: list[str], lo: int = 8, hi: int = 200) -> tuple[int, float]:
    """Smallest p where tags[i] == tags[i + p] holds for most of the sequence.

    A decoder layer is the same kernel sequence every time, so the tag sequence
    is periodic in the middle and irregular at the ends (embedding and the
    lm_head tail). Scoring over the middle 80% keeps the ends from hiding the
    period. Returns the best period and its match rate.
    """
    n = len(tags)
    if n < 2 * lo:
        return 0, 0.0
    a, b = int(n * 0.1), int(n * 0.9)
    best, best_score = 0, 0.0
    for p in range(lo, min(hi, (b - a) // 2) + 1):
        hits = sum(1 for i in range(a, b - p) if tags[i] == tags[i + p])
        score = hits / (b - p - a)
        # Strictly greater keeps the SMALLEST period that achieves the best
        # score. A multiple of the true period scores just as well and would
        # report two layers as one block.
        if score > best_score + 1e-9:
            best, best_score = p, score
    return best, best_score


def block_bounds(tags: list[str], period: int) -> tuple[int, int]:
    """First index from which the period holds cleanly, and the last.

    The head (embedding, position ids) and the tail (final norm, lm_head) are
    not part of any layer, and printing them inside the block would make the
    block wrong. Walk in from both ends while the periodic identity holds.
    """
    n = len(tags)
    start = 0
    while start + period < n and tags[start] != tags[start + period]:
        start += 1
    end = n - 1
    while end - period >= 0 and tags[end] != tags[end - period]:
        end -= 1
    return start, end


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

HEADER = (f"{'#':>5} {'launch us':>10} {'gpu us':>10} {'wait us':>9} "
          f"{'dur us':>9} {'gap us':>8} {'s':>1}  {'aten op':<26} kernel")


def format_rows(rows: list[Row], full_names: bool) -> list[str]:
    out = [HEADER, "-" * len(HEADER)]
    for r in rows:
        out.append(
            f"{r.idx:>5} {r.launch_ts:>10.1f} {r.gpu_ts:>10.1f} {r.wait:>9.2f} "
            f"{r.dur:>9.2f} {r.gap:>8.2f} {'*' if r.starved else ' ':>1}  "
            f"{r.op[:26]:<26} {r.raw if full_names else r.kernel}"
        )
    return out


def print_type_table(rows: list[Row], busy_us: float) -> None:
    agg: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for r in rows:
        a = agg[r.kernel]
        a[0] += 1
        a[1] += r.dur
    print(f"\n{'kernel':<44}{'n':>6}{'total ms':>11}{'% step':>9}{'us/inst':>10}")
    print("-" * 80)
    for name, (n, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{name[:44]:<44}{n:>6}{tot / 1000:>11.3f}{tot / busy_us * 100:>9.1f}{tot / n:>10.2f}")
    print("-" * 80)
    print(f"{len(agg)} distinct kernels, {len(rows)} events, {busy_us / 1000:.3f} ms device busy")


def print_block(rows: list[Row]) -> None:
    tags = [r.kernel for r in rows]
    period, score = find_period(tags)
    if not period or score < 0.9:
        print(f"\nno clean repeating block found (best period {period}, match {score:.0%})")
        return
    start, end = block_bounds(tags, period)
    n_blocks = (end - start + 1) // period
    head, tail = rows[:start], rows[start + n_blocks * period :]
    block = rows[start : start + period]
    per_block = sum(r.dur for r in block)

    print(f"\nRepeating block: {period} kernels, {n_blocks} repeats, "
          f"{score:.0%} match, {per_block / 1000:.3f} ms per repeat")
    print(f"  head before the first block: {len(head)} kernels, "
          f"{sum(r.dur for r in head) / 1000:.3f} ms")
    print(f"  tail after the last block:   {len(tail)} kernels, "
          f"{sum(r.dur for r in tail) / 1000:.3f} ms")
    print(f"  {period} x {n_blocks} + {len(head)} + {len(tail)} = "
          f"{period * n_blocks + len(head) + len(tail)} events")

    print(f"\nOne block, kernel by kernel:\n")
    print(f"{'#':>4} {'dur us':>9} {'gap us':>8}  {'aten op':<26} kernel")
    print("-" * 100)
    for i, r in enumerate(block):
        print(f"{i:>4} {r.dur:>9.2f} {r.gap:>8.2f}  {r.op[:26]:<26} {r.kernel}")
    print("-" * 100)

    if head:
        print("\nHead (before layer 0):")
        for r in head:
            print(f"{'':>4} {r.dur:>9.2f} {r.gap:>8.2f}  {r.op[:26]:<26} {r.kernel}")
    if tail:
        print("\nTail (after the last layer):")
        for r in tail:
            print(f"{'':>4} {r.dur:>9.2f} {r.gap:>8.2f}  {r.op[:26]:<26} {r.kernel}")


def print_starvation(rows: list[Row], facts: dict) -> None:
    """The launch-bound claim, as a count rather than an inference.

    Read the FRACTION here, not the milliseconds. Every gap in a profiled trace
    is inflated, because CUPTI's per-launch cost lands on the CPU side and the
    CPU side is what the device is waiting for: this step profiles at 71.8 ms
    against 40.7 ms clean, so its idle total is nearly double the published
    25.34 ms. What profiling cannot invent is the ORDER of two events. If the
    device finished a kernel before the CPU had issued the next launch, no
    amount of profiler overhead changes which came first, so the share of gaps
    that are starvation survives the overhead that their duration does not.
    """
    gaps = [r for r in rows if r.gap > 0]
    starved = [r for r in gaps if r.starved]
    gap_us = sum(r.gap for r in gaps)
    starved_us = sum(r.gap for r in starved)
    print("\nWhy the device idles")
    print("-" * 68)
    print(f"  gaps between device events:  {len(gaps)}")
    if gaps:
        print(f"  mean gap:                    {gap_us / len(gaps):.2f} us")
        print(f"  median gap:                  {sorted(r.gap for r in gaps)[len(gaps) // 2]:.2f} us")
    print(f"  total idle in this step:      {gap_us / 1000:.3f} ms, PROFILED and therefore")
    print(f"                                inflated. The publishable idle comes from")
    print(f"                                clean wall time minus busy time, not from here.")
    print(f"\n  gaps where the CPU had not yet issued the next launch:")
    print(f"    {len(starved)} of {len(gaps)} gaps, {len(starved) / max(len(gaps), 1) * 100:.1f}%")
    print(f"    ({starved_us / 1000:.3f} ms, {starved_us / max(gap_us, 1e-9) * 100:.1f}% of idle,"
          f" same inflation caveat)")
    print("  The device had finished and its next kernel did not exist yet. That")
    print("  is CPU latency, not queueing. This fraction is the direct evidence")
    print("  for launch-bound, and unlike the gap durations it is an ordering")
    print("  fact, which the profiler cannot manufacture.")

    # Second, independent reading of the same thing. If the device were the
    # bottleneck, launches would pile up and kernels would begin long after the
    # CPU issued them. A median wait near zero says the opposite: the launch is
    # what the device was waiting for, so it runs almost the moment it arrives.
    waits = sorted(r.wait for r in rows if r.wait == r.wait)
    if waits:
        n = len(waits)
        print("\n  launch return to kernel start, the same story from the queue's side:")
        print(f"    median {waits[n // 2]:.2f} us, p90 {waits[9 * n // 10]:.2f} us, "
              f"max {waits[-1]:.2f} us")
        print(f"    {sum(1 for w in waits if w < 5) / n * 100:.0f}% of kernels begin within "
              f"5 us of their launch returning,")
        print("    so the queue is empty when the work arrives. A device-bound step")
        print("    would show the reverse: launches queued far ahead of execution.")
        print("    The deep waits are the head of the step, where the CPU runs ahead")
        print("    before the first kernel lands and the device briefly has a backlog.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--step", default="median",
                   help="Step index, or 'median' to avoid the first (allocator "
                        "noise) and the last (possibly clipped).")
    p.add_argument("--segment", choices=["auto", "annotation", "graph", "kernel", "uniform"],
                   default="auto")
    p.add_argument("--marker", default="decode_step")
    p.add_argument("--marker-kernel", default="",
                   help="For traces with no annotations, e.g. vLLM. See analyze_trace.py.")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--full-names", action="store_true",
                   help="Print untruncated kernel symbols. Needed if you intend "
                        "to quote a template instantiation in the writeup.")
    p.add_argument("--head", type=int, default=60,
                   help="How many rows of the full listing to print to stdout. "
                        "The complete listing always goes to --out.")
    p.add_argument("--out", default="",
                   help="Write the complete ordered listing here.")
    args = p.parse_args()

    events = load_events(args.trace)
    starts, mode = step_starts(events, args.segment, args.marker, args.steps, args.marker_kernel)
    want = len(starts) // 2 if args.step == "median" else int(args.step)
    if not 0 <= want < len(starts):
        raise SystemExit(f"--step {want} out of range, trace has {len(starts)} steps")

    rows, facts = build_step(events, starts, want)

    print(f"\n{args.trace}")
    print(f"  segmented on:      {mode} ({len(starts)} steps in the trace), showing step {want}")
    print(f"  device events:     {facts['n_device']} ({facts['n_kernels']} kernels, "
          f"{facts['n_device'] - facts['n_kernels']} memcpy/memset)")
    print(f"  CPU launches:      {facts['n_launches']}"
          + (f" + {facts['n_graph_launches']} graph launches" if facts["n_graph_launches"] else ""))
    print(f"  device busy:       {facts['busy_us'] / 1000:.3f} ms (union of device intervals)")
    print(f"  first to last:     {facts['span_us'] / 1000:.3f} ms of device timeline")
    if facts["annotated_us"]:
        print(f"  annotated step:    {facts['annotated_us'] / 1000:.3f} ms "
              f"(PROFILED wall time, inflated. The publishable step time comes "
              f"from the clean pass in profile_decode.py.)")
    if facts["corr_fallback"]:
        print(f"  note: {facts['corr_fallback']} device events had no correlation id")

    print_type_table(rows, facts["busy_us"])
    print_block(rows)
    print_starvation(rows, facts)

    lines = format_rows(rows, args.full_names)
    if args.head:
        print(f"\nFirst {min(args.head, len(rows))} device events in order "
              f"(* = device idled waiting for the CPU):\n")
        print("\n".join(lines[: args.head + 2]))
        if len(rows) > args.head:
            print(f"... {len(rows) - args.head} more."
                  + ("" if args.out else " Pass --out to write the complete listing."))

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"# {args.trace}, step {want} of {len(starts)}, segmented on {mode}\n")
            f.write(f"# {facts['n_kernels']} kernels, {facts['n_launches']} CPU launches, "
                    f"{facts['busy_us'] / 1000:.3f} ms device busy\n")
            f.write("# gap us = device idle before this event. "
                    "s = * when the CPU had not yet issued the launch.\n")
            f.write("\n".join(lines) + "\n")
        print(f"\ncomplete listing written -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
