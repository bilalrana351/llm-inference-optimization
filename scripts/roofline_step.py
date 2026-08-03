"""Week 6 Part B: check every matmul in a decode step against the memory wall.

At batch 1 every linear layer is a matrix times a single vector, so each weight
element is read from memory, used for one multiply-add, and never touched again.
That is 2 FLOP per 2 bytes, an arithmetic intensity of 1.0, which puts the whole
decode step far on the memory-bound side of this card's roofline. If that is
true, then the time each matmul takes is predicted by one division:

    predicted us = weight bytes / achievable bandwidth

This script does that division for all seven projections in every layer plus
lm_head, and prints predicted against measured. It is the falsifiable form of
the "decode is memory-bound" claim: if the ratios came out scattered, the claim
would be wrong and no amount of arguing would save it.

The denominator is the MEASURED ceiling from scripts/measure_bandwidth.py
(291.5 GB/s on this box), not the theoretical 360 GB/s. A kernel cannot beat
what a streaming read achieves, so measuring against theoretical peak would
report every kernel as worse than it is. Both numbers carry roughly 2% of
error, so a kernel landing a couple of percent above 100% is at the wall, not
beyond it.

Mapping kernels to projections is done from structure, not from names. cuBLAS
kernel symbols do not say "gate_proj". What the trace gives is the repeating
per-layer block found by dump_step_kernels.py, and inside it the matmuls appear
in the order the architecture runs them: q, k, v, o, gate, up, down. Two
independent checks confirm the mapping rather than assuming it:

  - Qwen2 puts a bias on q, k and v only, so those three come through as
    aten::addmm and the other four as aten::mm.
  - No kernel may measure faster than its own byte floor, since bytes over
    bandwidth is a hard lower bound. If a projection appears to beat it, the
    mapping has attributed bytes to a kernel that never read them. --strict
    turns that into a non-zero exit. Being SLOWER than the floor is not an
    error, it is the result: k_proj and v_proj land 3x off because they are too
    small to saturate memory.

Usage:
    python scripts/roofline_step.py --trace results/trace_hf_decode.json.gz
    python scripts/roofline_step.py --trace results/trace_hf_decode.json.gz \
        --summary results/profile_summary.csv --csv results/roofline_step.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

from analyze_trace import KERNEL_CATS, RUNTIME_CATS, bucket, load_events, step_starts
from dump_step_kernels import block_bounds, find_period, op_of_launch, short_kernel_name

# Qwen2.5-1.5B. Weight shapes follow from these five numbers, and the check that
# they are the right five is that the per-step total has to equal the model's
# parameter bytes.
DEFAULTS = dict(hidden=1536, intermediate=8960, kv_dim=256, vocab=151936, layers=28)

# The order a Qwen2 decoder layer runs its projections, with the weight shape of
# each as (rows read, columns produced) and whether Qwen2 gives it a bias.
PROJECTIONS = [
    ("q_proj", "hidden", "hidden", True),
    ("k_proj", "hidden", "kv_dim", True),
    ("v_proj", "hidden", "kv_dim", True),
    ("o_proj", "hidden", "hidden", False),
    ("gate_proj", "hidden", "intermediate", False),
    ("up_proj", "hidden", "intermediate", False),
    ("down_proj", "intermediate", "hidden", False),
]


def collect_step(events: list[dict], starts: list[float], want: int) -> list[dict]:
    """Every kernel of one step, in device order, tagged with its aten op."""
    ops_by_tid: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("cat") == "cpu_op":
            ops_by_tid[e["tid"]].append(e)
    for lst in ops_by_tid.values():
        lst.sort(key=lambda o: o["ts"])

    corr_to_launch = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in RUNTIME_CATS and "correlation" in e.get("args", {})
    }

    out = []
    for e in events:
        if e.get("cat") not in KERNEL_CATS:
            continue
        launch = corr_to_launch.get(e.get("args", {}).get("correlation"))
        if launch is None or bucket(starts, launch["ts"]) != want:
            continue
        out.append({
            "ts": e["ts"],
            "dur": e["dur"],
            "op": op_of_launch(launch, ops_by_tid),
            "kernel": short_kernel_name(e["name"]),
        })
    out.sort(key=lambda r: r["ts"])
    return out


def layer_matmuls(rows: list[dict]) -> tuple[list[list[dict]], list[dict], int, list[list[dict]]]:
    """Per-layer lists of the matmul kernels, plus the tail, plus the block size.

    Splitting on the repeating block rather than on a name pattern means the
    same code works if a kernel is added or removed: the period changes and the
    grouping follows.
    """
    tags = [r["kernel"] for r in rows]
    period, score = find_period(tags)
    if not period or score < 0.9:
        raise SystemExit(f"no repeating per-layer block found (period {period}, match {score:.0%})")
    start, end = block_bounds(tags, period)
    n_layers = (end - start + 1) // period

    per_layer, epilogues = [], []
    for i in range(n_layers):
        blk = rows[start + i * period : start + (i + 1) * period]
        mm = [r for r in blk if r["op"].startswith("linear/")]
        # cuBLAS split-K emits a second kernel that sums the partial products,
        # and it comes through under the same aten::mm as the GEMM that needed
        # it. It is not an eighth projection. Separating it here rather than
        # dropping it keeps it in the accounting, because it is a real cost of
        # the shape that provoked it.
        per_layer.append([r for r in mm if not r["kernel"].startswith("splitKreduce")])
        epilogues.append([r for r in mm if r["kernel"].startswith("splitKreduce")])
    tail = rows[start + n_layers * period :]
    return per_layer, tail, period, epilogues


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="results/trace_hf_decode.json.gz")
    p.add_argument("--step", default="median")
    p.add_argument("--marker", default="decode_step")
    p.add_argument("--bandwidth-gbs", type=float, default=291.5,
                   help="MEASURED achievable read bandwidth, from "
                        "scripts/measure_bandwidth.py. Not the theoretical peak: "
                        "no kernel can beat a streaming read, so theoretical "
                        "would understate every kernel here.")
    p.add_argument("--bytes-per-element", type=int, default=2, help="fp16 weights.")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=int, default=v)
    p.add_argument("--summary", default="",
                   help="profile_summary.csv, to print the floor-to-measured ladder.")
    p.add_argument("--csv", default="", help="Write the per-projection table here.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if a projection measures faster than its own "
                        "byte floor, which means the kernel mapping is wrong. Being "
                        "slower than the floor is a result, not an error.")
    args = p.parse_args()

    dims = {k: getattr(args, k) for k in DEFAULTS}
    bw = args.bandwidth_gbs * 1e9
    esz = args.bytes_per_element

    events = load_events(args.trace)
    starts, mode = step_starts(events, "annotation", args.marker, 0, "")
    want = len(starts) // 2 if args.step == "median" else int(args.step)
    rows = collect_step(events, starts, want)
    per_layer, tail, period, epilogues = layer_matmuls(rows)

    n_layers = len(per_layer)
    if n_layers != dims["layers"]:
        print(f"warning: found {n_layers} repeats of the layer block but the model "
              f"config says {dims['layers']} layers")

    counts = {len(m) for m in per_layer}
    if counts != {len(PROJECTIONS)}:
        raise SystemExit(
            f"expected {len(PROJECTIONS)} matmuls per layer, found {sorted(counts)}. "
            "The kernel-to-projection mapping cannot be trusted, so nothing is printed."
        )

    print(f"\n{args.trace}, step {want} of {len(starts)}, segmented on {mode}")
    print(f"{n_layers} layers x {period}-kernel block, "
          f"{len(PROJECTIONS)} matmuls per layer")
    print(f"ceiling: {args.bandwidth_gbs:.1f} GB/s measured achievable "
          f"(scripts/measure_bandwidth.py)")

    header = (f"\n{'projection':<12}{'shape':>14}{'MB':>8}{'predicted':>11}"
              f"{'measured':>11}{'GB/s':>9}{'% ceiling':>11}{'op':>7}")
    print(header)
    print("-" * (len(header) - 1))

    table = []
    total_bytes = 0.0
    total_measured = 0.0
    bad = []

    for idx, (name, rows_dim, cols_dim, has_bias) in enumerate(PROJECTIONS):
        m, n = dims[rows_dim], dims[cols_dim]
        nbytes = m * n * esz
        # Median across layers. Layer-to-layer spread is small but a mean would
        # let one straggler move the number, and the claim is about the kernel,
        # not about the worst layer.
        durs = [layer[idx]["dur"] for layer in per_layer]
        measured = statistics.median(durs)
        ops = {layer[idx]["op"] for layer in per_layer}
        predicted = nbytes / bw * 1e6

        # Structural check: Qwen2 biases q, k and v only. If the op kind
        # disagrees with the architecture, the mapping has slipped.
        want_op = "addmm" if has_bias else "mm"
        op_ok = all(o.endswith(want_op) for o in ops)
        ratio = predicted / measured
        # Two checks of different strength. The first is physics: bytes over
        # bandwidth is a floor, so a kernel measuring meaningfully FASTER than
        # its prediction means it is not reading the bytes attributed to it, ie.
        # the mapping put a large matrix on a small kernel. The 5% allowance is
        # the joint error of the two measurements.
        if ratio > 1.05:
            bad.append(f"{name}: measured {measured:.1f} us beats its own byte floor of "
                       f"{predicted:.1f} us by {(ratio - 1) * 100:.0f}%, so this kernel "
                       f"is not reading {nbytes / 1e6:.2f} MB and the mapping is wrong")
        # The second is loose on purpose. A correctly mapped kernel can be
        # several times slower than its floor when it is too small to saturate
        # memory, which is exactly what k_proj and v_proj do here at ~3x. Only a
        # miss far beyond that suggests a misassignment.
        elif ratio < 1 / 8:
            bad.append(f"{name}: measured {measured:.1f} us against a {predicted:.1f} us "
                       f"floor, {1 / ratio:.1f}x slower, which is too far to be "
                       f"explained by low occupancy alone")
        if not op_ok:
            bad.append(f"{name}: expected aten::{want_op} (bias={has_bias}), saw {sorted(ops)}")

        total_bytes += nbytes * n_layers
        total_measured += sum(durs)
        table.append(dict(projection=name, rows=m, cols=n, mb=nbytes / 1e6,
                          predicted_us=predicted, measured_us=measured,
                          achieved_gbs=nbytes / measured * 1e6 / 1e9,
                          pct_ceiling=ratio * 100, aten_op=sorted(ops)[0],
                          instances_per_step=n_layers))
        print(f"{name:<12}{f'{m}x{n}':>14}{nbytes / 1e6:>8.2f}{predicted:>10.1f}us"
              f"{measured:>10.2f}us{nbytes / measured * 1e6 / 1e9:>9.1f}"
              f"{ratio * 100:>10.0f}%{sorted(ops)[0].split('/')[-1]:>7}")

    # lm_head lives in the tail, not in any layer, and is the largest single
    # matmul in the step by an order of magnitude.
    lm = max((r for r in tail if r["op"].startswith("linear/")),
             key=lambda r: r["dur"], default=None)
    if lm is not None:
        nbytes = dims["hidden"] * dims["vocab"] * esz
        predicted = nbytes / bw * 1e6
        ratio = predicted / lm["dur"]
        total_bytes += nbytes
        total_measured += lm["dur"]
        table.append(dict(projection="lm_head", rows=dims["hidden"], cols=dims["vocab"],
                          mb=nbytes / 1e6, predicted_us=predicted, measured_us=lm["dur"],
                          achieved_gbs=nbytes / lm["dur"] * 1e6 / 1e9,
                          pct_ceiling=ratio * 100, aten_op=lm["op"], instances_per_step=1))
        shape = f"{dims['hidden']}x{dims['vocab']}"
        print(f"{'lm_head':<12}{shape:>14}{nbytes / 1e6:>8.2f}"
              f"{predicted:>10.1f}us{lm['dur']:>10.2f}us"
              f"{nbytes / lm['dur'] * 1e6 / 1e9:>9.1f}{ratio * 100:>10.0f}%"
              f"{lm['op'].split('/')[-1]:>7}")
        if ratio > 1.0:
            print(f"  note: lm_head reads {nbytes / 1e6:.0f} MB, the longest streaming "
                  f"read in the step, and measures {ratio * 100:.0f}% of the ceiling. "
                  f"Both\n        that ceiling and this duration carry ~2% of error, so "
                  f"read it as at the wall, not past it.")

    # The split-K reduction, reported rather than absorbed. It exists only
    # because down_proj's reduction dimension is long and its output narrow, so
    # cuBLAS splits the dot products and has to add the pieces back together.
    epi_us = sum(r["dur"] for layer in epilogues for r in layer)
    n_epi = sum(len(layer) for layer in epilogues)
    if n_epi:
        med = statistics.median([r["dur"] for layer in epilogues for r in layer])
        print(f"{'  + splitKreduce':<12}{'(down_proj)':>14}{'':>8}{'':>11}"
              f"{med:>10.2f}us{'':>9}{'':>11}{'mm':>7}")
        print(f"  down_proj is the only projection that needs it: {n_epi} extra kernels, "
              f"{epi_us / 1000:.3f} ms per step.\n        Partial sums written to memory and "
              f"read back, which is why it lands furthest from the ceiling\n        of the "
              f"three 27.5 MB matmuls despite reading identical bytes.")

    busy = sum(r["dur"] for r in rows)
    n_mm = len(PROJECTIONS) * n_layers + (1 if lm is not None else 0) + n_epi
    print("-" * (len(header) - 1))
    print(f"weight bytes per token:  {total_bytes / 1e9:.3f} GB")
    print(f"floor at {args.bandwidth_gbs:.1f} GB/s:  {total_bytes / bw * 1e3:.2f} ms per token "
          f"= {bw / total_bytes:.1f} tok/s. Nothing on this card beats this at batch 1.")
    total_measured += epi_us
    print(f"in the {n_mm} matmuls:      {total_measured / 1000:.2f} ms of {busy / 1000:.2f} ms "
          f"device busy ({total_measured / busy * 100:.0f}%)")
    print(f"in the other {len(rows) - n_mm}:     {(busy - total_measured) / 1000:.2f} ms "
          f"({(busy - total_measured) / busy * 100:.0f}%), which is what fusion attacks")

    ai = 2 / esz
    print(f"\narithmetic intensity at batch 1: {ai:.1f} FLOP/byte "
          f"(one multiply-add per weight element, no reuse)")
    print(f"batching is the only lever on that number: at batch B it is {ai:.0f}B FLOP/byte "
          f"for the same weight traffic.")

    if args.summary and os.path.exists(args.summary):
        print(f"\nThe ladder, from physics to what was measured:")
        print(f"{'':<26}{'ms/token':>10}{'tok/s':>9}")
        print("-" * 45)
        floor_ms = total_bytes / bw * 1e3
        print(f"{'hardware floor':<26}{floor_ms:>10.2f}{1000 / floor_ms:>9.1f}")
        with open(args.summary) as f:
            seen = {}
            for row in csv.DictReader(f):
                seen[row["label"]] = row
        for label in ("vllm-graph", "hf"):
            if label not in seen:
                continue
            step = float(seen[label]["clean_step_ms"])
            busy_ms = float(seen[label]["gpu_busy_ms"])
            if label == "hf":
                print(f"{'HF device busy':<26}{busy_ms:>10.2f}{1000 / busy_ms:>9.1f}")
            print(f"{label + ' measured step':<26}{step:>10.2f}{1000 / step:>9.1f}")
        print(f"\nvLLM captures {floor_ms / float(seen['vllm-graph']['clean_step_ms']) * 100:.0f}% "
              f"of what the memory system can physically deliver at batch 1.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"\ntable written -> {args.csv}")

    if bad:
        print("\nMAPPING CHECKS FAILED:")
        for b in bad:
            print(f"  {b}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
