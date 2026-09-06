"""Plot measured format speedups and analytical-model error.

Usage:
    python scripts/plot_formats.py
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "dense_fp16": "C0",
    "int8": "C1",
    "nf4": "C2",
    "sparse_2to4": "C3",
    "cuda_sparse_2to4": "C4",
}
LABELS = {
    "dense_fp16": "dense fp16 (cuBLAS)",
    "int8": "INT8 (_int_mm + scale)",
    "nf4": "NF4 (bitsandbytes)",
    "sparse_2to4": "2:4 sparse (cuSPARSELt)",
    "cuda_sparse_2to4": "2:4 sparse (handwritten CUDA)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/formats.csv")
    parser.add_argument("--shapes-csv", default="results/roofline_step.csv")
    parser.add_argument("--summary-csv", default="results/formats_summary.csv")
    parser.add_argument("--speedup-png", default="results/formats_speedup.png")
    parser.add_argument("--model-png", default="results/formats_model_error.png")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    ok = data[data.status == "ok"].copy()
    dense = ok[ok.format == "dense_fp16"][["projection", "m", "measured_ms"]].rename(
        columns={"measured_ms": "dense_ms"}
    )
    speed = ok.merge(dense, on=["projection", "m"], how="left")
    speed["speedup"] = speed.dense_ms / speed.measured_ms

    shape_rows = pd.read_csv(args.shapes_csv)
    instances = dict(zip(shape_rows.projection, shape_rows.instances_per_step))
    speed["instances_per_step"] = speed.projection.map(instances)
    summaries = []
    for (format_name, m), group in speed.groupby(["format", "m"], sort=False):
        # A step total is valid only when this format ran all eight shapes.
        if set(group.projection) != set(shape_rows.projection):
            continue
        summaries.append({
            "format": format_name,
            "m": int(m),
            "shapes": len(group),
            "weighted_predicted_ms": (
                group.predicted_ms * group.instances_per_step
            ).sum(),
            "weighted_measured_ms": (
                group.measured_ms * group.instances_per_step
            ).sum(),
            "dense_weighted_ms": (
                group.dense_ms * group.instances_per_step
            ).sum(),
        })
    summary = pd.DataFrame(summaries)
    summary["speedup_vs_dense"] = summary.dense_weighted_ms / summary.weighted_measured_ms
    summary["measured_over_predicted"] = (
        summary.weighted_measured_ms / summary.weighted_predicted_ms
    )
    summary.to_csv(args.summary_csv, index=False)
    print(f"wrote {args.summary_csv}")

    projections = list(dict.fromkeys(data.projection))
    m_values = sorted(data.m.unique())
    formats = ["int8", "nf4", "sparse_2to4", "cuda_sparse_2to4"]
    columns = [(projection, m) for projection in projections for m in m_values]
    heat = np.full((len(formats), len(columns)), np.nan)
    for i, format_name in enumerate(formats):
        for j, (projection, m) in enumerate(columns):
            row = speed[(speed.format == format_name) &
                        (speed.projection == projection) & (speed.m == m)]
            if len(row):
                heat[i, j] = row.speedup.iloc[0]

    fig, ax = plt.subplots(figsize=(15, 3.8))
    image = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    ax.set_yticks(range(len(formats)))
    ax.set_yticklabels([LABELS[name] for name in formats])
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"{p}\nM={m}" for p, m in columns], rotation=90, fontsize=7)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.2f}x", ha="center", va="center", fontsize=6)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="gray")
    ax.set_title("Compressed-format speedup over dense fp16 (RTX 3060)")
    fig.colorbar(image, ax=ax, label="speedup over cuBLAS")
    fig.tight_layout()
    fig.savefig(args.speedup_png, dpi=170)
    print(f"wrote {args.speedup_png}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = {1: "o", 8: "s", 32: "^", 128: "D"}
    for format_name, group in ok.groupby("format", sort=False):
        for m, batch in group.groupby("m"):
            ax.scatter(
                batch.predicted_ms,
                batch.measured_ms,
                color=COLORS[format_name],
                marker=markers.get(int(m), "o"),
                s=45,
                alpha=0.8,
            )
    low = min(ok.predicted_ms.min(), ok.measured_ms.min()) * 0.75
    high = max(ok.predicted_ms.max(), ok.measured_ms.max()) * 1.35
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="bytes / bandwidth")
    for format_name in ok.format.unique():
        ax.scatter([], [], color=COLORS[format_name], label=LABELS[format_name])
    for m in sorted(ok.m.unique()):
        ax.scatter([], [], color="gray", marker=markers[int(m)], label=f"M={int(m)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("predicted runtime from representation bytes (ms)")
    ax.set_ylabel("measured device runtime (ms)")
    ax.set_title("Where compressed bytes do and do not predict runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.model_png, dpi=170)
    print(f"wrote {args.model_png}")


if __name__ == "__main__":
    main()
