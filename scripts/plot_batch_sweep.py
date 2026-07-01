"""Plot the vLLM batching sweep: throughput, latency, and the tradeoff curve.

Reads results/vllm_batch_sweep.csv (written by bench_vllm_batch.py), takes the
median over repeats at each (pool_config, batch_size), and writes three figures:

  - vllm_batch_throughput.png : end-to-end throughput vs batch size (log-x), one
    line per pool config, with a vertical marker at each config's analytical KV
    wall. Climbs, then flattens.
  - vllm_batch_latency.png    : per-token latency (TPOT) and, where available,
    p95 per-request latency vs batch size. Flat, then climbs.
  - vllm_batch_tradeoff.png   : throughput vs latency, parametric in batch size.
    The "what latency do I pay for X throughput" curve.

The three regions (memory-bound / compute-bound / KV wall) are read off these
shapes; see docs/batching-results.md for the reading.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _median_by_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeats to the median at each (series, batch_size).

    A series is one GPU and pool config, e.g. "3060 realistic", so the 3060 and
    3090 runs plot as separate lines instead of being averaged together.
    """
    df = df.copy()
    df["series"] = (df["gpu_name"].str.split().str[-1] + " " + df["pool_config"])
    num_cols = [
        "throughput_out_tok_s", "decode_throughput_tok_s", "tpot_ms",
        "ttft_s", "p50_latency_s", "p95_latency_s", "kv_capacity_seqs",
    ]
    keep = ["series", "batch_size"] + [c for c in num_cols if c in df.columns]
    g = df[keep].groupby(["series", "batch_size"], as_index=False).median()
    return g.sort_values(["series", "batch_size"])


def _kv_wall(df: pd.DataFrame, series: str) -> float | None:
    """The analytical KV wall (sequences) for a series, if vLLM exposed it."""
    vals = df.loc[df["series"] == series, "kv_capacity_seqs"]
    vals = vals[vals > 0]
    return float(vals.iloc[0]) if len(vals) else None


def plot_throughput(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for series, sub in df.groupby("series"):
        ax.plot(sub["batch_size"], sub["decode_throughput_tok_s"],
                marker="o", label=series)
        wall = _kv_wall(df, series)
        # Only mark the wall when it falls inside the swept range (constrained runs).
        if wall and wall <= sub["batch_size"].max():
            ax.axvline(wall, linestyle="--", alpha=0.5,
                       color=ax.lines[-1].get_color())
            ax.text(wall, ax.get_ylim()[1] * 0.02, f" KV wall ~{wall:.0f}",
                    rotation=90, va="bottom", fontsize=8, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent sequences (batch size)")
    ax.set_ylabel("decode throughput (output tok/s)")
    ax.set_title("vLLM decode throughput vs batch size (Qwen2.5-1.5B)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_latency(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for series, sub in df.groupby("series"):
        ax.plot(sub["batch_size"], sub["tpot_ms"], marker="o", label=f"{series}: TPOT")
        if "p95_latency_s" in sub and sub["p95_latency_s"].notna().any():
            ax.plot(sub["batch_size"], sub["p95_latency_s"] * 1000,
                    marker="x", linestyle=":", label=f"{series}: p95 end-to-end")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent sequences (batch size)")
    ax.set_ylabel("latency (ms), TPOT solid, p95 dotted")
    ax.set_title("vLLM latency vs batch size (flat, then climbing)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_tradeoff(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for series, sub in df.groupby("series"):
        ax.plot(sub["tpot_ms"], sub["decode_throughput_tok_s"], marker="o", label=series)
        for _, row in sub.iterrows():
            ax.annotate(f"{int(row['batch_size'])}",
                        (row["tpot_ms"], row["decode_throughput_tok_s"]),
                        fontsize=7, alpha=0.6,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("per-token latency TPOT (ms)")
    ax.set_ylabel("decode throughput (output tok/s)")
    ax.set_title("vLLM throughput vs latency tradeoff (labels are batch size)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/vllm_batch_sweep.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"{args.csv} not found; run bench_vllm_batch.py first.")

    df = _median_by_batch(pd.read_csv(args.csv))
    plot_throughput(df, os.path.join(args.out_dir, "vllm_batch_throughput.png"))
    plot_latency(df, os.path.join(args.out_dir, "vllm_batch_latency.png"))
    plot_tradeoff(df, os.path.join(args.out_dir, "vllm_batch_tradeoff.png"))


if __name__ == "__main__":
    main()
