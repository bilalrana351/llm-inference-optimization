"""Plots for the energy study (Phase 3 study 1).

Reads results/energy_hf.csv and results/energy_vllm.csv, aggregates repeats by
median per configuration, and writes two figures:

  results/energy_per_token.png   joules per generated token against batch size
                                 for vLLM, with the batch-1 HF fp16 and NF4
                                 points shown as horizontal reference lines
  results/energy_pareto.png      the trade-off view: decode tokens/sec against
                                 joules per token, one point per configuration

Usage:
    python scripts/plot_energy.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HF_CSV = "results/energy_hf.csv"
VLLM_CSV = "results/energy_vllm.csv"


def load(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"missing {path}, skipping")
        return None
    return pd.read_csv(path)


def median_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.groupby(keys, as_index=False)[list(num)].median()


def main() -> None:
    hf = load(HF_CSV)
    vl = load(VLLM_CSV)

    # --- figure 1: joules per token against batch (vLLM), HF as reference ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if vl is not None:
        m = median_by(vl, ["batch_size"]).sort_values("batch_size")
        ax.plot(m.batch_size, m.joules_per_token_gross * 1000, "o-",
                label="vLLM fp16, gross")
        ax.plot(m.batch_size, m.joules_per_token_net * 1000, "s--",
                label="vLLM fp16, net of idle floor")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(m.batch_size))
        ax.set_xticklabels([str(int(b)) for b in m.batch_size])
    if hf is not None:
        for dtype, style in [("float16", ":"), ("nf4", "-.")]:
            rows = hf[hf.dtype == dtype]
            if len(rows):
                val = rows.joules_per_token_gross.median() * 1000
                ax.axhline(val, linestyle=style, color="gray",
                           label=f"HF {dtype} @ batch 1 ({val:.0f} mJ/tok)")
    ax.set_xlabel("batch size (concurrent sequences)")
    ax.set_ylabel("energy per generated token (mJ)")
    ax.set_title("Energy per token against batch size (RTX 3060, Qwen2.5-1.5B)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/energy_per_token.png", dpi=150)
    print("wrote results/energy_per_token.png")

    # --- figure 2: the Pareto view, throughput against energy ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if hf is not None:
        for dtype, marker in [("float16", "o"), ("nf4", "D")]:
            rows = hf[hf.dtype == dtype]
            if len(rows):
                ax.scatter(rows.decode_tokens_per_sec.median(),
                           rows.joules_per_token_gross.median() * 1000,
                           marker=marker, s=70, label=f"HF {dtype} (batch 1)")
    if vl is not None:
        m = median_by(vl, ["batch_size"]).sort_values("batch_size")
        ax.plot(m.decode_tokens_per_sec, m.joules_per_token_gross * 1000,
                "o-", color="C2", label="vLLM fp16 (batch 1 to max)")
        label_offsets = {
            64: (0, 14),
            128: (8, 7),
        }
        for _, row in m.iterrows():
            batch = int(row.batch_size)
            offset = label_offsets.get(batch, (4, 4))
            ax.annotate(f"b{int(row.batch_size)}",
                        (row.decode_tokens_per_sec,
                         row.joules_per_token_gross * 1000),
                        textcoords="offset points", xytext=offset, fontsize=7,
                        ha="center" if batch == 64 else "left",
                        va="bottom")
    ax.set_xlabel("decode throughput (tokens/sec, aggregate)")
    ax.set_ylabel("energy per generated token (mJ)")
    ax.set_title("Throughput against energy per token (RTX 3060, Qwen2.5-1.5B)")
    ax.set_xscale("log")
    ax.margins(x=0.08, y=0.08)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/energy_pareto.png", dpi=150)
    print("wrote results/energy_pareto.png")


if __name__ == "__main__":
    main()
