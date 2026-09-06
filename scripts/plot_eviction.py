"""Plot the KV-cache eviction quality, memory, and speed curves."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "full": "full cache",
    "sink_window": "sinks + recent window",
    "score": "attention score + recent",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-csv", default="results/eviction_quality.csv")
    parser.add_argument("--speed-csv", default="results/eviction_speed.csv")
    parser.add_argument("--quality-summary-csv", default="results/eviction_quality_summary.csv")
    parser.add_argument("--speed-summary-csv", default="results/eviction_speed_summary.csv")
    parser.add_argument("--quality-png", default="results/eviction_quality.png")
    parser.add_argument("--systems-png", default="results/eviction_systems.png")
    args = parser.parse_args()

    quality = pd.read_csv(args.quality_csv)
    full = quality[quality.policy == "full"].iloc[0]
    quality["perplexity_delta"] = quality.perplexity - full.perplexity
    quality["perplexity_delta_pct"] = 100 * quality.perplexity_delta / full.perplexity
    quality["cache_saved_pct"] = 100 * (1 - quality.cache_mib / full.cache_mib)
    quality.to_csv(args.quality_summary_csv, index=False)
    print(f"wrote {args.quality_summary_csv}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy, group in quality[quality.policy != "full"].groupby("policy"):
        group = group.sort_values("budget")
        ax.plot(group.budget, group.perplexity, "o-", label=LABELS[policy])
    ax.axhline(full.perplexity, color="black", linestyle="--",
               label=f"full cache ({full.perplexity:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(quality[quality.budget > 0].budget.unique()))
    ax.set_xticklabels([str(int(x)) for x in ax.get_xticks()])
    ax.set_xlabel("retained KV-cache tokens")
    ax.set_ylabel("WikiText-2 perplexity (lower is better)")
    ax.set_title("Quality against KV-cache budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.quality_png, dpi=170)
    print(f"wrote {args.quality_png}")

    speed = pd.read_csv(args.speed_csv)
    med = speed.groupby(["policy", "budget", "context_tokens"], as_index=False).agg(
        decode_ms=("decode_ms", "median"),
        decode_tokens_per_sec=("decode_tokens_per_sec", "median"),
        cache_tokens=("cache_tokens", "median"),
        cache_mib=("cache_mib", "median"),
        allocated_above_model_mib=("allocated_above_model_mib", "median"),
        reserved_mib=("reserved_mib", "median"),
        peak_above_model_mib=("peak_above_model_mib", "median"),
    )
    full_speed = med[med.policy == "full"].set_index("context_tokens")
    med["speedup_vs_full"] = med.apply(
        lambda row: row.decode_tokens_per_sec
        / full_speed.loc[row.context_tokens, "decode_tokens_per_sec"],
        axis=1,
    )
    med["cache_saved_pct_vs_full"] = med.apply(
        lambda row: 100
        * (1 - row.cache_mib / full_speed.loc[row.context_tokens, "cache_mib"]),
        axis=1,
    )
    med.to_csv(args.speed_summary_csv, index=False)
    print(f"wrote {args.speed_summary_csv}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (policy, budget), group in med.groupby(["policy", "budget"]):
        label = LABELS[policy] if policy == "full" else f"window {int(budget)}"
        group = group.sort_values("context_tokens")
        axes[0].plot(group.context_tokens, group.cache_mib, "o-", label=label)
        axes[1].plot(group.context_tokens, group.decode_tokens_per_sec, "o-", label=label)
    contexts = sorted(med.context_tokens.unique())
    context_labels = [f"{int(x / 1000)}k" if x >= 1000 else str(int(x)) for x in contexts]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts)
        ax.set_xticklabels(context_labels)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlabel("original context length")
    axes[0].set_ylabel("live KV-cache memory (MiB)")
    axes[0].set_title("Bounded cache flattens memory growth")
    axes[1].set_xlabel("original context length")
    axes[1].set_ylabel("decode tokens/sec")
    axes[1].set_title("Bounded cache flattens decode slowdown")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.systems_png, dpi=170)
    print(f"wrote {args.systems_png}")


if __name__ == "__main__":
    main()
