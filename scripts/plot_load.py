"""Aggregate and plot the serving-under-load experiment."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "baseline": "baseline: seq64, memory 0.9, chunked",
    "seq16": "max_num_seqs 16",
    "memory50": "GPU memory 0.5",
    "no_chunk": "chunked prefill off",
}
ORDER = ["baseline", "seq16", "memory50", "no_chunk"]
PATTERNS = ["poisson", "bursty"]


def label(config: str) -> str:
    return LABELS.get(config, config)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "success_count",
        "error_count",
        "mean_prompt_tokens",
        "mean_output_tokens",
        "arrival_window_s",
        "run_wall_s",
        "realized_submit_rate_rps",
        "achieved_request_rate_rps",
        "output_throughput_tok_s",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "slo_ttft_ms",
        "slo_tpot_ms",
        "slo_pass_count",
        "slo_attainment_pct",
        "goodput_request_rps",
        "goodput_output_tok_s",
        "client_lag_p95_ms",
        "queue_mean",
        "queue_p95",
        "queue_max",
        "running_max",
        "kv_usage_max",
        "preemptions",
        "max_num_seqs",
        "gpu_mem_util",
    ]
    available = [column for column in metrics if column in runs.columns]
    summary = (
        runs.groupby(
            ["server_config", "arrival_pattern", "target_rate_rps"],
            as_index=False,
        )[available]
        .median()
        .sort_values(["server_config", "arrival_pattern", "target_rate_rps"])
    )
    chunked = (
        runs.groupby(
            ["server_config", "arrival_pattern", "target_rate_rps"],
            as_index=False,
        )["chunked_prefill"]
        .first()
    )
    return summary.merge(
        chunked,
        on=["server_config", "arrival_pattern", "target_rate_rps"],
        how="left",
    )


def derive_knees(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, pattern), group in summary.groupby(
        ["server_config", "arrival_pattern"]
    ):
        group = group.sort_values("target_rate_rps")
        within_slo = group[
            (group.slo_attainment_pct >= 90)
            & (group.ttft_p95_ms <= group.slo_ttft_ms)
            & (group.tpot_p95_ms <= group.slo_tpot_ms)
        ]
        overloaded = group.loc[~group.index.isin(within_slo.index)]
        peak = group.loc[group.goodput_request_rps.idxmax()]
        rows.append({
            "server_config": config,
            "arrival_pattern": pattern,
            "slo_capacity_rps": (
                within_slo.target_rate_rps.max() if len(within_slo) else float("nan")
            ),
            "first_overload_rate_rps": (
                overloaded.target_rate_rps.min() if len(overloaded) else float("nan")
            ),
            "peak_goodput_request_rps": peak.goodput_request_rps,
            "peak_goodput_offered_rate_rps": peak.target_rate_rps,
            "ttft_p95_at_slo_capacity_ms": (
                within_slo.iloc[-1].ttft_p95_ms if len(within_slo) else float("nan")
            ),
            "tpot_p95_at_slo_capacity_ms": (
                within_slo.iloc[-1].tpot_p95_ms if len(within_slo) else float("nan")
            ),
        })
    return pd.DataFrame(rows).sort_values(["arrival_pattern", "server_config"])


def plot_knees(summary: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for row, pattern in enumerate(PATTERNS):
        subset = summary[summary.arrival_pattern == pattern]
        for config in ORDER:
            group = subset[subset.server_config == config].sort_values(
                "target_rate_rps"
            )
            if group.empty:
                continue
            axes[row, 0].plot(
                group.target_rate_rps,
                group.goodput_request_rps,
                "o-",
                label=label(config),
            )
            axes[row, 1].plot(
                group.target_rate_rps,
                group.ttft_p95_ms,
                "o-",
                label=label(config),
            )
        rates = sorted(subset.target_rate_rps.unique())
        if rates:
            axes[row, 0].plot(rates, rates, "k--", alpha=0.45, label="ideal")
        slo = subset.slo_ttft_ms.iloc[0] if len(subset) else 1000
        axes[row, 1].axhline(slo, color="black", linestyle="--", alpha=0.6)
        axes[row, 0].set_ylabel(f"{pattern}\nSLO goodput (requests/sec)")
        axes[row, 1].set_ylabel(f"{pattern}\nTTFT p95 (ms)")
        for column in range(2):
            axes[row, column].grid(True, alpha=0.3)
            axes[row, column].set_xscale("log", base=2)
            axes[row, column].set_xticks(rates)
            axes[row, column].set_xticklabels([f"{rate:g}" for rate in rates])
    axes[0, 0].set_title("Goodput bends when requests miss the SLO")
    axes[0, 1].set_title("Queueing appears first in TTFT")
    axes[1, 0].set_xlabel("offered arrival rate (requests/sec)")
    axes[1, 1].set_xlabel("offered arrival rate (requests/sec)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print(f"wrote {path}")


def plot_scheduler(summary: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for row, pattern in enumerate(PATTERNS):
        subset = summary[summary.arrival_pattern == pattern]
        rates = sorted(subset.target_rate_rps.unique())
        for config in ORDER:
            group = subset[subset.server_config == config].sort_values(
                "target_rate_rps"
            )
            if group.empty:
                continue
            axes[row, 0].plot(
                group.target_rate_rps,
                group.queue_max,
                "o-",
                label=label(config),
            )
            axes[row, 1].plot(
                group.target_rate_rps,
                group.tpot_p95_ms,
                "o-",
                label=label(config),
            )
        slo = subset.slo_tpot_ms.iloc[0] if len(subset) else 100
        axes[row, 1].axhline(slo, color="black", linestyle="--", alpha=0.6)
        axes[row, 0].set_ylabel(f"{pattern}\nmax waiting requests")
        axes[row, 1].set_ylabel(f"{pattern}\nTPOT p95 (ms)")
        for column in range(2):
            axes[row, column].grid(True, alpha=0.3)
            axes[row, column].set_xscale("log", base=2)
            axes[row, column].set_xticks(rates)
            axes[row, column].set_xticklabels([f"{rate:g}" for rate in rates])
    axes[0, 0].set_title("Server queue depth")
    axes[0, 1].set_title("Per-token latency after admission")
    axes[1, 0].set_xlabel("offered arrival rate (requests/sec)")
    axes[1, 1].set_xlabel("offered arrival rate (requests/sec)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", default="results/load_runs.csv")
    parser.add_argument("--summary-csv", default="results/load_summary.csv")
    parser.add_argument("--knees-csv", default="results/load_knees.csv")
    parser.add_argument("--knees-png", default="results/load_knees.png")
    parser.add_argument("--scheduler-png", default="results/load_scheduler.png")
    args = parser.parse_args()

    runs = pd.read_csv(args.runs_csv)
    summary = aggregate(runs)
    knees = derive_knees(summary)
    summary.to_csv(args.summary_csv, index=False)
    knees.to_csv(args.knees_csv, index=False)
    print(f"wrote {args.summary_csv}")
    print(f"wrote {args.knees_csv}")
    plot_knees(summary, args.knees_png)
    plot_scheduler(summary, args.scheduler_png)


if __name__ == "__main__":
    main()
