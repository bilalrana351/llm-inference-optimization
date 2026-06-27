"""Compute per-step VRAM deviations from the OOM sweep CSV.

Reads peak_allocated_mib and peak_reserved_mib, computes the difference
between each adjacent 1000-token step, and reports the average deviation
excluding zero-valued steps.
"""

import csv
import sys
from pathlib import Path


def load_sweep(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def deviations(rows: list[dict], col: str) -> list[tuple[int, int, float]]:
    """Return (from_tokens, to_tokens, delta) for each adjacent pair where delta != 0."""
    result = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        curr = rows[i]
        # Skip OOM rows: new_tokens there is mid-step, not a clean 1000-token boundary.
        if curr["oom"] == "True":
            continue
        delta = float(curr[col]) - float(prev[col])
        if delta != 0.0:
            result.append((int(prev["new_tokens"]), int(curr["new_tokens"]), delta))
    return result


def report(label: str, devs: list[tuple[int, int, float]]) -> None:
    print(f"\n{label}")
    print(f"  {'from':>8}  {'to':>8}  {'delta (MiB)':>12}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}")
    for frm, to, d in devs:
        print(f"  {frm:>8}  {to:>8}  {d:>12.1f}")
    avg = sum(d for _, _, d in devs) / len(devs)
    print(f"\n  Non-zero steps : {len(devs)}")
    print(f"  Total delta    : {sum(d for _, _, d in devs):.1f} MiB")
    print(f"  Average delta  : {avg:.2f} MiB")


def main() -> None:
    csv_path = Path(__file__).parent.parent / "results" / "oom_sweep.csv"
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])

    rows = load_sweep(str(csv_path))
    # Keep only the hf rows, sorted by new_tokens.
    rows = [r for r in rows if r["engine"] == "hf"]
    rows.sort(key=lambda r: int(r["new_tokens"]))

    alloc_devs = deviations(rows, "peak_allocated_mib")
    resv_devs  = deviations(rows, "peak_reserved_mib")

    report("peak_allocated_mib deviations", alloc_devs)
    report("peak_reserved_mib deviations",  resv_devs)


if __name__ == "__main__":
    main()
