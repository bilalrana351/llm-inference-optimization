"""Check the handwritten 2:4 GEMV against an fp32 reference.

Every path multiplies the same pruned fp16 weight by the same fp16 vector:

  cuda_sparse_2to4 handwritten CUDA GEMV
  sparse_2to4      torch semi-structured fp16, backed by cuSPARSELt
  dense_fp16       torch.mv on the pruned dense matrix, the cuBLAS control

The reference is the same product computed in fp32 from those fp16 values. An
fp64 product is also computed to show that the fp32 reference itself is exact
enough to judge an fp16 result.

For each output y_i, the error is compared with the bound expected from fp32
accumulation followed by one rounding to fp16:

  |y_i - ref_i| <= u16 * |ref_i| + K * u32 * (|W| |x|)_i

where u16 = 2^-11 and u32 = 2^-24 are the unit roundoffs. The script also
reports the normwise relative error max|e| / max|ref| and the componentwise
error |e_i| / (|W| |x|)_i, which stays meaningful when ref_i is near zero.

The one-time weight preparation for both sparse paths is timed separately,
because neither benchmark included it in the per-call time.

Usage:
    python scripts/check_accuracy.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

import torch

from bench_common import print_env
from bench_formats import load_cuda_extension, load_shapes, make_valid_2to4, pack_2to4

U16 = 2.0**-11
U32 = 2.0**-24


def setup_ms(fn, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def error_stats(actual: torch.Tensor, ref: torch.Tensor, scale: torch.Tensor,
                k: int) -> dict[str, float]:
    err = (actual.double() - ref.double()).abs()
    bound = U16 * ref.double().abs() + k * U32 * scale
    return {
        "max_abs_error": err.max().item(),
        "normwise_rel_error": (err.max() / ref.double().abs().max()).item(),
        "max_componentwise_error": (err / scale).max().item(),
        "max_error_over_bound": (err / bound).max().item(),
        "fraction_within_bound": (err <= bound).double().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes-csv", default="results/roofline_step.csv")
    parser.add_argument("--csv", default="results/accuracy.csv")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device.")
    if os.path.exists(args.csv) and not args.overwrite:
        raise SystemExit(f"{args.csv} exists, pass --overwrite to replace it")

    print_env()
    extension = load_cuda_extension()
    shapes = load_shapes(args.shapes_csv, "")
    gpu = torch.cuda.get_device_name()
    rows: list[dict] = []

    for shape in shapes:
        projection = str(shape["projection"])
        k, n = int(shape["k"]), int(shape["n"])
        print(f"\n{projection}: W [{n}, {k}], x [{k}]")

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            weight = make_valid_2to4(n, k)
            x = torch.randn(k, device="cuda", dtype=torch.float16)

            ref32 = torch.mv(weight.float(), x.float())
            ref64 = torch.mv(weight.double(), x.double())
            scale = torch.mv(weight.double().abs(), x.double().abs())
            ref32_vs_64 = ((ref32.double() - ref64).abs() / scale).max().item()

            values, metadata = pack_2to4(weight)
            sparse_weight = torch.sparse.to_sparse_semi_structured(weight)
            outputs = {
                "cuda_sparse_2to4": extension.sparse_2to4_gemv(x, values, metadata),
                "sparse_2to4": torch.mm(x[None, :], sparse_weight.t())[0],
                "dense_fp16": torch.mv(weight, x),
            }

            setup = {}
            if seed == 0:
                setup = {
                    "cuda_sparse_2to4": setup_ms(lambda: pack_2to4(weight)),
                    "sparse_2to4": setup_ms(
                        lambda: torch.sparse.to_sparse_semi_structured(weight)),
                    "dense_fp16": 0.0,
                }

            for path, actual in outputs.items():
                stats = error_stats(actual, ref32, scale, k)
                rows.append({
                    "projection": projection, "k": k, "n": n, "seed": seed,
                    "path": path, **stats,
                    "ref32_vs_ref64_componentwise": ref32_vs_64,
                    "max_abs_ref": ref32.abs().max().item(),
                    "setup_ms": setup.get(path, ""),
                    "gpu_name": gpu, "torch_version": torch.__version__,
                })
                if seed == 0:
                    print(f"  {path:<17} max abs {stats['max_abs_error']:.4g}  "
                          f"normwise rel {stats['normwise_rel_error']:.3g}  "
                          f"err/bound {stats['max_error_over_bound']:.3g}  "
                          f"setup {setup[path]:.3g} ms")
            del weight, x, values, metadata, sparse_weight, outputs
            torch.cuda.empty_cache()

    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nworst case over {args.seeds} seeds, against the fp32 reference")
    print(f"{'projection':<10} {'path':<17} {'max abs':>9} {'normwise rel':>13} "
          f"{'err/bound':>10} {'within':>7}")
    for shape in shapes:
        for path in ("cuda_sparse_2to4", "sparse_2to4", "dense_fp16"):
            group = [r for r in rows
                     if r["projection"] == shape["projection"] and r["path"] == path]
            print(f"{shape['projection']:<10} {path:<17} "
                  f"{max(r['max_abs_error'] for r in group):>9.4g} "
                  f"{max(r['normwise_rel_error'] for r in group):>13.3g} "
                  f"{max(r['max_error_over_bound'] for r in group):>10.3g} "
                  f"{min(r['fraction_within_bound'] for r in group):>7.2%}")
    print(f"\nfp32 reference against fp64, worst componentwise error: "
          f"{max(r['ref32_vs_ref64_componentwise'] for r in rows):.3g}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
