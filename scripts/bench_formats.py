"""Benchmark compressed tensor formats on the real decode-step shapes.

The eight (K, N) shapes come from results/roofline_step.csv. Each format runs
the same matrix product, [M, K] by [K, N], over M in {1, 8, 32, 128}:

  dense_fp16       torch.mm, the cuBLAS control
  int8             torch._int_mm plus an fp16 scaling epilogue
  nf4              bitsandbytes.matmul_4bit, block size 64
  sparse_2to4      torch.sparse semi-structured fp16, backed by cuSPARSELt
  cuda_sparse_2to4 handwritten CUDA GEMV, M=1 only

The analytical model counts the minimum representation traffic: compressed
weights and metadata or scales, input, output, and any explicit intermediate.
Dividing by this box's measured streaming-read bandwidth predicts a floor. The
ratio between measured time and that floor is the study result.

A 64 MiB buffer is streamed immediately before every sample. Without this,
repeating one small projection benchmarks an L2-resident weight, unlike a real
decode step that walks through all model weights.

Usage:
    python scripts/bench_formats.py --overwrite
    python scripts/bench_formats.py --only-shape k_proj --m 1 32 \
        --repeats 3 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from bench_common import print_env


FORMAT_ORDER = ("dense_fp16", "int8", "nf4", "sparse_2to4")
CUDA_FORMAT = "cuda_sparse_2to4"


def load_shapes(path: str, only_shape: str) -> list[dict[str, int | str]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    shapes = [
        {"projection": row["projection"], "k": int(row["rows"]), "n": int(row["cols"])}
        for row in rows
        if not only_shape or row["projection"] == only_shape
    ]
    if not shapes:
        raise SystemExit(f"no shapes selected from {path}")
    return shapes


def stream_bytes(format_name: str, m: int, k: int, n: int) -> tuple[int, str]:
    """Minimum bytes moved by the measured execution path."""
    input_bytes = m * k * (1 if format_name == "int8" else 2)
    output_fp16_bytes = m * n * 2
    if format_name == "dense_fp16":
        weight_bytes = k * n * 2
        return weight_bytes + input_bytes + output_fp16_bytes, "fp16 weights + input + output"
    if format_name == "int8":
        weight_bytes = k * n
        scale_bytes = n * 4
        # _int_mm writes int32. The measured scaling epilogue reads that
        # intermediate and one fp32 scale per output column, then writes fp16.
        int32_write_and_read = 2 * m * n * 4
        return (
            weight_bytes + scale_bytes + input_bytes + int32_write_and_read + output_fp16_bytes,
            "int8 weights + fp32 scales + input + int32 write/read + fp16 output",
        )
    if format_name == "nf4":
        packed_weight_bytes = math.ceil(k * n / 2)
        absmax_bytes = math.ceil(k * n / 64) * 4
        codebook_bytes = 16 * 4
        return (
            packed_weight_bytes + absmax_bytes + codebook_bytes + input_bytes + output_fp16_bytes,
            "NF4 weights + fp32 absmax per 64 + codebook + input + output",
        )
    if format_name in ("sparse_2to4", CUDA_FORMAT):
        value_bytes = k * n
        metadata_bytes = k * n // 8
        return (
            value_bytes + metadata_bytes + input_bytes + output_fp16_bytes,
            "half the fp16 values + 4 metadata bits per group of four + input + output",
        )
    raise ValueError(format_name)


def benchmark(
    op: Callable[[], torch.Tensor],
    flush: torch.Tensor,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        # This launch is outside the event interval but ordered before it on the
        # same stream, so the tested weight starts outside L2.
        flush.add_(0.001)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = op()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    return samples


def make_valid_2to4(n: int, k: int) -> torch.Tensor:
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    groups = weight.view(n, k // 4, 4)
    keep = groups.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, keep, True)
    groups.mul_(mask)
    return weight


def pack_2to4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a valid dense 2:4 matrix for the handwritten CUDA kernel."""
    n, k = weight.shape
    groups = weight.view(n, k // 4, 4)
    indices = groups.abs().topk(2, dim=-1).indices.sort(dim=-1).values
    values = groups.gather(-1, indices).reshape(n, k // 2).contiguous()
    codes = (indices[..., 0] | (indices[..., 1] << 2)).to(torch.uint8)
    metadata = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return values, metadata


def load_cuda_extension():
    from torch.utils.cpp_extension import load

    scripts_dir = Path(__file__).resolve().parent
    return load(
        name="formats_cuda_ext",
        sources=[
            str(scripts_dir / "formats_cuda.cpp"),
            str(scripts_dir / "formats_cuda_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def result_row(
    projection: str,
    format_name: str,
    m: int,
    k: int,
    n: int,
    bandwidth_gbs: float,
    status: str,
    note: str,
    samples: list[float] | None = None,
    max_abs_error: float | None = None,
) -> dict:
    nbytes, model_note = stream_bytes(format_name, m, k, n)
    predicted_ms = nbytes / (bandwidth_gbs * 1e9) * 1e3
    measured_ms = statistics.median(samples) if samples else None
    return {
        "projection": projection,
        "format": format_name,
        "m": m,
        "k": k,
        "n": n,
        "output_elements": m * n,
        "model_bytes": nbytes,
        "bytes_per_output": nbytes / (m * n),
        "bandwidth_gbs": bandwidth_gbs,
        "predicted_ms": predicted_ms,
        "measured_ms": measured_ms,
        "min_ms": min(samples) if samples else None,
        "max_ms": max(samples) if samples else None,
        "measured_over_predicted": measured_ms / predicted_ms if measured_ms else None,
        "effective_gbs": nbytes / measured_ms / 1e6 if measured_ms else None,
        "status": status,
        "max_abs_error": max_abs_error,
        "model_note": model_note,
        "note": note,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes-csv", default="results/roofline_step.csv")
    parser.add_argument("--csv", default="results/formats.csv")
    parser.add_argument("--bandwidth-gbs", type=float, default=341.4)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--formats", nargs="+", choices=FORMAT_ORDER, default=list(FORMAT_ORDER))
    parser.add_argument("--only-shape", default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-cuda-kernel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device.")
    if os.path.exists(args.csv) and not args.overwrite:
        raise SystemExit(f"{args.csv} exists, pass --overwrite to replace it")
    if any(value <= 0 for value in args.m):
        raise SystemExit("all M values must be positive")

    torch.manual_seed(args.seed)
    print_env()
    print(f"bandwidth   {args.bandwidth_gbs:.1f} GB/s measured achievable")
    print(f"cache flush {args.flush_mib} MiB before every sample")
    shapes = load_shapes(args.shapes_csv, args.only_shape)
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 2, device="cuda", dtype=torch.float16
    ).normal_()
    extension = None if args.skip_cuda_kernel else load_cuda_extension()
    rows: list[dict] = []

    for shape in shapes:
        projection = str(shape["projection"])
        k, n = int(shape["k"]), int(shape["n"])
        print(f"\n{projection}: [M, {k}] x [{k}, {n}]")

        if "dense_fp16" in args.formats:
            # Match nn.Linear and the traced Qwen weights: store [N, K] and
            # multiply through a transpose view. cuBLAS dispatch differs at
            # M=1 for some large shapes if B is allocated directly as [K, N].
            weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
            for m in args.m:
                inp = torch.randn(m, k, device="cuda", dtype=torch.float16)
                samples = benchmark(
                    lambda: torch.mm(inp, weight.t()), flush, args.warmup, args.repeats
                )
                row = result_row(projection, "dense_fp16", m, k, n, args.bandwidth_gbs,
                                 "ok", "torch.mm (cuBLAS)", samples)
                rows.append(row)
                print(f"  dense_fp16      M={m:<3} {row['measured_ms']:.4f} ms")
                del inp
            del weight
            torch.cuda.empty_cache()

        if "int8" in args.formats:
            # The value range keeps the int32 accumulation representable in
            # fp16 after scaling. Values do not change tensor-core dispatch.
            weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
            scales = torch.rand(n, device="cuda", dtype=torch.float32)
            for m in args.m:
                if m <= 16:
                    note = "torch._int_mm requires M greater than 16 on torch 2.11.0+cu128"
                    rows.append(result_row(projection, "int8", m, k, n, args.bandwidth_gbs,
                                           "unsupported", note))
                    print(f"  int8             M={m:<3} unsupported by torch._int_mm")
                    continue
                inp = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
                output = torch.empty(m, n, device="cuda", dtype=torch.float16)

                def int8_op() -> torch.Tensor:
                    accum = torch._int_mm(inp, weight.t())
                    return torch.mul(accum, scales, out=output)

                try:
                    samples = benchmark(int8_op, flush, args.warmup, args.repeats)
                    row = result_row(projection, "int8", m, k, n, args.bandwidth_gbs,
                                     "ok", "torch._int_mm + fp16 scaling epilogue", samples)
                except RuntimeError as exc:
                    row = result_row(projection, "int8", m, k, n, args.bandwidth_gbs,
                                     "unsupported", str(exc).splitlines()[0])
                rows.append(row)
                if row["status"] == "ok":
                    print(f"  int8             M={m:<3} {row['measured_ms']:.4f} ms")
                else:
                    print(f"  int8             M={m:<3} unsupported")
                del inp, output
            del weight, scales
            torch.cuda.empty_cache()

        if "nf4" in args.formats:
            import bitsandbytes as bnb

            dense_weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
            packed, quant_state = bnb.functional.quantize_4bit(
                dense_weight, blocksize=64, quant_type="nf4"
            )
            del dense_weight
            for m in args.m:
                inp = torch.randn(m, k, device="cuda", dtype=torch.float16)
                op = lambda: bnb.matmul_4bit(inp, packed.t(), quant_state=quant_state)
                samples = benchmark(op, flush, args.warmup, args.repeats)
                row = result_row(projection, "nf4", m, k, n, args.bandwidth_gbs,
                                 "ok", "bitsandbytes.matmul_4bit, block size 64", samples)
                rows.append(row)
                print(f"  nf4              M={m:<3} {row['measured_ms']:.4f} ms")
                del inp
            del packed, quant_state
            torch.cuda.empty_cache()

        if "sparse_2to4" in args.formats:
            dense_weight = make_valid_2to4(n, k)
            sparse_weight = torch.sparse.to_sparse_semi_structured(dense_weight)
            for m in args.m:
                inp = torch.randn(m, k, device="cuda", dtype=torch.float16)
                op = lambda: torch.mm(inp, sparse_weight.t())
                try:
                    samples = benchmark(op, flush, args.warmup, args.repeats)
                    row = result_row(projection, "sparse_2to4", m, k, n,
                                     args.bandwidth_gbs, "ok",
                                     "torch sparse semi-structured (cuSPARSELt)", samples)
                except RuntimeError as exc:
                    row = result_row(projection, "sparse_2to4", m, k, n,
                                     args.bandwidth_gbs, "unsupported",
                                     str(exc).splitlines()[0])
                rows.append(row)
                if row["status"] == "ok":
                    print(f"  sparse_2to4      M={m:<3} {row['measured_ms']:.4f} ms")
                else:
                    print(f"  sparse_2to4      M={m:<3} unsupported")
                del inp

            if extension is not None and 1 in args.m:
                values, metadata = pack_2to4(dense_weight)
                inp = torch.randn(k, device="cuda", dtype=torch.float16)
                reference = torch.mv(dense_weight, inp)
                actual = extension.sparse_2to4_gemv(inp, values, metadata)
                max_abs_error = (reference - actual).abs().max().item()
                samples = benchmark(
                    lambda: extension.sparse_2to4_gemv(inp, values, metadata),
                    flush,
                    args.warmup,
                    args.repeats,
                )
                row = result_row(projection, CUDA_FORMAT, 1, k, n, args.bandwidth_gbs,
                                 "ok", "handwritten CUDA 2:4 GEMV, 256 threads/output",
                                 samples, max_abs_error)
                rows.append(row)
                print(f"  cuda_sparse_2to4 M=1   {row['measured_ms']:.4f} ms, "
                      f"max abs error {max_abs_error:.4f}")
                del values, metadata, inp, reference, actual

            del dense_weight, sparse_weight
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
