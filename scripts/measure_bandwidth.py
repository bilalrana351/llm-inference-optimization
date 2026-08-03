"""Measure this GPU's two ceilings: memory bandwidth and compute throughput.

Every "% of peak bandwidth" claim in this repo divides by a peak. Phase 1 used
360 GB/s, the RTX 3060 12GB spec figure, which is correct only if the card's
memory runs at its stock 15 Gbps. It does not always. The Vast.ai listing for the
box that produced the Week 6 profiling traces reports 287 GB/s, and 192 bits at
12 Gbps is exactly 288 GB/s, so that card's memory is roughly 20% below stock.

That is not a footnote. Device busy time for the same decode work came out
uniformly 1.22x higher on that box than on the earlier one (spread 1.4% across
four runs), and 360/287 is 1.25. Decode is memory-bound, so the throughput
difference between two nominally identical 3060s is mostly this number.

Two quantities, and they are not interchangeable:

  theoretical  memory clock x 2 (GDDR6 transfers twice per clock) x bus width,
               computed from what NVML reports the card is actually set to, not
               from the model name.
  achievable   what a large streaming access actually reaches. Always lower.
               80 to 90% of theoretical is the normal range.

Report the ratio between them so a "% of peak" claim elsewhere can say which peak
it means. Quoting 85% of theoretical when the achievable ceiling is 88% tells a
very different story from 85% of achievable.

The read number is the one that matters for decode. A batch-1 decode step reads
every weight once and writes almost nothing, so it is a near-pure streaming read.

The compute ceiling is here because a roofline needs both terms. Dividing FLOP/s
by GB/s gives the ridge point, the arithmetic intensity at which a kernel stops
being memory-bound, and both halves of docs/gate-phase2.md compare a kernel
against it: a batch-1 GEMV sits at 1.0 FLOP/byte and a tiled matmul at T/4. With
the bandwidth measured and the FLOP/s quoted from a spec sheet, the ridge was
half measurement and half hearsay.

Usage:
    python scripts/measure_bandwidth.py
    python scripts/measure_bandwidth.py --gib 3.0 --iters 50
    python scripts/measure_bandwidth.py --no-flops        # bandwidth only
"""

from __future__ import annotations

import argparse

import torch

from bench_common import print_env


def theoretical_bandwidth(device: int = 0) -> tuple[float, str]:
    """Peak GB/s from the clocks NVML reports, not from the model name.

    GDDR6 moves data on both clock edges, so the per-pin data rate is twice the
    reported memory clock. Peak bytes/sec is that rate times the bus width in
    bits, divided by 8.

    Returns (GB/s, description). Returns (0.0, reason) if NVML cannot supply the
    numbers, because a wrong peak is worse than an absent one.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        try:
            bus_bits = pynvml.nvmlDeviceGetMemoryBusWidth(h)
        except AttributeError:
            # Older pynvml has no accessor. 192 bits is the RTX 3060 12GB bus;
            # flagged in the description so a wrong card is visible in the output.
            bus_bits = 192
        pynvml.nvmlShutdown()
        gbps = mem_clock_mhz * 2 / 1000.0
        peak = gbps * bus_bits / 8.0
        return peak, f"{mem_clock_mhz} MHz x2 = {gbps:.1f} Gbps/pin over {bus_bits} bits"
    except Exception as exc:
        return 0.0, f"NVML unavailable ({type(exc).__name__})"


def measure_flops(n: int, iters: int, device: int) -> list[tuple[str, float, str]]:
    """Achievable FLOP/s for a large square GEMM, per precision and math mode.

    This exists for the same reason the bandwidth measurement does. Both halves
    of docs/gate-phase2.md place kernels against a roofline ridge point, and a
    ridge point is peak FLOP/s over achievable bandwidth. The bandwidth term is
    measured; until this function existed the FLOP/s term was a spec sheet
    figure, which is exactly the kind of number that already sent one section of
    docs/profiling.md down a wrong explanation.

    A large GEMM is the right probe because its arithmetic intensity is
    2N/(3*bytes_per_element), roughly 680 FLOP/byte at n=4096 in fp32, which is
    an order of magnitude above any plausible ridge point. So the measurement is
    compute-bound by construction and reports the compute ceiling rather than a
    mixture of the two.

    TF32 is forced off for the fp32 row. Ampere silently routes fp32 matmul
    through tensor cores at reduced mantissa by default, so a naive fp32
    measurement here would report the tensor core rate under an fp32 label, and
    the SIMT ridge point that a cuBLAS GEMV actually lives under would come out
    several times too high.
    """
    out = []
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    flop = 2.0 * n * n * n
    try:
        for label, dtype, tf32, note in (
            ("fp32 (SIMT)", torch.float32, False, "TF32 off, the SIMT ridge"),
            ("tf32 (tensor)", torch.float32, True, "same fp32 call, tensor cores"),
            ("fp16 (tensor)", torch.float16, False, "fp32 accumulate"),
        ):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            a = torch.randn(n, n, dtype=dtype, device=f"cuda:{device}")
            b = torch.randn(n, n, dtype=dtype, device=f"cuda:{device}")
            s = time_op(lambda: torch.matmul(a, b), iters, device)
            out.append((label, flop / s / 1e12, note))
            del a, b
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return out


def time_op(fn, iters: int, device: int) -> float:
    """Median seconds per call, synchronizing around the whole batch.

    Median rather than mean: a single scheduling hiccup should not move the
    reported bandwidth.
    """
    import time

    for _ in range(3):  # warmup, discarded
        fn()
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gib", type=float, default=2.0,
        help="Buffer size per tensor. Must be far larger than L2 (a few MB on "
        "sm_86) or the benchmark measures cache, not memory.",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--flops-n", type=int, default=4096,
        help="Square GEMM size for the compute ceiling. Must be large enough "
        "that the GEMM is compute-bound, which it is by a wide margin at 4096.",
    )
    parser.add_argument("--no-flops", action="store_true",
                        help="Skip the compute ceiling and measure bandwidth only.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device.")

    print_env()
    dev = args.device

    peak, how = theoretical_bandwidth(dev)
    print(f"\ntheoretical peak   {peak:8.1f} GB/s   ({how})")

    n = int(args.gib * (1024 ** 3) / 2)  # fp16 elements
    x = torch.randn(n, dtype=torch.float16, device=f"cuda:{dev}")
    y = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    print(f"buffer             {nbytes / 1e9:8.3f} GB per tensor, fp16")

    # Copy moves the buffer twice, once read and once written.
    copy_s = time_op(lambda: y.copy_(x), args.iters, dev)
    copy_bw = 2 * nbytes / copy_s / 1e9

    # Sum reads the buffer once and writes a scalar, so it is the read-only
    # case, and it is the pattern batch-1 decode actually has.
    read_s = time_op(lambda: torch.sum(x), args.iters, dev)
    read_bw = nbytes / read_s / 1e9

    print(f"\n{'op':10}{'ms':>10}{'GB/s':>10}{'% theo':>9}")
    print("-" * 39)
    print(f"{'copy':10}{copy_s * 1000:10.3f}{copy_bw:10.1f}"
          f"{copy_bw / peak * 100 if peak else 0:9.1f}")
    print(f"{'read':10}{read_s * 1000:10.3f}{read_bw:10.1f}"
          f"{read_bw / peak * 100 if peak else 0:9.1f}")
    print("-" * 39)

    best = max(copy_bw, read_bw)
    if peak:
        print(f"\nachievable / theoretical = {best / peak * 100:.1f}%")
        if best / peak > 0.95:
            print("  Suspiciously high. Check that the buffer really exceeds L2.")
        elif best / peak < 0.70:
            print("  Low. Check for another tenant on the card, or a power cap.")
    print("\nUse the READ figure as the denominator for decode bandwidth claims:")
    print("  a batch-1 decode step reads every weight once and writes almost nothing.")
    print(f"  achievable read bandwidth on this box: {read_bw:.1f} GB/s")

    if not args.no_flops:
        print(f"\ncompute ceiling, {args.flops_n}x{args.flops_n} GEMM "
              f"({2.0 * args.flops_n ** 3 / 1e9:.1f} GFLOP per call)")
        print(f"{'precision':16}{'TFLOP/s':>10}{'ridge':>10}  note")
        print("-" * 62)
        for label, tflops, note in measure_flops(args.flops_n, max(args.iters // 3, 5), dev):
            # Ridge point: the arithmetic intensity at which a kernel stops being
            # memory-bound. Both halves of the gate doc quote one of these.
            print(f"{label:16}{tflops:>10.2f}{tflops * 1e12 / (read_bw * 1e9):>10.1f}  {note}")
        print("-" * 62)
        print("  ridge = FLOP/byte at which compute and memory take equal time,")
        print(f"  computed against the measured {read_bw:.1f} GB/s read bandwidth above.")
        print("  A kernel below its ridge is memory-bound: a batch-1 GEMV sits at 1.0,")
        print("  and a tiled matmul with tile T sits at T/4. See docs/gate-phase2.md.")


if __name__ == "__main__":
    main()
