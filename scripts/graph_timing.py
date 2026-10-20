"""Time both 2:4 paths and cuBLAS with and without host overhead.

Each op is timed exactly as bench_formats.py does (L2 flush, 5 warmups,
median of 20 CUDA-event samples), once called eagerly and once replayed from
a CUDA graph, which removes CPU dispatch from the timed window. CPU launch
time per eager call is measured separately.
"""
import sys, time, statistics; sys.path.insert(0, "scripts")
import torch, csv
rows = []
from bench_formats import load_shapes, make_valid_2to4, load_cuda_extension, pack_2to4, benchmark
ext = load_cuda_extension()
torch.manual_seed(0)
flush = torch.empty(64 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16).normal_()
def cpu_us(op, n=200):
    for _ in range(20): op()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): op()
    cpu = (time.perf_counter() - t) / n * 1e6
    torch.cuda.synchronize(); return cpu
def graphed(op):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): op()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): op()
    return lambda: g.replay()
print(f"{'shape':<9} {'path':<10} {'event_us':>9} {'graph_event_us':>15} {'cpu_launch_us':>14}")
tot = {}
inst = {"lm_head": 1}
for s in load_shapes("results/roofline_step.csv", ""):
    k, n = s["k"], s["n"]
    w = make_valid_2to4(n, k); sw = torch.sparse.to_sparse_semi_structured(w); v, m = pack_2to4(w)
    x = torch.randn(1, k, device="cuda", dtype=torch.float16); x1 = x[0].contiguous()
    for name, op in [("cusparselt", lambda: torch.mm(x, sw.t())), ("custom", lambda: ext.sparse_2to4_gemv(x1, v, m)), ("cublas", lambda: torch.mm(x, w.t()))]:
        ev = statistics.median(benchmark(op, flush, 5, 20)) * 1e3
        try:
            gev = statistics.median(benchmark(graphed(op), flush, 5, 20)) * 1e3
        except Exception as e:
            gev = float("nan"); print("graph failed", name, str(e)[:100])
        cpu = cpu_us(op)
        w_ = inst.get(s["projection"], 28)
        tot.setdefault(name, [0, 0]); tot[name][0] += w_ * ev; tot[name][1] += w_ * gev
        print(f"{s['projection']:<9} {name:<10} {ev:>9.1f} {gev:>15.1f} {cpu:>14.1f}")
        rows.append(dict(projection=s["projection"], k=k, n=n, path=name, event_us=ev, graph_event_us=gev, cpu_launch_us=cpu, gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__))
for k_, (a, b) in tot.items(): print(f"step total {k_:<10} event {a/1e3:.2f} ms   graph {b/1e3:.2f} ms")

with open("results/graph_timing.csv","w",newline="") as f:
    wr=csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
