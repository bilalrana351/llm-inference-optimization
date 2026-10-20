"""List the GPU kernels and kernel time per call for both 2:4 paths (warm L2)."""
import sys; sys.path.insert(0, "scripts")
import torch
from torch.profiler import profile, ProfilerActivity
from bench_formats import load_shapes, make_valid_2to4, load_cuda_extension, pack_2to4
ext = load_cuda_extension()
torch.manual_seed(0)
for s in load_shapes("results/roofline_step.csv", ""):
    k, n = s["k"], s["n"]
    w = make_valid_2to4(n, k); sw = torch.sparse.to_sparse_semi_structured(w)
    v, m = pack_2to4(w)
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    for name, op in [("cusparselt", lambda: torch.mm(x, sw.t())), ("custom", lambda: ext.sparse_2to4_gemv(x[0], v, m))]:
        for _ in range(10): op()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(20): op()
            torch.cuda.synchronize()
        ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        names = {}
        for e in ev: names[e.name[:60]] = names.get(e.name[:60], 0) + e.device_time / 20
        tot = sum(names.values())
        print(f"{s['projection']:<9} {name:<10} gpu kernels/call {len(ev)/20:.0f}  gpu us/call {tot:.1f}  ", "; ".join(f"{k}={v:.1f}" for k, v in names.items()))
