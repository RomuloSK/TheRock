import argparse, json, sys, torch
p = argparse.ArgumentParser()
p.add_argument("--m", type=int, default=2048); p.add_argument("--n", type=int, default=2048)
p.add_argument("--k", type=int, default=2048); p.add_argument("--dtype", default="float16")
p.add_argument("--iters", type=int, default=200); p.add_argument("--check", action="store_true")
a = p.parse_args()
dt = getattr(torch, a.dtype)
g = torch.Generator(device="cuda").manual_seed(1)
A = torch.randn(a.m, a.k, device="cuda", dtype=dt, generator=g)
B = torch.randn(a.k, a.n, device="cuda", dtype=dt, generator=g)
if a.check:
    ref = (A.double() @ B.double())
    out = (A @ B).double()
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    tol = 2e-2 if dt != torch.float32 else 1e-3
    print(json.dumps({"max_rel_err": err}), file=sys.stderr)
    sys.exit(0 if err < tol else 1)
for _ in range(20): A @ B
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(a.iters): A @ B
e.record(); torch.cuda.synchronize()
ms = s.elapsed_time(e) / a.iters
print(json.dumps({"latency_ms": ms, "tflops": 2*a.m*a.n*a.k/ms/1e9}))
