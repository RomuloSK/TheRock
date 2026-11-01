import torch, triton, triton.language as tl, json, sys
@triton.jit
def gemv_kernel(x_ptr, w_ptr, y_ptr, K, N, stride_wk, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1)
    offs_n = pn*BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    k_per = tl.cdiv(K, SPLIT_K)
    k0 = pk*k_per
    for k in range(k0, min(k0+k_per, K), BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.).to(tl.float32)
        w = tl.load(w_ptr + offs_k[:, None]*stride_wk + offs_n[None, :], mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.).to(tl.float32)
        acc += tl.sum(w * x[:, None], axis=0)
    if SPLIT_K == 1: tl.store(y_ptr + offs_n, acc.to(tl.float16), mask=offs_n < N)
    else: tl.atomic_add(y_ptr + offs_n, acc, mask=offs_n < N)
def gemv(x, w, bn=128, bk=32, sk=1):
    K, N = w.shape
    y = torch.zeros(N, device=x.device, dtype=torch.float32 if sk > 1 else torch.float16)
    gemv_kernel[(triton.cdiv(N, bn), sk)](x, w, y, K, N, w.stride(0), BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk)
    return y.half()
def bench(f, n=400, copies=8, shape=None):
    K, N = shape; ws = [torch.randn(K, N, device="cuda", dtype=torch.half) for _ in range(copies)]; x = torch.randn(K, device="cuda", dtype=torch.half)
    ref = x.double() @ ws[0].double()
    out = f(x, ws[0]); err = ((out.double()-ref).abs().max()/ref.abs().max()).item()
    for i in range(20): f(x, ws[i % copies])
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for i in range(n): f(x, ws[i % copies])
    e.record(); torch.cuda.synchronize(); ms = s.elapsed_time(e)/n
    return ms*1e3, K*N*2/ms/1e6, err
if __name__ == "__main__":
    for shape in [(4096, 4096), (4096, 11008), (11008, 4096), (2048, 8192), (8192, 2048)]:
        base = bench(lambda x, w: x @ w, shape=shape); print(shape, "torch  us=%.1f GBps=%.0f err=%.1e" % base)
        best = None
        for bn in (64, 128, 256):
            for bk in (16, 32, 64):
                for sk in (1, 2, 4, 8):
                    try: r = bench(lambda x, w: gemv(x, w, bn, bk, sk), shape=shape)
                    except Exception as ex: continue
                    if r[2] < 5e-3 and (best is None or r[0] < best[0][0]): best = (r, (bn, bk, sk))
        print(shape, "triton us=%.1f GBps=%.0f err=%.1e" % best[0], "cfg bn,bk,sk=", best[1], "speedup=%.2fx" % (base[0]/best[0][0]))
