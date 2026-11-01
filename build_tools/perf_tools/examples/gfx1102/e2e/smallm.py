import torch, json, sys
def bench(fn, copies=8, n=300):
    for i in range(20): fn(i%copies)
    torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); s.record()
    for i in range(n): fn(i%copies)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n*1e3
rows=[]
for K,N in [(2048,2048),(2048,5632),(5632,2048),(2048,32000)]:
    Wt=[torch.randn(K,N,device="cuda",dtype=torch.half) for _ in range(8)]
    for M in (1,2,3,4,8,16):
        x=torch.randn(M,K,device="cuda",dtype=torch.half); us=bench(lambda i:x@Wt[i])
        rows.append((K,N,M,us,K*N*2/us/1e3))
last=None
for K,N,M,us,gb in rows:
    print(f"K={K:5d} N={N:5d} M={M:2d}: {us:7.1f} us  {gb:5.0f} GB/s")
