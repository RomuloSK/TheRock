import torch, sys, time
mode=sys.argv[1]
def f(x,w,g,u):
    v=x.float().pow(2).mean(-1,keepdim=True); y=(x*torch.rsqrt(v+1e-5)).to(x.dtype)*w
    return y+torch.nn.functional.silu(g)*u
cf=torch.compile(f,mode=None if mode=="default" else mode)
x=torch.randn(1,2048,device="cuda",dtype=torch.half); w=torch.randn(2048,device="cuda",dtype=torch.half); g=torch.randn(1,2048,device="cuda",dtype=torch.half); u=torch.randn_like(g)
t=time.perf_counter(); o=cf(x,w,g,u); torch.cuda.synchronize(); print(mode,"compile+run s=%.1f"%(time.perf_counter()-t),"maxdiff",(o-f(x,w,g,u)).abs().max().item())
