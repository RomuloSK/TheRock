import argparse, json, sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
p=argparse.ArgumentParser(); p.add_argument("--mode",default="prefill"); p.add_argument("--check",action="store_true")
p.add_argument("--layers",type=int,default=8); p.add_argument("--reps",type=int,default=5); p.add_argument("--ctx",type=int,default=2048)
a=p.parse_args(); D,H,FF,V=2048,16,8192,32000; dev="cuda"; dt=torch.half
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv=nn.Linear(D,3*D,bias=False); s.o=nn.Linear(D,D,bias=False)
        s.u=nn.Linear(D,FF,bias=False); s.d=nn.Linear(FF,D,bias=False); s.n1=nn.RMSNorm(D); s.n2=nn.RMSNorm(D)
    def forward(s,x,kv=None):
        b,t,_=x.shape; q,k,v=s.qkv(s.n1(x)).view(b,t,3,H,D//H).permute(2,0,3,1,4)
        if kv is not None: k=torch.cat([kv[0],k],2); v=torch.cat([kv[1],v],2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=(kv is None))
        x=x+s.o(y.transpose(1,2).reshape(b,t,D)); return x+s.d(F.gelu(s.u(s.n2(x)))),(k,v)
class LM(nn.Module):
    def __init__(s,n):
        super().__init__(); s.e=nn.Embedding(V,D); s.b=nn.ModuleList(Block() for _ in range(n)); s.h=nn.Linear(D,V,bias=False)
    def forward(s,ids,kvs=None):
        x=s.e(ids); new=[]
        for i,b in enumerate(s.b): x,kv=b(x,None if kvs is None else kvs[i]); new.append(kv)
        return s.h(x),new
torch.manual_seed(0); m=LM(a.layers).to(dev,dt).eval()
ids=torch.randint(0,V,(1,a.ctx),device=dev)

L=a.ctx+64
@torch.no_grad()
def go(graph):
    _,kvs=m(ids)
    K=[torch.zeros(1,H,L,D//H,device=dev,dtype=dt) for _ in kvs]; Vv=[torch.zeros_like(k) for k in K]
    for i,(k,v) in enumerate(kvs): K[i][:,:,:a.ctx]=k; Vv[i][:,:,:a.ctx]=v
    pos=torch.tensor([a.ctx],device=dev); tok=ids[:,-1:].clone()
    ar=torch.arange(L,device=dev)
    def step():
        x=m.e(tok)
        for i,b in enumerate(m.b):
            q,k,v=b.qkv(b.n1(x)).view(1,1,3,H,D//H).permute(2,0,3,1,4)
            K[i].index_copy_(2,pos,k); Vv[i].index_copy_(2,pos,v)
            mask=(ar<=pos).view(1,1,1,L)
            y=F.scaled_dot_product_attention(q,K[i],Vv[i],attn_mask=mask)
            x=x+b.o(y.transpose(1,2).reshape(1,1,D)); x=x+b.d(F.gelu(b.u(b.n2(x))))
        out=m.h(x); tok.copy_(out.argmax(-1)); pos.add_(1); return out
    for _ in range(3): step()
    pos.fill_(a.ctx)
    if graph:
        g=torch.cuda.CUDAGraph(); torch.cuda.synchronize()
        with torch.cuda.graph(g): step()
        pos.fill_(a.ctx); fn=g.replay
    else: fn=step
    n=64; s_,e_=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s_.record()
    for _ in range(n): fn()
    e_.record(); torch.cuda.synchronize(); return s_.elapsed_time(e_)/n
import os
print(json.dumps({"latency_ms": go(os.environ.get("GRAPH")=="1")}))
