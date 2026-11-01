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
@torch.no_grad()
def run():
    if a.check:
        o,_=m(ids)
        with sdpa_kernel(SDPBackend.MATH): r,_=m(ids)
        err=((o.float()-r.float()).abs().max()/r.float().abs().max()).item()
        agree=(o.argmax(-1)==r.argmax(-1)).float().mean().item()
        print(json.dumps({"rel_err":err,"argmax_agree":agree}),file=sys.stderr); sys.exit(0 if err<2e-2 and agree>0.98 else 1)
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    if a.mode=="prefill":
        m(ids); torch.cuda.synchronize(); s.record()
        for _ in range(a.reps): m(ids)
        e.record(); torch.cuda.synchronize(); print(json.dumps({"latency_ms":s.elapsed_time(e)/a.reps,"tok_s":a.ctx*a.reps/s.elapsed_time(e)*1e3}))
    else:
        _,kvs=m(ids); step=ids[:,-1:]; n=64
        for _ in range(4): m(step,kvs)   # (kv grows; fine)
        _,kvs=m(ids); torch.cuda.synchronize(); s.record()
        for _ in range(n): o,kvs=m(step,kvs)
        e.record(); torch.cuda.synchronize(); print(json.dumps({"latency_ms":s.elapsed_time(e)/n,"tok_s":n/s.elapsed_time(e)*1e3}))
run()
