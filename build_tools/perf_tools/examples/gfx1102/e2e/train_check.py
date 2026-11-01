import torch, torch.nn as nn, torch.nn.functional as F, json, time
torch.manual_seed(0); dev="cuda"; D,H,V,T=512,8,2000,512
class B(nn.Module):
    def __init__(s): super().__init__(); s.qkv=nn.Linear(D,3*D); s.o=nn.Linear(D,D); s.f=nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D)); s.n1=nn.LayerNorm(D); s.n2=nn.LayerNorm(D)
    def forward(s,x):
        b,t,_=x.shape; q,k,v=s.qkv(s.n1(x)).view(b,t,3,H,D//H).permute(2,0,3,1,4)
        x=x+s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,D)); return x+s.f(s.n2(x))
class M(nn.Module):
    def __init__(s): super().__init__(); s.e=nn.Embedding(V,D); s.b=nn.Sequential(*[B() for _ in range(4)]); s.h=nn.Linear(D,V)
    def forward(s,i): return s.h(s.b(s.e(i)))
m=M().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
g=torch.Generator(device=dev).manual_seed(1); data=torch.randint(0,V,(64,T+1),device=dev,generator=g)
losses=[]; torch.cuda.synchronize(); t0=time.perf_counter()
for i in range(60):
    x=data[(i*8)%64:(i*8)%64+8]; 
    with torch.autocast("cuda",dtype=torch.half): loss=F.cross_entropy(m(x[:,:-1]).float().view(-1,V),x[:,1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
torch.cuda.synchronize(); print(json.dumps({"first":losses[0],"last":losses[-1],"s_per_step":(time.perf_counter()-t0)/60,"l10":losses[10],"l30":losses[30]}))
