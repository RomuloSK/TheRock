import torch, sys
variant=sys.argv[1]
x=torch.randn(4096,4096,device="cuda",dtype=torch.half); w=torch.randn(4096,4096,device="cuda",dtype=torch.half)
buf=torch.zeros(1,64,4096,device="cuda",dtype=torch.half); pos=torch.zeros(1,dtype=torch.long,device="cuda")
def step():
    y=x[:1]@w; buf.index_copy_(1,pos,y[None]); pos.add_(1)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): step()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g): step()
torch.cuda.synchronize()
if variant=="eager_big":   # eager large GEMM + allocations after capture
    for n in (512,1024,2048):
        a=torch.randn(n,4096,device="cuda",dtype=torch.half); (a@w).sum().item()
if variant=="eager_attn":
    import torch.nn.functional as F
    q=torch.randn(1,32,512,64,device="cuda",dtype=torch.half); F.scaled_dot_product_attention(q,q,q,is_causal=True).sum().item()
g.replay(); torch.cuda.synchronize(); print(variant,"ok")
