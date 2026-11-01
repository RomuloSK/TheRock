import torch, torch.nn.functional as F, json, itertools
from torch.nn.attention import sdpa_kernel, SDPBackend
dev="cuda"; rows=[]
def mk(b,h,hk,q,kv,d,dt,seed=1):
    g=torch.Generator(device=dev).manual_seed(seed)
    r=lambda *s: torch.randn(*s,device=dev,dtype=dt,generator=g).requires_grad_()
    return r(b,h,q,d),r(b,hk,kv,d),r(b,hk,kv,d)
def ref_run(Q,K,V,mask,causal,gqa,go):
    q,k,v=[t.detach().double().cpu().requires_grad_() for t in (Q,K,V)]
    if gqa>1: k2,v2=k.repeat_interleave(gqa,1),v.repeat_interleave(gqa,1)
    else: k2,v2=k,v
    o=F.scaled_dot_product_attention(q,k2,v2,attn_mask=None if mask is None else (mask.cpu() if mask.dtype==torch.bool else mask.cpu().double()),is_causal=causal)
    o.backward(go.double().cpu()); return o.detach(),q.grad,k.grad,v.grad
def test(name,b,h,hk,q,kv,d,dt=torch.half,causal=False,mask_kind=None):
    Q,K,V=mk(b,h,hk,q,kv,d,dt); gqa=h//hk
    mask=None
    if mask_kind=="bool": mask=torch.rand(b,1,q,kv,device=dev)>0.3; mask[...,0]=True
    if mask_kind=="float": mask=torch.randn(b,1,q,kv,device=dev,dtype=dt)
    go=torch.randn(b,h,q,d,device=dev,dtype=dt)
    r={"name":name}
    for be in (SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH):
        for t in (Q,K,V): t.grad=None
        try:
            with sdpa_kernel(be):
                if gqa>1 and be!=SDPBackend.MATH: o=F.scaled_dot_product_attention(Q,K,V,attn_mask=mask,is_causal=causal,enable_gqa=True)
                elif gqa>1: o=F.scaled_dot_product_attention(Q,K,V,attn_mask=mask,is_causal=causal,enable_gqa=True)
                else: o=F.scaled_dot_product_attention(Q,K,V,attn_mask=mask,is_causal=causal)
                o.backward(go)
            r[be.name]={"o":o.detach(),"g":[t.grad.clone() for t in (Q,K,V)]}
        except Exception as e: r[be.name]={"err":str(e)[:70]}
    ref=ref_run(Q,K,V,mask,causal,gqa,go)
    out={"name":name}
    for be in ("FLASH_ATTENTION","EFFICIENT_ATTENTION","MATH"):
        x=r[be]
        if "err" in x: out[be]="UNSUPPORTED: "+x["err"]; continue
        def rel(a,b): return ((a.double().cpu()-b).abs().max()/(b.abs().max()+1e-9)).item()
        out[be]={"fwd":rel(x["o"],ref[0]),"dq":rel(x["g"][0],ref[1]),"dk":rel(x["g"][1],ref[2]),"dv":rel(x["g"][2],ref[3]),
                 "finite":all(bool(torch.isfinite(t).all()) for t in [x["o"]]+x["g"])}
    return out
cases=[("mha_causal",1,16,16,512,512,64,dict(causal=True)),
 ("mha_noncausal",2,16,16,384,384,128,{}),
 ("odd_len",2,8,8,1000,1000,64,dict(causal=True)),
 ("gqa4_causal",2,16,4,512,512,128,dict(causal=True)),
 ("mqa_decode",4,16,1,1,2048,128,{}),
 ("bool_mask",2,8,8,256,256,64,dict(mask_kind="bool")),
 ("float_mask",2,8,8,256,256,64,dict(mask_kind="float")),
 ("hd80",1,8,8,256,256,80,dict(causal=True)),
 ("bf16",1,16,16,512,512,128,dict(dt=torch.bfloat16,causal=True)),
 ("q_ne_kv_causal",1,8,8,128,512,64,dict(causal=True))]
res=[test(n,*a,**k) for n,*a,k in cases]
print(json.dumps(res,indent=1))
