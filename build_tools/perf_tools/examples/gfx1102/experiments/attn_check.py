import torch, torch.nn.functional as F, json
from torch.nn.attention import sdpa_kernel, SDPBackend
r={}
for name,(b,h,hk,q,kv,d,c) in {"prefill_2k":(1,32,32,2048,2048,128,True),"prefill_odd":(2,16,16,1000,1000,64,True),"decode_8k":(8,32,32,1,8192,128,False)}.items():
    g=torch.Generator(device="cuda").manual_seed(3)
    Q=torch.randn(b,h,q,d,device="cuda",dtype=torch.half,generator=g);K=torch.randn(b,h,kv,d,device="cuda",dtype=torch.half,generator=g);V=torch.randn(b,h,kv,d,device="cuda",dtype=torch.half,generator=g)
    ref=torch.cat([F.scaled_dot_product_attention(Q[:,i:i+4].double().cpu(),K[:,i:i+4].double().cpu(),V[:,i:i+4].double().cpu(),is_causal=c) for i in range(0,h,4)],1).cuda()
    for be in (SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION):
        with sdpa_kernel(be): o=F.scaled_dot_product_attention(Q,K,V,is_causal=c)
        with sdpa_kernel(SDPBackend.MATH): m=F.scaled_dot_product_attention(Q,K,V,is_causal=c)
        r[f"{name}_{be.name}"]={"max_abs_err_vs_fp64":(o.double()-ref).abs().max().item(),"math_fp16_err_vs_fp64":(m.double()-ref).abs().max().item(),"finite":bool(torch.isfinite(o).all())}
print(json.dumps(r,indent=1))
