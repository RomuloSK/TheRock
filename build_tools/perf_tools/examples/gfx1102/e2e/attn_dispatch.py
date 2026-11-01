import torch, torch.nn.functional as F, json
dev="cuda"; dt=torch.half; out={}
def t(*s): return torch.randn(*s,device=dev,dtype=dt,requires_grad=True)
cases={"gqa_mask":dict(h=16,hk=4,q=256,kv=256,mask=True,gqa=True),"qne_kv_causal":dict(h=8,hk=8,q=128,kv=512,causal=True),
"bool_mask":dict(h=8,hk=8,q=256,kv=256,mask=True),"dropout":dict(h=8,hk=8,q=512,kv=512,causal=True,p=0.1),
"fully_masked_row":dict(h=4,hk=4,q=64,kv=64,allmasked=True),"len1":dict(h=8,hk=8,q=1,kv=1,causal=False)}
for n,c in cases.items():
    try:
        Q,K,V=t(2,c["h"],c["q"],64),t(2,c["hk"],c["kv"],64),t(2,c["hk"],c["kv"],64)
        mask=None
        if c.get("mask"): mask=torch.rand(2,1,c["q"],c["kv"],device=dev)>0.3; mask[...,0]=True
        if c.get("allmasked"): mask=torch.ones(2,1,64,64,device=dev,dtype=torch.bool); mask[:,:,0,:]=False
        o=F.scaled_dot_product_attention(Q,K,V,attn_mask=mask,is_causal=c.get("causal",False),dropout_p=c.get("p",0.0),enable_gqa=c.get("gqa",False))
        o.sum().backward()
        out[n]="ok finite=%s"%bool(torch.isfinite(o).all() and torch.isfinite(Q.grad).all())
    except Exception as e: out[n]="ERR "+str(e)[:60]
print(json.dumps(out,indent=1))
