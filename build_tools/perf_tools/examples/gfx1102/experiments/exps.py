import torch, time, json, sys, os, statistics as st
dev="cuda"; out={}
def ev(fn, n):
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record(); fn(n); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n
def wall(fn):
    torch.cuda.synchronize(); t=time.perf_counter(); fn(); torch.cuda.synchronize(); return (time.perf_counter()-t)*1e3
def med(f,r=7): return st.median(f() for _ in range(r))

def e03():  # launch vs graph: chain of K tiny kernels
    res={}
    for K in (10,50,200):
        x=torch.randn(4096,device=dev)
        def chain(): 
            y=x
            for _ in range(K): y=y*1.0001+0.0001
            return y
        for _ in range(5): chain()
        direct=med(lambda: wall(lambda:[chain() for _ in range(20)]))/20
        g=torch.cuda.CUDAGraph(); torch.cuda.synchronize()
        with torch.cuda.graph(g): chain()
        for _ in range(5): g.replay()
        graph=med(lambda: wall(lambda:[g.replay() for _ in range(20)]))/20
        t=time.perf_counter(); g2=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2): chain()
        cap=(time.perf_counter()-t)*1e3
        res[f"K{K}"]={"direct_ms":direct,"graph_ms":graph,"speedup":direct/graph,"capture_ms":cap,"breakeven_replays":cap/max(direct-graph,1e-9)}
    # large kernels: graph should not matter
    A=torch.randn(4096,4096,device=dev,dtype=torch.float16)
    def big(): 
        y=A
        for _ in range(10): y=y@A*1e-3
    direct=med(lambda: wall(lambda:[big() for _ in range(3)]))/3
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): big()
    graph=med(lambda: wall(lambda:[g.replay() for _ in range(3)]))/3
    res["gemm10_4096"]={"direct_ms":direct,"graph_ms":graph,"speedup":direct/graph}
    return res

def e04():  # allocation: caching allocator vs raw sizes, churn
    res={}
    for mb in (1,16,256):
        n=mb*2**20//4
        def churn(k):
            for _ in range(k): t=torch.empty(n,device=dev); del t
        churn(5); res[f"alloc_free_{mb}MB_cached_us"]=ev(churn,200)*1e3
        def churn2(k):
            for _ in range(k):
                t=torch.empty(n+_%7*1024,device=dev); del t
        churn2(5); res[f"alloc_free_{mb}MB_varying_us"]=ev(churn2,200)*1e3
        torch.cuda.empty_cache()
        def nocache(k):
            for _ in range(k): t=torch.empty(n,device=dev); del t; torch.cuda.empty_cache()
        res[f"alloc_free_{mb}MB_nocache_us"]=ev(nocache,30)*1e3
    return res

def e05():  # copy/compute overlap, pinned vs pageable
    res={}; mb=256; n=mb*2**18
    h_page=torch.randn(n); h_pin=torch.randn(n).pin_memory(); d=torch.empty(n,device=dev)
    A=torch.randn(4096,4096,device=dev,dtype=torch.float16)
    for name,h in (("pageable",h_page),("pinned",h_pin)):
        d.copy_(h,non_blocking=True); torch.cuda.synchronize()
        res[f"h2d_{name}_GBps"]=mb/1024/(med(lambda: wall(lambda:d.copy_(h,non_blocking=True)))/1e3)
    cs=torch.cuda.Stream()
    def compute(): 
        for _ in range(8): A@A
    comp=med(lambda: wall(compute)); res["compute_alone_ms"]=comp
    for name,h in (("pageable",h_page),("pinned",h_pin)):
        cp=med(lambda: wall(lambda:d.copy_(h,non_blocking=True)))
        def both():
            with torch.cuda.stream(cs): d.copy_(h,non_blocking=True)
            compute()
        bt=med(lambda: wall(both))
        res[f"{name}_copy_ms"]=cp; res[f"{name}_overlapped_ms"]=bt; res[f"{name}_serial_sum_ms"]=cp+comp
        res[f"{name}_overlap_efficiency"]=(cp+comp-bt)/min(cp,comp)
    return res

def e06():  # attention backends / sizes (decode & prefill)
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    res={}
    for name,(b,h,q,kv,d) in {"prefill_2k":(1,32,2048,2048,128),"decode_8k":(8,32,1,8192,128),"decode_32k_b1":(1,32,1,32768,128)}.items():
        Q=torch.randn(b,h,q,d,device=dev,dtype=torch.float16);K=torch.randn(b,h,kv,d,device=dev,dtype=torch.float16);V=torch.randn_like(K)
        for be in (SDPBackend.EFFICIENT_ATTENTION,SDPBackend.FLASH_ATTENTION,SDPBackend.MATH):
            try:
                with sdpa_kernel(be):
                    f=lambda n:[F.scaled_dot_product_attention(Q,K,V,is_causal=(q>1)) for _ in range(n)]
                    f(3); ms=ev(f,20)
                res[f"{name}_{be.name}_ms"]=ms
                res[f"{name}_{be.name}_kvGBps"]=(2*b*h*kv*d*2)/ms/1e6
            except Exception as ex: res[f"{name}_{be.name}"]="unavailable: "+str(ex)[:60]
    return res

def e07():  # conv: default vs benchmark mode
    res={}
    import torch.nn as nn
    for name,(c,k,hw) in {"3x3_64":(64,64,56),"3x3_256":(256,256,28)}.items():
        m=nn.Conv2d(c,k,3,padding=1).to(dev).half(); x=torch.randn(8,c,hw,hw,device=dev,dtype=torch.half)
        for bm in (False,True):
            torch.backends.cudnn.benchmark=bm
            t=time.perf_counter(); m(x); torch.cuda.synchronize(); first=(time.perf_counter()-t)*1e3
            f=lambda n:[m(x) for _ in range(n)]
            f(5); res[f"{name}_benchmark{int(bm)}"]={"first_call_ms":first,"steady_ms":ev(f,100)}
    return res

def e01():  # prep reuse: new matmul shapes (cold) vs repeated; shape churn
    res={}
    shapes=[(m,4096,4096) for m in range(1,257,8)]
    B=torch.randn(4096,4096,device=dev,dtype=torch.half)
    t=[]; 
    for m,_,_ in shapes:
        A=torch.randn(m,4096,device=dev,dtype=torch.half); torch.cuda.synchronize(); s=time.perf_counter(); A@B; torch.cuda.synchronize(); t.append((time.perf_counter()-s)*1e3)
    res["first_call_shape_churn_ms"]={"median":st.median(t),"max":max(t),"first":t[0]}
    t=[]
    for m,_,_ in shapes:
        A=torch.randn(m,4096,device=dev,dtype=torch.half); torch.cuda.synchronize(); s=time.perf_counter(); A@B; torch.cuda.synchronize(); t.append((time.perf_counter()-s)*1e3)
    res["second_pass_ms"]={"median":st.median(t),"max":max(t)}
    return res

if __name__=="__main__":
    name=sys.argv[1]; print(json.dumps(globals()[name](),indent=1))
