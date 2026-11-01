import os, sys, json, time, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
if os.environ.get("FAST_LINEAR")=="1":
    from fast_linear import convert; convert(m)
ids=tok("The history of computing is a story of abstraction layers. "*100,return_tensors="pt").input_ids[:,:512].cuda()
mode=os.environ.get("MODE","eager")
kw=dict(do_sample=False,cache_implementation="static") if mode!="eager" else dict(do_sample=False)
if mode=="compile":
    m.forward=torch.compile(m.forward,mode="reduce-overhead",fullgraph=False)
def gen(n): 
    torch.cuda.synchronize(); t=time.perf_counter(); o=m.generate(ids,max_new_tokens=n,min_new_tokens=n,**kw); torch.cuda.synchronize(); return time.perf_counter()-t,o
with torch.no_grad():
    t0=time.perf_counter()
    for _ in range(3): gen(16)
    warm=time.perf_counter()-t0
    a,_=gen(1); b,o=gen(129)
print(json.dumps({"mode":mode,"warmup_s":warm,"latency_ms":(b-a)/128*1e3,"tokens":o[0,512:512+32].tolist()}))
