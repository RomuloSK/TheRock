import os, sys, json, time, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
if os.environ.get("FAST_LINEAR")=="1":
    from fast_linear import convert; convert(m)
text=("The history of computing is a story of abstraction layers. "*400)
ids=tok(text,return_tensors="pt").input_ids[:,:1800].cuda()
mode=sys.argv[1]
@torch.no_grad()
def timeit(fn,n):
    fn(); torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n
if mode=="prefill": print(json.dumps({"latency_ms":timeit(lambda:m(ids),5)}))
elif mode=="decode":
    with torch.no_grad():
        out=m.generate(ids[:,:512],max_new_tokens=8,do_sample=False)  # warm
        torch.cuda.synchronize(); t=time.perf_counter(); out=m.generate(ids[:,:512],max_new_tokens=129,min_new_tokens=129,do_sample=False); torch.cuda.synchronize()
        tt=time.perf_counter()-t
        t=time.perf_counter(); m.generate(ids[:,:512],max_new_tokens=1,do_sample=False); torch.cuda.synchronize(); t1=time.perf_counter()-t
    print(json.dumps({"latency_ms":(tt-t1)/128*1e3,"tokens":out[0,512:].tolist()[:64]}))
elif mode=="quality":
    with torch.no_grad():
        lg=m(ids).logits.float(); loss=torch.nn.functional.cross_entropy(lg[0,:-1],ids[0,1:]).item()
        g=m.generate(tok("Explain what a GPU does in one paragraph.",return_tensors="pt").input_ids.cuda(),max_new_tokens=48,do_sample=False)
    print(json.dumps({"loss":loss,"top1_hash":lg[0,-200:].argmax(-1).tolist(),"text":tok.decode(g[0],skip_special_tokens=True)}))
