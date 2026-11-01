import os, sys, time, json, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
from fast_linear import convert; from fast_decode import GraphedDecoder
name=os.environ.get("MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
prompts=["Explain what a GPU does in one paragraph.","The capital of France is","def fibonacci(n):","List three uses of a compiler:","Once upon a time in a small village"]
P=[tok(p,return_tensors="pt").input_ids.cuda() for p in prompts]
with torch.no_grad(): ref=[m.generate(p,max_new_tokens=48,do_sample=False,repetition_penalty=1.0,top_k=None,top_p=None,temperature=None)[0,p.shape[1]:] for p in P]
if os.environ.get("FAST_LINEAR")=="1": convert(m)
dec=GraphedDecoder(m,max_len=1024,compile=os.environ.get("COMPILE")=="1",use_graph=os.environ.get("GRAPH","1")=="1")
res=[]
for p,r in zip(P,ref):
    torch.cuda.synchronize(); t=time.perf_counter(); o=dec.generate(p,48)[0]; torch.cuda.synchronize(); dt=time.perf_counter()-t
    k=min(len(r),len(o)); agree=(o[:k]==r[:k]).float().mean().item(); res.append((round(agree,3),round(dt,2)))
print(json.dumps({"agree_vs_hf_generate_and_seconds_per_prompt":res}))
