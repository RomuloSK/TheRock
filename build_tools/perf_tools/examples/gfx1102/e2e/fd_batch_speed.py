import os, sys, time, json, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
from fast_linear import convert; from fast_decode import GraphedDecoder
name=os.environ.get("MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
base=tok("The history of computing is a story of abstraction layers. "*100,return_tensors="pt").input_ids[:,:512].cuda()
def tm(f):
    torch.cuda.synchronize(); t=time.perf_counter(); r=f(); torch.cuda.synchronize(); return time.perf_counter()-t,r
Bs=(1,2,4,8); stock={}; fork={}
with torch.no_grad():
    for B in Bs:
        ids=base.repeat(B,1); m.generate(ids,max_new_tokens=4,do_sample=False)
        a,_=tm(lambda:m.generate(ids,max_new_tokens=1,do_sample=False)); b,_=tm(lambda:m.generate(ids,max_new_tokens=65,min_new_tokens=65,do_sample=False))
        stock[B]=(b-a)/64*1e3
convert(m); dec=GraphedDecoder(m,max_len=640,max_batch=8,compile=True)
for B in Bs:
    ids=base.repeat(B,1); dec.generate(ids,4)
    a,_=tm(lambda:dec.generate(ids,1)); b,o=tm(lambda:dec.generate(ids,65)); fork[B]=(b-a)/64*1e3
print(json.dumps({"model":name,"rows":[{"B":B,"stock_ms_step":round(stock[B],2),"fork_ms_step":round(fork[B],2),"stock_tok_s":round(B*1e3/stock[B]),"fork_tok_s":round(B*1e3/fork[B]),"speedup":round(stock[B]/fork[B],2)} for B in Bs]}))
