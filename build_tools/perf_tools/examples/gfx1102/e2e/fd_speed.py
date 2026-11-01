import os, sys, time, json, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
from fast_linear import convert; from fast_decode import GraphedDecoder
name=os.environ.get("MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
ids=tok("The history of computing is a story of abstraction layers. "*100,return_tensors="pt").input_ids[:,:512].cuda()
def tm(f):
    torch.cuda.synchronize(); t=time.perf_counter(); r=f(); torch.cuda.synchronize(); return time.perf_counter()-t,r
with torch.no_grad():
    m.generate(ids,max_new_tokens=4,do_sample=False)
    a,_=tm(lambda:m.generate(ids,max_new_tokens=1,do_sample=False)); b,ref=tm(lambda:m.generate(ids,max_new_tokens=129,min_new_tokens=129,do_sample=False))
    stock=(b-a)/128*1e3; ref=ref[0,512:]
convert(m); dec=GraphedDecoder(m,max_len=768,compile=True)
dec.generate(ids,4)  # compile + capture
a,_=tm(lambda:dec.generate(ids,1)); b,o=tm(lambda:dec.generate(ids,129)); fork=(b-a)/128*1e3
agree=(o[0][:128]==ref[:128]).float().mean().item()
print(json.dumps({"model":name,"stock_eager_ms_per_token":round(stock,2),"fork_helper_ms_per_token":round(fork,2),"speedup":round(stock/fork,2),"token_agreement_128":round(agree,4)}))
