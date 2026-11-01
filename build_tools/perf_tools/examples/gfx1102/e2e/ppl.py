import os, sys, json, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
name=os.environ.get("MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0"); R="/home/romuloborges/projects/ROCm/therock-perf-implementation/"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,dtype=getattr(torch,os.environ.get("DT","float16")),attn_implementation="sdpa").cuda().eval()
if os.environ.get("FAST_LINEAR")=="1":
    from fast_linear import convert; convert(m)
text=open(R+"TESTING.md").read()+open(R+"RELEASES.md").read()+open(R+"README.md").read()
ids=tok(text,return_tensors="pt").input_ids[0]; tot=0;n=0
with torch.no_grad():
    for i in range(0,min(len(ids)-1,16384),2048):
        x=ids[i:i+2048].cuda()[None]
        if x.shape[1]<64: break
        lg=m(x).logits.float(); tot+=torch.nn.functional.cross_entropy(lg[0,:-1],x[0,1:],reduction="sum").item(); n+=x.shape[1]-1
print(json.dumps({"ppl":float(torch.exp(torch.tensor(tot/n))),"tokens":n}))
