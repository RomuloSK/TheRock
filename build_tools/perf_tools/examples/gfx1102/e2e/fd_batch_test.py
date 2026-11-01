import os, sys, time, json, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
from fast_linear import convert, fuse_projections; from fast_decode import GraphedDecoder; from fast_tune import load_tuning, tune_decode
name=os.environ.get("MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok=AutoTokenizer.from_pretrained(name,padding_side="left")
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
prompts=["Explain what a GPU does in one paragraph.","The capital of France is","def fibonacci(n):","List three uses of a compiler:"]
kw=dict(do_sample=False,repetition_penalty=1.0,top_k=None,top_p=None,temperature=None)
N=int(os.environ.get("NEW","32"))
refs=[]
for p in prompts:   # per-prompt (unbatched) reference
    i=tok(p,return_tensors="pt").to("cuda")
    with torch.no_grad(): refs.append(m.generate(**i,max_new_tokens=N,pad_token_id=tok.pad_token_id,**kw)[0,i.input_ids.shape[1]:])
if os.environ.get("FAST_LINEAR")=="1":
    convert(m)
    if os.environ.get("FUSE")=="1": fuse_projections(m)
tf=os.environ.get("TFILE")
if tf:
    if not os.path.exists(tf): tune_decode(m,(1,2,4),tf)
    load_tuning(tf)
dec=GraphedDecoder(m,max_len=512,max_batch=4,compile=os.environ.get("COMPILE")=="1",use_graph=os.environ.get("GRAPH","1")=="1")
res={}
for B in (1,2,4):
    enc=tok(prompts[:B],return_tensors="pt",padding=True).to("cuda")
    for rep in range(2):   # 2nd run reuses the captured graph
        o=dec.generate(enc.input_ids,N,attention_mask=enc.attention_mask)
    agree=[]
    for i in range(B):
        r=refs[i]; k=min(len(r),o.shape[1]); agree.append(round((o[i,:k]==r[:k]).float().mean().item(),3))
    res[f"B{B}"]=agree
print(json.dumps(res))
