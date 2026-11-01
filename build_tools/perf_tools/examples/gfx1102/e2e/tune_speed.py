import os, sys, time, json, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
from fast_linear import convert; from fast_decode import GraphedDecoder; from fast_tune import tune_decode, load_tuning
name=os.environ.get("MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0"); TUNE=os.environ.get("TUNE","0")=="1"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
base=tok("The history of computing is a story of abstraction layers. "*100,return_tensors="pt").input_ids[:,:512].cuda()
convert(m); f=os.environ.get("TFILE","/tmp/tune_%s.csv"%name.split("/")[-1])
if TUNE and not os.path.exists(f):
    t0=time.perf_counter(); tune_decode(m,(1,2,4,8),f); print("tuned in %.0fs, %d lines"%(time.perf_counter()-t0,sum(1 for _ in open(f))),file=sys.stderr)
    pass
if TUNE: load_tuning(f)
import faulthandler; faulthandler.dump_traceback_later(420,exit=True)
print("stage: tuned+loaded",file=sys.stderr,flush=True)
dec=GraphedDecoder(m,max_len=640,max_batch=8,compile=True)
def tm(fn):
    torch.cuda.synchronize(); t=time.perf_counter(); r=fn(); torch.cuda.synchronize(); return time.perf_counter()-t,r
out={}
for B in (1,2,4,8):
    ids=base.repeat(B,1); dec.generate(ids,4); a,_=tm(lambda:dec.generate(ids,1)); b,o=tm(lambda:dec.generate(ids,65)); out[B]=round((b-a)/64*1e3,2)
print(json.dumps({"tuned":TUNE,"ms_per_step":out}))
