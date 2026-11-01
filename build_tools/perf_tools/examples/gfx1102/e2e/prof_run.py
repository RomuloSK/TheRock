import os, sys, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM
from fast_linear import convert; from fast_decode import GraphedDecoder
name=os.environ["MODEL"]; tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
ids=tok("The history of computing is a story of abstraction layers. "*100,return_tensors="pt").input_ids[:,:512].cuda()
convert(m); dec=GraphedDecoder(m,max_len=640,compile=True); dec.generate(ids,4)
dec.generate(ids,65)
