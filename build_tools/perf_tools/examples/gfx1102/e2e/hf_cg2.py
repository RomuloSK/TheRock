import os, sys, json, time, torch
sys.path.insert(0,'/home/romuloborges/projects/ROCm/therock-perf-implementation/build_tools/perf_tools')
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
if os.environ.get("FAST_LINEAR")=="1":
    from fast_linear import convert; convert(m)
P=512; ids=tok("The history of computing is a story of abstraction layers. "*100,return_tensors="pt").input_ids[:,:P].cuda()
cache=StaticCache(config=m.config,max_batch_size=1,max_cache_len=P+160,device="cuda",dtype=torch.float16)
tokbuf=torch.zeros(1,1,dtype=torch.long,device="cuda"); pos=torch.zeros(1,dtype=torch.long,device="cuda")
out_tokens=[]
with torch.no_grad():
    lg=m(ids,past_key_values=cache,cache_position=torch.arange(P,device="cuda"),use_cache=True).logits
    tokbuf.copy_(lg[:,-1:].argmax(-1)); pos.fill_(P)
    def step():
        o=m(tokbuf,past_key_values=cache,cache_position=pos,use_cache=True).logits
        tokbuf.copy_(o[:,-1:].argmax(-1)); pos.add_(1)
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step()
    # reset and replay
    tokbuf.copy_(ids[:,-1:]); pos.fill_(P)  # restart decode from the prompt end (cache already holds the prompt)
    toks=[]; torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(128): g.replay(); toks.append(tokbuf.clone())
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/128*1e3
print(json.dumps({"latency_ms":dt,"tokens":[int(x) for x in torch.cat(toks,1)[0,:32]]}))
