import json, sys, time, urllib.request, hashlib, statistics
port=int(sys.argv[1]); out=sys.argv[2]; reps=int(sys.argv[3]) if len(sys.argv)>3 else 2
R="/home/romuloborges/projects/ROCm/therock-perf-implementation/"
corpus=(open(R+"TESTING.md").read()+open(R+"RELEASES.md").read()+open(R+"README.md").read()+open(R+"docs/performance/rx7600.md").read())
words=corpus.split()
def prompt(n_words, q): return " ".join(words[:n_words])+"\n\n"+q
cases=[("short",  "Write a short paragraph explaining what a GPU does and why memory bandwidth matters.", 160, 0),
       ("medium", prompt(900,"Summarize the text above in five bullet points."),               160, 1),
       ("long",   prompt(4500,"Summarize the text above in five bullet points."),               96, 2)]
def call(q, n, salt=""):
    body={"model":"m","messages":[{"role":"user","content":(salt+q)}],"temperature":0,"max_tokens":n,"stream":False,"seed":1}
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer 1234"})
    t=time.time(); r=json.loads(urllib.request.urlopen(req,timeout=3600).read()); return r, time.time()-t
call("Say hi.",8)   # warm-up
import os
sel=os.environ.get('CASES')
res=[]
for name,q,n,_ in cases:
    if sel and name not in sel.split(','): continue
    for rep in range(reps):
        r,dt=call(q,n,f"[request {name}-{rep}] "); tm=r.get("timings") or {}; u=r.get("usage") or {}
        m_=r["choices"][0]["message"]; txt=(m_.get("reasoning_content") or "")+"||"+(m_.get("content") or "")
        res.append({"case":name,"rep":rep,"wall_s":round(dt,2),"prompt_n":tm.get("prompt_n",u.get("prompt_tokens")),"prompt_tps":tm.get("prompt_per_second"),"pred_n":tm.get("predicted_n",u.get("completion_tokens")),"pred_tps":tm.get("predicted_per_second"),"text_sha":hashlib.sha1(txt.encode()).hexdigest()[:10],"text":txt[:2000]})
        print(name,rep,{k:res[-1][k] for k in ("wall_s","prompt_n","prompt_tps","pred_n","pred_tps","text_sha")},flush=True)
json.dump(res,open(out,"w"),indent=1)
