import json, glob, collections, statistics as st, re
d=collections.defaultdict(list)   # (model, test, fa, engine) -> [t/s per round]
for f in sorted(glob.glob("out/*.json")):
    eng,model,rnd=re.match(r"out/(rocm\d+)_(.+)_r(\d+)\.json",f).groups()
    for r in json.load(open(f)):
        test=f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        d[(model.replace("Qwen2.5-3B-Instruct-",""),test,r["flash_attn"],eng)].append(r["avg_ts"])
print(f"{'model':7s} {'test':7s} {'fa':3s} {'ROCm 7.10a t/s':>16s} {'ROCm 10.1 t/s':>16s} {'ratio':>7s}  per-round ratios")
for (m,t,fa,e) in sorted({(k[0],k[1],k[2],'x') for k in d}):
    a=d[(m,t,fa,"rocm7")]; b=d[(m,t,fa,"rocm101")]
    pr=[y/x for x,y in zip(a,b)]
    print(f"{m:7s} {t:7s} {str(fa):3s} {st.mean(a):9.1f}+-{st.pstdev(a):4.1f} {st.mean(b):9.1f}+-{st.pstdev(b):4.1f} {st.mean(b)/st.mean(a):7.3f}  {[round(x,3) for x in pr]}")
