import json, sys, glob, statistics as st, collections
runs=collections.OrderedDict()
for f in sys.argv[1:]: runs[f.replace(".results.json","")]=json.load(open(f))
def agg(res,case,key):
    v=[r[key] for r in res if r["case"]==case and r.get(key)]; return v
print(f"{'run':14s} {'case':7s} {'prefill tok/s':>14s} {'decode tok/s':>13s}  decode samples")
for name,res in runs.items():
    for case in ("short","medium","long"):
        p=agg(res,case,"prompt_tps"); d=agg(res,case,"pred_tps")
        print(f"{name:14s} {case:7s} {st.mean(p):14.1f} {st.mean(d):13.2f}  {[round(x,1) for x in d]}")
print()
print("overall decode (mean of all samples) / prefill (medium+long):")
for name,res in runs.items():
    d=[r["pred_tps"] for r in res if r.get("pred_tps")]; p=[r["prompt_tps"] for r in res if r["case"] in("medium","long")]
    print(f"  {name:14s} decode {st.mean(d):6.2f} tok/s (n={len(d)}, sd {st.pstdev(d):.2f})   prefill {st.mean(p):6.1f} tok/s")
