import json, sys
# usage: mkcfg.py NAME EXE LIBDIRS(colon) PORT
name, exe, libs, port = sys.argv[1:5]
W="/home/romuloborges/Documents/Codex/2026-10-31/setu/work"
cfg=json.load(open(W+"/strata-config.json"))
cfg["exe"]=exe; cfg["lib_dirs"]=[l for l in libs.split(":") if l]; cfg["port"]=int(port)
cfg["log"]=f"/home/romuloborges/rocm-env/strata-bench/{name}.engine.log"
import os
if os.environ.get("SPEC0")=="1":
    a=cfg["args"]; i=a.index("--spec"); a[i+1]="0"
    if "--spec-min-p" in a: j=a.index("--spec-min-p"); del a[j:j+2]
json.dump(cfg,open(f"/home/romuloborges/rocm-env/strata-bench/{name}.json","w"),indent=1)
print(f"/home/romuloborges/rocm-env/strata-bench/{name}.json")
