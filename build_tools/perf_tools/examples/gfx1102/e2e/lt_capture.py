import torch, sys, torch.cuda.tunable as t
variant=sys.argv[1]
t.enable(True); t.tuning_enable(False); t.set_filename("/home/romuloborges/rocm-env/exp/tm_0.csv",insert_device_ordinal=False); t.read_file("/home/romuloborges/rocm-env/exp/tm_0.csv")
x=torch.randn(4,5632,device="cuda",dtype=torch.half); w=torch.randn(5632,2048,device="cuda",dtype=torch.half); y=torch.empty(4,2048,device="cuda",dtype=torch.half)
def step(): torch.mm(x,w,out=y)
if variant=="none": pass
elif variant=="main_stream_warm":
    for _ in range(3): step()
    torch.cuda.synchronize()
elif variant=="side_stream_warm":
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
elif variant=="same_stream_warm_and_capture":
    st=torch.cuda.Stream()
    with torch.cuda.stream(st):
        for _ in range(3): step()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st): step()
    g.replay(); torch.cuda.synchronize(); print(variant,"capture ok"); sys.exit(0)
elif variant=="capture_stream_warm":   # warm inside a graph-capture-like private stream
    s=torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g): step()
g.replay(); torch.cuda.synchronize(); print(variant,"capture ok")
