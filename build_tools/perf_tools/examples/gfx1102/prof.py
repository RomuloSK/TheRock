import torch, sys
from torch.profiler import profile, ProfilerActivity
m,n,k=16,4096,4096
A=torch.randn(m,k,device="cuda",dtype=torch.float16);B=torch.randn(k,n,device="cuda",dtype=torch.float16)
for _ in range(30): A@B
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(100): A@B
    torch.cuda.synchronize()
for e in p.key_averages(): 
    if e.device_type.name=="CUDA" or e.self_device_time_total>0: print(e.key[:160], e.count, round(e.self_device_time_total/e.count,2),"us")
