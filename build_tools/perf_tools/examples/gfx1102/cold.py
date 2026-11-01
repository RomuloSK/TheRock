import torch, json, sys
m,n,k,copies=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),8
A=torch.randn(m,k,device="cuda",dtype=torch.float16)
Bs=[torch.randn(k,n,device="cuda",dtype=torch.float16) for _ in range(copies)]
for i in range(40): A@Bs[i%copies]
torch.cuda.synchronize()
s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
s.record()
for i in range(2400): A@Bs[i%copies]
e.record(); torch.cuda.synchronize()
ms=s.elapsed_time(e)/2400
print(json.dumps({"latency_ms":ms,"GBps":k*n*2/ms/1e6}))
