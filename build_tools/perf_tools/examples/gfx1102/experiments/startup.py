import time,sys
t=time.perf_counter()
import torch
t1=time.perf_counter(); x=torch.randn(512,512,device="cuda",dtype=torch.half); torch.cuda.synchronize(); t2=time.perf_counter(); y=x@x; torch.cuda.synchronize(); t3=time.perf_counter()
print(f'{{"import_s":{t1-t:.3f},"ctx_init_s":{t2-t1:.3f},"first_matmul_s":{t3-t2:.3f}}}')
