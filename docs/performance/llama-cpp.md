# llama.cpp (HIP) fully on the GPU: ROCm 10.1.0 vs ROCm 7.10.0a20251120 - RX 7600

Why this test: the Strata run (docs/performance/strata.md) was bounded by CPU/RAM expert compute, so it could not show GPU-side differences.
Here the whole model sits in the RX 7600's 8 GB VRAM (`-ngl 99`, 1.2-3.1 GB models), so CPU, RAM and PCIe are not on the critical path.

## What is compared (and what is not)
- Same llama.cpp source (the commit Strata pins, `3cf03257`), same CMake flags, `GPU_TARGETS=gfx1102`, built twice: against ROCm 7.10.0a20251120
  (TheRock wheels, what Strata's Linux package ships) and against ROCm 10.1.0 (stable wheels). Scripts: `build_tools/perf_tools/examples/gfx1102/llama_cpp/`.
- **The fork's own changes are not involved** (they are PyTorch-level). This fork adds no ROCm library or compiler code, so "ROCm 10.1.0 from the fork" is
  the same binaries as stock 10.1.0. The finding below is a difference between two stock ROCm releases.
- Model: Qwen2.5-3B-Instruct (bartowski GGUF quants). `llama-bench -ngl 99 -p 512 [-p 2048] -n 128 -fa 0/1 -r 5`; 3 rounds, engine order alternated per round.

## Results
Prompt processing (pp512) and generation (tg128), flash attention on, tokens/second (mean of 5 repetitions; Q8_0 and Q4_K_M from 3 rounds, others one run):
| quant | pp512 ROCm 7.10a | pp512 ROCm 10.1 | ratio | tg128 7.10a | tg128 10.1 | ratio |
|---|---|---|---|---|---|---|
| Q2_K | 2371 | 2298 | 0.97 | 107.4 | 109.9 | 1.02 |
| Q3_K_M | 2430 | 2388 | 0.98 | 91.2 | 98.2 | 1.08 |
| Q4_0 | 2673 | 2685 | 1.00 | 86.9 | 89.4 | 1.03 |
| Q4_K_M | 2604 | 2576 | 0.99 | 95.7 | 96.8 | 1.01 |
| Q5_K_M | 2587 | 2563 | 0.99 | 87.0 | 88.2 | 1.01 |
| Q6_K | 2452 | 2341 | 0.95 | 76.8 | 77.9 | 1.01 |
| **IQ4_XS** | **1082** | **2698** | **2.49** | 103.3 | 104.9 | 1.02 |
| **Q8_0** | **1001** | **2741** | **2.74** | 64.7 | 65.3 | 1.01 |

Three-round detail (`out/`, `summarize.py`): Q8_0 pp512 966 -> 2415 (fa off) / 1012 -> 2889 (fa on), ratios 2.49-2.93 with every round within 3% of the others; Q8_0 pp2048 916 -> 2288 / 997 -> 2926.
Q4_K_M prompt processing is 3-6% *slower* on 10.1 at pp512 (ratios 0.93-0.97 per round) and ~equal at pp2048. Decode is +0-1% for Q4_K_M/Q8_0, +1-3% elsewhere.

## Cause (profiled with rocprofv3, Q8_0 pp512, flash attention on)
| kernel | ROCm 7.10a | ROCm 10.1 |
|---|---|---|
| `mul_mat_q<GGML_TYPE_Q8_0, 128>` (llama.cpp's own MMQ kernel) | 1972 us avg, 93% of GPU time | 578 us avg, 80% |
| `unary_gated_op_kernel` (SiLU) | 268 us | 264 us |
| `flash_attn_ext_f16` | 214 us | 203 us |
| total kernel time | 1579 ms | 538 ms |
The same kernel from identical source and flags is ~3.4x faster; every other kernel is unchanged. Forcing the MMQ path (`GGML_CUDA_FORCE_MMQ=1`) changes nothing on
either build (1006 vs 1004 and 2767 vs 2745 tok/s), so rocBLAS/hipBLAS is not involved. This points at compiler code generation for the Q8_0 (and IQ4_XS) MMQ
instantiations on gfx1102 (the 7.10a LLVM produced much slower code). The exact compiler change was not identified.

## Numerical check (llama-perplexity, 24 chunks x 512 tokens of repo text)
Q8_0: 6.2974 +/- 0.2142 on both builds; IQ4_XS: 6.4416 on both (identical); Q4_K_M: 6.4191 vs 6.4368 (0.3%, well inside the +/-0.22 error).
The faster kernels give the same results.

## Limits
- One GPU (RX 7600), one model family and size, one llama.cpp commit; 3 rounds only for Q8_0/Q4_K_M. Single user-visible session timings.
- It is a stock-ROCm-release difference. It helps only where the older ROCm is in use and the workload is prompt-heavy with these quantizations.
- Strata itself does not benefit: its prompt path is dominated by CPU/RAM expert compute (see strata.md).
- Decode (memory-bound) is essentially unchanged: 1-3%.
