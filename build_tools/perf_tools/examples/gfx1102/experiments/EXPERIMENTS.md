# Plan experiments E01-E10 on one Radeon RX 7600 (gfx1102), ROCm 10.1.0, torch 2.14.0+rocm10.1.0

Host: Xeon E5-2680 v4, 8 GB VRAM, single GPU. Scripts here: `exps.py <e01|e03|e04|e05|e06|e07>`, `startup.py`, `attn_check.py`.
Method is weaker than the paired harness: medians of 7 runs / single sessions, no CIs, except where noted. Treat as
leads to confirm, not final claims. A/A noise on this host was ~+-0.5% for long runs (see ../RESULTS.md).

## Summary
| ID | Status | Result |
|---|---|---|
| E01 prep reuse | measured | New-shape first call: median 0.80 ms, max 176 ms (first call 114 ms); repeat pass median 0.23 ms. Shape churn costs ~3.5x median / up to 750x worst-case. |
| E02 GEMM selection | measured earlier | See ../RESULTS.md: small-M default tile (128x128) wasteful; cold-weight gain ~1.1x only. 2048^3: 1.086x. |
| E03 launch vs graph | measured | Chains of tiny kernels: graph replay 3.2-3.5x faster (K=10/50/200); capture pays back in 2-5 replays. 10x 4096^3 GEMM: 1.08x. |
| E04 allocation | measured | Cached alloc/free ~3 us at 1-256 MB; with empty_cache each cycle 123-131 us (1-16 MB), 3.77 ms (256 MB). Caching allocator works; never release per step. |
| E05 copy/compute overlap | measured | Pinned + side stream: overlap efficiency 0.999 (40 ms copy hidden behind 40 ms compute). Pageable: 0.14, effectively serial. H2D 6.3 GB/s either way (link reports PCIe 4.0 x8; host is Broadwell, so likely host-limited; not verified). |
| E06 attention | measured | **Default on gfx1102 falls back to the unfused MATH path** (fused kernels need `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`). With it: 2k causal prefill 56.8 -> 2.44 ms (23x); 8k decode b8 37.0 -> 3.86 ms (9.6x, ~278 GB/s ~ roofline); 32k decode b1 19.0 -> 2.25 ms (8.5x). Output error vs fp64 equals fp16 MATH error (1e-3 prefill, 3e-5 decode). |
| E07 convolution | measured | First conv call in a fresh process took 2.48 s (64ch, default mode); later calls in the same process took 0.26 ms. That is a one-time setup cost, but this run cannot attribute it (benchmark mode ran second, after the setup was already paid). 256ch: benchmark mode no gain (0.62 vs 0.56 ms). |
| E08 compiler cliff | **blocked** | Needs a hot kernel plus an LLVM build and ISA analysis; no candidate kernel found in E02/E06 (attention already near roofline). |
| E09 collectives | **blocked** | Single GPU; RCCL crossover needs a multi-GPU/node lab. |
| E10 startup | measured | Fresh process: import torch 1.64 s, ctx init 0.31 s, first matmul 0.15 s (very stable). Disabling COMGR cache doubled ctx init (0.31 -> 0.67 s), so the cache works. |

## Candidate upstream actions (ranked by measured value on this card)
1. Attention: the default SDPA path on RDNA3 (gfx1100/1101/1102/1103) silently uses the 8-23x slower math kernel. Enabling fused
   AOTriton kernels by default (a PyTorch build/runtime-policy decision, not a TheRock source change) or warning loudly is the
   biggest win. Needs broader numeric validation (GQA, masks, dropout, backward) before default-on.
2. Document/lint: pinned memory + side streams for overlap; never `empty_cache()` per step; use CUDA/HIP graphs for short-kernel chains.
3. First-call latency (E01/E07): persistent per-shape selection/ MIOpen caches shipped for common shapes.
4. Small-M GEMM tile choice (E02): ~10% when memory-bound.
Not done: backward attention, GQA/paged/ragged attention, multi-arch and CDNA validation, upstream PRs.
