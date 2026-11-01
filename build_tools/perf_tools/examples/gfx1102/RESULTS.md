# gfx1102 (Radeon RX 7600) measurements, ROCm 10.1.0 / torch 2.14.0+rocm10.1.0

Host: Xeon E5-2680 v4, Ubuntu 26.04, venv install from stable.repo.amd.com (`rocm[libraries,devel,device-gfx1102]`).
Raw reports: `aa2-report.json`, `tune-report.json`. Specs: `aa2.json`, `tune.json`. Benchmark: `gemm_bench.py`.

## A/A calibration (identical commands)
- Short runs (200 iters, skinny GEMM) produced a **false 1.08x "gain"** (CI 1.013-1.158).
- With 3000 iters/16 pairs, A/A CI is within +-0.5% on both shapes. Use >=3000 iters for launch-bound shapes.

## TunableOp (tuned) vs default GEMM selection, fp16, 10 pairs, 95% bootstrap CI
| shape (MxNxK) | speedup | CI |
|---|---|---|
| 2048x2048x2048 | 1.086x | 1.079-1.092 |
| 16x4096x4096 (cache-warm, see correction) | 1.826x | 1.779-1.870 |
| 1x11008x4096 | 0.999x | 0.998-1.000 |
| 512x4096x4096 | 1.005x | 1.000-1.009 |
| 4096x4096x4096 | 1.019x | 1.017-1.022 |

Caveats: one GPU, one consumer part, torch GEMM path only (not raw hipBLASLt), tuning file is shape-specific
and tied to library versions (see Validator rows in `tunable_0.csv`). Gains show where the default heuristic
picks a poor solution (small-M); this points to the heuristic/solution-selection as the thing to improve upstream.

## Correction: the 1.83x skinny-GEMM gain was a cache artifact
`gemm_bench.py` reuses one 33 MB weight matrix, which stays resident in the RX 7600's Infinity Cache, so
16x4096x4096 was not a realistic memory-bound decode case. `cold.py` rotates 8 weight copies (268 MB):

| variant | latency | effective weight bandwidth |
|---|---|---|
| default (rocBLAS/Tensile, MT128x128) | 0.143 ms | ~235 GB/s (~80% of ~288 GB/s peak) |
| tuned (hipBLASLt MT64x16) | 0.128-0.131 ms | ~255-262 GB/s |

Cold-weight gain is ~1.10x (3 runs, not a paired/CI result). Kernels seen via `prof.py`: the default launches a
128x128 tile for M=16 (compute-warm 135 us vs 83 us), but that tile waste matters little once HBM/GDDR-bound.
Lesson: every GEMM benchmark spec must state whether weights are cache-resident; use rotating buffers for decode.
The 2048^3 (1.09x) and 4096^3 results are compute-bound and unaffected by this issue.
