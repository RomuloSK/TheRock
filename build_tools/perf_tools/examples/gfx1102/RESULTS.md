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
| 16x4096x4096 | 1.826x | 1.779-1.870 |
| 1x11008x4096 | 0.999x | 0.998-1.000 |
| 512x4096x4096 | 1.005x | 1.000-1.009 |
| 4096x4096x4096 | 1.019x | 1.017-1.022 |

Caveats: one GPU, one consumer part, torch GEMM path only (not raw hipBLASLt), tuning file is shape-specific
and tied to library versions (see Validator rows in `tunable_0.csv`). Gains show where the default heuristic
picks a poor solution (small-M); this points to the heuristic/solution-selection as the thing to improve upstream.
