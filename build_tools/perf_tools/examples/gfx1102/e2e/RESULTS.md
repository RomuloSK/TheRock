# End-to-end: stock vs `rocm_perf_profile.py` (RX 7600 / gfx1102, ROCm 10.1.0, torch 2.14.0+rocm10.1.0)

Model: 8-layer decoder, d=2048, 16 heads, FFN 8192, fp16 (~0.5B params), random weights, ctx 2048.
Paired harness, 10 pairs, randomized order, 95% bootstrap CI, correctness gate on every variant
(logits vs unfused math path: rel err 7.8e-4, argmax agreement 99.8%). Raw: `llm-report.json`.

| case | stock | profile | speedup (95% CI) |
|---|---|---|---|
| prefill 2048 tokens | 312.8 ms (6.5k tok/s) | 102.5 ms (20.0k tok/s) | **3.04x** [3.035, 3.044] |
| decode (dynamic KV cat) | 11.9 ms/token | 8.2 ms/token | **1.45x** [1.452, 1.455] |

The profile only sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on gfx11 (before torch import); it never overrides a
user setting. No library was rebuilt: the fork's contribution is the detection, safety rules, tests and measurement.

## Follow-ups measured (single runs, no CI)
- Static preallocated KV cache + masked SDPA: 6.98 ms/token (vs 8.2 dynamic). Caller-side change; avoids per-step `torch.cat`.
- HIP graph of that step: 6.87 ms. Not launch-bound at this model size; graphs add ~nothing here.
- Decode time is dominated by rocBLAS `gemvt` (~175 us for a 33 MB weight, ~190 GB/s vs ~270 GB/s achievable on this card):
  a better M=1 GEMV kernel (rocBLAS) is the next real library-level candidate, ~1.3x on decode if it reached roofline. Not started.
- `TORCH_BLAS_PREFER_HIPBLASLT` 0/1: no difference on this card (checked 2048^3, 512/64x4096x4096, 4096^3).
Limits: random weights, one model shape, one GPU, forward only (no backward/training), experimental AOTriton kernels.
