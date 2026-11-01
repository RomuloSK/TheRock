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

## Safety validation of the fused-attention profile (added)
`attn_matrix.py` (forward + backward vs fp64 reference, `attn_matrix.json`), `attn_dispatch.py`, `train_check.py`.
- 10 configs: MHA causal/non-causal, odd length (1000), GQA 4:1, MQA decode, bool and float masks, head dim 80, bf16, q_len != kv_len.
  Fused fwd/dQ/dK/dV errors vs fp64 are <= ~1e-3 in fp16 (4e-3 bf16), 1-3x the unfused fp16 error; all finite.
- Some combinations are unsupported by one fused backend (FLASH: bool/float mask, q!=kv causal; EFFICIENT: GQA); the
  default dispatcher falls through to a supported backend. Confirmed for GQA+mask, q!=kv causal, bool mask, dropout 0.1,
  fully-masked row, length 1: all complete with finite outputs and gradients, with the flag on and off.
- Training smoke (4-layer d512, AdamW, autocast fp16, 60 steps): loss 7.7845 -> 4.8680 (stock) vs 4.8689 (profile);
  step time 90.6 ms -> 50.3 ms (1.8x, single run).
Not covered: dropout statistics, long-context (>8k) numerics, multi-epoch convergence, other gfx11 parts (only gfx1102 tested).
