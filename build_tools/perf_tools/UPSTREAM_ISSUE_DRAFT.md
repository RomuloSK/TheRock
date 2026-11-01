# DRAFT (not filed) - suggested target: pytorch/pytorch (label: module: rocm), cc ROCm/aotriton maintainers

**Title:** [ROCm] Fused SDPA (AOTriton) disabled by default on gfx1102 (RX 7600): 8-23x slower attention, forward/backward validated

**Summary.** On PyTorch 2.14.0+rocm10.1.0 (ROCm Core SDK 10.1.0 stable wheels), `scaled_dot_product_attention` on a Radeon RX 7600
(gfx1102) silently falls back to the math kernel unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is set. The flag gates gfx1101,
gfx1102, gfx1150, gfx1151 and gfx1200 (`torch/testing/_internal/common_cuda.py` lists them as experimental; gfx1100 is on by default).
The only hint is a UserWarning that most users never see in framework code paths.

**Measurements (gfx1102, 8 GB, ROCm 10.1.0; flag off -> on)**
- causal prefill, 32 heads, d=128, 2048 tokens: 56.8 ms -> 2.44 ms (23x)
- decode, batch 8, 32 heads, KV 8192: 37.0 -> 3.86 ms (9.6x, ~278 GB/s, near this card's memory roofline)
- decode, batch 1, KV 32768: 19.0 -> 2.25 ms (8.5x)
- 8-layer 0.5B decoder, paired runs, 95% CI: prefill 3.04x, decode 1.45x
- TinyLlama-1.1B (HF, sdpa): prefill 818 -> ~195 ms *together with a separate layout change*; attention-only share is in RESULTS.md
- training smoke (4-layer d512, AdamW, fp16 autocast, 60 steps): loss 4.8680 (off) vs 4.8689 (on); step 90.6 -> 50.3 ms

**Correctness checked with the flag on.** Forward + dQ/dK/dV vs an fp64 reference for MHA causal/non-causal, odd length (1000), GQA 4:1,
MQA decode, bool and float masks, head dim 80, bf16, q_len != kv_len: errors <= ~1e-3 (fp16) / 4e-3 (bf16), within 1-3x the unfused fp16
error, all finite. Unsupported combinations (FLASH with masks or q!=kv causal; EFFICIENT with GQA) fall through to another backend; GQA+mask,
dropout 0.1, fully-masked row and length-1 all completed with finite outputs and gradients. Perplexity on 16k tokens: TinyLlama 4.57627 vs
4.57635, Qwen2.5-0.5B 7.03619 vs 7.03653 (fp16), 7.04462 vs 7.04417 (bf16).

**Not tested (please do not read this as validation for them):** gfx1101/1150/1151/1200, long contexts (>8k) numerics, dropout statistics,
multi-GPU, multi-epoch convergence, other AOTriton versions.

**Ask.** (1) Is there a known correctness/perf reason gfx1102 remains experimental? (2) If not, enable it by default (or emit a once-per-process
warning naming the env var when the fallback to math is taken for a supported-by-AOTriton arch). 

**Repro.** `build_tools/perf_tools/examples/gfx1102/` in https://github.com/RomuloSK/TheRock/tree/perf/benchmark-gates
(`experiments/exps.py e06`, `experiments/attn_check.py`, `e2e/attn_matrix.py`, `e2e/attn_dispatch.py`, `e2e/train_check.py`).
