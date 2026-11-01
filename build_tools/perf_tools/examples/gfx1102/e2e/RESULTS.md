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

## fast_linear: pre-transposed weights for inference (added)
Stock `F.linear` ([out,in] weight) at M=1 reaches only 110-200 GB/s on this card; `x @ W_t` with an [in,out] layout reaches
250-275 GB/s (K=2048,N=6144: 2.24x; N=8192: 1.53x; K=8192,N=2048: 1.29x; head N=32000: 1.51x). A Triton GEMV (`gemv.py`)
gave no further gain: plain `x @ w` was already ~265-275 GB/s, so the loss was the layout, not the kernel.
`fast_linear.convert(model)` swaps each nn.Linear for an [in,out]-layout copy (state_dict stays standard; 6 unit tests).

Paired, 10 pairs, correctness gate on every run (`fl-report.json`, `both-report.json`):
| comparison | prefill 2048 | decode/token |
|---|---|---|
| profile + fast_linear vs profile only | 1.200x [1.198, 1.203] | 1.289x [1.288, 1.290] |
| **profile + fast_linear vs stock** | **3.649x** [3.640, 3.658] | **1.875x** [1.873, 1.878] |
Absolute: prefill 312.8 -> ~86 ms; decode 11.9 -> ~6.4 ms/token.
Caveat: INFERENCE ONLY. Training step (4-layer d512): loss identical (4.8688 vs 4.8689) but 50.5 -> 55.7 ms (10% slower),
so `convert()` warns in training mode. Memory is unchanged (weights are replaced, not duplicated).

## Real checkpoint: TinyLlama-1.1B-Chat (fp16, GQA 32/4 heads, untied head), HF transformers 5.22 sdpa
Single-session timings (not paired CIs). Prefill = 1800 tokens; decode = ms/token over 128 tokens after a 512-token prompt.
| config | prefill | decode (eager HF generate) | decode (CUDA-graph, static cache) |
|---|---|---|---|
| stock | 818.3 ms | 31.7 ms | 20.9 ms |
| profile + fast_linear | 195.4 ms (**4.2x**) | 22.7 ms (**1.40x**) | 13.9 ms (**2.28x vs stock eager**; 1.50x vs stock graphed) |
Quality (stock vs profile+fast_linear): perplexity on 16k tokens of repo prose 4.57627 vs 4.57635 (1.6e-5 rel);
loss on repeated text 0.03640 vs 0.03641; generated paragraph identical; 64 greedy tokens identical (eager); 32 graphed tokens identical.
(`hf_bench.py`, `ppl.py`, `hf_cg2.py`.)

## Stack reliability findings (reproduced on STOCK, i.e. not caused by the profile) - unresolved
1. `torch.compile(model.forward, mode="reduce-overhead")` with HF `cache_implementation="static"`: inductor warns
   "skipping cudagraphs due to mutated inputs (66 instances)", then the process segfaults (`hf_graph.py`, MODE=compile).
   HF static cache WITHOUT compile took 310 ms/token (not usable).
2. After capturing a CUDA graph of a StaticCache decode step, an eager prefill on the same cache faulted
   (`HSA_STATUS_ERROR_EXCEPTION ... 0x1016`, `hipErrorLaunchFailure`) with all 4 profile/fast_linear combinations (`hf_cg.py`),
   while replaying without the re-prefill works (`hf_cg2.py`). A minimal capture-then-eager repro (`repro_a.py`: GEMV +
   index_copy_ graph, then eager GEMM / SDPA, then replay) does NOT fault, so this is UNPROVEN as a stack bug and may be misuse of
   HF's cache API in `hf_cg.py` (e.g. host-side cache state across capture). Not reported upstream; needs root-causing first.
Note: fast_linear copies the weight, so models with tied embeddings (e.g. Qwen) keep one extra copy of the head; TinyLlama is untied.

## Second checkpoint: Qwen2.5-0.5B-Instruct (tied embeddings, QKV bias, GQA 14/2, head dim 64), single-session timings
(`MODEL=Qwen/Qwen2.5-0.5B-Instruct DT=float16|bfloat16`, `hf_bench.py`, `ppl.py`; stock = `ROCM_PERF_PROFILE=0`)
| dtype | prefill stock -> fork | decode eager stock -> fork | ppl stock / fork (16k tokens) |
|---|---|---|---|
| fp16 | 399.1 -> 103.5 ms (**3.86x**) | 37.3 -> 26.3 ms/token (**1.42x**) | 7.03619 / 7.03653 (4.8e-5 rel) |
| bf16 | 402.0 -> 109.8 ms (**3.66x**) | 37.6 -> 26.4 ms/token (**1.42x**) | 7.04462 / 7.04417 (6.4e-5 rel) |
Tied embeddings: fast_linear replaces the shared head with a transposed copy (about +136M params of memory for this model).
Graphed-decode profile attempt: torch profiler/CUPTI under graph replay reported only ~83 of ~155 GEMV launches per step, so
per-kernel attribution of the remaining 13.9 ms vs ~7.6 ms floor is NOT established; fusing QKV / gate-up GEMVs is an untested idea.

## Decode headroom analysis (synthetic, TinyLlama shapes: 22 layers, d=2048, kv=256, ffn=5632, graphed, cold weights)
Separate q/k/v + gate/up GEMVs: 8.63 ms/step (225 GB/s). Fused QKV and gate+up: 7.67 ms/step (253 GB/s): ~1 ms saved, ~7% of
the 13.9 ms real graphed step. The remaining ~5 ms is unfused elementwise/norm/rotary/attention work in HF eager code; capturing
it needs torch.compile (segfaults here, see above) or hand-written fused kernels. Neither is implemented; QKV fusion is not shipped.
