# Strata (Qwen3.8-Flash-Next Q2_0, RAM + VRAM) on ROCm 10.1 vs ROCm 7.10a - RX 7600

Real-workload test requested by the owner: the local Strata setup (`~/Documents/Codex/2026-10-31/setu/work`), serving
`Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0` (~66 GB GGUF, mixture of experts; experts held in system RAM, an expert cache and KV
in the RX 7600's 8 GB VRAM; MTP speculative decoding). Scripts and raw results: `build_tools/perf_tools/examples/gfx1102/strata/`.

## What this test can and cannot say about "our fork"
- Strata is a C++/HIP engine (llama.cpp/ggml based) with its own kernels and hipBLAS for prompt GEMMs. **None of this fork's changes touch it**:
  the fused-attention profile, `fast_linear`, `GraphedDecoder` and `fast_tune` are PyTorch-level. This fork changes no ROCm library or compiler
  code either, so "Strata built on the fork's ROCm" is the same binaries as stock ROCm 10.1.0 (installed from the stable wheels).
- So the question actually tested: **does Strata run faster when built and run on ROCm 10.1.0 instead of the ROCm 7.10.0a20251120 that its own
  Linux `rx7900` package ships?** To isolate the ROCm version, the *same Strata source* (checkout `c62d5e3`, pinned llama.cpp `3cf03257`,
  same CMake flags as `setup.py`, gfx1102 only) was built twice: against ROCm 7.10.0a20251120 (control) and against ROCm 10.1.0.
- The prebuilt bundle you run today (engine 0.1.55, ROCm 7.x) was measured too, as a reference. Its source is a different revision.

## Method
- Same machine, same pack/model/config as `strata-config.json` (`--spec 4 --spec-min-p 0.5 --mtp`, `--prefill 512`, `--kv int8`, `--resident-budget-gib 34`,
  13 pool workers) except port and log path; started through Strata's own `serve.server`; the owner's config and logs were not modified.
- `client.py`: OpenAI `/v1/chat/completions`, temperature 0, seed 1, three prompts (short 73 tokens/160 max; medium 1890 tokens/160; long 9412 tokens/96),
  each request prefixed with a unique tag so the conversation cache cannot reuse the prompt; timings are the server's own (`prompt_per_second`,
  `predicted_per_second`). Two repetitions per case after a warm-up, two passes per engine (A then B, order alternated), 12 decode samples per engine.

## Results (single RX 7600, single session)
| engine | decode tok/s (mean +- sd, n=12) | prefill tok/s (medium+long, n=8) |
|---|---|---|
| prebuilt bundle (ROCm 7.x, engine 0.1.55) | 25.3 (n=6) | 103.3 |
| source build, ROCm 7.10.0a20251120 (control) | 24.4 +- 1.0 | 103.6 |
| source build, **ROCm 10.1.0** | **26.9 +- 1.2** | 102.4 |

- Decode: ROCm 10.1 is **+9.9%** over the control (pooled bootstrap CI 1.06-1.13). That interval is optimistic: samples are clustered in only
  2 passes per engine. Prefill: -1.2% (no gain). Time to first token for long prompts is unchanged (~95 s for 9.4k tokens).
- The source-built control matches the prebuilt bundle (24.4-25.3 vs 25.3 tok/s decode, 103.6 vs 103.3 prefill), so the build recipe is sound.

### Where the decode difference comes from (from the engine logs)
| engine | drafts accepted / proposed | tokens per verification step | ms per verification step |
|---|---|---|---|
| control, ROCm 7.10a (2 passes) | 65% | 2.25 | 90.6, 94.3 |
| prebuilt bundle | 70% (664/954) | 2.27 | 89.1 |
| ROCm 10.1 (2 passes) | 69% | 2.34 | 87.7, 87.8 |
About 4% of the gain is more tokens per step (higher draft acceptance). That depends on the generated text, which differs between builds because GEMMs
round differently; it is not evidence of faster kernels and may not repeat on other prompts. The remaining ~4-5% is lower time per step (87.8 vs 90.6-94.3 ms),
which is about the size of the run-to-run drift inside the control (90.6 vs 94.3). **Honest summary: ROCm 10.1 is not slower and decode looks
a few percent faster, but this test cannot prove more than that.** Expert-cache slot count also varied between runs (811-861 slots, set by free VRAM);
decode was consistent for ROCm 10.1 across 811 and 861 slots, so it is not the main driver.

Outputs are not bit-identical between engines (different GEMM rounding, expected and documented by Strata), but the answers start identically and are coherent;
no quality metric (perplexity/benchmark) was run, so quality parity is not established beyond that.

## What would actually speed up this workload (not done)
Prefill (~103 tok/s) and decode (~25-27 tok/s) are bounded by CPU/RAM expert compute (AVX2, no AVX-512), PCIe and expert-cache hit rate (16%); the dense GPU GEMMs
are a small part. Candidates, in order of expected value, all untested here:
1. A **hipBLASLt tuning table for gfx1102** (`STRATA_HIPBLASLT_TABLE`, built with Strata's `tools/hip/tune_hipblaslt`): the shipped tables cover gfx1100 and gfx1151 only,
   and Strata reports +40-60% prompt speed from them on a 7900 XTX. This is the same idea as this repo's GEMM-selection finding (default choice vs tuned),
   but it only helps if dense prompt GEMMs matter here. Needs a prefill profile first.
2. A bigger expert cache (more free VRAM), more RAM-resident experts, or a faster CPU path for experts (the CPU lacks AVX-512).
3. Profile prefill with `rocprofv3` to see whether GPU time or host/PCIe time dominates before touching kernels.

## Reproduce
```bash
cd build_tools/perf_tools/examples/gfx1102/strata
./prep.sh                 # build tools, pinned llama.cpp, ROCm 7.10.0a20251120 wheels in a separate venv (~11 GB)
./build_strata.sh rocm7   <rocm7 site-packages> x     # control build (10-20 min)
./build_strata.sh rocm101 <rocm10.1 site-packages> x  # ROCm 10.1 build
./run_one.sh <name> <exe> <lib dirs> <port> <tag>      # start via serve.server, run client.py, stop
python3 summarize.py *.results.json
```
Paths inside the scripts are absolute to the author's machine; adapt them. Leftovers on disk: `~/rocm-env/strata-bench` (~11 GB for the ROCm 7.10a venv, 0.5 GB builds).
