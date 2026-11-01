#!/usr/bin/env bash
# usage: build_llama.sh NAME SITE_PACKAGES   (llama.cpp HIP backend for gfx1102 against that ROCm)
set -euo pipefail
NAME=$1; SP=$2; B=~/rocm-env/llama-bench; SRC=~/rocm-env/strata-bench/build/llama.cpp
ROOT=$SP/_rocm_sdk_devel; LIBS=$(ls -d $SP/_rocm_sdk_libraries* | head -1)
export HIP_PLATFORM=amd ROCM_PATH=$ROOT HIP_PATH=$ROOT
export LD_LIBRARY_PATH=$ROOT/lib:$LIBS/lib:${LD_LIBRARY_PATH:-}
export PATH=$HOME/rocm-env/strata-bench/bt/bin:$ROOT/bin:$ROOT/llvm/bin:$PATH
BITCODE=$ROOT/amdgcn/bitcode; [ -d "$BITCODE" ] || BITCODE=$ROOT/lib/llvm/amdgcn/bitcode
cmake -S $SRC -B $B/build/$NAME -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1102 -DAMDGPU_TARGETS=gfx1102 \
  -DCMAKE_HIP_COMPILER=$ROOT/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROOT -DCMAKE_C_COMPILER=$ROOT/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROOT/llvm/bin/clang++ \
  "-DCMAKE_PREFIX_PATH=$ROOT;$LIBS" "-DCMAKE_HIP_FLAGS=--rocm-path=$ROOT --rocm-device-lib-path=$BITCODE" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON
cmake --build $B/build/$NAME --target llama-bench llama-perplexity -j10
ls -la $B/build/$NAME/bin/llama-bench; echo BUILD_DONE_$NAME
