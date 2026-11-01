#!/usr/bin/env bash
# usage: build_strata.sh NAME SITE_PACKAGES_DIR LIBS_SUBSTR   (builds Strata HIP engine for gfx1102 against that ROCm)
set -euo pipefail
NAME=$1; SP=$2; FAM=$3
B=~/rocm-env/strata-bench; SRC=~/Documents/Codex/2026-10-31/setu/work/strata-src
ROOT=$SP/_rocm_sdk_devel
LIBS=$(ls -d $SP/_rocm_sdk_libraries* | head -1)
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$ROOT HIP_PATH=$ROOT
export LD_LIBRARY_PATH=$ROOT/lib:$LIBS/lib:${LD_LIBRARY_PATH:-}
export PATH=$B/bt/bin:$ROOT/bin:$ROOT/llvm/bin:$PATH
BITCODE=$ROOT/amdgcn/bitcode; [ -d "$BITCODE" ] || BITCODE=$ROOT/lib/llvm/amdgcn/bitcode
cmake -S $SRC -B $B/build/$NAME -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1102 \
  -DCMAKE_HIP_COMPILER=$ROOT/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROOT \
  "-DCMAKE_PREFIX_PATH=$ROOT;$LIBS" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$ROOT --rocm-device-lib-path=$BITCODE" \
  -DSTRATA_GGML_DIR=$B/build/llama.cpp
cmake --build $B/build/$NAME --target strata -j10
ls -la $B/build/$NAME/strata
echo BUILD_DONE_$NAME
