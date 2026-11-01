set -e
cd ~/rocm-env/strata-bench
# 1. build tools
[ -d bt ] || uv venv --python 3.12 bt
. bt/bin/activate && uv pip install cmake==4.4.3 ninja==1.13.2 >/dev/null && deactivate
# 2. pinned llama.cpp
if [ ! -d build/llama.cpp/ggml ]; then
  C=3cf03257f219afbe7334045ff7c6a06ac68c627d
  curl -fL -o build/llama.zip https://github.com/ggml-org/llama.cpp/archive/$C.zip
  rm -rf build/_u && mkdir build/_u && unzip -q build/llama.zip -x '*/tools/ui/*' -d build/_u
  mv build/_u/llama.cpp-$C build/llama.cpp
fi
# 3. control ROCm 7.10.0a20251120 (what Strata's own Linux rx7900 package uses)
[ -d rocm7 ] || uv venv --python 3.12 rocm7
. rocm7/bin/activate
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ "rocm[libraries,devel]==7.10.0a20251120" 
rocm-sdk path --root
echo PREP_DONE
