#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Reproduce the RX 7600 (gfx1102) test environment used in docs/performance/rx7600.md.
# Usage: setup_rx7600_env.sh [target_dir]   (default: ~/rocm-env). No root needed; requires `uv`.
set -euo pipefail
DIR="${1:-$HOME/rocm-env}"
INDEX="https://stable.repo.amd.com/rocm/whl-next/"
mkdir -p "$DIR" && cd "$DIR"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install --index-url "$INDEX" "rocm[libraries,devel,device-gfx1102]" "torch[device-gfx1102]"
uv pip install transformers accelerate numpy safetensors
echo "--- verify"
rocm-sdk targets | tr ';' '\n' | grep -x gfx1102 >/dev/null && echo "SDK knows gfx1102"
python - <<'PY'
import torch
print("torch", torch.__version__, "| gpu:", torch.cuda.get_device_name(0), "| available:", torch.cuda.is_available())
PY
echo "activate with: . $DIR/.venv/bin/activate"
