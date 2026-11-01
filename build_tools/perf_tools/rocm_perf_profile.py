# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Opt-in runtime performance profile for stock ROCm PyTorch wheels.

Sets environment defaults that stock PyTorch leaves off for some GPUs, *before*
torch is imported. Never overrides a variable the user already set.

    python rocm_perf_profile.py -- python my_script.py     # launcher
    import rocm_perf_profile; rocm_perf_profile.apply()     # before `import torch`

Profile (selected by GPU architecture, detected without importing torch):
  gfx1102 (validated): enable fused (AOTriton) scaled-dot-product attention. Measured on
  gfx1102 / ROCm 10.1.0: 8-23x faster attention vs the unfused math fallback, forward+backward
  checked against fp64 (see examples/gfx1102/). gfx1101/1150/1151/1200 are gated by the same flag in
  PyTorch but untested here: set ROCM_PERF_INCLUDE_UNVALIDATED=1 to include them at your own risk.
Use --dry-run / `describe()` to see what would be set. Set ROCM_PERF_PROFILE=0 to disable.
"""
import os
import re
import shutil
import subprocess
import sys

FUSED_ATTENTION = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
# Exact architectures for which PyTorch 2.14+rocm10.1.0 gates fused (AOTriton) SDPA behind the flag
# (torch/testing/_internal/common_cuda.py). gfx1100 is already enabled by default, so it is a no-op.
VALIDATED = ("gfx1102",)                                      # measured and checked in this repo
GATED_UNVALIDATED = ("gfx1101", "gfx1150", "gfx1151", "gfx1200")  # same gate, NOT tested here


def detect_arches(environ=None):
    environ = os.environ if environ is None else environ
    """GPU arch names from rocminfo (or ROCM_PERF_ARCH override); [] if unknown."""
    override = environ.get("ROCM_PERF_ARCH")
    if override:
        return override.split(";")
    exe = shutil.which("rocminfo")
    if not exe:
        try:
            import rocm_sdk  # noqa: F401  (only to locate bundled rocminfo)
            root = subprocess.run([sys.executable, "-m", "rocm_sdk", "path", "--bin"],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
            cand = os.path.join(root, "rocminfo")
            exe = cand if os.path.exists(cand) else None
        except Exception:
            exe = None
    if not exe:
        return []
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    return sorted(set(re.findall(r"^\s*Name:\s+(gfx[0-9a-f]+)\s*$", out, re.M)))


def settings_for(arches, include_unvalidated=False):
    targets = VALIDATED + (GATED_UNVALIDATED if include_unvalidated else ())
    return dict(FUSED_ATTENTION) if any(a in targets for a in arches) else {}


def describe(environ=None):
    environ = os.environ if environ is None else environ
    arches = detect_arches(environ)
    wanted = settings_for(arches, environ.get("ROCM_PERF_INCLUDE_UNVALIDATED") == "1")
    return {"arches": arches,
            "would_set": {k: v for k, v in wanted.items() if k not in environ},
            "user_overrides": {k: environ[k] for k in wanted if k in environ}}


def apply(environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("ROCM_PERF_PROFILE") == "0":
        return {}
    applied = describe(environ)["would_set"]
    environ.update(applied)
    return applied


def main(argv):
    dry = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    if dry:
        import json
        print(json.dumps(describe(), indent=1))
        return 0
    if "--" not in argv or argv.index("--") == len(argv) - 1:
        print("usage: rocm_perf_profile.py [--dry-run] -- command [args...]", file=sys.stderr)
        return 2
    cmd = argv[argv.index("--") + 1:]
    apply()
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
