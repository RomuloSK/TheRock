# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Opt-in runtime performance profile for stock ROCm PyTorch wheels.

Sets environment defaults that stock PyTorch leaves off for some GPUs, *before*
torch is imported. Never overrides a variable the user already set.

    python rocm_perf_profile.py -- python my_script.py     # launcher
    import rocm_perf_profile; rocm_perf_profile.apply()     # before `import torch`

Profiles (selected by GPU architecture, detected without importing torch):
  gfx11 (RDNA3): enable fused (AOTriton) scaled-dot-product attention. Measured on
  gfx1102 / ROCm 10.1.0: 8-23x faster attention vs the unfused math fallback, with
  fp16 error equal to the math path (forward only; see experiments/EXPERIMENTS.md).
Use --dry-run / `describe()` to see what would be set. Set ROCM_PERF_PROFILE=0 to disable.
"""
import os
import re
import shutil
import subprocess
import sys

PROFILES = {
    "gfx11": {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"},
}


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


def settings_for(arches):
    result = {}
    for arch in arches:
        for prefix, env in PROFILES.items():
            if arch.startswith(prefix):
                result.update(env)
    return result


def describe(environ=None):
    environ = os.environ if environ is None else environ
    arches = detect_arches(environ)
    wanted = settings_for(arches)
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
