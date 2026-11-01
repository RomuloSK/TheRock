# Offline paired performance gates

`paired_benchmark.py` runs a baseline and candidate as independent processes in
randomized pairs, saves raw observations, and compares them offline. It uses only
the Python standard library, so comparison does not depend on the results API.
It is currently an opt-in tool and is not yet wired into the existing nightly
benchmark adapters or CI pass/fail policy; integrate only after shadow runs and
hardware A/A calibration.

This is a harness, not a GPU profiler. `process_wall_seconds` includes process
startup and is appropriate only when end-to-end command latency is the desired
metric. For a device/application metric, configure the command to emit exactly
one JSON object on stdout and select a positive numeric field with
`json_stdout`. The command itself is responsible for measuring that metric (for
example, using synchronized HIP events); validate its timing method separately.
Send verbose diagnostics to stderr.

## Spec

```json
{
  "schema_version": 1,
  "benchmark": "example-gemm",
  "pairs": 10,
  "minimum_pairs": 10,
  "warmups": 2,
  "seed": 7301,
  "timeout_seconds": 300,
  "confidence_level": 0.95,
  "bootstrap_samples": 10000,
  "minimum_speedup": 1.03,
  "maximum_regression": 1.03,
  "working_directory": ".",
  "record_environment": ["ROCM_PATH", "HIP_VISIBLE_DEVICES"],
  "context": {
    "source_commit": "<immutable commit>",
    "gpu_target": "<verified target>",
    "driver_and_firmware": "<recorded versions>",
    "package_hashes": ["<baseline hash>", "<candidate hash>"]
  },
  "environment": {},
  "cases": [
    {
      "id": "gemm-f16-1024x1024x1024",
      "gate": "target",
      "direction": "lower",
      "unit": "ms",
      "metric": {"source": "json_stdout", "field": "latency_ms"},
      "baseline": {
        "command": ["./run-baseline", "--json"],
        "correctness_command": ["./check-baseline", "--same-contract"]
      },
      "candidate": {
        "command": ["./run-candidate", "--json"],
        "correctness_command": ["./check-candidate", "--same-contract"]
      }
    }
  ]
}
```

Correctness commands must return zero only after validating the same contract for
their respective variant. Omitting either command keeps the result `UNKNOWN`,
never a performance `PASS`. A failed correctness command is `FAIL`; process,
timeout, malformed metric, or measurement errors are not silently dropped.
Top-level environment settings apply equally to both variants. Do not put secrets
in the spec; retain the exact spec alongside results because results store its
SHA-256, not a copy of command arguments or environment values. Per-variant
environment overrides are available when isolating package/library versions;
keep the difference minimal and record it in the spec. The explicit
`record_environment` allowlist copies selected effective values into raw results
for auditability; never include secret-bearing variables in that allowlist.

## Run and compare

```bash
python build_tools/perf_tools/paired_benchmark.py run \
  --spec specs/example.json \
  --output results/example-run.json

python build_tools/perf_tools/paired_benchmark.py compare \
  --spec specs/example.json \
  --results results/example-run.json \
  --output results/example-comparison.json
```

An existing output or partial-output path is never overwritten. During a run, an
atomically updated `.partial` file preserves observations if execution is
interrupted. Preserve the spec, raw results, comparison report, source/package
locks, and correctness evidence together. Captured stdout/stderr excerpts may
contain sensitive application output; protect the result files accordingly.

The comparator uses paired log ratios and a deterministic percentile bootstrap
confidence interval. For `direction: lower`, speedup is baseline/candidate; for
`higher`, it is candidate/baseline. `PASS` requires the lower confidence bound
to meet `minimum_speedup`; a confidence interval wholly beyond the configured
regression limit is `REGRESSION`. Incomplete pairs, absent correctness checks,
and inconclusive intervals are `UNKNOWN`. A nonzero CLI status is returned for
all states except `PASS`: `FAIL`=1, `UNKNOWN`=2, input/execution `ERROR`=3, and
`REGRESSION`=4.

The default 10 pairs and 3% threshold are starting policy values, not universal
statistical guarantees. Calibrate them with A/A runs on the target host and
report independent process/day blocks for long-run conclusions. A passing
result is limited to the exact spec and recorded run; it is not evidence for
other shapes, architectures, packages, or applications.

The report gives per-case intervals and does not apply a multiple-comparison
correction across a large case matrix. Use a predeclared sentinel panel or an
explicit family-wise policy for hard CI gates; keep exploratory cases separate.
Mark at least one workload as `gate: "target"` (it must exceed the gain
threshold). Mark protected workloads as `gate: "guard"`; they pass only when the
lower confidence bound establishes that any slowdown remains within the
configured regression limit. This avoids requiring every protected shape to
improve by the target's 3% gain threshold.
