# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Paired process benchmark runner and offline regression gate.

This tool measures complete command invocations or consumes a positive numeric
metric from a command's JSON stdout. It does not configure GPUs, profile kernels,
or establish that a workload is representative; those remain the caller's
responsibility.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


SCHEMA_VERSION = 1
MAX_RETAINED_OUTPUT = 16 * 1024


class SpecError(ValueError):
    """Raised when a benchmark spec or result file is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError as error:
        raise SpecError(f"{name} must be finite and greater than zero") from error
    if not math.isfinite(number) or number <= 0:
        raise SpecError(f"{name} must be finite and greater than zero")
    return number


def _is_finite_positive_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return math.isfinite(number) and number > 0


def _greater_than_one(value: Any, name: str) -> float:
    number = _positive_number(value, name)
    if number <= 1:
        raise SpecError(f"{name} must be greater than one")
    return number


def _command(value: Any, name: str, required: bool) -> list[str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list) or not value:
        raise SpecError(f"{name} must be a non-empty array of strings")
    if any(not isinstance(arg, str) for arg in value) or not value[0]:
        raise SpecError(f"{name} must contain strings and a non-empty executable")
    if any("\0" in arg for arg in value):
        raise SpecError(f"{name} must not contain NUL characters")
    return value


def _environment(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or not isinstance(item, str)
        or "\0" in key
        or "\0" in item
        for key, item in value.items()
    ):
        raise SpecError(f"{name} must map valid environment variable names to strings")
    return value


def validate_spec(spec: Any) -> dict[str, Any]:
    """Validate the versioned benchmark document and return it unchanged."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise SpecError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(spec.get("benchmark"), str) or not spec["benchmark"].strip():
        raise SpecError("benchmark must be a non-empty string")

    for key, minimum in (("pairs", 1), ("warmups", 0)):
        value = spec.get(key)
        if type(value) is not int or value < minimum:
            raise SpecError(f"{key} must be an integer >= {minimum}")

    _positive_number(spec.get("timeout_seconds", 300), "timeout_seconds")

    seed = spec.get("seed", 7301)
    if type(seed) is not int:
        raise SpecError("seed must be an integer")

    confidence = spec.get("confidence_level", 0.95)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not _is_finite_positive_number(confidence)
        or not 0 < confidence < 1
    ):
        raise SpecError("confidence_level must be between zero and one")

    bootstrap_samples = spec.get("bootstrap_samples", 10000)
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise SpecError("bootstrap_samples must be an integer >= 100")

    minimum_pairs = spec.get("minimum_pairs", 10)
    if type(minimum_pairs) is not int or minimum_pairs < 2:
        raise SpecError("minimum_pairs must be an integer >= 2")

    _greater_than_one(spec.get("minimum_speedup", 1.03), "minimum_speedup")
    _greater_than_one(spec.get("maximum_regression", 1.03), "maximum_regression")

    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SpecError("cases must be a non-empty array")
    case_ids = set()
    has_target_case = False
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            raise SpecError(f"{prefix} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise SpecError(f"{prefix}.id must be a non-empty string")
        if case_id in case_ids:
            raise SpecError(f"duplicate case id: {case_id}")
        case_ids.add(case_id)
        gate = case.get("gate", "target")
        if not isinstance(gate, str) or gate not in {"target", "guard"}:
            raise SpecError(f"{prefix}.gate must be 'target' or 'guard'")
        has_target_case = has_target_case or gate == "target"
        if "minimum_speedup" in case:
            _greater_than_one(case["minimum_speedup"], f"{prefix}.minimum_speedup")
        if "maximum_regression" in case:
            _greater_than_one(
                case["maximum_regression"], f"{prefix}.maximum_regression"
            )
        if not isinstance(case.get("direction"), str) or case["direction"] not in {
            "lower",
            "higher",
        }:
            raise SpecError(f"{prefix}.direction must be 'lower' or 'higher'")

        metric = case.get("metric", {"source": "process_wall_seconds"})
        if not isinstance(metric, dict):
            raise SpecError(f"{prefix}.metric must be an object")
        source = metric.get("source", "process_wall_seconds")
        if source == "json_stdout":
            if not isinstance(metric.get("field"), str) or not metric["field"]:
                raise SpecError(f"{prefix}.metric.field is required for json_stdout")
        elif source != "process_wall_seconds":
            raise SpecError(
                f"{prefix}.metric.source must be 'process_wall_seconds' "
                "or 'json_stdout'"
            )
        unit = case.get("unit", "seconds" if source == "process_wall_seconds" else None)
        if not isinstance(unit, str) or not unit.strip():
            raise SpecError(f"{prefix}.unit must name the measurement unit")

        for variant_name in ("baseline", "candidate"):
            variant = case.get(variant_name)
            if not isinstance(variant, dict):
                raise SpecError(f"{prefix}.{variant_name} must be an object")
            _command(
                variant.get("command"),
                f"{prefix}.{variant_name}.command",
                required=True,
            )
            _command(
                variant.get("correctness_command"),
                f"{prefix}.{variant_name}.correctness_command",
                required=False,
            )
            if "environment" in variant:
                _environment(
                    variant["environment"],
                    f"{prefix}.{variant_name}.environment",
                )

    if not has_target_case:
        raise SpecError("at least one case must use gate 'target'")

    _environment(spec.get("environment", {}), "environment")
    record_environment = spec.get("record_environment", [])
    if not isinstance(record_environment, list) or any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\0" in key
        for key in record_environment
    ):
        raise SpecError("record_environment must be an array of variable names")
    if len(set(record_environment)) != len(record_environment):
        raise SpecError("record_environment must not contain duplicates")
    if "context" in spec and not isinstance(spec["context"], dict):
        raise SpecError("context must be an object")

    working_directory = spec.get("working_directory", ".")
    if not isinstance(working_directory, str) or not working_directory:
        raise SpecError("working_directory must be a non-empty path string")

    try:
        _json_bytes(spec)
    except (TypeError, ValueError) as error:
        raise SpecError(f"spec is not valid JSON data: {error}") from error
    return spec


def load_spec(path: Path) -> tuple[dict[str, Any], str]:
    """Load a spec and return it with a digest of its exact on-disk bytes."""
    try:
        raw = path.read_bytes()
        spec = json.loads(raw)
    except OSError as error:
        raise SpecError(f"cannot read spec {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SpecError(f"invalid JSON spec {path}: {error}") from error
    return validate_spec(spec), _sha256(raw)


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _excerpt(value: str) -> str:
    if len(value) <= MAX_RETAINED_OUTPUT:
        return value
    return value[:MAX_RETAINED_OUTPUT] + "\n[output truncated]"


def _run_process(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    start_ns = time.perf_counter_ns()
    returncode = None
    error = None
    stdout = ""
    stderr = ""
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        returncode = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exception:
        error = f"timed out after {timeout_seconds:g} seconds"
        stdout = _decode_output(exception.stdout)
        stderr = _decode_output(exception.stderr)
    except OSError as exception:
        error = f"could not execute command: {exception}"
    elapsed_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000
    return {
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }


def _measurement(
    process_result: dict[str, Any], metric: dict[str, Any]
) -> tuple[float | None, str | None]:
    if process_result["returncode"] != 0 or process_result["error"]:
        return None, None
    if metric.get("source", "process_wall_seconds") == "process_wall_seconds":
        value = process_result["elapsed_seconds"]
    else:
        try:
            output = json.loads(process_result["stdout"])
            value = output[metric["field"]]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            return None, f"could not read JSON stdout field: {error}"
    if not _is_finite_positive_number(value):
        return None, "measurement must be finite and greater than zero"
    return float(value), None


def _record_invocation(
    case_id: str,
    variant_name: str,
    variant: dict[str, Any],
    stage: str,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    metric: dict[str, Any],
    pair_index: int | None = None,
    order_index: int | None = None,
) -> dict[str, Any]:
    command = variant.get("correctness_command") if stage == "correctness" else variant[
        "command"
    ]
    command_environment = environment.copy()
    command_environment.update(variant.get("environment", {}))
    process_result = _run_process(
        command, cwd, command_environment, timeout_seconds
    )
    metric_value = None
    error = process_result["error"]
    if stage == "measurement":
        metric_value, metric_error = _measurement(process_result, metric)
        error = error or metric_error

    return {
        "case_id": case_id,
        "variant": variant_name,
        "stage": stage,
        "pair_index": pair_index,
        "order_index": order_index,
        "status": (
            "ERROR"
            if error or process_result["returncode"] is None
            else "PASS"
            if process_result["returncode"] == 0
            else "FAIL"
            if stage == "correctness"
            else "ERROR"
        ),
        "returncode": process_result["returncode"],
        "elapsed_seconds": process_result["elapsed_seconds"],
        "metric_value": metric_value,
        "command_sha256": _sha256(_json_bytes(command)),
        "error": error,
        "stdout_excerpt": _excerpt(process_result["stdout"]),
        "stderr_excerpt": _excerpt(process_result["stderr"]),
    }


def run_benchmark(
    spec: dict[str, Any],
    spec_sha256: str,
    spec_path: Path,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run correctness checks, warmups, and randomized paired measurements."""
    validate_spec(spec)
    cwd_value = Path(spec.get("working_directory", "."))
    cwd = cwd_value if cwd_value.is_absolute() else spec_path.parent / cwd_value
    cwd = cwd.resolve()
    if not cwd.is_dir():
        raise SpecError(f"working directory does not exist: {cwd}")

    environment = os.environ.copy()
    environment.update(spec.get("environment", {}))
    timeout_seconds = float(spec.get("timeout_seconds", 300))
    randomizer = random.Random(spec.get("seed", 7301))
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": spec["benchmark"],
        "spec_sha256": spec_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "context": spec.get("context", {}),
        "working_directory": str(cwd),
        "environments": {},
        "parameters": {
            "pairs": spec["pairs"],
            "warmups": spec["warmups"],
            "seed": spec.get("seed", 7301),
            "timeout_seconds": timeout_seconds,
        },
        "correctness": [],
        "runs": [],
    }

    def append_record(collection: str, record: dict[str, Any]) -> None:
        results[collection].append(record)
        if on_progress is not None:
            on_progress(results)

    for case in spec["cases"]:
        case_id = case["id"]
        metric = case.get("metric", {"source": "process_wall_seconds"})
        variants = ["baseline", "candidate"]
        case_correctness: dict[str, str] = {}
        results["environments"][case_id] = {}

        for variant_name in variants:
            variant = case[variant_name]
            effective_environment = environment.copy()
            effective_environment.update(variant.get("environment", {}))
            results["environments"][case_id][variant_name] = {
                key: effective_environment.get(key)
                for key in spec.get("record_environment", [])
            }
            if variant.get("correctness_command") is None:
                record = {
                    "case_id": case_id,
                    "variant": variant_name,
                    "configured": False,
                    "status": "UNKNOWN",
                    "error": "no correctness_command configured",
                }
            else:
                invocation = _record_invocation(
                    case_id,
                    variant_name,
                    variant,
                    "correctness",
                    cwd,
                    environment,
                    timeout_seconds,
                    metric,
                )
                record = {
                    "case_id": case_id,
                    "variant": variant_name,
                    "configured": True,
                    "status": invocation["status"],
                    "returncode": invocation["returncode"],
                    "elapsed_seconds": invocation["elapsed_seconds"],
                    "command_sha256": invocation["command_sha256"],
                    "error": invocation["error"],
                    "stdout_excerpt": invocation["stdout_excerpt"],
                    "stderr_excerpt": invocation["stderr_excerpt"],
                }
            case_correctness[variant_name] = record["status"]
            append_record("correctness", record)

        if any(status in {"FAIL", "ERROR"} for status in case_correctness.values()):
            continue

        warmup_failed = False
        for warmup_index in range(spec["warmups"]):
            order = variants.copy()
            randomizer.shuffle(order)
            for order_index, variant_name in enumerate(order):
                record = _record_invocation(
                    case_id,
                    variant_name,
                    case[variant_name],
                    "warmup",
                    cwd,
                    environment,
                    timeout_seconds,
                    metric,
                    pair_index=warmup_index,
                    order_index=order_index,
                )
                append_record("runs", record)
                warmup_failed = warmup_failed or record["status"] != "PASS"
        if warmup_failed:
            continue

        for pair_index in range(spec["pairs"]):
            order = variants.copy()
            randomizer.shuffle(order)
            for order_index, variant_name in enumerate(order):
                record = _record_invocation(
                    case_id,
                    variant_name,
                    case[variant_name],
                    "measurement",
                    cwd,
                    environment,
                    timeout_seconds,
                    metric,
                    pair_index=pair_index,
                    order_index=order_index,
                )
                append_record("runs", record)

    return results


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _bootstrap_interval(
    speedups: list[float],
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    log_speedups = [math.log(value) for value in speedups]
    estimate = math.exp(sum(log_speedups) / len(log_speedups))
    randomizer = random.Random(seed)
    bootstrapped = []
    for _ in range(bootstrap_samples):
        sampled = [randomizer.choice(log_speedups) for _ in log_speedups]
        bootstrapped.append(math.exp(sum(sampled) / len(sampled)))
    bootstrapped.sort()
    tail_probability = (1 - confidence_level) / 2
    return (
        estimate,
        _percentile(bootstrapped, tail_probability),
        _percentile(bootstrapped, 1 - tail_probability),
    )


def compare_results(
    spec: dict[str, Any], spec_sha256: str, results: Any
) -> dict[str, Any]:
    """Compare a result bundle using paired log ratios and a bootstrap CI."""
    validate_spec(spec)
    if not isinstance(results, dict):
        raise SpecError("results must be a JSON object")
    if results.get("schema_version") != SCHEMA_VERSION:
        raise SpecError(f"results schema_version must be {SCHEMA_VERSION}")
    if results.get("benchmark") != spec["benchmark"]:
        raise SpecError("benchmark name does not match spec")
    if results.get("spec_sha256") != spec_sha256:
        raise SpecError("results were produced from a different spec")
    parameters = results.get("parameters")
    expected_parameters = {
        "pairs": spec["pairs"],
        "warmups": spec["warmups"],
        "seed": spec.get("seed", 7301),
        "timeout_seconds": float(spec.get("timeout_seconds", 300)),
    }
    if not isinstance(parameters, dict) or any(
        parameters.get(key) != value for key, value in expected_parameters.items()
    ):
        raise SpecError("result run parameters do not match spec")
    if results.get("context") != spec.get("context", {}):
        raise SpecError("result context does not match spec")
    if not isinstance(results.get("working_directory"), str):
        raise SpecError("results are missing working-directory provenance")
    if not isinstance(results.get("host"), dict):
        raise SpecError("results are missing host provenance")
    environments = results.get("environments")
    if not isinstance(environments, dict):
        raise SpecError("results are missing recorded environments")
    recorded_names = set(spec.get("record_environment", []))
    if set(environments) != {case["id"] for case in spec["cases"]}:
        raise SpecError("recorded environment cases do not match spec")
    for case_id, variant_environments in environments.items():
        if not isinstance(variant_environments, dict) or set(variant_environments) != {
            "baseline",
            "candidate",
        }:
            raise SpecError(f"invalid recorded environments for case {case_id}")
        for recorded_values in variant_environments.values():
            if (
                not isinstance(recorded_values, dict)
                or set(recorded_values) != recorded_names
                or any(
                    value is not None and not isinstance(value, str)
                    for value in recorded_values.values()
                )
            ):
                raise SpecError(f"invalid environment values for case {case_id}")
    runs = results.get("runs")
    correctness = results.get("correctness")
    if not isinstance(runs, list) or not isinstance(correctness, list):
        raise SpecError("results must contain runs and correctness arrays")

    required_case_ids = {case["id"] for case in spec["cases"]}
    all_records = [*runs, *correctness]
    for record in all_records:
        if not isinstance(record, dict) or not isinstance(record.get("case_id"), str):
            raise SpecError(
                "each run and correctness record must have a string case_id"
            )
    result_case_ids = {record["case_id"] for record in all_records}
    unknown_ids = result_case_ids - required_case_ids
    if unknown_ids:
        raise SpecError(f"results contain unrequested cases: {unknown_ids}")

    minimum_pairs = spec.get("minimum_pairs", 10)
    requested_pairs = spec["pairs"]
    confidence_level = float(spec.get("confidence_level", 0.95))
    bootstrap_samples = spec.get("bootstrap_samples", 10000)
    default_minimum_speedup = float(spec.get("minimum_speedup", 1.03))
    default_maximum_regression = float(spec.get("maximum_regression", 1.03))
    base_seed = spec.get("seed", 7301)
    case_reports = []

    for case in spec["cases"]:
        case_id = case["id"]
        gate = case.get("gate", "target")
        metric = case.get("metric", {})
        unit = case.get(
            "unit",
            "seconds"
            if metric.get("source", "process_wall_seconds") == "process_wall_seconds"
            else None,
        )
        minimum_speedup = float(
            case.get("minimum_speedup", default_minimum_speedup)
        )
        maximum_regression = float(
            case.get("maximum_regression", default_maximum_regression)
        )
        case_runs = [
            record
            for record in runs
            if isinstance(record, dict)
            and record.get("case_id") == case_id
            and record.get("stage") == "measurement"
        ]
        case_checks = [
            record
            for record in correctness
            if isinstance(record, dict) and record.get("case_id") == case_id
        ]
        check_by_variant: dict[str, dict[str, Any]] = {}
        invalid_check = False
        for record in case_checks:
            variant = record.get("variant")
            if (
                not isinstance(variant, str)
                or variant not in {"baseline", "candidate"}
                or variant in check_by_variant
            ):
                invalid_check = True
                break
            check_by_variant[variant] = record

        if invalid_check:
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "ERROR",
                    "reason": "duplicate or invalid correctness record",
                }
            )
            continue

        if any(
            check.get("status") == "FAIL" for check in check_by_variant.values()
        ):
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "reason": "correctness check failed",
                }
            )
            continue
        if any(
            check.get("status") == "ERROR" for check in check_by_variant.values()
        ):
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "ERROR",
                    "reason": "correctness check errored",
                }
            )
            continue

        if any(
            case[variant].get("correctness_command") is not None
            and (
                variant not in check_by_variant
                or check_by_variant[variant].get("status") != "PASS"
            )
            for variant in ("baseline", "candidate")
        ):
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "UNKNOWN",
                    "reason": "correctness result missing",
                }
            )
            continue
        if any(
            case[variant].get("correctness_command") is None
            for variant in ("baseline", "candidate")
        ):
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "UNKNOWN",
                    "reason": "correctness command not configured for both variants",
                }
            )
            continue

        by_pair: dict[int, dict[str, dict[str, Any]]] = {}
        invalid_record = False
        for record in case_runs:
            variant = record.get("variant")
            pair_index = record.get("pair_index")
            if (
                not isinstance(variant, str)
                or variant not in {"baseline", "candidate"}
                or type(pair_index) is not int
                or pair_index < 0
                or pair_index >= requested_pairs
            ):
                invalid_record = True
                break
            pair = by_pair.setdefault(pair_index, {})
            if variant in pair:
                invalid_record = True
                break
            pair[variant] = record

        if invalid_record:
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "ERROR",
                    "reason": "duplicate or invalid paired sample",
                }
            )
            continue

        if any(
            record.get("status") != "PASS"
            for pair in by_pair.values()
            for record in pair.values()
        ):
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "ERROR",
                    "reason": "measurement command failed",
                }
            )
            continue

        if any(
            not _is_finite_positive_number(record.get("metric_value"))
            for pair in by_pair.values()
            for record in pair.values()
        ):
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "ERROR",
                    "reason": "measurement must be finite and greater than zero",
                }
            )
            continue

        complete_pairs = [
            index
            for index in range(requested_pairs)
            if index in by_pair
            and set(by_pair[index]) == {"baseline", "candidate"}
        ]
        if len(complete_pairs) != requested_pairs:
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "UNKNOWN",
                    "reason": (
                        f"incomplete pairs: {len(complete_pairs)}/{requested_pairs}"
                    ),
                    "complete_pairs": len(complete_pairs),
                }
            )
            continue
        if len(complete_pairs) < minimum_pairs:
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "UNKNOWN",
                    "reason": (
                        f"only {len(complete_pairs)} pairs; "
                        f"{minimum_pairs} required"
                    ),
                    "complete_pairs": len(complete_pairs),
                }
            )
            continue

        speedups = []
        for pair_index in complete_pairs:
            baseline = float(by_pair[pair_index]["baseline"]["metric_value"])
            candidate = float(by_pair[pair_index]["candidate"]["metric_value"])
            if case["direction"] == "lower":
                speedups.append(baseline / candidate)
            else:
                speedups.append(candidate / baseline)

        if any(not math.isfinite(speedup) or speedup <= 0 for speedup in speedups):
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "ERROR",
                    "reason": "invalid speedup ratio",
                }
            )
            continue

        case_seed = int(_sha256(case_id.encode("utf-8"))[:8], 16) ^ base_seed
        estimate, lower, upper = _bootstrap_interval(
            speedups, confidence_level, bootstrap_samples, case_seed
        )
        if gate == "guard" and lower >= 1 / maximum_regression:
            status = "PASS"
            reason = (
                f"lower confidence bound stays within "
                f"{maximum_regression:g}x regression limit"
            )
        elif gate == "target" and lower >= minimum_speedup:
            status = "PASS"
            reason = f"lower confidence bound meets {minimum_speedup:g}x target"
        elif upper <= 1 / maximum_regression:
            status = "REGRESSION"
            reason = (
                f"upper confidence bound exceeds {maximum_regression:g}x "
                "regression limit"
            )
        else:
            status = "UNKNOWN"
            reason = "evidence does not establish a gain or material regression"
        case_reports.append(
            {
                "case_id": case_id,
                "gate": gate,
                "unit": unit,
                "status": status,
                "reason": reason,
                "complete_pairs": len(complete_pairs),
                "geometric_mean_speedup": estimate,
                "confidence_level": confidence_level,
                "speedup_ci": [lower, upper],
            }
        )

    statuses = {report["status"] for report in case_reports}
    if "FAIL" in statuses:
        overall_status = "FAIL"
    elif "ERROR" in statuses:
        overall_status = "ERROR"
    elif "REGRESSION" in statuses:
        overall_status = "REGRESSION"
    elif "UNKNOWN" in statuses:
        overall_status = "UNKNOWN"
    else:
        overall_status = "PASS"

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": spec["benchmark"],
        "spec_sha256": spec_sha256,
        "overall_status": overall_status,
        "cases": case_reports,
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
        output.write(_json_bytes(value).decode("utf-8"))


def _write_json_snapshot(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(_json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _run_command(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec, spec_sha256 = load_spec(spec_path)
    output_path = Path(args.output).resolve()
    partial_path = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial_path.exists():
        raise SpecError(f"output or partial output already exists: {output_path}")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("x", encoding="utf-8"):
        pass

    result = run_benchmark(
        spec,
        spec_sha256,
        spec_path,
        on_progress=lambda snapshot: _write_json_snapshot(partial_path, snapshot),
    )
    _write_json_exclusive(output_path, result)
    partial_path.unlink(missing_ok=True)
    print(f"Wrote raw paired results to {output_path}")
    if any(
        record.get("status") in {"ERROR", "FAIL"}
        for record in [*result["runs"], *result["correctness"]]
    ):
        return 1
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    spec, spec_sha256 = load_spec(Path(args.spec).resolve())
    try:
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    except OSError as error:
        raise SpecError(f"cannot read results {args.results}: {error}") from error
    except json.JSONDecodeError as error:
        raise SpecError(f"invalid results JSON: {error}") from error

    report = compare_results(spec, spec_sha256, results)
    if args.output:
        _write_json_exclusive(Path(args.output).resolve(), report)

    print(f"{report['benchmark']}: {report['overall_status']}")
    for case in report["cases"]:
        summary = f"  {case['case_id']}: {case['status']} ({case['reason']})"
        if "geometric_mean_speedup" in case:
            low, high = case["speedup_ci"]
            summary += (
                f"; speedup={case['geometric_mean_speedup']:.4f}x"
                f"; {case['confidence_level']:.0%} CI=[{low:.4f}x, {high:.4f}x]"
            )
        print(summary)
    if args.output:
        print(f"Wrote comparison report to {Path(args.output).resolve()}")

    return {
        "PASS": 0,
        "FAIL": 1,
        "UNKNOWN": 2,
        "ERROR": 3,
        "REGRESSION": 4,
    }[report["overall_status"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser(
        "run", help="collect randomized paired samples"
    )
    run_parser.add_argument(
        "--spec", required=True, help="versioned JSON benchmark spec"
    )
    run_parser.add_argument("--output", required=True, help="new raw-results JSON path")
    run_parser.set_defaults(handler=_run_command)

    compare_parser = subparsers.add_parser(
        "compare", help="compare complete paired results offline"
    )
    compare_parser.add_argument(
        "--spec", required=True, help="the original benchmark spec"
    )
    compare_parser.add_argument(
        "--results", required=True, help="raw-results JSON path"
    )
    compare_parser.add_argument("--output", help="new JSON report path")
    compare_parser.set_defaults(handler=_run_compare)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (SpecError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
