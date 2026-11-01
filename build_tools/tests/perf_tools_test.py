# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf_tools.paired_benchmark import (  # noqa: E402
    SpecError,
    compare_results,
    main,
    run_benchmark,
    validate_spec,
)


def _python_json_command(value):
    return [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'latency_ms': %r}))" % value,
    ]


def _spec(baseline=10.0, candidate=5.0):
    return {
        "schema_version": 1,
        "benchmark": "synthetic",
        "pairs": 4,
        "minimum_pairs": 2,
        "warmups": 1,
        "seed": 73,
        "bootstrap_samples": 200,
        "confidence_level": 0.95,
        "minimum_speedup": 1.5,
        "maximum_regression": 1.05,
        "cases": [
            {
                "id": "case-a",
                "direction": "lower",
                "unit": "ms",
                "metric": {"source": "json_stdout", "field": "latency_ms"},
                "baseline": {
                    "command": _python_json_command(baseline),
                    "correctness_command": [sys.executable, "-c", "pass"],
                },
                "candidate": {
                    "command": _python_json_command(candidate),
                    "correctness_command": [sys.executable, "-c", "pass"],
                },
            }
        ],
    }


class PairedBenchmarkTest(unittest.TestCase):
    def _run(self, spec):
        validate_spec(spec)
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            result = run_benchmark(
                spec,
                "test-spec-digest",
                spec_path,
            )
        return result

    def test_complete_correct_paired_gain_passes(self):
        spec = _spec()
        result = self._run(spec)

        measurements = [run for run in result["runs"] if run["stage"] == "measurement"]
        self.assertEqual(len(measurements), spec["pairs"] * 2)
        for pair_index in range(spec["pairs"]):
            pair = [run for run in measurements if run["pair_index"] == pair_index]
            self.assertEqual(
                {run["variant"] for run in pair}, {"baseline", "candidate"}
            )

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "PASS")
        self.assertEqual(report["cases"][0]["geometric_mean_speedup"], 2.0)
        self.assertEqual(report["cases"][0]["unit"], "ms")

    def test_wrong_spec_digest_is_rejected(self):
        spec = _spec()
        with self.assertRaisesRegex(SpecError, "different spec"):
            compare_results(
                spec,
                "digest-a",
                {
                    "schema_version": 1,
                    "benchmark": spec["benchmark"],
                    "spec_sha256": "digest-b",
                    "runs": [],
                    "correctness": [],
                },
            )

    def test_missing_required_pair_is_unknown(self):
        spec = _spec()
        result = self._run(spec)
        result["runs"] = [
            record
            for record in result["runs"]
            if not (
                record["stage"] == "measurement"
                and record["pair_index"] == 0
                and record["variant"] == "candidate"
            )
        ]

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "UNKNOWN")
        self.assertIn("incomplete pairs", report["cases"][0]["reason"])

    def test_correctness_failure_blocks_performance_pass(self):
        spec = _spec()
        spec["cases"][0]["candidate"]["correctness_command"] = [
            sys.executable,
            "-c",
            "raise SystemExit(1)",
        ]
        result = self._run(spec)

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "FAIL")

    def test_unconfigured_correctness_is_unknown(self):
        spec = _spec()
        del spec["cases"][0]["candidate"]["correctness_command"]
        result = self._run(spec)

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "UNKNOWN")
        self.assertIn("not configured", report["cases"][0]["reason"])

    def test_material_regression_is_not_pass(self):
        spec = _spec(baseline=5.0, candidate=10.0)
        result = self._run(spec)

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "REGRESSION")

    def test_guard_can_pass_without_target_gain(self):
        spec = _spec()
        guard = copy.deepcopy(spec["cases"][0])
        guard["id"] = "protected-case"
        guard["gate"] = "guard"
        guard["baseline"]["command"] = _python_json_command(10.0)
        guard["candidate"]["command"] = _python_json_command(10.0)
        spec["cases"].append(guard)
        result = self._run(spec)

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "PASS")
        self.assertEqual(report["cases"][1]["status"], "PASS")

    def test_guard_regression_blocks_target_gain(self):
        spec = _spec()
        guard = copy.deepcopy(spec["cases"][0])
        guard["id"] = "protected-case"
        guard["gate"] = "guard"
        guard["baseline"]["command"] = _python_json_command(10.0)
        guard["candidate"]["command"] = _python_json_command(20.0)
        spec["cases"].append(guard)
        result = self._run(spec)

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "REGRESSION")

    def test_duplicate_correctness_record_is_error(self):
        spec = _spec()
        result = self._run(spec)
        result["correctness"].append(copy.deepcopy(result["correctness"][0]))

        report = compare_results(spec, "test-spec-digest", result)
        self.assertEqual(report["overall_status"], "ERROR")

    def test_records_only_allowlisted_effective_environment(self):
        spec = _spec()
        spec["environment"] = {"PERF_TEST_BASE": "base"}
        spec["record_environment"] = ["PERF_TEST_BASE", "PERF_TEST_OVERRIDE"]
        spec["cases"][0]["candidate"]["environment"] = {
            "PERF_TEST_OVERRIDE": "candidate"
        }
        result = self._run(spec)

        self.assertEqual(
            result["environments"]["case-a"]["baseline"],
            {"PERF_TEST_BASE": "base", "PERF_TEST_OVERRIDE": None},
        )
        self.assertEqual(
            result["environments"]["case-a"]["candidate"],
            {"PERF_TEST_BASE": "base", "PERF_TEST_OVERRIDE": "candidate"},
        )

    def test_cli_writes_raw_results_and_offline_report(self):
        spec = _spec()
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "spec.json"
            raw_path = Path(directory) / "raw.json"
            report_path = Path(directory) / "report.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                run_status = main(
                    ["run", "--spec", str(spec_path), "--output", str(raw_path)]
                )
                compare_status = main(
                    [
                        "compare",
                        "--spec",
                        str(spec_path),
                        "--results",
                        str(raw_path),
                        "--output",
                        str(report_path),
                    ]
                )

            self.assertEqual(run_status, 0)
            self.assertEqual(compare_status, 0)
            self.assertFalse(Path(str(raw_path) + ".partial").exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["overall_status"], "PASS")

    def test_invalid_spec_is_rejected(self):
        spec = _spec()
        duplicate = copy.deepcopy(spec["cases"][0])
        spec["cases"].append(duplicate)

        with self.assertRaisesRegex(SpecError, "duplicate case id"):
            validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
