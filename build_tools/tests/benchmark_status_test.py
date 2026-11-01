# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys
import types
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXTENDED_TESTS = REPOSITORY_ROOT / "tests" / "extended_tests"
BENCHMARK_SCRIPTS = EXTENDED_TESTS / "benchmark" / "scripts"
sys.path.insert(0, str(EXTENDED_TESTS))
sys.path.insert(0, str(BENCHMARK_SCRIPTS))

# benchmark_base.py's status logic does not need the external API client or
# PrettyTable implementation. Keep this unit test runnable with Python alone,
# while restoring sys.modules so these stubs cannot affect other tests.
missing = object()
previous_prettytable = sys.modules.get("prettytable", missing)
previous_extended_base = sys.modules.get("utils.extended_test_base", missing)
previous_utils = sys.modules.get("utils")
previous_extended_base_attribute = (
    getattr(previous_utils, "extended_test_base", missing)
    if previous_utils is not None
    else missing
)
prettytable_stub = types.ModuleType("prettytable")
prettytable_stub.PrettyTable = object
extended_test_base_stub = types.ModuleType("utils.extended_test_base")
extended_test_base_stub.ExtendedTestBase = type("ExtendedTestBase", (), {})
extended_test_base_stub.gha_append_step_summary = lambda _summary: None
sys.modules["prettytable"] = prettytable_stub
sys.modules["utils.extended_test_base"] = extended_test_base_stub
try:
    from benchmark_base import BenchmarkBase  # noqa: E402
finally:
    if previous_prettytable is missing:
        sys.modules.pop("prettytable", None)
    else:
        sys.modules["prettytable"] = previous_prettytable
    if previous_extended_base is missing:
        sys.modules.pop("utils.extended_test_base", None)
    else:
        sys.modules["utils.extended_test_base"] = previous_extended_base
    utils_package = sys.modules.get("utils")
    if utils_package is not None:
        if previous_extended_base_attribute is missing:
            if hasattr(utils_package, "extended_test_base"):
                delattr(utils_package, "extended_test_base")
        else:
            utils_package.extended_test_base = previous_extended_base_attribute


class FakeTable:
    def __init__(self, statuses):
        self.title = "benchmark comparison"
        self.field_names = ["FinalResult"]
        self._rows = [[status] for status in statuses]


class PositiveMetricWithoutBaseline(BenchmarkBase):
    def __init__(self):
        self.benchmark_name = "benchmark"
        self.display_name = "benchmark"
        self.script_dir = Path(".")

    def run_benchmarks(self):
        pass

    def parse_results(self):
        return ([{"status": "PASS"}], None)

    def calculate_statistics(self, _results):
        return {"passed": 1, "failed": 0, "total": 1, "overall_status": "PASS"}

    def upload_results(self, **_kwargs):
        return False

    def compare_with_lkg(self, _tables):
        return None


class BenchmarkStatusTest(unittest.TestCase):
    def setUp(self):
        self.benchmark = BenchmarkBase.__new__(BenchmarkBase)

    def test_missing_comparison_is_unknown(self):
        self.assertEqual(self.benchmark.determine_final_status(None), "UNKNOWN")
        self.assertEqual(self.benchmark.determine_final_status([]), "UNKNOWN")

    def test_empty_comparison_table_is_unknown(self):
        self.assertEqual(
            self.benchmark.determine_final_status(FakeTable([])), "UNKNOWN"
        )

    def test_pass_requires_a_comparison_result(self):
        self.assertEqual(
            self.benchmark.determine_final_status(FakeTable(["PASS"])), "PASS"
        )

    def test_failure_and_error_block_pass(self):
        self.assertEqual(
            self.benchmark.determine_final_status(FakeTable(["ERROR"])), "FAIL"
        )
        self.assertEqual(
            self.benchmark.determine_final_status(FakeTable(["FAIL"])), "FAIL"
        )

    def test_unknown_row_blocks_pass(self):
        self.assertEqual(
            self.benchmark.determine_final_status(FakeTable(["PASS", "UNKNOWN"])),
            "UNKNOWN",
        )

    def test_positive_raw_score_without_baseline_is_not_pass(self):
        self.assertEqual(PositiveMetricWithoutBaseline().run(), 1)


if __name__ == "__main__":
    unittest.main()
