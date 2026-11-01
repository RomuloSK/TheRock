# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import sys, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf_tools"))
import rocm_perf_profile as p


class ProfileTest(unittest.TestCase):
    def test_gfx11_enables_fused_attention(self):
        env = {"ROCM_PERF_ARCH": "gfx1102"}
        self.assertEqual(p.apply(env), {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"})
        self.assertEqual(env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"], "1")

    def test_user_setting_is_never_overridden(self):
        env = {"ROCM_PERF_ARCH": "gfx1102", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"}
        self.assertEqual(p.apply(env), {})
        self.assertEqual(env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"], "0")

    def test_other_arches_untouched(self):
        for arch in ("gfx942", "gfx90a", "gfx1201", "gfx1100", "gfx1103"):  # 1100 is already on by default
            self.assertEqual(p.apply({"ROCM_PERF_ARCH": arch}), {})

    def test_unvalidated_gated_arches_are_opt_in(self):
        for arch in ("gfx1101", "gfx1151", "gfx1200"):
            self.assertEqual(p.apply({"ROCM_PERF_ARCH": arch}), {})
            self.assertTrue(p.apply({"ROCM_PERF_ARCH": arch, "ROCM_PERF_INCLUDE_UNVALIDATED": "1"}))

    def test_mixed_arches_and_kill_switch(self):
        self.assertTrue(p.apply({"ROCM_PERF_ARCH": "gfx942;gfx1102"}))
        self.assertEqual(p.apply({"ROCM_PERF_ARCH": "gfx1102", "ROCM_PERF_PROFILE": "0"}), {})

    def test_unknown_hardware_sets_nothing(self):
        with mock.patch.object(p, "detect_arches", return_value=[]):
            self.assertEqual(p.apply({}), {})


if __name__ == "__main__":
    unittest.main()
