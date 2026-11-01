# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf_tools"))
try:
    import torch, torch.nn as nn
    from fast_linear import convert, FastLinear, fuse_projections
except ImportError:  # torch not installed in generic CI
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class FastLinearTest(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Sequential(nn.Linear(32, 8, bias=False)))

    def test_matches_reference_for_decode_and_batch_shapes(self):
        ref = self._model(); m = convert(self._model())
        for shape in ((1, 16), (5, 16), (2, 3, 16)):
            x = torch.randn(*shape)
            torch.testing.assert_close(m(x), ref(x), rtol=1e-5, atol=1e-5)

    def test_replaces_all_linears_and_preserves_bias_flag(self):
        m = convert(self._model())
        self.assertIsInstance(m[0], FastLinear); self.assertIsInstance(m[2][0], FastLinear)
        self.assertIsNotNone(m[0].bias); self.assertIsNone(m[2][0].bias)

    def test_state_dict_roundtrip_is_standard_layout(self):
        ref = self._model(); m = convert(self._model())
        sd = m.state_dict()
        self.assertEqual(tuple(sd["0.weight"].shape), (32, 16))
        self.assertEqual(set(sd), set(ref.state_dict()))
        ref2 = self._model(); ref2.load_state_dict(sd)           # standard model accepts it
        m2 = convert(self._model()); m2.load_state_dict(ref.state_dict())  # and vice versa
        x = torch.randn(3, 16); torch.testing.assert_close(m2(x), ref(x), rtol=1e-5, atol=1e-5)

    def test_gradients_match(self):
        ref = self._model(); m = convert(self._model()); x = torch.randn(4, 16)
        ref(x).sum().backward(); m(x).sum().backward()
        torch.testing.assert_close(m[0].weight_t.grad.t(), ref[0].weight.grad, rtol=1e-5, atol=1e-5)

    def test_warns_in_training_mode(self):
        with self.assertWarns(UserWarning):
            convert(self._model())
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error"); convert(self._model().eval())

    def test_load_rejects_wrong_shape(self):
        m = convert(self._model()); sd = self._model().state_dict(); sd["0.weight"] = torch.zeros(16, 32)
        with self.assertRaises(RuntimeError): m.load_state_dict(sd)


@unittest.skipIf(torch is None, "torch not installed")
class FuseProjectionsTest(unittest.TestCase):
    def _block(self, bias):
        class Attn(nn.Module):
            def __init__(s):
                super().__init__()
                s.q_proj, s.k_proj, s.v_proj = nn.Linear(16, 16, bias=bias), nn.Linear(16, 4, bias=bias), nn.Linear(16, 4, bias=bias)
            def forward(s, x):  # HF-style call order q, k, v
                return s.q_proj(x), s.k_proj(x), s.v_proj(x)
        class Mlp(nn.Module):
            def __init__(s):
                super().__init__()
                s.gate_proj, s.up_proj, s.down_proj = nn.Linear(16, 32, bias=False), nn.Linear(16, 32, bias=False), nn.Linear(32, 16, bias=False)
            def forward(s, x):
                return s.down_proj(torch.nn.functional.silu(s.gate_proj(x)) * s.up_proj(x))
        torch.manual_seed(1)
        return nn.ModuleDict({"attn": Attn(), "mlp": Mlp()})

    def test_fused_outputs_match_unfused_with_and_without_bias(self):
        for bias in (False, True):
            ref = self._block(bias); m = fuse_projections(convert(self._block(bias).eval()))
            for shape in ((1, 16), (3, 16), (2, 5, 16)):
                x = torch.randn(*shape)
                for r, o in zip(ref["attn"](x), m["attn"](x)):
                    torch.testing.assert_close(o, r, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(m["mlp"](x), ref["mlp"](x), rtol=1e-5, atol=1e-5)

    def test_runs_one_gemm_per_group_and_resets_state(self):
        m = fuse_projections(convert(self._block(False).eval())); x = torch.randn(1, 16)
        m["attn"](x); self.assertIsNone(m["attn"].fused_q_k_v._x)
        m["attn"](x)  # repeat call must not use a stale stash

    def test_out_of_order_calls_stay_correct(self):
        ref = self._block(True); m = fuse_projections(convert(self._block(True).eval())); x = torch.randn(2, 16)
        torch.testing.assert_close(m["attn"].v_proj(x), ref["attn"].v_proj(x), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(m["attn"].k_proj(x), ref["attn"].k_proj(x), rtol=1e-5, atol=1e-5)

    def test_non_matching_modules_untouched(self):
        m = fuse_projections(convert(nn.Sequential(nn.Linear(4, 4)).eval()))
        self.assertIsInstance(m[0], FastLinear)


if __name__ == "__main__":
    unittest.main()
