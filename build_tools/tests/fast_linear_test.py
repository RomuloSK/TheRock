# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf_tools"))
try:
    import torch, torch.nn as nn
    from fast_linear import convert, FastLinear
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


if __name__ == "__main__":
    unittest.main()
