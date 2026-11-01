# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Drop-in nn.Linear replacement that stores the weight as [in, out].

On gfx1102 (ROCm 10.1.0) F.linear with the stock [out, in] layout dispatches M=1..few
decode GEMMs to a transposed GEMV that reaches 110-200 GB/s, versus 250-275 GB/s for
`x @ W_t` with an [in, out] layout (1.3-2.2x faster decode GEMMs; see
examples/gfx1102/e2e/RESULTS.md). Outputs differ only by fp16 accumulation order.

    from fast_linear import convert
    convert(model)          # in place; replaces every nn.Linear, preserves bias/dtype/device
INFERENCE ONLY: training steps measured ~10% slower on gfx1102 (backward pays for the
layout); convert() warns if the model is in training mode.
Weights are *copied* in transposed form and the originals freed, so memory is unchanged.
state_dict() round-trips in the standard [out, in] layout (load/save are compatible).
"""
import warnings

import torch
import torch.nn as nn


class FastLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight_t = nn.Parameter(linear.weight.detach().t().contiguous(), requires_grad=linear.weight.requires_grad)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=linear.bias.requires_grad)

    def forward(self, x):
        y = x @ self.weight_t
        return y if self.bias is None else y + self.bias

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        w = self.weight_t if keep_vars else self.weight_t.detach()
        destination[prefix + "weight"] = w.t().contiguous() if not keep_vars else w.t()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        w = state_dict.pop(prefix + "weight", None)
        if w is None:
            missing_keys.append(prefix + "weight")
        elif tuple(w.shape) != (self.out_features, self.in_features):
            error_msgs.append(f"size mismatch for {prefix}weight: {tuple(w.shape)}")
        else:
            with torch.no_grad():
                self.weight_t.copy_(w.t())
        b = state_dict.pop(prefix + "bias", None)
        if self.bias is not None:
            if b is None:
                missing_keys.append(prefix + "bias")
            else:
                with torch.no_grad():
                    self.bias.copy_(b)
        elif b is not None:
            unexpected_keys.append(prefix + "bias")

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, layout=[in,out]"


def convert(module: nn.Module, _top: bool = True) -> nn.Module:
    if _top and module.training:
        warnings.warn("fast_linear.convert: model is in training mode; this layout is for inference "
                      "(measured ~10% slower training steps). Call model.eval() first.", stacklevel=2)
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            setattr(module, name, FastLinear(child))
        else:
            convert(child, _top=False)
    return module
