# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tune and load PyTorch TunableOp GEMM solutions for a model's *decode* shapes.

Why: on gfx1102 / ROCm 10.1.0 the default GEMM choice for M=2..16 rows (batched decode) reaches only
~135 GB/s on the FFN down-projection versus ~253 GB/s after tuning (examples/gfx1102/e2e/RESULTS.md).

    from fast_tune import tune_decode, load_tuning
    tune_decode(model, batch_sizes=(1, 2, 4, 8), filename="tune_tinyllama.csv")   # once, ~1-3 min
    load_tuning("tune_tinyllama.csv")                                               # every run, before building a GraphedDecoder

The tuning file is specific to GPU, PyTorch, ROCm/hipBLASLt versions and to the model's matrix shapes; PyTorch
ignores entries whose validators do not match. Never tune inside CUDA-graph capture: call these before GraphedDecoder runs.
"""
import torch


def _api():
    return torch.cuda.tunable


def save_tuning(filename):
    """Write validators + results in PyTorch's TunableOp CSV format (the public API only writes at process exit)."""
    t = _api()
    with open(filename, "w") as f:
        for key, value in t.get_validators():
            f.write(f"Validator,{key},{value}\n")
        for op_sig, param_sig, result, elapsed in t.get_results():
            f.write(f"{op_sig},{param_sig},{result},{elapsed}\n")


def load_tuning(filename):
    t = _api()
    t.enable(True)
    t.tuning_enable(False)
    t.set_filename(filename, insert_device_ordinal=False)
    t.read_file(filename)


@torch.no_grad()
def tune_decode(model, batch_sizes=(1, 2, 4, 8), filename="tunableop_decode.csv", prompt_len=16, steps=4, max_len=128):
    """Run a few eager decode steps per batch size with tuning enabled and save the results."""
    from fast_decode import GraphedDecoder
    t = _api()
    t.enable(True)
    t.tuning_enable(True)
    t.set_filename(filename, insert_device_ordinal=False)
    dev = next(model.parameters()).device
    dec = GraphedDecoder(model, max_len=max_len, max_batch=max(batch_sizes), compile=False, use_graph=False)
    vocab = model.config.vocab_size
    for b in batch_sizes:
        ids = torch.randint(3, vocab, (b, prompt_len), device=dev)
        dec.generate(ids, steps)
    t.tuning_enable(False)
    save_tuning(filename)
    return filename
