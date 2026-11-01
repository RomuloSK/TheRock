# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Greedy decoding for Hugging Face causal LMs with a static KV cache, an optional
torch.compile'd step, and a CUDA/HIP graph of the decode step. Supports batches with
left padding (pass `attention_mask`, as with `model.generate`).

    dec = GraphedDecoder(model, max_len=2048, max_batch=4, compile=True)
    tokens = dec.generate(input_ids, max_new_tokens=128, attention_mask=mask)   # [B, new]

Measured on gfx1102 / ROCm 10.1.0 (TinyLlama-1.1B): 31.7 ms/token stock eager generate ->
~11 ms/token with this + rocm_perf_profile + fast_linear (examples/gfx1102/e2e/RESULTS.md).
Limits: greedy only; models that accept `cache_position`, `position_ids`, a 2-D `attention_mask`
and `StaticCache` (Llama/Qwen2 style); a batch must be <= max_batch and uses one graph per distinct
batch size. Compilation costs ~1 min on first use per process. Finished rows keep emitting the
tokens the model produces after EOS (use `eos_token_id` to truncate/pad per row on return).
"""
import torch


class GraphedDecoder:
    def __init__(self, model, max_len=2048, max_batch=1, compile=True, use_graph=True):
        from transformers import StaticCache
        self.m, self.max_len, self.max_batch, self.use_graph = model.eval(), max_len, max_batch, use_graph
        p = next(model.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self._StaticCache, self._config = StaticCache, model.config
        self.fwd = torch.compile(model, dynamic=False) if compile else model
        self.state = {}  # batch size -> dict(cache, buffers, graph)

    # ---- per-batch-size state (static buffers, cache, graph) ----
    def _get(self, b):
        st = self.state.get(b)
        if st is None:
            st = dict(
                cache=self._StaticCache(config=self._config, max_batch_size=b, max_cache_len=self.max_len, device=self.dev, dtype=self.dtype),
                tok=torch.zeros(b, 1, dtype=torch.long, device=self.dev),
                pos=torch.zeros(1, dtype=torch.long, device=self.dev),          # cache slot to write (same for all rows)
                pid=torch.zeros(b, 1, dtype=torch.long, device=self.dev),       # per-row RoPE position (ignores left padding)
                mask=torch.zeros(b, self.max_len, dtype=torch.long, device=self.dev),
                graph=None,
            )
            self.state[b] = st
        return st

    def _step(self, st):
        # mark the slot being written as visible, run one token, advance
        st["mask"].index_fill_(1, st["pos"], 1)
        logits = self.fwd(st["tok"], past_key_values=st["cache"], cache_position=st["pos"], position_ids=st["pid"],
                          attention_mask=st["mask"], use_cache=True).logits
        st["tok"].copy_(logits[:, -1:].argmax(-1))
        st["pos"].add_(1)
        st["pid"].add_(1)

    @torch.no_grad()
    def _prefill(self, st, ids, am):
        b, n = ids.shape
        st["cache"].reset()  # in place (graph-safe): clears contents and HF's per-layer position counters
        st["mask"].zero_()
        st["mask"][:, :n] = am
        pid = (am.cumsum(-1) - 1).clamp(min=0)
        logits = self.m(ids, past_key_values=st["cache"], cache_position=torch.arange(n, device=self.dev), position_ids=pid,
                        attention_mask=st["mask"][:, :n], use_cache=True).logits
        st["tok"].copy_(logits[:, -1:].argmax(-1))
        st["pos"].fill_(n)
        st["pid"].copy_(am.sum(-1, keepdim=True))  # next RoPE position for each row

    def _capture(self, st, ids, am):
        # Warm up AND capture on the same stream: hipBLASLt handles (used by TunableOp-selected GEMMs) are created
        # per stream, and creating one during capture fails ("operation not permitted when stream is capturing")
        # and then hangs the process (examples/gfx1102/e2e/lt_capture.py).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warm up (and compile) outside the graph
                self._step(st)
        torch.cuda.synchronize()
        st["graph"] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(st["graph"], stream=s):
            self._step(st)
        torch.cuda.synchronize()
        self._prefill(st, ids, am)  # restore exact state: warmup steps advanced buffers and wrote cache slots

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, attention_mask=None, eos_token_id=None):
        b, n = input_ids.shape
        if b > self.max_batch:
            raise ValueError(f"batch {b} exceeds max_batch {self.max_batch}")
        if n + max_new_tokens > self.max_len:
            raise ValueError("prompt + max_new_tokens exceeds max_len")
        input_ids = input_ids.to(self.dev)
        am = torch.ones_like(input_ids) if attention_mask is None else attention_mask.to(self.dev).to(torch.long)
        st = self._get(b)
        self._prefill(st, input_ids, am)
        if self.use_graph and st["graph"] is None:
            self._capture(st, input_ids, am)
        out = [st["tok"].clone()]
        run = st["graph"].replay if (self.use_graph and st["graph"] is not None) else (lambda: self._step(st))
        done = torch.zeros(b, dtype=torch.bool, device=self.dev)
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None:
                done |= out[-1][:, 0] == eos_token_id
                if bool(done.all()):
                    break
            run()
            out.append(st["tok"].clone())
        return torch.cat(out, 1)
