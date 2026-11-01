# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Greedy decoding for Hugging Face causal LMs with a static KV cache, an optional
torch.compile'd step, and a CUDA/HIP graph of the decode step.

    dec = GraphedDecoder(model, max_len=2048, compile=True)
    tokens = dec.generate(input_ids, max_new_tokens=128)     # greedy, batch 1

Measured on gfx1102 / ROCm 10.1.0 (TinyLlama-1.1B): 31.7 ms/token stock eager generate ->
~11 ms/token with this + rocm_perf_profile + fast_linear (see examples/gfx1102/e2e/RESULTS.md).
Limits: batch size 1, greedy only, no stop tokens beyond `eos_token_id`, models that accept
`cache_position` + `StaticCache`. Compilation costs ~1 min on first use per process.
"""
import torch


class GraphedDecoder:
    def __init__(self, model, max_len=2048, compile=True, use_graph=True):
        from transformers import StaticCache
        self.m, self.max_len, self.use_graph = model.eval(), max_len, use_graph
        p = next(model.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self.cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_len, device=self.dev, dtype=self.dtype)
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.fwd = torch.compile(model, dynamic=False) if compile else model
        self.graph = None

    def _step(self):
        logits = self.fwd(self.tok, past_key_values=self.cache, cache_position=self.pos, use_cache=True).logits
        self.tok.copy_(logits[:, -1:].argmax(-1))
        self.pos.add_(1)

    @torch.no_grad()
    def _prefill(self, ids):
        n = ids.shape[1]
        if n + 1 > self.max_len:
            raise ValueError(f"prompt length {n} does not fit max_len {self.max_len}")
        logits = self.m(ids, past_key_values=self.cache, cache_position=torch.arange(n, device=self.dev), use_cache=True).logits
        self.tok.copy_(logits[:, -1:].argmax(-1))
        self.pos.fill_(n)

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        keep_tok, keep_pos = self.tok.clone(), self.pos.clone()
        with torch.cuda.stream(s):
            for _ in range(3):  # warm up (and compile) outside the graph
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()
        self.tok.copy_(keep_tok)
        self.pos.copy_(keep_pos)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, eos_token_id=None):
        if input_ids.shape[0] != 1:
            raise ValueError("GraphedDecoder supports batch size 1")
        if input_ids.shape[1] + max_new_tokens > self.max_len:
            raise ValueError("prompt + max_new_tokens exceeds max_len")
        input_ids = input_ids.to(self.dev)
        self.cache.reset()  # in place (graph-safe): clears contents and HF's per-layer position counters
        self._prefill(input_ids)
        if self.use_graph and self.graph is None:
            self._capture()
            self.cache.reset()
            self._prefill(input_ids)  # restore exact state: warmup steps wrote cache slots past the prompt
        out = [self.tok.clone()]
        run = self.graph.replay if (self.use_graph and self.graph is not None) else self._step
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and int(out[-1]) == eos_token_id:
                break
            run()
            out.append(self.tok.clone())
        return torch.cat(out, 1)
