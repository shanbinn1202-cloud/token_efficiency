"""Static-address CUDA graph for single-token decode; selection stays outside."""
import copy

import torch
from transformers.cache_utils import DynamicCache


class GraphKVLayer:
    def __init__(self, source, context):
        self.context = context
        shape = (1, source.keys.shape[1], context.capacity, source.keys.shape[-1])
        self.keys = torch.zeros(shape, device="cuda", dtype=source.keys.dtype)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True
        self.is_sliding = False
        self.is_compileable = True

    def update(self, keys, values, *args, **kwargs):
        self.keys.index_copy_(2, self.context.kv_index, keys)
        self.values.index_copy_(2, self.context.kv_index, values)
        return self.keys, self.values

    def get_seq_length(self):
        return self.context.length


class GraphDecoder:
    def __init__(self, runtime, capacity=8192):
        self.runtime = runtime
        self.capacity = capacity
        self.length = 0
        self.kv_index = torch.zeros(1, dtype=torch.long, device="cuda")
        self.input_ids = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        self.position_ids = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        self.ring_dest = torch.zeros(1, dtype=torch.long, device="cuda")
        self.mask = torch.zeros((1, capacity), dtype=torch.bool, device="cuda")
        self.slots = torch.arange(capacity, device="cuda")[None, :]
        self.rings = {index: torch.empty_like(ring) for index, ring in runtime.rings.items()}
        self.cache = DynamicCache(config=runtime.model.config)
        for index, source in enumerate(runtime.cache.layers):
            if index in runtime.layers:
                self.cache.layers[index] = GraphKVLayer(source, self)
            else:
                target = self.cache.layers[index]
                target.update_conv_state(source.conv_states[0], conv_kernel_size=4)
                target.update_recurrent_state(source.recurrent_states[0])
        self.graph = None
        self.replay_enabled = True

    def load(self, cache, rings):
        self.length = cache.layers[self.runtime.layers[0]].keys.shape[-2]
        for index, source in enumerate(cache.layers):
            target = self.cache.layers[index]
            if index in self.runtime.layers:
                target.keys[:, :, :self.length].copy_(source.keys)
                target.values[:, :, :self.length].copy_(source.values)
            else:
                target.conv_states[0].copy_(source.conv_states[0])
                target.recurrent_states[0].copy_(source.recurrent_states[0])
        for index, ring in rings.items():
            self.rings[index].copy_(ring)

    @torch.inference_mode()
    def activate(self):
        runtime = self.runtime
        source_cache, source_rings = runtime.cache, runtime.rings
        self.load(source_cache, source_rings)
        runtime.graph_context = self
        runtime.rings = self.rings
        if self.graph is None:
            self.kv_index.fill_(self.length)
            self.position_ids.fill_(self.length)
            self.mask.copy_(self.slots <= self.kv_index)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.invoke()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = self.invoke()
            # Warmup/capture must never consume actual history.
            self.load(source_cache, source_rings)
        runtime.cache = self.cache

    def invoke(self):
        return self.runtime.model(
            input_ids=self.input_ids, position_ids=self.position_ids,
            attention_mask={"full_attention": None, "linear_attention": None},
            past_key_values=self.cache, use_cache=True, logits_to_keep=1,
        ).logits[0, -1]

    @torch.inference_mode()
    def step(self, token, position):
        self.input_ids.fill_(token)
        self.position_ids.fill_(position)
        self.kv_index.fill_(self.length)
        self.ring_dest.copy_(self.runtime.ring_dest)
        self.mask.copy_(self.slots <= self.kv_index)
        if self.replay_enabled:
            self.graph.replay()
            logits = self.logits
        else:
            logits = self.invoke()
        self.length += 1
        self.runtime.peak_physical = max(self.runtime.peak_physical, self.length)
        return logits

    def deactivate(self):
        result = DynamicCache(config=self.runtime.model.config)
        for index, source in enumerate(self.cache.layers):
            if index in self.runtime.layers:
                result.layers[index].update(source.keys[:, :, :self.length].clone(), source.values[:, :, :self.length].clone())
            else:
                target = result.layers[index]
                target.update_conv_state(source.conv_states[0], conv_kernel_size=4)
                target.update_recurrent_state(source.recurrent_states[0])
        self.runtime.cache = result
        self.runtime.graph_context = None
