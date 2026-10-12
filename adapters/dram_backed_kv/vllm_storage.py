"""Tensor-layout boundary for the unmodified upstream DramKVRuntime.

Resident payloads may be views of native vLLM blocks. CPU copies always own
independent backing. A caller moving resident views must snapshot their values
before allocator zeroing, then rebind them to the destination native blocks.
"""
from dataclasses import dataclass

from .backend import load_upstream

upstream = load_upstream()
DramKVRuntime = upstream.DramKVRuntime


@dataclass
class PagePayload:
    layers: dict


class PageAdapter:
    def __init__(self, device):
        self.device = device

    def nbytes(self, payload):
        return sum(t.numel() * t.element_size() for t in payload.layers.values())

    def to_cpu(self, payload, *, pin):
        result = {}
        for name, tensor in payload.layers.items():
            copy = tensor.detach().to("cpu", copy=True)
            if pin:
                copy = copy.pin_memory()
            result[name] = copy
        return PagePayload(result)

    def to_device(self, payload):
        return PagePayload({name: tensor.to(self.device, copy=True)
                            for name, tensor in payload.layers.items()})


def page_id(start):
    if start < 0 or start % 16:
        raise ValueError("DRAM pages must start at a logical 16-token boundary")
    return f"p{start // 16:08d}:{start}-{start + 16}"


def native_page_view(cache, blocks, block_size, start):
    """Token-major K/V view of one 16-token page in FA3 packed storage."""
    if block_size % 16 or start % 16 or cache.ndim != 4:
        raise ValueError("Expected page-aligned FA3 packed KV storage")
    offset = start % block_size
    return cache[blocks[start // block_size], :, offset:offset + 16, :].permute(1, 0, 2)
