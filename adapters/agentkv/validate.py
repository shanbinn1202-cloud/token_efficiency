"""Check ring equivalence, attention causality, and per-head KV gathering."""
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .runtime import Runtime
from .upstream import factory, LayerEvictionView
from minisgl.eviction.query_ring import QueryRingBufferPool


@torch.inference_mode()
def main():
    torch.manual_seed(7)
    runtime = Runtime.__new__(Runtime)
    runtime.sdpa_backend = "auto"
    runtime.num_stages = 4
    runtime.ring_capacity = 8
    runtime.plan = factory("agentkv-8192")
    runtime.model = SimpleNamespace(model=SimpleNamespace(layers=[None, None]))
    runtime.rings = {}
    runtime.ring_positions = [0] * 4
    runtime.ring_sizes = [0] * 4
    runtime.peak_physical = 0
    original = QueryRingBufferPool(max_table_slots=1, num_layers=1, capacity=8,
        num_q_heads=4, head_dim=32, dtype=torch.float32, device=torch.device("cuda"), num_stages=4)
    position = 0
    for stages in [[0] * 13 + [1] * 9 + [2] * 3, [3] * 21 + [1] * 2, [2], [0, 3, 2, 1] * 6]:
        count = len(stages)
        q = torch.randn(1, 4, count, 32, device="cuda")
        k = torch.randn(1, 2, count + 19, 32, device="cuda")
        v = torch.randn_like(k)
        runtime.prepare_rings(stages)
        actual, _ = runtime.attention(SimpleNamespace(layer_idx=0, num_key_value_groups=2), q, k, v, None, scaling=32**-0.5)
        mask = torch.arange(count + 19, device="cuda")[None] <= torch.arange(count, device="cuda")[:, None] + 19
        expected = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=mask)
        torch.testing.assert_close(actual.transpose(1, 2), expected)
        metadata = original.prepare_write_metadata_from_reqs(
            [SimpleNamespace(extend_len=count, table_idx=0)],
            torch.arange(position, position + count, device="cuda"), torch.tensor(stages, device="cuda"))
        original.write_layer(layer_id=0, q=q[0].transpose(0, 1), metadata=metadata)
        for stage, size in enumerate(runtime.ring_sizes):
            torch.testing.assert_close(runtime.rings[0][stage * 8:stage * 8 + size], original.q[0, 0, stage, :size], rtol=0, atol=0)
        position += count
    runtime.plan = factory("agentkv-1024")
    keys = torch.randn(1, 2, 1400, 32, device="cuda")
    values = torch.randn_like(keys)
    linear_state = torch.randn(1, 2, 32, 32, device="cuda")
    runtime.layers = [0]
    runtime.cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values), SimpleNamespace(recurrent_states=[linear_state])])
    runtime.compactions = 0
    valid = [stage * 8 + i for stage, size in enumerate(runtime.ring_sizes) for i in range(size)]
    keep = runtime.plan.select_layer(LayerEvictionView(layer_id=0, physical_cached_len=1400, num_kv_heads=2,
        device=keys.device, k=keys[0].transpose(0, 1), v=values[0].transpose(0, 1), q_observation=runtime.rings[0][valid]))
    runtime.compact()
    for head in range(2):
        torch.testing.assert_close(runtime.cache.layers[0].keys[0, head], keys[0, head, keep[head]], rtol=0, atol=0)
        torch.testing.assert_close(runtime.cache.layers[0].values[0, head], values[0, head, keep[head]], rtol=0, atol=0)
    assert runtime.cache.layers[1].recurrent_states[0] is linear_state
    assert runtime.cache.layers[0].keys.shape[-2] == 1024
    print(json.dumps({"ring_matches_upstream_exactly": True, "attention_matches_sdpa": True,
        "kv_selection_matches_upstream_exactly": True, "linear_state_untouched": True}))


if __name__ == "__main__":
    main()
