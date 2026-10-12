"""Bound the workspace of the upstream full-history attention scorer.

Only the query dimension is tiled. Every softmax still sees all active and
evicted keys; no queries, heads, layers, or candidate positions are omitted.
"""
import math

import torch


SCORE_WORKSPACE_BYTES = 4 << 30


def score_evicted_positions_tiled(*, query_rows, evicted_cache,
                                 active_cache=None, pooling="max",
                                 workspace_bytes=SCORE_WORKSPACE_BYTES):
    if pooling not in {"max", "mean"}:
        raise ValueError(f"Unsupported pooling: {pooling}")
    if not len(evicted_cache):
        return {}
    if query_rows.ndim != 4 or query_rows.shape[0] != len(evicted_cache.kv):
        raise ValueError("Queries must contain every cache layer")
    if active_cache is not None and len(active_cache.kv) != len(evicted_cache.kv):
        raise ValueError("Active and evicted layer counts differ")
    if query_rows.device.type != "cpu" or query_rows.dtype != torch.float32:
        raise ValueError("The default upstream scorer requires host FP32 queries")

    layer_scores = []
    for layer_index, (key, _) in enumerate(evicted_cache.kv):
        query = query_rows[layer_index]
        heads, query_count, head_dim = query.shape
        if query_count == 0:
            raise ValueError("At least one query is required")
        keys = key.detach().to("cpu", dtype=torch.float32)[0]
        active_len = 0
        if active_cache is not None and len(active_cache):
            active_keys = active_cache.kv[layer_index][0].detach().to(
                "cpu", dtype=torch.float32)[0]
            active_len = active_keys.shape[1]
            keys = torch.cat((active_keys, keys), dim=1)
            del active_keys
        key_heads = keys.shape[0]
        if heads % key_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if heads != key_heads:
            keys = keys.repeat_interleave(heads // key_heads, dim=0)

        # Like upstream, normalize each query over the complete key set.
        # Dividing the key dimension would change the scoring algorithm.
        max_rows = max(1, workspace_bytes // (heads * keys.shape[1] * 4))
        tile_count = (query_count + max_rows - 1) // max_rows
        pooled_heads = None
        # Balance tiles to avoid a small tail with a slower CPU GEMM kernel.
        for tile in range(tile_count):
            start = query_count * tile // tile_count
            stop = query_count * (tile + 1) // tile_count
            logits = torch.matmul(query[:, start:stop], keys.transpose(-2, -1))
            logits = logits / math.sqrt(float(head_dim))
            attention = torch.softmax(logits, dim=-1)
            del logits
            evicted_attention = attention[:, :, active_len:]
            if pooling == "max":
                part = evicted_attention.amax(dim=1)
                pooled_heads = part if pooled_heads is None else torch.maximum(
                    pooled_heads, part)
            else:
                part = evicted_attention.sum(dim=1)
                pooled_heads = part if pooled_heads is None else pooled_heads + part
            del part, attention, evicted_attention
        if pooling == "mean":
            pooled_heads = pooled_heads / query_count
        layer_scores.append(pooled_heads.mean(dim=0))
        del keys, pooled_heads

    importance = torch.stack(layer_scores, dim=0).mean(dim=0)
    return dict(zip(evicted_cache.positions, importance.tolist()))
