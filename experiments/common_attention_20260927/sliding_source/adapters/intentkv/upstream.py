"""Run the original selector, retaining its query-head decisions verbatim."""
import importlib.util
from pathlib import Path
import torch
from torch.overrides import TorchFunctionMode

SOURCE = Path(__file__).resolve().parents[2] / "baselines/IntentKV/intentkv/intentkv_cluster.py"
_spec = importlib.util.spec_from_file_location("_intentkv_original_cluster", SOURCE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
IntentKVCluster = _module.IntentKVCluster


class _Queries:
    """A read-only view of the only Q rows consumed by upstream.

    Earlier prefill queries have already been discarded by the engine. Upstream
    reads only shape, dtype, and the last window; no tensor arithmetic is changed.
    """
    def __init__(self, tail, length):
        self.tail = tail
        self.shape = (*tail.shape[:2], length, tail.shape[-1])
        self.dtype = tail.dtype

    def __getitem__(self, index):
        assert index == (Ellipsis, slice(-self.tail.shape[-2], None), slice(None))
        return self.tail


class _GatherIndices(TorchFunctionMode):
    def __init__(self, kv_heads=None):
        self.indices = None
        self.kv_heads = kv_heads

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (self.kv_heads is not None and func is torch.Tensor.topk
                and args[0].ndim == 3):
            # The final block scores are [1, query_heads, blocks]. Preserve
            # upstream per-head JSD/query-window scoring, then average only
            # within each native GQA group before selecting shared blocks.
            scores = args[0]
            groups = scores.shape[1] // self.kv_heads
            grouped = scores.reshape(1, self.kv_heads, groups, -1).mean(dim=2)
            result = func(grouped, *args[1:], **kwargs)
            return torch.return_types.topk((
                result.values.repeat_interleave(groups, dim=1),
                result.indices.repeat_interleave(groups, dim=1),
            ))
        if func is torch.Tensor.gather and self.indices is None:
            assert kwargs["dim"] == 2
            self.indices = kwargs["index"][0, :, :, 0].clone()
        return func(*args, **kwargs)


@torch.no_grad()
def select_positions(keys, tail_queries, capacity, window=64, block_size=16, *, gqa=False):
    """keys: [tokens, KV heads, dim]; tail_queries: [window, Q heads, dim]."""
    length, kv_heads, dim = keys.shape
    heads = tail_queries.shape[1]
    if length < capacity:
        return torch.arange(length, device=keys.device).expand(kv_heads if gqa else heads, -1)
    assert heads % kv_heads == 0 and tail_queries.shape[0] == window
    # Exactly the upstream repeat_kv layout, before per-query-head selection.
    repeated = keys.transpose(0, 1)[None, :, None].expand(
        1, kv_heads, heads // kv_heads, length, dim).reshape(1, heads, length, dim)
    queries = _Queries(tail_queries.transpose(0, 1).unsqueeze(0), length)
    cluster = IntentKVCluster(window, capacity, block_size)
    with _GatherIndices(kv_heads if gqa else None) as captured:
        # V never enters the scores. Reusing K as its payload avoids another
        # expanded allocation; the real V is gathered later using these indices.
        cluster.update_kv(repeated, queries, repeated, None, heads // kv_heads)
    remainder = (length - window) % block_size
    tail = torch.arange(length - window - remainder, length, device=keys.device)
    selected = torch.cat((captured.indices, tail.expand(heads, -1)), dim=1)
    return selected[::heads // kv_heads] if gqa else selected


def storage_plan(selected, length, kv_heads, window=64, block_size=16):
    """Lossless union storage; attention still uses each original Q-head list."""
    import numpy as np
    selected = selected.cpu().numpy()
    heads, kept = selected.shape
    tail = window + (length - window) % block_size
    past = (kept - tail) // block_size
    blocks = selected[:, :past * block_size:block_size] // block_size
    groups = heads // kv_heads
    unions = [np.unique(blocks[h*groups:(h+1)*groups]) for h in range(kv_heads)]
    return dict(unions=unions, blocks=blocks, tail=tail, kept=kept,
                union_blocks=max(map(len, unions)), length=length)


def materialize_plan(plan, union_blocks, max_pages, block_size=16):
    import numpy as np
    unions, blocks = plan['unions'], plan['blocks']
    kv_heads, heads = len(unions), len(blocks)
    groups = heads // kv_heads
    keep = np.zeros((kv_heads, union_blocks * block_size + plan['tail']), dtype=np.int64)
    maps = np.empty((heads, max_pages), dtype=np.int32)
    past = blocks.shape[1]
    for kh, union in enumerate(unions):
        keep[kh, :len(union)*block_size] = (union[:, None]*block_size + np.arange(block_size)).ravel()
        keep[kh, union_blocks*block_size:] = np.arange(plan['length']-plan['tail'], plan['length'])
        for h in range(kh*groups, (kh+1)*groups):
            maps[h, :past] = np.searchsorted(union, blocks[h])
            maps[h, past:] = union_blocks + np.arange(max_pages-past)
    return torch.from_numpy(keep), torch.from_numpy(maps)
