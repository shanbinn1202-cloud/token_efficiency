"""Random Attention's published selector; engine-specific shape metadata only."""

from .config import TriAttentionRuntimeConfig
from .selector_random import build_random_pp_selector


def should_compact(*, physical, budget, new_tokens, decoding, window=64):
    # EvictLayer.post_attention_update: decode only, physical length modulo r.
    return decoding and new_tokens == 1 and physical > budget and physical % window == 0


class Policy:
    def __init__(self, budget):
        self.budget = budget
        self._triattention_active_req_id = None
        self.select, _, mode = build_random_pp_selector(
            TriAttentionRuntimeConfig(), self
        )
        assert mode == "random_pp"

    def keep(self, *, cache, physical, prefix, session, logical, layer):
        self._triattention_active_req_id = session
        # No KV content is read by the selector. A view supplies head/device metadata.
        shape_only = cache.new_empty(0).view(1, cache.shape[1], 0, 1)
        result = self.select(
            keys_dense=shape_only,
            total_tokens=physical,
            prefill_len=prefix,
            protect_prefill=True,
            layer_idx=layer,
            round_start=logical,
            budget_total=self.budget,
        )
        assert result is not None and result["mode"] == "per_head"
        return result["indices"]
