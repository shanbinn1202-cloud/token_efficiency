"""Keep the complete fixed prefix and most recent surviving KV rows.

This common-attention variant compacts at the end of prefill and every 64
physical decode tokens. B includes the prefix; it is a compaction target.
"""

import torch


def should_compact(*, physical, budget, new_tokens, decoding, window=64):
    return physical > budget and (
        not decoding or (new_tokens == 1 and physical % window == 0)
    )


class Policy:
    def __init__(self, budget):
        self.budget = budget

    def keep(self, *, cache, physical, prefix, **unused):
        assert 0 <= prefix < self.budget < physical
        indices = torch.cat((
            torch.arange(prefix, device=cache.device),
            torch.arange(physical - (self.budget - prefix), physical, device=cache.device),
        ))
        return indices.expand(cache.shape[1], -1)
