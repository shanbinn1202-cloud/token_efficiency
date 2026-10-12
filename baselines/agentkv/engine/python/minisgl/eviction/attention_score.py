from __future__ import annotations

import re

import torch


_LOWFREQ_RE = re.compile(r"^lowfreq(?P<percent>\d+)$")


def normalize_attention_score_channels(mode: str) -> str:
    mode = mode.lower()
    if mode == "full":
        return mode
    match = _LOWFREQ_RE.match(mode)
    if match is None:
        raise ValueError(
            "attention score channel mode must be 'full' or 'lowfreq{percent}', "
            f"got {mode!r}"
        )
    percent = int(match.group("percent"))
    if not 0 < percent <= 100:
        raise ValueError("lowfreq percent must be in (0, 100]")
    return mode


def project_attention_score_channels(
    q: torch.Tensor,
    k: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = normalize_attention_score_channels(mode)
    if mode == "full":
        return q, k

    match = _LOWFREQ_RE.match(mode)
    assert match is not None
    percent = int(match.group("percent"))

    head_dim = k.shape[-1]
    if head_dim % 2 != 0:
        raise RuntimeError("lowfreq scoring requires an even head dimension")
    if q.shape[-1] != head_dim:
        raise RuntimeError("q and k head dimensions must match for attention scoring")

    half_dim = head_dim // 2
    pair_count = max(1, half_dim * percent // 100)
    pair_start = half_dim - pair_count

    # FlashInfer RoPE uses GPT-NeoX layout by default: pair i is (i, i + half_dim).
    q_low = torch.cat(
        [q[..., pair_start:half_dim], q[..., half_dim + pair_start :]],
        dim=-1,
    )
    k_low = torch.cat(
        [k[..., pair_start:half_dim], k[..., half_dim + pair_start :]],
        dim=-1,
    )
    return q_low, k_low
