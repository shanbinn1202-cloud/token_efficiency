# SPDX-License-Identifier: Apache-2.0
"""Bridge DRAM-KV Selector V2 rankings into vLLM selective-load requests.

Xuanxue's implementation publishes one compact hidden-state vector per
16-token Page.  The Qwen3.6 vLLM configuration used in this project restores
full-attention KV in 784-token aligned pages.  Because both representations
use last-token pooling, the vector at the final 16-token Page of each
784-token interval is the compatible coarse-page representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from dram_kv.page_vectors import (
    PageVectorIndex,
    PageVectorProvenance,
    compact_cosine,
)
from dram_kv.schema import PageRecord


_POLICIES = frozenset({"observation_query", "goal_query"})


@dataclass(frozen=True, slots=True)
class VllmQueryPolicyPayload:
    policy: str
    fraction: float
    ranked_logical_pages: tuple[int, ...]
    query_hash: str

    def kv_transfer_params(self) -> dict[str, object]:
        return {
            "selective_page_fraction": self.fraction,
            "selective_page_policy": self.policy,
            "selective_page_ranked_pages": list(self.ranked_logical_pages),
            "selective_page_query_hash": self.query_hash,
        }


def build_observation_query_payload(
    *,
    index: PageVectorIndex,
    provenance: PageVectorProvenance,
    session_id: str,
    turn_id: int,
    pages: Sequence[PageRecord],
    fraction: float,
    vllm_page_tokens: int,
    recent_turns: int = 2,
    recent_quota: int = 1,
) -> VllmQueryPolicyPayload:
    return _build_payload(
        policy="observation_query",
        index=index,
        provenance=provenance,
        session_id=session_id,
        turn_id=turn_id,
        pages=pages,
        fraction=fraction,
        vllm_page_tokens=vllm_page_tokens,
        recent_turns=recent_turns,
        recent_quota=recent_quota,
    )


def build_goal_query_payload(
    *,
    index: PageVectorIndex,
    provenance: PageVectorProvenance,
    session_id: str,
    turn_id: int,
    pages: Sequence[PageRecord],
    fraction: float,
    vllm_page_tokens: int,
    recent_turns: int = 2,
    recent_quota: int = 1,
) -> VllmQueryPolicyPayload:
    return _build_payload(
        policy="goal_query",
        index=index,
        provenance=provenance,
        session_id=session_id,
        turn_id=turn_id,
        pages=pages,
        fraction=fraction,
        vllm_page_tokens=vllm_page_tokens,
        recent_turns=recent_turns,
        recent_quota=recent_quota,
    )


def _build_payload(
    *,
    policy: str,
    index: PageVectorIndex,
    provenance: PageVectorProvenance,
    session_id: str,
    turn_id: int,
    pages: Sequence[PageRecord],
    fraction: float,
    vllm_page_tokens: int,
    recent_turns: int,
    recent_quota: int,
) -> VllmQueryPolicyPayload:
    if policy not in _POLICIES:
        raise ValueError(f"unsupported query policy {policy!r}")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if vllm_page_tokens <= 0 or vllm_page_tokens % provenance.page_size:
        raise ValueError(
            "vllm_page_tokens must be a positive multiple of the DRAM-KV page size"
        )
    if recent_turns < 0 or recent_quota < 0:
        raise ValueError("recent safety bounds cannot be negative")

    history = tuple(page for page in pages if page.turn_id < turn_id)
    groups: dict[int, list[PageRecord]] = {}
    for page in history:
        coarse_idx = page.logical_position_start // vllm_page_tokens
        if page.logical_position_end > (coarse_idx + 1) * vllm_page_tokens:
            raise ValueError(
                f"DRAM-KV page {page.page_id} crosses a vLLM page boundary"
            )
        groups.setdefault(coarse_idx, []).append(page)

    representatives = {
        coarse_idx: max(group, key=lambda page: page.logical_position_end)
        for coarse_idx, group in groups.items()
        if max(page.logical_position_end for page in group)
        == (coarse_idx + 1) * vllm_page_tokens
    }
    if not representatives:
        raise ValueError("no complete vLLM-aligned history pages are available")

    snapshot = index.snapshot(
        session_id,
        turn_id,
        (page.page_id for page in representatives.values()),
        provenance,
    )
    if not snapshot.complete or snapshot.query is None:
        reasons = []
        if snapshot.query is None:
            reasons.append("query missing or incompatible")
        if snapshot.missing_page_ids:
            reasons.append("missing pages=" + ",".join(snapshot.missing_page_ids))
        if snapshot.mismatched_page_ids:
            reasons.append(
                "mismatched pages=" + ",".join(snapshot.mismatched_page_ids)
            )
        raise ValueError("incomplete Page-vector snapshot: " + "; ".join(reasons))

    query = snapshot.query.vector
    scores = {
        coarse_idx: compact_cosine(
            query,
            snapshot.pages[page.page_id].vector,
        )
        for coarse_idx, page in representatives.items()
    }

    mandatory = sorted(
        coarse_idx
        for coarse_idx, group in groups.items()
        if coarse_idx in representatives and any(page.mandatory for page in group)
    )
    mandatory_set = set(mandatory)

    cutoff = max(0, turn_id - recent_turns)
    recent_candidates = [
        coarse_idx
        for coarse_idx, group in groups.items()
        if coarse_idx in representatives
        and coarse_idx not in mandatory_set
        and max(page.turn_id for page in group) >= cutoff
    ]
    recent_candidates.sort(
        key=lambda coarse_idx: (
            -max(page.turn_id for page in groups[coarse_idx]),
            -coarse_idx,
        )
    )
    recent = recent_candidates[:recent_quota]
    safety_set = mandatory_set | set(recent)

    semantic = sorted(
        (
            coarse_idx
            for coarse_idx in representatives
            if coarse_idx not in safety_set
        ),
        key=lambda coarse_idx: (-scores[coarse_idx], coarse_idx),
    )
    ranking = tuple((*mandatory, *recent, *semantic))
    selected_count = int(len(representatives) * fraction)
    if selected_count <= 0:
        raise ValueError("fraction would select zero vLLM pages")
    if len(safety_set) > selected_count:
        raise ValueError(
            "mandatory/recent safety Pages exceed the selective Page budget"
        )

    query_hash = sha256(
        (
            policy
            + ":"
            + provenance.provenance_hash
            + ":"
            + query.content_hash
        ).encode("ascii")
    ).hexdigest()
    return VllmQueryPolicyPayload(
        policy=policy,
        fraction=fraction,
        ranked_logical_pages=ranking,
        query_hash=query_hash,
    )


__all__ = [
    "VllmQueryPolicyPayload",
    "build_goal_query_payload",
    "build_observation_query_payload",
]
