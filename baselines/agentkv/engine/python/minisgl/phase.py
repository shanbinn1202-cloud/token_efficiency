from __future__ import annotations

from enum import IntEnum

import torch


class SemanticPhase(IntEnum):
    UNKNOWN = -1
    SYSTEM = 0
    USER = 1
    ASSISTANT_TEXT = 2
    ASSISTANT_TOOL_CALL = 3
    TOOL_RESPONSE = 4


class QueryRingRoute(IntEnum):
    IGNORE = -1
    STAGE0 = 0
    STAGE1 = 1
    STAGE2 = 2
    STAGE3 = 3


_STAGEQ_ROUTE_MAPS: dict[str, dict[SemanticPhase, QueryRingRoute]] = {
    "fix1": {
        SemanticPhase.SYSTEM: QueryRingRoute.STAGE0,
        SemanticPhase.ASSISTANT_TEXT: QueryRingRoute.STAGE1,
        SemanticPhase.ASSISTANT_TOOL_CALL: QueryRingRoute.STAGE1,
        SemanticPhase.TOOL_RESPONSE: QueryRingRoute.STAGE2,
        SemanticPhase.USER: QueryRingRoute.STAGE3,
    },
    "fix2": {
        SemanticPhase.SYSTEM: QueryRingRoute.STAGE0,
        SemanticPhase.ASSISTANT_TEXT: QueryRingRoute.STAGE1,
        SemanticPhase.TOOL_RESPONSE: QueryRingRoute.STAGE2,
        SemanticPhase.ASSISTANT_TOOL_CALL: QueryRingRoute.STAGE3,
        SemanticPhase.USER: QueryRingRoute.IGNORE,
    },
    "fix3": {
        SemanticPhase.SYSTEM: QueryRingRoute.STAGE0,
        SemanticPhase.ASSISTANT_TEXT: QueryRingRoute.STAGE1,
        SemanticPhase.ASSISTANT_TOOL_CALL: QueryRingRoute.STAGE1,
        SemanticPhase.TOOL_RESPONSE: QueryRingRoute.STAGE2,
        SemanticPhase.USER: QueryRingRoute.IGNORE,
    },
}


def route_semantic_phases_to_query_stages(
    semantic_phase_ids: torch.Tensor,
    mapping: str,
) -> torch.Tensor:
    if mapping not in _STAGEQ_ROUTE_MAPS:
        raise ValueError(f"Unknown Stage-Q query-stage mapping: {mapping}")
    routes = _STAGEQ_ROUTE_MAPS[mapping]
    stage_ids = torch.full_like(
        semantic_phase_ids,
        int(QueryRingRoute.IGNORE),
        dtype=torch.int32,
    )
    for semantic_phase, route in routes.items():
        stage_ids = torch.where(
            semantic_phase_ids == int(semantic_phase),
            torch.full_like(stage_ids, int(route)),
            stage_ids,
        )
    return stage_ids
