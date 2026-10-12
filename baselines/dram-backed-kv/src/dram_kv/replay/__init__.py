"""Deterministic recorded-trajectory replay helpers."""

from .recorded_loop import (
    ActiveBudget,
    BudgetDecision,
    BudgetExceededError,
    OracleEvidence,
    RecordedTrajectory,
    RecordedTurn,
    RecordedTurnArtifact,
    RecordedTurnLoop,
    RenderedSpan,
    build_selection_plan,
    oracle_evidence_from_annotation,
)

__all__ = [
    "ActiveBudget",
    "BudgetDecision",
    "BudgetExceededError",
    "OracleEvidence",
    "RecordedTrajectory",
    "RecordedTurn",
    "RecordedTurnArtifact",
    "RecordedTurnLoop",
    "RenderedSpan",
    "build_selection_plan",
    "oracle_evidence_from_annotation",
]
from .goal_shadow import (
    SHADOW_CONFIGS,
    ShadowRun,
    load_shadow_replay,
    run_shadow_configuration,
    run_shadow_matrix,
)
from .swe_agent import (
    RecordedBoundary,
    SweAgentBundle,
    build_semantic_shadow_payload,
    load_swe_agent_bundle,
    message_map_sha256,
    recorded_boundaries,
    recorded_message_map,
    validate_oracle_binding,
)

__all__ += [
    "SHADOW_CONFIGS",
    "ShadowRun",
    "load_shadow_replay",
    "run_shadow_configuration",
    "run_shadow_matrix",
    "RecordedBoundary",
    "SweAgentBundle",
    "build_semantic_shadow_payload",
    "load_swe_agent_bundle",
    "message_map_sha256",
    "recorded_boundaries",
    "recorded_message_map",
    "validate_oracle_binding",
]
