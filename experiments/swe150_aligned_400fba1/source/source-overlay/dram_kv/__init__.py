"""DRAM-backed goal-conditioned KV working-set experiment framework."""

from .pipeline import ExperimentPipeline
from .page_vectors import (
    CompactVector,
    PageVectorIndex,
    PageVectorProvenance,
    PageVectorPublisher,
    SimulatedPrefillPageVectorPublisher,
)
from .schema import (
    GoalPolicyHint,
    GoalRecord,
    PageCatalog,
    PageRecord,
    RetrievalGoal,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)

__all__ = [
    "ExperimentPipeline",
    "CompactVector",
    "GoalPolicyHint",
    "GoalRecord",
    "PageCatalog",
    "PageRecord",
    "PageVectorIndex",
    "PageVectorProvenance",
    "PageVectorPublisher",
    "RetrievalGoal",
    "SelectionPlan",
    "SelectionReason",
    "SimulatedPrefillPageVectorPublisher",
    "TurnContext",
]
