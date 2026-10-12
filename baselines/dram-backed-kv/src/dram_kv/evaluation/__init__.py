"""Offline, dependency-free evaluation helpers."""

from .shadow import (
    TRACE_SCHEMA_VERSION,
    RunSummary,
    ShadowTurnTrace,
    build_turn_trace,
    summarize_turn_traces,
    write_shadow_trace,
)
from .calibration import calibrate_recorded_thresholds

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "RunSummary",
    "ShadowTurnTrace",
    "build_turn_trace",
    "summarize_turn_traces",
    "write_shadow_trace",
    "calibrate_recorded_thresholds",
]
