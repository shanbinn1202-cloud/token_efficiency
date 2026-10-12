"""Self-contained JSONL traces and metrics for offline Shadow experiments.

Every aggregate emitted here can be recomputed from the preceding
``shadow_turn`` records.  In particular, effective Goal updates compare the
retrieval-only hash rather than the selector hash, so a phase-policy change is
not accidentally counted as a retrieval-Goal update.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypedDict

from ..schema import GoalRecord, SelectionPlan, TurnContext


TRACE_SCHEMA_VERSION = "1.0.0"


class ShadowTurnTrace(TypedDict, total=False):
    """JSON-compatible envelope for exactly one turn boundary."""

    record_type: str
    schema_version: str
    run_id: str
    config: Any
    session_id: str
    instance_id: str
    turn_id: int
    context_boundary: dict[str, Any]
    goal: dict[str, Any]
    root_goal_hash: str
    retrieval_goal_hash: str
    policy_hash: str
    selector_hash: str
    selector_goal_hash: str
    selection_plan: dict[str, Any]
    backend_metrics: dict[str, Any]


class RunSummary(TypedDict, total=False):
    """JSON-compatible aggregate appended after a run's turn records."""

    record_type: str
    schema_version: str
    run_id: str | None
    config: Any
    turn_count: int
    session_count: int
    maintenance_boundaries: int


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy of a record value."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=_stable_json)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _text_boundary_summary(value: str | None, *, excerpt_chars: int) -> dict[str, Any]:
    text = value or ""
    return {
        "chars": len(text),
        "sha256": sha256(text.encode("utf-8")).hexdigest(),
        "excerpt": text[:excerpt_chars],
        "truncated": len(text) > excerpt_chars,
    }


def build_turn_trace(
    context: TurnContext,
    goal: GoalRecord,
    selection_plan: SelectionPlan,
    backend_metrics: Mapping[str, Any],
    *,
    run_id: str,
    config: str | Mapping[str, Any],
    schema_version: str = TRACE_SCHEMA_VERSION,
) -> ShadowTurnTrace:
    """Build one complete, bounded Shadow turn envelope.

    The latest action and observation are represented by bounded excerpts plus
    hashes and lengths.  The complete bounded Goal record (including its
    feature, phase, gate and SLM diagnostics), SelectionPlan, and backend
    metrics are retained so the summary can be independently recomputed.
    """

    if not run_id.strip():
        raise ValueError("run_id is required")
    if not schema_version.strip():
        raise ValueError("schema_version is required")
    if not isinstance(config, (str, Mapping)):
        raise TypeError("config must be a name or a mapping")
    if goal.session_id != context.session_id or goal.turn_id != context.turn_id:
        raise ValueError("Goal and context must describe the same turn")
    if (
        selection_plan.session_id != context.session_id
        or selection_plan.turn_id != context.turn_id
    ):
        raise ValueError("SelectionPlan and context must describe the same turn")

    structured_state = _json_value(context.structured_state)
    structured_keys = sorted(str(key) for key in context.structured_state)
    context_summary = {
        "message_source": context.message_source,
        "historical_turn_count": len(context.historical_turn_ids),
        "historical_page_count": len(context.historical_page_ids),
        "hbm_resident_page_count": len(context.hbm_resident_page_ids),
        "previous_goal_present": context.previous_goal_text is not None,
        "explicit_requirement_present": context.explicit_requirement is not None,
        "previous_action": _text_boundary_summary(
            context.previous_action, excerpt_chars=512
        ),
        "current_observation": _text_boundary_summary(
            context.current_observation, excerpt_chars=1_024
        ),
        "structured_state": {
            "key_count": len(structured_keys),
            "keys": structured_keys[:64],
            "keys_truncated": len(structured_keys) > 64,
            "sha256": sha256(_stable_json(structured_state).encode("utf-8")).hexdigest(),
            "tool_name": structured_state.get("tool_name"),
            "return_code": structured_state.get("return_code"),
        },
    }
    selector_hash = goal.version_hash
    return ShadowTurnTrace(
        record_type="shadow_turn",
        schema_version=schema_version,
        run_id=run_id,
        config=_json_value(config),
        session_id=context.session_id,
        instance_id=context.instance_id,
        turn_id=context.turn_id,
        context_boundary=context_summary,
        goal=_json_value(asdict(goal)),
        root_goal_hash=goal.root_goal_hash,
        retrieval_goal_hash=goal.retrieval_goal_hash,
        policy_hash=goal.policy_hash,
        selector_hash=selector_hash,
        # Explicit alias makes the relation to SelectionPlan.goal_hash clear.
        selector_goal_hash=selector_hash,
        selection_plan=_json_value(asdict(selection_plan)),
        backend_metrics=_json_value(backend_metrics),
    )


def _number(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else default


def _integer(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear-interpolated percentile, defined for singleton samples."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    sample = list(values)
    return {
        "count": len(sample),
        "sum": sum(sample),
        "p50": _percentile(sample, 0.50),
        "p95": _percentile(sample, 0.95),
    }


def _goal(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    value = trace.get("goal")
    return value if isinstance(value, Mapping) else {}


def _diagnostics(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _goal(trace).get("diagnostics")
    return value if isinstance(value, Mapping) else {}


def _slm_costs(traces: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "input_tokens": sum(_integer(_goal(trace).get("input_tokens")) for trace in traces),
        "output_tokens": sum(
            _integer(_goal(trace).get("output_tokens")) for trace in traces
        ),
        "model_cost": sum(_number(_goal(trace).get("model_cost")) for trace in traces),
    }


def _latencies(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = [_number(_goal(trace).get("latency_ms")) for trace in traces]
    rule = [_number(_diagnostics(trace).get("rule_latency_ms")) for trace in traces]
    slm_amortized = [
        _number(_diagnostics(trace).get("slm_latency_ms")) for trace in traces
    ]
    slm_calls = [
        _number(_diagnostics(trace).get("slm_latency_ms"))
        for trace in traces
        if bool(_goal(trace).get("slm_attempted"))
    ]
    return {
        "total": _distribution(total),
        "rule": _distribution(rule),
        "slm": _distribution(slm_calls),
        "slm_amortized": _distribution(slm_amortized),
        "percentile_method": "linear_interpolation",
    }


def summarize_turn_traces(
    turns: Iterable[Mapping[str, Any]],
) -> RunSummary:
    """Recompute run metrics solely from ``shadow_turn`` envelopes.

    A maintenance boundary is every turn except the chronologically first turn
    of each session.  Empty-evidence boundaries do not enter either Shadow
    Recall@Budget denominator.
    """

    records = [dict(turn) for turn in turns]
    for record in records:
        if record.get("record_type") != "shadow_turn":
            raise ValueError("summaries accept only shadow_turn records")

    run_ids = {str(record.get("run_id")) for record in records}
    configs = {_stable_json(record.get("config")) for record in records}
    schemas = {str(record.get("schema_version")) for record in records}
    if len(run_ids) > 1 or len(configs) > 1 or len(schemas) > 1:
        raise ValueError("all turn records must share run_id, config, and schema_version")

    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_boundaries: set[tuple[str, int]] = set()
    for record in records:
        session_id = str(record.get("session_id", ""))
        turn_id = record.get("turn_id")
        if not session_id or isinstance(turn_id, bool) or not isinstance(turn_id, int):
            raise ValueError("each turn requires string session_id and integer turn_id")
        boundary = (session_id, turn_id)
        if boundary in seen_boundaries:
            raise ValueError(f"duplicate turn boundary {boundary!r}")
        seen_boundaries.add(boundary)
        by_session[session_id].append(record)

    initial: list[Mapping[str, Any]] = []
    maintenance: list[Mapping[str, Any]] = []
    previous_by_boundary: dict[tuple[str, int], Mapping[str, Any]] = {}
    for session_id, session_records in by_session.items():
        ordered = sorted(session_records, key=lambda item: int(item["turn_id"]))
        initial.append(ordered[0])
        for previous, current in zip(ordered, ordered[1:]):
            maintenance.append(current)
            previous_by_boundary[(session_id, int(current["turn_id"]))] = previous

    decision_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    effective_updates = 0
    policy_only_updates = 0
    zero_retrieval_changes = 0
    keep_boundaries = 0
    keep_with_retrieval_change = 0
    for trace in maintenance:
        goal = _goal(trace)
        decision_counts[str(goal.get("gate_decision", "UNKNOWN"))] += 1
        applied_counts[str(goal.get("applied_decision", "UNKNOWN"))] += 1
        triggers = goal.get("triggers", ())
        if isinstance(triggers, (list, tuple)):
            trigger_counts.update(str(trigger) for trigger in triggers)
        previous = previous_by_boundary[(str(trace["session_id"]), int(trace["turn_id"]))]
        retrieval_changed = (
            trace.get("retrieval_goal_hash") != previous.get("retrieval_goal_hash")
        )
        selector_changed = trace.get("selector_hash") != previous.get("selector_hash")
        if retrieval_changed:
            effective_updates += 1
            if goal.get("gate_decision") == "KEEP":
                keep_with_retrieval_change += 1
        elif selector_changed:
            policy_only_updates += 1
        if not retrieval_changed:
            zero_retrieval_changes += 1
        if goal.get("gate_decision") == "KEEP":
            keep_boundaries += 1

    def slm_counts(scope: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        attempts = sum(bool(_goal(trace).get("slm_attempted")) for trace in scope)
        successes = sum(bool(_goal(trace).get("slm_succeeded")) for trace in scope)
        fallbacks = sum(_goal(trace).get("fallback_reason") is not None for trace in scope)
        return {
            "boundaries": len(scope),
            "attempts": attempts,
            "successes": successes,
            "fallbacks": fallbacks,
            "generation_rate": _rate(attempts, len(scope)),
            "success_rate_per_attempt": _rate(successes, attempts),
        }

    root_hash_violations: list[dict[str, Any]] = []
    diagnostic_violations: list[dict[str, Any]] = []
    root_union: set[tuple[str, int]] = set()
    for session_id, session_records in by_session.items():
        ordered = sorted(session_records, key=lambda item: int(item["turn_id"]))
        frozen_hash = ordered[0].get("root_goal_hash")
        for trace in ordered:
            turn_id = int(trace["turn_id"])
            boundary = (session_id, turn_id)
            if trace.get("root_goal_hash") != frozen_hash:
                root_hash_violations.append(
                    {"session_id": session_id, "turn_id": turn_id}
                )
                root_union.add(boundary)
            if _diagnostics(trace).get("root_invariant") is False:
                diagnostic_violations.append(
                    {"session_id": session_id, "turn_id": turn_id}
                )
                root_union.add(boundary)

    recalls: list[float] = []
    evidence_total = 0
    found_total = 0
    labeled_boundaries = 0
    for trace in records:
        metrics = trace.get("backend_metrics")
        if not isinstance(metrics, Mapping):
            continue
        evidence = _integer(metrics.get("evidence_pages"))
        if evidence == 0:
            continue
        found = _integer(metrics.get("evidence_pages_found"))
        if found > evidence:
            raise ValueError("evidence_pages_found cannot exceed evidence_pages")
        labeled_boundaries += 1
        evidence_total += evidence
        found_total += found
        recalls.append(found / evidence)

    config_value = records[0].get("config") if records else None
    summary = RunSummary(
        record_type="run_summary",
        schema_version=(str(records[0]["schema_version"]) if records else TRACE_SCHEMA_VERSION),
        run_id=(str(records[0]["run_id"]) if records else None),
        config=_json_value(config_value),
        turn_count=len(records),
        session_count=len(by_session),
        initialization_boundaries=len(initial),
        maintenance_boundaries=len(maintenance),
        decision_counts=dict(sorted(decision_counts.items())),
        applied_decision_counts=dict(sorted(applied_counts.items())),
        trigger_counts=dict(sorted(trigger_counts.items())),
        effective_retrieval_goal_updates=effective_updates,
        effective_retrieval_goal_update_rate=_rate(
            effective_updates, len(maintenance)
        ),
        zero_retrieval_goal_change_rate=_rate(
            zero_retrieval_changes, len(maintenance)
        ),
        keep_invariance={
            "boundaries": keep_boundaries,
            "retrieval_goal_change_violations": keep_with_retrieval_change,
            "invariant": keep_with_retrieval_change == 0,
        },
        policy_only_updates=policy_only_updates,
        slm={
            "maintenance": slm_counts(maintenance),
            "initialization": slm_counts(initial),
        },
        latency_ms={
            "all": _latencies(records),
            "maintenance": _latencies(maintenance),
            "initialization": _latencies(initial),
        },
        token_cost_totals={
            "all": _slm_costs(records),
            "maintenance": _slm_costs(maintenance),
            "initialization": _slm_costs(initial),
        },
        root_invariance={
            "root_hash_violation_count": len(root_hash_violations),
            "diagnostic_violation_count": len(diagnostic_violations),
            "violation_count": len(root_union),
            "invariant": not root_union,
            "root_hash_violations": root_hash_violations,
            "diagnostic_violations": diagnostic_violations,
        },
        shadow_evidence_recall_at_budget={
            "labeled_boundaries": labeled_boundaries,
            "unlabeled_boundaries": len(records) - labeled_boundaries,
            "evidence_pages": evidence_total,
            "evidence_pages_found": found_total,
            "macro": (sum(recalls) / len(recalls) if recalls else None),
            "micro": _rate(found_total, evidence_total),
        },
    )
    return summary


def write_shadow_trace(
    path: str | PathLike[str],
    turns: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write turn envelopes followed by one summary as UTF-8 JSONL."""

    turn_records = [_json_value(dict(turn)) for turn in turns]
    if any(turn.get("record_type") != "shadow_turn" for turn in turn_records):
        raise ValueError("turns must contain only shadow_turn records")
    summary_record = _json_value(dict(summary))
    if summary_record.get("record_type") != "run_summary":
        raise ValueError("summary must be a run_summary record")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in (*turn_records, summary_record):
            handle.write(_stable_json(record))
            handle.write("\n")
