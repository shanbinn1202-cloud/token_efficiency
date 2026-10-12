"""Offline, dependency-free Goal Constructor Shadow replay matrix."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..artifacts import turn_context_from_dict
from ..backends import ShadowBackend
from ..evaluation.shadow import build_turn_trace, summarize_turn_traces
from ..goals import (
    CommandGoalRewriter,
    DeterministicFakeRewriter,
    GoalRewriter,
    MaintainedGoalConstructor,
    PersistentCommandGoalRewriter,
    RawObjectiveGoalConstructor,
    StatelessHeuristicGoalConstructor,
)
from ..goals.state import PhaseTrackerConfig
from ..paging import TokenSegment, build_page_catalog
from ..pipeline import ExperimentPipeline
from ..selectors import LexicalMetadataSelector


SHADOW_CONFIGS = (
    "raw_objective",
    "stateless_heuristic",
    "heuristic_only",
    "heuristic_slm",
)


@dataclass(frozen=True)
class ShadowRun:
    run_id: str
    config_name: str
    turns: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def load_shadow_replay(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Shadow replay must be a JSON object")
    return value


def _segments(payload: Mapping[str, Any]) -> tuple[TokenSegment, ...]:
    result = []
    for raw in payload.get("segments", ()):  # type: ignore[assignment]
        if not isinstance(raw, Mapping):
            raise ValueError("each segment must be an object")
        data = dict(raw)
        for name in ("files", "symbols", "tests", "errors"):
            data[name] = tuple(data.get(name, ()))
        result.append(TokenSegment(**data))
    if not result:
        raise ValueError("Shadow replay requires segments")
    return tuple(result)


def _evidence(payload: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
    raw = payload.get("evidence_page_ids", {})
    if not isinstance(raw, Mapping):
        raise ValueError("evidence_page_ids must be an object")
    return {int(turn): tuple(str(item) for item in page_ids) for turn, page_ids in raw.items()}


def _run_id(
    payload: Mapping[str, Any], config_name: str, rewriter_name: str,
    constructor_variant: Mapping[str, Any] | None = None,
) -> str:
    stable = json.dumps(
        {"payload": payload, "config": config_name, "rewriter": rewriter_name,
         "constructor_variant": constructor_variant},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{config_name}-{sha256(stable.encode('utf-8')).hexdigest()[:12]}"


def run_shadow_configuration(
    payload: Mapping[str, Any],
    config_name: str,
    *,
    slm_command: Sequence[str] | None = None,
    goal_rewriter: GoalRewriter | None = None,
    initialize_with_rewriter: bool = True,
    phase_config: PhaseTrackerConfig | None = None,
    entity_drift_steps: int = 2,
    slm_timeout_seconds: float = 30.0,
    slm_persistent: bool = False,
    slm_startup_timeout_seconds: float = 300.0,
) -> ShadowRun:
    """Run one configuration against a fixed catalog, labels, and budget."""

    if config_name not in SHADOW_CONFIGS:
        raise ValueError(f"unsupported Shadow configuration {config_name!r}")
    if goal_rewriter is not None and config_name != "heuristic_slm":
        raise ValueError("goal_rewriter injection requires heuristic_slm")
    if goal_rewriter is not None and slm_command:
        raise ValueError("goal_rewriter and slm_command are mutually exclusive")
    if payload.get("schema_name") != "dram_kv_goal_shadow_replay":
        raise ValueError("unsupported Shadow replay schema")
    session_id = str(payload.get("session_id", ""))
    if not session_id:
        raise ValueError("Shadow replay requires session_id")
    page_size = int(payload.get("page_size", 128))
    bytes_per_token = int(payload.get("bytes_per_token", 0))
    catalog = build_page_catalog(
        session_id,
        _segments(payload),
        page_size=page_size,
        bytes_per_token=bytes_per_token,
    )
    selector_raw = payload.get("selector", {})
    if not isinstance(selector_raw, Mapping):
        raise ValueError("selector must be an object")
    selector = LexicalMetadataSelector(
        page_budget=int(selector_raw.get("page_budget", 8)),
        recent_turns=int(selector_raw.get("recent_turns", 2)),
        recent_quota=int(selector_raw.get("recent_quota", 1)),
    )

    rewriter_name = "none"
    persistent_rewriter: PersistentCommandGoalRewriter | None = None
    if config_name == "raw_objective":
        constructor = RawObjectiveGoalConstructor()
    elif config_name == "stateless_heuristic":
        constructor = StatelessHeuristicGoalConstructor()
    elif config_name == "heuristic_only":
        constructor = MaintainedGoalConstructor(
            phase_config=phase_config,
            entity_drift_steps=entity_drift_steps,
        )
    else:
        if goal_rewriter is not None:
            rewriter = goal_rewriter
        elif slm_command:
            if slm_persistent:
                rewriter = PersistentCommandGoalRewriter(
                    tuple(slm_command),
                    timeout_seconds=slm_timeout_seconds,
                    startup_timeout_seconds=slm_startup_timeout_seconds,
                )
                persistent_rewriter = rewriter
            else:
                rewriter = CommandGoalRewriter(
                    tuple(slm_command), timeout_seconds=slm_timeout_seconds
                )
        else:
            rewriter = DeterministicFakeRewriter()
        rewriter_name = type(rewriter).__name__
        constructor = MaintainedGoalConstructor(
            rewriter,
            phase_config=phase_config,
            entity_drift_steps=entity_drift_steps,
            initialize_with_rewriter=initialize_with_rewriter,
        )

    evidence = _evidence(payload)
    backend = ShadowBackend({(session_id, turn): pages for turn, pages in evidence.items()})
    pipeline = ExperimentPipeline(constructor, selector, backend)
    constructor_variant = (
        constructor._config() if isinstance(constructor, MaintainedGoalConstructor) else None
    )
    if constructor_variant is not None and slm_command:
        constructor_variant = {
            **constructor_variant,
            "slm_command_sha256": sha256(
                json.dumps(list(slm_command), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "slm_timeout_seconds": slm_timeout_seconds,
            "slm_persistent": slm_persistent,
            "slm_startup_timeout_seconds": slm_startup_timeout_seconds,
        }
    run_id = _run_id(payload, config_name, rewriter_name, constructor_variant)
    config = {
        "name": config_name,
        "goal_constructor": constructor.name,
        "selector": selector.name,
        "selector_config": {
            "page_budget": selector.page_budget,
            "recent_turns": selector.recent_turns,
            "recent_quota": selector.recent_quota,
        },
        "rewriter": rewriter_name,
        "initialize_with_rewriter": (
            initialize_with_rewriter if config_name == "heuristic_slm" else False
        ),
        "source": payload.get("source"),
        "constructor_config": constructor_variant,
    }
    traces: list[Mapping[str, Any]] = []
    raw_turns = payload.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError("Shadow replay requires a non-empty turns array")
    for raw in raw_turns:
        if not isinstance(raw, Mapping):
            raise ValueError("each Shadow replay turn must be an object")
        context = turn_context_from_dict(raw)
        if context.session_id != session_id:
            raise ValueError("turn session_id differs from replay session_id")
        result = pipeline.run_turn(context, catalog)
        traces.append(build_turn_trace(
            context,
            result.goal,
            result.selection,
            result.backend_metrics,
            run_id=run_id,
            config=config,
        ))
    summary = summarize_turn_traces(traces)
    if persistent_rewriter is not None:
        persistent_rewriter.close()
    return ShadowRun(run_id, config_name, tuple(traces), summary)


def run_shadow_matrix(
    payload: Mapping[str, Any],
    *,
    configs: Iterable[str] = SHADOW_CONFIGS,
    slm_command: Sequence[str] | None = None,
    goal_rewriter: GoalRewriter | None = None,
    initialize_with_rewriter: bool = True,
    phase_config: PhaseTrackerConfig | None = None,
    entity_drift_steps: int = 2,
    slm_timeout_seconds: float = 30.0,
    slm_persistent: bool = False,
    slm_startup_timeout_seconds: float = 300.0,
) -> tuple[ShadowRun, ...]:
    names = tuple(configs)
    if len(names) != len(set(names)):
        raise ValueError("Shadow configuration names must be unique")
    return tuple(
        run_shadow_configuration(
            payload,
            name,
            slm_command=slm_command,
            goal_rewriter=goal_rewriter,
            initialize_with_rewriter=initialize_with_rewriter,
            phase_config=phase_config,
            entity_drift_steps=entity_drift_steps,
            slm_timeout_seconds=slm_timeout_seconds,
            slm_persistent=slm_persistent,
            slm_startup_timeout_seconds=slm_startup_timeout_seconds,
        )
        for name in names
    )


__all__ = [
    "SHADOW_CONFIGS",
    "ShadowRun",
    "load_shadow_replay",
    "run_shadow_configuration",
    "run_shadow_matrix",
]
