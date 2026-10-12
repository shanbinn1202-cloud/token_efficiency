"""Session-scoped heuristic maintenance with conditional retrieval-Goal rewrite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from threading import RLock
import time
from typing import Iterable

from ..schema import GoalPolicyHint, GoalRecord, RetrievalGoal, TurnContext
from .features import DeterministicFeatureExtractor, FeatureSnapshot
from .rewriter import GoalRewriter, RewriteRequest, validate_rewrite_candidate
from .state import PhaseSnapshot, PhaseTracker, PhaseTrackerConfig


_ENTITY_FIELDS = ("files", "symbols", "tests", "errors")
_MAX_ENTITIES = {"files": 32, "symbols": 32, "tests": 24, "errors": 16}


def _bounded_union(
    existing: Iterable[str], additions: Iterable[str], *, limit: int
) -> tuple[str, ...]:
    values = list(dict.fromkeys((*existing, *additions)))
    return tuple(values[-limit:])


def _error_signature(value: str) -> str:
    return value.split(":", 1)[0].strip()


def _goal_entities(goal: RetrievalGoal) -> set[str]:
    return {
        item
        for name in _ENTITY_FIELDS
        for item in getattr(goal, name)
    }


def _feature_entities(features: FeatureSnapshot) -> set[str]:
    return {
        item
        for name in _ENTITY_FIELDS
        for item in (
            features.confirmed_errors if name == "errors" else getattr(features, name)
        )
    }


def _changed_fields(before: RetrievalGoal, after: RetrievalGoal) -> tuple[str, ...]:
    return tuple(
        item.name for item in fields(RetrievalGoal)
        if getattr(before, item.name) != getattr(after, item.name)
    )


def _policy_for_phase(phase: str) -> GoalPolicyHint:
    values = {
        "exploration": (20_000, "high", 0.85, 0.50,
                        ("task_anchor", "recent_tail", "diverse_history")),
        "hypothesis": (16_000, "medium", 0.50, 0.90,
                       ("diagnostic_output", "stack_trace", "error_entities")),
        "implementation": (12_000, "low", 0.20, 0.95,
                           ("edited_files", "symbols", "dependency_neighbors")),
        "verification": (16_000, "medium", 0.20, 0.65,
                         ("test_results", "reproduction_verdict", "edit_records")),
        "stuck": (20_000, "high", 0.95, 0.50,
                  ("recent_success", "failed_attempts", "verification_records",
                   "task_anchor")),
    }
    budget, slack, diversity, structure, priorities = values[phase]
    return GoalPolicyHint(
        phase=phase,
        active_token_budget=budget,
        slack=slack,
        diversity_weight=diversity,
        structure_weight=structure,
        priorities=priorities,
    )


def _initial_evidence(phase: str) -> tuple[str, ...]:
    return {
        "exploration": ("relevant files and symbols",),
        "hypothesis": ("error signature and diagnostic evidence",),
        "implementation": ("edited files and dependency context",),
        "verification": ("test and reproduction verdict",),
        "stuck": ("recent successful state", "failed attempts"),
    }[phase]


@dataclass
class _SessionState:
    instance_id: str
    root_goal: str
    root_goal_hash: str
    retrieval_goal: RetrievalGoal
    phase_tracker: PhaseTracker
    last_turn: int
    last_top_frame: str | None
    drift_streak: int = 0


class MaintainedGoalConstructor:
    """Maintain a bounded selector Goal and invoke an SLM only on hard rewrites.

    The object may serve multiple sessions. Every mutable item is keyed by
    ``session_id`` and guarded by a lock; a session ID cannot silently switch
    instance or root objective.
    """

    name = "heuristic_conditional_slm_v1"

    def __init__(
        self,
        rewriter: GoalRewriter | None = None,
        *,
        phase_config: PhaseTrackerConfig | None = None,
        entity_drift_steps: int = 2,
        initialize_with_rewriter: bool = False,
        max_description_chars: int = 1_200,
    ) -> None:
        if entity_drift_steps <= 0 or max_description_chars <= 0:
            raise ValueError("Goal maintenance bounds must be positive")
        self.rewriter = rewriter
        self.phase_config = phase_config or PhaseTrackerConfig()
        self.entity_drift_steps = entity_drift_steps
        self.initialize_with_rewriter = initialize_with_rewriter
        self.max_description_chars = max_description_chars
        self.feature_extractor = DeterministicFeatureExtractor()
        self._sessions: dict[str, _SessionState] = {}
        self._lock = RLock()

    def construct(self, context: TurnContext) -> GoalRecord:
        started = time.perf_counter()
        with self._lock:
            features = self._extract(context)
            state = self._sessions.get(context.session_id)
            if state is None:
                return self._initialize(context, features, started)
            self._validate_boundary(state, context)
            phase = state.phase_tracker.update(features)
            record = self._maintain(context, state, features, phase, started)
            state.last_turn = context.turn_id
            if features.confirmed_top_frame:
                state.last_top_frame = features.confirmed_top_frame
            state.retrieval_goal = record.retrieval_goal
            return record

    def release_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _extract(self, context: TurnContext) -> FeatureSnapshot:
        raw_return_code = context.structured_state.get("return_code")
        return_code = (
            raw_return_code
            if isinstance(raw_return_code, int) and not isinstance(raw_return_code, bool)
            else None
        )
        raw_tool_name = context.structured_state.get("tool_name")
        tool_name = raw_tool_name if isinstance(raw_tool_name, str) else None
        observation = (
            context.current_observation
            if context.message_source in {"tool", "assistant"}
            else ""
        )
        return self.feature_extractor.extract(
            context.previous_action,
            observation,
            tool_name=tool_name,
            return_code=return_code,
        )

    def _initialize(
        self, context: TurnContext, features: FeatureSnapshot, started: float
    ) -> GoalRecord:
        root_goal = context.task_objective.strip()
        if not root_goal:
            raise ValueError("task_objective is required to freeze root_goal")
        tracker = PhaseTracker(self.phase_config)
        phase = tracker.update(features)
        trusted_constraint = (
            (context.explicit_requirement,)
            if context.explicit_requirement is not None
            else ()
        )
        goal = RetrievalGoal(
            description=(f"Retrieve historical evidence relevant to: {root_goal}")[
                : self.max_description_chars
            ],
            phase_hint=(
                phase.stable_phase if phase.stable_phase != "stuck" else "exploration"
            ),
            files=features.files,
            symbols=features.symbols,
            tests=features.tests,
            errors=features.errors,
            evidence_needed=_initial_evidence(phase.stable_phase),
            constraints=trusted_constraint,
        )
        slm_attempted = False
        slm_succeeded = False
        fallback_reason = None
        input_tokens = output_tokens = 0
        model_cost = slm_latency_ms = 0.0
        slm_request_latency_ms = slm_startup_latency_ms = 0.0
        slm_generation_latency_ms = 0.0
        if self.initialize_with_rewriter and self.rewriter is not None:
            slm_attempted = True
            slm_started = time.perf_counter()
            try:
                result = self.rewriter.rewrite(RewriteRequest(
                    root_goal=root_goal,
                    current_goal=goal,
                    latest_action=context.previous_action or "",
                    latest_observation=(
                        context.current_observation
                        if context.message_source == "tool" else ""
                    ),
                    features=self._rewrite_features(features, context),
                    trigger="initialization",
                    stats=phase.stats.to_dict(),
                    allow_constraint_updates=False,
                ))
                goal = validate_rewrite_candidate(
                    result.candidate,
                    goal,
                    trigger="initialization",
                    allow_constraint_updates=False,
                )
                slm_succeeded = True
                slm_request_latency_ms = result.latency_ms
                slm_startup_latency_ms = result.startup_latency_ms
                slm_generation_latency_ms = result.generation_latency_ms
                slm_latency_ms = result.latency_ms + result.startup_latency_ms
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens
                model_cost = result.model_cost
            except Exception as error:  # adapters are untrusted experiment plugins
                elapsed_slm_ms = (time.perf_counter() - slm_started) * 1_000
                slm_request_latency_ms = float(getattr(error, "latency_ms", 0.0))
                slm_startup_latency_ms = float(
                    getattr(error, "startup_latency_ms", 0.0)
                )
                slm_generation_latency_ms = float(
                    getattr(error, "generation_latency_ms", 0.0)
                )
                input_tokens = int(getattr(error, "input_tokens", 0))
                output_tokens = int(getattr(error, "output_tokens", 0))
                slm_latency_ms = (
                    slm_request_latency_ms + slm_startup_latency_ms
                    or elapsed_slm_ms
                )
                fallback_reason = f"{type(error).__name__}: {error}"[-1_000:]

        root_hash = sha256(root_goal.encode("utf-8")).hexdigest()
        self._sessions[context.session_id] = _SessionState(
            instance_id=context.instance_id,
            root_goal=root_goal,
            root_goal_hash=root_hash,
            retrieval_goal=goal,
            phase_tracker=tracker,
            last_turn=context.turn_id,
            last_top_frame=features.confirmed_top_frame,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        return GoalRecord(
            session_id=context.session_id,
            turn_id=context.turn_id,
            constructor=self.name,
            constructor_config=self._config(),
            text=goal.description,
            root_goal=root_goal,
            retrieval_goal=goal,
            gate_decision="INITIALIZE",
            applied_decision="INITIALIZE",
            changed_fields=tuple(item.name for item in fields(RetrievalGoal)),
            policy_hint=_policy_for_phase(phase.stable_phase),
            slm_attempted=slm_attempted,
            slm_succeeded=slm_succeeded,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_cost=model_cost,
            diagnostics={
                "features": features.to_dict(),
                "phase": phase.to_dict(),
                "root_invariant": True,
                "initialization_slm": self.initialize_with_rewriter,
                "rule_latency_ms": max(0.0, latency_ms - slm_latency_ms),
                "slm_latency_ms": slm_latency_ms,
                "slm_request_latency_ms": slm_request_latency_ms,
                "slm_startup_latency_ms": slm_startup_latency_ms,
                "slm_generation_latency_ms": slm_generation_latency_ms,
                "entity_overlap": None,
                "drift_streak": 0,
            },
        )

    def _validate_boundary(self, state: _SessionState, context: TurnContext) -> None:
        if context.instance_id != state.instance_id:
            raise ValueError("session_id cannot be reused for another instance")
        if context.task_objective.strip() != state.root_goal:
            raise ValueError("immutable root_goal changed within the session")
        if context.turn_id <= state.last_turn:
            raise ValueError("Goal maintenance turns must be strictly increasing")

    def _maintain(
        self,
        context: TurnContext,
        state: _SessionState,
        features: FeatureSnapshot,
        phase: PhaseSnapshot,
        started: float,
    ) -> GoalRecord:
        current = state.retrieval_goal
        current_entities = _feature_entities(features)
        goal_entities = _goal_entities(current)
        overlap = (
            len(current_entities & goal_entities) / len(current_entities)
            if current_entities else None
        )
        if current_entities and goal_entities and not (current_entities & goal_entities):
            state.drift_streak += 1
        else:
            state.drift_streak = 0

        decision, trigger = self._gate(context, state, current, features, phase)
        applied = decision
        updated = current
        slm_attempted = False
        slm_succeeded = False
        fallback_reason = None
        slm_latency_ms = model_cost = 0.0
        slm_request_latency_ms = slm_startup_latency_ms = 0.0
        slm_generation_latency_ms = 0.0
        input_tokens = output_tokens = 0

        if decision == "KEEP":
            pass
        elif decision == "PATCH":
            updated = self._deterministic_patch(
                current, features, trigger, context.explicit_requirement
            )
        else:
            if self.rewriter is None:
                fallback_reason = "rewriter_disabled"
                updated = self._deterministic_patch(
                    current, features, trigger, context.explicit_requirement
                )
                applied = "PATCH" if updated != current else "KEEP"
            else:
                slm_attempted = True
                slm_started = time.perf_counter()
                try:
                    result = self.rewriter.rewrite(RewriteRequest(
                        root_goal=state.root_goal,
                        current_goal=current,
                        latest_action=context.previous_action or "",
                        latest_observation=(
                            context.current_observation
                            if context.message_source == "tool" else ""
                        ),
                        features=self._rewrite_features(features, context),
                        trigger=trigger,
                        stats={
                            **phase.stats.to_dict(),
                            "entity_overlap": overlap,
                            "drift_streak": state.drift_streak,
                        },
                        allow_constraint_updates=trigger == "user_request",
                    ))
                    updated = validate_rewrite_candidate(
                        result.candidate,
                        current,
                        trigger=trigger,
                        allow_constraint_updates=trigger == "user_request",
                    )
                    slm_succeeded = True
                    slm_request_latency_ms = result.latency_ms
                    slm_startup_latency_ms = result.startup_latency_ms
                    slm_generation_latency_ms = result.generation_latency_ms
                    slm_latency_ms = result.latency_ms + result.startup_latency_ms
                    input_tokens = result.input_tokens
                    output_tokens = result.output_tokens
                    model_cost = result.model_cost
                except Exception as error:  # fail closed to deterministic fields
                    elapsed_slm_ms = (time.perf_counter() - slm_started) * 1_000
                    slm_request_latency_ms = float(
                        getattr(error, "latency_ms", 0.0)
                    )
                    slm_startup_latency_ms = float(
                        getattr(error, "startup_latency_ms", 0.0)
                    )
                    slm_generation_latency_ms = float(
                        getattr(error, "generation_latency_ms", 0.0)
                    )
                    input_tokens = int(getattr(error, "input_tokens", 0))
                    output_tokens = int(getattr(error, "output_tokens", 0))
                    slm_latency_ms = (
                        slm_request_latency_ms + slm_startup_latency_ms
                        or elapsed_slm_ms
                    )
                    fallback_reason = f"{type(error).__name__}: {error}"[-1_000:]
                    updated = self._deterministic_patch(
                        current, features, trigger, context.explicit_requirement
                    )
                    applied = "PATCH" if updated != current else "KEEP"

        changed = _changed_fields(current, updated)
        latency_ms = (time.perf_counter() - started) * 1_000
        return GoalRecord(
            session_id=context.session_id,
            turn_id=context.turn_id,
            constructor=self.name,
            constructor_config=self._config(),
            text=updated.description,
            root_goal=state.root_goal,
            retrieval_goal=updated,
            reused_previous=not changed,
            gate_decision=decision,
            applied_decision=applied,
            triggers=(trigger,),
            changed_fields=changed,
            policy_hint=_policy_for_phase(phase.stable_phase),
            slm_attempted=slm_attempted,
            slm_succeeded=slm_succeeded,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_cost=model_cost,
            diagnostics={
                "features": features.to_dict(),
                "phase": phase.to_dict(),
                "root_invariant": (
                    sha256(state.root_goal.encode("utf-8")).hexdigest()
                    == state.root_goal_hash
                ),
                "rule_latency_ms": max(0.0, latency_ms - slm_latency_ms),
                "slm_latency_ms": slm_latency_ms,
                "slm_request_latency_ms": slm_request_latency_ms,
                "slm_startup_latency_ms": slm_startup_latency_ms,
                "slm_generation_latency_ms": slm_generation_latency_ms,
                "entity_overlap": overlap,
                "drift_streak": state.drift_streak,
            },
        )

    def _gate(
        self,
        context: TurnContext,
        state: _SessionState,
        goal: RetrievalGoal,
        features: FeatureSnapshot,
        phase: PhaseSnapshot,
    ) -> tuple[str, str]:
        if context.explicit_requirement is not None:
            return "REWRITE", "user_request"
        if phase.triggers:
            return "PATCH", "stuck_recovery"
        goal_error_types = {_error_signature(item) for item in goal.errors}
        feature_error_types = {
            _error_signature(item) for item in features.confirmed_errors
        }
        new_error = bool(feature_error_types - goal_error_types)
        top_frame_changed = bool(
            state.last_top_frame and features.confirmed_top_frame
            and state.last_top_frame != features.confirmed_top_frame
        )
        if new_error or top_frame_changed:
            return "REWRITE", "hypothesis_invalidated"
        if state.drift_streak >= self.entity_drift_steps:
            return "REWRITE", "goal_invalid"
        if features.test_verdict == "passed":
            return "PATCH", "evidence_satisfied"
        if goal.status == "blocked" and features.return_code == 0:
            return "PATCH", "blocker_resolved"
        if features.return_code is not None and features.return_code != 0:
            if goal.status != "blocked" or "failure diagnostics" not in goal.evidence_needed:
                return "PATCH", "execution_failed"

        current_entities = _feature_entities(features)
        existing_entities = _goal_entities(goal)
        if not existing_entities and current_entities:
            return "PATCH", "entities_discovered"
        if current_entities & existing_entities and current_entities - existing_entities:
            return "PATCH", "entities_extended"
        return "KEEP", "no_boundary"

    def _deterministic_patch(
        self,
        goal: RetrievalGoal,
        features: FeatureSnapshot,
        trigger: str,
        explicit_requirement: str | None,
    ) -> RetrievalGoal:
        values = asdict(goal)
        for name in _ENTITY_FIELDS:
            additions = (
                features.confirmed_errors if name == "errors" else getattr(features, name)
            )
            values[name] = _bounded_union(
                getattr(goal, name), additions, limit=_MAX_ENTITIES[name]
            )
        evidence = goal.evidence_needed
        status = goal.status
        constraints = goal.constraints
        if trigger == "stuck_recovery":
            status = "blocked"
            evidence = _bounded_union(
                evidence,
                ("recent successful state", "failed attempts",
                 "verification records", "root task anchor"),
                limit=24,
            )
        elif trigger in {"hypothesis_invalidated", "execution_failed"}:
            status = "blocked"
            evidence = _bounded_union(evidence, ("failure diagnostics",), limit=24)
        elif trigger == "evidence_satisfied":
            status = "done"
            evidence = tuple(
                item for item in evidence
                if "test" not in item.lower() and "verdict" not in item.lower()
            )
        elif trigger == "blocker_resolved":
            status = "active"
            evidence = tuple(item for item in evidence if item != "failure diagnostics")
        elif trigger == "goal_invalid":
            evidence = _bounded_union(evidence, ("goal boundary evidence",), limit=24)
        elif trigger == "user_request" and explicit_requirement is not None:
            constraints = _bounded_union(constraints, (explicit_requirement,), limit=24)
            evidence = _bounded_union(
                evidence, ("evidence for explicit user requirement",), limit=24
            )
        values.update(status=status, evidence_needed=evidence, constraints=constraints)
        return RetrievalGoal(**values)

    @staticmethod
    def _rewrite_features(
        features: FeatureSnapshot, context: TurnContext
    ) -> dict[str, object]:
        value = {
            "files": list(features.files),
            "symbols": list(features.symbols),
            "tests": list(features.tests),
            "errors": list(features.confirmed_errors),
            "top_frame": features.confirmed_top_frame,
            "return_code": features.return_code,
            "test_verdict": features.test_verdict,
            "has_diff": features.has_diff,
            "action_kind": features.action_kind,
        }
        if context.explicit_requirement is not None:
            value["constraints"] = [context.explicit_requirement]
        return value

    def _config(self) -> dict[str, object]:
        return {
            "entity_drift_steps": self.entity_drift_steps,
            "initialize_with_rewriter": self.initialize_with_rewriter,
            "max_description_chars": self.max_description_chars,
            "phase": asdict(self.phase_config),
            "rewriter": type(self.rewriter).__name__ if self.rewriter else None,
        }
