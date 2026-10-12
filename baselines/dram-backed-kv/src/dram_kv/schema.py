"""Versioned, runtime-independent records shared by every experiment stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.2.0"

GOAL_PHASES = frozenset({
    "exploration", "hypothesis", "implementation", "verification", "stuck",
})
GOAL_STATUSES = frozenset({"active", "blocked", "verifying", "done"})
GOAL_DECISIONS = frozenset({"BASELINE", "INITIALIZE", "KEEP", "PATCH", "REWRITE"})
MESSAGE_SOURCES = frozenset({"tool", "assistant", "user", "system"})


def _unique(values: Iterable[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique")
    return result


@dataclass(frozen=True)
class TurnContext:
    session_id: str
    instance_id: str
    turn_id: int
    task_objective: str
    previous_goal_text: str | None
    previous_action: str | None
    current_observation: str
    structured_state: Mapping[str, Any] = field(default_factory=dict)
    historical_turn_ids: tuple[int, ...] = ()
    historical_page_ids: tuple[str, ...] = ()
    hbm_resident_page_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    message_source: str = "tool"
    explicit_requirement: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.instance_id:
            raise ValueError("session_id and instance_id are required")
        if self.turn_id <= 0:
            raise ValueError("turn_id must be positive")
        if self.message_source not in MESSAGE_SOURCES:
            raise ValueError(f"unsupported message_source {self.message_source!r}")
        if self.explicit_requirement and self.message_source not in {"user", "system"}:
            raise ValueError("explicit_requirement requires a user/system message source")
        if self.explicit_requirement is not None and not self.explicit_requirement.strip():
            raise ValueError("explicit_requirement cannot be blank")
        if any(turn >= self.turn_id for turn in self.historical_turn_ids):
            raise ValueError("historical_turn_ids cannot contain current or future turns")
        object.__setattr__(self, "historical_page_ids", _unique(
            self.historical_page_ids, "historical_page_ids"
        ))
        object.__setattr__(self, "hbm_resident_page_ids", _unique(
            self.hbm_resident_page_ids, "hbm_resident_page_ids"
        ))
        unknown = set(self.hbm_resident_page_ids) - set(self.historical_page_ids)
        if unknown:
            raise ValueError(f"resident Pages are absent from history: {sorted(unknown)}")


@dataclass(frozen=True)
class RetrievalGoal:
    """Selector-side retrieval control signal, never an execution plan."""

    description: str
    phase_hint: str = "exploration"
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    evidence_needed: tuple[str, ...] = ()
    status: str = "active"
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("retrieval Goal description is required")
        if self.phase_hint not in GOAL_PHASES - {"stuck"}:
            raise ValueError(f"unsupported retrieval Goal phase {self.phase_hint!r}")
        if self.status not in GOAL_STATUSES:
            raise ValueError(f"unsupported retrieval Goal status {self.status!r}")
        for name in (
            "files", "symbols", "tests", "errors", "evidence_needed", "constraints",
        ):
            values = getattr(self, name)
            object.__setattr__(self, name, _unique(values, f"retrieval_goal.{name}"))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"retrieval_goal.{name} must contain non-empty strings")


@dataclass(frozen=True)
class GoalPolicyHint:
    """Advisory working-set policy selected by the phase detector."""

    phase: str
    active_token_budget: int
    slack: str
    diversity_weight: float
    structure_weight: float
    priorities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in GOAL_PHASES:
            raise ValueError(f"unsupported policy phase {self.phase!r}")
        if self.active_token_budget <= 0:
            raise ValueError("active_token_budget must be positive")
        if self.slack not in {"low", "medium", "high"}:
            raise ValueError(f"unsupported slack setting {self.slack!r}")
        for name in ("diversity_weight", "structure_weight"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        object.__setattr__(self, "priorities", _unique(self.priorities, "policy priorities"))
        if any(not isinstance(value, str) or not value.strip() for value in self.priorities):
            raise ValueError("policy priorities must contain non-empty strings")


@dataclass(frozen=True)
class GoalRecord:
    session_id: str
    turn_id: int
    constructor: str
    constructor_config: Mapping[str, Any]
    text: str
    reused_previous: bool = False
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_cost: float = 0.0
    schema_version: str = SCHEMA_VERSION
    root_goal: str | None = None
    retrieval_goal: RetrievalGoal | None = None
    gate_decision: str = "INITIALIZE"
    applied_decision: str = "INITIALIZE"
    triggers: tuple[str, ...] = ()
    changed_fields: tuple[str, ...] = ()
    policy_hint: GoalPolicyHint | None = None
    slm_attempted: bool = False
    slm_succeeded: bool = False
    fallback_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.turn_id <= 0 or not self.constructor:
            raise ValueError("GoalRecord requires a positive turn and constructor")
        if self.latency_ms < 0 or self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("Goal costs cannot be negative")
        if self.model_cost < 0:
            raise ValueError("model_cost cannot be negative")
        if self.gate_decision not in GOAL_DECISIONS:
            raise ValueError(f"unsupported gate_decision {self.gate_decision!r}")
        if self.applied_decision not in GOAL_DECISIONS:
            raise ValueError(f"unsupported applied_decision {self.applied_decision!r}")
        if self.slm_succeeded and not self.slm_attempted:
            raise ValueError("SLM cannot succeed without an attempt")
        if self.root_goal is not None and not self.root_goal.strip():
            raise ValueError("root_goal cannot be blank")
        object.__setattr__(self, "triggers", _unique(self.triggers, "Goal triggers"))
        object.__setattr__(self, "changed_fields", _unique(
            self.changed_fields, "Goal changed_fields"
        ))

        retrieval = self.retrieval_goal
        if retrieval is None:
            retrieval = RetrievalGoal(
                description=self.text,
                files=tuple(self.files),
                symbols=tuple(self.symbols),
                tests=tuple(self.tests),
                errors=tuple(self.errors),
                constraints=tuple(self.constraints),
            )
            object.__setattr__(self, "retrieval_goal", retrieval)
        else:
            if self.text and self.text != retrieval.description:
                raise ValueError("GoalRecord.text must match retrieval_goal.description")
            for name in ("files", "symbols", "tests", "errors", "constraints"):
                legacy = tuple(getattr(self, name))
                canonical = tuple(getattr(retrieval, name))
                if legacy and legacy != canonical:
                    raise ValueError(f"GoalRecord.{name} must match retrieval_goal.{name}")
                object.__setattr__(self, name, canonical)
            object.__setattr__(self, "text", retrieval.description)

    @property
    def root_goal_hash(self) -> str:
        value = self.root_goal if self.root_goal is not None else ""
        return sha256(value.encode("utf-8")).hexdigest()

    @property
    def selector_payload(self) -> Mapping[str, Any]:
        """Every Goal field that is permitted to influence a Selector."""

        return {
            "schema_version": self.schema_version,
            "root_goal": self.root_goal,
            "retrieval_goal": asdict(self.retrieval_goal),
            "policy_hint": asdict(self.policy_hint) if self.policy_hint else None,
        }

    @staticmethod
    def _stable_hash(value: Mapping[str, Any]) -> str:
        stable = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(stable.encode("utf-8")).hexdigest()

    @property
    def retrieval_goal_hash(self) -> str:
        return self._stable_hash({
            "schema_version": self.schema_version,
            "root_goal": self.root_goal,
            "retrieval_goal": asdict(self.retrieval_goal),
        })

    @property
    def policy_hash(self) -> str:
        return self._stable_hash({
            "schema_version": self.schema_version,
            "policy_hint": asdict(self.policy_hint) if self.policy_hint else None,
        })

    @property
    def version_hash(self) -> str:
        return self._stable_hash(self.selector_payload)


@dataclass(frozen=True)
class PageRecord:
    page_id: str
    turn_id: int
    role: str
    token_start: int
    token_end: int
    logical_position_start: int
    logical_position_end: int
    byte_size: int
    mandatory: bool = False
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    summary: str | None = None
    embedding_ref: str | None = None
    dram_location: str | None = None
    hbm_resident: bool = False
    kv_provenance: str = "exact"
    page_vector_ref: str | None = None
    lexical_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.page_id:
            raise ValueError("page_id is required")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("Page token span must be non-empty")
        if self.logical_position_start < 0:
            raise ValueError("logical positions cannot be negative")
        if self.logical_position_end - self.logical_position_start != self.token_count:
            raise ValueError("logical and token spans must have the same length")
        if self.byte_size < 0:
            raise ValueError("byte_size cannot be negative")
        object.__setattr__(
            self,
            "lexical_handles",
            _unique(self.lexical_handles, "Page lexical_handles"),
        )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.lexical_handles
        ):
            raise ValueError("Page lexical_handles must contain non-empty strings")
        if self.page_vector_ref is not None:
            prefix = "page-vector-v2:"
            digest = self.page_vector_ref.removeprefix(prefix)
            if (
                not self.page_vector_ref.startswith(prefix)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("page_vector_ref must identify a Selector V2 vector")
        if self.kv_provenance not in {
            "exact", "selection_conditioned", "retained", "reactivated"
        }:
            raise ValueError(f"unsupported KV provenance {self.kv_provenance!r}")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class PageCatalog:
    session_id: str
    pages: tuple[PageRecord, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        ids = [page.page_id for page in self.pages]
        if len(ids) != len(set(ids)):
            raise ValueError("Page IDs must be stable and unique")
        if list(self.pages) != sorted(
            self.pages, key=lambda page: (page.logical_position_start, page.page_id)
        ):
            raise ValueError("PageCatalog must be in original logical-position order")
        for previous, current in zip(self.pages, self.pages[1:]):
            if previous.logical_position_end > current.logical_position_start:
                raise ValueError("Page logical spans cannot overlap")

    def by_id(self) -> dict[str, PageRecord]:
        return {page.page_id: page for page in self.pages}

    def historical_for_turn(self, turn_id: int) -> tuple[PageRecord, ...]:
        return tuple(page for page in self.pages if page.turn_id < turn_id)


@dataclass(frozen=True)
class SelectionReason:
    source: str
    score: float
    detail: str = ""


@dataclass(frozen=True)
class SelectionPlan:
    session_id: str
    turn_id: int
    goal_hash: str
    selector: str
    selector_config: Mapping[str, Any]
    ordered_page_ids: tuple[str, ...]
    reasons: Mapping[str, SelectionReason]
    mandatory_page_ids: tuple[str, ...]
    recent_page_ids: tuple[str, ...]
    retrieved_page_ids: tuple[str, ...]
    active_token_budget: int | None = None
    active_page_budget: int | None = None
    active_byte_budget: int | None = None
    confidence: float | None = None
    latency_ms: float = 0.0
    added_page_ids: tuple[str, ...] = ()
    retained_page_ids: tuple[str, ...] = ()
    removed_page_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    standby_page_ids: tuple[str, ...] = ()
    resident_page_budget: int | None = None

    def __post_init__(self) -> None:
        selected = _unique(self.ordered_page_ids, "ordered_page_ids")
        for name in ("mandatory_page_ids", "recent_page_ids", "retrieved_page_ids"):
            values = _unique(getattr(self, name), name)
            unknown = set(values) - set(selected)
            if unknown:
                raise ValueError(f"{name} contains unselected Pages: {sorted(unknown)}")
        standby = _unique(self.standby_page_ids, "standby_page_ids")
        overlap = set(standby) & set(selected)
        if overlap:
            raise ValueError(f"standby Pages cannot be attention-active: {sorted(overlap)}")
        if self.resident_page_budget is not None:
            if self.resident_page_budget <= 0:
                raise ValueError("resident_page_budget must be positive")
            if len(selected) + len(standby) > self.resident_page_budget:
                raise ValueError("plan exceeds resident_page_budget")
        if set(self.reasons) != set(selected):
            raise ValueError("Selection reasons must cover exactly the active Pages")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")

    def validate_against(self, context: TurnContext, catalog: PageCatalog) -> None:
        if self.session_id != context.session_id or catalog.session_id != context.session_id:
            raise ValueError("session mismatch between plan, context, and catalog")
        if self.turn_id != context.turn_id:
            raise ValueError("plan and context turn mismatch")
        pages = catalog.by_id()
        unknown = set(self.ordered_page_ids) - set(context.historical_page_ids)
        if unknown:
            raise ValueError(f"plan selects Pages outside sealed history: {sorted(unknown)}")
        unknown_standby = set(self.standby_page_ids) - set(context.historical_page_ids)
        if unknown_standby:
            raise ValueError(
                f"plan retains standby Pages outside sealed history: {sorted(unknown_standby)}"
            )
        ordered = sorted(
            self.ordered_page_ids, key=lambda page_id: pages[page_id].logical_position_start
        )
        if list(self.ordered_page_ids) != ordered:
            raise ValueError("active Pages must preserve original logical order")
        ordered_standby = sorted(
            self.standby_page_ids,
            key=lambda page_id: pages[page_id].logical_position_start,
        )
        if list(self.standby_page_ids) != ordered_standby:
            raise ValueError("standby Pages must preserve original logical order")
        selected_pages = [pages[page_id] for page_id in self.ordered_page_ids]
        if self.active_page_budget is not None and len(selected_pages) > self.active_page_budget:
            raise ValueError("plan exceeds active_page_budget")
        tokens = sum(page.token_count for page in selected_pages)
        if self.active_token_budget is not None and tokens > self.active_token_budget:
            raise ValueError("plan exceeds active_token_budget")
        byte_size = sum(page.byte_size for page in selected_pages)
        if self.active_byte_budget is not None and byte_size > self.active_byte_budget:
            raise ValueError("plan exceeds active_byte_budget")


def record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a record to JSON-compatible data without losing schema fields."""
    return asdict(record)
