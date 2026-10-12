"""Dependency-free orchestration for fixed-token trajectory replay.

The replay boundary is deliberately strict::

    plan -> restore/bind -> formal prefill+generation -> seal new Pages

The module never tokenizes text and never imports a model framework.  Callers
must supply canonical rendered token spans and model-specific callbacks.  This
keeps message-to-token mapping out of the experiment policy and makes a replay
artifact sufficient to audit exactly which logical positions were available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import floor
from typing import Any, Callable, Generic, Literal, Mapping, Sequence, TypeVar

from ..paging import TokenSegment, active_set_delta, build_page_catalog
from ..schema import (
    GoalRecord,
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


PayloadT = TypeVar("PayloadT")
PreparedT = TypeVar("PreparedT")
ModelResultT = TypeVar("ModelResultT")
BudgetUnit = Literal["pages", "tokens", "ratio"]
ReplayPolicy = Literal["dense", "full_active", "oracle", "goal"]


@dataclass(frozen=True)
class RenderedSpan:
    """One already-tokenized contiguous part of a recorded model call."""

    role: str
    token_ids: tuple[int, ...]
    mandatory: bool = False
    source_message_ids: tuple[int, ...] = ()
    summary: str | None = None
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.role or not self.token_ids:
            raise ValueError("rendered span requires a role and at least one token")
        if len(set(self.source_message_ids)) != len(self.source_message_ids):
            raise ValueError("source_message_ids must be unique")


@dataclass(frozen=True)
class RecordedTurn:
    """Fixed input/output tokens for one recorded assistant turn.

    ``prefill_spans`` are passed to the formal prefill. ``generated_spans`` are
    the recorded continuation (or teacher-forced equivalent).  All spans become
    sealed history only after the model callback completes.
    """

    turn_id: int
    prefill_spans: tuple[RenderedSpan, ...]
    generated_spans: tuple[RenderedSpan, ...]
    action_summary: str = ""
    action_text: str = ""
    observation_text: str = ""
    return_code: int | None = None
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if self.turn_id <= 0:
            raise ValueError("turn_id must be positive")
        if not self.prefill_spans:
            raise ValueError("each turn requires a non-empty formal prefill")

    @property
    def prefill_token_count(self) -> int:
        return sum(len(span.token_ids) for span in self.prefill_spans)

    @property
    def appended_token_count(self) -> int:
        return self.prefill_token_count + sum(
            len(span.token_ids) for span in self.generated_spans
        )


@dataclass(frozen=True)
class RecordedTrajectory:
    session_id: str
    instance_id: str
    task_objective: str
    turns: tuple[RecordedTurn, ...]

    def __post_init__(self) -> None:
        if not self.session_id or not self.instance_id or not self.turns:
            raise ValueError("trajectory identifiers and turns are required")
        actual = tuple(turn.turn_id for turn in self.turns)
        expected = tuple(range(1, len(self.turns) + 1))
        if actual != expected:
            raise ValueError(f"turn IDs must be contiguous: expected={expected} actual={actual}")

    def catalog(self, *, page_size: int, bytes_per_token: int) -> PageCatalog:
        """Build immutable Page records without retokenizing any content."""

        cursor = 0
        segments: list[TokenSegment] = []
        for turn in self.turns:
            for span in turn.prefill_spans + turn.generated_spans:
                end = cursor + len(span.token_ids)
                segments.append(TokenSegment(
                    turn_id=turn.turn_id,
                    role=span.role,
                    token_start=cursor,
                    token_end=end,
                    mandatory=span.mandatory,
                    files=span.files,
                    symbols=span.symbols,
                    tests=span.tests,
                    errors=span.errors,
                    summary=span.summary,
                ))
                cursor = end
        return build_page_catalog(
            self.session_id,
            segments,
            page_size=page_size,
            bytes_per_token=bytes_per_token,
        )

    def page_source_message_ids(
        self, *, page_size: int, bytes_per_token: int
    ) -> Mapping[str, tuple[int, ...]]:
        """Return the explicit annotation-message provenance of every Page."""

        catalog = self.catalog(page_size=page_size, bytes_per_token=bytes_per_token)
        result: dict[str, tuple[int, ...]] = {}
        cursor = 0
        page_index = 0
        for turn in self.turns:
            for span in turn.prefill_spans + turn.generated_spans:
                end = cursor + len(span.token_ids)
                while page_index < len(catalog.pages):
                    page = catalog.pages[page_index]
                    if page.logical_position_start >= end:
                        break
                    if page.logical_position_start < cursor or page.logical_position_end > end:
                        raise RuntimeError("Page crosses a rendered source-message boundary")
                    result[page.page_id] = span.source_message_ids
                    page_index += 1
                cursor = end
        if page_index != len(catalog.pages):
            raise RuntimeError("not every Page received source-message provenance")
        return result

    def physical_page_layout(
        self, *, page_size: int, bytes_per_token: int
    ) -> tuple[PageCatalog, Mapping[str, tuple[int, ...]]]:
        """Build true fixed-size Pages that may cross rendered-span boundaries.

        ``catalog()`` intentionally preserves source-span boundaries for the
        older replay engine. Selector V2 instead needs physical 16-token Pages:
        a Page remains open across message and turn boundaries and becomes
        eligible only after its final token has been produced. The unfilled
        trajectory tail is not sealed.
        """

        if page_size <= 0 or bytes_per_token < 0:
            raise ValueError("page_size must be positive and bytes_per_token non-negative")
        buckets: dict[int, dict[str, Any]] = {}
        cursor = 0
        for turn in self.turns:
            for span in turn.prefill_spans + turn.generated_spans:
                span_start = cursor
                span_end = cursor + len(span.token_ids)
                first_page = span_start // page_size
                last_page = (span_end - 1) // page_size
                for page_index in range(first_page, last_page + 1):
                    overlap_start = max(span_start, page_index * page_size)
                    overlap_end = min(span_end, (page_index + 1) * page_size)
                    bucket = buckets.setdefault(page_index, {
                        "covered": 0,
                        "turn_id": turn.turn_id,
                        "roles": [],
                        "mandatory": False,
                        "files": [],
                        "symbols": [],
                        "tests": [],
                        "errors": [],
                        "lexical_handles": [],
                        "source_message_ids": [],
                    })
                    bucket["covered"] += overlap_end - overlap_start
                    bucket["turn_id"] = turn.turn_id
                    if span.role not in bucket["roles"]:
                        bucket["roles"].append(span.role)
                    bucket["mandatory"] = bucket["mandatory"] or span.mandatory
                    for field_name in ("files", "symbols", "tests", "errors"):
                        for value in getattr(span, field_name):
                            if value not in bucket[field_name]:
                                bucket[field_name].append(value)
                            if value not in bucket["lexical_handles"]:
                                bucket["lexical_handles"].append(value)
                    for message_id in span.source_message_ids:
                        if message_id not in bucket["source_message_ids"]:
                            bucket["source_message_ids"].append(message_id)
                cursor = span_end

        pages: list[PageRecord] = []
        sources: dict[str, tuple[int, ...]] = {}
        full_page_count = cursor // page_size
        for page_index in range(full_page_count):
            bucket = buckets[page_index]
            if bucket["covered"] != page_size:
                raise RuntimeError("physical Page token coverage is not contiguous")
            start = page_index * page_size
            end = start + page_size
            page_id = f"p{page_index:08d}:{start}-{end}"
            roles = tuple(bucket["roles"])
            pages.append(PageRecord(
                page_id=page_id,
                turn_id=int(bucket["turn_id"]),
                role=roles[0] if len(roles) == 1 else "mixed",
                token_start=start,
                token_end=end,
                logical_position_start=start,
                logical_position_end=end,
                byte_size=page_size * bytes_per_token,
                mandatory=bool(bucket["mandatory"]),
                files=tuple(bucket["files"]),
                symbols=tuple(bucket["symbols"]),
                tests=tuple(bucket["tests"]),
                errors=tuple(bucket["errors"]),
                lexical_handles=tuple(bucket["lexical_handles"]),
            ))
            sources[page_id] = tuple(bucket["source_message_ids"])
        return PageCatalog(self.session_id, tuple(pages)), sources

    def turn_bounds(self, turn_id: int) -> tuple[int, int, int]:
        """Return ``(history_end, prefill_end, generated_end)`` positions."""

        if turn_id <= 0 or turn_id > len(self.turns):
            raise KeyError(f"unknown recorded turn {turn_id}")
        history_end = sum(turn.appended_token_count for turn in self.turns[: turn_id - 1])
        turn = self.turns[turn_id - 1]
        prefill_end = history_end + turn.prefill_token_count
        return history_end, prefill_end, history_end + turn.appended_token_count

    def token_ids_through(self, turn_id: int) -> tuple[int, ...]:
        values: list[int] = []
        for turn in self.turns[:turn_id]:
            for span in turn.prefill_spans + turn.generated_spans:
                values.extend(span.token_ids)
        return tuple(values)


@dataclass(frozen=True)
class ActiveBudget:
    """A single unambiguous cap. Ratio is of all historical tokens."""

    unit: BudgetUnit
    value: int | float

    def __post_init__(self) -> None:
        if self.unit == "ratio":
            if not isinstance(self.value, (int, float)) or not 0 < self.value <= 1:
                raise ValueError("ratio budget must be in (0, 1]")
        elif self.unit in {"pages", "tokens"}:
            if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value < 0:
                raise ValueError(f"{self.unit} budget must be a non-negative integer")
        else:
            raise ValueError(f"unsupported budget unit {self.unit!r}")


@dataclass(frozen=True)
class OracleEvidence:
    """Offline evidence turns copied from an annotation, never inferred."""

    selected_older_turn_ids: Mapping[int, tuple[int, ...]]
    recent_turns: int = 2
    mandatory_message_ids: tuple[int, ...] = (0, 1)
    turn_message_ids: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    boundary_input_message_ids: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.recent_turns < 0:
            raise ValueError("recent_turns cannot be negative")
        for target, evidence in self.selected_older_turn_ids.items():
            if target <= 0 or any(turn <= 0 or turn >= target for turn in evidence):
                raise ValueError("oracle evidence must reference positive past turns")
        if len(set(self.mandatory_message_ids)) != len(self.mandatory_message_ids):
            raise ValueError("mandatory_message_ids must be unique")
        for turn_id, message_ids in self.turn_message_ids.items():
            if turn_id <= 0 or len(set(message_ids)) != len(message_ids):
                raise ValueError("Oracle turn message mapping is invalid")
        for turn_id, message_ids in self.boundary_input_message_ids.items():
            if turn_id <= 0 or len(set(message_ids)) != len(message_ids):
                raise ValueError("Oracle boundary input mapping is invalid")

    def message_sets_for_turn(self, target_turn: int) -> tuple[set[int], set[int], set[int]]:
        """Map semantic completed turns to canonical action/observation messages.

        Message ``2*t-1`` is the current live input for target turn ``t`` and
        is explicitly excluded even if it appears in the recent window.
        """

        cutoff = max(1, target_turn - self.recent_turns)
        if self.turn_message_ids:
            live = set(self.boundary_input_message_ids.get(target_turn, ()))
            recent = {
                message_id
                for turn_id in range(cutoff, target_turn)
                for message_id in self.turn_message_ids.get(turn_id, ())
                if message_id not in live
            }
            older = {
                message_id
                for turn_id in self.selected_older_turn_ids.get(target_turn, ())
                for message_id in self.turn_message_ids.get(turn_id, ())
            }
            return set(self.mandatory_message_ids), recent, older

        live_message = 2 * target_turn - 1
        recent = {
            message_id
            for turn_id in range(cutoff, target_turn)
            for message_id in (2 * turn_id, 2 * turn_id + 1)
            if message_id != live_message
        }
        older = {
            message_id
            for turn_id in self.selected_older_turn_ids.get(target_turn, ())
            for message_id in (2 * turn_id, 2 * turn_id + 1)
        }
        return set(self.mandatory_message_ids), recent, older


def oracle_evidence_from_annotation(value: Mapping[str, Any]) -> OracleEvidence:
    """Read old or hash-bound v2 semantic evidence without turn arithmetic."""

    schema_name = value.get("schema_name")
    if schema_name not in {"kv_drift_turn_oracle", "dram_kv_recorded_oracle"}:
        raise ValueError("unsupported Oracle annotation schema")
    policy = value.get(
        "retrieval_policy" if schema_name == "dram_kv_recorded_oracle" else "oracle_policy"
    )
    turns = value.get("turns")
    if not isinstance(policy, Mapping) or not isinstance(turns, Sequence):
        raise ValueError("Oracle annotation lacks oracle_policy or turns")
    selected: dict[int, tuple[int, ...]] = {}
    for item in turns:
        if not isinstance(item, Mapping):
            raise ValueError("Oracle annotation turn must be an object")
        target = int(item["turn_id"])
        retrieval = item.get("retrieval")
        raw_selected = (
            retrieval.get("selected_older_turn_ids", ())
            if isinstance(retrieval, Mapping)
            else item.get("selected_older_turn_ids", ())
        )
        selected[target] = tuple(int(turn) for turn in raw_selected)

    turn_messages: dict[int, tuple[int, ...]] = {}
    boundary_inputs: dict[int, tuple[int, ...]] = {}
    if schema_name == "dram_kv_recorded_oracle":
        message_map = value.get("message_map")
        boundaries = message_map.get("boundaries") if isinstance(message_map, Mapping) else None
        if not isinstance(boundaries, Sequence):
            raise ValueError("recorded Oracle lacks canonical message-map boundaries")
        for boundary in boundaries:
            if not isinstance(boundary, Mapping):
                raise ValueError("message-map boundary must be an object")
            turn_id = int(boundary["turn_id"])
            assistant_id = int(boundary["assistant_message_id"])
            result_ids = tuple(int(item) for item in boundary.get("result_message_ids", ()))
            turn_messages[turn_id] = (assistant_id, *result_ids)
            boundary_inputs[turn_id] = tuple(
                int(item) for item in boundary.get("input_message_ids", ())
            )
    return OracleEvidence(
        selected,
        recent_turns=int(policy.get("recent_complete_turn_window", 2)),
        mandatory_message_ids=tuple(int(item) for item in policy.get("mandatory_message_ids", ())),
        turn_message_ids=turn_messages,
        boundary_input_message_ids=boundary_inputs,
    )


@dataclass(frozen=True)
class BudgetDecision:
    policy: ReplayPolicy
    budget_unit: str | None
    budget_value: int | float | None
    capacity: int | None
    historical_pages: int
    historical_tokens: int
    selected_pages: int
    selected_tokens: int
    omitted_recent_page_ids: tuple[str, ...] = ()
    omitted_oracle_page_ids: tuple[str, ...] = ()
    filler_page_ids: tuple[str, ...] = ()
    unused_capacity: int | None = None
    plan_selector: str | None = None


class BudgetExceededError(ValueError):
    """Raised when indivisible mandatory Pages cannot fit the declared cap."""


def build_selection_plan(
    context: TurnContext,
    goal: GoalRecord,
    catalog: PageCatalog,
    *,
    policy: ReplayPolicy,
    evidence: OracleEvidence | None = None,
    budget: ActiveBudget | None = None,
    previous_active: Sequence[str] = (),
    ever_selected: Sequence[str] = (),
    page_source_message_ids: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[SelectionPlan, BudgetDecision]:
    """Create dense/full-active or budgeted Oracle plans.

    Pages are indivisible. Mandatory Pages are admitted first and cause an
    explicit :class:`BudgetExceededError` when they cannot fit. Recent safety
    Pages are considered newest-first, then Oracle Pages in annotation order;
    omitted safety/evidence Pages are reported in ``BudgetDecision``. Output is
    always restored to original logical order.
    """

    if context.session_id != catalog.session_id or goal.session_id != context.session_id:
        raise ValueError("session mismatch while building replay plan")
    if goal.turn_id != context.turn_id:
        raise ValueError("goal/context turn mismatch")
    history = catalog.historical_for_turn(context.turn_id)
    expected_ids = tuple(page.page_id for page in history)
    if context.historical_page_ids != expected_ids:
        raise ValueError("context history must exactly match sealed catalog history")
    total_tokens = sum(page.token_count for page in history)

    if policy in {"dense", "full_active"}:
        if budget is not None:
            raise ValueError(f"{policy} does not accept an active budget")
        selected = history
        sources = {page.page_id: policy for page in selected}
        omitted_recent: tuple[str, ...] = ()
        omitted_oracle: tuple[str, ...] = ()
        filler_ids: tuple[str, ...] = ()
        capacity = None
    elif policy == "oracle":
        if evidence is None or budget is None:
            raise ValueError("oracle policy requires evidence and an active budget")
        if page_source_message_ids is None:
            raise ValueError("oracle policy requires explicit Page source-message mapping")
        missing_sources = set(expected_ids) - set(page_source_message_ids)
        if missing_sources:
            raise ValueError(f"historical Pages lack source-message mapping: {sorted(missing_sources)}")
        empty_sources = tuple(
            page_id for page_id in expected_ids if not page_source_message_ids[page_id]
        )
        if empty_sources:
            raise ValueError(
                f"historical Pages have empty source-message mapping: {list(empty_sources)}"
            )
        live_message = 2 * context.turn_id - 1
        leaked_live = tuple(
            page_id for page_id in expected_ids
            if live_message in page_source_message_ids[page_id]
        )
        if leaked_live:
            raise ValueError(
                "current live-input message appears in sealed history: "
                f"message_id={live_message} pages={list(leaked_live)}"
            )
        selected, sources, omitted_recent, omitted_oracle, filler_ids, capacity = _budgeted_oracle(
            context.turn_id, history, evidence, budget, page_source_message_ids
        )
    else:
        raise ValueError(f"unsupported replay policy {policy!r}")

    ordered_ids = tuple(page.page_id for page in selected)
    mandatory_ids = tuple(
        page.page_id for page in selected
        if page.mandatory or sources[page.page_id] == "mandatory"
    )
    recent_ids = tuple(page.page_id for page in selected if sources[page.page_id] == "recent")
    retrieved_ids = tuple(
        page.page_id for page in selected
        if sources[page.page_id] in {"oracle_evidence", "budget_filler", "dense", "full_active"}
    )
    reasons = {
        page.page_id: SelectionReason(sources[page.page_id], 1.0)
        for page in selected
    }
    delta = active_set_delta(ordered_ids, previous_active, ever_selected)
    token_budget = None
    page_budget = None
    if policy == "oracle" and budget is not None:
        if budget.unit == "pages":
            page_budget = int(capacity)
        else:
            token_budget = int(capacity)
    plan = SelectionPlan(
        session_id=context.session_id,
        turn_id=context.turn_id,
        goal_hash=goal.version_hash,
        selector=f"recorded_{policy}",
        selector_config={
            "policy": policy,
            "future_leakage": policy == "oracle",
            "budget_unit": budget.unit if budget else None,
            "budget_value": budget.value if budget else None,
            "page_indivisible": True,
        },
        ordered_page_ids=ordered_ids,
        reasons=reasons,
        mandatory_page_ids=mandatory_ids,
        recent_page_ids=recent_ids,
        retrieved_page_ids=retrieved_ids,
        active_token_budget=token_budget,
        active_page_budget=page_budget,
        confidence=1.0,
        added_page_ids=delta.added,
        retained_page_ids=delta.retained,
        removed_page_ids=delta.removed,
    )
    plan.validate_against(context, catalog)
    decision = BudgetDecision(
        policy=policy,
        budget_unit=budget.unit if budget else None,
        budget_value=budget.value if budget else None,
        capacity=capacity,
        historical_pages=len(history),
        historical_tokens=total_tokens,
        selected_pages=len(selected),
        selected_tokens=sum(page.token_count for page in selected),
        omitted_recent_page_ids=omitted_recent,
        omitted_oracle_page_ids=omitted_oracle,
        filler_page_ids=filler_ids,
        unused_capacity=(capacity - (len(selected) if budget and budget.unit == "pages" else sum(page.token_count for page in selected))) if capacity is not None else None,
    )
    return plan, decision


def _budgeted_oracle(
    target_turn: int,
    history: tuple[PageRecord, ...],
    evidence: OracleEvidence,
    budget: ActiveBudget,
    page_sources: Mapping[str, tuple[int, ...]],
) -> tuple[
    tuple[PageRecord, ...], dict[str, str], tuple[str, ...], tuple[str, ...], tuple[str, ...], int
]:
    total_tokens = sum(page.token_count for page in history)
    capacity = (
        floor(total_tokens * float(budget.value))
        if budget.unit == "ratio"
        else int(budget.value)
    )
    selected: set[str] = set()
    sources: dict[str, str] = {}
    used = 0

    def cost(page: PageRecord) -> int:
        return 1 if budget.unit == "pages" else page.token_count

    mandatory_messages, recent_messages, older_messages = evidence.message_sets_for_turn(target_turn)

    def sourced_from(page: PageRecord, message_ids: set[int]) -> bool:
        return bool(set(page_sources[page.page_id]) & message_ids)

    mandatory = tuple(
        page for page in history
        if page.mandatory or sourced_from(page, mandatory_messages)
    )
    mandatory_cost = sum(cost(page) for page in mandatory)
    if mandatory_cost > capacity:
        raise BudgetExceededError(
            "mandatory Pages exceed active budget: "
            f"required={mandatory_cost} capacity={capacity} unit={budget.unit}"
        )
    for page in mandatory:
        selected.add(page.page_id)
        sources[page.page_id] = "mandatory"
        used += cost(page)

    recent = tuple(
        page for page in reversed(history)
        if page.page_id not in selected and sourced_from(page, recent_messages)
    )
    omitted_recent: list[str] = []
    for page in recent:
        if page.page_id in selected:
            continue
        if used + cost(page) <= capacity:
            selected.add(page.page_id)
            sources[page.page_id] = "recent"
            used += cost(page)
        else:
            omitted_recent.append(page.page_id)

    candidates = tuple(
        page for page in history
        if page.page_id not in selected and sourced_from(page, older_messages)
    )
    omitted_oracle: list[str] = []
    for page in candidates:
        if page.page_id in selected:
            continue
        if used + cost(page) <= capacity:
            selected.add(page.page_id)
            sources[page.page_id] = "oracle_evidence"
            used += cost(page)
        else:
            omitted_oracle.append(page.page_id)

    # Fill the declared capacity with newest inactive history. This isolates
    # active-budget effects even after all Oracle evidence already fits.
    filler: list[str] = []
    for page in reversed(history):
        if page.page_id in selected:
            continue
        if used + cost(page) <= capacity:
            selected.add(page.page_id)
            sources[page.page_id] = "budget_filler"
            filler.append(page.page_id)
            used += cost(page)

    ordered = tuple(page for page in history if page.page_id in selected)
    return (
        ordered, sources, tuple(omitted_recent), tuple(omitted_oracle),
        tuple(filler), capacity,
    )


@dataclass(frozen=True)
class RecordedTurnArtifact:
    session_id: str
    instance_id: str
    turn_id: int
    history_token_end: int
    prefill_token_end: int
    generated_token_end: int
    plan: Mapping[str, Any]
    budget: Mapping[str, Any]
    preparation: Mapping[str, Any]
    sealed_page_ids: tuple[str, ...]
    backend_metrics: Mapping[str, Any]
    event_order: tuple[str, ...]


class RecordedTurnLoop(Generic[PayloadT]):
    """Stateful turn loop enforcing restore/bind before model execution."""

    def __init__(self, trajectory: RecordedTrajectory, catalog: PageCatalog, backend: Any):
        if catalog.session_id != trajectory.session_id:
            raise ValueError("trajectory/catalog session mismatch")
        self.trajectory = trajectory
        self.catalog = catalog
        self.backend = backend
        self._next_turn = 1
        self._sealed: set[str] = set()

    def context_for_turn(self, turn_id: int, *, resident_page_ids: Sequence[str] = ()) -> TurnContext:
        history = self.catalog.historical_for_turn(turn_id)
        expected_sealed = {page.page_id for page in history}
        if expected_sealed != self._sealed:
            missing = tuple(sorted(expected_sealed - self._sealed))
            extra = tuple(sorted(self._sealed - expected_sealed))
            raise RuntimeError(f"sealed history mismatch: missing={missing} extra={extra}")
        resident = tuple(page.page_id for page in history if page.page_id in set(resident_page_ids))
        previous = self.trajectory.turns[turn_id - 2] if turn_id > 1 else None
        current = self.trajectory.turns[turn_id - 1]
        previous_action = None
        if previous is not None:
            previous_action = previous.action_text or previous.action_summary or None
        return TurnContext(
            session_id=self.trajectory.session_id,
            instance_id=self.trajectory.instance_id,
            turn_id=turn_id,
            task_objective=self.trajectory.task_objective,
            previous_goal_text=None,
            previous_action=previous_action,
            current_observation=current.observation_text,
            structured_state={
                "return_code": current.return_code,
                "tool_name": current.tool_name,
            },
            historical_turn_ids=tuple(range(1, turn_id)),
            historical_page_ids=tuple(page.page_id for page in history),
            hbm_resident_page_ids=resident,
            message_source="tool" if turn_id > 1 else "user",
        )

    def run_turn(
        self,
        context: TurnContext,
        plan: SelectionPlan,
        decision: BudgetDecision,
        *,
        restore_and_bind: Callable[[TurnContext, PageCatalog, SelectionPlan], PreparedT],
        prefill_and_generate: Callable[[PreparedT, RecordedTurn], ModelResultT],
        extract_payload: Callable[[ModelResultT, PageRecord], PayloadT],
        keep_resident: bool = False,
    ) -> RecordedTurnArtifact:
        """Execute one recorded turn and seal its newly produced exact KV."""

        if context.turn_id != self._next_turn:
            raise RuntimeError(
                f"replay expected turn {self._next_turn}, got {context.turn_id}"
            )
        expected = self.context_for_turn(context.turn_id, resident_page_ids=context.hbm_resident_page_ids)
        if context != expected:
            raise ValueError("context differs from deterministic recorded trajectory context")
        plan.validate_against(context, self.catalog)
        selected_tokens = sum(
            self.catalog.by_id()[page_id].token_count
            for page_id in plan.ordered_page_ids
        )
        if (
            decision.selected_pages != len(plan.ordered_page_ids)
            or decision.selected_tokens != selected_tokens
            or plan.selector != (decision.plan_selector or f"recorded_{decision.policy}")
        ):
            raise ValueError("budget decision does not describe the supplied plan")
        turn = self.trajectory.turns[context.turn_id - 1]
        history_end, prefill_end, generated_end = self.trajectory.turn_bounds(context.turn_id)
        events: list[str] = ["plan"]
        prepared = restore_and_bind(context, self.catalog, plan)
        events.append("restore_bind_complete")
        model_result = prefill_and_generate(prepared, turn)
        events.append("prefill_generation_complete")

        current_pages = tuple(
            page for page in self.catalog.pages if page.turn_id == context.turn_id
        )
        metrics: dict[str, Any] = {}
        sealed_ids: list[str] = []
        staged_payloads = tuple(
            (page, extract_payload(model_result, page)) for page in current_pages
        )
        for page, payload in staged_payloads:
            if page.page_id in self._sealed:
                raise RuntimeError(f"Page {page.page_id!r} was already sealed")
            operation = self.backend.seal_after_generation(
                context.session_id,
                page.page_id,
                payload,
                keep_resident=keep_resident,
            )
            sealed_ids.append(page.page_id)
            metrics[page.page_id] = asdict(operation) if hasattr(operation, "__dataclass_fields__") else operation
        events.append("seal_complete")
        self._sealed.update(sealed_ids)
        self._next_turn += 1
        return RecordedTurnArtifact(
            session_id=context.session_id,
            instance_id=context.instance_id,
            turn_id=context.turn_id,
            history_token_end=history_end,
            prefill_token_end=prefill_end,
            generated_token_end=generated_end,
            plan=asdict(plan),
            budget=asdict(decision),
            preparation=_portable_preparation(prepared),
            sealed_page_ids=tuple(sealed_ids),
            backend_metrics=metrics,
            event_order=tuple(events),
        )


def _portable_preparation(prepared: Any) -> Mapping[str, Any]:
    if hasattr(prepared, "receipt"):
        receipt = getattr(prepared, "receipt")
        return {
            "session_id": getattr(receipt, "session_id", None),
            "turn_id": getattr(receipt, "turn_id", None),
            "preparation_id": getattr(receipt, "preparation_id", None),
            "ordered_page_ids": tuple(getattr(receipt, "ordered_page_ids", ())),
            "active_logical_positions": tuple(
                getattr(prepared, "active_logical_positions", ())
            ),
            "suffix_logical_positions": tuple(
                getattr(prepared, "suffix_logical_positions", ())
            ),
        }
    if hasattr(prepared, "__dataclass_fields__"):
        return asdict(prepared)
    return {"type": type(prepared).__name__}
