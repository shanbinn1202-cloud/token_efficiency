# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU candidate for Xuanxue Selector V2 in the SWE-bench path.

This module is deliberately NOT_INSTALLABLE.  It executes the audited upstream
``PrefillPageEmbeddingSelectorV2`` after adapting 16-token physical Page
records to one synthetic 16-token selector unit per 784-token vLLM restore
page.  The synthetic geometry is explicit in the receipt; it is not claimed to
be byte-identical to upstream physical-Page selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence


UPSTREAM_PACKAGE = Path(__file__).resolve().parents[1] / "dram_kv"

from dram_kv.artifacts import goal_record_from_dict, turn_context_from_dict
from dram_kv.page_vectors import PageVectorIndex, PageVectorProvenance, PageVectorPublisher
from dram_kv.schema import PageCatalog, PageRecord, SelectionPlan, TurnContext
from dram_kv.selectors.prefill_v2 import PrefillPageEmbeddingSelectorV2
from xuanxue_vllm_policy_adapter import (
    build_goal_query_payload as legacy_build_goal_query_payload,
    build_observation_query_payload as legacy_build_observation_query_payload,
)


_QUERY_POLICIES = frozenset({"goal_query", "observation_query"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def observation_source_sha256(goal_output: Mapping[str, object]) -> str:
    """Hash the exact raw current observation accepted by the Goal chronology."""

    contexts = goal_output.get("contexts")
    _require(isinstance(contexts, list) and bool(contexts), "Goal contexts required")
    observation = contexts[-1].get("current_observation")
    _require(isinstance(observation, str), "current observation must be raw text")
    return sha256(observation.encode("utf-8")).hexdigest()


def _constructor_source_hashes() -> dict[str, str]:
    files = [UPSTREAM_PACKAGE / "schema.py", *sorted((UPSTREAM_PACKAGE / "goals").glob("*.py"))]
    return {
        path.relative_to(UPSTREAM_PACKAGE).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _canonical_goal_probe(goal: object) -> tuple[dict[str, object], str, str]:
    """Rebuild the only accepted probe from the parsed upstream GoalRecord."""

    payload = json.loads(json.dumps(
        goal.selector_payload, ensure_ascii=False, sort_keys=True, allow_nan=False
    ))
    probe = "Retrieve historical evidence for this task Goal:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return payload, probe, sha256(probe.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QueryBinding:
    """Trusted control-plane binding for one already-produced query vector."""

    session_id: str
    generation: str
    turn_id: int
    policy: str
    query_source: str
    producer_provenance_hash: str
    source_sha256: str
    goal_hash: str
    query_content_hash: str | None

    def __post_init__(self) -> None:
        _require(self.policy in _QUERY_POLICIES, "unsupported query policy")
        expected = "goal_hidden" if self.policy == "goal_query" else "observation_hidden"
        _require(self.query_source == expected, "query source does not match policy")
        _require(bool(self.session_id) and self.turn_id > 0, "invalid query session/turn")
        for label, value in (
            ("generation", self.generation),
            ("producer provenance", self.producer_provenance_hash),
            ("source", self.source_sha256),
            ("Goal", self.goal_hash),
        ):
            _require(_is_sha(value), f"{label} hash must be SHA256")
        _require(
            self.query_content_hash is None or _is_sha(self.query_content_hash),
            "query content hash must be SHA256 or absent for lexical fallback",
        )


@dataclass(frozen=True, slots=True)
class CoarseAdaptation:
    catalog: PageCatalog
    index: PageVectorIndex
    representative_fine_page_ids: Mapping[str, str]
    coarse_index_by_page_id: Mapping[str, int]
    vector_status: str
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SelectorDecision:
    policy: str
    plan: SelectionPlan
    ranked_logical_pages: tuple[int, ...]
    selected_logical_pages: tuple[int, ...]
    query_hash: str
    kv_transfer_params: Mapping[str, object]
    adaptation_receipt: Mapping[str, object]
    system_valid: bool = False
    live_multi_turn_supported: bool = False
    installable: bool = False


def _complete_group(
    group: Sequence[PageRecord], *, coarse_index: int, fine_page_tokens: int,
    vllm_page_tokens: int,
) -> tuple[PageRecord, ...]:
    ordered = tuple(sorted(group, key=lambda page: (page.logical_position_start, page.page_id)))
    expected_start = coarse_index * vllm_page_tokens
    expected = list(range(expected_start, expected_start + vllm_page_tokens, fine_page_tokens))
    starts = [page.logical_position_start for page in ordered]
    _require(starts == expected, f"coarse page {coarse_index} has incomplete/noncontiguous fine geometry")
    _require(
        all(
            page.token_count == fine_page_tokens
            and page.logical_position_end == page.logical_position_start + fine_page_tokens
            and page.token_start >= 0
            for page in ordered
        ),
        f"coarse page {coarse_index} contains incompatible fine Pages",
    )
    return ordered


def adapt_16_to_vllm_pages(
    *, index: PageVectorIndex, provenance: PageVectorProvenance,
    context: TurnContext, pages: Sequence[PageRecord], binding: QueryBinding,
    vllm_page_tokens: int, expected_vllm_pages: int,
) -> CoarseAdaptation:
    """Build explicit synthetic selector units and copy only compatible vectors.

    Each complete vLLM page is represented by a synthetic 16-token PageRecord.
    Its vector is the final 16-token Page vector in that vLLM interval, matching
    upstream last-token pooling.  Missing or provenance-incompatible vectors are
    intentionally left unpublished so the real upstream selector takes its
    deterministic lexical fallback for the whole decision.
    """

    _require(provenance.page_size == 16, "audited Selector V2 provenance must use 16-token Pages")
    _require(provenance.pooling == "last_token",
             "16-to-vLLM endpoint adaptation requires last_token pooling")
    _require(
        type(vllm_page_tokens) is int and vllm_page_tokens > 0
        and vllm_page_tokens % provenance.page_size == 0,
        "vLLM page size must be a positive multiple of 16",
    )
    _require(type(expected_vllm_pages) is int and expected_vllm_pages > 0,
             "positive expected vLLM page count required")
    _require(binding.session_id == context.session_id and binding.turn_id == context.turn_id,
             "stale query binding for session/turn")
    _require(binding.producer_provenance_hash == provenance.provenance_hash,
             "query binding provenance differs from configured producer")

    history = tuple(page for page in pages if page.turn_id < context.turn_id)
    _require(bool(history), "sealed historical fine Pages required")
    _require(len({page.page_id for page in history}) == len(history),
             "fine Page IDs must be stable and unique")
    groups: dict[int, list[PageRecord]] = {}
    for page in history:
        _require(page.logical_position_start >= 0, "negative Page geometry")
        coarse_index = page.logical_position_start // vllm_page_tokens
        _require(coarse_index < expected_vllm_pages, "fine Page lies outside bound vLLM universe")
        _require(page.logical_position_end <= (coarse_index + 1) * vllm_page_tokens,
                 "fine Page crosses a vLLM page boundary")
        groups.setdefault(coarse_index, []).append(page)
    _require(set(groups) == set(range(expected_vllm_pages)),
             "fine Page catalog does not exactly cover expected vLLM pages")
    complete = {
        coarse_index: _complete_group(
            group, coarse_index=coarse_index, fine_page_tokens=provenance.page_size,
            vllm_page_tokens=vllm_page_tokens,
        )
        for coarse_index, group in groups.items()
    }
    representatives = {
        coarse_index: group[-1] for coarse_index, group in complete.items()
    }
    fine_snapshot = index.snapshot(
        context.session_id, context.turn_id,
        (representatives[idx].page_id for idx in range(expected_vllm_pages)), provenance,
    )
    if fine_snapshot.query is not None:
        _require(
            binding.query_content_hash == fine_snapshot.query.vector.content_hash,
            "query binding content hash differs from published vector",
        )
    else:
        _require(binding.query_content_hash is None,
                 "query binding claims content for a missing/incompatible vector")

    coarse_index_store = PageVectorIndex()
    publisher = PageVectorPublisher(coarse_index_store, provenance)
    if fine_snapshot.query is not None and not fine_snapshot.query_mismatch:
        publisher.publish_query(context.session_id, context.turn_id, fine_snapshot.query.vector)

    records: list[PageRecord] = []
    representative_ids: dict[str, str] = {}
    coarse_indices: dict[str, int] = {}
    vector_issues: list[str] = []
    for coarse_index in range(expected_vllm_pages):
        group = complete[coarse_index]
        representative = representatives[coarse_index]
        coarse_id = f"vllm-page-{coarse_index:08d}"
        representative_ids[coarse_id] = representative.page_id
        coarse_indices[coarse_id] = coarse_index
        fine_entry = fine_snapshot.pages.get(representative.page_id)
        compatible = fine_entry is not None
        if compatible and representative.page_vector_ref is not None:
            compatible = representative.page_vector_ref == fine_entry.vector_ref
        vector_ref = None
        if compatible:
            vector_ref = publisher.publish_page(
                context.session_id, coarse_id, fine_entry.vector
            )
        else:
            vector_issues.append(representative.page_id)
        summaries = _ordered_unique(tuple(page.summary or "" for page in group))
        records.append(PageRecord(
            page_id=coarse_id,
            turn_id=max(page.turn_id for page in group),
            role=representative.role,
            # Synthetic Selector V2 geometry, deliberately not vLLM token geometry.
            token_start=coarse_index * provenance.page_size,
            token_end=(coarse_index + 1) * provenance.page_size,
            logical_position_start=coarse_index * provenance.page_size,
            logical_position_end=(coarse_index + 1) * provenance.page_size,
            byte_size=sum(page.byte_size for page in group),
            mandatory=any(page.mandatory for page in group),
            files=_ordered_unique(tuple(value for page in group for value in page.files)),
            symbols=_ordered_unique(tuple(value for page in group for value in page.symbols)),
            tests=_ordered_unique(tuple(value for page in group for value in page.tests)),
            errors=_ordered_unique(tuple(value for page in group for value in page.errors)),
            lexical_handles=_ordered_unique(tuple(
                value for page in group for value in page.lexical_handles
            )),
            summary="\n".join(summaries) or None,
            page_vector_ref=vector_ref,
        ))

    status = "complete"
    if fine_snapshot.query is None or fine_snapshot.query_mismatch:
        status = "query_unavailable_lexical_fallback"
    elif vector_issues or fine_snapshot.missing_page_ids or fine_snapshot.mismatched_page_ids:
        status = "page_vector_unavailable_lexical_fallback"
    receipt = {
        "schema_version": 1,
        "adapter": "explicit_16_to_784_last_fine_page_v1",
        "fine_page_tokens": provenance.page_size,
        "vllm_page_tokens": vllm_page_tokens,
        "expected_vllm_pages": expected_vllm_pages,
        "synthetic_selector_unit_tokens": provenance.page_size,
        "representative_rule": "last complete 16-token Page in each vLLM interval",
        "byte_identical_to_upstream_physical_page_selection": False,
        "vector_status": status,
        "vector_issues": sorted(set(vector_issues)),
        "fine_provenance_hash": provenance.provenance_hash,
    }
    return CoarseAdaptation(
        catalog=PageCatalog(context.session_id, tuple(records)),
        index=coarse_index_store,
        representative_fine_page_ids=representative_ids,
        coarse_index_by_page_id=coarse_indices,
        vector_status=status,
        receipt=receipt,
    )


class XuanxueSelectorRuntime:
    """Single-authority, bounded-turn execution candidate for query policies."""

    def __init__(self, *, max_sessions: int = 128,
                 selector_factory: Callable[..., object] = PrefillPageEmbeddingSelectorV2):
        _require(type(max_sessions) is int and max_sessions > 0, "positive session bound required")
        self.max_sessions = max_sessions
        self.selector_factory = selector_factory
        self._sessions: dict[str, dict[str, object]] = {}
        self.selector_calls = 0

    def open_session(self, *, session_id: str, policy: str, generation: str,
                     first_turn: int) -> None:
        _require(policy in _QUERY_POLICIES, "query runtime accepts only Goal/Observation policies")
        _require(isinstance(session_id, str) and 0 < len(session_id) <= 128,
                 "bounded session ID required")
        _require(_is_sha(generation), "generation must be SHA256")
        _require(type(first_turn) is int and first_turn > 0, "positive first turn required")
        _require(session_id not in self._sessions, "session ID cannot be reused")
        _require(len(self._sessions) < self.max_sessions, "session capacity reached")
        self._sessions[session_id] = {
            "policy": policy, "generation": generation,
            "next_turn": first_turn, "state": "open",
        }

    def close_session(self, *, session_id: str, generation: str) -> None:
        state = self._sessions.get(session_id)
        _require(state is not None and state["generation"] == generation,
                 "unknown or stale session generation")
        state["state"] = "closed"

    def select(
        self, *, policy: str, goal_output: Mapping[str, object], binding: QueryBinding,
        fine_pages: Sequence[PageRecord], fine_index: PageVectorIndex,
        provenance: PageVectorProvenance, selected_count: int | None = None,
        active_token_budget: int | None = None,
        vllm_page_tokens: int, expected_vllm_pages: int, prefix_identity: str,
        recent_turns: int = 2, recent_quota: int = 1,
        previous_active: Sequence[str] = (), ever_selected: Sequence[str] = (),
    ) -> SelectorDecision:
        state = self._sessions.get(binding.session_id)
        _require(state is not None and state["state"] == "open", "session is not open")
        _require(state["generation"] == binding.generation, "stale session generation")
        _require(state["policy"] == policy == binding.policy, "session policy mismatch")
        _require(state["next_turn"] == binding.turn_id, "stale, repeated, or skipped turn")
        _require((selected_count is None) != (active_token_budget is None),
                 "set exactly one of selected_count or active_token_budget")
        if active_token_budget is not None:
            _require(type(active_token_budget) is int and active_token_budget > 0
                     and active_token_budget % vllm_page_tokens == 0,
                     "active token budget must be a positive whole number of vLLM pages")
            selected_count = active_token_budget // vllm_page_tokens
        _require(type(selected_count) is int and 0 < selected_count < expected_vllm_pages,
                 "selective page budget required")
        _require(_is_sha(prefix_identity), "prefix identity must be SHA256")
        _require(type(recent_turns) is int and type(recent_quota) is int
                 and recent_turns >= 0 and recent_quota >= 0,
                 "nonnegative recent safety bounds required")

        _require(goal_output.get("kind") == "maintained_goal_constructor_output",
                 "actual maintained Goal constructor output required")
        _require(goal_output.get("system_valid") is False
                 and goal_output.get("task_accuracy_valid") is False
                 and goal_output.get("live_multi_turn_supported") is False,
                 "offline Goal candidate validity flags changed")
        _require(goal_output.get("execution_status_policy") == "untrusted_text_status_disabled_v1",
                 "Goal provenance adaptation changed")
        rule = goal_output.get("goal_construction_rule")
        _require(isinstance(rule, str) and "MaintainedGoalConstructor(rewriter=None)" in rule
                 and "text-derived return_code/test_verdict disabled" in rule,
                 "Goal constructor heuristic/provenance contract changed")
        goal_adapter = importlib.import_module("goal_adapter")
        _require(goal_output.get("adapter_source_sha256")
                 == sha256(Path(goal_adapter.__file__).read_bytes()).hexdigest(),
                 "Goal provenance adapter source changed")
        _require(goal_output.get("constructor_source_sha256") == _constructor_source_hashes(),
                 "audited upstream Goal constructor sources changed")
        records, contexts = goal_output.get("goal_records"), goal_output.get("contexts")
        _require(isinstance(records, list) and isinstance(contexts, list)
                 and len(records) == len(contexts) and bool(records),
                 "complete Goal record/context chronology required")
        goal = goal_record_from_dict(records[-1])
        context = turn_context_from_dict(contexts[-1])
        _require(goal.session_id == context.session_id == binding.session_id
                 and goal.turn_id == context.turn_id == binding.turn_id,
                 "stale Goal/context/session binding")
        _require(goal.version_hash == goal_output.get("goal_hash") == binding.goal_hash,
                 "Goal hash differs from maintained Goal record")
        canonical_payload, canonical_probe, canonical_probe_hash = _canonical_goal_probe(goal)
        _require(goal_output.get("goal_payload") == canonical_payload,
                 "Goal payload differs from final maintained Goal record")
        _require(goal_output.get("goal_probe_text") == canonical_probe,
                 "Goal probe text is not the canonical selector-payload probe")
        _require(goal_output.get("goal_probe_text_sha256") == canonical_probe_hash,
                 "Goal probe hash was not independently recomputed")
        if policy == "goal_query":
            _require(binding.source_sha256 == canonical_probe_hash,
                     "Goal query is not bound to the maintained Goal probe")
        else:
            _require(binding.source_sha256 == observation_source_sha256(goal_output),
                     "Observation query is not bound to the current raw observation")

        coarse_ids = tuple(f"vllm-page-{idx:08d}" for idx in range(expected_vllm_pages))
        context = replace(
            context,
            historical_page_ids=coarse_ids,
            hbm_resident_page_ids=tuple(
                page_id for page_id in previous_active if page_id in set(coarse_ids)
            ),
        )
        adaptation = adapt_16_to_vllm_pages(
            index=fine_index, provenance=provenance, context=context, pages=fine_pages,
            binding=binding, vllm_page_tokens=vllm_page_tokens,
            expected_vllm_pages=expected_vllm_pages,
        )
        trace_hash = _digest({
            "goal_records": records,
            "contexts": contexts,
            "source_inventory_sha256": goal_output.get("source_inventory_sha256"),
        })
        selector = self.selector_factory(
            adaptation.index, provenance,
            frozen_goal_trace_hash=trace_hash,
            page_budget=selected_count,
            recent_turns=recent_turns,
            recent_quota=recent_quota,
            query_source=binding.query_source,
        )
        self.selector_calls += 1
        plan = selector.select(context, goal, adaptation.catalog, previous_active, ever_selected)
        _require(plan.selector == "prefill_page_embedding_v2",
                 "active path did not execute audited upstream Selector V2")
        _require(plan.selector_config.get("query_source") == binding.query_source,
                 "selector query source changed")
        selected_ids = tuple(plan.ordered_page_ids)
        _require(len(selected_ids) == selected_count,
                 "selector did not satisfy exact vLLM page budget")
        selected_indices = tuple(adaptation.coarse_index_by_page_id[page_id] for page_id in selected_ids)
        remaining = tuple(index for index in range(expected_vllm_pages)
                          if index not in set(selected_indices))
        ranking = selected_indices + remaining
        _require(len(ranking) == expected_vllm_pages and set(ranking) == set(range(expected_vllm_pages)),
                 "adapter did not produce complete vLLM page permutation")
        legacy_equivalence = "not_applicable_vector_fallback"
        if adaptation.vector_status == "complete":
            legacy_builder = (legacy_build_goal_query_payload if policy == "goal_query"
                              else legacy_build_observation_query_payload)
            legacy_payload = legacy_builder(
                index=fine_index,
                provenance=provenance,
                session_id=binding.session_id,
                turn_id=binding.turn_id,
                pages=fine_pages,
                # The legacy adapter recomputes int(num_pages * fraction).
                # Preserve the exact upstream budget (e.g. 15 / 22 must not
                # round down to fourteen through binary floating point).
                fraction=Fraction(selected_count, expected_vllm_pages),
                vllm_page_tokens=vllm_page_tokens,
                recent_turns=recent_turns,
                # Upstream clips recent safety to the capacity remaining after
                # mandatory Pages. The older adapter rejects an unclipped quota.
                recent_quota=len(plan.recent_page_ids),
            )
            _require(
                set(legacy_payload.ranked_logical_pages[:selected_count])
                == set(selected_indices),
                "existing Xuanxue vLLM adapter disagrees with upstream SelectionPlan",
            )
            legacy_equivalence = "pass"
        query_hash = _digest({
            "policy": policy,
            "query_source": binding.query_source,
            "query_content_hash": binding.query_content_hash,
            "source_sha256": binding.source_sha256,
            "goal_hash": binding.goal_hash,
            "plan_config": dict(plan.selector_config),
            "selected": selected_indices,
            "adaptation": dict(adaptation.receipt),
        })
        kv = {
            "selective_page_contract_version": 2,
            "selective_page_policy": policy,
            "selective_page_num_pages": expected_vllm_pages,
            "selective_page_page_tokens": vllm_page_tokens,
            "selective_page_selected_count": selected_count,
            "selective_page_prefix_identity": prefix_identity,
            "selective_page_ranked_pages": list(ranking),
            "selective_page_query_hash": query_hash,
        }
        state["next_turn"] = binding.turn_id + 1
        return SelectorDecision(
            policy=policy,
            plan=plan,
            ranked_logical_pages=ranking,
            selected_logical_pages=tuple(sorted(selected_indices)),
            query_hash=query_hash,
            kv_transfer_params=kv,
            adaptation_receipt={
                **dict(adaptation.receipt),
                "selector_called": True,
                "selector": plan.selector,
                "existing_xuanxue_adapter_equivalence": legacy_equivalence,
                "effective_mode": plan.selector_config.get("effective_mode"),
                "requested_active_token_budget": active_token_budget,
                "derived_vllm_page_budget": selected_count,
                "system_valid": False,
                "live_multi_turn_supported": False,
                "installable": False,
            },
        )


class ActiveFourPolicyRouter:
    """Keep Full/Random50 handlers untouched; query policies call Selector V2."""

    def __init__(self, *, full_handler: Callable[..., object],
                 random50_handler: Callable[..., object],
                 query_runtime: XuanxueSelectorRuntime):
        self.full_handler = full_handler
        self.random50_handler = random50_handler
        self.query_runtime = query_runtime

    def execute(self, policy: str, **kwargs: object) -> object:
        if policy == "full":
            return self.full_handler(**kwargs)
        if policy == "random50":
            return self.random50_handler(**kwargs)
        _require(policy in _QUERY_POLICIES, "unknown policy")
        return self.query_runtime.select(policy=policy, **kwargs)


__all__ = [
    "ActiveFourPolicyRouter",
    "CoarseAdaptation",
    "QueryBinding",
    "SelectorDecision",
    "XuanxueSelectorRuntime",
    "adapt_16_to_vllm_pages",
    "observation_source_sha256",
]
