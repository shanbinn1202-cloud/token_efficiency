"""Portable JSONL artifacts for replay across backends."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

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


Record = TypeVar("Record", TurnContext, GoalRecord, PageCatalog, SelectionPlan)


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            value = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def turn_context_from_dict(value: Mapping[str, Any]) -> TurnContext:
    data = dict(value)
    for key in (
        "historical_turn_ids", "historical_page_ids", "hbm_resident_page_ids"
    ):
        data[key] = tuple(data.get(key, ()))
    return TurnContext(**data)


def goal_record_from_dict(value: Mapping[str, Any]) -> GoalRecord:
    data = dict(value)
    for key in ("files", "symbols", "tests", "errors", "constraints"):
        data[key] = tuple(data.get(key, ()))
    raw_retrieval = data.get("retrieval_goal")
    if isinstance(raw_retrieval, Mapping):
        retrieval = dict(raw_retrieval)
        for key in (
            "files", "symbols", "tests", "errors", "evidence_needed", "constraints",
        ):
            retrieval[key] = tuple(retrieval.get(key, ()))
        data["retrieval_goal"] = RetrievalGoal(**retrieval)
    raw_policy = data.get("policy_hint")
    if isinstance(raw_policy, Mapping):
        policy = dict(raw_policy)
        policy["priorities"] = tuple(policy.get("priorities", ()))
        data["policy_hint"] = GoalPolicyHint(**policy)
    for key in ("triggers", "changed_fields"):
        data[key] = tuple(data.get(key, ()))
    return GoalRecord(**data)


def page_catalog_from_dict(value: Mapping[str, Any]) -> PageCatalog:
    data = dict(value)
    pages = []
    for raw in data.pop("pages"):
        page_data = dict(raw)
        for key in ("files", "symbols", "tests", "errors", "lexical_handles"):
            page_data[key] = tuple(page_data.get(key, ()))
        pages.append(PageRecord(**page_data))
    return PageCatalog(pages=tuple(pages), **data)


def selection_plan_from_dict(value: Mapping[str, Any]) -> SelectionPlan:
    data = dict(value)
    data["reasons"] = {
        page_id: SelectionReason(**reason)
        for page_id, reason in data["reasons"].items()
    }
    for key in (
        "ordered_page_ids", "mandatory_page_ids", "recent_page_ids",
        "retrieved_page_ids", "added_page_ids", "retained_page_ids",
        "removed_page_ids", "standby_page_ids",
    ):
        data[key] = tuple(data.get(key, ()))
    return SelectionPlan(**data)
