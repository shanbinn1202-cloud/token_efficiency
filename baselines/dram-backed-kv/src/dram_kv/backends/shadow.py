"""Record selection quality/cost without changing model context."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..schema import PageCatalog, SelectionPlan, TurnContext


class ShadowBackend:
    name = "shadow"

    def __init__(
        self,
        evidence_page_ids: Mapping[int | tuple[str, int], Iterable[str]] | None = None,
    ) -> None:
        self._global_evidence: dict[int, frozenset[str]] = {}
        self._session_evidence: dict[tuple[str, int], frozenset[str]] = {}
        self._resident: dict[str, tuple[str, ...]] = {}
        self._ever_resident: dict[str, set[str]] = {}
        self._ever_active: dict[str, set[str]] = {}
        for key, page_ids in (evidence_page_ids or {}).items():
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise ValueError("session evidence key must be (session_id, turn_id)")
                session_id, turn_id = key
                self._session_evidence[(str(session_id), int(turn_id))] = frozenset(page_ids)
            else:
                self._global_evidence[int(key)] = frozenset(page_ids)

    def execute(
        self, context: TurnContext, catalog: PageCatalog, plan: SelectionPlan
    ) -> Mapping[str, Any]:
        plan.validate_against(context, catalog)
        page_map = catalog.by_id()
        active = set(plan.ordered_page_ids)
        evidence = self._session_evidence.get(
            (context.session_id, context.turn_id),
            self._global_evidence.get(context.turn_id, frozenset()),
        )
        unknown = evidence - set(context.historical_page_ids)
        if unknown:
            raise ValueError(f"Shadow evidence is outside sealed history: {sorted(unknown)}")
        found = active & evidence
        ever_active = self._ever_active.setdefault(context.session_id, set())
        active_reactivated = tuple(
            page_id for page_id in plan.added_page_ids if page_id in ever_active
        )
        previous_resident = self._resident.get(context.session_id, ())
        previous_set = set(previous_resident)
        resident = tuple(dict.fromkeys((*plan.ordered_page_ids, *plan.standby_page_ids)))
        resident_set = set(resident)
        ever_resident = self._ever_resident.setdefault(context.session_id, set())
        resident_added = tuple(page_id for page_id in resident if page_id not in previous_set)
        resident_removed = tuple(
            page_id for page_id in previous_resident if page_id not in resident_set
        )
        resident_reactivated = tuple(
            page_id for page_id in resident_added if page_id in ever_resident
        )
        h2d_bytes = sum(page_map[page_id].byte_size for page_id in resident_added)
        self._resident[context.session_id] = resident
        ever_resident.update(resident)
        ever_active.update(active)
        active_union = active | set(plan.removed_page_ids)
        active_jaccard_churn = (
            len(set(plan.added_page_ids) | set(plan.removed_page_ids)) / len(active_union)
            if active_union else 0.0
        )
        return {
            "backend": self.name,
            "session_id": context.session_id,
            "turn_id": context.turn_id,
            "selector": plan.selector,
            "goal_hash": plan.goal_hash,
            "dense_context_unchanged": True,
            "active_pages": len(active),
            "active_tokens": sum(page_map[page_id].token_count for page_id in active),
            "active_bytes": sum(page_map[page_id].byte_size for page_id in active),
            "added_pages": len(plan.added_page_ids),
            "retained_pages": len(plan.retained_page_ids),
            "removed_pages": len(plan.removed_page_ids),
            "active_reactivated_pages": len(active_reactivated),
            "standby_pages": len(plan.standby_page_ids),
            "resident_pages": len(resident),
            "resident_added_pages": len(resident_added),
            "resident_removed_pages": len(resident_removed),
            "resident_reactivated_pages": len(resident_reactivated),
            "simulated_h2d_pages": len(resident_added),
            "simulated_h2d_bytes": h2d_bytes,
            "active_set_jaccard_churn": active_jaccard_churn,
            "evidence_pages": len(evidence),
            "evidence_pages_found": len(found),
            "evidence_page_ids": sorted(evidence),
            "evidence_page_ids_found": sorted(found),
            "evidence_recall": len(found) / len(evidence) if evidence else None,
        }

    def release_session(self, session_id: str) -> None:
        self._resident.pop(session_id, None)
        self._ever_resident.pop(session_id, None)
        self._ever_active.pop(session_id, None)
