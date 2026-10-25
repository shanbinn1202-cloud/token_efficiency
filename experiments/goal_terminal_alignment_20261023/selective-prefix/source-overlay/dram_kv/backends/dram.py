"""SelectionPlan-driven DRAM restore boundary executed before model prefill."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from threading import RLock
from typing import Any, Generic, Mapping, TypeVar

from ..runtime import DramKVRuntime, OperationMetrics, RestoreResult, RuntimeStats
from ..schema import PageCatalog, SelectionPlan, TurnContext


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class PreparedTurn(Generic[PayloadT]):
    session_id: str
    turn_id: int
    preparation_id: str
    result: RestoreResult[PayloadT]


class DramRuntimeBackend(Generic[PayloadT]):
    """Bridge the experiment pipeline to physical Page lifecycle operations.

    ``execute`` is synchronous: it returns only after every selected Page is
    resident and ordered. The model integration must call it before formal
    prefill, then obtain the exact payload tuple with ``prepared_payloads``.
    New live-tail Pages are sealed with ``seal_after_generation`` only after
    generation has completed.
    """

    name = "dram_hbm"

    def __init__(self, runtime: DramKVRuntime[PayloadT]) -> None:
        self.runtime = runtime
        self._prepared: dict[str, PreparedTurn[PayloadT]] = {}
        self._lock = RLock()

    def seal_after_generation(
        self,
        session_id: str,
        page_id: str,
        gpu_payload: PayloadT,
        *,
        keep_resident: bool = False,
    ) -> OperationMetrics:
        return self.runtime.offload_sealed(
            session_id, page_id, gpu_payload, keep_resident=keep_resident
        )

    def prepare_before_prefill(
        self, context: TurnContext, catalog: PageCatalog, plan: SelectionPlan
    ) -> RestoreResult[PayloadT]:
        with self._lock:
            result = self.runtime.restore_selected(context, catalog, plan)
            preparation_id = self._preparation_id(
                context.session_id,
                context.turn_id,
                plan.goal_hash,
                result.ordered_page_ids,
            )
            self._prepared[context.session_id] = PreparedTurn(
                context.session_id, context.turn_id, preparation_id, result
            )
        return result

    def prepare_full_before_prefill(
        self, context: TurnContext, catalog: PageCatalog
    ) -> RestoreResult[PayloadT]:
        with self._lock:
            result = self.runtime.restore_all(context, catalog)
            preparation_id = self._preparation_id(
                context.session_id,
                context.turn_id,
                "dense_full_restore",
                result.ordered_page_ids,
            )
            self._prepared[context.session_id] = PreparedTurn(
                context.session_id, context.turn_id, preparation_id, result
            )
        return result

    def execute(
        self, context: TurnContext, catalog: PageCatalog, plan: SelectionPlan
    ) -> Mapping[str, Any]:
        result = self.prepare_before_prefill(context, catalog, plan)
        preparation_id = self.preparation_id(context.session_id, context.turn_id)
        return {
            "backend": self.name,
            "phase": "pre_prefill_restore_complete",
            "session_id": context.session_id,
            "turn_id": context.turn_id,
            "selector": plan.selector,
            "goal_hash": plan.goal_hash,
            "ordered_active_page_ids": list(result.ordered_page_ids),
            "preparation_id": preparation_id,
            "active_pages": len(result.ordered_page_ids),
            **result.as_metrics(),
        }

    def preparation_id(self, session_id: str, turn_id: int) -> str:
        with self._lock:
            prepared = self._prepared.get(session_id)
            if prepared is None:
                raise RuntimeError(f"session {session_id!r} has no prepared working set")
            if prepared.turn_id != turn_id:
                raise RuntimeError(
                    f"prepared working set is for turn {prepared.turn_id}, not {turn_id}"
                )
            return prepared.preparation_id

    def prepared_payloads(
        self, session_id: str, turn_id: int, *, preparation_id: str
    ) -> tuple[PayloadT, ...]:
        with self._lock:
            prepared = self._prepared.get(session_id)
            if prepared is None:
                raise RuntimeError(f"session {session_id!r} has no prepared working set")
            if prepared.turn_id != turn_id:
                raise RuntimeError(
                    f"prepared working set is for turn {prepared.turn_id}, not {turn_id}"
                )
            if prepared.preparation_id != preparation_id:
                raise RuntimeError("prepared working set was superseded by another plan")
            return prepared.result.ordered_payloads

    def session_metrics(self, session_id: str) -> RuntimeStats:
        return self.runtime.session_metrics(session_id)

    def release_session(self, session_id: str) -> Mapping[str, int | float]:
        with self._lock:
            self._prepared.pop(session_id, None)
            return asdict(self.runtime.release_session(session_id))

    @staticmethod
    def _preparation_id(
        session_id: str,
        turn_id: int,
        goal_hash: str,
        ordered_page_ids: tuple[str, ...],
    ) -> str:
        stable = "\0".join((session_id, str(turn_id), goal_hash, *ordered_page_ids))
        return sha256(stable.encode("utf-8")).hexdigest()
