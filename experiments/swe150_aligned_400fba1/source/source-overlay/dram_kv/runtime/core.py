"""Transactional DRAM-backed storage and HBM working-set management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from threading import RLock
from time import perf_counter
from typing import Generic, Mapping, TypeVar

from ..schema import PageCatalog, SelectionPlan, TurnContext
from .tensor import TensorAdapter


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class RuntimeStats:
    dram_bytes: int = 0
    hbm_resident_bytes: int = 0
    peak_hbm_managed_bytes: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    restore_count: int = 0
    offload_count: int = 0
    release_count: int = 0
    reuse_count: int = 0
    reactivation_count: int = 0
    restore_latency_ms: float = 0.0
    offload_latency_ms: float = 0.0


@dataclass(frozen=True)
class OperationMetrics:
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    restore_count: int = 0
    offload_count: int = 0
    release_count: int = 0
    reuse_count: int = 0
    reactivation_count: int = 0
    transient_hbm_managed_bytes: int = 0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class RestoreResult(Generic[PayloadT]):
    ordered_page_ids: tuple[str, ...]
    ordered_payloads: tuple[PayloadT, ...]
    operation: OperationMetrics
    session: RuntimeStats

    def as_metrics(self) -> Mapping[str, int | float]:
        result = {f"operation_{key}": value for key, value in asdict(self.operation).items()}
        result.update({f"session_{key}": value for key, value in asdict(self.session).items()})
        return result


@dataclass
class _Session(Generic[PayloadT]):
    dram: dict[str, PayloadT]
    dram_sizes: dict[str, int]
    resident: dict[str, PayloadT]
    resident_sizes: dict[str, int]
    ever_resident: set[str]
    stats: RuntimeStats


class DramKVRuntime(Generic[PayloadT]):
    """Own exact sealed KV in DRAM and expose an ordered HBM working set.

    Transfer work is staged before session dictionaries and counters are
    replaced. Consequently an adapter exception cannot publish a partial page
    set or partially advance accounting.
    """

    def __init__(self, adapter: TensorAdapter[PayloadT], *, pin_cpu: bool = False) -> None:
        self.adapter = adapter
        self.pin_cpu = pin_cpu
        self._sessions: dict[str, _Session[PayloadT]] = {}
        self._lock = RLock()

    @staticmethod
    def _new_session() -> _Session[PayloadT]:
        return _Session({}, {}, {}, {}, set(), RuntimeStats())

    def offload_sealed(
        self,
        session_id: str,
        page_id: str,
        gpu_payload: PayloadT,
        *,
        keep_resident: bool = False,
    ) -> OperationMetrics:
        if not session_id or not page_id:
            raise ValueError("session_id and page_id are required")
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None and page_id in state.dram:
                raise ValueError(f"Page {page_id!r} is already sealed")
            started = perf_counter()
            size = self.adapter.nbytes(gpu_payload)
            if size < 0:
                raise ValueError("payload byte size cannot be negative")
            cpu_payload = self.adapter.to_cpu(gpu_payload, pin=self.pin_cpu)
            latency = (perf_counter() - started) * 1000.0

            if state is None:
                state = self._new_session()
                self._sessions[session_id] = state
            new_dram = dict(state.dram)
            new_dram_sizes = dict(state.dram_sizes)
            new_resident = dict(state.resident)
            new_resident_sizes = dict(state.resident_sizes)
            new_ever = set(state.ever_resident)
            new_dram[page_id] = cpu_payload
            new_dram_sizes[page_id] = size
            if keep_resident:
                new_resident[page_id] = gpu_payload
                new_resident_sizes[page_id] = size
                new_ever.add(page_id)
            else:
                # The runtime does not retain the supplied GPU payload. The
                # caller must drop its own reference after this call returns.
                new_resident.pop(page_id, None)
                new_resident_sizes.pop(page_id, None)
            hbm_bytes = sum(new_resident_sizes.values())
            state.dram = new_dram
            state.dram_sizes = new_dram_sizes
            state.resident = new_resident
            state.resident_sizes = new_resident_sizes
            state.ever_resident = new_ever
            state.stats = replace(
                state.stats,
                dram_bytes=sum(new_dram_sizes.values()),
                hbm_resident_bytes=hbm_bytes,
                peak_hbm_managed_bytes=max(
                    state.stats.peak_hbm_managed_bytes, hbm_bytes
                ),
                d2h_bytes=state.stats.d2h_bytes + size,
                offload_count=state.stats.offload_count + 1,
                offload_latency_ms=state.stats.offload_latency_ms + latency,
            )
            return OperationMetrics(
                d2h_bytes=size,
                offload_count=1,
                transient_hbm_managed_bytes=hbm_bytes,
                latency_ms=latency,
            )

    def restore_selected(
        self, context: TurnContext, catalog: PageCatalog, plan: SelectionPlan
    ) -> RestoreResult[PayloadT]:
        plan.validate_against(context, catalog)
        return self._restore(
            context.session_id,
            plan.ordered_page_ids,
            standby_page_ids=plan.standby_page_ids,
        )

    def restore_all(
        self, context: TurnContext, catalog: PageCatalog
    ) -> RestoreResult[PayloadT]:
        if context.session_id != catalog.session_id:
            raise ValueError("session mismatch between context and catalog")
        page_map = catalog.by_id()
        unknown = set(context.historical_page_ids) - set(page_map)
        if unknown:
            raise ValueError(f"history contains unknown Pages: {sorted(unknown)}")
        ordered = tuple(
            page.page_id
            for page in catalog.pages
            if page.page_id in set(context.historical_page_ids)
        )
        if ordered != context.historical_page_ids:
            raise ValueError("historical Pages must preserve original logical order")
        return self._restore(context.session_id, ordered)

    def _restore(
        self,
        session_id: str,
        ordered_page_ids: tuple[str, ...],
        *,
        standby_page_ids: tuple[str, ...] = (),
    ) -> RestoreResult[PayloadT]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"unknown session {session_id!r}")
            resident_page_ids = tuple(dict.fromkeys((*ordered_page_ids, *standby_page_ids)))
            unknown = [
                page_id for page_id in resident_page_ids
                if page_id not in state.resident and page_id not in state.dram
            ]
            if unknown:
                raise KeyError(f"Pages have no sealed backing: {unknown}")

            started = perf_counter()
            staged: dict[str, PayloadT] = {}
            staged_sizes: dict[str, int] = {}
            restore_ids: list[str] = []
            reuse_ids: list[str] = []
            reactivated_ids: list[str] = []
            previous_ids = set(state.resident)
            for page_id in resident_page_ids:
                if page_id in state.resident:
                    staged[page_id] = state.resident[page_id]
                    staged_sizes[page_id] = state.resident_sizes[page_id]
                    reuse_ids.append(page_id)
                else:
                    payload = self.adapter.to_device(state.dram[page_id])
                    staged[page_id] = payload
                    staged_sizes[page_id] = state.dram_sizes[page_id]
                    restore_ids.append(page_id)
                    if page_id in state.ever_resident:
                        reactivated_ids.append(page_id)
            latency = (perf_counter() - started) * 1000.0
            released = previous_ids - set(resident_page_ids)
            h2d_bytes = sum(staged_sizes[page_id] for page_id in restore_ids)
            hbm_bytes = sum(staged_sizes.values())
            # Reused tensors are already counted in the old resident set.
            # Newly restored tensors coexist with that old set until commit.
            transient_hbm_bytes = sum(state.resident_sizes.values()) + h2d_bytes

            # Single commit point after every potentially failing adapter call.
            state.resident = staged
            state.resident_sizes = staged_sizes
            state.ever_resident.update(resident_page_ids)
            state.stats = replace(
                state.stats,
                hbm_resident_bytes=hbm_bytes,
                peak_hbm_managed_bytes=max(
                    state.stats.peak_hbm_managed_bytes,
                    transient_hbm_bytes,
                    hbm_bytes,
                ),
                h2d_bytes=state.stats.h2d_bytes + h2d_bytes,
                restore_count=state.stats.restore_count + len(restore_ids),
                release_count=state.stats.release_count + len(released),
                reuse_count=state.stats.reuse_count + len(reuse_ids),
                reactivation_count=state.stats.reactivation_count + len(reactivated_ids),
                restore_latency_ms=state.stats.restore_latency_ms + latency,
            )
            operation = OperationMetrics(
                h2d_bytes=h2d_bytes,
                restore_count=len(restore_ids),
                release_count=len(released),
                reuse_count=len(reuse_ids),
                reactivation_count=len(reactivated_ids),
                transient_hbm_managed_bytes=transient_hbm_bytes,
                latency_ms=latency,
            )
            return RestoreResult(
                ordered_page_ids=ordered_page_ids,
                ordered_payloads=tuple(staged[page_id] for page_id in ordered_page_ids),
                operation=operation,
                session=state.stats,
            )

    def resident_page_ids(self, session_id: str) -> tuple[str, ...]:
        with self._lock:
            state = self._sessions.get(session_id)
            return tuple(state.resident) if state is not None else ()

    def active_payloads(self, session_id: str) -> tuple[PayloadT, ...]:
        with self._lock:
            state = self._sessions.get(session_id)
            return tuple(state.resident.values()) if state is not None else ()

    def session_metrics(self, session_id: str) -> RuntimeStats:
        with self._lock:
            state = self._sessions.get(session_id)
            return state.stats if state is not None else RuntimeStats()

    def release_session(self, session_id: str) -> RuntimeStats:
        """Drop runtime-owned payloads and return zeroed capacity gauges.

        Transfer bytes, latency, and operation counts remain lifetime totals.
        ``release_count`` counts resident GPU entries dropped by the runtime;
        it is not a CUDA allocator measurement.
        """
        with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                return RuntimeStats()
            return replace(
                state.stats,
                dram_bytes=0,
                hbm_resident_bytes=0,
                release_count=state.stats.release_count + len(state.resident),
            )
