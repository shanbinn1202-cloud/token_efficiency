from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from minisgl.core import Req
from minisgl.env import ENV
from minisgl.kvcache import (
    compact_all_layers_kv_heads_inplace_sorted_triton,
    compact_all_layers_kv_heads_triton,
    compact_layer_kv_heads,
)
from minisgl.message import TokenizerErrorMsg
from minisgl.utils import init_logger

from .base import (
    LayerEvictionView,
    SelectionProfile,
    profile_add,
    profile_end_cuda,
    profile_finalize_cuda,
    profile_json_dict,
    profile_start_cuda,
)
from . import create_eviction_plan

if TYPE_CHECKING:
    from minisgl.engine import Engine
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.config import SchedulerConfig


logger = init_logger(__name__)


class EvictionManager:
    def __init__(
        self,
        *,
        config: SchedulerConfig,
        engine: Engine,
        cache_manager: CacheManager,
        token_pool: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.engine = engine
        self.cache_manager = cache_manager
        self.token_pool = token_pool
        self.device = device
        self.disabled = config.eviction_plan.strip().lower() in {"none", "off"}
        self.plan = create_eviction_plan(config.eviction_plan)
        if self.plan.requires_page_size_one and config.page_size != 1:
            raise ValueError(f"Eviction plan {self.plan.name!r} requires page_size == 1")
        self.compact_interval = getattr(config, "compact_interval", 128)
        self.compact_after_prefill = getattr(config, "compact_after_prefill", False)
        self.compact_before_park = getattr(config, "compact_before_park", False)
        self.compact_debug = config.compact_debug
        self.compact_debug_dir = config.compact_debug_dir
        self.compact_debug_jsonl = config.compact_debug_jsonl
        self.compact_debug_max_items = config.compact_debug_max_items
        self.eviction_trace = getattr(config, "eviction_trace", False)
        self.eviction_trace_dir = getattr(config, "eviction_trace_dir", "")
        self._last_query_ring_profile = self._query_ring_profile_snapshot()

    @property
    def compacts_active_req(self) -> bool:
        return not self.disabled

    def should_compact(self, req: Req, *, finished: bool) -> bool:
        if self.disabled:
            return False
        if req.session_id is None:
            return False
        return self._should_compact(req, finished=finished)

    def maybe_compact(
        self,
        req: Req,
        *,
        finished: bool,
        probe_q_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> TokenizerErrorMsg | None:
        if not self.should_compact(req, finished=finished):
            return None

        profile_timing = self.compact_debug_jsonl or self.compact_debug
        profile_split = profile_timing and bool(ENV.STAGEQ_PROFILE_SPLIT)
        selection_profile: SelectionProfile | None = {} if profile_split else None
        total_cpu_start = time.perf_counter() if profile_timing else None
        select_cpu_start = time.perf_counter() if profile_timing else None
        select_gpu_start = None
        select_gpu_end = None
        if profile_timing and self.device.type == "cuda":
            select_gpu_start = torch.cuda.Event(enable_timing=True)
            select_gpu_end = torch.cuda.Event(enable_timing=True)
            select_gpu_start.record(torch.cuda.current_stream(self.device))
        keep_by_layer = self._select_compaction_indices(
            req,
            probe_q_by_layer=probe_q_by_layer,
            selection_profile=selection_profile,
        )
        select_cpu_ms = (
            (time.perf_counter() - select_cpu_start) * 1000
            if select_cpu_start is not None
            else None
        )
        select_gpu_ms = None
        if profile_timing and self.device.type == "cuda":
            if select_gpu_end is not None:
                select_gpu_end.record(torch.cuda.current_stream(self.device))
            torch.cuda.current_stream(self.device).synchronize()
            if select_gpu_start is not None and select_gpu_end is not None:
                select_gpu_ms = select_gpu_start.elapsed_time(select_gpu_end)
        profile_finalize_cuda(selection_profile)
        if keep_by_layer is None:
            return None

        compact_len = keep_by_layer[0].shape[1]
        if compact_len > self.cache_manager.available_pages and not self._can_compact_in_place(
            req,
            compact_len,
        ):
            error = (
                "SESSION_CACHE_COMPACT_OOM: aborted current session request because "
                f"eviction plan {self.plan.name!r} could not allocate "
                f"{compact_len} new KV pages "
                f"(available_pages={self.cache_manager.available_pages}, uid={req.uid}, "
                f"session_id={req.session_id})"
            )
            logger.warning_rank0(error)
            self._write_compact_debug_summary(
                req=req,
                status="failed",
                error_code="SESSION_CACHE_COMPACT_OOM",
                error_message=error,
                trigger=_compact_trigger(req, finished),
                old_physical_cached_len=req.physical_cached_len,
                new_physical_cached_len=compact_len,
                available_pages_before=self.cache_manager.available_pages,
                available_pages_after=None,
                artifact_npz=None,
                compact_cpu_ms=None,
                compact_gpu_ms=None,
                select_cpu_ms=select_cpu_ms,
                select_gpu_ms=select_gpu_ms,
                selection_profile=profile_json_dict(selection_profile),
                prepare_cpu_ms=None,
                prepare_normalize_cpu_ms=None,
                prepare_representative_cpu_ms=None,
                prepare_alloc_cpu_ms=None,
                prepare_request_state_cpu_ms=None,
                prepare_position_state_cpu_ms=None,
                total_cpu_ms=(
                    (time.perf_counter() - total_cpu_start) * 1000
                    if total_cpu_start is not None
                    else None
                ),
            )
            return TokenizerErrorMsg(uid=req.uid, error=error, status_code=503)

        self.compact_session_req(
            req,
            keep_by_layer,
            finished=finished,
            select_cpu_ms=select_cpu_ms,
            select_gpu_ms=select_gpu_ms,
            selection_profile=profile_json_dict(selection_profile),
            total_cpu_start=total_cpu_start,
        )
        return None

    def _can_compact_in_place(self, req: Req, compact_len: int) -> bool:
        return (
            getattr(self.cache_manager, "page_size", 1) == 1
            and 0 <= compact_len <= req.physical_cached_len
        )

    def _should_compact(self, req: Req, *, finished: bool) -> bool:
        return (
            (finished and self.compact_before_park)
            or (req.generated_len == 1 and self.compact_after_prefill)
            or (
                req.generated_len > 1
                and self.compact_interval > 0
                and req.generated_len % self.compact_interval == 0
            )
        )

    def _select_compaction_indices(
        self,
        req: Req,
        *,
        probe_q_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        selection_profile: SelectionProfile | None = None,
    ) -> list[torch.Tensor] | None:
        keep_by_layer: list[torch.Tensor] = []
        compact_len: int | None = None

        for layer_id in range(self.engine.kv_cache.num_layers):
            k_layer = self.engine.kv_cache.k_cache(layer_id)
            num_heads = k_layer.shape[2]
            view = self._make_layer_eviction_view(
                req,
                layer_id=layer_id,
                num_kv_heads=num_heads,
                device=k_layer.device,
                probe_q_by_layer=probe_q_by_layer,
                selection_profile=selection_profile,
            )
            select_layer_start = time.perf_counter() if selection_profile is not None else None
            keep = self.plan.select_layer(view)
            if select_layer_start is not None:
                profile_add(
                    selection_profile,
                    "plan_select_cpu_ms",
                    (time.perf_counter() - select_layer_start) * 1000,
                )
            if keep.ndim != 2:
                raise RuntimeError(
                    f"Eviction plan {self.plan.name!r} returned {keep.ndim}D "
                    f"indices for layer {layer_id}; expected 2D"
                )
            if keep.device != k_layer.device or keep.dtype != torch.long:
                raise RuntimeError(
                    f"Eviction plan {self.plan.name!r} returned indices on "
                    f"{keep.device}/{keep.dtype}; expected "
                    f"{k_layer.device}/torch.int64"
                )
            if keep.shape[0] != num_heads:
                raise RuntimeError(
                    f"Eviction plan {self.plan.name!r} returned {keep.shape[0]} "
                    f"heads for layer {layer_id}; expected {num_heads}"
                )
            if compact_len is None:
                compact_len = keep.shape[1]
            elif keep.shape[1] != compact_len:
                raise RuntimeError(
                    f"Eviction plan {self.plan.name!r} returned compact length "
                    f"{keep.shape[1]} for layer {layer_id}; expected {compact_len}"
                )
            keep_by_layer.append(keep)

        return keep_by_layer

    def _make_layer_eviction_view(
        self,
        req: Req,
        *,
        layer_id: int,
        num_kv_heads: int,
        device: torch.device,
        probe_q_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        selection_profile: SelectionProfile | None = None,
    ) -> LayerEvictionView:
        profile_add(selection_profile, "layers", 1)
        profile_add(selection_profile, "physical_cached_len_total", req.physical_cached_len)
        logical_positions_by_head = None
        if self.plan.requires_logical_positions:
            logical_positions_by_head = _retained_positions_for_layer(
                req,
                layer_id=layer_id,
                num_kv_heads=num_kv_heads,
            )

        k_view = None
        v_view = None
        if self.plan.requires_kv_view:
            cpu_start = time.perf_counter() if selection_profile is not None else None
            cuda_token = profile_start_cuda(selection_profile, "kv_view_gpu_ms", device)
            table_row = self.engine.page_table[req.table_idx, : req.physical_cached_len].to(
                dtype=torch.long
            )
            k_layer = self.engine.kv_cache.k_cache(layer_id)
            v_layer = self.engine.kv_cache.v_cache(layer_id)
            flat_k = k_layer.reshape(k_layer.shape[0] * k_layer.shape[1], *k_layer.shape[2:])
            flat_v = v_layer.reshape(v_layer.shape[0] * v_layer.shape[1], *v_layer.shape[2:])
            k_view = flat_k[table_row]
            v_view = flat_v[table_row]
            profile_end_cuda(cuda_token)
            if cpu_start is not None:
                profile_add(
                    selection_profile,
                    "kv_view_cpu_ms",
                    (time.perf_counter() - cpu_start) * 1000,
                )
            profile_add(selection_profile, "kv_view_rows", req.physical_cached_len)

        q_observation = None
        q_observation_positions = None
        if self.plan.requires_query_observation:
            cpu_start = time.perf_counter() if selection_profile is not None else None
            cuda_token = profile_start_cuda(selection_profile, "q_read_gpu_ms", device)
            q_observation, q_observation_positions = self.engine.query_ring.get_unordered(
                table_idx=req.table_idx,
                layer_id=layer_id,
                stages=getattr(self.plan, "query_ring_read_stages", None),
            )
            profile_end_cuda(cuda_token)
            if cpu_start is not None:
                profile_add(
                    selection_profile,
                    "q_read_cpu_ms",
                    (time.perf_counter() - cpu_start) * 1000,
                )
            profile_add(selection_profile, "q_observation_rows", q_observation.shape[0])
            if probe_q_by_layer is not None and layer_id in probe_q_by_layer:
                probe_q, probe_positions = probe_q_by_layer[layer_id]
                if getattr(self.plan, "summary_probe_mode", "append") == "replace":
                    q_observation = probe_q
                    q_observation_positions = probe_positions
                else:
                    q_observation = torch.cat([q_observation, probe_q], dim=0)
                    q_observation_positions = torch.cat(
                        [q_observation_positions, probe_positions],
                        dim=0,
                    )

        kv_phase_ids = None
        if getattr(self.plan, "requires_kv_phase_ids", False):
            assert req.phase_ids is not None
            kv_phase_ids = req.phase_ids[: req.physical_cached_len].to(
                device=device,
                dtype=torch.long,
            )

        return LayerEvictionView(
            layer_id=layer_id,
            physical_cached_len=req.physical_cached_len,
            num_kv_heads=num_kv_heads,
            device=device,
            num_layers=self.engine.kv_cache.num_layers,
            logical_current_position=getattr(req, "logical_cached_len", None),
            logical_positions_by_head=logical_positions_by_head,
            k=k_view,
            v=v_view,
            kv_phase_ids=kv_phase_ids,
            q_observation=q_observation,
            q_observation_positions=q_observation_positions,
            profile=selection_profile,
        )

    def compact_session_req(
        self,
        req: Req,
        keep_relative_indices_by_layer_head: list[torch.Tensor] | torch.Tensor,
        *,
        finished: bool = False,
        select_cpu_ms: float | None = None,
        select_gpu_ms: float | None = None,
        selection_profile: dict[str, float] | None = None,
        total_cpu_start: float | None = None,
    ) -> None:
        assert req.session_id is not None
        assert req.retained_positions is not None
        assert req.logical_cached_len is not None
        assert req.logical_device_len is not None

        profile_timing = self.compact_debug_jsonl or self.compact_debug
        prepare_cpu_start = time.perf_counter() if profile_timing else None
        prepare_step_start = prepare_cpu_start
        keep_by_layer = _normalize_layer_keep_indices(
            keep_relative_indices_by_layer_head,
            num_layers=self.engine.kv_cache.num_layers,
            kv_cache=self.engine.kv_cache,
        )
        prepare_normalize_cpu_ms = (
            (time.perf_counter() - prepare_step_start) * 1000
            if prepare_step_start is not None
            else None
        )
        prepare_step_start = time.perf_counter() if profile_timing else None
        compact_len = keep_by_layer[0].shape[1]
        representative_keep = keep_by_layer[0][0].to(dtype=torch.long, device="cpu")
        prepare_representative_cpu_ms = (
            (time.perf_counter() - prepare_step_start) * 1000
            if prepare_step_start is not None
            else None
        )
        prepare_step_start = time.perf_counter() if profile_timing else None
        old_physical_cached_len = req.physical_cached_len
        old_slots = self.engine.page_table[req.table_idx, :old_physical_cached_len].clone()
        available_pages_before = self.cache_manager.available_pages
        in_place_compact = (
            compact_len > available_pages_before
            and self._can_compact_in_place(req, compact_len)
        )
        if in_place_compact:
            new_slots = old_slots[:compact_len].clone()
        else:
            new_slots = self.cache_manager.allocate_token_slots(compact_len)
        prepare_alloc_cpu_ms = (
            (time.perf_counter() - prepare_step_start) * 1000
            if prepare_step_start is not None
            else None
        )
        prepare_step_start = time.perf_counter() if profile_timing else None
        tail_ids = req.input_ids[old_physical_cached_len : req.device_len].clone()
        retained_ids = req.input_ids[:old_physical_cached_len][representative_keep].clone()
        tail_phase_ids = req.phase_ids[old_physical_cached_len : req.device_len].clone()
        retained_phase_ids = req.phase_ids[:old_physical_cached_len][representative_keep].clone()
        tail_semantic_phase_ids = req.semantic_phase_ids[
            old_physical_cached_len : req.device_len
        ].clone()
        retained_semantic_phase_ids = req.semantic_phase_ids[
            :old_physical_cached_len
        ][representative_keep].clone()
        retained_positions = req.retained_positions[representative_keep].clone()
        old_cached_input_ids = None
        old_cached_phase_ids = None
        old_cached_retained_positions = None
        if self.eviction_trace:
            old_cached_input_ids = req.input_ids[:old_physical_cached_len].clone()
            old_cached_phase_ids = req.phase_ids[:old_physical_cached_len].clone()
            old_cached_retained_positions = req.retained_positions[:old_physical_cached_len].clone()
        prepare_request_state_cpu_ms = (
            (time.perf_counter() - prepare_step_start) * 1000
            if prepare_step_start is not None
            else None
        )
        prepare_step_start = time.perf_counter() if profile_timing else None

        needs_position_state = self.compact_debug
        old_positions_by_layer_head = None
        new_positions_by_layer_head = None
        if needs_position_state:
            old_positions_by_layer_head = _retained_positions_by_layer_head(
                req,
                keep_by_layer=keep_by_layer,
            )
            new_positions_by_layer_head = torch.stack(
                [
                    old_positions_by_layer_head[layer_id].gather(1, keep.cpu())
                    for layer_id, keep in enumerate(keep_by_layer)
                ],
                dim=0,
            )
        prepare_position_state_cpu_ms = (
            (time.perf_counter() - prepare_step_start) * 1000
            if prepare_step_start is not None
            else None
        )

        prepare_cpu_ms = (
            (time.perf_counter() - prepare_cpu_start) * 1000
            if prepare_cpu_start is not None
            else None
        )
        cpu_start = time.perf_counter() if profile_timing else None
        gpu_start = None
        gpu_end = None
        if profile_timing and self.device.type == "cuda":
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record(torch.cuda.current_stream(self.device))

        in_place_triton = (
            in_place_compact
            and self.engine.page_table.is_cuda
            and _keep_indices_are_sorted(keep_by_layer)
        )
        if in_place_triton:
            compact_backend = "triton_in_place_sorted"
        elif in_place_compact:
            compact_backend = "torch_in_place"
        else:
            compact_backend = "triton_all_layers" if self.engine.page_table.is_cuda else "torch"
        if compact_backend == "triton_all_layers":
            compact_all_layers_kv_heads_triton(
                kv_cache=self.engine.kv_cache,
                page_table=self.engine.page_table,
                table_idx=req.table_idx,
                physical_cached_len=old_physical_cached_len,
                keep_relative_indices_by_layer_head=torch.stack(keep_by_layer, dim=0),
                new_slots=new_slots,
            )
        elif compact_backend == "triton_in_place_sorted":
            new_slots = compact_all_layers_kv_heads_inplace_sorted_triton(
                kv_cache=self.engine.kv_cache,
                page_table=self.engine.page_table,
                table_idx=req.table_idx,
                physical_cached_len=old_physical_cached_len,
                keep_relative_indices_by_layer_head=torch.stack(keep_by_layer, dim=0),
            )
        else:
            for layer_id, keep in enumerate(keep_by_layer):
                compact_layer_kv_heads(
                    kv_cache=self.engine.kv_cache,
                    page_table=self.engine.page_table,
                    table_idx=req.table_idx,
                    physical_cached_len=old_physical_cached_len,
                    layer_id=layer_id,
                    keep_relative_indices_by_head=keep,
                    new_slots=new_slots,
                )

        compact_gpu_ms = None
        if profile_timing and self.device.type == "cuda":
            if gpu_end is not None:
                gpu_end.record(torch.cuda.current_stream(self.device))
            torch.cuda.current_stream(self.device).synchronize()
            if gpu_start is not None and gpu_end is not None:
                compact_gpu_ms = gpu_start.elapsed_time(gpu_end)
        self.engine.page_table[req.table_idx, :compact_len].copy_(new_slots)
        if compact_len > 0:
            self.token_pool[req.table_idx, :compact_len].copy_(
                retained_ids, non_blocking=False
            )
        if len(tail_ids) > 0:
            self.token_pool[req.table_idx, compact_len : compact_len + len(tail_ids)].copy_(
                tail_ids, non_blocking=False
            )

        if in_place_compact:
            self.cache_manager.free_token_slots(old_slots[compact_len:])
        else:
            self.cache_manager.free_token_slots(old_slots)
        req.input_ids = torch.cat([retained_ids, tail_ids])
        req.phase_ids = torch.cat([retained_phase_ids, tail_phase_ids])
        req.semantic_phase_ids = torch.cat(
            [retained_semantic_phase_ids, tail_semantic_phase_ids]
        )
        req.physical_cached_len = compact_len
        req.device_len = compact_len + len(tail_ids)
        req.max_device_len = req.device_len + req.remain_len
        req.retained_positions = retained_positions
        req.retained_positions_by_layer_head = new_positions_by_layer_head
        compact_cpu_ms = (
            (time.perf_counter() - cpu_start) * 1000 if cpu_start is not None else None
        )

        if self.eviction_trace:
            assert old_cached_input_ids is not None
            assert old_cached_phase_ids is not None
            assert old_cached_retained_positions is not None
            self._write_eviction_trace_artifact(
                req=req,
                trigger=_compact_trigger(req, finished),
                old_physical_cached_len=old_physical_cached_len,
                new_physical_cached_len=req.physical_cached_len,
                keep_by_layer=keep_by_layer,
                old_cached_input_ids=old_cached_input_ids,
                old_cached_phase_ids=old_cached_phase_ids,
                old_cached_retained_positions=old_cached_retained_positions,
            )

        artifact_npz = None
        if self.compact_debug:
            if old_positions_by_layer_head is None:
                old_positions_by_layer_head = _retained_positions_by_layer_head(
                    req,
                    keep_by_layer=keep_by_layer,
                    physical_cached_len=old_physical_cached_len,
                    retained_positions=retained_positions,
                )
            if new_positions_by_layer_head is None:
                new_positions_by_layer_head = torch.stack(
                    [
                        old_positions_by_layer_head[layer_id].gather(1, keep)
                        for layer_id, keep in enumerate(keep_by_layer)
                    ],
                    dim=0,
                )
            artifact_npz = self._write_compact_debug_npz(
                req=req,
                old_slots=old_slots,
                new_slots=new_slots,
                keep_by_layer=keep_by_layer,
                old_positions_by_layer_head=old_positions_by_layer_head,
                new_positions_by_layer_head=new_positions_by_layer_head,
            )

        self._write_compact_debug_summary(
            req=req,
            status="committed",
            error_code=None,
            error_message=None,
            trigger=_compact_trigger(req, finished),
            old_physical_cached_len=old_physical_cached_len,
            new_physical_cached_len=req.physical_cached_len,
            available_pages_before=available_pages_before,
            available_pages_after=self.cache_manager.available_pages,
            needed_new_pages=0 if in_place_compact else compact_len,
            artifact_npz=artifact_npz,
            compact_cpu_ms=compact_cpu_ms,
            compact_gpu_ms=compact_gpu_ms,
            compact_backend=compact_backend,
            select_cpu_ms=select_cpu_ms,
            select_gpu_ms=select_gpu_ms,
            selection_profile=selection_profile,
            prepare_cpu_ms=prepare_cpu_ms,
            prepare_normalize_cpu_ms=prepare_normalize_cpu_ms,
            prepare_representative_cpu_ms=prepare_representative_cpu_ms,
            prepare_alloc_cpu_ms=prepare_alloc_cpu_ms,
            prepare_request_state_cpu_ms=prepare_request_state_cpu_ms,
            prepare_position_state_cpu_ms=prepare_position_state_cpu_ms,
            total_cpu_ms=(
                (time.perf_counter() - total_cpu_start) * 1000
                if total_cpu_start is not None
                else None
            ),
        )

        logger.info_rank0(
            "Compacted session KV uid=%s session_id=%s "
            "old_physical_cached_len=%s new_physical_cached_len=%s "
            "logical_cached_len=%s",
            req.uid,
            req.session_id,
            old_physical_cached_len,
            req.physical_cached_len,
            req.logical_cached_len,
        )

    def _write_compact_debug_summary(
        self,
        *,
        req: Req,
        status: str,
        error_code: str | None,
        error_message: str | None,
        trigger: str,
        old_physical_cached_len: int,
        new_physical_cached_len: int,
        available_pages_before: int,
        available_pages_after: int | None,
        needed_new_pages: int | None = None,
        artifact_npz: str | None,
        compact_cpu_ms: float | None,
        compact_gpu_ms: float | None,
        compact_backend: str = "unknown",
        select_cpu_ms: float | None = None,
        select_gpu_ms: float | None = None,
        selection_profile: dict[str, float] | None = None,
        prepare_cpu_ms: float | None = None,
        prepare_normalize_cpu_ms: float | None = None,
        prepare_representative_cpu_ms: float | None = None,
        prepare_alloc_cpu_ms: float | None = None,
        prepare_request_state_cpu_ms: float | None = None,
        prepare_position_state_cpu_ms: float | None = None,
        total_cpu_ms: float | None = None,
    ) -> None:
        if not self.compact_debug_jsonl and not self.compact_debug:
            return

        import json

        debug_dir = self._compact_debug_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        query_ring_profile_delta = self._query_ring_profile_delta()
        event = {
            "event": "session_cache_compact",
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            "uid": req.uid,
            "session_id": req.session_id,
            "table_idx": req.table_idx,
            "trigger": trigger,
            "eviction_plan": self.plan.name,
            "compact_backend": compact_backend,
            "logical_len": len(req.full_input_ids) if req.full_input_ids is not None else len(req.input_ids),
            "logical_cached_len": req.logical_cached_len,
            "old_physical_cached_len": old_physical_cached_len,
            "new_physical_cached_len": new_physical_cached_len,
            "generated_len": req.generated_len,
            "available_pages_before": available_pages_before,
            "needed_new_pages": (
                new_physical_cached_len if needed_new_pages is None else needed_new_pages
            ),
            "available_pages_after": available_pages_after,
            "kept_count": new_physical_cached_len,
            "dropped_count": old_physical_cached_len - new_physical_cached_len,
            "artifact_npz": artifact_npz,
            "compact_cpu_ms": compact_cpu_ms,
            "compact_gpu_ms": compact_gpu_ms,
            "select_cpu_ms": select_cpu_ms,
            "select_gpu_ms": select_gpu_ms,
            "select_profile": selection_profile,
            "query_ring_profile_delta": query_ring_profile_delta,
            "prepare_cpu_ms": prepare_cpu_ms,
            "prepare_normalize_cpu_ms": prepare_normalize_cpu_ms,
            "prepare_representative_cpu_ms": prepare_representative_cpu_ms,
            "prepare_alloc_cpu_ms": prepare_alloc_cpu_ms,
            "prepare_request_state_cpu_ms": prepare_request_state_cpu_ms,
            "prepare_position_state_cpu_ms": prepare_position_state_cpu_ms,
            "total_cpu_ms": total_cpu_ms,
        }
        with (debug_dir / "compact_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _write_compact_debug_npz(
        self,
        *,
        req: Req,
        old_slots: torch.Tensor,
        new_slots: torch.Tensor,
        keep_by_layer: list[torch.Tensor],
        old_positions_by_layer_head: torch.Tensor,
        new_positions_by_layer_head: torch.Tensor,
    ) -> str:
        import numpy as np

        debug_dir = self._compact_debug_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1_000_000)
        path = debug_dir / f"compact_uid{req.uid}_gen{req.generated_len}_{stamp}.npz"
        keep = torch.stack(keep_by_layer, dim=0)
        keep_mask = torch.zeros(
            old_positions_by_layer_head.shape,
            dtype=torch.bool,
            device=keep.device,
        )
        keep_mask.scatter_(2, keep, True)
        np.savez_compressed(
            path,
            old_slots=old_slots.cpu().numpy(),
            new_slots=new_slots.cpu().numpy(),
            keep_relative_indices_by_layer_head=keep.cpu().numpy(),
            old_logical_positions_by_layer_head=old_positions_by_layer_head.cpu().numpy(),
            kept_logical_positions_by_layer_head=new_positions_by_layer_head.cpu().numpy(),
            keep_mask_by_layer_head=keep_mask.cpu().numpy(),
        )
        return str(path)

    def _compact_debug_dir(self) -> Path:
        return Path(self.compact_debug_dir or "compact_debug")

    def _query_ring_profile_snapshot(self) -> dict[str, int]:
        query_ring = getattr(self.engine, "query_ring", None)
        if query_ring is None or not hasattr(query_ring, "profile_snapshot"):
            return {}
        return query_ring.profile_snapshot()

    def _query_ring_profile_delta(self) -> dict[str, int]:
        query_ring = getattr(self.engine, "query_ring", None)
        if query_ring is None or not hasattr(query_ring, "profile_delta_since"):
            return {}
        delta = query_ring.profile_delta_since(self._last_query_ring_profile)
        self._last_query_ring_profile = query_ring.profile_snapshot()
        return delta

    def _write_eviction_trace_artifact(
        self,
        *,
        req: Req,
        trigger: str,
        old_physical_cached_len: int,
        new_physical_cached_len: int,
        keep_by_layer: list[torch.Tensor],
        old_cached_input_ids: torch.Tensor,
        old_cached_phase_ids: torch.Tensor,
        old_cached_retained_positions: torch.Tensor,
    ) -> str:
        import json
        import numpy as np

        trace_dir = self._eviction_trace_dir()
        trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1_000_000)
        stem = f"eviction_uid{req.uid}_gen{req.generated_len}_{stamp}"
        path = trace_dir / f"{stem}.npz"

        old_tokens = old_cached_input_ids[:old_physical_cached_len].to(device="cpu")
        old_phases = old_cached_phase_ids[:old_physical_cached_len].to(device="cpu")
        old_positions = old_cached_retained_positions[:old_physical_cached_len].to(device="cpu")

        keep_by_layer_head = torch.stack(
            [layer_keep.to(dtype=torch.long, device="cpu") for layer_keep in keep_by_layer],
            dim=0,
        )
        keep_mask_by_layer_head = torch.zeros(
            (
                keep_by_layer_head.shape[0],
                keep_by_layer_head.shape[1],
                old_physical_cached_len,
            ),
            dtype=torch.bool,
        )
        keep_mask_by_layer_head.scatter_(2, keep_by_layer_head, True)
        dropped_mask_by_layer_head = ~keep_mask_by_layer_head
        old_positions_broadcast = old_positions.view(1, 1, -1)
        old_tokens_broadcast = old_tokens.view(1, 1, -1)
        old_phases_broadcast = old_phases.view(1, 1, -1)
        kept_logical_positions_by_layer_head = old_positions_broadcast.expand(
            keep_by_layer_head.shape[0],
            keep_by_layer_head.shape[1],
            -1,
        ).gather(2, keep_by_layer_head)
        kept_token_ids_by_layer_head = old_tokens_broadcast.expand(
            keep_by_layer_head.shape[0],
            keep_by_layer_head.shape[1],
            -1,
        ).gather(2, keep_by_layer_head)
        kept_phase_ids_by_layer_head = old_phases_broadcast.expand(
            keep_by_layer_head.shape[0],
            keep_by_layer_head.shape[1],
            -1,
        ).gather(2, keep_by_layer_head)

        np.savez_compressed(
            path,
            keep_relative_indices_by_layer_head=keep_by_layer_head.to(dtype=torch.int32).numpy(),
            kept_logical_positions_by_layer_head=kept_logical_positions_by_layer_head.to(dtype=torch.int64).numpy(),
            kept_token_ids_by_layer_head=kept_token_ids_by_layer_head.to(dtype=torch.int32).numpy(),
            kept_phase_ids_by_layer_head=kept_phase_ids_by_layer_head.to(dtype=torch.int16).numpy(),
            keep_mask_by_layer_head=keep_mask_by_layer_head.numpy(),
            dropped_mask_by_layer_head=dropped_mask_by_layer_head.numpy(),
            old_logical_positions=old_positions.to(dtype=torch.int64).numpy(),
            old_token_ids=old_tokens.to(dtype=torch.int32).numpy(),
            old_phase_ids=old_phases.to(dtype=torch.int16).numpy(),
            old_physical_cached_len=np.array([old_physical_cached_len], dtype=np.int64),
            new_physical_cached_len=np.array([new_physical_cached_len], dtype=np.int64),
        )

        event = {
            "event": "eviction_trace",
            "schema_version": 2,
            "trace_scope": "all_layer_head",
            "num_layers": int(keep_by_layer_head.shape[0]),
            "num_kv_heads": int(keep_by_layer_head.shape[1]),
            "uid": req.uid,
            "session_id": req.session_id,
            "table_idx": req.table_idx,
            "trigger": trigger,
            "eviction_plan": self.plan.name,
            "generated_len": req.generated_len,
            "logical_len": len(req.full_input_ids) if req.full_input_ids is not None else len(req.input_ids),
            "logical_cached_len": req.logical_cached_len,
            "old_physical_cached_len": old_physical_cached_len,
            "new_physical_cached_len": new_physical_cached_len,
            "kept_count_per_layer_head": new_physical_cached_len,
            "dropped_count_per_layer_head": old_physical_cached_len - new_physical_cached_len,
            "artifact_npz": str(path),
        }
        with (trace_dir / "eviction_trace_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        return str(path)

    def _eviction_trace_dir(self) -> Path:
        return Path(self.eviction_trace_dir or "eviction_trace")


def _compact_trigger(req: Req, finished: bool) -> str:
    if finished:
        return "before_park"
    if req.generated_len == 1:
        return "after_prefill"
    return "decode_interval"


def _normalize_layer_keep_indices(
    keep: list[torch.Tensor] | torch.Tensor,
    *,
    num_layers: int,
    kv_cache,
) -> list[torch.Tensor]:
    if isinstance(keep, torch.Tensor):
        if keep.ndim == 1:
            return [
                keep.to(dtype=torch.long, device=kv_cache.k_cache(layer_id).device)
                .unsqueeze(0)
                .expand(kv_cache.k_cache(layer_id).shape[2], -1)
                for layer_id in range(num_layers)
            ]
        if keep.ndim == 3:
            if keep.shape[0] != num_layers:
                raise ValueError(
                    f"keep tensor has {keep.shape[0]} layers, expected {num_layers}"
                )
            return [
                keep[layer_id].to(dtype=torch.long, device=kv_cache.k_cache(layer_id).device)
                for layer_id in range(num_layers)
            ]
        raise ValueError(f"keep tensor must be 1D or 3D, got {keep.ndim}D")

    if len(keep) != num_layers:
        raise ValueError(f"keep list has {len(keep)} layers, expected {num_layers}")
    return [
        layer_keep.to(dtype=torch.long, device=kv_cache.k_cache(layer_id).device)
        for layer_id, layer_keep in enumerate(keep)
    ]


def _keep_indices_are_sorted(keep_by_layer: list[torch.Tensor]) -> bool:
    for keep in keep_by_layer:
        if keep.shape[1] > 1 and not bool(torch.all(keep[:, 1:] >= keep[:, :-1]).item()):
            return False
    return True


def _retained_positions_for_layer(
    req: Req,
    *,
    layer_id: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if req.retained_positions_by_layer_head is not None:
        return req.retained_positions_by_layer_head[layer_id]
    assert req.retained_positions is not None
    return req.retained_positions.unsqueeze(0).expand(num_kv_heads, -1)


def _retained_positions_by_layer_head(
    req: Req,
    *,
    keep_by_layer: list[torch.Tensor],
    physical_cached_len: int | None = None,
    retained_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    if req.retained_positions_by_layer_head is not None:
        return req.retained_positions_by_layer_head
    base_positions = retained_positions if retained_positions is not None else req.retained_positions
    assert base_positions is not None
    if physical_cached_len is not None and len(base_positions) != physical_cached_len:
        base_positions = torch.arange(physical_cached_len, dtype=torch.int32, device="cpu")
    return torch.stack(
        [base_positions.unsqueeze(0).expand(keep.shape[0], -1) for keep in keep_by_layer],
        dim=0,
    )
