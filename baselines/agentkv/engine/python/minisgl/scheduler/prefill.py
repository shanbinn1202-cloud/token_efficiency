from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


@dataclass
class PrefillAdder:
    token_budget: int
    prefill_chunk_size: int
    output_reserve_tokens: int | None
    reserve_remaining_prompt: bool
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _output_reserve(self, req: PendingReq) -> int:
        if self.output_reserve_tokens is None or self.output_reserve_tokens <= 0:
            return req.output_len
        return min(req.output_len, self.output_reserve_tokens)

    def _try_allocate_one(self, req: PendingReq) -> Tuple[int, int, int, bool] | None:
        if req.session_id is None:
            raise RuntimeError("session_id is required")

        physical_cached_len = req.physical_cached_len
        remain_len = req.input_len - physical_cached_len
        chunk_cap = self.prefill_chunk_size if self.prefill_chunk_size > 0 else self.token_budget
        chunk_size = min(remain_len, self.token_budget, chunk_cap)
        if chunk_size <= 0:
            return None
        prefill_only = chunk_size < remain_len
        if req.session_req is None:
            if self.table_manager.available_size == 0:
                return None
            table_idx = self.table_manager.allocate()
        else:
            table_idx = req.session_req.table_idx

        prompt_reserve = remain_len if self.reserve_remaining_prompt else chunk_size
        estimated_len = prompt_reserve + self._output_reserve(req)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            if req.session_req is None:
                self.table_manager.free(table_idx)
            return None
        return table_idx, physical_cached_len, chunk_size, prefill_only

    def _add_one_req(
        self,
        pending_req: PendingReq,
        table_idx: int,
        physical_cached_len: int,
        chunk_size: int,
        prefill_only: bool,
    ) -> Req:
        remain_len = pending_req.input_len - physical_cached_len
        prompt_reserve = remain_len if self.reserve_remaining_prompt else chunk_size
        self.token_budget -= chunk_size
        self.reserved_size += prompt_reserve + self._output_reserve(pending_req)
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(physical_cached_len, physical_cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        current_device_len = physical_cached_len + chunk_size
        session_prompt_len = pending_req.session_prompt_len or pending_req.logical_input_len
        return Req(
            input_ids=pending_req.input_ids[:current_device_len],
            table_idx=table_idx,
            physical_cached_len=physical_cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            session_id=pending_req.session_id,
            session_text=pending_req.rendered_text,
            session_prompt_len=session_prompt_len,
            full_input_ids=pending_req.full_input_ids,
            phase_ids=(
                pending_req.phase_ids[:current_device_len]
                if pending_req.phase_ids is not None
                else None
            ),
            full_phase_ids=pending_req.full_phase_ids,
            semantic_phase_ids=(
                pending_req.semantic_phase_ids[:current_device_len]
                if pending_req.semantic_phase_ids is not None
                else None
            ),
            retained_positions=pending_req.retained_positions,
            retained_positions_by_layer_head=pending_req.retained_positions_by_layer_head,
            logical_cached_len=pending_req.logical_cached_len,
            logical_device_len=pending_req.logical_cached_len + chunk_size,
            sampling_params=pending_req.sampling_params,
            phase_text_tail=pending_req.phase_text_tail,
            prefill_only=prefill_only,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if resource := self._try_allocate_one(pending_req):
            table_idx, physical_cached_len, chunk_size, prefill_only = resource
            return self._add_one_req(
                pending_req=pending_req,
                table_idx=table_idx,
                physical_cached_len=physical_cached_len,
                chunk_size=chunk_size,
                prefill_only=prefill_only,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    inflight_partials: dict[int, PendingReq] = field(default_factory=dict)

    def add_pending_req(self, req: PendingReq) -> None:
        self.pending_list.append(req)

    def schedule_next_batch(
        self,
        prefill_budget: int,
        prefill_chunk_size: int = 0,
        reserved_decode_tokens: int | None = None,
        output_reserve_tokens: int | None = None,
        reserve_remaining_prompt: bool = True,
    ) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            prefill_chunk_size=prefill_chunk_size,
            output_reserve_tokens=output_reserve_tokens,
            reserve_remaining_prompt=reserve_remaining_prompt,
            reserved_size=(
                self.decode_manager.inflight_tokens
                if reserved_decode_tokens is None
                else reserved_decode_tokens
            ),
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        allow_multi_chunk_batch = prefill_chunk_size > 0
        reqs: List[Req] = []
        scheduled_count = 0
        for pending_req in self.pending_list:
            if reqs and adder.token_budget <= 0:
                break
            if req := adder.try_add_one(pending_req):
                reqs.append(req)
                scheduled_count += 1
                if req.prefill_only:
                    self.inflight_partials[req.uid] = pending_req
                    if not allow_multi_chunk_batch:
                        break
                if pending_req.session_req is not None and not allow_multi_chunk_batch:
                    break
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = self.pending_list[scheduled_count:]
        return Batch(reqs=reqs, phase="prefill")

    def complete_partial_req(self, req: Req) -> None:
        pending_req = self.inflight_partials.pop(req.uid)
        pending_req.session_req = req
        pending_req.physical_cached_len = req.physical_cached_len
        pending_req.retained_positions = req.retained_positions
        pending_req.retained_positions_by_layer_head = req.retained_positions_by_layer_head
        assert req.logical_cached_len is not None
        pending_req.logical_cached_len = req.logical_cached_len
        self.pending_list.insert(0, pending_req)

    def abort_req(self, uid: int) -> Req | PendingReq | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.session_req or req
        if uid in self.inflight_partials:
            req = self.inflight_partials.pop(uid)
            return req.session_req or req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
