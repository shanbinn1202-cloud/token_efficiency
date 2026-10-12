from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.eviction import EvictionManager
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    CloseSessionBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    TokenizerErrorMsg,
    UserMsg,
)
from minisgl.phase import SemanticPhase, route_semantic_phases_to_query_stages
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import PrefillManager
from .session import SessionError, SessionManager
from .table import TableManager
from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (req_mapping, physical_positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.session_manager = SessionManager(config.page_size)
        self.engine.query_ring.set_profile_enabled(
            config.compact_debug_jsonl or config.compact_debug
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.prefill_chunk_size = config.prefill_chunk_size
        self.compact_aware_kv_reserve = config.compact_aware_kv_reserve
        self.compact_aware_prefill_reserve = config.compact_aware_prefill_reserve
        self.eviction_manager = EvictionManager(
            config=config,
            engine=self.engine,
            cache_manager=self.cache_manager,
            token_pool=self.token_pool,
            device=self.device,
        )
        self.query_stage_mapping = getattr(
            self.eviction_manager.plan,
            "query_stage_mapping",
            "legacy",
        )
        self.scheduler_profile_path = (
            Path(config.compact_debug_dir or "compact_debug") / "scheduler_batches.jsonl"
            if ENV.SCHEDULER_PROFILE
            else None
        )
        self.scheduler_admission_profile_path = (
            Path(config.compact_debug_dir or "compact_debug") / "scheduler_admission.jsonl"
            if ENV.SCHEDULER_PROFILE
            else None
        )
        self.scheduler_forward_profile_path = (
            Path(config.compact_debug_dir or "compact_debug") / "scheduler_forward.jsonl"
            if ENV.SCHEDULER_PROFILE
            else None
        )
        self.scheduler_process_profile_path = (
            Path(config.compact_debug_dir or "compact_debug") / "scheduler_process.jsonl"
            if ENV.SCHEDULER_PROFILE
            else None
        )
        self.scheduler_profile_step = 0
        self.scheduler_admission_profile_step = 0
        self.scheduler_forward_profile_step = 0
        self.scheduler_process_profile_step = 0
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def _encode_session_suffix(self, text: str) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(text, add_special_tokens=False),
            dtype=torch.int32,
            device="cpu",
        )

    def _advance_phase_state(self, req: Req, next_token: int) -> tuple[int, str]:
        assert req.phase_ids is not None
        current_phase = int(req.phase_ids[-1].item()) if len(req.phase_ids) > 0 else 0
        token_text = self.tokenizer.decode([next_token])
        tail = (req.phase_text_tail + token_text)[-128:]
        return _phase_from_tail(tail, current_phase), tail

    def _advance_semantic_phase_state(self, req: Req, next_token: int, tail: str) -> int:
        assert req.semantic_phase_ids is not None
        current_phase = (
            int(req.semantic_phase_ids[-1].item())
            if len(req.semantic_phase_ids) > 0
            else int(SemanticPhase.UNKNOWN)
        )
        return _semantic_phase_from_tail(tail, current_phase)

    def _session_full_text(self, req: Req, *, drop_last_token: bool) -> str | None:
        if req.session_text is None or req.session_prompt_len is None:
            return None
        req.materialize_full_host()
        full_input_ids = req.full_input_ids if req.full_input_ids is not None else req.input_ids
        end = len(full_input_ids) - (1 if drop_last_token else 0)
        output_ids = full_input_ids[req.session_prompt_len : end]
        output_text = self.tokenizer.decode(
            output_ids.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return req.session_text + output_text

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity(owned_pages=self.session_manager.owned_pages())

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Compaction mutates the active request's page table, so process the
        # previous batch before scheduling another forward for the same request.
        if self.eviction_manager.compacts_active_req:
            self._process_last_data(last_data)
            last_data = None

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        if self.scheduler_process_profile_path is not None:
            self._process_last_data_profiled(last_data)
            return

        batch, forward_output = last_data[0].batch, last_data[1]
        next_tokens_cpu = forward_output.next_tokens_cpu
        copy_done = forward_output.copy_done_event
        copy_done.synchronize()
        self._record_forward_profile(batch, forward_output)
        reply: List[BaseTokenizerMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if req.prefill_only:
                    self.prefill_manager.complete_partial_req(req)
                    continue
                next_token = next_tokens_cpu[i]
                next_token = int(next_token.item())
                phase_id, phase_text_tail = self._advance_phase_state(req, next_token)
                semantic_phase_id = self._advance_semantic_phase_state(
                    req,
                    next_token,
                    phase_text_tail,
                )
                req.append_host(
                    next_tokens_cpu[i].unsqueeze(0),
                    phase_id=phase_id,
                    semantic_phase_id=semantic_phase_id,
                    phase_text_tail=phase_text_tail,
                )
                finished = not req.can_decode
                finished_by_eos = False
                if not req.sampling_params.ignore_eos:
                    finished_by_eos = next_token == self.eos_token_id
                    finished |= finished_by_eos

                compact_error = self._maybe_compact_session_req(req, finished=finished)
                if compact_error is not None:
                    self.decode_manager.remove_req(req)
                    if req.session_id is not None:
                        self.session_manager.remove_running(req)
                    self._free_req_resources(req)
                    reply.append(compact_error)
                    new_finished_reqs.add(req)
                    continue

                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    if req.session_id is not None:
                        should_free = self.session_manager.finish_request(
                            req,
                            drop_last_token=finished_by_eos,
                            full_text=self._session_full_text(
                                req, drop_last_token=finished_by_eos
                            ),
                        )
                        if should_free:
                            self._free_req_resources(req)
                    else:
                        self._free_req_resources(req)
                    new_finished_reqs.add(req)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_last_data_profiled(self, last_data: ForwardData) -> None:
        batch, forward_output = last_data[0].batch, last_data[1]
        next_tokens_cpu = forward_output.next_tokens_cpu
        copy_done = forward_output.copy_done_event
        process_timing: dict[str, float] = {}
        counts: dict[str, int] = {
            "prefill_only_reqs": 0,
            "decode_reqs": 0,
            "finished_reqs": 0,
            "compact_error_reqs": 0,
            "freed_reqs": 0,
            "reply_count": 0,
        }

        def add_elapsed(name: str, started_s: float) -> None:
            process_timing[name] = process_timing.get(name, 0.0) + (
                time.perf_counter() - started_s
            ) * 1000

        total_started_s = time.perf_counter()
        step_time_s = total_started_s
        section_started_s = time.perf_counter()
        copy_done.synchronize()
        add_elapsed("copy_wait_ms", section_started_s)
        section_started_s = time.perf_counter()
        self._record_forward_profile(batch, forward_output)
        add_elapsed("record_forward_profile_ms", section_started_s)

        reply: List[BaseTokenizerMsg] = []
        new_finished_reqs: Set[Req] = set()
        lazy_region_started_s = time.perf_counter()
        with self.cache_manager.lazy_free_region():
            for req_index, req in enumerate(batch.reqs):
                if req.prefill_only:
                    counts["prefill_only_reqs"] += 1
                    section_started_s = time.perf_counter()
                    self.prefill_manager.complete_partial_req(req)
                    add_elapsed("prefill_complete_ms", section_started_s)
                    continue
                counts["decode_reqs"] += 1
                section_started_s = time.perf_counter()
                next_token_tensor = next_tokens_cpu[req_index]
                next_token = int(next_token_tensor.item())
                add_elapsed("token_extract_ms", section_started_s)

                section_started_s = time.perf_counter()
                phase_id, phase_text_tail = self._advance_phase_state(req, next_token)
                semantic_phase_id = self._advance_semantic_phase_state(
                    req,
                    next_token,
                    phase_text_tail,
                )
                add_elapsed("phase_update_ms", section_started_s)

                section_started_s = time.perf_counter()
                req.append_host(
                    next_token_tensor.unsqueeze(0),
                    phase_id=phase_id,
                    semantic_phase_id=semantic_phase_id,
                    phase_text_tail=phase_text_tail,
                )
                add_elapsed("append_host_ms", section_started_s)

                section_started_s = time.perf_counter()
                finished = not req.can_decode
                finished_by_eos = False
                if not req.sampling_params.ignore_eos:
                    finished_by_eos = next_token == self.eos_token_id
                    finished |= finished_by_eos
                add_elapsed("finish_check_ms", section_started_s)

                section_started_s = time.perf_counter()
                compact_error = self._maybe_compact_session_req(req, finished=finished)
                add_elapsed("compact_ms", section_started_s)
                if compact_error is not None:
                    counts["compact_error_reqs"] += 1
                    section_started_s = time.perf_counter()
                    self.decode_manager.remove_req(req)
                    if req.session_id is not None:
                        self.session_manager.remove_running(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    counts["freed_reqs"] += 1
                    add_elapsed("compact_error_cleanup_ms", section_started_s)

                    section_started_s = time.perf_counter()
                    reply.append(compact_error)
                    add_elapsed("reply_build_ms", section_started_s)
                    continue

                section_started_s = time.perf_counter()
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))
                add_elapsed("reply_build_ms", section_started_s)

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    counts["finished_reqs"] += 1
                    section_started_s = time.perf_counter()
                    self.decode_manager.remove_req(req)
                    add_elapsed("decode_remove_ms", section_started_s)
                    if req.session_id is not None:
                        section_started_s = time.perf_counter()
                        full_text = self._session_full_text(
                            req, drop_last_token=finished_by_eos
                        )
                        add_elapsed("session_full_text_ms", section_started_s)
                        section_started_s = time.perf_counter()
                        should_free = self.session_manager.finish_request(
                            req,
                            drop_last_token=finished_by_eos,
                            full_text=full_text,
                        )
                        add_elapsed("session_finish_ms", section_started_s)
                        if should_free:
                            section_started_s = time.perf_counter()
                            self._free_req_resources(req)
                            counts["freed_reqs"] += 1
                            add_elapsed("free_req_resources_ms", section_started_s)
                    else:
                        section_started_s = time.perf_counter()
                        self._free_req_resources(req)
                        counts["freed_reqs"] += 1
                        add_elapsed("free_req_resources_ms", section_started_s)
                    new_finished_reqs.add(req)

        add_elapsed("lazy_free_region_total_ms", lazy_region_started_s)
        counts["reply_count"] = len(reply)
        section_started_s = time.perf_counter()
        self.finished_reqs = new_finished_reqs
        self.send_result(reply)
        add_elapsed("send_result_ms", section_started_s)
        add_elapsed("total_ms", total_started_s)
        self._record_process_profile(
            batch,
            time_s=step_time_s,
            process_timing=process_timing,
            counts=counts,
        )

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                error = (
                    f"Input sequence length {input_len} exceeds maximum sequence length "
                    f"{max_seq_len}."
                )
                logger.warning_rank0("%s Request %s is rejected.", error, msg.uid)
                self.send_result([TokenizerErrorMsg(uid=msg.uid, error=error, status_code=400)])
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            pending_req = None
            try:
                pending_req = self.session_manager.prepare_request(
                    msg, encode_suffix=self._encode_session_suffix
                )
                input_len = pending_req.logical_input_len
                max_output_len = max_seq_len - input_len
                if max_output_len <= 0:
                    error = (
                        f"Input sequence length {input_len} exceeds maximum sequence length "
                        f"{max_seq_len}."
                    )
                    raise SessionError(error)
                if msg.sampling_params.max_tokens > max_output_len:
                    msg.sampling_params.max_tokens = max_output_len
                    logger.warning_rank0(
                        f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                    )
                self._validate_pending_session_req(pending_req)
            except SessionError as exc:
                logger.warning_rank0("Request %s is rejected: %s", msg.uid, exc)
                self.send_result(
                    [TokenizerErrorMsg(uid=msg.uid, error=str(exc), status_code=400)]
                )
                return
            except RuntimeError as exc:
                logger.warning_rank0("Request %s is rejected: %s", msg.uid, exc)
                if pending_req is not None:
                    self.session_manager.cancel_prepare(pending_req)
                self.send_result(
                    [TokenizerErrorMsg(uid=msg.uid, error=str(exc), status_code=400)]
                )
                return
            self.prefill_manager.add_pending_req(pending_req)
        elif isinstance(msg, CloseSessionBackendMsg):
            logger.debug_rank0("Closing session %s", msg.session_id)
            try:
                req = self.session_manager.close(msg.session_id)
            except SessionError as exc:
                self.send_result(
                    [TokenizerErrorMsg(uid=msg.uid, error=str(exc), status_code=400)]
                )
                return
            if req is not None:
                self.cache_manager.free_req_pages(req)
                self.table_manager.free(req.table_idx)
            self.send_result(
                [DetokenizeMsg(uid=msg.uid, next_token=self.eos_token_id, finished=True)]
            )
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            if isinstance(req_to_free, PendingReq):
                self.session_manager.cancel_prepare(req_to_free)
                return
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                if req_to_free.session_id is not None:
                    self.session_manager.remove_running(req_to_free)
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.engine.query_ring.reset(req.table_idx)
        self.table_manager.free(req.table_idx)
        self.cache_manager.free_req_pages(req)

    def _validate_pending_session_req(self, req) -> None:
        suffix_len = req.input_len - req.physical_cached_len
        if suffix_len <= 0:
            self.session_manager.cancel_prepare(req)
            raise RuntimeError("Session request has no new prompt tokens to prefill")
        if req.session_req is None and self.table_manager.available_size == 0:
            self.session_manager.cancel_prepare(req)
            raise RuntimeError("No free request table slots")

    def _maybe_compact_session_req(
        self,
        req: Req,
        *,
        finished: bool,
    ) -> TokenizerErrorMsg | None:
        probe_q_by_layer = None
        if (
            self.eviction_manager.should_compact(req, finished=finished)
            and getattr(self.eviction_manager.plan, "requires_summary_probe", False)
        ):
            probe_q_by_layer = self._run_summary_probe(req)
        return self.eviction_manager.maybe_compact(
            req,
            finished=finished,
            probe_q_by_layer=probe_q_by_layer,
        )

    def _summary_probe_token_ids(self, max_tokens: int) -> torch.Tensor:
        probe_text = "\nThe summary of the current context is:"
        token_ids = self.tokenizer.encode(probe_text, add_special_tokens=False)
        if max_tokens > 0:
            token_ids = token_ids[:max_tokens]
        return torch.tensor(token_ids, dtype=torch.int32, device="cpu")

    def _run_summary_probe(self, req: Req) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
        probe_tokens = int(getattr(self.eviction_manager.plan, "summary_probe_tokens", 0))
        probe_ids = self._summary_probe_token_ids(probe_tokens)
        if probe_ids.numel() == 0 or self.engine.query_ring.capacity <= 0:
            return None
        if self.table_manager.available_size <= 0:
            logger.warning_rank0(
                "Skipping summary probe for uid=%s because no request table slot is free",
                req.uid,
            )
            return None
        if self.cache_manager.available_pages < probe_ids.numel():
            logger.warning_rank0(
                "Skipping summary probe for uid=%s because KV pages are low "
                "(need=%s, available=%s)",
                req.uid,
                probe_ids.numel(),
                self.cache_manager.available_pages,
            )
            return None

        temp_slot = self.table_manager.allocate()
        old_physical_cached_len = req.physical_cached_len
        probe_len = int(probe_ids.numel())
        device_probe_ids = probe_ids.pin_memory().to(self.device, non_blocking=True)
        allocated_probe_pages = False
        try:
            if old_physical_cached_len > 0:
                self.engine.page_table[temp_slot, :old_physical_cached_len].copy_(
                    self.engine.page_table[req.table_idx, :old_physical_cached_len]
                )
            self.token_pool[
                temp_slot,
                old_physical_cached_len : old_physical_cached_len + probe_len,
            ].copy_(device_probe_ids, non_blocking=True)

            prefix_ids = torch.zeros(old_physical_cached_len, dtype=torch.int32, device="cpu")
            input_ids = torch.cat([prefix_ids, probe_ids])
            assert req.retained_positions is not None
            assert req.logical_cached_len is not None
            if req.phase_ids is None:
                prefix_phase_ids = torch.zeros(
                    old_physical_cached_len,
                    dtype=torch.int32,
                    device="cpu",
                )
            else:
                prefix_phase_ids = req.phase_ids[:old_physical_cached_len].clone()
            probe_stage_ids = (
                torch.arange(probe_len, dtype=torch.int32, device="cpu")
                % max(1, self.engine.query_ring.num_stages)
            )
            phase_ids = torch.cat([prefix_phase_ids, probe_stage_ids])
            semantic_phase_ids = phase_ids.clone()
            fake_req = Req(
                input_ids=input_ids,
                table_idx=temp_slot,
                physical_cached_len=old_physical_cached_len,
                output_len=1,
                uid=-1,
                sampling_params=req.sampling_params,
                session_id=req.session_id,
                phase_ids=phase_ids,
                semantic_phase_ids=semantic_phase_ids,
                logical_cached_len=req.logical_cached_len,
                logical_device_len=req.logical_cached_len + probe_len,
                retained_positions=req.retained_positions.clone(),
            )
            batch = Batch(reqs=[fake_req], phase="prefill")
            forward_input = self._prepare_batch(batch)
            allocated_probe_pages = True
            if forward_input is None:
                return None
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                self._forward(forward_input)
            self.engine.stream.synchronize()

            probe_q_by_layer = {}
            for layer_id in range(self.engine.kv_cache.num_layers):
                q, positions = self.engine.query_ring.get_unordered(
                    table_idx=temp_slot,
                    layer_id=layer_id,
                    stages=None,
                )
                probe_q_by_layer[layer_id] = (q.clone(), positions.clone())
            return probe_q_by_layer
        finally:
            end = old_physical_cached_len + probe_len
            if allocated_probe_pages and probe_len > 0:
                probe_slots = self.engine.page_table[temp_slot, old_physical_cached_len:end].clone()
                self.cache_manager.free_token_slots(probe_slots)
            self.engine.query_ring.reset(temp_slot)
            self.engine.page_table[temp_slot].zero_()
            self.table_manager.free(temp_slot)

    def _compact_session_req(self, req: Req, keep_relative_indices: torch.Tensor) -> None:
        self.eviction_manager.compact_session_req(req, keep_relative_indices)

    def _prepare_batch(self, batch: Batch) -> ForwardInput | None:
        prepare_start_s = time.perf_counter()
        prepare_timing: dict[str, float] = {}
        if batch.is_decode:
            self._drop_oldest_decode_reqs_until_kv_fits(batch)
        prepare_timing["drop_kv_ms"] = (time.perf_counter() - prepare_start_s) * 1000
        if not batch.reqs:
            return None
        cache_available_before = self.cache_manager.available_size
        cache_pages_before = self.cache_manager.available_pages
        cache_needed_pages = self.cache_manager.needed_pages(batch.reqs)
        cuda_graph_candidate = self.engine.graph_runner.can_use_cuda_graph(batch)
        section_start_s = time.perf_counter()
        self.engine.graph_runner.pad_batch(batch)
        prepare_timing["pad_batch_ms"] = (time.perf_counter() - section_start_s) * 1000
        section_start_s = time.perf_counter()
        self.cache_manager.allocate_paged(batch.reqs)
        prepare_timing["allocate_paged_ms"] = (time.perf_counter() - section_start_s) * 1000
        section_start_s = time.perf_counter()
        batch.positions = _make_positions(batch, self.device)
        prepare_timing["make_positions_ms"] = (time.perf_counter() - section_start_s) * 1000
        section_start_s = time.perf_counter()
        if self.engine.query_ring.capacity <= 0:
            batch.stage_ids = torch.zeros_like(batch.positions, dtype=torch.int32)
        else:
            batch.stage_ids = _make_stage_ids(batch, self.device, self.query_stage_mapping)
        prepare_timing["make_stage_ids_ms"] = (time.perf_counter() - section_start_s) * 1000
        section_start_s = time.perf_counter()
        batch.query_ring_metadata = self.engine.query_ring.prepare_write_metadata_from_reqs(
            batch.padded_reqs,
            batch.positions,
            batch.stage_ids,
        )
        prepare_timing["query_ring_metadata_ms"] = (
            time.perf_counter() - section_start_s
        ) * 1000
        self.engine.query_ring.record_write_metadata_stats(
            phase=batch.phase,
            active_rows=sum(req.extend_len for req in batch.reqs),
            padded_rows=sum(req.extend_len for req in batch.padded_reqs),
            written_rows=batch.query_ring_metadata.table_indices_by_token.numel(),
        )
        section_start_s = time.perf_counter()
        input_mapping = _make_input_tuple(batch, self.device)
        prepare_timing["make_input_tuple_ms"] = (
            time.perf_counter() - section_start_s
        ) * 1000
        section_start_s = time.perf_counter()
        write_mapping = _make_write_tuple(batch, self.device)
        prepare_timing["make_write_tuple_ms"] = (
            time.perf_counter() - section_start_s
        ) * 1000
        section_start_s = time.perf_counter()
        batch.out_loc = self.engine.page_table[input_mapping]
        prepare_timing["out_loc_ms"] = (time.perf_counter() - section_start_s) * 1000
        section_start_s = time.perf_counter()
        self.engine.attn_backend.prepare_metadata(batch)
        prepare_timing["attn_prepare_metadata_ms"] = (
            time.perf_counter() - section_start_s
        ) * 1000
        prepare_timing["total_ms"] = (time.perf_counter() - prepare_start_s) * 1000
        self._record_scheduler_profile(
            batch,
            cache_available_before=cache_available_before,
            cache_pages_before=cache_pages_before,
            cache_needed_pages=cache_needed_pages,
            cuda_graph_candidate=cuda_graph_candidate,
            time_s=prepare_start_s,
            prepare_timing=prepare_timing,
        )
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _drop_oldest_decode_reqs_until_kv_fits(self, batch: Batch) -> None:
        while (
            batch.reqs
            and self.cache_manager.needed_pages(batch.reqs) > self.cache_manager.available_pages
        ):
            needed_pages = self.cache_manager.needed_pages(batch.reqs)
            available_pages = self.cache_manager.available_pages
            victim = min(batch.reqs, key=lambda req: req.uid)
            batch.reqs.remove(victim)
            self.decode_manager.remove_req(victim)
            if victim.session_id is not None:
                self.session_manager.remove_running(victim)
            self._free_req_resources(victim)
            error = (
                "SESSION_CACHE_KV_PRESSURE: aborted oldest running decode request "
                "because KV pages were exhausted while scheduling decode "
                f"(victim_uid={victim.uid}, needed_pages={needed_pages}, "
                f"available_pages={available_pages})"
            )
            logger.warning_rank0(error)
            self.send_result(
                [TokenizerErrorMsg(uid=victim.uid, error=error, status_code=503)]
            )

    def _record_scheduler_profile(
        self,
        batch: Batch,
        *,
        cache_available_before: int,
        cache_pages_before: int,
        cache_needed_pages: int,
        cuda_graph_candidate: bool,
        time_s: float,
        prepare_timing: dict[str, float],
    ) -> None:
        if self.scheduler_profile_path is None:
            return
        self.scheduler_profile_step += 1
        event = {
            "event": "scheduler_batch",
            "step": self.scheduler_profile_step,
            "time_s": time_s,
            "phase": batch.phase,
            "batch_size": batch.size,
            "padded_size": batch.padded_size,
            "active_extend_tokens": sum(req.extend_len for req in batch.reqs),
            "padded_extend_tokens": sum(req.extend_len for req in batch.padded_reqs),
            "prefill_only_reqs": sum(1 for req in batch.reqs if req.prefill_only),
            "pending_prefill_reqs": len(self.prefill_manager.pending_list),
            "inflight_partial_reqs": len(self.prefill_manager.inflight_partials),
            "decode_running_reqs": len(self.decode_manager.running_reqs),
            "cache_available_before": int(cache_available_before),
            "cache_pages_before": int(cache_pages_before),
            "cache_needed_pages": int(cache_needed_pages),
            "cuda_graph_candidate": bool(cuda_graph_candidate),
            "prepare_timing_ms": prepare_timing,
        }
        self.scheduler_profile_path.parent.mkdir(parents=True, exist_ok=True)
        with self.scheduler_profile_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _record_forward_profile(self, batch: Batch, forward_output: ForwardOutput) -> None:
        path = getattr(self, "scheduler_forward_profile_path", None)
        if path is None:
            return
        self.scheduler_forward_profile_step += 1
        forward_ms = forward_output.forward_start_event.elapsed_time(
            forward_output.logits_ready_event
        )
        sample_copy_ms = forward_output.logits_ready_event.elapsed_time(
            forward_output.copy_done_event
        )
        total_ms = forward_output.forward_start_event.elapsed_time(
            forward_output.copy_done_event
        )
        event = {
            "event": "scheduler_forward",
            "step": self.scheduler_forward_profile_step,
            "time_s": time.perf_counter(),
            "phase": batch.phase,
            "batch_size": batch.size,
            "padded_size": batch.padded_size,
            "active_extend_tokens": sum(req.extend_len for req in batch.reqs),
            "used_cuda_graph": bool(forward_output.used_cuda_graph),
            "forward_ms": float(forward_ms),
            "sample_copy_ms": float(sample_copy_ms),
            "total_ms": float(total_ms),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _record_process_profile(
        self,
        batch: Batch,
        *,
        time_s: float,
        process_timing: dict[str, float],
        counts: dict[str, int],
    ) -> None:
        path = getattr(self, "scheduler_process_profile_path", None)
        if path is None:
            return
        self.scheduler_process_profile_step += 1
        event = {
            "event": "scheduler_process",
            "step": self.scheduler_process_profile_step,
            "time_s": time_s,
            "phase": batch.phase,
            "batch_size": batch.size,
            "padded_size": batch.padded_size,
            "active_extend_tokens": sum(req.extend_len for req in batch.reqs),
            "process_timing_ms": process_timing,
            "counts": counts,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _record_admission_profile(
        self,
        *,
        reason: str,
        delay: bool,
        pending_req: PendingReq,
        remaining_prompt: int,
        prompt_reserve: int,
        pending_output_reserve: int,
        compact_budget: int,
        reserved_decode_tokens: int,
        required_headroom: int,
    ) -> None:
        path = getattr(self, "scheduler_admission_profile_path", None)
        if path is None:
            return
        self.scheduler_admission_profile_step += 1
        event = {
            "event": "scheduler_admission",
            "step": self.scheduler_admission_profile_step,
            "reason": reason,
            "delay": delay,
            "pending_uid": pending_req.uid,
            "pending_has_session_req": pending_req.session_req is not None,
            "remaining_prompt": int(remaining_prompt),
            "prompt_reserve": int(prompt_reserve),
            "pending_output_reserve": int(pending_output_reserve),
            "compact_budget": int(compact_budget),
            "reserved_decode_tokens": int(reserved_decode_tokens),
            "required_headroom": int(required_headroom),
            "cache_available_size": int(self.cache_manager.available_size),
            "decode_running_reqs": len(self.decode_manager.running_reqs),
            "pending_prefill_reqs": len(self.prefill_manager.pending_list),
            "inflight_partial_reqs": len(self.prefill_manager.inflight_partials),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        if self._should_delay_prefill_for_compact_headroom():
            batch = (
                self.decode_manager.schedule_next_batch()
                or self.prefill_manager.schedule_next_batch(
                    self.prefill_budget,
                    self.prefill_chunk_size,
                    reserved_decode_tokens=self._decode_reserved_tokens_for_prefill(),
                    output_reserve_tokens=self._pending_output_reserve_tokens(),
                    reserve_remaining_prompt=not self._uses_compact_aware_prefill_reserve(),
                )
            )
        else:
            batch = (
                self.prefill_manager.schedule_next_batch(
                    self.prefill_budget,
                    self.prefill_chunk_size,
                    reserved_decode_tokens=self._decode_reserved_tokens_for_prefill(),
                    output_reserve_tokens=self._pending_output_reserve_tokens(),
                    reserve_remaining_prompt=not self._uses_compact_aware_prefill_reserve(),
                )
                or self.decode_manager.schedule_next_batch()
            )
        return self._prepare_batch(batch) if batch else None

    def _compact_reserve_budget(self) -> int:
        compact_budget = int(getattr(self.eviction_manager.plan, "budget", 0) or 0)
        if not self.compact_aware_kv_reserve:
            return compact_budget
        if getattr(self.cache_manager, "page_size", 1) == 1:
            return 0
        return compact_budget

    def _pending_output_reserve_tokens(self) -> int | None:
        if not self._uses_compact_aware_kv_reserve():
            return None
        interval = int(getattr(self.eviction_manager, "compact_interval", 0) or 0)
        return interval if interval > 0 else None

    def _pending_prompt_reserve_tokens(self, pending_req: PendingReq) -> int:
        remaining_prompt = pending_req.input_len - pending_req.physical_cached_len
        if not self._uses_compact_aware_prefill_reserve():
            return remaining_prompt
        chunk_cap = self.prefill_chunk_size if self.prefill_chunk_size > 0 else self.prefill_budget
        return min(remaining_prompt, self.prefill_budget, chunk_cap)

    def _decode_reserved_tokens_for_prefill(self) -> int:
        if not self._uses_compact_aware_kv_reserve():
            return self.decode_manager.inflight_tokens
        page_reserve = (getattr(self.cache_manager, "page_size", 1) - 1) * len(
            self.decode_manager.running_reqs
        )
        return page_reserve + sum(
            self._decode_req_output_reserve(req) for req in self.decode_manager.running_reqs
        )

    def _decode_req_output_reserve(self, req: Req) -> int:
        interval = int(getattr(self.eviction_manager, "compact_interval", 0) or 0)
        if interval <= 0:
            return req.remain_len
        if req.generated_len <= 0 and self.eviction_manager.compact_after_prefill:
            return min(req.remain_len, 1)
        tokens_into_interval = req.generated_len % interval
        tokens_until_compact = interval - tokens_into_interval if tokens_into_interval else interval
        return min(req.remain_len, tokens_until_compact)

    def _uses_compact_aware_kv_reserve(self) -> bool:
        return (
            self.compact_aware_kv_reserve
            and not self.eviction_manager.disabled
            and int(getattr(self.eviction_manager, "compact_interval", 0) or 0) > 0
        )

    def _uses_compact_aware_prefill_reserve(self) -> bool:
        return self.compact_aware_prefill_reserve and self._uses_compact_aware_kv_reserve()

    def _should_delay_prefill_for_compact_headroom(self) -> bool:
        if not self.decode_manager.runnable or not self.prefill_manager.pending_list:
            return False
        if self.eviction_manager.disabled or not self.eviction_manager.compact_after_prefill:
            return False

        pending_req = self.prefill_manager.pending_list[0]
        compact_budget = self._compact_reserve_budget()
        if compact_budget <= 0:
            compact_budget = 0

        remaining_prompt = pending_req.input_len - pending_req.physical_cached_len
        plan_budget = int(getattr(self.eviction_manager.plan, "budget", 0) or 0)
        if plan_budget > 0 and pending_req.input_len <= plan_budget:
            self._record_admission_profile(
                reason="small_prompt",
                delay=False,
                pending_req=pending_req,
                remaining_prompt=remaining_prompt,
                prompt_reserve=remaining_prompt,
                pending_output_reserve=pending_req.output_len,
                compact_budget=compact_budget,
                reserved_decode_tokens=self._decode_reserved_tokens_for_prefill(),
                required_headroom=remaining_prompt + pending_req.output_len + compact_budget,
            )
            return False
        if pending_req.session_req is not None and remaining_prompt > self.prefill_budget:
            self._record_admission_profile(
                reason="continue_partial",
                delay=False,
                pending_req=pending_req,
                remaining_prompt=remaining_prompt,
                prompt_reserve=min(remaining_prompt, self.prefill_budget),
                pending_output_reserve=0,
                compact_budget=compact_budget,
                reserved_decode_tokens=self._decode_reserved_tokens_for_prefill(),
                required_headroom=min(remaining_prompt, self.prefill_budget),
            )
            return False

        output_reserve_tokens = self._pending_output_reserve_tokens()
        pending_output_reserve = (
            pending_req.output_len
            if output_reserve_tokens is None
            else min(pending_req.output_len, output_reserve_tokens)
        )
        prompt_reserve = self._pending_prompt_reserve_tokens(pending_req)
        required_headroom = prompt_reserve + pending_output_reserve + compact_budget
        reserved_decode_tokens = self._decode_reserved_tokens_for_prefill()
        delay = required_headroom + reserved_decode_tokens > self.cache_manager.available_size
        self._record_admission_profile(
            reason="headroom",
            delay=delay,
            pending_req=pending_req,
            remaining_prompt=remaining_prompt,
            prompt_reserve=prompt_reserve,
            pending_output_reserve=pending_output_reserve,
            compact_budget=compact_budget,
            reserved_decode_tokens=reserved_decode_tokens,
            required_headroom=required_headroom,
        )
        return delay

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        write_positions = output_mapping[1]
        write_mask = write_positions >= 0
        if torch.any(write_mask):
            self.token_pool[(output_mapping[0][write_mask], write_positions[write_mask])] = (
                forward_output.next_tokens_gpu[write_mask]
            )
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        start = req.logical_cached_len if req.logical_cached_len is not None else req.physical_cached_len
        end = req.logical_device_len if req.logical_device_len is not None else req.device_len
        if end - start != length:
            raise RuntimeError(
                f"logical position range length {end - start} does not match extend_len "
                f"{length} for request {req.uid}"
            )
        torch.arange(
            start,
            end,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_stage_ids(
    batch: Batch,
    device: torch.device,
    query_stage_mapping: str = "legacy",
) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    stages_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        start = req.physical_cached_len
        end = req.device_len
        assert req.phase_ids is not None
        assert req.semantic_phase_ids is not None
        if end - start != length:
            raise RuntimeError(
                f"physical phase range length {end - start} does not match extend_len "
                f"{length} for request {req.uid}"
            )
        if query_stage_mapping == "legacy":
            stage_slice = req.phase_ids[start:end]
        else:
            semantic_slice = req.semantic_phase_ids[start:end]
            stage_slice = route_semantic_phases_to_query_stages(
                semantic_slice,
                query_stage_mapping,
            )
        if len(stage_slice) != length:
            if req.uid >= 0:
                raise RuntimeError(
                    f"stage id slice length {len(stage_slice)} does not match extend_len "
                    f"{length} for request {req.uid}"
                )
            stages_host[offset : offset + length].zero_()
        else:
            stages_host[offset : offset + length].copy_(stage_slice)
        offset += length
    return stages_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    mapping_host = torch.empty(needed_size, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_size, dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        start = req.physical_cached_len
        end = req.device_len
        if end - start != length:
            raise RuntimeError(
                f"physical position range length {end - start} does not match extend_len "
                f"{length} for request {req.uid}"
            )
        mapping_host[offset : offset + length].fill_(req.table_idx)
        torch.arange(
            start,
            end,
            dtype=torch.int64,
            out=positions_host[offset : offset + length],
        )
        offset += length
    return mapping_host.to(device, non_blocking=True), positions_host.to(device, non_blocking=True)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)


def _phase_from_tail(tail: str, current_phase: int) -> int:
    markers = (
        ("<|im_start|>system", 0),
        ("<|im_start|>user", 0),
        ("<|im_start|>assistant", 1),
        ("<think>", 1),
        ("</think>", 2),
        ("<tool_call>", 2),
        ("</tool_call>", 2),
        ("<tool_response>", 3),
        ("</tool_response>", 3),
        ("<|im_end|>", 0),
    )
    phase = current_phase
    last_pos = -1
    for marker, marker_phase in markers:
        pos = tail.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            phase = marker_phase
    return phase


def _semantic_phase_from_tail(tail: str, current_phase: int) -> int:
    markers = (
        ("<|im_start|>system", int(SemanticPhase.SYSTEM)),
        ("<|im_start|>user", int(SemanticPhase.USER)),
        ("<|im_start|>assistant", int(SemanticPhase.ASSISTANT_TEXT)),
        ("<think>", int(SemanticPhase.ASSISTANT_TEXT)),
        ("</think>", int(SemanticPhase.ASSISTANT_TEXT)),
        ("<tool_response>", int(SemanticPhase.TOOL_RESPONSE)),
        ("</tool_response>", int(SemanticPhase.TOOL_RESPONSE)),
        ("<tool_call>", int(SemanticPhase.ASSISTANT_TOOL_CALL)),
        ("</tool_call>", int(SemanticPhase.ASSISTANT_TOOL_CALL)),
        ("<|im_end|>", current_phase),
    )
    phase = current_phase
    last_pos = -1
    for marker, marker_phase in markers:
        pos = tail.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            phase = marker_phase
    return phase
