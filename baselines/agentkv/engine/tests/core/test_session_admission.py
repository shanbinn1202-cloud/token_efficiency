from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from minisgl.core import Req, SamplingParams
from minisgl.message import UserMsg
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.session import SessionManager, SessionState
from minisgl.scheduler.utils import PendingReq


class _SessionManager:
    def __init__(self) -> None:
        self.cancelled = 0

    def cancel_prepare(self, req: PendingReq) -> None:
        self.cancelled += 1


class _TableManager:
    available_size = 1


def _scheduler_for_validation(prefill_budget: int = 8) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = prefill_budget
    scheduler.session_manager = _SessionManager()
    scheduler.table_manager = _TableManager()
    return scheduler


def _scheduler_for_admission_control(*, available_size: int, inflight_tokens: int) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 4
    scheduler.prefill_chunk_size = 0
    scheduler.compact_aware_kv_reserve = False
    scheduler.compact_aware_prefill_reserve = False
    scheduler.cache_manager = SimpleNamespace(available_size=available_size, page_size=1)
    scheduler.decode_manager = SimpleNamespace(
        runnable=inflight_tokens > 0,
        inflight_tokens=inflight_tokens,
        running_reqs=[],
    )
    scheduler.eviction_manager = SimpleNamespace(
        disabled=False,
        compact_after_prefill=True,
        compact_interval=4,
        plan=SimpleNamespace(budget=4),
    )
    scheduler.prefill_manager = SimpleNamespace(
        pending_list=[
            PendingReq(
                uid=1,
                input_ids=torch.arange(10, dtype=torch.int32),
                sampling_params=SamplingParams(max_tokens=4),
                session_id="session-a",
            )
        ]
    )
    return scheduler


def test_session_admission_allows_waiting_for_kv_capacity() -> None:
    scheduler = _scheduler_for_validation()
    req = PendingReq(
        uid=1,
        input_ids=torch.arange(6, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1024),
        session_id="session-a",
        physical_cached_len=4,
    )

    scheduler._validate_pending_session_req(req)

    assert scheduler.session_manager.cancelled == 0


def test_session_admission_allows_suffix_larger_than_prefill_budget_for_chunking() -> None:
    scheduler = _scheduler_for_validation(prefill_budget=2)
    req = PendingReq(
        uid=1,
        input_ids=torch.arange(6, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        physical_cached_len=3,
    )

    scheduler._validate_pending_session_req(req)

    assert scheduler.session_manager.cancelled == 0


def test_admission_control_delays_new_long_prefill_until_compact_headroom_exists() -> None:
    """Long prompt admission must leave enough temporary KV room for post-prefill compact."""
    scheduler = _scheduler_for_admission_control(available_size=20, inflight_tokens=8)

    assert scheduler._should_delay_prefill_for_compact_headroom()


def test_admission_control_allows_new_long_prefill_with_sufficient_headroom() -> None:
    """The gate should not throttle new long prompts when compact scratch space is available."""
    scheduler = _scheduler_for_admission_control(available_size=40, inflight_tokens=8)

    assert not scheduler._should_delay_prefill_for_compact_headroom()


def test_admission_control_does_not_stall_active_partial_prefill() -> None:
    """Already-admitted partial chunks must continue so the request can reach compact."""
    scheduler = _scheduler_for_admission_control(available_size=20, inflight_tokens=8)
    scheduler.prefill_manager.pending_list[0].session_req = object()

    assert not scheduler._should_delay_prefill_for_compact_headroom()


def test_admission_control_delays_final_prefill_chunk_without_compact_headroom() -> None:
    """The final prefill chunk should wait if post-prefill compact would OOM."""
    scheduler = _scheduler_for_admission_control(available_size=18, inflight_tokens=8)
    pending = scheduler.prefill_manager.pending_list[0]
    pending.session_req = object()
    pending.physical_cached_len = 7

    assert scheduler._should_delay_prefill_for_compact_headroom()


def test_compact_aware_admission_reserves_only_until_next_compact() -> None:
    """Interval compact keeps future decode KV bounded before max_tokens is exhausted."""
    scheduler = _scheduler_for_admission_control(available_size=20, inflight_tokens=8)
    scheduler.compact_aware_kv_reserve = True
    scheduler.decode_manager.running_reqs = [
        SimpleNamespace(remain_len=8, generated_len=1),
    ]

    assert not scheduler._should_delay_prefill_for_compact_headroom()


def test_compact_aware_admission_still_requires_prompt_pages() -> None:
    """Compact-aware reserve must not admit a prompt that cannot fit its remaining suffix."""
    scheduler = _scheduler_for_admission_control(available_size=12, inflight_tokens=8)
    scheduler.compact_aware_kv_reserve = True
    scheduler.decode_manager.running_reqs = [
        SimpleNamespace(remain_len=8, generated_len=1),
    ]

    assert scheduler._should_delay_prefill_for_compact_headroom()


def test_compact_aware_prefill_reserve_uses_current_chunk() -> None:
    """Chunked prefill admission should not require the full remaining prompt upfront."""
    scheduler = _scheduler_for_admission_control(available_size=12, inflight_tokens=8)
    scheduler.compact_aware_kv_reserve = True
    scheduler.compact_aware_prefill_reserve = True
    scheduler.decode_manager.running_reqs = [
        SimpleNamespace(remain_len=8, generated_len=1),
    ]

    assert not scheduler._should_delay_prefill_for_compact_headroom()


def test_compact_aware_prefill_reserve_still_requires_current_chunk() -> None:
    """The relaxed prompt reserve must still keep enough space for the next chunk."""
    scheduler = _scheduler_for_admission_control(available_size=10, inflight_tokens=8)
    scheduler.compact_aware_kv_reserve = True
    scheduler.compact_aware_prefill_reserve = True
    scheduler.decode_manager.running_reqs = [
        SimpleNamespace(remain_len=8, generated_len=1),
    ]

    assert scheduler._should_delay_prefill_for_compact_headroom()


def test_compact_aware_prefill_reserve_requires_compact_aware_kv_reserve() -> None:
    """Prompt-reserve relaxation is unsafe unless interval-compact KV reserve is active."""
    scheduler = _scheduler_for_admission_control(available_size=12, inflight_tokens=8)
    scheduler.compact_aware_prefill_reserve = True
    scheduler.decode_manager.running_reqs = [
        SimpleNamespace(remain_len=8, generated_len=1),
    ]

    assert scheduler._should_delay_prefill_for_compact_headroom()


def test_scheduler_admission_profile_records_headroom_decision() -> None:
    """Scheduler profile must explain when the compact-headroom gate delays prefill."""
    scheduler = _scheduler_for_admission_control(available_size=10, inflight_tokens=8)
    scheduler.compact_aware_kv_reserve = True
    scheduler.compact_aware_prefill_reserve = True
    scheduler.decode_manager.running_reqs = [
        SimpleNamespace(remain_len=8, generated_len=1),
    ]
    scheduler.prefill_manager.inflight_partials = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "scheduler_admission.jsonl"
        scheduler.scheduler_admission_profile_path = path
        scheduler.scheduler_admission_profile_step = 0

        assert scheduler._should_delay_prefill_for_compact_headroom()

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["reason"] == "headroom"
    assert rows[0]["delay"] is True
    assert rows[0]["prompt_reserve"] == scheduler.prefill_budget


def test_session_continuation_uses_server_tokens_when_text_prefix_matches() -> None:
    """Detokenize/re-tokenize can change token IDs; parked KV must stay authoritative."""
    manager = SessionManager(page_size=1)
    old_req = Req(
        input_ids=torch.tensor([10, 20, 30], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=2,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        session_text="ABC",
        session_prompt_len=2,
    )
    manager.sessions["session-a"] = SessionState(
        session_id="session-a",
        status="PARKED",
        req=old_req,
        full_token_ids=torch.tensor([10, 20, 30], dtype=torch.int32),
        retained_token_ids=torch.tensor([10, 20], dtype=torch.int32),
        retained_positions=torch.tensor([0, 1], dtype=torch.int32),
        logical_cached_len=2,
        full_text="ABC",
    )
    msg = UserMsg(
        uid=2,
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        rendered_text="ABCD",
    )

    pending = manager.prepare_request(
        msg,
        encode_suffix=lambda text: torch.tensor([40], dtype=torch.int32),
    )

    assert pending.physical_cached_len == 2
    assert pending.rendered_text == "ABCD"
    assert pending.input_ids.tolist() == [10, 20, 30, 40]
    assert pending.full_input_ids.tolist() == [10, 20, 30, 40]
    assert pending.retained_positions.tolist() == [0, 1]
    assert pending.logical_cached_len == 2


def test_session_continuation_uses_retained_tokens_after_compaction() -> None:
    """Continuation must decode from physical retained KV, not full logical history."""
    manager = SessionManager(page_size=1)
    old_req = Req(
        input_ids=torch.tensor([10, 50, 60, 30], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=3,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        session_text="ABCDE",
        session_prompt_len=5,
        full_input_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),
        logical_cached_len=5,
        logical_device_len=6,
        retained_positions=torch.tensor([0, 3, 4], dtype=torch.int32),
    )
    manager.sessions["session-a"] = SessionState(
        session_id="session-a",
        status="PARKED",
        req=old_req,
        full_token_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),
        retained_token_ids=torch.tensor([10, 40, 50], dtype=torch.int32),
        retained_positions=torch.tensor([0, 3, 4], dtype=torch.int32),
        logical_cached_len=5,
        full_text="ABCDE",
    )
    msg = UserMsg(
        uid=2,
        input_ids=torch.tensor([10, 20, 30, 40, 50, 70], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        rendered_text="ABCDEF",
    )

    pending = manager.prepare_request(
        msg,
        encode_suffix=lambda text: torch.tensor([70], dtype=torch.int32),
    )

    assert pending.physical_cached_len == 3
    assert pending.input_ids.tolist() == [10, 40, 50, 70]
    assert pending.full_input_ids.tolist() == [10, 20, 30, 40, 50, 70]
    assert pending.retained_positions.tolist() == [0, 3, 4]
    assert pending.logical_cached_len == 5


def test_session_continuation_carries_retained_phase_ids_after_compaction() -> None:
    """StageQ suffix prefill must preserve phases for retained anchors and new suffix."""
    manager = SessionManager(page_size=1)
    old_req = Req(
        input_ids=torch.tensor([10, 50, 60, 30], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=3,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        session_text="ABCDE",
        session_prompt_len=5,
        full_input_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),
        phase_ids=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
        full_phase_ids=torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32),
        logical_cached_len=5,
        logical_device_len=6,
        retained_positions=torch.tensor([0, 3, 4], dtype=torch.int32),
    )
    manager.sessions["session-a"] = SessionState(
        session_id="session-a",
        status="PARKED",
        req=old_req,
        full_token_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),
        retained_token_ids=torch.tensor([10, 40, 50], dtype=torch.int32),
        retained_phase_ids=torch.tensor([0, 2, 3], dtype=torch.int32),
        retained_positions=torch.tensor([0, 3, 4], dtype=torch.int32),
        full_phase_ids=torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32),
        logical_cached_len=5,
        full_text="ABCDE",
    )
    msg = UserMsg(
        uid=2,
        input_ids=torch.tensor([10, 20, 30, 40, 50, 70], dtype=torch.int32),
        phase_ids=torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        rendered_text="ABCDEF",
    )

    pending = manager.prepare_request(
        msg,
        encode_suffix=lambda text: torch.tensor([70], dtype=torch.int32),
    )

    assert pending.phase_ids is not None
    assert pending.full_phase_ids is not None
    assert pending.phase_ids.tolist() == [0, 2, 3, 3]
    assert pending.full_phase_ids.tolist() == [0, 0, 1, 2, 3, 3]


def test_session_continuation_carries_semantic_phase_ids_independently() -> None:
    """Stage-Q can route semantic phases without changing KV phase labels."""
    manager = SessionManager(page_size=1)
    old_req = Req(
        input_ids=torch.tensor([10, 50, 60, 30], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=3,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        session_text="ABCDE",
        session_prompt_len=5,
        full_input_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),
        phase_ids=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
        full_phase_ids=torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32),
        semantic_phase_ids=torch.tensor([0, 2, -1, 1], dtype=torch.int32),
        logical_cached_len=5,
        logical_device_len=6,
        retained_positions=torch.tensor([0, 3, 4], dtype=torch.int32),
    )
    manager.sessions["session-a"] = SessionState(
        session_id="session-a",
        status="PARKED",
        req=old_req,
        full_token_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),
        retained_token_ids=torch.tensor([10, 40, 50], dtype=torch.int32),
        retained_phase_ids=torch.tensor([0, 2, 3], dtype=torch.int32),
        retained_semantic_phase_ids=torch.tensor([0, 2, -1], dtype=torch.int32),
        retained_positions=torch.tensor([0, 3, 4], dtype=torch.int32),
        full_phase_ids=torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32),
        logical_cached_len=5,
        full_text="ABCDE",
    )
    msg = UserMsg(
        uid=2,
        input_ids=torch.tensor([10, 20, 30, 40, 50, 70], dtype=torch.int32),
        phase_ids=torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.int32),
        semantic_phase_ids=torch.tensor([0, -1, 1, 2, -1, 3], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        rendered_text="ABCDEF",
    )

    pending = manager.prepare_request(
        msg,
        encode_suffix=lambda text: torch.tensor([70], dtype=torch.int32),
    )

    assert pending.semantic_phase_ids is not None
    assert pending.semantic_phase_ids.tolist() == [0, 2, -1, 3]


def test_session_text_prefix_fallback_preserves_suffix_phase_ids() -> None:
    """StageQ must not collapse re-encoded suffix tokens into the previous phase."""
    manager = SessionManager(page_size=1)
    old_req = Req(
        input_ids=torch.tensor([10, 20, 30], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=2,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        session_text="ABC",
        session_prompt_len=2,
    )
    manager.sessions["session-a"] = SessionState(
        session_id="session-a",
        status="PARKED",
        req=old_req,
        full_token_ids=torch.tensor([10, 20, 30], dtype=torch.int32),
        retained_token_ids=torch.tensor([10, 20], dtype=torch.int32),
        retained_phase_ids=torch.tensor([0, 1], dtype=torch.int32),
        retained_positions=torch.tensor([0, 1], dtype=torch.int32),
        full_phase_ids=torch.tensor([0, 1, 1], dtype=torch.int32),
        logical_cached_len=2,
        full_text="ABC",
    )
    msg = UserMsg(
        uid=2,
        input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        phase_ids=torch.tensor([0, 1, 1, 3, 3], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        session_id="session-a",
        rendered_text="ABCDE",
    )

    pending = manager.prepare_request(
        msg,
        encode_suffix=lambda text: torch.tensor([40, 50], dtype=torch.int32),
    )

    assert pending.phase_ids is not None
    assert pending.full_phase_ids is not None
    assert pending.full_input_ids.tolist() == [10, 20, 30, 40, 50]
    assert pending.full_phase_ids.tolist() == [0, 1, 1, 3, 3]
    assert pending.phase_ids.tolist() == [0, 1, 1, 3, 3]
