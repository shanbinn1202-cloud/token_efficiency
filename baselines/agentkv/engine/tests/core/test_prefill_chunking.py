from __future__ import annotations

from types import SimpleNamespace

import torch

from minisgl.core import SamplingParams
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import _make_write_tuple
from minisgl.scheduler.utils import PendingReq


class _CacheManager:
    available_size = 128


class _SmallCacheManager:
    available_size = 16


class _DecodeManager:
    inflight_tokens = 0


class _TableManager:
    def __init__(self, slots: int = 2) -> None:
        self._free_slots = list(range(1, slots))
        self.token_pool = torch.zeros((slots, 32), dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)


def _pending_req() -> PendingReq:
    return PendingReq(
        uid=7,
        input_ids=torch.arange(10, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
        session_id="session-a",
        phase_ids=torch.arange(10, dtype=torch.int32),
        semantic_phase_ids=torch.arange(10, dtype=torch.int32) + 100,
    )


def test_prefill_manager_requeues_intermediate_prompt_chunks() -> None:
    """Intermediate chunk prefill must advance cached prompt state without decoding."""
    manager = PrefillManager(
        cache_manager=_CacheManager(),
        table_manager=_TableManager(),
        decode_manager=_DecodeManager(),
    )
    pending = _pending_req()
    manager.add_pending_req(pending)

    batch = manager.schedule_next_batch(prefill_budget=4)

    assert batch is not None
    req = batch.reqs[0]
    assert req.prefill_only
    assert not req.can_decode
    assert req.input_ids.tolist() == [0, 1, 2, 3]
    assert req.phase_ids is not None
    assert req.semantic_phase_ids is not None
    assert req.phase_ids.tolist() == [0, 1, 2, 3]
    assert req.semantic_phase_ids.tolist() == [100, 101, 102, 103]
    assert req.physical_cached_len == 0
    assert manager.pending_list == []

    req.complete_prefill_chunk()
    manager.complete_partial_req(req)

    assert manager.pending_list == [pending]
    assert pending.session_req is req
    assert pending.physical_cached_len == 4
    assert pending.logical_cached_len == 4
    assert pending.retained_positions.tolist() == [0, 1, 2, 3]
    assert req.generated_len == 0
    assert req.device_len == 4

    next_batch = manager.schedule_next_batch(prefill_budget=8)

    assert next_batch is not None
    final_req = next_batch.reqs[0]
    assert not final_req.prefill_only
    assert final_req.can_decode
    assert final_req.physical_cached_len == 4
    assert final_req.input_ids.tolist() == list(range(10))


def test_prefill_only_chunks_have_no_output_write_slot() -> None:
    """Intermediate prefill chunks must not write a sampled token into token_pool."""
    manager = PrefillManager(
        cache_manager=_CacheManager(),
        table_manager=_TableManager(),
        decode_manager=_DecodeManager(),
    )
    pending = _pending_req()
    manager.add_pending_req(pending)
    batch = manager.schedule_next_batch(prefill_budget=4)

    assert batch is not None
    batch.padded_reqs = batch.reqs
    _, write_positions = _make_write_tuple(batch, torch.device("cpu"))

    assert write_positions.tolist() == [-1]


def test_final_chunk_of_chunked_request_does_not_admit_next_long_prompt() -> None:
    """Post-prefill compact needs KV headroom before another long prompt chunk is admitted."""
    manager = PrefillManager(
        cache_manager=_CacheManager(),
        table_manager=_TableManager(),
        decode_manager=_DecodeManager(),
    )
    final_chunk = PendingReq(
        uid=1,
        input_ids=torch.arange(10, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
        session_id="session-a",
        session_req=SimpleNamespace(table_idx=0),
        physical_cached_len=8,
        logical_cached_len=8,
    )
    next_long = PendingReq(
        uid=2,
        input_ids=torch.arange(20, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
        session_id="session-b",
    )
    manager.add_pending_req(final_chunk)
    manager.add_pending_req(next_long)

    batch = manager.schedule_next_batch(prefill_budget=8)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [1]
    assert not batch.reqs[0].prefill_only
    assert manager.pending_list == [next_long]


def test_prefill_chunk_size_batches_multiple_partial_prompts() -> None:
    """A per-request chunk cap should turn one 8192-token budget into multiple chunks."""
    manager = PrefillManager(
        cache_manager=_CacheManager(),
        table_manager=_TableManager(slots=4),
        decode_manager=_DecodeManager(),
    )
    reqs = [
        PendingReq(
            uid=uid,
            input_ids=torch.arange(10, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=2),
            session_id=f"session-{uid}",
            phase_ids=torch.arange(10, dtype=torch.int32),
            semantic_phase_ids=torch.arange(10, dtype=torch.int32),
        )
        for uid in (1, 2, 3)
    ]
    for req in reqs:
        manager.add_pending_req(req)

    batch = manager.schedule_next_batch(prefill_budget=8, prefill_chunk_size=4)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [1, 2]
    assert [req.extend_len for req in batch.reqs] == [4, 4]
    assert all(req.prefill_only for req in batch.reqs)
    assert manager.pending_list == [reqs[2]]
    assert sorted(manager.inflight_partials) == [1, 2]


def test_prefill_chunk_reserve_batches_by_current_chunk() -> None:
    """Compact-aware admission can reserve current chunks instead of whole prompts."""
    manager = PrefillManager(
        cache_manager=_SmallCacheManager(),
        table_manager=_TableManager(slots=4),
        decode_manager=_DecodeManager(),
    )
    reqs = [
        PendingReq(
            uid=uid,
            input_ids=torch.arange(10, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=4),
            session_id=f"session-{uid}",
            phase_ids=torch.arange(10, dtype=torch.int32),
            semantic_phase_ids=torch.arange(10, dtype=torch.int32),
        )
        for uid in (1, 2)
    ]
    for req in reqs:
        manager.add_pending_req(req)

    full_prompt_batch = manager.schedule_next_batch(
        prefill_budget=8,
        prefill_chunk_size=4,
        output_reserve_tokens=4,
        reserve_remaining_prompt=True,
    )

    assert full_prompt_batch is not None
    assert [req.uid for req in full_prompt_batch.reqs] == [1]

    manager = PrefillManager(
        cache_manager=_SmallCacheManager(),
        table_manager=_TableManager(slots=4),
        decode_manager=_DecodeManager(),
    )
    for req in reqs:
        manager.add_pending_req(req)

    current_chunk_batch = manager.schedule_next_batch(
        prefill_budget=8,
        prefill_chunk_size=4,
        output_reserve_tokens=4,
        reserve_remaining_prompt=False,
    )

    assert current_chunk_batch is not None
    assert [req.uid for req in current_chunk_batch.reqs] == [1, 2]
