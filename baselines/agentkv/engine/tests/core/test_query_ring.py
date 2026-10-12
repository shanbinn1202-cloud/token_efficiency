from __future__ import annotations

import torch

from minisgl.eviction.query_ring import QueryRingBufferPool
from minisgl.phase import (
    QueryRingRoute,
    SemanticPhase,
    route_semantic_phases_to_query_stages,
)
from minisgl.scheduler.scheduler import _phase_from_tail, _semantic_phase_from_tail
from minisgl.tokenizer.tokenize import _char_phase_ids, _char_semantic_phase_ids


class _Req:
    def __init__(self, *, table_idx: int, extend_len: int) -> None:
        self.table_idx = table_idx
        self.extend_len = extend_len


def test_disabled_query_ring_returns_empty_metadata() -> None:
    """Capacity 0 is the default benchmark path and must not touch hot-path state."""
    pool = QueryRingBufferPool(
        max_table_slots=2,
        num_layers=1,
        capacity=0,
        num_q_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    metadata = pool.prepare_write_metadata_from_reqs(
        [_Req(table_idx=0, extend_len=2)],
        torch.tensor([0, 1], dtype=torch.int32),
    )

    assert metadata.table_indices_by_token.numel() == 0
    assert metadata.ring_locations_by_token.numel() == 0
    assert metadata.positions_by_token.numel() == 0


def test_query_ring_profile_counts_rows_by_scheduler_phase() -> None:
    """Collect-Q pruning needs before/after row counts split by prefill and decode."""
    pool = QueryRingBufferPool(
        max_table_slots=2,
        num_layers=3,
        capacity=4,
        num_q_heads=1,
        head_dim=1,
        num_stages=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool.set_profile_enabled(True)

    metadata = pool.prepare_write_metadata_from_reqs(
        [_Req(table_idx=0, extend_len=3), _Req(table_idx=1, extend_len=1)],
        positions_by_token=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        stage_ids_by_token=torch.tensor([0, -1, 1, 0], dtype=torch.int32),
    )
    pool.record_write_metadata_stats(
        phase="prefill",
        active_rows=3,
        padded_rows=4,
        written_rows=metadata.table_indices_by_token.numel(),
    )
    pool.record_write_metadata_stats(
        phase="decode",
        active_rows=1,
        padded_rows=2,
        written_rows=2,
    )

    snapshot = pool.profile_snapshot()

    assert snapshot["prefill_batches"] == 1
    assert snapshot["prefill_active_rows"] == 3
    assert snapshot["prefill_padded_rows"] == 4
    assert snapshot["prefill_written_rows"] == 3
    assert snapshot["prefill_estimated_layer_write_rows"] == 9
    assert snapshot["decode_batches"] == 1
    assert snapshot["decode_estimated_layer_write_rows"] == 6
    assert pool.profile_delta_since({"prefill_written_rows": 1})["prefill_written_rows"] == 2


def test_query_ring_appends_flat_tokens_and_preserves_positions() -> None:
    """Flat writer must preserve logical positions for later suffix-prefill checks."""
    pool = QueryRingBufferPool(
        max_table_slots=4,
        num_layers=2,
        capacity=4,
        num_q_heads=2,
        head_dim=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    metadata = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([1, 1, 1], dtype=torch.long),
        positions_by_token=torch.tensor([10, 11, 12], dtype=torch.int32),
    )
    q_layer0 = torch.arange(3 * 2 * 3, dtype=torch.float32).view(3, 2, 3)
    q_layer1 = q_layer0 + 100
    pool.write_layer(layer_id=0, q=q_layer0, metadata=metadata)
    pool.write_layer(layer_id=1, q=q_layer1, metadata=metadata)

    q0, positions = pool.get_ordered(table_idx=1, layer_id=0)
    q1, _ = pool.get_ordered(table_idx=1, layer_id=1)

    assert positions.tolist() == [10, 11, 12]
    assert torch.equal(q0, q_layer0)
    assert torch.equal(q1, q_layer1)
    assert int(pool.write_pos[1].item()) == 3
    assert int(pool.size[1].item()) == 3


def test_query_ring_wraps_and_ordered_read_restores_chronology() -> None:
    """Ring overwrite is allowed, but ordered reads must expose old-to-new observations."""
    pool = QueryRingBufferPool(
        max_table_slots=2,
        num_layers=1,
        capacity=4,
        num_q_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    first = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0, 0], dtype=torch.long),
        positions_by_token=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
        metadata=first,
    )
    second = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0, 0], dtype=torch.long),
        positions_by_token=torch.tensor([3, 4, 5], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[3.0]], [[4.0]], [[5.0]]]),
        metadata=second,
    )

    q_ordered, positions_ordered = pool.get_ordered(table_idx=0, layer_id=0)
    q_unordered, positions_unordered = pool.get_unordered(table_idx=0, layer_id=0)

    assert positions_ordered.tolist() == [2, 3, 4, 5]
    assert q_ordered.flatten().tolist() == [2.0, 3.0, 4.0, 5.0]
    assert sorted(positions_unordered.tolist()) == [2, 3, 4, 5]
    assert sorted(q_unordered.flatten().tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_query_ring_keeps_independent_stage_buffers() -> None:
    """StageQ needs one FIFO query ring per phase, carried by the table slot."""
    pool = QueryRingBufferPool(
        max_table_slots=1,
        num_layers=1,
        capacity=2,
        num_q_heads=1,
        head_dim=1,
        num_stages=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    first = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        stage_ids_by_token=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        positions_by_token=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]]),
        metadata=first,
    )
    second = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0], dtype=torch.long),
        stage_ids_by_token=torch.tensor([0, 1], dtype=torch.int32),
        positions_by_token=torch.tensor([4, 5], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[4.0]], [[5.0]]]),
        metadata=second,
    )

    q_ordered, positions_ordered = pool.get_ordered(table_idx=0, layer_id=0)

    assert positions_ordered.tolist() == [2, 4, 3, 5]
    assert q_ordered.flatten().tolist() == [2.0, 4.0, 3.0, 5.0]
    assert pool.write_pos[0].tolist() == [1, 1]
    assert pool.size[0].tolist() == [2, 2]


def test_query_ring_full_unordered_read_uses_flat_view() -> None:
    """Hot Stage-Q reads full rings; this must avoid gather+cat GPU work."""
    pool = QueryRingBufferPool(
        max_table_slots=1,
        num_layers=1,
        capacity=2,
        num_q_heads=1,
        head_dim=1,
        num_stages=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    metadata = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        stage_ids_by_token=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        positions_by_token=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]]),
        metadata=metadata,
    )

    q, positions = pool.get_unordered(table_idx=0, layer_id=0)

    assert positions.tolist() == [0, 2, 1, 3]
    assert q.flatten().tolist() == [0.0, 2.0, 1.0, 3.0]
    assert q.data_ptr() == pool.q[0, 0, 0, 0].data_ptr()
    assert positions.data_ptr() == pool.positions[0, 0, 0].data_ptr()


def test_query_ring_can_read_stage_subset_for_ablation() -> None:
    """StageQ phase ablations keep all rings written but expose only selected phases."""
    pool = QueryRingBufferPool(
        max_table_slots=1,
        num_layers=1,
        capacity=2,
        num_q_heads=1,
        head_dim=1,
        num_stages=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    metadata = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        stage_ids_by_token=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        positions_by_token=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[10.0]], [[11.0]], [[12.0]], [[13.0]]]),
        metadata=metadata,
    )

    q, positions = pool.get_ordered(table_idx=0, layer_id=0, stages=(0, 3))

    assert positions.tolist() == [10, 13]
    assert q.flatten().tolist() == [10.0, 13.0]
    assert pool.size[0].tolist() == [1, 1, 1, 1]


def test_query_ring_ignores_negative_stage_ids_without_polluting_stage_zero() -> None:
    """Ignored query-stage tokens must not overwrite real Stage-Q anchors."""
    pool = QueryRingBufferPool(
        max_table_slots=1,
        num_layers=1,
        capacity=2,
        num_q_heads=1,
        head_dim=1,
        num_stages=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    first = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0], dtype=torch.long),
        stage_ids_by_token=torch.tensor([0, 1], dtype=torch.int32),
        positions_by_token=torch.tensor([10, 11], dtype=torch.int32),
    )
    pool.write_layer(layer_id=0, q=torch.tensor([[[10.0]], [[11.0]]]), metadata=first)
    ignored = pool.prepare_write_metadata(
        table_indices_by_token=torch.tensor([0, 0], dtype=torch.long),
        stage_ids_by_token=torch.tensor([-1, -1], dtype=torch.int32),
        positions_by_token=torch.tensor([20, 21], dtype=torch.int32),
    )
    pool.write_layer(layer_id=0, q=torch.tensor([[[20.0]], [[21.0]]]), metadata=ignored)

    q, positions = pool.get_ordered(table_idx=0, layer_id=0)

    assert ignored.table_indices_by_token.numel() == 0
    assert positions.tolist() == [10, 11]
    assert q.flatten().tolist() == [10.0, 11.0]
    assert pool.size[0].tolist() == [1, 1, 0, 0]


def test_query_ring_carries_stage_buffers_across_suffix_writes() -> None:
    """Session suffix prefill appends new stage anchors without resetting old rings."""
    pool = QueryRingBufferPool(
        max_table_slots=1,
        num_layers=1,
        capacity=2,
        num_q_heads=1,
        head_dim=1,
        num_stages=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    first = pool.prepare_write_metadata_from_reqs(
        [_Req(table_idx=0, extend_len=2)],
        stage_ids_by_token=torch.tensor([0, 1], dtype=torch.int32),
        positions_by_token=torch.tensor([0, 1], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[0.0]], [[1.0]]]),
        metadata=first,
    )
    suffix = pool.prepare_write_metadata_from_reqs(
        [_Req(table_idx=0, extend_len=2)],
        stage_ids_by_token=torch.tensor([3, 3], dtype=torch.int32),
        positions_by_token=torch.tensor([10, 11], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[10.0]], [[11.0]]]),
        metadata=suffix,
    )

    q_ordered, positions_ordered = pool.get_ordered(table_idx=0, layer_id=0)

    assert positions_ordered.tolist() == [0, 1, 10, 11]
    assert q_ordered.flatten().tolist() == [0.0, 1.0, 10.0, 11.0]
    assert pool.size[0].tolist() == [1, 1, 0, 2]
    assert pool.write_pos[0].tolist() == [1, 1, 0, 0]


def test_query_ring_prunes_overwritten_prefill_writes_exactly() -> None:
    """Long prefill only needs to scatter Q rows that remain visible in the ring."""
    pool = QueryRingBufferPool(
        max_table_slots=1,
        num_layers=1,
        capacity=3,
        num_q_heads=1,
        head_dim=1,
        num_stages=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    metadata = pool.prepare_write_metadata_from_reqs(
        [_Req(table_idx=0, extend_len=7)],
        positions_by_token=torch.arange(7, dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.arange(7, dtype=torch.float32).view(7, 1, 1),
        metadata=metadata,
    )

    q_ordered, positions_ordered = pool.get_ordered(table_idx=0, layer_id=0)

    assert metadata.source_token_indices.tolist() == [4, 5, 6]
    assert metadata.ring_locations_by_token.tolist() == [1, 2, 0]
    assert pool.write_pos[0].tolist() == [1]
    assert pool.size[0].tolist() == [3]
    assert positions_ordered.tolist() == [4, 5, 6]
    assert q_ordered.flatten().tolist() == [4.0, 5.0, 6.0]


def test_single_stage_query_ring_ignores_stage_ids_and_keeps_dummy_slot_separate() -> None:
    """SnapKV uses one stage, so phase labels and graph padding must be semantic no-ops."""
    pool = QueryRingBufferPool(
        max_table_slots=3,
        num_layers=1,
        capacity=4,
        num_q_heads=1,
        head_dim=1,
        num_stages=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    metadata = pool.prepare_write_metadata_from_reqs(
        [_Req(table_idx=0, extend_len=2), _Req(table_idx=2, extend_len=1)],
        positions_by_token=torch.tensor([10, 11, 999], dtype=torch.int32),
        stage_ids_by_token=torch.tensor([3, 2, 1], dtype=torch.int32),
    )
    pool.write_layer(
        layer_id=0,
        q=torch.tensor([[[10.0]], [[11.0]], [[999.0]]]),
        metadata=metadata,
    )

    q_real, positions_real = pool.get_ordered(table_idx=0, layer_id=0)
    q_dummy, positions_dummy = pool.get_ordered(table_idx=2, layer_id=0)

    assert positions_real.tolist() == [10, 11]
    assert q_real.flatten().tolist() == [10.0, 11.0]
    assert positions_dummy.tolist() == [999]
    assert q_dummy.flatten().tolist() == [999.0]
    assert pool.write_pos[0].tolist() == [2]
    assert pool.write_pos[2].tolist() == [1]


def test_phase_detector_routes_chat_markers_to_stage_ids() -> None:
    """Tokenizer-visible markers define the phase id written into the query ring."""
    prompt = (
        "<|im_start|>system\npolicy<|im_end|>\n"
        "<|im_start|>assistant\n<think>x</think><tool_call>{}</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>ok</tool_response><|im_end|>\n"
    )

    phase_ids = _char_phase_ids(prompt)

    assert phase_ids[prompt.index("policy")] == 0
    assert phase_ids[prompt.index("x")] == 1
    assert phase_ids[prompt.index("{}")] == 2
    assert phase_ids[prompt.index("ok")] == 3


def test_semantic_phase_detector_routes_bfcl_tool_result_user_message() -> None:
    """BFCL encodes tool results as a user repr; detect semantics before ring routing."""
    prompt = (
        "<|im_start|>system\npolicy<|im_end|>\n"
        "<|im_start|>user\nquestion<|im_end|>\n"
        "<|im_start|>assistant\n<think>x</think><tool_call>{}</tool_call><|im_end|>\n"
        "<|im_start|>user\n[{'role': 'tool', 'name': 'search', 'content': 'ok'}]<|im_end|>\n"
    )

    semantic = _char_semantic_phase_ids(prompt)

    assert semantic[prompt.index("policy")] == int(SemanticPhase.SYSTEM)
    assert semantic[prompt.index("question")] == int(SemanticPhase.USER)
    assert semantic[prompt.index("x")] == int(SemanticPhase.ASSISTANT_TEXT)
    assert semantic[prompt.index("{}")] == int(SemanticPhase.ASSISTANT_TOOL_CALL)
    assert semantic[prompt.index("ok")] == int(SemanticPhase.TOOL_RESPONSE)


def test_semantic_phase_detector_routes_bfcl_assistant_python_call_as_tool_call() -> None:
    """BFCL prompt-mode stores executed tool calls as assistant text in Python-call syntax."""
    prompt = (
        "<|im_start|>system\npolicy<|im_end|>\n"
        "<|im_start|>user\nquestion<|im_end|>\n"
        "<|im_start|>assistant\n[ls(a=True), grep(pattern='x', file_name='a.txt')]<|im_end|>\n"
        "<|im_start|>assistant\nThe task is done.<|im_end|>\n"
    )

    semantic = _char_semantic_phase_ids(prompt)

    assert semantic[prompt.index("ls(a=True)")] == int(SemanticPhase.ASSISTANT_TOOL_CALL)
    assert semantic[prompt.index("The task is done")] == int(SemanticPhase.ASSISTANT_TEXT)


def test_stageq_fix_mappings_route_semantic_phases_to_ring_stages() -> None:
    """Stage-Q fixes are routing policies over canonical semantic phases."""
    semantic = torch.tensor(
        [
            int(SemanticPhase.SYSTEM),
            int(SemanticPhase.USER),
            int(SemanticPhase.ASSISTANT_TEXT),
            int(SemanticPhase.ASSISTANT_TOOL_CALL),
            int(SemanticPhase.TOOL_RESPONSE),
        ],
        dtype=torch.int32,
    )

    fix1 = route_semantic_phases_to_query_stages(semantic, "fix1")
    fix2 = route_semantic_phases_to_query_stages(semantic, "fix2")
    fix3 = route_semantic_phases_to_query_stages(semantic, "fix3")

    assert fix1.tolist() == [
        int(QueryRingRoute.STAGE0),
        int(QueryRingRoute.STAGE3),
        int(QueryRingRoute.STAGE1),
        int(QueryRingRoute.STAGE1),
        int(QueryRingRoute.STAGE2),
    ]
    assert fix2.tolist() == [
        int(QueryRingRoute.STAGE0),
        int(QueryRingRoute.IGNORE),
        int(QueryRingRoute.STAGE1),
        int(QueryRingRoute.STAGE3),
        int(QueryRingRoute.STAGE2),
    ]
    assert fix3.tolist() == [
        int(QueryRingRoute.STAGE0),
        int(QueryRingRoute.IGNORE),
        int(QueryRingRoute.STAGE1),
        int(QueryRingRoute.STAGE1),
        int(QueryRingRoute.STAGE2),
    ]


def test_decode_phase_state_advances_when_marker_completes() -> None:
    """Generated markers update the current phase used by subsequent query writes."""
    assert _phase_from_tail("abc</think>", 1) == 2
    assert _phase_from_tail("abc<tool_response>", 0) == 3


def test_decode_semantic_phase_state_detects_tool_call_independently() -> None:
    """Decode detects semantic phase; Stage-Q mapping decides final ring stage later."""
    assert _semantic_phase_from_tail(
        "abc<tool_call>",
        int(SemanticPhase.ASSISTANT_TEXT),
    ) == int(SemanticPhase.ASSISTANT_TOOL_CALL)
