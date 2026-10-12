import torch

from minisgl.core import Req, SamplingParams


def test_append_host_defers_full_logical_history_until_materialized() -> None:
    req = Req(
        input_ids=torch.tensor([10, 20, 30], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=2,
        output_len=4,
        uid=7,
        sampling_params=SamplingParams(max_tokens=4),
        session_id="session-a",
        full_input_ids=torch.tensor([100, 200, 300], dtype=torch.int32),
        phase_ids=torch.tensor([1, 1, 2], dtype=torch.int32),
        full_phase_ids=torch.tensor([4, 4, 5], dtype=torch.int32),
        semantic_phase_ids=torch.tensor([8, 8, 9], dtype=torch.int32),
        logical_cached_len=2,
        logical_device_len=3,
    )

    req.append_host(
        torch.tensor([99], dtype=torch.int32),
        phase_id=3,
        semantic_phase_id=10,
        phase_text_tail="tail",
    )

    assert req.input_ids.tolist() == [10, 20, 30, 99]
    assert req.phase_ids.tolist() == [1, 1, 2, 3]
    assert req.semantic_phase_ids.tolist() == [8, 8, 9, 10]
    assert req.phase_text_tail == "tail"
    assert req.full_input_ids.tolist() == [100, 200, 300]
    assert req.full_phase_ids.tolist() == [4, 4, 5]

    req.materialize_full_host()

    assert req.full_input_ids.tolist() == [100, 200, 300, 99]
    assert req.full_phase_ids.tolist() == [4, 4, 5, 3]
    assert req._full_input_ids_tail == []
    assert req._full_phase_ids_tail == []
