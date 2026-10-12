"""Compaction-aware recompute must preserve historical attention decisions."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines/vllm_six"))
from vllm.tebench import kv_recompute as replay
from vllm.tebench.kv_session import OFFSETS, gather_kv, write_dense


@pytest.fixture(autouse=True)
def clear_state():
    for m in [replay.BOUNDARIES, replay.REMOVED, replay.CURSORS, replay.HISTORIES, OFFSETS]:
        m.clear()
    yield
    for m in [replay.BOUNDARIES, replay.REMOVED, replay.CURSORS, replay.HISTORIES, OFFSETS]:
        m.clear()


def request(computed=0):
    return NS(request_id="r", sampling_params=NS(extra_args={"agentkv": {}}),
              num_computed_tokens=computed)


def test_recompute_stops_at_every_historical_cut_then_allows_new_cuts():
    for p in [10, 20]:
        replay.record_boundary("r", p)
    req = request(25)
    OFFSETS['r'] = 12
    replay.begin_recompute(req)
    req.num_computed_tokens = 0
    assert OFFSETS['r'] == 0 and replay.limit_tokens(req, 8192) == 10
    req.num_computed_tokens = 10
    replay.on_worker_update("r", dict(replayed=True, logical=10))
    assert replay.limit_tokens(req, 8192) == 10
    req.num_computed_tokens = 20
    replay.on_worker_update("r", dict(replayed=True, logical=20))
    assert replay.limit_tokens(req, 100) == 100
    replay.on_worker_update("r", dict(record_compaction=True, logical=30))
    req.num_computed_tokens = 30
    assert replay.limit_tokens(req, 1) == 1
    replay.begin_recompute(req)
    req.num_computed_tokens = 0
    assert replay.limit_tokens(req, 100) == 10


def test_empty_history_uses_native_recompute_without_stale_cursor():
    req = request()
    replay.begin_recompute(req)
    replay.record_boundary('r', 10)
    req.num_computed_tokens = 10
    assert replay.limit_tokens(req, 1) == 1


def test_replay_admission_reserves_peak_instead_of_one_chunk_or_full_history():
    req = request()
    replay.record_boundary('r', 20000, 4000)
    replay.record_boundary('r', 50000, 34000)
    replay.begin_recompute(req)
    # 60K logical history ends at 26K physical, but the second cut needs
    # 46K live slots immediately before compaction. An 8K chunk is unsafe.
    assert replay.limit_tokens(req, 8192) == 8192
    assert replay.admission_tokens(req, 60000) == 46000
    # A prefill ending before that cut must not reserve future events.
    assert replay.admission_tokens(req, 30000) == 26000
    # Re-admission after the first replay applies the current offset once.
    OFFSETS['r'] = 4000
    req.num_computed_tokens = 20000
    replay.on_worker_update('r', dict(replayed=True, logical=20000))
    assert replay.admission_tokens(req, 60000) - OFFSETS['r'] == 46000


def test_replay_admission_native_and_missing_metadata_are_conservative():
    req = request()
    assert replay.admission_tokens(req, 60000) == 60000
    replay.record_boundary('r', 20000)
    replay.begin_recompute(req)
    assert replay.admission_tokens(req, 60000) == 60000




@pytest.mark.parametrize('method', ['agentkv', 'random_attention'])
def test_rebuilt_pages_replay_exact_per_head_or_window_selection(method):
    import vllm.tebench.agentkv as agentkv
    assert agentkv.STATE is None
    cache = torch.full((6, 2, 4, 6), -1.)
    dense = torch.arange(10 * 2 * 6).reshape(10, 2, 6).float()
    blocks = [4, 1, 3]
    write_dense(cache, blocks, 4, dense)
    req = request(6)
    req.sampling_params.extra_args = {method: {}}
    req.block_ids = [blocks[:]]
    group = NS(kv_cache_spec=NS(block_size=4), layer_names=['layer'])
    runner = NS(requests={'r': req}, kv_cache_config=NS(kv_cache_groups=[group]),
        compilation_config=NS(static_forward_context={'layer': NS(kv_cache=cache)}),
        input_batch=NS(req_id_to_index={}))
    keep = torch.tensor([[0, 3, 7, 9], [1, 2, 5, 9]], dtype=torch.int32)
    replay.record_agent('r', 10, 6, [(0, 'layer', keep)])
    h = replay.HISTORIES['r']; h.cursor=0; h.replay_until=12
    output = NS(req_id_to_index={'r': 0}, sampled_token_ids=[[]])
    updates = replay.after_sample(runner, NS(num_scheduled_tokens={'r': 4}), output)
    expected = torch.stack([dense[keep[head].long(), head] for head in range(2)], dim=1)
    assert torch.equal(gather_kv(cache, req.block_ids[0], 4, 4), expected)
    assert torch.all(cache[[0, 2, 5]] == -1)
    assert updates['r'] == dict(removed=6, physical=4, logical=10, replayed=True)
    assert h.replayed_compactions == 1 and h.recomputed_tokens == 4
    assert len(h.events) == 1 and OFFSETS['r'] == 6
    req.num_computed_tokens = 10
    replay.after_sample(runner, NS(num_scheduled_tokens={'r': 2}), output)
    assert h.cursor is None and h.recomputed_tokens == 6










def test_agent_uses_native_recompute_without_output_reservation():
    from vllm.v1.request import RequestStatus
    from vllm.v1.core.sched.scheduler import Scheduler
    req=request(100)
    req.max_tokens=4096
    req.num_output_tokens=96
    assert replay.admission_tokens(req,1000)==1000
    req.status=RequestStatus.RUNNING
    req.spec_token_ids=[]
    req.drop_stale_output=False
    req.num_stale_output_tokens=0
    req.num_in_flight_tokens=2
    req.num_output_placeholders=2
    req.num_preemptions=0
    freed=[]; waiting=[]
    scheduler=NS(_free_request_blocks=lambda r:freed.append(r.request_id),
        encoder_cache_manager=NS(free=lambda r:None),_inflight_prefills=NS(discard=lambda r:None),
        log_stats=False,waiting=NS(prepend_request=lambda r:waiting.append(r.request_id)),
        reset_preempted_req_ids=set())
    Scheduler._preempt_request(scheduler,req,0.)
    assert req.status == RequestStatus.PREEMPTED
    assert req.num_computed_tokens == req.num_output_placeholders == 0
    assert req.num_stale_output_tokens == 2 and req.num_preemptions == 1
    assert freed == waiting == ['r']
