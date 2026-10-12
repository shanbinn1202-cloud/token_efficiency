"""Async batch execution, compaction fences, and common scheduling policy."""
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('method', ['vanilla', 'sliding_window', 'agentdiet',
                                  'agentkv', 'repairkv', 'intentkv', 'random_attention'])
def test_common_native_async_scheduler(method):
    from tebench.config import load
    from tebench.serve import command
    from tebench.methods import policy_snapshot
    from vllm.config.scheduler import SchedulerConfig
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    c = load(overrides=['backend.name=vllm'])
    cmd = command(c, method)
    assert '--async-scheduling' in cmd and '--no-async-scheduling' not in cmd
    assert cmd[cmd.index('--scheduling-policy') + 1] == 'fcfs'
    assert cmd[cmd.index('--max-num-seqs') + 1] == '8'
    assert policy_snapshot(method, c)['engine_overrides']['async_scheduling']
    assert SchedulerConfig.default_factory(async_scheduling=True).get_scheduler_cls() is AsyncScheduler


def engine_stub(output):
    calls = []
    sample_future = Future(); sample_future.set_result(NS())
    execute_future = Future(); execute_future.set_result(None)
    engine = NS(batch_queue=deque(), batch_queue_size=2,
        scheduler=NS(has_requests=lambda:True, schedule=lambda _:output,
                     get_grammar_bitmask=lambda _:None, update_from_output=lambda *a:{}),
        model_executor=NS(execute_model=lambda *a, **k: execute_future,
                          sample_tokens=lambda *a, **k:(calls.append('sample') or sample_future)),
        _should_throttle_prefills=lambda:False, is_pooling_model=False, is_ec_consumer=True,
        log_error_detail=lambda _:nullcontext(), capture_iteration_details=lambda _:nullcontext(),
        _process_aborts_queue=lambda:None, _attach_iteration_details=lambda *a:None)
    return engine, calls


def test_async_decoder_executes_sampling_and_drains_commit_before_next_batch():
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.core.sched.output import SchedulerOutput
    output = SchedulerOutput.make_empty()
    output.total_num_scheduled_tokens = 1
    output.kv_commit_fence = True
    engine, calls = engine_stub(output)
    assert EngineCore.step_with_batch_queue(engine) == (None, True)
    assert calls == ['sample'] and len(engine.batch_queue) == 1
    # A pending physical mutation prevents scheduling against obsolete pages.
    engine.scheduler.schedule = lambda _:pytest.fail('scheduled across a commit fence')
    assert EngineCore.step_with_batch_queue(engine) == ({}, False)
    assert not engine.batch_queue


def test_ordinary_decode_keeps_two_batches_in_flight():
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.core.sched.output import SchedulerOutput
    output = SchedulerOutput.make_empty(); output.total_num_scheduled_tokens = 1
    engine, calls = engine_stub(output)
    EngineCore.step_with_batch_queue(engine)
    EngineCore.step_with_batch_queue(engine)
    assert calls == ['sample', 'sample'] and len(engine.batch_queue) == 1


def test_explicit_reset_retains_native_scheduler_delegation():
    from vllm.v1.engine.core import EngineCore
    calls = []
    scheduler = NS(reset_prefix_cache=lambda *args: calls.append(args) or True)
    assert EngineCore.reset_prefix_cache(NS(scheduler=scheduler), True, True)
    assert calls == [(True, True)]


def test_fences_only_at_original_agent_interval_and_replay_cuts(monkeypatch):
    from vllm.tebench.kv_async import needs_commit
    from vllm.tebench.kv_recompute import BOUNDARIES, CURSORS
    from vllm.tebench.kv_session import OFFSETS
    monkeypatch.setenv('VLLM_AGENTKV', '1')
    monkeypatch.setenv('VLLM_AGENTKV_BUDGET', '8192')
    monkeypatch.setenv('VLLM_AGENTKV_OPTIONS', '{"eviction_interval":128}')
    req = NS(num_prompt_tokens=9000, num_computed_tokens=9125)
    sched = NS(requests={'fence-test':req})
    out = NS(num_scheduled_tokens={'fence-test':1})
    assert not needs_commit(sched, out)
    req.num_computed_tokens += 1
    assert needs_commit(sched, out)
    req.num_computed_tokens += 1
    assert not needs_commit(sched, out)
    monkeypatch.setitem(BOUNDARIES, 'fence-test', [9128])
    monkeypatch.setitem(CURSORS, 'fence-test', 0)
    assert needs_commit(sched, out)


def test_agent_checkpoint_restores_exact_kv_and_ring_with_new_native_blocks(monkeypatch):
    import torch
    import vllm.tebench.agentkv as agentkv
    from vllm.tebench.agentkv_connector import snapshot, restore_after_zero, restore_ring
    from adapters.agentkv.upstream import QueryRingBufferPool
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.tebench.kv_session import write_dense, gather_kv
    cache = torch.zeros(8, 2, 4, 6)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float32)
    ring = QueryRingBufferPool(max_table_slots=2,num_layers=1,capacity=4,num_stages=2,
        num_q_heads=2,head_dim=3,dtype=torch.float32,device=torch.device('cpu'))
    state = NS(slot=1,physical=7)
    ring.q[1].normal_(); ring.positions[1].copy_(torch.arange(8).reshape(2,4))
    ring.size[1].fill_(4); ring.write_pos[1].fill_(2)
    source = torch.randn(7,2,6)
    req = NS(block_ids=[[1,3]], req_id='r')
    write_dense(cache, req.block_ids[0], 4, source)
    runner = NS(kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=['l'])]),
        compilation_config=NS(static_forward_context={'l':NS(kv_cache=cache)}))
    monkeypatch.setattr(agentkv, 'STATE', NS(ring=ring, sessions={'r':state}))
    monkeypatch.setattr(agentkv, 'ENABLED', True)
    snapshot(runner, state, req)
    cache.fill_(-9); ring.q.fill_(-8); state.slot=0; req.block_ids=[[5,2]]
    restore_ring(state)
    restore_after_zero(runner, NS(scheduled_new_reqs=[req]))
    assert torch.equal(gather_kv(cache, [5,2],4,7), source)
    assert torch.equal(ring.q[0], state.ring_checkpoint['q'])
    assert torch.equal(ring.positions[0], state.ring_checkpoint['positions'])
    assert torch.all(cache[1] == -9)


def test_random_fence_uses_physical_decode_boundary(monkeypatch):
    from vllm.tebench.kv_async import needs_commit
    from vllm.tebench.kv_session import OFFSETS
    monkeypatch.setenv('VLLM_RANDOM_ATTENTION', '1')
    monkeypatch.setenv('VLLM_RANDOM_ATTENTION_BUDGET', '8192')
    monkeypatch.setitem(OFFSETS, 'random-fence', 1000)
    req = NS(num_prompt_tokens=9200, num_computed_tokens=9254)
    sched = NS(requests={'random-fence': req})
    out = NS(num_scheduled_tokens={'random-fence': 1})
    assert not needs_commit(sched, out)
    req.num_computed_tokens = 9255
    assert needs_commit(sched, out)
    req.num_prompt_tokens = 9256
    assert not needs_commit(sched, out)
