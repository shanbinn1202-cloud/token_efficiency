"""State corruption checks at the vLLM/RepairKV boundary."""
from types import SimpleNamespace as NS
import torch
import pytest

from adapters.repairkv.policy import append_rows, base_partition, repair_positions
from adapters.repairkv.upstream import (PositionTrackedCache, slice_kv,
    build_turn_n_keep_plan, materialize_context_partition)
from tebench.config import load


@pytest.mark.parametrize('fail', [False, True])
def test_original_cpu_execution_restores_vllm_threads(monkeypatch, fail):
    from vllm.tebench import repairkv
    changes = []
    monkeypatch.setattr(repairkv.torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(repairkv.torch, 'set_num_threads', changes.append)
    monkeypatch.setattr(repairkv.os, 'sched_getaffinity', lambda pid: set(range(8)))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '32')
    try:
        with repairkv.upstream_cpu_execution() as threads:
            assert threads == 8
            if fail:
                raise RuntimeError('CPU operation failure')
    except RuntimeError:
        assert fail
    assert changes == [8, 1]


@pytest.mark.parametrize('pooling', ['max', 'mean'])
def test_repair_calls_original_default_cpu_scorer_and_selector(pooling):
    from adapters.repairkv import policy
    from phases.phase6_repair.src.selectors import score_evicted_positions, select_idlekv_positions
    assert policy.score_evicted_positions is score_evicted_positions
    assert policy.select_idlekv_positions is select_idlekv_positions
    assert not getattr(score_evicted_positions, '_score_on_gpu', False)
    torch.manual_seed(23)
    pool = PositionTrackedCache(tuple((torch.randn(1, 2, 93, 16), torch.randn(1, 2, 93, 16)) for _ in range(3)), list(range(93)))
    active = slice_kv(pool, list(range(0,93,3)))
    evicted = slice_kv(pool, [p for p in range(93) if p % 3])
    queries = torch.randn(3,6,37,16)
    options = dict(load()['methods']['repairkv']['options'], repair_tokens=11, pooling=pooling)
    expected_scores = score_evicted_positions(query_rows=queries, active_cache=active,
        evicted_cache=evicted, pooling=pooling)
    expected_selected = select_idlekv_positions(evicted_positions=evicted.positions,
        q2_scores=expected_scores, turn_n_scores={}, k=11, left=2, right=20)
    info = {}
    positions, selected, scores = repair_positions(active, evicted, queries,
        scores={}, options=options, device='cpu', telemetry=info)
    assert scores == expected_scores and selected == expected_selected
    assert positions == sorted(active.positions+selected)
    assert info['score_device'] == 'cpu' and info['score_cpu_fallbacks'] == 0


@pytest.mark.parametrize('pooling', ['max', 'mean'])
@pytest.mark.parametrize('query_tile', [1, 7, 16])
def test_query_tiling_preserves_full_normalization_scores_and_selected_rows(pooling, query_tile):
    """Large queries must not OOM or change full-history attention competition."""
    from adapters.repairkv.scoring import score_evicted_positions_tiled
    from adapters.repairkv.upstream import score_evicted_positions, select_idlekv_positions
    torch.manual_seed(431)
    pool = PositionTrackedCache(tuple((torch.randn(1, 2, 83, 16),
        torch.randn(1, 2, 83, 16)) for _ in range(3)), list(range(83)))
    active = slice_kv(pool, list(range(0, 83, 3)))
    evicted = slice_kv(pool, [p for p in range(83) if p % 3])
    queries = torch.randn(3, 6, 37, 16)
    expected = score_evicted_positions(query_rows=queries, active_cache=active,
        evicted_cache=evicted, pooling=pooling)
    actual = score_evicted_positions_tiled(query_rows=queries, active_cache=active,
        evicted_cache=evicted, pooling=pooling, workspace_bytes=query_tile * 6 * 83 * 4)
    assert actual.keys() == expected.keys()
    torch.testing.assert_close(torch.tensor(list(actual.values())),
        torch.tensor(list(expected.values())), rtol=2e-6, atol=1e-8)
    choose = lambda scores: select_idlekv_positions(evicted_positions=evicted.positions,
        q2_scores=scores, turn_n_scores={}, k=11, left=2, right=20)
    assert choose(actual) == choose(expected)


def test_restore_uses_original_cpu_fragment_transfer_and_injection():
    from adapters.repairkv.policy import restore_selected, upstream_restore_positions
    from phases.phase6_repair.src.runner import _restore_positions
    assert upstream_restore_positions is _restore_positions
    torch.manual_seed(24)
    pool = PositionTrackedCache(((torch.randn(1,2,20,8),torch.randn(1,2,20,8)),),list(range(20)))
    active, evicted = slice_kv(pool,[0,3,7,19]), slice_kv(pool,[i for i in range(20) if i not in {0,3,7,19}])
    restored = restore_selected(active,evicted,[2,8],device='cpu')
    expected = slice_kv(pool,[0,2,3,7,8,19])
    assert restored.positions == expected.positions
    for pair,reference in zip(restored.kv,expected.kv):
        assert all(torch.equal(a,b) for a,b in zip(pair,reference))


def test_paper_budget_includes_growing_dialogue_and_restores_exactly_k():
    torch.manual_seed(7)
    options = load()['methods']['repairkv']['options']
    kv = tuple((torch.randn(1, 2, 380, 8), torch.randn(1, 2, 380, 8)) for _ in range(3))
    pool = PositionTrackedCache(kv, list(range(380)))
    plan, active, evicted = base_partition(pool, answer_tokens=18,
        budget=256, options=options, device='cpu')
    assert len(active) == 256-96
    # Even if only the first 8 tokens were the initial prompt, later dialogue
    # participates in eviction once the accumulated history crosses budget.
    assert set(evicted.positions) & set(range(8, 380))
    assert set(active.positions).isdisjoint(evicted.positions)
    queries = torch.randn(3, 4, 11, 8)
    positions, selected, scores = repair_positions(active, evicted, queries,
        scores=plan.importance_scores, options=options, device='cpu')
    assert len(selected) == 96 and len(positions) == 256
    assert set(selected) <= set(evicted.positions)
    assert len(positions) == len(set(positions))
    assert positions == sorted(positions)
    assert set(scores) == set(evicted.positions)
    assert pool.positions == list(range(380))


def test_full_pool_reselection_readmits_old_rows_and_refreshes_their_scores():
    options = dict(load()['methods']['repairkv']['options'], repair_tokens=1,
        sink_size=1, recency_window=1)
    keys = torch.zeros(1, 1, 8, 2)
    keys[0, 0, 2] = torch.tensor([5., 0.])
    keys[0, 0, [4, 6, 7]] = torch.tensor([0., 5.])
    pool = PositionTrackedCache(((keys, keys.clone()),), list(range(8)))
    old_plan, old_base, old_evicted = base_partition(pool,
        answer_tokens=2, budget=4, options=options, device='cpu')
    assert 2 in old_evicted.positions and 2 not in old_base.positions
    original = keys.clone()
    # A new answer now favors the previously evicted row 2. It must return
    # through BASE selection, before any repair or next-query probe occurs.
    new_keys = torch.tensor([[[[5., 0.], [5., 0.]]]])
    pool = append_rows(pool, ((new_keys, new_keys.clone()),), 8)
    expected_plan = build_turn_n_keep_plan(post_q1_cache=pool,
        q1_answer_ids=torch.zeros(2, dtype=torch.long), context_len=len(pool),
        sink_size=1, recency_window=1, pooling=options['pooling'])
    expected = materialize_context_partition(full_post_q1_cache=pool,
        keep_plan=expected_plan, context_budget=3)
    plan, base, evicted = base_partition(pool, answer_tokens=2,
        budget=4, options=options, device='cpu')
    assert base.positions == expected.compressed.positions == [0, 2, 9]
    assert 2 not in evicted.positions
    assert plan.importance_scores == expected_plan.importance_scores
    assert plan.importance_scores[2] > old_plan.importance_scores[2]
    assert set(plan.importance_scores) == set(range(10))
    assert set(base.positions) | set(evicted.positions) == set(range(10))
    assert set(base.positions).isdisjoint(evicted.positions)
    assert torch.equal(pool.kv[0][0][:, :, :8], original)


def test_full_pool_base_oom_fallback_uses_the_same_original_plan(monkeypatch):
    from adapters.repairkv import policy
    torch.manual_seed(17)
    options = dict(load()['methods']['repairkv']['options'], repair_tokens=3,
        sink_size=1, recency_window=2)
    pool = PositionTrackedCache(((torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8)),), list(range(20)))
    expected = base_partition(pool, answer_tokens=3, budget=12, options=options, device='cpu')
    original, attempts, cleanups = policy._partition_full_pool, [], []
    def constrained(*args, **kwargs):
        attempts.append(kwargs['device'].type)
        if kwargs['device'].type == 'cuda':
            raise torch.OutOfMemoryError('Injected base scoring allocation failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(policy, '_partition_full_pool', constrained)
    monkeypatch.setattr(torch.accelerator, 'empty_cache', lambda: cleanups.append(True))
    info = {}
    actual = base_partition(pool, answer_tokens=3, budget=12, options=options, device='cuda', telemetry=info)
    assert attempts == ['cuda', 'cuda', 'cpu'] and len(cleanups) == 2
    assert info['base_oom_retries'] == 2 and info['base_cpu_fallbacks'] == 1
    assert actual[0] == expected[0]
    assert actual[1].positions == expected[1].positions
    assert actual[2].positions == expected[2].positions


@pytest.mark.parametrize('audit', [None, '0'])
def test_repair_moves_promoted_rows_without_disabled_audit_io(monkeypatch, audit):
    from vllm.tebench import repairkv
    options = dict(load()['methods']['repairkv']['options'], repair_tokens=3,
        sink_size=1, recency_window=2)
    pool = PositionTrackedCache(((torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8)),), list(range(20)))
    plan, base, evicted = base_partition(pool,
        answer_tokens=3, budget=12, options=options, device='cpu')
    state = NS(base=base, evicted=evicted, scores=plan.importance_scores,
        query_chunks=[torch.randn(1, 4, 2, 8)], repairs=0, restored_tokens=0,
        score_seconds=0., checkpoint_logical=20, recurrent={}, last_compression={})
    if audit is None:
        monkeypatch.delenv('VLLM_REPAIRKV_AUDIT', raising=False)
    else:
        monkeypatch.setenv('VLLM_REPAIRKV_AUDIT', audit)
    monkeypatch.setattr(torch, 'save', lambda *a, **kw: pytest.fail('Disabled audit wrote tensors'))
    monkeypatch.setattr(repairkv, 'STATE', NS(external={'r': state}, device='cpu', options=options, budget=12))
    result = repairkv.select(None, 'r')
    assert result['physical'] == 12
    assert state.last_compression['resumed_tokens'] == 12
    assert len(state.evicted) == 8
    assert set(state.pending_positions).isdisjoint(state.evicted.positions)
    assert set(state.pending_positions) | set(state.evicted.positions) == set(range(20))


def test_host_history_refuses_probe_or_duplicate_rows():
    kv = ((torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)),)
    pool = append_rows(None, kv, 0)
    original = pool.kv[0][0].clone()
    result = append_rows(pool, kv, 8)
    assert result.positions == list(range(16))
    assert torch.equal(result.kv[0][0][:, :, :8], original)
    with pytest.raises(ValueError, match='committed logical history'):
        append_rows(result, kv, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocation recovery')
def test_committed_history_survives_cuda_append_oom_and_future_turns(monkeypatch):
    torch.manual_seed(817)
    layers = tuple((torch.randn(1, 2, 9, 8, device='cuda'),
                    torch.randn(1, 2, 9, 8, device='cuda')) for _ in range(2))
    pool = append_rows(None, layers, 0)
    cat = torch.cat
    attempts = []
    def fail_gpu_once(tensors, *args, **kwargs):
        if not attempts and tensors[0].device.type == 'cuda':
            attempts.append(True)
            raise torch.OutOfMemoryError('Injected private-history allocation failure')
        return cat(tensors, *args, **kwargs)
    monkeypatch.setattr(torch, 'cat', fail_gpu_once)
    grown = append_rows(pool, layers, 9)
    assert grown.device.type == 'cpu' and grown.positions == list(range(18))
    grown = append_rows(grown, layers, 18)
    assert grown.positions == list(range(27))
    for actual, source in zip(grown.kv, layers):
        for value, original in zip(actual, source):
            assert torch.equal(value, cat([original.cpu()] * 3, dim=2))


def test_restore_survives_allocator_zero_without_touching_other_sessions(monkeypatch):
    from vllm.tebench import repairkv
    from vllm.tebench.kv_session import gather_kv
    torch.manual_seed(2)
    cache = torch.randn(8, 2, 4, 8)
    original = cache.clone()
    k, v = torch.randn(1, 2, 20, 4), torch.randn(1, 2, 20, 4)
    pool = PositionTrackedCache(((k, v),), list(range(20)))
    state = NS(pool=pool, positions=[0, 2, 5, 6, 9, 13, 15], stage_recurrent={}, rollbacks=0,
        stage_restore_cache=slice_kv(pool, [0, 2, 5, 6, 9, 13, 15]))
    monkeypatch.setattr(repairkv, 'ENABLED', True)
    monkeypatch.setattr(repairkv, 'STATE', NS(sessions={'r': state}, device='cpu'))
    monkeypatch.setattr(repairkv, '_layers', lambda runner: [('a', 0, 4)])
    monkeypatch.setattr(repairkv, '_recurrent', lambda *args: None)
    runner = NS(compilation_config=NS(static_forward_context={'a': NS(kv_cache=cache)}))
    req = NS(req_id='r', block_ids=[[2, 5]], sampling_params=NS(extra_args={
        'repairkv': dict(restore=True, rewind=True)}))
    cache[5].zero_()
    repairkv.restore_after_zero(runner, NS(scheduled_new_reqs=[req]))
    expected = torch.cat((k[0, :, state.positions], v[0, :, state.positions]), dim=-1).permute(1, 0, 2)
    assert torch.equal(gather_kv(cache, [2, 5], 4, 7), expected)
    assert torch.equal(cache[[0, 1, 3, 4, 6, 7]], original[[0, 1, 3, 4, 6, 7]])
    assert state.rollbacks == 1


def test_eval_releases_each_repairkv_session_when_worker_is_reused(tmp_path, monkeypatch):
    from threading import local
    import datasets
    import tebench.run as workflow
    import tebench.backends as backends
    import tebench.provenance as provenance
    import tebench.metrics as metrics
    import baselines.common.runner as runner
    import baselines.common.environment as environment
    import baselines.common.harness as harness

    state = local()
    resets = []
    def complete(**kwargs):
        state.session = 'active'
    def reset():
        resets.append(getattr(state, 'session', None))
        state.session = None
    backend = NS(execution_info={}, complete=complete, reset=reset, close=lambda: None)
    config = load(overrides=['backend.name=vllm', 'evaluation.workers=1', 'dataset.count=2'])
    monkeypatch.setattr(workflow, 'load', lambda *a: config)
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **k: [{'instance_id': 'a'}, {'instance_id': 'b'}])
    monkeypatch.setattr(environment, 'resolve_instance_image', lambda *a: 'fake-image')
    monkeypatch.setattr(provenance, 'snapshot', lambda *a: {'method': 'repairkv'})
    monkeypatch.setattr(backends, 'create', lambda *a: backend)
    monkeypatch.setattr(harness, 'run_apptainer_harness', lambda **kw: None)
    monkeypatch.setattr(metrics, 'aggregate', lambda *a: {'eligible_for_comparison': True})
    def run_instance(row, *a, **kw):
        assert getattr(state, 'session', None) is None, 'Previous instance leaked its session'
        complete()
        return dict(instance_id=row['instance_id'], exit_status='Submitted', model_patch='')
    monkeypatch.setattr(runner, 'run_instance', run_instance)
    workflow.main(['--method', 'repairkv', '--output', str(tmp_path/'run')])
    assert resets == ['active', 'active', 'active']  # Warmup, first task, second task.


def test_native_preemption_frees_pages_and_requeues_repairkv():
    from vllm import SamplingParams
    from vllm.tebench import repairkv
    from vllm.tebench.kv_session import OFFSETS
    from vllm.tebench.kv_recompute import admission_tokens, limit_tokens, on_worker_update
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.tebench.repairkv_connector import RepairKVConnector
    params = SamplingParams(max_tokens=4096, extra_args={
        'repairkv': dict(logical=20, physical=13)})
    req = Request('repair-test', list(range(25)), params, None)
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = 24
    freed, waiting = [], []
    scheduler = NS(_free_request_blocks=lambda r: freed.append(r),
        encoder_cache_manager=NS(free=lambda r: None), _inflight_prefills=set(),
        log_stats=False, waiting=NS(prepend_request=lambda r: waiting.append(r)),
        reset_preempted_req_ids=set(), running=[req])
    try:
        OFFSETS[req.request_id] = 7
        assert admission_tokens(req, 25) == 25
        assert RepairKVConnector.get_num_new_matched_tokens(None, req, 0) == (20, False)
        Scheduler._preempt_request(scheduler, req, 0.)
        assert freed == waiting == [req]
        assert req.status == RequestStatus.PREEMPTED and req.num_computed_tokens == 0
        assert req.num_preemptions == 1 and OFFSETS[req.request_id] == 0
        assert RepairKVConnector.get_num_new_matched_tokens(None, req, 0) == (0, False)
        assert limit_tokens(req, 8192) == 20
        on_worker_update(req.request_id, dict(repair_recomputed_boundary=True))
        assert limit_tokens(req, 8192) == 8192
    finally:
        OFFSETS.pop(req.request_id, None)
        repairkv.REPLAY_BOUNDARIES.pop(req.request_id, None)


@pytest.mark.parametrize('finished', [False, True])
@pytest.mark.parametrize('mode', ['generate', 'boundary', 'probe'])
def test_late_preemption_preserves_completed_repairkv_stage(monkeypatch, finished, mode):
    """Native async output may finish before its preemption reaches the worker.

    The finished stage's durable KV/probe result must remain available to the
    next turn. An unfinished stage must still reset and use native recompute.
    """
    from vllm.tebench import repairkv
    from vllm.tebench.kv_session import OFFSETS
    pool = PositionTrackedCache(((torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2)),), list(range(4)))
    query = torch.randn(1, 1, 2, 2)
    state = NS(rid='late-preempt', mode=mode, stage_finished=finished,
        logical=4, physical=3, positions=[0, 2, 3], pool=pool,
        query_chunks=[query], stage_logical=2, preemptions=0,
        recompute_until=0, restore_at=0)
    monkeypatch.setattr(repairkv, 'ENABLED', True)
    monkeypatch.setattr(repairkv, 'STATE', NS(sessions={state.rid: state},
        capture_count=torch.zeros(1, dtype=torch.int32)))
    output = NS(finished_req_ids=set(), preempted_req_ids={state.rid},
        scheduled_new_reqs=[], num_scheduled_tokens={}, total_num_scheduled_tokens=0)
    try:
        OFFSETS[state.rid] = 1
        repairkv.worker_update(None, output)
        assert state.preemptions == 1
        assert state.pool is pool
        if finished:
            assert (state.logical, state.physical, state.positions) == (4, 3, [0, 2, 3])
            assert state.query_chunks[0] is query
            repairkv._harvest(None, state)  # Already committed; no native pages needed.
        else:
            assert (state.logical, state.physical, state.positions) == (0, 0, [])
            assert not state.query_chunks
            assert (state.recompute_until, state.restore_at) == (4, 2)
            assert OFFSETS[state.rid] == 0
    finally:
        OFFSETS.pop(state.rid, None)
