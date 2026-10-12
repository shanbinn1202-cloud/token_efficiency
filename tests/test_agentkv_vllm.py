"""Independent policy, layout and scheduler invariants for AgentKV/vLLM."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines/vllm_six"))
from vllm.tebench.kv_session import gather_kv, compact_heads, apply_worker_updates, OFFSETS
from adapters.agentkv.vllm_policy import plan_for, should_compact
from adapters.agentkv.upstream import LayerEvictionView, QueryRingBufferPool


def test_original_plan_defaults_and_trigger_reset_each_reply():
    p = plan_for(8192, {})
    assert (p.anchor_budget, p.num_stages, p.observation_window, p.pooling_kernel_size,
            p.sink_tokens, p.query_group_pooling, p.query_stage_mapping) == (32, 4, 8, 1, 16, "mean", "legacy")
    assert [n for n in range(1, 260) if should_compact(n, 128)] == [128, 256]
    assert not should_compact(1, 1)


def test_independent_per_head_score_reference():
    torch.manual_seed(11)
    k = torch.randn(90, 3, 16)
    q = torch.randn(32, 6, 16)
    p = plan_for(40, {})
    view = LayerEvictionView(layer_id=0, physical_cached_len=90, num_kv_heads=3,
                            device=k.device, k=k, q_observation=q)
    actual = p.select_layer(view)
    expected = []
    for h in range(3):
        scores = []
        for obs in q:
            for head in obs[2*h:2*h+2]:
                scores.append(torch.softmax(k[16:82, h] @ head / 4, dim=0))
        ranked = torch.stack(scores).mean(0).topk(16).indices + 16
        expected.append(torch.cat([torch.arange(16), ranked, torch.arange(82,90)]).sort().values)
    assert torch.equal(actual, torch.stack(expected))
    assert not torch.equal(actual[0], actual[1])


def test_packed_kv_per_head_copy_preserves_external_pages_and_both_kv():
    cache = torch.arange(9*3*16*8).reshape(9,3,16,8).clone()
    original = cache.clone()
    pages = [7,2,5,1]
    selected = torch.tensor([[0,1,17,31,50], [0,4,18,30,62], [0,2,19,33,49]])
    compact_heads(cache,pages,16,selected)
    for h in range(3):
        for dst,src in enumerate(selected[h].tolist()):
            assert torch.equal(cache[pages[dst//16],h,dst%16],original[pages[src//16],h,src%16])
    assert torch.equal(cache[[0,3,4,6,8]],original[[0,3,4,6,8]])


def test_scheduler_releases_after_worker_copy_and_preserves_linear_state():
    freed=[]
    m=type('FullAttentionManager',(),{})()
    m.block_size=16
    m.req_to_blocks={'s':[NS(block_id=i,ref_cnt=1,is_null=False) for i in range(5)]}
    m.block_pool=NS(free_blocks=lambda bs:freed.extend(b.block_id for b in bs))
    linear=NS(req_to_blocks={'s':['state']})
    scheduler=NS(requests={'s':NS(num_computed_tokens=140)},kv_cache_manager=NS(
        coordinator=NS(single_type_managers=[m,linear])))
    apply_worker_updates(scheduler,{'s':{'removed':109,'physical':31}})
    assert OFFSETS.pop('s')==109
    assert len(m.req_to_blocks['s'])==2 and sorted(freed)==[2,3,4]
    assert linear.req_to_blocks['s']==['state']


def test_session_slot_rings_survive_batch_reordering_and_wrap():
    ring=QueryRingBufferPool(max_table_slots=2,num_layers=1,capacity=8,num_stages=4,
        num_q_heads=2,head_dim=4,dtype=torch.float32,device=torch.device('cpu'))
    positions=torch.arange(60,dtype=torch.int32)
    phases=torch.tensor([1]*25+[2]*5+[3]*30,dtype=torch.int32)
    meta=ring.prepare_write_metadata_from_reqs([NS(table_idx=1,extend_len=30),
        NS(table_idx=0,extend_len=30)],positions,phases)
    q=torch.arange(60*2*4,dtype=torch.float32).reshape(60,2,4)
    ring.write_layer(layer_id=0,q=q,metadata=meta)
    for slot, expected in [(1,set(range(17,30))),(0,set(range(52,60)))]:
        seen,pos=ring.get_unordered(table_idx=slot,layer_id=0)
        assert set(pos.tolist())==expected
        assert torch.equal(seen,q[pos.long()])
    ring.reset(1)
    assert ring.get_unordered(table_idx=0,layer_id=0)[0].shape[0]==8


def test_agentkv_service_config_matches_client_budget():
    import json
    from tebench.config import load
    from tebench.serve import command, environment
    from tebench.methods import check_support, policy_snapshot
    c=load(overrides=['backend.name=vllm','cache.kv_budget=16384'])
    check_support('agentkv',c)
    cmd=command(c,'agentkv'); env=environment('agentkv',c)
    assert '--async-scheduling' in cmd and '--no-enable-prefix-caching' in cmd
    assert env['VLLM_AGENTKV_BUDGET']=='16384'
    assert json.loads(env['VLLM_AGENTKV_OPTIONS'])==c['methods']['agentkv']['options']
    assert policy_snapshot('agentkv',c)['budget_scope']=='decode_compaction_target'


def test_statistics_match_explicit_external_session_id(monkeypatch):
    import vllm.tebench.agentkv as module
    state=NS(external_id='external',compactions=2,physical=32,logical=99,
             prompt_physical=34,prefill=4,last_kept=None, rid='external-randomsuffix',
             history=NS(events=[],preemptions=0,running_preemptions=0,parked_reclaims=0,
                 recomputed_tokens=0,replayed_compactions=0),checkpoint_bytes=123,stage_finished=True)
    monkeypatch.setattr(module,'STATE',NS(external={'external':state}))
    OFFSETS['external-randomsuffix']=67
    try:
        stats=module.get_stats(None,'external')
        assert stats['total_evicted_tokens']==67 and stats['cached_tokens']==30
        with pytest.raises(KeyError):
            module.get_stats(None,'not-this-session')
    finally:
        OFFSETS.pop('external-randomsuffix')


def test_live_model_audit_has_separate_compile_cache_key(monkeypatch):
    import json
    from tebench.config import load
    from tebench.serve import command
    c=load(overrides=['backend.name=vllm'])
    monkeypatch.delenv('VLLM_AGENTKV_MODEL_AUDIT', raising=False)
    normal=command(c,'agentkv')
    monkeypatch.setenv('VLLM_AGENTKV_MODEL_AUDIT','1')
    audit=command(c,'agentkv')
    key=lambda cmd:json.loads(cmd[cmd.index('--additional-config')+1])['kv_session_extension']
    assert key(normal)['model_audit'] is False
    assert key(audit)['model_audit'] is True


def test_resume_keeps_raw_byte_sample_and_full_decoded_span():
    """A length limit inside UTF-8 must not decode/reencode the pending ID."""
    import asyncio
    from vllm.tebench.server import Session

    class Tokenizer:
        eos_token_id = 9
        def convert_tokens_to_ids(self, text): return 9
        def decode(self, ids, **kwargs):
            if ids == [500, 245]: return '汉'
            return ''.join({1: 'A', 2: 'B', 3: 'C', 500: '�', 245: '�', 9: '<eos>'}[i] for i in ids)
        def __call__(self, text, **kwargs):
            return {'input_ids': [{'A': 1, 'B': 2, 'C': 3, '�': 5513}[c] for c in text],
                    'offset_mapping': [(i, i+1) for i in range(len(text))]}

    async def run():
        tokenizer = Tokenizer()
        engine = NS(model_config=NS(max_model_len=1000), get_tokenizer=lambda: tokenizer)
        session = Session(engine, 'agentkv')
        async def stats(*args, **kwargs):
            return [{'logical_computed_tokens': session.logical, 'retained_kv_tokens': session.logical, 'boundaries': []}]
        engine.collective_rpc = stats
        seen = []
        sequences = iter(([500, 245], [3, 9], [3, 9]))
        async def generate(prompt, params, rid):
            seen.append(NS(prompt=prompt, sampling_params=params, request_id=rid))
            outputs = next(sequences)
            yield NS(outputs=[NS(token_ids=outputs, text='', finish_reason='length' if outputs[-1] != 9 else 'stop')])
        engine.generate = generate
        body = dict(max_tokens=2, temperature=0)
        await session.complete_agentkv(dict(body, input_ids=[1]))
        assert session.prompt_text == 'A汉' and session.pending_token == 245
        with pytest.raises(ValueError, match='Restored token differs'):
            await session.complete_agentkv(dict(body, input_ids=[5513, 2]))
        await session.complete_agentkv(dict(body, input_ids=[245, 2]))
        assert session.prompt_text == 'A汉BC' and session.pending_token is None
        assert seen[1].prompt['prompt_token_ids'] == [1, 500, 245, 2]
        assert seen[0].request_id != seen[1].request_id
        assert seen[1].sampling_params.extra_args['agentkv']['restored_tokens'] == 1
        assert seen[1].sampling_params.extra_args['agentkv']['phase_ids'][0] == -1
        await session.complete_agentkv(dict(body, input_ids=[2]))
        assert seen[2].sampling_params.extra_args['agentkv']['restored_tokens'] == 0
        await session.close()
    asyncio.run(run())


def test_restored_sample_uses_parked_generation_phase(monkeypatch):
    import vllm.tebench.agentkv as module
    from adapters.agentkv.upstream import advance_phase
    from vllm.tebench.kv_recompute import History
    old = NS(phases=[0, 1], tail='<think>unfinished', generated=256, slot=0, history=History())
    import vllm.tebench.agentkv_connector as connector
    monkeypatch.setattr(connector, 'restore_ring', lambda state: None)
    tokenizer = NS(decode=lambda ids, **kwargs: '�')
    monkeypatch.setattr(module, 'ENABLED', True)
    monkeypatch.setattr(module, 'STATE', NS(sessions={}, external={'external': old}, tokenizer=tokenizer))
    expected = advance_phase(old.tail + '�', old.phases[-1])
    meta = dict(phase_ids=[-1, 2], restored_tokens=1, phase_tail='new tool prompt',
                session_id='external', logical=2, physical=2)
    req = NS(req_id='r', num_computed_tokens=2, prompt_token_ids=[10, 11, 245, 12],
             sampling_params=NS(extra_args={'agentkv': meta}))
    module.worker_update(None, NS(finished_req_ids=[], scheduled_new_reqs=[req]))
    assert old.phases == [0, 1, expected, 2]
    assert old.tail == 'new tool prompt' and old.generated == 0
    assert old.prefill == 2
