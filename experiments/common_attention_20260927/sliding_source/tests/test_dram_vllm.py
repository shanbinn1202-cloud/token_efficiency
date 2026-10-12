"""DRAM transport checks; these do not establish selector equivalence."""
import asyncio
import pytest
from types import SimpleNamespace as NS
from threading import local

from tebench.backends.dram_vllm import DramVLLMBackend
from vllm.kv_session_server import Session


def test_reset_releases_remote_session_before_local_state():
    calls = []
    backend = DramVLLMBackend.__new__(DramVLLMBackend)
    backend._local = local()
    backend._local.session = {"id": "task-session"}
    backend.client = NS(post=lambda path, json: (
        calls.append((path, json)) or NS(raise_for_status=lambda: None)))
    backend.reset()
    backend.reset()
    assert calls == [("kv-session/close", {"session_id": "task-session"})]
    assert backend._local.session is None


def test_usage_counts_input_before_generation(monkeypatch):
    import json
    from tebench.config import load
    monkeypatch.setenv('VLLM_DRAM_KV_OPTIONS', json.dumps(load()['methods']['dram_backed_kv']['options']))
    async def scenario():
        async def rpc(*args, **kwargs):
            return [{"retained_kv_tokens": 9, "logical_computed_tokens": 9,
                     "vectors": [], "resident": [], "sealed_end": 0}]

        tokenizer = NS(decode=lambda ids, **kw: 'answer')
        c = NS(hidden_size=8, layer_types=['full_attention'], num_key_value_heads=1, head_dim=8)
        engine = NS(model_config=NS(max_model_len=128, hf_text_config=c, model='fake'),
                    collective_rpc=rpc, get_tokenizer=lambda: tokenizer)
        session = Session(engine, "dram_backed_kv")

        async def consume(params):
            await session.queue.get()
            session.future.set_result(([4, 5, 6, 7], "answer", "length"))

        session.consume = consume
        result = await session.complete_dram(dict(input_ids=[1, 2, 3, 8, 9, 10],
            max_tokens=4, temperature=0, header_tokens=2, prefix_tokens=4, budget=16,
            messages=[{'role': 'system', 'content': 'You are an assistant.'},
                      {'role': 'user', 'content': 'Fix a unit test.'}]))
        assert result["prompt_tokens"] == 6
        assert result["cached_tokens"] == 0
        assert result["completion_tokens"] == 4
        assert result["prefill_tokens"] == 6
        await session.close()

    asyncio.run(scenario())


def test_native_views_offload_independently_and_runtime_reuses_residents():
    import torch
    from adapters.dram_backed_kv.vllm_storage import (
        DramKVRuntime, PageAdapter, PagePayload, native_page_view, page_id)

    cache = torch.arange(4 * 2 * 32 * 8).reshape(4, 2, 32, 8).float()
    blocks = [3, 1]
    payload = PagePayload({"layer0": native_page_view(cache, blocks, 32, 16)})
    expected = payload.layers["layer0"].clone()
    runtime = DramKVRuntime(PageAdapter("cpu"), pin_cpu=False)
    runtime.offload_sealed("task", page_id(16), payload, keep_resident=True)
    reused = runtime._restore("task", (page_id(16),))
    assert reused.ordered_payloads[0] is payload
    assert reused.operation.reuse_count == 1
    assert reused.operation.restore_count == 0
    runtime._restore("task", ())
    cache.zero_()
    restored = runtime._restore("task", (page_id(16),))
    assert torch.equal(restored.ordered_payloads[0].layers["layer0"], expected)
    assert restored.operation.restore_count == 1
    assert restored.operation.reactivation_count == 1
    assert restored.operation.h2d_bytes == expected.numel() * expected.element_size()
    released = runtime.release_session("task")
    assert released.dram_bytes == released.hbm_resident_bytes == 0


def test_captured_projection_matches_original_decoder_output_projection():
    import torch
    from adapters.dram_backed_kv.vllm_storage import upstream
    from vllm.dramkv import _capture_impl
    torch.manual_seed(73)
    h = torch.randn(7, 64).bfloat16()
    residual = torch.randn_like(h)
    projection, _ = upstream._projection(torch, 64, 128, 20260904)
    values = torch.empty(7, 128, dtype=torch.int8)
    scales = torch.empty(7)
    _capture_impl(h, residual, projection, values, scales)
    owner = NS(torch=torch, projection=projection)
    for i, row in enumerate(h + residual):
        expected = upstream.OnlineSelectorSession._compact_last(owner, row)
        assert values[i].tolist() == list(expected.values)
        # Batched GEMM/normalization and single-row reduction can differ by
        # one float32 ULP; quantized coordinates must still match exactly.
        torch.testing.assert_close(scales[i], torch.tensor(expected.scale), rtol=1e-6, atol=0)


@pytest.mark.parametrize('expand', [False, True])
def test_restore_survives_allocator_zeroing_with_native_page_ownership(monkeypatch, expand):
    import torch
    import vllm.dramkv as worker
    from vllm.kv_session import OFFSETS, gather_kv
    from adapters.dram_backed_kv.vllm_storage import (
        DramKVRuntime, PageAdapter, PagePayload, native_page_view, page_id)
    cache = torch.arange(6 * 2 * 32 * 8).reshape(6, 2, 32, 8).float()
    original = cache.clone()
    old_blocks = [2, 4]
    runtime = DramKVRuntime(PageAdapter('cpu'), pin_cpu=False)
    for start in range(0, 64, 16):
        payload = PagePayload({'layer': native_page_view(cache, old_blocks, 32, start)})
        runtime.offload_sealed('r', page_id(start), payload, keep_resident=True)
    if expand:
        runtime._restore('r', (page_id(0),))
        old_blocks = [2]
        selected = [page_id(0), page_id(32), page_id(48)]
        destinations = [2, 4]
        positions = list(range(16))
    else:
        selected = [page_id(0), page_id(32)]
        destinations = [2]
        positions = list(range(64))
    state = NS(logical=64, physical=len(positions), positions=positions, selections=0,
               pending_restore=None)
    monkeypatch.setattr(worker, 'ENABLED', True)
    monkeypatch.setattr(worker, 'STATE', NS(runtime=runtime, sessions={'r': state}))
    group = NS(layer_names=['layer'], kv_cache_spec=NS(block_size=32))
    monkeypatch.setattr(worker, '_full_groups', lambda runner: [(0, group)])
    runner = NS(compilation_config=NS(static_forward_context={'layer': NS(kv_cache=cache)}))
    meta = {'logical': 64, 'selected': selected, 'tail_start': 64,
            'source_blocks': {'0': old_blocks}, 'session_id': 'external'}
    req = NS(req_id='r', num_computed_tokens=64, block_ids=[destinations],
             sampling_params=NS(extra_args={'dram_kv': meta}))
    output = NS(finished_req_ids=[], scheduled_new_reqs=[req])
    try:
        worker.worker_update(runner, output)
        cache[4].zero_()  # Recycled/new block is zeroed between the two hooks.
        worker.restore_after_zero(runner, output)
        expected = torch.cat([native_page_view(original, [2, 4], 32,
                              int(pid.split(':')[1].split('-')[0])) for pid in selected])
        actual = gather_kv(cache, destinations, 32, len(expected))
        assert torch.equal(actual, expected)
        assert torch.equal(cache[[0, 1, 3, 5]], original[[0, 1, 3, 5]])
        assert OFFSETS['r'] == 64 - len(expected)
        assert runtime.session_metrics('r').restore_count == (2 if expand else 0)
    finally:
        OFFSETS.pop('r', None)
