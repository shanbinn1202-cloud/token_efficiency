"""Storage and policy invariants for the local vLLM extension (CPU tests)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

PATH = Path(__file__).resolve().parents[1] / "engines/vllm/vllm/kv_window.py"
sys.path.insert(0, str(PATH.parents[1]))
spec = importlib.util.spec_from_file_location("tested_kv_window", PATH)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


@pytest.mark.parametrize("budget,expected", [(120, (10, 3, 20)), (90, (40, 2, 20)),
                                            (55, (75, 1, 20)), (30, (100, 0, 20)), (25, (100, 0, 15))])
def test_exact_token_suffix_budget(budget, expected):
    assert w.select_window(10, [10, 40, 80], 100, 20, 5, budget) == expected


def test_prefix_cannot_be_truncated():
    with pytest.raises(ValueError, match="prefix"):
        w.select_window(10, [10], 100, 200, 5, 14)


def test_dummy_graph_rows_have_zero_offsets():
    runner = NS(input_batch=NS(req_ids=[]))
    assert w.offsets_cpu(runner, 8, 8).tolist() == [0] * 8


@pytest.mark.parametrize("prefix,start,end", [(7, 29, 60), (0, 13, 54), (10, 60, 60), (16, 16, 64)])
def test_non_page_aligned_compaction_exact(prefix, start, end):
    import torch
    cache = torch.arange(8 * 2 * 16 * 4).reshape(8, 2, 16, 4).clone()
    old = cache.clone()
    blocks = [5, 1, 3, 7]
    w.compact_tensor(cache, blocks, 16, prefix, start, end)
    selected = list(range(prefix)) + list(range(start, end))
    for j, i in enumerate(selected):
        assert torch.equal(cache[blocks[j // 16], :, j % 16], old[blocks[i // 16], :, i % 16])
    # Other sessions' pages are untouched.
    assert torch.equal(cache[[0, 2, 4, 6]], old[[0, 2, 4, 6]])


def test_allocator_releases_only_tail_and_keeps_logical_positions():
    released = []
    manager = type("FullAttentionManager", (), {})()
    manager.block_size = 16
    manager.req_to_blocks = {"s": [NS(block_id=i, ref_cnt=1, is_null=False) for i in range(1, 5)]}
    manager.block_pool = NS(free_blocks=lambda blocks: released.extend(blocks))
    mamba = NS(req_to_blocks={"s": ["state"]})
    scheduler = NS(kv_cache_manager=NS(enable_caching=False,
                   coordinator=NS(single_type_managers=[manager, mamba])))
    session = NS(request_id="s", num_computed_tokens=60)
    meta = dict(computed=60, logical_computed=60, prefix=7, suffix_start=29, budget=60)
    update = NS(max_tokens=8192, prompt_token_ids=[1] * 10, sampling_params=NS(extra_args={"kv_window": meta}))
    w.OFFSETS.clear()
    w.PENDING.clear()
    w.resume_window(scheduler, session, update)
    assert session.num_computed_tokens == 60
    assert w.OFFSETS["s"] == 22
    assert len(manager.req_to_blocks["s"]) == 3
    assert [b.block_id for b in released] == [4]
    assert mamba.req_to_blocks["s"] == ["state"]
    assert w.allocation_offset(manager, "s") == 22
    assert w.allocation_offset(mamba, "s") == 0
    assert len(w.drain_compactions({"s": session})) == 1
    assert not w.drain_compactions({})
    assert not w.OFFSETS


def test_server_uses_local_engine_and_explicit_session_requirements():
    from tebench.config import ROOT, load
    from tebench.serve import command, environment
    cmd = command(load(), "sliding_window")
    from tebench.engine import engine_root
    assert cmd != command(load(), "vanilla")
    assert "--async-scheduling" in cmd
    assert "--middleware" in cmd
    assert environment("sliding_window")["VLLM_KV_WINDOW"] == "0"
    assert environment("sliding_window")["VLLM_NATIVE_WINDOW"] == "0"
    assert environment("sliding_window")["VLLM_COMPACT_WINDOW"] == "1"
    assert environment("sliding_window")["PYTHONPATH"].split(":")[0] == str(engine_root())


def test_logical_context_limit_stops_agent_without_retry_or_failed_inference():
    from tebench.backends.base import Backend
    from tebench.backends.kv_window import KVWindowBackend
    from tebench.config import load
    from tebench.methods import create_model
    from minisweagent.exceptions import InterruptAgentFlow
    config = load(overrides=["backend.name=vllm"])
    backend = KVWindowBackend.__new__(KVWindowBackend)
    Backend.__init__(backend, config)
    backend.tokenizer = NS(
        apply_chat_template=lambda messages, **kw: "prefix" + ("header" if kw["add_generation_prompt"] else ""),
        encode=lambda text, **kw: list(text.encode()))
    sent = []
    def reject(*args, **kwargs):
        sent.append(kwargs)
        return NS(is_error=True, status_code=400,
                  json=lambda: {"error": "Logical RoPE context limit reached"})
    backend.client = NS(post=reject)
    from baselines.sliding_window.model import KVSlidingWindowModel
    model = KVSlidingWindowModel(backend=backend, model_name="openai/Qwen3.6-27B")
    with pytest.raises(InterruptAgentFlow) as caught:
        model.query([{"role": "user", "content": "continue"}])
    assert caught.value.messages[0]["extra"]["exit_status"] == "LimitsExceeded"
    assert len(sent) == 1
    assert model.requests == []


def test_window_matches_literal_prefix_plus_latest_tokens_even_for_large_new_input():
    prefix, header = 10, 5
    for computed in [15, 29, 60, 100]:
        for incoming in [5, 19, 200]:
            for budget in [15, 30, 59, 250]:
                start, _, used = w.select_window(prefix, [prefix], computed, incoming, header, budget)
                old=list(range(computed));fresh=list(range(computed,computed+incoming))
                actual=old[:prefix]+old[start:]+fresh[-used:]
                entire=old+fresh
                expected=entire if len(entire)<=budget else entire[:prefix]+entire[-(budget-prefix):]
                assert actual==expected
