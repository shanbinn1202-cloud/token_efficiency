"""The persistent prefix window must preserve KV and recycle only masked pages."""

import pytest
import torch
from types import SimpleNamespace as NS

from vllm.tebench import native_window as window
from vllm.tebench import window_manager as storage
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import RSWASpec


def test_invisible_gap_never_reads_pages_recycled_to_another_request(monkeypatch):
    monkeypatch.setattr(window, "ENABLED", True)
    monkeypatch.setattr(window, "BUDGET", 20)
    monkeypatch.setitem(window.SESSIONS, "r", NS(prefix=7))
    blocks = list(range(1, 29))
    written = []
    spec = RSWASpec(
        block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float32, rswa_window=20
    )
    runner = NS(
        requests={"r": NS(block_ids=[blocks])},
        kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec)]),
        input_batch=NS(
            req_ids=["r"],
            num_computed_tokens_cpu=[101],
            block_table=[NS(add_row=lambda values, row: written.append(values))],
        ),
    )
    window.prepare_block_tables(runner)
    actual = written[0]
    assert blocks == list(
        range(1, 29)
    )  # No allocator ownership or logical mapping changes.
    first, last = window.gap_blocks(7, 101, 20, 4)
    assert actual[first:last] == [blocks[0]] * (last - first)
    for query in range(101, 112):
        for key in range(query):
            if key < 7 or query - key < 13:
                assert actual[key // 4] == blocks[key // 4]


@pytest.mark.parametrize("prefix", [0, 1, 7, 16])
def test_checkpoint_contains_every_key_visible_to_next_query(prefix):
    budget = 20
    for computed in [0, 5, 19, 20, 21, 67, 129]:
        expected = [
            k for k in range(computed) if k < prefix or computed - k < budget - prefix
        ]
        assert window.retained_positions(prefix, computed, budget) == expected
        assert len(expected) <= budget - 1
        first, last = window.gap_blocks(prefix, computed, budget, 4)
        assert not any(first <= k // 4 < last for k in expected)


def test_restored_window_allocates_only_live_pages_and_keeps_prefix(monkeypatch):
    budget, prefix, bs, computed = 20, 7, 4, 101
    monkeypatch.setattr(storage, "BUDGET", budget)
    monkeypatch.setitem(window.PREFIXES, "r", prefix)
    pool = BlockPool(num_gpu_blocks=40, enable_caching=False, hash_block_size=bs)
    spec = RSWASpec(
        block_size=bs,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        rswa_window=budget,
    )
    manager = storage.NativeWindowManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=bs,
        needs_kv_cache_zeroing=True,
    )
    reserved = manager.get_num_blocks_to_allocate(
        "r", computed, [], computed, 0, computed
    )
    assert reserved < (computed + bs - 1) // bs
    manager.add_local_computed_blocks("r", [], 0, computed)
    manager.allocate_external_computed_blocks("r", 0, computed)
    blocks = manager.req_to_blocks["r"]
    assert sum(not b.is_null for b in blocks) == reserved
    assert len(manager.take_new_block_ids()) == reserved
    prefix_pages = [b.block_id for b in blocks[:2]]
    # Native requests with zero initial cache hit need not enter this mapping.
    manager.num_cached_block.pop("r", None)
    for end in range(computed + 1, computed + 81):
        manager.remove_skipped_blocks("r", end - 1, num_prompt_tokens=end - 3)
        manager.get_num_blocks_to_allocate("r", end, [], end - 1, end - 1, end)
        manager.allocate_new_blocks("r", end, end)
        assert [b.block_id for b in blocks[:2]] == prefix_pages
        assert sum(not b.is_null for b in blocks) <= 7
        assert all(
            not blocks[k // bs].is_null
            for k in window.retained_positions(prefix, end, budget)
        )
    manager.free("r")
    assert pool.get_num_free_blocks() == 39


def test_native_window_launch_and_frontend_preserve_raw_history():
    from baselines.sliding_window.model import KVSlidingWindowModel
    from tebench.config import load
    from tebench.methods import create_model, policy_snapshot
    from tebench.serve import command, environment

    config = load(overrides=["backend.name=vllm"])
    cmd = command(config, "sliding_window")
    assert "--async-scheduling" in cmd and "--no-enable-prefix-caching" in cmd
    assert environment("sliding_window", config)["VLLM_NATIVE_WINDOW"] == "0"
    assert environment("sliding_window", config)["VLLM_COMPACT_WINDOW"] == "1"
    assert "--hf-overrides" not in cmd
    assert isinstance(
        create_model("sliding_window", config, None, model_name="openai/Qwen3.6-27B"),
        KVSlidingWindowModel,
    )
    assert policy_snapshot("sliding_window", config)["linear_state"] == "retained"


@pytest.mark.parametrize("budget", [256,8192,12288,16384])
@pytest.mark.parametrize("device", ["cpu",pytest.param("cuda",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="GPU test"))])
def test_compact_window_keeps_exact_prefix_and_latest_kv_in_native_pages(budget,device):
    from adapters.sliding_window.policy import Policy
    from vllm.tebench.kv_session import compact_heads, gather_kv, write_dense
    prefix,heads,dim,block = 53,4,8,16
    dense = torch.randn(budget+192,heads,2*dim,device=device,dtype=torch.bfloat16)
    blocks = list(reversed(range((len(dense)+block-1)//block)))
    cache = torch.full((len(blocks)+2,heads,block,2*dim),float('nan'),device=device,dtype=dense.dtype)
    policy = Policy(budget)
    for _ in range(3):
        write_dense(cache,blocks,block,dense)
        keep = policy.keep(cache=cache,physical=len(dense),prefix=prefix)
        assert torch.equal(keep,keep[:1].expand(heads,-1))
        expected = torch.cat((dense[:prefix],dense[-(budget-prefix):]))
        compact_heads(cache,blocks,block,keep,dense=dense)
        actual = gather_kv(cache,blocks,block,budget)
        assert torch.equal(actual,expected)
        assert torch.isnan(cache[-2:]).all()
        dense = torch.cat((actual,torch.randn(64,heads,2*dim,device=device,dtype=dense.dtype)))


def test_compact_window_prefill_and_decode_boundaries():
    from adapters.sliding_window.policy import should_compact
    assert should_compact(physical=9000,budget=8192,new_tokens=808,decoding=False)
    assert not should_compact(physical=8192,budget=8192,new_tokens=1,decoding=True)
    assert not should_compact(physical=8255,budget=8192,new_tokens=1,decoding=True)
    assert should_compact(physical=8256,budget=8192,new_tokens=1,decoding=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native FA3 requires GPU")
@pytest.mark.parametrize("budget", [8192,12288,16384])
def test_compact_window_ordinary_gqa_reads_only_retained_kv(budget):
    from adapters.sliding_window.policy import Policy
    from vllm.tebench.kv_session import compact_heads,write_dense
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    block,heads,kv_heads,dim,prefix,count = 784,24,4,256,53,17
    length = budget+301
    dense = torch.randn(length,kv_heads,2*dim,device="cuda",dtype=torch.bfloat16)
    blocks = list(range((length+block-1)//block))
    raw = torch.zeros(len(blocks),block,kv_heads,2*dim,device="cuda",dtype=dense.dtype)
    cache = raw.transpose(1,2)
    write_dense(cache,blocks,block,dense)
    keep = Policy(budget).keep(cache=cache,physical=length,prefix=prefix)
    compact_heads(cache,blocks,block,keep,dense=dense)
    extra = torch.randn(count,kv_heads,2*dim,device="cuda",dtype=dense.dtype)
    write_dense(cache,blocks,block,extra,start=budget)
    expected_kv = torch.cat((dense[:prefix],dense[-(budget-prefix):],extra))
    # Unused physical pages are poisoned; the standard FA3 call has no mask
    # or virtual heads and is told only the compacted physical sequence length.
    raw.flatten(0,1)[budget+count:] = float('nan')
    k,v = raw.split(dim,dim=-1)
    q = torch.randn(count,heads,dim,device="cuda",dtype=dense.dtype)
    actual = flash_attn_varlen_func(q=q,k=k,v=v,
        cu_seqlens_q=torch.tensor([0,count],device="cuda",dtype=torch.int32),
        seqused_k=torch.tensor([budget+count],device="cuda",dtype=torch.int32),
        max_seqlen_q=count,max_seqlen_k=budget+count,
        block_table=torch.tensor([blocks],device="cuda",dtype=torch.int32),
        softmax_scale=dim**-0.5,causal=True,fa_version=3)
    ek,ev = expected_kv.split(dim,dim=-1)
    ek = ek.repeat_interleave(heads//kv_heads,dim=1).transpose(0,1).float()
    ev = ev.repeat_interleave(heads//kv_heads,dim=1).transpose(0,1).float()
    logits = q.transpose(0,1).float() @ ek.transpose(-1,-2) * dim**-0.5
    mask = torch.arange(budget+count,device="cuda")[None,:] <= torch.arange(budget,budget+count,device="cuda")[:,None]
    expected = (logits.masked_fill(~mask,-torch.inf).softmax(-1) @ ev).transpose(0,1)
    torch.testing.assert_close(actual.float(),expected,atol=.025,rtol=.02)
