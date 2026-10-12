"""Guard against changing the published selector or corrupting selected KV.

Compare the actual original HF implementation using the same random scores;
the copied published vLLM selector consumes a different-shaped RNG draw.
This tests selection and compaction, including repeated eviction and GQA heads.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from adapters.random_attention.policy import Policy, should_compact
from adapters.random_attention.selector_random import _stable_seed
from vllm.tebench.kv_session import compact_heads, gather_kv, write_dense

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "baselines/Random-Attention"


def upstream_layer():
    spec = importlib.util.spec_from_file_location("random_reference", UPSTREAM / "kvcompress/engine/cache_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvictLayer


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test"))])
@pytest.mark.parametrize("budget", [256, 8192, 12288, 16384])
def test_paged_policy_equals_unmodified_HF_with_matched_random_scores(device, budget):
    Layer = upstream_layer()
    policy = Policy(budget)
    prefix, heads, dim, page = 53, 4, 8, 16
    length = budget + 192
    dense = torch.randn(length, heads, 2 * dim, device=device, dtype=torch.bfloat16)
    blocks = list(reversed(range((length+page-1)//page)))
    cache = torch.empty(len(blocks)+2, heads, page, 2*dim, device=device, dtype=dense.dtype)
    cache.fill_(float('nan'))
    write_dense(cache, blocks, page, dense)
    for event in range(3):
        logical = length + event * 64
        keep = policy.keep(cache=cache, physical=len(dense), prefix=prefix,
                           session="same-request", logical=logical, layer=3)
        g = torch.Generator(device=device).manual_seed(_stable_seed(1234, "same-request", logical, 3))
        scores = torch.rand((heads, len(dense)), device=device, generator=g)
        ref = Layer(3, budget, 64, "random_pp", 0.5, False, False)
        ref.device = torch.device(device)
        ref.k_cache = dense[..., :dim].unsqueeze(0).clone()
        ref.v_cache = dense[..., dim:].unsqueeze(0).clone()
        ref.cache_seqlens = torch.tensor([len(dense)], device=device)
        ref.prompt_len = prefix
        with patch.object(torch, "rand", return_value=scores[:, :-64].unsqueeze(0).clone()):
            ref._run_eviction(len(dense))
        compact_heads(cache, blocks, page, keep, dense=dense)
        expected = torch.cat((ref.k_cache[0, :budget], ref.v_cache[0, :budget]), dim=-1)
        actual = gather_kv(cache, blocks, page, budget)
        assert torch.equal(actual, expected)
        assert torch.equal(actual[:prefix], dense[:prefix])
        assert torch.equal(actual[-64:], dense[-64:])
        assert not torch.equal(keep[0], keep[1]), "Independent KV heads collapsed"
        assert torch.isnan(cache[-2:]).all(), "Touched unowned pages"
        dense = torch.cat((actual, torch.randn(64, heads, 2*dim, device=device, dtype=dense.dtype)))
        write_dense(cache, blocks, page, dense)


def test_trigger_matches_original_physical_length_and_decode_condition():
    Layer = upstream_layer()
    for count in [1, 7, 64]:
        for physical in [8192, 8193, 8255, 8256, 8319, 8320]:
            ref = Layer(0, 8192, 64, "random_pp", 0.5, False, False)
            ref.cache_seqlens = torch.tensor([physical-count])
            called = []
            ref._run_eviction = lambda n: called.append(n)
            ref.post_attention_update(count)
            actual = should_compact(physical=physical, budget=8192, new_tokens=count, decoding=True)
            assert actual == bool(called)
    assert not should_compact(physical=8256, budget=8192, new_tokens=1, decoding=False)


def test_published_selector_is_copied_verbatim():
    assert (ROOT / "adapters/random_attention/selector_random.py").read_bytes() == (UPSTREAM / "scripts/vllm_rp_bench/selector_random.py").read_bytes()
