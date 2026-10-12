"""The KV algorithm no longer replaces native request scheduling."""
from types import SimpleNamespace as NS
import pytest
from vllm import agentkv_schedule as policy


def test_no_method_specific_batch_phase():
    assert policy.phase(None) is None


def test_context_limit_clamps_output_like_upstream():
    from adapters.agentkv.vllm_policy import output_limit
    assert output_limit(64000,4096,65536)==1536
    assert output_limit(32768,4096,65536)==4096
    with pytest.raises(ValueError,match="context limit"):
        output_limit(65536,4096,65536)

