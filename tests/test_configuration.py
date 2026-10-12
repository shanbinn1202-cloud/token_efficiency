import json
import pytest
from tebench.config import load, runner_config
from tebench.methods import METHODS, check_support, policy_snapshot
from tebench.serve import command


def test_defaults_and_overrides_propagate_to_runner_and_service():
    c = load(overrides=["cache.kv_budget=16384", "generation.max_tokens=2048", "generation.max_turns=77",
        "generation.max_window_size=65536"])
    rc = runner_config(c, "/tmp/test-run")
    assert rc.agent.max_tokens == 2048 and rc.agent.step_limit == 77
    assert rc.evaluation.trajectory_max_workers == 8
    assert policy_snapshot("sliding_window", c)["kv_budget"] == 16384
    assert policy_snapshot("sliding_window", c)["budget_scope"] == "input"
    cmd = command(c)
    assert cmd[cmd.index("--max-model-len") + 1] == "65536"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "8"
    assert policy_snapshot("agentkv", c)["kv_budget"] == 16384
    assert load()["cache"]["kv_budget"] == 16384


def test_aligned_defaults():
    c = load()
    assert c["model"]["enable_thinking"] is True
    assert c["generation"]["max_tokens"] == 4096
    assert c["generation"]["max_turns"] == 120
    assert c["generation"]["max_window_size"] == 131072
    assert c["generation"]["temperature"] == 0
    b = c["backend"]
    assert b["cuda_graph"] and b["torch_compile"]
    assert b["full_attention"] == "sdpa" and b["sdpa_kernel"] == "auto"
    assert b["linear_attention"] == "native" and b["prefill_chunk_size"] is None
    assert c["methods"]["dram_backed_kv"]["options"]["query_source"] == "goal"


@pytest.mark.parametrize("method", ["vanilla", "sliding_window", "agentkv"])
def test_explicit_kv_pool_reaches_service_without_changing_token_budget(method):
    c = load(overrides=["backend.name=vllm", "backend.kv_cache_memory_bytes=5368709120"])
    check_support(method, c)
    cmd = command(c, method)
    assert cmd[cmd.index("--kv-cache-memory-bytes") + 1] == "5368709120"
    assert c["cache"]["kv_budget"] == 16384
    assert c["generation"]["max_tokens"] == 4096
    assert "--kv-cache-memory-bytes" not in command(load(), method)


@pytest.mark.parametrize("value", ["0", "-1", "true", "5.5", "5GiB"])
def test_invalid_kv_pool_rejected(value):
    with pytest.raises(ValueError, match="kv_cache_memory_bytes"):
        load(overrides=[f"backend.kv_cache_memory_bytes={value}"])


def test_transformers_cannot_silently_ignore_explicit_kv_pool():
    with pytest.raises(ValueError, match="explicit GPU KV pool"):
        check_support("vanilla", load(overrides=["backend.kv_cache_memory_bytes=5368709120"]))


@pytest.mark.parametrize("setting", ["generation.max_tokens=0", "generation.max_tokes=1", "backend.name=other",
    "methods.sliding_window.options.recent=-1", "model.enable_thinking=1", "dataset.start=-1",
    "generation.max_window_size=100", "methods.agentkv.options.anchor_budget=31"])
def test_invalid_settings_rejected(setting):
    with pytest.raises(ValueError):
        load(overrides=[setting])


def test_policy_override_does_not_mutate_variant_registry():
    from baselines.group.memory import INLINE_SCOPE_POLICIES
    before = INLINE_SCOPE_POLICIES["scope_tool_v30"]
    c = load(overrides=["methods.group.options.max_active_turns=5"])
    assert policy_snapshot("group", c)["max_active_turns"] == 5
    assert INLINE_SCOPE_POLICIES["scope_tool_v30"] is before
    assert before.max_active_turns is None


def test_window_turn_limit_is_removed_from_config_and_policy():
    c = load()
    assert c["methods"]["sliding_window"] == {"variant": "sliding_window", "options": {}}
    assert policy_snapshot("sliding_window", c) == {
        "kv_budget": 16384, "budget_scope": "input", "selection": "recent_tokens",
        "storage": "rerendered_tokens", "linear_state": "recomputed"}
    with pytest.raises(ValueError, match="no turn-limit options"):
        load(overrides=["methods.sliding_window.options.recent=10"])
    old = load(overrides=["methods.sliding_window.variant=sliding_window_10"])
    with pytest.raises(ValueError, match="Unknown sliding_window variant"):
        policy_snapshot("sliding_window", old)


def test_backend_capability_matrix_does_not_silently_substitute_methods():
    c = load()
    for method in METHODS:
        with pytest.raises(ValueError, match="evaluation.workers=1"):
            check_support(method, c)
    c["evaluation"]["workers"] = 1
    c["backend"]["cuda_graph"] = False
    c["backend"]["torch_compile"] = False
    for method in METHODS:
        if method in {"repairkv", "intentkv", "random_attention"}:
            with pytest.raises(ValueError, match="vLLM TP=1"):
                check_support(method, c)
            continue
        check_support(method, c)
        policy_snapshot(method, c)
    c["backend"]["name"] = "vllm"
    c["evaluation"]["workers"] = 8
    c["backend"]["cuda_graph"] = True
    c["backend"]["torch_compile"] = True
    for method in METHODS:
        check_support(method, c)


@pytest.mark.parametrize("graph,compile_model,mode,graph_mode", [
    (True, True, "VLLM_COMPILE", "FULL_AND_PIECEWISE"),
    (False, True, "VLLM_COMPILE", "NONE"),
    (True, False, "NONE", "FULL_DECODE_ONLY"),
    (False, False, "NONE", "NONE"),
])
def test_vllm_graph_and_compile_controls_are_independent(graph, compile_model, mode, graph_mode):
    c = load()
    c["backend"].update(cuda_graph=graph, torch_compile=compile_model)
    cmd = command(c)
    assert json.loads(cmd[cmd.index("--compilation-config") + 1]) == {
        "mode": mode, "cudagraph_mode": graph_mode}
    assert ("--enforce-eager" in cmd) == (not graph and not compile_model)


def test_yaml_overlay_is_partial_and_strict(tmp_path):
    p = tmp_path / "local.yaml"
    p.write_text("generation:\n  max_turns: 80\ncache:\n  kv_budget: 16384\n")
    c = load(p)
    assert c["generation"]["max_tokens"] == 4096
    assert c["generation"]["max_turns"] == 80
    assert policy_snapshot("sliding_window", c)["kv_budget"] == 16384


@pytest.mark.parametrize("method", ["threshold", "intent_memory", "generic_scope"])
def test_removed_methods_are_rejected(method):
    with pytest.raises(ValueError):
        load(overrides=[f"methods.{method}.options.recent=3"])
    with pytest.raises(ValueError, match="Unknown method"):
        check_support(method, load())
