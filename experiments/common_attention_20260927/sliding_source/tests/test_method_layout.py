"""Guard method isolation and unified method construction."""
import json
import subprocess
import sys

from tebench.config import load
from tebench.methods import HISTORY_METHODS, METHODS, create_model
from baselines.sliding_window.model import SlidingWindowMemory, POLICIES


def test_window_import_does_not_load_other_methods():
    code = '''
import sys
from baselines.sliding_window.model import SlidingWindowModel
assert not any(n.startswith(("baselines.group", "baselines.threshold", "baselines.generic_scope", "baselines.intent_memory")) for n in sys.modules)
m = SlidingWindowModel(model_name="openai/test")
assert m.variant == "sliding_window"
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_method_registry_routes_to_owned_adapters():
    config = load()
    assert set(METHODS) == {"vanilla", "sliding_window", "group", "agentkv", "agentdiet", "dram_backed_kv", "repairkv", "intentkv", "random_attention"}
    assert set(config["methods"]) == set(METHODS)
    for method, (module, name, _, _) in HISTORY_METHODS.items():
        model = create_model(method, config, backend=None, model_name="openai/test")
        assert type(model).__module__ == module
        assert type(model).__name__ == name
        assert model.variant == config["methods"][method]["variant"]


def test_token_window_can_cut_through_a_turn_without_mutating_archive():
    ids = list(range(500))
    selector = SlidingWindowMemory(POLICIES['sliding_window'], kv_budget=127)
    selected, trace = selector.select_tokens(ids, prefix_tokens=20, header_tokens=5)
    assert selected == ids[:20] + ids[-107:]
    assert trace['prompt_tokens'] == 127
    assert ids == list(range(500))
