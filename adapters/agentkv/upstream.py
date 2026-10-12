"""Import upstream cache/session code without importing the MiniSGL model engine.

Namespace packages skip engine/server-only optional dependencies; every exported
implementation below is loaded from the immutable upstream checkout.
"""
import ast
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "baselines/agentkv/engine/python/minisgl"
for name, directory in [("minisgl", ENGINE), *[
    (f"minisgl.{name}", ENGINE / name)
    for name in ("eviction", "scheduler", "utils", "kernel")
]]:
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package

from minisgl.utils.misc import div_ceil, div_even
from minisgl.utils.logger import init_logger
sys.modules["minisgl.utils"].div_ceil = div_ceil
sys.modules["minisgl.utils"].div_even = div_even
sys.modules["minisgl.utils"].init_logger = init_logger
from minisgl.kernel.store import store_cache
sys.modules["minisgl.kernel"].store_cache = store_cache

from minisgl.eviction.base import LayerEvictionView
from minisgl.eviction.stageq import StageQSnapKVEvictionPlan
from minisgl.phase import SemanticPhase


def original_function(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


factory = original_function(
    ENGINE / "eviction/__init__.py", {"create_eviction_plan"},
    {"EvictionPlan": object, "StageQSnapKVEvictionPlan": StageQSnapKVEvictionPlan},
)["create_eviction_plan"]
tokenizer_functions = original_function(
    ENGINE / "tokenizer/tokenize.py",
    {node.name for node in ast.parse((ENGINE / "tokenizer/tokenize.py").read_text()).body
     if isinstance(node, ast.FunctionDef)},
    {"torch": torch, "List": list, "SemanticPhase": SemanticPhase},
)
phase_ids = tokenizer_functions["_phase_ids_from_offsets"]
semantic_phase_ids = tokenizer_functions["_semantic_phase_ids_from_offsets"]
advance_phase = original_function(
    ENGINE / "scheduler/scheduler.py", {"_phase_from_tail"}, {},
)["_phase_from_tail"]
advance_semantic_phase = original_function(ENGINE / "scheduler/scheduler.py",
    {"_semantic_phase_from_tail"}, {"SemanticPhase": SemanticPhase})["_semantic_phase_from_tail"]

sys.modules["minisgl.eviction"].create_eviction_plan = factory
from minisgl.core import Req, SamplingParams
from minisgl.message import UserMsg
from minisgl.distributed import get_tp_info, set_tp_info, try_get_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.eviction.query_ring import QueryRingBufferPool
from minisgl.eviction.manager import EvictionManager
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.session import SessionManager, SessionError
