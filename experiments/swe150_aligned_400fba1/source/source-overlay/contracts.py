"""Development-only loader of the unchanged shared stdlib runtime contracts."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent / "swebench_baseline"
NAME = "_swebench_dynamic_baseline"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)

from _swebench_dynamic_baseline.bootstrap import ManifestContextProvider
from _swebench_dynamic_baseline.coordinator import (RuntimeCoordinator, RuntimeBinding,
    CaptureSnapshot, QuerySnapshot, PolicyRuntimeAbort, canonical_prompt_hash)
from _swebench_dynamic_baseline.goal_contract_v13 import ProducerProvenance
from _swebench_dynamic_baseline.observation_runtime import (ObservationRuntime,
    ObservationTransition, PreparedObservation, RecomputeReceipt, _hash, _history, _turn, _vector)
from _swebench_dynamic_baseline.goal_runtime import GoalRuntimeAdapter
