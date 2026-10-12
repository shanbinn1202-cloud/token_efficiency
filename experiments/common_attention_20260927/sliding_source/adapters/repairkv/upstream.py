"""Import the pinned RepairKV checkout without modifying its sources."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "baselines/RepairKV"
sys.path.insert(0, str(ROOT))

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, slice_kv
from phases.phase6_repair.src.protocol import build_turn_n_keep_plan, materialize_context_partition
from phases.phase6_repair.src.selectors import score_evicted_positions, select_idlekv_positions
from phases.phase6_repair.src.runner import _restore_positions as restore_positions
