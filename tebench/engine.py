"""Select and identify the independently checked-out vLLM engine."""
import os
from pathlib import Path

from tebench.config import ROOT


def engine_root():
    path = Path(os.environ.get("TEBENCH_VLLM_ROOT", ROOT / "engines/vllm_six")).resolve()
    if not (path / "vllm/v1/engine/core.py").is_file():
        raise FileNotFoundError(f"vLLM source checkout missing: {path}")
    return path


def has_policy_hooks():
    return (engine_root() / "vllm/tebench/__init__.py").is_file()
