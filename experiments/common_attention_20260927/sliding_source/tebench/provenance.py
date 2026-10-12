"""Fingerprint the configuration, own sources and externally owned repositories."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import platform
import os
from tebench.config import ROOT
from tebench.methods import policy_snapshot
from tebench.engine import engine_root

PROTECTED = ("baselines/Random-Attention", "baselines/IntentKV", "baselines/RepairKV", "baselines/agentkv", "baselines/dram-backed-kv", "mini-swe-agent", "external/SWE-bench")


def snapshot(config, method, ids):
    sources = {}
    runtime = engine_root()
    for base in ("tebench", "adapters", "baselines", "configs", str(runtime.relative_to(ROOT) / "vllm")):
        for parent, directories, files in os.walk(ROOT / base):
            directories[:] = sorted(d for d in directories if d not in {".git", "__pycache__"}
                and str((Path(parent) / d).relative_to(ROOT)) not in PROTECTED)
            for name in sorted(files):
                p = Path(parent) / name
                if p.suffix in {".py", ".yaml", ".sh"}:
                    sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    external = {}
    for path in PROTECTED:
        def git(*args):
            r = subprocess.run(["git", "-C", str(ROOT / path), *args], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else None
        external[path] = {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
                          "submodules": git("submodule", "status", "--recursive")}
    runtime_commit = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    runtime_diff = subprocess.check_output(["git", "-C", str(runtime), "diff", "HEAD", "--", "vllm"])
    versions = {}
    for name in ("torch", "transformers", "vllm", "mini-swe-agent", "litellm", "flash-linear-attention"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    from huggingface_hub import try_to_load_from_cache
    model_source = Path(config["model"]["path"])
    if not model_source.is_dir():
        cached = try_to_load_from_cache(config["model"]["path"], "config.json")
        model_source = Path(cached).parent if isinstance(cached, str) else None
    model_files = {}
    if model_source:
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
            p = model_source / name
            if p.is_file():
                model_files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"schema_version": 1, "configuration": config, "method": method,
        "policy": policy_snapshot(method, config), "instance_ids": ids,
        "source_sha256": sources, "external": external, "versions": versions,
        "model_source": str(model_source), "model_files_sha256": model_files,
        "architecture": platform.machine(),
        "engine": {"path": str(runtime), "upstream_commit": runtime_commit,
                   "tracked_diff_sha256": hashlib.sha256(runtime_diff).hexdigest()}}
