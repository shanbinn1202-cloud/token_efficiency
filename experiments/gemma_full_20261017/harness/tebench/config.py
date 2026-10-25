"""Strict layered configuration, independent of GPU and agent libraries."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"


def merge(base, overrides, prefix=""):
    result = deepcopy(base)
    for key, value in overrides.items():
        path = f"{prefix}.{key}".strip(".")
        if key not in result and not prefix.endswith(".options"):
            raise ValueError(f"Unknown configuration key: {path}")
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value, path)
        else:
            result[key] = deepcopy(value)
    return result


def set_value(config, assignment):
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError("Overrides must have the form section.key=value")
    keys = key.split(".")
    node = config
    for part in keys[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(f"Unknown configuration key: {key}")
        node = node[part]
    if keys[-1] not in node and (len(keys) < 2 or keys[-2] != "options"):
        raise ValueError(f"Unknown configuration key: {key}")
    node[keys[-1]] = yaml.safe_load(raw)


def validate(c):
    for section, keys in {
        "generation": ("max_tokens", "max_turns", "max_window_size", "request_timeout", "max_consecutive_format_errors"),
        "dataset": ("count",), "evaluation": ("workers", "grading_workers", "timeout"),
        "cache": ("kv_budget",), "backend": ("tensor_parallel_size",),
    }.items():
        for key in keys:
            if type(c[section][key]) is not int or c[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if c["generation"]["max_tokens"] >= c["generation"]["max_window_size"]:
        raise ValueError("max_tokens must be smaller than max_window_size")
    if c["generation"]["temperature"] < 0:
        raise ValueError("temperature must be nonnegative")
    if not 0 < c["generation"]["top_p"] <= 1 or c["generation"]["repetition_penalty"] <= 0:
        raise ValueError("Require 0 < top_p <= 1 and positive repetition_penalty")
    if type(c["generation"]["top_k"]) is not int or c["generation"]["top_k"] == 0 or c["generation"]["top_k"] < -1:
        raise ValueError("top_k must be -1 (disabled) or a positive integer")
    if type(c["dataset"]["start"]) is not int or c["dataset"]["start"] < 0:
        raise ValueError("dataset.start must be a nonnegative integer")
    if c["backend"]["name"] not in {"transformers", "vllm"}:
        raise ValueError("backend.name must be transformers or vllm")
    if not 0 < c["backend"]["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    pool = c["backend"]["kv_cache_memory_bytes"]
    if pool is not None and (type(pool) is not int or pool <= 0):
        raise ValueError("backend.kv_cache_memory_bytes must be null or a positive integer")
    for section, key in [("model", "enable_thinking"), ("model", "local_files_only"),
                         ("backend", "prefix_caching"), ("backend", "async_scheduling"),
                         ("backend", "cuda_graph"), ("backend", "torch_compile")]:
        if type(c[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be a boolean")
    if c["backend"]["full_attention"] != "sdpa":
        raise ValueError("Current aligned Transformers execution requires full_attention=sdpa")
    if c["backend"]["sdpa_kernel"] not in {"auto", "math", "flash", "efficient"}:
        raise ValueError("sdpa_kernel must be auto, math, flash or efficient")
    if c["backend"]["linear_attention"] not in {"native", "fla", "torch"}:
        raise ValueError("linear_attention must be native, fla or torch")
    chunk = c["backend"]["prefill_chunk_size"]
    if c["backend"]["vllm_attention_backend"] != "TRITON_ATTN":
        raise ValueError("The six-method v0.24 profile requires TRITON_ATTN")
    if c["backend"]["gdn_prefill_backend"] != "triton":
        raise ValueError("The six-method v0.24 profile requires Triton GDN")
    batch = c["backend"]["max_num_batched_tokens"]
    if type(batch) is not int or batch < c['evaluation']['workers']:
        raise ValueError("max_num_batched_tokens must cover all request slots")
    if chunk is not None and (type(chunk) is not int or chunk <= 0):
        raise ValueError("prefill_chunk_size must be null or a positive integer")
    for name, method in c["methods"].items():
        if not isinstance(method.get("options"), dict):
            raise ValueError(f"methods.{name}.options must be a mapping")
    if c["methods"]["sliding_window"]["options"]:
        raise ValueError("sliding_window has no turn-limit options; token suffix is determined by cache.kv_budget")
    diet = c["methods"]["agentdiet"]["options"]
    if set(diet) - {"delay_steps", "window_before_steps", "token_threshold", "reflection_enable_thinking"}:
        raise ValueError("Unknown AgentDiet options")
    if any(type(v) is not int or v < 0 for k, v in diet.items() if k != "reflection_enable_thinking"):
        raise ValueError("AgentDiet options must be nonnegative integers")
    if type(diet["reflection_enable_thinking"]) is not bool:
        raise ValueError("AgentDiet reflection_enable_thinking must be a boolean")
    kv = c["methods"]["agentkv"]["options"]
    allowed = {"eviction_interval", "anchor_budget", "observation_window", "sink_tokens", "pooling_kernel_size"}
    if set(kv) - allowed:
        raise ValueError(f"Unknown AgentKV options: {set(kv) - allowed}")
    if any(type(v) is not int or v < (0 if k == "sink_tokens" else 1) for k, v in kv.items()):
        raise ValueError("AgentKV options must be positive integers (sink_tokens may be zero)")
    if kv["pooling_kernel_size"] % 2 == 0:
        raise ValueError("AgentKV pooling_kernel_size must be odd")
    if kv["anchor_budget"] % 4:
        raise ValueError("AgentKV anchor_budget must divide evenly across four stages")
    if c["cache"]["kv_budget"] <= kv["sink_tokens"] + kv["observation_window"]:
        raise ValueError("kv_budget must exceed AgentKV mandatory sink/recent tokens")
    return c


def load(path=None, overrides=()):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    explicit_pool = False
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        profile = yaml.safe_load(Path(path).read_text()) or {}
        explicit_pool = "kv_cache_memory_bytes" in profile.get("backend", {})
        config = merge(config, profile)
    for assignment in overrides:
        set_value(config, assignment)
        explicit_pool |= assignment.partition("=")[0] == "backend.kv_cache_memory_bytes"
    # The shared pool default applies to vLLM; Transformers has no such pool.
    # Explicit non-vLLM pool settings still reach check_support and are rejected.
    if config["backend"]["name"] != "vllm" and not explicit_pool:
        config["backend"]["kv_cache_memory_bytes"] = None
    return validate(config)


def runner_config(c, output):
    """Translate once at the existing environment/agent boundary."""
    from baselines.common.settings import Config, AgentSettings, DataSettings, EvaluationSettings
    g, e, d, b = c["generation"], c["evaluation"], c["dataset"], c["backend"]
    return Config(
        agent=AgentSettings(model="openai/" + c["model"]["name"].split("/")[-1],
            api_endpoint=b["endpoint"], api_key=b["api_key"], temperature=g["temperature"],
            max_tokens=g["max_tokens"], step_limit=g["max_turns"], request_timeout=g["request_timeout"],
            max_consecutive_format_errors=g["max_consecutive_format_errors"]),
        data=DataSettings(eval_dataset=d["name"], eval_split=d["split"], max_instances=d["count"]),
        evaluation=EvaluationSettings(apptainer_executable=e["apptainer"],
            apptainer_image_manifest=str(ROOT / e["image_manifest"]),
            swebench_max_workers=e["grading_workers"], trajectory_max_workers=e["workers"],
            instance_timeout=e["timeout"], output_dir=str(output)),
    )
