"""Shared runtime settings for vanilla and history-selection baselines."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


from tebench.config import DEFAULT_CONFIG
DEFAULT_CONFIG_PATH = str(DEFAULT_CONFIG)
_DEFAULTS = yaml.safe_load(DEFAULT_CONFIG.read_text())


@dataclass
class AgentSettings:
    model: str = "openai/" + _DEFAULTS["model"]["name"].split("/")[-1]
    api_endpoint: str = _DEFAULTS["backend"]["endpoint"]
    api_key: str = _DEFAULTS["backend"]["api_key"]
    temperature: float = _DEFAULTS["generation"]["temperature"]
    max_tokens: int = _DEFAULTS["generation"]["max_tokens"]
    step_limit: int = _DEFAULTS["generation"]["max_turns"]
    request_timeout: float = _DEFAULTS["generation"]["request_timeout"]
    max_consecutive_format_errors: int = _DEFAULTS["generation"]["max_consecutive_format_errors"]


@dataclass
class DataSettings:
    eval_dataset: str = _DEFAULTS["dataset"]["name"]
    eval_split: str = _DEFAULTS["dataset"]["split"]
    max_instances: int = _DEFAULTS["dataset"]["count"]


@dataclass
class EvaluationSettings:
    apptainer_executable: str = _DEFAULTS["evaluation"]["apptainer"]
    apptainer_image_manifest: str = _DEFAULTS["evaluation"]["image_manifest"]
    swebench_max_workers: int = _DEFAULTS["evaluation"]["grading_workers"]
    trajectory_max_workers: int = _DEFAULTS["evaluation"]["workers"]
    instance_timeout: int = _DEFAULTS["evaluation"]["timeout"]
    output_dir: str = _DEFAULTS["evaluation"]["output_root"] + "/vanilla__transformers"


@dataclass
class Config:
    agent: AgentSettings = field(default_factory=AgentSettings)
    data: DataSettings = field(default_factory=DataSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH):
        data = yaml.safe_load(Path(path).read_text()) or {}
        if "agent" not in data and "data" not in data:
            from tebench.config import load, runner_config
            c = load(path)
            return runner_config(c, c["evaluation"]["output_root"] + "/vanilla__transformers")
        if set(data) - {"agent", "data", "evaluation"}:
            raise ValueError("Unknown configuration section")
        config = cls(
            AgentSettings(**data.get("agent", {})),
            DataSettings(**data.get("data", {})),
            EvaluationSettings(**data.get("evaluation", {})),
        )
        if (
            min(
                config.agent.max_tokens,
                config.agent.step_limit,
                config.data.max_instances,
                config.evaluation.trajectory_max_workers,
                config.evaluation.swebench_max_workers,
                config.evaluation.instance_timeout,
                config.agent.request_timeout,
                config.agent.max_consecutive_format_errors,
            )
            <= 0
        ):
            raise ValueError(
                "Token, step, instance, worker and timeout limits must be positive"
            )
        return config
