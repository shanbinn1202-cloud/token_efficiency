"""Shared instance execution, patch recovery and token accounting."""

import argparse
import json
import logging
import os
import shlex
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path, PurePosixPath

from datasets import load_dataset
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from rich.live import Live

from baselines.common.settings import DEFAULT_CONFIG_PATH, Config
from baselines.common.concurrency import iter_parallel_completed
from baselines.common.environment import (
    check_apptainer,
    create_apptainer_environment,
    native_architecture,
)
from baselines.common.harness import run_apptainer_harness
from baselines.common.environment import resolve_instance_image

logger = logging.getLogger(__name__)


class HistoryReductionAgentMixin:
    """Run delayed history policies after tool execution, before the next turn."""
    def execute_actions(self, message):
        result = super().execute_actions(message)
        self.model.reduce_history(self.messages)
        return result


def is_recoverable_source_path(raw_path: str) -> bool:
    """Exclude benchmark artifacts and likely scratch files from fallback patches."""
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    lowered = tuple(part.lower() for part in path.parts)
    if any(part in {"test", "tests", "testing", "__tests__"} for part in lowered[:-1]):
        return False
    name = lowered[-1]
    if name.startswith(("test_", "test-", "debug_", "repro_", "check_")):
        return False
    if name.endswith(("_test.py", ".pyc", ".pyo")):
        return False
    if name in {
        "patch.txt",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "requirements.txt",
        "package-lock.json",
        "yarn.lock",
    }:
        return False
    return ".git" not in lowered


def recover_worktree_patch(env) -> str:
    """Create a best-effort source patch after a terminal agent interruption."""
    status = env.execute(
        {"command": "cd /testbed && git status --porcelain=v1 -z --untracked-files=all"}
    )
    if status.get("returncode") != 0:
        return ""
    records = status.get("output", "").split("\0")
    paths = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4:
            continue
        code, path = record[:2], record[3:]
        if "R" in code or "C" in code:
            index += 1  # porcelain -z stores the original path next
        if is_recoverable_source_path(path):
            paths.append(path)
    paths = list(dict.fromkeys(paths))
    if not paths:
        return ""
    quoted = " ".join(shlex.quote(path) for path in paths)
    # Intent-to-add makes new source files visible to git diff.  The sandbox is
    # disposable, so changing its index does not affect user repositories.
    env.execute({"command": f"cd /testbed && git add -N -- {quoted}"})
    diff = env.execute(
        {"command": f"cd /testbed && git diff --no-ext-diff --binary HEAD -- {quoted}"}
    )
    patch = diff.get("output", "") if diff.get("returncode") == 0 else ""
    return patch if patch.lstrip().startswith("diff --git ") else ""


def recover_terminal_result(result: dict, env) -> dict:
    """Preserve source edits when a recoverable terminal condition stops the agent."""
    status = result.get("exit_status")
    if result.get("submission") or status not in {
        "RepeatedFormatError",
        "LimitsExceeded",
        "TimeExceeded",
    }:
        return result
    recovered = recover_worktree_patch(env)
    return (
        {**result, "exit_status": f"RecoveredPatchAfter{status}", "submission": recovered}
        if recovered
        else result
    )


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def token_usage(trajectory: dict) -> dict:
    prompt = completion = calls = cached = 0
    for message in trajectory.get("messages", []):
        if message.get("role") != "assistant":
            continue
        usage = message.get("extra", {}).get("response", {}).get("usage") or {}
        if usage:
            calls += 1
            prompt += usage.get("prompt_tokens", 0)
            completion += usage.get("completion_tokens", 0)
            cached += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cached_prompt_tokens": cached,
        "calls_with_usage": calls,
    }


def run_instance(
    row: dict, config: Config, output: Path,
    progress: RunBatchProgressManager | None = None,
    model_factory=LitellmModel,
) -> dict:
    instance_id = row["instance_id"]
    path = output / "trajectories" / f"{instance_id}.json"
    defaults = get_config_from_spec(builtin_config_dir / "benchmarks/swebench.yaml")
    env = agent = None
    exit_status = "error"
    if progress is not None:
        progress.on_instance_start(instance_id)
        progress.update_instance_status(instance_id, "Starting ARM64 environment")
    try:
        logger.info("Starting %s", instance_id)
        env = create_apptainer_environment(
            row["image_name"], executable=config.evaluation.apptainer_executable
        )
        model = model_factory(
            model_name=config.agent.model,
            cost_tracking="ignore_errors",
            observation_template=defaults["model"]["observation_template"],
            format_error_template=defaults["model"]["format_error_template"],
            model_kwargs={
                "api_base": config.agent.api_endpoint,
                "api_key": config.agent.api_key,
                "temperature": config.agent.temperature,
                "max_tokens": config.agent.max_tokens,
                "timeout": config.agent.request_timeout,
                "parallel_tool_calls": True,
            },
        )
        agent_class = ProgressTrackingAgent if progress is not None else DefaultAgent
        if hasattr(model, "reduce_history"):
            agent_class = type("HistoryReductionAgent", (HistoryReductionAgentMixin, agent_class), {})
        progress_kwargs = (
            {"progress_manager": progress, "instance_id": instance_id}
            if progress is not None else {}
        )
        agent = agent_class(
            model,
            env,
            **progress_kwargs,
            **{
                **defaults["agent"],
                "step_limit": config.agent.step_limit,
                "cost_limit": 0,
                "max_consecutive_format_errors": config.agent.max_consecutive_format_errors,
                "output_path": path,
            },
        )
        result = agent.run(row["problem_statement"])
        recovered = recover_terminal_result(result, env)
        if recovered is not result:
            result = recovered
            agent.messages[-1]["content"] = result["exit_status"]
            agent.messages[-1]["extra"].update(result)
        trajectory = agent.serialize({"info": {"instance_id": instance_id}})
        trajectory["usage"] = token_usage(trajectory)
        save_json(path, trajectory)
        exit_status = result.get("exit_status", "unknown")
        return {
            "instance_id": instance_id,
            "model_name_or_path": config.agent.model,
            "model_patch": result.get("submission", ""),
            "exit_status": result.get("exit_status", ""),
        }
    except Exception as exc:
        exit_status = f"Error: {type(exc).__name__}"
        logger.exception("Agent failed for %s", instance_id)
        if agent is not None:
            save_json(
                path,
                agent.serialize(
                    {"info": {"instance_id": instance_id, "error": str(exc)}}
                ),
            )
        return {
            "instance_id": instance_id,
            "model_name_or_path": config.agent.model,
            "model_patch": "",
            "exit_status": "error",
            "error": str(exc),
        }
    finally:
        try:
            if env is not None:
                env.cleanup()
        finally:
            if progress is not None:
                progress.on_instance_end(instance_id, exit_status)

