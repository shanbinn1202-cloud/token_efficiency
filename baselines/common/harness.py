"""Run official SWE-bench test scripts and grading with Apptainer, not Docker.

The execution adapter uses Apptainer. Test specifications, patch application
commands and FAIL_TO_PASS/PASS_TO_PASS grading come from the pinned SWE-bench
submodule. Every prediction is graded in a fresh image, never the agent sandbox.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from rich.live import Live
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.test_spec.test_spec import make_test_spec

from baselines.common.concurrency import iter_parallel_completed
from baselines.common.environment import (
    check_apptainer,
    check_swebench_platform,
    create_apptainer_environment,
    native_architecture,
)
from baselines.common.environment import resolve_instance_image

logger = logging.getLogger(__name__)


class InvalidModelPatch(RuntimeError):
    """All patch applications failed normally; this is a model outcome."""


def normalize_legacy_test_ids(instance_id: str, output: str) -> str:
    """Match pytest 3.3.1's empty-ID fallback to the dataset's test name.

    Astropy's ids=_unit_as_str returns '' for the dimensionless unit. Pytest
    3.3.1 treats that as false and falls back to unit0. Preserve the actual
    status; this mapping must never turn a failed/missing test into a pass.
    """
    if instance_id != "astropy__astropy-7606":
        return output
    node = "astropy/units/tests/test_units.py::test_compose_roundtrip"
    return re.sub(
        rf"^{re.escape(node)}\[unit0\](?=\s+(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b)",
        f"{node}[]",
        output,
        flags=re.MULTILINE,
    )


def run_apptainer_harness(
    *,
    predictions: list[dict[str, Any]],
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    max_workers: int,
    timeout: int,
    run_id: str,
    report_path: Path,
    executable: str,
    image_manifest: str = "",
) -> Path:
    check_apptainer(executable)
    check_swebench_platform()
    requested = set(instance_ids)
    prediction_map = {
        p["instance_id"]: p for p in predictions if p["instance_id"] in requested
    }
    instances = {
        row["instance_id"]: dict(row)
        for row in load_dataset(dataset_name, split=split)
        if row["instance_id"] in requested
    }
    if missing := requested - instances.keys():
        raise ValueError(
            f"Instances missing from evaluation dataset: {sorted(missing)}"
        )
    if any(
        not re.fullmatch(r"[\w.-]+", key) or key in {".", ".."} for key in requested
    ):
        raise ValueError("Unsafe instance ID for evaluation artifact directory")

    log_root = report_path.parent / "instances"
    log_root.mkdir(parents=True, exist_ok=True)
    progress = RunBatchProgressManager(len(requested), report_path.parent / "eval_statuses.yaml")

    def grade(instance_id: str) -> tuple[str, str]:
        prediction = prediction_map.get(instance_id)
        log_dir = log_root / instance_id
        log_dir.mkdir(parents=True, exist_ok=True)
        # Always grade the current patch. In particular, never reuse a report
        # left by a previous --resume/--force-rerun with different predictions.
        for filename in (
            "report.json",
            "error.txt",
            "test_output.txt",
            "test_output.raw.txt",
            "patch.diff",
            "eval.sh",
            "patch_apply.log",
        ):
            (log_dir / filename).unlink(missing_ok=True)
        if prediction is None:
            return instance_id, "incomplete"
        if not (prediction.get("model_patch") or "").strip():
            return instance_id, "empty_patch"

        env = None
        try:
            spec = make_test_spec(
                instances[instance_id], namespace="swebench", arch=native_architecture()
            )
            patch = prediction["model_patch"]
            (log_dir / "patch.diff").write_text(patch, encoding="utf-8")
            (log_dir / "eval.sh").write_text(spec.eval_script, encoding="utf-8")
            image = resolve_instance_image(
                instances[instance_id], spec.instance_image_key, image_manifest
            )
            progress.update_instance_status(instance_id, "Starting ARM64 environment")
            env = create_apptainer_environment(image, executable=executable)
            # /tmp is mounted separately under --containall. Put these files at
            # the sandbox root so they are visible inside without host binds.
            (env.sandbox_dir / "coact_patch.diff").write_text(patch, encoding="utf-8")
            (env.sandbox_dir / "coact_eval.sh").write_text(
                spec.eval_script, encoding="utf-8"
            )
            apply_log = []
            progress.update_instance_status(instance_id, "Applying patch")
            for command in GIT_APPLY_CMDS:
                result = env.execute({"command": f"{command} /coact_patch.diff"})
                apply_log.append(f"$ {command}\n{result['output']}\n")
                (log_dir / "patch_apply.log").write_text(
                    "".join(apply_log), encoding="utf-8"
                )
                if result["returncode"] == 0:
                    break
                if result.get("exception_info"):
                    raise RuntimeError(result["exception_info"])
            else:
                raise InvalidModelPatch(
                    "Model patch could not be applied; see patch_apply.log"
                )

            progress.update_instance_status(instance_id, "Running SWE-bench tests")
            result = env.execute(
                {"command": "/bin/bash /coact_eval.sh"}, timeout=timeout
            )
            test_log = log_dir / "test_output.txt"
            (log_dir / "test_output.raw.txt").write_text(result["output"], encoding="utf-8")
            test_log.write_text(
                normalize_legacy_test_ids(instance_id, result["output"]), encoding="utf-8"
            )
            if result.get("exception_info"):
                raise RuntimeError(result["exception_info"])
            # A failing test may produce a nonzero exit code; verdicts must come
            # from the official test parser rather than the shell exit code.
            report = get_eval_report(
                test_spec=spec,
                prediction=prediction,
                test_log_path=str(test_log),
                include_tests_status=True,
            )
            (log_dir / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return instance_id, "resolved" if report[instance_id][
                "resolved"
            ] else "unresolved"
        except InvalidModelPatch as exc:
            (log_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            return instance_id, "invalid_patch"
        except Exception as exc:
            logger.exception("Apptainer evaluation failed for %s", instance_id)
            (log_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            return instance_id, "error"
        finally:
            if env is not None:
                env.cleanup()

    statuses: dict[str, list[str]] = {
        key: []
        for key in ("resolved", "unresolved", "empty_patch", "invalid_patch", "error", "incomplete")
    }
    def grade_with_progress(instance_id):
        progress.on_instance_start(instance_id)
        status = "error"
        try:
            _, status = grade(instance_id)
            return instance_id, status
        finally:
            progress.on_instance_end(instance_id, status)

    with Live(progress.render_group, refresh_per_second=4) as live:
        live.console.print("[bold cyan]SWE-bench evaluation (ARM64 / Apptainer)[/bold cyan]")
        for instance_id, status in iter_parallel_completed(
            sorted(requested), grade_with_progress, max_workers
        ):
            statuses[status].append(instance_id)
    completed = statuses["resolved"] + statuses["unresolved"]
    report = {
        "execution_backend": "apptainer",
        "architecture": native_architecture(),
        "image_manifest": image_manifest,
        "run_id": run_id,
        "total_instances": len(requested),
        "submitted_instances": len(prediction_map),
        "completed_instances": len(completed),
        "submitted_ids": sorted(prediction_map),
        "completed_ids": sorted(completed),
    }
    for status, ids in statuses.items():
        report[f"{status}_instances"] = len(ids)
        report[f"{status}_ids"] = sorted(ids)
    # Preserve SWE-bench's broad error count, and separately identify model
    # patch failures so metric eligibility does not filter out bad predictions.
    report["infrastructure_error_ids"] = sorted(statuses["error"])
    report["infrastructure_error_instances"] = len(statuses["error"])
    report["error_ids"] = sorted(statuses["error"] + statuses["invalid_patch"])
    report["error_instances"] = len(report["error_ids"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path
