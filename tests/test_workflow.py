import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from baselines.common import harness
from baselines.common.environment import native_architecture
from baselines.common.environment import resolve_instance_image
from baselines.common.prepare import legacy_setup_script, repo_setup_script
from baselines.common.runner import (
    is_recoverable_source_path,
    recover_terminal_result,
    recover_worktree_patch,
    token_usage,
)


def test_manifest_checks_commit_architecture_and_missing_images(tmp_path):
    image = tmp_path / "test.sif"
    image.touch()
    path = tmp_path / "manifest.json"
    manifest = {
        "architecture": native_architecture(),
        "images": {"repo__1": {"image": str(image), "base_commit": "abc"}},
    }
    path.write_text(json.dumps(manifest))
    assert resolve_instance_image(
        {"instance_id": "repo__1", "base_commit": "abc"}, "", str(path)
    ) == str(image)
    with pytest.raises(ValueError, match="commit"):
        resolve_instance_image(
            {"instance_id": "repo__1", "base_commit": "wrong"}, "", str(path)
        )
    with pytest.raises(ValueError, match="missing"):
        resolve_instance_image({"instance_id": "repo__2"}, "", str(path))
    image.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_instance_image(
            {"instance_id": "repo__1", "base_commit": "abc"}, "", str(path)
        )
    manifest["architecture"] = "not-native"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="architecture"):
        resolve_instance_image(
            {"instance_id": "repo__1", "base_commit": "abc"}, "", str(path)
        )


def test_native_build_does_not_include_gold_or_future_commits():
    row = {
        "instance_id": "astropy__astropy-7166",
        "repo": "astropy/astropy",
        "version": "1.3",
        "base_commit": "a" * 40,
        "problem_statement": "task",
        "test_patch": "",
        "patch": "SECRET_REFERENCE_PATCH",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
    }
    environment = legacy_setup_script(row)
    repository = repo_setup_script(row)
    assert "python=3.6.13" in environment and "numpy==1.16.0" in environment
    assert "setuptools_scm==4.1.2" in environment
    assert "ld_impl_linux-64" not in environment
    assert "SECRET_REFERENCE_PATCH" not in environment + repository
    assert f"git fetch --depth 1 origin {row['base_commit']}" in repository
    assert "git remote remove origin" in repository
    with pytest.raises(ValueError, match="recipe"):
        legacy_setup_script({**row, "version": "unknown"})


def test_usage_keeps_cached_tokens_in_prompt_total():
    message = {
        "role": "assistant",
        "extra": {
            "response": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 80},
                }
            }
        },
    }
    assert token_usage({"messages": [message, message, {"role": "assistant"}]}) == {
        "prompt_tokens": 200,
        "completion_tokens": 20,
        "total_tokens": 220,
        "cached_prompt_tokens": 160,
        "calls_with_usage": 2,
    }


def test_fallback_patch_path_filter_keeps_source_only():
    assert is_recoverable_source_path("src/pkg/core.py")
    assert is_recoverable_source_path("module.py")
    for path in (
        "/absolute.py",
        "../outside.py",
        "tests/test_core.py",
        "src/__tests__/core.ts",
        "test_repro.py",
        "debug_issue.py",
        "pyproject.toml",
        "patch.txt",
    ):
        assert not is_recoverable_source_path(path)


def test_fallback_patch_includes_untracked_source_and_excludes_artifacts():
    class Environment:
        def __init__(self):
            self.commands = []

        def execute(self, action):
            command = action["command"]
            self.commands.append(command)
            if "status --porcelain" in command:
                return {
                    "returncode": 0,
                    "output": " M src/core.py\0?? src/new.py\0?? test_repro.py\0 M pyproject.toml\0",
                }
            if "git diff" in command:
                assert "src/core.py" in command and "src/new.py" in command
                assert "test_repro.py" not in command and "pyproject.toml" not in command
                return {"returncode": 0, "output": "diff --git a/src/core.py b/src/core.py\n"}
            return {"returncode": 0, "output": ""}

    env = Environment()
    assert recover_worktree_patch(env).startswith("diff --git ")
    assert any("git add -N" in command for command in env.commands)


def test_terminal_patch_recovery_covers_limits_and_leaves_other_exits_alone():
    class Environment:
        def execute(self, action):
            if "status --porcelain" in action["command"]:
                return {"returncode": 0, "output": " M src/core.py\0"}
            if "git diff" in action["command"]:
                return {"returncode": 0, "output": "diff --git a/src/core.py b/src/core.py\n"}
            return {"returncode": 0, "output": ""}

    for status in ("RepeatedFormatError", "LimitsExceeded", "TimeExceeded"):
        assert recover_terminal_result(
            {"exit_status": status, "submission": ""}, Environment()
        ) == {
            "exit_status": f"RecoveredPatchAfter{status}",
            "submission": "diff --git a/src/core.py b/src/core.py\n",
        }
    submitted = {"exit_status": "Submitted", "submission": "patch"}
    assert recover_terminal_result(submitted, Environment()) is submitted


@pytest.mark.parametrize("status", ["PASSED", "FAILED", "ERROR", "SKIPPED"])
def test_legacy_empty_test_id_preserves_real_status(status):
    node = "astropy/units/tests/test_units.py::test_compose_roundtrip"
    log = f"{node}[unit0] {status}\n{node}[m] PASSED\n"
    normalized = harness.normalize_legacy_test_ids("astropy__astropy-7606", log)
    assert normalized == f"{node}[] {status}\n{node}[m] PASSED\n"
    assert harness.normalize_legacy_test_ids("astropy__astropy-7671", log) == log
    assert harness.normalize_legacy_test_ids("astropy__astropy-7606", "") == ""


def test_official_grader_scope_failures_and_fresh_regrading(tmp_path, monkeypatch):
    ids = ["pass", "fail", "timeout", "badpatch", "empty"]
    monkeypatch.setattr(harness, "check_apptainer", lambda executable: None)
    monkeypatch.setattr(
        harness,
        "load_dataset",
        lambda *a, **k: [{"instance_id": key} for key in ids + ["outside"]],
    )
    monkeypatch.setattr(
        harness,
        "resolve_instance_image",
        lambda row, default, manifest: row["instance_id"],
    )
    monkeypatch.setattr(
        harness,
        "make_test_spec",
        lambda row, **k: SimpleNamespace(
            instance_id=row["instance_id"],
            instance_image_key=row["instance_id"],
            eval_script="test script",
            repo="astropy/astropy",
            version="4.3",
            FAIL_TO_PASS=["test_fix"],
            PASS_TO_PASS=["test_keep"],
        ),
    )
    environments = []

    class Environment:
        def __init__(self, image, **kwargs):
            self.image = image
            self.sandbox_dir = tmp_path / uuid4().hex
            self.sandbox_dir.mkdir()
            self.cleaned = False
            environments.append(self)

        def execute(self, action, *, timeout=None):
            if action["command"] != "/bin/bash /coact_eval.sh":
                assert (self.sandbox_dir / "coact_patch.diff").read_text() == "patch"
                return {
                    "returncode": 1 if self.image == "badpatch" else 0,
                    "output": "apply",
                }
            assert timeout == 77
            if self.image == "timeout":
                return {
                    "returncode": -1,
                    "output": "partial",
                    "exception_info": "timed out",
                }
            status = "FAILED" if self.image == "fail" else "PASSED"
            return {
                "returncode": 1 if status == "FAILED" else 0,
                "output": f"{START_TEST_OUTPUT}\n{status} test_fix\nPASSED test_keep\n{END_TEST_OUTPUT}",
            }

        def cleanup(self):
            self.cleaned = True

    monkeypatch.setattr(harness, "create_apptainer_environment", Environment)
    predictions = [
        {
            "instance_id": key,
            "model_name_or_path": "m",
            "model_patch": "" if key == "empty" else "patch",
        }
        for key in ids + ["outside"]
    ]
    kwargs = {
        "predictions": predictions,
        "dataset_name": "verified",
        "split": "test",
        "instance_ids": ids,
        "max_workers": 8,
        "timeout": 77,
        "run_id": "test",
        "report_path": tmp_path / "report.json",
        "executable": "apptainer",
    }
    report = json.loads(harness.run_apptainer_harness(**kwargs).read_text())
    assert (
        report["total_instances"],
        report["resolved_instances"],
        report["unresolved_instances"],
        report["empty_patch_instances"],
        report["error_instances"],
    ) == (5, 1, 1, 1, 2)
    assert report['invalid_patch_ids'] == ['badpatch']
    assert report['infrastructure_error_ids'] == ['timeout']
    assert "outside" not in report["submitted_ids"]
    assert len(environments) == 4 and all(env.cleaned for env in environments)
    predictions[0]["model_patch"] = ""
    assert (
        json.loads(harness.run_apptainer_harness(**kwargs).read_text())[
            "resolved_instances"
        ]
        == 0
    )
    assert not (tmp_path / "instances/pass/report.json").exists()
