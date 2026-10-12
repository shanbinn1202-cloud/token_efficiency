"""Exercise the unified runner's save/resume/evaluation boundaries without a GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('method,backend_name', [('vanilla', 'transformers'),
    ('sliding_window', 'vllm'), ('agentkv', 'vllm'), ('agentdiet', 'vllm')])
def test_runner_evaluates_all_cases_and_rejects_changed_resume(tmp_path, monkeypatch, method, backend_name):
    import datasets
    import baselines.common.environment as images
    import baselines.common.runner as runner
    import baselines.common.harness as harness
    import tebench.backends as backends
    import tebench.provenance as provenance
    from tebench.run import main
    from tebench.methods import policy_snapshot
    rows = [{"instance_id": "a"}, {"instance_id": "b"}]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(images, "resolve_instance_image", lambda *a: "/tmp/test.sif")
    monkeypatch.setattr(provenance, "snapshot", lambda c, m, ids: {
        "configuration": c, "method": m, "policy": policy_snapshot(m, c), "instance_ids": ids})
    created = []
    import threading
    state = threading.local()
    def reset():
        state.active = False
    monkeypatch.setattr(backends, "create", lambda *a: created.append(1) or SimpleNamespace(
        complete=lambda **k: None, reset=reset, close=lambda: None))
    def run(row, config, output, **kwargs):
        assert not getattr(state, 'active', False), 'Previous task session was not released'
        state.active = True
        runner.save_json(output / "trajectories" / f"{row['instance_id']}.json", {"memory": {"requests": [
            {"kind": "agent", "seconds": 1, "usage": {"prompt_tokens": 3, "completion_tokens": 2}}]}})
        return {"instance_id": row["instance_id"], "exit_status": "Submitted", "model_patch": "patch"}
    monkeypatch.setattr(runner, "run_instance", run)
    def grade(**kwargs):
        assert kwargs["instance_ids"] == ["a", "b"]
        runner.save_json(kwargs["report_path"], {"resolved_ids": ["a"], "error_instances": 0, "incomplete_instances": 0})
    monkeypatch.setattr(harness, "run_apptainer_harness", grade)
    args = ["--method", method, "--backend", backend_name,
            "--output", str(tmp_path / "out"), "--set", "dataset.count=2",
            "--set", "evaluation.workers=1", "--set", "backend.cuda_graph=false",
            "--set", "backend.torch_compile=false"]
    main(args)
    metrics = json.loads((tmp_path / "out/metrics.json").read_text())
    assert metrics["accuracy"] == 0.5 and metrics["eligible_for_comparison"]
    main(args + ["--resume"])
    main(args + ["--eval-only"])
    assert len(created) == 1
    with pytest.raises(ValueError, match="changed"):
        main(args + ["--resume", "--set", "cache.kv_budget=8192"])


def test_dry_run_never_creates_backend(monkeypatch, capsys):
    import tebench.backends as backends
    from tebench.run import main
    monkeypatch.setattr(backends, "create", lambda *a: pytest.fail("dry-run loaded a model"))
    main(["--method", "sliding_window", "--dry-run", "--set", "cache.kv_budget=4096",
          "--set", "evaluation.workers=1", "--set", "backend.cuda_graph=false",
          "--set", "backend.torch_compile=false"])
    result = json.loads(capsys.readouterr().out)
    assert result["policy"]["kv_budget"] == 4096
    assert result["policy"]["selection"] == "recent_tokens"
