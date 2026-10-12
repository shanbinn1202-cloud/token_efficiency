import json
import pytest
from tebench.metrics import aggregate
from tebench.compare import compare


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def fixture(p):
    write(p / "predictions.json", [{"instance_id": "a", "exit_status": "Submitted"}, {"instance_id": "b", "exit_status": "LimitsExceeded"}])
    write(p / "timings.json", {"instances": {"a": {"seconds": 10}, "b": {"seconds": 20}}, "segments": {"x": 20}})
    write(p / "report.json", {"resolved_ids": ["a"], "error_instances": 0, "incomplete_instances": 0})
    for iid in ["a", "b"]:
        write(p / "trajectories" / f"{iid}.json", {"memory": {"requests": [
            {"kind": "agent", "seconds": 5, "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            {"kind": "gate", "seconds": 1, "usage": {"prompt_tokens": 20, "completion_tokens": 1}}]}})


def test_metric_denominators_and_concurrent_wall_time(tmp_path):
    fixture(tmp_path)
    m = aggregate(tmp_path, ["a", "b"])
    assert m["accuracy"] == 0.5
    assert m["e2e_latency_seconds_mean"] == 15
    assert m["throughput_instances_per_hour"] == 360
    assert m["throughput_output_tokens_per_second"] == 1.1
    assert m["avg_turns"] == 1 and m["avg_context_tokens"] == 100
    assert m["auxiliary_calls"] == 2
    assert m["eligible_for_comparison"]


def test_missing_cases_stay_in_accuracy_denominator(tmp_path):
    fixture(tmp_path)
    m = aggregate(tmp_path, ["a", "b", "c"])
    assert m["accuracy"] == 1 / 3
    assert m["e2e_latency_seconds_mean"] is None
    assert not m["eligible_for_comparison"]


def test_hidden_goal_queries_are_counted_without_double_counting_request_time(tmp_path):
    fixture(tmp_path)
    p = tmp_path / 'trajectories/a.json'
    d = json.loads(p.read_text())
    d['memory']['requests'][0]['backend'] = {'query_embedding': {
        'source': 'goal', 'published': True, 'cache_hit': False,
        'token_count': 35, 'forward_ms': 200}}
    write(p, d)
    m = aggregate(tmp_path, ['a', 'b'])
    assert m['goal_query_forwards'] == 1 and m['auxiliary_calls'] == 3
    assert m['goal_query_input_tokens'] == 35
    assert m['goal_query_request_seconds'] == .2
    assert m['model_request_seconds'] == 12 and m['output_tokens'] == 22


def test_invalid_model_patch_is_a_failure_not_an_excluded_run(tmp_path):
    fixture(tmp_path)
    report = {'resolved_ids': ['a'], 'error_instances': 1, 'error_ids': ['b'],
              'invalid_patch_ids': ['b'], 'infrastructure_error_instances': 0,
              'infrastructure_error_ids': [], 'incomplete_instances': 0}
    write(tmp_path/'report.json', report)
    m = aggregate(tmp_path, ['a', 'b'])
    assert m['accuracy'] == .5 and m['eligible_for_comparison']
    assert m['invalid_patch_ids'] == ['b'] and m['grading_errors'] == []
    report.update(infrastructure_error_instances=1, infrastructure_error_ids=['b'],
                  invalid_patch_ids=[])
    write(tmp_path/'report.json', report)
    assert not aggregate(tmp_path, ['a', 'b'])['eligible_for_comparison']


def test_comparison_rejects_kernel_or_policy_mismatch(tmp_path, capsys):
    from copy import deepcopy
    from tebench.config import load
    c = load()
    run = {"configuration": c, "method": "sliding_window", "policy": {"budget_scope": "input"},
           "instance_ids": ["a", "b"], "external": {}, "source_sha256": {}}
    for name in ["one", "two"]:
        p = tmp_path / name
        fixture(p)
        write(p / "metrics.json", aggregate(p, ["a", "b"]))
        write(p / "run.json", run)
    compare([tmp_path / "one", tmp_path / "two"])
    assert "sliding_window" in capsys.readouterr().out
    changed = deepcopy(run)
    changed["configuration"]["backend"]["sdpa_kernel"] = "math"
    write(tmp_path / "two/run.json", changed)
    with pytest.raises(ValueError, match="execution"):
        compare([tmp_path / "one", tmp_path / "two"])
    changed = deepcopy(run)
    changed["policy"]["budget_scope"] = "input_and_generation"
    write(tmp_path / "two/run.json", changed)
    with pytest.raises(ValueError, match="policy"):
        compare([tmp_path / "one", tmp_path / "two"])
