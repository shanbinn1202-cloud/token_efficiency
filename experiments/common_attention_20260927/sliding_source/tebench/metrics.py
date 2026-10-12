"""One metric definition for every method and backend; no missing-case filtering."""
import json
from pathlib import Path


def ratio(a, b):
    return a / b if b else None


def aggregate(output, instance_ids):
    output = Path(output)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    predictions = {p["instance_id"]: p for p in json.loads((output / "predictions.json").read_text())}
    timing = json.loads((output / "timings.json").read_text())
    cases, requests = [], []
    for iid in instance_ids:
        path = output / "trajectories" / f"{iid}.json"
        trajectory = json.loads(path.read_text()) if path.exists() else {}
        calls = trajectory.get("memory", {}).get("requests", [])
        requests.extend(calls)
        agent = [r for r in calls if r["kind"] == "agent"]
        cases.append({"instance_id": iid, "exit_status": predictions.get(iid, {}).get("exit_status", "missing"),
            "resolved": iid in report.get("resolved_ids", []) if report else None,
            "e2e_seconds": timing["instances"].get(iid, {}).get("seconds"),
            "turns": len(agent), "agent_prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in agent),
            "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in calls)})
    n = len(instance_ids)
    calls = [r for r in requests if r["kind"] == "agent"]
    measured_calls = [r for r in calls if r.get("usage")]
    rewriter_tokens = sum((r.get("backend", {}).get("goal") or {}).get("slm_output_tokens", 0) or 0 for r in requests)
    rewriter_calls = sum(bool((r.get("backend", {}).get("goal") or {}).get("slm_attempted")) for r in requests)
    queries = [(r.get("backend", {}).get("query_embedding") or {}) for r in requests]
    query_forwards = [q for q in queries if q.get("source") == "goal"
                      and q.get("published") and not q.get("cache_hit")]
    tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in requests) + rewriter_tokens
    service = sum(r.get("seconds", 0) for r in requests)
    agent_tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in calls)
    diet_calls = [r for r in requests if r["kind"] == "agentdiet"]
    wall = sum(timing["segments"].values())
    complete = all(c["exit_status"] not in {"missing", "error"} and c["e2e_seconds"] is not None for c in cases)
    # Invalid model patches remain failures in the accuracy denominator.
    # Legacy reports without this classification retain the conservative rule.
    grading_error_count = report.get("infrastructure_error_instances", report.get("error_instances", 0)) if report else None
    return {
        "schema_version": 1, "instances": n,
        "accuracy": ratio(len(set(report.get("resolved_ids", [])) & set(instance_ids)), n) if report else None,
        "e2e_latency_seconds_mean": ratio(sum(c["e2e_seconds"] or 0 for c in cases), n) if complete else None,
        "e2e_latency_seconds_total": sum(c["e2e_seconds"] or 0 for c in cases),
        "batch_wall_seconds": wall,
        "throughput_instances_per_hour": ratio(n * 3600, wall) if complete else None,
        # Keep both denominators: batch wall throughput includes queueing and
        # tool/environment time. Request-time throughput includes queueing,
        # prefill, decode and policy work; it is not pure GPU decode speed.
        "throughput_output_tokens_per_second": ratio(tokens, wall),
        "batch_output_tokens_per_second": ratio(tokens, wall),
        "agent_output_tokens": agent_tokens,
        "agent_output_tokens_per_second": ratio(agent_tokens, wall),
        "agentdiet_calls": len(diet_calls),
        "agentdiet_output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in diet_calls),
        "agentdiet_request_seconds": sum(r.get("seconds", 0) for r in diet_calls),
        "inference_output_tokens_per_second": ratio(tokens, service),
        "avg_turns": ratio(len(calls), n) if complete else None,
        "avg_context_tokens": ratio(sum(r["usage"].get("prompt_tokens", 0) for r in measured_calls), len(measured_calls)),
        "agent_calls_with_usage": len(measured_calls),
        "agent_calls": len(calls), "auxiliary_calls": len(requests) - len(calls) + rewriter_calls + len(query_forwards),
        "goal_rewriter_output_tokens": rewriter_tokens,
        "goal_query_forwards": len(query_forwards),
        "goal_query_input_tokens": sum(q.get("token_count", 0) or 0 for q in query_forwards),
        "goal_query_request_seconds": sum(q.get("forward_ms", 0) or 0 for q in query_forwards) / 1000,
        "output_tokens": tokens, "model_request_seconds": service,
        "eligible_for_comparison": bool(complete and requests and report and not grading_error_count and not report.get("incomplete_instances")
                                        and all(r.get("usage") and not r.get("error") for r in requests)),
        "grading_errors": report.get("infrastructure_error_ids", report.get("error_ids", [])) if report else None,
        "invalid_patch_ids": report.get("invalid_patch_ids", []) if report else None,
        "cases": cases,
    }
