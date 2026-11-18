"""Validate each formal SWE-150 entry point on real tasks in one interactive allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import benchmark

COUNTS = {"random": 24, "sliding_window": 1, "agentkv": 1, "claude_compression": 1}
STEPS = 2
CHECKS = HERE / "probes.py"


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def peak_overlap(intervals):
    events = []
    for start, end in intervals:
        require(end > start, "Invalid recorded interval")
        events.extend([(start, 1), (end, -1)])
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def task_evidence(evaluation, row):
    iid = row["instance_id"]
    completed = load(evaluation / (iid + ".completed.json"))
    prediction, timing = completed["prediction"], completed["timing"]
    require(timing["instance_id"] == prediction["instance_id"] == iid and
            not timing.get("error") and not timing.get("close_error"), f"{iid}: session failed or did not close")
    allowed_exits = {"Submitted", "LimitsExceeded", "LimitsExceeded_InformalSubmission"}
    require(prediction["exit_status"] in allowed_exits, f"{iid}: agent terminated with {prediction['exit_status']}")
    calls = [json.loads(line) for line in (evaluation / (iid + ".requests.jsonl")).read_text().splitlines()]
    main = [call for call in calls if call["kind"] == "main"]
    require(0 < len(main) <= STEPS and all(not c.get("error") for c in calls),
            f"{iid}: missing main completion or received a request error")
    main_tokens = sum(call["usage"]["completion_tokens"] for call in main)
    require(main_tokens == timing["main_completion_tokens"] and main_tokens > 0, f"{iid}: invalid token accounting")
    messages = load(evaluation / (iid + ".traj.json"))["messages"]
    commands = [action["command"] for m in messages if m["role"] == "assistant"
                for action in m.get("extra", {}).get("actions", []) if action.get("command")]
    observations = [m["extra"] for m in messages if m["role"] == "user" and "returncode" in m.get("extra", {})]
    submitted = prediction["exit_status"] == "Submitted"
    tool_executed = bool(commands and (submitted or any(not obs.get("exception_info") for obs in observations)))
    checkout = load(evaluation / (iid + ".environment.json"))["testbed"]
    require(checkout["returncode"] == 0 and row["base_commit"] in checkout["output"] and
            "/opt/miniconda3/envs/testbed/bin/python" in checkout["output"], f"{iid}: unexpected checkout or interpreter")
    report = load(evaluation / "grades" / iid / "report.json")
    require(report["submitted_ids"] == [iid] and not report["infrastructure_error_ids"],
            f"{iid}: missing scorer receipt")
    return {"instance_id": iid, "repo": row["repo"], "exit_status": prediction["exit_status"],
            "turns": timing["turns"], "commands": len(commands), "observations": len(observations),
            "tool_executed": tool_executed, "main_completion_tokens": main_tokens,
            "nonempty_patch": bool(prediction["model_patch"].strip()),
            "task_interval": [timing["start"], timing["end"]],
            "request_intervals": [[c["start"], c["end"]] for c in main]}


def validate_agent(method, out, job):
    evaluation = out / "evaluation"
    count = COUNTS[method]
    rows = load(ROOT / "assets/dataset-first150.json")[:count]
    ids = [row["instance_id"] for row in rows]
    spec = load(evaluation / "run-spec.json")
    require(spec == {"method": method, "expected": count, "instance_ids": ids,
                     "step_limit": STEPS, "pool_size": 24}, "Unexpected pilot task list or limits")
    details = [task_evidence(evaluation, row) for row in rows]
    repos = sorted({t["repo"] for t in details})
    tool_coverage = {r: sum(t["tool_executed"] for t in details if t["repo"] == r) for r in repos}
    require(all(tool_coverage.values()), "At least one repository has no successful real testbed tool execution")
    main_tokens = sum(t["main_completion_tokens"] for t in details)
    result = load(evaluation / "result.json")
    require(result["complete"] and result["graded"] == count and result["completed"] == count and
            result["main_completion_tokens"] == main_tokens and not result["infrastructure_error_ids"] and
            not result["failed_ids"] and result["output_tokens_per_second"] > 0, "Grading or accounting failed")

    placements = {role: load(out / (role + "-placement.json")) for role in ("engine", "agent", "grade")}
    require(all(p["job_id"] == job and p["role"] == role and len(set(p["physical_cores"])) == 16
                and p["scratch"]["nvme_devices"] for role, p in placements.items()), "Missing split placement")
    engine, agent = placements["engine"], placements["agent"]
    require(engine["hostname"] == agent["hostname"] == placements["grade"]["hostname"] and
            agent["peer_step_id"] == engine["step_id"] and
            set(engine["physical_cores"]).isdisjoint(agent["physical_cores"]), "Engine and agent cores overlap")
    config = load(out / "effective-config.json")
    require(config["scheduler_config"] == {"max_num_seqs": 24, "max_num_batched_tokens": 16384, "async_scheduling": True}
            and config["model_config"] == {"max_model_len": 262144, "seed": 42}
            and config["cache_config"]["kv_cache_memory_bytes"] == 25 * 2**30, "Unexpected vLLM settings")
    if method in benchmark.POLICIES:
        subprocess.run([sys.executable, "-B", str(CHECKS), "native", "--out", str(out),
                        "--method", method], check=True)
        engine_check = load(out / "native-check.json")
        require(engine_check["passed"] and engine_check["method"] == method and
                engine_check["policy"] == benchmark.POLICIES[method], "Native selector postcheck failed")
        eviction = load(out / "native-live-check.json")
        require(eviction["passed"], "The dedicated long-history selector probe did not show an actual eviction")
        engine_check = {"pilot_tasks": engine_check, "long_history": eviction}
    else:
        engine_check = load(out / "engine-smoke.json")
        require(engine_check["passed"] and engine_check["method"] == method, "Engine smoke test failed")
        if method == "claude_compression":
            compact = load(out / "compact-check.json")
            require(compact["passed"] and compact["version"] == "2.1.325" and
                    compact["command"] == "/compact" and compact["before_tokens"] > 8192 and
                    compact["auxiliary_completion_tokens"] > 0 and compact["continued_main_tokens"] > 0,
                    "Live /compact or post-compaction continuation failed")
            engine_check = {"thinking_history_smoke": engine_check, "official_compact": compact}
    return {"passed": True, "tasks": count, "task_results": details, "graded": result["graded"],
            "resolved": result["resolved"], "repo_tool_coverage": tool_coverage,
            "peak_active_task_intervals": peak_overlap(t["task_interval"] for t in details),
            "peak_inflight_main_requests": peak_overlap(span for t in details for span in t["request_intervals"]),
            "turns": result["turns"], "main_completion_tokens": main_tokens,
            "auxiliary_completion_tokens": result["auxiliary_completion_tokens"], "hostname": engine["hostname"],
            "memory_enforcement": {r: p["cgroup"]["enforcement"] for r, p in placements.items()},
            "engine_check": engine_check}


def gold_control(out):
    out.mkdir()
    command = benchmark.slurm_step("grade", [sys.executable, "-B", str(CHECKS), "gold",
                                             "--out", str(out)], out, Path(os.environ["TMPDIR"]))
    with (out / "grading.log").open("x") as log:
        subprocess.run(command, env=benchmark.step_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    result = load(out / "gold-check.json")
    require(result["passed"] and result["graded"] == result["resolved"] == 1 and result["accuracy_percent"] == 100,
            "The official reference patch did not pass grading")
    return result


def error_evidence():
    return {"passed": False, "error": traceback.format_exc(), "time": time.time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("local", type=Path)
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    partition = os.environ["SLURM_JOB_PARTITION"]
    out, local = args.out.resolve(), args.local.resolve()
    require(partition == "ghx4-interactive" and out == HERE / "runs" / job, "Expected isolated interactive run")
    require(local == Path(os.environ["GOAL_LOCAL_ROOT"]).resolve(), "Unexpected staged runtime path")
    manifest_sha256 = hashlib.sha256((ROOT / "files.json").read_bytes()).hexdigest()
    summary = {"job_id": job, "partition": partition, "manifest_sha256": manifest_sha256, "started": time.time(),
               "finished": False, "passed": False, "methods": {}, "gold": None}
    receipt = out / "verification.json"
    require(not receipt.exists(), "A verification receipt already exists in this allocation")
    benchmark.write(receipt, summary)
    for method in benchmark.METHODS:
        run = out / method
        print(f"Starting {method}", flush=True)
        try:
            run.mkdir()
            with benchmark.local_port() as port:
                benchmark.run(method, run, local, port, probe_count=COUNTS[method], probe_steps=STEPS)
            entry = validate_agent(method, run, job)
        except Exception:
            entry = error_evidence()
            print(entry["error"], file=sys.stderr, flush=True)
        summary["methods"][method] = entry
        benchmark.write(receipt, summary)
        print(f"{method} passed={entry['passed']}", flush=True)
        # The first completed scoring pass also exercises the official tests on a
        # known-correct patch. Truncated two-turn agent trials can yield empty patches.
        if summary["gold"] is None and (run / "evaluation/result.json").exists():
            try:
                summary["gold"] = gold_control(out / "gold")
            except Exception:
                summary["gold"] = error_evidence()
                print(summary["gold"]["error"], file=sys.stderr, flush=True)
            benchmark.write(receipt, summary)
    summary["finished"] = True
    summary["end_time"] = time.time()
    summary["passed"] = bool(all(x["passed"] for x in summary["methods"].values()) and
                             summary["gold"] and summary["gold"]["passed"] and
                             hashlib.sha256((ROOT / "files.json").read_bytes()).hexdigest() == manifest_sha256)
    benchmark.write(receipt, summary)
    print(f"Interactive verification passed={summary['passed']}: {receipt}", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
