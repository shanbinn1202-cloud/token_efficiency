"""Interactive GPU pilot controls and independent, pinned selection-receipt checks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clients"))
POLICIES = {"random": "random_fill", "sliding_window": "recent_fill"}
CONTROL_INDEX = 100  # Reserved for the synthetic observation, outside every pilot worklist.


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def gold(out):
    row = read(ROOT / "assets/dataset-first150.json")[0]
    iid = row["instance_id"]
    require(bool(row["patch"].strip()), "The official patch is missing")
    work = out / "reference-evaluation"
    work.mkdir()
    save(work / "metrics.json", {"expected": 1, "output_tokens_per_second": None,
                                 "e2e_seconds": None})
    save(work / "predictions.json", [{"instance_id": iid, "model_name_or_path": "official-reference-control",
                                      "model_patch": row["patch"]}])
    subprocess.run([sys.executable, "-B", str(ROOT / "clients/local_grade.py"), "--workers", "1",
                    "--output", str(work), "--manifest", str(ROOT / "local_images/manifest.json")], check=True)
    result = read(work / "result.json")
    require(result["complete"] and result["graded"] == 1 and result["resolved_ids"] == [iid]
            and result["accuracy_percent"] == 100 and not result["infrastructure_error_ids"],
            "Official reference patch did not resolve on the interactive node")
    return {"passed": True, "instance_id": iid, "graded": 1, "resolved": 1, "accuracy_percent": 100,
            "source": "SWE-bench Verified gold patch, independent of model output"}


def make_client(method, model_path, port, work, index):
    import yaml
    from jinja2 import Template, StrictUndefined
    from local_common import FORMAL, SRC, dataset
    from local_run import Model
    from minisweagent.utils.serialize import recursive_merge
    from transformers import AutoTokenizer

    rows, pins = dataset()
    row = rows[index]
    iid = row["instance_id"]
    config = recursive_merge(yaml.safe_load((FORMAL / "mini242/minisweagent/config/benchmarks/swebench_xml.yaml").read_text()),
                             yaml.safe_load((SRC / "swebench_prompt_testbed.yaml").read_text()))
    kwargs = dict(config["model"])
    kwargs.pop("model_class", None)
    kwargs.update(model_name="openai/selective-prefix", cost_tracking="ignore_errors", model_kwargs={})
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    work.mkdir()
    client = Model(method, f"http://127.0.0.1:{port}/v1", tokenizer, work, iid, **kwargs)
    context = {"task": row["problem_statement"], "image": pins[iid].split("@")[0] + ":latest",
               "step_limit": 120}
    messages = [{"role": role, "content": Template(config["agent"][key], undefined=StrictUndefined).render(**context)}
                for role, key in [("system", "system_template"), ("user", "instance_template")]]
    return client, iid, messages


def long_observation(client):
    # Keep stdout below Mini-SWE's 10,000-character elision boundary while crossing
    # the token budget. Its production observation formatter is used for every byte.
    line = "log: 龘齎𠀀𠮷𠮷\n"
    encode = lambda s: len(client.tokenizer.encode(s, add_special_tokens=False))
    copies = 1 + 9050 // max(1, encode(line))
    value = line * copies
    while encode(value) <= 9000 and len(value) < 9900:
        value += line * 4
    require(9000 < encode(value) < 11200 and len(value) < 10000, "Probe stdout missed the tokenizer bounds")
    return value


def feedback(client, previous, stdout, step):
    return client.format_observation_messages(previous, [{"returncode": 0, "output": stdout, "exception_info": ""}],
                                              {"step_limit": 120, "n_model_calls": step})


def compact(out, model_path, port):
    client, iid, messages = make_client("claude_compression", model_path, port, out / "compact-control", CONTROL_INDEX)
    assistant = {"role": "assistant", "reasoning_content": "Check the debug output before editing.",
                 "content": "THOUGHT: Inspect the captured file.\n<mswea_bash_command>cat /tmp/debug.log</mswea_bash_command>"}
    try:
        messages.append(assistant)
        messages.extend(feedback(client, assistant, long_observation(client), 1))
        require(client.count(messages) > 8192, "Claude probe does not cross the 8K trigger")
        # Use the actual policy and production native_compact(), preserving the two reasoning fields.
        next_input = client.compactor.prepare(messages)
        events = client.compactor.events
        require(len(events) == 1 and next_input[0] == messages[0] and len(next_input) >= 2,
                "The official summary was not applied to the continuation")
        event = events[0]
        require(event["command"] == "/compact" and event["version"] == "2.1.325" and event["before_tokens"] > 8192,
                "Unexpected Claude Code version, operation, or trigger")
        # Already verified a real summary. The short continuation tests main inference
        # without charging a full 4096-token reply to the separate control.
        response = client.post(next_input, max_tokens=32)
        aux = sum(c["usage"]["completion_tokens"] for c in client.calls if c["kind"] == "summary")
        main = response.usage.completion_tokens
        require(aux > 0 and main > 0 and bool(next_input[1]["content"].strip()),
                "No live Qwen summary or main continuation output")
        return {"passed": True, "instance_id": iid, "source": "synthetic long tool observation; live Qwen service",
                "version": event["version"], "command": event["command"], "before_tokens": event["before_tokens"],
                "after_tokens": event["after_tokens"], "auxiliary_completion_tokens": aux, "continued_main_tokens": main}
    finally:
        client.close()


def selection_evidence(out, method, ids):
    sys.path.insert(0, str(ROOT / "source/harnesses/perf-swe32"))
    import swe_checks as checks

    evidence = out / "server"
    ready = read(evidence / "ready.json")
    desired = POLICIES[method]
    contract = ready["selection_contract"]
    require(ready["policy"] == desired and ready["native_http_ready"] and contract["active_token_budget"] == 8192,
            "Unexpected live native server or selection contract")
    sessions = {entry["session_id"]: entry["instance_id"] for entry in ready["sessions"]}
    by_session = defaultdict(list)
    for row in checks.runtime_rows(evidence):
        iid = sessions.get(row["session_id"])
        if iid in ids:
            by_session[iid].append(row)
    require(set(by_session) == set(ids), "Missing runtime requests for some pilot sessions")
    reports = []
    for iid in ids:
        initial_end = None
        ends, spans = [], []
        for turn, row in enumerate(sorted(by_session[iid], key=lambda r: r["turn"])):
            path = checks.native_path(evidence, row["request_id"])
            require(checks.sc.sha_file(path) == row["native_evidence_sha256"], "Native receipt hash mismatch")
            native = read(path)
            require(row["turn"] == turn and native["request_id"] == row["request_id"]
                    and row["session_id"] == native["session_id"] and row["policy"] == desired,
                    "Native request ID, policy or turn binding mismatch")
            if not turn:
                initial_end = native["stable_end"]
            if native["local_turn"] == 0:
                ends, spans = [], []
            selection = checks.check_selection(desired, row, native, contract, initial_end, 16,
                                               history=(list(ends), list(spans)))
            ends.append(native["stable_end"])
            if native["local_turn"]:
                spans.append(native["observation_span"])
            reports.append({"instance_id": iid, "turn": turn, "request_id": row["request_id"], **selection})
    return {"passed": True, "method": method, "policy": desired, "active_token_budget": 8192,
            "runtime_requests": len(reports), "signed_native_receipts": len(reports),
            "active_selection_turns": sum(bool(x.get("active")) for x in reports),
            "eviction_turns": sum(x["selected"] < x["candidate_pages"] for x in reports), "turns": reports}


def native_live(out, method, model_path, port):
    work = out / "native-live-control"
    client, iid, messages = make_client(method, model_path, port, work, CONTROL_INDEX)
    returned = []
    try:
        for turn in range(3):
            response = client.post(messages)  # native frontend pins the production 4096 output bound
            returned.append(response.usage.completion_tokens)
            if turn < 2:
                reply = response.choices[0].message.model_dump(exclude_none=True)
                messages.append(reply)
                stdout = long_observation(client) if turn == 0 else "Probe diagnostic completed."
                messages.extend(feedback(client, reply, stdout, turn + 1))
    finally:
        client.close()
    require(all(t > 0 for t in returned), "The native service did not generate a reply on each of the three turns")
    evidence = selection_evidence(out, method, [iid])
    require(evidence["eviction_turns"] > 0 and evidence["active_selection_turns"] > 0,
            "The native selector did not exercise over-budget page removal")
    return {**evidence, "source": "synthetic >8K tool observation on separately registered task",
            "registered_index": CONTROL_INDEX, "completion_tokens_by_turn": returned}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="control", required=True)
    for name in ["gold", "native", "native-live", "compact"]:
        a = sub.add_parser(name)
        a.add_argument("--out", type=Path, required=True)
        if name.startswith("native"):
            a.add_argument("--method", choices=POLICIES, required=True)
        if name in {"native-live", "compact"}:
            a.add_argument("--model", type=Path, required=True)
            a.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.control == "gold":
        result = gold(out)
    elif args.control == "compact":
        result = compact(out, args.model, args.port)
    elif args.control == "native-live":
        result = native_live(out, args.method, args.model, args.port)
    else:
        spec = read(out / "evaluation/run-spec.json")
        ids = spec["instance_ids"]
        desired = read(ROOT / "assets/dataset-first150.json")[CONTROL_INDEX]["instance_id"]
        require(spec["method"] == args.method and desired not in ids, "Pilot and synthetic sessions overlap")
        result = selection_evidence(out, args.method, ids)
    result["wall_seconds"] = time.time() - started
    save(out / (args.control + "-check.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "turns"}), flush=True)


if __name__ == "__main__":
    main()
