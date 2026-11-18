"""Offline constructor integration; no rendering, HTTP, runtime install or GPU."""
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "dram_kv"
RUNTIME = ROOT / "swebench_baseline"


def _package(name, path):
    # Isolate packages; bypass their broad __init__ exports. Imported modules
    # below remain the original upstream source, not copied/reimplemented logic.
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package


_package("_goal_integration_upstream", UPSTREAM)
_package("_goal_integration_upstream.goals", UPSTREAM / "goals")
_package("_goal_integration_runtime", RUNTIME)
MaintainedGoalConstructor = importlib.import_module(
    "_goal_integration_upstream.goals.maintained").MaintainedGoalConstructor
TurnContext = importlib.import_module("_goal_integration_upstream.schema").TurnContext
DeterministicFeatureExtractor = importlib.import_module(
    "_goal_integration_upstream.goals.features").DeterministicFeatureExtractor
_contract = importlib.import_module("_goal_integration_runtime.goal_contract_v13")
runtime_digest = _contract.digest
ProducerProvenance = _contract.ProducerProvenance
ManifestContextProvider = importlib.import_module(
    "_goal_integration_runtime.bootstrap").ManifestContextProvider


class CapturedTextFeatureExtractor(DeterministicFeatureExtractor):
    """Retain semantic evidence without trusting status strings in tool text.

    The capture-v2 input has no authenticated structured execution-status
    channel. Its stdout may quote arbitrary source, logs, or fixture strings.
    Upstream's default extractor infers return_code/test_verdict from that text;
    these two fields must be cleared BEFORE its state tracker and Goal gates run.
    This instance-local wrapper does not alter upstream modules or raw evidence.
    """

    def extract(self, action, observation, *, tool_name=None, return_code=None):
        require(return_code is None,
                "capture adapter has no trusted structured execution-status channel")
        features = super().extract(action, observation, tool_name=tool_name)
        return replace(features, return_code=None,
            test_verdict="unknown" if features.action_kind == "test" else "not_run")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def capture_digest(value, *, ordered=False):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
        sort_keys=not ordered, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _tokens(value, label):
    require(type(value) is list and value and
            all(type(token) is int and token >= 0 for token in value),
            label + " requires actual nonempty nonnegative token IDs")
    return value


def _request(replay, index):
    require(type(replay) is dict and replay.get("schema_version") == 2
            and type(replay.get("schema_version")) is int
            and replay.get("kind") == "frozen_agent_message_inputs"
            and replay.get("condition") == "full"
            and _sha(replay.get("source_inventory_sha256")), "Full v2 capture export required")
    calls = replay.get("calls")
    require(type(calls) is list and type(index) is int and 0 <= index < len(calls),
            "invalid captured call index")
    call = calls[index]
    require(type(call) is dict, "invalid captured call")
    request = call.get("request")
    require(type(request) is dict and request.get("extra_body") is None,
            "Full request required")
    require(call.get("request_sha256") == capture_digest(request)
            and call.get("request_order_sha256") == capture_digest(request, ordered=True),
            "captured request content/order digest mismatch")
    messages = request.get("messages")
    require(type(messages) is list and messages and all(type(m) is dict for m in messages),
            "captured messages required")
    require(all(m.get("role") in {"system", "user", "assistant", "tool"}
                and (m.get("content") is None or type(m.get("content")) is str)
                for m in messages), "only text messages supported")
    observations = [{"message_index": i, "message": m} for i, m in enumerate(messages)
                    if m.get("role") == "tool"]
    require(call.get("observation_messages") == observations,
            "captured observation export mismatch")
    return request


def _latest_context(messages):
    latest = messages[-1]
    if latest.get("role") == "user" and not any(m.get("role") == "tool" for m in messages):
        return None, "", "user", {}
    require(latest.get("role") == "tool" and type(latest.get("content")) is str
            and latest["content"].strip(), "latest message must be a nonempty tool observation")
    require(len(messages) >= 2 and messages[-2].get("role") == "assistant",
            "parallel/unpaired tool observations unsupported")
    calls = messages[-2].get("tool_calls")
    require(type(calls) is list and len(calls) == 1 and type(calls[0]) is dict,
            "exactly one preceding tool call required")
    call = calls[0]
    require(call.get("type") == "function" and type(call.get("id")) is str
            and call["id"] and call["id"] == latest.get("tool_call_id"),
            "tool observation/action ID mismatch")
    fn = call.get("function")
    require(type(fn) is dict and fn.get("name") == "bash"
            and type(fn.get("arguments")) is str, "only captured bash function arguments supported")
    args = read_json_text(fn["arguments"])
    require(type(args) is dict and set(args) == {"command"}
            and type(args["command"]) is str and args["command"].strip(),
            "bash arguments require exactly one nonempty command")
    # Raw tool text is evidence, not a trusted explicit_requirement or status.
    # No return code is guessed from arbitrary stdout/XML/text.
    return args["command"], latest["content"], "tool", {"tool_name": "bash"}


def construct_goal(replay, task, call_index, *, session_id):
    """Replay every captured decision through the actual maintained constructor.

    A fresh instance per invocation makes retry deterministic and prevents partial
    failure from corrupting another session. Constructor state is maintained
    across prefix calls here; runtime growing-history state is NOT implemented.
    """
    require(type(task) is dict and type(task.get("problem_statement")) is str
            and task["problem_statement"].strip()
            and task.get("instance_id") == replay.get("instance_id"),
            "matching explicit task.json issue/instance required")
    require(type(session_id) is str and bool(re.fullmatch(r"[A-Za-z0-9._-]{1,64}", session_id)),
            "invalid session ID")
    _request(replay, call_index)
    first = _request(replay, 0)
    require(any(m.get("role") == "user" and task["problem_statement"] in (m.get("content") or "")
                for m in first["messages"]), "task issue absent from initial user prompt")
    constructor = MaintainedGoalConstructor(rewriter=None)
    constructor.feature_extractor = CapturedTextFeatureExtractor()
    previous_messages, previous_record = None, None
    records, contexts = [], []
    for index in range(call_index + 1):
        request = _request(replay, index)
        messages = request["messages"]
        if previous_messages is not None:
            require(len(messages) == len(previous_messages) + 2
                    and capture_digest(messages[:len(previous_messages)], ordered=True)
                    == capture_digest(previous_messages, ordered=True),
                    "capture history must append one assistant/tool pair per call")
        action, observation, source, state = _latest_context(messages)
        context = TurnContext(session_id=session_id, instance_id=task["instance_id"],
            turn_id=index + 1, task_objective=task["problem_statement"],
            previous_goal_text=previous_record.text if previous_record else None,
            previous_action=action, current_observation=observation,
            message_source=source, structured_state=state,
            historical_turn_ids=tuple(range(1, index + 1)))
        record = constructor.construct(context)
        contexts.append(asdict(context))
        records.append(asdict(record))
        previous_messages, previous_record = messages, record
    # All semantic fields remain available; policy hints are encoded but never
    # automatically converted to the experiment's fixed page budget/ranking.
    payload = snapshot(previous_record.selector_payload)
    probe_text = "Retrieve historical evidence for this task Goal:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return snapshot(dict(schema_version=1, kind="maintained_goal_constructor_output",
        source_request_sha256=capture_digest(request),
        source_request_order_sha256=capture_digest(request, ordered=True),
        source_inventory_sha256=replay["source_inventory_sha256"],
        task_sha256=capture_digest(task), instance_id=task["instance_id"], call_index=call_index,
        session_id=session_id, goal_payload=payload, goal_hash=runtime_digest(payload),
        goal_probe_text=probe_text, goal_probe_text_sha256=hashlib.sha256(probe_text.encode()).hexdigest(),
        execution_status_policy="untrusted_text_status_disabled_v1",
        goal_construction_rule="MaintainedGoalConstructor(rewriter=None) with instance-local "
            "CapturedTextFeatureExtractor; chronological captured requests; "
            "immutable task.problem_statement; paired bash command and raw tool observation; "
            "text-derived return_code/test_verdict disabled before phase/gate updates; "
            "no trusted structured execution status available; "
            "selector_payload encoded as independent plain-text probe",
        adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contexts=contexts, goal_records=records,
        constructor_source_sha256={p.relative_to(UPSTREAM).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [UPSTREAM / "schema.py", *sorted((UPSTREAM / "goals").glob("*.py"))]},
        system_valid=False, task_accuracy_valid=False, live_multi_turn_supported=False))


def build_goal_candidate(goal, *, history_receipt, probe_receipt, provenance,
                         num_pages, selected_count, generation, projection_seed=42):
    """Build static startup-manifest candidate from external renderer attestations.

    Receipt hash/identity matching is not authentication. A production dispatcher
    must obtain these from its trusted installed-model renderer, never public HTTP.
    This function has no fallback tokenizer and cannot create missing token IDs.
    """
    require(type(goal) is dict and goal.get("kind") == "maintained_goal_constructor_output",
            "constructor output required")
    require(type(goal.get("session_id")) is str and bool(re.fullmatch(
        r"[A-Za-z0-9._-]{1,64}", goal["session_id"])), "invalid session ID")
    require(goal.get("goal_hash") == runtime_digest(goal["goal_payload"]), "Goal payload changed")
    expected_text = "Retrieve historical evidence for this task Goal:\n" + json.dumps(
        goal["goal_payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(goal.get("goal_probe_text") == expected_text and goal.get("goal_probe_text_sha256") ==
            hashlib.sha256(expected_text.encode()).hexdigest(), "Goal probe text changed")
    require(type(history_receipt) is dict and type(probe_receipt) is dict,
            "both history and independent Goal probe renderer receipts required")
    require(history_receipt.get("schema_version") == 2, "history renderer v2 receipt required")
    for key in ("source_request_sha256", "source_request_order_sha256"):
        require(history_receipt.get(key) == goal[key], "history receipt request binding mismatch")
    for key in ("model", "model_revision", "tokenizer_revision", "renderer_identity_sha256"):
        value = history_receipt.get(key)
        require(type(value) is str and value.strip() and value not in {"main", "master", "local"}
                and probe_receipt.get(key) == value, "history/probe renderer identity mismatch: " + key)
    require(_sha(history_receipt["renderer_identity_sha256"]), "renderer identity must be SHA256")
    require(probe_receipt.get("schema_version") == 1
            and probe_receipt.get("kind") == "plain_text_goal_probe_renderer_receipt"
            and probe_receipt.get("goal_hash") == goal["goal_hash"]
            and probe_receipt.get("text_sha256") == goal["goal_probe_text_sha256"]
            and probe_receipt.get("add_special_tokens") is False,
            "probe receipt must bind exact semantic Goal and plain text with no extra tokens")
    prompt = _tokens(history_receipt.get("rendered_prompt_token_ids"), "history receipt")
    probe = _tokens(probe_receipt.get("rendered_prompt_token_ids"), "probe receipt")
    require(probe != prompt, "Goal probe must differ from historical prompt")
    p = ProducerProvenance(**provenance)
    require((p.model, p.revision) == (history_receipt["model"], history_receipt["model_revision"]),
            "renderer/producer model identity mismatch")
    require(type(num_pages) is int and type(selected_count) is int
            and 0 < selected_count < num_pages
            and num_pages * p.page_tokens < len(prompt) < (num_pages + 1) * p.page_tokens,
            "frozen candidate requires complete history plus a partial suffix and selective budget")
    require(_sha(generation), "trusted administrator generation must be SHA256")
    lineage = dict(goal_hash=goal["goal_hash"], query_payload_hash=runtime_digest(
        dict(prompt_token_ids=probe, add_special_tokens=False)))
    entries = [dict(policy="goal_query", stage=stage, session_id=goal["session_id"],
        generation=generation, turn=turn, prompt_hash=runtime_digest(token_ids),
        num_pages=count, metadata=metadata) for stage, turn, token_ids, count, metadata in (
            ("capture", 1, prompt, num_pages, {}), ("query", 2, probe, 0, lineage),
            ("restore", 3, prompt, num_pages, dict(lineage, selected_count=selected_count)))]
    manifest = dict(version=1, provenance=asdict(p), projection_seed=projection_seed,
                    selected_count=selected_count, requests=entries)
    ManifestContextProvider(manifest, None)
    return snapshot(dict(schema_version=1, kind="frozen_goal_manifest_candidate",
        goal=goal, manifest=manifest,
        selector_goal_fields=dict(goal_payload=goal["goal_payload"],
            goal_construction_rule=goal["goal_construction_rule"], goal_probe_token_ids=probe,
            goal_probe_renderer_identity_sha256=history_receipt["renderer_identity_sha256"]),
        kv_transfer_params=dict(policy_runtime=dict(version=1, policy="goal_query")),
        stage_token_ids=dict(capture=prompt, query=probe, restore=prompt),
        history_receipt=history_receipt, probe_receipt=probe_receipt,
        token_binding="EXTERNAL_RENDERER_ATTESTATION_NOT_INDEPENDENTLY_VERIFIED",
        manifest_schema_validated=True, system_valid=False, task_accuracy_valid=False,
        live_multi_turn_supported=False, http_execution_implemented=False))


def read_json_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-inputs", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--call-index", type=int, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    result = construct_goal(read_json_text(args.replay_inputs.read_text(encoding="utf-8")),
        read_json_text(args.task.read_text(encoding="utf-8")), args.call_index, session_id=args.session_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
