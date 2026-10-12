"""Dependency-free adapters for recorded mini-SWE-agent trajectories.

Oracle annotations are read only to construct evaluation labels.  They never
contribute text, entities, or policy hints to an online Goal/selector input.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

from ..goals.features import DeterministicFeatureExtractor, FeatureSnapshot
from ..paging import TokenSegment, build_page_catalog


_PR_DESCRIPTION_RE = re.compile(
    r"<pr_description>\s*(?:Consider the following PR description:\s*)?"
    r"(.*?)\s*</pr_description>",
    re.DOTALL,
)


@dataclass(frozen=True)
class SweAgentBundle:
    trajectory: Mapping[str, Any]
    oracle: Mapping[str, Any]
    trajectory_sha256: str
    oracle_sha256: str


@dataclass(frozen=True)
class RecordedBoundary:
    """One recorded model-call boundary with explicit message provenance.

    A boundary is keyed by the assistant message that will be replayed.  Its
    input messages are every message added since the preceding assistant
    response.  This intentionally supports harness feedback messages in
    addition to the ordinary one-tool-result-per-action layout.
    """

    turn_id: int
    assistant_message_id: int
    input_message_ids: tuple[int, ...]
    result_message_ids: tuple[int, ...]
    tool_call_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "assistant_message_id": self.assistant_message_id,
            "input_message_ids": list(self.input_message_ids),
            "result_message_ids": list(self.result_message_ids),
            "tool_call_ids": list(self.tool_call_ids),
        }


def recorded_boundaries(trajectory: Mapping[str, Any]) -> tuple[RecordedBoundary, ...]:
    """Normalize model calls without assuming messages live at ``2 * turn``."""

    messages = _messages(trajectory)
    assistant_ids = [
        index for index, message in enumerate(messages)
        if str(message.get("role", "")) == "assistant"
    ]
    if not assistant_ids:
        raise ValueError("trajectory has no assistant messages")
    boundaries: list[RecordedBoundary] = []
    previous_assistant = -1
    for offset, assistant_id in enumerate(assistant_ids):
        next_assistant = (
            assistant_ids[offset + 1] if offset + 1 < len(assistant_ids) else len(messages)
        )
        assistant = messages[assistant_id]
        raw_calls = assistant.get("tool_calls")
        call_ids: list[str] = []
        if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)):
            for call in raw_calls:
                if isinstance(call, Mapping) and str(call.get("id", "")).strip():
                    call_ids.append(str(call["id"]))
        if not call_ids:
            extra = assistant.get("extra")
            actions = extra.get("actions") if isinstance(extra, Mapping) else None
            if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
                for action in actions:
                    if isinstance(action, Mapping) and str(action.get("tool_call_id", "")).strip():
                        call_ids.append(str(action["tool_call_id"]))
        input_start = 1 if offset == 0 else previous_assistant + 1
        input_ids = tuple(
            index for index in range(input_start, assistant_id)
            if str(messages[index].get("role", "")) in {"user", "tool"}
        )
        result_ids = tuple(
            index for index in range(assistant_id + 1, next_assistant)
            if str(messages[index].get("role", "")) in {"tool", "exit"}
        )
        boundaries.append(RecordedBoundary(
            turn_id=offset + 1,
            assistant_message_id=assistant_id,
            input_message_ids=input_ids,
            result_message_ids=result_ids,
            tool_call_ids=tuple(dict.fromkeys(call_ids)),
        ))
        previous_assistant = assistant_id
    return tuple(boundaries)


def recorded_message_map(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical message map covered by the Oracle binding."""

    messages = _messages(trajectory)
    terminal = [
        index for index, message in enumerate(messages)
        if str(message.get("role", "")) == "exit"
    ]
    return {
        "terminal_exit_message_id": terminal[0] if len(terminal) == 1 else None,
        "boundaries": [
            boundary.to_dict() for boundary in recorded_boundaries(trajectory)
        ],
    }


def message_map_sha256(trajectory: Mapping[str, Any]) -> str:
    """Hash the exact turn/message interpretation used by replay and labels."""

    stable = json.dumps(
        recorded_message_map(trajectory),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(stable.encode("utf-8")).hexdigest()


def _oracle_selected_turns(entry: Mapping[str, Any]) -> tuple[int, ...]:
    retrieval = entry.get("retrieval")
    raw = (
        retrieval.get("selected_older_turn_ids", ())
        if isinstance(retrieval, Mapping)
        else entry.get("selected_older_turn_ids", ())
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("selected_older_turn_ids must be an array")
    return tuple(int(item) for item in raw)


def validate_oracle_binding(
    trajectory: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    trajectory_sha256: str,
) -> None:
    """Fail closed when an Oracle was authored for another exact trajectory."""

    instance_id = str(trajectory.get("instance_id", ""))
    annotated = oracle.get("trajectory")
    if not isinstance(annotated, Mapping):
        raise ValueError("Oracle lacks trajectory binding")
    if not instance_id or instance_id != str(annotated.get("instance_id", "")):
        raise ValueError("trajectory/oracle instance IDs differ")
    declared_hash = str(annotated.get("sha256", ""))
    if declared_hash and declared_hash != trajectory_sha256:
        raise ValueError("Oracle trajectory SHA-256 differs from loaded trajectory")
    strict = oracle.get("schema_name") == "dram_kv_recorded_oracle"
    if strict and not declared_hash:
        raise ValueError("recorded Oracle requires an exact trajectory SHA-256")
    declared_map_hash = str(annotated.get("message_map_sha256", ""))
    actual_map_hash = message_map_sha256(trajectory)
    if declared_map_hash and declared_map_hash != actual_map_hash:
        raise ValueError("Oracle message-map SHA-256 differs from loaded trajectory")
    if strict and not declared_map_hash:
        raise ValueError("recorded Oracle requires an exact message-map SHA-256")
    if strict:
        declared_map = oracle.get("message_map")
        if not isinstance(declared_map, Mapping):
            raise ValueError("recorded Oracle requires its canonical message map")
        actual_map = recorded_message_map(trajectory)
        # Some unit fixtures have no terminal exit and omit the corresponding
        # null field.  Canonicalize that benign representation before compare.
        comparable = dict(declared_map)
        comparable.setdefault("terminal_exit_message_id", None)
        if comparable != actual_map:
            raise ValueError("Oracle message map differs from loaded trajectory")

    boundaries = recorded_boundaries(trajectory)
    by_turn = {boundary.turn_id: boundary for boundary in boundaries}
    entries = oracle.get("turns")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Oracle requires non-empty turns")
    policy = oracle.get("retrieval_policy")
    recent_window = (
        int(policy.get("recent_complete_turn_window", 0))
        if isinstance(policy, Mapping) else 0
    )
    if recent_window < 0:
        raise ValueError("recent_complete_turn_window cannot be negative")
    seen: set[int] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Oracle turn must be an object")
        turn_id = int(entry["turn_id"])
        if turn_id in seen or turn_id not in by_turn:
            raise ValueError(f"Oracle contains duplicate or unknown turn {turn_id}")
        seen.add(turn_id)
        boundary = by_turn[turn_id]
        if "assistant_message_id" in entry and int(entry["assistant_message_id"]) != boundary.assistant_message_id:
            raise ValueError(f"Oracle assistant message mismatch at turn {turn_id}")
        if "input_message_ids" in entry:
            actual = tuple(int(item) for item in entry["input_message_ids"])
            if actual != boundary.input_message_ids:
                raise ValueError(f"Oracle input message mismatch at turn {turn_id}")
        selected = _oracle_selected_turns(entry)
        if len(set(selected)) != len(selected):
            raise ValueError(f"Oracle repeats evidence turns at turn {turn_id}")
        if any(source <= 0 or source >= turn_id for source in selected):
            raise ValueError(f"Oracle evidence must be older than turn {turn_id}")
        if any(source >= turn_id - recent_window for source in selected):
            raise ValueError(
                f"Oracle selected_older evidence overlaps the recent window at turn {turn_id}"
            )


def load_swe_agent_bundle(archive: Path) -> SweAgentBundle:
    """Load and checksum the single trajectory/oracle pair in ``archive``."""

    with ZipFile(archive) as handle:
        names = handle.namelist()
        trajectories = [name for name in names if name.endswith(".traj.json")]
        oracles = [name for name in names if name.endswith(".oracle.json")]
        if len(trajectories) != 1 or len(oracles) != 1:
            raise ValueError("archive must contain exactly one trajectory and one oracle")
        trajectory_bytes = handle.read(trajectories[0])
        oracle_bytes = handle.read(oracles[0])
        checksum_names = [name for name in names if name.endswith("SHA256SUMS")]
        declared: dict[str, str] = {}
        if len(checksum_names) == 1:
            for line in handle.read(checksum_names[0]).decode("utf-8").splitlines():
                digest, _, filename = line.partition("  ")
                if digest and filename:
                    declared[filename.strip()] = digest.strip()
    trajectory = json.loads(trajectory_bytes)
    oracle = json.loads(oracle_bytes)
    if not isinstance(trajectory, Mapping) or not isinstance(oracle, Mapping):
        raise ValueError("trajectory and oracle must be JSON objects")
    trajectory_hash = sha256(trajectory_bytes).hexdigest()
    oracle_hash = sha256(oracle_bytes).hexdigest()
    for name, digest in (
        (Path(trajectories[0]).name, trajectory_hash),
        (Path(oracles[0]).name, oracle_hash),
    ):
        if name in declared and declared[name] != digest:
            raise ValueError(f"archive checksum mismatch for {name}")
    validate_oracle_binding(
        trajectory,
        oracle,
        trajectory_sha256=trajectory_hash,
    )
    return SweAgentBundle(
        trajectory,
        oracle,
        trajectory_hash,
        oracle_hash,
    )


def root_objective(trajectory: Mapping[str, Any]) -> str:
    messages = _messages(trajectory)
    text = str(messages[1].get("content", ""))
    match = _PR_DESCRIPTION_RE.search(text)
    return (match.group(1) if match else text).strip()


def assistant_action(message: Mapping[str, Any]) -> str:
    """Prefer executable actions over assistant chain-of-thought prose."""

    extra = message.get("extra")
    if isinstance(extra, Mapping):
        actions = extra.get("actions")
        if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
            commands = [
                str(item.get("command", "")).strip()
                for item in actions
                if isinstance(item, Mapping) and str(item.get("command", "")).strip()
            ]
            if commands:
                return "\n".join(commands)
    return str(message.get("content", ""))


def tool_observation(message: Mapping[str, Any]) -> tuple[str, int | None]:
    extra = message.get("extra")
    if isinstance(extra, Mapping):
        raw = extra.get("raw_output")
        code = extra.get("returncode")
        if raw is not None:
            return str(raw), code if isinstance(code, int) and not isinstance(code, bool) else None
    return str(message.get("content", "")), None


def completed_turn_features(
    trajectory: Mapping[str, Any], turn_id: int,
    *, extractor: DeterministicFeatureExtractor | None = None,
) -> FeatureSnapshot:
    messages = _messages(trajectory)
    by_turn = {boundary.turn_id: boundary for boundary in recorded_boundaries(trajectory)}
    if turn_id not in by_turn:
        raise KeyError(f"unknown recorded turn {turn_id}")
    boundary = by_turn[turn_id]
    action = assistant_action(messages[boundary.assistant_message_id])
    observation, return_code = _messages_observation(messages, boundary.result_message_ids)
    return (extractor or DeterministicFeatureExtractor()).extract(
        action, observation, tool_name="shell", return_code=return_code
    )


def _messages_observation(
    messages: Sequence[Mapping[str, Any]], message_ids: Sequence[int]
) -> tuple[str, int | None]:
    parts: list[str] = []
    return_code: int | None = None
    for message_id in message_ids:
        if message_id < 0 or message_id >= len(messages):
            raise ValueError(f"message ID lies outside trajectory: {message_id}")
        text, code = tool_observation(messages[message_id])
        if text.strip():
            parts.append(text)
        if code is not None:
            return_code = code
    return "\n\n".join(parts), return_code


def build_semantic_shadow_payload(
    bundle: SweAgentBundle,
    *, page_budget: int = 8,
    recent_turns: int = 2,
    recent_quota: int = 2,
) -> dict[str, Any]:
    """Build a real-message, semantic-Page Goal Shadow replay.

    One completed SWE-agent turn is one Page.  This deliberately evaluates the
    control plane at the Oracle's semantic selection unit; it is not presented
    as token-level KV or latency measurement.
    """

    trajectory, oracle = bundle.trajectory, bundle.oracle
    messages = _messages(trajectory)
    entries = oracle.get("turns")
    if oracle.get("schema_name") not in {
        "kv_drift_turn_oracle", "dram_kv_recorded_oracle",
    } or not isinstance(entries, list):
        raise ValueError("unsupported or incomplete Oracle annotation")
    validate_oracle_binding(
        trajectory,
        oracle,
        trajectory_sha256=bundle.trajectory_sha256,
    )
    instance_id = str(trajectory["instance_id"])
    session_id = f"recorded-{instance_id}"
    page_size = 1
    extractor = DeterministicFeatureExtractor()
    segments: list[dict[str, Any]] = [{
        "turn_id": 0,
        "role": "task_anchor",
        "token_start": 0,
        "token_end": 1,
        "mandatory": True,
        "summary": root_objective(trajectory),
    }]
    boundaries = recorded_boundaries(trajectory)
    boundaries_by_turn = {boundary.turn_id: boundary for boundary in boundaries}
    for entry in entries:
        turn_id = int(entry["turn_id"])
        if turn_id not in boundaries_by_turn:
            raise ValueError(f"Oracle references unknown turn {turn_id}")
        boundary = boundaries_by_turn[turn_id]
        features = completed_turn_features(trajectory, turn_id, extractor=extractor)
        action = assistant_action(messages[boundary.assistant_message_id])
        observation, _ = _messages_observation(messages, boundary.result_message_ids)
        summary = " | ".join(filter(None, (
            features.action.excerpt,
            features.observation.excerpt,
        )))[:2400]
        segments.append({
            "turn_id": turn_id,
            "role": "completed_turn",
            "token_start": turn_id,
            "token_end": turn_id + 1,
            "files": list(features.files),
            "symbols": list(features.symbols),
            "tests": list(features.tests),
            "errors": list(features.errors),
            "summary": summary or (action + "\n" + observation)[:2400],
        })

    catalog = build_page_catalog(
        session_id,
        tuple(TokenSegment(**segment) for segment in segments),
        page_size=page_size,
        bytes_per_token=0,
    )
    pages_by_turn = {page.turn_id: page.page_id for page in catalog.pages}
    objective = root_objective(trajectory)
    turns: list[dict[str, Any]] = []
    evidence: dict[str, list[str]] = {}
    for entry in entries:
        target = int(entry["turn_id"])
        boundary = boundaries_by_turn[target]
        previous = boundaries_by_turn.get(target - 1)
        previous_action = (
            assistant_action(messages[previous.assistant_message_id])
            if previous is not None else None
        )
        if target > 1:
            observation, return_code = _messages_observation(
                messages, boundary.input_message_ids
            )
            source = str(messages[boundary.input_message_ids[-1]].get("role", "tool"))
        else:
            observation, return_code, source = "", None, "user"
        history = catalog.historical_for_turn(target)
        turns.append({
            "session_id": session_id,
            "instance_id": instance_id,
            "turn_id": target,
            "task_objective": objective,
            "previous_goal_text": None,
            "previous_action": previous_action,
            "current_observation": observation,
            "structured_state": {"return_code": return_code, "tool_name": "shell"},
            "historical_turn_ids": sorted({page.turn_id for page in history}),
            "historical_page_ids": [page.page_id for page in history],
            "message_source": source,
        })
        evidence[str(target)] = [
            pages_by_turn[turn] for turn in _oracle_selected_turns(entry)
        ]
    return {
        "schema_name": "dram_kv_goal_shadow_replay",
        "schema_version": "1.1.0",
        "source": {
            "kind": "recorded_swe_agent_semantic_pages",
            "trajectory_sha256": bundle.trajectory_sha256,
            "oracle_sha256": bundle.oracle_sha256,
            "oracle_usage": "evaluation_labels_only",
            "token_level": False,
        },
        "session_id": session_id,
        "page_size": page_size,
        "bytes_per_token": 0,
        "selector": {
            "page_budget": page_budget,
            "recent_turns": recent_turns,
            "recent_quota": recent_quota,
        },
        "segments": segments,
        "turns": turns,
        "evidence_page_ids": evidence,
    }


def _messages(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError("trajectory lacks recorded messages")
    if any(not isinstance(message, Mapping) for message in messages):
        raise ValueError("trajectory messages must be objects")
    return messages


__all__ = [
    "RecordedBoundary",
    "SweAgentBundle",
    "assistant_action",
    "build_semantic_shadow_payload",
    "completed_turn_features",
    "load_swe_agent_bundle",
    "message_map_sha256",
    "recorded_boundaries",
    "recorded_message_map",
    "root_objective",
    "tool_observation",
    "validate_oracle_binding",
]
