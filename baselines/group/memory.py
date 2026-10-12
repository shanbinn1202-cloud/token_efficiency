"""Scope policies, entity resolution and history selection."""

import copy
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from baselines.common.history import split_turns


PATH = re.compile(r"(?<![\w/])(?:\./|/testbed/)?([\w.-]+(?:/[\w.-]+)+\.(?:py|pyx|pxd|c|h|cpp|js|ts|rs|go))\b")
PHASES = {
    "verify": {"verify", "verification", "test", "tests", "testing", "validate", "validation", "submit", "submission", "cleanup"},
    "implement": {"implement", "implementation", "edit", "patch", "fix", "modify"},
    "reproduce": {"reproduce", "reproduction", "repro"},
    "diagnose": {"diagnose", "diagnosis", "explore", "investigate", "understand", "locate", "read", "search", "inspect", "analyze"},
}


def component(path):
    parts = list(PurePosixPath(path).parts)
    # A component's source and tests belong together. Ignore scratch scripts
    # without a package path; changing the script name is not a new objective.
    parts = [p for p in parts if p not in {"tests", "test", ".", "testbed", "/"}]
    return "/".join(parts[:-1]) if len(parts) > 1 else None


def evidence(turn, proposed):
    commands = []
    for message in turn:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"])
            except (KeyError, TypeError, ValueError):
                continue
            commands.append(str(args.get("command") or ""))
    entities = {component(p) for p in PATH.findall("\n".join(commands))} - {None}
    # Phase comes from the agent's objective, not the shell verb: reading a
    # file during verification does not send the task back to diagnosis.
    words = re.split(r"[^a-z]+", proposed["scope"].lower())
    phase = next((phase for word in words for phase, names in PHASES.items() if word in names), None)
    return phase, entities


class EntityScopeResolver:
    def __init__(self, symbols=False):
        self.use_symbols = symbols
        self.records = {}
        self.previous = None
        self.last_trace = {}

    def resolve(self, turn, proposed):
        phase, entities = evidence(turn, proposed)
        symbols = set()
        if self.use_symbols:
            # Root package files are weak evidence of a new component. A
            # package initializer must not split its own subpackage's work.
            entities = {entity for entity in entities if "/" in entity}
            label = proposed["scope"]
            if re.search(r"(?:create|prepare|review|inspect|final).*patch", label):
                phase = "verify"
            # Metadata names the action's focus. Do not extract every symbol
            # from a reproduction script or bulk source output.
            text = label + " " + proposed.get("event", "")
            symbols = set(re.findall(r"\b_*[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b|\b[A-Z][a-z]+(?:[A-Z][a-zA-Z0-9]*)+\b", text))
            symbols = {symbol.lower() for symbol in symbols}
            methods = {symbol for symbol in symbols if "_" in symbol}
            if methods:
                symbols = methods
        previous = self.records.get(self.previous)
        phase = phase or (previous["phase"] if previous else "diagnose")
        candidates = [key for key, record in self.records.items() if record["phase"] == phase]
        def related(left, right):
            return bool(left & right) or bool(self.use_symbols and any(
                a.startswith(b + "/") or b.startswith(a + "/") for a in left for b in right
            ))
        def compatible_symbols(record):
            return not symbols or not record["symbols"] or bool(symbols & record["symbols"])
        matching = [key for key in candidates if compatible_symbols(self.records[key]) and (
            related(entities, self.records[key]["entities"])
            or (self.use_symbols and symbols & self.records[key]["symbols"] and not entities)
        )]
        if self.previous in matching:
            key, reason = self.previous, "same_phase_shared_component"
        elif matching:
            key = max(matching, key=lambda k: len(entities & self.records[k]["entities"]))
            reason = "return_to_component"
        elif previous and previous["phase"] == phase and compatible_symbols(previous) and (not entities or not previous["entities"]):
            key, reason = self.previous, "no_disjoint_entity_evidence"
        elif not entities and not symbols and candidates:
            # Generic test/review commands often omit paths; use the last
            # episode of that phase instead of inventing one per command.
            key, reason = candidates[-1], "return_to_phase_without_new_entity"
        else:
            target = sorted(entities)[0] if entities else proposed["scope"]
            if symbols:
                target += ":" + sorted(symbols)[0]
            key = f"{phase}-{target}"
            reason = "phase_or_disjoint_component"
            self.records[key] = {"phase": phase, "entities": set(), "symbols": set()}
        self.records[key]["entities"].update(entities)
        self.records[key]["symbols"].update(symbols)
        self.previous = key
        self.last_trace = {
            "proposed_scope": proposed["scope"],
            "resolved_scope": key,
            "phase": phase,
            "entities": sorted(entities),
            "symbols": sorted(symbols),
            "boundary_reason": reason,
        }
        return key


SCOPE = re.compile(r"[^a-z0-9._/-]+")

SCOPE_TOOL_INSTRUCTION = """
Every bash call must include scope, event, and scope_changed.

Scope is a stable decision episode and must use one of these forms:
- diagnose-<specific-bug>: search, read, reproduce, and establish one root cause.
- implement-<specific-decision>: edit and inspect the chosen fix.
- verify-<specific-decision>: run focused/broad tests, review the diff, and submit.

Keep the exact previous scope across commands and files in one episode. A different
command, file, test, or retry is not a change. Reuse an exact earlier scope when
returning to it. Change scope only for a phase transition, an independent required
change, or a new failure that invalidates the current decision. scope_changed is
true on the first call and otherwise exactly when scope differs from the preceding
call. event is a short factual statement of what this action is doing.
""".strip()

SCOPE_TOOL_V3_INSTRUCTION = """
Every bash call must include scope, event, and scope_changed.

Scope is the current decision episode, not the whole issue:
- diagnose-<specific-hypothesis>: gather evidence for one concrete cause or mechanism.
  Keep it while reading or testing that hypothesis. If evidence rejects it and the
  next action tests a different mechanism, start a new diagnose scope.
- implement-<specific-decision>: edit and inspect one chosen source change.
- verify-<specific-risk>: keep one scope for one focused behavior, then change when
  moving to a different regression risk, the broad suite, or final patch review.

Do not change scope merely for another command/file/retry supporting the same
hypothesis, decision, or risk. Reuse an exact earlier label when returning to it.
scope_changed is true on the first call and otherwise exactly when scope differs
from the preceding call. event briefly states this action's factual purpose.
""".strip()

SCOPE_TOOL_V4_INSTRUCTION = (
    SCOPE_TOOL_V3_INSTRUCTION
    + """

Before final submission, audit semantic consistency end to end: if a recognizer,
parser, normalization rule, or accepted input was broadened, inspect every
downstream consumer/comparison of that value and apply the same semantics where
needed. Run the issue example plus a focused regression test before submitting.
""".rstrip()
)

SCOPE_TOOL_V6_INSTRUCTION = """
Every bash call must include scope, event, and scope_changed.

A scope names the stable question or decision that the action advances:
- diagnose-<specific-hypothesis>: keep the exact label while gathering evidence
  that could support or reject one concrete cause. Start a new diagnose scope only
  after evidence rejects/resolves it and reasoning moves to a different mechanism.
- implement-<specific-decision>: keep the exact label through all edits and source
  inspection for one chosen fix.
- verify-<specific-fix>: keep the exact label through focused tests, regression
  tests, diff review, cleanup, and submission of that fix.

Default to continuity: scope_changed should usually be false and scope must then
copy the preceding label exactly. A new command, file, search, repro script, test,
retry, cleanup, or action verb is not a scope change. A normal coding task usually
has 3-8 scopes, not one per turn. Reuse an exact earlier label when returning.
scope_changed is true on the first call and otherwise exactly when the underlying
hypothesis or chosen decision changes. event is a short factual action purpose.

Before final submission, audit semantic consistency end to end. If accepted input,
matching, parsing, or normalization was broadened, inspect downstream consumers and
comparisons for the same semantics. Run the issue example and focused regression.
""".strip()

SCOPE_TOOL_V7_INSTRUCTION = SCOPE_TOOL_V6_INSTRUCTION.split(
    "\n\nBefore final submission", 1
)[0]
SCOPE_TOOL_V17_INSTRUCTION = (
    SCOPE_TOOL_V7_INSTRUCTION
    + """

event must describe the new evidence or state change expected from this action.
Before issuing a command, check the visible calls in the current scope. Do not
repeat an identical command after it produced the same result. Act on that result
with a different evidence source or implementation step; if it exposes a distinct
unresolved question, decide whether to change scope yourself.
""".rstrip()
)
SCOPE_VERIFY_AUDIT_INSTRUCTION = """
Before final submission, audit semantic consistency end to end. If accepted input,
matching, parsing, or normalization was broadened, inspect downstream consumers and
comparisons for the same semantics. Run the issue example and focused regression.
""".strip()
SCOPE_VERIFY_AUDIT_V12_INSTRUCTION = """
Before final submission, audit the intended semantics across the full boundary,
not only the reported example. If a format or protocol is described as
case-insensitive or normalized, determine whether that property applies to the
whole parser and every symbolic token class rather than one matched branch. Test
round trips and representative variants, then run the focused regression module.
Follow each newly accepted representation through post-match conversion and
dispatch. Search downstream for exact literal comparisons, case-sensitive keys,
or normalization assumptions; recognizer acceptance alone is not a complete fix.
""".strip()
SCOPE_STAGNATION_INSTRUCTION = """
The same command has now produced the same result at least three times in the
recent history. Do not repeat it again. Act on the evidence with a different
implementation or diagnostic step; decide yourself whether this changes scope.
""".strip()
SCOPE_TOOL_V19_INSTRUCTION = (
    SCOPE_TOOL_V7_INSTRUCTION
    + """

These metadata fields are bookkeeping and must not add investigation steps.
Keep reasoning concise and always finish the response with one bash call before
the output limit. Once evidence identifies a fix, implement it; once focused tests
and the diff are sufficient, submit. You decide both transitions and sufficiency.
When scope and scope_changed could conflict, scope_changed expresses your decision:
false means copy the preceding scope exactly; true means use the new scope.
""".rstrip()
)
SCOPE_OUTPUT_LIMIT_INSTRUCTION = """
Your immediately preceding response exhausted its output budget before the tool
call. For this retry, skip recap and extended analysis and emit the next concrete
bash call immediately, with scope, event, and scope_changed chosen by you.
""".strip()

INLINE_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute one shell command and label its stable coding scope.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "scope": {
                    "type": "string",
                    "description": (
                        "Stable question/decision; usually copy the exact previous label."
                    ),
                },
                "event": {
                    "type": "string",
                    "description": "Short factual purpose or result of this action.",
                },
                "scope_changed": {
                    "type": "boolean",
                    "description": (
                        "Usually false; true only when the underlying hypothesis/decision changed."
                    ),
                },
            },
            "required": ["command", "scope", "event", "scope_changed"],
            "additionalProperties": False,
        },
    },
}
INLINE_BASH_TOOL_STRICT = copy.deepcopy(INLINE_BASH_TOOL)
INLINE_BASH_TOOL_STRICT["function"]["strict"] = True
INLINE_BASH_TOOL_EXACT_CHANGE = copy.deepcopy(INLINE_BASH_TOOL_STRICT)
INLINE_BASH_TOOL_EXACT_CHANGE["function"]["parameters"]["properties"]["scope_changed"]["description"] = (
    "After choosing scope: true on the first call or if this scope label differs "
    "from the preceding call; false if the label is unchanged."
)


@dataclass(frozen=True)
class InlineScopePolicy:
    max_scope_chars: int = 80
    max_event_chars: int = 200
    recent: int = 0
    change_authoritative: bool = False
    max_active_turns: int | None = None
    entity_boundaries: bool = False
    symbol_boundaries: bool = False


INLINE_SCOPE_POLICIES = {
    "scope_tool_v1": InlineScopePolicy(),
    "scope_tool_v2": InlineScopePolicy(),
    "scope_tool_v3": InlineScopePolicy(recent=3),
    "scope_tool_v4": InlineScopePolicy(recent=3),
    "scope_tool_v6": InlineScopePolicy(recent=3),
    "scope_tool_v7": InlineScopePolicy(recent=3),
    "scope_tool_v8": InlineScopePolicy(recent=3),
    # V9 follows the scope retrieval rule exactly: all turns in the agent's
    # current scope plus the latest turn from every other scope.
    "scope_tool_v9": InlineScopePolicy(),
    "scope_tool_v10": InlineScopePolicy(),
    "scope_tool_v11": InlineScopePolicy(recent=3),
    "scope_tool_v12": InlineScopePolicy(recent=3),
    "scope_tool_v13": InlineScopePolicy(recent=3),
    "scope_tool_v14": InlineScopePolicy(recent=3),
    "scope_tool_v15": InlineScopePolicy(recent=3),
    "scope_tool_v16": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v17": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v18": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v19": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v20": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v26": InlineScopePolicy(recent=3, entity_boundaries=True, symbol_boundaries=True),
    "scope_tool_v25": InlineScopePolicy(recent=3, entity_boundaries=True, symbol_boundaries=True),
    "scope_tool_v24": InlineScopePolicy(recent=3, entity_boundaries=True),
    "scope_tool_v32": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v31": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v30": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v29": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v28": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v27": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v21": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v22": InlineScopePolicy(recent=3, change_authoritative=True),
    "scope_tool_v23": InlineScopePolicy(recent=3, change_authoritative=True, max_active_turns=8),
}


def repeated_action_result(messages, repeats=3, recent=8):
    """Detect an exact repeated command/result pair without changing its scope."""
    _, turns = split_turns(messages)
    pairs = []
    for turn in turns[-recent:]:
        assistant = next((m for m in turn if m.get("role") == "assistant"), {})
        tool = next((m for m in reversed(turn) if m.get("role") == "tool"), {})
        calls = assistant.get("tool_calls") or []
        if not calls or not tool:
            continue
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            continue
        command = arguments.get("command")
        if isinstance(command, str):
            pairs.append((command, str(tool.get("content") or "")))
    return bool(pairs and pairs.count(pairs[-1]) >= repeats)


def canonical_scope(value, max_chars=80) -> str:
    scope = SCOPE.sub("-", str(value or "").strip().lower()).strip("-")
    return scope[:max_chars]


def parse_scope_label(value, policy: InlineScopePolicy):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    scope = canonical_scope(value.get("scope"), policy.max_scope_chars)
    event = re.sub(r"\s+", " ", str(value.get("event") or "")).strip()
    changed = value.get("scope_changed")
    if not scope or not event or not isinstance(changed, bool):
        return None
    return {
        "scope": scope,
        "event": event[: policy.max_event_chars],
        "scope_changed": changed,
    }


def parse_turn_context(turn, policy):
    assistant = next(
        (message for message in turn if message.get("role") == "assistant"), {}
    )
    for call in assistant.get("tool_calls") or []:
        arguments = (call.get("function") or {}).get("arguments")
        parsed = parse_scope_label(arguments, policy)
        if parsed is not None:
            return parsed
    return None


def strip_scope_metadata(messages):
    """Remove index metadata from an API-input copy, preserving stored messages."""
    cleaned = copy.deepcopy(messages)
    for message in cleaned:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "")
            except (TypeError, ValueError):
                continue
            for key in ("scope", "event", "scope_changed"):
                arguments.pop(key, None)
            function["arguments"] = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            )
    return cleaned


class InlineScopeMemory:
    def __init__(self, policy):
        self.policy = policy
        self.scopes = []
        self.events = []
        self.declared_changes = []
        self.valid = []
        self.goal_changes = 1
        self.entity_resolver = EntityScopeResolver(symbols=policy.symbol_boundaries) if policy.entity_boundaries else None

    def scope_endpoints(self):
        return {scope: i for i, scope in enumerate(self.scopes)}

    def _add(self, turn):
        previous = self.scopes[-1] if self.scopes else ""
        parsed = parse_turn_context(turn, self.policy)
        if parsed is None:
            scope, event, changed, valid = (
                previous or "task",
                "scope metadata unavailable",
                False,
                False,
            )
        else:
            changed = parsed["scope_changed"]
            # The agent's scope label is the retrieval decision.  Keep
            # scope_changed as telemetry; never use it to override the label.
            scope = (
                parsed["scope"]
                if not self.policy.change_authoritative or changed or not previous
                else previous
            )
            if self.entity_resolver is not None:
                scope = self.entity_resolver.resolve(turn, parsed)
            event, valid = parsed["event"], True
        transitioned = bool(previous and scope != previous)
        if transitioned:
            self.goal_changes += 1
        self.scopes.append(scope)
        self.events.append(event)
        self.declared_changes.append(changed)
        self.valid.append(valid)

    def select(self, messages, _unused=None):
        prefix, turns = split_turns(messages)
        for turn in turns[len(self.scopes) :]:
            self._add(turn)
        if not turns:
            return messages, {
                "available": 0,
                "kept": [],
                "active_scope": None,
                "scope_endpoints": {},
                "valid_annotations": 0,
            }
        active = self.scopes[-1]
        endpoints = self.scope_endpoints()
        active_indices = [i for i, scope in enumerate(self.scopes) if scope == active]
        if self.policy.max_active_turns is not None:
            active_indices = active_indices[-self.policy.max_active_turns:]
        kept = sorted(
            set(active_indices)
            | set(endpoints.values())
            | set(range(max(0, len(turns) - self.policy.recent), len(turns)))
        )
        return prefix + [message for i in kept for message in turns[i]], {
            "available": len(turns),
            "kept": kept,
            "active_scope": active,
            "scope_endpoints": dict(sorted(endpoints.items())),
            "scope_count": len(endpoints),
            "active_kept": active_indices,
            "entity_boundary": dict(self.entity_resolver.last_trace) if self.entity_resolver else None,
            "scope_changed": self.declared_changes[-1],
            "scope_transitioned": bool(
                len(self.scopes) == 1 or self.scopes[-1] != self.scopes[-2]
            ),
            "scope_change_consistent": bool(
                self.valid[-1]
                and self.declared_changes[-1]
                == (len(self.scopes) == 1 or self.scopes[-1] != self.scopes[-2])
            ),
            "event": self.events[-1],
            "annotation_valid": self.valid[-1],
            "valid_annotations": sum(self.valid),
        }


SCOPE_TOOL_V22_INSTRUCTION = """
Every bash call includes scope, event, and scope_changed. Scope groups evidence for
one concrete question or decision, not an entire issue or an individual command.
Use diagnose-<hypothesis>, implement-<decision>, or verify-<behavior>.
Keep the exact label across commands/files answering that same question. Once the
question is answered, move to the next decision episode: reproducing a symptom,
locating its cause, implementing the chosen fix, and verifying it are distinct.
A rejected hypothesis and a newly discovered independent failure start new scopes.
Do not split for a different command, file, retry, or wording alone. Reuse an earlier
label only when its original question becomes relevant again.
event states the latest established finding/decision and this action's purpose;
include concrete identifiers or outcomes needed after leaving this scope.
scope_changed is true on the first call and whenever scope differs from the prior
call; otherwise copy the previous scope exactly and use false.
Your next input retains all turns in the current scope, the last turn of each other
scope, and the latest three turns. Conclude an episode with a useful decision in
your normal reasoning; do not run additional commands just to create a boundary.
""".strip()


ACTIVITIES = {'verify', 'verification', 'test', 'tests', 'testing', 'existing', 'regression', 'regressions', 'edge', 'cases', 'case', 'fix', 'final', 'diff', 'review', 'cleanup', 'clean', 'up', 'files', 'file', 'create', 'creating', 'prepare', 'preparing', 'patch', 'patches', 'submit', 'submitting', 'submission', 'output', 'content', 'contents', 'check'}
NOISE = ACTIVITIES | {'diagnose','reproduce','reproduction','issue','bug','works'}


def entity_words(label):
    return set(re.findall(r'[a-z0-9_]+', label.lower())) - NOISE


def alias_scope(proposed, previous, history, command):
    words = set(proposed.split('-'))
    verify = next((s for s in reversed(history) if s.startswith('verify-')), None)
    # Commands consisting only of the submission protocol are unambiguous even
    # if the agent incorrectly names them after an old implementation scope.
    cmd = command.strip()
    submission = bool(re.fullmatch(r'echo\s+COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\s*&&\s*cat\s+patch\.txt', cmd))
    patch_write = bool(re.fullmatch(r'git\s+diff\s+[^\n;&|]*>\s*patch\.txt', cmd))
    patch_read = bool(re.fullmatch(r'cat\s+patch\.txt', cmd))
    if verify and (submission or patch_write or patch_read):
        return verify, 'final_patch_protocol'
    if verify and words and words <= ACTIVITIES:
        return verify, 'activity_only_verification_or_handoff'
    if proposed.startswith('reproduce-') and previous.startswith('diagnose-'):
        a, b = entity_words(proposed), entity_words(previous)
        if a and a == b:
            return previous, 'same_question_reproduction'
    return proposed, None


class SemanticScopeMemory(InlineScopeMemory):
    def __init__(self, policy):
        super().__init__(policy)
        self.raw_scopes = []
        self.aliases = []

    def _add(self, turn):
        parsed = parse_turn_context(turn, self.policy)
        previous = self.scopes[-1] if self.scopes else ''
        before_changes = self.goal_changes
        super()._add(turn)
        raw = parsed['scope'] if parsed else (self.raw_scopes[-1] if self.raw_scopes else 'task')
        self.raw_scopes.append(raw)
        command = ''
        for message in turn:
            for call in message.get('tool_calls') or []:
                try:
                    args = json.loads(call['function']['arguments'])
                    command = args.get('command', '')
                except (ValueError, KeyError, TypeError):
                    continue
                break
            if command:
                break
        proposed = self.scopes[-1]
        effective, reason = alias_scope(proposed, previous, self.scopes[:-1], command)
        self.scopes[-1] = effective
        self.goal_changes = before_changes + bool(previous and previous != effective)
        self.aliases.append({'raw_scope':raw,'effective_scope':effective,'reason':reason})

    def select(self, messages, _unused=None):
        selected, trace = super().select(messages, _unused)
        trace['activity_alias'] = self.aliases[-1] if self.aliases else None
        return selected, trace
