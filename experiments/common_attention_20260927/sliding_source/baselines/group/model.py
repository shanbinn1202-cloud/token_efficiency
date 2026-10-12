"""Group-specific scope prompts, recovery and tool-call validation."""
import copy
import time
from pathlib import Path
import litellm
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from baselines.group.memory import (
    INLINE_BASH_TOOL,
    INLINE_BASH_TOOL_STRICT,
    INLINE_BASH_TOOL_EXACT_CHANGE,
    INLINE_SCOPE_POLICIES,
    SCOPE_TOOL_INSTRUCTION,
    SCOPE_TOOL_V3_INSTRUCTION,
    SCOPE_TOOL_V4_INSTRUCTION,
    SCOPE_TOOL_V6_INSTRUCTION,
    SCOPE_TOOL_V7_INSTRUCTION,
    SCOPE_TOOL_V17_INSTRUCTION,
    SCOPE_TOOL_V19_INSTRUCTION,
    SCOPE_TOOL_V22_INSTRUCTION,
    SCOPE_STAGNATION_INSTRUCTION,
    SCOPE_OUTPUT_LIMIT_INSTRUCTION,
    SCOPE_VERIFY_AUDIT_INSTRUCTION,
    SCOPE_VERIFY_AUDIT_V12_INSTRUCTION,
    InlineScopeMemory,
    parse_scope_label,
    repeated_action_result,
    strip_scope_metadata,
)
from baselines.group.memory import SemanticScopeMemory
from baselines.group.feedback import SCOPE_V31_INSTRUCTION
from baselines.group.feedback import recovery_evidence, RECOVERY_INSTRUCTION
from baselines.group.feedback import complete_examples, repair_feedback

from baselines.common.model import HistoryModel

class GroupModel(HistoryModel):
    def __init__(self, *, variant="scope_tool_v30", policy=None, stop_path=None, **kwargs):
        super().__init__(variant=variant, stop_path=stop_path, **kwargs)
        self.variant = variant
        self.behavior_variant = "scope_tool_v16" if variant in {"scope_tool_v21", "scope_tool_v22", "scope_tool_v23", "scope_tool_v24", "scope_tool_v25", "scope_tool_v26", "scope_tool_v27", "scope_tool_v28", "scope_tool_v29", "scope_tool_v30", "scope_tool_v31", "scope_tool_v32"} else variant
        self.stop_path = Path(stop_path) if stop_path else None
        memory_class = SemanticScopeMemory if variant == "scope_tool_v32" else InlineScopeMemory
        self.memory = memory_class(policy or INLINE_SCOPE_POLICIES[variant])
        self.requests = []
        self.selections = []
        self.scope_consistency_retry_pending = False
        self.verify_audit_seen = variant == "scope_tool_v28"
        self.recovery_notified_turns = set()


    def _prepare_messages_for_api(self, messages):
        selected, trace = self.memory.select(messages, self.gate)
        if self.variant in {"scope_tool_v21", "scope_tool_v22", "scope_tool_v23", "scope_tool_v24", "scope_tool_v25", "scope_tool_v26", "scope_tool_v27", "scope_tool_v28", "scope_tool_v29", "scope_tool_v30", "scope_tool_v31", "scope_tool_v32"}:
            selected = copy.deepcopy(selected)
        if self.variant in {"scope_tool_v30", "scope_tool_v31", "scope_tool_v32"}:
            # Only framework instructions, never the issue text or observed history.
            for message in selected:
                content = message.get("content")
                if message.get("role") == "user" and isinstance(content, str) and "</pr_description>" in content:
                    issue, instructions = content.split("</pr_description>", 1)
                    message["content"] = issue + "</pr_description>" + complete_examples(instructions, semantic=self.variant in {"scope_tool_v31", "scope_tool_v32"})
                    break
        self.selections.append(trace)
        if self.variant in INLINE_SCOPE_POLICIES:
            stagnation_prompt = (
                self.behavior_variant in {"scope_tool_v18", "scope_tool_v20"}
                and repeated_action_result(messages)
            )
            trace["stagnation_prompt"] = stagnation_prompt
            if self.variant == "scope_tool_v1":
                selected = strip_scope_metadata(selected)
            system = next(
                (message for message in selected if message.get("role") == "system"),
                None,
            )
            if system is not None:
                instruction = (
                    SCOPE_TOOL_V17_INSTRUCTION
                    if self.variant == "scope_tool_v17"
                    else SCOPE_TOOL_V19_INSTRUCTION
                    if self.variant == "scope_tool_v19"
                    else SCOPE_TOOL_V7_INSTRUCTION
                    if self.behavior_variant in {"scope_tool_v7", "scope_tool_v8", "scope_tool_v9", "scope_tool_v10", "scope_tool_v11", "scope_tool_v12", "scope_tool_v13", "scope_tool_v14", "scope_tool_v15", "scope_tool_v16", "scope_tool_v18", "scope_tool_v20"}
                    else SCOPE_TOOL_V6_INSTRUCTION
                    if self.variant == "scope_tool_v6"
                    else SCOPE_TOOL_V4_INSTRUCTION
                    if self.variant == "scope_tool_v4"
                    else SCOPE_TOOL_V3_INSTRUCTION
                    if self.variant == "scope_tool_v3"
                    else SCOPE_TOOL_INSTRUCTION
                )
                if self.variant in {"scope_tool_v29", "scope_tool_v30", "scope_tool_v31", "scope_tool_v32"}:
                    instruction = instruction.replace(
                        "scope_changed is true on the first call and otherwise exactly when the underlying\n"
                        "hypothesis or chosen decision changes.",
                        "After choosing scope, set scope_changed to true on the first call or when the scope\n"
                        "label differs from the preceding call; otherwise set it to false.",
                    )
                if self.variant in {"scope_tool_v31", "scope_tool_v32"}:
                    instruction = SCOPE_V31_INSTRUCTION
                if self.variant == "scope_tool_v26":
                    instruction += (
                        "\n\nevent records the latest established evidence and the next "
                        "decision, rather than merely naming a command. A scope is complete "
                        "when its concrete question is answered. In verification, state the "
                        "behavior required by the issue and compare it with the observed "
                        "result of the original reproducer. Passing existing tests alone "
                        "does not establish that the reported issue is fixed. Do not invent "
                        "an observation before the tool returns; record it on the next call."
                    )
                if self.variant == "scope_tool_v22":
                    instruction = SCOPE_TOOL_V22_INSTRUCTION
                if (
                    self.behavior_variant in {"scope_tool_v7", "scope_tool_v8"}
                    and str(trace.get("active_scope") or "").startswith("verify-")
                ):
                    instruction += "\n\n" + SCOPE_VERIFY_AUDIT_INSTRUCTION
                if self.variant in {"scope_tool_v27", "scope_tool_v28", "scope_tool_v29", "scope_tool_v30", "scope_tool_v31", "scope_tool_v32"}:
                    self.verify_audit_seen |= str(trace.get("active_scope") or "").startswith("verify-")
                    trace["persistent_verify_audit"] = self.verify_audit_seen
                if (
                    self.behavior_variant in {"scope_tool_v13", "scope_tool_v14", "scope_tool_v15", "scope_tool_v16", "scope_tool_v17", "scope_tool_v18", "scope_tool_v19", "scope_tool_v20"}
                    and (str(trace.get("active_scope") or "").startswith("verify-")
                         or (self.variant in {"scope_tool_v27", "scope_tool_v28", "scope_tool_v29", "scope_tool_v30", "scope_tool_v31", "scope_tool_v32"} and self.verify_audit_seen))
                ):
                    instruction += "\n\n" + SCOPE_VERIFY_AUDIT_V12_INSTRUCTION
                if stagnation_prompt:
                    instruction += "\n\n" + SCOPE_STAGNATION_INSTRUCTION
                output_limit_prompt = bool(
                    self.variant == "scope_tool_v20"
                    and messages
                    and messages[-1].get("role") == "user"
                    and (messages[-1].get("extra") or {}).get("interrupt_type")
                    == "FormatError"
                    and "output token limit"
                    in str(messages[-1].get("content") or "").lower()
                )
                trace["output_limit_prompt"] = output_limit_prompt
                if output_limit_prompt:
                    instruction += "\n\n" + SCOPE_OUTPUT_LIMIT_INSTRUCTION
                if (
                    self.behavior_variant in {"scope_tool_v9", "scope_tool_v10", "scope_tool_v11", "scope_tool_v12"}
                    and trace.get("scope_transitioned")
                    and str(trace.get("active_scope") or "").startswith("verify-")
                ):
                    instruction += "\n\n" + (
                        SCOPE_VERIFY_AUDIT_V12_INSTRUCTION
                        if self.variant == "scope_tool_v12"
                        else SCOPE_VERIFY_AUDIT_INSTRUCTION
                    )
                if self.variant == "scope_tool_v30":
                    evidence = recovery_evidence(messages, self.memory.scopes)
                    turn_index = len(self.memory.scopes)
                    if evidence and turn_index not in self.recovery_notified_turns:
                        self.recovery_notified_turns.add(turn_index)
                        trace["scope_recovery_evidence"] = evidence
                        instruction += "\n\n" + RECOVERY_INSTRUCTION
                system["content"] = (
                    str(system.get("content") or "")
                    + "\n\nSCOPE METADATA\n"
                    + instruction
                )
        return LitellmModel._prepare_messages_for_api(self, selected)


    def _query(self, messages, **kwargs):
        start = time.monotonic()
        try:
            response = self.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[
                    INLINE_BASH_TOOL_EXACT_CHANGE
                    if self.variant in {"scope_tool_v29", "scope_tool_v30", "scope_tool_v31", "scope_tool_v32"}
                    else INLINE_BASH_TOOL_STRICT
                    if self.behavior_variant in {"scope_tool_v8", "scope_tool_v9", "scope_tool_v10", "scope_tool_v11", "scope_tool_v12", "scope_tool_v13", "scope_tool_v14", "scope_tool_v15", "scope_tool_v16", "scope_tool_v17", "scope_tool_v18", "scope_tool_v19", "scope_tool_v20"}
                    else INLINE_BASH_TOOL
                ],
                **(self.config.model_kwargs | kwargs),
            )
        except Exception as exc:
            self.requests.append({"kind": "agent", "seconds": time.monotonic() - start,
                                  "error": str(exc), "usage": {}})
            raise
        # Record before action parsing, including billed FormatError responses.
        self.record(response, "agent", time.monotonic() - start)
        return response


    def _parse_actions(self, response):
        try:
            return self._parse_scope_actions(response)
        except FormatError as error:
            if self.variant in {"scope_tool_v30", "scope_tool_v31", "scope_tool_v32"}:
                error.messages[0]["content"] = repair_feedback(
                    error.messages[0]["content"], response,
                    self.memory.scopes[-1] if self.memory.scopes else None,
                )
            raise


    def _parse_scope_actions(self, response):
        if self.behavior_variant in {"scope_tool_v10", "scope_tool_v11", "scope_tool_v12", "scope_tool_v13", "scope_tool_v14", "scope_tool_v15", "scope_tool_v16", "scope_tool_v17", "scope_tool_v18", "scope_tool_v19", "scope_tool_v20"}:
            calls = response.choices[0].message.tool_calls or []
            parsed_calls = [
                parse_scope_label(
                    call.function.arguments, self.memory.policy
                )
                for call in calls
                if call.function.name == "bash"
            ]
            invalid = any(parsed is None for parsed in parsed_calls)
            if invalid:
                raise FormatError(
                    {
                        "role": "user",
                        "content": (
                            "The bash call was not executed because it omitted valid "
                            "scope, event, or scope_changed fields. Resend the same "
                            "intended action once, filling all three fields yourself."
                        ),
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )
            if self.variant not in {"scope_tool_v24", "scope_tool_v25", "scope_tool_v26"} and self.behavior_variant in {"scope_tool_v14", "scope_tool_v15", "scope_tool_v16", "scope_tool_v17", "scope_tool_v18", "scope_tool_v20"}:
                scope_history = self.memory.raw_scopes if self.variant == "scope_tool_v32" else self.memory.scopes
                previous = scope_history[-1] if scope_history else ""
                for parsed in parsed_calls:
                    transitioned = not previous or parsed["scope"] != previous
                    if parsed["scope_changed"] != transitioned:
                        if (
                            self.behavior_variant in {"scope_tool_v16", "scope_tool_v17", "scope_tool_v18", "scope_tool_v20"}
                            and self.scope_consistency_retry_pending
                        ):
                            self.scope_consistency_retry_pending = False
                            previous = (
                                parsed["scope"]
                                if parsed["scope_changed"] or not previous
                                else previous
                            )
                            continue
                        self.scope_consistency_retry_pending = True
                        proposed = parsed["scope"]
                        guidance = (
                            "Decide yourself and resend the same action once: "
                            f"choose either scope='{previous}' with "
                            "scope_changed=false (continuity), or "
                            f"scope='{proposed}' with scope_changed=true "
                            "(transition)."
                            if self.behavior_variant in {"scope_tool_v15", "scope_tool_v16", "scope_tool_v17", "scope_tool_v18", "scope_tool_v20"}
                            else (
                                "Decide yourself and resend the same action once: "
                                "for continuity copy the exact previous scope "
                                f"'{previous}' and use false; for a real transition "
                                "use the new scope and true."
                            )
                        )
                        raise FormatError(
                            {
                                "role": "user",
                                "content": (
                                    "The bash call was not executed because scope and "
                                    "scope_changed disagree. " + guidance
                                ),
                                "extra": {"interrupt_type": "FormatError"},
                            }
                        )
                    self.scope_consistency_retry_pending = False
                    previous = parsed["scope"]
        return super()._parse_actions(response)
