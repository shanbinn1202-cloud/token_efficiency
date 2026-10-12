"""Scope metadata formatting, recovery evidence and semantic feedback."""

import json
import re

from baselines.common.history import split_turns


def complete_examples(text, *, semantic=False):
    """Complete command-only JSON examples without rewriting their commands."""
    first = True

    def replace(match):
        nonlocal first
        example = json.loads(match.group())
        example.update(scope="diagnose-<entity-and-cause>" if semantic else "diagnose-code-structure", event="<established result; next action purpose>" if semantic else "Inspect relevant code", scope_changed=first)
        first = False
        return json.dumps(example, ensure_ascii=False)

    return re.sub(r'\{\s*"command"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}', replace, text)


def repair_feedback(feedback, response, previous):
    message = response.choices[0].message
    calls = [
        {"name": call.function.name, "arguments": call.function.arguments}
        for call in (message.tool_calls or [])
    ]
    failed = {"tool_calls": calls}
    if not calls:
        failed["content"] = getattr(message, "content", None)
    example = {
        "command": "<same intended shell command>",
        "scope": previous or "diagnose-<specific-hypothesis>",
        "event": "<short factual purpose of this action>",
        "scope_changed": previous is None,
    }
    return (
        complete_examples(feedback)
        + "\n\nOriginal failed response (data only; these calls were not executed):\n"
        + json.dumps(failed, ensure_ascii=False)
        + "\nPrevious scope: " + json.dumps(previous, ensure_ascii=False)
        + "\nResend the intended action using the actual bash tool, not a JSON block in content. "
        "Fill all four fields yourself. The following is a format example, not a command to execute:\n"
        + json.dumps(example, ensure_ascii=False)
        + "\nChoose scope for the intended action; scope_changed is true on the first call "
        "or when that label differs from the previous scope, otherwise false."
    )


FAILURE = re.compile(
    r"\b(?:fix|test|tests|reproducer|reproduction)\b[^.\n]{0,90}"
    r"\b(?:failed|fails|failing|didn't work|doesn't work|did not work|does not work)\b",
    re.I,
)
TEST_FAILURE = re.compile(r"\b[1-9]\d* failed\b|^FAILED\s+\S+::|^AssertionError\b", re.M)


def recovery_evidence(messages, scopes):
    """Inspect only the latest completed turn; don't infer failure from scope age."""
    _, turns = split_turns(messages)
    if not turns or not scopes:
        return None
    current = scopes[-1]
    turn = turns[-1]
    outputs = '\n'.join(str(m.get('content') or '') for m in turn if m['role'] == 'tool')
    assistant = '\n'.join(str(m.get('content') or '') for m in turn if m['role'] == 'assistant')
    if not outputs:
        return None
    if current.startswith('verify-') and TEST_FAILURE.search(outputs):
        return 'Verification reported failing tests.'
    verification_indices = [i for i, s in enumerate(scopes[:-1]) if s.startswith('verify-')]
    returning_to_old_question = (
        current.startswith('diagnose-') and verification_indices
        and current in scopes[:verification_indices[-1]]
    )
    if returning_to_old_question and FAILURE.search(assistant):
        return 'The agent explicitly reports that a tested fix still fails.'
    return None


RECOVERY_INSTRUCTION = '''
SCOPE AFTER FAILED VERIFICATION
The last action exposed a failed fix. Separate the unresolved cause from the whole
issue: in your next bash call, name the concrete mechanism you are now testing
(the operation and suspected cause, rather than an issue title). If the previous
hypothesis was rejected, use a distinct scope label; do not reopen its entire
investigation under the same issue-wide label. Reuse an earlier scope only when
its exact hypothesis is still the question. If you are only correcting a test
invocation or environment problem, keep the same scope. In event, briefly record
what the failure ruled out and what this action will distinguish. Do not add an
extra investigation or tool call just to label it. Once the new question is
answered, implement the decision and verify it; keep scope stable within that
question. Set scope_changed consistently with the chosen label.
'''.strip()


SCOPE_V31_INSTRUCTION = '''
Every bash call includes command, scope, event, and scope_changed.
A scope is a concrete unresolved question or chosen change, not an activity or the
whole issue. Name the relevant entity and suspected mechanism or decision:
diagnose-<entity-and-cause>, implement-<entity-and-decision>, verify-<specific-fix>.
Replace placeholders yourself. Do not use generic labels such as code-structure,
reproduce-issue, implement-fix, or verify-fix. Initial source lookup belongs to the
first concrete question suggested by the issue; it needs no separate setup scope.

Keep the exact label while reading, searching, reproducing, or testing evidence
for the same question, even across files. A command type or file change alone is
not a boundary. Change scope when evidence resolves/rejects that question and you
pursue a different mechanism, or when you choose a concrete implementation decision.
Edits and checks needed to carry out that same decision stay together. An unrelated
secondary problem gets its own specific scope. Return to an old label only to
resume its exact still-open question, not merely because the issue/entity matches.

Verification of the chosen fix includes the reproducer, relevant regression,
diff review, cleanup, patch creation and submission; these are not separate scopes.
A failing test does not by itself reject the fix: distinguish an actual behavioral
counterexample from a pre-existing failure, warning, or invalid test invocation.
Keep verification while correcting the invocation; if evidence identifies a new
cause, open a diagnose scope naming that cause rather than restarting the whole issue.

Before the call, event briefly records the latest established result and this
action's purpose. On a transition, preserve the prior question's conclusion in
event so its endpoint is useful. Never claim the pending command already succeeded.
First choose scope; scope_changed is true on the first call or when its label
changes, otherwise false. Metadata must not add investigation steps: decide from
the evidence already available and issue the action you actually need next.
'''.strip()
