import pytest
from types import SimpleNamespace

from minisweagent.exceptions import FormatError, InterruptAgentFlow

from baselines.group.model import GroupModel as MemoryModel


def test_recorded_pruning_stops_before_any_model_or_gate_request(tmp_path):
    stop_path = tmp_path / "pruned.json"
    stop_path.write_text('{"reason": "experiment dominated by complete control"}')
    model = MemoryModel(model_name="openai/test", stop_path=stop_path)
    with pytest.raises(InterruptAgentFlow) as caught:
        model.query([])
    assert caught.value.messages[0]["extra"]["exit_status"] == "TrialPruned"
    assert model.requests == model.selections == []


def response(arguments):
    function = SimpleNamespace(name="bash", arguments=arguments)
    call = SimpleNamespace(function=function, id="call-1")
    message = SimpleNamespace(tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice])


def test_scope_v10_rejects_missing_metadata_before_execution():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v10")
    with pytest.raises(FormatError) as caught:
        model._parse_actions(response('{"command":"pwd"}'))
    assert "not executed" in caught.value.messages[0]["content"]


def test_scope_v10_accepts_agent_generated_metadata():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v10")
    actions = model._parse_actions(
        response(
            '{"command":"pwd","scope":"diagnose-path","event":"Inspect path",'
            '"scope_changed":true}'
        )
    )
    assert actions == [{"command": "pwd", "tool_call_id": "call-1"}]


def test_scope_v14_asks_agent_to_resolve_inconsistent_change_fields():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v14")
    model.memory.scopes.append("diagnose-old")
    with pytest.raises(FormatError) as caught:
        model._parse_actions(
            response(
                '{"command":"pwd","scope":"diagnose-new",'
                '"event":"Inspect path","scope_changed":false}'
            )
        )
    assert "Decide yourself" in caught.value.messages[0]["content"]


def test_scope_v15_gives_two_concrete_consistency_options():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v15")
    model.memory.scopes.append("diagnose-old")
    with pytest.raises(FormatError) as caught:
        model._parse_actions(
            response(
                '{"command":"pwd","scope":"diagnose-new",'
                '"event":"Inspect path","scope_changed":false}'
            )
        )
    message = caught.value.messages[0]["content"]
    assert "scope='diagnose-old'" in message
    assert "scope='diagnose-new'" in message


def test_scope_v16_retries_inconsistency_only_once_then_trusts_agent():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v16")
    model.memory.scopes.append("diagnose-old")
    inconsistent = response(
        '{"command":"pwd","scope":"diagnose-new",'
        '"event":"Inspect path","scope_changed":false}'
    )
    with pytest.raises(FormatError):
        model._parse_actions(inconsistent)
    assert model._parse_actions(inconsistent) == [
        {"command": "pwd", "tool_call_id": "call-1"}
    ]


def test_scope_v19_observes_inconsistency_without_blocking_agent_action():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v19")
    model.memory.scopes.append("diagnose-old")
    actions = model._parse_actions(
        response(
            '{"command":"pwd","scope":"diagnose-new",'
            '"event":"Inspect path","scope_changed":false}'
        )
    )
    assert actions == [{"command": "pwd", "tool_call_id": "call-1"}]


@pytest.mark.parametrize('variant', ['scope_tool_v21', 'scope_tool_v22', 'scope_tool_v24', 'scope_tool_v25', 'scope_tool_v26', 'scope_tool_v27', 'scope_tool_v28', 'scope_tool_v29'])
def test_scope_prompt_does_not_accumulate_in_archive_or_retries(variant):
    import copy
    model = MemoryModel(model_name='openai/test', variant=variant)
    messages = [{'role': 'system', 'content': 'Base rules'},
                {'role': 'user', 'content': 'Fix the bug'}]
    before = copy.deepcopy(messages)
    first = model._prepare_messages_for_api(messages)
    second = model._prepare_messages_for_api(messages)
    assert messages == before
    assert first == second
    assert first[0]['content'].count('SCOPE METADATA') == 1


@pytest.mark.parametrize("variant", ["scope_tool_v24", "scope_tool_v25", "scope_tool_v26"])
def test_entity_scope_does_not_retry_agent_label_against_rule_generated_key(variant):
    model = MemoryModel(model_name="openai/test", variant=variant)
    model.memory.scopes.append("diagnose-astropy/table")
    actions = model._parse_actions(response(
        '{"command":"pwd","scope":"diagnose-table","event":"Inspect",'
        '"scope_changed":false}'
    ))
    assert actions == [{"command": "pwd", "tool_call_id": "call-1"}]


def test_v27_retains_v16_audit_after_leaving_verify_without_duplication():
    import copy
    from baselines.group.memory import SCOPE_VERIFY_AUDIT_V12_INSTRUCTION
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v27")
    phases = iter(["diagnose-root", "verify-fix", "diagnose-regression"])
    model.memory.select = lambda messages, gate: (messages, {"active_scope": next(phases)})
    messages = [{"role": "system", "content": "Base rules"},
                {"role": "user", "content": "Fix the bug"}]
    before = copy.deepcopy(messages)
    first = model._prepare_messages_for_api(messages)
    second = model._prepare_messages_for_api(messages)
    third = model._prepare_messages_for_api(messages)
    assert SCOPE_VERIFY_AUDIT_V12_INSTRUCTION not in first[0]["content"]
    assert second[0]["content"].count(SCOPE_VERIFY_AUDIT_V12_INSTRUCTION) == 1
    assert third[0]["content"] == second[0]["content"]
    assert third[0]["content"].count("SCOPE METADATA") == 1
    assert messages == before
    assert model.memory.policy.change_authoritative
    assert not model.memory.policy.entity_boundaries
    assert model.memory.policy.max_active_turns is None


def test_v28_enables_same_audit_from_first_request():
    from baselines.group.memory import SCOPE_VERIFY_AUDIT_V12_INSTRUCTION
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v28")
    result = model._prepare_messages_for_api([
        {"role": "system", "content": "Base rules"},
        {"role": "user", "content": "Fix the bug"},
    ])
    assert result[0]["content"].count(SCOPE_VERIFY_AUDIT_V12_INSTRUCTION) == 1


@pytest.mark.parametrize("variant", ["scope_tool_v27", "scope_tool_v28", "scope_tool_v29"])
def test_v16_descendants_preserve_single_consistency_retry(variant):
    model = MemoryModel(model_name="openai/test", variant=variant)
    model.memory.scopes.append("diagnose-old")
    reply = response('{"command":"pwd","scope":"diagnose-new",'
                     '"event":"Inspect","scope_changed":false}')
    with pytest.raises(FormatError):
        model._parse_actions(reply)
    assert model._parse_actions(reply) == [{"command": "pwd", "tool_call_id": "call-1"}]


def test_v29_uses_exact_change_definition_in_prompt_and_tool_schema(monkeypatch):
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v29")
    prepared = model._prepare_messages_for_api([
        {"role": "system", "content": "Base rules"},
        {"role": "user", "content": "Fix the bug"},
    ])
    assert "After choosing scope, set scope_changed" in prepared[0]["content"]
    captured = []
    monkeypatch.setattr("baselines.group.model.litellm.completion",
                        lambda **kwargs: captured.append(kwargs) or SimpleNamespace(usage=None))
    model._query(prepared)
    tool = captured[0]["tools"][0]["function"]
    assert tool["strict"]
    assert "label is unchanged" in tool["parameters"]["properties"]["scope_changed"]["description"]
    assert model.memory.policy.change_authoritative
    assert not model.memory.policy.entity_boundaries


import json

from baselines.group.memory import (
    InlineScopeMemory,
    InlineScopePolicy,
    canonical_scope,
    parse_scope_label,
    repeated_action_result,
    strip_scope_metadata,
)


def turn(scope, event, changed, observation):
    arguments = json.dumps(
        {
            "command": f"command-{observation}",
            "scope": scope,
            "event": event,
            "scope_changed": changed,
        }
    )
    return [
        {
            "role": "assistant",
            "content": "Use a tool.",
            "tool_calls": [{"function": {"name": "bash", "arguments": arguments}}],
        },
        {"role": "tool", "content": observation},
    ]


def messages(turns):
    return [{"role": "user", "content": "Fix the bug."}] + [
        message for bundle in turns for message in bundle
    ]


def test_label_parser_validates_and_canonicalizes():
    policy = InlineScopePolicy()
    assert parse_scope_label(
        '{"scope":"Diagnose None Return","event":"Found the failing path",'
        '"scope_changed":true}',
        policy,
    ) == {
        "scope": "diagnose-none-return",
        "event": "Found the failing path",
        "scope_changed": True,
    }
    assert parse_scope_label('{"scope":"x","event":"y"}', policy) is None
    assert canonical_scope("  Verify: API / Result  ") == "verify-api-/-result"


def test_active_scope_and_historical_endpoints_select_raw_turns():
    turns = [
        turn("diagnose-none", "locate", True, "0"),
        turn("diagnose-none", "confirm", False, "1"),
        turn("implement-guard", "patch", True, "2"),
        turn("implement-guard", "inspect", False, "3"),
        turn("verify-guard", "pass", True, "4"),
    ]
    selected, trace = InlineScopeMemory(InlineScopePolicy()).select(messages(turns))
    assert trace["scope_endpoints"] == {
        "diagnose-none": 1,
        "implement-guard": 3,
        "verify-guard": 4,
    }
    assert trace["kept"] == [1, 3, 4]
    assert selected == messages([turns[1], turns[3], turns[4]])


def test_scope_label_controls_retrieval_even_if_changed_flag_is_inconsistent():
    turns = [
        turn("verify-fix", "test", True, "0"),
        turn("diagnose-regression", "inspect failure", False, "1"),
    ]
    _, trace = InlineScopeMemory(InlineScopePolicy()).select(messages(turns))
    assert trace["active_scope"] == "diagnose-regression"
    assert trace["scope_transitioned"] is True
    assert trace["scope_changed"] is False
    assert trace["scope_change_consistent"] is False
    assert trace["scope_count"] == 2


def test_authoritative_change_policy_uses_agents_boolean_judgment():
    policy = InlineScopePolicy(change_authoritative=True)
    turns = [
        turn("diagnose-old", "inspect", True, "0"),
        turn("renamed-by-mistake", "continue", False, "1"),
    ]
    _, trace = InlineScopeMemory(policy).select(messages(turns))
    assert trace["active_scope"] == "diagnose-old"
    assert trace["scope_count"] == 1


def test_return_to_scope_restores_all_raw_turns():
    turns = [
        turn("diagnose-none", "locate", True, "source"),
        turn("diagnose-none", "cause", False, "cause"),
        turn("implement-guard", "edit", True, "edited"),
        turn("diagnose-none", "revisit", True, "new evidence"),
    ]
    selected, trace = InlineScopeMemory(InlineScopePolicy()).select(messages(turns))
    assert trace["kept"] == [0, 1, 2, 3]
    assert trace["scope_count"] == 2
    assert selected == messages(turns)


def test_metadata_is_removed_only_from_api_copy():
    raw = messages([turn("diagnose-none", "locate", True, "source")])
    cleaned = strip_scope_metadata(raw)
    raw_args = json.loads(raw[1]["tool_calls"][0]["function"]["arguments"])
    clean_args = json.loads(cleaned[1]["tool_calls"][0]["function"]["arguments"])
    assert set(raw_args) == {"command", "scope", "event", "scope_changed"}
    assert clean_args == {"command": "command-source"}
    assert cleaned[0] == raw[0]


def test_invalid_metadata_inherits_scope():
    turns = [
        turn("diagnose-none", "inspect", True, "source"),
        turn("", "", False, "raw observation"),
    ]
    selected, trace = InlineScopeMemory(InlineScopePolicy()).select(messages(turns))
    assert trace["active_scope"] == "diagnose-none"
    assert trace["annotation_valid"] is False
    assert trace["kept"] == [0, 1]
    assert selected == messages(turns)


def test_recent_turn_safety_net_survives_scope_transition():
    turns = [
        turn("diagnose-a", "one", True, "0"),
        turn("diagnose-a", "two", False, "1"),
        turn("diagnose-a", "three", False, "2"),
        turn("diagnose-a", "decision", False, "3"),
        turn("implement-b", "edit", True, "4"),
    ]
    _, trace = InlineScopeMemory(InlineScopePolicy(recent=3)).select(messages(turns))
    assert trace["kept"] == [2, 3, 4]


def test_exact_repeated_action_result_detects_third_occurrence_only():
    same = turn("diagnose-a", "inspect", False, "unchanged")
    other = turn("diagnose-a", "other", False, "different")
    assert not repeated_action_result(messages([same, other, same]))
    assert repeated_action_result(messages([same, other, same, same]))


def test_active_scope_window_retains_distant_other_scope_endpoint():
    turns = [turn('diagnose-root', 'cause found', True, 'evidence')]
    turns += [turn('implement-fix', 'edit', i == 0, str(i)) for i in range(12)]
    _, trace = InlineScopeMemory(InlineScopePolicy(recent=3, max_active_turns=8)).select(messages(turns))
    assert trace['kept'] == [0] + list(range(5, 13))
    assert trace['scope_endpoints']['diagnose-root'] == 0


import json

from baselines.group.memory import EntityScopeResolver
from baselines.group.memory import INLINE_SCOPE_POLICIES, InlineScopeMemory


def entity_turn(scope, command, output="ok"):
    return [
        {"role": "assistant", "content": "Inspect evidence", "tool_calls": [
            {"function": {"name": "bash", "arguments": json.dumps({
                "scope": scope, "event": "Inspect evidence", "scope_changed": False,
                "command": command,
            })}}
        ]},
        {"role": "tool", "content": output},
    ]


def resolve(resolver, scope, command, output="ok"):
    return resolver.resolve(entity_turn(scope, command, output), {"scope": scope})


def test_sibling_source_and_test_files_share_component_despite_label_changes():
    resolver = EntityScopeResolver()
    first = resolve(resolver, "diagnose-column", "cat astropy/table/column.py")
    assert resolve(resolver, "diagnose-table", "cat astropy/table/table.py") == first
    assert resolve(resolver, "diagnose-regression", "cat astropy/table/tests/test_column.py") == first


def test_disjoint_component_changes_scope_even_with_identical_agent_label():
    resolver = EntityScopeResolver()
    first = resolve(resolver, "explore-codebase", "cat astropy/table/column.py")
    second = resolve(resolver, "explore-codebase", "cat astropy/io/fits/column.py")
    assert first != second
    assert resolve(resolver, "explore-codebase", "cat astropy/table/table.py") == first


def test_output_paths_do_not_create_entities_and_phase_is_not_shell_verb():
    resolver = EntityScopeResolver()
    first = resolve(resolver, "verify-table", "pytest astropy/table/tests/test_table.py")
    assert resolve(resolver, "verify-diff", "git diff", "astropy/io/fits/column.py") == first
    assert resolve(resolver, "verify-source", "cat astropy/table/table.py") == first
    assert resolve(resolver, "implement-table", "cat astropy/table/table.py") != first


def test_scope_history_is_unbounded_and_restored_on_return():
    memory = InlineScopeMemory(INLINE_SCOPE_POLICIES["scope_tool_v24"])
    prefix = [{"role": "user", "content": "Fix the issue"}]
    history = [entity_turn("diagnose-table", "cat astropy/table/table.py") for _ in range(15)]
    history += [entity_turn("diagnose-fits", "cat astropy/io/fits/column.py")]
    history += [entity_turn("diagnose-table", "cat astropy/table/column.py")]
    messages = prefix + [m for t in history for m in t]
    selected, trace = memory.select(messages)
    assert trace["active_kept"] == list(range(15)) + [16]
    assert trace["kept"] == list(range(17))
    assert selected == messages
    assert trace["entity_boundary"]["boundary_reason"] == "return_to_component"


def test_symbol_scope_separates_questions_in_one_file_and_restores_previous():
    resolver = EntityScopeResolver(symbols=True)
    first = resolve(resolver, "diagnose-_line_type", "cat astropy/io/ascii/qdp.py")
    second = resolve(resolver, "diagnose-_get_tables_from_qdp_file", "cat astropy/io/ascii/qdp.py")
    assert first != second
    assert resolve(resolver, "diagnose-_line_type", "cat astropy/io/ascii/qdp.py") == first


def test_symbol_scope_keeps_parent_child_component_and_patch_review_phase():
    resolver = EntityScopeResolver(symbols=True)
    first = resolve(resolver, "diagnose-wcs", "cat astropy/wcs/wcsapi/low_level_api.py")
    assert resolve(resolver, "diagnose-wcs", "cat astropy/wcs/wcsapi/wrappers/sliced_wcs.py") == first
    assert resolve(resolver, "diagnose-wcs", "cat astropy/__init__.py") == first
    verify = resolve(resolver, "verify-wcs", "pytest astropy/wcs/wcsapi/tests/test_api.py")
    assert resolve(resolver, "create-patch", "git diff astropy/wcs/wcsapi/wrappers/sliced_wcs.py") == verify


def test_method_focus_takes_precedence_over_shared_class_name():
    resolver = EntityScopeResolver(symbols=True)
    one = resolver.resolve(entity_turn("diagnose-method", "cat pkg/core/object.py"),
                           {"scope": "diagnose-method", "event": "Inspect SkyCoord.__getattr__"})
    two = resolver.resolve(entity_turn("diagnose-method", "cat pkg/core/object.py"),
                           {"scope": "diagnose-method", "event": "Inspect SkyCoord.__getattribute__"})
    assert one != two


import json
from baselines.group.memory import alias_scope, SemanticScopeMemory
from baselines.group.memory import INLINE_SCOPE_POLICIES
from baselines.group.model import GroupModel as MemoryModel


def test_aliases_only_unambiguous_activities():
    history=['diagnose-array-cause','implement-array-copy','verify-array-copy']
    for label in ['verify-edge-cases','verify-existing-tests','cleanup-test-files','create-patch','submit-fix']:
        assert alias_scope(label, history[-1], history, 'pwd')[0]==history[-1]
    for label in ['diagnose-array-shape','verify-other-conversion','implement-new-cache','diagnose-import-error']:
        assert alias_scope(label, history[-1], history, 'pwd')==(label,None)
    assert alias_scope('reproduce-array-cause',history[0],history[:1],'python repro.py')[0]==history[0]
    assert alias_scope('reproduce-other-cause',history[0],history[:1],'python repro.py')[1] is None


def test_submission_command_does_not_match_other_work():
    h=['verify-array-copy']
    assert alias_scope('implement-old',h[-1],h,'git diff -- file.py > patch.txt')[0]==h[-1]
    assert alias_scope('implement-old',h[-1],h,'git diff -- file.py > patch.txt && python edit.py')[1] is None


def test_raw_labels_retained_while_selection_uses_aliases_without_cap():
    m=SemanticScopeMemory(INLINE_SCOPE_POLICIES['scope_tool_v32']);msgs=[]
    labels=['diagnose-array-cause','reproduce-array-cause','implement-array-copy','verify-array-copy','verify-edge-cases','verify-edge-cases','diagnose-import-error']
    for i,label in enumerate(labels):
        args=json.dumps(dict(command='pwd',scope=label,event='Inspect evidence',scope_changed=i==0 or label!=labels[i-1]))
        msgs += [{'role':'assistant','tool_calls':[{'function':{'name':'bash','arguments':args}}]}, {'role':'tool','content':'ok'}]
    m.select(msgs)
    assert m.raw_scopes==labels
    assert m.scopes==['diagnose-array-cause','diagnose-array-cause','implement-array-copy','verify-array-copy','verify-array-copy','verify-array-copy','diagnose-import-error']
    assert m.policy.max_active_turns is None
    assert MemoryModel(model_name='openai/test',variant='scope_tool_v32').behavior_variant=='scope_tool_v16'


import copy
import json
from types import SimpleNamespace

import pytest
from minisweagent.exceptions import FormatError

from baselines.group.model import GroupModel as MemoryModel


def test_examples_change_only_framework_instructions_and_not_archive():
    task = '<pr_description>{"command":"issue example"}</pr_description>\n{"command":"pwd"}'
    messages = [{"role": "system", "content": "System"}, {"role": "user", "content": task}]
    original = copy.deepcopy(messages)
    old = MemoryModel(model_name="openai/test", variant="scope_tool_v29")
    new = MemoryModel(model_name="openai/test", variant="scope_tool_v30")
    a = old._prepare_messages_for_api(messages)
    b = new._prepare_messages_for_api(messages)
    assert a[0] == b[0]
    assert a[1]["content"] == task
    assert b[1]["content"].split('</pr_description>')[0] == task.split('</pr_description>')[0]
    assert json.loads(b[1]["content"].split('</pr_description>')[1])["scope_changed"] is True
    assert new._prepare_messages_for_api(messages) == b
    assert messages == original
    assert old.selections[0] == new.selections[0]


def test_missing_metadata_feedback_contains_exact_failed_arguments_and_format():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v30")
    arguments = json.dumps({"command": "printf 'a\\nb'"})
    with pytest.raises(FormatError) as caught:
        model._parse_actions(response(arguments))
    content = caught.value.messages[0]["content"]
    assert json.dumps(arguments) in content
    for field in ('command', 'scope', 'event', 'scope_changed'):
        assert f'"{field}"' in content
    # Original response is feedback data, not an executable assistant call.
    assert caught.value.messages[0]["role"] == "user"
    assert model.requests == []


def test_no_tool_feedback_preserves_failed_text():
    model = MemoryModel(model_name="openai/test", variant="scope_tool_v30")
    r = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(tool_calls=[], content="I need to inspect code."))])
    with pytest.raises(FormatError) as caught:
        model._parse_actions(r)
    assert 'I need to inspect code.' in caught.value.messages[0]['content']
    assert 'scope_changed' in caught.value.messages[0]['content']


def test_success_and_consistency_retry_unchanged():
    old = MemoryModel(model_name="openai/test", variant="scope_tool_v29")
    new = MemoryModel(model_name="openai/test", variant="scope_tool_v30")
    valid = response('{"command":"pwd","scope":"diagnose-path","event":"Inspect path","scope_changed":true}')
    assert old._parse_actions(valid) == new._parse_actions(valid)
    invalid = response('{"command":"pwd","scope":"diagnose-new","event":"Inspect path","scope_changed":false}')
    for model in (old, new):
        model.memory.scopes.append("diagnose-path")
        with pytest.raises(FormatError):
            model._parse_actions(invalid)
    assert old._parse_actions(invalid) == new._parse_actions(invalid)


def test_v31_uses_question_placeholders_without_changing_v30():
    messages = [{'role':'system','content':'System'}, {'role':'user','content':'<pr_description>Issue</pr_description>\n{"command":"pwd"}'}]
    model = MemoryModel(model_name='openai/test', variant='scope_tool_v31')
    result = model._prepare_messages_for_api(messages)
    example = json.loads(result[1]['content'].split('</pr_description>')[1])
    assert example['scope'] == 'diagnose-<entity-and-cause>'
    assert 'A scope is a concrete unresolved question' in result[0]['content']
    assert 'SCOPE AFTER FAILED VERIFICATION' not in result[0]['content']
    assert model.memory.policy.max_active_turns is None
    assert model.memory.policy.recent == 3
    assert model.behavior_variant == 'scope_tool_v16'
    assert model._prepare_messages_for_api(messages) == result
    old = MemoryModel(model_name='openai/test',variant='scope_tool_v30')._prepare_messages_for_api(messages)
    assert 'diagnose-code-structure' in old[1]['content']


from baselines.group.feedback import recovery_evidence


def history(text, output):
    return [{'role':'assistant','content':text}, {'role':'tool','content':output}]


def test_failed_regression_triggers_in_verify():
    assert recovery_evidence(history('', '2 failed, 9 passed'), ['verify-fix'])


def test_failed_fix_returning_to_diagnose_triggers():
    assert recovery_evidence(history("The fix didn't work. Let me investigate.", 'trace'), ['diagnose-issue','verify-fix','diagnose-issue'])


def test_normal_progress_environment_and_boolean_output_do_not_trigger():
    for text,output,scopes in [('', '19 passed', ['verify-fix']), ('', 'ImportError: missing dependency', ['verify-fix']), ('', 'False', ['verify-fix']), ('Test failed', 'trace', ['diagnose-issue']), ('', '1 failed', ['diagnose-issue'])]:
        assert recovery_evidence(history(text, output), scopes) is None
