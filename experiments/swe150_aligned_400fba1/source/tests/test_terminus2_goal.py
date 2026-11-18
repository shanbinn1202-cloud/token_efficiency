"""goal_query's Goal inputs on the chat-text protocol for Terminus 2 (remote_ra_live_v2.terminus2_goal,
PROVISIONAL): the pinned template, the extraction rules and their refusals. CPU only.

The fixtures are synthetic: made-up instructions and terminal output, formatted with the
pinned template (``T2.TEMPLATE``, whose SHA-256 is Harbor 0.21.0's terminus-json-plain.txt)
and with this file's own transcription of Terminus 2.0.0's prompts (terminus_2.py, read on
the agent host); no task instruction or trajectory of a real run is used.
``REMOTE_RA_HARBOR_TERMINUS2_DIR`` (a directory laid out like harbor/agents/terminus_2/,
opt-in) also checks the pins and every literal against Harbor's own files: their SHA-256,
and each literal a constant part of one of their f-strings.

``Terminus2`` is test_chat_text_protocol's Terminus client with Terminus 2.0.0's exact
summarization prompts (test_chat_text_e2e's goal_query sessions use it).
"""
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))

import test_chat_text_protocol as T  # noqa: E402  the Terminus client
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import terminus2_goal as T2  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402

HARBOR = os.environ.get('REMOTE_RA_HARBOR_TERMINUS2_DIR')

# ---- Terminus 2.0.0's prompts, transcribed from terminus_2.py (Harbor 0.21.0) --------------------

SUMMARY = """You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""

QUESTIONS = """You are picking up work from a previous AI agent on this task:

**Original Task:** {instruction}

**Summary from Previous Agent:**
{summary}

**Current Terminal Screen:**
{screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the solution that are not answered in the summary
from the prior agent. After you ask these questions you will be on your own,
so ask everything you need to know."""

ANSWERS = 'The next agent has a few questions for you, please answer each of them one by one in detail:\n\n'
HANDOFF = ('Here are the answers the other agent provided.\n\n%s\n\nContinue working on this task from where the '
           'previous agent left off. You can no longer ask questions. Please follow the spec to interact with '
           'the terminal.')
SHORT = 'Briefly continue this task: %s\n\nCurrent state: %s\n\nNext steps (2-3 sentences):'
PARSE_ERROR = ('Previous response had parsing errors:\n%s\n\nPlease fix these issues and provide a proper '
               'JSON response.')
COMPLETION = ('Current terminal state:\n%s\n\nAre you sure you want to mark the task as complete? This will trigger '
              "your solution to be graded and you won't be able to make any further corrections. If so, include "
              '"task_complete": true in your JSON response again.')
OUTPUT_LIMIT = ('ERROR!! NONE of the actions you just requested were performed because you exceeded %s. Your '
                'outputs must be less than %s. Re-issue this request, breaking it into chunks each of which is '
                'less than %s.')


def summary_prompt(instruction):
    return SUMMARY.format(instruction=instruction)


def question_prompt(instruction, summary, screen):
    return QUESTIONS.format(instruction=instruction, summary=summary, screen=screen)


def first_prompt(instruction, screen='root@4f0c2a9d81be:/app# \n\n\n'):
    """Terminus 2's first message: the pinned template with the instruction and the first screen
    (``_limit_output_length(get_incremental_output())``: 'Current Terminal Screen:' at first)."""
    return T2.TEMPLATE.format(instruction=instruction, terminal_state='Current Terminal Screen:\n' + screen)


def new_output(text):
    return 'New Terminal Output:\n' + text


def warnings(feedback, output):
    """_run_agent_loop: ``f"Previous response had warnings:\\n{feedback}\\n\\n{output}"``, feedback
    'WARNINGS: ' + the parser's '- ...' lines."""
    return 'Previous response had warnings:\nWARNINGS: %s\n\n%s' % (feedback, output)


def timeout(command, seconds, state):
    return T2.TIMEOUT_TEMPLATE.format(command=command, timeout_sec=seconds, terminal_state=state)


def user(content):
    return dict(role='user', content=content)


def assistant(content):
    return dict(role='assistant', content=content)


class Terminus2(T.Terminus):
    """test_chat_text_protocol's Terminus client with Terminus 2.0.0's exact summarization prompts
    (its unwinding, fallbacks, length errors and proactive summary are already 2.0.0's)."""

    async def summarize(self):
        if not self.chat:
            return None
        self.events.append('summary')
        prompt = summary_prompt(self.instruction)
        summary = await self.call(prompt, self.chat)
        questions_prompt = question_prompt(self.instruction, summary, self.screen)
        questions = await self.call(questions_prompt, [])
        answers = await self.call(ANSWERS + questions, self.chat + [user(prompt), assistant(summary)])
        self.chat = [self.chat[0], user(questions_prompt), assistant(questions)]
        return HANDOFF % answers


INSTRUCTION = ('Repair the CSV loader in /app/loader.py so that quoted fields keep their commas; run '
               'pytest -q tests/ until it passes. Résumé files use UTF-8 {braces} too.')
P0 = first_prompt(INSTRUCTION)


def conversation(*turns):
    """[P0, reply, user, reply, user, ...] with the given user messages after the first."""
    messages = [user(P0)]
    for k, content in enumerate(turns):
        messages += [assistant('{"analysis": "step %d", "plan": "next", "commands": []}' % k), user(content)]
    return messages


class TemplateTests(unittest.TestCase):
    def test_pins_and_pieces(self):
        self.assertEqual(sha256(T2.TEMPLATE.encode()).hexdigest(), T2.TEMPLATE_SHA256)
        self.assertEqual(len(T2.TEMPLATE.encode()), 2916)
        self.assertEqual(sha256(T2.TIMEOUT_TEMPLATE.encode()).hexdigest(), T2.TIMEOUT_SHA256)
        self.assertEqual(T2.TEMPLATE_SHA256, '89a3dc3a15752b748a99fb907c9251ab31a464e13386583fb48476b078b54a34')
        self.assertTrue(T2.TEMPLATE_HEAD.endswith('- Commands array can be empty if you want to wait without taking '
                                                  'action\n\nTask Description:\n'))
        self.assertEqual((T2.TEMPLATE_SEP, T2.TEMPLATE_TAIL), ('\n\nCurrent terminal state:\n', '\n'))
        self.assertNotIn('{{', T2.TEMPLATE_HEAD)
        self.assertIn('{\n  "analysis"', T2.TEMPLATE_HEAD)
        self.assertEqual(P0, T2.TEMPLATE_HEAD + INSTRUCTION + T2.TEMPLATE_SEP + 'Current Terminal Screen:\n'
                         + 'root@4f0c2a9d81be:/app# \n\n\n' + T2.TEMPLATE_TAIL)
        self.assertEqual((T2.TIMEOUT_HEAD, T2.TIMEOUT_TAIL), ('Previous command:\n', ' '))
        # This file's transcription of terminus_2.py equals the module's literals.
        self.assertEqual(summary_prompt('X'), T2.SUMMARY_HEAD + 'X' + T2.SUMMARY_TAIL)
        self.assertEqual(question_prompt('X', 'S', 'W'), T2.QUESTIONS_HEAD + 'X' + T2.QUESTIONS_SUMMARY_SEP + 'S'
                         + T2.QUESTIONS_SCREEN_SEP + 'W' + T2.QUESTIONS_TAIL)
        self.assertEqual((ANSWERS, HANDOFF % 'A'), (T2.ANSWERS_HEAD, T2.HANDOFF_HEAD + 'A' + T2.HANDOFF_TAIL))
        self.assertEqual(SHORT % ('X', 'W'), T2.SHORT_HEAD + 'X' + T2.SHORT_STATE_SEP + 'W' + T2.SHORT_TAIL)
        self.assertEqual(PARSE_ERROR % 'E', T2.PARSE_ERROR_HEAD + 'E' + T2.PARSE_ERROR_TAIL)
        self.assertEqual(COMPLETION % 'O', T2.COMPLETION_HEAD + 'O' + T2.COMPLETION_TAIL)
        self.assertTrue((OUTPUT_LIMIT % (('1536 tokens',) * 3)).startswith(T2.OUTPUT_LIMIT_HEAD))
        # test_chat_text_protocol's client sends the same hand-off, answers, short and length prompts.
        self.assertEqual((T.HANDOFF, T.ANSWER_PROMPT % 'q', T.OUTPUT_LIMIT),
                         (HANDOFF, ANSWERS + 'q', OUTPUT_LIMIT % (('the maximum output length',) * 3)))

    def test_record_and_config(self):
        record = T2.record()
        self.assertEqual((record['status'], record['agent_version'], record['harbor'], record['template_sha256']),
                         ('provisional', '2.0.0', '0.21.0', T2.TEMPLATE_SHA256))
        agents = 'harbor/agents/terminus_2/'
        self.assertEqual(record['sources'], {
            agents + 'templates/timeout.txt': T2.TIMEOUT_SHA256,
            agents + 'terminus_2.py': 'c6b72b8c6289809b2ff3a3009b8f118fa1edb1da205f48d0b56c6938a49cb12f',
            agents + 'tmux_session.py': '38642b794c3358809dabddfb36ea53d57fadf3e1e6fd736d82b39da8f3ce7976'})
        self.assertEqual((record['requests'], record['requirement_rules'], record['observation_rules']),
                         (sorted(T2.REQUESTS), sorted(T2.REQUIREMENT_RULES), sorted(T2.OBSERVATION_RULES)))
        self.assertEqual(json.loads(json.dumps(record)), record)
        # Every rule but the defaults has its place in the documented order.
        self.assertEqual(set(T2.ORDER), (set(T2.REQUESTS) - {'main'}) | (set(T2.OBSERVATION_RULES) - {'subcall_none'}))
        chat = C.record()
        config = dict(protocol=C.PROTOCOL, chat_text=chat, tasks=C.slot_tasks(512), task=C.slot_tasks(1)[0],
                      max_running_sessions=20, policy='goal_query', chat_text_goal=record)
        self.assertEqual(C.check_config(config), chat)
        self.assertEqual(T2.check_config(config), record)
        for change in (dict(chat_text_goal=None), dict(chat_text_goal=dict(record, status='confirmed')),
                       dict(chat_text_goal=dict(record, template_sha256='0' * 64)),
                       dict(policy='observation_query'), dict(policy='full')):
            with self.subTest(change=sorted(change)), self.assertRaises(LiveV2Error):
                C.check_config(dict(config, **change))
        without = {k: v for k, v in config.items() if k != 'chat_text_goal'}
        with self.assertRaises(LiveV2Error):
            C.check_config(without)                          # goal_query needs the record
        for policy in ('full', 'random50', 'observation_query', 'random_fill'):
            self.assertEqual(C.check_config(dict(without, policy=policy)), chat)


def inputs(messages):
    g = T2.classify(messages)
    return (g.request, g.requirement_rule, g.requirement, g.observation_rule, g.observation)


class ExtractionTests(unittest.TestCase):
    maxDiff = None

    # ---- rule 1: the requirement, the template match and its refusal -----------------------------
    def test_first_turn_requirement_is_the_instruction(self):
        g = T2.classify([user(P0)])
        self.assertEqual((g.request, g.requirement_rule, g.requirement, g.requirement_span, g.observation_rule,
                          g.observation, g.observation_span, g.rewrite, g.handoff),
                         ('main', 'template_instruction', INSTRUCTION, (0, len(T2.TEMPLATE_HEAD),
                          len(T2.TEMPLATE_HEAD) + len(INSTRUCTION)), 'first_turn', '', None, True, False))
        self.assertEqual(T2.context_fields(g), dict(objective=INSTRUCTION, observation='', message_source='user',
                                                    structured_state=dict(tool_name=None, return_code=None)))
        # Multi-line and non-ASCII instructions, a large first screen, an instruction that itself
        # holds the template's separator (the instruction ends at the last one).
        for instruction in ('Line one.\n\nLine two: héllo 世界\n- a list\n', 'Say "Current terminal state:"',
                            'A\n\nCurrent terminal state:\nB'):
            for screen in ('', 'root@x:/# \n' * 40):
                with self.subTest(instruction=instruction, screen=len(screen)):
                    self.assertEqual(T2.classify([user(first_prompt(instruction, screen))]).requirement, instruction)

    def test_a_first_message_off_the_template_is_refused(self):
        bad = [
            user(T.first_prompt(INSTRUCTION)),                       # test_chat_text_protocol's short shape
            user(P0.replace('Linux environment', 'Linux  environment', 1)),
            user(first_prompt(INSTRUCTION, 'root@x:/app# ')[:-1]),    # the template's final newline missing
            user(P0 + ' '),
            user(' ' + P0),
            user(P0.replace('Current Terminal Screen:\n', 'Screen:\n')),     # no tmux label
            user(T2.TEMPLATE),                                        # unformatted
            user(INSTRUCTION),
            user(first_prompt('   \n ')),                             # an empty instruction
            user(question_prompt('', 'S', 'W')),                      # a questions prompt without a task
        ]
        for messages in [[m] for m in bad] + [[dict(role='system', content='You are helpful.'), user(P0)]]:
            with self.subTest(first=messages[0]['content'][:40]), \
                    self.assertRaisesRegex(T2.GoalInputsRefused, 'goal_query on chat-text serves Harbor 0.21.0 '
                                           r'Terminus 2 \(2.0.0, json parser\) only'):
                T2.classify(messages)
        with self.assertRaisesRegex(T2.GoalInputsRefused, 'system message'):
            T2.classify([dict(role='system', content='s'), user(P0)])
        # The refusal does not depend on what follows the first message.
        with self.assertRaises(T2.GoalInputsRefused):
            T2.classify([bad[0], assistant('{}'), user(new_output('ok'))])

    # ---- rule 2: the latest turn's terminal output, the wrappers removed --------------------------
    def test_observation_wrappers(self):
        out = 'root@4f0c2a9d81be:/app# pytest -q\n3 failed, 1 passed\nroot@4f0c2a9d81be:/app# '
        trace = 'Traceback (most recent call last):\n  File "loader.py", line 9, in load\nValueError: bad quote\n'
        cases = [
            (new_output(out), 'terminal_output', out),
            ('Current Terminal Screen:\n' + out, 'terminal_output', out),
            (new_output('\n' + trace), 'terminal_output', '\n' + trace),
            (warnings('- Extra text detected before JSON object', new_output(out)), 'warnings_terminal_output', out),
            (warnings('- AUTO-CORRECTED: fix - please fix this in future responses\n- Command 1: Missing duration '
                      'field, using default 1.0', 'Current Terminal Screen:\n' + out), 'warnings_terminal_output', out),
            (COMPLETION % new_output(out), 'completion_terminal_output', out),
            (timeout('make -j8\n', 60.0, new_output(out)), 'timeout_terminal_state', out),
            (timeout('cat\n', 1.0, 'Current Terminal Screen:\n' + out), 'timeout_terminal_state', out),
            (warnings('- Extra text detected after JSON object', timeout('sleep 99\n', 0.5, new_output(out))),
             'warnings_terminal_output', out),
            (COMPLETION % timeout('x\n', 2.0, new_output(out)), 'completion_terminal_output', out),
            (PARSE_ERROR % 'ERROR: No valid JSON found in response', 'parse_error_none', ''),
            (PARSE_ERROR % "ERROR: Missing required fields: plan\nWARNINGS: - Extra text detected before JSON object",
             'parse_error_none', ''),
            (OUTPUT_LIMIT % (('1536 tokens',) * 3), 'output_limit_none', ''),
            (OUTPUT_LIMIT % (('the maximum output length',) * 3) + '\n\nParser warnings from your truncated '
             'response:\n- Extra text detected before JSON object', 'output_limit_none', ''),
            ('Something Terminus 2.0.0 never sends', 'raw_message', 'Something Terminus 2.0.0 never sends'),
        ]
        # Terminus's _limit_output_length keeps the first and last 5,000 bytes; the marker stays.
        long_ = 'x' * 5000 + '\n[... output limited to 10000 bytes; 123 interior bytes omitted ...]\n' + 'y' * 5000
        cases.append((new_output(long_), 'terminal_output', long_))
        for content, rule, expected in cases:
            with self.subTest(rule=rule, content=content[:50]):
                messages = conversation(new_output('older output'), content)
                g = T2.classify(messages)
                self.assertEqual((g.request, g.requirement, g.observation_rule, g.observation),
                                 ('main', INSTRUCTION, rule, expected))
                i, start, end = g.observation_span or (len(messages) - 1, 0, 0)
                self.assertEqual((i, messages[i]['content'][start:end]), (len(messages) - 1, expected))
                self.assertEqual(T2.context_fields(g)['message_source'], 'tool')
                self.assertEqual(T2.context_fields(g)['observation'], expected)

    def test_observation_is_the_latest_turn_only(self):
        messages = conversation(new_output('first output'), new_output('second output'), new_output('third'))
        self.assertEqual(inputs(messages), ('main', 'template_instruction', INSTRUCTION, 'terminal_output', 'third'))
        # A rebuilt or rebound session (local turn 0 with history) takes its newest message too.
        self.assertEqual(T2.context_fields(T2.classify(messages))['observation'], 'third')

    # ---- rule 3: summaries -----------------------------------------------------------------------
    def test_summary_sub_calls_and_the_hand_off(self):
        history = conversation(new_output('a'), new_output('b'))
        summary = 'I cloned the repo and found the bug in loader.py.\n\n**Current Terminal Screen:**\nquoted'
        screen = 'root@4f0c2a9d81be:/app# '
        prompt = summary_prompt(INSTRUCTION)
        questions = question_prompt(INSTRUCTION, summary, screen)
        calls = {
            'summary': history + [assistant('{}'), user(prompt)],
            'questions': [user(questions)],
            'answers': history + [assistant('{}'), user(prompt), assistant(summary), user(ANSWERS + '1. Which?')],
            'unwound': [user(P0), assistant('{}'), user(prompt)],            # after a context-length refusal
        }
        expected = dict(summary='summary_call', questions='questions_call', answers='answers_call',
                        unwound='summary_call')
        # The sub-calls on the main session's history keep its last Goal; the others the requirement's.
        goals = dict(summary='session_last', answers='session_last', unwound='session_last',
                     questions='requirement_only')
        for name, messages in calls.items():
            with self.subTest(name=name):
                g = T2.classify(messages)
                self.assertEqual((g.request, g.requirement, g.observation_rule, g.observation, g.rewrite,
                                  g.subcall_goal), (expected[name], INSTRUCTION, 'subcall_none', '', False, goals[name]))
                self.assertEqual(T2.context_fields(g), dict(objective=INSTRUCTION, observation='',
                    message_source='user', structured_state=dict(tool_name=None, return_code=None)))
        self.assertEqual(T2.classify(calls['questions']).requirement_rule, 'questions_task')
        # The hand-off: [P0, questions prompt, questions, hand-off]; the same first message (a
        # rebuild of the same client session), the instruction from P0, the summary as the observation.
        handoff = [user(P0), user(questions), assistant('1. Which file?\n2. Which test?'),
                   user(HANDOFF % 'The loader is /app/loader.py.')]
        g = T2.classify(handoff)
        self.assertEqual((g.request, g.requirement_rule, g.requirement, g.observation_rule, g.observation,
                          g.rewrite, g.handoff), ('main', 'template_instruction', INSTRUCTION, 'handoff_summary',
                                                  summary, True, True))
        i, start, end = g.observation_span
        self.assertEqual((i, handoff[1]['content'][start:end]), (1, summary))
        self.assertEqual((T2.context_fields(g)['message_source'], g.subcall_goal), ('tool', None))
        self.assertTrue(T2.evidence(g, 'maintained')['goal_restarted'])
        # After the hand-off the conversation goes on with terminal output.
        self.assertEqual(inputs(handoff + [assistant('{}'), user(new_output('c'))])[3:], ('terminal_output', 'c'))
        # The short-summary sub-call and hand-off, and the no-LLM fallback (context-length path).
        short = [user(SHORT % (INSTRUCTION, screen))]
        self.assertEqual(inputs(short), ('short_summary_call', 'short_summary_task', INSTRUCTION, 'subcall_none', ''))
        self.assertEqual(T2.classify(short).subcall_goal, 'requirement_only')
        unwound = [user(P0), assistant('{}')]
        self.assertEqual(inputs(unwound + [user('%s\n\nSummary: %s' % (INSTRUCTION, 'Run pytest next.'))]),
                         ('main', 'template_instruction', INSTRUCTION, 'short_summary', 'Run pytest next.'))
        self.assertEqual(inputs(unwound + [user('%s\n\nCurrent state: %s' % (INSTRUCTION, screen))]),
                         ('main', 'template_instruction', INSTRUCTION, 'fallback_state', screen))
        # A hand-off whose questions prompt names another task is no summary hand-off.
        other = [user(P0), user(question_prompt('Another task.', summary, screen)), assistant('?'),
                 user(HANDOFF % 'x')]
        self.assertEqual(T2.classify(other).observation_rule, 'raw_message')

    def test_evidence_and_row(self):
        g = T2.classify(conversation(new_output('done')))
        n = len(T2.TEMPLATE_HEAD)
        self.assertEqual(T2.evidence(g, 'maintained'), dict(
            definition=T2.DEFINITION, status='provisional', request='main',
            requirement=dict(rule='template_instruction', span=dict(message=0, start=n, end=n + len(INSTRUCTION)),
                             chars=len(INSTRUCTION)),
            observation=dict(rule='terminal_output', span=dict(message=2, start=len('New Terminal Output:\n'),
                                                                end=len(new_output('done'))), chars=4),
            rewriter='allowed', goal_source='maintained', goal_restarted=False))
        self.assertEqual(T2.row(g), dict(definition=T2.DEFINITION, request='main', requirement='template_instruction',
                                         observation='terminal_output', rewriter='allowed'))
        sub = T2.classify([user(question_prompt(INSTRUCTION, 's', 'w'))])
        self.assertEqual((T2.evidence(sub, 'requirement_only')['rewriter'],
                          T2.evidence(sub, 'requirement_only')['observation']),
                         ('skipped', dict(rule='subcall_none', span=None, chars=0)))
        # The engine's Goal source must fit the request: maintained for main, a sub-call's otherwise.
        for inputs_, source in ((g, 'session_last'), (g, 'requirement_only'), (sub, 'maintained'), (g, 'other')):
            with self.assertRaises(LiveV2Error):
                T2.evidence(inputs_, source)

    def test_sub_call_goals(self):
        """query_selection's sub-call Goal (SUBCALL_GOALS): the session's last Goal for the summary
        and answers calls, unchanged in everything the selector reads (hashes, query text), recorded
        as a KEEP of this turn without the rewriter; the requirement's when the session has none, and
        always for the questions and short-summary calls."""
        from remote_ra_live_v2 import query_selection as QS
        from remote_ra_live_v2 import selection_contract as SC
        _, schema, goals, _ = QS.upstream()
        context = lambda turn, observation='', source='user': schema.TurnContext(
            'slot-1', 'chat-text-slot-0001', turn, INSTRUCTION, None, None, observation, message_source=source,
            structured_state=dict(tool_name=None, return_code=None))
        maintained = goals.MaintainedGoalConstructor(None)        # the session's maintained Goal
        maintained.construct(context(1))
        trace = 'Traceback (most recent call last):\n  File "loader.py", line 9, in load\nValueError: bad quote\n'
        last = maintained.construct(context(2, trace, 'tool'))
        self.assertTrue(last.errors)

        class Selection:                       # the authority's state subcall_goal reads
            contract = dict(entity_drift_steps=3)
            requirement_goal = QS.QuerySelectionAuthority.requirement_goal
            subcall_goal = QS.QuerySelectionAuthority.subcall_goal
        selection = Selection()
        summary = T2.classify(conversation(new_output('a')) + [assistant('{}'), user(summary_prompt(INSTRUCTION))])
        questions = T2.classify([user(question_prompt(INSTRUCTION, 's', 'w'))])
        # No last Goal yet: the requirement's (an INITIALIZE without features or the rewriter).
        alone, source = selection.subcall_goal('slot-1', context(3), summary)
        self.assertEqual((source, alone.gate_decision, alone.slm_attempted, alone.errors, alone.text),
                         ('requirement_only', 'INITIALIZE', False, (),
                          'Retrieve historical evidence relevant to: ' + INSTRUCTION))
        selection.last_goals = {'slot-1': last}
        kept, source = selection.subcall_goal('slot-1', context(3), summary)
        self.assertEqual(source, 'session_last')
        self.assertEqual((kept.version_hash, kept.retrieval_goal_hash, kept.selector_payload),
                         (last.version_hash, last.retrieval_goal_hash, last.selector_payload))
        self.assertEqual(SC.goal_query_text(kept, 'full_goal'), SC.goal_query_text(last, 'full_goal'))
        self.assertEqual((kept.turn_id, kept.gate_decision, kept.applied_decision, kept.slm_attempted,
                          kept.fallback_reason, kept.reused_previous, kept.triggers, kept.changed_fields,
                          kept.diagnostics), (3, 'KEEP', 'KEEP', False, None, True, ('summarization_subcall',), (),
                                              {'kept_from_turn': 2}))
        # Another session's Goal is never taken; the questions call takes the requirement's.
        self.assertEqual(selection.subcall_goal('slot-2', context(3), summary)[1], 'requirement_only')
        self.assertEqual(selection.subcall_goal('slot-1', context(3), questions)[1], 'requirement_only')
        # The requirement-only constructor keeps no session.
        self.assertEqual(selection.requirement_constructor.session_count(), 0)

    def test_a_terminus_run_classifies_every_request(self):
        """Terminus2 against test_chat_text_protocol's protocol-level server: plain turns, the
        proactive summary, a context-length refusal with the full summary and the short-summary
        fallback; every request has Goal inputs and the expected kinds appear in order."""
        kit = T.char_kit()
        server = T.ProtocolServer(kit)
        tokens = lambda messages: len(kit.encode(''.join(m['content'] for m in messages)))
        agent = Terminus2(INSTRUCTION, send=server.send, count=tokens, context_limit=5600, threshold=600)
        prompt = P0
        for k in range(14):
            T.run(agent.step(prompt))
            prompt = new_output('root@x:/app# step %d\n' % k + 'line\n' * 6)
        self.assertIn('summary', agent.events)
        kinds = [T2.classify(messages).request for messages, status, _ in agent.requests]
        rules = [T2.classify(messages).observation_rule for messages, _, _ in agent.requests]
        self.assertTrue(all(status == 200 for _, status, _ in agent.requests))
        start = kinds.index('summary_call')
        self.assertEqual(kinds[start:start + 4], ['summary_call', 'questions_call', 'answers_call', 'main'])
        self.assertEqual(rules[start + 3], 'handoff_summary')
        self.assertEqual(rules[0], 'first_turn')
        self.assertEqual(set(rules) - {'first_turn', 'handoff_summary', 'subcall_none'}, {'terminal_output'})


@unittest.skipUnless(HARBOR, 'set REMOTE_RA_HARBOR_TERMINUS2_DIR to a copy of harbor/agents/terminus_2 (opt-in)')
class HarborSourceTests(unittest.TestCase):
    """The pins against Harbor 0.21.0's own files (read-only copies): SHA-256, and every literal of
    terminus2_goal a constant part of an f-string (or string expression) of those files."""

    def files(self):
        root = Path(HARBOR)
        return {name: (root / name).read_bytes() for name in ('templates/terminus-json-plain.txt',
                                                              'templates/timeout.txt', 'terminus_2.py',
                                                              'tmux_session.py')}

    @staticmethod
    def fragments(source):
        """Every maximal run of constant text in the string expressions of ``source``: f-string
        constant parts and string concatenations, the formatted values cutting the runs."""
        def flat(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return [node.value]
            if isinstance(node, ast.JoinedStr):
                return [part for value in node.values for part in (flat(value) if isinstance(value, ast.Constant)
                                                                   else [None])]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return flat(node.left) + flat(node.right)
            return [None]
        found = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
                run = ''
                for part in flat(node) + [None]:
                    if part is None:
                        found.add(run)
                        run = ''
                    else:
                        run += part
        return found - {''}

    def test_pins(self):
        files = self.files()
        self.assertEqual(files['templates/terminus-json-plain.txt'].decode(), T2.TEMPLATE)
        self.assertEqual(files['templates/timeout.txt'].decode(), T2.TIMEOUT_TEMPLATE)
        record = T2.record()
        for name, raw in files.items():
            pin = record['template_sha256'] if name.endswith('json-plain.txt') else \
                record['sources']['harbor/agents/terminus_2/' + name]
            self.assertEqual(sha256(raw).hexdigest(), pin, name)

    def test_literals_are_harbor_constants(self):
        files = self.files()
        terminus = self.fragments(files['terminus_2.py'].decode())
        tmux = self.fragments(files['tmux_session.py'].decode())
        self.assertIn('2.0.0', terminus)                                      # version()
        self.assertIn('JSON response', terminus)                              # _get_error_response_type, json
        for label in T2.TERMINAL_LABELS:
            self.assertIn(label, tmux)
        for name in ('SUMMARY_HEAD', 'SUMMARY_TAIL', 'QUESTIONS_HEAD', 'QUESTIONS_SUMMARY_SEP', 'QUESTIONS_SCREEN_SEP',
                     'QUESTIONS_TAIL', 'ANSWERS_HEAD', 'HANDOFF_HEAD', 'HANDOFF_TAIL', 'SHORT_HEAD', 'SHORT_STATE_SEP',
                     'SHORT_TAIL', 'SHORT_HANDOFF_SEP', 'STATE_HANDOFF_SEP', 'PARSE_ERROR_HEAD', 'WARNINGS_SEP',
                     'OUTPUT_LIMIT_HEAD', 'COMPLETION_HEAD', 'COMPLETION_TAIL'):
            with self.subTest(name=name):
                self.assertIn(getattr(T2, name), terminus)
        # Two literals join a constant and a formatted value of Terminus 2.0.0's json parser.
        self.assertIn('\n\nPlease fix these issues and provide a proper ', terminus)
        self.assertEqual(T2.PARSE_ERROR_TAIL, '\n\nPlease fix these issues and provide a proper ' + 'JSON response'
                         + '.')
        self.assertIn('Previous response had warnings:\n', terminus)
        self.assertIn('WARNINGS: ', terminus)
        self.assertEqual(T2.WARNINGS_HEAD, 'Previous response had warnings:\n' + 'WARNINGS: ')
        # This file's transcriptions are Harbor's too (built from the same constants).
        self.assertIn(SUMMARY.split('{instruction}')[0], terminus)
        self.assertIn(QUESTIONS.split('{screen}')[1], terminus)


if __name__ == '__main__':
    unittest.main()
