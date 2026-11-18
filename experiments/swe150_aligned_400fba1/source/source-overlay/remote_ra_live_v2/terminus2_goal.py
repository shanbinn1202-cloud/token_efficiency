"""goal_query's Goal inputs on the chat-text protocol for Harbor 0.21.0's Terminus 2 (PROVISIONAL).

PROVISIONAL: the definition below is the team's proposal and waits for the Goal owner's
confirmation. Every choice is one named rule of the tables REQUESTS, REQUIREMENT_RULES,
OBSERVATION_RULES and SUBCALL_GOALS; ``classify`` applies them in a fixed order and nothing
else in the runtime decides what a Terminus Goal is built from. The Goal construction itself (the
maintained constructor, the Qwen3-0.6B rewriter, its prompt, schema, gate and fallbacks)
is the SWE-bench goal_query path, unchanged; only its inputs come from here:

- the requirement (the TurnContext's root objective) is the task instruction, cut out of
  Terminus 2's fixed prompt template in the session's first message;
- the observation (the TurnContext's current observation) is the terminal output of the
  latest turn only (the newest user message), with Terminus 2's fixed wrapper text
  removed; nothing is accumulated across turns;
- after a summary hand-off the Goal is built again from the instruction plus the summary
  text as the observation;
- Terminus 2's summarization sub-calls run no rewriter (``rewrite`` False): those on the
  main session's history (summary, answers) keep the session's last Goal, those that come
  as client sessions of their own (questions, short summary) take the requirement alone
  (SUBCALL_GOALS).

Pinned sources (Harbor 0.21.0 as installed on the agent host, harbor/agents/terminus_2/;
read there, never modified):

- templates/terminus-json-plain.txt, SHA-256 ``TEMPLATE_SHA256``: ``TEMPLATE`` below,
  byte for byte (checked when this module loads), the prompt Terminus 2 formats with
  ``str.format(instruction=..., terminal_state=...)`` as its first message;
- templates/timeout.txt, SHA-256 ``TIMEOUT_SHA256``: ``TIMEOUT_TEMPLATE`` below;
- terminus_2.py, SHA-256 ``TERMINUS_2_SHA256`` (``version()`` "2.0.0", parser "json"):
  the summarization prompts and the observation wrappers, the literals ``*_HEAD``,
  ``*_TAIL`` and ``*_SEP`` below, each a constant part of one of its f-strings;
- tmux_session.py, SHA-256 ``TMUX_SESSION_SHA256``: the two labels of
  ``get_incremental_output`` (``TERMINAL_LABELS``).

What Terminus 2.0.0 sends (no system message; the JSON parser; litellm adds the user
prompt to the history it passes):

- the main conversation: ``[P0, reply, observation, reply, ...]``, P0 the formatted
  template; the session key is P0, so every request of a trial has the same first message;
- a summary (proactive when fewer than 8,000 tokens are left, or after a context-length
  refusal and an unwind) makes three sub-calls: the summary prompt on the main history
  (key P0), the questions prompt alone (its own key: a new client session), and the
  answers prompt on the summary call's history (key P0); then the main conversation
  continues as ``[P0, questions prompt, questions, hand-off]``: the same first message,
  a rebuild of the same client session (not a new first message);
- if the full summary fails after a context-length refusal: the short-summary sub-call
  (its own key) and the hand-off ``instruction + "\\n\\nSummary: " + text``, or, if that
  fails too, ``instruction + "\\n\\nCurrent state: " + screen``.

Nothing here runs per request but string prefix/suffix tests and slicing of the newest
message (and of the first two messages); no hashing.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .contract import LiveV2Error, require
import policy_registry

STATUS = 'provisional'
DEFINITION = 'terminus2-goal-inputs-provisional-v2'   # v2: summary/answers sub-calls keep the session's Goal
RECORD_KEY = 'chat_text_goal'          # the startup config's record (serve.py --protocol chat-text --policy goal_query)
HARBOR_VERSION = '0.21.0'
AGENT = 'terminus-2'
AGENT_VERSION = '2.0.0'
PARSER = 'json'

TEMPLATE_FILE = 'harbor/agents/terminus_2/templates/terminus-json-plain.txt'
TEMPLATE_SHA256 = '89a3dc3a15752b748a99fb907c9251ab31a464e13386583fb48476b078b54a34'
TIMEOUT_FILE = 'harbor/agents/terminus_2/templates/timeout.txt'
TIMEOUT_SHA256 = '32bf9aa7b157a6a0e0ba0d33b81e59dd67204276f158afc1fe6cb684ced68872'
TERMINUS_2_FILE = 'harbor/agents/terminus_2/terminus_2.py'
TERMINUS_2_SHA256 = 'c6b72b8c6289809b2ff3a3009b8f118fa1edb1da205f48d0b56c6938a49cb12f'
TMUX_SESSION_FILE = 'harbor/agents/terminus_2/tmux_session.py'
TMUX_SESSION_SHA256 = '38642b794c3358809dabddfb36ea53d57fadf3e1e6fd736d82b39da8f3ce7976'

# terminus-json-plain.txt (Harbor 0.21.0), 2,916 bytes, verbatim.
TEMPLATE = (
    'You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will '
    'be given a task description and the output from previously executed commands. Your goal is to '
    'solve the task by providing batches of shell commands.\n'
    '\n'
    'Format your response as JSON with the following structure:\n'
    '\n'
    '{{\n'
    '  "analysis": "Analyze the current state based on the terminal output provided. What do you '
    'see? What has been accomplished? What still needs to be done?",\n'
    '  "plan": "Describe your plan for the next steps. What commands will you run and why? Be '
    'specific about what you expect each command to accomplish.",\n'
    '  "commands": [\n'
    '    {{\n'
    '      "keystrokes": "ls -la\\n",\n'
    '      "duration": 0.1\n'
    '    }},\n'
    '    {{\n'
    '      "keystrokes": "cd project\\n",\n'
    '      "duration": 0.1\n'
    '    }}\n'
    '  ],\n'
    '  "task_complete": true\n'
    '}}\n'
    '\n'
    'Required fields:\n'
    '- "analysis": Your analysis of the current situation\n'
    '- "plan": Your plan for the next steps\n'
    '- "commands": Array of command objects to execute\n'
    '\n'
    'Optional fields:\n'
    '- "task_complete": Boolean indicating if the task is complete (defaults to false if not '
    'present)\n'
    '\n'
    'Command object structure:\n'
    '- "keystrokes": String containing the exact keystrokes to send to the terminal (required)\n'
    '- "duration": Number of seconds to wait for the command to complete before the next command '
    'will be executed (defaults to 1.0 if not present)\n'
    '\n'
    'IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write '
    'commands exactly as you want them sent to the terminal:\n'
    '- You must end every command with a newline (\\n) or it will not execute.\n'
    '- For special key sequences, use tmux-style escape sequences:\n'
    '  - C-c for Ctrl+C\n'
    '  - C-d for Ctrl+D\n'
    '\n'
    'The "duration" attribute specifies the number of seconds to wait for the command to complete '
    '(default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, '
    'echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration '
    'of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set '
    'an appropriate duration as you determine necessary.\n'
    '\n'
    'It is better to set a smaller duration than a longer duration. It is always possible to wait '
    'again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} '
    'on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see '
    'intermediate result status.\n'
    '\n'
    'Important notes:\n'
    "- Each command's keystrokes are sent exactly as written to the terminal\n"
    "- Do not include extra whitespace before or after the keystrokes unless it's part of the "
    'intended command\n'
    '- Extra text before or after the JSON will generate warnings but be tolerated\n'
    '- The JSON must be valid - use proper escaping for quotes and special characters within strings\n'
    '- Commands array can be empty if you want to wait without taking action\n'
    '\n'
    'Task Description:\n'
    '{instruction}\n'
    '\n'
    'Current terminal state:\n'
    '{terminal_state}\n')

# timeout.txt (Harbor 0.21.0), 365 bytes, verbatim (it ends in a space, no newline).
TIMEOUT_TEMPLATE = (
    'Previous command:\n'
    '{command}\n'
    '\n'
    'The previous command timed out after {timeout_sec} seconds\n'
    '\n'
    'It is possible that the command is not yet finished executing. If that is the case, then do '
    'nothing. It is also possible that you have entered an interactive shell and should continue '
    'sending keystrokes as normal.\n'
    '\n'
    'Here is the current state of the terminal:\n'
    '\n'
    '{terminal_state} ')


def _fields(template, names):
    """The literal pieces of a str.format template around its fields ``names`` (in order),
    braces unescaped as str.format does."""
    pieces, rest = [], template
    for name in names:
        head, sep, rest = rest.partition('{%s}' % name)
        require(sep and '{' not in head.replace('{{', '').replace('}}', ''), 'template field %s' % name)
        pieces.append(head.replace('{{', '{').replace('}}', '}'))
    require('{' not in rest.replace('{{', ''), 'template has more fields')
    pieces.append(rest.replace('{{', '{').replace('}}', '}'))
    return tuple(pieces)


require(sha256(TEMPLATE.encode('utf-8')).hexdigest() == TEMPLATE_SHA256
        and sha256(TIMEOUT_TEMPLATE.encode('utf-8')).hexdigest() == TIMEOUT_SHA256,
        'Terminus 2 templates differ from their pinned SHA-256')
# P0 = TEMPLATE_HEAD + instruction + TEMPLATE_SEP + terminal_state + TEMPLATE_TAIL.
TEMPLATE_HEAD, TEMPLATE_SEP, TEMPLATE_TAIL = _fields(TEMPLATE, ('instruction', 'terminal_state'))
# The timeout observation:
# TIMEOUT_HEAD + command + TIMEOUT_SEP + seconds + TIMEOUT_STATE + terminal_state + TIMEOUT_TAIL.
TIMEOUT_HEAD, TIMEOUT_SEP, TIMEOUT_STATE, TIMEOUT_TAIL = _fields(TIMEOUT_TEMPLATE,
                                                                 ('command', 'timeout_sec', 'terminal_state'))
require(TEMPLATE.format(instruction='\x00i', terminal_state='\x00t')
        == TEMPLATE_HEAD + '\x00i' + TEMPLATE_SEP + '\x00t' + TEMPLATE_TAIL
        and TIMEOUT_TEMPLATE.format(command='c', timeout_sec=1.0, terminal_state='t')
        == TIMEOUT_HEAD + 'c' + TIMEOUT_SEP + '1.0' + TIMEOUT_STATE + 't' + TIMEOUT_TAIL,
        'Terminus 2 template pieces do not reassemble the template')

# tmux_session.py get_incremental_output: the label in front of every terminal output.
TERMINAL_LABELS = ('New Terminal Output:\n', 'Current Terminal Screen:\n')

# terminus_2.py 2.0.0, _summarize: sub-call 1 (summary), 2 (questions), 3 (answers), the hand-off.
SUMMARY_HEAD = ('You are about to hand off your work to another AI agent.\n'
                '            Please provide a comprehensive summary of what you have\n'
                '            accomplished so far on this task:\n\nOriginal Task: ')
SUMMARY_TAIL = ('\n\nBased on the conversation history, please provide a detailed summary covering:\n'
                '1. **Major Actions Completed** - List each significant command you executed\n'
                '            and what you learned from it.\n'
                '2. **Important Information Learned** - A summary of crucial findings, file\n'
                '            locations, configurations, error messages, or system state discovered.\n'
                '3. **Challenging Problems Addressed** - Any significant issues you\n'
                '            encountered and how you resolved them.\n'
                '4. **Current Status** - Exactly where you are in the task completion process.\n\n\n'
                'Be comprehensive and detailed. The next agent needs to understand everything\n'
                '            that has happened so far in order to continue.')
QUESTIONS_HEAD = 'You are picking up work from a previous AI agent on this task:\n\n**Original Task:** '
QUESTIONS_SUMMARY_SEP = '\n\n**Summary from Previous Agent:**\n'
QUESTIONS_SCREEN_SEP = '\n\n**Current Terminal Screen:**\n'
QUESTIONS_TAIL = ('\n\nPlease begin by asking several questions (at least five, more if necessary)\n'
                  'about the current state of the solution that are not answered in the summary\n'
                  'from the prior agent. After you ask these questions you will be on your own,\n'
                  'so ask everything you need to know.')
ANSWERS_HEAD = ('The next agent has a few questions for you, please answer each of them one by one in '
                'detail:\n\n')
HANDOFF_HEAD = 'Here are the answers the other agent provided.\n\n'
HANDOFF_TAIL = ('\n\nContinue working on this task from where the previous agent left off. You can no longer '
                'ask questions. Please follow the spec to interact with the terminal.')
# _query_llm after a context-length refusal: the short-summary sub-call and the two fallbacks.
SHORT_HEAD = 'Briefly continue this task: '
SHORT_STATE_SEP = '\n\nCurrent state: '
SHORT_TAIL = '\n\nNext steps (2-3 sentences):'
SHORT_HANDOFF_SEP = '\n\nSummary: '
STATE_HANDOFF_SEP = '\n\nCurrent state: '
# _run_agent_loop, _query_llm, _get_completion_confirmation_message: the observation wrappers.
PARSE_ERROR_HEAD = 'Previous response had parsing errors:\n'
PARSE_ERROR_TAIL = '\n\nPlease fix these issues and provide a proper JSON response.'
WARNINGS_HEAD = 'Previous response had warnings:\nWARNINGS: '
WARNINGS_SEP = '\n\n'
OUTPUT_LIMIT_HEAD = 'ERROR!! NONE of the actions you just requested were performed because you exceeded '
COMPLETION_HEAD = 'Current terminal state:\n'
COMPLETION_TAIL = ('\n\nAre you sure you want to mark the task as complete? This will trigger your solution to be '
                   "graded and you won't be able to make any further corrections. If so, include "
                   '"task_complete": true in your JSON response again.')

# ---- the rules (the provisional definition; each name is recorded with every Goal) ----------------

# What kind of Terminus request it is, and whether its Goal may call the rewriter.
REQUESTS = {
    'main': 'the main conversation: messages[0] is the pinned template (P0); the Goal path of goal_query',
    'summary_call': 'summarization sub-call 1: the newest message is the summary prompt (on the main history); '
                    'no rewriter, the Goal of SUBCALL_GOALS',
    'answers_call': 'summarization sub-call 3: the newest message is the answers prompt after the summary '
                    "call's prompt and reply (on the main history); no rewriter, the Goal of SUBCALL_GOALS",
    'questions_call': 'summarization sub-call 2: the questions prompt alone (its own client session); no '
                      'rewriter, the Goal of SUBCALL_GOALS',
    'short_summary_call': 'the short-summary sub-call after a context-length refusal (its own client session); '
                          'no rewriter, the Goal of SUBCALL_GOALS',
}
# The Goal of a summarization sub-call (none calls the rewriter): the sub-calls on the main
# session's history keep the session's last Goal; those that arrive as client sessions of
# their own take the requirement alone.
SUBCALL_GOALS = {
    'summary_call': 'session_last',
    'answers_call': 'session_last',
    'questions_call': 'requirement_only',
    'short_summary_call': 'requirement_only',
}
GOAL_SOURCES = {
    'maintained': 'the main conversation: the maintained Goal of the SWE-bench path (it may call the rewriter)',
    'session_last': "the engine session's last Goal (its previous turn's, kept across a rebuild on the same slot), "
                    'unchanged: recorded as a KEEP without the rewriter; requirement_only when the session has none '
                    '(its first request, or a fresh slot after a rebind)',
    'requirement_only': 'the deterministic Goal of the requirement alone (a rewriter-less constructor on a session '
                        "of its own); the session's maintained Goal is not touched",
}
REQUIREMENT_RULES = {
    'template_instruction': 'P0 = TEMPLATE_HEAD + instruction + TEMPLATE_SEP + terminal_state + TEMPLATE_TAIL, '
                            'terminal_state starting with a tmux label: the instruction (up to the last '
                            'TEMPLATE_SEP), as Terminus formats it into {instruction}',
    'questions_task': 'questions prompt: the Original Task between QUESTIONS_HEAD and the first '
                      'QUESTIONS_SUMMARY_SEP',
    'short_summary_task': 'short-summary prompt: the task between SHORT_HEAD and the first SHORT_STATE_SEP',
}
OBSERVATION_RULES = {
    'first_turn': 'P0 alone: no turn output yet (empty; message source user, as the first turn of the SWE path)',
    'terminal_output': 'the newest message is a terminal output: the text after its tmux label',
    'timeout_terminal_state': 'the timeout template: its terminal_state (after the tmux label); the command is '
                              'the agent\'s, not terminal output',
    'warnings_terminal_output': 'WARNINGS_HEAD + warnings + "\\n\\n" + terminal output (or timeout): that '
                                'output, unwrapped as above',
    'completion_terminal_output': 'COMPLETION_HEAD + terminal output (or timeout) + COMPLETION_TAIL: that '
                                  'output, unwrapped as above',
    'parse_error_none': 'the parse-error prompt: no terminal output (empty)',
    'output_limit_none': 'the reply-length error prompt: no terminal output (empty)',
    'handoff_summary': 'the summary hand-off [P0, questions prompt, questions, hand-off]: the summary text '
                       '(the Summary from Previous Agent in messages[1]); the Goal is built again here',
    'short_summary': 'the short-summary hand-off instruction + SHORT_HANDOFF_SEP + text: that text; the Goal is '
                     'built again here',
    'fallback_state': 'the no-LLM hand-off instruction + STATE_HANDOFF_SEP + screen: that screen; the Goal is '
                      'built again here',
    'raw_message': 'none of the above: the whole newest message',
    'subcall_none': 'a summarization sub-call: no observation (its Goal is SUBCALL_GOALS\')',
}
# The rule order of classify (sub-calls first, then the hand-offs, then the turn outputs).
ORDER = ('questions_call', 'short_summary_call', 'summary_call', 'answers_call', 'handoff_summary',
         'short_summary', 'fallback_state', 'first_turn', 'parse_error_none', 'output_limit_none',
         'completion_terminal_output', 'warnings_terminal_output', 'timeout_terminal_state', 'terminal_output',
         'raw_message')
HANDOFFS = frozenset({'handoff_summary', 'short_summary', 'fallback_state'})


class GoalInputsRefused(LiveV2Error):
    """A request whose first message is neither Terminus 2.0.0's pinned prompt template nor one of
    its summarization sub-call prompts: goal_query has no Goal definition for it."""


@dataclass(frozen=True)
class GoalInputs:
    """The Goal inputs of one request. Spans are (message index, start, end) into that
    message's content as sent."""
    request: str                  # a REQUESTS key
    requirement: str
    requirement_rule: str         # a REQUIREMENT_RULES key
    requirement_span: tuple
    observation: str
    observation_rule: str         # an OBSERVATION_RULES key
    observation_span: tuple | None

    @property
    def rewrite(self):
        """False for a summarization sub-call: no rewriter, the Goal of SUBCALL_GOALS."""
        return self.request == 'main'

    @property
    def subcall_goal(self):
        """A sub-call's Goal (SUBCALL_GOALS: 'session_last' or 'requirement_only'); None for main."""
        return SUBCALL_GOALS.get(self.request)

    @property
    def handoff(self):
        """A summary hand-off: the Goal is built again (instruction plus the summary text)."""
        return self.observation_rule in HANDOFFS


def record():
    """The startup config's ``chat_text_goal`` record: what this release builds a Terminus Goal from."""
    return dict(status=STATUS, definition=DEFINITION, harbor=HARBOR_VERSION, agent=AGENT, agent_version=AGENT_VERSION,
                parser=PARSER, template=TEMPLATE_FILE, template_sha256=TEMPLATE_SHA256,
                sources={TIMEOUT_FILE: TIMEOUT_SHA256, TERMINUS_2_FILE: TERMINUS_2_SHA256,
                         TMUX_SESSION_FILE: TMUX_SESSION_SHA256},
                requests=sorted(REQUESTS), requirement_rules=sorted(REQUIREMENT_RULES),
                observation_rules=sorted(OBSERVATION_RULES), subcall_goals=dict(SUBCALL_GOALS))


def check_config(config):
    """Startup only (chat_text_protocol.check_config): a goal_query chat-text config pins exactly
    this release's record; every other config none."""
    value = config.get(RECORD_KEY)
    if policy_registry.goal_text(config.get('policy')):
        require(value == record(), 'goal_query on chat-text needs the startup record %s of this release '
                '(Terminus 2 %s, Harbor %s; provisional)' % (RECORD_KEY, AGENT_VERSION, HARBOR_VERSION))
    else:
        require(RECORD_KEY not in config, '%s applies to goal_query only' % RECORD_KEY)
    return value


# ---- extraction -------------------------------------------------------------------------------------

NOT_TERMINUS = ('the first message is neither the pinned Terminus 2 prompt template (%s, sha256 %s...) nor one of '
                'its summarization sub-call prompts' % (TEMPLATE_FILE.rsplit('/', 1)[-1], TEMPLATE_SHA256[:16]))
SYSTEM_MESSAGE = 'a system message was sent (Terminus 2 sends none)'


def _refuse(why):
    raise GoalInputsRefused('goal_query on chat-text serves Harbor %s Terminus 2 (%s, %s parser) only (provisional '
                            'Goal definition %s): %s' % (HARBOR_VERSION, AGENT_VERSION, PARSER, DEFINITION, why))


def instruction_span(content):
    """(start, end) of the instruction in P0 (rule template_instruction), or None when
    ``content`` is not the pinned template formatted with an instruction and a terminal state."""
    if not content.startswith(TEMPLATE_HEAD) or not content.endswith(TEMPLATE_TAIL):
        return None
    end = content.rfind(TEMPLATE_SEP, len(TEMPLATE_HEAD), len(content) - len(TEMPLATE_TAIL))
    if end < 0 or not content.startswith(TERMINAL_LABELS, end + len(TEMPLATE_SEP)):
        return None
    return len(TEMPLATE_HEAD), end


def _between(content, head, sep, tail, last=False):
    """(start, end) of the text between ``head`` and ``sep`` (its first occurrence, or the last
    with ``last``) in a message that ends with ``tail``, else None."""
    if not content.startswith(head) or not content.endswith(tail) or len(content) < len(head) + len(tail):
        return None
    limit = len(content) - len(tail)
    end = content.rfind(sep, len(head), limit) if last else content.find(sep, len(head), limit)
    return None if end < 0 else (len(head), end)


def _terminal(content, start, end):
    """(rule, start, end) of the terminal output in content[start:end]: a timeout template's
    terminal state or a tmux-labelled output, without the label; None if neither."""
    text = content[start:end]
    if text.startswith(TIMEOUT_HEAD) and text.endswith(TIMEOUT_TAIL):
        state = text.rfind(TIMEOUT_STATE, len(TIMEOUT_HEAD), len(text) - len(TIMEOUT_TAIL))
        seconds = text.rfind(TIMEOUT_SEP, len(TIMEOUT_HEAD), max(state, 0))
        if state >= 0 and seconds >= 0:
            inner = start + state + len(TIMEOUT_STATE)
            stop = end - len(TIMEOUT_TAIL)
            label = next((x for x in TERMINAL_LABELS if content.startswith(x, inner)), '')
            return 'timeout_terminal_state', inner + len(label), stop
    label = next((x for x in TERMINAL_LABELS if text.startswith(x)), None)
    if label is not None:
        return 'terminal_output', start + len(label), end
    return None


def _observation(messages, instruction):
    """(rule, span) of a main-conversation request's observation (OBSERVATION_RULES, in ORDER)."""
    if len(messages) == 1:
        return 'first_turn', None
    i = len(messages) - 1
    content = messages[i]['content']
    n = len(content)
    if len(messages) == 4 and content.startswith(HANDOFF_HEAD) and content.endswith(HANDOFF_TAIL) \
            and messages[1]['role'] == 'user':
        span = _between(messages[1]['content'], QUESTIONS_HEAD + instruction + QUESTIONS_SUMMARY_SEP,
                        QUESTIONS_SCREEN_SEP, QUESTIONS_TAIL, last=True)
        if span is not None:
            return 'handoff_summary', (1,) + span
    for rule, sep in (('short_summary', SHORT_HANDOFF_SEP), ('fallback_state', STATE_HANDOFF_SEP)):
        if content.startswith(instruction + sep):
            return rule, (i, len(instruction) + len(sep), n)
    if content.startswith(PARSE_ERROR_HEAD) and content.endswith(PARSE_ERROR_TAIL):
        return 'parse_error_none', None
    if content.startswith(OUTPUT_LIMIT_HEAD):
        return 'output_limit_none', None
    if content.startswith(COMPLETION_HEAD) and content.endswith(COMPLETION_TAIL):
        found = _terminal(content, len(COMPLETION_HEAD), n - len(COMPLETION_TAIL))
        if found is not None:
            return 'completion_terminal_output', (i,) + found[1:]
    if content.startswith(WARNINGS_HEAD):
        k = content.find(WARNINGS_SEP, len(WARNINGS_HEAD))
        while k >= 0:
            found = _terminal(content, k + len(WARNINGS_SEP), n)
            if found is not None:
                return 'warnings_terminal_output', (i,) + found[1:]
            k = content.find(WARNINGS_SEP, k + 1)
    found = _terminal(content, 0, n)
    if found is not None:
        return found[0], (i,) + found[1:]
    return 'raw_message', (i, 0, n)


def classify(messages):
    """The Goal inputs of one chat-text request (role/content messages as validated by
    chat_text_protocol.validate_messages). Raises GoalInputsRefused when the first message is
    neither the pinned template nor a summarization sub-call prompt."""
    if messages[0]['role'] != 'user':
        _refuse(SYSTEM_MESSAGE)
    first = messages[0]['content']

    def made(request, requirement_rule, span, rule='subcall_none', observation=None):
        text = messages[span[0]]['content'][span[1]:span[2]]
        if not text.strip():
            _refuse('its task instruction is empty')
        seen = '' if observation is None else messages[observation[0]]['content'][observation[1]:observation[2]]
        return GoalInputs(request, text, requirement_rule, span, seen, rule, observation)
    if len(messages) == 1:
        span = _between(first, QUESTIONS_HEAD, QUESTIONS_SUMMARY_SEP, QUESTIONS_TAIL)
        if span is not None:
            return made('questions_call', 'questions_task', (0,) + span)
        span = _between(first, SHORT_HEAD, SHORT_STATE_SEP, SHORT_TAIL)
        if span is not None:
            return made('short_summary_call', 'short_summary_task', (0,) + span)
    span = instruction_span(first)
    if span is None:
        _refuse(NOT_TERMINUS)
    requirement = (0,) + span
    instruction = first[span[0]:span[1]]
    summary = lambda m: m['role'] == 'user' and m['content'].startswith(SUMMARY_HEAD) \
        and m['content'].endswith(SUMMARY_TAIL)
    if len(messages) > 1 and summary(messages[-1]):
        return made('summary_call', 'template_instruction', requirement)
    if len(messages) >= 4 and messages[-1]['content'].startswith(ANSWERS_HEAD) and summary(messages[-3]):
        return made('answers_call', 'template_instruction', requirement)
    rule, observation = _observation(messages, instruction)
    return made('main', 'template_instruction', requirement, rule, observation)


def context_fields(inputs):
    """The TurnContext fields of a request's Goal: the requirement as the root objective; for
    the main conversation the observation (message source 'tool', except on the first turn,
    'user' as on the SWE path's first turn); for a sub-call none ('user', empty: the context of
    a requirement-only Goal). No tool name or return code (Terminus output has none) and no
    explicit requirement."""
    tool = inputs.rewrite and inputs.observation_rule != 'first_turn'
    return dict(objective=inputs.requirement.strip(), observation=inputs.observation if tool else '',
                message_source='tool' if tool else 'user',
                structured_state={'tool_name': None, 'return_code': None})


def evidence(inputs, goal_source):
    """goals.jsonl ``inputs`` of one turn: the definition, the rules applied and the spans
    (into the request's messages; no text, no hash); ``goal_source`` (GOAL_SOURCES: where the
    engine took this turn's Goal from); ``goal_restarted`` on a summary hand-off (the Goal is
    initialized there, from the instruction and the summary text)."""
    require(goal_source in GOAL_SOURCES and (goal_source == 'maintained') == inputs.rewrite,
            'Goal source differs from the request kind')
    span = lambda value: None if value is None else dict(message=value[0], start=value[1], end=value[2])
    return dict(definition=DEFINITION, status=STATUS, request=inputs.request,
                requirement=dict(rule=inputs.requirement_rule, span=span(inputs.requirement_span),
                                 chars=len(inputs.requirement)),
                observation=dict(rule=inputs.observation_rule, span=span(inputs.observation_span),
                                 chars=len(inputs.observation)),
                rewriter='allowed' if inputs.rewrite else 'skipped', goal_source=goal_source,
                goal_restarted=inputs.handoff)


def row(inputs):
    """The runtime row's ``chat_text.goal_inputs`` (the frontend's classification)."""
    return dict(definition=DEFINITION, request=inputs.request, requirement=inputs.requirement_rule,
                observation=inputs.observation_rule, rewriter='allowed' if inputs.rewrite else 'skipped')
