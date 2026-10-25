"""Generic OpenAI chat text protocol (``chat_text_v1``, serve.py --protocol chat-text).

For chat agents that send their whole text history each turn and parse the reply
themselves, such as Harbor 0.21.0's Terminus 2 on Terminal-Bench 2.0 (JSON
actions, no system message). Thinking stays off.

Wire (``POST /v1/chat/completions``, nonstreaming, one choice):

- ``messages``: an optional system message, then user and assistant text messages;
  the first message after the system is a user message and the last is one. Only
  ``role`` and ``content`` (a string) are read; ``tools``, ``tool_calls`` and
  ``tool`` messages are refused.
- The server decodes greedily, stops on EOS/pad only (no action marker), and
  replies with at most ``max_tokens`` new tokens: the client's ``max_tokens`` (or
  ``max_completion_tokens``) from 1 to the pinned reply bound (``--max-tokens``),
  else the bound itself clamped to the context left. Other sampling fields are
  ignored (recorded in the wire hash only).
- The reply: ``content`` the decoded generated tokens (special tokens skipped),
  ``finish_reason`` ``stop`` on EOS/pad and ``length`` at the bound, ``usage``
  prompt/completion/total tokens.
- A prompt that does not fit (prompt + the requested reply above the context
  limit, or no room for one token) is refused with HTTP 400 and vLLM's text "This
  model's maximum context length is ..." before any session state changes, so
  litellm and Harbor map it to a context-length error.

Token stream of one session (the reference: the pinned chat template with
``enable_thinking=False`` and ``preserve_thinking=True``):

- A fresh or rebuilt session renders the whole conversation through the pinned
  template and appends the generation header ``<|im_start|>assistant\\n<think>\\n\\n
  </think>\\n\\n``. The mandatory prefix (``initial_length``) is the template's
  render of the initial segment: the system message, if any, and the first user
  message.
- A request that repeats the previous request, then the previous reply (compared
  after ``strip()``), then one new user message continues the session: the
  previous reply keeps its exact generated token ids and the boundary
  ``<|im_end|>\\n`` (``\\n`` after EOS/pad), and the new message is
  ``<|im_start|>user\\n{content.strip()}<|im_end|>\\n``, which is what the template
  renders for a user message (``check_template`` verifies it once at startup).
- Any other request (Terminus's summary hand-off, an unwound history, a resent
  request) rebuilds the session from text, as the XML protocol does.

When every reply re-encodes to its own ids and carries no surrounding whitespace,
the continuing stream equals the template's render of the whole conversation
(test_chat_text_protocol, pinned tokenizer).

Gemma 4 31B IT (model profile gemma4-31b-it, gemma-tb): the same protocol with
Gemma's turns (``GEMMA4_CHAT``; the startup record pins its header and template
arguments). The initial segment goes through the pinned template as OpenAI text
parts with ``enable_thinking=False`` (the form in which vLLM hands string content to
that template: a system message gets one more space), a later user message is
``<|turn>user\\n{content.strip()}<turn|>\\n``, the generation header is
``<|turn>model\\n<|channel>thought\\n<channel|>`` (the template's thinking-off
prompt), a reply's boundary is ``<turn|>\\n`` (``\\n`` after a stop id; the stop ids
are 0, 1, 50 and 106) and a past reply keeps the empty thought channel in front of
it: Gemma's template drops that channel when it re-renders a conversation (4 tokens
per past reply), the stream is append-only. A rebuild renders the initial segment
through the template and every later message as the continuing stream holds it (a
run of assistant messages is one model turn, as the template renders it, after the
empty thought channel). When every reply re-encodes to its own ids, each prompt is
the template's render of its messages with that channel inserted after every past
``<|turn>model\\n`` (test_gemma_chat_text, pinned tokenizer).

Sessions (``ChatTextRegistry``, frontend only): the startup config registers
``slots`` sessions of a session pool (``slot_tasks``; at most K open,
``--max-running-sessions``). A client session is keyed by its initial segment and,
when the client sends one, its ``X-Session-ID`` header: the first request with a
new key binds the key to the next unused slot. Terminus sends no header by
default, and its first message (the task prompt, with the container's host name)
differs from trial to trial. Under one header id a request whose initial segment
differs from the header's first one (Terminus's summary "questions" call, sent
without history) is its own session (``kind`` aside), so it never resets the main
session. A slot that can no longer serve its key (failed by an earlier request,
closed for idleness, at the per-slot request bound) is retired and the key moves
to a fresh slot, whose first request renders from text (``rebind``). Slots are
never reused, so ``slots`` bounds the sessions and rebinds of one server lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import heapq
import json
import re
import time

from .contract import require

PROTOCOL = 'chat_text_v1'
RECORD_KEY = 'chat_text'
SOURCE_SCHEMA = 'remote-ra-chat-text-source-v1'   # runtime_renderer.CHAT_SOURCE_SCHEMA
# The generation header with thinking off (the template's add_generation_prompt with
# enable_thinking=False); equal to the XML protocol's.
HEADER_TEXT = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
TEMPLATE_KWARGS = {'enable_thinking': False, 'preserve_thinking': True}
SESSION_HEADER = 'x-session-id'
SESSION_ID = re.compile(r'[\x21-\x7e]{1,256}\Z')        # printable ASCII, no spaces
KEY_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
ROLES = frozenset({'system', 'user', 'assistant'})
MAX_MESSAGES = 4096
# Policies of a chat-text server: goal_query's Goal is SWE-centric (<pr_description>,
# <returncode>, file/symbol/test entities) and undefined for these agents.
POLICIES = ('full', 'random50', 'observation_query', 'random_fill')
# Sessions a chat-text server binds over its lifetime (registered slot tasks): one
# per client session, plus one per rebind and per aside. Terminal-Bench 2.0 has 89
# tasks; its Terminus runs add a few asides and rebinds.
DEFAULT_SLOTS = 512
MAX_SLOTS = 1024
# Idle closes (no close call arrives on the default path): an open session idle for
# idle_timeout_s closes; while a request waits for an open session, the least
# recently used session idle for evict_after_s closes.
DEFAULT_IDLE_TIMEOUT_S = 600
DEFAULT_EVICT_AFTER_S = 30
TIMEOUT_RANGE = (1, 86400)
# A request of a session that has a request queued or in flight waits this long for
# it to end (a client retry after its own timeout) before 409 SESSION_BUSY.
BUSY_WAIT_S = 30
SLOT_INSTANCE = 'chat-text-slot-%04d'
SLOT_STATEMENT = 'chat-text slot %04d, bound to one client session at its first request'
CONTEXT_TEXT = ("This model's maximum context length is %d tokens. However, you requested %d output tokens "
                "and your prompt contains %d input tokens, for a total of %d tokens. Please reduce the length "
                "of the input prompt or the number of requested output tokens.")


class ChatTextError(RuntimeError):
    """A refused chat-text request: HTTP ``status`` with an OpenAI/vLLM error body."""

    def __init__(self, kind, message, status=400):
        super().__init__(message)
        self.kind, self.status = kind, status

    def body(self):
        names = {400: 'BadRequestError', 404: 'NotFoundError', 409: 'ConflictError', 500: 'InternalServerError'}
        return {'error': {'message': str(self), 'type': names.get(self.status, 'BadRequestError'),
                          'param': self.kind, 'code': self.status}}


def fail(kind, message, status=400):
    raise ChatTextError(kind, message, status)


def _int(value, low=None, high=None):
    return type(value) is int and (low is None or value >= low) and (high is None or value <= high)


# ---- startup ---------------------------------------------------------------------------------

def slot_tasks(count):
    """The session pool's registered tasks of a chat-text server: one per slot."""
    require(_int(count, 1, MAX_SLOTS), 'chat-text slots must be 1..%d' % MAX_SLOTS)
    return [dict(instance_id=SLOT_INSTANCE % k, problem_statement=SLOT_STATEMENT % k) for k in range(count)]


def record(*, slots=DEFAULT_SLOTS, idle_timeout_s=DEFAULT_IDLE_TIMEOUT_S, evict_after_s=DEFAULT_EVICT_AFTER_S,
           profile=None):
    """The startup config's ``chat_text`` record (serve.py --protocol chat-text).

    ``profile``: the startup config's model profile (None: Qwen, the record as it was;
    'gemma4-31b-it': Gemma's generation header and template arguments)."""
    low, high = TIMEOUT_RANGE
    grammar = grammar_of(profile)
    require(_int(slots, 1, MAX_SLOTS), '--chat-text-slots must be 1..%d' % MAX_SLOTS)
    require(_int(idle_timeout_s, low, high), '--chat-text-idle-timeout must be %d..%d seconds' % TIMEOUT_RANGE)
    require(_int(evict_after_s, low, idle_timeout_s),
            '--chat-text-evict-after must be %d seconds up to --chat-text-idle-timeout' % low)
    return dict(slots=slots, idle_timeout_s=idle_timeout_s, evict_after_s=evict_after_s,
                header_text=grammar.header, template_kwargs=dict(grammar.template_kwargs),
                session_header='X-Session-ID', binding='initial_segment', busy_wait_s=BUSY_WAIT_S)


def check_config(config):
    """Startup only (remote_ra_http.config.load): the chat-text record, the slot tasks,
    the session pool and the policy of a ``chat_text_v1`` config. Returns the record.
    The record's turn pins are those of the config's model profile (none: Qwen)."""
    require(config.get('protocol') == PROTOCOL, 'chat-text record on another protocol')
    value = config.get(RECORD_KEY)
    profile = config.get('model_profile')
    require(profile is None or profile in GRAMMARS, 'chat-text has no turns for model profile %r' % (profile,))
    require(type(value) is dict and set(value) == set(record()),
            'chat-text startup record {%s} required' % ', '.join(sorted(record())))
    require(value == record(slots=value['slots'], idle_timeout_s=value['idle_timeout_s'],
                            evict_after_s=value['evict_after_s'], profile=profile),
            'chat-text record differs from the protocol pins')
    require(config.get('tasks') == slot_tasks(value['slots']) and config.get('task') == slot_tasks(1)[0],
            'chat-text server registers exactly its slot tasks')
    require(_int(config.get('max_running_sessions'), 1),
            'chat-text needs a session pool (--max-running-sessions K)')
    require(config.get('policy') in POLICIES,
            'chat-text serves %s (goal_query\'s Goal is SWE-centric)' % ', '.join(POLICIES))
    return value


def options(config):
    """What this server runs, for GET /v1/chat-text/info."""
    from .context_limits import max_model_len, max_tokens
    value = dict(config[RECORD_KEY], protocol=PROTOCOL, policy=config['policy'], page_tokens=config['page_tokens'],
                 max_tokens=max_tokens(config), max_model_len=max_model_len(config),
                 max_running_sessions=config['max_running_sessions'])
    contract = config.get('selection_contract')
    if contract is not None:
        value.update(active_token_budget=contract['active_token_budget'], active_ratio=contract['active_ratio'],
                     switch_threshold_tokens=contract['switch_threshold_tokens'])
    if config.get('model_profile') is not None:
        # Gemma (gemma-tb): the model profile; a Qwen server's options are unchanged.
        value.update(model_profile=config['model_profile'])
    return value


def user_piece(content):
    """One later user message, as the pinned template renders it (content trimmed)."""
    return '<|im_start|>user\n' + content.strip() + '<|im_end|>\n'


def boundary_text(generated, stop_ids):
    """The reply's closing delimiter: EOS/pad already closed the turn."""
    return '\n' if generated and generated[-1] in stop_ids else '<|im_end|>\n'


def check_template(render_template, template):
    """Startup only: the pinned template renders a user message as ``user_piece``, an
    initial user segment without any default system text, a history reply with the
    empty thought block (preserve_thinking) and the thinking-off generation header
    HEADER_TEXT. Returns HEADER_TEXT."""
    kwargs = dict(TEMPLATE_KWARGS)
    probe = ' chat-text probe {"keystrokes": "ls -la\\n", "duration": 0.1} '
    first = render_template(template, [dict(role='user', content=probe)], dict(kwargs, add_generation_prompt=False))
    require(first == user_piece(probe), 'pinned template renders a lone user message differently from user_piece')
    turns = [dict(role='user', content=probe), dict(role='assistant', content='{"a": 1}'),
             dict(role='user', content=' next \n')]
    whole = render_template(template, turns, dict(kwargs, add_generation_prompt=True))
    expected = (user_piece(probe) + HEADER_TEXT + '{"a": 1}' + '<|im_end|>\n' + user_piece(' next \n')
                + HEADER_TEXT)
    require(whole == expected, 'pinned template history/header render differs from the chat-text stream')
    system = render_template(template, [dict(role='system', content=' s '), dict(role='user', content='u')],
                             dict(kwargs, add_generation_prompt=False))
    require(system == '<|im_start|>system\ns<|im_end|>\n' + user_piece('u'),
            'pinned template system render differs')
    return HEADER_TEXT


# ---- the turns of each model profile ---------------------------------------------------------------

class QwenChat:
    """Qwen3.6-27B's chat-text turns: the functions above (every Qwen server, unchanged).
    A fresh or rebuilt session renders the whole conversation through the template."""
    profile = None
    header = HEADER_TEXT
    template_kwargs = TEMPLATE_KWARGS
    whole_render = True

    @staticmethod
    def user_piece(content):
        return user_piece(content)

    @staticmethod
    def boundary_text(generated, stop_ids):
        return boundary_text(generated, stop_ids)


GEMMA4_PROFILE = 'gemma4-31b-it'          # model_profile.GEMMA4
GEMMA4_HEADER_TEXT = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA4_TEMPLATE_KWARGS = {'enable_thinking': False}
GEMMA4_MODEL_TURN = '<|turn>model\n'      # the template's model turn (no thought channel)
GEMMA4_END_OF_TURN = '<turn|>\n'
GEMMA4_BOS = '<bos>'


class Gemma4Chat:
    """Gemma 4 31B IT's chat-text turns (model profile gemma4-31b-it, gemma-tb).

    Every piece is what the pinned template renders for that turn (thinking off, content
    as OpenAI text parts), except that a model turn keeps the empty thought channel of
    the generation header: the template drops it from history (4 tokens per past reply),
    the stream is append-only. A fresh or rebuilt session renders only the initial
    segment through the template; every later message is a piece below (a run of
    assistant messages is one model turn, as the template renders it)."""
    profile = GEMMA4_PROFILE
    header = GEMMA4_HEADER_TEXT
    template_kwargs = GEMMA4_TEMPLATE_KWARGS
    whole_render = False

    @staticmethod
    def user_piece(content):
        """A later user message: the template trims it (Jinja ``trim`` is ``str.strip``)."""
        return '<|turn>user\n' + content.strip() + GEMMA4_END_OF_TURN

    @staticmethod
    def model_piece(contents):
        """A run of assistant messages (one model turn) after the kept empty thought channel;
        the template's ``strip_thinking`` of each content."""
        from .mswea_protocol import strip_thinking
        return GEMMA4_HEADER_TEXT + ''.join(strip_thinking(c) for c in contents) + GEMMA4_END_OF_TURN

    def history_pieces(self, messages):
        """The pieces of the messages after the initial segment (user and assistant)."""
        pieces, run = [], []
        for m in messages:
            require(m['role'] in ('user', 'assistant'), 'chat-text history holds user and assistant messages')
            if m['role'] == 'assistant':
                run.append(m['content'])
                continue
            if run:
                pieces.append(self.model_piece(run))
                run = []
            pieces.append(self.user_piece(m['content']))
        if run:
            pieces.append(self.model_piece(run))
        return pieces

    @staticmethod
    def boundary_text(generated, stop_ids):
        """The reply's closing delimiter: a stop id (<turn|> 106, or 0, 1, 50) already ended it."""
        return '\n' if generated and generated[-1] in stop_ids else GEMMA4_END_OF_TURN

    def check_template(self, render):
        """Startup only: ``render(messages, add_generation_prompt)`` is the pinned template's
        text with this profile's arguments and the content as OpenAI text parts. It renders
        <bos> and a lone user message as ``user_piece``, a past reply as ``model_piece``
        without the thought channel (a run of replies as one model turn), the thinking-off
        generation header, and a system message as text parts (one more space). Returns
        the header text."""
        probe = ' chat-text probe {"keystrokes": "ls -la\\n", "duration": 0.1} '
        first = render([dict(role='user', content=probe)], False)
        require(first == GEMMA4_BOS + self.user_piece(probe),
                'pinned Gemma template renders a lone user message differently from user_piece')
        reply = ' {"a": 1} '
        turns = [dict(role='user', content=probe), dict(role='assistant', content=reply),
                 dict(role='user', content=' next \n')]
        whole = render(turns, True)
        model = self.model_piece([reply])
        require(model.startswith(GEMMA4_HEADER_TEXT) and self.header.startswith(GEMMA4_MODEL_TURN),
                'Gemma model turn and header differ')
        expected = (GEMMA4_BOS + self.user_piece(probe) + GEMMA4_MODEL_TURN + model[len(GEMMA4_HEADER_TEXT):]
                    + self.user_piece(' next \n') + self.header)
        require(whole == expected, 'pinned Gemma template history/header render differs from the chat-text stream')
        run = [dict(role='user', content='u'), dict(role='assistant', content=' x '),
               dict(role='assistant', content='y\n'), dict(role='user', content='v')]
        merged = self.model_piece(['x', 'y'])
        require(render(run, True) == (GEMMA4_BOS + self.user_piece('u') + GEMMA4_MODEL_TURN
                                      + merged[len(GEMMA4_HEADER_TEXT):] + self.user_piece('v') + self.header),
                'pinned Gemma template renders a run of replies differently from model_piece')
        system = render([dict(role='system', content=' s '), dict(role='user', content='u')], False)
        require(system == GEMMA4_BOS + '<|turn>system\ns <turn|>\n' + self.user_piece('u'),
                'pinned Gemma template system render (text parts) differs')
        return self.header


QWEN_CHAT = QwenChat()
GEMMA4_CHAT = Gemma4Chat()
GRAMMARS = {None: QWEN_CHAT, GEMMA4_PROFILE: GEMMA4_CHAT}


def grammar_of(profile):
    """The chat-text turns of a startup config's model profile (None: Qwen)."""
    require(profile in GRAMMARS, 'chat-text has no turns for model profile %r' % (profile,))
    return GRAMMARS[profile]


# ---- messages ------------------------------------------------------------------------------------

def initial_count(messages):
    """Messages in the initial segment: the system message, if any, and the first user."""
    return 2 if messages and messages[0].get('role') == 'system' else 1


def validate_messages(messages):
    """The request shape; returns role/content copies (other message keys are ignored)."""
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        fail('BAD_MESSAGES', 'chat-text messages must be a list of 1..%d messages' % MAX_MESSAGES)
    plain = []
    for index, m in enumerate(messages):
        if type(m) is not dict:
            fail('BAD_MESSAGES', 'chat-text messages must be objects')
        role, content = m.get('role'), m.get('content')
        if role == 'tool' or m.get('tool_calls') or m.get('tool_call_id') is not None:
            fail('TOOLS_UNSUPPORTED', 'chat-text serves text messages only: no tool calls or tool messages')
        if role not in ROLES or (role == 'system' and index):
            fail('BAD_MESSAGES', 'chat-text messages are an optional system message, then user and assistant')
        if type(content) is not str:
            fail('BAD_MESSAGES', 'chat-text message content must be a string')
        plain.append(dict(role=role, content=content))
    first = initial_count(plain) - 1
    if len(plain) <= first or plain[first]['role'] != 'user':
        fail('BAD_MESSAGES', 'the first message after the optional system message must be a user message')
    if plain[-1]['role'] != 'user':
        fail('BAD_MESSAGES', 'a chat-text request ends with a user message')
    return plain


# Body keys that would name server authority; a chat-text body may carry session_id
# (litellm's extra_body copy of the X-Session-ID header).
FORBIDDEN = frozenset({'policy', 'session', 'epoch', 'issue', 'goal', 'query', 'ranking', 'ranked_pages',
                       'selected_pages', 'kv_transfer_params', 'policy_runtime', 'extra_args'})


def check_wire(body):
    """The body's shape (FrontendEvidence.begin via config.validate_wire): nonstreaming,
    one choice, no server authority, no tools, chat-text messages. Returns the messages."""
    require(type(body) is dict and body.get('stream', False) is False and body.get('n', 1) in (None, 1),
            'nonstreaming single completion required')
    require(not FORBIDDEN.intersection(body), 'client selector authority forbidden')
    require(not body.get('tools') and body.get('tool_choice') in (None, 'none', 'auto'),
            'chat-text protocol forbids tools')
    validate_messages(body.get('messages'))
    return body['messages']


def session_header(value):
    """The client's X-Session-ID (header or extra_body ``session_id``), or None."""
    if value is None:
        return None
    if type(value) is not str or not SESSION_ID.match(value):
        fail('BAD_SESSION_ID', 'X-Session-ID must be 1-256 printable ASCII characters without spaces')
    return value


def key_of(messages, header):
    """The routing key of a request: its header id and its initial segment (strings
    as sent; Python's dict hashing, nothing hashed by the protocol)."""
    return (header, tuple((m['role'], m['content']) for m in messages[:initial_count(messages)]))


def key_sha256(key):
    """Evidence name of a key (computed once, when it binds): SHA-256 of the canonical
    JSON of its initial segment (role/content), as a harness can recompute it from
    the first request; the header id is recorded next to it."""
    segment = [dict(role=role, content=content) for role, content in key[1]]
    raw = json.dumps(segment, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256(raw).hexdigest()


def requested_max_tokens(request):
    """The client's reply bound (max_completion_tokens before max_tokens), or None."""
    for name in ('max_completion_tokens', 'max_tokens'):
        value = getattr(request, name, None)
        if value is not None:
            return value
    return None


def effective_max_tokens(requested, prompt_length, bound, max_model_len):
    """The reply bound of one request; refuses one that does not fit the context.

    ``requested`` (the client's, or None): from 1 to ``bound`` (the pinned
    --max-tokens) and prompt + requested within the context; None takes the bound,
    clamped to the context left. vLLM's own refusal text either way."""
    if requested is not None:
        if not _int(requested, 1, bound):
            fail('MAX_TOKENS', 'max_tokens must be 1..%d (the server reply bound, --max-tokens)' % bound)
        if prompt_length + requested > max_model_len:
            fail('CONTEXT_LENGTH', CONTEXT_TEXT % (max_model_len, requested, prompt_length,
                                                   prompt_length + requested))
        return requested
    left = max_model_len - prompt_length
    if left < 1:
        fail('CONTEXT_LENGTH', CONTEXT_TEXT % (max_model_len, 1, prompt_length, prompt_length + 1))
    return min(bound, left)


def check_sampling(params, stop_ids, bound, prompt_length, max_model_len):
    """EngineCore's check of a chat-text request (private_endpoint.render_claim): greedy,
    EOS/pad stops, 1 <= max_tokens <= the pinned bound within the context, no other
    control; the route set exactly these (chat_text_route.prepare_request)."""
    value = lambda name, default=None: getattr(params, name, default)
    stops = set(value('stop_token_ids') or ()) | {value('eos_token_id')}
    stops.discard(None)
    limit = value('max_tokens')
    wrong = [name for name, ok in (
        ('n', value('n', 1) == 1), ('temperature', value('temperature') == 0),
        ('max_tokens', _int(limit, 1, bound) and prompt_length + limit <= max_model_len),
        ('min_tokens', value('min_tokens', 0) == 0), ('stop', not value('stop')),
        ('ignore_eos', not value('ignore_eos', False)), ('stop_token_ids', stops == set(stop_ids)),
        ('repetition_penalty', value('repetition_penalty', 1.0) == 1.0),
        ('presence_penalty', value('presence_penalty', 0.0) == 0.0),
        ('frequency_penalty', value('frequency_penalty', 0.0) == 0.0),
        ('logit_bias', not value('logit_bias')), ('allowed_token_ids', value('allowed_token_ids') is None),
        ('bad_words', not value('bad_words')), ('structured_outputs', value('structured_outputs') is None),
        ('logits_processors', not value('logits_processors'))) if not ok]
    require(not wrong, 'chat-text requires greedy generation within the reply bound; differs: '
            + ', '.join('%s=%r' % (name, value(name)) for name in wrong))
    return True


def finish_reason(generated, stop_ids, max_tokens):
    """'stop' on an EOS/pad token, 'length' when the reply used its whole bound."""
    if generated and generated[-1] in stop_ids:
        return 'stop'
    require(len(generated) == max_tokens, 'reply ended without a stop token before its bound')
    return 'length'


def goal_inputs(messages, first):
    """TurnContext fields of one turn (the selection's evidence-only Goal): the root
    objective is the first user message; the observation is the new user message."""
    observation = messages[-1]['content']
    return dict(objective=messages[initial_count(messages) - 1]['content'].strip(), observation=observation,
                message_source='user' if first else 'tool', explicit_requirement=None,
                structured_state={'tool_name': None, 'return_code': None})


# ---- the token transcript ------------------------------------------------------------------------

class ChatTextTranscript:
    """Server-owned token stream of one chat-text session (frontend and engine).

    ``render_chat(messages)`` renders role/content messages through the pinned
    template (TEMPLATE_KWARGS, no generation prompt) to token ids; ``encode``
    tokenizes text without adding special tokens; ``decode`` skips special tokens.
    Attribute names follow mswea_protocol.Transcript where the controller
    checkpoints them (controller.TRANSCRIPT_STATE). ``plan`` computes a request's
    tokens without changing anything (the route's context check); ``render`` then
    reuses that plan.

    ``grammar``: the turns, ``QWEN_CHAT`` (the default) or ``GEMMA4_CHAT``; with Gemma's
    ``render_chat`` renders the initial segment only (OpenAI text parts)."""
    protocol = PROTOCOL
    allow_rebuild = True

    def __init__(self, *, encode, render_chat, decode, stop_ids, grammar=QWEN_CHAT):
        require(callable(encode) and callable(render_chat) and callable(decode)
                and type(stop_ids) is frozenset and stop_ids, 'chat-text transcript callbacks required')
        require(grammar is QWEN_CHAT or grammar is GEMMA4_CHAT, 'known chat-text turns required')
        self.encode, self.render_chat, self.decode, self.stop_ids = encode, render_chat, decode, stop_ids
        self.grammar = grammar
        self.header = tuple(self.encode(grammar.header))
        require(self.header, 'chat-text generation header is empty')
        self.sessions = 0        # sessions opened, including rebuilds
        self._plan = None        # (state, messages, plan) of the last plan
        self._reset()

    def _reset(self):
        self.history = None      # messages of the last committed request
        self.turns = []          # dict(observation, text, ids) per committed turn
        self.stream = ()         # every token of this session so far
        self.initial_length = 0  # template tokens of the initial segment (mandatory prefix)
        self.pending = None

    @property
    def turn(self):
        return len(self.turns)

    def extends(self, messages):
        history = self.history
        return (history is not None and len(messages) == len(history) + 2
                and messages[:len(history)] == history
                and messages[-2]['role'] == 'assistant'
                and messages[-2]['content'].strip() == self.turns[-1]['text'].strip())

    def _state(self):
        return (self.turn, len(self.stream), self.sessions, id(self.history))

    def plan(self, messages):
        """The request's prompt (tokens, stable end, observation span, rebuild, mandatory
        prefix) on this transcript, without changing it."""
        messages = validate_messages(messages)
        cached = self._plan
        if cached is not None and cached[0] == self._state() and cached[1] == messages:
            return cached[2]
        rebuild = self.turn > 0 and not self.extends(messages)
        if self.turn == 0 or rebuild:
            count = initial_count(messages)
            head = tuple(self.render_chat(messages[:count]))
            require(head and all(type(t) is int and t >= 0 for t in head), 'initial template render is empty')
            if len(messages) == count:
                complete = head
            elif self.grammar.whole_render:
                complete = tuple(self.render_chat(messages))
            else:
                # Gemma: the later messages as the continuing stream holds them (each piece
                # starts with a special token, so the pieces encode on their own).
                complete = head + tuple(t for piece in self.grammar.history_pieces(messages[count:])
                                        for t in self.encode(piece))
            require(complete[:len(head)] == head, 'template render of the initial segment is not a token prefix')
            tokens, stable, span, initial = complete + self.header, len(complete), None, len(head)
        else:
            observed = tuple(self.encode(self.grammar.user_piece(messages[-1]['content'])))
            stable = len(self.stream)
            tokens, span, initial = self.stream + observed + self.header, (stable, stable + len(observed)), \
                self.initial_length
        value = dict(messages=messages, tokens=tokens, stable=stable, span=span, rebuild=rebuild,
                     initial_length=initial, observation=messages[-1]['content'])
        self._plan = (self._state(), messages, value)
        return value

    def prompt_length(self, messages):
        return len(self.plan(messages)['tokens'])

    def render(self, messages):
        """Prompt tokens, stable prefix end and observation span for the next request."""
        require(self.pending is None, 'previous chat-text turn neither committed nor abandoned')
        plan = self.plan(messages)
        if plan['rebuild']:
            self._reset()
        if plan['rebuild'] or self.turn == 0:
            self.initial_length = plan['initial_length']
        self.pending = dict(messages=plan['messages'], tokens=plan['tokens'], observation=plan['observation'],
                            rebuild=plan['rebuild'])
        self._plan = None
        return plan['tokens'], plan['stable'], plan['span']

    def abandon(self):
        self.pending = None

    def release(self):
        """The session closed for good: drop its stream, history and any pending render."""
        self._reset()
        self._plan = None

    def commit(self, prompt_tokens, generated):
        """Append the generated tokens and the boundary; return the reply."""
        pending = self.pending
        require(pending is not None and tuple(prompt_tokens) == pending['tokens'],
                'committed prompt differs from the rendered chat-text turn')
        generated = tuple(generated)
        require(generated and all(type(t) is int and t >= 0 for t in generated), 'generated token IDs required')
        boundary = tuple(self.encode(self.grammar.boundary_text(generated, self.stop_ids)))
        text = self.decode(list(generated))
        if not self.turns:
            self.sessions += 1
        header_start = len(pending['tokens']) - len(self.header)
        ids = pending['tokens'][header_start:] + generated + boundary
        self.turns.append(dict(observation=pending['observation'], text=text, ids=ids))
        self.stream = pending['tokens'] + generated + boundary
        self.history = pending['messages']
        self.pending = None
        return dict(text=text, boundary=boundary, generated=generated, rebuild=pending['rebuild'])

    def previous_action(self):
        """The decoded reply of this session's prior turn."""
        return self.turns[-1]['text'] if self.turns else None


def chat(transcript):
    """Whether ``transcript`` is a chat-text transcript."""
    return getattr(transcript, 'protocol', None) == PROTOCOL


# ---- client sessions -> slots (frontend) ---------------------------------------------------------

@dataclass
class ClientSession:
    """One client session (a routing key) and the slots it used, newest last."""
    name: str
    key: tuple
    key_sha256: str
    header: str | None
    kind: str                       # 'session' (no header), 'main' or 'aside' (under a header id)
    slots: list = field(default_factory=list)
    requests: int = 0               # requests served (completed) for this key
    attempts: int = 0               # requests admitted to a slot (completed or failed)
    closed: bool = False
    retired: dict = field(default_factory=dict)   # slot -> why it stopped serving this key

    @property
    def slot(self):
        return self.slots[-1] if self.slots else None


class ChatTextRegistry:
    """Frontend map of client sessions to the pool's slots, and the slots' idle clock.

    ``target`` names the slot a request would use, changing nothing; ``bind`` makes
    it so. A slot serves one key; a key moves to a fresh slot (``rebind``) when its
    slot was failed by a request, closed by the server (idle, evicted) or reached the
    per-slot request bound. A slot never binds twice: the lifetime is ``slots``
    sessions and rebinds. Header ids closed by the client are never served again."""

    def __init__(self, slots, clock=time.monotonic):
        require(_int(slots, 1, MAX_SLOTS), 'chat-text slots must be 1..%d' % MAX_SLOTS)
        self.slots, self.clock = slots, clock
        self.sessions = {}           # key -> ClientSession
        self.by_slot = {}            # slot -> ClientSession
        self.by_header = {}          # header id -> [ClientSession] (the first is 'main')
        self.closed_headers = {}     # header id -> the first close answer
        self.closed_keys = {}        # key_sha256 -> the first close answer (POST close by key)
        self.unused = list(range(slots))   # a heap: slots never bound
        self.last_active = {}        # slot -> monotonic seconds of its last request end
        self.closing = {}            # slot -> why the server closes it (set by the reaper)
        self.ordinal = 0

    def usable(self, live, slot, max_turns):
        """Why ``slot`` cannot serve its key's next request, or None."""
        session = live.sessions[slot]
        if session.ended:
            return self.closing.get(slot, 'closed')
        if session.closed:
            return 'failed'
        if session.turn >= max_turns:
            return 'turn_bound'
        return None

    def target(self, key, live, max_turns):
        """(client session or None, slot, rebind reason or None) of a request; no change."""
        header = key[0]
        if header is not None and header in self.closed_headers:
            fail('SESSION_CLOSED', 'session %s was closed by the client; it is never served again' % header)
        client = self.sessions.get(key)
        if client is not None and client.closed:
            fail('SESSION_CLOSED', 'session %s was closed by the client; it is never served again' % client.name)
        reason = None
        if client is not None:
            reason = self.usable(live, client.slot, max_turns)
            if reason is None:
                return client, client.slot, None
        if not self.unused:
            fail('SESSION_LIMIT', 'this server binds at most %d sessions over its lifetime (--chat-text-slots); '
                 'every slot has been used' % self.slots)
        return client, self.unused[0], reason

    def bind(self, key, slot, reason):
        """Bind ``key`` to ``slot`` (the slot ``target`` named); returns (client, event)."""
        client = self.sessions.get(key)
        if client is not None and client.slot == slot:
            return client, None
        require(self.unused and self.unused[0] == slot, 'chat-text slot is not the next unused one')
        heapq.heappop(self.unused)
        if client is None:
            header = key[0]
            kind = 'session' if header is None else ('aside' if self.by_header.get(header) else 'main')
            self.ordinal += 1
            client = ClientSession('cs-%05d' % self.ordinal, key, key_sha256(key), header, kind)
            self.sessions[key] = client
            if header is not None:
                self.by_header.setdefault(header, []).append(client)
        else:
            client.retired[client.slot] = reason
        previous = client.slot
        client.slots.append(slot)
        self.by_slot[slot] = client
        event = dict(event='bind' if previous is None else 'rebind', slot=slot, previous_slot=previous,
                     reason=reason)
        return client, event

    def touch(self, slot):
        self.last_active[slot] = self.clock()

    def idle_seconds(self, slot, now=None):
        last = self.last_active.get(slot)
        return None if last is None else (self.clock() if now is None else now) - last

    def clients_of_header(self, header):
        return list(self.by_header.get(header, ()))

    def info(self):
        """Counts for GET /v1/chat-text/info."""
        return dict(slots=self.slots, used_slots=self.slots - len(self.unused),
                    client_sessions=len(self.sessions), closed_session_ids=len(self.closed_headers),
                    rebinds=sum(len(c.slots) - 1 for c in self.sessions.values()))
