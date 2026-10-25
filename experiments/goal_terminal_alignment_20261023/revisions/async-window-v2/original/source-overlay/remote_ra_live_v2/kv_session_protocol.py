"""MemoryAgentBench token-level KV-session protocol (``kv_session_v1``, serve.py --protocol kv-session).

The wire format is the tebench KV-session client's (method ``dram_backed_kv``):

- ``GET /v1/kv-session/info`` returns ``{method, budget, options}``: this server's
  method name, its attention budget in tokens (null without a token budget) and
  the options it really runs (``info``), nothing else.
- ``POST /v1/kv-session/complete`` carries ``{method, session_id, input_ids,
  header_tokens, prefix_tokens, budget, max_tokens, temperature, top_p, top_k,
  repetition_penalty, seed, messages, boundary_tokens}`` and returns ``{text,
  output_ids, prompt_tokens, completion_tokens, cached_tokens, finish_reason}``
  (plus fields named apart from those), or 400 ``{"error", "type"}``.
- ``POST /v1/kv-session/close`` ``{session_id}`` returns 200 ``{"closed": true}``,
  also for an id this server never saw or already closed.

The client renders the chat template and parses replies; the server never
renders. A session's token transcript is the stream of every prompt token it was
sent and every token it generated: ``input_ids`` is the whole first prompt at turn
0, and after that only the new tail: the reply's closing delimiter (with the
reply's last sampled token restored in front when it was not an EOS token), the
new messages, and the next generation header. Hence ``input_ids[0]`` always
repeats the previous reply's last output id, the one token whose KV no forward has
computed yet. That is the stage-B turn geometry's C (``remote_ra_native_growing.
contract.turn_geometry``), so a turn maps onto phase 1 / phase 2 as::

    T = stream + input_ids[1:]       N = len(T)      C = len(stream) - 1
    o = C + boundary_tokens          e = N - header_tokens
    turn 0:  T = input_ids,  o = e = prefix_tokens  (the mandatory prefix)

Phase 1 computes [C, e) (the boundary and the new messages), selection ranks the
sealed Pages, phase 2 computes the header [e, N) and decodes. ``C < o < e < N``
holds for every request the client builds (boundary >= 1, at least one message
token, header >= 1).

Checks are structural and O(new tokens): the header suffix against the pinned
header ids, ``input_ids[boundary] == <|im_start|>``, ids below the vocabulary,
the continuation token, the unchanged prefix. Nothing is hashed per request; the
header ids are pinned once at startup from the pinned template (``check_header``).

``messages`` (the client's full normalized history) never makes tokens. The
frontend keeps only the messages new since the previous request and the session's
root objective, for the Goal of goal_query (``goal_inputs``): in MemAgent the
question arrives on the last request only, as a user message after the last tool
message, and becomes the Goal's ``explicit_requirement``.

Sessions: the startup config registers ``slots`` sessions of a session pool
(``slot_tasks``; at most K open at a time, ``--max-running-sessions``). A client
``session_id`` binds to the next unused slot at its first (turn-0) request
(``KvSessionRegistry``); a closed id is never served again and a slot is never
reused. Capacity (which sessions keep GPU pages) is the pool's and the engine's
business, not this protocol's.

Gemma 4 31B IT (model profile gemma4-31b-it, gemma-mab): the same wire format and
geometry with Gemma's turn grammar (record ``grammar`` ``gemma4``), thinking off only.
The pinned template's generation header (``GEMMA4_HEADER``, 7 tokens) follows a user
turn; after tool responses the model turn continues and the template adds no header,
so the client derives ``header_tokens`` 0. Such a tail ends with ``<tool_response|>``,
which phase 2 computes (e = N - 1).
The new messages open with ``<|turn>`` at ``input_ids[boundary]`` (a new turn) or
right after ``<|tool_response>`` at ``input_ids[boundary - 1]`` (tool responses): a
reply that stops on ``<|tool_response>`` (id 50, one of the stop ids {0, 1, 50, 106})
is that opener itself, so its boundary is 1. The client restores the last sample in
front whenever it is not the tail's first token (``<|tool_response>`` after a tool
call, ``<turn|>`` after text), as Qwen's client does for every id but ``<|im_end|>``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .contract import require

PROTOCOL = 'kv_session_v1'
METHOD = 'dram_backed_kv'
RECORD_KEY = 'kv_session'
SOURCE_SCHEMA = 'remote-ra-kv-session-source-v1'
# The client recognizes exactly this error text as the context limit.
CONTEXT_LIMIT = 'Logical RoPE context limit reached'
THINKING = ('on', 'off')
# The generation header the client's template appends (enable_thinking on/off);
# check_header verifies it against the pinned template once, at startup.
HEADERS = {'on': '<|im_start|>assistant\n<think>\n',
           'off': '<|im_start|>assistant\n<think>\n\n</think>\n\n'}
OBSERVATION_START = '<|im_start|>'
# Gemma 4 (gemma-mab): the grammar record value, thinking off only, its header after a
# user turn, the turn opener and the tool-response delimiters.
GEMMA4 = 'gemma4'
GEMMA4_THINKING = 'off'
GEMMA4_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA4_TURN_START = '<|turn>'
GEMMA4_TOOL_OPEN = '<|tool_response>'
GEMMA4_TOOL_CLOSE = '<tool_response|>'
COMPLETE_KEYS = frozenset({'method', 'session_id', 'input_ids', 'header_tokens', 'prefix_tokens', 'budget',
                           'max_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'seed',
                           'messages', 'boundary_tokens'})
CLOSE_KEYS = frozenset({'session_id'})
SESSION_ID = re.compile(r'[0-9a-f]{32}\Z')
ROLES = frozenset({'system', 'user', 'assistant', 'tool'})
MAX_MESSAGES = 4096
# Sessions a kv-session server binds over its lifetime: the session pool's
# registered tasks (sessions.MAX_TASKS). One MemoryAgentBench allocation opens
# about 50 (warm-up, validation, 32 cases).
DEFAULT_SLOTS = 64
MAX_SLOTS = 64
SLOT_INSTANCE = 'kv-session-slot-%03d'
SLOT_STATEMENT = 'kv-session slot %03d, bound to one client session at its first request'
# The engine's source (input_ids plus the new messages) has the source cap of every
# protocol, runtime_renderer.MAX_SOURCE_BYTES (1 MiB; doubled by a 262144-token
# server's context_limits.pin_request_caps), so a frame stays within its own cap.
RETURN_CODE = re.compile(r'<returncode>(-?\d+)</returncode>')


class KvSessionError(RuntimeError):
    """A refused kv-session request: HTTP ``status`` with body ``{"error", "type"}``."""

    def __init__(self, kind, message, status=400):
        super().__init__(message)
        self.kind, self.status = kind, status

    def body(self):
        return {'error': str(self), 'type': self.kind}


def fail(kind, message, status=400):
    raise KvSessionError(kind, message, status)


def _int(value, low=None):
    return type(value) is int and (low is None or value >= low)


def _number(value):
    return type(value) in (int, float)


# ---- startup ---------------------------------------------------------------------------------

def slot_tasks(count):
    """The session pool's registered tasks of a kv-session server: one per slot, in
    slot order (no statement contains another, so sessions.validate_tasks accepts them)."""
    require(_int(count) and 1 <= count <= MAX_SLOTS, 'kv-session slots must be 1..%d' % MAX_SLOTS)
    return [dict(instance_id=SLOT_INSTANCE % k, problem_statement=SLOT_STATEMENT % k) for k in range(count)]


def budget_of(contract):
    """The attention budget in tokens this server applies: the selection contract's
    token budget, or None (Full/Random50, or a ratio budget)."""
    if contract is None:
        return None
    return contract.get('active_token_budget')


def record(*, contract, thinking='on', slots=DEFAULT_SLOTS, grammar=None):
    """The startup config's ``kv_session`` record (serve.py --protocol kv-session).

    ``grammar`` ``gemma4`` (the Gemma 4 profile, gemma-mab): thinking off only, Gemma's
    header, and the record names the grammar; Qwen's record has no such key."""
    require(thinking in THINKING, '--kv-session-thinking must be on or off')
    require(_int(slots) and 1 <= slots <= MAX_SLOTS, '--kv-session-slots must be 1..%d' % MAX_SLOTS)
    if grammar is None:
        return dict(method=METHOD, budget=budget_of(contract), thinking=thinking, header_text=HEADERS[thinking],
                    slots=slots)
    require(grammar == GEMMA4, 'unknown kv-session grammar %r' % (grammar,))
    require(thinking == GEMMA4_THINKING, 'the Gemma 4 kv-session serves --kv-session-thinking off only '
            '(thinking off, as decided for MemoryAgentBench)')
    return dict(method=METHOD, budget=budget_of(contract), thinking=thinking, header_text=GEMMA4_HEADER,
                slots=slots, grammar=grammar)


def check_config(config):
    """Startup only (remote_ra_http.config.load): the kv-session record, the slot
    tasks and the session pool of a ``kv_session_v1`` config. Returns the record.
    A Gemma 4 config (``model_profile``) pins the Gemma grammar, every other none."""
    require(config.get('protocol') == PROTOCOL, 'kv-session record on another protocol')
    value = config.get(RECORD_KEY)
    gemma4 = config.get('model_profile') is not None
    keys = {'method', 'budget', 'thinking', 'header_text', 'slots'} | ({'grammar'} if gemma4 else set())
    require(type(value) is dict and set(value) == keys,
            'kv-session startup record {%s} required' % ', '.join(sorted(keys)))
    require(value == record(contract=config.get('selection_contract'), thinking=value['thinking'],
                            slots=value['slots'], grammar=GEMMA4 if gemma4 else None),
            'kv-session record differs from the selection contract or the pinned header')
    require(config.get('tasks') == slot_tasks(value['slots']) and config.get('task') == slot_tasks(1)[0],
            'kv-session server registers exactly its slot tasks')
    require(_int(config.get('max_running_sessions'), 1),
            'kv-session needs a session pool (--max-running-sessions K)')
    return value


def options(config):
    """What this server runs, for /v1/kv-session/info: derived from the startup config."""
    from .context_limits import max_model_len, max_tokens
    kv = config[RECORD_KEY]
    value = dict(policy=config['policy'], page_tokens=config['page_tokens'], max_tokens=max_tokens(config),
                 max_model_len=max_model_len(config), thinking=kv['thinking'],
                 max_running_sessions=config['max_running_sessions'], slots=kv['slots'])
    if 'grammar' in kv:
        value['grammar'] = kv['grammar']     # the Gemma 4 profile's turn grammar (gemma-mab)
    contract = config.get('selection_contract')
    if contract is not None:
        rewriter = contract.get('goal_rewriter')
        value.update(budget_mode='fixed' if contract['active_token_budget'] is not None else 'ratio',
                     active_token_budget=contract['active_token_budget'], active_ratio=contract['active_ratio'],
                     switch_threshold_tokens=contract['switch_threshold_tokens'],
                     recent_turns=contract['recent_turns'], recent_quota_pages=contract['recent_quota_pages'],
                     mandatory=contract['mandatory'], hidden_state=contract['hidden_state'],
                     projection_dimension=contract['projection_dimension'],
                     projection_seed=contract['projection_seed'],
                     goal_query_variant=contract['goal_query_variant'],
                     goal_rewriter=None if rewriter is None else rewriter['model'])
        if 'random_fill' in contract:
            value['fill_seed'] = contract['random_fill']['seed']
    return value


def info(config):
    """GET /v1/kv-session/info: method, budget and this server's options."""
    kv = config[RECORD_KEY]
    return dict(method=kv['method'], budget=kv['budget'], options=options(config))


def check_header(render_template, template, thinking):
    """Startup only: the pinned template's generation header (enable_thinking per
    ``thinking``) is HEADERS[thinking], the text the client derives the same way
    (render with and without the generation prompt)."""
    probe = [dict(role='user', content='kv-session header probe')]
    flag = dict(enable_thinking=thinking == 'on')
    without = render_template(template, probe, dict(flag, add_generation_prompt=False))
    with_header = render_template(template, probe, dict(flag, add_generation_prompt=True))
    require(type(without) is str and type(with_header) is str and with_header.startswith(without)
            and with_header[len(without):] == HEADERS[thinking],
            'pinned template generation header differs from the kv-session header (%s)' % thinking)
    return HEADERS[thinking]


# A tool call and its response, the continuation probe of the Gemma template.
_PROBE_CALL = dict(id='call_kv_probe', type='function', function=dict(name='probe', arguments={}))


def check_gemma4_header(render_template, template):
    """Startup only (the Gemma 4 profile): with thinking off the pinned template's
    generation header after a user turn is GEMMA4_HEADER, and after tool responses
    it is empty, the render ending with ``<tool_response|>`` (the model turn goes on)."""
    flag = dict(enable_thinking=False)
    probe = [dict(role='user', content='kv-session header probe')]
    without = render_template(template, probe, dict(flag, add_generation_prompt=False))
    with_header = render_template(template, probe, dict(flag, add_generation_prompt=True))
    require(type(without) is str and type(with_header) is str and with_header.startswith(without)
            and with_header[len(without):] == GEMMA4_HEADER,
            'pinned Gemma template generation header differs from the kv-session header')
    probe = probe + [dict(role='assistant', content='', tool_calls=[_PROBE_CALL]),
                     dict(role='tool', tool_call_id=_PROBE_CALL['id'], content='kv-session probe result')]
    without = render_template(template, probe, dict(flag, add_generation_prompt=False))
    with_header = render_template(template, probe, dict(flag, add_generation_prompt=True))
    require(type(without) is str and with_header == without and without.endswith(GEMMA4_TOOL_CLOSE)
            and without.count(GEMMA4_TOOL_OPEN) == 1,
            'pinned Gemma template does not continue the model turn after a tool response')
    return GEMMA4_HEADER


# ---- requests --------------------------------------------------------------------------------

@dataclass(frozen=True)
class KvRequest:
    """One validated complete body. ``messages``: the client's whole list on the
    frontend, the new messages only on the engine (``from_source``)."""
    session_id: str
    input_ids: tuple
    header_tokens: int
    prefix_tokens: int
    boundary_tokens: int
    max_tokens: int
    seed: object
    messages: tuple
    objective: str | None = None     # engine: the root objective a turn-0 source carries

    @property
    def first(self):
        """A turn-0 body: the whole first prompt, no boundary."""
        return self.boundary_tokens == 0


def check_session_id(value):
    if type(value) is not str or not SESSION_ID.match(value):
        fail('BAD_SESSION_ID', 'session_id must be 32 lowercase hex characters')
    return value


def _message(m):
    if type(m) is not dict or m.get('role') not in ROLES:
        fail('BAD_MESSAGES', 'messages must be objects with role system, user, assistant or tool')
    content = m.get('content')
    if not (content is None or type(content) is str) or (content is None and m['role'] != 'assistant'):
        fail('BAD_MESSAGES', 'message content must be text (assistant content may be absent)')
    return m


def check_messages(messages):
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        fail('BAD_MESSAGES', 'messages must be a list of 1..%d messages' % MAX_MESSAGES)
    for m in messages:
        _message(m)
    return messages


def check_wire(body):
    """The body's shape (FrontendEvidence.begin via config.validate_wire): exact keys,
    a session id and a message list. Returns the messages."""
    if type(body) is not dict:
        fail('BAD_REQUEST', 'JSON object body required')
    missing, extra = sorted(COMPLETE_KEYS - set(body)), sorted(set(body) - COMPLETE_KEYS)
    if missing or extra:
        fail('BAD_REQUEST', 'complete body keys differ: missing %s, unknown %s' % (missing, extra))
    check_session_id(body['session_id'])
    return check_messages(body['messages'])


def parse_complete(body, *, budget, bound, max_model_len, empty_header=False):
    """POST /v1/kv-session/complete: the validated KvRequest, or KvSessionError.

    ``budget``: the server's (info); ``bound``: its pinned reply bound (--max-tokens);
    the server decodes greedily, so the sampling fields must be the neutral ones.
    ``empty_header`` (the Gemma grammar): a later turn may carry header_tokens 0 (a
    model turn that goes on after tool responses); its tail still ends with one token
    phase 2 computes (``<tool_response|>``, checked against the session)."""
    messages = check_wire(body)
    if body['method'] != METHOD:
        fail('METHOD_MISMATCH', 'this server runs %s, not %r' % (METHOD, body['method']))
    if body['budget'] != budget or type(body['budget']) is not type(budget):
        fail('BUDGET_MISMATCH', 'this server runs budget %r, not %r' % (budget, body['budget']))
    ids = body['input_ids']
    if type(ids) is not list or not 1 <= len(ids) <= max_model_len or not all(_int(t, 0) for t in ids):
        fail('BAD_INPUT_IDS', 'input_ids must be 1..%d token ids' % max_model_len)
    header, prefix, boundary = body['header_tokens'], body['prefix_tokens'], body['boundary_tokens']
    if empty_header and _int(boundary, 1) and header == 0 and type(header) is int:
        suffix = 1           # a continuation's last token, computed by phase 2
    elif not (_int(header, 1) and _int(prefix, 1) and _int(boundary, 0)):
        fail('BAD_GEOMETRY', 'header_tokens and prefix_tokens must be positive, boundary_tokens >= 0')
    else:
        suffix = header
    if not _int(prefix, 1):
        fail('BAD_GEOMETRY', 'header_tokens and prefix_tokens must be positive, boundary_tokens >= 0')
    if boundary == 0:
        if len(ids) != prefix + header:
            fail('BAD_GEOMETRY', 'a turn-0 body is the whole prompt: len(input_ids) = prefix_tokens + '
                 'header_tokens (%d != %d + %d)' % (len(ids), prefix, header))
    elif boundary + suffix >= len(ids):
        fail('BAD_GEOMETRY', 'input_ids must hold the boundary (%d), at least one message token and the '
             'header (%d); it has %d tokens' % (boundary, header, len(ids)))
    limit = body['max_tokens']
    if not (_int(limit) and 1 <= limit <= bound):
        fail('MAX_TOKENS', 'max_tokens must be 1..%d (the server reply bound)' % bound)
    temperature, top_p, top_k, penalty = (body[k] for k in ('temperature', 'top_p', 'top_k',
                                                             'repetition_penalty'))
    if not (_number(temperature) and temperature == 0 and _number(top_p) and top_p == 1
            and _int(top_k) and top_k in (-1, 0) and _number(penalty) and penalty == 1):
        fail('SAMPLING', 'the server decodes greedily: temperature 0, top_p 1, top_k -1, '
             'repetition_penalty 1')
    seed = body['seed']
    if not (seed is None or _int(seed)):
        fail('SAMPLING', 'seed must be an integer or null')
    return KvRequest(body['session_id'], tuple(ids), header, prefix, boundary, limit, seed, tuple(messages))


def check_close(body):
    """POST /v1/kv-session/close: the session id."""
    if type(body) is not dict or set(body) != CLOSE_KEYS:
        fail('BAD_REQUEST', 'close body must be {"session_id": ...}')
    return check_session_id(body['session_id'])


def effective_max_tokens(request, prompt_length, max_model_len):
    """The client's max_tokens clamped to the context left after the prompt; the
    context-limit refusal (before any model work) when none is left."""
    remaining = max_model_len - prompt_length
    if remaining < 1:
        fail('CONTEXT_LIMIT', CONTEXT_LIMIT)
    return min(request.max_tokens, remaining)


def check_sampling(params, stop_ids, bound, prompt_length, max_model_len):
    """EngineCore's check of a kv-session request (private_endpoint.render_claim): greedy,
    EOS/pad stops, 1 <= max_tokens <= the pinned bound and within the context, no other
    control; the route set exactly these (kv_session_route.prepare_request)."""
    value = lambda name, default=None: getattr(params, name, default)
    stops = set(value('stop_token_ids') or ()) | {value('eos_token_id')}
    stops.discard(None)
    limit = value('max_tokens')
    wrong = [name for name, ok in (
        ('n', value('n', 1) == 1), ('temperature', value('temperature') == 0),
        ('max_tokens', _int(limit) and 1 <= limit <= bound and prompt_length + limit <= max_model_len),
        ('min_tokens', value('min_tokens', 0) == 0), ('stop', not value('stop')),
        ('ignore_eos', not value('ignore_eos', False)), ('stop_token_ids', stops == set(stop_ids)),
        ('repetition_penalty', value('repetition_penalty', 1.0) == 1.0),
        ('presence_penalty', value('presence_penalty', 0.0) == 0.0),
        ('frequency_penalty', value('frequency_penalty', 0.0) == 0.0),
        ('logit_bias', not value('logit_bias')), ('allowed_token_ids', value('allowed_token_ids') is None),
        ('bad_words', not value('bad_words')), ('structured_outputs', value('structured_outputs') is None),
        ('logits_processors', not value('logits_processors'))) if not ok]
    require(not wrong, 'kv-session requires greedy generation within the reply bound; differs: '
            + ', '.join('%s=%r' % (name, value(name)) for name in wrong))
    return True


def finish_reason(generated, stop_ids, max_tokens):
    """'stop' on an EOS/pad token, 'length' when the reply used its whole bound."""
    if generated[-1] in stop_ids:
        return 'stop'
    require(len(generated) == max_tokens, 'reply ended without a stop token before its bound')
    return 'length'


# ---- the new messages and the Goal inputs ------------------------------------------------------

def _plain(m):
    """A message of the engine source: role and text, an assistant's tool names."""
    value = dict(role=m['role'], content=m.get('content') or '')
    if m['role'] == 'assistant':
        names = []
        for call in m.get('tool_calls') or ():
            function = call.get('function') if type(call) is dict else None
            name = function.get('name') if type(function) is dict else None
            names.append(name if type(name) is str else '')
        value['tool_names'] = names
    return value


def objective_of(messages):
    """The session's root objective: its first user message (stripped)."""
    for m in messages:
        if m['role'] == 'user' and (m.get('content') or '').strip():
            return m['content'].strip()
    return ''


def new_messages(request, history, first):
    """The messages since the previous request: everything at turn 0 (no assistant
    message), else (the client's own rule) the previous reply, then only tool/user
    feedback. Nothing older is compared: tokens never come from messages."""
    messages = request.messages
    if first:
        if any(m['role'] == 'assistant' for m in messages):
            fail('BAD_MESSAGES', 'a turn-0 body carries no assistant message')
        return [_plain(m) for m in messages]
    new = messages[history:]
    if not new or any(m['role'] == 'assistant' for m in new[1:]):
        fail('BAD_MESSAGES', 'messages must extend the previous request by one reply and its tool/user '
             'feedback (%d messages before, %d now)' % (history, len(messages)))
    return [_plain(m) for m in new]


def goal_inputs(messages, first):
    """TurnContext fields of one turn from its new messages: the observation (the new
    tool/user texts), its source ('user' when a real user message arrived, as the
    question and a format-error nudge do), the explicit requirement (that user
    message) and the structured state (tool name, <returncode>)."""
    feedback = [m for m in messages if m['role'] != 'assistant']
    observation = '\n'.join(m['content'] for m in feedback)
    users = [m['content'].strip() for m in feedback if m['role'] in ('user', 'system')]
    replies = [m for m in messages if m['role'] == 'assistant']
    names = replies[0].get('tool_names', []) if replies else []
    tools = [m['content'] for m in feedback if m['role'] == 'tool']
    code = RETURN_CODE.search(tools[0]) if tools else None
    state = dict(tool_name=names[0] if names and names[0] else None,
                 return_code=int(code.group(1)) if code else None)
    if first:
        return dict(observation=observation, message_source='user', explicit_requirement=None,
                    structured_state=state)
    requirement = users[-1] if users and users[-1] else None
    return dict(observation=observation, message_source='user' if users else 'tool',
                explicit_requirement=requirement, structured_state=state)


def source(request, messages, objective):
    """The engine's source of one request: the token body, the new messages and,
    at turn 0, the root objective."""
    value = dict(schema=SOURCE_SCHEMA, session_id=request.session_id, input_ids=list(request.input_ids),
                 header_tokens=request.header_tokens, prefix_tokens=request.prefix_tokens,
                 boundary_tokens=request.boundary_tokens, messages=messages, objective=objective)
    return validate_source(value)


def validate_source(value):
    """The engine's check of a kv-session source (frontend-built, frame-authenticated)."""
    require(type(value) is dict and set(value) == {'schema', 'session_id', 'input_ids', 'header_tokens',
                                                   'prefix_tokens', 'boundary_tokens', 'messages', 'objective'}
            and value['schema'] == SOURCE_SCHEMA, 'exact kv-session source schema required')
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    require(len(raw.encode('utf-8')) <= source_bytes(), 'kv-session source exceeds limit')
    value = json.loads(raw)
    require(type(value['session_id']) is str and SESSION_ID.match(value['session_id']) is not None
            and type(value['input_ids']) is list and value['input_ids']
            and all(_int(t, 0) for t in value['input_ids'])
            and _int(value['header_tokens'], 0) and _int(value['prefix_tokens'], 1)
            and _int(value['boundary_tokens'], 0)
            # header 0: a later Gemma turn that goes on after tool responses (the transcript checks it)
            and (value['header_tokens'] > 0 or value['boundary_tokens'] > 0)
            and (value['objective'] is None) == (value['boundary_tokens'] > 0)
            and (value['objective'] is None or type(value['objective']) is str)
            and type(value['messages']) is list and 1 <= len(value['messages']) <= MAX_MESSAGES
            and all(type(m) is dict and m.get('role') in ROLES and type(m.get('content')) is str
                    and set(m) <= {'role', 'content', 'tool_names'} for m in value['messages']),
            'kv-session source fields invalid')
    return value


def source_bytes():
    """The source cap now (runtime_renderer.MAX_SOURCE_BYTES, as pinned at startup)."""
    import runtime_renderer
    return runtime_renderer.MAX_SOURCE_BYTES


def from_source(value):
    """The engine's KvRequest of a validated source (messages: the new ones)."""
    return KvRequest(value['session_id'], tuple(value['input_ids']), value['header_tokens'],
                     value['prefix_tokens'], value['boundary_tokens'], 0, None, tuple(value['messages']),
                     value['objective'])


# ---- the token transcript ----------------------------------------------------------------------

class KvTranscript:
    """Server-owned token stream of one kv-session session (frontend and engine).

    The same logic runs in the API server (it builds the prompt token IDs that the
    native request carries) and in EngineCore (it re-derives them from its own
    record of the generated tokens and the request's source). Attribute names
    follow mswea_protocol.Transcript where the controller checkpoints them
    (controller.TRANSCRIPT_STATE); ``history`` is the client's message count of the
    last committed request. A kv-session never rebuilds: ``extends`` is always True.

    ``tools`` (the Gemma grammar): the ids of ``<|tool_response>`` and
    ``<tool_response|>``. A later request may then open its new messages right after
    ``<|tool_response>`` (``input_ids[boundary - 1]``) and carry no header when the
    model turn goes on after tool responses (its tail ends with ``<tool_response|>``,
    the one token phase 2 computes); ``im_start`` is ``<|turn>``.
    """
    protocol = PROTOCOL

    def __init__(self, *, header, im_start, vocab, decode, stop_ids, tools=None):
        require(type(header) is tuple and header and all(_int(t, 0) for t in header)
                and _int(im_start, 0) and _int(vocab, 1) and callable(decode)
                and type(stop_ids) is frozenset and stop_ids, 'kv-session transcript pins required')
        require(tools is None or (type(tools) is tuple and len(tools) == 2 and all(_int(t, 0) for t in tools)
                                  and tools[1] != header[-1]),
                'kv-session tool-response delimiters must be two ids (the Gemma grammar)')
        self.header, self.im_start, self.vocab = header, im_start, vocab
        self.decode, self.stop_ids = decode, stop_ids
        self.tools = tools
        self.sessions = 0
        self.staged = None       # frontend: the validated request the next capture renders
        self._reset()

    def _reset(self):
        self.history = 0         # client messages of the last committed request
        self.turns = []          # dict(text, count) per committed turn
        self.stream = ()         # every token of this session so far
        self.initial_length = 0  # prefix_tokens: the mandatory prefix
        self.pending = None
        self.objective = None    # the root objective (the first user message)

    @property
    def turn(self):
        return len(self.turns)

    def extends(self, messages=None):
        """Append-only by construction: no request rebuilds a kv-session."""
        return True

    def prompt_length(self, request):
        """len(T) of ``request`` on this transcript (the route's context-limit check)."""
        if self.turn == 0:
            return len(request.input_ids)
        return len(self.stream) + len(request.input_ids) - 1

    def suffix(self, request):
        """Tokens of ``request``'s tail that phase 2 computes: its header, or the closing
        ``<tool_response|>`` of a Gemma continuation (header_tokens 0)."""
        return request.header_tokens or 1

    def phase2_suffix(self, tokens):
        """The phase-2 suffix of a prompt this transcript renders, from its tokens alone
        (the capacity gate of the window layout): the pinned header, or one token after
        Gemma tool responses (the header's last token is never ``<tool_response|>``)."""
        H = len(self.header)
        if self.tools is not None and tuple(tokens[-H:]) != self.header and tokens[-1] == self.tools[1]:
            return 1
        return H

    def check(self, request):
        """The request's structure against this session, without changing anything."""
        ids, H = request.input_ids, len(self.header)
        if self.tools is not None and request.header_tokens == 0 and not request.first and self.turn > 0:
            # Gemma: the model turn goes on after tool responses, no generation header.
            if ids[-1] != self.tools[1] or ids[request.boundary_tokens - 1] != self.tools[0]:
                fail('HEADER_MISMATCH', 'a request without a header continues the model turn after tool '
                     'responses: <|tool_response> ends its boundary and <tool_response|> ends input_ids')
        elif request.header_tokens != H or ids[-H:] != self.header:
            fail('HEADER_MISMATCH', 'input_ids must end with the pinned %d-token generation header' % H)
        if self.turn == 0:
            if not request.first:
                fail('SESSION_STATE', 'the first request of a session must be a turn-0 body (boundary_tokens 0)')
            if max(ids) >= self.vocab:
                fail('BAD_INPUT_IDS', 'token ids must be below %d' % self.vocab)
            return
        if request.first:
            fail('SESSION_STATE', 'turn-0 body for a session at turn %d' % self.turn)
        if request.prefix_tokens != self.initial_length:
            fail('PREFIX_MISMATCH', 'prefix_tokens %d differs from the session prefix %d'
                 % (request.prefix_tokens, self.initial_length))
        if ids[0] != self.stream[-1]:
            fail('CONTINUATION_MISMATCH', 'input_ids[0] must repeat the last output id of the previous reply')
        if max(ids[1:]) >= self.vocab:
            fail('BAD_INPUT_IDS', 'token ids must be below %d' % self.vocab)
        if self.tools is not None:
            # Gemma: a new turn (<|turn>) or tool responses (after <|tool_response>).
            if ids[request.boundary_tokens] != self.im_start and ids[request.boundary_tokens - 1] != self.tools[0]:
                fail('BOUNDARY_MISMATCH', 'the new messages must open at input_ids[boundary_tokens] (<|turn>) '
                     'or right after input_ids[boundary_tokens - 1] (<|tool_response>)')
        elif ids[request.boundary_tokens] != self.im_start:
            fail('BOUNDARY_MISMATCH', 'input_ids[boundary_tokens] must open the new messages (<|im_start|>)')

    def stage(self, request):
        """Frontend: the request the next capture renders (one request per session)."""
        require(type(request) is KvRequest and self.pending is None, 'kv-session turn already pending')
        self.staged = request

    def render(self, request=None):
        """Prompt tokens, stable prefix end (o) and message span [o, e) of ``request``."""
        request = self.staged if request is None else request
        require(type(request) is KvRequest, 'kv-session request required')
        require(self.pending is None, 'previous kv-session turn neither committed nor abandoned')
        self.check(request)
        ids = request.input_ids
        if self.turn == 0:
            tokens, stable, span = ids, request.prefix_tokens, None
            self.initial_length = stable
        else:
            tokens = self.stream + ids[1:]
            o = len(self.stream) - 1 + request.boundary_tokens
            stable, span = o, (o, len(tokens) - self.suffix(request))
        self.pending = dict(request=request, tokens=tokens)
        return tokens, stable, span

    def abandon(self):
        self.pending = None

    def release(self):
        """The session closed for good: drop its stream, history and any pending render."""
        self._reset()
        self.staged = None

    def commit(self, prompt_tokens, generated, message_count=None):
        """Append the generated tokens (the last sample included); return the reply."""
        pending = self.pending
        require(pending is not None and tuple(prompt_tokens) == pending['tokens'],
                'committed prompt differs from the rendered kv-session turn')
        generated = tuple(generated)
        require(generated and all(_int(t, 0) for t in generated), 'generated token IDs required')
        request = pending['request']
        text = self.decode(list(generated))
        if not self.turns:
            self.sessions += 1
            self.objective = (request.objective if request.objective is not None
                              else objective_of(request.messages))
        self.turns.append(dict(text=text, count=len(generated)))
        self.stream = pending['tokens'] + generated
        if message_count is not None:
            self.history = message_count
        self.pending = None
        self.staged = None
        return dict(text=text, generated=generated, stop=generated[-1] in self.stop_ids)

    def previous_action(self):
        """The decoded previous reply (reasoning and tool-call text included)."""
        return self.turns[-1]['text'] if self.turns else None


def kv(transcript):
    """Whether ``transcript`` is a kv-session transcript."""
    return getattr(transcript, 'protocol', None) == PROTOCOL


# ---- client sessions -> slots ------------------------------------------------------------------

class KvSessionRegistry:
    """Frontend map of client session ids to the server's slots.

    A client id binds to the next unused slot at its first request, which must be a
    turn-0 body; a request of an unbound id that is not one is refused. A closed id
    is never served again (its first close's answer is repeated), and a slot never
    binds twice: the lifetime is ``slots`` sessions."""

    def __init__(self, slots):
        require(_int(slots) and 1 <= slots <= MAX_SLOTS, 'kv-session slots must be 1..%d' % MAX_SLOTS)
        self.slots = slots
        self.bound = {}          # client id -> slot index
        self.closed = {}         # client id -> the close answer
        self.next = 0

    def resolve(self, request):
        sid = request.session_id
        if sid in self.closed:
            fail('SESSION_CLOSED', 'session %s was closed; a session id is never served again' % sid)
        index = self.bound.get(sid)
        if index is not None:
            return index
        if not request.first:
            fail('SESSION_UNKNOWN', 'unknown session %s: its first request must be a turn-0 body '
                 '(boundary_tokens 0)' % sid)
        if self.next >= self.slots:
            fail('SESSION_LIMIT', 'this server serves at most %d sessions over its lifetime '
                 '(--kv-session-slots)' % self.slots)
        index = self.bound[sid] = self.next
        self.next += 1
        return index

    def slot(self, sid):
        """The slot of a bound, unclosed client id, else None."""
        return None if sid in self.closed else self.bound.get(sid)

    def closed_answer(self, sid):
        return self.closed.get(sid)

    def mark_closed(self, sid, answer):
        self.closed[sid] = answer
