"""Thinking ON for the Mini-SWE-Agent XML and chat-text protocols (stage 1; serve.py --enable-thinking).

Qwen3.6-27B (below) and Gemma 4 31B IT (the last section), opt-in. Off (the default) none of
this runs: no startup config key, no evidence key, the same header, template arguments,
responses and transcripts as before. The kv-session protocol has its own
--kv-session-thinking (its client renders and parses the replies itself).

On (startup config ``thinking``, ``record``), Qwen:

- Rendering. The generation header is the template's thinking-on prompt
  ``<|im_start|>assistant\\n<think>\\n`` (5 tokens; off: the empty think block, 7). The XML
  protocol renders system + task with ``enable_thinking=True`` (the same text: the flag only
  changes the generation prompt); chat-text renders a fresh or rebuilt session with
  ``enable_thinking=True, preserve_thinking=True`` and passes each assistant message's
  reasoning to the template as ``reasoning_content``.
- The reply. The model generates ``R\\n</think>\\n\\nC`` (``</think>``, id 248069, is not a
  special token, so the decoded text keeps it). ``split`` cuts the decoded text at its first
  ``</think>``: ``reasoning`` is the text before it, ``content`` the text after it, both as
  generated (no strip; reasoning + ``</think>`` + content is the whole text). A reply without
  ``</think>`` (cut at the reply bound inside the reasoning, or stopped on EOS before it) is
  all reasoning with empty content, as vLLM's Qwen3 reasoning parser is believed to treat a
  thinking-enabled reply without an end token (its source is not part of this repository;
  the overlay's serving.py hands that parser the request's chat_template_kwargs). The
  response carries ``content`` = C and the reasoning twice, ``reasoning_content`` and
  ``reasoning`` (litellm reads vLLM's ``reasoning`` into ``reasoning_content``; clients differ).
  Content is never null: a client gets ``""`` (Mini-SWE-Agent then reports a format error,
  Terminus 2 a parse error, both as for any reply without an action).
- The XML action marker. MarkerStop arms only after ``</think>``: it stops once the text after
  the first ``</think>`` holds ``</mswea_bash_command>``; a marker inside R never stops the
  reply. Nothing else in the server scans a reply for protocol markers.
- History. A client sends the reasoning back as a separate field of its assistant message:
  Mini-SWE-Agent 2.4.2 through litellm and Terminus 2 with ``interleaved_thinking`` send
  ``reasoning_content``; a top-level ``reasoning`` is read too (``client_reasoning``:
  ``reasoning_content`` first). vLLM's request model keeps only ``reasoning`` of an assistant
  message, so the routes read the field from the request's raw JSON body (``attach``) and
  hand the transcripts ``{role, content, reasoning}``. A request continues the session (an
  append: the server's stream keeps R's exact token ids) when it repeats the previous
  request's messages exactly (reasoning fields included), then the previous reply with its
  ``content`` equal to the returned C and its reasoning equal to the returned R, both
  compared after ``strip()`` (the template trims both; a missing reasoning field counts as
  ``""``), then one new user message. On the XML protocol (keep-rebuild-1116) a request that
  repeats the previous request exactly and adds one user message (the client dropped the
  previous reply: Mini-SWE-Agent after a format error) is a rollback: its prompt is the session's
  stream up to the dropped reply's header (exact ids, every earlier R included) + the new
  message + the header; the engine re-prefills it as a rebuild (local turn 0), and the row's
  ``history`` is ``rollback``. Anything else takes the protocol's rebuild: the XML protocol
  renders system + task through the template and every later message as the template renders
  it with ``preserve_thinking=True`` (``keep_message_text``: R where the client sent it, trimmed
  text; on Gemma the template's own render, which has no R); before keep-rebuild-1116 it
  re-encoded every later message from its content text (C; R dropped), so a run before that
  commit lost every earlier R at a session's first format error. Rollback and rebuild rows add
  ``kept_tokens`` and ``reencoded_replies``. Chat-text renders the whole conversation through the
  template with the thinking-on arguments (R where the client sent it). A client that drops R (Terminus 2
  with interleaved_thinking off) therefore rebuilds every turn; stage 2 (cut the last reply
  and re-prefill) is the separate option --thinking-history strip (remote_ra_live_v2.thinking_cut).
- Reply bound. max_tokens counts every generated token, the reasoning included, as vLLM
  does; the capacity rule reserves ceil((prompt + max_tokens) / P) Pages for a turn, and the
  context checks count the stream, R's tokens included, as prompt.
- The Goal's previous reply (query_selection: ``transcript.previous_action()``) is C only, the
  action, as with thinking off. FLAGGED for the Goal owner: this is a Goal input under the
  option (the full text R + C is the alternative).
- Evidence. Every runtime.jsonl row of such a server adds ``thinking`` (``evidence``): the
  reply's token split (``reasoning_tokens`` before the token that completes the first
  ``</think>``, ``think_end_tokens`` 1 or 0, ``content_tokens`` after it), whether the reply
  closed its reasoning, and how the request's prompt was built (``history``). With the pinned
  template and ``enable_thinking=True, preserve_thinking=True`` the offline reference render
  of a conversation whose replies kept R (``reference_messages``) equals the server's stream
  when every reply closed its reasoning and re-encodes to its own ids (and, on the XML
  protocol, every later user message is trimmed, as the template trims it).

Startup only (``pinned``, ``loaded``): nothing is checked per request beyond the protocol's
own comparisons.

History strip (serve.py --thinking-history strip, chat-text only; option (a) form E,
``remote_ra_live_v2.thinking_cut``): the record adds ``history: strip``, its template arguments
are the stock default ``enable_thinking=True`` (no ``preserve_thinking``: every past reply renders
as its content only) and its append rule is ``thinking_cut.RULE``. Without the option the record
is the one above (stage 1's keep mode).

Gemma 4 31B IT (model profile gemma4-31b-it; docs/thinking-gemma/DESIGN.md): the same rules
with Gemma's format, on its XML protocol (mswea_xml_gemma4_v1) and chat-text. The record
(``record(..., profile=GEMMA4_PROFILE)``; Qwen's is unchanged) pins the thinking-on header after a
user turn ``<|turn>model\\n`` (3 tokens; the model opens the channel itself), the template
arguments ``enable_thinking=True`` (no ``preserve_thinking``: it changes nothing for text turns;
the initial segment gains the template's ``<|think|>`` system turn), the channel markers, the
``thought\\n`` prefix and the split rule. The reply ``<|channel>thought\\n R <channel|> C`` is
split on its ids (the markers are special tokens, which the decoded text skips) by
``Gemma4Split``: the rules of vLLM 0.24.0's gemma4 reasoning parser (read at the v0.24.0 tag,
vllm/parser/gemma4.py and vllm/parser/engine/), which the stock control runs. MarkerStop counts
the marker only in the content so far; the evidence's token split counts the channel ids.
"""
from __future__ import annotations

import json
import os

from .contract import require

CONFIG_KEY = 'thinking'                 # startup config, ready.json, runtime.jsonl rows, chat-text info
XML = 'mswea_xml_v1'
CHAT = 'chat_text_v1'
PROTOCOLS = (XML, CHAT)
HEADER_TEXT = '<|im_start|>assistant\n<think>\n'     # = kv_session_protocol.HEADERS['on']
THINK_END = '</think>'
TEMPLATE_KWARGS = {XML: {'enable_thinking': True},
                   CHAT: {'enable_thinking': True, 'preserve_thinking': True}}
# The response's reasoning fields (both set) and the client fields read back (the first string wins).
RESPONSE_FIELDS = ('reasoning_content', 'reasoning')
CLIENT_FIELDS = ('reasoning_content', 'reasoning')
KEY = 'reasoning'                       # the reasoning of an assistant message in the transcripts' messages
APPEND_RULE = 'previous_messages_exact+reply_content_and_reasoning_stripped'
# How a request's prompt was built (evidence ``history``).
FRESH, APPEND = 'fresh', 'append'
DROPPED, DIFFERS, OTHER = 'rebuild_reasoning_dropped', 'rebuild_reasoning_differs', 'rebuild'
# keep-rebuild-1116: the previous request + one new user message (the client dropped the previous reply, as
# Mini-SWE-Agent does after a format error): the session's exact tokens up to the dropped reply's header, then
# the new message (``Transcript.rolls_back``).
ROLLBACK = 'rollback'
# The extra evidence keys of a rollback or rebuild row (``evidence``): the prompt tokens taken from the session's
# own stream (exact ids, generated ones included) and the assistant messages encoded from their text.
KEPT_TOKENS, REENCODED = 'kept_tokens', 'reencoded_replies'


# ---- Gemma 4 (model profile gemma4-31b-it) --------------------------------------------------------

GEMMA4_PROFILE = 'gemma4-31b-it'        # model_profile.GEMMA4
GEMMA4_XML = 'mswea_xml_gemma4_v1'      # mswea_protocol.GEMMA4_PROTOCOL
GEMMA4_PROTOCOLS = (GEMMA4_XML, CHAT)
GEMMA4_HEADER_TEXT = '<|turn>model\n'   # the template's thinking-on prompt after a user turn (3 tokens)
GEMMA4_TEMPLATE_KWARGS = {'enable_thinking': True}
# The initial segment's thinking-on system turn without a system message (7 tokens); with one
# the template puts <|think|>\n in front of its text (2 tokens).
GEMMA4_THINK_SYSTEM = '<bos><|turn>system\n<|think|>\n<turn|>\n'
CHANNEL_OPEN, CHANNEL_CLOSE = '<|channel>', '<channel|>'
THOUGHT_PREFIX = 'thought\n'            # vLLM's _GEMMA4_THOUGHT_PREFIX (removed from the reasoning)
GEMMA4_SPLIT = 'vllm-0.24.0-gemma4-parser-on-ids'


# ---- the pin (startup only) ----------------------------------------------------------------------

HISTORY_KEY = 'history'                 # the record's history mode: absent (keep, stage 1) or 'strip'


def protocols_of(profile):
    """The protocols thinking on serves for a model profile (None: Qwen)."""
    require(profile in (None, GEMMA4_PROFILE), 'thinking on is for the Qwen and Gemma 4 profiles only')
    return PROTOCOLS if profile is None else GEMMA4_PROTOCOLS


def record(protocol, history=None, profile=None):
    """The startup config's ``thinking`` record of a server of ``protocol``. ``history``
    (serve.py --thinking-history): None, stage 1's keep mode (the record as it was), or
    'strip' (chat-text only, thinking_cut). ``profile``: the config's model profile (None:
    Qwen, the record as it was; 'gemma4-31b-it': Gemma's header, arguments and split)."""
    require(protocol in protocols_of(profile), 'thinking on serves the mswea-xml and chat-text protocols only')
    if profile is None:
        value = dict(enable_thinking=True, header_text=HEADER_TEXT, template_kwargs=dict(TEMPLATE_KWARGS[protocol]),
                     think_end=THINK_END, response_fields=list(RESPONSE_FIELDS), client_fields=list(CLIENT_FIELDS),
                     append=APPEND_RULE)
    else:
        value = dict(enable_thinking=True, header_text=GEMMA4_HEADER_TEXT,
                     template_kwargs=dict(GEMMA4_TEMPLATE_KWARGS), channel=[CHANNEL_OPEN, CHANNEL_CLOSE],
                     thought_prefix=THOUGHT_PREFIX, split=GEMMA4_SPLIT, response_fields=list(RESPONSE_FIELDS),
                     client_fields=list(CLIENT_FIELDS), append=APPEND_RULE)
    if history is None:
        return value
    from . import thinking_cut
    require(history == thinking_cut.STRIP and protocol == CHAT,
            '--thinking-history strip serves the chat-text protocol only (mswea-xml and kv-session keep R)')
    value.update(template_kwargs=dict(thinking_cut.TEMPLATE_KWARGS if profile is None else
                                      thinking_cut.GEMMA4_TEMPLATE_KWARGS), append=thinking_cut.RULE)
    value[HISTORY_KEY] = thinking_cut.STRIP
    return value


def pinned(config):
    """The record a startup config pins, else None (the key is absent: thinking off)."""
    if config is None or CONFIG_KEY not in config:
        return None
    profile = config.get('model_profile')
    require(config.get('protocol') in protocols_of(profile),
            'thinking on serves the mswea-xml and chat-text protocols only (kv-session: --kv-session-thinking)')
    value = config[CONFIG_KEY]
    history = value.get(HISTORY_KEY) if type(value) is dict else None
    require(value == record(config['protocol'], history, profile),
            'pinned thinking record differs from the protocol pins; off is pinned by leaving it out')
    return dict(value)


def enabled(config):
    return pinned(config) is not None


def history_mode(config):
    """'strip' when the startup config pins --thinking-history strip, else None (stage 1's
    keep mode, or thinking off). Startup only."""
    value = pinned(config)
    return None if value is None else value.get(HISTORY_KEY)


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def check_header(render_template, template):
    """Startup only: the pinned template's thinking-on generation prompt is HEADER_TEXT."""
    probe = [dict(role='user', content='thinking header probe')]
    flag = dict(enable_thinking=True)
    without = render_template(template, probe, dict(flag, add_generation_prompt=False))
    with_header = render_template(template, probe, dict(flag, add_generation_prompt=True))
    require(type(without) is str and type(with_header) is str and with_header.startswith(without)
            and with_header[len(without):] == HEADER_TEXT,
            'pinned template thinking-on generation header differs from %r' % HEADER_TEXT)
    return HEADER_TEXT


def check_header_gemma4(render):
    """Startup only (the Gemma 4 profile): ``render(messages, add_generation_prompt)`` is the
    pinned template's text with the thinking-on arguments (content as OpenAI text parts). The
    initial segment opens with the ``<|think|>`` system turn, and the generation prompt after a
    user turn is GEMMA4_HEADER_TEXT."""
    probe = [dict(role='user', content='thinking header probe')]
    without, with_header = render(probe, False), render(probe, True)
    require(type(without) is str and type(with_header) is str and without.startswith(GEMMA4_THINK_SYSTEM)
            and with_header == without + GEMMA4_HEADER_TEXT,
            'pinned Gemma template thinking-on system turn or generation header differs from %r'
            % GEMMA4_HEADER_TEXT)
    return GEMMA4_HEADER_TEXT


# ---- the reply -----------------------------------------------------------------------------------

def split(text):
    """(reasoning, content, closed) of a decoded reply: cut at the first ``</think>``; none
    (``closed`` False): all reasoning, empty content."""
    require(type(text) is str, 'decoded reply text required')
    reasoning, end, content = text.partition(THINK_END)
    if not end:
        return text, '', False
    return reasoning, content, True


def marker_after_think(text, marker):
    """MarkerStop's predicate with thinking on: ``marker`` in the text after the first ``</think>``."""
    return marker in text.partition(THINK_END)[2]


def token_split(generated, decode, end_ids=None):
    """(reasoning_tokens, think_end_tokens, content_tokens) of a reply's generated ids: the
    tokens before the one that completes the first ``</think>`` in the decoded text, that
    token (1, or 0 without ``</think>``), the rest. ``end_ids``: ``</think>`` encoded (the
    pinned tokenizer: one added token, 248069); when it is one id, its first occurrence decides
    with one decode of the text before it. Otherwise a decoded prefix only grows, so the token
    is found by bisection (about log2(n) decodes). The frontend's evidence only."""
    generated = list(generated)
    n = len(generated)
    if end_ids is not None and len(end_ids) == 1 and end_ids[0] in generated:
        k = generated.index(end_ids[0])
        if THINK_END not in decode(generated[:k]):
            return k, 1, n - k - 1
    if THINK_END not in decode(generated):
        return n, 0, 0
    low, high = 0, n               # THINK_END not in decode(generated[:low]), in decode(generated[:high])
    while high - low > 1:
        middle = (low + high) // 2
        if THINK_END in decode(generated[:middle]):
            high = middle
        else:
            low = middle
    return high - 1, 1, n - high


CONTENT, REASONING = 'content', 'reasoning'


class Gemma4Split:
    """The split of a Gemma 4 reply's generated ids into reasoning and content (``__call__``):
    the rules of vLLM 0.24.0's gemma4 reasoning parser in its non-streaming ``extract_reasoning``
    (vllm/parser/gemma4.py on vllm/parser/engine/), applied to the ids instead of the text it
    decodes with special tokens kept:

    - the trailing stop id is not part of the reply text (vLLM's text excludes it);
    - the state starts in content; ``<|channel>`` (one id) in content opens reasoning,
      ``<channel|>`` in reasoning closes it, ``<channel|>`` in content is swallowed, and a
      ``<|channel>`` inside reasoning stays as its text; a reasoning still open at the end stays
      reasoning (a reply cut at its bound inside the channel: all reasoning, content ``""``);
    - when the reply has no ``<|channel>`` but holds ``<channel|>``, or its text starts with
      ``thought\\n`` or is ``thought``, the parser puts a ``<|channel>`` in front (its first-feed
      injection): the reply starts in reasoning;
    - the reasoning is the reasoning runs' text with trailing whitespace stripped
      (strip_trailing_reasoning_whitespace, on by default) and then a leading ``thought\\n``
      removed (``thought`` alone: no reasoning); the content is the content runs' text.

    The runs between channel ids decode on their own (the decode skips special tokens, as the
    reply text does; a run boundary is a special token, so the runs join to the reply's text).
    Differences from vLLM: an empty content or reasoning is ``""`` where vLLM sends null (the
    rule of both profiles here: content is never null); the markers are matched as ids (vLLM
    also splits at a plain-text spelling of them); other special tokens are skipped in the text
    (as in every reply here); ``<|tool_call>`` (vLLM: the end of reasoning) does not occur on
    these protocols, which serve no tools.

    ``encode``/``decode``: the transcript's (the markers must be one id each); ``stop_ids``:
    the profile's. ``markers``: the two channel ids (MarkerStop's incremental check decodes the
    whole output after one)."""

    def __init__(self, encode, decode, stop_ids):
        require(callable(encode) and callable(decode) and type(stop_ids) is frozenset and stop_ids,
                'Gemma split callbacks required')
        ids = [tuple(encode(text)) for text in (CHANNEL_OPEN, CHANNEL_CLOSE)]
        require(all(len(i) == 1 for i in ids) and ids[0] != ids[1] and not set(ids[0] + ids[1]) & stop_ids,
                'the Gemma channel markers must be one token each, not stop ids')
        self.open, self.close = ids[0][0], ids[1][0]
        self.markers = frozenset((self.open, self.close))
        self.decode, self.stop_ids = decode, stop_ids

    def __call__(self, generated):
        """dict(reasoning, content, closed, tokens): ``closed`` is True when the reply ends
        outside reasoning (no channel at all, or every channel closed); ``tokens`` the evidence's
        token split, ``reasoning_tokens`` (the ids in reasoning, ``thought\\n`` included),
        ``think_start_tokens`` (the ``<|channel>`` ids that opened reasoning),
        ``think_end_tokens`` (the ``<channel|>`` ids) and ``content_tokens`` (the rest, the stop
        id included); together the reply's length."""
        ids = list(generated)
        stop = 1 if ids and ids[-1] in self.stop_ids else 0
        body = ids[:len(ids) - stop]
        state = CONTENT
        if body and self.open not in body:
            text = self.decode(body)
            if self.close in body or text.startswith(THOUGHT_PREFIX) or text == THOUGHT_PREFIX.rstrip('\n'):
                state = REASONING       # the parser's injected <|channel>
        parts = {CONTENT: [], REASONING: []}
        counts = {CONTENT: stop, REASONING: 0, 'open': 0, 'close': 0}
        run = []                        # the ids since the last marker (the state changes only at one)
        for token in body:
            if token not in self.markers:
                run.append(token)
                counts[state] += 1
                continue
            if run:
                parts[state].append(self.decode(run))
                run = []
            if token == self.open and state == CONTENT:
                state = REASONING
                counts['open'] += 1
            elif token == self.open:
                parts[REASONING].append(CHANNEL_OPEN)      # no transition: the terminal's text
                counts[REASONING] += 1
            else:
                state = CONTENT                              # closes reasoning, or is swallowed
                counts['close'] += 1
        if run:
            parts[state].append(self.decode(run))
        reasoning =''.join(parts[REASONING]).rstrip()
        if reasoning.startswith(THOUGHT_PREFIX):
            reasoning = reasoning[len(THOUGHT_PREFIX):]
        elif reasoning == THOUGHT_PREFIX.rstrip('\n'):
            reasoning = ''
        tokens = dict(reasoning_tokens=counts[REASONING], think_start_tokens=counts['open'],
                      think_end_tokens=counts['close'], content_tokens=counts[CONTENT])
        return dict(reasoning=reasoning, content=''.join(parts[CONTENT]), closed=state == CONTENT, tokens=tokens)

    def content(self, tokens):
        """The content of a reply so far (MarkerStop's text with thinking on)."""
        return self(tokens)['content']


def splitter(grammar_profile, encode, decode, stop_ids):
    """The reply split of a thinking-on transcript: None for Qwen (``split`` on the decoded
    text), a ``Gemma4Split`` for the Gemma 4 profile."""
    if grammar_profile is None:
        return None
    require(grammar_profile == GEMMA4_PROFILE, 'thinking on is for the Qwen and Gemma 4 profiles only')
    return Gemma4Split(encode, decode, stop_ids)


def commit_split(transcript, text, generated):
    """(reasoning, content, closed, extra) of a committed reply on a thinking-on transcript:
    Qwen's ``split`` of the text, or the transcript's Gemma split of the ids (``extra``: the
    committed reply's ``split``, the evidence's token split)."""
    gemma = getattr(transcript, 'splitter', None)
    if gemma is None:
        reasoning, content, closed = split(text)
        return reasoning, content, closed, {}
    value = gemma(generated)
    return value['reasoning'], value['content'], value['closed'], dict(split=value['tokens'])


def set_field(message, name, value):
    """Set a response message field; vLLM's ChatMessage allows extra fields (pydantic
    extra='allow'), a model without the attribute keeps it in its extras."""
    try:
        setattr(message, name, value)
    except (AttributeError, TypeError, ValueError):
        extra = getattr(message, '__pydantic_extra__', None)
        require(type(extra) is dict, 'response message cannot carry %s' % name)
        extra[name] = value
    require(getattr(message, name, (getattr(message, '__pydantic_extra__', None) or {}).get(name)) == value,
            'response message did not keep %s' % name)


def respond(message, committed):
    """The response message of a thinking-on reply: content C, the reasoning in both fields."""
    message.content = committed['content']
    for name in RESPONSE_FIELDS:
        set_field(message, name, committed['reasoning'])


def evidence(committed, decode, encode=None):
    """The runtime.jsonl row's ``thinking`` record of one committed thinking-on reply
    (``encode``: the transcript's, for the one-id ``</think>`` of ``token_split``)."""
    if 'split' in committed:
        # The Gemma 4 profile: the transcript's split of the ids (Gemma4Split), with its channel ids.
        value = dict(committed['split'], closed=committed['closed'], reasoning_chars=len(committed['reasoning']),
                     content_chars=len(committed['content']), history=committed['history'])
    else:
        end_ids = None if encode is None else tuple(encode(THINK_END))
        reasoning, end, content = token_split(committed['generated'], decode, end_ids)
        value = dict(reasoning_tokens=reasoning, think_end_tokens=end, content_tokens=content,
                     closed=committed['closed'], reasoning_chars=len(committed['reasoning']),
                     content_chars=len(committed['content']), history=committed['history'])
    if 'cut' in committed:
        # --thinking-history strip (thinking_cut): where the turn was cut, and its re-prefill.
        value.update(cut=committed['cut'], reprefill=committed['reprefill'])
    if KEPT_TOKENS in committed:
        # keep-rebuild-1116 (the XML transcripts): a rollback or rebuild row's exact tokens and re-encoded replies.
        value.update({KEPT_TOKENS: committed[KEPT_TOKENS], REENCODED: committed[REENCODED]})
    return value


# ---- the client's messages -----------------------------------------------------------------------

def client_reasoning(message):
    """The reasoning a client sent back with an assistant message, or None."""
    for name in CLIENT_FIELDS:
        value = message.get(name)
        if type(value) is str:
            return value
    return None


def attach(messages, body):
    """Put each assistant message's reasoning from the raw JSON body (``body``, the request's
    wire) on the request's messages as ``reasoning`` (removed where the body has none). The
    two lists must name the same messages."""
    wire = json.loads(body) if type(body) in (bytes, str) else body
    sent = wire.get('messages') if type(wire) is dict else None
    require(type(messages) is list and type(sent) is list and len(sent) == len(messages)
            and all(type(m) is dict for m in messages) and all(type(m) is dict for m in sent),
            'thinking on: request messages differ from the request body')
    for message, raw in zip(messages, sent):
        require(message.get('role') == raw.get('role') and (type(message.get('content')) is not str
                or message.get('content') == raw.get('content')),
                'thinking on: request messages differ from the request body')
        if message.get('role') != 'assistant':
            continue
        value = client_reasoning(raw)
        if value is None:
            message.pop(KEY, None)
        else:
            message[KEY] = value
    return messages


def plain(message, thinking):
    """One message as the transcripts take it: role and content, and with thinking on an
    assistant message's reasoning (a string) when the client sent one."""
    value = dict(role=message.get('role'), content=message['content'])
    if thinking and value['role'] == 'assistant' and type(message.get(KEY)) is str:
        value[KEY] = message[KEY]
    return value


def check_keys(message, thinking):
    """A transcript message's keys: role and content, and with thinking on a string
    reasoning on an assistant message."""
    keys = set(message)
    if KEY in keys:
        require(thinking and message.get('role') == 'assistant' and type(message[KEY]) is str,
                'reasoning is an assistant message field of a thinking-on server')
        keys.discard(KEY)
    return keys


def matches(message, turn):
    """The append rule for the previous reply: content and reasoning as returned, compared
    after strip() (a missing reasoning field is "")."""
    return (message['content'].strip() == turn['content'].strip()
            and (message.get(KEY) or '').strip() == turn['reasoning'].strip())


def history(transcript, messages, extends, rollback=False):
    """How a request's prompt is built on ``transcript`` (evidence): fresh, append, rollback (the
    XML transcripts: the previous request + one user message), or why it rebuilds."""
    if transcript.turn == 0:
        return FRESH
    if extends:
        return APPEND
    if rollback:
        return ROLLBACK
    previous = transcript.history
    if (previous is not None and len(messages) == len(previous) + 2 and messages[:len(previous)] == previous
            and messages[-2]['role'] == 'assistant'
            and messages[-2]['content'].strip() == transcript.turns[-1]['content'].strip()):
        return DROPPED if messages[-2].get(KEY) is None else DIFFERS
    return OTHER


def keep_message_text(message):
    """One message after system + task as the pinned Qwen3.6 template renders it with
    ``enable_thinking=True, preserve_thinking=True`` (the keep render; keep-rebuild-1116): a user
    message ``<|im_start|>user\\n`` + its content trimmed + ``<|im_end|>\\n``; an assistant message
    ``<|im_start|>assistant\\n<think>\\n`` + R trimmed + ``\\n</think>\\n\\n`` + C trimmed +
    ``<|im_end|>\\n``, R the message's reasoning, or without one (the template's own rule) the text
    before the last ``</think>`` of the trimmed content (after its last ``<think>``, newlines
    trimmed) and C the text after it, else R empty. Jinja's ``trim`` is ``str.strip``. Each piece
    starts and ends at a special token, so it encodes alone as inside a whole render."""
    role, content = message['role'], message['content']
    require(role in ('user', 'assistant') and type(content) is str,
            'XML history holds user and assistant text messages only')
    text = content.strip()
    if role == 'user':
        return '<|im_start|>user\n' + text + '<|im_end|>\n'
    reasoning = message.get(KEY)
    if type(reasoning) is not str:
        reasoning = ''
        if THINK_END in text:
            reasoning = text.split(THINK_END)[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
            text = text.split(THINK_END)[-1].lstrip('\n')
    return '<|im_start|>assistant\n<think>\n' + reasoning.strip() + '\n</think>\n\n' + text + '<|im_end|>\n'


def reference_messages(messages):
    """Messages for the pinned template's reference render: each assistant reasoning as
    ``reasoning_content`` (render with TEMPLATE_KWARGS[chat_text_v1])."""
    out = []
    for m in messages:
        m = dict(m)
        if KEY in m:
            m['reasoning_content'] = m.pop(KEY)
        out.append(m)
    return out
