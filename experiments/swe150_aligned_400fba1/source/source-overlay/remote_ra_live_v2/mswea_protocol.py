"""Mini-SWE-Agent XML single-action protocol, token-identical to the dram-backed-kv server.

Upstream reference: ``scripts/serve_online_selector_v2.py`` (``_render``,
``_generate``, ``_append_boundary``, ``_root_objective``, ``_context``).

* ``messages[:2]`` (system + task) go through the chat template with
  ``add_generation_prompt=False`` and ``enable_thinking=False``, no tools.
* Every later message is ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` (user
  and assistant only).  A historical assistant message is replaced by the exact
  token IDs of its generation header, generated tokens and boundary, matched by
  ``content.strip()``.
* The generation header is ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``.
* Generation stops once the decoded text contains ``</mswea_bash_command>`` or
  on EOS/pad; the returned content is the decoded text of every generated
  token.  The boundary ``<|im_end|>\\n`` (``\\n`` after EOS/pad) joins the
  token stream but is never returned.

The same ``Transcript`` logic runs in the API server (to build the prompt token
IDs) and in EngineCore (to re-derive and verify them from its own record of
the generated tokens).

The turn literals above are the ``QWEN_XML`` grammar, the default. The Gemma 4
model profile (stage 1) uses ``GEMMA4_XML`` (protocol ``mswea_xml_gemma4_v1``):
a later message is ``<|turn>{role}\\n{content.strip()}<turn|>\\n`` with role
``model`` for an assistant (rebuilds only) after the empty thought channel, the
generation header is ``<|turn>model\\n<|channel>thought\\n<channel|>`` (Gemma's
thinking-off generation prompt) and the boundary is ``<turn|>\\n`` (``\\n``
after a stop id). History keeps the empty thought channel of every reply, which
Gemma's own template drops when it re-renders a conversation: the transcript is
append-only (design D-6).

Thinking on (serve.py --enable-thinking; ``thinking``): the header is
``<|im_start|>assistant\\n<think>\\n``, a reply splits at its first ``</think>``
into reasoning and content, the marker stop arms after ``</think>``, and an
append also needs the previous reply's reasoning back (``thinking.matches``).
On the Gemma 4 grammar the header is ``<|turn>model\\n``, a reply splits on its
channel ids (``thinking.Gemma4Split``) and the marker counts in its content only.
Off, everything above is unchanged.

Thinking on, keep-rebuild-1116: a request that repeats the previous request and adds one
user message (the client dropped the previous reply, as Mini-SWE-Agent does after a format
error) is a rollback: the session's stream up to the dropped reply's header (exact ids, every
earlier reasoning kept), the new user message, the header; the engine re-prefills it as a
rebuild. Every later message of a fresh or rebuilt session, and every user message of an
append, is the template's keep render (``grammar.keep_message_text``: Qwen with
preserve_thinking, R where the client sent it, text trimmed). Before this commit a rebuild
re-encoded every later message from its content, so every earlier R was lost from a
session's first format error on.
"""
from __future__ import annotations

import re

from .context_limits import MAX_TOKENS_CHOICES
from .contract import require
from . import thinking as _thinking

PROTOCOL = 'mswea_xml_v1'
MARKER = '</mswea_bash_command>'
GENERATION_HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
# The PoC's reply bound, also the server default. The reply bound is a server
# setting (serve.py --max-tokens, context_limits), not part of this protocol:
# check_sampling takes the server's pinned value.
MAX_TOKENS = 1536
RETURN_CODE = re.compile(r'<returncode>(-?\d+)</returncode>')
TASK = re.compile(r'<pr_description>\s*(.*?)\s*</pr_description>', re.DOTALL)
MAX_MESSAGES = 4096


def message_text(role, content):
    return f'<|im_start|>{role}\n{content}<|im_end|>\n'


def root_objective(messages):
    """Upstream ``_root_objective``: the PR description inside the task message."""
    require(len(messages) >= 2, 'a system message and task message are required')
    content = str(messages[1].get('content', '')).strip()
    match = TASK.search(content)
    return (match.group(1) if match else content).strip()


def return_code(observation):
    match = RETURN_CODE.search(observation)
    return int(match.group(1)) if match else None


def validate_messages(messages, problem_statement=None, thinking=False):
    """Upstream request shape: system, task, then user/assistant text, ending with a user.
    ``thinking`` (serve.py --enable-thinking): an assistant message may carry its reasoning."""
    require(type(messages) is list and 2 <= len(messages) <= MAX_MESSAGES,
            'XML protocol conversation required')
    for index, message in enumerate(messages):
        allowed = ('system',) if index == 0 else ('user',) if index == 1 else ('user', 'assistant')
        require(type(message) is dict and message.get('role') in allowed
                and type(message.get('content')) is str
                and (set(message) if not thinking else _thinking.check_keys(message, True)) <= {'role', 'content'},
                'XML protocol messages are system, task, then user/assistant text')
    require(messages[-1]['role'] == 'user', 'XML protocol request must end with a user message')
    if problem_statement is not None:
        require(problem_statement in messages[1]['content'],
                'initial task message must contain the frozen issue')
    return messages


def check_sampling(params, stop_ids, max_tokens=MAX_TOKENS):
    """Upstream greedy generation: argmax, ``max_tokens`` new tokens (the server's
    pinned reply bound, 1536 by default), EOS/pad stop, no other controls.

    The action marker is not a vLLM stop string: vLLM checks stop strings in the
    frontend and aborts the engine request afterwards.  ``MarkerStop`` ends the
    request inside the engine instead.

    ``skip_special_tokens`` is not part of the contract: it only shapes vLLM's
    own detokenized text, which the route replaces with the transcript's
    upstream decode.  vLLM's Qwen3 parser engine sets it to False while
    rendering, after the route has fixed the request.
    """
    require(type(max_tokens) is int and max_tokens in MAX_TOKENS_CHOICES, 'pinned reply bound required')
    value = lambda name, default=None: getattr(params, name, default)
    stops = set(value('stop_token_ids') or ()) | {value('eos_token_id')}
    stops.discard(None)
    wrong = [name for name, ok in (
        ('n', value('n', 1) == 1), ('temperature', value('temperature') == 0),
        ('max_tokens', value('max_tokens') == max_tokens), ('min_tokens', value('min_tokens', 0) == 0),
        ('stop', not value('stop')), ('ignore_eos', not value('ignore_eos', False)),
        ('stop_token_ids', stops == set(stop_ids)),
        ('repetition_penalty', value('repetition_penalty', 1.0) == 1.0),
        ('presence_penalty', value('presence_penalty', 0.0) == 0.0),
        ('frequency_penalty', value('frequency_penalty', 0.0) == 0.0),
        ('logit_bias', not value('logit_bias')), ('allowed_token_ids', value('allowed_token_ids') is None),
        ('bad_words', not value('bad_words')), ('structured_outputs', value('structured_outputs') is None),
        ('logits_processors', not value('logits_processors'))) if not ok]
    require(not wrong, 'XML protocol requires the upstream greedy generation contract; differs: '
            + ', '.join('%s=%r' % (name, value(name)) for name in wrong))
    return True


class MarkerStop:
    """Engine-side upstream stop: the decoded output contains the action marker.

    ``incremental`` (--decode-fastpath): the scheduler calls this once per
    appended output token, and a request whose decoded output held the marker is
    stopped. The marker is ASCII and ends in '>'; a byte-level decode maps ASCII
    bytes to themselves wherever they sit (UTF-8 never uses them inside a
    multi-byte sequence) and an output only grows at its end, so if the previous
    output of this request (one token shorter) was decoded without the marker,
    the marker can newly appear only if the new token's own bytes hold its final
    '>', which the token decoded alone shows. Such a token, and any call that does
    not follow the request's previous call by exactly one token, decodes the
    whole output as before. The answer is the same in every case."""

    # Requests remembered at most (output length checked without the marker); past
    # this the table starts over and the next call per request decodes in full.
    INCREMENTAL_REQUESTS = 4096
    # ``token_cache``: token ids remembered at most; past this the cache starts over.
    TOKEN_CACHE = 1 << 18

    def __init__(self, decode, incremental=False, token_cache=False, thinking=False, content=None, marker_ids=()):
        """``token_cache`` (--decode-fastpath2, with ``incremental``): remember per token id
        whether the token decoded alone holds a '>' (a token decodes to the same text
        every time, so the answer is the one the decode gives).

        ``thinking`` (--enable-thinking): the marker counts only in the text after the first
        ``</think>`` (a marker inside the reasoning never stops the reply). The incremental
        argument holds as it is: ``</think>`` also ends in '>', so the text after it can
        change only when the new token holds a '>'.

        ``content`` (with ``thinking``, the Gemma 4 profile: ``thinking.Gemma4Split.content``):
        the marker counts only in the content of the output so far, split on its ids, and
        ``marker_ids`` (the channel ids) are tokens that can move earlier text between reasoning
        and content: after one the whole output is decoded, as after a token holding '>'. Any
        other token adds text only to the end of the reasoning or of the content, so the
        argument above holds for it."""
        require(callable(decode), 'decoder required')
        require(type(incremental) is bool, 'incremental is a switch')
        require(type(token_cache) is bool and (incremental or not token_cache),
                'token_cache is a switch of the incremental check')
        require(type(thinking) is bool, 'thinking is a switch')
        require((content is None and not marker_ids) or (thinking and callable(content)
                                                          and type(marker_ids) is frozenset and marker_ids),
                'a content split needs thinking on and its marker ids')
        self.decode = decode
        self.incremental = incremental
        self.thinking = thinking
        self.content = content
        self.marker_ids = frozenset(marker_ids)
        self._checked = {}
        self._gt = {} if token_cache else None

    def _found(self, tokens):
        """The stop condition on a request's whole decoded output."""
        if self.content is not None:
            return MARKER in self.content(list(tokens))
        if self.thinking:
            return _thinking.marker_after_think(self.decode(list(tokens)), MARKER)
        return MARKER in self.decode(list(tokens))

    def _closes(self, token):
        """``token_cache``: True when ``token`` decoded alone holds a '>' (the marker's last character)."""
        cache = self._gt
        found = cache.get(token)
        if found is None:
            if len(cache) >= self.TOKEN_CACHE:
                cache.clear()
            found = cache[token] = '>' in self.decode([token])
        return found

    def _absent(self, request):
        """True when the marker is known absent from the request's decoded output."""
        tokens = request.output_token_ids
        n = len(tokens)
        rid = (request.request_id, id(request))     # this request object, never a reused id
        if n > 0 and self._checked.get(rid) == n - 1 and tokens[-1] not in self.marker_ids and (
                '>' not in self.decode([tokens[-1]]) if self._gt is None else not self._closes(tokens[-1])):
            absent = True
        else:
            absent = not self._found(tokens)
        if absent:
            if len(self._checked) >= self.INCREMENTAL_REQUESTS and rid not in self._checked:
                self._checked.clear()
            self._checked[rid] = n
        else:
            self._checked.pop(rid, None)
        return absent

    def __call__(self, request):
        if self.incremental:
            if self._absent(request):
                return False
        elif not self._found(request.output_token_ids):
            return False
        from vllm.v1.request import RequestStatus
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = MARKER
        return True


def boundary_text(generated, stop_ids):
    """Upstream ``_append_boundary``: EOS/pad already closed the turn."""
    return '\n' if generated and generated[-1] in stop_ids else '<|im_end|>\n'


class QwenGrammar:
    """The turn literals of the upstream protocol (Qwen3.6-27B): the functions above."""
    protocol = PROTOCOL
    profile = None
    header = GENERATION_HEADER
    thinking_header = _thinking.HEADER_TEXT      # --enable-thinking

    @staticmethod
    def message_text(role, content):
        return message_text(role, content)

    @staticmethod
    def thinking_message_text(role, content):
        """--enable-thinking before keep-rebuild-1116: a rebuild re-encoded every later message as
        with thinking off (content only, R dropped). Kept for reference; ``keep_message_text`` renders."""
        return message_text(role, content)

    @staticmethod
    def keep_message_text(message):
        """--enable-thinking: a later message as the pinned template renders it with
        preserve_thinking (thinking.keep_message_text: R where the client sent it, text trimmed)."""
        return _thinking.keep_message_text(message)

    @staticmethod
    def boundary_text(generated, stop_ids):
        return boundary_text(generated, stop_ids)


GEMMA4_PROTOCOL = 'mswea_xml_gemma4_v1'
GEMMA4_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA4_END_OF_TURN = '<turn|>\n'


def strip_thinking(text):
    """Gemma 4 template ``strip_thinking``: drop every ``<|channel>...<channel|>`` part,
    then ``trim`` (Python ``str.strip``)."""
    result = ''
    for part in text.split('<channel|>'):
        result += part.split('<|channel>')[0] if '<|channel>' in part else part
    return result.strip()


class Gemma4Grammar:
    """The Gemma 4 turn literals (model profile gemma4-31b-it, stage 1).

    Every piece equals what the pinned template renders for that turn, except that a
    model turn keeps the empty thought channel of its generation header (the template
    drops it from history; the transcript is append-only). The stage-0 stock-vLLM
    append reference (job 3364469) is reproduced token for token from these literals.

    Thinking on (--enable-thinking; remote_ra_live_v2.thinking, Gemma's record): the header is
    the template's thinking-on prompt ``<|turn>model\\n`` (the model opens the thought channel
    itself), and a rebuild re-encodes a past reply from its content after ``<|turn>model\\n``
    without a channel (the reasoning is not re-rendered, as on Qwen; that is also what the
    template renders for this protocol's history, whose observations are user messages)."""
    protocol = GEMMA4_PROTOCOL
    profile = _thinking.GEMMA4_PROFILE
    header = GEMMA4_HEADER
    thinking_header = _thinking.GEMMA4_HEADER_TEXT

    @staticmethod
    def message_text(role, content):
        require(role in ('user', 'assistant'), 'Gemma XML history holds user and assistant messages only')
        if role == 'assistant':
            # A rebuild re-encodes a reply from text: the kept channel, then the
            # template's strip_thinking (what the continuing transcript holds when
            # the reply round-trips).
            return GEMMA4_HEADER + strip_thinking(content) + GEMMA4_END_OF_TURN
        return '<|turn>user\n' + content.strip() + GEMMA4_END_OF_TURN

    @staticmethod
    def thinking_message_text(role, content):
        """--enable-thinking: a later message of a rebuild (a past reply without a channel)."""
        require(role in ('user', 'assistant'), 'Gemma XML history holds user and assistant messages only')
        if role == 'assistant':
            return _thinking.GEMMA4_HEADER_TEXT + strip_thinking(content) + GEMMA4_END_OF_TURN
        return '<|turn>user\n' + content.strip() + GEMMA4_END_OF_TURN

    @classmethod
    def keep_message_text(cls, message):
        """--enable-thinking: a later message as the pinned Gemma template renders it with the
        thinking-on arguments: ``thinking_message_text`` (the template keeps no R of a past reply)."""
        return cls.thinking_message_text(message['role'], message['content'])

    @staticmethod
    def boundary_text(generated, stop_ids):
        return '\n' if generated and generated[-1] in stop_ids else GEMMA4_END_OF_TURN


QWEN_XML = QwenGrammar()
GEMMA4_XML = Gemma4Grammar()
GRAMMARS = {PROTOCOL: QWEN_XML, GEMMA4_PROTOCOL: GEMMA4_XML}


def xml_transcripts(renderer, protocol, count, thinking=False):
    """One transcript per session of a server whose startup config pins an XML protocol
    (``protocol``: Qwen's or Gemma's turn grammar), else (). Startup only.

    Each transcript's grammar must be the pinned protocol's: a Gemma release renders
    with Gemma's grammar, every other release with Qwen's (renderers without a
    ``grammar`` attribute render Qwen's). ``thinking`` (the startup config's ``thinking``
    record, serve.py --enable-thinking): the transcripts with thinking on (either grammar)."""
    if protocol not in GRAMMARS:
        return ()
    require(type(thinking) is bool, 'thinking is a switch')
    options = dict(thinking=True) if thinking else {}
    transcripts = tuple(renderer.transcript(**options) for _ in range(count))
    require(all(getattr(t, 'grammar', QWEN_XML) is GRAMMARS[protocol] for t in transcripts),
            "the renderer's turn grammar differs from the pinned protocol")
    require(all(getattr(t, 'thinking', False) is thinking for t in transcripts),
            "the renderer's transcripts differ from the pinned thinking switch")
    return transcripts


class Transcript:
    """Server-owned token stream of one XML session, with upstream session semantics.

    ``encode(text)`` tokenizes without adding special tokens (special-token text
    inside is parsed, exactly as upstream ``_encode``); ``render_initial`` renders
    system + task through the chat template; ``decode`` skips special tokens.

    A request extends the session only if it repeats the previous request, then
    the reply (compared after ``strip()``), then one new user message; that is
    exactly when upstream's exact-ID render is a strict token extension.  Any
    other request (a format-error retry that drops the reply, compaction, a
    resend) makes upstream open a fresh cold session whose render re-encodes
    every assistant message from text.  ``allow_rebuild`` enables that path;
    otherwise it fails closed with ``SESSION_REBUILD_REQUIRED``.

    ``grammar``: the turn literals, ``QWEN_XML`` (the default) or ``GEMMA4_XML``.

    ``thinking`` (serve.py --enable-thinking; remote_ra_live_v2.thinking): the grammar's
    thinking-on header; a committed reply is split into reasoning and content (Qwen: at the
    first ``</think>`` of its text; Gemma: ``thinking.Gemma4Split`` on its ids, ``splitter``);
    an assistant message may carry its reasoning, and the previous reply must come back with
    both (``thinking.matches``) to extend the session; the previous request + one user message
    is a rollback (``rolls_back``: the exact stream up to the dropped reply's header, then that
    message; keep-rebuild-1116); any other rebuild renders every later message as the template
    does with preserve_thinking (``grammar.keep_message_text``; before keep-rebuild-1116 from
    its content text only); ``previous_action`` is the content.
    """
    splitter = None          # thinking on, Gemma: the reply split on ids (thinking.Gemma4Split)

    def __init__(self, *, encode, render_initial, decode, stop_ids, allow_rebuild=False, grammar=QWEN_XML,
                 thinking=False):
        require(callable(encode) and callable(render_initial) and callable(decode)
                and type(stop_ids) is frozenset and stop_ids, 'transcript tokenizer callbacks required')
        require(grammar is QWEN_XML or grammar is GEMMA4_XML, 'known XML grammar required')
        require(type(thinking) is bool, 'thinking is a switch')
        self.encode, self.render_initial, self.decode = encode, render_initial, decode
        self.stop_ids, self.allow_rebuild, self.grammar = stop_ids, allow_rebuild, grammar
        self.thinking = thinking
        self.header = tuple(self.encode(grammar.thinking_header if thinking else grammar.header))
        self._message_text = grammar.thinking_message_text if thinking else grammar.message_text
        if thinking:
            self.splitter = _thinking.splitter(grammar.profile, encode, decode, stop_ids)
        self.sessions = 0        # sessions opened, including rebuilds
        self._reset()

    def _reset(self):
        self.history = None      # messages of the last committed request
        self.turns = []          # dict(observation, text, ids) per committed turn
        self.stream = ()         # every token of this session so far
        self.initial_length = 0  # template tokens of system + task (mandatory prefix)
        # (role, start, end, content) of every message after system + task that the first request of this
        # session encoded from text (a rebuild's history; empty on a fresh session): the query hook's input
        # (ledger-policy-1112); the tokens are unchanged. Thinking on (keep-rebuild-1116): (role, start, end,
        # content, reasoning or None) of every message the session's first request holds after system + task
        # (a rollback's history keeps its exact ids; a rebuild's is the keep render).
        self.rebuilt = ()
        # Thinking on (keep-rebuild-1116): (role, start, end, content, reasoning or None) of every message of
        # ``history`` after system + task in ``stream``, and the start of the last reply's header (the end of
        # ``history``'s render): what a rollback keeps. Off: unused.
        self.spans = ()
        self.reply_start = None
        self.pending = None

    @property
    def turn(self):
        return len(self.turns)

    def extends(self, messages):
        # The client echoes the reply as it stored it (possibly trimmed); later
        # requests must repeat that echo verbatim.
        history = self.history
        if self.thinking:
            # Thinking on: the reply's content and reasoning as returned (thinking.matches).
            return (history is not None and len(messages) == len(history) + 2
                    and messages[:len(history)] == history
                    and messages[-2]['role'] == 'assistant'
                    and _thinking.matches(messages[-2], self.turns[-1]))
        return (history is not None and len(messages) == len(history) + 2
                and messages[:len(history)] == history
                and messages[-2]['role'] == 'assistant'
                and messages[-2]['content'].strip() == self.turns[-1]['text'].strip())

    def rolls_back(self, messages):
        """Thinking on (keep-rebuild-1116): ``messages`` repeat the previous request exactly (reasoning
        fields included) and add one user message, i.e. the client dropped the previous reply (Mini-SWE-Agent
        after a format error: a reply cut inside its reasoning, or one without an action). The session's
        stream up to that reply's header is then the render of the repeated messages, token for token."""
        history = self.history
        return (self.thinking and self.turn > 0 and history is not None and self.reply_start is not None
                and len(messages) == len(history) + 1 and messages[:len(history)] == history
                and messages[-1]['role'] == 'user' and len(self.spans) == len(history) - 2)

    def drops_reply(self, messages):
        """goal-rollback-1116: ``messages`` repeat the previous request and add one user message, i.e. the
        client dropped the previous reply (Mini-SWE-Agent after a format error), thinking on or off. On it
        the session's Goal goes on from the previous turn (query_selection.QuerySelectionAuthority
        .reset_session); the tokens are unchanged: with thinking on it is exactly a ``rolls_back`` render,
        off an ordinary rebuild that re-encodes every reply from its text."""
        history = self.history
        return (self.turn > 0 and history is not None and len(messages) == len(history) + 1
                and messages[:len(history)] == history and messages[-1]['role'] == 'user')

    def _later_text(self, message):
        """A message after system + task as a fresh or rebuilt session encodes it: thinking on, the
        template's keep render (grammar.keep_message_text); off, upstream's text (unchanged)."""
        if self.thinking:
            return self.grammar.keep_message_text(message)
        return self._message_text(message['role'], message['content'])

    def render(self, messages):
        """Prompt tokens, stable prefix end and observation span for the next request."""
        require(self.pending is None, 'previous XML turn neither committed nor abandoned')
        validate_messages(messages, thinking=self.thinking)
        if self.thinking:
            messages = [_thinking.plain(m, True) for m in messages]
        else:
            messages = [dict(role=m['role'], content=m['content']) for m in messages]
        extends = self.turn > 0 and self.extends(messages)
        rollback = not extends and self.rolls_back(messages)
        history = _thinking.history(self, messages, extends, rollback) if self.thinking else None
        rebuild = self.turn > 0 and not extends
        # goal-rollback-1116: thinking on, a rollback; off, a rebuild of the same message shape.
        drops = rollback if self.thinking else (rebuild and self.drops_reply(messages))
        if rebuild:
            require(self.allow_rebuild, 'SESSION_REBUILD_REQUIRED: request is not an append-only '
                    'extension; upstream opens a fresh cold session here')
        # A rollback or rebuild resets the session at render time; ``abandon`` (a failure before the engine
        # opens the new namespace) puts it back, so a retry renders against the session the engine still holds.
        restore = {name: getattr(self, name) for name in self.RESTORED} if rebuild else None
        extra = {}
        if rollback:
            # keep-rebuild-1116: the session's exact tokens of the repeated request (its render up to the
            # dropped reply's header, every earlier reasoning and generated id kept), then the new user
            # message. The engine re-prefills it as a rebuild (a fresh namespace at local turn 0).
            kept, spans, initial_length = self.stream[:self.reply_start], self.spans, self.initial_length
            self._reset()
            observed = tuple(self.encode(self._later_text(messages[-1])))
            complete = kept + observed
            rebuilt = spans + (('user', len(kept), len(complete), messages[-1]['content'], None),)
            tokens, stable, span = complete + self.header, len(complete), None
            self.initial_length = initial_length
            self.rebuilt = rebuilt
            extra = {_thinking.KEPT_TOKENS: len(kept), _thinking.REENCODED: 0}
        else:
            if rebuild:
                self._reset()
            if self.turn == 0:
                # Fresh or rebuilt session: template for system + task, every later message (assistants
                # too) encoded from text, as upstream _render does without exact serializations; thinking
                # on, as the template renders it with preserve_thinking (keep-rebuild-1116).
                initial = tuple(self.render_initial([dict(m) for m in messages[:2]]))
                require(initial, 'initial template render is empty')
                complete, rebuilt = initial, []
                for message in messages[2:]:
                    start = len(complete)
                    complete += tuple(self.encode(self._later_text(message)))
                    row = (message['role'], start, len(complete), message['content'])
                    rebuilt.append(row + (message.get(_thinking.KEY),) if self.thinking else row)
                tokens, stable, span = complete + self.header, len(complete), None
                self.initial_length = len(initial)
                self.rebuilt = rebuilt = tuple(rebuilt)
                if rebuild:
                    extra = {_thinking.KEPT_TOKENS: 0,
                             _thinking.REENCODED: sum(m['role'] == 'assistant' for m in messages[2:])}
            else:
                observed = tuple(self.encode(self._later_text(messages[-1]) if self.thinking else
                                             self._message_text('user', messages[-1]['content'])))
                stable = len(self.stream)
                span = (stable, stable + len(observed))
                tokens = self.stream + observed + self.header
                if self.thinking:
                    # The reply as the server split it (its turn record), as the query hook's live rows hold it.
                    reply = self.turns[-1]
                    rebuilt = self.spans + (
                        ('assistant', self.reply_start, stable, reply['content'], reply['reasoning']),
                        ('user', stable, span[1], messages[-1]['content'], None))
        self.pending = dict(messages=messages, tokens=tokens, observation=messages[-1]['content'],
                            rebuild=rebuild, drops_reply=drops)
        if restore is not None:
            self.pending['restore'] = restore
        if self.thinking:
            self.pending.update(history=history, spans=rebuilt, **extra)
        return tokens, stable, span

    # What a rollback or rebuild render resets and ``abandon`` restores.
    RESTORED = ('history', 'turns', 'stream', 'initial_length', 'rebuilt', 'spans', 'reply_start')

    def abandon(self):
        """Drop a render that will not be committed. A rollback or rebuild render reset the session; its
        state before the render comes back (the engine has not opened a new namespace for it)."""
        pending, self.pending = self.pending, None
        if pending is not None and 'restore' in pending:
            for name, value in pending['restore'].items():
                setattr(self, name, value)

    def release(self):
        """Session pool: the session closed for good; drop its history, token stream
        and any render a failed request left pending. Nothing renders on it again."""
        self._reset()

    def commit(self, prompt_tokens, generated):
        """Append the generated tokens and the upstream boundary; return the reply."""
        pending = self.pending
        require(pending is not None and tuple(prompt_tokens) == pending['tokens'],
                'committed prompt differs from the rendered XML turn')
        generated = tuple(generated)
        require(all(type(t) is int and t >= 0 for t in generated), 'generated token IDs required')
        boundary = tuple(self.encode(self.grammar.boundary_text(generated, self.stop_ids)))
        text = self.decode(list(generated))
        if not self.turns:
            self.sessions += 1
        header_start = len(pending['tokens']) - len(self.header)
        ids = pending['tokens'][header_start:] + generated + boundary
        turn = dict(observation=pending['observation'], text=text, ids=ids)
        committed = dict(text=text, boundary=boundary, generated=generated, rebuild=pending['rebuild'])
        if self.thinking:
            reasoning, content, closed, extra = _thinking.commit_split(self, text, generated)
            turn.update(reasoning=reasoning, content=content, closed=closed)
            committed.update(reasoning=reasoning, content=content, closed=closed, history=pending['history'], **extra)
            for key in (_thinking.KEPT_TOKENS, _thinking.REENCODED):
                if key in pending:
                    committed[key] = pending[key]       # keep-rebuild-1116: rollback and rebuild rows
            self.spans, self.reply_start = pending['spans'], header_start
        self.turns.append(turn)
        self.stream = pending['tokens'] + generated + boundary
        self.history = pending['messages']
        self.pending = None
        return committed

    def previous_action(self):
        """Upstream ``previous_action``: the decoded reply of this session's prior turn.
        Thinking on: its content (the action), without the reasoning (FLAGGED: a Goal input)."""
        if self.thinking:
            return self.turns[-1]['content'] if self.turns else None
        return self.turns[-1]['text'] if self.turns else None
