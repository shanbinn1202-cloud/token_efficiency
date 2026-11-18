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
"""
from __future__ import annotations

import re

from .context_limits import MAX_TOKENS_CHOICES
from .contract import require

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


def validate_messages(messages, problem_statement=None):
    """Upstream request shape: system, task, then user/assistant text, ending with a user."""
    require(type(messages) is list and 2 <= len(messages) <= MAX_MESSAGES,
            'XML protocol conversation required')
    for index, message in enumerate(messages):
        allowed = ('system',) if index == 0 else ('user',) if index == 1 else ('user', 'assistant')
        require(type(message) is dict and message.get('role') in allowed
                and type(message.get('content')) is str and set(message) <= {'role', 'content'},
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

    def __init__(self, decode, incremental=False, token_cache=False):
        """``token_cache`` (--decode-fastpath2, with ``incremental``): remember per token id
        whether the token decoded alone holds a '>' (a token decodes to the same text
        every time, so the answer is the one the decode gives)."""
        require(callable(decode), 'decoder required')
        require(type(incremental) is bool, 'incremental is a switch')
        require(type(token_cache) is bool and (incremental or not token_cache),
                'token_cache is a switch of the incremental check')
        self.decode = decode
        self.incremental = incremental
        self._checked = {}
        self._gt = {} if token_cache else None

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
        if n > 0 and self._checked.get(rid) == n - 1 and (
                '>' not in self.decode([tokens[-1]]) if self._gt is None else not self._closes(tokens[-1])):
            absent = True
        else:
            absent = MARKER not in self.decode(list(tokens))
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
        elif MARKER not in self.decode(list(request.output_token_ids)):
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
    header = GENERATION_HEADER

    @staticmethod
    def message_text(role, content):
        return message_text(role, content)

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
    append reference (job 3364469) is reproduced token for token from these literals."""
    protocol = GEMMA4_PROTOCOL
    header = GEMMA4_HEADER

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
    def boundary_text(generated, stop_ids):
        return '\n' if generated and generated[-1] in stop_ids else GEMMA4_END_OF_TURN


QWEN_XML = QwenGrammar()
GEMMA4_XML = Gemma4Grammar()
GRAMMARS = {PROTOCOL: QWEN_XML, GEMMA4_PROTOCOL: GEMMA4_XML}


def xml_transcripts(renderer, protocol, count):
    """One transcript per session of a server whose startup config pins an XML protocol
    (``protocol``: Qwen's or Gemma's turn grammar), else (). Startup only.

    Each transcript's grammar must be the pinned protocol's: a Gemma release renders
    with Gemma's grammar, every other release with Qwen's (renderers without a
    ``grammar`` attribute render Qwen's)."""
    if protocol not in GRAMMARS:
        return ()
    transcripts = tuple(renderer.transcript() for _ in range(count))
    require(all(getattr(t, 'grammar', QWEN_XML) is GRAMMARS[protocol] for t in transcripts),
            "the renderer's turn grammar differs from the pinned protocol")
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
    """

    def __init__(self, *, encode, render_initial, decode, stop_ids, allow_rebuild=False, grammar=QWEN_XML):
        require(callable(encode) and callable(render_initial) and callable(decode)
                and type(stop_ids) is frozenset and stop_ids, 'transcript tokenizer callbacks required')
        require(grammar is QWEN_XML or grammar is GEMMA4_XML, 'known XML grammar required')
        self.encode, self.render_initial, self.decode = encode, render_initial, decode
        self.stop_ids, self.allow_rebuild, self.grammar = stop_ids, allow_rebuild, grammar
        self.header = tuple(self.encode(grammar.header))
        self.sessions = 0        # sessions opened, including rebuilds
        self._reset()

    def _reset(self):
        self.history = None      # messages of the last committed request
        self.turns = []          # dict(observation, text, ids) per committed turn
        self.stream = ()         # every token of this session so far
        self.initial_length = 0  # template tokens of system + task (mandatory prefix)
        self.pending = None

    @property
    def turn(self):
        return len(self.turns)

    def extends(self, messages):
        # The client echoes the reply as it stored it (possibly trimmed); later
        # requests must repeat that echo verbatim.
        history = self.history
        return (history is not None and len(messages) == len(history) + 2
                and messages[:len(history)] == history
                and messages[-2]['role'] == 'assistant'
                and messages[-2]['content'].strip() == self.turns[-1]['text'].strip())

    def render(self, messages):
        """Prompt tokens, stable prefix end and observation span for the next request."""
        require(self.pending is None, 'previous XML turn neither committed nor abandoned')
        validate_messages(messages)
        messages = [dict(role=m['role'], content=m['content']) for m in messages]
        rebuild = self.turn > 0 and not self.extends(messages)
        if rebuild:
            require(self.allow_rebuild, 'SESSION_REBUILD_REQUIRED: request is not an append-only '
                    'extension; upstream opens a fresh cold session here')
            self._reset()
        if self.turn == 0:
            # Fresh or rebuilt session: template for system + task, every later
            # message (assistants too) encoded from text, as upstream _render
            # does without exact serializations.
            initial = tuple(self.render_initial([dict(m) for m in messages[:2]]))
            require(initial, 'initial template render is empty')
            complete = initial
            for message in messages[2:]:
                complete += tuple(self.encode(self.grammar.message_text(message['role'], message['content'])))
            tokens, stable, span = complete + self.header, len(complete), None
            self.initial_length = len(initial)
        else:
            observed = tuple(self.encode(self.grammar.message_text('user', messages[-1]['content'])))
            stable = len(self.stream)
            span = (stable, stable + len(observed))
            tokens = self.stream + observed + self.header
        self.pending = dict(messages=messages, tokens=tokens, observation=messages[-1]['content'],
                            rebuild=rebuild)
        return tokens, stable, span

    def abandon(self):
        self.pending = None

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
        self.turns.append(dict(observation=pending['observation'], text=text, ids=ids))
        self.stream = pending['tokens'] + generated + boundary
        self.history = pending['messages']
        self.pending = None
        return dict(text=text, boundary=boundary, generated=generated, rebuild=pending['rebuild'])

    def previous_action(self):
        """Upstream ``previous_action``: the decoded reply of this session's prior turn."""
        return self.turns[-1]['text'] if self.turns else None
