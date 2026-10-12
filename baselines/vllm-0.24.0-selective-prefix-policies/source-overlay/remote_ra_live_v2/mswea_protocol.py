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
"""
from __future__ import annotations

import re

from .contract import require

PROTOCOL = 'mswea_xml_v1'
MARKER = '</mswea_bash_command>'
GENERATION_HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
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


def check_sampling(params, stop_ids):
    """Upstream greedy generation: argmax, 1536 tokens, EOS/pad stop, no other controls.

    The action marker is not a vLLM stop string: vLLM checks stop strings in the
    frontend and aborts the engine request afterwards.  ``MarkerStop`` ends the
    request inside the engine instead.

    ``skip_special_tokens`` is not part of the contract: it only shapes vLLM's
    own detokenized text, which the route replaces with the transcript's
    upstream decode.  vLLM's Qwen3 parser engine sets it to False while
    rendering, after the route has fixed the request.
    """
    value = lambda name, default=None: getattr(params, name, default)
    stops = set(value('stop_token_ids') or ()) | {value('eos_token_id')}
    stops.discard(None)
    wrong = [name for name, ok in (
        ('n', value('n', 1) == 1), ('temperature', value('temperature') == 0),
        ('max_tokens', value('max_tokens') == MAX_TOKENS), ('min_tokens', value('min_tokens', 0) == 0),
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
    """Engine-side upstream stop: the decoded output contains the action marker."""

    def __init__(self, decode):
        require(callable(decode), 'decoder required')
        self.decode = decode

    def __call__(self, request):
        if MARKER not in self.decode(list(request.output_token_ids)):
            return False
        from vllm.v1.request import RequestStatus
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = MARKER
        return True


def boundary_text(generated, stop_ids):
    """Upstream ``_append_boundary``: EOS/pad already closed the turn."""
    return '\n' if generated and generated[-1] in stop_ids else '<|im_end|>\n'


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
    """

    def __init__(self, *, encode, render_initial, decode, stop_ids, allow_rebuild=False):
        require(callable(encode) and callable(render_initial) and callable(decode)
                and type(stop_ids) is frozenset and stop_ids, 'transcript tokenizer callbacks required')
        self.encode, self.render_initial, self.decode = encode, render_initial, decode
        self.stop_ids, self.allow_rebuild = stop_ids, allow_rebuild
        self.header = tuple(self.encode(GENERATION_HEADER))
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
                complete += tuple(self.encode(message_text(message['role'], message['content'])))
            tokens, stable, span = complete + self.header, len(complete), None
            self.initial_length = len(initial)
        else:
            observed = tuple(self.encode(message_text('user', messages[-1]['content'])))
            stable = len(self.stream)
            span = (stable, stable + len(observed))
            tokens = self.stream + observed + self.header
        self.pending = dict(messages=messages, tokens=tokens, observation=messages[-1]['content'],
                            rebuild=rebuild)
        return tokens, stable, span

    def abandon(self):
        self.pending = None

    def commit(self, prompt_tokens, generated):
        """Append the generated tokens and the upstream boundary; return the reply."""
        pending = self.pending
        require(pending is not None and tuple(prompt_tokens) == pending['tokens'],
                'committed prompt differs from the rendered XML turn')
        generated = tuple(generated)
        require(all(type(t) is int and t >= 0 for t in generated), 'generated token IDs required')
        boundary = tuple(self.encode(boundary_text(generated, self.stop_ids)))
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
