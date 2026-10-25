"""kv-session protocol (MemoryAgentBench, serve.py --protocol kv-session): token transcript,
request mapping, Goal inputs and client sessions. CPU only.

The client here is this project's own re-implementation of the KV-session client
behaviour (plan memoryagentbench-kvsession-plan.md, sections 2.3-2.6): it renders
the pinned Qwen3.6 chat template itself, sends only the new tail after the first
turn (the previous reply's closing delimiter, with a non-EOS last sample restored
in front, the new messages and the next generation header), and parses replies.
The agent loop is a re-implementation of the read_next MemAgent loop (plan 1.2):
history chunks arrive as tool messages, the question as a user message in the same
request as the last chunk, a reply without a tool call is a format error until
every chunk is read, and the first reply without one after that is the answer.

Two tokenizers: a character-level stand-in that keeps the pinned added-token ids
(always available; the template is rendered with jinja2 like Transformers does),
and the pinned Qwen3.6 tokenizer files (source-overlay/remote_ra_linux/
model-evidence) with Transformers' own apply_chat_template, when Transformers is
installed (the remote-ra-tok venv). Nothing here is GPU evidence.
"""
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))

from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402

EVIDENCE = ROOT / 'source-overlay' / 'remote_ra_linux' / 'model-evidence'
CONFIG = json.loads((EVIDENCE / 'tokenizer-config.json').read_text(encoding='utf-8'))
TEMPLATE = CONFIG['chat_template']
ADDED = {row['content']: (int(k), row['special']) for k, row in CONFIG['added_tokens_decoder'].items()}
IM_START, IM_END, PAD = ADDED['<|im_start|>'][0], ADDED['<|im_end|>'][0], ADDED['<|endoftext|>'][0]
HAVE_TRANSFORMERS = bool(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers'))

BASH = {'type': 'function', 'function': {'name': 'bash', 'description': 'Run one shell command',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string',
                       'description': 'the command'}}, 'required': ['command']}}}
SYSTEM = ('You answer questions about a long history that you read in chunks. Call the bash tool '
          'with the command read_next to read the next chunk; answer only after the last chunk.')
FIRST_USER = 'Read the whole history with read_next, then answer the question that follows it.'
NUDGE = 'History reading is not complete. Call bash with {"command": "read_next"} to read the next chunk.'
CALL = '<tool_call>\n<function=bash>\n<parameter=command>\n%s\n</parameter>\n</function>\n</tool_call>'
END_NOTE = '\n[End of the history. The question follows in the next user message.]'


# ---- tokenizers and the template ----------------------------------------------------------------

def jinja_render(template, messages, tools=None, add_generation_prompt=False, **kwargs):
    """The template rendered as Transformers renders chat templates (sandboxed jinja2,
    trim/lstrip blocks, loop controls, its tojson and raise_exception)."""
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    return env.from_string(template).render(messages=messages, tools=tools,
                                            add_generation_prompt=add_generation_prompt, **kwargs)


class CharTokenizer:
    """Character-level stand-in keeping the pinned ids of every added token (matched in
    text like the HF fast tokenizer); characters are ids above every real one."""
    BASE = 300000
    vocab = BASE + 0x110000

    def __init__(self):
        self.added = sorted(ADDED, key=len, reverse=True)
        self.names = {i: (t, special) for t, (i, special) in ADDED.items()}

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for token in self.added:
                if text.startswith(token, i):
                    ids.append(ADDED[token][0])
                    i += len(token)
                    break
            else:
                ids.append(ord(text[i]) + self.BASE)
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for t in ids:
            if t in self.names:
                token, special = self.names[t]
                out.append('' if special and skip_special_tokens else token)
            else:
                out.append(chr(t - self.BASE) if t >= self.BASE else '<id%d>' % t)
        return ''.join(out)


class Kit:
    """One tokenizer and the template renderer its tests use."""

    def __init__(self, name, encode, decode, render, vocab, im_start=IM_START, eos=IM_END, pad=PAD):
        self.name, self._encode, self._decode, self._render, self.vocab = name, encode, decode, render, vocab
        self.im_start, self.eos, self.pad = im_start, eos, pad      # eos is <|im_end|>, as in Qwen3.6
        self.stop_ids = frozenset({eos, pad})

    def encode(self, text):
        return list(self._encode(text))

    def decode(self, ids):
        return self._decode(list(ids))

    def render(self, messages, tools=None, generation=False, **kwargs):
        return self._render(messages, tools, generation, kwargs)

    def header(self, thinking='on'):
        return tuple(self.encode(K.HEADERS[thinking]))

    def transcript(self, thinking='on'):
        return K.KvTranscript(header=self.header(thinking), im_start=self.im_start, vocab=self.vocab,
                              decode=self.decode, stop_ids=self.stop_ids)


def fake_kit():
    tok = CharTokenizer()
    return Kit('char', tok.encode, tok.decode,
               lambda messages, tools, generation, kwargs: jinja_render(
                   TEMPLATE, messages, tools, generation, **kwargs), tok.vocab)


_REAL = []


def real_kit():
    """The pinned Qwen3.6 tokenizer with Transformers' apply_chat_template, or None."""
    if not HAVE_TRANSFORMERS:
        return None
    if not _REAL:
        from transformers import AutoTokenizer
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for target, name in (('config.json', 'model-config.json'), ('tokenizer.json', 'tokenizer.json'),
                                 ('tokenizer_config.json', 'tokenizer-config.json')):
                shutil.copyfile(EVIDENCE / name, work / target)
            tok = AutoTokenizer.from_pretrained(str(work), local_files_only=True)
        assert tok.chat_template == TEMPLATE
        assert tok.convert_tokens_to_ids('<|im_start|>') == IM_START and tok.eos_token_id == IM_END

        def render(messages, tools, generation, kwargs):
            return tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                           add_generation_prompt=generation, **kwargs)
        _REAL.append(Kit('qwen3.6', lambda text: tok.encode(text, add_special_tokens=False),
                         lambda ids: tok.decode(ids, skip_special_tokens=False), render, len(tok)))
    return _REAL[0]


# ---- the client (re-implemented) ----------------------------------------------------------------

class ContextLimit(Exception):
    """The client's mapping of the server's context-limit refusal (the agent stops)."""


class ServerError(RuntimeError):
    def __init__(self, status, body):
        super().__init__('KV session server: %d %s' % (status, body))
        self.status, self.body = status, body


def normalize(messages):
    """The client's message normalization: JSON copies, tool-call arguments as objects."""
    messages = json.loads(json.dumps(messages))
    for m in messages:
        m.pop('extra', None)
        for call in m.get('tool_calls') or ():
            if isinstance(call['function'].get('arguments'), str):
                call['function']['arguments'] = json.loads(call['function']['arguments'])
    return messages


def parse_reply(text, finish_reason):
    """The client's reply parse: reasoning before </think>, XML tool calls, the rest as content."""
    reasoning, content = '', text
    if '</think>' in content:
        reasoning, content = content.split('</think>', 1)
        reasoning = reasoning.split('<think>')[-1].strip()
    calls = []
    for body in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.S):
        name = re.search(r'<function=([^>\n]+)>', body)
        args = {k: v.strip('\n') for k, v in re.findall(r'<parameter=([^>\n]+)>(.*?)</parameter>', body, re.S)}
        calls.append({'id': 'call_' + uuid.uuid4().hex[:12], 'type': 'function',
                      'function': {'name': name.group(1) if name else '', 'arguments': json.dumps(args)}})
    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.S)
    content = content.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    message = {'role': 'assistant', 'content': content, 'reasoning_content': reasoning}
    if calls:
        message['tool_calls'] = calls
    return message, ('tool_calls' if calls else finish_reason)


class KvClient:
    """One agent's KV session: the request construction of plan 2.3-2.5, re-implemented.

    ``send(path, body)`` returns (HTTP status, JSON body)."""

    def __init__(self, kit, send, *, tools=None, thinking=True, budget=32768, max_tokens=4096, new_id=None):
        self.kit, self.send, self.tools = kit, send, tools
        self.template_kwargs = {'enable_thinking': thinking}
        self.budget, self.max_tokens = budget, max_tokens
        self.new_id = new_id or (lambda: uuid.uuid4().hex)     # a fresh uuid4 hex per session
        self.state, self.bodies, self.responses = None, [], []

    def render(self, messages, generation):
        return self.kit.render(messages, self.tools, generation, **self.template_kwargs)

    def build(self, messages, max_tokens=None):
        """The complete body of the next request (the state is set before it is sent)."""
        normalized = normalize(messages)
        enc = self.kit.encode
        boundary = 0
        if self.state is None:
            stable, text = self.render(normalized, False), self.render(normalized, True)
            assert text.startswith(stable), 'unstable generation header'
            header, ids = enc(text[len(stable):]), enc(text)
            self.state = dict(id=self.new_id(), messages=[], last=[], prefix=len(ids) - len(header))
        else:
            previous = self.state['messages']
            assert normalized[:len(previous)] == previous, 'append-only history required'
            new = normalized[len(previous):]
            assert new and not any(m['role'] == 'assistant' for m in new[1:]), 'one reply, then feedback'
            rendered = (normalized if new[0]['role'] == 'assistant' else
                        normalized[:len(previous)] + [self.state['last_message']] + new)
            marker = 'KV_TAIL_MARKER_' + uuid.uuid4().hex
            marked = copy.deepcopy(rendered)
            marked[len(previous)] = {'role': 'assistant', 'content': marker}
            with_header, without = self.render(marked, True), self.render(marked, False)
            assert with_header.count(marker) == 1 and with_header.startswith(without)
            header = enc(with_header[len(without):])
            tail = with_header.split(marker, 1)[1]
            ids = enc(tail)
            assert '<|im_start|>' in tail, 'the new messages need an explicit boundary'
            boundary = len(enc(tail.split('<|im_start|>', 1)[0]))
            last = self.state['last'][-1:]
            if last and last[0] != self.kit.eos:
                ids = last + ids           # the uncomputed last sample, restored
                boundary += 1
        body = dict(method=K.METHOD, session_id=self.state['id'], input_ids=ids, header_tokens=len(header),
                    prefix_tokens=self.state['prefix'], budget=self.budget,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens, temperature=0, top_p=1.0,
                    top_k=-1, repetition_penalty=1.0, seed=42, messages=normalized, boundary_tokens=boundary)
        return body, normalized

    def accept(self, normalized, status, data):
        """The response of the last body: the reply message, or the client's exceptions."""
        if status != 200:
            if status == 400 and data.get('error') == K.CONTEXT_LIMIT:
                raise ContextLimit(data)
            raise ServerError(status, data)
        self.responses.append(data)
        self.state.update(messages=copy.deepcopy(normalized), last=list(data['output_ids']))
        message, _finish = parse_reply(data['text'], data['finish_reason'])
        self.state['last_message'] = message
        return message, data

    def complete(self, messages, max_tokens=None):
        """One request; returns the parsed assistant message and the response body."""
        body, normalized = self.build(messages, max_tokens)
        self.bodies.append(body)
        return self.accept(normalized, *self.send('complete', body))

    async def acomplete(self, messages, max_tokens=None):
        """``complete`` over an asynchronous ``send``."""
        body, normalized = self.build(messages, max_tokens)
        self.bodies.append(body)
        return self.accept(normalized, *(await self.send('complete', body)))

    def closed(self, status, data):
        if status != 200:
            raise ServerError(status, data)
        self.state = None
        return data

    def close(self):
        if self.state is not None:
            return self.closed(*self.send('close', {'session_id': self.state['id']}))
        return None

    async def aclose(self):
        if self.state is not None:
            return self.closed(*(await self.send('close', {'session_id': self.state['id']})))
        return None

    def stream(self):
        """The client's own construction of the server's T for its latest request."""
        return client_stream(self.bodies, self.responses[:len(self.bodies) - 1])


def client_stream(bodies, responses):
    """T of request len(bodies) - 1: every request's ids and every reply but its last
    sample, which the next request repeats in front."""
    assert len(bodies) == len(responses) + 1
    tokens = list(bodies[0]['input_ids'])
    for response, body in zip(responses, bodies[1:]):
        tokens += response['output_ids'][:-1] + body['input_ids']
    return tokens


# ---- the read_next agent loop (re-implemented) --------------------------------------------------

def tool_message(call, code, text):
    return dict(role='tool', tool_call_id=call['id'],
                content='<returncode>%d</returncode>\n<output>\n%s</output>' % (code, text))


def reply_text(kind, k=0, calls=1, reasoning=None):
    """A scripted reply: a read (n read_next calls), an invalid call, a format error or the answer."""
    reason = reasoning if reasoning is not None else 'Turn %d: I keep reading the history.' % k
    if kind == 'read':
        return reason + '\n</think>\n\n' + '\n'.join(CALL % 'read_next' for _ in range(calls))
    if kind == 'invalid':
        return reason + '\n</think>\n\n' + CALL % 'cat history.txt'
    if kind == 'format':
        return reason + '\n</think>\n\nI will continue reading the history.'
    return reason + '\n</think>\n\nAnswer: chunk 2 names the harbour city.'


class Agent:
    """MemAgent's read_next loop over ``chunks`` (plan 1.2)."""

    def __init__(self, client, chunks, question, *, system=SYSTEM, first_user=FIRST_USER):
        self.client, self.chunks, self.question = client, list(chunks), question
        self.messages = ([dict(role='system', content=system)] if system is not None else []) + [
            dict(role='user', content=first_user)]
        self.delivered, self.errors, self.answer, self.turns = 0, 0, None, []

    def step(self, max_tokens=None):
        return self.after(*self.client.complete(self.messages, max_tokens))

    async def astep(self, max_tokens=None):
        return self.after(*(await self.client.acomplete(self.messages, max_tokens)))

    def after(self, message, data):
        """The loop's reaction to one reply: an answer, a format error, or tool results."""
        self.turns.append(data)
        self.messages.append(message)
        calls = message.get('tool_calls') or []
        if not calls:
            if self.delivered == len(self.chunks):
                self.answer = message['content']
                return 'answer'
            self.errors += 1
            self.messages.append(dict(role='user', content=NUDGE))
            return 'format'
        self.errors = 0
        question = False
        for call in calls:
            args = json.loads(call['function']['arguments'])
            if call['function']['name'] == 'bash' and args.get('command') == 'read_next' \
                    and self.delivered < len(self.chunks):
                self.delivered += 1
                text = '[History chunk %d/%d]\n%s' % (self.delivered, len(self.chunks), self.chunks[self.delivered - 1])
                if self.delivered == len(self.chunks):
                    text += END_NOTE
                    question = True
                self.messages.append(tool_message(call, 0, text))
            else:
                self.messages.append(tool_message(call, 1, 'unknown command: only read_next is available'))
        if question:
            self.messages.append(dict(role='user', content=self.question))
        return 'read'


# ---- a protocol-level server: frontend and engine transcripts, scripted model ---------------------

class MiniServer:
    """The protocol pieces as the route and the engine use them, with a scripted model.

    Each slot has a frontend transcript (built from the body) and an engine transcript
    (re-derived from the source the frontend sends); both must agree on every turn."""

    def __init__(self, kit, *, slots=8, budget=32768, bound=4096, max_model_len=131072, thinking='on'):
        self.kit, self.budget, self.bound, self.max_model_len = kit, budget, bound, max_model_len
        self.registry = K.KvSessionRegistry(slots)
        self.frontend = [kit.transcript(thinking) for _ in range(slots)]
        self.engine = [kit.transcript(thinking) for _ in range(slots)]
        self.script = {}                # client session id -> list of (text, end) replies
        self.turns = []                 # one record per served request
        self.closes = []

    def info(self):
        return dict(method=K.METHOD, budget=self.budget, options={})

    def send(self, path, body):
        try:
            if path == 'close':
                return 200, self.close(K.check_close(body))
            return 200, self.complete(body)
        except K.KvSessionError as error:
            return error.status, error.body()

    def close(self, sid):
        answer = self.registry.closed_answer(sid)
        if answer is not None:
            return dict(answer, repeated=True)
        index = self.registry.slot(sid)
        if index is None:
            return dict(closed=True, session_id=sid, known=False)
        answer = dict(closed=True, session_id=sid, known=True, turns=self.frontend[index].turn)
        for transcript in (self.frontend[index], self.engine[index]):
            transcript.release()
        self.registry.mark_closed(sid, answer)
        self.closes.append(sid)
        return answer

    def generate(self, sid, effective):
        text, end = self.script[sid].pop(0)
        ids = self.kit.encode(text)
        if end == 'eos':
            ids = ids + [self.kit.eos]
        elif end == 'pad':
            ids = ids + [self.kit.pad]
        return ids[:effective]

    def complete(self, body):
        request = K.parse_complete(body, budget=self.budget, bound=self.bound, max_model_len=self.max_model_len)
        index = self.registry.resolve(request)
        front, engine = self.frontend[index], self.engine[index]
        before = front.stream
        front.check(request)
        effective = K.effective_max_tokens(request, front.prompt_length(request), self.max_model_len)
        first = front.turn == 0
        messages = K.new_messages(request, front.history, first)
        source = K.source(request, messages, K.objective_of(request.messages) if first else None)
        front.stage(request)
        tokens, stable, span = front.render()
        # The engine re-derives the prompt from its own stream and the source.
        echo = K.from_source(K.validate_source(json.loads(json.dumps(source))))
        etokens, estable, espan = engine.render(echo)
        assert (etokens, estable, espan) == (tokens, stable, span)
        context = K.goal_inputs(echo.messages, first)
        generated = self.generate(request.session_id, effective)
        committed = front.commit(tokens, generated, len(request.messages))
        engine.commit(etokens, generated)
        assert engine.stream == front.stream and engine.previous_action() == front.previous_action()
        assert engine.objective == front.objective
        finish = K.finish_reason(generated, front.stop_ids, effective)
        self.turns.append(dict(index=index, sid=request.session_id, before=before, tokens=tokens, stable=stable,
                               span=span, request=request, generated=generated, context=context,
                               objective=front.objective, effective=effective, source=source))
        return dict(text=committed['text'], output_ids=list(generated), prompt_tokens=len(tokens),
                    completion_tokens=len(generated), cached_tokens=len(tokens) - len(request.input_ids),
                    finish_reason=finish)


def episode_script(n_chunks, *, invalid_at=None, format_at=None, double_at=None, length_at=None):
    """Replies of one MemAgent case: a read per chunk (two chunks at ``double_at``), an
    invalid call and a format error where asked, a length-limited read, the answer."""
    replies, read, k = [], 0, 0
    while read < n_chunks:
        if k == invalid_at:
            replies.append(('invalid', 1, 'eos'))
        elif k == format_at:
            replies.append(('format', 1, 'eos'))
        else:
            calls = 2 if k == double_at and read + 2 <= n_chunks else 1
            replies.append(('read', calls, 'length' if k == length_at else 'eos'))
            read += calls
        k += 1
    replies.append(('answer', 0, 'eos'))
    return [(reply_text(kind, j, calls), end) for j, (kind, calls, end) in enumerate(replies)]


CHUNKS = ['The harbour city of Velmora was founded on a river delta.',
          'In chunk two the archive names Velmora as the harbour city.',
          'Merchants traded salt, amber and glass there for centuries.',
          'A storm in the third century flooded the lower town.',
          'The council rebuilt the quays with stone from the hills.']
QUESTION = 'Question: which harbour city does the history name? Answer with the city name.'


def play(server, kit, sid_script, agent, *, length_at=None):
    """Run ``agent`` to its answer; a length-limited reply's max_tokens is its own length."""
    k = 0
    while agent.answer is None:
        text, end = sid_script[k] if k < len(sid_script) else (None, None)
        max_tokens = len(kit.encode(text)) if end == 'length' else None
        agent.step(max_tokens)
        k += 1
    return agent


def start(server, kit, script, chunks=CHUNKS, question=QUESTION, tools=(BASH,), **kwargs):
    """A new agent whose session id the server scripts ``script`` against."""
    sid = uuid.uuid4().hex
    client = KvClient(kit, server.send, tools=list(tools) if tools else None, new_id=lambda: sid, **kwargs)
    server.script[sid] = list(script)
    return client, Agent(client, chunks, question), sid


# ---- tests ---------------------------------------------------------------------------------------

def body_for(kit, **changes):
    """A valid turn-0 body for a user-only conversation, with fields changed."""
    client = KvClient(kit, None)
    body, _ = client.build([dict(role='user', content='Reply OK.')], max_tokens=2)
    body.update(changes)
    return body


class ParseTests(unittest.TestCase):
    kit = fake_kit()

    def parse(self, body, budget=32768, bound=4096, max_model_len=131072):
        return K.parse_complete(body, budget=budget, bound=bound, max_model_len=max_model_len)

    def refused(self, kind, body, **kwargs):
        with self.assertRaises(K.KvSessionError) as caught:
            self.parse(body, **kwargs)
        self.assertEqual((caught.exception.kind, caught.exception.status), (kind, 400), str(caught.exception))
        self.assertEqual(caught.exception.body(), {'error': str(caught.exception), 'type': kind})
        return caught.exception

    def test_valid_turn_zero_body(self):
        body = body_for(self.kit)
        request = self.parse(body)
        self.assertTrue(request.first)
        self.assertEqual((request.session_id, request.input_ids, request.header_tokens, request.prefix_tokens,
                          request.boundary_tokens, request.max_tokens, request.seed),
                         (body['session_id'], tuple(body['input_ids']), len(self.kit.header()),
                          len(body['input_ids']) - len(self.kit.header()), 0, 2, 42))
        self.assertEqual(request.messages, tuple(body['messages']))
        # Neutral sampling in its other spellings, a null seed.
        for changes in (dict(temperature=0.0, top_p=1, top_k=0, repetition_penalty=1), dict(seed=None)):
            self.parse(dict(body, **changes))
        self.parse(dict(body, budget=None), budget=None)

    def test_refusals(self):
        body = body_for(self.kit)
        self.refused('BAD_REQUEST', [body])
        self.refused('BAD_REQUEST', {k: v for k, v in body.items() if k != 'seed'})
        self.refused('BAD_REQUEST', dict(body, stream=False))
        self.refused('METHOD_MISMATCH', dict(body, method='sliding_window'))
        for value in (16384, 32768.0, None):
            self.refused('BUDGET_MISMATCH', dict(body, budget=value))
        self.refused('BUDGET_MISMATCH', body, budget=None)
        for sid in ('X' * 32, body['session_id'][:31], body['session_id'].upper(), 7, None):
            self.refused('BAD_SESSION_ID', dict(body, session_id=sid))
        for ids in ([], [1.0], [-1], [True], 'abc', [1] * 20):
            self.refused('BAD_INPUT_IDS' if ids != [1] * 20 else 'BAD_INPUT_IDS', dict(body, input_ids=ids),
                         max_model_len=16)
        for changes in (dict(header_tokens=0), dict(prefix_tokens=0), dict(boundary_tokens=-1),
                        dict(header_tokens=True), dict(prefix_tokens=body['prefix_tokens'] + 1),
                        dict(boundary_tokens=2, input_ids=[1, 2] + list(self.kit.header()))):
            self.refused('BAD_GEOMETRY', dict(body, **changes))
        for value in (0, 4097, 2.0, None):
            self.refused('MAX_TOKENS', dict(body, max_tokens=value))
        self.refused('MAX_TOKENS', dict(body, max_tokens=2000), bound=1536)
        for changes in (dict(temperature=0.7), dict(top_p=0.9), dict(top_k=20), dict(repetition_penalty=1.05),
                        dict(temperature=False), dict(top_k=-1.0), dict(seed='42'), dict(seed=True)):
            self.refused('SAMPLING', dict(body, **changes))
        for messages in ([], 'x', [dict(role='robot', content='x')], [dict(role='user')],
                         [dict(role='tool', content=None)], [dict(role='user', content=3)]):
            self.refused('BAD_MESSAGES', dict(body, messages=messages))
        # An assistant message may come without content (the client drops None fields).
        self.parse(dict(body, messages=body['messages'] + [dict(role='assistant', tool_calls=[])]))

    def test_close_body_context_limit_and_finish(self):
        sid = uuid.uuid4().hex
        self.assertEqual(K.check_close({'session_id': sid}), sid)
        for bad in ({}, {'session_id': sid, 'x': 1}, {'session_id': 'nope'}, [sid]):
            with self.assertRaises(K.KvSessionError):
                K.check_close(bad)
        request = self.parse(body_for(self.kit, max_tokens=4096))
        self.assertEqual(K.effective_max_tokens(request, 1000, 131072), 4096)
        self.assertEqual(K.effective_max_tokens(request, 131000, 131072), 72)
        self.assertEqual(K.effective_max_tokens(request, 131071, 131072), 1)
        for length in (131072, 131073):
            with self.assertRaises(K.KvSessionError) as caught:
                K.effective_max_tokens(request, length, 131072)
            self.assertEqual(caught.exception.body(), {'error': 'Logical RoPE context limit reached',
                                                       'type': 'CONTEXT_LIMIT'})
        stops = frozenset({IM_END, PAD})
        self.assertEqual([K.finish_reason(g, stops, 4) for g in ([5, IM_END], [5, PAD], [5, 6, 7, IM_END],
                                                               [5, 6, 7, 8])], ['stop', 'stop', 'stop', 'length'])
        with self.assertRaises(LiveV2Error):
            K.finish_reason([5, 6], stops, 4)

    def test_slot_tasks_record_info_and_header(self):
        from remote_ra_live_v2.sessions import MAX_TASKS, validate_tasks
        from remote_ra_live_v2 import selection_contract as sc
        tasks = K.slot_tasks(K.MAX_SLOTS)
        self.assertEqual((K.MAX_SLOTS, K.DEFAULT_SLOTS), (MAX_TASKS, MAX_TASKS))
        self.assertEqual(validate_tasks(tasks, MAX_TASKS), tasks)
        self.assertEqual(tasks[7]['instance_id'], 'kv-session-slot-007')
        for bad in (0, 65, True, 2.0):
            with self.assertRaises(LiveV2Error):
                K.slot_tasks(bad)
        goal = sc.for_policy(sc.default_contract(16, active_ratio=None, active_token_budget=32768,
                                                 vector_format=sc.COMPACT_INT8_V1), 'goal_query')
        ratio = sc.for_policy(sc.default_contract(16), 'observation_query')
        fill = sc.for_policy(sc.default_contract(16, active_ratio=None, active_token_budget=32768,
                                                 random_fill=sc.random_fill_pin(7)), 'random_fill')
        self.assertEqual(K.record(contract=goal), dict(method='dram_backed_kv', budget=32768, thinking='on',
                                                       header_text='<|im_start|>assistant\n<think>\n', slots=64))
        self.assertEqual(K.record(contract=None, thinking='off', slots=3)['budget'], None)
        self.assertEqual(K.record(contract=ratio)['budget'], None)
        for kwargs in (dict(thinking='maybe'), dict(slots=0), dict(slots=65)):
            with self.assertRaises(LiveV2Error):
                K.record(contract=None, **kwargs)

        def config(policy, contract, **extra):
            value = dict(protocol=K.PROTOCOL, policy=policy, page_tokens=16, selection_contract=contract,
                         max_running_sessions=16, tasks=K.slot_tasks(64), task=K.slot_tasks(1)[0],
                         effective_server_config=dict(max_model_len=131072), max_tokens=4096,
                         kv_session=K.record(contract=contract))
            value.update(extra)
            return value
        for policy, contract in (('goal_query', goal), ('observation_query', ratio), ('random_fill', fill),
                                 ('full', None)):
            value = config(policy, contract)
            self.assertEqual(K.check_config(value), value['kv_session'])
            info = K.info(value)
            self.assertEqual(set(info), {'method', 'budget', 'options'})
            self.assertEqual((info['method'], info['budget']), ('dram_backed_kv', K.budget_of(contract)))
            options = info['options']
            self.assertEqual((options['policy'], options['max_tokens'], options['max_model_len'],
                              options['max_running_sessions'], options['slots'], options['thinking']),
                             (policy, 4096, 131072, 16, 64, 'on'))
            if contract is not None:
                self.assertEqual((options['projection_seed'], options['recent_turns'], options['goal_rewriter']),
                                 (20260904, 2, None))
                self.assertEqual(options['budget_mode'], 'fixed' if contract['active_token_budget'] else 'ratio')
            self.assertEqual(options.get('fill_seed'), 7 if policy == 'random_fill' else None)
            json.dumps(info, allow_nan=False)
        good = config('goal_query', goal)
        for bad in (dict(good, protocol='mswea_xml_v1'), dict(good, kv_session=dict(good['kv_session'], budget=1)),
                    dict(good, kv_session=dict(good['kv_session'], header_text='x')),
                    dict(good, kv_session=dict(good['kv_session'], extra=1)),
                    dict(good, tasks=K.slot_tasks(63)), {k: v for k, v in good.items() if k != 'max_running_sessions'}):
            with self.assertRaises(LiveV2Error):
                K.check_config(bad)
        # The header check renders the pinned template like the client does.
        render = lambda template, messages, kwargs: jinja_render(template, messages, None, **kwargs)
        for thinking in ('on', 'off'):
            self.assertEqual(K.check_header(render, TEMPLATE, thinking), K.HEADERS[thinking])
        with self.assertRaises(LiveV2Error):
            K.check_header(render, TEMPLATE.replace("'<think>\\n' }}", "'<think>\\n\\n' }}"), 'on')


class TranscriptTests(unittest.TestCase):
    kit = fake_kit()

    def first(self, transcript, body=None):
        body = body or body_for(self.kit)
        request = K.parse_complete(body, budget=32768, bound=4096, max_model_len=131072)
        return request, transcript.render(request)

    def test_geometry_of_turn_zero_and_later_turns(self):
        t = self.kit.transcript()
        H = len(t.header)
        request, (tokens, stable, span) = self.first(t)
        self.assertEqual((tokens, stable, span, t.initial_length), (request.input_ids, request.prefix_tokens,
                                                                   None, request.prefix_tokens))
        generated = self.kit.encode('\n</think>\n\nOK') + [IM_END]
        committed = t.commit(tokens, generated, 1)
        self.assertEqual((committed['text'], committed['stop'], t.turn, t.history, t.objective, t.sessions),
                         ('\n</think>\n\nOK<|im_end|>', True, 1, 1, 'Reply OK.', 1))
        self.assertEqual(t.stream, tokens + tuple(generated))
        # The next tail: <|im_end|>\n, then a user message, then the header.
        tail = self.kit.encode('<|im_end|>\n<|im_start|>user\nAgain.<|im_end|>\n') + list(t.header)
        nxt = K.KvRequest(request.session_id, tuple(tail), H, request.prefix_tokens, 2, 16, 42, ())
        self.assertEqual(t.prompt_length(nxt), len(t.stream) + len(tail) - 1)
        tokens2, stable2, span2 = t.render(nxt)
        C = len(t.stream) - 1
        N = len(tokens2)
        self.assertEqual(tokens2, t.stream + tuple(tail[1:]))
        self.assertEqual((stable2, span2), (C + 2, (C + 2, N - H)))
        self.assertTrue(C < stable2 < span2[1] < N)
        self.assertEqual((tokens2[C], tokens2[stable2], tokens2[span2[1]:]), (IM_END, IM_START, t.header))
        self.assertEqual(self.kit.decode(tokens2[span2[0]:span2[1]]), '<|im_start|>user\nAgain.<|im_end|>\n')
        # A length-limited reply: its last sample restored in front, boundary 3.
        t.commit(tokens2, self.kit.encode('abc'), 3)
        tail3 = [self.kit.encode('c')[0]] + self.kit.encode('<|im_end|>\n<|im_start|>user\nX<|im_end|>\n') + list(t.header)
        tokens3, stable3, span3 = t.render(K.KvRequest(request.session_id, tuple(tail3), H, request.prefix_tokens,
                                                       3, 16, 42, ()))
        C3 = len(t.stream) - 1
        self.assertEqual((tokens3[C3], stable3, tokens3[stable3]), (self.kit.encode('c')[0], C3 + 3, IM_START))
        self.assertEqual(t.previous_action(), 'abc')

    def test_refusals_change_nothing(self):
        t = self.kit.transcript()
        H = len(t.header)
        request, (tokens, _, _) = self.first(t)
        t.commit(tokens, [5, IM_END], 1)
        stream = t.stream
        good = self.kit.encode('<|im_end|>\n<|im_start|>user\nAgain.<|im_end|>\n') + list(t.header)

        def req(ids, boundary=2, prefix=request.prefix_tokens, header=H):
            return K.KvRequest(request.session_id, tuple(ids), header, prefix, boundary, 16, 42, ())
        cases = (('HEADER_MISMATCH', req(good, header=H - 1)), ('HEADER_MISMATCH', req(good[:-1] + [7])),
                 ('SESSION_STATE', req(good, boundary=0)), ('PREFIX_MISMATCH', req(good, prefix=request.prefix_tokens + 1)),
                 ('CONTINUATION_MISMATCH', req([PAD] + good[1:])), ('BOUNDARY_MISMATCH', req(good, boundary=3)),
                 ('BAD_INPUT_IDS', req(good[:3] + [self.kit.vocab] + good[4:])))
        for kind, bad in cases:
            with self.assertRaises(K.KvSessionError) as caught:
                t.render(bad)
            self.assertEqual(caught.exception.kind, kind)
            self.assertEqual((t.stream, t.pending, t.turn), (stream, None, 1))
        # A session's first request must be a turn-0 body; its ids below the vocabulary.
        fresh = self.kit.transcript()
        with self.assertRaises(K.KvSessionError) as caught:
            fresh.render(req(good))
        self.assertEqual(caught.exception.kind, 'SESSION_STATE')
        ids = list(request.input_ids)
        ids[0] = self.kit.vocab + 5
        with self.assertRaises(K.KvSessionError):
            fresh.render(K.KvRequest(request.session_id, tuple(ids), H, request.prefix_tokens, 0, 2, 42, ()))
        # Render, abandon, render again; a pending render refuses a second one and a foreign commit.
        tokens2, _, _ = t.render(req(good))
        with self.assertRaises(LiveV2Error):
            t.render(req(good))
        with self.assertRaises(LiveV2Error):
            t.commit(tokens2[:-1], [5, IM_END])
        with self.assertRaises(LiveV2Error):
            t.commit(tokens2, [])
        t.abandon()
        self.assertEqual(t.render(req(good))[0], tokens2)
        # The controller's capacity-wait checkpoint restores exactly these attributes.
        from remote_ra_live_v2.controller import TRANSCRIPT_STATE
        saved = {name: (list(t.turns) if name == 'turns' else getattr(t, name)) for name in TRANSCRIPT_STATE}
        t.abandon()
        for name, value in saved.items():
            setattr(t, name, value)
        self.assertTrue(t.extends(None) and t.extends([1]))
        t.release()
        self.assertEqual((t.stream, t.turns, t.pending, t.history, t.objective, t.staged), ((), [], None, 0, None, None))

    def test_stage_then_render(self):
        t = self.kit.transcript()
        request = K.parse_complete(body_for(self.kit), budget=32768, bound=4096, max_model_len=131072)
        t.stage(request)
        self.assertIs(t.staged, request)
        tokens, _, _ = t.render()
        with self.assertRaises(LiveV2Error):
            t.stage(request)
        t.commit(tokens, [5, IM_END])
        self.assertIsNone(t.staged)
        with self.assertRaises(LiveV2Error):
            t.stage('not a request')
        with self.assertRaises(LiveV2Error):
            K.KvTranscript(header=(), im_start=1, vocab=10, decode=str, stop_ids=frozenset({1}))


class RegistryTests(unittest.TestCase):
    kit = fake_kit()

    def request(self, first=True):
        body = body_for(self.kit, session_id=uuid.uuid4().hex)
        request = K.parse_complete(body, budget=32768, bound=4096, max_model_len=131072)
        return request if first else K.KvRequest(request.session_id, request.input_ids, request.header_tokens,
                                                 request.prefix_tokens, 2, 16, 42, ())

    def test_binding_closing_and_the_lifetime(self):
        registry = K.KvSessionRegistry(3)
        a, b = self.request(), self.request()
        self.assertEqual((registry.resolve(a), registry.resolve(b), registry.resolve(a)), (0, 1, 0))
        with self.assertRaises(K.KvSessionError) as caught:
            registry.resolve(self.request(first=False))
        self.assertEqual(caught.exception.kind, 'SESSION_UNKNOWN')
        self.assertEqual((registry.slot(a.session_id), registry.slot('f' * 32)), (0, None))
        registry.mark_closed(a.session_id, dict(closed=True))
        self.assertEqual((registry.slot(a.session_id), registry.closed_answer(a.session_id)), (None, dict(closed=True)))
        with self.assertRaises(K.KvSessionError) as caught:
            registry.resolve(a)                 # a closed id is never served again, not even turn 0
        self.assertEqual(caught.exception.kind, 'SESSION_CLOSED')
        self.assertEqual(registry.resolve(self.request()), 2)
        with self.assertRaises(K.KvSessionError) as caught:
            registry.resolve(self.request())    # every slot used once: the lifetime is over
        self.assertEqual(caught.exception.kind, 'SESSION_LIMIT')
        self.assertEqual(registry.next, 3)
        for bad in (0, 65):
            with self.assertRaises(LiveV2Error):
                K.KvSessionRegistry(bad)


class GoalInputTests(unittest.TestCase):
    def test_each_turn_type(self):
        call = dict(id='c1', type='function', function=dict(name='bash', arguments={'command': 'read_next'}))
        reply = dict(role='assistant', content='', reasoning_content='r', tool_calls=[call])
        first = [K._plain(m) for m in (dict(role='system', content=SYSTEM), dict(role='user', content=FIRST_USER))]
        self.assertEqual(K.objective_of(first), FIRST_USER)
        self.assertEqual(K.goal_inputs(first, True), dict(observation=SYSTEM + '\n' + FIRST_USER,
            message_source='user', explicit_requirement=None, structured_state=dict(tool_name=None, return_code=None)))
        chunk = tool_message(call, 0, '[History chunk 1/5]\nText')
        read = [K._plain(m) for m in (reply, chunk)]
        self.assertEqual(read[0], dict(role='assistant', content='', tool_names=['bash']))
        self.assertEqual(K.goal_inputs(read, False), dict(observation=chunk['content'], message_source='tool',
            explicit_requirement=None, structured_state=dict(tool_name='bash', return_code=0)))
        invalid = [K._plain(m) for m in (reply, tool_message(call, 1, 'unknown command'))]
        self.assertEqual(K.goal_inputs(invalid, False)['structured_state'], dict(tool_name='bash', return_code=1))
        nudge = [K._plain(m) for m in (dict(role='assistant', content='I continue.'), dict(role='user', content=NUDGE))]
        self.assertEqual(K.goal_inputs(nudge, False), dict(observation=NUDGE, message_source='user',
            explicit_requirement=NUDGE, structured_state=dict(tool_name=None, return_code=None)))
        last = tool_message(call, 0, '[History chunk 5/5]\nText' + END_NOTE)
        question = [K._plain(m) for m in (reply, last, dict(role='user', content=QUESTION))]
        self.assertEqual(K.goal_inputs(question, False), dict(observation=last['content'] + '\n' + QUESTION,
            message_source='user', explicit_requirement=QUESTION, structured_state=dict(tool_name='bash', return_code=0)))
        two = [K._plain(m) for m in (dict(reply, tool_calls=[call, dict(call, id='c2')]), chunk,
                                     tool_message(dict(call, id='c2'), 0, 'chunk 2'))]
        self.assertEqual(K.goal_inputs(two, False)['message_source'], 'tool')
        # new_messages: the client's own rule (one reply, then feedback only).
        request = K.KvRequest('a' * 32, (1,), 1, 1, 2, 1, None, tuple([dict(role='user', content='q'), reply, chunk]))
        self.assertEqual(K.new_messages(request, 1, False), [K._plain(reply), K._plain(chunk)])
        for history, messages in ((3, request.messages), (0, request.messages + (reply,))):
            with self.assertRaises(K.KvSessionError):
                K.new_messages(K.KvRequest('a' * 32, (1,), 1, 1, 2, 1, None, tuple(messages)), history, False)
        with self.assertRaises(K.KvSessionError):
            K.new_messages(request, 0, True)
        # The engine source: validated, the objective at turn 0 only.
        req0 = K.KvRequest('a' * 32, (1, 2, 3), 1, 2, 0, 1, None, tuple(first))
        value = K.source(req0, first, FIRST_USER)
        self.assertEqual(K.from_source(value).objective, FIRST_USER)
        for bad in (dict(value, objective=None), dict(value, extra=1), dict(value, messages=[dict(role='x', content='')]),
                    dict(value, input_ids=[]), dict(value, session_id='A' * 32)):
            with self.assertRaises(LiveV2Error):
                K.validate_source(bad)
        with self.assertRaises(LiveV2Error):
            K.validate_source(dict(value, input_ids=[1] * (K.source_bytes() // 2)))
        import runtime_renderer
        self.assertEqual(K.source_bytes(), runtime_renderer.MAX_SOURCE_BYTES)


class EpisodeChecks:
    """One served turn, as the transcript should map it (geometry, tokens, Goal inputs)."""

    def check_turn(self, kit, turn, client_stream):
        request, tokens, before = turn['request'], turn['tokens'], turn['before']
        H = len(kit.header())
        self.assertEqual(list(tokens), client_stream)
        if request.first:
            self.assertEqual((turn['stable'], turn['span']), (request.prefix_tokens, None))
            self.assertEqual(tokens, request.input_ids)
            return
        C, N = len(before) - 1, len(tokens)
        o, e = turn['span']
        self.assertEqual((turn['stable'], o, e), (C + request.boundary_tokens, C + request.boundary_tokens, N - H))
        self.assertTrue(C < o < e < N)
        self.assertEqual((tokens[:len(before)], tokens[C], tokens[o], tokens[e:]),
                         (before, before[-1], kit.im_start, kit.header()))
        self.assertEqual(request.input_ids[0], before[-1])


class FakeEpisodeTests(EpisodeChecks, unittest.TestCase):
    """The MemAgent loop and the harness's warm-up/validation shapes on the stand-in tokenizer."""
    kit = fake_kit()

    def run_case(self, kit, script_kwargs, check_render=True):
        server = MiniServer(kit)
        script = episode_script(len(CHUNKS), **script_kwargs)
        client, agent, sid = start(server, kit, script)
        play(server, kit, script, agent)
        self.assertEqual(agent.answer, 'Answer: chunk 2 names the harbour city.')
        turns = [t for t in server.turns if t['sid'] == sid]
        self.assertEqual(len(turns), len(script))
        for k, turn in enumerate(turns):
            self.check_turn(kit, turn, client_stream(client.bodies[:k + 1], client.responses[:k]))
        if check_render:
            # Non-empty reasoning: the raw stream is the template's render with
            # preserve_thinking, as text (and token for token on the real tokenizer).
            final = turns[-1]['tokens']
            text = kit.render(normalize(agent.messages[:-1]), [BASH], True, enable_thinking=True,
                              preserve_thinking=True)
            self.assertEqual(kit.decode(final), text)
            if kit.name != 'char':
                self.assertEqual(list(final), kit.encode(text))
        return server, client, agent, turns

    def test_memagent_case_every_turn_type(self):
        server, client, agent, turns = self.run_case(self.kit, dict(invalid_at=2, format_at=3, double_at=4))
        kinds = [t['context']['message_source'] for t in turns]
        # turn 0 user; reads; after the invalid call (rc 1); the nudge; the question.
        self.assertEqual(kinds[0], 'user')
        self.assertEqual(turns[3]['context']['structured_state']['return_code'], 1)
        self.assertEqual((turns[4]['context']['message_source'], turns[4]['context']['explicit_requirement']),
                         ('user', NUDGE))
        self.assertEqual((turns[-1]['context']['message_source'], turns[-1]['context']['explicit_requirement']),
                         ('user', QUESTION))
        self.assertIn('[History chunk 5/5]', turns[-1]['context']['observation'])
        self.assertTrue(all(t['context']['message_source'] == 'tool' for t in turns[1:-1]
                            if t not in (turns[4],)))
        self.assertEqual({t['objective'] for t in turns}, {FIRST_USER})
        self.assertEqual([b['boundary_tokens'] for b in client.bodies][:3], [0, 2, 2])
        self.assertTrue(all(b['header_tokens'] == len(self.kit.header()) for b in client.bodies))
        self.assertEqual({b['prefix_tokens'] for b in client.bodies}, {client.bodies[0]['prefix_tokens']})
        responses = client.responses
        self.assertTrue(all(r['finish_reason'] == 'stop' and r['completion_tokens'] == len(r['output_ids'])
                            and r['cached_tokens'] == r['prompt_tokens'] - len(b['input_ids'])
                            for r, b in zip(responses, client.bodies)))
        # Two chunks in one reply: one user block with two tool responses.
        double = client.bodies[5]
        self.assertEqual(self.kit.decode(double['input_ids']).count('<tool_response>'), 2)
        client.close()
        self.assertEqual(server.closes, [client.bodies[0]['session_id']])

    def test_length_limited_reply_restores_its_last_sample(self):
        server, client, agent, turns = self.run_case(self.kit, dict(length_at=1))
        self.assertEqual(client.responses[1]['finish_reason'], 'length')
        after = client.bodies[2]
        self.assertEqual((after['boundary_tokens'], after['input_ids'][0]),
                         (3, client.responses[1]['output_ids'][-1]))
        self.assertEqual(after['input_ids'][1], IM_END)

    def test_warm_up_and_transport_validation_shapes(self):
        server = MiniServer(self.kit)
        # Warm-up: user only, no tools, max_tokens 2 (a length-limited reply).
        sid = uuid.uuid4().hex
        client = KvClient(self.kit, server.send, new_id=lambda: sid)
        server.script[sid] = [('OK, I will reply now with a short answer.', 'length')]
        message, data = client.complete([dict(role='user', content='Reply OK.')], 2)
        self.assertEqual((data['finish_reason'], data['completion_tokens'], data['cached_tokens']), ('length', 2, 0))
        self.assertEqual(client.close()['known'], True)
        # Two user-only turns with max_tokens 32, sessions closed and gone.
        for k in range(3):
            sid = uuid.uuid4().hex
            client = KvClient(self.kit, server.send, new_id=lambda: sid)
            server.script[sid] = [('\n</think>\n\n3', 'eos'), ('\n</think>\n\n3', 'eos')]
            messages = [dict(role='user', content='Reply with the number %d.' % k)]
            message, data = client.complete(messages, 32)
            messages += [message, dict(role='user', content='Repeat that number.')]
            message, data2 = client.complete(messages, 32)
            self.assertEqual((client.bodies[1]['boundary_tokens'], data2['finish_reason']), (2, 'stop'))
            self.assertEqual(data2['prompt_tokens'], len(client.stream()))
            client.close()
        self.assertEqual(len(server.closes), 4)

    def test_sessions_interleave_without_leaking(self):
        server = MiniServer(self.kit, slots=6)
        cases = []
        for k in range(5):
            chunks = ['session %d chunk %d: %s' % (k, j, CHUNKS[j]) for j in range(3 + k % 2)]
            script = episode_script(len(chunks), format_at=1 if k % 2 else None)
            client, agent, sid = start(server, self.kit, script, chunks=chunks, question=QUESTION + ' (%d)' % k)
            cases.append((client, agent, sid, script))
        # Round robin, one request each, until every agent answered.
        while any(agent.answer is None for _, agent, _, _ in cases):
            for client, agent, sid, script in cases:
                if agent.answer is None:
                    agent.step()
        order = [t['index'] for t in server.turns]
        self.assertEqual(order[:5], [0, 1, 2, 3, 4])            # slots bind in first-request order
        for client, agent, sid, script in cases:
            turns = [t for t in server.turns if t['sid'] == sid]
            self.assertEqual(len(turns), len(script))
            for k, turn in enumerate(turns):
                self.check_turn(self.kit, turn, client_stream(client.bodies[:k + 1], client.responses[:k]))
            other = [c for c in cases if c[2] != sid]
            final = self.kit.decode(turns[-1]['tokens'])
            for _, other_agent, _, _ in other:
                self.assertNotIn(other_agent.question, final)
        for client, _, _, _ in cases:
            client.close()
        self.assertEqual(server.registry.next, 5)

    def test_close_is_idempotent_and_ids_are_never_served_again(self):
        server = MiniServer(self.kit, slots=2)
        sid = uuid.uuid4().hex
        self.assertEqual(server.send('close', {'session_id': sid}), (200, dict(closed=True, session_id=sid, known=False)))
        script = episode_script(1)
        client, agent, sid = start(server, self.kit, script, chunks=CHUNKS[:1])
        agent.step()
        body = client.bodies[-1]
        status, first = server.send('close', {'session_id': sid})
        self.assertEqual((status, first['known'], first['turns']), (200, True, 1))
        self.assertEqual(server.send('close', {'session_id': sid}), (200, dict(first, repeated=True)))
        for request in (body, dict(body, boundary_tokens=2)):
            status, data = server.send('complete', request)
            self.assertEqual((status, data['type']), (400, 'SESSION_CLOSED'))
        self.assertEqual(server.send('close', {'session_id': 'bad'})[0], 400)

    def test_errors_leave_the_session_usable(self):
        server = MiniServer(self.kit)
        script = episode_script(3)
        client, agent, sid = start(server, self.kit, script, chunks=CHUNKS[:3])
        agent.step()
        agent.step()
        body, _ = client.build(agent.messages)
        good = copy.deepcopy(body)
        H = body['header_tokens']
        cases = [('CONTINUATION_MISMATCH', dict(input_ids=[PAD] + body['input_ids'][1:])),
                 ('PREFIX_MISMATCH', dict(prefix_tokens=body['prefix_tokens'] - 1)),
                 ('HEADER_MISMATCH', dict(input_ids=body['input_ids'][:-1] + [IM_START])),
                 ('BOUNDARY_MISMATCH', dict(boundary_tokens=1)),
                 ('SESSION_STATE', dict(boundary_tokens=0, input_ids=[1] * body['prefix_tokens'] + body['input_ids'][-H:])),
                 ('SESSION_UNKNOWN', dict(session_id=uuid.uuid4().hex)),
                 ('MAX_TOKENS', dict(max_tokens=4097)), ('SAMPLING', dict(temperature=0.5)),
                 ('BUDGET_MISMATCH', dict(budget=16384)), ('BAD_MESSAGES', dict(messages=body['messages'][:2]))]
        before = (server.frontend[0].stream, server.engine[0].stream, server.frontend[0].turn)
        for kind, change in cases:
            status, data = server.send('complete', dict(good, **change))
            self.assertEqual((status, data['type']), (400, kind), (kind, data))
            self.assertEqual((server.frontend[0].stream, server.engine[0].stream, server.frontend[0].turn), before)
            self.assertIsNone(server.frontend[0].pending)
        while agent.answer is None:
            agent.step()
        self.assertEqual(agent.answer, 'Answer: chunk 2 names the harbour city.')

    def test_context_limit_refusal(self):
        script = episode_script(len(CHUNKS))
        probe = MiniServer(self.kit)
        client, agent, sid = start(probe, self.kit, script)
        play(probe, self.kit, script, agent)
        lengths = [t['prompt_tokens'] for t in client.responses]
        # A server whose context ends inside the fourth request's prompt.
        limit = lengths[3] - 1
        server = MiniServer(self.kit, max_model_len=limit)
        client, agent, sid = start(server, self.kit, episode_script(len(CHUNKS)))
        for _ in range(3):
            agent.step()
        self.assertEqual(client.responses[2]['prompt_tokens'], lengths[2])
        stream = server.frontend[0].stream
        with self.assertRaises(ContextLimit) as caught:
            agent.step()
        self.assertEqual(caught.exception.args[0], {'error': K.CONTEXT_LIMIT, 'type': 'CONTEXT_LIMIT'})
        self.assertEqual((server.frontend[0].stream, server.frontend[0].pending), (stream, None))
        # Exactly at the limit minus one: the reply is clamped to one token.
        server = MiniServer(self.kit, max_model_len=lengths[3] + 1)
        client, agent, sid = start(server, self.kit, episode_script(len(CHUNKS)))
        for _ in range(3):
            agent.step()
        client.complete(agent.messages)
        self.assertEqual((client.responses[-1]['completion_tokens'], client.responses[-1]['finish_reason']),
                         (1, 'length'))
        self.assertEqual(server.turns[-1]['effective'], 1)

    def test_pad_stop_is_restored_like_a_length_stop(self):
        server = MiniServer(self.kit)
        script = [(reply_text('read', 0), 'pad')] + episode_script(1)[1:]
        client, agent, sid = start(server, self.kit, script, chunks=CHUNKS[:1])
        play(server, self.kit, script, agent)
        self.assertEqual(client.responses[0]['output_ids'][-1], PAD)
        self.assertEqual((client.bodies[1]['input_ids'][:2], client.bodies[1]['boundary_tokens']), ([PAD, IM_END], 3))
        self.check_turn(self.kit, server.turns[1], client_stream(client.bodies[:2], client.responses[:1]))


@unittest.skipUnless(HAVE_TRANSFORMERS, 'transformers and tokenizers not installed (use the remote-ra-tok venv)')
class QwenTokenizerTests(FakeEpisodeTests):
    """The same episodes on the pinned Qwen3.6 tokenizer and Transformers' own template
    rendering: header 5 tokens, boundary 2 after EOS and 3 after a restored sample, and
    the raw stream equal, token for token, to the preserve_thinking render."""

    @classmethod
    def setUpClass(cls):
        cls.kit = real_kit()

    def test_measured_shapes(self):
        kit = self.kit
        self.assertEqual(len(kit.header('on')), 5)
        self.assertEqual(kit.decode(kit.header('on')), '<|im_start|>assistant\n<think>\n')
        render = lambda template, messages, kwargs: kit.render(messages, None, kwargs.pop('add_generation_prompt'),
                                                               **kwargs)
        for thinking in ('on', 'off'):
            self.assertEqual(K.check_header(render, TEMPLATE, thinking), K.HEADERS[thinking])
        server, client, agent, turns = self.run_case(kit, dict(invalid_at=2, format_at=3, double_at=4, length_at=5))
        self.assertEqual({b['header_tokens'] for b in client.bodies}, {5})
        self.assertEqual([b['boundary_tokens'] for b in client.bodies[1:]],
                         [3 if r['finish_reason'] == 'length' else 2 for r in client.responses[:-1]])

    def test_stand_in_template_rendering_equals_transformers(self):
        fake, real = fake_kit(), self.kit
        server = MiniServer(fake)
        script = episode_script(len(CHUNKS), invalid_at=2, format_at=3, double_at=4)
        client, agent, sid = start(server, fake, script)
        play(server, fake, script, agent)
        agent_messages = normalize(agent.messages)
        for k in range(1, len(agent_messages) + 1):
            for generation in (False, True):
                for kwargs in (dict(enable_thinking=True), dict(enable_thinking=True, preserve_thinking=True),
                               dict(enable_thinking=False)):
                    part = agent_messages[:k]
                    if not any(m['role'] == 'user' for m in part):
                        continue
                    self.assertEqual(fake.render(part, [BASH], generation, **kwargs),
                                     real.render(part, [BASH], generation, **kwargs))


if __name__ == '__main__':
    unittest.main()
