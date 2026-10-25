"""kv-session protocol with the Gemma 4 grammar (gemma-mab): client geometry, transcript, launcher. CPU only.

The client here is this project's own re-implementation of the KV-session client's request
construction for Gemma 4 31B IT, thinking off (the MemoryAgentBench decision), next to the
Qwen one of test_kv_session_protocol (whose read_next agent loop it reuses):

* turn 0 sends the whole rendered prompt; its header is the pinned template's generation
  header, ``<|turn>model\\n<|channel>thought\\n<channel|>`` (7 tokens);
* a later turn sends the tail after the newest reply. Gemma's template renders a reply's
  tool calls, then its tool responses, then its text, and drops tool messages that follow
  a reply without tool calls, so Qwen's marker (the reply replaced by a text marker) does
  not find the tail of a tool-call reply. The client renders a stand-in reply instead:
  the reply's tool calls without text; without the new messages the template closes such
  a reply with ``<|tool_response>`` (``tool_call_tail``), with them the tail goes on from
  that token. A reply without tool calls takes Qwen's marker;
* the header is the render with the generation prompt less the one without: 7 tokens after
  a user turn, none after tool responses (the model turn goes on; header_tokens 0);
* the boundary ends where the new messages open: before ``<|turn>`` or right after
  ``<|tool_response>``, whichever comes first; the previous reply's last output id is
  restored in front (+1) when it is not the tail's first token (``<|tool_response>``
  after a tool call, ``<turn|>`` after text): the EOS rule of Gemma's stop ids
  {0, 1, 50, 106}.

The served stream is then the template's render of the conversation with every model
turn's empty thought channel kept (the header the server generated after, which Gemma's
template drops when it re-renders history; ``reference_text``). On the pinned Gemma
tokenizer (``REMOTE_RA_GEMMA4_METADATA`` and Transformers 5.13, the remote-ra-tok venv)
it is equal token for token; stock vLLM's Vanilla re-render differs from the first model
turn on by those 3 tokens per turn. Without the metadata the tests use a character
tokenizer with Gemma's special ids and a Python stand-in of the template for these
message shapes (``gemma_render``), which the metadata tests compare with the real one.
"""
import copy
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
import unittest
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_kv_session_protocol as KP  # noqa: E402  the read_next agent loop, the Qwen client's helpers
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402

META = os.environ.get('REMOTE_RA_GEMMA4_METADATA')
HAVE_REAL = bool(META) and KP.HAVE_TRANSFORMERS

# Gemma 4's special tokens (the pinned tokenizer.json) the grammar uses.
SPECIAL = {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<|tool>': 46, '<tool|>': 47, '<|tool_call>': 48,
           '<tool_call|>': 49, '<|tool_response>': 50, '<tool_response|>': 51, '<|"|>': 52, '<|think|>': 98,
           '<|channel>': 100, '<channel|>': 101, '<|turn>': 105, '<turn|>': 106}
PAD, EOS, TOOL_CALL, TOOL_CALL_END = 0, 1, 48, 49
TOOL_OPEN, TOOL_CLOSE, TURN, TURN_END = 50, 51, 105, 106
STOP_IDS = frozenset({PAD, EOS, TOOL_OPEN, TURN_END})
Q = '<|"|>'
HEADER_TEXT = K.GEMMA4_HEADER
CHANNEL = '<|channel>thought\n<channel|>'
BASH = {'type': 'function', 'function': {'name': 'bash', 'description': 'Execute a bash command',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string',
                       'description': 'The bash command to execute'}}, 'required': ['command']}}}


# ---- the template: a Python stand-in for the message shapes of the read_next loop -----------------

def _argument(value):
    """format_argument(value, escape_keys=False) of the pinned template."""
    if value is None:
        return 'null'
    if type(value) is str:
        return Q + value + Q
    if type(value) is bool:
        return 'true' if value else 'false'
    if type(value) is dict:
        return '{' + ','.join(k + ':' + _argument(v) for k, v in sorted(value.items())) + '}'
    if type(value) in (list, tuple):
        return '[' + ','.join(_argument(v) for v in value) + ']'
    return str(value)


def _declaration(tool):
    """format_function_declaration of a tool whose parameters are string properties."""
    function = tool['function']
    params = function.get('parameters') or {}
    out = 'declaration:' + function['name'] + '{description:' + Q + function['description'] + Q
    if params:
        out += ',parameters:{'
        if params.get('properties'):
            parts = []
            for key, value in sorted(params['properties'].items()):
                assert value['type'] == 'string' and set(value) <= {'type', 'description'}, value
                part = key + ':{'
                if value.get('description'):
                    part += 'description:' + Q + value['description'] + Q + ','
                parts.append(part + 'type:' + Q + 'STRING' + Q + '}')
            out += 'properties:{' + ','.join(parts) + '},'
        if params.get('required'):
            out += 'required:[' + ','.join(Q + r + Q for r in params['required']) + '],'
        if params.get('type'):
            out += 'type:' + Q + params['type'].upper() + Q + '}'
    return out + '}'


def _strip_thinking(text):
    result = ''
    for part in text.split('<channel|>'):
        result += part.split('<|channel>')[0] if '<|channel>' in part else part
    return result.strip()


def gemma_render(messages, tools=None, add_generation_prompt=False, enable_thinking=False):
    """The pinned Gemma 4 chat template's text for system/user/assistant/tool messages with
    string content, string-argument tool calls and no reasoning text, thinking off."""
    assert enable_thinking is False
    out, loop, prev_type = ['<bos>'], list(messages), None
    if tools or (messages and messages[0]['role'] in ('system', 'developer')):
        out.append('<|turn>system\n')
        if messages and messages[0]['role'] in ('system', 'developer'):
            out.append(messages[0]['content'].strip())
            loop = loop[1:]
        for tool in tools or ():
            out.append('<|tool>' + _declaration(tool) + '<tool|>')
        prev_type = 'tool' if tools else None
        out.append('<turn|>\n')
    prev_role = None
    for i, m in enumerate(loop):
        if m['role'] == 'tool':
            continue
        assert not (m.get('reasoning') or m.get('reasoning_content')), 'no reasoning text with thinking off'
        prev_type = None
        role = 'model' if m['role'] == 'assistant' else m['role']
        if not (role == 'model' and prev_role == 'assistant'):
            out.append('<|turn>' + role + '\n')
        calls = m.get('tool_calls') or []
        for call in calls:
            arguments = call['function']['arguments']
            assert type(arguments) is dict
            out.append('<|tool_call>call:' + call['function']['name'] + '{'
                       + ','.join(k + ':' + _argument(v) for k, v in sorted(arguments.items())) + '}<tool_call|>')
            prev_type = 'tool_call'
        responded = False
        if calls:
            for follow in loop[i + 1:]:
                if follow['role'] != 'tool':
                    break
                name = follow.get('name') or 'unknown'
                for call in calls:
                    if call.get('id') == follow.get('tool_call_id'):
                        name = call['function']['name']
                out.append('<|tool_response>response:' + name + '{value:' + _argument(follow['content'])
                           + '}<tool_response|>')
                responded, prev_type = True, 'tool_response'
        content = m.get('content') or ''
        content = _strip_thinking(content) if role == 'model' else content.strip()
        out.append(content)
        following = next((f['role'] for f in loop[i + 1:] if f['role'] != 'tool'), None)
        continues = role == 'model' and following == 'assistant' and (not calls or responded)
        if prev_type == 'tool_call' and not responded:
            out.append('<|tool_response>')
        elif continues:
            pass
        elif not (responded and not content.strip() and following is None):
            out.append('<turn|>\n')
        prev_role = m['role']
    if add_generation_prompt and prev_type not in ('tool_response', 'tool_call'):
        out.append('<|turn>model\n<|channel>thought\n<channel|>')
    return ''.join(out)


def reference_text(messages, tools=None):
    """The served stream of ``messages`` (their last message a tool or user one) as text:
    the template's render with the generation prompt, every model turn's empty thought
    channel kept after its ``<|turn>model\\n`` (the header the reply was generated after)."""
    text = gemma_render(messages, tools, True)
    if text.endswith(HEADER_TEXT):
        body, header = text[:-len(HEADER_TEXT)], HEADER_TEXT
    else:
        body, header = text, ''
    return body.replace('<|turn>model\n', '<|turn>model\n' + CHANNEL) + header


# ---- tokenizers -----------------------------------------------------------------------------------

class CharTokenizer:
    """Characters as ids above every real one; Gemma's special tokens keep their pinned ids."""
    BASE = 300000
    vocab = BASE + 0x110000

    def __init__(self):
        self.order = sorted(SPECIAL, key=len, reverse=True)
        self.names = {i: t for t, i in SPECIAL.items()}
        self.splitter = re.compile('(' + '|'.join(re.escape(t) for t in self.order) + ')')

    def encode(self, text):
        ids = []
        for part in self.splitter.split(text):
            if part in SPECIAL:
                ids.append(SPECIAL[part])
            elif part:
                ids.extend(ord(ch) + self.BASE for ch in part)
        return ids

    def decode(self, ids):
        return ''.join(self.names[t] if t in self.names else chr(t - self.BASE) for t in ids)


class Kit:
    """One tokenizer and the template renderer (the Gemma grammar's)."""

    def __init__(self, name, encode, decode, render, vocab):
        self.name, self._encode, self._decode, self._render, self.vocab = name, encode, decode, render, vocab
        self.im_start, self.eos, self.pad = TURN, EOS, PAD
        self.stop_ids = STOP_IDS

    def encode(self, text):
        return list(self._encode(text))

    def decode(self, ids):
        return self._decode(list(ids))

    def render(self, messages, tools=None, generation=False, **kwargs):
        return self._render(messages, tools, generation, dict(kwargs, enable_thinking=False))

    def header(self, thinking='off'):
        return tuple(self.encode(HEADER_TEXT))

    def transcript(self, thinking='off'):
        return K.KvTranscript(header=self.header(), im_start=TURN, vocab=self.vocab, decode=self.decode,
                              stop_ids=STOP_IDS, tools=(TOOL_OPEN, TOOL_CLOSE))


def fake_kit():
    tok = CharTokenizer()
    return Kit('char', tok.encode, tok.decode,
               lambda messages, tools, generation, kwargs: gemma_render(messages, tools, generation, **kwargs),
               tok.vocab)


_REAL = []


def real_kit():
    """The pinned Gemma 4 tokenizer and template through Transformers (the metadata files)."""
    if not HAVE_REAL:
        return None
    if not _REAL:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(META, local_files_only=True)
        assert tok.chat_template == (Path(META) / 'chat_template.jinja').read_text(encoding='utf-8')
        assert all(tok.convert_tokens_to_ids(t) == i for t, i in SPECIAL.items())

        def render(messages, tools, generation, kwargs):
            return tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                           add_generation_prompt=generation, **kwargs)
        _REAL.append((Kit('gemma4', lambda text: tok.encode(text, add_special_tokens=False),
                          lambda ids: tok.decode(ids, skip_special_tokens=False), render, len(tok)), tok))
    return _REAL[0][0]


# ---- the client (re-implemented for Gemma's grammar) -----------------------------------------------

def parse_arguments(text):
    """Gemma's tool-call arguments ``key:<|"|>text<|"|>,key:123`` as a dict."""
    out, i = {}, 0
    while i < len(text):
        colon = text.index(':', i)
        key = text[i:colon]
        i = colon + 1
        if text.startswith(Q, i):
            end = text.index(Q, i + len(Q))
            out[key] = text[i + len(Q):end]
            i = end + len(Q)
        else:
            end = text.find(',', i)
            end = len(text) if end < 0 else end
            raw = text[i:end]
            out[key] = json.loads(raw) if raw in ('true', 'false', 'null') or re.fullmatch(r'-?\d+(\.\d+)?', raw) \
                else raw
            i = end
        if text.startswith(',', i):
            i += 1
    return out


def parse_reply(text, finish_reason):
    """The client's parse of a Gemma reply: a thought channel (none with thinking off), the
    ``<|tool_call>call:name{...}<tool_call|>`` calls, the rest as text without special tokens."""
    reasoning = ''
    match = re.search(r'<\|channel>thought\n(.*?)<channel\|>', text, re.S)
    if match:
        reasoning, text = match.group(1).strip(), text[:match.start()] + text[match.end():]
    calls = []
    for name, arguments in re.findall(r'<\|tool_call>call:([\w.\-]+)\{(.*?)\}<tool_call\|>', text, re.S):
        try:
            parsed = parse_arguments(arguments)
        except ValueError:
            continue
        calls.append({'id': 'call_' + uuid.uuid4().hex[:12], 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(parsed)}})
    content = re.sub(r'<\|tool_call>.*?<tool_call\|>', '', text, flags=re.S)
    for token in ('<turn|>', '<|tool_response>', '<eos>', '<pad>'):
        content = content.replace(token, '')
    message = {'role': 'assistant', 'content': content.strip(), 'reasoning_content': reasoning}
    if calls:
        message['tool_calls'] = calls
    return message, ('tool_calls' if calls else finish_reason)


def gemma_tail(render, previous, reply, new):
    """(tail text, header text) of the request after ``reply`` (the newest assistant message)
    whose new messages are ``new``: Gemma's grammar, thinking off."""
    if reply.get('tool_calls'):
        stand_in = {'role': 'assistant', 'content': '', 'tool_calls': copy.deepcopy(reply['tool_calls'])}
        base = render(previous + [stand_in], False)
        with_header, without = render(previous + [stand_in] + new, True), render(previous + [stand_in] + new, False)
        if not (base.endswith(K.GEMMA4_TOOL_OPEN) and without.startswith(base) and with_header.startswith(without)):
            raise ValueError('Unsupported template continuation after tool calls')
        return with_header[len(base) - len(K.GEMMA4_TOOL_OPEN):], with_header[len(without):]
    marker = 'KV_TAIL_MARKER_' + uuid.uuid4().hex
    marked = previous + [{'role': 'assistant', 'content': marker}] + new
    with_header, without = render(marked, True), render(marked, False)
    if with_header.count(marker) != 1 or not with_header.startswith(without):
        raise ValueError('Unsupported template continuation')
    return with_header.split(marker, 1)[1], with_header[len(without):]


def gemma_boundary(enc, tail):
    """Tokens of ``tail`` before its new messages: up to ``<|turn>`` or through ``<|tool_response>``."""
    turn, tool = tail.find(K.GEMMA4_TURN_START), tail.find(K.GEMMA4_TOOL_OPEN)
    if tool >= 0 and (turn < 0 or tool < turn):
        return len(enc(tail[:tool + len(K.GEMMA4_TOOL_OPEN)]))
    if turn < 0:
        raise ValueError('the new messages need an explicit boundary')
    return len(enc(tail[:turn]))


class GemmaKvClient(KP.KvClient):
    """One agent's KV session (test_kv_session_protocol.KvClient) with Gemma's request construction."""

    def __init__(self, kit, send, *, tools=None, budget=32768, max_tokens=4096, new_id=None):
        super().__init__(kit, send, tools=tools, thinking=False, budget=budget, max_tokens=max_tokens, new_id=new_id)

    def build(self, messages, max_tokens=None):
        normalized = KP.normalize(messages)
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
            reply, feedback = (new[0], new[1:]) if new[0]['role'] == 'assistant' else (
                KP.normalize([self.state['last_message']])[0], new)
            tail, header_text = gemma_tail(self.render, previous, reply, feedback)
            header = enc(header_text)
            ids = enc(tail)
            boundary = gemma_boundary(enc, tail)
            last = self.state['last'][-1:]
            if last and last[0] != ids[0]:
                ids = last + ids            # the uncomputed last sample, restored
                boundary += 1
        body = dict(method=K.METHOD, session_id=self.state['id'], input_ids=ids, header_tokens=len(header),
                    prefix_tokens=self.state['prefix'], budget=self.budget,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens, temperature=0, top_p=1.0,
                    top_k=-1, repetition_penalty=1.0, seed=42, messages=normalized, boundary_tokens=boundary)
        return body, normalized

    def accept(self, normalized, status, data):
        if status != 200:
            if status == 400 and data.get('error') == K.CONTEXT_LIMIT:
                raise KP.ContextLimit(data)
            raise KP.ServerError(status, data)
        self.responses.append(data)
        self.state.update(messages=copy.deepcopy(normalized), last=list(data['output_ids']))
        message, _finish = parse_reply(data['text'], data['finish_reason'])
        self.state['last_message'] = message
        return message, data


# ---- a protocol-level server with Gemma's transcripts ----------------------------------------------

class MiniServer(KP.MiniServer):
    """test_kv_session_protocol.MiniServer with Gemma's grammar: header 0 after tool responses."""

    def generate(self, sid, effective):
        text, end = self.script[sid].pop(0)
        ids = self.kit.encode(text) + {'call': [TOOL_OPEN], 'turn': [TURN_END], 'eos': [EOS], 'pad': [PAD],
                                       'length': []}[end]
        return ids[:effective]

    def complete(self, body):
        request = K.parse_complete(body, budget=self.budget, bound=self.bound, max_model_len=self.max_model_len,
                                   empty_header=True)
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
        echo = K.from_source(K.validate_source(json.loads(json.dumps(source))))
        etokens, estable, espan = engine.render(echo)
        assert (etokens, estable, espan) == (tokens, stable, span)
        assert engine.phase2_suffix(etokens) == front.suffix(request)
        context = K.goal_inputs(echo.messages, first)
        generated = self.generate(request.session_id, effective)
        committed = front.commit(tokens, generated, len(request.messages))
        engine.commit(etokens, generated)
        assert engine.stream == front.stream and engine.previous_action() == front.previous_action()
        finish = K.finish_reason(generated, front.stop_ids, effective)
        self.turns.append(dict(index=index, sid=request.session_id, before=before, tokens=tokens, stable=stable,
                               span=span, request=request, generated=generated, context=context,
                               objective=front.objective, effective=effective, source=source))
        return dict(text=committed['text'], output_ids=list(generated), prompt_tokens=len(tokens),
                    completion_tokens=len(generated), cached_tokens=len(tokens) - len(request.input_ids),
                    finish_reason=finish)


CALL = '<|tool_call>call:bash{command:' + Q + '%s' + Q + '}<tool_call|>'


def reply_text(kind, calls=1):
    """A scripted Gemma reply (thinking off: no thought channel) and how it stops."""
    if kind == 'read':
        return ''.join(CALL % 'read_next' for _ in range(calls)), 'call'
    if kind == 'invalid':
        return CALL % 'cat history.txt', 'call'
    if kind == 'format':
        return 'I will continue reading the history.', 'turn'
    return 'Answer: chunk 2 names the harbour city.', 'turn'


def episode_script(n_chunks, *, invalid_at=None, format_at=None, double_at=None, stops=None):
    """Replies of one MemAgent case (test_kv_session_protocol.episode_script with Gemma replies);
    ``stops`` overrides the stop of reply k (eos, pad, turn after a call, length)."""
    replies, read, k = [], 0, 0
    while read < n_chunks:
        if k == invalid_at:
            replies.append(reply_text('invalid'))
        elif k == format_at:
            replies.append(reply_text('format'))
        else:
            calls = 2 if k == double_at and read + 2 <= n_chunks else 1
            replies.append(reply_text('read', calls))
            read += calls
        k += 1
    replies.append(reply_text('answer'))
    return [(text, (stops or {}).get(j, end)) for j, (text, end) in enumerate(replies)]


def start(server, kit, script, chunks=KP.CHUNKS, question=KP.QUESTION, tools=(BASH,), **kwargs):
    sid = uuid.uuid4().hex
    client = GemmaKvClient(kit, server.send, tools=list(tools) if tools else None, new_id=lambda: sid, **kwargs)
    server.script[sid] = list(script)
    return client, KP.Agent(client, chunks, question), sid


def play(kit, script, agent):
    k = 0
    while agent.answer is None:
        text, end = script[k]
        agent.step(len(kit.encode(text)) if end == 'length' else None)
        k += 1
    return agent


# ---- tests -----------------------------------------------------------------------------------------

class GemmaEpisodeChecks:
    def check_turn(self, kit, turn, client_stream):
        request, tokens, before = turn['request'], turn['tokens'], turn['before']
        H = len(kit.header())
        self.assertEqual(list(tokens), client_stream)
        if request.first:
            self.assertEqual((turn['stable'], turn['span'], request.header_tokens), (request.prefix_tokens, None, H))
            self.assertEqual((tokens, tuple(tokens[-H:])), (request.input_ids, kit.header()))
            return
        C, N = len(before) - 1, len(tokens)
        o, e = turn['span']
        suffix = request.header_tokens or 1
        self.assertIn(request.header_tokens, (0, H))
        self.assertEqual((turn['stable'], o, e), (C + request.boundary_tokens, C + request.boundary_tokens, N - suffix))
        self.assertTrue(C < o < e < N)
        self.assertEqual((tokens[:len(before)], tokens[C]), (before, before[-1]))
        self.assertTrue(tokens[o] == TURN or tokens[o - 1] == TOOL_OPEN, (tokens[o - 1], tokens[o]))
        if request.header_tokens:
            self.assertEqual(tuple(tokens[e:]), kit.header())
        else:
            self.assertEqual((tokens[e:], tokens[o - 1]), ((TOOL_CLOSE,), TOOL_OPEN))
        self.assertEqual(request.input_ids[0], before[-1])


class GemmaFakeEpisodeTests(GemmaEpisodeChecks, unittest.TestCase):
    """The MemAgent loop on the character tokenizer and the template stand-in."""
    kit = fake_kit()

    def run_case(self, script_kwargs, *, chunks=KP.CHUNKS, check_reference=True):
        kit = self.kit
        server = MiniServer(kit, thinking='off')
        script = episode_script(len(chunks), **script_kwargs)
        client, agent, sid = start(server, kit, script, chunks=chunks)
        play(kit, script, agent)
        self.assertEqual(agent.answer, 'Answer: chunk 2 names the harbour city.')
        turns = [t for t in server.turns if t['sid'] == sid]
        self.assertEqual(len(turns), len(script))
        for k, turn in enumerate(turns):
            self.check_turn(kit, turn, KP.client_stream(client.bodies[:k + 1], client.responses[:k]))
            if check_reference:
                # Every served prompt: the template's render of its messages with every model
                # turn's thought channel kept, token for token (as text on the stand-in).
                messages = KP.normalize(client.bodies[k]['messages'])
                text = reference_text(messages, [BASH])
                self.assertEqual(kit.decode(turn['tokens']), text)
                self.assertEqual(list(turn['tokens']), kit.encode(text))
        return server, client, agent, turns

    def test_memagent_case_every_turn_type(self):
        server, client, agent, turns = self.run_case(dict(invalid_at=2, format_at=3, double_at=4))
        bodies = client.bodies
        H = len(self.kit.header())
        # turn 0 (header 7), reads (no header, boundary 1), the invalid call (rc 1), the
        # format error (text stopped on <turn|>: boundary 2 before <|turn>), the question (header 7).
        self.assertEqual([b['header_tokens'] for b in bodies], [H, 0, 0, 0, H, 0, H])
        self.assertEqual([b['boundary_tokens'] for b in bodies], [0, 1, 1, 1, 2, 1, 1])
        self.assertEqual(turns[3]['context']['structured_state']['return_code'], 1)
        self.assertEqual((turns[4]['context']['message_source'], turns[4]['context']['explicit_requirement']),
                         ('user', KP.NUDGE))
        self.assertEqual((turns[-1]['context']['message_source'], turns[-1]['context']['explicit_requirement']),
                         ('user', KP.QUESTION))
        self.assertIn('[History chunk 5/5]', turns[-1]['context']['observation'])
        self.assertEqual({t['objective'] for t in turns}, {KP.FIRST_USER})
        self.assertEqual({b['prefix_tokens'] for b in bodies}, {bodies[0]['prefix_tokens']})
        self.assertTrue(all(r['finish_reason'] == 'stop' for r in client.responses))
        # The reads stop on <|tool_response>, which opens the next request's tool responses.
        self.assertEqual([r['output_ids'][-1] for r in client.responses],
                         [TOOL_OPEN] * 3 + [TURN_END] + [TOOL_OPEN] * 2 + [TURN_END])
        double = bodies[5]
        self.assertEqual(self.kit.decode(double['input_ids']).count('<|tool_response>'), 2)
        # The Goal source: the tool text of a read, the question on the last turn.
        self.assertEqual(turns[1]['context']['message_source'], 'tool')
        client.close()
        self.assertEqual(server.closes, [bodies[0]['session_id']])

    def test_restored_samples(self):
        # A read stopped on <turn|>, on <eos>, on <pad> and at its bound; a text reply on <eos>.
        stops = {0: 'turn', 1: 'eos', 2: 'pad', 3: 'length'}
        server, client, agent, turns = self.run_case(dict(format_at=4, stops={**stops, 4: 'eos'}),
                                                     check_reference=False)
        for k in range(5):
            last, body = client.responses[k]['output_ids'][-1], client.bodies[k + 1]
            restored = body['input_ids'][0] == last and body['input_ids'][1] in (TOOL_OPEN, TURN_END)
            self.assertTrue(restored, (k, body['input_ids'][:3]))
        self.assertEqual([b['boundary_tokens'] for b in client.bodies[1:6]], [2, 2, 2, 2, 3])
        self.assertEqual(client.responses[3]['finish_reason'], 'length')
        # A canonical stop needs no restore: the stream then is the reference again.
        self.assertEqual(client.bodies[6]['input_ids'][0], TOOL_OPEN)

    def test_vanilla_rerender_drops_the_thought_channel(self):
        server, client, agent, turns = self.run_case(dict())
        final = list(turns[-1]['tokens'])
        vanilla = self.kit.encode(gemma_render(KP.normalize(client.bodies[-1]['messages']), [BASH], True))
        first = next(i for i, (a, b) in enumerate(zip(final, vanilla)) if a != b)
        prefix = client.bodies[0]['prefix_tokens']
        # The first model turn: <|turn>model\n follows the prefix, then the thought channel
        # (3 tokens on the pinned tokenizer); every read continues that one model turn.
        self.assertEqual(first, prefix + len(self.kit.encode('<|turn>model\n')))
        self.assertEqual(len(final) - len(vanilla), len(self.kit.encode(CHANNEL)))

    def test_refusals_of_the_gemma_grammar(self):
        server = MiniServer(self.kit, thinking='off')
        script = episode_script(3)
        client, agent, sid = start(server, self.kit, script, chunks=KP.CHUNKS[:3])
        agent.step()
        body, _ = client.build(agent.messages)
        self.assertEqual((body['header_tokens'], body['boundary_tokens'], body['input_ids'][0], body['input_ids'][-1]),
                         (0, 1, TOOL_OPEN, TOOL_CLOSE))
        good = copy.deepcopy(body)
        H = len(self.kit.header())
        # A request without a header must continue after tool responses: <|tool_response>
        # ends its boundary and <tool_response|> its input_ids.
        cases = [('HEADER_MISMATCH', dict(input_ids=body['input_ids'][:-1] + [TURN])),
                 ('HEADER_MISMATCH', dict(header_tokens=H)),
                 ('HEADER_MISMATCH', dict(boundary_tokens=2)),
                 ('HEADER_MISMATCH', dict(input_ids=[PAD] + body['input_ids'][1:])),
                 ('CONTINUATION_MISMATCH', dict(boundary_tokens=2, input_ids=[PAD] + body['input_ids'])),
                 ('BAD_GEOMETRY', dict(header_tokens=-1)),
                 ('BAD_GEOMETRY', dict(input_ids=[TOOL_OPEN, TOOL_CLOSE]))]
        before = (server.frontend[0].stream, server.engine[0].stream, server.frontend[0].turn)
        for kind, change in cases:
            status, data = server.send('complete', dict(good, **change))
            self.assertEqual((status, data['type']), (400, kind), (kind, data))
            self.assertEqual((server.frontend[0].stream, server.engine[0].stream, server.frontend[0].turn), before)
        # A turn-0 body without a header: the whole prompt has one.
        status, data = server.send('complete', dict(good, session_id=uuid.uuid4().hex, boundary_tokens=0))
        self.assertEqual((status, data['type']), (400, 'BAD_GEOMETRY'))
        # A request with the header opens its new messages at <|turn> or after <|tool_response>.
        agent.step()                                  # the second read: its reply stops on <|tool_response>
        agent.step()                                  # the last chunk and the question: header 7
        question, _ = client.build(agent.messages)
        self.assertEqual((question['header_tokens'], question['boundary_tokens']), (H, 1))
        for change in (dict(boundary_tokens=2), dict(boundary_tokens=3)):
            status, data = server.send('complete', dict(question, **change))
            self.assertEqual((status, data['type']), (400, 'BOUNDARY_MISMATCH'), (change, data))
        while agent.answer is None:
            agent.step()
        # Qwen's parse refuses an empty header outright (its grammar has none).
        with self.assertRaises(K.KvSessionError) as caught:
            K.parse_complete(good, budget=32768, bound=4096, max_model_len=131072)
        self.assertEqual(caught.exception.kind, 'BAD_GEOMETRY')

    def test_transcript_suffix_and_the_gate_figure(self):
        kit = self.kit
        t = kit.transcript()
        H = kit.header()
        self.assertEqual(t.phase2_suffix((1, 2) + H), len(H))
        self.assertEqual(t.phase2_suffix((1, 2, TOOL_CLOSE)), 1)
        qwen = KP.fake_kit().transcript('off')
        self.assertEqual(qwen.phase2_suffix((1, 2) + qwen.header), len(qwen.header))
        with self.assertRaisesRegex(Exception, 'two ids'):
            K.KvTranscript(header=H, im_start=TURN, vocab=kit.vocab, decode=kit.decode, stop_ids=STOP_IDS,
                           tools=(TOOL_OPEN, H[-1]))

    def test_record_info_and_header_check(self):
        record = K.record(contract=None, thinking='off', slots=4, grammar=K.GEMMA4)
        self.assertEqual(record, dict(method=K.METHOD, budget=None, thinking='off', header_text=HEADER_TEXT, slots=4,
                                      grammar='gemma4'))
        with self.assertRaisesRegex(Exception, 'thinking off only'):
            K.record(contract=None, thinking='on', slots=4, grammar=K.GEMMA4)
        config = dict(protocol=K.PROTOCOL, kv_session=record, tasks=K.slot_tasks(4), task=K.slot_tasks(1)[0],
                      max_running_sessions=2, selection_contract=None, model_profile='gemma4-31b-it',
                      policy='full', page_tokens=16, effective_server_config=dict(max_model_len=131072))
        self.assertEqual(K.check_config(config), record)
        with self.assertRaisesRegex(Exception, 'record'):
            K.check_config(dict(config, kv_session=dict(record, header_text=K.HEADERS['off'])))
        # A Qwen config carries no grammar; a Gemma one must.
        with self.assertRaisesRegex(Exception, 'required'):
            K.check_config({k: v for k, v in config.items() if k != 'model_profile'})
        with self.assertRaisesRegex(Exception, 'required'):
            K.check_config(dict(config, kv_session={k: v for k, v in record.items() if k != 'grammar'}))
        self.assertEqual((K.options(config)['grammar'], K.info(config)['method']), ('gemma4', K.METHOD))

        def render(template, messages, kwargs):
            return gemma_render(messages, None, kwargs['add_generation_prompt'], kwargs['enable_thinking'])
        self.assertEqual(K.check_gemma4_header(render, None), HEADER_TEXT)
        # A template that adds a header after tool responses too.
        broken = lambda template, messages, kwargs: render(template, messages, kwargs) + (
            HEADER_TEXT if kwargs['add_generation_prompt'] and messages[-1]['role'] == 'tool' else '')
        with self.assertRaisesRegex(Exception, 'does not continue'):
            K.check_gemma4_header(broken, None)


@unittest.skipUnless(HAVE_REAL, 'set REMOTE_RA_GEMMA4_METADATA and install Transformers (the remote-ra-tok venv)')
class GemmaTokenizerTests(GemmaFakeEpisodeTests):
    """The same episodes on the pinned Gemma 4 tokenizer and its template through Transformers:
    the 7-token header, the client's geometry, and every served prompt equal, token for token,
    to the template's render with the thought channels kept."""

    @classmethod
    def setUpClass(cls):
        cls.kit = real_kit()

    def test_measured_shapes(self):
        kit, tok = real_kit(), _REAL[0][1]
        self.assertEqual(len(kit.header()), 7)
        self.assertEqual(kit.decode(kit.header()), HEADER_TEXT)
        template = tok.chat_template

        def render(template_, messages, kwargs):
            kwargs = dict(kwargs)
            return kit.render(messages, None, kwargs.pop('add_generation_prompt'), **kwargs)
        self.assertEqual(K.check_gemma4_header(render, template), HEADER_TEXT)
        for text in (CALL % 'read_next', '<turn|>\n<|turn>user\n', K.GEMMA4_TOOL_OPEN, K.GEMMA4_TOOL_CLOSE):
            self.assertEqual(kit.decode(kit.encode(text)), text)
        self.assertEqual(kit.encode(K.GEMMA4_TOOL_OPEN + K.GEMMA4_TOOL_CLOSE), [TOOL_OPEN, TOOL_CLOSE])

    def test_stand_in_template_rendering_equals_transformers(self):
        fake, real = fake_kit(), real_kit()
        server = MiniServer(fake, thinking='off')
        script = episode_script(len(KP.CHUNKS), invalid_at=2, format_at=3, double_at=4)
        client, agent, sid = start(server, fake, script)
        play(fake, script, agent)
        messages = KP.normalize(agent.messages)
        for k in range(1, len(messages) + 1):
            for generation in (False, True):
                part = messages[:k]
                for tools in ([BASH], None):
                    self.assertEqual(fake.render(part, tools, generation), real.render(part, tools, generation),
                                     (k, generation))
        # The client's stand-in (a reply's tool calls without text) as Transformers renders it.
        previous, reply = messages[:2], dict(messages[2], content='')
        self.assertEqual(fake.render(previous + [reply], [BASH], False), real.render(previous + [reply], [BASH], False))


class StartupCheckTests(unittest.TestCase):
    """vLLM's own startup check of the window layout (the overlay's kv_cache_utils) with the
    selective rule's request_pages: one request then needs those global Pages, not
    max_model_len's (test_gemma_kv_pools: 16 windows at 28 GiB and 131,072 tokens are refused
    without it)."""

    def test_request_pages_replace_one_request_at_the_context_limit(self):
        import dataclasses
        import test_gemma_kv_pools as GKP
        from remote_ra_live_v2 import kv_layout as I0
        from remote_ra_live_v2 import window_capacity as WC
        w = GKP.gemma_world()
        page, block = GKP.GEOMETRY.page_bytes(16), GKP.GEOMETRY.sliding_block_bytes(16)
        for goal, gib in ((0, 28), (2048, 28), (0, 24)):
            base = I0.for_window_server(16, 16, 1024, 8192, goal)
            request = WC.pinned_request_pages(base, 32768, 4096, gib << 30)
            layout = dataclasses.replace(base, request_pages=request)
            self.assertEqual(I0.from_vllm_config(NS(additional_config=I0.to_additional_config(layout))), layout)
            pool1 = I0.window_pool_blocks(base, 5) * block
            for max_len in (131072, 262144):
                with self.subTest(goal=goal, gib=gib, max_len=max_len):
                    vc = GKP.vconfig(base, max_len=max_len)
                    groups = w.utils.get_kv_cache_groups(vc, GKP.gemma_specs(w, 16))
                    # vLLM's own figure: one request at max_len in pool 0 plus pool 1.
                    self.assertEqual(w.utils._max_memory_usage_bytes_from_groups(vc, groups),
                                     max_len // 16 * page + pool1)
                    with self.assertRaisesRegex(ValueError, 'max seq len'):
                        GKP.plan(w, base, gib * GKP.GiB, max_len=max_len)
                    vc = GKP.vconfig(layout, max_len=max_len)
                    groups = w.utils.get_kv_cache_groups(vc, GKP.gemma_specs(w, 16))
                    self.assertEqual(w.utils._max_memory_usage_bytes_from_groups(vc, groups),
                                     request * page + pool1)
                    if gib == 28:
                        _, config = GKP.plan(w, layout, gib * GKP.GiB, max_len=max_len)
                        self.assertEqual(config.num_blocks_by_pool,
                                         ((gib * GKP.GiB - pool1) // page, I0.window_pool_blocks(base, 5)))
                        self.assertGreater(config.num_blocks_by_pool[0] - 1, request)
            # 24 GiB cannot hold 16 windows and one 32K-budget turn (the launcher refuses it too).
            if gib == 24:
                self.assertGreater(request * page + pool1, 24 * GKP.GiB)
        # Without the field nothing changes (a Qwen layout has none either).
        layout = I0.for_window_server(16, 4, 1024, 8192)
        self.assertEqual(layout.request_pages, 0)
        self.assertNotIn(I0.REQUEST_FIELD, I0.to_additional_config(layout)[I0.LAYOUT_KEY])
        with self.assertRaisesRegex(ValueError, 'request_pages must be positive'):
            I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: dict(
                I0.to_additional_config(layout)[I0.LAYOUT_KEY], request_pages=0)}))


class GemmaLauncherTests(unittest.TestCase):
    """serve.py's Gemma kv-session path and its startup checks (test_gemma_startup's fixture)."""
    maxDiff = None

    def setUp(self):
        import test_gemma_startup as GS
        self.gs = GS.GemmaStartupTests('test_load_accepts_the_gemma_startup_config')
        self.gs.setUp()
        self.addCleanup(self.gs.doCleanups)

    def prepare(self, policy='observation_query', selection=None, **kw):
        kw = dict(dict(protocol='kv-session', kv_session_thinking='off', max_running=16, max_model_len=262144,
                       release_idle_pages=True, dram_arena_gib=150, host_memory_shares=2), **kw)
        if selection is None and policy != 'full':
            selection = dict(active_token_budget=32768)
            if policy == 'goal_query':
                selection.update(no_goal_rewriter=True)
        return self.gs.prepare(None, policy, selection, **kw)

    def test_sixteen_sessions_at_262144(self):
        from remote_ra_live_v2 import window_capacity as WC
        from remote_ra_http import config as HC
        for policy, gib in (('observation_query', 28), ('goal_query', 28), ('random_fill', 28),
                            ('observation_query', 32)):
            with self.subTest(policy=policy, gib=gib):
                _, config, command, _ = self.prepare(policy, kv_cache_gib=gib)
                self.assertEqual((config['protocol'], config['kv_session']['grammar'],
                                  config['kv_session']['header_text'], config['kv_session']['slots']),
                                 ('kv_session_v1', 'gemma4', HEADER_TEXT, 64))
                layout = HC.kv_layout(self.gs.load(config))
                # The selective rule's figure is vLLM's startup check for one request.
                active = 32768
                self.assertEqual(layout.request_pages, WC.pinned_request_pages(layout, active, 4096, gib << 30))
                self.assertEqual(config['kv_layout']['request_pages'], layout.request_pages)
                self.assertIn('--max-model-len', command)
                self.assertEqual(command[command.index('--max-model-len') + 1], '262144')
                value = self.gs.load(config)
                self.assertEqual(value['kv_session']['grammar'], 'gemma4')

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'thinking off only'):
            self.prepare(kv_session_thinking='on')
        # Without the selective rule pool 0 must hold one session at 262,144 tokens.
        with self.assertRaisesRegex(ValueError, 'one session at the 262144-token limit'):
            self.prepare('full', kv_cache_gib=28)
        # One share cannot pin a 150 GiB arena; two can.
        with self.assertRaisesRegex(ValueError, 'more than the 80.9 GiB a --mem=113560M job'):
            self.prepare(host_memory_shares=1, kv_cache_gib=28)
        with self.assertRaisesRegex(ValueError, 'must be 1 or 2'):
            self.prepare(host_memory_shares=3, kv_cache_gib=28)
        # A Qwen release keeps its budget.
        with self.assertRaisesRegex(ValueError, 'Gemma 4 profile only'):
            self.gs.prepare(None, 'full', None, release=self.gs.qwen, kv_cache_gib=24, page_tokens=16,
                            max_tokens=4096, max_model_len=131072, language_model_only=False,
                            protocol='kv-session', max_running=2, host_memory_shares=2)

    def test_one_session_without_releases(self):
        from remote_ra_http import config as HC
        _, config, _, _ = self.prepare('full', max_running=1, release_idle_pages=False, dram_arena_gib=None,
                                       host_memory_shares=None, max_model_len=131072, kv_session_slots=4)
        value = self.gs.load(config)
        self.assertNotIn('request_pages', config['kv_layout'])
        self.assertEqual((value['kv_session']['slots'], HC.kv_layout(value).request_pages), (4, 0))

    def test_readme_command(self):
        # The Gemma kv-session launch of README section 4.14, in both languages, as a dry run.
        import contextlib
        import io
        import shlex
        import serve
        from unittest.mock import patch
        commands = []
        for name in ('README.md', 'README.zh-CN.md'):
            text = (ROOT / name).read_text(encoding='utf-8')
            found = [b for b in re.findall(r'```sh\n(.*?)```', text, re.S)
                     if '--protocol kv-session' in b and '$RELEASE-gemma4' in b]
            self.assertEqual(len(found), 1, name)
            commands.append(found[0])
        self.assertEqual(commands[0], commands[1])
        output = self.gs.root / 'readme-gemma-mab'
        values = {'$RELEASE-gemma4': str(self.gs.release), '$RUNS/gemma-mab-obs': str(output)}
        argv = [values.get(v, v) for v in shlex.split(commands[0].replace('\\\n', ' '))]
        self.assertEqual(argv[:3], ['python', '-B', 'serve.py'])
        out = io.StringIO()
        with patch.object(sys, 'argv', ['serve.py'] + argv[3:]), contextlib.redirect_stdout(out), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            serve.main()
        dry = json.loads(out.getvalue())
        command = dry['command']
        self.assertEqual(dry['status'], 'SERVICE_DRY_RUN')
        for flag, value in (('--max-num-seqs', '16'), ('--max-model-len', '262144'), ('--block-size', '16'),
                            ('--attention-backend', 'TRITON_ATTN')):
            self.assertEqual(command[command.index(flag) + 1], value)
        layout = json.loads(command[command.index('--additional-config') + 1])['remote_ra_kv_layout']
        self.assertIn('request_pages', layout)

    def test_load_refuses_a_changed_request_pages(self):
        _, config, _, _ = self.prepare(kv_cache_gib=28)
        changed = copy.deepcopy(config)
        changed['kv_layout']['request_pages'] += 1
        with self.assertRaisesRegex(RuntimeError, 'KV layout differs from the server layout'):
            self.gs.load(changed)
        changed = copy.deepcopy(config)
        changed['kv_session'].pop('grammar')
        with self.assertRaisesRegex(RuntimeError, 'kv-session startup record'):
            self.gs.load(changed)


if __name__ == '__main__':
    unittest.main()
