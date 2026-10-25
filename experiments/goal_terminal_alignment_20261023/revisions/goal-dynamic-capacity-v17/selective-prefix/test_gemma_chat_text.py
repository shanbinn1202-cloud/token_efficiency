"""Gemma 4 31B IT on the chat-text protocol (gemma-tb): turns, transcripts, launcher, startup. CPU only.

serve.py --protocol chat-text on a release of the Gemma 4 profile serves Terminus 2 on
Terminal-Bench 2.0 with Gemma's turns (remote_ra_live_v2.chat_text_protocol.GEMMA4_CHAT):
the initial segment through the pinned template as OpenAI text parts with thinking off,
later user messages ``<|turn>user\\n{content.strip()}<turn|>\\n``, the header
``<|turn>model\\n<|channel>thought\\n<channel|>``, the boundary ``<turn|>\\n`` (``\\n``
after a stop id 0, 1, 50 or 106), past replies kept after the empty thought channel.

Two kits, as in test_chat_text_protocol: a character-level stand-in (``GTOK``, which keeps
the pinned ids of Gemma's special tokens) with a stand-in of the pinned template rendered by
jinja2 (``STAND_IN``: the template's text turns, OpenAI text parts, thinking off), and, with
REMOTE_RA_GEMMA4_METADATA and Transformers (the remote-ra-tok venv), the pinned Gemma
tokenizer and template through Transformers' apply_chat_template, as vLLM renders them.

The reference of every served prompt is the template's render of its messages with the
empty thought channel inserted after every past ``<|turn>model\\n`` (``GemmaKit.reference``);
the same prompt without those channels is stock vLLM's prompt (``GemmaKit.stock``: the
template's render, tokenized), token for token. With the pinned tokenizer the chat-text
transcript also reproduces the prompt ids that stock vLLM 0.24.0 served on the GPU in the
stage-0 append session (job 3364469, testdata/gemma4-stage0-append.json) on all 10 turns.
The launcher and startup part runs serve.prepare on the installer's Gemma fixture with the
Terminal-Bench settings (K = 20, 65,536 tokens, 4,096-token replies).
"""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import gemma4_testing as G  # noqa: E402
import install  # noqa: E402
import serve  # noqa: E402
import test_chat_text_protocol as T  # noqa: E402  Terminus client, prompts, protocol-level server
import test_kv_session_protocol as KP  # noqa: E402  jinja render
from test_gemma_install import gemma_fixture  # noqa: E402
from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from remote_ra_live_v2.mswea_protocol import GEMMA4_HEADER, strip_thinking  # noqa: E402

CHANNEL = '<|channel>thought\n<channel|>'
MODEL_TURN = '<|turn>model\n'
TURN_END = 106
SPECIAL = {'<pad>': MP.PAD, '<eos>': MP.EOS, '<bos>': MP.BOS, '<|tool_response>': MP.TOOL_RESPONSE,
           '<|channel>': MP.CHANNEL_OPEN, '<channel|>': MP.CHANNEL_CLOSE, '<|turn>': MP.TURN_OPEN,
           '<turn|>': MP.TURN_CLOSE}


class GemmaCharTokenizer:
    """Character-level stand-in of the pinned Gemma tokenizer: its special tokens keep their
    pinned ids (parsed in text like the HF fast tokenizer), characters are ord + 1000."""
    BASE = 1000

    def __init__(self):
        self.added = sorted(SPECIAL, key=len, reverse=True)
        self.names = {i: t for t, i in SPECIAL.items()}

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for token in self.added:
                if text.startswith(token, i):
                    ids.append(SPECIAL[token])
                    i += len(token)
                    break
            else:
                ids.append(ord(text[i]) + self.BASE)
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for t in ids:
            if t in self.names:
                out.append('' if skip_special_tokens else self.names[t])
            else:
                out.append(chr(t - self.BASE) if t >= self.BASE else '<id%d>' % t)
        return ''.join(out)


GTOK = GemmaCharTokenizer()

# The pinned template's text turns (system, user, assistant; string content or OpenAI text
# parts; a run of assistant messages is one model turn; thinking off), for the stand-in kit.
# With the pinned files, StandInTests checks that it renders what the real template renders.
STAND_IN = r"""{%- macro strip_thinking(text) -%}
    {%- set ns = namespace(result='') -%}
    {%- for part in text.split('<channel|>') -%}
        {%- if '<|channel>' in part -%}
            {%- set ns.result = ns.result + part.split('<|channel>')[0] -%}
        {%- else -%}
            {%- set ns.result = ns.result + part -%}
        {%- endif -%}
    {%- endfor -%}
    {{- ns.result | trim -}}
{%- endmacro -%}
{%- set loop_messages = messages -%}
{{- bos_token -}}
{%- if messages and messages[0]['role'] == 'system' -%}
    {{- '<|turn>system\n' -}}
    {%- if messages[0]['content'] is string -%}
        {{- messages[0]['content'] | trim -}}
    {%- else -%}
        {%- for item in messages[0]['content'] -%}{{- item['text'] | trim + ' ' -}}{%- endfor -%}
    {%- endif -%}
    {{- '<turn|>\n' -}}
    {%- set loop_messages = messages[1:] -%}
{%- endif -%}
{%- set ns = namespace(prev=None) -%}
{%- for message in loop_messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {%- if not (role == 'model' and ns.prev == 'assistant') -%}{{- '<|turn>' + role + '\n' -}}{%- endif -%}
    {%- if message['content'] is string -%}
        {%- if role == 'model' -%}{{- strip_thinking(message['content']) -}}{%- else -%}{{- message['content'] | trim -}}{%- endif -%}
    {%- else -%}
        {%- for item in message['content'] -%}
            {%- if role == 'model' -%}{{- strip_thinking(item['text']) -}}{%- else -%}{{- item['text'] | trim -}}{%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if not (role == 'model' and loop.nextitem is defined and loop.nextitem['role'] == 'assistant') -%}{{- '<turn|>\n' -}}{%- endif -%}
    {%- set ns.prev = message['role'] -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
    {%- if not enable_thinking -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}
{%- endif -%}"""


def openai_parts(messages):
    from gemma4_renderer import openai_parts as parts
    return parts(messages)


class GemmaKit:
    """One tokenizer and the pinned template (or its stand-in) as the server and stock vLLM
    use them: content as OpenAI text parts, enable_thinking False, no tools."""

    def __init__(self, name, encode, decode, render, template):
        self.name, self._encode, self._decode, self._render, self.template = name, encode, decode, render, template
        self.stop_ids = MP.GEMMA4_STOP_IDS
        self.channel = self.encode(CHANNEL)
        self.model_turn = self.encode(MODEL_TURN)

    def encode(self, text):
        return list(self._encode(text))

    def decode(self, ids):
        return self._decode(list(ids))

    def render(self, messages, generation=False):
        """The template's text of plain messages, as vLLM renders them (text parts)."""
        return self._render(openai_parts(messages), generation)

    def render_template(self, template, messages, kwargs):
        """The server's render_template callback (Gemma4SourceRenderer passes text parts)."""
        assert template == self.template
        kwargs = dict(kwargs)
        generation = kwargs.pop('add_generation_prompt')
        assert kwargs == C.GEMMA4_TEMPLATE_KWARGS, kwargs
        return self._render(messages, generation)

    def render_chat(self, messages):
        """Gemma4SourceRenderer.render_chat on this kit: the initial segment only."""
        messages = C.validate_messages(messages)
        assert len(messages) == C.initial_count(messages)
        return self.encode(self.render(messages, False))

    def stock(self, messages):
        """Stock vLLM's prompt ids of a request: the template's render with the generation
        prompt, tokenized without added special tokens."""
        return self.encode(self.render(messages, True))

    def reference(self, messages):
        """The chat-text stream's prompt: the template's render with the empty thought channel
        kept after every past model turn."""
        assert not any('<|turn>' in m['content'] for m in messages), 'turn markers inside content'
        text = self.render(messages, True)
        assert text.endswith(GEMMA4_HEADER)
        body = text[:-len(GEMMA4_HEADER)]
        return self.encode(body.replace(MODEL_TURN, MODEL_TURN + CHANNEL) + GEMMA4_HEADER)

    def without_channels(self, ids):
        """``ids`` without the thought channel after every model turn except the last (the header)."""
        ids, out, k = list(ids), [], 0
        m, c = self.model_turn, self.channel
        header = len(ids) - len(m) - len(c)
        while k < len(ids):
            if k < header and ids[k:k + len(m)] == m and ids[k + len(m):k + len(m) + len(c)] == c:
                out += m
                k += len(m) + len(c)
            else:
                out.append(ids[k])
                k += 1
        return out

    def transcript(self):
        return C.ChatTextTranscript(encode=self.encode, render_chat=self.render_chat, decode=self.decode,
                                    stop_ids=self.stop_ids, grammar=C.GEMMA4_CHAT)

    def reply(self, text, end='turn'):
        """Generated ids of a reply: its text's ids, then <turn|> (``end`` 'turn'), <eos>, <pad>
        or nothing (a reply cut at its bound)."""
        stop = dict(turn=[MP.TURN_CLOSE], eos=[MP.EOS], pad=[MP.PAD], length=[])[end]
        return self.encode(text) + stop


def stand_in_render(messages, generation):
    return KP.jinja_render(STAND_IN, messages, None, generation, bos_token='<bos>', **C.GEMMA4_TEMPLATE_KWARGS)


def char_kit():
    return GemmaKit('gtok', GTOK.encode, lambda ids: GTOK.decode(ids, skip_special_tokens=True),
                    stand_in_render, STAND_IN)


HF = G.tokenizer() if G.metadata_dir() is not None else None
_REAL = []


def real_kit():
    """The pinned Gemma tokenizer and Transformers' apply_chat_template, or None."""
    if HF is None:
        return None
    if not _REAL:
        template = HF.chat_template

        def render(messages, generation):
            return HF.apply_chat_template(messages, chat_template=template, tokenize=False,
                                          add_generation_prompt=generation, **C.GEMMA4_TEMPLATE_KWARGS)
        kit = GemmaKit('gemma4', lambda text: HF.encode(text, add_special_tokens=False),
                       lambda ids: HF.decode(ids, skip_special_tokens=True), render, template)
        kit.tokenizer = HF
        _REAL.append(kit)
    return _REAL[0]


# ---- the turns ------------------------------------------------------------------------------------

class TurnTests(unittest.TestCase):
    def test_gemma_literals_and_the_qwen_turns_unchanged(self):
        g, q = C.GEMMA4_CHAT, C.QWEN_CHAT
        self.assertEqual((g.profile, g.header, g.template_kwargs, g.whole_render),
                         (MP.GEMMA4, GEMMA4_HEADER, {'enable_thinking': False}, False))
        self.assertEqual(g.header, MODEL_TURN + CHANNEL)
        self.assertEqual((q.profile, q.header, q.template_kwargs, q.whole_render),
                         (None, C.HEADER_TEXT, C.TEMPLATE_KWARGS, True))
        self.assertEqual(C.GRAMMARS, {None: q, MP.GEMMA4: g})
        self.assertIs(C.grammar_of(None), q)
        with self.assertRaises(LiveV2Error):
            C.grammar_of('qwen3.6-27b')
        self.assertEqual(g.user_piece('  {"a": 1}\n\n'), '<|turn>user\n{"a": 1}<turn|>\n')
        self.assertEqual(q.user_piece(' x '), C.user_piece(' x '))
        self.assertEqual(g.model_piece([' {"a": 1} ']), GEMMA4_HEADER + '{"a": 1}<turn|>\n')
        self.assertEqual(g.model_piece(['a<|channel>thought\nhidden<channel|>b', ' c ']), GEMMA4_HEADER + 'abc<turn|>\n')
        u, a = (lambda c: dict(role='user', content=c)), (lambda c: dict(role='assistant', content=c))
        self.assertEqual(g.history_pieces([a('x'), a('y'), u('u'), u('v'), a('z')]),
                         [g.model_piece(['x', 'y']), g.user_piece('u'), g.user_piece('v'), g.model_piece(['z'])])
        stops = MP.GEMMA4_STOP_IDS
        for generated, text in (((5, 106), '\n'), ((5, 1), '\n'), ((5, 0), '\n'), ((5, 50), '\n'),
                                ((5,), '<turn|>\n'), ((), '<turn|>\n')):
            self.assertEqual(g.boundary_text(generated, stops), text)
        for generated in ((), (5,), (5, 7), (7, 5)):
            self.assertEqual(q.boundary_text(generated, frozenset({7})), C.boundary_text(generated, frozenset({7})))

    def test_startup_record_and_config(self):
        qwen = C.record()
        self.assertEqual(qwen, dict(slots=512, idle_timeout_s=600, evict_after_s=30, header_text=C.HEADER_TEXT,
                                    template_kwargs=dict(enable_thinking=False, preserve_thinking=True),
                                    session_header='X-Session-ID', binding='initial_segment', busy_wait_s=30))
        gemma = C.record(profile=MP.GEMMA4)
        self.assertEqual(gemma, dict(qwen, header_text=GEMMA4_HEADER, template_kwargs=dict(enable_thinking=False)))
        self.assertEqual(C.record(slots=9, idle_timeout_s=60, evict_after_s=5, profile=MP.GEMMA4),
                         dict(gemma, slots=9, idle_timeout_s=60, evict_after_s=5))
        base = dict(protocol=C.PROTOCOL, tasks=C.slot_tasks(512), task=C.slot_tasks(1)[0], max_running_sessions=20,
                    policy='observation_query')
        self.assertEqual(C.check_config(dict(base, chat_text=qwen)), qwen)
        self.assertEqual(C.check_config(dict(base, chat_text=gemma, model_profile=MP.GEMMA4)), gemma)
        for config in (dict(base, chat_text=gemma), dict(base, chat_text=qwen, model_profile=MP.GEMMA4),
                       dict(base, chat_text=gemma, model_profile='qwen3.6-27b'),
                       dict(base, chat_text=dict(gemma, template_kwargs=dict(enable_thinking=True)),
                            model_profile=MP.GEMMA4)):
            with self.subTest(config=config), self.assertRaises(LiveV2Error):
                C.check_config(config)
        options = C.options(dict(base, chat_text=gemma, model_profile=MP.GEMMA4, page_tokens=16, max_tokens=4096,
                                 effective_server_config=dict(max_model_len=65536), selection_contract=None))
        self.assertEqual(options['model_profile'], MP.GEMMA4)
        self.assertNotIn('model_profile', C.options(dict(base, chat_text=qwen, page_tokens=16, max_tokens=4096,
                                                         effective_server_config=dict(max_model_len=65536),
                                                         selection_contract=None)))

    def test_transcript_refuses_unknown_turns(self):
        kit = char_kit()
        with self.assertRaises(LiveV2Error):
            C.ChatTextTranscript(encode=kit.encode, render_chat=kit.render_chat, decode=kit.decode,
                                 stop_ids=kit.stop_ids, grammar=NS(header='x'))


class TemplateCheckTests(unittest.TestCase):
    def check(self, render):
        return C.GEMMA4_CHAT.check_template(lambda messages, generation: render(openai_parts(messages), generation))

    def test_stand_in_passes_and_changed_templates_are_refused(self):
        self.assertEqual(self.check(stand_in_render), GEMMA4_HEADER)
        changes = (
            ('drops <bos>', lambda m, g: stand_in_render(m, g)[len('<bos>'):]),
            ('a default system turn', lambda m, g: stand_in_render(
                [dict(role='system', content=[dict(type='text', text='You are helpful.')])] + m
                if m[0]['role'] != 'system' else m, g)),
            ('thinking on', lambda m, g: KP.jinja_render(STAND_IN, m, None, g, bos_token='<bos>', enable_thinking=True)),
            ('keeps the history channel', lambda m, g: stand_in_render(m, g).replace(
                '<|turn>model\n{', '<|turn>model\n' + CHANNEL + '{')),
            ('runs not merged', lambda m, g: stand_in_render(m, g).replace('xy', 'x<turn|>\n<|turn>model\ny')),
            ('system as a string', lambda m, g: stand_in_render(
                [dict(x, content=x['content'][0]['text']) if x['role'] == 'system' else x for x in m], g)))
        for name, render in changes:
            with self.subTest(change=name), self.assertRaises(LiveV2Error):
                self.check(render)


# ---- transcripts --------------------------------------------------------------------------------

class TranscriptChecks:
    """Transcript invariants on ``self.kit`` (the stand-in, or the pinned tokenizer)."""

    def play(self, transcript, instruction, replies, observations, ends=None, system=None):
        """Turn by turn: render, check against the reference and stock vLLM, commit the reply's ids."""
        kit = self.kit
        messages = ([dict(role='system', content=system)] if system else []) + \
            [dict(role='user', content=T.first_prompt(instruction))]
        prompts = []
        for k, reply in enumerate(replies):
            plan = transcript.plan(messages)
            tokens, stable, span = transcript.render(messages)
            self.assertEqual(tokens, plan['tokens'])
            prompts.append(tokens)
            self.assertEqual(list(tokens), kit.reference(messages), 'turn %d differs from the template' % k)
            self.assertEqual(kit.without_channels(tokens), kit.stock(messages), 'turn %d differs from stock' % k)
            self.assertEqual(len(tokens) - len(kit.stock(messages)), len(kit.channel) * k)
            self.assertEqual(tokens[-len(transcript.header):], transcript.header)
            self.assertEqual(list(transcript.header), kit.encode(GEMMA4_HEADER))
            if k == 0:
                self.assertEqual(tokens[0], MP.BOS)
                self.assertEqual((stable, span, transcript.initial_length),
                                 (len(tokens) - len(transcript.header), None,
                                  len(kit.encode(kit.render(messages[:C.initial_count(messages)])))))
            else:
                self.assertEqual(span, (stable, len(tokens) - len(transcript.header)))
                self.assertEqual(list(tokens[span[0]:span[1]]),
                                 kit.encode(C.GEMMA4_CHAT.user_piece(messages[-1]['content'])))
            end = 'turn' if ends is None else ends[k]
            generated = kit.reply(reply, end)
            committed = transcript.commit(tokens, generated)
            self.assertEqual(committed['text'], kit.decode(generated))
            self.assertFalse(committed['rebuild'])
            self.assertEqual(list(committed['boundary']), kit.encode('\n' if end != 'length' else '<turn|>\n'))
            messages = messages + [dict(role='assistant', content=committed['text']),
                                   dict(role='user', content=observations[k])]
        return messages, prompts

    def test_terminus_turns_equal_the_template(self):
        transcript = self.kit.transcript()
        replies = [T.action(k, keystrokes=ks) for k, ks in enumerate(
            ['ls -la\n', 'cat > /app/a.json << \'EOF\'\n{"key": "va\\"l\\\\ue"}\nEOF\n', 'echo "\\t tab"\n',
             'python3 -c "print(1)"\n'])]
        observations = [T.observation(k, size=k * 3) for k in range(4)]
        messages, prompts = self.play(transcript, 'Write /app/a.json with one key.', replies, observations)
        self.assertEqual(transcript.turn, 4)
        self.assertEqual(list(transcript.plan(messages)['tokens']), self.kit.reference(messages))
        # A rebuild of the same conversation (a fresh session) gives the continuing stream.
        fresh = self.kit.transcript()
        self.assertEqual(fresh.plan(messages)['tokens'], transcript.plan(messages)['tokens'])

    def test_system_message_and_non_ascii(self):
        transcript = self.kit.transcript()
        instruction = ('把 /app/数据.csv 转成 JSON，保留 "名称" 字段；Ümlaut café naïve — résumé; 日本語のテスト；'
                       'emoji 🚀✨ and math ∑∫√')
        replies = [T.action(0, analysis='分析：先列出文件 🚀'), T.action(1, keystrokes='iconv -f utf-8 数据.csv\n')]
        messages, prompts = self.play(transcript, instruction, replies, [T.observation(0), '新的终端输出：\n完成 ✓\n'],
                                      system='  You solve tasks.  ')
        # The system message is rendered as text parts: one more space than a plain string.
        plain = self.kit.encode(self.kit._render(messages[:2], False))
        self.assertEqual(len(self.kit.render_chat(messages[:2])), len(plain) + 1)

    def test_stops_and_a_length_cut(self):
        transcript = self.kit.transcript()
        replies = [T.action(0)[:40], T.action(1), T.action(2), T.action(3)]
        # A length-cut reply (no stop id) gets <turn|>\n, which the template renders after its text too;
        # <eos> and <pad> close the turn like <turn|>: the boundary is \n.
        self.play(transcript, 'Count the files.', replies,
                  [T.OUTPUT_LIMIT, T.observation(1), T.observation(2), T.observation(3)],
                  ends=['length', 'turn', 'turn', 'turn'])
        self.assertEqual(transcript.turns[0]['ids'][-2:], tuple(self.kit.encode('<turn|>\n')))
        for end, stop in (('eos', MP.EOS), ('pad', MP.PAD)):
            t = self.kit.transcript()
            messages = [dict(role='user', content=T.first_prompt('Stop %s.' % end))]
            tokens, _, _ = t.render(messages)
            committed = t.commit(tokens, self.kit.reply(T.action(0), end))
            self.assertEqual(list(committed['boundary']), self.kit.encode('\n'))
            self.assertEqual(C.finish_reason(committed['generated'], self.kit.stop_ids, 4096), 'stop')
            self.assertEqual(t.stream[-2:], (stop, self.kit.encode('\n')[0]))

    def test_rebuilds_runs_and_refusals_change_nothing(self):
        kit, transcript = self.kit, self.kit.transcript()
        replies = [T.action(k) for k in range(3)]
        messages, _ = self.play(transcript, 'Fix the build.', replies, [T.observation(k) for k in range(3)])
        stream = transcript.stream
        handoff = [messages[0], dict(role='user', content=T.QUESTION_PROMPT % ('Fix the build.', 'done', 'screen')),
                   dict(role='assistant', content='1. Which files?'), dict(role='user', content=T.HANDOFF % 'answers')]
        plan = transcript.plan(handoff)
        self.assertTrue(plan['rebuild'])
        self.assertEqual((transcript.stream, transcript.turn, transcript.pending), (stream, 3, None))
        with self.assertRaises(C.ChatTextError):
            transcript.plan(handoff[:-1])
        nxt = messages + [dict(role='assistant', content=T.action(3)), dict(role='user', content=T.observation(3))]
        self.assertFalse(transcript.plan(messages)['rebuild'])
        transcript.render(messages)
        transcript.abandon()
        self.assertEqual((transcript.stream, transcript.turn, transcript.pending), (stream, 3, None))
        self.assertTrue(transcript.plan(nxt)['rebuild'])
        tokens, stable, span = transcript.render(handoff)
        self.assertEqual((list(tokens), span, transcript.turn), (kit.reference(handoff), None, 0))
        self.assertEqual(kit.without_channels(tokens), kit.stock(handoff))
        committed = transcript.commit(tokens, kit.reply(T.action(9)))
        self.assertTrue(committed['rebuild'])
        self.assertEqual((transcript.turn, transcript.sessions), (1, 2))
        echo = handoff + [dict(role='assistant', content='  ' + committed['text'] + '\n'),
                          dict(role='user', content=T.observation(10))]
        self.assertFalse(transcript.plan(echo)['rebuild'])
        # A run of assistant messages and two user messages in a row: one model turn, two user turns.
        odd = [messages[0], dict(role='assistant', content=' first part '),
               dict(role='assistant', content='second <|channel>thought\nx<channel|>part'),
               dict(role='user', content='u1'), dict(role='user', content=' u2 ')]
        fresh = kit.transcript()
        self.assertEqual(list(fresh.plan(odd)['tokens']), kit.reference(odd))
        self.assertEqual(kit.without_channels(fresh.plan(odd)['tokens']), kit.stock(odd))
        transcript.release()
        self.assertEqual((transcript.stream, transcript.turn, transcript.history), ((), 0, None))

    def test_exact_reply_ids_survive_a_noncanonical_split(self):
        kit, transcript = self.kit, self.kit.transcript()
        messages = [dict(role='user', content=T.first_prompt('Split check.'))]
        tokens, _, _ = transcript.render(messages)
        text = T.action(0)
        split = next((i for i in range(1, len(text)) if kit.encode(text[:i]) + kit.encode(text[i:])
                      != kit.encode(text)), None)
        generated = (kit.encode(text) if split is None else kit.encode(text[:split]) + kit.encode(text[split:])) \
            + [TURN_END]
        committed = transcript.commit(tokens, generated)
        nxt = messages + [dict(role='assistant', content=committed['text']), dict(role='user', content=T.observation(1))]
        plan = transcript.plan(nxt)
        self.assertFalse(plan['rebuild'])
        self.assertEqual(list(plan['tokens'][len(tokens):len(tokens) + len(generated)]), generated)


class CharTranscriptTests(TranscriptChecks, unittest.TestCase):
    def setUp(self):
        self.kit = char_kit()


@unittest.skipUnless(HF is not None, 'pinned Gemma 4 tokenizer (%s) and Transformers required' % G.ENV)
class GemmaTranscriptTests(TranscriptChecks, unittest.TestCase):
    """The same invariants on the pinned Gemma tokenizer and Transformers' template."""

    def setUp(self):
        self.kit = real_kit()

    def test_literal_ids(self):
        kit = self.kit
        self.assertEqual(kit.encode(GEMMA4_HEADER), [105, 4368, 107, 100, 45518, 107, 101])
        self.assertEqual(kit.channel, [100, 45518, 107, 101])        # 4 tokens per past reply
        self.assertEqual(kit.encode('<turn|>\n'), [106, 107])
        self.assertEqual(kit.encode('\n'), [107])
        self.assertEqual(kit.encode('<|turn>user\nhello<turn|>\n'), [105, 2364, 107, 23391, 106, 107])

    def test_pinned_template_check_through_the_renderer(self):
        from gemma4_renderer import Gemma4SourceRenderer
        import test_gemma_renderer as TR
        renderer = Gemma4SourceRenderer(model_dir=G.metadata_dir(), tokenizer=HF, render_template=TR.hf_render(HF),
                                        verify_identity=lambda: True, page_tokens=16)
        record = C.record(profile=MP.GEMMA4)
        make = renderer.chat_transcripts(record)
        t = make()
        self.assertIs(t.grammar, C.GEMMA4_CHAT)
        self.assertEqual(t.stop_ids, frozenset({0, 1, 50, 106}))
        messages = [dict(role='system', content=' s '), dict(role='user', content=T.first_prompt('x'))]
        self.assertEqual(renderer.render_chat(messages), self.kit.render_chat(messages))
        with self.assertRaises(Exception):
            renderer.render_chat(messages + [dict(role='assistant', content='a'), dict(role='user', content='b')])
        for bad in (C.record(), dict(record, template_kwargs=dict(enable_thinking=False, preserve_thinking=True))):
            with self.assertRaises(Exception):
                renderer.chat_transcripts(bad)
        # The engine renders a chat-text source only on a Gemma chat-text transcript.
        from runtime_renderer import CHAT_SOURCE_SCHEMA
        source = dict(schema=CHAT_SOURCE_SCHEMA, messages=[dict(role='user', content='x')])
        with self.assertRaisesRegex(Exception, "another model's turns"):
            qwen_turns = C.ChatTextTranscript(encode=GTOK.encode, render_chat=lambda m: [2], decode=GTOK.decode,
                                              stop_ids=frozenset({1}))
            renderer.render_growing(source, None, None, transcript=qwen_turns, session=0, turn=0, local_turn=0,
                                    generation='g', model=None, renderer=None, native_prefix=None)

    def test_stand_in_renders_what_the_pinned_template_renders(self):
        conversations = [
            [dict(role='user', content=T.first_prompt('任务 ✓'))],
            [dict(role='system', content='  sys  '), dict(role='user', content=' u ')],
            [dict(role='user', content=T.first_prompt('t')), dict(role='assistant', content=T.action(0)),
             dict(role='user', content=T.observation(0, size=2))],
            [dict(role='user', content='u'), dict(role='assistant', content=' a1 '),
             dict(role='assistant', content='a2 <|channel>thought\nz<channel|> b'), dict(role='user', content='v')],
            [dict(role='system', content='s'), dict(role='user', content='u'), dict(role='user', content='w ')]]
        for messages in conversations:
            for generation in (False, True):
                with self.subTest(messages=messages, generation=generation):
                    self.assertEqual(char_kit().render(messages, generation), self.kit.render(messages, generation))
        self.assertEqual(C.GEMMA4_CHAT.check_template(
            lambda m, g: self.kit._render(openai_parts(m), g)), GEMMA4_HEADER)

    def test_chat_text_transcript_reproduces_the_stage0_stock_reference(self):
        """The stage-0 append session (stock vLLM 0.24.0 on the GPU, job 3364469): system, task
        and nine observations. The chat-text transcript's initial segment, user pieces, header and
        boundaries reproduce every served prompt's ids, as the XML transcript does."""
        stage0 = G.Stage0()
        t = self.kit.transcript()
        decode = lambda ids: HF.decode(list(ids), skip_special_tokens=True)
        equal = []
        for k, turn in enumerate(stage0.turns):
            messages = stage0.messages(k, decode)
            tokens, stable, span = t.render(messages)
            equal.append((len(tokens), G.ids_sha(tokens)) == (turn['prompt_len'], turn['prompt_sha256']))
            if k:
                self.assertFalse(t.pending['rebuild'])
            out = t.commit(tokens, turn['completion_ids'])
            self.assertEqual(list(out['boundary']), turn['boundary_ids'])
        self.assertEqual(equal, [True] * 10)
        fresh = self.kit.transcript()
        tokens = fresh.plan(stage0.messages(9, decode))['tokens']
        self.assertEqual((len(tokens), G.ids_sha(tokens)), (stage0.turns[9]['prompt_len'],
                                                            stage0.turns[9]['prompt_sha256']))

    def test_json_replies_round_trip(self):
        kit = self.kit
        for k, keystrokes in enumerate(['ls -la\n', "grep -n '\"x\"' a.py\n", 'printf "\\u00e9\\n"\n',
                                        'cat <<EOF > t.txt\nä ö ü — 中文\nEOF\n', '\x03', 'C-c', '  spaced  \n']):
            text = T.action(k, keystrokes=keystrokes)
            ids = kit.encode(text)
            self.assertEqual(kit.decode(ids + [TURN_END]), text)
            self.assertEqual(json.loads(kit.decode(ids + [TURN_END]))['commands'][0]['keystrokes'], keystrokes)


# ---- the Terminus client against the protocol-level server ----------------------------------------

class TerminusTests(unittest.TestCase):
    """A Terminus 2 client against the protocol-level server with Gemma's turns: every prompt is
    the reference render (and stock vLLM's without the channels), the summary hand-off a
    rebuild, the questions call its own session, a context refusal changes nothing."""

    def setUp(self):
        self.kit = real_kit() or char_kit()

    def check_prompts(self, server):
        cut = set()
        for entry in server.log:
            if entry['rebuild'] or entry['binding'] is not None:
                cut.discard(entry['slot'])
            if entry['slot'] not in cut:
                self.assertEqual(entry['prompt'], self.kit.reference(entry['messages']))
                self.assertEqual(self.kit.without_channels(entry['prompt']), self.kit.stock(entry['messages']))
            if entry['finish'] == 'length':
                cut.add(entry['slot'])

    def test_episode_summary_and_handoff(self):
        server = T.ProtocolServer(self.kit)
        tokens = lambda messages: len(self.kit.encode(''.join(m['content'] for m in messages)))
        limit = 4000 if self.kit.name == 'gemma4' else 12000
        agent = T.Terminus('Build it.', send=server.send, count=tokens, context_limit=limit, threshold=limit * 2 // 3)
        prompt = T.first_prompt('Build it.')
        for k in range(10):
            T.run(agent.step(prompt))
            prompt = T.observation(k, size=20)
        self.assertIn('summary', agent.events)
        kinds = ['aside' if e['messages'][0]['content'].startswith('You are picking up') else
                 'rebuild' if e['rebuild'] else 'extend' for e in server.log]
        start = kinds.index('aside')
        self.assertEqual(kinds[start - 1:start + 3], ['extend', 'aside', 'extend', 'rebuild'])
        self.check_prompts(server)

    def test_context_refusal_and_output_length(self):
        # Gemma's tokenizer packs several characters a token: longer observations than the stand-in's,
        # and a context above Terminus's 4,000-token unwind target, so that history is kept.
        limit, bound, size = (6000, 1024, 40) if self.kit.name == 'gemma4' else (3000, 1536, 4)
        server = T.ProtocolServer(self.kit, limit=limit, bound=bound)
        agent = T.Terminus('Grow.', send=server.send, max_tokens=None,
                           count=lambda m: len(self.kit.encode(''.join(x['content'] for x in m))),
                           context_limit=limit, threshold=0)
        calls = {'n': 0}

        def model(messages, slot):
            calls['n'] += 1
            return T.action(len(messages)) * (60 if calls['n'] == 2 else 1)
        server.model = model
        prompt = T.first_prompt('Grow.')
        for k in range(12):
            T.run(agent.step(prompt))
            prompt = T.observation(k, size=size)
        self.assertIn('output_length', agent.events)
        self.assertIn('context_error', agent.events)
        refused = [r for r in agent.requests if r[1] == 400]
        self.assertTrue(refused and all(T.harbor_context_error(r[2]) for r in refused))
        self.assertTrue(all(len(e['prompt']) <= limit for e in server.log))
        self.assertTrue(any(e['rebuild'] for e in server.log))
        self.check_prompts(server)


# ---- the launcher and the startup config ----------------------------------------------------------

TB = dict(protocol='chat-text', kv_cache_gib=28, page_tokens=16, max_tokens=4096, max_model_len=65536,
          language_model_only=True, max_running=20, dram_arena_gib=64, connector_pool_gib=1, release_idle_pages=True,
          cuda_graphs='full', decode_fastpath=True, chat_text_slots=512, chat_text_idle_timeout=600,
          chat_text_evict_after=30)
BUDGET = dict(active_token_budget=8192)


class LauncherTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, self.release = gemma_fixture(self.root)
            install.install(site, gemma, self.release, package_root=package, apply=True, model_profile=MP.GEMMA4)
            self.qwen = self.root / 'qwen-release'
            install.install(site, model, self.qwen, package_root=package, apply=True)
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))
        self.n = 0

    def prepare(self, policy='observation_query', selection=BUDGET, release=None, **kw):
        self.n += 1
        return serve.prepare(release or self.release, None, policy, self.root / ('out%d' % self.n), 8000, selection,
                             **dict(TB, **kw))

    def load(self, config):
        self.n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                   REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
        with patch.dict(os.environ, env):
            return HC.load()

    def test_terminal_bench_servers(self):
        for policy, selection, kw in (('observation_query', BUDGET, {}), ('random_fill', BUDGET, {}),
                                      ('full', None, dict(max_running=19)), ('full', None, dict(kv_cache_gib=32)),
                                      ('random50', None, dict(kv_cache_gib=32))):
            with self.subTest(policy=policy, kw=kw):
                _, config, command, env = self.prepare(policy, selection, **kw)
                K = kw.get('max_running', 20)
                self.assertEqual((config['protocol'], config['model_profile'], config['max_running_sessions']),
                                 (C.PROTOCOL, MP.GEMMA4, K))
                self.assertEqual(config['chat_text'], C.record(profile=MP.GEMMA4))
                self.assertEqual((config['tasks'], config['task']), (C.slot_tasks(512), C.slot_tasks(1)[0]))
                layout = I0.for_window_server(16, K, 1024, 8192)
                if selection is not None:
                    # The selective rule (gemma-mab): the layout pins its figure for vLLM's own
                    # startup check, 2,630 global Pages at K = 20, 28 GiB and an 8,192-token budget.
                    import dataclasses
                    from remote_ra_live_v2.window_capacity import pinned_request_pages
                    contract = config['selection_contract']
                    active = max(contract['active_token_budget'], contract['switch_threshold_tokens'])
                    layout = dataclasses.replace(layout, request_pages=pinned_request_pages(
                        layout, active, TB['max_tokens'], TB['kv_cache_gib'] << 30))
                    self.assertEqual(layout.request_pages, 2630)
                self.assertEqual(config['kv_layout'], I0.to_additional_config(layout)[I0.LAYOUT_KEY])
                self.assertIs(config['release_idle_pages'], True)
                argv = command[command.index('--model'):]
                self.assertEqual(argv[argv.index('--max-num-seqs') + 1], str(K))
                self.assertEqual(argv[argv.index('--attention-backend') + 1], 'TRITON_ATTN')
                self.assertEqual(argv[argv.index('--chat-template-content-format') + 1], 'openai')
                value = self.load(config)
                self.assertEqual((HC.model_profile(value), HC.chat_text(value), HC.running_sessions(value)),
                                 (MP.GEMMA4, C.record(profile=MP.GEMMA4), K))
                self.assertEqual(HC.kv_layout(value), layout)

    def test_refusals(self):
        cases = (
            (dict(policy='full', selection=None), 'use at most 19 sessions or --kv-cache-gib 32'),
            (dict(policy='goal_query', selection=dict(BUDGET, no_goal_rewriter=True)), "goal_query's Goal is SWE-centric"),
            (dict(max_running=25), 'max_running_sessions must be 1..24'),
            (dict(max_running=24), 'use at most 23 sessions or --kv-cache-gib 32'),
            (dict(max_running=None), 'needs --max-running-sessions K'),
            (dict(chat_text_slots=1025), '--chat-text-slots must be 1..1024'),
            (dict(chat_text_evict_after=601), '--chat-text-evict-after must be'),
            (dict(dram_arena_gib=None), 'more than the 80.9 GiB a --mem=113560M job leaves'),
            (dict(language_model_only=False), 'requires --language-model-only'),
            (dict(decode_fastpath2=True, decode_fastpath3=True), 'decode-fastpath3 is not available'),
            # kv-session runs since gemma-mab (test_gemma_kv_session), with thinking off only.
            (dict(protocol='kv-session', chat_text_slots=None, chat_text_idle_timeout=None, chat_text_evict_after=None),
             'kv-session serves --kv-session-thinking off only'))
        for change, pattern in cases:
            change = dict(change)
            policy, selection = change.pop('policy', 'observation_query'), change.pop('selection', BUDGET)
            with self.subTest(change=change), self.assertRaises(Exception) as caught:
                self.prepare(policy, selection, **change)
            self.assertIn(pattern, str(caught.exception))
        with self.assertRaisesRegex(ValueError, 'takes no --task'):
            serve.prepare(self.release, [dict(instance_id='a', problem_statement='b')], 'full', self.root / 'tasked',
                          8000, None, **dict(TB, kv_cache_gib=32))

    def test_startup_config_pins(self):
        _, config, _, _ = self.prepare()

        def refused(pattern, edit):
            value = copy.deepcopy(config)
            edit(value)
            with self.assertRaisesRegex(RuntimeError, pattern):
                self.load(value)
        refused('chat-text record differs from the protocol pins', lambda v: v.update(chat_text=C.record()))
        refused('chat-text record differs', lambda v: v['chat_text'].update(template_kwargs=dict(enable_thinking=True)))
        refused('pins its XML grammar, the kv-session or the chat-text protocol',
                lambda v: v.update(protocol='mswea_xml_v1'))
        # The Gemma profile serves kv-session too (gemma-mab): a chat-text config is not one
        # (its 512 slot tasks are more than a kv-session server registers).
        refused('startup task list of 1..64 tasks required', lambda v: v.update(protocol='kv_session_v1'))
        refused('KV layout differs from the server layout', lambda v: v['kv_layout'].update(window_slots=16))
        refused('goal_query|chat-text serves', lambda v: v.update(policy='goal_query'))
        # Without the profile key the config is Qwen's: the Gemma record (and layout) is refused.
        refused('chat-text record differs|pinned KV layout', lambda v: v.pop('model_profile'))

    def test_readme_command(self):
        # The Terminal-Bench launch of README section 4.14, in both languages, as a dry run.
        import contextlib
        import re
        import shlex
        commands = []
        for name in ('README.md', 'README.zh-CN.md'):
            text = (ROOT / name).read_text(encoding='utf-8')
            found = [b for b in re.findall(r'```sh\n(.*?)```', text, re.S)
                     if '--protocol chat-text' in b and '$RELEASE-gemma4' in b]
            self.assertEqual(len(found), 1, name)
            commands.append(found[0])
        self.assertEqual(commands[0], commands[1])
        values = {'"$RELEASE-gemma4"': str(self.release), '$RELEASE-gemma4': str(self.release),
                  '$RUNS/tb-gemma-obs': str(self.root / 'readme-out')}
        argv = [values.get(v, v) for v in shlex.split(commands[0].replace('\\\n', ' '))]
        self.assertEqual(argv[:3], ['python', '-B', 'serve.py'])
        out = io.StringIO()
        with patch.object(sys, 'argv', ['serve.py'] + argv[3:]), contextlib.redirect_stdout(out):
            serve.main()
        command = json.loads(out.getvalue())['command']
        for flag, value in (('--max-num-seqs', '20'), ('--max-model-len', '65536'), ('--block-size', '16'),
                            ('--kv-cache-memory-bytes', str(28 << 30)), ('--attention-backend', 'TRITON_ATTN'),
                            ('--max-num-batched-tokens', '8192')):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_the_qwen_chat_text_server_is_unchanged(self):
        # The Qwen Terminal-Bench server's flags (perf-tb16).
        qwen = dict(protocol='chat-text', page_tokens=16, layout='decoupled', kv_cache_gib=32, max_tokens=4096,
                    max_model_len=65536, max_running=20, dram_arena_gib=64, connector_pool_gib=1, cuda_graphs='full',
                    decode_fastpath=True, language_model_only=True, chat_text_slots=512, chat_text_idle_timeout=600,
                    chat_text_evict_after=30)
        _, config, command, _ = serve.prepare(self.qwen, None, 'observation_query', self.root / 'qwen-out', 8000,
                                              BUDGET, **qwen)
        self.assertNotIn('model_profile', config)
        self.assertEqual(config['chat_text'], C.record())
        self.assertEqual(config['chat_text']['header_text'], C.HEADER_TEXT)
        self.assertEqual(HC.chat_text(self.load(config)), C.record())


if __name__ == '__main__':
    unittest.main()
