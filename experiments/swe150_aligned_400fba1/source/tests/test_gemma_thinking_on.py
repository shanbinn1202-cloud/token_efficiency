"""Thinking ON for the Gemma 4 31B IT profile (gemma-thinking, Tier 1): serve.py --enable-thinking on
Gemma's XML protocol and chat-text (the keep mode) and --kv-session-thinking on. CPU only; not GPU
evidence. Design: docs/thinking-gemma/DESIGN.md.

- the pins: the Gemma ``thinking`` record (header ``<|turn>model\\n``, ``enable_thinking=True``, the
  channel markers, the split rule), the chat_text and kv_session records with thinking on; the
  Qwen records and Gemma's thinking-off records unchanged (literal values); the launcher adds the
  record and no vLLM argument; refusals (kv-session --enable-thinking; strip on Gemma in Tier 1);
- the reply split on ids (thinking.Gemma4Split) against a model of vLLM 0.24.0's gemma4 reasoning
  parser on the text it decodes with special tokens kept (``vllm_gemma4``): fixed cases (no channel,
  closed, unclosed, a channel inside C, the parser's injection cases) and random replies;
- MarkerStop counts the marker in the content only, in all three of its modes and in async C's
  prediction, the incremental answers equal to the full ones on random outputs;
- the transcripts: XML and chat-text appends when the reasoning comes back and rebuilds when it does
  not; a rebuild without the reasoning is the stock thinking-on render (the template with
  enable_thinking=True), one with it reproduces the append stream; the initial segment's
  ``<|think|>`` system turn; the routes' responses and evidence;
- kv-session: the two thinking-on headers (checks, refusals, phase-2 suffix), and with the pinned
  template a MemoryAgentBench-shaped session whose every prompt is the template's render with
  enable_thinking=True and preserve_thinking=True;
- end to end: Terminus 2 sessions on test_gemma_chat_text_e2e's server (the packaged router, the
  frontend and the window engine) with thinking on; a MemoryAgentBench-shaped kv-session session
  with the two headers through test_gemma_kv_session_e2e's route server.

Two kits: a character-level stand-in of the Gemma tokenizer (the pinned ids of its special tokens)
with a stand-in of the pinned template's text turns (``STAND_IN_ON``, checked against the real
template when it is there), and, with REMOTE_RA_GEMMA4_METADATA and Transformers (the remote-ra-tok
venv), the pinned tokenizer and template through apply_chat_template and the real
Gemma4SourceRenderer. All reasoning and content is synthetic.
"""
import asyncio
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import gemma4_testing as G  # noqa: E402
import serve  # noqa: E402
import test_chat_text_protocol as T  # noqa: E402  Terminus prompts and actions
import test_gemma_chat_text as GT  # noqa: E402  the Gemma char tokenizer's specials, the pinned tokenizer
import test_kv_session_protocol as KP  # noqa: E402  jinja render

from remote_ra_http import chat_text_route as CR  # noqa: E402
from remote_ra_http import kv_session_route as KR  # noqa: E402
from remote_ra_http import mswea_route  # noqa: E402
from remote_ra_live_v2 import async_sched as A  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2 import mswea_protocol as XML  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

GEMMA = MP.GEMMA4
OPEN, CLOSE, TURN_CLOSE = MP.CHANNEL_OPEN, MP.CHANNEL_CLOSE, MP.TURN_CLOSE
HEADER = '<|turn>model\n'
THOUGHT = '<|channel>thought\n'
MARKER = XML.MARKER
STOPS = MP.GEMMA4_STOP_IDS
ON_XML = TH.record(TH.GEMMA4_XML, profile=GEMMA)
ON_CHAT = TH.record(TH.CHAT, profile=GEMMA)
SYSTEM = 'You are a helpful assistant that can interact with a computer shell.'
TASK = '<pr_description>\nFix the index bug in target.py\n</pr_description>'


# ---- the kits ---------------------------------------------------------------------------------------

# The char stand-in's special tokens: test_gemma_chat_text's and the thinking / tool ones (pinned ids).
SPECIAL = dict(GT.SPECIAL, **{'<|think|>': 98, '<|tool_call>': 48, '<tool_call|>': 49, '<tool_response|>': 51,
                              '<|"|>': 52})


class ThinkTokenizer:
    """Character-level stand-in of the pinned Gemma tokenizer with the thinking tokens: special
    tokens keep their pinned ids (parsed in text), characters are ord + 1000."""
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
        return ''.join(('' if skip_special_tokens else self.names[t]) if t in self.names else chr(t - self.BASE)
                       for t in ids)


TOK = ThinkTokenizer()

# The pinned template's text turns with enable_thinking (system with <|think|>, user and assistant
# with string content or OpenAI text parts, a run of assistant messages as one model turn, every
# reasoning dropped: in these protocols every model message precedes the last user message). With
# the pinned files StandInTests checks it renders what the real template renders.
STAND_IN_ON = r"""{%- macro strip_thinking(text) -%}
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
{%- set enable_thinking = enable_thinking | default(false) -%}
{{- bos_token -}}
{%- if enable_thinking or (messages and messages[0]['role'] == 'system') -%}
    {{- '<|turn>system\n' -}}
    {%- if enable_thinking -%}{{- '<|think|>\n' -}}{%- endif -%}
    {%- if messages and messages[0]['role'] == 'system' -%}
        {%- if messages[0]['content'] is string -%}
            {{- messages[0]['content'] | trim -}}
        {%- else -%}
            {%- for item in messages[0]['content'] -%}{{- item['text'] | trim + ' ' -}}{%- endfor -%}
        {%- endif -%}
        {%- set loop_messages = messages[1:] -%}
    {%- endif -%}
    {{- '<turn|>\n' -}}
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


class Kit:
    """One tokenizer and the pinned template (or its stand-in): ``encode`` without added special
    tokens, ``decode`` skipping them (the server's reply text), ``keep`` keeping them (the text
    vLLM's gemma4 parser reads), ``render_template`` the server's callback."""

    def __init__(self, name, encode, decode, keep, render_template, template):
        self.name, self._encode, self._decode, self._keep = name, encode, decode, keep
        self.render_template, self.template = render_template, template

    def encode(self, text):
        return list(self._encode(text))

    def decode(self, ids):
        return self._decode(list(ids))

    def keep(self, ids):
        return self._keep(list(ids))

    def render(self, messages, generation=False, **kwargs):
        """The template's text of plain messages as vLLM renders them (OpenAI text parts)."""
        return self.render_template(self.template, openai_parts(messages),
                                    dict(dict(TH.GEMMA4_TEMPLATE_KWARGS, **kwargs), add_generation_prompt=generation))

    def stock(self, messages, **kwargs):
        """Stock vLLM's thinking-on prompt ids of a request (the template with enable_thinking=True)."""
        return self.encode(self.render(messages, True, **kwargs))


def char_render(template, messages, kwargs):
    kwargs = dict(kwargs)
    generation = kwargs.pop('add_generation_prompt')
    kwargs.pop('preserve_thinking', None)
    return KP.jinja_render(template, messages, None, generation, bos_token='<bos>', **kwargs)


def char_kit():
    return Kit('gtok', TOK.encode, lambda ids: TOK.decode(ids, True), lambda ids: TOK.decode(ids, False),
               char_render, STAND_IN_ON)


HF = GT.HF
_REAL = []


def real_kit():
    """The pinned Gemma tokenizer and template (Transformers' apply_chat_template), or None."""
    if HF is None:
        return None
    if not _REAL:
        import test_gemma_renderer as TR
        kit = Kit('gemma4', lambda text: HF.encode(text, add_special_tokens=False),
                  lambda ids: HF.decode(ids, skip_special_tokens=True),
                  lambda ids: HF.decode(ids, skip_special_tokens=False), TR.hf_render(HF), HF.chat_template)
        _REAL.append(kit)
    return _REAL[0]


def kits():
    real = real_kit()
    return [char_kit()] + ([real] if real is not None else [])


def renderer_of(kit):
    """The Gemma engine source renderer over a kit: the real Gemma4SourceRenderer with the pinned
    files for the pinned kit, else the class over the stand-ins (its startup checks included)."""
    from gemma4_renderer import Gemma4SourceRenderer
    if kit.name == 'gemma4':
        return Gemma4SourceRenderer(model_dir=G.metadata_dir(), tokenizer=HF, render_template=kit.render_template,
                                    verify_identity=lambda: True, page_tokens=16)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return kit.encode(text)

        def decode(self, ids, skip_special_tokens=False):
            return kit.decode(ids) if skip_special_tokens else kit.keep(ids)

        def convert_tokens_to_ids(self, token):
            ids = kit.encode(token)
            return ids[0] if len(ids) == 1 else None

        def __len__(self):
            return 262144
    renderer = Gemma4SourceRenderer.__new__(Gemma4SourceRenderer)
    renderer.tokenizer, renderer.template, renderer._check_identity = Tokenizer(), kit.template, lambda: None
    renderer._render_template, renderer._stop_ids = kit.render_template, STOPS
    return renderer


def splitter(kit):
    return TH.Gemma4Split(kit.encode, kit.decode, STOPS)


def raw(reasoning, content):
    """A thinking reply as the model writes it: the thought channel, then the content."""
    return THOUGHT + reasoning + '<channel|>' + content


# ---- vLLM 0.24.0's gemma4 reasoning parser, on the text (the model of decision 4) -----------------

def vllm_gemma4(text):
    """(reasoning, content, closed) as vLLM 0.24.0's Gemma4Parser.extract_reasoning gives them for a
    model output ``text`` decoded with special tokens kept (stop token excluded): its first-feed
    injection, the CONTENT/REASONING transitions of its engine config on <|channel> and <channel|>
    (a <channel|> in content swallowed, a <|channel> in reasoning kept as text), the trailing
    whitespace of the reasoning stripped, then its leading 'thought\\n' (None for 'thought' alone);
    empty values None. ``closed``: the engine ended in content."""
    start, end = TH.CHANNEL_OPEN, TH.CHANNEL_CLOSE
    if text and start not in text and (end in text or text.startswith('thought\n') or text == 'thought'):
        text = start + text
    state, parts, i = 'content', dict(content=[], reasoning=[]), 0
    while i < len(text):
        if text.startswith(start, i):
            if state == 'content':
                state = 'reasoning'
            else:
                parts['reasoning'].append(start)
            i += len(start)
        elif text.startswith(end, i):
            state = 'content'
            i += len(end)
        else:
            parts[state].append(text[i])
            i += 1
    reasoning = ''.join(parts['reasoning']).rstrip()
    if reasoning.startswith('thought\n'):
        reasoning = reasoning[len('thought\n'):]
    elif reasoning == 'thought':
        reasoning = None
    return reasoning or None, ''.join(parts['content']) or None, state == 'content'


PIECES = ('<|channel>', '<channel|>', 'thought', '\n', 'thought\n', ' ', 'Let me look.', '{"a": 1}', '\n\n', 'x',
          '中文 😀', ' THOUGHT: y\n', '<mswea_bash_command>ls</mswea_bash_command>', '  ', 'though', 't\n')


def random_reply(kit, rng, pieces=PIECES, longest=12):
    """Generated ids: random pieces (each encoded alone) and, mostly, a stop id."""
    ids = []
    for _ in range(rng.randrange(0, longest)):
        ids += kit.encode(rng.choice(pieces))
    return ids + rng.choice(([TURN_CLOSE], [TURN_CLOSE], [MP.EOS], [MP.TOOL_RESPONSE], []))


# ---- 1. the pins -----------------------------------------------------------------------------------

class RecordTests(unittest.TestCase):
    maxDiff = None

    def test_gemma_record(self):
        fields = dict(response_fields=['reasoning_content', 'reasoning'], client_fields=['reasoning_content', 'reasoning'],
                      append='previous_messages_exact+reply_content_and_reasoning_stripped')
        gemma = dict(enable_thinking=True, header_text='<|turn>model\n', template_kwargs={'enable_thinking': True},
                     channel=['<|channel>', '<channel|>'], thought_prefix='thought\n',
                     split='vllm-0.24.0-gemma4-parser-on-ids', **fields)
        self.assertEqual(ON_XML, gemma)
        self.assertEqual(ON_CHAT, gemma)
        for bad in (TH.XML, 'kv_session_v1'):
            with self.subTest(protocol=bad), self.assertRaisesRegex(LiveV2Error, 'mswea-xml and chat-text'):
                TH.record(bad, profile=GEMMA)
        with self.assertRaisesRegex(LiveV2Error, 'Qwen and Gemma 4 profiles only'):
            TH.record(TH.CHAT, profile='another')

    def test_the_qwen_records_are_unchanged(self):
        """Identity: the Qwen thinking, chat_text and kv_session records, literal (as at ba4d1ba)."""
        fields = dict(response_fields=['reasoning_content', 'reasoning'], client_fields=['reasoning_content', 'reasoning'])
        header = '<|im_start|>assistant\n<think>\n'
        self.assertEqual(TH.record(TH.XML), dict(enable_thinking=True, header_text=header,
                         template_kwargs={'enable_thinking': True}, think_end='</think>',
                         append='previous_messages_exact+reply_content_and_reasoning_stripped', **fields))
        self.assertEqual(TH.record(TH.CHAT), dict(TH.record(TH.XML), template_kwargs={'enable_thinking': True,
                                                                                     'preserve_thinking': True}))
        self.assertEqual(TH.record(TH.CHAT, 'strip'), dict(
            TH.record(TH.XML), history='strip',
            append='previous_messages_exact+reply_content_stripped;cut_at_header_start+reprefill_reply_sent_back'))
        base = dict(slots=512, idle_timeout_s=600, evict_after_s=30, session_header='X-Session-ID',
                    binding='initial_segment', busy_wait_s=30)
        self.assertEqual(C.record(), dict(base, header_text='<|im_start|>assistant\n<think>\n\n</think>\n\n',
                                          template_kwargs={'enable_thinking': False, 'preserve_thinking': True}))
        self.assertEqual(C.record(thinking=True), dict(base, header_text=header,
                         template_kwargs={'enable_thinking': True, 'preserve_thinking': True}))
        self.assertEqual(C.record(thinking=True, strip=True), dict(base, header_text=header,
                         template_kwargs={'enable_thinking': True}))
        for thinking, text in (('on', header), ('off', '<|im_start|>assistant\n<think>\n\n</think>\n\n')):
            self.assertEqual(K.record(contract=None, thinking=thinking, slots=4),
                             dict(method='dram_backed_kv', budget=None, thinking=thinking, header_text=text, slots=4))

    def test_gemma_thinking_off_records_are_unchanged(self):
        """Identity: the Gemma 4 records with thinking off, literal (as at ba4d1ba)."""
        off = '<|turn>model\n<|channel>thought\n<channel|>'
        base = dict(slots=512, idle_timeout_s=600, evict_after_s=30, session_header='X-Session-ID',
                    binding='initial_segment', busy_wait_s=30)
        self.assertEqual(C.record(profile=GEMMA), dict(base, header_text=off, template_kwargs={'enable_thinking': False}))
        self.assertEqual(K.record(contract=None, thinking='off', slots=4, grammar=K.GEMMA4),
                         dict(method='dram_backed_kv', budget=None, thinking='off', header_text=off, slots=4,
                              grammar='gemma4'))
        self.assertEqual((XML.GEMMA4_XML.header, XML.GEMMA4_XML.message_text('assistant', ' a<|channel>x<channel|>b ')),
                         (off, off + 'ab<turn|>\n'))

    def test_chat_and_kv_records_with_thinking_on(self):
        off = C.record(profile=GEMMA)
        on = C.record(profile=GEMMA, thinking=True)
        self.assertEqual(on, dict(off, header_text='<|turn>model\n', template_kwargs={'enable_thinking': True}))
        base = dict(protocol=C.PROTOCOL, tasks=C.slot_tasks(512), task=C.slot_tasks(1)[0], max_running_sessions=20,
                    policy='observation_query', model_profile=GEMMA)
        self.assertEqual(C.check_config(dict(base, chat_text=on, thinking=ON_CHAT)), on)
        for config in (dict(base, chat_text=on), dict(base, chat_text=off, thinking=ON_CHAT),
                       dict(base, chat_text=C.record(thinking=True), thinking=ON_CHAT)):
            with self.subTest(config=sorted(config)), self.assertRaises(LiveV2Error):
                C.check_config(config)
        self.assertEqual(TH.pinned(dict(base, chat_text=on, thinking=ON_CHAT)), ON_CHAT)
        for value in (TH.record(TH.CHAT), dict(ON_CHAT, header_text=HEADER + THOUGHT), dict(ON_CHAT, split='text')):
            with self.subTest(value=value), self.assertRaisesRegex(LiveV2Error, 'differs from the protocol pins'):
                TH.pinned(dict(base, thinking=value))
        kv = K.record(contract=None, thinking='on', slots=4, grammar=K.GEMMA4)
        self.assertEqual(kv, dict(method='dram_backed_kv', budget=None, thinking='on', header_text='<|turn>model\n',
                                  slots=4, grammar='gemma4', tool_header_text='<|channel>thought\n'))
        config = dict(protocol=K.PROTOCOL, kv_session=kv, tasks=K.slot_tasks(4), task=K.slot_tasks(1)[0],
                      max_running_sessions=2, selection_contract=None, model_profile=GEMMA, policy='full',
                      page_tokens=16, effective_server_config=dict(max_model_len=131072))
        self.assertEqual(K.check_config(config), kv)
        for bad in ({k: v for k, v in kv.items() if k != 'tool_header_text'}, dict(kv, tool_header_text=HEADER),
                    dict(K.record(contract=None, thinking='off', slots=4, grammar=K.GEMMA4), tool_header_text=THOUGHT)):
            with self.subTest(bad=sorted(bad)), self.assertRaises(LiveV2Error):
                K.check_config(dict(config, kv_session=bad))
        self.assertEqual(K.options(config)['thinking'], 'on')
        # The route's header-less continuation is thinking off's only.
        frontend = lambda record: KR.frontend(dict(config, kv_session=record))
        self.assertIs(frontend(kv).empty_header, False)
        self.assertIs(frontend(K.record(contract=None, thinking='off', slots=4, grammar=K.GEMMA4)).empty_header, True)


# ---- 2. the reply split -------------------------------------------------------------------------------

class SplitTests(unittest.TestCase):
    maxDiff = None

    def case(self, kit, text, stop=(TURN_CLOSE,)):
        ids = kit.encode(text) + list(stop)
        return ids, splitter(kit)(ids)

    def test_the_cases_of_decision_four_and_of_vllm(self):
        """(text, reasoning, content, closed, think_start_tokens, think_end_tokens)."""
        cases = (
            ('{"a": 1}', '', '{"a": 1}', True, 0, 0),                              # no channel: content only
            (raw('Let me look.\n', ' {"a": 1}'), 'Let me look.', ' {"a": 1}', True, 1, 1),
            (raw('R', ''), 'R', '', True, 1, 1),
            (THOUGHT + 'still thinking <mswea_bash_command>x', 'still thinking <mswea_bash_command>x', '', False, 1, 0),
            ('<|channel>thought\n<channel|>C', '', 'C', True, 1, 1),               # the empty channel
            ('<|channel>thought<channel|>C', '', 'C', True, 1, 1),                 # 'thought' alone: no reasoning
            ('<|channel>thought\n\nR<channel|>C', '\nR', 'C', True, 1, 1),         # 'thought\n' removed once
            ('<|channel>plain<channel|>C', 'plain', 'C', True, 1, 1),              # no thought prefix: kept
            ('Before <|channel>thought\nmid<channel|> after', 'mid', 'Before  after', True, 1, 1),
            (raw('a', 'b') + raw('c', 'd'), 'athought\nc', 'bd', True, 2, 2),       # every channel's reasoning
            ('<|channel>thought\nx<|channel>y<channel|>z', 'x<|channel>y', 'z', True, 1, 1),   # open inside: text
            ('C<channel|>D', 'C', 'D', True, 0, 1),                                # a close without an open
            (raw('r', 'c<channel|>d'), 'r', 'cd', True, 1, 2),                     # a stray close in content
            ('thought\nno channel, all reasoning', 'no channel, all reasoning', '', False, 0, 0),
            ('thought', '', '', False, 0, 0),
            ('thoughts and a plain reply', '', 'thoughts and a plain reply', True, 0, 0),
            (raw('R  \n\t', 'C '), 'R', 'C ', True, 1, 1),                         # reasoning rstripped, content not
            ('', '', '', True, 0, 0))
        for kit in kits():
            for text, reasoning, content, closed, opens, closes in cases:
                with self.subTest(kit=kit.name, text=text):
                    ids, value = self.case(kit, text)
                    self.assertEqual((value['reasoning'], value['content'], value['closed']), (reasoning, content, closed))
                    tokens = value['tokens']
                    self.assertEqual((tokens['think_start_tokens'], tokens['think_end_tokens']), (opens, closes))
                    self.assertEqual(sum(tokens.values()), len(ids))
                    r, c, done = vllm_gemma4(kit.keep(ids[:-1]))
                    self.assertEqual((r or '', c or '', done), (reasoning, content, closed))
        kit = char_kit()
        # The stop id is not part of the text; a reply cut at its bound has none; a stop id only.
        for stop in ([TURN_CLOSE], [MP.EOS], [MP.PAD], [MP.TOOL_RESPONSE], []):
            ids, value = self.case(kit, raw('R', 'C'), stop)
            self.assertEqual((value['reasoning'], value['content'], value['tokens']['content_tokens']),
                             ('R', 'C', len(kit.encode('C')) + len(stop)))
        self.assertEqual(splitter(kit)([TURN_CLOSE])['tokens'], dict(reasoning_tokens=0, think_start_tokens=0,
                                                                     think_end_tokens=0, content_tokens=1))
        # The token split of the evidence: thought\n and R in reasoning, the channel ids apart.
        ids, value = self.case(kit, raw('R1', 'C2'))
        self.assertEqual(value['tokens'], dict(reasoning_tokens=len(kit.encode('thought\nR1')), think_start_tokens=1,
                                               think_end_tokens=1, content_tokens=len(kit.encode('C2')) + 1))

    def test_random_replies_equal_the_vllm_parser(self):
        rng = random.Random(4)
        for kit, count in [(char_kit(), 3000)] + ([(real_kit(), 600)] if real_kit() else []):
            split = splitter(kit)
            closed = opened = 0
            for _ in range(count):
                ids = random_reply(kit, rng)
                value = split(ids)
                body = ids[:-1] if ids and ids[-1] in STOPS else ids
                r, c, done = vllm_gemma4(kit.keep(body))
                with self.subTest(kit=kit.name, ids=ids):
                    self.assertEqual((value['reasoning'], value['content'], value['closed']), (r or '', c or '', done))
                    self.assertEqual(sum(value['tokens'].values()), len(ids))
                    self.assertEqual(split.content(ids), value['content'])
                closed += done
                opened += value['tokens']['think_start_tokens'] > 0
            self.assertGreater(opened, count // 4)
            self.assertLess(closed, count)

    def test_markers_must_be_single_ids(self):
        with self.assertRaisesRegex(LiveV2Error, 'one token each'):
            TH.Gemma4Split(lambda text: [ord(c) for c in text], lambda ids: '', STOPS)
        with self.assertRaisesRegex(LiveV2Error, 'callbacks'):
            TH.Gemma4Split(TOK.encode, None, STOPS)
        self.assertIsNone(TH.splitter(None, TOK.encode, TOK.decode, STOPS))
        self.assertIsInstance(TH.splitter(GEMMA, TOK.encode, TOK.decode, STOPS), TH.Gemma4Split)


# ---- 3. MarkerStop -----------------------------------------------------------------------------------

class MarkerStopTests(unittest.TestCase):
    def setUp(self):
        status = ModuleType('vllm.v1.request')
        status.RequestStatus = NS(FINISHED_STOPPED='FINISHED_STOPPED')
        modules = {'vllm': ModuleType('vllm'), 'vllm.v1': ModuleType('vllm.v1'), 'vllm.v1.request': status}
        self.enterContext(patch.dict(sys.modules, modules))

    def stop(self, kit, **options):
        split = splitter(kit)
        return XML.MarkerStop(kit.decode, thinking=True, content=split.content, marker_ids=split.markers, **options)

    def run_stop(self, stop, tokens):
        request = NS(request_id='r', output_token_ids=[], status=None, stop_reason=None)
        for k, token in enumerate(tokens):
            request.output_token_ids.append(token)
            if stop(request):
                return k + 1
        return None

    def test_the_marker_counts_in_the_content_only(self):
        for kit in kits():
            draft = 'Draft: <mswea_bash_command>echo DRAFT</mswea_bash_command> then the real one.'
            action = 'THOUGHT: look.\n\n<mswea_bash_command>ls -la</mswea_bash_command>'
            tokens = kit.encode(raw(draft, action)) + kit.encode(' trailing')
            split = splitter(kit)
            expected = next(k for k in range(1, len(tokens) + 1) if MARKER in split(tokens[:k])['content'])
            self.assertTrue(kit.decode(tokens[:expected]).endswith(action))
            off = next(k for k in range(1, len(tokens) + 1) if MARKER in kit.decode(tokens[:k]))
            self.assertLess(off, expected)                    # thinking off would stop inside R
            for options in ({}, dict(incremental=True), dict(incremental=True, token_cache=True)):
                with self.subTest(kit=kit.name, options=options):
                    self.assertEqual(self.run_stop(self.stop(kit, **options), tokens), expected)
            # A reply cut inside the channel, and a reply opening with plain 'thought\n' (vLLM: reasoning).
            for text in (THOUGHT + draft, 'thought\n' + draft):
                self.assertIsNone(self.run_stop(self.stop(kit, incremental=True), kit.encode(text)))
            # Without a channel the marker counts everywhere (as with thinking off).
            plain = kit.encode(action)
            self.assertEqual(self.run_stop(self.stop(kit), plain), len(plain))
        with self.assertRaisesRegex(LiveV2Error, 'content split needs thinking on'):
            XML.MarkerStop(TOK.decode, content=lambda ids: '', marker_ids=frozenset({OPEN}))
        with self.assertRaisesRegex(LiveV2Error, 'content split needs thinking on'):
            XML.MarkerStop(TOK.decode, thinking=True, content=lambda ids: '')

    def test_incremental_answers_equal_the_full_ones(self):
        """Random outputs (channel ids, the marker in pieces, the parser's injection cases and a close
        before a late open, which moves earlier text into the content): the incremental modes and
        async C's prediction stop exactly where a full check of every prefix does."""
        kit = char_kit()
        pieces = PIECES + ('<', '/', 'mswea_bash_command', '>', '</mswea_bash_command>', 'x>')
        rng = random.Random(11)
        split = splitter(kit)
        stops = 0
        for k in range(1500):
            tokens = random_reply(kit, rng, pieces, 18)
            if rng.random() < .3:
                # A plain 'thought\n' start (reasoning by the parser's injection) holding the marker,
                # until a late <|channel> moves it into the content: that id must decode the output.
                tokens = kit.encode('thought\nx</mswea_bash_command>y') + tokens + [OPEN] + kit.encode('d')
            expected = next((j for j in range(1, len(tokens) + 1) if MARKER in split(tokens[:j])['content']), None)
            for options in ({}, dict(incremental=True), dict(incremental=True, token_cache=True)):
                with self.subTest(k=k, options=options):
                    self.assertEqual(self.run_stop(self.stop(kit, **options), tokens), expected)
            policy = self.stop(kit, incremental=True, token_cache=True)
            request = NS(request_id='a%d' % k, output_token_ids=[], status=None, stop_reason=None)
            for token in tokens:
                table = dict(policy._checked)
                predicted = A.marker_stops(policy, request, token)
                self.assertEqual(policy._checked, table)
                request.output_token_ids.append(token)
                self.assertEqual(predicted, policy(request))
                if predicted:
                    break
            stops += expected is not None
        self.assertGreater(stops, 200)

    def test_compose_engine_arms_the_stop_with_the_split(self):
        source = (ROOT / 'source-overlay' / 'remote_ra_live_v2' / 'composition.py').read_text(encoding='utf-8')
        for text in ('armed = dict(thinking=True) if thinking else {}', 'splitter = getattr(transcript, "splitter", None)',
                     'armed.update(content=splitter.content, marker_ids=splitter.markers)'):
            self.assertIn(text, source)


# ---- 4. the XML transcript ----------------------------------------------------------------------------

def xml_messages(turns, trim=True):
    """System, task, then (reasoning or None, content, observation) per turn."""
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=TASK)]
    for reasoning, content, observation in turns:
        reply = dict(role='assistant', content=content)
        if reasoning is not None:
            reply['reasoning'] = reasoning
        messages += [reply, dict(role='user', content=observation.strip() if trim else observation)]
    return messages


def plain(messages):
    return [dict(role=m['role'], content=m['content']) for m in messages]


R1 = 'The PR wants the index fixed. Draft: <mswea_bash_command>echo DRAFT</mswea_bash_command>.'
C1 = 'THOUGHT: look around.\n\n<mswea_bash_command>ls -la</mswea_bash_command>'
R2 = 'Second step; the file is there.'
C2 = 'THOUGHT: edit it.\n\n<mswea_bash_command>sed -i s/a/b/ target.py</mswea_bash_command>'


class XmlTranscriptTests(unittest.TestCase):
    maxDiff = None

    def play(self, kit, sent_back):
        ours = renderer_of(kit).transcript(thinking=True)
        self.assertTrue(ours.thinking and ours.grammar is XML.GEMMA4_XML)
        self.assertEqual(ours.header, tuple(kit.encode(HEADER)))
        turns, served = [], []
        script = [(R1, C1, '<returncode>0</returncode>\n<output>\na.py\n</output>'),
                  (R2, C2, '<returncode>0</returncode>\n<output>\n</output>\n'), ('Done.', 'THOUGHT: x', 'ok')]
        for k, (reasoning, content, observation) in enumerate(script):
            messages = xml_messages(turns)
            tokens, stable, span = ours.render(messages)
            served.append((messages, tokens, ours.pending['history'], ours.pending['rebuild']))
            committed = ours.commit(tokens, kit.encode(raw(reasoning, content)) + [TURN_CLOSE])
            self.assertEqual((committed['reasoning'], committed['content'], committed['closed']), (reasoning, content, True))
            self.assertEqual(committed['split']['think_start_tokens'], 1)
            self.assertEqual(ours.previous_action(), content)            # the Goal's previous reply: C
            turns.append((sent_back(k, committed['reasoning']), ' ' + committed['content'] + '\n', observation))
        return ours, served

    def test_append_with_the_reasoning_sent_back(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                ours, served = self.play(kit, lambda k, reasoning: reasoning)
                self.assertEqual([h for _, _, h, _ in served], ['fresh', 'append', 'append'])
                self.assertEqual(ours.sessions, 1)
                # The initial segment: system and task with the <|think|> system turn (2 more tokens).
                first = served[0][1]
                initial = kit.encode(kit.render(xml_messages([])[:2]))
                self.assertEqual(list(first), initial + kit.encode(HEADER))
                self.assertTrue(kit.render(xml_messages([])[:2]).startswith('<bos><|turn>system\n<|think|>\n'))
                off = kit.encode(kit.render(xml_messages([])[:2], enable_thinking=False))
                if kit.name == 'gemma4':
                    self.assertEqual(len(initial) - len(off), 2)
                # Every append is the stream, the observation as the template renders it, the header.
                for (_, before, _, _), (messages, after, _, _) in zip(served, served[1:]):
                    reply = messages[-2]
                    stream = list(before) + kit.encode(raw(reply['reasoning'], reply['content'].strip())) + [TURN_CLOSE] \
                        + kit.encode('\n')
                    self.assertEqual(list(after), stream + kit.encode('<|turn>user\n' + messages[-1]['content'].strip()
                                                                      + '<turn|>\n' + HEADER))

    def test_a_rebuild_is_the_stock_render(self):
        """A client that drops R (or changes it) rebuilds: every later message from its content after
        <|turn>model\\n, which is the stock thinking-on render of the conversation."""
        for kit in kits():
            for sent, reason in ((lambda k, r: None, TH.DROPPED), (lambda k, r: 'another reasoning', TH.DIFFERS)):
                with self.subTest(kit=kit.name, reason=reason):
                    ours, served = self.play(kit, sent)
                    self.assertEqual([h for _, _, h, _ in served], ['fresh', reason, reason])
                    self.assertEqual(ours.sessions, 3)
                    for messages, tokens, _, _ in served:
                        self.assertEqual(list(tokens), kit.stock(plain(messages)))

    def test_replies_without_a_channel_or_cut_inside_it(self):
        kit = char_kit()
        ours = renderer_of(kit).transcript(thinking=True)
        tokens, _, _ = ours.render(xml_messages([]))
        committed = ours.commit(tokens, kit.encode(C1) + [TURN_CLOSE])
        self.assertEqual((committed['reasoning'], committed['content'], committed['closed']), ('', C1, True))
        tokens, _, _ = ours.render(xml_messages([(None, C1, 'obs')]))
        self.assertEqual(ours.pending['history'], 'append')              # a missing reasoning counts as ""
        cut = kit.encode(THOUGHT + 'I am still thinking <mswea_bash_command>ls</mswea_bash_command> and')
        committed = ours.commit(tokens, cut)
        self.assertEqual((committed['reasoning'], committed['content'], committed['closed']),
                         ('I am still thinking <mswea_bash_command>ls</mswea_bash_command> and', '', False))
        self.assertEqual(committed['boundary'], tuple(kit.encode('<turn|>\n')))
        ours.render(xml_messages([(None, C1, 'obs'), (committed['reasoning'], '', 'Format error: no action')]))
        self.assertEqual((ours.pending['history'], ours.pending['rebuild']), ('append', False))
        self.assertEqual(ours.previous_action(), '')

    def test_transcripts_and_header_check(self):
        kit = char_kit()
        renderer = renderer_of(kit)
        made = XML.xml_transcripts(renderer, XML.GEMMA4_PROTOCOL, 2, thinking=True)
        self.assertTrue(len(made) == 2 and all(t.thinking and t.splitter is not None for t in made))
        self.assertFalse(any(t.thinking for t in XML.xml_transcripts(renderer, XML.GEMMA4_PROTOCOL, 2)))
        self.assertIsNone(XML.xml_transcripts(renderer, XML.GEMMA4_PROTOCOL, 1)[0].splitter)
        render = lambda messages, generation: kit.render(messages, generation)
        self.assertEqual(TH.check_header_gemma4(render), HEADER)
        for bad in (lambda m, g: kit.render(m, g, enable_thinking=False),
                    lambda m, g: kit.render(m, g).replace('<|think|>\n', '')):
            with self.assertRaisesRegex(LiveV2Error, 'thinking-on system turn or generation header'):
                TH.check_header_gemma4(bad)


# ---- 5. the chat-text transcript ----------------------------------------------------------------------

def chat_transcript(kit):
    return renderer_of(kit).chat_transcripts(C.record(profile=GEMMA, thinking=True), thinking=True)()


class ChatTranscriptTests(unittest.TestCase):
    maxDiff = None

    def play(self, kit, sent_back, turns=3, system=False):
        ours = chat_transcript(kit)
        messages = ([dict(role='system', content=' s ')] if system else []) + [
            dict(role='user', content=T.first_prompt('Thinking task.'))]
        served = []
        for k in range(turns):
            tokens, _, _ = ours.render([dict(m) for m in messages])
            served.append(([dict(m) for m in messages], tokens, ours.pending['history']))
            reasoning = 'Turn %d: the JSON {"x": %d} drafts it.' % (k, k)
            committed = ours.commit(tokens, kit.encode(raw(reasoning, T.action(k))) + [TURN_CLOSE])
            self.assertEqual((committed['content'], committed['reasoning']), (T.action(k), reasoning))
            self.assertEqual(ours.previous_action(), T.action(k))
            reply = dict(role='assistant', content=committed['content'])
            value = sent_back(k, committed['reasoning'])
            if value is not None:
                reply['reasoning'] = value
            messages += [reply, dict(role='user', content=T.observation(k))]
        return ours, served

    def test_a_client_that_drops_the_reasoning_gets_the_stock_prompt(self):
        """Terminus 2's default (interleaved_thinking off): every turn rebuilds, and every prompt is
        the stock thinking-on render, with no 4-token channel difference."""
        for kit in kits():
            for system in (False, True):
                with self.subTest(kit=kit.name, system=system):
                    ours, served = self.play(kit, lambda k, r: None, system=system)
                    self.assertEqual([h for _, _, h in served], ['fresh', TH.DROPPED, TH.DROPPED])
                    for messages, tokens, _ in served:
                        self.assertEqual(list(tokens), kit.stock(messages))

    def test_the_reasoning_sent_back_appends_and_a_rebuild_reproduces_the_stream(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                ours, served = self.play(kit, lambda k, r: r)
                self.assertEqual([h for _, _, h in served], ['fresh', 'append', 'append'])
                self.assertEqual(ours.sessions, 1)
                for messages, tokens, _ in served:
                    fresh = chat_transcript(kit)
                    self.assertEqual(fresh.render(messages)[0], tokens)       # the keep pieces with R
                # The initial segment holds the <|think|> system turn: 7 more tokens than thinking off.
                first = served[0][0]
                on, off = (kit.encode(kit.render(first, enable_thinking=e)) for e in (True, False))
                self.assertEqual(on[:len(kit.encode('<bos><|turn>system\n<|think|>\n<turn|>\n'))],
                                 kit.encode('<bos><|turn>system\n<|think|>\n<turn|>\n'))
                if kit.name == 'gemma4':
                    self.assertEqual(len(on) - len(off), 7)
                    self.assertEqual(on[1:8], [105, 9731, 107, 98, 107, 106, 107])

    def test_the_keep_pieces(self):
        g = C.GEMMA4_CHAT
        a = lambda content, reasoning=None: dict(role='assistant', content=content,
                                                 **({} if reasoning is None else dict(reasoning=reasoning)))
        self.assertEqual(g.model_piece_thinking([a(' {"a": 1} ', 'R')]),
                         '<|turn>model\n<|channel>thought\nR<channel|>{"a": 1}<turn|>\n')
        self.assertEqual(g.model_piece_thinking([a('x'), a(' y<|channel>thought\nz<channel|> ', '')]),
                         '<|turn>model\nxy<turn|>\n')
        self.assertEqual(g.history_pieces([a('x', 'r'), dict(role='user', content=' u ')], thinking=True),
                         ['<|turn>model\n<|channel>thought\nr<channel|>x<turn|>\n', '<|turn>user\nu<turn|>\n'])
        self.assertEqual(g.history_pieces([a('x', 'r')]), [g.model_piece(['x'])])       # thinking off: as before
        self.assertEqual((g.thinking_header, g.thinking_kwargs, g.strip_kwargs()),
                         ('<|turn>model\n', {'enable_thinking': True}, {'enable_thinking': True}))

    def test_template_checks(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                render = lambda messages, generation: kit.render(messages, generation)
                self.assertEqual(C.GEMMA4_CHAT.check_template_thinking(render), HEADER)
                with self.assertRaisesRegex(LiveV2Error, 'thinking-on lone user message'):
                    C.GEMMA4_CHAT.check_template_thinking(lambda m, g: kit.render(m, g, enable_thinking=False))
        kit = char_kit()
        renderer = renderer_of(kit)
        with self.assertRaisesRegex(RuntimeError, 'differ from the startup record'):
            renderer.chat_transcripts(C.record(profile=GEMMA), thinking=True)
        with self.assertRaisesRegex(RuntimeError, 'differ from the startup record'):
            renderer.chat_transcripts(C.record(profile=GEMMA, thinking=True))
        self.assertTrue(renderer.chat_transcripts(C.record(profile=GEMMA, thinking=True), thinking=True)().thinking)
        with self.assertRaisesRegex(RuntimeError, 'strip needs thinking'):
            renderer.chat_transcripts(C.record(profile=GEMMA), strip=True)
        # The strip transcript with Gemma's turns: test_gemma_thinking_cut.
        strip = C.ChatTextTranscript(encode=kit.encode, render_chat=lambda m: [1], decode=kit.decode, stop_ids=STOPS,
                                     grammar=C.GEMMA4_CHAT, thinking=True, strip=True)
        self.assertTrue(strip.strip and strip.splitter is not None)

    def test_route_response_and_evidence(self):
        kit = char_kit()
        ours = chat_transcript(kit)
        tokens, _, _ = ours.render([dict(role='user', content='u')])
        generated = kit.encode(raw('Think {"keystrokes": "rm"} first.\n', '{"keystrokes": "ls"}')) + [TURN_CLOSE]
        message = NS(content=None, tool_calls=None, reasoning=None)
        result = NS(choices=[NS(message=message, token_ids=list(generated), finish_reason='stop', stop_reason=None)],
                    usage=NS(prompt_tokens=0, completion_tokens=0, total_tokens=0))
        reply = CR.finish_response(result, NS(capture=NS(tokens=tokens)), ours, 4096)
        self.assertEqual((message.content, message.reasoning_content, message.reasoning),
                         ('{"keystrokes": "ls"}', 'Think {"keystrokes": "rm"} first.', 'Think {"keystrokes": "rm"} first.'))
        self.assertEqual(reply['finish_reason'], 'stop')
        row = mswea_route.evidence_fields(ours, reply)['thinking']
        self.assertEqual(row, dict(reasoning_tokens=len(kit.encode('thought\nThink {"keystrokes": "rm"} first.\n')),
                                   think_start_tokens=1, think_end_tokens=1,
                                   content_tokens=len(kit.encode('{"keystrokes": "ls"}')) + 1, closed=True,
                                   reasoning_chars=len('Think {"keystrokes": "rm"} first.'),
                                   content_chars=len('{"keystrokes": "ls"}'), history='fresh'))
        self.assertEqual(sum(v for k, v in row.items() if k.endswith('_tokens')), len(generated))
        # A reply cut at the bound inside its channel: all reasoning, content "".
        ours = chat_transcript(kit)
        tokens, _, _ = ours.render([dict(role='user', content='u')])
        generated = kit.encode(THOUGHT + 'thinking {"keystrokes": "ls"} and')
        message = NS(content=None, tool_calls=None, reasoning=None)
        result = NS(choices=[NS(message=message, token_ids=list(generated), finish_reason='length', stop_reason=None)],
                    usage=NS(prompt_tokens=0, completion_tokens=0, total_tokens=0))
        reply = CR.finish_response(result, NS(capture=NS(tokens=tokens)), ours, len(generated))
        self.assertEqual((message.content, message.reasoning, reply['finish_reason']),
                         ('', 'thinking {"keystrokes": "ls"} and', 'length'))
        self.assertEqual(mswea_route.evidence_fields(ours, reply)['thinking']['closed'], False)


# ---- 6. kv-session ------------------------------------------------------------------------------------

def kv_transcript(kit=None):
    kit = kit or char_kit()
    enc = kit.encode
    return K.KvTranscript(header=tuple(enc(HEADER)), im_start=enc('<|turn>')[0], vocab=262144, decode=kit.keep,
                          stop_ids=STOPS, tools=(enc('<|tool_response>')[0], enc('<tool_response|>')[0]),
                          tool_header=tuple(enc(THOUGHT)))


def body(ids, header, prefix, boundary, messages=None):
    return dict(method=K.METHOD, session_id='a' * 32, input_ids=list(ids), header_tokens=header,
                prefix_tokens=prefix, boundary_tokens=boundary, budget=None, max_tokens=256, temperature=0, top_p=1,
                top_k=-1, repetition_penalty=1, seed=None, messages=messages or [dict(role='user', content='q')])


def parse(value):
    return K.parse_complete(value, budget=None, bound=4096, max_model_len=131072)


class KvSessionTests(unittest.TestCase):
    maxDiff = None

    def test_the_two_headers(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                self.two_headers(kit)

    def two_headers(self, kit):
        t = kv_transcript(kit)
        H, TOOL = len(t.header), len(t.tool_header)          # 3 and 3 on the pinned tokenizer
        if kit.name == 'gemma4':
            self.assertEqual((t.header, t.tool_header), ((105, 4368, 107), (100, 45518, 107)))
        first = kit.encode('<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nRead, then answer.<turn|>\n') \
            + kit.encode(HEADER)
        prefix = len(first) - H
        t.render(parse(body(first, H, prefix, 0)))
        self.assertEqual(t.phase2_suffix(first), H)
        reply = kit.encode(THOUGHT + 'Read the next chunk.\n<channel|><|tool_call>call:read_next{}<tool_call|>') \
            + [MP.TOOL_RESPONSE]
        t.commit(t.pending['tokens'], reply)
        tail = [MP.TOOL_RESPONSE] + kit.encode('response:read_next{value:<|"|>chunk 1<|"|>}<tool_response|>') \
            + kit.encode(THOUGHT)
        tokens, stable, span = t.render(parse(body(tail, TOOL, prefix, 1)))
        self.assertEqual(t.phase2_suffix(tokens), TOOL)
        self.assertEqual(span[1], len(tokens) - TOOL)
        t.commit(tokens, kit.encode('Chunk 1 read.\n<channel|><|tool_call>call:read_next{}<tool_call|>')
                 + [MP.TOOL_RESPONSE])
        # The question: a user turn after the tool responses, the user header.
        question = [MP.TOOL_RESPONSE] + kit.encode('response:read_next{value:<|"|>chunk 2<|"|>}<tool_response|>'
                                                   '<turn|>\n<|turn>user\nWhich gate?<turn|>\n') + kit.encode(HEADER)
        boundary = question.index(MP.TURN_OPEN)
        t.check(parse(body(question, H, prefix, boundary)))
        self.assertEqual(t.phase2_suffix(question), H)
        # Refusals: no header-less continuation with thinking on; the tool header after anything but
        # <tool_response|>; a header of another length.
        bare = tail[:-TOOL]
        refused = (('BAD_GEOMETRY', lambda: parse(body(bare, 0, prefix, 1))),
                   ('HEADER_MISMATCH', lambda: t.check(K.parse_complete(body(bare, 0, prefix, 1), budget=None,
                                                                        bound=4096, max_model_len=131072,
                                                                        empty_header=True))),
                   ('HEADER_MISMATCH', lambda: t.check(parse(body(bare[:-1] + kit.encode(THOUGHT), TOOL, prefix, 1)))),
                   ('HEADER_MISMATCH', lambda: t.check(parse(body(tail, TOOL + 1, prefix, 1)))))
        for kind, call in refused:
            with self.subTest(kind=kind), self.assertRaises(K.KvSessionError) as caught:
                call()
            self.assertEqual(caught.exception.kind, kind)
        with self.assertRaisesRegex(LiveV2Error, 'ids of the Gemma grammar'):
            K.KvTranscript(header=(1,), im_start=2, vocab=9, decode=kit.keep, stop_ids=STOPS, tool_header=(3,))

    def test_header_check(self):
        def render(template, messages, kwargs):
            on = kwargs['enable_thinking']
            text = '<bos>' + ('<|turn>system\n<|think|>\n<turn|>\n' if on else '') + '<|turn>user\nprobe<turn|>\n'
            if messages[-1]['role'] == 'tool':
                text = text[:-len('<turn|>\n')] + '<turn|>\n<|turn>model\n<|tool_call>call:probe{}<tool_call|>' \
                    '<|tool_response>response:probe{value:x}<tool_response|>'
                return text + (THOUGHT if on and kwargs['add_generation_prompt'] else '')
            if kwargs['add_generation_prompt']:
                text += HEADER if on else K.GEMMA4_HEADER
            return text
        self.assertEqual(K.check_gemma4_headers_on(render, None), (HEADER, THOUGHT))
        self.assertEqual(K.check_gemma4_header(render, None), K.GEMMA4_HEADER)        # thinking off as before
        broken = lambda template, messages, kwargs: render(template, messages, dict(kwargs, add_generation_prompt=False))
        with self.assertRaisesRegex(LiveV2Error, 'thinking-on generation header differs'):
            K.check_gemma4_headers_on(broken, None)

    @unittest.skipUnless(HF is not None, 'pinned Gemma 4 tokenizer (%s) and transformers required' % G.ENV)
    def test_a_memoryagentbench_session_is_the_preserve_thinking_render(self):
        """The client renders the template with enable_thinking=True and preserve_thinking=True and
        sends the new tail; the model's reply writes its reasoning as the template renders it
        (R + '\\n<channel|>'). The server's stream equals the template's render at every turn, the
        read_next loop and the question; the headers come from the real renderer's startup check."""
        kit = real_kit()
        renderer = renderer_of(kit)
        record = K.record(contract=None, thinking='on', slots=4, grammar=K.GEMMA4)
        t = renderer.kv_transcripts(record)()
        self.assertEqual((t.header, t.tool_header), ((105, 4368, 107), (100, 45518, 107)))
        kwargs = dict(enable_thinking=True, preserve_thinking=True)
        render = lambda messages: kit.encode(kit.render_template(kit.template, messages,
                                                                 dict(kwargs, add_generation_prompt=True)))
        messages = [dict(role='system', content='You are a memory agent.'),
                    dict(role='user', content='Read the chunks with read_next, then answer.')]
        prompt = render(messages)
        prefix = len(prompt) - 3
        request = parse(body(prompt, 3, prefix, 0, messages))
        stream = None
        for k in range(4):
            if stream is not None:
                full = render(messages)
                self.assertEqual(full[:len(stream)], list(stream))            # the client's render extends it
                tail = [stream[-1]] + full[len(stream):]
                header = 3
                boundary = 1
                request = parse(body(tail, header, prefix, boundary, messages))
            tokens, _, _ = t.render(request)
            self.assertEqual(list(tokens), render(messages))
            reasoning = 'Chunk %d: nothing about the gate yet.' % k
            call = dict(id='c%d' % k, type='function', function=dict(name='read_next', arguments={}))
            opener = '' if k else THOUGHT          # the first turn's model opens its channel itself
            reply = kit.encode(opener + reasoning + '\n<channel|><|tool_call>call:read_next{}<tool_call|>') \
                + [MP.TOOL_RESPONSE]
            t.commit(tokens, reply)
            stream = t.stream
            messages = messages + [dict(role='assistant', content='', reasoning_content=reasoning, tool_calls=[call]),
                                   dict(role='tool', tool_call_id=call['id'], content='chunk %d text' % k)]
        # The question after the loop: a user message; preserve_thinking keeps every R of the tool calls.
        messages = messages + [dict(role='user', content='Which gate opens first?')]
        full = render(messages)
        self.assertEqual(full[:len(stream)], list(stream))
        self.assertEqual(full[-3:], [105, 4368, 107])
        tail = [stream[-1]] + full[len(stream):]
        boundary = next(j for j in range(1, len(tail)) if tail[j] == 105)
        tokens, _, _ = t.render(parse(body(tail, 3, prefix, boundary, messages)))
        self.assertEqual(list(tokens), full)
        # Without preserve_thinking the template drops every R at the question (the Vanilla control).
        stock = kit.encode(kit.render_template(kit.template, messages, dict(enable_thinking=True,
                                                                            add_generation_prompt=True)))
        self.assertLess(len(stock), len(full))


# ---- 7. the launcher ----------------------------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        import test_gemma_startup as GS
        self.gs = GS.GemmaStartupTests('test_load_accepts_the_gemma_startup_config')
        self.gs.setUp()
        self.addCleanup(self.gs.doCleanups)

    def prepare(self, **kw):
        return self.gs.prepare(**kw)

    def pair(self, **kw):
        """(off, on) launches of the same options and output directory."""
        off = self.prepare(**kw)
        self.gs.n -= 1
        return off, self.prepare(enable_thinking=True, **kw)

    def test_mswea_xml_and_chat_text(self):
        import test_gemma_startup as GS
        for name, kw, value in (
                ('xml', dict(task=GS.tasks(4), policy='observation_query', selection=dict(active_token_budget=8192)),
                 ON_XML),
                ('chat', dict(task=None, policy='observation_query', selection=dict(active_token_budget=8192),
                              protocol='chat-text', max_running=4), ON_CHAT)):
            with self.subTest(name):
                (m0, c0, a0, e0), (m1, c1, a1, e1) = self.pair(**kw)
                self.assertEqual((m1, a1), (m0, a0))                     # no vLLM argument changes
                expected = dict(c0, thinking=value)
                if name == 'chat':
                    expected['chat_text'] = C.record(slots=512, profile=GEMMA, thinking=True)
                self.assertEqual(c1, expected)
                loaded = self.gs.load(c1)
                self.assertEqual((loaded['thinking'], TH.pinned(loaded)), (value, value))
                self.assertIsNone(TH.pinned(self.gs.load(c0)))
                self.assertEqual(TH.history_mode(loaded), None)
                if name == 'chat':
                    self.assertTrue(CR.frontend(loaded).thinking)
                    self.assertEqual(C.options(loaded)['thinking'], ON_CHAT)

    def test_kv_session_thinking_on(self):
        kw = dict(task=None, policy='full', protocol='kv-session', max_running=4)
        m0, c0, a0, _ = self.prepare(kv_session_thinking='off', **kw)
        self.gs.n -= 1
        m1, c1, a1, _ = self.prepare(kv_session_thinking='on', **kw)
        self.assertEqual((m1, a1), (m0, a0))
        self.assertEqual(c1, dict(c0, kv_session=K.record(contract=None, thinking='on', slots=64, grammar=K.GEMMA4)))
        self.assertEqual(self.gs.load(c1)['kv_session']['tool_header_text'], THOUGHT)
        self.assertNotIn('thinking', c1)

    def test_refusals(self):
        import test_gemma_startup as GS
        cases = ((dict(task=None, policy='full', protocol='kv-session', max_running=4, enable_thinking=True),
                  'kv-session: --kv-session-thinking'),
                 (dict(task=GS.tasks(4), enable_thinking=True, thinking_history='strip'),
                  'strip serves --protocol chat-text only'),
                 # strip on chat-text: test_gemma_thinking_cut (refused there with --async-scheduling).
                 (dict(task=None, protocol='chat-text', max_running=4, thinking_history='strip'),
                  'requires --enable-thinking'),
                 (dict(task=GS.tasks(4), enable_thinking=1), 'enable-thinking is a switch'))
        for kw, pattern in cases:
            with self.subTest(kw=sorted(kw)), self.assertRaisesRegex(ValueError, pattern):
                self.prepare(**kw)
        xml = self.prepare(task=GS.tasks(4), enable_thinking=True)[1]
        for name, value in (('the Qwen record', dict(xml, thinking=TH.record(TH.XML))),
                            ('the chat record on XML', dict(xml, thinking=dict(ON_XML, split='x'))),
                            ('a record of true', dict(xml, thinking=True))):
            with self.subTest(name), self.assertRaises(RuntimeError):
                self.gs.load(value)

    def test_dry_run_main(self):
        task = self.gs.root / 'dry-task.json'
        task.write_text(json.dumps(dict(instance_id='a', problem_statement='fix a')))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.gs.release), '--output', str(self.gs.root / 'dry'),
                    '--port', '8231', '--policy', 'full', '--kv-cache-gib', '28', '--page-tokens', '16',
                    '--max-model-len', '131072', '--language-model-only', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stream), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(stream.getvalue())
        for extra, record in ((('--task', str(task)), ON_XML),
                              (('--protocol', 'chat-text', '--max-running-sessions', '4'), ON_CHAT)):
            with self.subTest(extra=extra):
                plain, flagged = main(*extra), main(*extra, '--enable-thinking')
                self.assertEqual(flagged['thinking'], record)
                self.assertNotIn('thinking', plain)
                # The reply bound's speed default with thinking on: 4096 (thinking off: 1536).
                bound = lambda dry: dry['speed_defaults']['max_tokens']
                self.assertEqual((bound(plain)['value'], bound(flagged)['value']), (1536, 4096))
                self.assertEqual(flagged['command'], plain['command'])        # no vLLM argument changes


# ---- 8. the stand-in template against the pinned one ---------------------------------------------------

@unittest.skipUnless(HF is not None, 'pinned Gemma 4 tokenizer (%s) and transformers required' % G.ENV)
class StandInTests(unittest.TestCase):
    def test_the_stand_in_renders_what_the_pinned_template_renders(self):
        real = real_kit()
        conversations = [[dict(role='user', content=' u ')],
                         [dict(role='system', content=' s '), dict(role='user', content='u')],
                         [dict(role='user', content=T.first_prompt('x')), dict(role='assistant', content=' {"a": 1} '),
                          dict(role='user', content=T.observation(0)), dict(role='assistant', content='x'),
                          dict(role='assistant', content='y<|channel>thought\nz<channel|>'), dict(role='user', content='v')],
                         xml_messages([(R1, C1, 'obs'), (None, C2, ' o \n')])]
        for messages in conversations:
            for generation in (False, True):
                for on in (True, False):
                    with self.subTest(messages=len(messages), generation=generation, on=on):
                        parts = openai_parts(plain(messages))
                        kwargs = dict(enable_thinking=on, add_generation_prompt=generation)
                        self.assertEqual(char_render(STAND_IN_ON, parts, dict(kwargs)),
                                         real.render_template(real.template, parts, kwargs))


# ---- 9. end to end ------------------------------------------------------------------------------------

def e2e_render_chat(messages):
    """Gemma4SourceRenderer.render_chat_thinking over the e2e stand-ins (test_gemma_chat_text's GTOK)."""
    messages = C.validate_messages(messages)
    assert len(messages) == C.initial_count(messages)
    text = char_render(STAND_IN_ON, openai_parts(messages), dict(enable_thinking=True, add_generation_prompt=False))
    return GT.GTOK.encode(text)


def e2e_stock(messages):
    text = char_render(STAND_IN_ON, openai_parts(plain(messages)), dict(enable_thinking=True,
                                                                        add_generation_prompt=True))
    return GT.GTOK.encode(text)


def e2e_transcripts(record, thinking=True, strip=False):
    assert (record['header_text'], record['template_kwargs'], thinking, strip) == (
        HEADER, {'enable_thinking': True}, True, False)
    return lambda: C.ChatTextTranscript(encode=GT.GTOK.encode, render_chat=e2e_render_chat,
                                        decode=lambda ids: GT.GTOK.decode(list(ids)), stop_ids=STOPS,
                                        grammar=C.GEMMA4_CHAT, thinking=True)


def thinking_model(original):
    """The scripted thinking-on model: a thought channel with a draft action and JSON, then the
    original reply text, which is therefore the reply's content as before."""
    def model(messages):
        return (THOUGHT + 'Reasoning %d: draft {"keystrokes": "rm -rf /"} first.' % len(messages) + '<channel|>'
                + original(messages))
    return model


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChatEndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self):
        import test_ms_e2e as E2E
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    def server(self, policy='full', slots=8, limit=4):
        import test_chat_text_e2e as CE
        import test_gemma_chat_text_e2e as GCE
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value, layout = GCE.config(policy, output, slots, limit, True, thinking=True)
        value['thinking'] = ON_CHAT

        class Renderer(GCE.S.Renderer):
            def chat_transcripts(self, record, thinking=False, strip=False):
                return e2e_transcripts(record, thinking, strip)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        self.enterContext(patch.object(CE, 'model', thinking_model(CE.model)))
        with patch.object(GCE, 'transcripts', e2e_transcripts), patch.object(GCE, 'ChatRenderer', Renderer):
            srv = GCE.GemmaChatServer(policy, value, layout, output, self.router, self.Completion,
                                      fa_blocks=GCE.GE.FA_BLOCKS)
        self.addAsyncCleanup(srv.stop)
        dump = srv.Completion.model_dump

        def model_dump(result):
            value = dump(result)
            for choice, row in zip(result.choices, value['choices']):
                row['message'].update(reasoning=getattr(choice.message, 'reasoning', None),
                                      reasoning_content=getattr(choice.message, 'reasoning_content', None))
            return value
        srv.Completion = type('ThinkingCompletion', (srv.Completion,), dict(model_dump=model_dump))
        return srv

    async def episode(self, srv, k, keep, turns=3):
        """A Terminus-shaped session: the client sends the reasoning back (``keep``; Terminus 2 with
        interleaved_thinking) or drops it (its default)."""
        import test_chat_text_e2e as CE
        messages = [dict(role='user', content='Task %d: list the files in /app.\nroot@h%04d:/app# ' % (k, k))]
        for j in range(turns):
            status, response = await srv.chat_request(messages)
            self.assertEqual(status, 200)
            message = response['choices'][0]['message']
            self.assertEqual(message['content'], CE.compact(len(messages)))
            self.assertEqual(message['reasoning'], 'Reasoning %d: draft {"keystrokes": "rm -rf /"} first.' % len(messages))
            self.assertEqual(message['reasoning_content'], message['reasoning'])
            reply = dict(role='assistant', content=message['content'])
            if keep:
                reply['reasoning_content'] = message['reasoning_content']
            messages = messages + [reply, dict(role='user', content=T.observation(j, host='x'))]

    async def test_terminus_sessions_with_thinking_on(self):
        srv = self.server()
        await asyncio.gather(*(self.episode(srv, k, keep=bool(k % 2)) for k in range(4)))
        self.assertFalse(srv.rows('failures.jsonl'))
        rows = srv.rows()
        self.assertEqual(len(rows), 12)
        stock = 0
        for entry in srv.served:
            self.assertEqual(srv.native(entry['rid'])['prompt_token_ids'], entry['tokens'])
            if not any('reasoning' in m or 'reasoning_content' in m for m in entry['messages']):
                # The reasoning dropped (Terminus 2's default): the stock thinking-on prompt.
                self.assertEqual(entry['tokens'], e2e_stock(entry['messages']))
                stock += 1
        self.assertEqual(stock, 8)
        histories = sorted(r['thinking']['history'] for r in rows)
        self.assertEqual(histories, ['append'] * 4 + ['fresh'] * 4 + [TH.DROPPED] * 4)
        for row in rows:
            t = row['thinking']
            self.assertEqual(t['reasoning_tokens'] + t['think_start_tokens'] + t['think_end_tokens'] + t['content_tokens'],
                             row['chat_text']['completion_tokens'])
            self.assertEqual((t['closed'], t['think_start_tokens'], t['think_end_tokens']), (True, 1, 1))
        for transcript in srv.engine.endpoint.transcripts.values():
            if transcript.turns:
                self.assertTrue(transcript.thinking and transcript.splitter is not None)
                self.assertNotIn('Reasoning', transcript.previous_action())


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class KvRouteTests(unittest.IsolatedAsyncioTestCase):
    """--kv-session-thinking on through test_gemma_kv_session_e2e's route server (the packaged router, the
    live controller, the frontend capture, the batched engine and worker fakes): a MemoryAgentBench-shaped
    session whose read turns end with the tool header and whose question ends with the user header; the
    served prompts are the client's stream, the replies' text keeps the channel markers (the client parses
    them), and a header-less continuation is refused."""
    maxDiff = None

    async def asyncSetUp(self):
        import test_kv_session_e2e as KE
        import test_ms_e2e as E2E
        import test_gemma_kv_session_e2e as GKE
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)
        self.enterContext(patch.object(self.router, 'ChatCompletionRequest', KE.chat_request))
        self.tokenizer = GKE.GemmaTokenizer()

    def server(self):
        import test_gemma_kv_session as GK
        import test_gemma_kv_session_e2e as GKE
        import test_kv_session_e2e as KE
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = KE.config('observation_query', output, 4, 3, thinking='on')
        value.update(model_profile=GEMMA, kv_session=K.record(contract=value['selection_contract'], thinking='on',
                                                              slots=4, grammar=K.GEMMA4))
        tok = GK.CharTokenizer()
        make = lambda: K.KvTranscript(header=tuple(tok.encode(HEADER)), im_start=GK.TURN, vocab=tok.vocab,
                                      decode=tok.decode, stop_ids=GK.STOP_IDS, tools=(GK.TOOL_OPEN, GK.TOOL_CLOSE),
                                      tool_header=tuple(tok.encode(THOUGHT)))
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        self.enterContext(patch.object(GKE, '_gemma_transcripts', lambda record: make))
        srv = GKE.GemmaKvServer('observation_query', value, output, self.router, self.Completion, self.tokenizer)
        self.addAsyncCleanup(srv.stop)
        self.assertIs(srv.kv.empty_header, False)
        return srv, tok

    async def test_a_read_loop_and_the_question(self):
        import test_gemma_kv_session as GK
        srv, tok = self.server()
        status, info = await srv.send('info', None)
        self.assertEqual((status, info['options']['grammar'], info['options']['thinking']), (200, 'gemma4', 'on'))
        sid = 'b' * 32
        call = lambda k: dict(id='c%d' % k, type='function', function=dict(name='read_next', arguments={}))
        messages = [dict(role='system', content='You are a memory agent.'),
                    dict(role='user', content='Read the chunks, then answer.')]
        first = tok.encode('<bos><|turn>system\n<|think|>\nYou are a memory agent.<turn|>\n<|turn>user\n'
                           'Read the chunks, then answer.<turn|>\n' + HEADER)
        H, TOOL = len(tok.encode(HEADER)), len(tok.encode(THOUGHT))
        prefix = len(first) - H
        replies = [THOUGHT + 'R0: read on.\n<channel|><|tool_call>call:read_next{}<tool_call|>',
                   'R1: chunk 0 is about ships.\n<channel|><|tool_call>call:read_next{}<tool_call|>',
                   THOUGHT + 'The gate is in chunk 1.\n<channel|>Gate 3.']
        srv.scripts[sid] = [tok.encode(replies[0]) + [GK.TOOL_OPEN], tok.encode(replies[1]) + [GK.TOOL_OPEN],
                            tok.encode(replies[2]) + [GK.TURN_END]]

        def body(ids, header, boundary, history):
            return dict(method=K.METHOD, session_id=sid, input_ids=list(ids), header_tokens=header,
                        prefix_tokens=prefix, boundary_tokens=boundary, budget=info['budget'], max_tokens=256,
                        temperature=0, top_p=1, top_k=-1, repetition_penalty=1, seed=None, messages=history)
        stream = list(first)
        status, out = await srv.send('complete', body(first, H, 0, messages))
        self.assertEqual((status, out['text'], out['finish_reason']), (200, replies[0] + '<|tool_response>', 'stop'))
        stream += out['output_ids']
        for k, (tail_text, header_text) in enumerate(((
                'response:read_next{value:<|"|>chunk 0<|"|>}<tool_response|>', THOUGHT), (
                'response:read_next{value:<|"|>chunk 1<|"|>}<tool_response|><turn|>\n<|turn>user\nWhich gate?'
                '<turn|>\n', HEADER))):
            messages = messages + [dict(role='assistant', content='', tool_calls=[call(k)]),
                                   dict(role='tool', tool_call_id='c%d' % k, content='chunk %d' % k)]
            if k == 1:
                messages = messages + [dict(role='user', content='Which gate?')]
            tail = [stream[-1]] + tok.encode(tail_text) + tok.encode(header_text)
            boundary = 1          # the new messages (tool responses, then the question) open after <|tool_response>
            header = len(tok.encode(header_text))
            if k == 0:
                # A header-less continuation (thinking off's shape) is refused, the session unchanged.
                bare = dict(body(tail[:-TOOL], 0, 1, messages))
                status, data = await srv.send('complete', bare)
                self.assertEqual((status, data['type']), (400, 'BAD_GEOMETRY'), data)
            status, out = await srv.send('complete', body(tail, header, boundary, messages))
            self.assertEqual(status, 200, out)
            self.assertEqual(out['prompt_tokens'], len(stream) + len(tail) - 1)
            stream += tail[1:] + out['output_ids']
            self.assertEqual(out['text'], replies[k + 1] + ('<|tool_response>' if k == 0 else '<turn|>'))
        rows = srv.rows()
        # The one failure row is the refused header-less continuation.
        self.assertEqual([f['error'] for f in srv.rows('failures.jsonl')], ['KvSessionError'])
        self.assertEqual([r['kv_session']['header_tokens'] for r in rows], [H, TOOL, H])
        self.assertEqual(list(srv.transcripts[0].stream), stream)


if __name__ == '__main__':
    unittest.main()
