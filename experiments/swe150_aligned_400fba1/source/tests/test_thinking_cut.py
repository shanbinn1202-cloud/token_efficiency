"""serve.py --thinking-history strip (remote_ra_live_v2.thinking_cut): thinking on, option (a),
form E ("commit at the cut"), on the chat-text protocol of the Qwen profile. CPU only; not GPU
evidence (docs/thinking-cut/DESIGN.md, checks G1 and G2 are the GPU's).

1. The template: ``thinking_cut.reply_piece(c)`` is what the pinned template renders for a past
   reply with the stock default arguments (enable_thinking=True, no preserve_thinking), for
   contents with surrounding whitespace, Unicode, a ``</think>`` inside (one or several), an
   empty content (a reply cut at its bound inside the reasoning); with the jinja2 stand-in and,
   with Transformers (the remote-ra-tok venv), the pinned Qwen3.6 tokenizer, where the reply head
   is the header's first K = 3 tokens. The startup check and what it refuses.
2. The transcript: every prompt equals the stock render of its messages with every reasoning
   dropped (clients that send R back or not, echoes with whitespace, ``</think>`` in C, cut
   replies); after a commit the stream is prompt[:X], X = N - |header|; the re-prefill geometry
   (stable X + K, the span to e, the piece |A|); another history rebuilds; the K-token check
   fails closed; the capacity gate's copy (reasoning kept) extends alike.
3. Pins and refusals: the records, the startup checks, the launcher (off changes nothing, on adds
   the records only and no vLLM argument), the dry run, ready.json, the worker flag and echo,
   the attachments (Gemma, kv-session, mswea-xml, no --enable-thinking, --gdn-device-handoff,
   --phase1-in-step refused).
4. End to end on the engine fakes (test_chat_text_e2e's server: the packaged router, the chat-text
   route and session pool, the frontend, the real engine and worker over test_stageb_worker's
   FakeRunner, whose every forward checks its GDN chain and KV against the dense state of its own
   prompt): each completion commits at X (seal floor(X/P), the tail at X, no phase-2 vectors, the
   carry's identity token the phase-1 state's at a continuing turn, a split capture at local turn
   0 whose prefill forward ends at X under 'page' and whole-prompt chunks, no C' capture); the
   next turn's geometry and re-prefill; the evidence (plan commit_end, truncate with the dropped
   ids, reprefill, the row's committed ends, phase2_end and truncate); every prompt is the stock
   render and every decision is the one a dense recomputation of the same prompt gives (the
   selection's candidate vectors and query, its ranking, the Goal inputs); --admission-overlap,
   the three decode fast paths, --deferred-completion-io, --release-idle-pages under pressure,
   the capacity gate, goal_query and observation_query; and test_chat_text_e2e's scenarios
   (sessions decoding together, summary hand-offs, asides, rebinds after a cancel, idle closes,
   goal_query with releases) on a strip server.
5. Mutations: each documented mistake (the carry at C', a sealed phase-2 Page, a phase-2 vector,
   a stable end at X, the reasoning kept, no split at local turn 0, the template's
   preserve_thinking) makes these checks fail.
6. The ATIF replay (opt-in: REMOTE_RA_TB_ATIF_DIR, the pinned tokenizer): the 178 Terminal-Bench
   Terminus 2 trials of test_chat_text_replay with reasoning inserted into every reply; every
   served prompt equals the stock render, cut replies included.
"""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
import test_align_ctx as AC  # noqa: E402
import test_chat_text_protocol as T  # noqa: E402  kits, Terminus prompts and client
import test_kv_session_protocol as KP  # noqa: E402  the pinned template, jinja render
import test_mswea_protocol as X  # noqa: E402  the e2e harness tokenizer

from remote_ra_http import chat_text_route as CR  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2 import thinking_cut as CUT  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error, closed_pages  # noqa: E402
from runtime_renderer import EngineSourceRenderer  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

STRIP = TH.record(TH.CHAT, history=CUT.STRIP)
KW = CUT.TEMPLATE_KWARGS
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'


def plain(messages):
    """The messages as the reference renders them: role and content, every reasoning dropped."""
    return [dict(role=m['role'], content=m['content']) for m in messages]


def raw_reply(reasoning, content):
    return reasoning + '\n' + TH.THINK_END + '\n\n' + content


def kits():
    found = [T.char_kit()]
    real = T.real_kit()
    if real is not None:
        found.append(real)
    return found


def reference(kit, messages):
    """The stock reference prompt: the pinned template with enable_thinking=True and no
    preserve_thinking, every reasoning dropped, with the generation header."""
    return kit.encode(kit.render(plain(messages), True, **KW))


def renderer_of(kit):
    """EngineSourceRenderer over a kit (the template callback is the kit's render)."""
    class Tokenizer:
        eos_token_id, pad_token_id = T.EOS, T.PAD

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return kit.encode(text)

        def decode(self, ids, skip_special_tokens=False):
            assert skip_special_tokens is True
            return kit.decode(ids)
    renderer = EngineSourceRenderer.__new__(EngineSourceRenderer)
    renderer.tokenizer, renderer.template = Tokenizer(), KP.TEMPLATE
    renderer._check_identity = lambda: None
    renderer._render_template = kit.render_template
    return renderer


def strip_transcript(kit):
    return renderer_of(kit).chat_transcripts(C.record(thinking=True, strip=True), thinking=True, strip=True)()


CONTENTS = ['{"a": 1}', '  {"a": 1}\n\n', '\n\n' + T.action(3), 'R</think>\n\n{"b": 2}', '', '   ',
            'x</think>y</think>\n\n  z ', 'Unicode: é中文 \U0001f600 tld.उदा',
            '{"keystrokes": "cat <<EOF\\n</think>\\nEOF\\n"}', '\t\tindented\n', '<think>inner</think>after',
            '\n\n</think>\n\n']


def random_contents(seed, count=40):
    rng = random.Random(seed)
    alphabet = ['a', 'b', ' ', '\n', '\t', '{', '}', '"', 'é', '中', '</think>', '<think>', '\r\n', '\\n',
                '<|im_start|>', 'x' * 7]
    return [''.join(rng.choice(alphabet) for _ in range(rng.randrange(0, 18))) for _ in range(count)]


# ---- 1. the template's history render -------------------------------------------------------------

class TemplateTests(unittest.TestCase):
    maxDiff = None

    def test_reply_piece_is_the_template_render(self):
        for kit in kits():
            for content in CONTENTS + random_contents(1):
                if '<|im_start|>' in content:
                    continue          # a special token inside content: not a reply (the template renders it raw)
                with self.subTest(kit=kit.name, content=content):
                    messages = [dict(role='user', content='u'), dict(role='assistant', content=content),
                                dict(role='user', content='v')]
                    text = kit.render(messages, True, **KW)
                    self.assertEqual(text, C.user_piece('u') + CUT.reply_piece(content) + C.user_piece('v')
                                     + TH.HEADER_TEXT)
                    # The pieces encode alone as inside the whole render.
                    self.assertEqual(kit.encode(text), kit.encode(C.user_piece('u')) +
                                     kit.encode(CUT.reply_piece(content)) + kit.encode(C.user_piece('v')) +
                                     kit.encode(TH.HEADER_TEXT))

    def test_strip_content(self):
        self.assertEqual(CUT.strip_content('  R</think>\n\n{"a": 1} '), '{"a": 1}')
        self.assertEqual(CUT.strip_content('a</think>b</think>\n\n c'), ' c')
        self.assertEqual(CUT.strip_content('\n\n{"a": 1}\n'), '{"a": 1}')
        self.assertEqual(CUT.strip_content(''), '')
        self.assertEqual(CUT.reply_piece('x'), '<|im_start|>assistant\nx<|im_end|>\n')
        with self.assertRaises(LiveV2Error):
            CUT.strip_content(None)

    def test_the_reply_head_is_three_pinned_tokens(self):
        real = T.real_kit()
        if real is None:
            self.skipTest('the pinned tokenizer needs Transformers (remote-ra-tok venv)')
        head, header = real.encode(CUT.REPLY_HEAD), real.encode(TH.HEADER_TEXT)
        self.assertEqual((len(head), len(header), header[:3]), (3, 5, head))
        transcript = strip_transcript(real)
        self.assertEqual(transcript.reply_head, tuple(head))
        # </think> is one added token: the evidence's think_end is its index, no decode.
        self.assertEqual(transcript.think_end_ids, (248069,))
        # Every content's piece starts with those three ids (the per-turn check never refuses them).
        for content in CONTENTS + random_contents(2, 200):
            if '<|im_start|>' in content:
                continue
            piece = real.encode(CUT.reply_piece(content))
            self.assertEqual(piece[:3], head, content)
        # A content that starts with '\r' after its last </think> (the pre-token '\n\r\n'): the
        # pinned BPE still keeps the head's '\n' as its own id.
        self.assertEqual(real.encode(CUT.reply_piece('a</think>\r\nb'))[:3], head)

    def test_startup_check_and_refusals(self):
        kit = T.char_kit()
        self.assertEqual(C.check_template_strip(kit.render_template, KP.TEMPLATE), TH.HEADER_TEXT)
        for name, render in (('preserve_thinking', lambda t, m, k: kit.render_template(
                                  t, m, dict(k, preserve_thinking=True))),
                             ('thinking off', lambda t, m, k: kit.render_template(t, m, dict(k, enable_thinking=False)))):
            with self.subTest(name), self.assertRaisesRegex(LiveV2Error, 'differs'):
                C.check_template_strip(render, KP.TEMPLATE)
        renderer = renderer_of(kit)
        transcript = renderer.chat_transcripts(C.record(thinking=True, strip=True), thinking=True, strip=True)()
        self.assertTrue(transcript.strip and transcript.thinking)
        with self.assertRaisesRegex(RuntimeError, 'strip needs thinking'):
            renderer.chat_transcripts(C.record(), strip=True)
        with self.assertRaisesRegex(LiveV2Error, 'requires --enable-thinking'):
            C.record(strip=True)
        with self.assertRaisesRegex(LiveV2Error, 'requires thinking on'):
            C.ChatTextTranscript(encode=kit.encode, render_chat=lambda m: [1], decode=kit.decode, stop_ids=kit.stop_ids,
                                 strip=True)
        # Gemma's turns strip since gemma-thinking (test_gemma_thinking_cut), with Gemma's tokenizer: its
        # channel markers must be single ids.
        with self.assertRaisesRegex(LiveV2Error, 'channel markers must be one token each'):
            C.ChatTextTranscript(encode=kit.encode, render_chat=lambda m: [1], decode=kit.decode, stop_ids=kit.stop_ids,
                                 grammar=C.GEMMA4_CHAT, thinking=True, strip=True)
        self.assertEqual(C.record(thinking=True, strip=True),
                         dict(C.record(thinking=True), template_kwargs={'enable_thinking': True}))


# ---- 2. the transcript --------------------------------------------------------------------------

class TranscriptTests(unittest.TestCase):
    maxDiff = None

    def play(self, kit, *, turns=5, sent_back=None, echo=None, replies=None, bound=None):
        """One client session on a strip transcript: the client sends ``echo(content)`` back as the
        assistant message (with ``sent_back(reasoning)`` as reasoning_content when not None).
        Returns (transcript, [(messages, tokens, stable, span, committed)])."""
        ours = strip_transcript(kit)
        messages = [dict(role='user', content=T.first_prompt('Strip task.'))]
        served = []
        for k in range(turns):
            before = ours.stream
            tokens, stable, span = ours.render([dict(m) for m in messages])
            text = replies(k) if replies is not None else raw_reply('Turn %d drafts {"x": %d}.' % (k, k), T.action(k))
            generated = kit.encode(text) + [T.EOS]
            if bound is not None:
                generated = generated[:bound(k)]
            committed = ours.commit(tokens, generated)
            served.append(([dict(m) for m in messages], tokens, stable, span, committed, before))
            reply = dict(role='assistant', content=(echo or (lambda c: c))(committed['content']))
            if sent_back is not None and sent_back(committed['reasoning']) is not None:
                reply['reasoning_content'] = sent_back(committed['reasoning'])
            messages = messages + [reply, dict(role='user', content=T.observation(k))]
        return ours, served

    def check(self, kit, ours, served, histories=None):
        header = kit.encode(TH.HEADER_TEXT)
        K = len(kit.encode(CUT.REPLY_HEAD))
        for k, (messages, tokens, stable, span, committed, before) in enumerate(served):
            self.assertEqual(list(tokens), reference(kit, messages), k)
            X = len(tokens) - len(header)
            self.assertEqual((committed['cut'], list(tokens[X:])), (X, header))
            if committed['history'] == CUT.CUT:
                # Re-prefill: the stream before, A = the reply piece, the user piece, the header.
                piece = kit.encode(CUT.reply_piece(messages[-2]['content']))
                user = kit.encode(C.user_piece(messages[-1]['content']))
                C0 = len(before)
                self.assertEqual(list(tokens), list(before) + piece + user + header)
                self.assertEqual((stable, span), (C0 + K, (C0 + K, C0 + len(piece) + len(user))))
                self.assertEqual(committed['reprefill'], dict(start=C0, piece_end=C0 + len(piece)))
                self.assertEqual(list(tokens[C0:C0 + K]), header[:K])
            else:
                self.assertIsNone(committed['reprefill'])
        self.assertEqual(list(ours.stream), list(served[-1][1][:served[-1][4]['cut']]))
        if histories is not None:
            self.assertEqual([c['history'] for *_, c, _ in served], histories)

    def test_every_prompt_is_the_stock_render(self):
        cases = {'dropped': dict(), 'sent back': dict(sent_back=lambda r: r),
                 'stripped echo': dict(echo=str.strip), 'padded echo': dict(echo=lambda c: '  ' + c + '\n '),
                 'think end in C': dict(replies=lambda k: raw_reply('R%d' % k, 'C%d </think>\n\n%s' % (k, T.action(k)))),
                 'no think end': dict(replies=lambda k: 'reasoning only %d, never closed' % k),
                 'cut at the bound': dict(bound=lambda k: 7 + k)}
        for kit in kits():
            for name, options in cases.items():
                with self.subTest(kit=kit.name, case=name):
                    ours, served = self.play(kit, **options)
                    self.check(kit, ours, served, ['fresh'] + ['cut'] * 4)
                    self.assertEqual(ours.sessions, 1)
                    self.assertEqual(ours.previous_action(), served[-1][4]['content'])

    def test_reasoning_never_reaches_the_prompt(self):
        kit = T.char_kit()
        a, sa = self.play(kit, turns=3)
        b, sb = self.play(kit, turns=3, sent_back=lambda r: 'something else entirely')
        self.assertEqual([list(x[1]) for x in sa], [list(x[1]) for x in sb])
        self.assertEqual([x[4]['history'] for x in sb], ['fresh', 'cut', 'cut'])

    def test_another_history_rebuilds_from_text(self):
        kit = T.char_kit()
        ours, served = self.play(kit, turns=3)
        messages = served[-1][0] + [dict(role='assistant', content=served[-1][4]['content']),
                                    dict(role='user', content='next')]
        # A different earlier reply (Terminus unwinding, a summary hand-off): a rebuild from text.
        changed = [dict(m) for m in messages]
        changed[2] = dict(role='assistant', content='rewritten')
        plan = ours.plan(changed)
        self.assertEqual((plan['rebuild'], plan['history'], plan['reprefill']), (True, CUT.REBUILD, None))
        tokens, stable, span = ours.render(changed)
        self.assertEqual((list(tokens), stable, span), (reference(kit, changed), len(tokens) - len(ours.header), None))
        committed = ours.commit(tokens, kit.encode(raw_reply('r', 'c')) + [T.EOS])
        self.assertEqual((committed['rebuild'], committed['history'], ours.turn, ours.sessions), (True, 'rebuild', 1, 2))
        self.assertEqual(list(ours.stream), list(tokens[:committed['cut']]))
        # A reply that does not come back as returned (another content) rebuilds too.
        fresh = strip_transcript(kit)
        first = [dict(role='user', content='u')]
        fresh.render(first)
        fresh.commit(fresh.pending['tokens'], kit.encode(raw_reply('r', 'said')) + [T.EOS])
        self.assertTrue(fresh.plan(first + [dict(role='assistant', content='not what you said'),
                                            dict(role='user', content='o')])['rebuild'])
        self.assertFalse(fresh.plan(first + [dict(role='assistant', content=' said '),
                                             dict(role='user', content='o')])['rebuild'])

    def test_head_check_fails_closed(self):
        kit = T.char_kit()
        ours, served = self.play(kit, turns=1)
        messages = served[0][0] + [dict(role='assistant', content=served[0][4]['content']),
                                   dict(role='user', content='o')]
        encode = ours.encode

        def shifted(text):
            ids = encode(text)
            return ids[1:] + ids[:1] if text.startswith(CUT.REPLY_HEAD) else ids
        ours.encode = shifted
        with self.assertRaises(C.ChatTextError) as caught:
            ours.plan(messages)
        self.assertEqual((caught.exception.kind, caught.exception.status), ('STRIP_REPLY_HEAD', 500))
        self.assertIsNone(ours.pending)
        ours.encode = encode
        self.assertEqual(ours.plan(messages)['history'], CUT.CUT)

    def test_extends_ignores_the_reasoning_of_the_capacity_copy(self):
        kit = T.char_kit()
        ours, served = self.play(kit, turns=2)
        messages = served[-1][0] + [dict(role='assistant', content=served[-1][4]['content'], reasoning='R'),
                                    dict(role='user', content='o')]
        self.assertTrue(ours.extends(messages))
        self.assertTrue(ours.extends(plain(messages)))
        self.assertEqual(ours.plan(messages)['history'], CUT.CUT)
        self.assertFalse(ours.extends(messages[:-2] + [dict(role='assistant', content='other'),
                                                       dict(role='user', content='o')]))


# ---- 3. pins and refusals -------------------------------------------------------------------------

class PinTests(unittest.TestCase):
    def config(self, **extra):
        return dict(dict(protocol='chat_text_v1', thinking=STRIP), **extra)

    def test_records(self):
        self.assertEqual(STRIP, dict(TH.record(TH.CHAT), history='strip', template_kwargs={'enable_thinking': True},
                                     append=CUT.RULE))
        self.assertEqual(TH.record(TH.CHAT), TH.record(TH.CHAT, None))
        for protocol in (TH.XML, 'kv_session_v1'):
            with self.subTest(protocol), self.assertRaises(LiveV2Error):
                TH.record(protocol, history='strip')
        with self.assertRaises(LiveV2Error):
            TH.record(TH.CHAT, history='keep')
        self.assertEqual(TH.history_mode(self.config()), 'strip')
        self.assertIsNone(TH.history_mode(dict(protocol='chat_text_v1', thinking=TH.record(TH.CHAT))))
        self.assertIsNone(TH.history_mode({}))
        self.assertTrue(CUT.pinned(self.config()))
        self.assertIsNone(CUT.pinned(dict(protocol='chat_text_v1', thinking=TH.record(TH.CHAT))))
        self.assertIsNone(CUT.pinned({}))

    def test_refusals(self):
        refused = {'Gemma': self.config(model_profile='gemma4-31b-it'),
                   'mswea-xml': self.config(protocol='mswea_xml_v1'),
                   'kv-session': self.config(protocol='kv_session_v1'),
                   'handoff': self.config(admission=dict(gdn_device_handoff=True)),
                   'in step': self.config(admission=dict(phase1_in_step=True)),
                   'another record': self.config(thinking=dict(STRIP, template_kwargs=TH.TEMPLATE_KWARGS[TH.CHAT]))}
        for name, config in refused.items():
            with self.subTest(name), self.assertRaises(LiveV2Error):
                CUT.check_config(config)
        from remote_ra_live_v2 import admission_flags
        self.assertTrue(CUT.check_config(self.config(admission=admission_flags.record(admission_overlap=4))))

    def test_worker_flag_and_attachments(self):
        self.assertIsNone(CUT.worker_flag({}))
        self.assertTrue(CUT.worker_flag({CUT.WORKER_KEY: 'strip'}))
        with self.assertRaises(LiveV2Error):
            CUT.worker_flag({CUT.WORKER_KEY: True})
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                patch('remote_ra_http.config.load', lambda: dict(protocol='chat_text_v1')):
            with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned'):
                CUT.worker_flag({CUT.WORKER_KEY: 'strip'})
            self.assertIsNone(CUT.worker_flag({}))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                patch('remote_ra_http.config.load', lambda: self.config()):
            self.assertTrue(CUT.loaded())
            with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned'):
                CUT.worker_flag({})
        adapter, bridge = NS(), NS()
        CUT.attach_engine(adapter, bridge)
        self.assertEqual((adapter.commit_at_cut, adapter.cut_records, bridge.commit_at_cut), (True, {}, True))
        with self.assertRaisesRegex(LiveV2Error, 'collision'):
            CUT.attach_engine(adapter, bridge)
        # The window layout (Gemma) is handed to the held cut window (window_hold; test_gemma_thinking_cut,
        # test_gemma_hold): an adapter with a window but no window layout is refused there.
        with self.assertRaisesRegex(LiveV2Error, 'the window hold needs the window layout'):
            CUT.attach_engine(NS(window=object()), NS())
        owner, qwen, backend = NS(), NS(), NS()
        CUT.attach_worker(owner, qwen, backend)
        self.assertTrue(owner.commit_at_cut and qwen.commit_at_cut and backend.commit_at_cut)
        with self.assertRaisesRegex(LiveV2Error, 'gdn-device-handoff'):
            CUT.attach_worker(NS(handoff=True), NS(), NS())
        with self.assertRaisesRegex(LiveV2Error, 'phase1-in-step'):
            CUT.attach_worker(NS(phase1_in_step=True), NS(), NS())

    def test_split(self):
        a0 = NS(local_turn=0, observation_end=100)
        self.assertEqual([CUT.split(a0, s, c) for s, c in ((0, 100), (0, 101), (96, 16), (100, 5), (64, 32), (0, 99))],
                         [100, 100, 4, 5, 32, 99])
        self.assertEqual(CUT.split(NS(local_turn=1, observation_end=100), 96, 16), 16)
        from remote_ra_live_v2.composition import EngineBridge
        bridge = EngineBridge()
        request = NS(request_id='r')
        bridge.assignments['r'] = NS(assignment=a0)
        self.assertEqual(bridge.scheduler_split(request, 96, 16), 16)       # off: vLLM's chunk
        bridge.commit_at_cut = True
        self.assertEqual((bridge.scheduler_split(request, 96, 16), bridge.scheduler_split(NS(request_id='x'), 96, 16)),
                         (4, 16))

    def test_think_end(self):
        self.assertEqual(CUT.think_end([5, 7, 9, 7], (7,)), 1)
        self.assertIsNone(CUT.think_end([5, 9], (7,)))
        self.assertIsNone(CUT.think_end([5, 7], (7, 8)))
        self.assertIsNone(CUT.think_end([5, 7], None))

    def test_native_and_row_fields(self):
        """The native file's truncate and reprefill (from the engine's cut record and its
        transcript's committed turn) and the runtime row's counts; a disagreement refuses."""
        record = dict(cut=5, end=12, header_ids=(1, 2, 3, 4, 5), generated_ids=[9, 248069, 8])
        turn = dict(cut=5, ids=(1, 2, 3, 4, 5, 9, 248069, 8), reprefill=dict(start=2, piece_end=4))
        transcript = NS(turns=[turn], think_end_ids=(248069,))
        core = NS(_remote_ra_growing_adapter=NS(commit_at_cut=True, cut_records={'r': record}),
                  _remote_ra_private_v2_endpoint=NS(transcripts={'s': transcript}))
        binding = NS(request='r', session='s')
        fields = CUT.native_fields(core, binding)
        self.assertEqual(fields, dict(truncate=dict(cut=5, dropped=dict(start=5, end=12, header_ids=[1, 2, 3, 4, 5],
                                                                         generated_ids=[9, 248069, 8], think_end=1)),
                                      reprefill=dict(start=2, piece_end=4)))
        native = dict(fields, worker_frames=[dict(operation='complete', computed_end=11)])
        self.assertEqual(CUT.row_fields(native), (dict(phase2_end=11), dict(
            cut=5, dropped_tokens=7, header_tokens=5, generated_tokens=3, think_end=1,
            reprefill=dict(start=2, piece_end=4))))
        self.assertEqual(CUT.row_fields(dict(worker_frames=[])), ({}, None))
        turn['reprefill'] = None
        self.assertNotIn('reprefill', CUT.native_fields(core, binding))
        turn['ids'] = (1, 2, 3, 4, 5, 9, 9, 8)
        with self.assertRaisesRegex(LiveV2Error, 'differs from the committed cut'):
            CUT.native_fields(core, binding)
        with self.assertRaisesRegex(LiveV2Error, 'without its cut record'):
            CUT.native_fields(core, NS(request='x', session='s'))
        self.assertEqual(CUT.native_fields(NS(_remote_ra_growing_adapter=NS()), binding), {})


class LauncherTests(AC.Release, unittest.TestCase):
    maxDiff = None

    def chat(self, policy='observation_query', out=None, **options):
        type(self).count += 1
        out = out or 'chat%d' % self.count
        selection = dict(active_token_budget=8192) if policy in ('observation_query', 'random_fill') else None
        options.setdefault('max_running', 4)
        options.setdefault('max_tokens', 4096)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, None, policy, self.root / out, 8231, selection, 'chat-text', 24, 16,
                                 None, **options)

    def test_off_and_keep_are_unchanged(self):
        for policy in ('full', 'observation_query'):
            with self.subTest(policy=policy):
                plain = self.chat(policy, out='off')
                self.assertEqual(self.chat(policy, out='off', thinking_history=None), plain)
                keep = self.chat(policy, out='keep', enable_thinking=True)
                self.assertEqual(self.chat(policy, out='keep', enable_thinking=True, thinking_history=None), keep)
                self.assertNotIn('history', keep[1]['thinking'])

    def test_strip_adds_the_records_and_no_vllm_argument(self):
        for policy in ('full', 'observation_query', 'goal_query', 'random_fill'):
            with self.subTest(policy=policy):
                selection = {}
                if policy == 'goal_query':
                    continue          # needs a rewriter model in the fixture; the records are the same
                m0, c0, a0, e0 = self.chat(policy, out='on', enable_thinking=True)
                m1, c1, a1, e1 = self.chat(policy, out='on', enable_thinking=True, thinking_history='strip')
                self.assertEqual((m1, a1), (m0, a0))
                self.assertEqual(c1, dict(c0, thinking=STRIP, chat_text=C.record(slots=512, thinking=True, strip=True)))
                self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                self.assertEqual(c1['chat_text']['template_kwargs'], {'enable_thinking': True})
                loaded = self.load(c1)
                self.assertEqual((TH.pinned(loaded), TH.history_mode(loaded), CUT.pinned(loaded)),
                                 (STRIP, 'strip', True))
                self.assertEqual(C.options(loaded)['thinking'], STRIP)
                self.assertTrue(CR.frontend(loaded).thinking)
                del selection

    def test_refusals(self):
        task = AC.tasks(1)[0]
        for options, text in ((dict(thinking_history='strip'), 'requires --enable-thinking'),
                              (dict(enable_thinking=True, thinking_history='keep'), 'must be strip'),
                              (dict(enable_thinking=True, thinking_history=True), 'must be strip'),
                              (dict(enable_thinking=True, thinking_history='strip', gdn_device_handoff=True),
                               'refuses --gdn-device-handoff'),
                              (dict(enable_thinking=True, thinking_history='strip', phase1_in_step=True),
                               'refuses --gdn-device-handoff and --phase1-in-step')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, text):
                self.chat(**options)
        # review-fixes-1110 (runtime review 1.6): the two policies that read a reply's classes from the
        # stream are refused with strip; with the keep mode, with thinking off and for recent_fill with
        # strip the launch is as before.
        for policy in ('recent_tools', 'akv_rule'):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, 'strip refuses --policy recent_tools and akv_rule'):
                    self.chat(policy, enable_thinking=True, thinking_history='strip')
                self.assertEqual(self.chat(policy, enable_thinking=True)[1]['policy'], policy)
                self.assertEqual(self.chat(policy)[1]['policy'], policy)
        self.assertEqual(self.chat('recent_fill', enable_thinking=True, thinking_history='strip')[1]['thinking'], STRIP)
        with self.assertRaisesRegex(ValueError, 'serves --protocol chat-text only'):
            self.prepare(task, enable_thinking=True, thinking_history='strip')
        with self.assertRaisesRegex(ValueError, 'requires --enable-thinking|serves --protocol chat-text only'):
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.prepare(self.release, None, 'full', self.root / 'kvs', 8231, None, 'kv-session', 24, 16, None,
                              max_running=4, thinking_history='strip')
        strip = self.chat(out='rs', enable_thinking=True, thinking_history='strip')[1]
        keep = self.chat(out='rk', enable_thinking=True)[1]
        refused = {'the keep chat record': dict(strip, chat_text=keep['chat_text']),
                   'the strip chat record on keep': dict(keep, chat_text=strip['chat_text']),
                   'a strip record with preserve': dict(strip, thinking=dict(STRIP, template_kwargs=keep['thinking'][
                       'template_kwargs'])),
                   'handoff': dict(strip, admission=dict(gdn_device_handoff=True))}
        for name, value in refused.items():
            with self.subTest(name), self.assertRaises(RuntimeError):
                self.load(value)

    def test_dry_run_main(self):
        def main(*extra):
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--output', str(self.root / 'dry'),
                    '--port', '8231', '--policy', 'full', '--protocol', 'chat-text', '--max-running-sessions', '4',
                    '--page-tokens', '16', '--kv-cache-gib', '24', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(stream.getvalue())
        keep = main('--enable-thinking')
        strip = main('--enable-thinking', '--thinking-history', 'strip')
        self.assertEqual(strip, dict(keep, thinking=STRIP))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main('--enable-thinking', '--thinking-history', 'keep')

    def test_compose_engine_sends_the_record_once_and_checks_the_echo(self):
        import test_integrate_flags as IF
        strip = self.chat(out='ce', enable_thinking=True, thinking_history='strip')[1]
        keep = self.chat(out='ck', enable_thinking=True)[1]
        record, error = IF.engine_record(strip)
        plain, plain_error = IF.engine_record(keep)
        self.assertEqual((error, plain_error), (None, None))
        self.assertEqual(record, dict(plain, thinking_history='strip'))
        echo = {key: record[key] for key in ('context', 'admission') if key in record}
        receipt_check = 'loaded CUDA worker attachment receipt'      # the first check after the echoes
        self.assertIn(receipt_check, IF.engine_record(strip, dict(echo, thinking_history='strip'))[1])
        self.assertIn('another thinking_history', IF.engine_record(strip, echo)[1])
        self.assertIn('another thinking_history', IF.engine_record(keep, dict(echo, thinking_history='strip'))[1])
        self.assertIn(receipt_check, IF.engine_record(keep, echo)[1])

    def test_worker_attaches_and_echoes(self):
        import test_integrate_flags as IF
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            plain = IF.attach()[0]
            evidence, runner = IF.attach({CUT.WORKER_KEY: 'strip'})
            self.assertNotIn(CUT.WORKER_KEY, plain)
            self.assertEqual(evidence, dict(plain, thinking_history='strip'))
            owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
            self.assertTrue(owner.commit_at_cut and bridge.qwen.commit_at_cut)
            for flag in ('gdn_device_handoff', 'phase1_in_step'):
                with self.subTest(flag), self.assertRaisesRegex(LiveV2Error, 'refuses --gdn-device-handoff'):
                    IF.attach({CUT.WORKER_KEY: 'strip', 'admission': {flag: True}})
        strip = self.chat(out='cw', enable_thinking=True, thinking_history='strip')[1]
        with self.startup(strip):
            self.assertTrue(CUT.worker_flag({CUT.WORKER_KEY: 'strip'}))
            with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned'):
                CUT.worker_flag({})

    def test_ready_records_the_pin_and_checks_the_echo(self):
        import pathlib
        import test_ms_frontend as MF
        from remote_ra_http.evidence import FrontendEvidence
        readiness = dict(native_ready=True)
        authority = NS(identity=NS(epoch=MF.EPOCH, manifest='m' * 64), policy='full', session_id=MF.SERVER,
                       verify_admission=lambda: True, live_v2_readiness=readiness,
                       loaded=NS(manifest=dict(qualification_parent_manifest='q' * 64)))
        read_text = pathlib.Path.read_text

        def proc(path, *args, **kwargs):
            name = str(path).replace('\\', '/')
            if name == '/proc/sys/kernel/random/boot_id':
                return 'boot\n'
            if name == '/proc/self/stat':
                return '1 (python) ' + ' '.join(['0'] * 30)
            return read_text(path, *args, **kwargs)

        def ready(config, echo):
            loaded = self.load(config)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                value = dict(loaded, output_directory=str(output))
                pools = dict(kv_layout=loaded['kv_layout'], num_blocks_by_pool=[10, 20], free_blocks_by_pool=[9, 19])
                receipt = dict(tensor_geometry={}, source_closure={}, **echo)
                (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                    epoch=MF.EPOCH, report=readiness, kv_pools=pools, receipt=receipt)).encode())
                with patch('remote_ra_http.evidence.load', lambda: value), \
                        patch.object(pathlib.Path, 'read_text', proc):
                    evidence = FrontendEvidence(authority, None, dict(loaded['effective_server_config']))
                    evidence.write_ready(NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]),
                                         NS(started=True))
                return json.loads((output / 'ready.json').read_bytes())
        strip = self.chat('full', out='ready', enable_thinking=True, thinking_history='strip', max_tokens=1536)[1]
        keep = self.chat('full', out='ready', enable_thinking=True, max_tokens=1536)[1]
        on, off = ready(strip, dict(thinking_history='strip')), ready(keep, {})
        self.assertEqual((on['thinking'], off['thinking']), (STRIP, TH.record(TH.CHAT)))
        differ = ('thinking', 'server_run_id', 'effective_server_config', 'server_config_sha256', 'worker_receipt_sha256')
        self.assertEqual({k: v for k, v in on.items() if k not in differ}, {k: v for k, v in off.items() if k not in differ})
        for config, echo in ((strip, {}), (keep, dict(thinking_history='strip'))):
            with self.subTest(echo=echo), self.assertRaisesRegex(RuntimeError, 'thinking_history mismatch'):
                ready(config, echo)


class GemmaRefusalTests(unittest.TestCase):
    """The Gemma 4 profile takes the strip history since gemma-thinking (its transcript side,
    test_gemma_thinking_cut; its engine side holds the sliding window at the cut, test_gemma_hold): with
    --enable-thinking its records and the layout's hold slots are pinned, without it and with
    --async-scheduling it is refused."""

    def test_gemma_refuses(self):
        import test_gemma_capacity as GC
        case = GC.LauncherTests('test_vllm_command')
        case.setUp()
        try:
            chat = dict(protocol='chat-text', max_running=4)
            for options, pattern in ((dict(thinking_history='strip'), 'requires --enable-thinking'),
                                     (dict(enable_thinking=True, thinking_history='strip', async_scheduling=True,
                                           decode_fastpath=True, decode_fastpath2=True), 'refuses --async-scheduling')):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, pattern):
                    case.prepare(None, 'observation_query', dict(active_token_budget=8192), **chat, **options)
            config = case.prepare(None, 'observation_query', dict(active_token_budget=8192), **chat,
                                  enable_thinking=True, thinking_history='strip')[1]
            self.assertEqual(config['thinking'], TH.record(TH.CHAT, 'strip', profile='gemma4-31b-it'))
            self.assertGreaterEqual(config['kv_layout']['hold_slots'], 1)
        finally:
            case.doCleanups()


# ---- 4. end to end on the engine fakes ----------------------------------------------------------

HEADER = tuple(X.TOK.encode(TH.HEADER_TEXT))
HEAD = len(X.TOK.encode(CUT.REPLY_HEAD))        # K of the e2e tokenizer (characters): 11; Qwen3.6: 3
P16 = 16


def e2e_render_chat(messages):
    """EngineSourceRenderer.render_chat_strip over the e2e harness tokenizer."""
    return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, C.validate_messages(messages), None, False, **KW))


def e2e_reference(messages):
    """The stock reference prompt over the e2e harness tokenizer (every reasoning dropped)."""
    return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, plain(messages), None, True, **KW))


def e2e_transcripts(record, thinking=True, strip=True):
    import test_chat_text_e2e as CE
    assert (record['header_text'], record['template_kwargs'], thinking, strip) == (TH.HEADER_TEXT, KW, True, True)
    return lambda: C.ChatTextTranscript(encode=X.TOK.encode, render_chat=e2e_render_chat,
                                        decode=lambda ids: X.TOK.decode(list(ids)), stop_ids=CE.STOPS,
                                        thinking=True, strip=True)


def thinking_model(original):
    """The scripted thinking-on model: a reasoning with a draft action and JSON, then ``</think>``
    and the original reply text, which is therefore the reply's content as before."""
    def model(messages):
        return ('Reasoning %d: draft {"keystrokes": "rm -rf /"} first.\n' % len(messages) + TH.THINK_END
                + original(messages))
    return model


@contextlib.contextmanager
def strip_patches():
    """test_chat_text_e2e's server with strip transcripts on both ends and the thinking model."""
    import test_chat_text_e2e as CE

    class Renderer(CE.S.Renderer):
        def chat_transcripts(self, record, thinking=False, strip=False):
            return e2e_transcripts(record, thinking, strip)
    with patch.object(CE, 'transcripts', e2e_transcripts), patch.object(CE, 'ChatRenderer', Renderer), \
            patch.object(CE, 'model', thinking_model(CE.model)):
        yield


def dense_row(tokens, x):
    """The FakeRunner's hidden row of position x of a prompt, [x + 1, T[x] + 1, 1] (final-norm
    gain 1), normalised: what a dense recomputation of the prompt gives."""
    from remote_ra_live_v2.query_selection import normalized
    return normalized([float(x + 1), float(tokens[x] + 1), 1.0])


class Recorder:
    """What a strip engine and worker did, checked as it happens: every completion's mirrors
    (engine and worker) at the cut; every selection's candidate vectors and query against a
    dense recomputation of the request's prompt; the Goal inputs; the carries' sources."""

    def __init__(self, h):
        self.h, self.final, self.splits, self.phase1, self.contexts = h, 0, [], {}, {}
        self.mismatches, self.dense, self.completions = [], 0, 0
        backend, adapter = h.backend, h.adapter
        final, split, hold = backend.capture_final_hybrid, backend.capture_split_carry, backend.hold_phase1_carry

        def counted_final(*args, **kwargs):
            self.final += 1
            return final(*args, **kwargs)

        def recorded_split(identity, slots, end, generation):
            self.splits.append((identity.request, end))
            return split(identity, slots, end, generation)

        def recorded_hold(identity, end, generation):
            candidate = hold(identity, end, generation)
            self.phase1[identity.request] = (end, candidate.digest)
            return candidate
        backend.capture_final_hybrid, backend.capture_split_carry = counted_final, recorded_split
        backend.hold_phase1_carry = recorded_hold
        complete = adapter.complete

        def checked(binding):
            receipt = complete(binding)
            self.completed(binding)
            return receipt
        adapter.complete = checked
        selection = h.selection
        if selection is not None:
            plan = selection.plan

            def recorded_plan(binding, goal, context, entries, query, n, vector_space=None, local_turn=None):
                self.check_dense(binding, entries, query, n)
                return plan(binding, goal, context, entries, query, n, vector_space, local_turn)
            selection.plan = recorded_plan
            for name in ('chat_context', 'chat_goal_context'):
                setattr(selection, name, self.context_of(getattr(selection, name), name == 'chat_goal_context'))

    def context_of(self, original, goal):
        def wrapped(binding, geometry, conversation, task, transcript):
            value = original(binding, geometry, conversation, task, transcript)
            context = value[0] if goal else value
            self.contexts[binding.request] = (context, [dict(m) for m in conversation], geometry.local_turn, goal)
            return value
        return wrapped

    def completed(self, binding):
        h, P = self.h, self.h.adapter.page_tokens
        s = h.adapter.sessions[binding.session]
        T, X_ = binding.tokens, len(binding.tokens) - len(HEADER)
        Q = X_ // P * P
        w = h.owner.sessions[s.namespace]
        tail = h.adapter.owner.tails.get(s.namespace)
        carry = h.backend.carry[binding.session]
        checks = dict(
            engine=(s.computed_tokens == T[:X_] and s.next_token == T[X_] and s.sealed_tokens == T[:Q] and
                    set(s.catalog) == set(range(Q // P)) and all(p < Q // P for p in s.resident)),
            tail=((tail is None) == (X_ == Q) and (tail is None or (tail.page, tail.start, tail.valid_end) ==
                                                    (Q // P, Q, X_)) and
                  (s.tail is None or (s.tail['page'], s.tail['valid_end']) == (Q // P, X_))),
            worker=(w['computed_end'] == X_ and w['sealed_end'] == Q and tuple(w['resident']) == tuple(s.resident) and
                    set(w['catalog']) == set(s.catalog) and (w['tail'] is None) == (s.tail is None) and
                    (w['tail'] is None or (w['tail']['page'], w['tail']['valid_end']) == (Q // P, X_))),
            carry=(carry.computed_end == X_ and carry.digest == s.boundary == w['carry_digest']))
        selection = h.selection
        if selection is not None and s.policy in ('observation_query', 'goal_query', 'random_fill'):
            checks['vectors'] = set(selection.vector_catalog.get(binding.session, {})) == set(range(Q // P))
        bad = sorted(k for k, ok in checks.items() if not ok)
        if bad:
            self.mismatches.append(('completion', binding.request, bad))
        self.completions += 1

    def check_dense(self, binding, entries, query, n):
        h = self.h
        P, T, policy = h.adapter.page_tokens, binding.tokens, h.adapter.sessions[binding.session].policy
        selection = h.selection
        if n == 0 or binding.local_turn == 0:
            return
        compactor = getattr(h.backend, 'compactor', None)
        if compactor is not None:
            # Decoupled: the worker's compaction of the dense rows (the same device code path).
            rows = torch.tensor([[float((p + 1) * P), float(T[(p + 1) * P - 1] + 1), 1.0] for p in range(n)])
            dense = {p: selection.page_vector((list(v.values), v.scale)) for p, v in enumerate(compactor.compact(rows))}
        else:
            dense = {p: selection.page_vector(dense_row(T, (p + 1) * P - 1)) for p in range(n)}
        if dict(entries) != dense:
            self.mismatches.append(('candidates', binding.request,
                                    sorted(p for p in range(n) if entries.get(p) != dense[p])))
        if policy == 'observation_query' and query not in (None, (1.0,)):
            e = binding.observation_span[1]
            if tuple(query) != tuple(selection.query_vector(dense_row(T, e - 1))):
                self.mismatches.append(('query', binding.request))
        self.dense += 1


def attach_strip(h):
    """compose_engine's and attach_loaded_worker's strip attachments on a fake engine and worker."""
    CUT.attach_engine(h.adapter, h.core._remote_ra_live_v2_bridge)
    CUT.attach_worker(h.owner, h.qwen, h.backend)
    h.recorder = Recorder(h)
    return h


def strip_server(test, policy='full', slots=8, limit=4, blocks=4000, via_config=False, release=False, goals=False,
                 extra=None, **record):
    """test_chat_text_e2e's ChatServer on a strip server: the strip records pinned, strip
    transcripts at both ends, the thinking model, the engine and the worker attached."""
    import test_chat_text_e2e as CE
    output = Path(test.enterContext(tempfile.TemporaryDirectory()))
    value = CE.config(policy, output, slots, limit, thinking=True, strip=True, **record)
    value['thinking'] = STRIP
    value.update(extra or {})
    test.enterContext(patch('remote_ra_http.config.load', lambda: value))
    test.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
    test.enterContext(strip_patches())
    srv = CE.ChatServer(policy, value, output, test.router, test.Completion, blocks=blocks, via_config=via_config,
                        release=release, goals=goals)
    test.addAsyncCleanup(srv.stop)
    attach_strip(srv.engine)
    return srv


def check_strip(test, srv, *, completions=None):
    """Every served request of a strip server: the prompt is the stock render of its messages
    (engine and frontend), the completion committed at the cut, the evidence (native plan,
    truncate, reprefill, worker frames, runtime row), the carry's source, the local turn 0
    split, the next turn's geometry, and the recorder's checks (mirrors, dense decisions)."""
    h = srv.engine
    P = h.adapter.page_tokens
    rows = {r['request_id']: r for r in srv.rows()}
    previous = {}          # (engine session, namespace generation) -> the cut of its last turn
    checked = 0
    for entry in srv.served:
        rid, messages, tokens = entry['rid'], entry['messages'], entry['tokens']
        test.assertEqual(tokens, e2e_reference(messages), rid)
        if rid not in rows:
            continue          # cancelled or failed after scheduling
        native, row = srv.native(rid), rows[rid]
        test.assertEqual(native['prompt_token_ids'], tokens)
        N, g = len(tokens), row['chat_text']['completion_tokens']
        X_ = N - len(HEADER)
        Q = X_ // P * P
        plan, done = native['plan'], native['worker_frames'][-1]
        local = plan['local_turn']
        test.assertEqual((plan['commit_end'], plan['seal_at_completion']['rule']), (X_, 'floor(X/P)'))
        test.assertEqual((done['operation'], done['computed_end'], done['sealed_end'], done['carry']['computed_end']),
                         ('complete', N + g - 1, Q, X_))
        test.assertEqual(done['dram'], list(range(Q // P)))
        test.assertEqual(done['tail'] and (done['tail']['page'], done['tail']['start'], done['tail']['valid_end']),
                         (Q // P, Q, X_) if X_ > Q else None)
        test.assertEqual(sorted(int(p) for p in done['page_vectors']),
                         list(closed_pages(plan['phase2_start'], max(plan['phase2_start'], X_), P)))
        test.assertEqual(sorted(int(p) for p in done['new_pages']),
                         sorted(set(range(plan['candidate_end'] // P, Q // P)) |
                                ({plan['sealed_start'] // P} if plan['tail_candidate'] else set())))
        generated = native['truncate']['dropped']['generated_ids']
        test.assertEqual(native['truncate'], dict(cut=X_, dropped=dict(start=X_, end=N + g, header_ids=tokens[X_:],
                                                                       generated_ids=generated, think_end=None)))
        test.assertEqual(len(generated), g)
        stage = row['stage_b']
        test.assertEqual((stage['computed_end'], stage['sealed_end'], stage['phase2_end']), (X_, Q, N + g - 1))
        test.assertEqual(row['truncate'], dict(cut=X_, dropped_tokens=N + g - X_, header_tokens=len(HEADER),
                                               generated_tokens=g, think_end=None, reprefill=native.get('reprefill')))
        test.assertEqual((row['thinking']['cut'], row['thinking']['reprefill']), (X_, native.get('reprefill')))
        key = (native['session_id'], plan['ns_generation'])
        if local:
            # The re-prefill: [C, e) = the reply piece (the client's echo as the template renders
            # it), the user piece; the stable end K tokens past the cut; the carry is the phase-1
            # state at e (its identity token).
            C0 = plan['computed_start']
            test.assertEqual(C0, previous.get(key, C0))
            piece = X.TOK.encode(CUT.reply_piece(messages[-2]['content']))
            test.assertEqual(native['reprefill'], dict(start=C0, piece_end=C0 + len(piece)))
            test.assertEqual(tokens[C0:C0 + len(piece)], piece)
            test.assertEqual((native['stable_end'], native['observation_span']), (C0 + HEAD, [C0 + HEAD, X_]))
            phase1 = [f for f in native['worker_frames'] if f['operation'] == 'phase1']
            test.assertEqual((phase1[0]['start'], phase1[0]['end']), (C0, X_))
            test.assertEqual(h.recorder.phase1.get(rid), (X_, done['carry']['digest']))
            test.assertEqual(phase1[0]['state_digest'], done['carry']['digest'])
            test.assertEqual(row['thinking']['history'], CUT.CUT)
        else:
            # Local turn 0: a prefill forward ended at X, where the carry was captured.
            test.assertNotIn('reprefill', native)
            test.assertIn((rid, X_), h.recorder.splits)
            test.assertIn(X_, [s + c for step in h.runner.steps for r, s, c in step if r == rid])
            test.assertIn(row['thinking']['history'], (CUT.FRESH, CUT.REBUILD))
        previous[key] = X_
        checked += 1
    test.assertEqual(h.recorder.final, 0)             # no C' capture: the carry is positioned
    test.assertEqual(h.recorder.mismatches, [])
    if completions is not None:
        test.assertEqual(checked, completions)
    return checked


def dense_goal_inputs(test, srv):
    """The Goal inputs of every chat-text turn equal those of its messages: the root objective,
    the newest observation, and the previous action the client sent back (local turn > 0)."""
    count = 0
    for rid, (context, conversation, local, goal) in srv.engine.recorder.contexts.items():
        if goal:
            continue          # terminus2_goal's inputs: test_chat_text_e2e's goal_query checks
        fields = C.goal_inputs(conversation, local == 0)
        test.assertEqual((context.task_objective, context.current_observation),
                         (fields['objective'], fields['observation']))
        if local:
            test.assertEqual(context.previous_action.strip(), conversation[-2]['content'].strip())
        count += 1
    return count


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self):
        import test_ms_e2e as E2E
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def send(self, srv, messages, max_tokens=None):
        """One request: reasoning_content on the wire; vLLM's request model without it."""
        import test_kv_session_e2e as KE
        body = dict(model='selective-prefix', messages=messages)
        if max_tokens is not None:
            body['max_tokens'] = max_tokens
        wire = json.dumps(body).encode()

        async def read():
            return wire
        raw = NS(app=NS(state=srv.state), headers={}, receive=asyncio.Queue().get, body=read,
                 client=NS(host='127.0.0.1'))
        request = KE.chat_request(**dict(body, messages=[{k: v for k, v in m.items() if k != 'reasoning_content'}
                                                         for m in messages]))
        response = await KE.progressing(srv.router.create_chat_completion(request, raw), srv.progress)
        return getattr(response, 'status_code', 200), json.loads(json.dumps(response.content))

    async def episode(self, srv, k, turns=4, *, keep=False, pad=0, sizes=None, max_tokens=None):
        """A client that sends each reply's content back (and a reasoning when ``keep``): the
        first prompt padded by ``pad`` characters, observation j of ``sizes(j)`` lines."""
        messages = [dict(role='user', content=T.first_prompt('Task %d%s.' % (k, 'x' * pad), host='h%04d' % k))]
        replies = []
        for j in range(turns):
            status, body = await self.send(srv, messages, None if max_tokens is None else max_tokens(j))
            self.assertEqual(status, 200, body)
            message = body['choices'][0]['message']
            replies.append(body)
            reply = dict(role='assistant', content=message['content'])
            if keep:
                reply['reasoning_content'] = 'whatever the client keeps'
            size = (sizes or (lambda j: j))(j)
            messages = messages + [reply, dict(role='user', content=T.observation(j, host='h%04d' % k, size=size)
                                               + 'y' * ((size + k) % 7))]
        return replies

    async def test_sessions_commit_at_the_cut(self):
        """Three clients decode together on observation_query (selection active from the second
        turn), one sending a reasoning back (dropped): every completion commits at the cut,
        every prompt is the stock render, every decision the dense one."""
        srv = strip_server(self, 'observation_query', slots=8, limit=4)
        await asyncio.gather(*(self.episode(srv, k, turns=5, keep=k == 1, pad=k) for k in range(3)))
        rows = srv.rows()
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(len(rows), 15)
        self.assertEqual(check_strip(self, srv, completions=15), 15)
        self.assertGreater(srv.engine.recorder.dense, 5)
        self.assertEqual(dense_goal_inputs(self, srv), 15)
        self.assertFalse(any(r['stage_b']['rebuild'] for r in rows))
        active = [r for r in rows if r['query']['selection']['reason'] is None]
        self.assertTrue(active)
        self.assertEqual(srv.engine.recorder.completions, 15)

    async def test_every_tail_geometry_and_cut_offset(self):
        """Sessions whose cuts fall at every offset mod P: the tail page across the cut is none
        (X % P == 0), live (1..P-K-1) or a phase-1 candidate (P-K..P-1)."""
        srv = strip_server(self, 'observation_query', slots=24, limit=8)
        for k in range(8):
            await self.episode(srv, k, turns=4, pad=2 * k, sizes=lambda j, k=k: (j * 3 + k) % 5)
        rows = srv.rows()
        check_strip(self, srv, completions=32)
        offsets = {(r['stage_b']['computed_end']) % P16 for r in rows}
        kinds = {(r['stage_b']['tail_live'], r['stage_b']['tail_candidate'], r['stage_b']['computed_start'] % P16 == 0)
                 for r in rows if r['stage_b']['local_turn']}
        self.assertGreaterEqual(len(offsets), 10)
        self.assertTrue({(True, False, False), (False, True, False)} <= kinds, kinds)

    async def test_local_turn_zero_splits_its_prefill_at_the_cut(self):
        """Fresh sessions of every cut offset mod P, prefilled one Page per step, in vLLM's
        align-mode chunks (the last M multiple, here M = P) and in one chunk: the prefill forward
        that holds X ends there and the next starts there (also when the 19-token header of the
        character tokenizer crosses an M multiple: the next chunk relocates the state at X). The
        default pass runs the align-mode chunks; REMOTE_RA_SLOW_TESTS=1 all three."""
        for chunk in (('page', 'aligned', 'whole') if SLOW else ('aligned',)):
            with self.subTest(chunk=chunk):
                srv = strip_server(self, 'full', slots=20, limit=16)
                srv.chunk = lambda tokens, chunk=chunk: chunk
                residues = set()
                for k in range(P16):
                    await self.episode(srv, k, turns=2, pad=k)
                check_strip(self, srv, completions=2 * P16)
                for entry in srv.served:
                    residues.add((len(entry['tokens']) - len(HEADER)) % P16)
                self.assertEqual(residues, set(range(P16)))

    async def test_cut_replies_and_think_end_inside_the_content(self):
        """A reply cut at its bound inside the reasoning (content '', closed false), then replies
        with a second </think> inside their content: the next prompts are the stock render."""
        import test_chat_text_e2e as CE
        srv = strip_server(self, 'observation_query', slots=4, limit=2)
        replies = await self.episode(srv, 0, turns=3, max_tokens=lambda j: 9 if j == 0 else None)
        self.assertEqual((replies[0]['choices'][0]['finish_reason'], replies[0]['choices'][0]['message']['content']),
                         ('length', ''))
        original = CE.model
        self.enterContext(patch.object(CE, 'model', lambda m: original(m) + ' </think> trailing'))
        await self.episode(srv, 1, turns=3)
        rows = srv.rows()
        self.assertEqual([r['thinking']['closed'] for r in rows], [False] + [True] * 5)
        check_strip(self, srv, completions=6)

    async def test_rebuild_after_a_cut_and_a_cancel_in_phase_2(self):
        import test_ms2_e2e as B2
        srv = strip_server(self, 'observation_query', slots=4, limit=2)
        h = srv.engine
        messages = [dict(role='user', content=T.first_prompt('Rebuild.'))]
        status, body = await self.send(srv, messages)
        messages += [dict(role='assistant', content=body['choices'][0]['message']['content']),
                     dict(role='user', content=T.observation(0))]
        status, body = await self.send(srv, messages)
        # An unwound history (another earlier reply): a rebuild, a local turn 0 split again.
        unwound = messages[:1] + [dict(role='assistant', content='{"rewritten": true}'),
                                  dict(role='user', content=T.observation(1))]
        status, body = await self.send(srv, unwound)
        self.assertEqual(status, 200)
        row = srv.rows()[-1]
        self.assertEqual((row['stage_b']['rebuild'], row['stage_b']['local_turn'], row['thinking']['history']),
                         (True, 0, CUT.REBUILD))
        nxt = unwound + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                         dict(role='user', content=T.observation(2))]
        # A cancel while phase 2 decodes: the session keeps its committed cut and is poisoned;
        # the client's retry rebinds to a fresh slot.
        hold = srv.hold(0, 3)
        receive = asyncio.Queue()
        task = asyncio.create_task(srv.chat_request(nxt, receive=receive))
        await asyncio.wait_for(hold.parked.wait(), 10)
        s = h.adapter.sessions[srv.live.sessions[0].session_id]
        committed = (s.computed_tokens, s.boundary, s.tail)
        receive.put_nowait(dict(type='http.disconnect'))
        await B2.settle(lambda: any(e[0] == 'abort' for e in h.events))
        hold.gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((s.computed_tokens, s.boundary, s.tail, s.poisoned), committed + (True,))
        status, body = await self.send(srv, nxt)
        self.assertEqual((status, srv.rows()[-1]['chat_text']['rebind_reason']), (200, 'failed'))
        check_strip(self, srv, completions=4)


# ---- 4b. test_chat_text_e2e's scenarios and the flags, on a strip server ---------------------------

SCENARIOS = ('test_sixteen_sessions_decode_together', 'test_header_aside_handoff_and_close',
             'test_cancel_retry_rebinds_and_the_failed_slot_closes', 'test_idle_close_eviction_turn_bound_and_lifetime',
             'test_goal_query_sessions_with_summaries_and_release', 'test_a_lone_turn_waits_for_room_instead_of_running_out',
             'test_frontend_from_the_config_and_the_background_reaper')


def strip_chat_case():
    """test_chat_text_e2e.ChatTextEndToEndTests on a strip server (strip_server), every served
    prompt checked against the stock render and every completion at its cut (check_strip). Built
    here, not at module level, so that discovery does not collect the inherited tests."""
    import test_chat_text_e2e as CE

    class StripChat(CE.ChatTextEndToEndTests):
        def server(self, policy='full', slots=8, limit=4, blocks=4000, via_config=False, release=False, goals=False,
                   **record):
            self.enterContext(patch.object(CE, 'reference', e2e_reference))
            return strip_server(self, policy, slots, limit, blocks, via_config, release, goals, **record)

        def check_prompts(self, srv):
            check_strip(self, srv)
            self.strip_checked = getattr(self, 'strip_checked', 0) + 1
    return StripChat


def run_scenarios(test, names, wrap=None):
    """Run test_chat_text_e2e scenarios ``names`` on a strip server (``wrap``: a context manager
    around the run, e.g. a patched engine class); every one must pass."""
    import test_integrate_flags as IF
    with (wrap or contextlib.nullcontext)():
        outcomes = IF.run_tests(strip_chat_case(), names)
    for name, *outcome in outcomes:
        with test.subTest(name):
            test.assertEqual(outcome, [1, [], [], []])


@contextlib.contextmanager
def admission_overlap(permits):
    """--admission-overlap K on test_chat_text_e2e's server, as compose_engine and compose_frontend
    attach it (test_kv_session_e2e's overlap test): the pinned record, the Goal admissions table
    with K slots, the ledger and the lease, the frontend's permits."""
    import functools
    import test_chat_text_e2e as CE
    import test_kv_session_e2e as KE
    from remote_ra_live_v2 import admission_flags as AF, admission_overlap as AO, goal_async as GA
    from remote_ra_live_v2.composition import compose_frontend
    import test_ms2_e2e as B2
    plain_config, plain_engine = CE.config, KE.KvEngine
    attached = []
    add = B2.BatchLLM.registered_add_async

    async def clamped(llm, frame, timeout):
        # The fake keeps vLLM's former 300-s bound (test_kv_session_e2e's overlap test).
        return await add(llm, frame, min(timeout, 300))

    def overlapped(*args, **kwargs):
        return dict(plain_config(*args, **kwargs), admission=AF.record(admission_overlap=permits))

    class Engine(plain_engine):
        def __init__(self, policy, output, server, **kwargs):
            # goal_query parks its Goals (several at once); the other policies admit in line.
            with patch.object(GA, 'attach', functools.partial(GA.attach, overlap=permits)):
                super().__init__(policy, output, server, **kwargs)
            self.lifecycle.lease = AF.timeouts(permits)[1]
            AO.attach_engine(self.core, self.lifecycle, AF.record(admission_overlap=permits))
            attached.append(self)
    with patch.object(CE, 'config', overlapped), patch.object(KE, 'KvEngine', Engine), \
            patch.object(B2.BatchLLM, 'registered_add_async', clamped), \
            patch.object(CE, 'compose_frontend', functools.partial(compose_frontend, admission_overlap=permits)):
        yield attached
    for engine in attached:
        goals = engine._remote_ra_goal_admissions
        if goals is not None:
            for executor in list(getattr(goals, '_threads', {}).values()) + [goals._thread]:
                executor.shutdown(wait=True, cancel_futures=True)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ScenarioTests(unittest.TestCase):
    """test_chat_text_e2e's scenarios on a strip server: sessions decoding together, the summary
    hand-off (a rebuild) and its aside, a cancel and the rebinds, idle closes and evictions,
    goal_query Terminus 2 clients with --release-idle-pages under pressure, the capacity gate's
    CAPACITY_WAIT, the frontend from the config. Then with the three decode fast paths, with
    --deferred-completion-io, and with --admission-overlap."""
    maxDiff = None

    def test_scenarios(self):
        names = SCENARIOS if SLOW else ('test_header_aside_handoff_and_close',
                                        'test_goal_query_sessions_with_summaries_and_release',
                                        'test_a_lone_turn_waits_for_room_instead_of_running_out')
        run_scenarios(self, names)

    @unittest.skipUnless(SLOW, 'about a minute more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_scenarios_with_the_decode_fast_paths(self):
        import test_integrate_flags as IF
        for fast in (1, 2, 3):
            with self.subTest(fast=fast):
                with IF.kv_engines(fast) as engines:
                    run_scenarios(self, ('test_sixteen_sessions_decode_together', 'test_header_aside_handoff_and_close'))
                self.assertTrue(engines and all(e.adapter.commit_at_cut for e in engines))
                if fast > 1:
                    self.assertTrue(all('allocation_batch' in e.operations for e in engines))

    def test_three_decode_fast_paths(self):
        import test_integrate_flags as IF
        with IF.kv_engines(3) as engines:
            run_scenarios(self, ('test_header_aside_handoff_and_close',))
        self.assertTrue(engines and all(e.adapter.commit_at_cut and e.bridge.decode_fastpath3 for e in engines))
        self.assertTrue(all('allocation_batch' in e.operations and 'allocation_update' not in e.operations
                            for e in engines))
        self.assertGreater(min(min(e.fast3.values()) for e in engines), 10)

    def test_deferred_completion_io(self):
        import test_deferred_completion as TD
        import test_kv_session_e2e as KE
        import test_integrate_flags as IF
        engines = []
        root = Path(tempfile.mkdtemp(dir=os.environ.get('TMP')))
        try:
            Engine = TD.batch_engine(KE.KvEngine, engines, shadow_root=root)
            with patch.object(KE, 'KvEngine', Engine):
                outcomes = IF.run_tests(strip_chat_case(), ['test_header_aside_handoff_and_close'])
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)
        self.assertEqual([list(o[1:]) for o in outcomes], [[1, [], [], []]])
        # Every native file the writer wrote equals the synchronous path's, truncate included.
        self.assertTrue(engines)
        for engine in engines:
            self.assertEqual((engine.writer.mismatches, engine.writer.stats()['errors']), ([], 0))
        self.assertGreater(sum(e.writer.compared for e in engines), 4)

    def test_admission_overlap(self):
        """--admission-overlap 4: four observation_query sessions admitted while the others decode
        (the one phase-1 lease at a time holds a phase 1 longer by the re-prefilled reply); in the
        slow pass also goal_query Terminus 2 clients with their Goals parked on Goal threads, several
        at once."""
        case = strip_chat_case()

        class Overlap(case):
            async def test_overlap(self):
                srv = self.server('observation_query', slots=8, limit=4, blocks=6000)
                agents = [self.agent(srv, k, short=True) for k in range(4)]
                await asyncio.gather(*(self.play(a, 4) for a in agents))
                self.assertFalse(srv.rows('failures.jsonl'))
                self.assertEqual(check_strip(self, srv, completions=16), 16)
                self.assertGreater(srv.engine.recorder.dense, 4)
                self.assertTrue(srv.engine.endpoint.phase1.overlap)
                self.quiescent(srv)
        import test_integrate_flags as IF
        with admission_overlap(4) as engines:
            outcomes = IF.run_tests(Overlap, ['test_overlap'])
        self.assertEqual([list(o[1:]) for o in outcomes], [[1, [], [], []]])
        if SLOW:
            with admission_overlap(4) as more:
                run_scenarios(self, ('test_goal_query_sessions_with_summaries_and_release',))
            engines += more
            from remote_ra_live_v2 import goal_async as GA
            self.assertIsInstance(engines[1]._remote_ra_goal_admissions, GA.OverlapGoalAdmissions)
        self.assertEqual(len(engines), 2 if SLOW else 1)
        self.assertTrue(all(e.endpoint.phase1.overlap and e.adapter.commit_at_cut for e in engines))


# ---- 4c. the decoupled layout: (P, M) = (784, 784) legacy and decoupled, (16, 784) -------------------

def decoupled_harness(case):
    """test_decouple_e2e's single-session harness at (P, M) serving the chat-text protocol with the
    strip history: the strip transcripts on both ends (engine and frontend), the chat-text source,
    the prefill in vLLM's align-mode chunks (each to the next M multiple), then the split hook; the
    worker compacts the Page vectors (decoupled) and copies Pages through the batched IO."""
    import base64
    import test_decouple_e2e as DE
    import test_stageb_e2e as S
    from contracts import canonical_prompt_hash
    from runtime_renderer import CHAT_SOURCE_SCHEMA

    class Renderer(S.Renderer):
        def chat_transcripts(self, record, thinking=False, strip=False):
            return e2e_transcripts(record, thinking, strip)

    class Harness(DE.Harness):
        def __init__(self, policy, output, case):
            with patch.object(S, 'Renderer', Renderer):
                super().__init__(policy, output, case)
            attach_strip(self)
            self.frontend = e2e_transcripts(C.record(thinking=True, strip=True))()
            self.messages = [dict(role='user', content=T.first_prompt('Decoupled strip task.'))]
            self.ordinal = 0

        def submit(self, messages):
            tokens, stable, span = self.frontend.render(messages)
            rid = 'r%d-%s' % (self.ordinal, self.policy)
            self.natives[rid.encode()] = NS(request_id=rid, external_req_id='x' + rid,
                prompt_token_ids=list(tokens), sampling_params=NS(n=1), client_index=0, current_wave=0,
                data_parallel_rank=None, mm_features=None, prompt_embeds=None, prompt_is_token_ids=None,
                lora_request=None, pooling_params=None)
            body = dict(request=rid, session='s', turn=self.ordinal, generation='g%d' % self.ordinal,
                        core=base64.b64encode(rid.encode()).decode(), external_id='x' + rid,
                        prompt_hash=canonical_prompt_hash(tokens),
                        source=dict(schema=CHAT_SOURCE_SCHEMA, messages=[dict(m) for m in messages]))
            wire = json.dumps(dict(model='m', messages=messages)).encode()

            class Raw:
                async def body(self):
                    return wire
            asyncio.run(self.evidence.begin(Raw(), NS(_remote_ra_live_four_policy=NS(session=self.session))))
            info = dict(rid=rid, tokens=tokens, stable=stable, span=span, body=body,
                        messages=[dict(m) for m in messages], mark=self.mark(), namespace=self.session.namespace)
            self.ordinal += 1
            info['view'] = self.lifecycle.admit(body)
            claim = self.endpoint.claims[rid]
            info.update(request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                        namespace=self.session.namespace, ident=(rid, 's', body['turn'], body['generation']))
            return info

        def generate(self, info, outputs, chunk='mamba', stop_after=None):
            request, N, M = info['request'], len(info['request'].prompt_token_ids), self.case.M
            self.runner.samples[request.request_id] = iter(outputs)
            bridge = self.core._remote_ra_live_v2_bridge
            while request.num_computed_tokens < N:
                start = request.num_computed_tokens
                count = bridge.scheduler_split(request, start, min(N, (start // M + 1) * M) - start)
                self.step(request, count, start == 0 or start == info['geo'].phase2_start)
            while len(request.output_token_ids) < len(outputs):
                self.step(request, 1, False)

        def complete(self, info):
            rid = info['rid']
            self.lifecycle.completed(rid)
            self.free(info['request'])
            info['committed'] = self.frontend.commit(info['tokens'], list(info['request'].output_token_ids))
            self.evidence.complete(NS(internal_id=rid, engine_complete=True), NS(id='resp-' + rid))
            info.update(native=self.native_file(rid),
                        row=json.loads((self.output / 'runtime.jsonl').read_bytes().splitlines()[-1]))
            return info

        def turn(self, messages, reply):
            info = self.submit(messages)
            outputs = X.TOK.encode(reply) + [X.EOS]
            self.generate(info, outputs)
            self.complete(info)
            return info
    return Harness


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class DecoupledTests(unittest.TestCase):
    """The strip commit on test_decouple_e2e's (P, M) matrix: legacy 784, decoupled 784 and the
    16-token Pages with 784-token Mamba blocks (compact Page vectors, the Page arena's batched
    copies). Prompts whose cut falls before and after the last M multiple, and a rebuild, so the
    local turn 0 prefill ends a forward at X in each relation to the M blocks; the runner checks
    every forward's GDN chain and KV against the dense state of its prompt."""
    maxDiff = None

    def run_case(self, case, policy, target):
        """Seven turns, the first prompt padded so that its N is ``target`` mod M."""
        import test_decouple_e2e as DE
        with contextlib.ExitStack() as stack:
            DE.enter_case(stack, case)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = DE.config(policy, output, case)
            value.update(protocol=C.PROTOCOL, chat_text=C.record(slots=1, thinking=True, strip=True), thinking=STRIP)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = decoupled_harness(case)(policy, output, case)
            P, M = case.P, case.M
            first = lambda pad: [dict(role='user', content=T.first_prompt('Decoupled strip task.' + 'x' * pad))]
            pad = (target - len(e2e_reference(first(0)))) % M
            infos, messages = [], first(pad)
            relations = []
            for j in range(7):
                if j == 5:
                    # A rebuild: another earlier reply; its local turn 0 splits again.
                    messages = messages[:1] + [dict(role='assistant', content='{"rewritten": %d}' % j),
                                               dict(role='user', content=T.observation(j, size=3))]
                reply = raw_reply('Reasoning %d {"draft": "x"}' % j, T.action(j))
                info = h.turn([dict(m) for m in messages], reply)
                infos.append(info)
                tokens, native = list(info['tokens']), info['native']
                N = len(tokens)
                X_ = N - len(HEADER)
                self.assertEqual(tokens, e2e_reference(messages), j)
                self.assertEqual((native['plan']['commit_end'], native['truncate']['cut']), (X_, X_))
                done = native['worker_frames'][-1]
                self.assertEqual((done['carry']['computed_end'], done['sealed_end']), (X_, X_ // P * P))
                self.assertEqual(sorted(int(p) for p in done['page_vectors']),
                                 list(closed_pages(native['plan']['phase2_start'],
                                                   max(native['plan']['phase2_start'], X_), P)))
                if not info['geo'].local_turn:
                    self.assertIn((info['rid'], X_), h.recorder.splits)
                    self.assertIn(X_, [s + c for step in h.runner.steps for r, s, c in step if r == info['rid']])
                    relations.append('before the last M multiple' if X_ < N - N % M else 'in the last M block')
                messages = messages + [dict(role='assistant', content=info['committed']['content']),
                                       dict(role='user', content=T.observation(j, size=10 * (j + 1) + 3 * j))]
            self.assertEqual(h.recorder.final, 0)
            self.assertEqual(h.recorder.mismatches, [])
            self.assertEqual([i['geo'].local_turn for i in infos], [0, 1, 2, 3, 4, 0, 1])
            h.close()
            return relations

    def test_every_case(self):
        import test_decouple_e2e as DE
        cases = DE.CASES if SLOW else DE.CASES[::2]
        for case in cases:
            # (goal_query needs Terminus 2's exact prompts: test_chat_text_e2e's scenario in ScenarioTests.)
            for policy in (('full', 'observation_query') if SLOW else ('observation_query',)):
                # N mod M below the header's length: X precedes the last M multiple (the split chunk,
                # then vLLM's own cut at that multiple); above it: X in the last M block.
                for target, relation in ((10, 'before the last M multiple'), (400, 'in the last M block')):
                    with self.subTest(case=case.name, policy=policy, target=target):
                        self.assertEqual(self.run_case(case, policy, target)[0], relation)


# ---- 5. mutations -----------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MutationTests(unittest.IsolatedAsyncioTestCase):
    """Each documented mistake makes the strip checks fail (a request refused or failed closed,
    or check_strip's comparison with the stock render and the dense state)."""
    maxDiff = None

    async def asyncSetUp(self):
        import test_ms_e2e as E2E
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def outcome(self, mutate=None, policy='observation_query', turns=3, rebuild=False):
        """One client of ``turns`` turns on a strip server mutated by ``mutate(srv)``: 'ok', or the
        first failure (an HTTP status other than 200 with its message, or check_strip's)."""
        srv = strip_server(self, policy, slots=6, limit=2)
        if mutate is not None:
            mutate(srv)
        messages = [dict(role='user', content=T.first_prompt('Mutation.'))]
        send = EndToEndTests.send
        for j in range(turns):
            status, body = await send(self, srv, messages)
            if status != 200:
                return 'status %d: %s' % (status, body['error']['message'])
            messages = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                                   dict(role='user', content=T.observation(j, size=j))]
            if rebuild and j == 1:
                messages[2] = dict(role='assistant', content='{"unwound": 1}')
        try:
            check_strip(self, srv)
        except AssertionError as failure:
            return 'check: %s' % str(failure)[:300]
        return 'ok'

    async def test_unmutated_passes(self):
        self.assertEqual(await self.outcome(), 'ok')
        self.assertEqual(await self.outcome(rebuild=True, turns=4), 'ok')

    async def test_the_old_carry_relabelled_at_the_cut(self):
        def mutate(srv):
            backend = srv.engine.backend

            def hold(identity, end, generation):
                old = backend.carry[identity.session]
                return backend.hold_carry(identity, end, 'phase1', old.payload, old.digest, generation)
            backend.hold_phase1_carry = hold
        found = await self.outcome(mutate)
        self.assertIn('GDN discontinuity', found)

    async def test_phase2_vectors_published(self):
        def mutate(srv):
            qwen = srv.engine.qwen
            after = qwen.after_forward

            def uncut(*args, **kwargs):
                qwen.commit_at_cut = False
                try:
                    return after(*args, **kwargs)
                finally:
                    qwen.commit_at_cut = True
            qwen.after_forward = uncut
        self.assertIn('phase-2 Page publication incomplete', await self.outcome(mutate))

    async def test_the_engine_forgets_the_cut(self):
        def mutate(srv):
            srv.engine.adapter.commit_at_cut = False
        self.assertIn('reservation plan differs', await self.outcome(mutate))

    async def test_no_split_at_local_turn_zero(self):
        def mutate(srv):
            srv.engine.core._remote_ra_live_v2_bridge.commit_at_cut = False
        found = await self.outcome(mutate)
        self.assertIn('positioned carry', found)

    async def test_a_stable_end_at_the_cut(self):
        def mutate(srv):
            for transcript in list(srv.transcripts) + list(srv.engine.endpoint.transcripts.values()):
                transcript.reply_head = ()
        found = await self.outcome(mutate)
        # The turn geometry refuses it (worker_payload). Until review-fixes-1110 the refusal's cleanup
        # failed too ('partial native admission and cleanup both failed', runtime review 1.4).
        self.assertTrue(found.startswith('status 500') and 'OBSERVATION_GEOMETRY' in found
                        and 'cleanup' not in found, found)

    async def test_the_template_keeps_the_reasoning_block(self):
        def keep(messages):
            return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, C.validate_messages(messages), None, False,
                                                **TH.TEMPLATE_KWARGS[TH.CHAT]))

        def mutate(srv):
            for transcript in list(srv.transcripts) + list(srv.engine.endpoint.transcripts.values()):
                transcript.render_chat = keep
        found = await self.outcome(mutate, rebuild=True, turns=4)
        self.assertTrue(found.startswith('check:'), found)

    async def test_a_tail_at_the_reply_end(self):
        """The worker's tail at C' (the cut's tail dropped): the engine refuses the receipt."""
        def mutate(srv):
            owner = srv.engine.owner
            complete = owner._complete

            def late(rec, payload):
                command = complete(rec, payload)

                def run(backend, qwen):
                    result = command(backend, qwen)
                    if result.get('tail') is not None:
                        result['tail'] = dict(result['tail'], valid_end=payload['computed_end'])
                    return result
                return run
            owner._complete = late
        self.assertIn('worker tail differs', await self.outcome(mutate, turns=4))


# ---- 6. the ATIF replay (opt-in) -----------------------------------------------------------------------

ATIF = os.environ.get('REMOTE_RA_TB_ATIF_DIR')


class StripKit:
    """A ChatKit on strip transcripts whose reference is the stock render (every reasoning dropped)."""

    def __init__(self, kit):
        self.kit, self.name, self.stop_ids = kit, kit.name, kit.stop_ids

    def __getattr__(self, name):
        return getattr(self.kit, name)

    def transcript(self):
        return strip_transcript(self.kit)

    def reference(self, messages):
        return reference(self.kit, messages)


class StripProtocolServer(T.ProtocolServer):
    """test_chat_text_protocol's protocol-level server whose replies are thinking-on (reasoning,
    ``</think>``, the action) and whose response content is C, as the strip route returns it."""

    def send(self, messages, header, max_tokens):
        model = self.model
        self.model = lambda m, slot: raw_reply('I check the screen {"draft": %d} first.' % len(m), model(m, slot))
        try:
            status, body = super().send(messages, header, max_tokens)
        finally:
            self.model = model
        if status == 200:
            committed = self.transcripts[self.log[-1]['slot']].turns[-1]
            body['choices'][0]['message']['content'] = committed['content']
        return status, body


@unittest.skipUnless(ATIF, 'set REMOTE_RA_TB_ATIF_DIR to the copied ATIF job directories (not in the repository)')
class AtifReplayTests(unittest.TestCase):
    """test_chat_text_replay's 178 Terminal-Bench 2.0 Terminus 2 trials on a strip server with the
    pinned tokenizer, reasoning inserted into every reply: every served prompt equals the stock
    render of its messages, the replies cut at their bound included (their content is empty)."""

    def test_every_trial(self):
        import test_chat_text_replay as RP
        kit = T.real_kit()
        if kit is None:
            self.skipTest('the replay needs the pinned tokenizer (Transformers, remote-ra-tok venv)')
        paths = RP.trajectories(ATIF)
        self.assertTrue(paths)
        total = dict(trials=0, requests=0, rebuilds=0, cut=0, mismatches=0)
        with patch.object(T, 'ProtocolServer', StripProtocolServer):
            for path in paths:
                with self.subTest(trial=str(path.parent.parent.relative_to(ATIF))):
                    counts, server, agent = RP.replay(StripKit(kit), path)
                    for entry in server.log:
                        self.assertEqual(entry['prompt'], reference(kit, entry['messages']))
                        if not entry['rebuild'] and entry['binding'] is None:
                            transcript = server.transcripts[entry['slot']]
                            self.assertIsNotNone(transcript)
                    total['trials'] += 1
                    for name in ('requests', 'rebuilds', 'cut', 'mismatches'):
                        total[name] += counts[name]
        self.assertEqual(total['mismatches'], 0)
        print('ATIF strip replay totals:', total)


if __name__ == '__main__':
    unittest.main()
