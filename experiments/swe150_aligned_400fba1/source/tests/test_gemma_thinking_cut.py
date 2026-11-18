"""--thinking-history strip on the Gemma 4 profile, the transcript side (gemma-thinking Tier 2). CPU only;
not GPU evidence. Design: docs/thinking-gemma/DESIGN.md sections 3.3, 4.1 and 4.5.

- T1, the reply piece: for random contents (inline channels, whitespace, Unicode, an empty one) the
  stock render (the template with enable_thinking=True) of a past reply is ``<|turn>model\\n`` +
  strip_thinking(C) + ``<turn|>\\n``, which encodes alone as inside the whole render, and its head is
  the whole header: K = |H| (3 on the pinned tokenizer);
- the transcript: each completion commits the stream at X = N - |H|, the next prompt re-prefills the
  reply the client sent back as the template renders it, its stable end X + K and its observation span
  as on Qwen; the head is compared and a mismatch refused; another history rebuilds from text; the
  reply's split (vLLM's gemma4 rules) is what the route returns, the reasoning never reaches a prompt;
  the evidence (the row's cut and re-prefill, the dropped branch's ``<channel|>`` index);
- T2, Terminus-shaped sessions through Gemma's strip turns with synthetic reasoning (first prompts,
  observations, the proactive summary hand-off and its aside, a reply cut at its bound, context
  refusals): every served prompt equals the stock render of its messages, so no 4-token channel
  difference remains; opt-in (REMOTE_RA_TB_ATIF_DIR and the pinned tokenizer), the 178 Terminal-Bench
  trials of test_gemma_chat_text_replay with synthetic reasoning in every reply (counts only);
- the pins and refusals: the record (history strip, ``enable_thinking=True``), the launcher (the
  records and the window layout's hold slots, --window-hold-slots H or the most that fit, in vLLM's
  --additional-config and nothing else; --async-scheduling, mswea-xml and kv-session refused; H = 0
  refused), the startup check (a Gemma strip config needs hold slots), the dry run's ``window_hold``
  and the READY record, and thinking_cut's attachments, which hand the window layout to the held cut
  window (window_hold; end to end in test_gemma_hold and test_gemma_strip_e2e).

All reasoning and content is synthetic.
"""
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
import test_chat_text_protocol as T  # noqa: E402  Terminus client, prompts, the protocol-level server
import test_gemma_thinking_on as GTO  # noqa: E402  the Gemma kits, the renderer, the vLLM parser model

from remote_ra_http import mswea_route  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2 import thinking_cut as CUT  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from remote_ra_live_v2.mswea_protocol import strip_thinking  # noqa: E402

GEMMA = MP.GEMMA4
HEADER, THOUGHT = GTO.HEADER, GTO.THOUGHT
STOPS = MP.GEMMA4_STOP_IDS
STRIP = TH.record(TH.CHAT, 'strip', profile=GEMMA)
kits, raw = GTO.kits, GTO.raw


def strip_transcript(kit):
    record = C.record(profile=GEMMA, thinking=True, strip=True)
    return GTO.renderer_of(kit).chat_transcripts(record, thinking=True, strip=True)()


def random_contents(seed, count=60):
    rng = random.Random(seed)
    pieces = ('<|channel>thought\nR<channel|>', '<|channel>', '<channel|>', ' ', '\n', '\t', '{"a": 1}', 'x',
              '中文 😀', 'thought\n', '<|channel>thought\nlong reasoning\n', '  {"keystrokes": "ls\\n"}  ', '\r\n')
    return [''] + [''.join(rng.choice(pieces) for _ in range(rng.randrange(1, 8))) for _ in range(count)]


# ---- T1: the reply piece -------------------------------------------------------------------------------

class TemplateTests(unittest.TestCase):
    maxDiff = None

    def test_the_reply_piece_is_the_stock_render(self):
        g = C.GEMMA4_CHAT
        for kit in kits():
            header = kit.encode(HEADER)
            head = kit.encode(g.reply_head)
            self.assertEqual(head, header)                     # K = |H|: the reply head is the whole header
            if kit.name == 'gemma4':
                self.assertEqual(header, [105, 4368, 107])
            for content in random_contents(7, 60 if kit.name == 'gemma4' else 200):
                with self.subTest(kit=kit.name, content=content):
                    piece = g.reply_piece(content)
                    self.assertEqual(piece, HEADER + strip_thinking(content) + '<turn|>\n')
                    messages = [dict(role='user', content=' u '), dict(role='assistant', content=content),
                                dict(role='user', content='v\n')]
                    before = TH.GEMMA4_THINK_SYSTEM + g.user_piece(' u ')
                    after = g.user_piece('v\n') + HEADER
                    self.assertEqual(kit.render(messages, True), before + piece + after)
                    ids = kit.encode(piece)
                    self.assertEqual(kit.stock(messages), kit.encode(before) + ids + kit.encode(after))
                    self.assertTrue(len(ids) > len(head) and ids[:len(head)] == head)
        self.assertEqual(CUT.gemma4_strip_content(' a<|channel>thought\nr<channel|>b \n'), 'ab')
        self.assertEqual(CUT.gemma4_reply_piece(' c '), '<|turn>model\nc<turn|>\n')
        # Qwen's pieces are unchanged.
        self.assertEqual((C.QWEN_CHAT.reply_head, C.QWEN_CHAT.reply_piece(' x </think>\n\ny ')),
                         ('<|im_start|>assistant\n', '<|im_start|>assistant\ny<|im_end|>\n'))

    def test_template_check(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                render = lambda messages, generation: kit.render(messages, generation)
                self.assertEqual(C.GEMMA4_CHAT.check_template_strip(render), HEADER)
                with self.assertRaisesRegex(LiveV2Error, 'thinking-on lone user message'):
                    C.GEMMA4_CHAT.check_template_strip(lambda m, g: kit.render(m, g, enable_thinking=False))
        kit = GTO.char_kit()
        # A template that kept the channels of a past reply would not be the strip history.
        keeps = lambda m, g: kit.render(m, g).replace('<|turn>model\nxz<turn|>',
                                                      '<|turn>model\nx<|channel>thought\ny<channel|>z<turn|>')
        with self.assertRaisesRegex(LiveV2Error, 'stock thinking-on history'):
            C.GEMMA4_CHAT.check_template_strip(keeps)
        renderer = GTO.renderer_of(kit)
        with self.assertRaisesRegex(RuntimeError, 'differ from the startup record'):
            renderer.chat_transcripts(C.record(profile=GEMMA), thinking=True, strip=True)
        # On Gemma the keep and strip chat_text records are the same (header and arguments); the thinking
        # record's history tells them apart.
        self.assertEqual(C.record(profile=GEMMA, thinking=True), C.record(profile=GEMMA, thinking=True, strip=True))
        t = strip_transcript(kit)
        self.assertTrue(t.strip and t.thinking and t.grammar is C.GEMMA4_CHAT)
        self.assertEqual((t.header, t.reply_head), (tuple(kit.encode(HEADER)), tuple(kit.encode(HEADER))))
        self.assertEqual(t.think_end_ids, tuple(kit.encode('<channel|>')))


# ---- the transcript ------------------------------------------------------------------------------------

class TranscriptTests(unittest.TestCase):
    maxDiff = None

    def play(self, kit, *, turns=5, sent_back=None, echo=None, replies=None, bound=None, system=False):
        """One client session on a strip transcript (as test_thinking_cut.TranscriptTests.play)."""
        ours = strip_transcript(kit)
        messages = ([dict(role='system', content='You are an agent.')] if system else []) + [
            dict(role='user', content=T.first_prompt('Strip task.'))]
        served = []
        for k in range(turns):
            before = ours.stream
            tokens, stable, span = ours.render([dict(m) for m in messages])
            text = replies(k) if replies is not None else raw('Turn %d drafts {"x": %d}.' % (k, k), T.action(k))
            generated = kit.encode(text) + [MP.TURN_CLOSE]
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
        header = kit.encode(HEADER)
        K = len(header)
        for k, (messages, tokens, stable, span, committed, before) in enumerate(served):
            self.assertEqual(list(tokens), kit.stock(GTO.plain(messages)), k)
            X = len(tokens) - len(header)
            self.assertEqual((committed['cut'], list(tokens[X:])), (X, header))
            if committed['history'] == CUT.CUT:
                piece = kit.encode(C.GEMMA4_CHAT.reply_piece(messages[-2]['content']))
                user = kit.encode(C.GEMMA4_CHAT.user_piece(messages[-1]['content']))
                C0 = len(before)
                self.assertEqual(list(tokens), list(before) + piece + user + header)
                self.assertEqual((stable, span), (C0 + K, (C0 + K, C0 + len(piece) + len(user))))
                self.assertEqual(committed['reprefill'], dict(start=C0, piece_end=C0 + len(piece)))
                self.assertEqual(list(tokens[C0:C0 + K]), header)
            else:
                self.assertIsNone(committed['reprefill'])
        self.assertEqual(list(ours.stream), list(served[-1][1][:served[-1][4]['cut']]))
        if histories is not None:
            self.assertEqual([c['history'] for *_, c, _ in served], histories)

    def test_every_prompt_is_the_stock_render(self):
        cases = {'dropped': dict(), 'sent back': dict(sent_back=lambda r: r),
                 'stripped echo': dict(echo=str.strip), 'padded echo': dict(echo=lambda c: '  ' + c + '\n '),
                 'a channel in C': dict(replies=lambda k: raw('R%d' % k, 'C%d <|channel>x<channel|>\n%s'
                                                                % (k, T.action(k)))),
                 'no channel': dict(replies=lambda k: T.action(k)),
                 'never closed': dict(replies=lambda k: THOUGHT + 'reasoning only %d, never closed' % k),
                 'cut at the bound': dict(bound=lambda k: 7 + k),
                 'a system message': dict(system=True)}
        for kit in kits():
            for name, options in cases.items():
                with self.subTest(kit=kit.name, case=name):
                    ours, served = self.play(kit, **options)
                    self.check(kit, ours, served, ['fresh'] + ['cut'] * 4)
                    self.assertEqual(ours.sessions, 1)
                    self.assertEqual(ours.previous_action(), served[-1][4]['content'])

    def test_the_split_of_the_cut_reply(self):
        kit = GTO.char_kit()
        cases = ((raw('R ', ' {"a": 1}'), 'R', ' {"a": 1}', True), (T.action(0), '', T.action(0), True),
                 (THOUGHT + 'cut', 'cut', '', False), ('x<channel|>y', 'x', 'y', True))
        for text, reasoning, content, closed in cases:
            ours = strip_transcript(kit)
            tokens, _, _ = ours.render([dict(role='user', content='u')])
            committed = ours.commit(tokens, kit.encode(text) + [MP.TURN_CLOSE])
            self.assertEqual((committed['reasoning'], committed['content'], committed['closed']),
                             (reasoning, content, closed))
            self.assertEqual(committed['split'], GTO.splitter(kit)(kit.encode(text) + [MP.TURN_CLOSE])['tokens'])
            row = mswea_route.evidence_fields(ours, committed)['thinking']
            self.assertEqual((row['cut'], row['reprefill'], row['history']), (len(tokens) - len(ours.header), None,
                                                                              'fresh'))
            self.assertEqual(sum(v for k, v in row.items() if k.endswith('_tokens')), len(kit.encode(text)) + 1)

    def test_reasoning_never_reaches_the_prompt(self):
        kit = GTO.char_kit()
        a, sa = self.play(kit, turns=3)
        b, sb = self.play(kit, turns=3, sent_back=lambda r: 'something else entirely')
        self.assertEqual([list(x[1]) for x in sa], [list(x[1]) for x in sb])
        self.assertEqual([x[4]['history'] for x in sb], ['fresh', 'cut', 'cut'])
        self.assertFalse(any('drafts' in kit.decode(x[1]) for x in sa))

    def test_another_history_rebuilds_and_a_wrong_head_is_refused(self):
        kit = GTO.char_kit()
        ours, served = self.play(kit, turns=3)
        messages = served[-1][0] + [dict(role='assistant', content=served[-1][4]['content']),
                                    dict(role='user', content='next')]
        changed = [dict(m) for m in messages]
        changed[2] = dict(role='assistant', content='rewritten')
        plan = ours.plan(changed)
        self.assertEqual((plan['rebuild'], plan['history'], plan['reprefill']), (True, CUT.REBUILD, None))
        tokens, stable, span = ours.render(changed)
        self.assertEqual((list(tokens), stable, span), (kit.stock(changed), len(tokens) - len(ours.header), None))
        committed = ours.commit(tokens, kit.encode(raw('r', 'c')) + [MP.TURN_CLOSE])
        self.assertEqual((committed['rebuild'], ours.turn, ours.sessions), (True, 1, 2))
        # A re-encoded reply whose head is not the cut header's (a tokenizer that merged it): refused.
        ours, served = self.play(kit, turns=1)
        nxt = served[-1][0] + [dict(role='assistant', content=served[-1][4]['content']), dict(role='user', content='o')]
        with patch.object(C.Gemma4Chat, 'reply_piece', staticmethod(lambda c: '<|turn>user\n' + c + '<turn|>\n')):
            with self.assertRaises(C.ChatTextError) as caught:
                ours.plan(nxt)
        self.assertEqual((caught.exception.kind, caught.exception.status), ('STRIP_REPLY_HEAD', 500))
        self.assertEqual(ours.plan(nxt)['history'], CUT.CUT)

    def test_the_header_check_allows_k_equal_to_the_header_on_gemma_only(self):
        kit = GTO.char_kit()
        make = lambda grammar, head: C.ChatTextTranscript(encode=kit.encode, render_chat=lambda m: [1],
                                                          decode=kit.decode, stop_ids=STOPS, grammar=grammar,
                                                          thinking=True, strip=True)
        self.assertEqual(len(make(C.GEMMA4_CHAT, None).reply_head), len(kit.encode(HEADER)))
        # Qwen keeps K < |H| (its reply head is the header's first 3 of 5 tokens).
        with patch.object(C.QwenChat, 'reply_head', C.QwenChat.thinking_header):
            with self.assertRaisesRegex(LiveV2Error, 'must start with the reply head'):
                C.ChatTextTranscript(encode=kit.encode, render_chat=lambda m: [1], decode=kit.decode,
                                     stop_ids=STOPS, thinking=True, strip=True)
        with patch.object(C.Gemma4Chat, 'reply_head', '<|turn>user\n'):
            with self.assertRaisesRegex(LiveV2Error, 'must start with the reply head'):
                make(C.GEMMA4_CHAT, None)


# ---- T2: Terminus-shaped sessions ----------------------------------------------------------------------

class StripKit:
    """A Gemma kit on strip transcripts for test_chat_text_protocol's ProtocolServer and Terminus client:
    replies end with <turn|>, the reference is the stock thinking-on render."""

    def __init__(self, kit):
        self.kit, self.name, self.stop_ids = kit, kit.name, STOPS

    def encode(self, text):
        return self.kit.encode(text)

    def decode(self, ids):
        return self.kit.decode(ids)

    def transcript(self):
        return strip_transcript(self.kit)

    def reply(self, text):
        return self.kit.encode(text) + [MP.TURN_CLOSE]

    def reference(self, messages):
        return self.kit.stock(GTO.plain(messages))


class StripProtocolServer(T.ProtocolServer):
    """The protocol-level server with thinking-on replies (a thought channel holding a draft action and
    JSON, then the action) whose response content is C, as the strip route returns it."""

    def send(self, messages, header, max_tokens):
        model = self.model
        self.model = lambda m, slot: raw('I check the screen {"draft": %d} first.' % len(m), model(m, slot))
        try:
            status, body = super().send(messages, header, max_tokens)
        finally:
            self.model = model
        if status == 200:
            committed = self.transcripts[self.log[-1]['slot']].turns[-1]
            body['choices'][0]['message']['content'] = committed['content']
        return status, body


class TerminusTests(unittest.TestCase):
    maxDiff = None

    def check(self, kit, server):
        for entry in server.log:
            self.assertEqual(entry['prompt'], kit.reference(entry['messages']))
            if any('<|channel>' in m['content'] for m in entry['messages']):
                self.fail('a channel reached the messages')

    def test_plain_episode(self):
        for kit in [StripKit(k) for k in kits()]:
            with self.subTest(kit=kit.name):
                server = StripProtocolServer(kit)
                agent = T.Terminus('List the files.', send=server.send, count=lambda m: 0)
                prompt = T.first_prompt('List the files.')
                for k in range(6):
                    T.run(agent.step(prompt))
                    prompt = T.observation(k, size=k)
                self.assertEqual([e['rebuild'] for e in server.log], [False] * 6)
                self.check(kit, server)
                # Every turn committed at its cut X = N - |H| (3 on the pinned tokenizer).
                transcript = server.transcripts[0]
                self.assertEqual([t['cut'] for t in transcript.turns],
                                 [len(e['prompt']) - len(kit.encode(HEADER)) for e in server.log])
                self.assertEqual(list(transcript.stream), server.log[-1]['prompt'][:transcript.turns[-1]['cut']])

    def test_summary_handoff_cut_replies_and_context_refusals(self):
        kit = StripKit(GTO.char_kit())
        server = StripProtocolServer(kit)
        tokens = lambda messages: len(kit.encode(''.join(m['content'] for m in messages)))
        agent = T.Terminus('Build it.', send=server.send, count=tokens, context_limit=16000, threshold=8000)
        prompt = T.first_prompt('Build it.')
        for k in range(10):
            T.run(agent.step(prompt))
            prompt = T.observation(k, size=20)
        self.assertIn('summary', agent.events)
        self.assertTrue(any(e['rebuild'] for e in server.log))
        self.check(kit, server)
        # A reply cut at its bound (inside its channel: content "") and context refusals.
        limit = 6000
        server = StripProtocolServer(kit, limit=limit, bound=200)
        agent = T.Terminus('Grow.', send=server.send, max_tokens=None, count=tokens, context_limit=limit,
                           threshold=0)
        longer = {'n': 0}

        def model(messages, slot):
            longer['n'] += 1
            return T.action(len(messages)) * (60 if longer['n'] == 2 else 1)
        server.model = model
        prompt = T.first_prompt('Grow.')
        for k in range(12):
            T.run(agent.step(prompt))
            prompt = T.observation(k, size=4)
        self.assertIn('output_length', agent.events)
        self.assertTrue(all(len(e['prompt']) <= limit for e in server.log))
        self.check(kit, server)


class Terminus2Tests(unittest.TestCase):
    """Terminus 2.0.0's exact prompts (test_terminus2_goal.Terminus2: the task template, the proactive
    summary hand-off with its three sub-calls, the unwound history) through Gemma's strip turns on every
    kit: with REMOTE_RA_GEMMA4_METADATA and Transformers, the pinned tokenizer and the pinned template.
    Every served prompt equals the stock thinking-on render of its messages (the reference of the engine
    fakes' sessions in test_gemma_strip_e2e, whose template is the stand-in)."""
    maxDiff = None

    def test_sessions_with_the_summary_handoff_on_every_kit(self):
        import test_terminus2_goal as TG
        for kit in [StripKit(k) for k in kits()]:
            with self.subTest(kit=kit.name):
                server = StripProtocolServer(kit)
                chars = lambda messages: sum(len(m['content']) for m in messages)
                agents = []
                for k in range(3):
                    instruction = 'Task %d: build /app/job%d.py and make ./run %d print ok.' % (k, k, k)
                    agent = TG.Terminus2(instruction, send=server.send, count=chars, context_limit=4200,
                                         threshold=600, header='t%02d__agent' % k if k % 2 else None)
                    agent.first = TG.first_prompt(instruction, 'root@h%04d:/app# \n' % k)
                    agents.append(agent)
                for j in range(6):
                    for k, agent in enumerate(agents):
                        T.run(agent.step(agent.first if j == 0 else TG.new_output(
                            'root@h%04d:/app# ./run %d\nok %d\nroot@h%04d:/app# ' % (k, j, j, k))))
                self.assertTrue(all('summary' in agent.events for agent in agents))
                self.assertTrue(all(status == 200 for agent in agents for _, status, _ in agent.requests))
                self.assertEqual(len(server.log), sum(len(agent.requests) for agent in agents))
                header = kit.encode(HEADER)
                rebuilds = 0
                for entry in server.log:
                    self.assertEqual(entry['prompt'], kit.reference(entry['messages']))
                    self.assertEqual(entry['prompt'][-len(header):], header)
                    self.assertNotIn(MP.CHANNEL_OPEN, entry['prompt'])
                    self.assertNotIn(MP.CHANNEL_CLOSE, entry['prompt'])
                    rebuilds += bool(entry['rebuild'])
                self.assertGreater(rebuilds, 0)                 # the unwound history after a hand-off
                if kit.name == 'gemma4':
                    self.assertEqual(header, [105, 4368, 107])
                # Every transcript's stream is its last prompt up to the cut X = N - |H|.
                for transcript in server.transcripts:
                    if transcript.turns:
                        last = [e for e in server.log if e['slot'] == server.transcripts.index(transcript)][-1]
                        self.assertEqual(list(transcript.stream), last['prompt'][:len(last['prompt']) - len(header)])


ATIF = os.environ.get('REMOTE_RA_TB_ATIF_DIR')


@unittest.skipUnless(ATIF, 'set REMOTE_RA_TB_ATIF_DIR to the copied ATIF job directories (not in the repository)')
class AtifReplayTests(unittest.TestCase):
    """test_gemma_chat_text_replay's Terminal-Bench 2.0 Terminus 2 trials on a Gemma strip server with the
    pinned tokenizer and template, synthetic reasoning in every reply: every served prompt equals the stock
    thinking-on render of its messages (no channel difference), the replies cut at their bound included.
    Only counts are printed; nothing of the trajectories is kept."""

    def test_every_trial(self):
        import test_chat_text_replay as RP
        real = GTO.real_kit()
        if real is None:
            self.skipTest('the replay needs the pinned Gemma tokenizer (%s, Transformers)' % GTO.G.ENV)
        kit = StripKit(real)
        paths = RP.trajectories(ATIF)
        self.assertTrue(paths)
        total = dict(trials=0, requests=0, rebuilds=0, cut=0, refused=0, mismatches=0)
        with patch.object(T, 'ProtocolServer', StripProtocolServer):
            for path in paths:
                with self.subTest(trial=str(path.parent.parent.relative_to(ATIF))):
                    counts, server, agent = RP.replay(kit, path)
                    for entry in server.log:
                        if entry['prompt'] != kit.reference(entry['messages']):
                            total['mismatches'] += 1
                    total['trials'] += 1
                    for name in ('requests', 'rebuilds', 'cut', 'refused'):
                        total[name] += counts[name]
        self.assertEqual(total['mismatches'], 0)
        print('Gemma ATIF strip replay totals:', total)


# ---- the pins and refusals -----------------------------------------------------------------------------

class PinTests(unittest.TestCase):
    maxDiff = None

    def test_the_record(self):
        self.assertEqual(STRIP, dict(GTO.ON_CHAT, history='strip', template_kwargs={'enable_thinking': True},
                                     append=CUT.RULE))
        self.assertEqual(C.record(profile=GEMMA, thinking=True, strip=True),
                         dict(C.record(profile=GEMMA), header_text=HEADER, template_kwargs={'enable_thinking': True}))
        with self.assertRaisesRegex(LiveV2Error, 'serves the chat-text protocol only'):
            TH.record(TH.GEMMA4_XML, 'strip', profile=GEMMA)

    HELD = I0.to_additional_config(I0.WindowLayout(16, 1024, 4, 8192, hold_slots=2))[I0.LAYOUT_KEY]

    def config(self, **extra):
        value = dict(protocol=C.PROTOCOL, model_profile=GEMMA, thinking=STRIP, kv_layout=dict(self.HELD),
                     chat_text=C.record(profile=GEMMA, thinking=True, strip=True))
        value.update(extra)
        return value

    def test_check_config(self):
        self.assertIs(CUT.check_config(self.config()), True)
        self.assertEqual(TH.history_mode(self.config()), 'strip')
        with self.assertRaisesRegex(LiveV2Error, 'refuses --async-scheduling'):
            CUT.check_config(self.config(async_scheduling=True))
        with self.assertRaisesRegex(LiveV2Error, 'differs from the protocol pins'):
            CUT.check_config(self.config(thinking=TH.record(TH.CHAT, 'strip')))
        with self.assertRaisesRegex(LiveV2Error, 'mswea-xml and chat-text protocols only'):
            CUT.check_config(self.config(protocol='kv_session_v1'))
        self.assertIsNone(CUT.check_config(self.config(thinking=GTO.ON_CHAT)))

    def test_a_gemma_strip_config_needs_hold_slots(self):
        """H = 0 is refused at startup, clearly: no hold slot, no strip reply could decode. The keep mode
        and thinking off need none; Qwen's strip config has no window and needs none."""
        plain = {k: v for k, v in self.HELD.items() if k != I0.HOLD_FIELD}
        for layout in (plain, None, dict(plain, hold_slots=0), dict(plain, hold_slots='2'), dict(plain, hold_slots=True)):
            with self.subTest(layout=layout), self.assertRaisesRegex(
                    LiveV2Error, r'needs hold slots for the sliding window at the cut \(kv_layout.hold_slots >= 1, '
                                 r'serve.py --window-hold-slots\): with 0 hold slots no strip reply could decode'):
                CUT.check_config(self.config(kv_layout=layout))
        self.assertIs(CUT.check_config(self.config(kv_layout=dict(plain, hold_slots=1))), True)
        self.assertIsNone(CUT.check_config(self.config(thinking=GTO.ON_CHAT, kv_layout=plain)))
        qwen = dict(protocol=C.PROTOCOL, thinking=TH.record(TH.CHAT, 'strip'),
                    chat_text=C.record(thinking=True, strip=True))
        self.assertIs(CUT.check_config(qwen), True)

    def test_the_engine_attachment_hands_the_window_layout_to_the_hold(self):
        """The refusal GT1 left for the window layout is gone: thinking_cut.attach_engine attaches the held
        cut window (window_hold) on a window adapter, and refuses one whose layout has no hold slot."""
        import dataclasses
        import test_gemma_engine as GEN
        case = GEN.AdapterLayoutTests('test_window_layout_gets_window_pins')
        adapter = case.adapter(dataclasses.replace(I0.for_window_server(16, 2, 1024, 8192), hold_slots=2))
        bridge = NS()
        CUT.attach_engine(adapter, bridge)
        self.assertEqual((adapter.commit_at_cut, adapter.window.hold_slots, adapter.cut_records), (True, 2, {}))
        self.assertFalse(getattr(bridge, 'commit_at_cut', False))      # no split at local turn 0 on Gemma
        bare = case.adapter(I0.for_window_server(16, 2, 1024, 8192))
        with self.assertRaisesRegex(LiveV2Error, r'needs hold slots in its KV layout \(hold_slots >= 1\)'):
            CUT.attach_engine(bare, NS())
        self.assertFalse(bare.commit_at_cut)
        source = (ROOT / 'source-overlay' / 'remote_ra_live_v2' / 'thinking_cut.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('HOLD-PENDING'), 0)       # the marked refusal is gone
        qwen = NS(window=None)
        CUT.attach_engine(qwen, bridge)
        self.assertTrue(qwen.commit_at_cut and bridge.commit_at_cut)


class LauncherTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        import test_gemma_startup as GS
        self.gs = GS.GemmaStartupTests('test_load_accepts_the_gemma_startup_config')
        self.gs.setUp()
        self.addCleanup(self.gs.doCleanups)

    def chat(self, **kw):
        return self.gs.prepare(**dict(dict(task=None, policy='observation_query',
                                           selection=dict(active_token_budget=8192), protocol='chat-text',
                                           max_running=4), **kw))

    def test_strip_adds_the_records_and_the_hold_slots(self):
        m0, c0, a0, _ = self.chat(enable_thinking=True)
        self.gs.n -= 1
        m1, c1, a1, _ = self.chat(enable_thinking=True, thinking_history='strip')
        self.assertEqual(m1, m0)
        # The window layout gains its hold slots (the most that fit: all four sessions' here), in the
        # config's kv_layout and in vLLM's --additional-config; no other vLLM argument changes.
        layout = dict(c0['kv_layout'], hold_slots=4)
        self.assertEqual(c1['kv_layout'], layout)
        at = a0.index('--additional-config') + 1
        self.assertEqual(json.loads(a1[at]), {I0.LAYOUT_KEY: layout})
        self.assertEqual(a1[:at] + a1[at + 1:], a0[:at] + a0[at + 1:])
        effective = dict(c0['effective_server_config'], additional_config={I0.LAYOUT_KEY: layout})
        self.assertEqual(c1, dict(c0, thinking=STRIP, kv_layout=layout, effective_server_config=effective,
                                  server_config_sha256=serve.pin(effective),
                                  chat_text=C.record(slots=512, profile=GEMMA, thinking=True, strip=True)))
        loaded = self.gs.load(c1)
        self.assertEqual((TH.pinned(loaded), TH.history_mode(loaded), CUT.pinned(loaded)), (STRIP, 'strip', True))
        # --window-hold-slots H pins that H.
        self.gs.n -= 1
        c2 = self.chat(enable_thinking=True, thinking_history='strip', window_hold_slots=1)[1]
        self.assertEqual(c2['kv_layout'], dict(c0['kv_layout'], hold_slots=1))
        self.gs.load(c2)

    def test_the_hold_slot_option(self):
        import test_gemma_startup as GS
        strip = dict(enable_thinking=True, thinking_history='strip')
        for kw, pattern in ((dict(strip, window_hold_slots=0), 'must be 1 to 4 .*with 0 hold slots no reply could decode'),
                            (dict(strip, window_hold_slots=5), 'must be 1 to 4'),
                            (dict(strip, window_hold_slots='2'), 'must be 1 to 4'),
                            (dict(enable_thinking=True, window_hold_slots=2), 'applies to --thinking-history strip only'),
                            (dict(window_hold_slots=2), 'applies to --thinking-history strip only'),
                            (dict(protocol='mswea-xml', task=GS.tasks(4), enable_thinking=True, window_hold_slots=1),
                             'applies to --thinking-history strip only')):
            with self.subTest(kw=sorted(kw)), self.assertRaisesRegex(ValueError, pattern):
                self.chat(**kw)
        # A Qwen release refuses the option (its strip commit holds a GDN state, no window).
        with self.assertRaisesRegex(ValueError, 'applies to the Gemma 4 profile with --thinking-history strip only'):
            self.gs.prepare(GS.tasks(2), 'full', release=self.gs.qwen, kv_cache_gib=8, page_tokens=784,
                            max_tokens=1536, max_model_len=65536, language_model_only=False, window_hold_slots=1)

    def test_dry_run_and_ready_record_the_hold_slots(self):
        """The dry run names H and where it came from (``window_hold``); startup.json pins it in kv_layout
        and ready.json echoes the worker's layout (the same record)."""
        import contextlib
        import io
        import pathlib
        import tempfile
        import test_ms_frontend as MF
        from remote_ra_http.evidence import FrontendEvidence

        def dry(*extra):
            self.gs.n += 1
            argv = ['serve.py', '--release-dir', str(self.gs.release), '--policy', 'observation_query',
                    '--active-token-budget', '8192', '--output', str(self.gs.root / ('dry%d' % self.gs.n)),
                    '--protocol', 'chat-text', '--max-running-sessions', '4', '--kv-cache-gib', '28',
                    '--page-tokens', '16', '--max-model-len', '131072', '--enable-thinking', *extra]
            out = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(out):
                serve.main()
            return json.loads(out.getvalue())
        keep = dry()
        self.assertNotIn('window_hold', keep)
        auto = dry('--thinking-history', 'strip')
        self.assertEqual(auto['window_hold'], dict(hold_slots=4, source='auto: the most that fit'))
        pinned = dry('--thinking-history', 'strip', '--window-hold-slots', '2')
        self.assertEqual(pinned['window_hold'], dict(hold_slots=2, source='--window-hold-slots'))
        at = pinned['command'].index('--additional-config') + 1
        self.assertEqual(json.loads(pinned['command'][at])[I0.LAYOUT_KEY]['hold_slots'], 2)
        self.assertEqual(pinned['thinking'], STRIP)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dry('--thinking-history', 'strip', '--window-hold-slots', 'auto')
        # ready.json: the pinned layout with its hold slots, the thinking record, the worker's echo.
        config = self.chat(enable_thinking=True, thinking_history='strip', window_hold_slots=2)[1]
        loaded = self.gs.load(config)
        readiness = dict(native_ready=True)
        authority = NS(identity=NS(epoch=MF.EPOCH, manifest='m' * 64), policy='observation_query',
                       session_id=MF.SERVER, verify_admission=lambda: True, live_v2_readiness=readiness,
                       loaded=NS(manifest=dict(qualification_parent_manifest='q' * 64)))
        read_text = pathlib.Path.read_text

        def proc(path, *args, **kwargs):
            name = str(path).replace('\\', '/')
            if name == '/proc/sys/kernel/random/boot_id':
                return 'boot\n'
            if name == '/proc/self/stat':
                return '1 (python) ' + ' '.join(['0'] * 30)
            return read_text(path, *args, **kwargs)

        def ready(echo, layout):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                value = dict(loaded, output_directory=str(output))
                pools = dict(kv_layout=layout, num_blocks_by_pool=[10, 20], free_blocks_by_pool=[9, 19])
                receipt = dict(tensor_geometry={}, source_closure={}, **echo)
                (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                    epoch=MF.EPOCH, report=readiness, kv_pools=pools, receipt=receipt)).encode())
                with patch('remote_ra_http.evidence.load', lambda: value), \
                        patch.object(pathlib.Path, 'read_text', proc):
                    evidence = FrontendEvidence(authority, None, dict(loaded['effective_server_config']))
                    evidence.write_ready(NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]),
                                         NS(started=True))
                return json.loads((output / 'ready.json').read_bytes())
        from remote_ra_live_v2.context_limits import RECORD_KEY, record as context_record
        from remote_ra_live_v2.language_model_only import CONFIG_KEY as TEXT_ONLY, record as text_only_record
        echo = {RECORD_KEY: context_record(loaded), TEXT_ONLY: text_only_record(loaded), 'thinking_history': 'strip'}
        echo = {k: v for k, v in echo.items() if v is not None}
        value = ready(echo, loaded['kv_layout'])
        self.assertEqual((value['kv_layout']['hold_slots'], value['thinking']), (2, STRIP))
        self.assertEqual(value['kv_layout'], config['kv_layout'])
        with self.assertRaisesRegex(RuntimeError, 'KV layout mismatch'):
            ready(echo, {k: v for k, v in loaded['kv_layout'].items() if k != 'hold_slots'})
        with self.assertRaisesRegex(RuntimeError, 'thinking_history mismatch'):
            ready({k: v for k, v in echo.items() if k != 'thinking_history'}, loaded['kv_layout'])

    def test_refusals(self):
        import test_gemma_startup as GS
        for kw, pattern in ((dict(thinking_history='strip'), 'requires --enable-thinking'),
                            (dict(enable_thinking=True, thinking_history='strip', async_scheduling=True,
                                  decode_fastpath=True, decode_fastpath2=True), 'refuses --async-scheduling'),
                            (dict(protocol='kv-session', enable_thinking=True, thinking_history='strip', selection=None,
                                  policy='full'), 'mswea-xml and chat-text only'),
                            (dict(protocol='mswea-xml', task=GS.tasks(4), enable_thinking=True,
                                  thinking_history='strip'), 'strip serves --protocol chat-text only')):
            with self.subTest(kw=sorted(kw)), self.assertRaisesRegex(ValueError, pattern):
                self.chat(**kw)
        # --async-scheduling with the keep mode is taken (decision 7 refuses it with strip only).
        config = self.chat(enable_thinking=True, async_scheduling=True, decode_fastpath=True, decode_fastpath2=True)[1]
        self.assertEqual((config['thinking'], config['async_scheduling']), (GTO.ON_CHAT, True))


if __name__ == '__main__':
    unittest.main()
