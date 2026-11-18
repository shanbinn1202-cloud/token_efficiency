"""keep-rebuild-1116: the thinking-on XML history after a format error (CPU only; not GPU evidence).

Before this branch a thinking-on mini-swe-agent session kept every reply's reasoning R only until its first format
error: the client drops the rejected reply and adds a format-error message, which is no append, and the XML
rebuild re-encoded every later message from its content (R gone for the rest of the session). Now:

- the keep render of one later message (thinking.keep_message_text) is the pinned Qwen3.6 template's render with
  preserve_thinking, with the jinja2 stand-in and, when Transformers is installed, the pinned tokenizer;
- a rollback (the previous request + one user message) keeps the session's exact ids up to the dropped reply's
  header and adds the message: a reply cut inside its reasoning, a closed reply without an action, several format
  errors in a row, the session's first reply dropped; every prompt equals the keep render, and the ids before the
  dropped reply's header are the previous prompt's;
- any other rebuild renders the keep render (R where the client sent it), counted in the record;
- thinking off, and the transcript without allow_rebuild, behave as before;
- the record (thinking.evidence): ``history`` ``rollback``, and ``kept_tokens`` / ``reencoded_replies`` on rollback
  and rebuild rows only (fresh and append rows keep their keys);
- the controller's capacity-wait checkpoint restores what a rollback keeps;
- the Gemma 4 grammar: a rollback keeps the exact ids (the reply's thought channel) too;
- the query hook's history after a rollback: every earlier reply with its reasoning and its call start;
- tools/keep_render_check.py: the keep render, the trajectory's requests, the per-call comparison.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import test_chat_text_protocol as T  # noqa: E402
import test_kv_session_protocol as KP  # noqa: E402
import test_thinking_on as TO  # noqa: E402

import keep_render_check as KRC  # noqa: E402
from remote_ra_live_v2 import controller as CTL  # noqa: E402
from remote_ra_live_v2 import mswea_protocol as XML  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from remote_ra_live_v2.query_selection import QuerySelectionAuthority  # noqa: E402
from remote_ra_live_v2 import selection_contract as sc  # noqa: E402

R1, C1, R2, C2 = TO.R1, TO.C1, TO.R2, TO.C2
FORMAT_ERROR = ('Format error:\n\n<error>\nExpected exactly 1 action, found 0.\n</error>\n\n'
                'Please always provide EXACTLY ONE action in triple backticks.')
OBS = ['<returncode>0</returncode>\n<output>\na.py\n</output>', '<returncode>0</returncode>\n<output>\n</output>',
       '<returncode>1</returncode>\n<output>\nboom\n</output>', '<returncode>0</returncode>']
EVIDENCE_KEYS = {'reasoning_tokens', 'think_end_tokens', 'content_tokens', 'closed', 'reasoning_chars',
                 'content_chars', 'history'}


def keep(kit, messages):
    """The keep render of a request (the pinned template, preserve_thinking, R as reasoning_content)."""
    return TO.reference(kit, messages)


class Agent:
    """A mini-swe-agent 2.4.2 session against a transcript: a kept reply is sent back with its content (trimmed)
    and reasoning_content, then the observation; a rejected reply (a format error) is dropped and the format-error
    message added. ``served`` holds (messages, tokens, history, rebuild, committed) per call."""

    def __init__(self, kit, transcript):
        self.kit, self.t = kit, transcript
        self.messages = TO.xml_messages([])
        self.served = []

    def call(self, text, kept=True, observation=None, generated=None):
        tokens, stable, span = self.t.render([dict(m) for m in self.messages])
        pending = dict(self.t.pending)
        committed = self.t.commit(tokens, self.kit.encode(text) if generated is None else generated)
        self.served.append(NS(messages=[dict(m) for m in self.messages], tokens=list(tokens), stable=stable, span=span,
                              history=pending.get('history'), rebuild=pending['rebuild'], committed=committed))
        if kept:
            reply = dict(role='assistant', content=committed['content'].strip() if self.t.thinking
                         else committed['text'].strip())
            if self.t.thinking:
                reply['reasoning'] = committed['reasoning']
            self.messages += [reply, dict(role='user', content=observation)]
        else:
            self.messages += [dict(role='user', content=observation or FORMAT_ERROR)]
        return committed


def kits():
    return TO.kits()


def transcript(kit, thinking=True, allow_rebuild=True):
    made = TO.xml_transcript(kit, thinking=thinking)
    made.allow_rebuild = allow_rebuild
    return made


# ---- 1. the keep render of one message ----------------------------------------------------------------------

class KeepPieceTests(unittest.TestCase):
    maxDiff = None

    def test_pieces_equal_the_template(self):
        """Each later message's piece is what the template renders for it inside a whole conversation."""
        assistants = [dict(role='assistant', content=C1, reasoning=R1),
                      dict(role='assistant', content='\n\n  ' + C1 + '\n', reasoning='\n ' + R1 + '\n\n'),
                      dict(role='assistant', content='', reasoning='cut inside the reasoning'),
                      dict(role='assistant', content=C1),                                       # no reasoning sent
                      dict(role='assistant', content='<think>\nr\n</think>\n\n' + C1),          # R inside the content
                      dict(role='assistant', content='a</think>b</think>\n\nc'),
                      dict(role='assistant', content='x</think>y', reasoning='given')]          # the field wins
        users = ['plain', '  surrounding whitespace \n', OBS[0], '']
        for kit in kits():
            for message in assistants + [dict(role='user', content=u) for u in users]:
                with self.subTest(kit=kit.name, message=message):
                    messages = TO.xml_messages([]) + [message] + ([] if message['role'] == 'user'
                                                                  else [dict(role='user', content='next')])
                    whole = kit.render(TH.reference_messages(messages), False, **TO.KW_CHAT)
                    before = kit.render(TH.reference_messages(messages[:2]), False, **TO.KW_CHAT)
                    piece = TH.keep_message_text(message)
                    self.assertTrue(whole.startswith(before + piece), (whole, piece))
                    self.assertEqual(kit.encode(whole)[:len(kit.encode(before + piece))],
                                     kit.encode(before) + kit.encode(piece))

    def test_grammar_pieces(self):
        self.assertEqual(XML.QWEN_XML.keep_message_text(dict(role='user', content=' o ')), '<|im_start|>user\no<|im_end|>\n')
        self.assertEqual(XML.GEMMA4_XML.keep_message_text(dict(role='assistant', content=' c ', reasoning='r')),
                         XML.GEMMA4_XML.thinking_message_text('assistant', ' c '))
        with self.assertRaisesRegex(LiveV2Error, 'user and assistant text'):
            TH.keep_message_text(dict(role='system', content='s'))


# ---- 2. the rollback ----------------------------------------------------------------------------------------

class RollbackTests(unittest.TestCase):
    maxDiff = None

    def check_keep(self, agent):
        for k, call in enumerate(agent.served):
            self.assertEqual(call.tokens, keep(agent.kit, call.messages), 'call %d is not the keep render' % k)

    def test_a_reply_cut_inside_its_reasoning(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                agent = Agent(kit, transcript(kit))
                agent.call(TO.raw_reply(R1, C1), observation=OBS[0])
                agent.call(TO.raw_reply(R2, C2), observation=OBS[1])
                cut = agent.call('I keep thinking about <mswea_bash_command>ls</mswea_bash_command> and', kept=False)
                self.assertEqual((cut['closed'], cut['content']), (False, ''))
                agent.call(TO.raw_reply('Back on track.', C1), observation=OBS[2])
                agent.call(TO.raw_reply('Last.', C2), observation=OBS[3])
                histories = [c.history for c in agent.served]
                self.assertEqual(histories, ['fresh', 'append', 'append', TH.ROLLBACK, 'append'])
                self.assertEqual([c.rebuild for c in agent.served], [False, False, False, True, False])
                self.check_keep(agent)
                before, rolled = agent.served[2], agent.served[3]
                X = len(before.tokens) - len(agent.t.header)
                # The exact ids up to the dropped reply's header, then the format-error message and the header.
                self.assertEqual(rolled.tokens[:X], before.tokens[:X])
                self.assertEqual(rolled.tokens[X:], kit.encode(TH.keep_message_text(dict(role='user', content=FORMAT_ERROR))
                                                               + TH.HEADER_TEXT))
                self.assertEqual((rolled.stable, rolled.span), (len(rolled.tokens) - len(agent.t.header), None))
                text = kit.decode(rolled.tokens)
                self.assertIn(R1, text)
                self.assertIn(R2, text)
                self.assertNotIn('I keep thinking', text)
                self.assertEqual((rolled.committed[TH.KEPT_TOKENS], rolled.committed[TH.REENCODED]), (X, 0))
                # Every later call appends to the rolled-back session; R stays.
                self.assertIn(R1, kit.decode(agent.served[4].tokens))
                self.assertEqual(agent.t.sessions, 2)

    def test_a_closed_reply_without_an_action_and_errors_in_a_row(self):
        kit = T.char_kit()
        agent = Agent(kit, transcript(kit))
        agent.call(TO.raw_reply(R1, C1), observation=OBS[0])
        agent.call(TO.raw_reply('No action.', 'THOUGHT: I will think more.'), kept=False)
        agent.call('cut again', kept=False, observation=FORMAT_ERROR + ' (2)')
        agent.call(TO.raw_reply('Also none.', 'THOUGHT: still nothing'), kept=False, observation=FORMAT_ERROR + ' (3)')
        agent.call(TO.raw_reply(R2, C2), observation=OBS[1])
        agent.call(TO.raw_reply('Done.', C1), observation=OBS[2])
        self.assertEqual([c.history for c in agent.served],
                         ['fresh', 'append', TH.ROLLBACK, TH.ROLLBACK, TH.ROLLBACK, 'append'])
        self.check_keep(agent)
        for previous, rolled in zip(agent.served[1:4], agent.served[2:5]):
            X = len(previous.tokens) - len(agent.t.header)
            self.assertEqual(rolled.tokens[:X], previous.tokens[:X])
            self.assertEqual(rolled.committed[TH.KEPT_TOKENS], X)
        self.assertEqual(agent.t.rebuilt[-1][0], 'user')
        self.assertEqual([r[0] for r in agent.t.rebuilt], ['assistant', 'user', 'user', 'user', 'user'])

    def test_the_first_reply_dropped(self):
        kit = T.char_kit()
        agent = Agent(kit, transcript(kit))
        agent.call('thinking, cut', kept=False)
        agent.call(TO.raw_reply(R1, C1), observation=OBS[0])
        agent.call(TO.raw_reply(R2, C2), observation=OBS[1])
        self.assertEqual([c.history for c in agent.served], ['fresh', TH.ROLLBACK, 'append'])
        self.check_keep(agent)
        self.assertEqual(agent.served[1].committed[TH.KEPT_TOKENS], agent.t.initial_length)

    def test_other_rebuilds_render_the_keep_history(self):
        """A resend, a rewritten history or a reply sent back without R: the keep render of what the client sent,
        the re-encoded replies counted."""
        kit = T.char_kit()
        for name, edit, history, replies in (
                ('dropped R', lambda m: m[:-2] + [dict(m[-2], reasoning=None), m[-1]], TH.DROPPED, 2),
                ('compacted', lambda m: m[:2] + [dict(role='user', content='summary')], TH.OTHER, 0),
                ('resend', lambda m: m[:-1], TH.OTHER, 1)):
            with self.subTest(name=name):
                agent = Agent(kit, transcript(kit))
                agent.call(TO.raw_reply(R1, C1), observation=OBS[0])
                agent.call(TO.raw_reply(R2, C2), observation=OBS[1])
                agent.messages = [{k: v for k, v in m.items() if v is not None} for m in edit(agent.messages)]
                if agent.messages[-1]['role'] != 'user':
                    agent.messages = agent.messages[:-1]
                agent.call(TO.raw_reply('Third.', C1), observation=OBS[2])
                call = agent.served[-1]
                self.assertEqual((call.history, call.rebuild), (history, True))
                self.assertEqual(call.tokens, keep(kit, call.messages))
                self.assertEqual((call.committed[TH.KEPT_TOKENS], call.committed[TH.REENCODED]), (0, replies))
                self.assertEqual(R1 in kit.decode(call.tokens), name != 'compacted')
                self.assertEqual(R2 in kit.decode(call.tokens), False)

    def test_thinking_off_is_unchanged(self):
        """Off: a format error rebuilds from the content text, as upstream (no rollback, no new record)."""
        kit = T.char_kit()
        agent = Agent(kit, transcript(kit, thinking=False))
        agent.call(C1, observation=OBS[0])
        agent.call('THOUGHT: no action', kept=False)
        agent.call(C2, observation=OBS[1])
        call = agent.served[-1]
        self.assertEqual((call.history, call.rebuild), (None, True))
        expected = kit.encode(kit.render(call.messages[:2], False, enable_thinking=False))
        for m in call.messages[2:]:
            expected += kit.encode(XML.message_text(m['role'], m['content']))
        self.assertEqual(call.tokens, expected + kit.encode(XML.GENERATION_HEADER))
        self.assertNotIn(TH.KEPT_TOKENS, call.committed)
        self.assertEqual(agent.t.rebuilt[0], ('assistant', agent.t.initial_length, agent.t.rebuilt[0][2], C1))

    def test_without_allow_rebuild_a_rollback_is_refused(self):
        kit = T.char_kit()
        agent = Agent(kit, transcript(kit, allow_rebuild=False))
        agent.call(TO.raw_reply(R1, C1), kept=False)
        with self.assertRaisesRegex(LiveV2Error, 'SESSION_REBUILD_REQUIRED'):
            agent.t.render(agent.messages)
        self.assertEqual(agent.t.turn, 1)          # nothing changed

    def test_abandon_and_the_capacity_wait_checkpoint(self):
        kit = T.char_kit()
        agent = Agent(kit, transcript(kit))
        agent.call(TO.raw_reply(R1, C1), observation=OBS[0])
        agent.call('cut', kept=False)
        live = NS(transcript=agent.t)
        saved = CTL.LiveFourPolicyController.checkpoint(NS(session_of=lambda request: live), None)
        self.assertTrue(set(CTL.TRANSCRIPT_STATE) | set(CTL.XML_TRANSCRIPT_STATE) <= set(saved))
        first, _, _ = agent.t.render(agent.messages)
        self.assertEqual(agent.t.pending['history'], TH.ROLLBACK)
        # CAPACITY_WAIT: the controller restores the attributes, and the retry renders the same rollback.
        for name, value in saved.items():
            setattr(agent.t, name, value)
        again, _, _ = agent.t.render(agent.messages)
        self.assertEqual((again, agent.t.pending['history']), (first, TH.ROLLBACK))
        # An abandoned rollback render (a failure before the engine opens the new namespace) restores the
        # session as it was before the render: the retry renders the same rollback.
        agent.t.abandon()
        self.assertEqual({name: getattr(agent.t, name) for name in XML.Transcript.RESTORED}, dict(
            {k: v for k, v in saved.items() if k in XML.Transcript.RESTORED}))
        self.assertIsNone(agent.t.pending)
        self.assertEqual(agent.t.turn, 2)
        retry, _, _ = agent.t.render(agent.messages)
        self.assertEqual((retry, agent.t.pending['history'], agent.t.pending['rebuild']), (first, TH.ROLLBACK, True))
        agent.t.abandon()

    def test_an_abandoned_rebuild_restores_the_session(self):
        """Every rollback or rebuild render (thinking on or off) resets the session; abandon puts it back."""
        kit = T.char_kit()
        for thinking in (True, False):
            for name, edit in (('rollback', None), ('rebuild', lambda m: m[:2] + [dict(role='user', content='x')])):
                if name == 'rollback' and not thinking:
                    continue
                with self.subTest(thinking=thinking, name=name):
                    agent = Agent(kit, transcript(kit, thinking=thinking))
                    agent.call(TO.raw_reply(R1, C1) if thinking else C1, observation=OBS[0])
                    agent.call(TO.raw_reply(R2, C2) if thinking else C2, kept=edit is not None, observation=OBS[1])
                    messages = agent.messages if edit is None else edit(agent.messages)
                    state = {k: (list(v) if k == 'turns' else v) for k, v in
                             ((k, getattr(agent.t, k)) for k in XML.Transcript.RESTORED)}
                    stream = agent.t.stream
                    agent.t.render(messages)
                    self.assertTrue(agent.t.pending['rebuild'])
                    self.assertEqual(agent.t.turn, 0)                  # reset at render time
                    agent.t.abandon()
                    self.assertEqual({k: (list(v) if k == 'turns' else v) for k, v in
                                      ((k, getattr(agent.t, k)) for k in XML.Transcript.RESTORED)}, state)
                    self.assertIs(agent.t.stream, stream)
                    # The session is the one before the render: the original next request appends to it (or
                    # rolls back again).
                    agent.t.render(agent.messages)
                    self.assertEqual(agent.t.pending['rebuild'], edit is None)
                    agent.t.abandon()
                    self.assertEqual(agent.t.turn, 2)

    def test_an_append_trims_the_new_user_message(self):
        """keep-rebuild-1116 changes ordinary thinking-on appends: the new user message is encoded trimmed, as the
        template renders it (before, untrimmed: an observation with leading or trailing whitespace made that
        append's prompt differ from the template render). Thinking off keeps the untrimmed encoding."""
        for kit in kits():
            with self.subTest(kit=kit.name):
                for thinking in (True, False):
                    agent = Agent(kit, transcript(kit, thinking=thinking))
                    observation = OBS[0] + '\n'                       # a trailing newline
                    agent.call(TO.raw_reply(R1, C1) if thinking else C1, observation=observation)
                    stream = list(agent.t.stream)
                    tokens, _, span = agent.t.render(agent.messages)
                    untrimmed = kit.encode(XML.message_text('user', observation))
                    trimmed = kit.encode(XML.message_text('user', observation.strip()))
                    header = kit.encode(TH.HEADER_TEXT if thinking else XML.GENERATION_HEADER)
                    self.assertEqual(list(tokens), stream + (trimmed if thinking else untrimmed) + header)
                    self.assertEqual(span, (len(stream), len(stream) + len(trimmed if thinking else untrimmed)))
                    self.assertNotEqual(untrimmed, trimmed)
                    if thinking:
                        self.assertEqual(list(tokens), keep(kit, agent.messages))
                    agent.t.abandon()

    def test_evidence_keys(self):
        kit = T.char_kit()
        agent = Agent(kit, transcript(kit))
        agent.call(TO.raw_reply(R1, C1), observation=OBS[0])
        agent.call(TO.raw_reply(R2, C2), kept=False)
        agent.call(TO.raw_reply('ok', C1), observation=OBS[1])
        agent.messages = agent.messages[:2] + [dict(role='user', content='x')]
        agent.call(TO.raw_reply('again', C1), observation=OBS[2])
        rows = [TH.evidence(c.committed, kit.decode, kit.encode) for c in agent.served]
        self.assertEqual([r['history'] for r in rows], ['fresh', 'append', TH.ROLLBACK, TH.OTHER])
        for row in rows[:2]:
            self.assertEqual(set(row), EVIDENCE_KEYS)
        for row in rows[2:]:
            self.assertEqual(set(row), EVIDENCE_KEYS | {TH.KEPT_TOKENS, TH.REENCODED})
        self.assertEqual(rows[3][TH.REENCODED], 0)


# ---- 3. Gemma 4 ---------------------------------------------------------------------------------------------

class GemmaRollbackTests(unittest.TestCase):
    def test_a_rollback_keeps_the_thought_channel(self):
        import test_gemma_thinking_on as G
        for kit in G.kits():
            with self.subTest(kit=kit.name):
                ours = G.renderer_of(kit).transcript(thinking=True)
                messages = G.xml_messages([])
                tokens, _, _ = ours.render(messages)
                committed = ours.commit(tokens, kit.encode(G.raw(G.R1, G.C1)) + [G.TURN_CLOSE])
                messages += [dict(role='assistant', content=committed['content'], reasoning=committed['reasoning']),
                             dict(role='user', content='obs')]
                second, _, _ = ours.render(messages)
                ours.commit(second, kit.encode(G.THOUGHT + 'cut inside the channel'))
                messages.append(dict(role='user', content=FORMAT_ERROR))
                third, _, _ = ours.render(messages)
                self.assertEqual((ours.pending['history'], ours.pending['rebuild']), (TH.ROLLBACK, True))
                X = len(second) - len(ours.header)
                self.assertEqual(third[:X], second[:X])
                self.assertEqual(list(third[X:]), kit.encode('<|turn>user\n' + FORMAT_ERROR.strip() + '<turn|>\n'
                                                             + G.HEADER))
                # The first reply's thought channel (R) is still there, as the append stream held it.
                self.assertEqual(list(third[:len(tokens)]), list(tokens))
                self.assertIn(G.R1, kit.decode(list(third)))


# ---- 4. the query hook's history after a rollback -------------------------------------------------------------

class HistoryMessageTests(unittest.TestCase):
    def test_rows_after_a_rollback(self):
        kit = T.char_kit()
        agent = Agent(kit, transcript(kit))
        agent.call(TO.raw_reply(R1, C1), observation=OBS[0])
        agent.call(TO.raw_reply(R2, C2), observation=OBS[1])
        agent.call('cut', kept=False)
        agent.call(TO.raw_reply('Next.', C1), observation=OBS[2])
        tokens, stable, span = agent.t.render(agent.messages)
        self.assertFalse(agent.t.pending['rebuild'])
        rebuilt = agent.t.rebuilt
        self.assertEqual([r[0] for r in rebuilt], ['assistant', 'user', 'assistant', 'user', 'user'])
        self.assertTrue(all(len(r) == 5 for r in rebuilt))
        self.assertEqual([r[4] for r in rebuilt if r[0] == 'assistant'], [R1 + '\n', R2 + '\n'])
        self.assertEqual(rebuilt[0][1], agent.t.initial_length)
        self.assertTrue(all(x[2] == y[1] for x, y in zip(rebuilt, rebuilt[1:])))
        ends, spans = [agent.served[3].stable], []      # the session's local turn 0 (the rollback), then this turn
        owner = NS(bound=lambda sid: (None, agent.t))
        owner.call_starts = lambda sid, replies: QuerySelectionAuthority.call_starts(owner, sid, replies)
        rows = QuerySelectionAuthority.history_messages(owner, NS(session='s', stable_end=stable), ends, spans)
        self.assertEqual([m.role for m in rows], ['assistant', 'user', 'assistant', 'user', 'user', 'assistant'])
        think_end = tuple(kit.encode(sc.THINK_END))
        for m in rows:
            if m.role == 'assistant':
                self.assertIsNotNone(m.reasoning)
                self.assertEqual(tuple(agent.t.stream[m.call - len(think_end):m.call]), think_end)
        self.assertEqual(rows[0].reasoning, R1 + '\n')
        # The server's own split of each reply (as the live rows), not the client's trimmed echo.
        self.assertEqual(rows[0].text, '\n\n' + C1)
        self.assertEqual(rows[0].text, agent.served[0].committed['content'])


# ---- 5. tools/keep_render_check.py ---------------------------------------------------------------------------

def checker(kit):
    """The tool's renderer over a kit: its decode keeps special tokens."""
    if kit.name == 'char':
        tok = KP.CharTokenizer()
        decode = lambda ids: tok.decode(list(ids), skip_special_tokens=False)
    else:
        decode = lambda ids: kit.tokenizer.decode(list(ids), skip_special_tokens=False)
    return KRC.KeepRenderer(kit.encode, kit.render_template, KP.TEMPLATE, decode)


def trajectory(agent):
    """The session's messages as mini-swe-agent writes them (reasoning as reasoning_content, extras ignored)."""
    out = []
    for m in agent.messages:
        m = dict(m, extra={'cost': 0})
        if 'reasoning' in m:
            m['reasoning_content'] = m.pop('reasoning')
        out.append(m)
    return out


def spelled(kit, text):
    """A reply's ids with every ``</think>`` spelled character by character (the char kit): a valid but
    non-canonical tokenization of the same text, as a model may generate."""
    head, _, tail = text.partition(TH.THINK_END)
    return kit.encode(head) + [ord(c) + KP.CharTokenizer.BASE for c in TH.THINK_END] + kit.encode(tail)


class KeepRenderCheckTests(unittest.TestCase):
    maxDiff = None

    def session(self, kit, generated=None):
        agent = Agent(kit, transcript(kit))
        replies = [TO.raw_reply(R1, C1), 'cut', TO.raw_reply(R2, C2), TO.raw_reply('Done.', C1)]
        for k, (text, kept, observation) in enumerate(zip(replies, (True, False, True, True), (OBS[0], None, OBS[1],
                                                                                            OBS[2]))):
            agent.call(text, kept=kept, observation=observation,
                       generated=None if generated is None else generated(k, text))
        return agent

    def test_the_new_runtime_passes(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                agent = self.session(kit)
                requests = KRC.requests_from_trajectory(trajectory(agent))
                self.assertEqual(len(requests), len(agent.served) + 1)
                served = [c.tokens for c in agent.served]
                results = KRC.compare(checker(kit), requests[:len(served)], served)
                self.assertEqual(KRC.summary(results), dict(calls=4, equal=4, text_equal=4, exact_equal=4, failed=0,
                                                            first_failed=None))
                lengths = KRC.compare(checker(kit), requests[:len(served)], [len(s) for s in served])
                self.assertTrue(all(r['equal'] and r['exact_equal'] is None for r in lengths))
                # Without a decode the verdict is the plain token equality.
                plain = KRC.KeepRenderer(kit.encode, kit.render_template, KP.TEMPLATE)
                self.assertEqual(KRC.summary(KRC.compare(plain, requests[:4], served))['failed'], 0)

    def test_replies_kept_as_non_canonical_ids(self):
        """The server keeps a reply's exact ids, here with </think> spelled in pieces: not token-equal to the
        re-encoded keep render, but text-equal, and exact-equal with the reply's own ids."""
        kit = T.char_kit()
        agent = self.session(kit, generated=lambda k, text: spelled(kit, text))
        requests = KRC.requests_from_trajectory(trajectory(agent))[:4]
        served = [c.tokens for c in agent.served]
        results = KRC.compare(checker(kit), requests, served)
        self.assertEqual([r['equal'] for r in results], [True, False, False, False])
        self.assertEqual([r['text_equal'] for r in results], [True] * 4)
        self.assertEqual([r['exact_equal'] for r in results], [True] * 4)
        self.assertEqual(KRC.summary(results)['failed'], 0)

    def test_the_old_rebuild_and_other_differences_are_caught(self):
        """The content-only rebuild of the runtime before this branch: the first difference is the first
        reasoning after the task. A changed observation token and a reply whose kept text is not the template's
        fail too."""
        kit = T.char_kit()
        agent = self.session(kit)
        requests = KRC.requests_from_trajectory(trajectory(agent))[:4]
        served = [c.tokens for c in agent.served]
        old = kit.encode(kit.render(requests[2][:2], False, enable_thinking=True))
        for m in requests[2][2:]:
            old += kit.encode(XML.message_text(m['role'], m['content']))
        results = KRC.compare(checker(kit), requests, served[:2] + [old + kit.encode(TH.HEADER_TEXT)] + served[3:])
        self.assertEqual([r['exact_equal'] for r in results], [True, True, False, True])
        initial = len(kit.encode(kit.render(requests[2][:2], False, enable_thinking=True)))
        self.assertEqual(results[2]['first_diff'], initial + len(kit.encode('<|im_start|>assistant\n')))
        self.assertEqual(KRC.summary(results)['first_failed'], 2)
        bad = KRC.compare(checker(kit), requests, [len(s) for s in served[:2]] + [len(old) + 5, len(served[3])])
        self.assertEqual(bad[2]['first_diff'], 'length')
        # One observation token changed.
        changed = [list(s) for s in served]
        at = len(changed[3]) - len(agent.t.header) - 3
        changed[3][at] += 1
        results = KRC.compare(checker(kit), requests, changed)
        self.assertEqual(([r['exact_equal'] for r in results], results[3]['first_diff']), ([True] * 3 + [False], at))
        # The server kept a reply with surrounding whitespace in its content (the template trims it).
        loose = self.session(kit, generated=lambda k, text: kit.encode(text + ('  ' if k == 2 else '')))
        served = [c.tokens for c in loose.served]
        results = KRC.compare(checker(kit), KRC.requests_from_trajectory(trajectory(loose))[:4], served)
        self.assertEqual([r['exact_equal'] for r in results], [True, True, True, False])
        self.assertEqual(results[3]['text_equal'], False)

    def test_a_fault_shared_by_the_tool_and_the_runtime_fails(self):
        """The verdict needs text_equal too: pieces that trim only newlines, in the tool and in the served prompt
        alike, agree with each other (exact_equal) but not with the template (text_equal), so the call fails."""
        from unittest.mock import patch
        kit = T.char_kit()
        agent = self.session(kit)
        requests = KRC.requests_from_trajectory(trajectory(agent))[:4]
        requests[3] = requests[3][:-1] + [dict(requests[3][-1], content=' ' + requests[3][-1]['content'])]
        served = [list(c.tokens) for c in agent.served]
        real = KRC.piece_text

        def faulty(message):
            text = real(message)
            if message['role'] == 'user':
                text = '<|im_start|>user\n' + message['content'].strip('\n') + '<|im_end|>\n'
            return text
        header = kit.encode(TH.HEADER_TEXT)
        good = kit.encode(real(requests[3][-1]))
        served[3] = served[3][:len(served[3]) - len(header) - len(good)] + kit.encode(faulty(requests[3][-1])) + header
        with patch.object(KRC, 'piece_text', faulty):
            results = KRC.compare(checker(kit), requests, served)
        self.assertEqual((results[3]['exact_equal'], results[3]['text_equal']), (True, False))
        self.assertFalse(KRC.verdict(results[3]))
        self.assertEqual((KRC.summary(results)['failed'], KRC.summary(results)['first_failed']), (1, 3))

    def test_interface_errors_and_helpers(self):
        self.assertEqual(KRC.first_difference([1, 2], [1, 2, 3]), 2)
        self.assertIsNone(KRC.first_difference([1], [1]))
        self.assertEqual(KRC.normalize(dict(role='assistant', content='c', reasoning='r', tool_calls=None)),
                         dict(role='assistant', content='c', reasoning_content='r'))
        self.assertEqual(KRC.normalize(dict(role='user', content='u', reasoning_content='x')),
                         dict(role='user', content='u'))
        with self.assertRaises(ValueError):
            KRC.normalize(dict(role='user', content=None))
        with self.assertRaises(ValueError):
            KRC.requests_from_trajectory([dict(role='user', content='x')])
        with self.assertRaises(ValueError):
            KRC.compare(checker(T.char_kit()), [[]], [])
        with self.assertRaises(ValueError):
            KRC.piece_text(dict(role='system', content='s'))
        for message in (dict(role='user', content=' u '), dict(role='assistant', content=' c ', reasoning_content=' r '),
                        dict(role='assistant', content='<think>r</think>c')):
            self.assertEqual(KRC.piece_text(message), TH.keep_message_text(TH.plain(dict(
                message, **({'reasoning': message['reasoning_content']} if 'reasoning_content' in message else {})),
                True)))

    @unittest.skipUnless(KP.HAVE_TRANSFORMERS, 'Transformers (the remote-ra-tok venv) required')
    def test_cli_with_the_pinned_tokenizer(self):
        kit = T.real_kit()
        agent = self.session(kit)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            import shutil
            for target, name in (('config.json', 'model-config.json'), ('tokenizer.json', 'tokenizer.json'),
                                 ('tokenizer_config.json', 'tokenizer-config.json')):
                shutil.copyfile(KP.EVIDENCE / name, work / target)
            (work / 'template.jinja').write_text(KP.TEMPLATE, encoding='utf-8')
            (work / 'traj.json').write_text(json.dumps(dict(messages=trajectory(agent))), encoding='utf-8')
            (work / 'served.json').write_text(json.dumps([c.tokens for c in agent.served]), encoding='utf-8')
            run = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'tools' / 'keep_render_check.py'), str(work),
                                  str(work / 'template.jinja'), str(work / 'traj.json'), str(work / 'served.json')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            value = json.loads(run.stdout)
            self.assertEqual((value['exact_equal'], value['text_equal'], value['failed'], value['verdict_rule']),
                             (4, 4, 0, 'exact_equal and text_equal'))


if __name__ == '__main__':
    unittest.main()
