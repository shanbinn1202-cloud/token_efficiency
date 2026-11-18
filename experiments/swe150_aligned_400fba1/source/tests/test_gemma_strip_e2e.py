"""--thinking-history strip on the Gemma 4 profile end to end: the integration of gemma-thinking (the strip
transcript) and gemma-thinking-hold (the sliding window held at the cut, option G). CPU only; not GPU
evidence. Design: docs/thinking-gemma/DESIGN.md sections 3.3 and 4.4.

Everything runs on test_gemma_hold's server: test_gemma_chat_text_e2e's API server (the packaged router, the
chat-text route, the frontend, the capacity gate) over the window layout's real engine and worker, with the
runtime's own Gemma strip transcripts over a stand-in tokenizer whose thinking-on header is the pinned
[105, 4368, 107] (K = |H| = 3), attached through the runtime's entry points (thinking_cut.attach_engine and
attach_worker, which hand the window layout to window_hold). The fake geometry is test_gemma_e2e's (P = 16,
W = 64, one native forward of 1,024 tokens); on the server W = 1,024, so "shorter than a block", "about a
window" and "longer than a window" here are replies of fewer than 16, about 1,000 and more than 1,024 tokens
there. All reasoning and content is synthetic.

1. Reply lengths: replies shorter than a block, just under a window and longer than a window, a reply
   without a channel and one cut inside its channel, at local turn 0 and at continuing turns of every cut
   offset: each completion commits at X with the window held at X, the hold's ``left`` is what the window
   moved, the next turn re-prefills the content the client sends back (an empty one too).
2. Terminus 2 sessions (test_terminus2_goal's client with 2.0.0's exact prompts: main turns, the proactive
   summary hand-off and its asides, session evictions and rebinds) with H < K hold slots: every served
   prompt equals the stock thinking-on render of its messages; every completion is checked at its cut.
3. A re-prefill longer than one native forward: phase 1 runs in chunks and the window it ended in is held.
4. --goal-overlap with the hold (test_gemma_chat_text_goal's goal_query scenario: Terminus 2 clients with
   summaries, Goals parked on the Goal thread, --admission-overlap, --release-idle-pages under pressure, the
   three decode fast paths) on a strip server: every check of the scenario and of the strip commit holds,
   every Goal that needed a forward ran it as its own lease after phase 1's drain, and every session's
   prompts, replies, selections and Goals equal the same server without --goal-overlap.
5. The wiring: compose_engine sends the strip record to the window worker and checks its echo; the window
   worker attaches and echoes it and refuses it without hold slots or with --async-scheduling; mutations a
   broken wiring must fail (the engine attached without the hold, the worker or the engine not attached, a
   layout without hold slots, a re-prefilled piece that keeps the reasoning).

With the pinned tokenizer and template (REMOTE_RA_GEMMA4_METADATA and Transformers; no torch there, so not on
the engine fakes) test_gemma_thinking_cut replays Terminus-shaped sessions on the strip transcripts against
the pinned template's render.
"""
import asyncio
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_gemma_hold as GH  # noqa: E402  the Gemma strip server on the window fakes and its checks
import test_chat_text_e2e as CE  # noqa: E402  the scripted model
import test_chat_text_protocol as T  # noqa: E402  prompts and observations
import test_gemma_chat_text_e2e as GCE  # noqa: E402
import test_gemma_chat_text_goal as GCG  # noqa: E402  the goal_query scenario
import test_gemma_e2e as GE  # noqa: E402
import test_gemma_flags as GF  # noqa: E402  compose_engine's worker record on a Gemma config
import test_gemma_graphs as GG  # noqa: E402  the window worker's attachment with its loaded seams faked
import test_gemma_thinking_on as GTO  # noqa: E402
import test_ms_e2e as E2E  # noqa: E402
import test_terminus2_goal as TG  # noqa: E402  Terminus 2.0.0's exact prompts
import test_thinking_cut as TC  # noqa: E402  the clients (send, episode)

from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import goal_overlap as GO  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2 import thinking_cut as CUT  # noqa: E402
from remote_ra_live_v2 import window_hold as WH  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from remote_ra_live_v2.sliding_window import kept_blocks  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

FW, FP, HEADER = GH.FW, GH.FP, len(GH.HEADER_IDS)
CONTENT = '{"k":1}'
ON = {'goal_overlap': True}


def sized(total):
    """A thinking reply of exactly ``total`` generated tokens, its stop id included: the channel opening
    (3 tokens), a reasoning, ``<channel|>`` and the content; 0: the content alone, no channel."""
    if not total:
        return CONTENT
    fixed = len(GH.KIT.encode(GTO.THOUGHT)) + 1 + len(GH.KIT.encode(CONTENT)) + 1
    assert total > fixed
    return GTO.raw('r' * (total - fixed), CONTENT)


SHORT, NEAR, LONG = 13, FW - 4, FW + 2 * FP + 8      # < a block; just under a window; longer than a window


# ---- 1. reply lengths ------------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ReplyLengthTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None
    send, episode = TC.EndToEndTests.send, TC.EndToEndTests.episode

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    def scripted(self, length):
        """The model replies with ``length(client, turn)`` generated tokens (``sized``). Installed after
        the server (whose patches wrap test_chat_text_e2e's model in the thinking model)."""
        def model(messages):
            k = int(re.search(r'Task (\d+)', messages[0]['content']).group(1))
            return sized(length(k, (len(messages) - 1) // 2))
        self.enterContext(patch.object(CE, 'model', model))

    def lefts(self, srv):
        """Each completion's hold against what the window moved while its reply decoded: the blocks of
        kept_blocks(X) before the first block the row still needs at its last allocation."""
        rows = {r['request_id']: r for r in srv.rows()}
        out = []
        for rid, done in srv.engine.recorder.completions.items():
            hold, g = done['hold'], rows[rid]['chat_text']['completion_tokens']
            blocks = hold['last'] - hold['first']
            moved = {min(blocks, max(0, kept_blocks(done['X'] + HEADER + g - d, FW, FP)[0] - hold['first']))
                     for d in (0, 1, 2)}
            if hold['source'] != 'row':
                self.assertIn(hold['left'], moved, (rid, hold, g))
            out.append((g, bool(done['local']), hold['source'], hold['left'], blocks))
        return out

    async def test_short_near_window_and_long_replies(self):
        """Every client's turns cycle through the three lengths, with first prompts of every offset mod P:
        a reply shorter than a block leaves the held window in the row (at most its first block has left),
        one just under a window has moved most of it out, one longer than a window all of it."""
        srv = GH.strip_window_server(self, 'observation_query', slots=24, limit=8, hold=3)
        self.scripted(lambda k, j: (SHORT, NEAR, LONG)[(j + k) % 3])
        count = 6
        for k in range(count):
            replies = await self.episode(srv, k, turns=4, pad=3 * k)
            self.assertEqual([r['choices'][0]['message']['content'] for r in replies], [CONTENT] * 4)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(GH.check_window_strip(self, srv, completions=4 * count), 4 * count)
        seen = self.lefts(srv)
        self.assertEqual({g for g, *_ in seen}, {SHORT, NEAR, LONG})
        for g, local, source, left, blocks in seen:
            if local:
                self.assertEqual(source, 'install')
                self.assertTrue(left <= 1 if g == SHORT else 0 < left < blocks if g == NEAR else left == blocks,
                                (g, left, blocks))
            elif g == LONG and source == 'prefill':
                self.assertEqual(left, blocks)
        # Every reply is split on the channel ids: 1 opening, the reasoning, 1 close, the content.
        for row in srv.rows():
            t = row['thinking']
            self.assertEqual((t['closed'], t['think_start_tokens'], t['think_end_tokens']), (True, 1, 1))
            self.assertEqual(t['reasoning_tokens'] + t['content_tokens'] + 2, row['chat_text']['completion_tokens'])
        self.assertGreaterEqual(len({done['X'] % FP for done in srv.engine.recorder.completions.values()}), 8)
        GH.boundary(self, srv)

    async def test_a_reply_without_a_channel_and_a_reply_cut_inside_its_channel(self):
        """No channel: the reply is content only. Cut at its bound inside the channel: all reasoning, the
        content is "" and the client sends "" back; the next prompt re-prefills ``<|turn>model\\n<turn|>\\n``
        (the stock render of an empty reply) on the window held at that cut."""
        srv = GH.strip_window_server(self, 'full', slots=8, limit=2)
        self.scripted(lambda k, j: (0, LONG, 0, LONG)[j])
        replies = await self.episode(srv, 0, turns=4, max_tokens=lambda j: 9 if j == 1 else None)
        messages = [r['choices'][0]['message'] for r in replies]
        self.assertEqual([m['content'] for m in messages], [CONTENT, '', CONTENT, CONTENT])
        self.assertEqual([r['choices'][0]['finish_reason'] for r in replies], ['stop', 'length', 'stop', 'stop'])
        self.assertEqual(GH.check_window_strip(self, srv, completions=4), 4)
        rows = srv.rows()
        self.assertEqual([(r['thinking']['closed'], r['thinking']['think_start_tokens'], r['thinking']['content_tokens'])
                          for r in rows], [(True, 0, 8), (False, 1, 0), (True, 0, 8), (True, 1, 8)])
        # The turn after the cut reply: its re-prefilled piece is the header and the end of turn alone.
        after = srv.native(rows[2]['request_id'])
        start, end = after['reprefill']['start'], after['reprefill']['piece_end']
        self.assertEqual(after['prompt_token_ids'][start:end], GH.KIT.encode(CUT.gemma4_reply_piece('')))
        self.assertEqual(end - start, HEADER + 2)
        self.assertEqual([srv.native(r['request_id'])['truncate']['dropped']['think_end'] for r in rows][:2], [None, None])
        GH.boundary(self, srv)


# ---- 2. Terminus 2 sessions --------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class TerminusTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def test_terminus_2_sessions_are_served_the_stock_thinking_on_prompts(self):
        """Four Terminus 2 clients with the proactive summary hand-off (its three sub-calls, the unwound
        history) on a strip server of four open sessions and two hold slots: every served prompt is the
        stock thinking-on render of its messages (no reasoning, no channel), every completion committed at
        its cut with the held window, and at most two replies decode at once."""
        count, episodes = 4, 6
        srv = GH.strip_window_server(self, 'observation_query', slots=count * (episodes + 1) * 3, limit=4, hold=2)
        srv.chat.autostart, srv.chat.tick_s = True, 0.02
        srv.chat.record.update(evict_after_s=0.05)
        chars = lambda messages: sum(len(m['content']) for m in messages)
        agents = []
        for k in range(count):
            instruction = 'Task %d: build /app/job%d.py and make ./run %d print ok.' % (k, k, k)
            agent = TG.Terminus2(instruction, asend=srv.send, count=chars, context_limit=4200, threshold=600,
                                 header='t%02d__agent' % k if k % 2 else None)
            agent.first = TG.first_prompt(instruction, 'root@h%04d:/app# \n' % k)
            agents.append(agent)

        async def play(k):
            for j in range(episodes):
                await agents[k].step(agents[k].first if j == 0 else TG.new_output(
                    'root@h%04d:/app# ./run %d\nok %d\nroot@h%04d:/app# ' % (k, j, j, k)))
        await asyncio.gather(*(play(k) for k in range(count)))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertTrue(all(status == 200 for a in agents for _, status, _ in a.requests))
        self.assertTrue(all('summary' in a.events for a in agents))
        rows = srv.rows()
        self.assertEqual(len(rows), sum(len(a.requests) for a in agents))
        self.assertEqual(GH.check_window_strip(self, srv, completions=len(rows)), len(rows))
        for entry in srv.served:
            self.assertEqual(entry['tokens'], GH.reference(entry['messages']))
            text = GH.KIT.keep(entry['tokens'])
            self.assertNotIn('<|channel>', text)
            self.assertNotIn('Reasoning', text)
            self.assertEqual(text.count('<|think|>'), 1)
        self.assertEqual({r['chat_text']['kind'] for r in rows}, {'session', 'main', 'aside'})
        histories = {r['thinking']['history'] for r in rows}
        self.assertEqual(histories, {CUT.FRESH, CUT.CUT, CUT.REBUILD})
        # The client never saw a reasoning in a content, and never sent one back.
        for agent in agents:
            for messages, _, body in agent.requests:
                self.assertNotIn('Reasoning', body['choices'][0]['message']['content'])
        self.assertLessEqual(srv.engine.recorder.most, 2)
        self.assertEqual(srv.engine.adapter.window.hold_slots, 2)


# ---- 3. a re-prefill longer than one native forward -----------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChunkedPhase1Tests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None
    send = TC.EndToEndTests.send

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def test_phase_1_in_chunks_ends_in_the_held_window(self):
        """The re-prefill [C, X) of a continuing turn longer than one native forward runs as chunked forwards
        on one lease (test_gemma_chunked; forwards of 256 tokens here): the window its last forward ends in is
        the window at X the request holds from its install, and the next turn's phase 1 borrows it."""
        import copy
        import importlib
        WP = importlib.import_module('remote_ra_live_v2.window_phase1')
        leases, parse = [], WP.parse_lease

        def recording(raw, **kw):
            leases.append(copy.deepcopy(raw))
            return parse(raw, **kw)
        self.enterContext(patch.object(WP, 'parse_lease', recording))
        srv = GH.strip_window_server(self, 'observation_query', slots=4, limit=2)
        srv.engine.endpoint.phase1.chunk_tokens = 256
        messages = [dict(role='user', content=T.first_prompt('Chunks.'))]
        lengths = []
        for j in range(3):
            status, body = await self.send(srv, messages)
            self.assertEqual(status, 200, body)
            # An observation of about two native forwards (the fakes' forward is 1,024 tokens).
            observation = T.observation(j, size=2) + ('line %d of a long build log\n' % j) * 90
            messages = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                                   dict(role='user', content=observation)]
            lengths.append(len(GH.KIT.encode(observation)))
        self.assertGreater(min(lengths), 2 * GE.N)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(GH.check_window_strip(self, srv, completions=3), 3)
        rows = srv.rows()
        cuts = [row['truncate']['cut'] for row in rows]
        chunked = [lease for lease in leases if 'chunks' in lease]
        # Both continuing turns' phase 1 ran in chunks of at most 256 tokens and ended at the turn's cut.
        self.assertEqual([lease['end'] for lease in chunked], cuts[1:])
        for lease in chunked:
            bounds = [lease['start']] + [c[0] for c in lease['chunks']] + [lease['end']]
            self.assertTrue(all(0 < b - a <= 256 for a, b in zip(bounds, bounds[1:])))
            self.assertGreater(lease['end'] - lease['start'], 2 * GE.N)
        for row in rows[1:]:
            self.assertEqual(srv.native(row['request_id'])['truncate']['window_hold']['source'], 'install')
        GH.boundary(self, srv)


# ---- 4. --goal-overlap with the hold ----------------------------------------------------------------------------

def strip_goal_run(count, episodes, blocks, *, goal_overlap, hold, **options):
    """test_gemma_chat_text_goal.goal_run (Terminus 2 clients with summaries on a Gemma goal_query chat-text
    server, Goals parked on the Goal thread) on a strip server with ``hold`` hold slots, attached as a strip
    server attaches, and with ``goal_overlap`` --goal-overlap as compose_engine and the window worker attach
    it. The scenario's prompt check becomes the strip one (GH.check_window_strip: the stock thinking-on
    render, the commit at the cut with the held window). Returns goal_run's outcome and findings, and
    ``checked``: the strip completions checked, the turns' Goal frames, the hold figures."""
    checked = {}
    plain_config = GCG.goal_config

    def goal_config(output, slots, limit, release, overlap, **kwargs):
        value, layout = plain_config(output, slots, limit, release, overlap, **kwargs)
        layout = dataclasses.replace(layout, hold_slots=hold)
        value['kv_layout'] = I0.to_additional_config(layout)[I0.LAYOUT_KEY]
        value['chat_text'] = C.record(slots=slots, profile=MP.GEMMA4, thinking=True, strip=True)
        value['thinking'] = GH.STRIP
        if goal_overlap:
            value['admission'] = dict(value.get('admission') or {}, **ON)
        assert CUT.check_config(value) is True
        return value, layout

    class Server(GH.StripChatServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            h = self.engine
            GH.attach_strip(h)
            if goal_overlap:
                assert GO.attach_engine(h.core, dict(ON)) is True
                h.owner.goal_overlap = True

    def check_prompts(case, srv, cut=()):
        h = srv.engine
        checked['completions'] = GH.check_window_strip(case, srv)
        case.assertEqual(bool(h.core._remote_ra_private_v2_endpoint.goal_overlap), goal_overlap)
        forwards = 0
        for row in srv.rows():
            native = srv.native(row['request_id'])
            operations = [f['operation'] for f in native['worker_frames']]
            if goal_overlap:
                # Every phase-1 frame ran without a Goal forward; a Goal that needed one ran it as its own
                # lease after phase 1's drain, before the request's install (where the window is held).
                for frame in native['worker_frames']:
                    if frame['operation'] == 'phase1':
                        case.assertIsNone(frame.get('goal_vector'))
                if 'goal_forward' in operations:
                    case.assertGreater(operations.index('goal_forward'), operations.index('phase1'))
                    forwards += 1
            else:
                case.assertNotIn('goal_forward', operations)
        checked.update(forwards=forwards, most=h.recorder.most, hold_slots=h.adapter.window.hold_slots,
                       sources=sorted({c['hold']['source'] for c in h.recorder.completions.values()}))
        GH.boundary(case, srv)

    with GH.strip_patches(), patch.object(GCG, 'goal_config', goal_config), \
            patch.object(GCE, 'GemmaChatServer', Server), \
            patch.object(GCE.GemmaChatTextEndToEndTests, 'check_prompts', check_prompts):
        outcome, found = GCG.goal_run(count, episodes, blocks, **options)
    return outcome, found, checked


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GoalOverlapTests(unittest.TestCase):
    """goal_query with --goal-overlap on a Gemma strip server: the overlap changes when phase 1 runs (beside
    the Goal's construction) and adds the Goal forward's own lease; the strip commit holds the window phase 1
    ended in from the request's install. The two compose: nothing a turn decides or commits changes."""
    maxDiff = None
    # test_gemma_chat_text_goal's pool-0 sizes: a server that never releases holds every open session's pins
    # and one turn; under pressure (with --release-idle-pages) every first turn fits, not every pin.
    REFERENCE, PRESSURE = GCG.EndToEndTests.REFERENCE, GCG.EndToEndTests.PRESSURE

    def run_pair(self, blocks, **options):
        runs = {}
        for overlap in (True, False):
            outcome, found, checked = strip_goal_run(4, 6, blocks, goal_overlap=overlap, hold=2, **options)
            self.assertEqual(outcome, (1, [], [], []), outcome)
            self.assertEqual(checked['completions'], sum(len(v) for v in found['sessions'].values()))
            self.assertEqual(checked['hold_slots'], 2)
            self.assertLessEqual(checked['most'], 2)
            runs[overlap] = (found, checked)
        (on, on_checked), (off, _) = runs[True], runs[False]
        self.assertGreater(on_checked['forwards'], 0)
        self.assertIn('install', on_checked['sources'])
        # Every client session's turns: prompts, selections, attended Pages, Goal inputs, decisions, query
        # hashes and rewriter requests, and every reply, equal the server without --goal-overlap.
        self.assertEqual(on['replies'], off['replies'])
        self.assertEqual(on['sessions'], off['sessions'])
        return on

    def test_strip_with_the_goal_overlap_and_the_terminal_bench_flags(self):
        """--admission-overlap 16, --release-idle-pages under pool pressure and the three decode fast paths
        (the Terminal-Bench servers' flags), two hold slots for twenty open sessions."""
        found = self.run_pair(self.PRESSURE, overlap=16, release=True, fast=3)
        self.assertGreater(found['width'], 1)                      # sessions decode together
        self.assertTrue({'evict', 'readmit'} <= {e['event'] for e in found['events']})

    def test_strip_with_the_goal_overlap_alone(self):
        """Without --admission-overlap: a parked Goal waits after its phase 1; one admission at a time."""
        self.run_pair(self.REFERENCE)


# ---- 5. the wiring ----------------------------------------------------------------------------------------------

class RecordTests(unittest.TestCase):
    """The strip record between the engine and the window worker (compose_engine, attach_window_worker)."""
    maxDiff = None

    def setUp(self):
        import test_gemma_startup as GS
        self.gs = GS.GemmaStartupTests('test_load_accepts_the_gemma_startup_config')
        self.gs.setUp()
        self.addCleanup(self.gs.doCleanups)

    def launched(self, **kw):
        import serve
        options = dict(dict(task=None, policy='observation_query', selection=dict(active_token_budget=8192),
                            protocol='chat-text', max_running=4, enable_thinking=True), **kw)
        config = self.gs.prepare(**options)[1]
        self.gs.n += 1
        path = self.gs.root / ('startup%d.json' % self.gs.n)
        path.write_bytes(serve.wire(config))
        return config, dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))

    def run_engine(self, launched, mutate=None):
        """launch -> compose_engine's worker startup record -> attach_window_worker (its loaded seams faked)
        -> its evidence (``mutate``d) back to compose_engine: the record, the outcome, the attached runner."""
        from remote_ra_http import config as HC
        config, env = launched
        with patch.dict(os.environ, env):
            value = HC.load()
            layout = HC.kv_layout(value)
        attached = []

        def worker(record):
            attached.append(GG.WorkerAttachTests().attach(record, *GF.compiled('off'), layout=layout))
            evidence = dict(attached[-1][0])
            return mutate(evidence) if mutate is not None else evidence
        record, outcome = GF.engine_record(value, env, worker)
        return record, outcome, attached, layout

    def test_engine_worker_and_echo(self):
        strip = self.launched(thinking_history='strip', window_hold_slots=2)
        keep = self.launched()
        record, outcome, ((evidence, runner),), layout = self.run_engine(strip)
        plain, plain_outcome, ((plain_evidence, plain_runner),), _ = self.run_engine(keep)
        self.assertEqual((outcome, plain_outcome), ('accepted', 'accepted'))
        # The engine sends the strip record once; the window worker attaches the hold and echoes it.
        self.assertEqual((record[CUT.WORKER_KEY], CUT.WORKER_KEY in plain), ('strip', False))
        self.assertEqual({k: v for k, v in record.items() if k not in (CUT.WORKER_KEY, 'kv_layout')},
                         {k: v for k, v in plain.items() if k != 'kv_layout'})
        self.assertEqual((layout.hold_slots, record['kv_layout']['hold_slots']), (2, 2))
        self.assertEqual((evidence[CUT.WORKER_KEY], CUT.WORKER_KEY in plain_evidence), ('strip', False))
        self.assertEqual(evidence['kv_layout'][I0.LAYOUT_KEY]['hold_slots'], 2)
        owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
        self.assertTrue(owner.commit_at_cut and bridge.qwen.commit_at_cut and owner.backend.commit_at_cut)
        plain_owner = plain_runner._remote_ra_live_v2_allocation_owner
        self.assertFalse(getattr(plain_owner, 'commit_at_cut', False) or
                         getattr(plain_owner.backend, 'commit_at_cut', False))
        # A worker that attached without it (no echo), and one that echoes it to a keep server: refused.
        drop = lambda evidence: {k: v for k, v in evidence.items() if k != CUT.WORKER_KEY}
        self.assertIn('another thinking_history', self.run_engine(strip, drop)[1])
        self.assertIn('another thinking_history',
                      self.run_engine(keep, lambda evidence: dict(evidence, thinking_history='strip'))[1])

    def test_the_window_worker_refusals(self):
        strip = self.launched(thinking_history='strip', window_hold_slots=2)
        record, _, _, layout = self.run_engine(strip)
        attach = lambda startup, layout=layout: GG.WorkerAttachTests().attach(startup, *GF.compiled('off'),
                                                                               layout=layout)
        attach(record)
        # Without hold slots in the worker's layout, and with --async-scheduling (v1): refused there too.
        with self.assertRaisesRegex(LiveV2Error, r'needs hold slots \(kv_layout.hold_slots >= 1\)'):
            attach(record, dataclasses.replace(layout, hold_slots=0))
        with self.assertRaisesRegex(LiveV2Error, 'refuses --async-scheduling'):
            attach(dict(record, async_scheduling=True, decode_fastpath=True, decode_fastpath2=True))
        with self.assertRaisesRegex(LiveV2Error, 'must be strip or absent'):
            attach(dict(record, thinking_history='keep'))
        # On the HTTP runtime the record is compared with the pinned startup config, once.
        with patch.dict(os.environ, strip[1]):
            self.assertIs(CUT.worker_flag({CUT.WORKER_KEY: 'strip'}), True)
            with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
                CUT.worker_flag({})
        with patch.dict(os.environ, self.launched()[1]), \
                self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
            CUT.worker_flag({CUT.WORKER_KEY: 'strip'})


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class WiringMutationTests(unittest.IsolatedAsyncioTestCase):
    """A broken wiring fails: a request refused or failed closed, a startup refusal, or the strip checks."""
    maxDiff = None
    send = TC.EndToEndTests.send

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def outcome(self, attach=None, turns=3, started=None, **kw):
        """Three turns of one client on a strip server attached by ``attach(h)`` (None: as the runtime);
        ``started(srv)`` runs once the server is up."""
        srv = GH.strip_window_server(self, 'observation_query', slots=6, limit=2, attach=attach is None, **kw)
        if attach is not None:
            attach(srv.engine)
        if started is not None:
            started(srv)
        messages = [dict(role='user', content=T.first_prompt('Wiring.'))]
        for j in range(turns):
            status, body = await self.send(srv, messages)
            if status != 200:
                return 'status %d: %s' % (status, body['error']['message'])
            messages = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                                   dict(role='user', content=T.observation(j, size=j))]
        failures = srv.rows('failures.jsonl')
        if failures:
            return 'failure: %s' % json.dumps(failures[0])[:600]
        try:
            self.assertTrue(hasattr(srv.engine, 'recorder'), 'no strip attachment')
            GH.check_window_strip(self, srv)
            GH.boundary(self, srv)
        except (AssertionError, KeyError) as failure:
            return 'check: %s' % str(failure)[:300]
        return 'ok'

    async def test_unmutated_passes(self):
        self.assertEqual(await self.outcome(), 'ok')

    async def test_the_engine_attached_without_the_hold(self):
        """thinking_cut.attach_engine as before the integration (Qwen's attachment on a window adapter): the
        completion would commit at the cut with no hold slot; the first continuing turn is refused."""
        def attach(h):
            h.adapter.commit_at_cut, h.adapter.cut_records = True, {}
            h.core._remote_ra_live_v2_bridge.commit_at_cut = True
            CUT.attach_worker(h.owner, h.qwen, h.backend)
            h.recorder = GH.Recorder(h)
        found = await self.outcome(attach)
        self.assertNotEqual(found, 'ok')
        self.assertIn('without a free hold slot', found)

    async def test_the_worker_not_attached(self):
        def attach(h):
            CUT.attach_engine(h.adapter, h.core._remote_ra_live_v2_bridge)
            h.recorder = GH.Recorder(h)
        self.assertNotEqual(await self.outcome(attach), 'ok')

    async def test_the_engine_not_attached(self):
        def attach(h):
            CUT.attach_worker(h.owner, h.qwen, h.backend)
        self.assertNotEqual(await self.outcome(attach), 'ok')

    async def test_nothing_attached_serves_the_thinking_off_commit(self):
        """Neither side attached: the transcripts cut their streams but the engine commits at C'; refused."""
        self.assertNotEqual(await self.outcome(lambda h: None), 'ok')

    async def test_a_layout_without_hold_slots(self):
        with self.assertRaisesRegex(LiveV2Error, 'needs hold slots for the sliding window at the cut'):
            GH.strip_window_server(self, 'observation_query', slots=6, limit=2, hold=0)
        # Past the config check (mutated away), the engine attachment still refuses it.
        with patch.object(CUT, 'check_config', lambda config: True), \
                self.assertRaisesRegex(LiveV2Error, r'needs hold slots in its KV layout \(hold_slots >= 1\)'):
            GH.strip_window_server(self, 'observation_query', slots=6, limit=2, hold=0)

    async def test_a_reprefilled_piece_that_keeps_the_reasoning(self):
        """A keep-style piece (a thought channel before the content) at both ends. At startup the transcript's
        template check refuses it (the piece is not the stock thinking-on render of a past reply); swapped in
        after startup, the server runs but its prompts are no longer the stock render."""
        keep = staticmethod(lambda content: CUT.GEMMA4_REPLY_HEAD + '<|channel>thought\nkept<channel|>'
                            + CUT.gemma4_strip_content(content) + CUT.GEMMA4_REPLY_END)
        with patch.object(C.Gemma4Chat, 'reply_piece', keep), self.assertRaisesRegex(
                LiveV2Error, 'stock thinking-on history/header render differs from the strip stream'):
            await self.outcome()
        found = await self.outcome(started=lambda srv: self.enterContext(
            patch.object(C.Gemma4Chat, 'reply_piece', keep)))
        self.assertTrue(found.startswith('check: '), found)

    async def test_a_hold_released_at_the_install(self):
        """The hold taken and released at once (no reference through phase 2): a reply longer than the
        window loses the window at the cut before the commit."""
        def attach(h):
            GH.attach_strip(h)
            pins = h.adapter.window
            hold = pins.hold

            def dropped(ns, rid, end, source, cursor):
                value = hold(ns, rid, end, source, cursor)
                pins.release(rid)
                return value
            pins.hold = dropped
        found = await self.outcome(attach)
        self.assertNotEqual(found, 'ok')


if __name__ == '__main__':
    unittest.main()
