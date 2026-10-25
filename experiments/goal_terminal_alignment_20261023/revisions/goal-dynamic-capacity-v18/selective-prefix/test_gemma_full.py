"""Gemma 4 31B IT, `full` at 16 sessions with --release-idle-pages (gemma-goal). CPU only.

`full` keeps every Page of a session's history resident, so 16 idle `full` sessions pin
far more than any pool 0. With --release-idle-pages the capacity gate releases idle
sessions' global Pages under pressure (their sliding windows stay in pool 1) and each
released session readmits its whole history before its next phase 1. The launcher then
sizes pool 0 by the release rule (window_capacity.release_pages): one session's worst
turn at the context limit, the other sessions' tails (never released) and the watermark.
Covered:

- the launcher: the release rule for `full` (and every policy without a fixed budget) at
  16 sessions, 131,072 tokens and 4,096-token replies: 9,018 / 9,051 / 9,083 global Pages
  needed against 4,138 / 7,415 / 10,692 at 24 / 28 / 32 GiB, so 32 GiB (16 sessions need
  30.02 GiB, 30,738 MiB; 24, 28 and 32 GiB hold at most 8, 13 and 18); at 65,536 tokens 28 GiB
  holds 19. full runs at --kv-cache-gib 32 (serve.GEMMA4_FULL_KV_CACHE_GIB, the user's decision,
  as Qwen): 1,608 Pages above the rule, and a smaller size's refusal names it;
- full never starts the Goal rewriter: no selection contract, no rewriter record, no Goal
  transient in the window layout, and the engine's selection authority holds none;
- end to end (test_gemma_release's sixteen sessions on the engine and worker fakes, pool 0
  at the release rule's minimum for a test context limit of 8,192 tokens): every `full`
  turn completes and reads exactly what it reads with a pool that never releases; victims
  are idle sessions only; a readmission restores exactly the released Pages (every sealed
  Page of the session) with their KV; windows and pool 1 are untouched; both pools return
  to their baselines. Without the switch the same pool cannot serve the sixteen sessions.
  The default suite runs six sessions of four turns at the release rule's minimum; the
  sixteen sessions, compared with a pool that never releases, run with REMOTE_RA_SLOW_TESTS=1.
"""
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve
import test_gemma_goal as GG
import test_gemma_e2e as GE
import test_gemma_release as GR
import test_gemma_startup as GS
import test_stageb_worker as W

from remote_ra_live_v2 import idle_release as R
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import window_capacity as WC

try:
    import torch
except ImportError:
    torch = None

GiB, MiB = 1024 ** 3, 1024 ** 2
G = MP.GEMMA4_GEOMETRY
BLOCK, PAGE = G.sliding_block_bytes(16), G.page_bytes(16)
TEST_LIMIT = 8192                  # the test geometry's context limit for the release rule


def release_blocks(layout, limit=TEST_LIMIT, reply=GR.REPLY_BOUND):
    """The smallest pool 0 the release rule accepts for the test geometry."""
    blocks = 1
    while blocks - 1 < WC.release_pages(layout, limit, reply, blocks):
        blocks += 1
    return blocks


class LauncherTests(unittest.TestCase):
    maxDiff = None
    setUp, prepare, load = GS.GemmaStartupTests.setUp, GS.GemmaStartupTests.prepare, GS.GemmaStartupTests.load

    def test_release_rule_for_sixteen_full_sessions(self):
        sixteen = dict(task=GS.tasks(16), policy='full', release_idle_pages=True, dram_arena_gib=79)
        self.assertEqual(serve.GEMMA4_FULL_KV_CACHE_GIB, 32)
        _, config, command, _ = self.prepare(kv_cache_gib=serve.GEMMA4_FULL_KV_CACHE_GIB, **sixteen)
        self.assertIs(config['release_idle_pages'], True)
        self.assertEqual(command[command.index('--max-num-seqs') + 1], '16')
        self.assertIsNone(config['selection_contract'])
        self.load(config)
        for gib, need, most in ((24, 9018, 8), (28, 9051, 13)):
            with self.assertRaisesRegex(ValueError, 'one session at the 131072-token limit and 15 other tails needs '
                                                    '%d: use at most %d sessions or --kv-cache-gib 32$'
                                        % (need, most)):
                self.prepare(kv_cache_gib=gib, **sixteen)
        # Every policy without a fixed budget takes the same rule (random50, --active-ratio).
        self.prepare(kv_cache_gib=32, **dict(sixteen, policy='random50'))
        self.prepare(kv_cache_gib=32, **dict(sixteen, policy='observation_query', selection=dict(active_ratio=.4)))
        # Without the switch the strict rule (no tails) holds, and 16 sessions fit 32 GiB too,
        # but then nothing frees an idle session's Pages.
        _, config, _, _ = self.prepare(kv_cache_gib=32, **dict(sixteen, release_idle_pages=False))
        self.assertNotIn('release_idle_pages', config)

    def test_figures(self):
        layout = I0.for_window_server(16, 16, 1024, 8192)
        common = dict(page_tokens=16, window_tokens=1024, forward_tokens=8192, groups=5, block_bytes=BLOCK,
                      page_bytes=PAGE, max_model_len=131072, max_tokens=4096)
        rows = {}
        for gib in (24, 28, 32):
            blocks = WC.pool_zero_blocks(gib * GiB, layout, 5, BLOCK, PAGE)
            rows[gib] = (blocks, WC.release_pages(layout, 131072, 4096, blocks),
                         WC.one_session_pages(layout, 131072, 4096, blocks))
        self.assertEqual(rows, {24: (4138, 9018, 9003), 28: (7415, 9051, 9036), 32: (10692, 9083, 9068)})
        self.assertEqual({gib: WC.max_sessions(gib * GiB, release=True, **common) for gib in (24, 28, 32)},
                         {24: 8, 28: 13, 32: 18})
        self.assertEqual({gib: WC.max_sessions(gib * GiB, release=True, **dict(common, max_model_len=65536))
                          for gib in (24, 28, 32)}, {24: 14, 28: 19, 32: 24})
        smallest = lambda kv: WC.pool_zero_blocks(kv, layout, 5, BLOCK, PAGE)
        enough = lambda kv: smallest(kv) - 1 >= WC.release_pages(layout, 131072, 4096, smallest(kv))
        self.assertTrue(enough(int(30.02 * GiB)))
        self.assertFalse(enough(int(30.01 * GiB)))
        # The exact minimum (1 MiB steps) and the margin of the pinned 32 GiB.
        self.assertEqual((enough(30738 * MiB), enough(30737 * MiB)), (True, False))
        self.assertEqual(rows[32][0] - 1 - rows[32][1], 1608)
        self.assertAlmostEqual(1608 * PAGE / GiB, 1.96, places=2)

    def test_full_never_starts_the_goal_rewriter(self):
        sixteen = dict(task=GS.tasks(16), policy='full', release_idle_pages=True, dram_arena_gib=79)
        _, config, command, _ = self.prepare(kv_cache_gib=serve.GEMMA4_FULL_KV_CACHE_GIB, **sixteen)
        self.assertIsNone(config['selection_contract'])
        self.assertIsNone(config['goal_rewriter'])
        self.assertNotIn('goal_rewriter_batch', config)
        self.assertNotIn(I0.GOAL_FIELD, config['effective_server_config']['additional_config'][I0.LAYOUT_KEY])
        self.assertFalse([a for a in command if 'rewriter' in a.lower()])
        self.load(config)
        with self.assertRaisesRegex(ValueError, 'goal-rewriter-batch'):
            self.prepare(kv_cache_gib=32, goal_rewriter_batch=4, **sixteen)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class SixteenFullSessionsTests(unittest.TestCase):
    maxDiff = None
    TURNS = 5
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.layout = GE.window_layout('full', GR.SESSIONS)
        self.small = release_blocks(self.layout)

    def readmissions(self, readmitted):
        """An observer: full's readmission brings back every sealed Page, with its KV."""
        def observe(h):
            release = h.core._remote_ra_idle_release
            readmit = release.readmit

            def readmitting(binding, session):
                sealed = session.sealed_end // GE.P
                pages = readmit(binding, session)
                self.assertEqual(list(pages), list(range(sealed)))
                lane = next(l for l in h.lanes if l.sid == binding.session)
                stream = list(lane.frontend.stream)
                for page in pages:
                    block = h.adapter.owner.sessions[session.namespace][page][0][0].block_id
                    self.assertEqual(h.store.kv[(GE.FA, block)],
                                     [(x, stream[x]) for x in range(page * GE.P, (page + 1) * GE.P)])
                readmitted.append(len(pages))
                return pages
            release.readmit = readmitting
        return observe

    def test_full_sessions_with_releases(self):
        # The default suite's variant: six sessions of four turns at the release rule's minimum.
        readmitted = []
        small = release_blocks(GE.window_layout('full', GG.SHORT_SESSIONS))
        # full never starts the Goal rewriter (selection_contract.uses_goal_rewriter).
        QS = importlib.import_module('remote_ra_live_v2.query_selection')
        with patch.object(QS, 'goal_rewriter', side_effect=AssertionError('full started a Goal rewriter')):
            _, h, events, cleanup = GG.pressure_run(self, 'full', GG.SHORT_SESSIONS, GG.SHORT_TURNS, small,
                                                    observe=self.readmissions(readmitted))
        self.assertIsNone(h.selection.rewriter)
        self.assertLessEqual(max(len(i['tokens']) for i in h.done) + GR.REPLY_BOUND, TEST_LIMIT)
        for info in h.done:
            self.assertEqual(info['plan']['selected'], list(range(info['geo'].n)))
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue([e for e in events if e['event'] == 'evict'] and readmissions)
        self.assertEqual([e['pages'] for e in readmissions], readmitted)
        self.assertEqual(cleanup['idle_release']['readmitted_pages'], sum(readmitted))

    def run_case(self, blocks, release, observe=None):
        from remote_ra_http.evidence import engine_closed
        with ExitStack() as stack:
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = GE.config('full', output, GR.tasks(GR.SESSIONS), self.layout)
            if release:
                value.update(release_idle_pages=True)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = GR.Pressure('full', output, GR.tasks(GR.SESSIONS), blocks, release, self.TURNS)
            if observe is not None:
                observe(h)
            h.serve(self.check)
            self.assertEqual(len(h.done), GR.SESSIONS * self.TURNS)
            self.boundary(h)
            # full keeps every sealed Page resident.
            for info in h.done:
                self.assertEqual(info['plan']['selected'], list(range(info['geo'].n)))
            result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                    h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane]
                      for lane in h.lanes}
            prompts = max(len(i['tokens']) for i in h.done)
            path = output / R.EVENTS
            events = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
            h.final_pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
            h.lifecycle.disconnect()
            h.sessions.close()
            engine_closed(h.core)
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual(h.adapter.window.pinned_blocks(), 0)
            self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                             (h.baseline, h.window_baseline))
            return result, h, events, cleanup, prompts

    @unittest.skipUnless(GG.SLOW, 'about 30 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_full_sessions_with_releases(self):
        reference, big, events, cleanup, prompts = self.run_case(GR.REFERENCE_BLOCKS, True)
        self.assertEqual(events, [])
        self.assertLessEqual(prompts + GR.REPLY_BOUND, TEST_LIMIT)        # the test's limit holds every turn
        self.assertGreater(big.final_pins, self.small)                     # the idle pins exceed the pool
        readmitted = []
        result, h, events, cleanup, _ = self.run_case(self.small, True, self.readmissions(readmitted))
        self.assertEqual(result, reference)
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        self.assertEqual([e['pages'] for e in readmissions], readmitted)
        self.assertEqual(cleanup['idle_release']['readmitted_pages'], sum(readmitted))
        for e in events:
            self.assertEqual(e['free_blocks_by_pool'][1], e['free_blocks_by_pool_before'][1])
        # Without the switch the same pool cannot serve the sixteen sessions: once the idle sessions'
        # pins leave a turn no room, nothing runs and the gate refuses every lane (integrate-1018:
        # the window layout checks a lone turn under other sessions' pins). Before, such a turn was
        # admitted unchecked and found no blocks ("insufficient native blocks", "found no free block").
        with self.assertRaisesRegex(AssertionError, 'no request runs and the gate refused every lane'):
            self.run_case(self.small, False)


if __name__ == '__main__':
    unittest.main()
