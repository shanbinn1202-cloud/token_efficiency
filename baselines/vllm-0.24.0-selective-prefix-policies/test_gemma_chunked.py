"""Gemma 4 31B IT, chunked phase 1 (gemma-goal, design WP10). CPU only.

A phase 1 longer than one native forward (8,192 tokens: about 10K-token observations,
which failed with PHASE1_TOO_LONG before) runs as forwards of at most 4,096 tokens on one
lease. The full-attention row holds the whole run from the first forward on; each
forward's sliding rows hold the window it reads and its new tokens, and the fresh sliding
blocks whose tokens left the window are reused for later tokens, so phase 1 holds at most
ceil((W - 1 + 4,096) / P) + 1 fresh blocks per sliding group (321 at P = 16, within the
one forward's transient of 512). Covered:

- the arithmetic: the forwards of a run, the fresh blocks they hold at once, the gate's
  figures for a 10K-token observation on the Gemma geometry;
- the lease: parse_lease accepts the engine's chunked lease and refuses later forwards
  that break the tiling, drop the carried window, reuse the borrowed window or a block
  still in the window, or take another group's block; the worker's owner check takes the
  phase-2 window from the last forward;
- end to end (test_gemma_e2e's engine and worker with vLLM's own KV manager and a runner
  that checks every full-attention and sliding read; forwards of 256 tokens, one native
  forward of 1,024): observations of up to about 2,600 tokens for full, observation_query
  and goal_query (whose Goal forward follows the chunks) over three sessions; every token
  is computed once, every forward reads exactly the last W - 1 tokens of each sliding group,
  the gate's figures bound what each turn takes from pool 1, and every session's tokens,
  selections, attended Pages and replies equal those of a server whose one native forward
  holds the whole observation. The default suite runs goal_query (the Goal forward follows
  the chunks); full and observation_query run with REMOTE_RA_SLOW_TESTS=1.
"""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_gemma_e2e as GE
import test_stageb_e2e as S
import test_stageb_worker as W

from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import sliding_window as SW
from remote_ra_live_v2 import window_capacity as WC
from remote_ra_live_v2.contract import LiveV2Error, parse_lease

try:
    import torch
except ImportError:
    torch = None

CHUNK = 256
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
LONG = [S.OUT % (0, 'q' * 2600), S.OUT % (1, 'Traceback: long ' + 'r' * 1500), S.OUT % (0, 'short'),
        S.OUT % (0, 's' * 1200 + ' src/app.py'), 'ok']


class ArithmeticTests(unittest.TestCase):
    def test_chunks_and_ring(self):
        self.assertEqual(SW.phase1_chunks(100, 8292, 8192, 4096), [(100, 8292)])
        self.assertEqual(SW.phase1_chunks(100, 8293, 8192, 4096), [(100, 4196), (4196, 8292), (8292, 8293)])
        self.assertEqual(MP.GEMMA4_PHASE1_CHUNK_TOKENS, 4096)
        # One forward holds every fresh block; chunks hold at most a window and a chunk.
        self.assertEqual(SW.phase1_ring(100, [(100, 8292)], 1024, 16), 519 - 7)
        self.assertEqual(SW.phase1_ring(100, SW.phase1_chunks(100, 8293, 8192, 4096), 1024, 16), 321)
        with self.assertRaises(LiveV2Error):
            SW.phase1_chunks(5, 5, 8192, 4096)

    def test_gate_figures_of_a_10k_observation(self):
        layout = I0.for_window_server(16, 16, 1024, 8192)
        need = WC.sliding_turn_need(layout, 5, cold=False, C=50000, e=60200, suffix=7)
        self.assertEqual(need, (5 * 320, 5 * 1))
        self.assertLessEqual(321, layout.transient_blocks)
        # Below one forward nothing changes: the run's fresh blocks.
        self.assertEqual(WC.sliding_turn_need(layout, 5, cold=False, C=50000, e=54300)[0], 5 * 269)
        self.assertEqual(WC.ledger_parts(layout, 5, C=50000, e=60200, suffix=7), (5 * 2, 5 * 320))


def harness(test, policy, tasks, n=None):
    output = Path(test.enterContext(tempfile.TemporaryDirectory()))
    layout = GE.window_layout(policy, len(tasks))
    value = GE.config(policy, output, list(tasks), layout)
    test.enterContext(patch('remote_ra_http.config.load', lambda: value))
    test.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
    return GE.Harness(policy, output, list(tasks))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChunkedEndToEndTests(unittest.TestCase):
    maxDiff = None
    check, windows = GE.GemmaEndToEndTests.check, GE.GemmaEndToEndTests.windows

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def play(self, policy, chunk=None, forward=None, leases=None, turns=5):
        """Three sessions, five turns each (``turns``), the long observations; returns per lane
        the tokens, replies, selections and attended Pages of every turn."""
        import importlib
        # The module sys.modules holds now (an earlier test's patch.dict may have dropped it).
        WP = importlib.import_module('remote_ra_live_v2.window_phase1')
        with patch.object(GE, 'N', forward or GE.N):
            if leases is not None:
                parse = WP.parse_lease

                def recording(raw, **kw):
                    leases.append(copy.deepcopy(raw))
                    return parse(raw, **kw)
                self.enterContext(patch.object(WP, 'parse_lease', recording))
            h = harness(self, policy, (GE.A, GE.B, GE.C))
            if chunk is not None:
                h.endpoint.phase1.chunk_tokens = chunk
            out = {lane.index: [] for lane in h.lanes}
            for k in range(turns):
                for lane in h.lanes:
                    observation = None if k == 0 else LONG[(k - 1 + lane.index) % len(LONG)]
                    info = h.turn(lane, observation, target=GE.TARGETS[(k + lane.index) % len(GE.TARGETS)],
                                  eos=(k + lane.index) % 3 == 2)
                    self.check(h, info)
                    out[lane.index].append((list(info['tokens']), info['committed']['text'], info['plan']['selected'],
                                            h.runner.visible[info['rid']], info['geo'].e - info['geo'].C))
            GE.GemmaEndToEndTests.close(self, h)
            return out, h

    def test_long_observations_run_in_chunks(self):
        self.long_observations(('goal_query',))

    @unittest.skipUnless(SLOW, 'about 25 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_long_observations_run_in_chunks_full_and_observation_query(self):
        self.long_observations(('full', 'observation_query'))

    def long_observations(self, policies):
        for policy in policies:
            with self.subTest(policy=policy):
                leases = []
                result, h = self.play(policy, chunk=CHUNK, leases=leases)
                chunked = [l for l in leases if 'chunks' in l]
                self.assertTrue(chunked)
                self.assertTrue(any(r[-1] > GE.N for runs in result.values() for r in runs))
                for lease in chunked:
                    bounds = [lease['start']] + [c[0] for c in lease['chunks']] + [lease['end']]
                    self.assertTrue(all(0 < b - a <= CHUNK for a, b in zip(bounds, bounds[1:])))
                    self.assertGreater(lease['end'] - lease['start'], GE.N)
                if policy == 'goal_query':
                    self.assertTrue(any(l['goal'] is not None for l in chunked))
                reference, _ = self.play(policy, forward=4096)
                self.assertEqual(result, reference)

    def test_the_lease_refuses_a_broken_chunk(self):
        leases = []
        self.play('observation_query', chunk=CHUNK, leases=leases)
        raw = next(l for l in leases if len(l.get("chunks", ())) >= 3)
        layout = GE.window_layout('observation_query', 3)
        lease = parse_lease(raw, attention_groups=(GE.FA,), layout=layout)
        self.assertEqual([f[:2] for f in lease.forwards()],
                         [(raw['start'], raw['chunks'][0][0])] + [tuple(c[:2]) for c in raw['chunks']])
        self.assertEqual(lease.final_rows, lease.forwards()[-1][2])
        keys = lease.block_keys()
        self.assertTrue(all((1, b) in keys for c in raw['chunks'] for row in c[2] for b in row if b))

        def refused(edit, pattern):
            value = copy.deepcopy(raw)
            edit(value)
            with self.assertRaisesRegex(LiveV2Error, pattern):
                parse_lease(value, attention_groups=(GE.FA,), layout=layout)
        last = len(raw['chunks']) - 1
        refused(lambda v: v['chunks'][1].__setitem__(0, v['chunks'][1][0] + 1), 'tile')
        refused(lambda v: v['chunks'].pop(), 'end at e')
        # A later forward's window must be the blocks the previous forward wrote.
        def drop(v):
            row = v['chunks'][1][2][0]
            head = SW.kept_blocks(v['chunks'][1][0], GE.WINDOW, GE.P)[0]
            row[head], row[-1] = row[-1], row[head]
        refused(drop, 'carry the window')
        # Reusing a block that is still in the window aliases two tokens.
        def alias(v):
            row = v['chunks'][1][2][0]
            head = SW.kept_blocks(v['chunks'][1][0], GE.WINDOW, GE.P)[0]
            row[-1] = row[head]
        refused(alias, 'distinct')
        # The borrowed window is the session's, never reused.
        def borrowed(v):
            head, (b0, b1), _ = SW.phase1_row(v['start'], v['chunks'][0][0], GE.WINDOW, GE.P)
            v['chunks'][last][2][0][-1] = v['rows'][GE.SLIDING[0]][b0]
        if SW.phase1_row(raw['start'], raw['chunks'][0][0], GE.WINDOW, GE.P)[1][1] > \
                SW.phase1_row(raw['start'], raw['chunks'][0][0], GE.WINDOW, GE.P)[1][0]:
            refused(borrowed, 'borrowed window')
        # Another sliding group's block.
        def foreign(v):
            other = v['chunks'][last][2][1]
            v['chunks'][last][2][0][-1] = other[-1]
        refused(foreign, "another group's block|distinct")
        # Chunks are the window layout's: a Mamba lease takes none.
        with self.assertRaisesRegex(LiveV2Error, 'window-layout lease'):
            parse_lease(raw, attention_groups=(GE.FA,), layout=I0.for_server(16, 3))


if __name__ == '__main__':
    unittest.main()
