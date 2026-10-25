"""Gemma 4 31B IT, goal_query (gemma-goal): the Goal forward's own sliding transient. CPU only.

goal_query's Goal forward is a separate cold forward of the maintained Goal's
``<retrieval_query>`` text (README 5). It runs after phase 1, inside the same lease, while
phase 1's fresh sliding blocks are still held, so on the window layout it needs blocks
of pool 1 beyond the one forward's transient that phase 1 uses. The layout reserves them:
``WindowLayout.goal_tokens`` G (``model_profile.GEMMA4_GOAL_TOKENS`` = 2,048 tokens),
ceil(G / P) blocks per sliding group. The Goal itself (its maintenance, the rewriter, the
prompts, the gate and its content) is unchanged: the engine feeds it the same TurnContext
from Gemma's transcript as from Qwen's. Covered:

- the layout: G validated (0, or a multiple of P up to one native forward); pool 1 = null
  + groups x (S slots + one forward's transient + ceil(G / P)); the vLLM switch and the
  startup config carry ``goal_tokens`` only when G > 0, so every other server's command,
  config and pools are unchanged; the record round-trips;
- the launcher: goal_query runs with the Goal transient pinned; the rewriter choice as for
  Qwen; at 16 sessions, 131,072 tokens, an 8,192-token budget and 4,096-token replies the
  transient adds 640 block ids (1,600 MiB) to pool 1, which leaves pool 0 2,858 Pages at
  24 GiB and 6,135 at 28 GiB, against 2,606 and 2,639 the selective rule needs (16, 21
  and 26 sessions fit 24, 28 and 32 GiB); the strict rule counts a Goal forward's Pages
  where they exceed a reply's; ``remote_ra_http.config.load`` checks the pin both ways;
- the gate: a goal_query turn whose selection is active asks ceil(G / P) global Pages
  and ceil(G / P) blocks per sliding group for its Goal forward; phase 1 refuses a longer
  Goal (PHASE1_TOO_LONG) before it allocates anything;
- end to end (test_gemma_release's sixteen sessions on the engine and worker fakes, pool
  0 at the selective rule's minimum, idle sessions' Pages released and readmitted): every
  goal_query turn completes and decides exactly as with a pool that never releases; the
  Goal forwards run in their own transient (at most ceil(G / P) blocks per sliding group,
  all freed at the drain); each Goal query vector is the pre-final-norm last row of its
  Goal forward, ranked in the Page vectors' projection (D-9). Without the Goal transient
  the same pools fail. The default suite runs six sessions of four turns; the sixteen
  sessions, compared with a pool that never releases, run with REMOTE_RA_SLOW_TESTS=1.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve
import test_gemma_capacity as GC
import test_gemma_e2e as GE
import test_gemma_release as GR
import test_gemma_startup as GS
import test_mswea_protocol as X
import test_stageb_worker as W

from remote_ra_http import config as HC
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import window_capacity as WC
from remote_ra_live_v2.contract import LiveV2Error

try:
    import torch
except ImportError:
    torch = None

SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
SHORT_SESSIONS, SHORT_TURNS = 6, 4      # the default suite's pressure runs
GiB, MiB = 1024 ** 3, 1024 ** 2
G = MP.GEMMA4_GEOMETRY
GOAL = MP.GEMMA4_GOAL_TOKENS
BLOCK, PAGE = G.sliding_block_bytes(16), G.page_bytes(16)


def pressure_run(test, policy, sessions, turns, blocks, release=True, observe=None, make=None, extra=None):
    """test_gemma_release's Pressure run with ``sessions`` lanes of ``turns`` turns (its checks after
    every turn and at the end): (per-lane result, harness, capacity events, engine cleanup). ``make``
    builds another harness from (policy, output, tasks, blocks, release, turns); ``extra`` updates
    the startup config."""
    import json
    import tempfile
    from contextlib import ExitStack
    from remote_ra_http.evidence import engine_closed
    with ExitStack() as stack:
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        tasks = GR.tasks(sessions)
        layout = GE.window_layout(policy, sessions)
        if policy in GE.QUERY:
            value = GR.config(policy, output, tasks, layout, release)
        else:
            value = GE.config(policy, output, tasks, layout)
            if release:
                value.update(release_idle_pages=True)
        value.update(extra or {})
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        h = (make or GR.Pressure)(policy, output, tasks, blocks, release, turns)
        if observe is not None:
            observe(h)
        h.serve(test.check)
        test.assertEqual(len(h.done), sessions * turns)
        test.boundary(h)
        result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane]
                  for lane in h.lanes}
        path = output / 'capacity.jsonl'
        events = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
        h.final_pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
        test.assertEqual(h.adapter.window.pinned_blocks(), 0)
        test.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                         (h.baseline, h.window_baseline))
        return result, h, events, cleanup


def layout(sessions=16, goal=GOAL, P=16):
    return I0.for_window_server(P, sessions, 1024, 8192, goal)


class LayoutTests(unittest.TestCase):
    def test_goal_transient(self):
        self.assertEqual(GOAL, 2048)
        plain, goal = layout(goal=0), layout()
        self.assertEqual((plain.goal_tokens, plain.goal_blocks), (0, 0))
        self.assertEqual((goal.goal_tokens, goal.goal_blocks), (2048, 128))
        self.assertEqual(layout(P=32).goal_blocks, 64)
        # Pool 1 = null + 5 x (16 x 65 + 512 + 128): 640 block ids (1,600 MiB) more.
        self.assertEqual((I0.window_pool_blocks(plain, 5), I0.window_pool_blocks(goal, 5)), (7761, 8401))
        self.assertEqual((I0.window_pool_blocks(goal, 5) - I0.window_pool_blocks(plain, 5)) * BLOCK, 1600 * MiB)
        self.assertEqual(5 * layout(P=32).goal_blocks * G.sliding_block_bytes(32), 1600 * MiB)
        for bad in (-16, 8, 2040, 8192 + 16):
            with self.assertRaisesRegex(ValueError, 'goal_tokens must be 0 or a multiple'):
                layout(goal=bad)
        with self.assertRaisesRegex(ValueError, 'goal_tokens must be an int'):
            layout(goal=2048.0)
        self.assertEqual(layout(goal=8192).goal_blocks, 512)

    def test_switch_record(self):
        plain, goal = layout(goal=0), layout()
        # Every other server's record is the one it was: no goal_tokens key.
        self.assertEqual(I0.to_additional_config(plain)[I0.LAYOUT_KEY],
                         dict(page_tokens=16, state='sliding', window_tokens=1024, window_slots=16,
                              forward_tokens=8192))
        record = I0.to_additional_config(goal)
        self.assertEqual(record[I0.LAYOUT_KEY]['goal_tokens'], 2048)
        parse = lambda value: I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: value}))
        self.assertEqual(parse(record[I0.LAYOUT_KEY]), goal)
        self.assertEqual(parse(I0.to_additional_config(plain)[I0.LAYOUT_KEY]), plain)
        for value, pattern in ((dict(record[I0.LAYOUT_KEY], goal_tokens=0), 'must be positive'),
                               (dict(record[I0.LAYOUT_KEY], goal_tokens=100), 'goal_tokens must be 0 or'),
                               (dict(record[I0.LAYOUT_KEY], goal=2048), 'exactly the keys')):
            with self.assertRaisesRegex(ValueError, pattern):
                parse(value)


class LauncherTests(unittest.TestCase):
    maxDiff = None
    setUp, prepare, load = GS.GemmaStartupTests.setUp, GS.GemmaStartupTests.prepare, GS.GemmaStartupTests.load

    def sixteen(self, **kw):
        value = dict(task=GS.tasks(16), policy='goal_query', release_idle_pages=True, dram_arena_gib=79,
                     selection=dict(active_token_budget=8192, no_goal_rewriter=True), kv_cache_gib=24)
        value.update(kw)
        return self.prepare(**value)

    def test_goal_query_is_served_with_its_goal_transient(self):
        for gib in (24, 28):
            _, config, command, env = self.sixteen(kv_cache_gib=gib)
            argv = command[command.index('--model'):]
            additional = json.loads(argv[argv.index('--additional-config') + 1])
            # The selective rule (gemma-mab) also pins its figure for vLLM's startup check.
            request = WC.pinned_request_pages(layout(), 8192, 4096, gib << 30)
            self.assertEqual(additional, {I0.LAYOUT_KEY: dict(page_tokens=16, state='sliding', window_tokens=1024,
                                                              window_slots=16, forward_tokens=8192,
                                                              goal_tokens=2048, request_pages=request)})
            self.assertEqual(config['kv_layout'], additional[I0.LAYOUT_KEY])
            self.assertEqual(config['effective_server_config']['additional_config'], additional)
            self.assertEqual(config['policy'], 'goal_query')
            self.assertEqual(config['goal_rewriter'], dict(used=False, reason='disabled by --no-goal-rewriter',
                                                           requested_model=None))
            self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
            value = self.load(config)
            import dataclasses
            self.assertEqual(HC.kv_layout(value), dataclasses.replace(layout(), request_pages=request))
        # The rewriter is pinned as for Qwen (its checkpoint's config.json).
        rewriter = self.root / 'qwen3-0.6b'
        rewriter.mkdir()
        (rewriter / 'config.json').write_text('{"model_type": "qwen3"}')
        _, config, _, _ = self.sixteen(selection=dict(active_token_budget=8192, goal_rewriter_model=str(rewriter)))
        self.assertIs(config['goal_rewriter']['used'], True)
        self.assertEqual(config['selection_contract']['goal_rewriter']['model'], str(rewriter.resolve()))
        self.load(config)
        with self.assertRaisesRegex(ValueError, 'goal_query requires --goal-rewriter-model'):
            self.sixteen(selection=dict(active_token_budget=8192))
        # Every other policy's layout, command and config stay without the Goal transient.
        _, other, command, _ = self.sixteen(policy='observation_query', selection=dict(active_token_budget=8192))
        self.assertNotIn('goal_tokens', other['kv_layout'])
        self.assertNotIn('goal_tokens', command[command.index('--additional-config') + 1])

    def test_load_checks_the_goal_pin_both_ways(self):
        _, config, _, _ = self.sixteen()
        _, other, _, _ = self.sixteen(policy='observation_query', selection=dict(active_token_budget=8192))
        for value in (dict(config, kv_layout={k: v for k, v in config['kv_layout'].items() if k != 'goal_tokens'}),
                      dict(config, kv_layout=dict(config['kv_layout'], goal_tokens=4096)),
                      dict(other, kv_layout=dict(other['kv_layout'], goal_tokens=2048))):
            with self.assertRaisesRegex(RuntimeError, 'KV layout differs from the server layout'):
                self.load(value)
        # The layout must also be the pinned vLLM argument.
        value = dict(config)
        value['effective_server_config'] = dict(config['effective_server_config'],
                                                additional_config=other['effective_server_config']['additional_config'])
        value['server_config_sha256'] = serve.pin(value['effective_server_config'])
        with self.assertRaisesRegex(RuntimeError, 'pinned vLLM arguments'):
            self.load(value)

    def test_memory_and_session_limits(self):
        common = dict(page_tokens=16, window_tokens=1024, forward_tokens=8192, groups=5, block_bytes=BLOCK,
                      page_bytes=PAGE, max_model_len=131072, max_tokens=4096)
        rows = {}
        for goal in (0, GOAL):
            out = rows[goal] = {}
            for gib in (24, 28, 32):
                blocks = WC.pool_zero_blocks(gib * GiB, layout(goal=goal), 5, BLOCK, PAGE)
                out[gib] = (blocks, WC.selective_pages(layout(goal=goal), 8192, 4096, blocks))
            out['most'] = {gib: WC.max_sessions(gib * GiB, active_tokens=8192, goal_tokens=goal, **common)
                           for gib in (24, 28, 32)}
        # 1,600 MiB of pool 1 are 1,280 global Pages of 1.25 MiB at every KV size.
        self.assertEqual({gib: rows[0][gib][0] - rows[GOAL][gib][0] for gib in (24, 28, 32)},
                         {24: 1280, 28: 1280, 32: 1280})
        self.assertEqual((rows[GOAL][24], rows[GOAL][28]), ((2858, 2606), (6135, 2639)))
        self.assertEqual((rows[0][24], rows[0][28]), ((4138, 2619), (7415, 2652)))
        self.assertEqual(rows[GOAL]['most'], {24: 16, 28: 21, 32: 26})
        self.assertEqual(rows[0]['most'], {24: 18, 28: 23, 32: 28})
        # The selective turn itself is unchanged: its phase-2 peak exceeds phase 1 with a Goal.
        self.assertEqual(WC.selective_turn_pages(layout(), 8192, 4096), 2563)
        # 16 goal_query sessions need 23.70 GiB (observation_query: 22.13).
        smallest = lambda kv: WC.pool_zero_blocks(kv, layout(), 5, BLOCK, PAGE)
        enough = lambda kv: smallest(kv) - 1 >= WC.selective_pages(layout(), 8192, 4096, smallest(kv))
        self.assertTrue(enough(int(23.70 * GiB)))
        self.assertFalse(enough(int(23.69 * GiB)))
        # A Goal transient of one whole forward (8,192 tokens, 6.25 GiB) would not fit 16 sessions at 28 GiB.
        self.assertEqual(WC.max_sessions(28 * GiB, active_tokens=8192, goal_tokens=8192, **common), 15)
        # The launcher's refusal names the goal_query limit.
        with patch.object(serve, 'GEMMA4_KV_CACHE_GIB', (20, 24, 28, 32)), \
                self.assertRaisesRegex(ValueError, r'\(8401 block ids\).*use at most 11 sessions'):
            self.sixteen(kv_cache_gib=20)

    def test_strict_rule_counts_the_goal_pages(self):
        # One session at the context limit: a goal_query phase 1 holds the Goal forward's 128
        # Pages where a 1,536-token reply holds 97; with 4,096-token replies (257) nothing changes.
        self.assertEqual(WC.one_session_pages(layout(1), 131072, 1536, 0) -
                         WC.one_session_pages(layout(1, goal=0), 131072, 1536, 0), 128 - 97)
        self.assertEqual(WC.one_session_pages(layout(1), 131072, 4096, 0),
                         WC.one_session_pages(layout(1, goal=0), 131072, 4096, 0))
        self.prepare(GS.tasks(6), 'goal_query', dict(no_goal_rewriter=True), kv_cache_gib=24, dram_arena_gib=60)
        with self.assertRaisesRegex(ValueError, 'one session at the 131072-token limit.*use at most 6 sessions'):
            self.prepare(GS.tasks(7), 'goal_query', dict(no_goal_rewriter=True), kv_cache_gib=24, dram_arena_gib=60)


class GateTests(unittest.TestCase):
    """capacity.turn_need_by_pool for a goal_query turn on the window layout."""

    def core(self, goal_tokens, contract):
        base = GC.GateTests()
        base.layout = layout(12, goal=goal_tokens)
        core = base.core({'s': GC.GateTests.session(50000, policy='goal_query', resident=range(100))},
                         {'s': GC.GateTests.transcript()})
        endpoint = core._remote_ra_private_v2_endpoint
        endpoint.selection = NS(contract=contract)
        endpoint.phase1.goal_tokens = goal_tokens
        return core

    def test_a_goal_forward_takes_its_transient(self):
        from remote_ra_live_v2 import capacity as CAP
        from remote_ra_live_v2 import selection_contract as sc
        contract = sc.default_contract(16, vector_format=sc.COMPACT_INT8_V1, active_ratio=None,
                                       active_token_budget=8192)
        tokens = [1] * 54307
        (fa1, s1), (fa2, s2) = CAP.turn_need_by_pool(self.core(GOAL, contract), 's', tokens, None, 4096)
        (fa0, s0), (fb2, sb2) = CAP.turn_need_by_pool(self.core(0, contract), 's', tokens, None, 4096)
        # Phase 1: 269 fresh blocks per group and the Goal forward's 128 (global and sliding).
        self.assertEqual((fa1 - fa0, s1 - s0, s1), (128, 5 * 128, 5 * (269 + 128)))
        self.assertEqual((fa2, s2), (fb2, sb2))
        # Below the switch threshold no Goal forward runs: nothing is asked for it.
        idle = sc.default_contract(16, vector_format=sc.COMPACT_INT8_V1)
        idle['switch_threshold_tokens'] = 1 << 20
        self.assertEqual(CAP.turn_need_by_pool(self.core(GOAL, idle), 's', tokens, None, 4096)[0][1], 5 * 269)
        # The sliding need refuses a Goal beyond the transient.
        with self.assertRaisesRegex(LiveV2Error, 'exceeds the Goal transient'):
            WC.sliding_turn_need(layout(12), 5, cold=False, C=50000, e=50300, goal_tokens=GOAL + 1)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class SixteenGoalSessionsTests(unittest.TestCase):
    """test_gemma_release's sixteen sessions with goal_query: the Goal transient of the test
    geometry is 384 tokens (24 blocks of 16 per sliding group; the fakes' Goals reach 359)."""
    maxDiff = None
    TURNS = 5
    TEST_GOAL = 384
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of
    run_case = GR.SixteenSessionTests.run_case

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.enterContext(patch.object(GE, 'GOAL', self.TEST_GOAL))
        self.layout = GE.window_layout('goal_query', GR.SESSIONS)
        self.assertEqual(self.layout.goal_blocks, 24)
        self.small = GR.selective_blocks(self.layout)
        self.short = GR.selective_blocks(GE.window_layout('goal_query', SHORT_SESSIONS))

    def watch(self, records):
        """Record each Goal forward (its tokens, sliding rows and query vector) and every
        selection's projection."""
        from runtime_renderer import PLAIN_GOAL_SCHEMA

        def observe(h):
            run, plan = h.endpoint.phase1.run, h.selection.plan_turn

            def running(binding, geo, *, allocation_generation, goal_payload=None, _lease_only=False):
                free = h.window_pool.get_num_free_blocks()
                result = run(binding, geo, allocation_generation=allocation_generation, goal_payload=goal_payload,
                             _lease_only=_lease_only)
                if goal_payload is not None:
                    tokens = list(h.endpoint.phase1.renderer(dict(schema=PLAIN_GOAL_SCHEMA, text=goal_payload)).token_ids)
                    records.append(dict(kind='goal', request=binding.request, tokens=tokens, before=free,
                                        after=h.window_pool.get_num_free_blocks()))
                return result

            def planning(binding, geo, generation, query, phase1):
                out = plan(binding, geo, generation, query, phase1)
                selection = out[2]['body']['selection']
                records.append(dict(kind='plan', request=binding.request, goal=None if phase1 is None else
                                    phase1.goal_vector, text=query.text, forward=out[2]['body']['goal']['forward'],
                                    projection=selection.get('projection_sha256')))
                return out
            h.endpoint.phase1.run = running
            h.selection.plan_turn = planning
            self.assertEqual(type(h.endpoint.phase1).__name__, 'WindowPhase1Coordinator')
        return observe

    def goal_checks(self, records, at_least):
        """Goal forwards ran, each in the Goal transient (its blocks were taken during phase 1
        and none is held after the drain: the boundary check at the end of every turn), and
        each Goal query vector is the pre-final-norm last row of its Goal forward (the fake
        runner's hidden row [x + 1, token + 1, 1] of its last token, gain 1), ranked in the
        projection of the Page vectors (D-9)."""
        from remote_ra_live_v2.page_compaction import PageCompactor
        from remote_ra_live_v2.query_selection import normalized
        goals = [r for r in records if r['kind'] == 'goal']
        plans = {r['request']: r for r in records if r['kind'] == 'plan'}
        self.assertGreaterEqual(len(goals), at_least)
        projection = PageCompactor(GE.HIDDEN, GE.P).projection_sha256
        for record in goals:
            plan, tokens = plans[record['request']], record['tokens']
            self.assertLessEqual(len(tokens), self.TEST_GOAL)
            self.assertTrue(plan['forward'])
            self.assertEqual(list(plan['goal']), list(normalized([float(len(tokens)), tokens[-1] + 1., 1.])))
            self.assertEqual(plan['projection'], projection)
            self.assertEqual(tokens, list(X.TOK.encode(plan['text'])))
        return goals

    def test_goal_query_sessions_with_releases(self):
        # The default suite's variant: six sessions of four turns at the selective rule's minimum.
        records = []
        _, h, events, cleanup = pressure_run(self, 'goal_query', SHORT_SESSIONS, SHORT_TURNS, self.short,
                                             observe=self.watch(records))
        self.assertTrue([e for e in events if e['event'] == 'evict'] and
                        [e for e in events if e['event'] == 'readmit'])
        self.assertEqual(cleanup['idle_release']['evictions'],
                         len([e for e in events if e['event'] == 'evict']))
        self.goal_checks(records, 3)

    @unittest.skipUnless(SLOW, 'about 25 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_goal_query_sessions_with_releases(self):
        reference, big, events, _, cleanup = self.run_case('goal_query', GR.REFERENCE_BLOCKS, True)
        self.assertEqual(events, [])
        self.assertGreater(big.final_pins, self.small)
        records = []
        result, h, events, natives, cleanup = self.run_case('goal_query', self.small, True, self.watch(records))
        # Every session saw, selected, attended and replied exactly as with the large pool.
        self.assertEqual(result, reference)
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        self.assertEqual(cleanup['idle_release']['evictions'], len(evictions))
        goals = self.goal_checks(records, 11)
        self.assertGreater(max(len(r['tokens']) for r in goals), self.TEST_GOAL - 2 * GE.P)

    def test_without_the_goal_transient_the_same_pools_fail(self):
        # Hold the Goal transient's blocks from the start: pool 1 as without it. The other sessions
        # pin their windows and Pages, so the gate checks a Goal turn even while nothing runs
        # (integrate-1018) and refuses it for the sliding pool: every lane waits. Before, such a turn
        # was admitted unchecked and its phase 1 failed ("insufficient native blocks for phase 1:
        # FA .. free, sliding ..").
        def steal(h):
            self.stolen = h.window_pool.get_new_blocks(len(GE.SLIDING) * self.layout.goal_blocks)
        with self.assertRaisesRegex(AssertionError, 'no request runs and the gate refused every lane'):
            pressure_run(self, 'goal_query', SHORT_SESSIONS, SHORT_TURNS, self.short, observe=steal)

    def test_a_goal_longer_than_the_transient_fails_before_it_allocates(self):
        def shrink(h):
            h.endpoint.phase1.goal_tokens = GE.P
            free = []
            run = h.endpoint.phase1.run

            def running(binding, geo, **kw):
                free.append(tuple(h.kv.get_num_free_blocks_by_pool()))
                try:
                    return run(binding, geo, **kw)
                except LiveV2Error:
                    self.assertEqual(tuple(h.kv.get_num_free_blocks_by_pool()), free[-1])
                    raise
            h.endpoint.phase1.run = running
        with self.assertRaisesRegex(Exception, r'PHASE1_TOO_LONG: Goal forward of \d+ tokens exceeds the Goal '
                                               r'transient of the window layout \(16 tokens\)'):
            pressure_run(self, 'goal_query', SHORT_SESSIONS, SHORT_TURNS, GR.REFERENCE_BLOCKS, observe=shrink)


if __name__ == '__main__':
    unittest.main()
