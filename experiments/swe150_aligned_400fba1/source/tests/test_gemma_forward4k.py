"""Gemma 4 31B IT with 4,096-token native forwards (gemma-forward4k). CPU only.

The group's servers run max_num_batched_tokens 4,096. The Gemma profile's one native forward
is 8,192 tokens by default (unchanged); serve.py --max-num-batched-tokens 4096 sets vLLM's
--max-num-batched-tokens and the window layout's forward_tokens to 4,096. Covered:

- the layout: pool 1 holds per sliding group one window slot per session and one 4,096-token
  forward's transient, 256 blocks at P = 16 (8,192: 512), so 16 sessions' pool 1 is 6,481 block
  ids, 15.82 GiB (18.95 GiB at 8,192) and the global pool gains 3.125 GiB (2,560 Pages); vLLM's
  planner sizes the pools so; the pool-0 rules take the 4,096-token forward; the capacity figures
  for 16 sessions at 131,072 tokens at 24, 25, 28 and 32 GiB;
- chunked phase 1: forwards of N - W - P tokens (3,056 at P = 16, 3,040 at P = 32), the largest
  whose ring of fresh sliding blocks fits the forward's transient (8,192: 4,096, unchanged);
  4,096-token chunks would hold up to 321 blocks per group, more than the transient of 256, so
  with every window slot pinned such a turn could never be admitted. On the engine and worker
  fakes at the same ratio (N = 4 W) the long observations run in those chunks, every turn equal to
  a one-forward server's, while chunks of one native forward wait for the sliding pool with nothing
  running (CAPACITY_WAIT, forever: no release frees a window);
- the launcher: 4096 changes exactly --max-num-batched-tokens, effective_server_config's
  max_num_batched_tokens and the layout's forward_tokens (under the selective rule also its
  request_pages) of test_gemma_flags' launches, the batched Goal rewriter's and the kv-session and
  chat-text protocols'; 8192 (or no option) gives the default launch byte for byte; other values
  and a Qwen release are refused; the dry-run CLI;
- startup: remote_ra_http.config.load and the frontend's argument check accept the 4,096 launches
  and refuse a layout with another forward or a --max-num-batched-tokens that differs from it.
"""
from contextlib import redirect_stdout
import copy
import dataclasses
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_gemma_chunked as GC  # noqa: E402
import test_gemma_e2e as GE  # noqa: E402
import test_gemma_flags as GF  # noqa: E402
import test_gemma_kv_pools as GK  # noqa: E402
from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import capacity as CAP  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2 import sliding_window as SW  # noqa: E402
from remote_ra_live_v2 import window_capacity as WC  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
GiB, MiB = 1024 ** 3, 1024 ** 2
G = MP.GEMMA4_GEOMETRY
W_ = G.sliding_window
FORWARD = 4096
GROUPS = len(G.sliding_groups())


def layout(P=16, sessions=16, forward=FORWARD, goal=0):
    return I0.for_window_server(P, sessions, W_, forward, goal)


def common(P=16, forward=FORWARD, goal=0):
    return dict(page_tokens=P, window_tokens=W_, forward_tokens=forward, groups=GROUPS,
                block_bytes=G.sliding_block_bytes(P), page_bytes=G.page_bytes(P), max_model_len=131072,
                max_tokens=4096, goal_tokens=goal)


# ---- the layout and its capacity ----------------------------------------------------------------

class LayoutTests(unittest.TestCase):
    def test_choices(self):
        self.assertEqual(MP.GEMMA4_FORWARD_TOKENS, 8192)
        self.assertEqual(MP.GEMMA4_FORWARD_TOKEN_CHOICES, (8192, 4096))
        self.assertEqual(MP.GEMMA4_PHASE1_CHUNK_TOKENS, 4096)
        self.assertLessEqual(MP.GEMMA4_GOAL_TOKENS, FORWARD)

    def test_pool_one(self):
        # P = 16: a slot is 65 blocks per sliding group; one 4,096-token forward adds
        # ceil((1,023 + 4,096) / 16) + 1 - 65 = 256 (8,192: 512). 16 windows: 6,481 block ids of
        # 2.5 MiB, 15.82 GiB; goal_query's Goal transient adds 640 (1,600 MiB) as at 8,192.
        for P, slot, transient, ids, goal_ids in ((16, 65, 256, 6481, 7121), (32, 33, 128, 3281, 3601)):
            with self.subTest(P=P):
                lay, before = layout(P), layout(P, forward=8192)
                self.assertEqual((lay.window_blocks, lay.transient_blocks), (slot, transient))
                self.assertEqual(before.transient_blocks, 2 * transient)
                self.assertEqual(I0.window_pool_blocks(lay, GROUPS), ids)
                self.assertEqual(I0.window_pool_blocks(layout(P, goal=MP.GEMMA4_GOAL_TOKENS), GROUPS), goal_ids)
                freed = (I0.window_pool_blocks(before, GROUPS) - ids) * G.sliding_block_bytes(P)
                self.assertEqual(freed, 3200 * MiB)                       # 3.125 GiB at either Page size
                self.assertEqual(freed // G.page_bytes(P), 2560 * 16 // P)  # global Pages it gives pool 0
        self.assertAlmostEqual(6481 * G.sliding_block_bytes(16) / GiB, 15.82, places=2)
        self.assertAlmostEqual(7121 * G.sliding_block_bytes(16) / GiB, 17.39, places=2)
        # The record names the forward; it parses back.
        record = I0.to_additional_config(lay := layout())
        self.assertEqual(record[I0.LAYOUT_KEY]['forward_tokens'], FORWARD)
        self.assertEqual(I0.from_vllm_config(NS(additional_config=record)), lay)

    def test_vllm_planner(self):
        # vLLM's planner (the overlay's decoupled path) reserves pool 1 for the 4,096-token forward
        # and gives pool 0 the rest; its own startup check (one request at max_model_len next to
        # pool 1) needs 10 + 15.82 GiB for 16 windows at 131,072 tokens (8,192: 28.95 GiB), and the
        # selective rule's request_pages (2,141 at 25 GiB) 18.44 GiB.
        w = GK.gemma_world()
        lay = layout()
        _, config = GK.plan(w, lay, 32 * GiB, batched=FORWARD)
        self.assertEqual(config.num_blocks_by_pool, ((32 * GiB - 6481 * G.sliding_block_bytes(16))
                                                     // G.page_bytes(16), 6481))
        self.assertEqual(config.num_blocks_by_pool[0], 13252)
        with self.assertRaisesRegex(ValueError, 'max seq len'):
            GK.plan(w, lay, 25 * GiB, batched=FORWARD)
        R = WC.pinned_request_pages(lay, 8192, 4096, 25 * GiB)
        self.assertEqual(R, 2141)
        _, config = GK.plan(w, dataclasses.replace(lay, request_pages=R), 25 * GiB, batched=FORWARD)
        self.assertEqual(config.num_blocks_by_pool, (7518, 6481))
        self.assertAlmostEqual((R * G.page_bytes(16) + 6481 * G.sliding_block_bytes(16)) / GiB, 18.44, places=2)

    def test_pool_zero_rules(self):
        # The rules take one 4,096-token forward: 256 Pages less per forward term at P = 16.
        lay, before = layout(), layout(forward=8192)
        blocks = 7518
        self.assertEqual(WC.one_session_pages(before, 131072, 4096, blocks) - WC.one_session_pages(lay, 131072, 4096, blocks),
                         256)
        # One selective turn: 2 x (512 + ceil((4,096 + 4,096) / 16) + 1) + 1 = 2,051 (8,192: 2,563).
        self.assertEqual(WC.selective_turn_pages(lay, 8192, 4096), 2051)
        self.assertEqual(WC.selective_turn_pages(before, 8192, 4096), 2563)

    def test_capacity_of_sixteen_sessions(self):
        # 16 sessions at 131,072 tokens, 4,096-token replies, an 8,192-token budget, P = 16:
        # (global Pages of pool 0, the selective rule's need, the most sessions it holds) by KV GiB.
        rows = {
            # observation_query and random_fill (8,192: 4,138/4,958/7,415/10,692 Pages, 18/19/23/28)
            0: ({24: 6698, 25: 7518, 28: 9975, 32: 13252}, {24: 2132, 25: 2141, 28: 2165, 32: 2198},
                {24: 23, 25: 24, 28: 28, 32: 33}),
            # goal_query with the 2,048-token Goal transient (8,192: 2,858/3,678/6,135/9,412, 16/17/21/26)
            MP.GEMMA4_GOAL_TOKENS: ({24: 5418, 25: 6238, 28: 8695, 32: 11972}, {24: 2120, 25: 2128, 28: 2152,
                                                                                32: 2185},
                                    {24: 21, 25: 22, 28: 26, 32: 31})}
        for goal, (pages, needs, most) in rows.items():
            lay = layout(goal=goal)
            for gib in (24, 25, 28, 32):
                with self.subTest(goal=goal, gib=gib):
                    blocks = WC.pool_zero_blocks(gib * GiB, lay, GROUPS, G.sliding_block_bytes(16), G.page_bytes(16))
                    self.assertEqual(blocks, pages[gib])
                    self.assertEqual(WC.selective_pages(lay, 8192, 4096, blocks), needs[gib])
                    self.assertEqual(WC.max_sessions(gib * GiB, active_tokens=8192, **common(goal=goal)), most[gib])
                    before = layout(forward=8192, goal=goal)
                    self.assertEqual(blocks - WC.pool_zero_blocks(gib * GiB, before, GROUPS, G.sliding_block_bytes(16),
                                                                  G.page_bytes(16)), 2560)
        # full under the release rule (one session at the limit, 15 tails): 8,820 Pages at 28 GiB
        # against 9,975, so 16 full sessions fit 28 GiB (8,192: 32 GiB only); 24, 25, 28 and 32 GiB
        # hold 12, 14, 17 and 22 (8,192: 8, 9, 13 and 18).
        lay = layout()
        self.assertEqual({gib: WC.release_pages(lay, 131072, 4096, WC.pool_zero_blocks(
            gib * GiB, lay, GROUPS, G.sliding_block_bytes(16), G.page_bytes(16))) for gib in (24, 25, 28, 32)},
            {24: 8787, 25: 8796, 28: 8820, 32: 8853})
        self.assertEqual({gib: WC.max_sessions(gib * GiB, release=True, **common()) for gib in (24, 25, 28, 32)},
                         {24: 12, 25: 14, 28: 17, 32: 22})
        self.assertEqual({gib: WC.max_sessions(gib * GiB, release=True, **common(forward=8192))
                          for gib in (24, 25, 28, 32)}, {24: 8, 25: 9, 28: 13, 32: 18})


# ---- chunked phase 1 ----------------------------------------------------------------------------

class ChunkTests(unittest.TestCase):
    def test_chunk_tokens(self):
        for P, chunk in ((16, 3056), (32, 3040)):
            with self.subTest(P=P):
                self.assertEqual(WC.phase1_chunk_tokens(layout(P, forward=8192)), 4096)   # unchanged
                self.assertEqual(WC.phase1_chunk_tokens(layout(P)), chunk)
                self.assertEqual(WC.phase1_chunk_tokens(layout(P, goal=MP.GEMMA4_GOAL_TOKENS)), chunk)
                self.assertEqual(chunk, FORWARD - W_ - P)
                # A smaller native forward than the layout's caps the chunk too.
                self.assertEqual(WC.phase1_chunk_tokens(layout(P, forward=8192), 2048), 2048)
        # The CPU fakes' geometry (W = 64, N = 1,024): 944; a layout with no room keeps min(4,096, N).
        self.assertEqual(WC.phase1_chunk_tokens(I0.for_window_server(16, 3, 64, 1024)), 944)
        self.assertEqual(WC.phase1_chunk_tokens(I0.for_window_server(16, 3, 64, 64)), 64)

    def test_the_ring_fits_one_forward_s_transient(self):
        # Every run up to 12,000 tokens from every offset within two Pages: the fresh blocks the
        # forwards hold at once never exceed the transient. 4,096-token chunks would hold up to
        # ceil((1,023 + 4,096) / 16) + 1 = 321 blocks per group at N = 4,096.
        for P in (16, 32):
            for forward in (8192, FORWARD):
                lay = layout(P, forward=forward)
                chunk = WC.phase1_chunk_tokens(lay)
                worst = 0
                for C in range(50000, 50000 + 2 * P):
                    for n in (1, 300, forward - 1, forward, forward + 1, 6000, 8192, 8193, 10200, 12000):
                        e = C + n
                        ring = SW.phase1_ring(C, SW.phase1_chunks(C, e, forward, chunk), W_, P)
                        worst = max(worst, ring)
                        self.assertLessEqual(ring, lay.transient_blocks, (P, forward, C, n))
                with self.subTest(P=P, forward=forward):
                    # One whole forward of N tokens is the largest; the chunks stay within it.
                    self.assertEqual(worst, lay.transient_blocks)
        lay = layout()
        old = max(SW.phase1_ring(C, SW.phase1_chunks(C, C + 10200, FORWARD, FORWARD), W_, 16)
                  for C in range(50000, 50032))
        self.assertEqual(old, 321)
        self.assertGreater(old, lay.transient_blocks)

    def test_gate_figures(self):
        # A run of at most one forward is one forward: 256 fresh blocks per group for 4,096 tokens.
        lay = layout()
        self.assertEqual(SW.phase1_chunks(50000, 54096, FORWARD, 3056), [(50000, 54096)])
        self.assertEqual(WC.sliding_turn_need(lay, GROUPS, cold=False, C=50000, e=54096, suffix=7)[0], 5 * 256)
        # A 10,200-token observation: four forwards (3,056 x 3 and 1,032); 255 fresh blocks per
        # group at once; phase 2 adds one block per group for the header.
        self.assertEqual(SW.phase1_chunks(50000, 60200, FORWARD, 3056),
                         [(50000, 53056), (53056, 56112), (56112, 59168), (59168, 60200)])
        self.assertEqual(WC.sliding_turn_need(lay, GROUPS, cold=False, C=50000, e=60200, suffix=7), (5 * 255, 5 * 1))
        self.assertEqual(WC.ledger_parts(lay, GROUPS, C=50000, e=60200, suffix=7), (5 * 2, 5 * 255))
        # With goal_query's Goal forward the Goal transient comes on top, as at 8,192.
        goal = layout(goal=MP.GEMMA4_GOAL_TOKENS)
        self.assertEqual(WC.sliding_turn_need(goal, GROUPS, cold=False, C=50000, e=60200, suffix=7,
                                              goal_tokens=MP.GEMMA4_GOAL_TOKENS)[0], 5 * (255 + 128))
        # A cold prefill in 4,096-token forwards holds at most a slot plus the transient (321).
        self.assertEqual(WC.cold_peak(lay, 60000), 65 + 256)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChunkedEndToEndTests(unittest.TestCase):
    """test_gemma_chunked's three sessions with long observations (up to about 2,660 tokens of run)
    on the window engine and worker fakes, at N = 4 W (the 4,096-token layout's ratio: N = 256 with
    the fakes' W = 64; 512 for goal_query, whose Goal forwards there take up to about 400 tokens)."""
    maxDiff = None
    check, windows = GE.GemmaEndToEndTests.check, GE.GemmaEndToEndTests.windows
    play = GC.ChunkedEndToEndTests.play
    setUp = GC.ChunkedEndToEndTests.setUp

    def run_policy(self, policy, forward):
        with patch.object(GE, 'GOAL', forward if policy == 'goal_query' else GE.GOAL):
            leases = []
            result, h = self.play(policy, forward=forward, leases=leases)
            chunk = forward - GE.WINDOW - GE.P
            self.assertEqual(h.endpoint.phase1.chunk_tokens, chunk)
            self.assertEqual(h.endpoint.phase1.chunk_tokens, WC.phase1_chunk_tokens(h.layout))
            chunked = [l for l in leases if 'chunks' in l]
            self.assertTrue(chunked)
            for lease in chunked:
                bounds = [lease['start']] + [c[0] for c in lease['chunks']] + [lease['end']]
                self.assertTrue(all(0 < b - a <= chunk for a, b in zip(bounds, bounds[1:])))
            if policy == 'goal_query':
                self.assertTrue(any(l['goal'] is not None for l in chunked))
            reference, _ = self.play(policy, forward=4096)
            self.assertEqual(result, reference)

    def test_chunks_fit_the_transient(self):
        self.run_policy('observation_query', 256)
        self.run_policy('goal_query', 512)

    @unittest.skipUnless(SLOW, 'about 6 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1')
    def test_chunks_fit_the_transient_full(self):
        self.run_policy('full', 256)

    def test_chunks_of_one_forward_never_fit(self):
        # The rule before gemma-forward4k (chunks of min(4,096, N) = N tokens): with the other
        # sessions' windows pinned, a chunked phase 1 needs more of pool 1 than is free while
        # nothing runs, and no release frees a window.
        with self.assertRaisesRegex(CAP.CapacityWait, 'CAPACITY_WAIT: sliding pool'):
            self.play('observation_query', chunk=256, forward=256)


# ---- the launcher -------------------------------------------------------------------------------

K = GF.K
# A sample of test_gemma_flags' combinations: (cuda_graphs, decode fast paths, release, overlap).
COMBINATIONS = (GF.PLAIN, ('full', GF.ALL3, True, K), ('off', (), True, 1), ('off', GF.BOTH, False, K),
                ('piecewise', (GF.FP,), True, 1), ('full-decode', GF.ALL3, False, 1))


def as_forward4k(value, active=None, max_tokens=4096, kv_bytes=GF.COMMON['kv_cache_gib'] << 30):
    """The [config, command, env] of a launch at 8,192 changed to 4,096-token forwards: the
    layout's forward_tokens (and request_pages under the selective rule, ``active`` its selection),
    effective_server_config's max_num_batched_tokens and additional_config, its pin and the flag."""
    config, command, env = copy.deepcopy(value)
    raw = dict(config['kv_layout'], forward_tokens=FORWARD)
    if 'request_pages' in raw:
        lay = I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: dict(raw, request_pages=1)}))
        raw['request_pages'] = WC.pinned_request_pages(lay, active, max_tokens, kv_bytes)
    additional = {I0.LAYOUT_KEY: raw}
    effective = dict(config['effective_server_config'], max_num_batched_tokens=FORWARD, additional_config=additional)
    config.update(kv_layout=raw, effective_server_config=effective, server_config_sha256=serve.pin(effective))
    command[command.index('--max-num-batched-tokens') + 1] = str(FORWARD)
    command[command.index('--additional-config') + 1] = json.dumps(additional, sort_keys=True, separators=(',', ':'))
    return [config, command, env]


def active_of(base):
    selection = GF.BASES[base][1] or {}
    return selection.get('active_token_budget')


class LauncherTests(GF.Release, unittest.TestCase):
    maxDiff = None

    def test_only_the_forward_changes(self):
        for base in GF.BASES:
            for combination in COMBINATIONS:
                with self.subTest(base=base, combination=combination):
                    plain = self.launched(base, combination)[0]
                    status, value, config = self.launch(base, combination, max_num_batched_tokens=FORWARD)
                    self.assertEqual(status, 'ok', value)
                    self.assertEqual(value, as_forward4k(plain, active_of(base)))
                    self.assertEqual(config['kv_layout']['forward_tokens'], FORWARD)
                    self.assertEqual('request_pages' in config['kv_layout'], GF.selective(base, combination[2]))
                    # 8192 is the default: the same launch byte for byte.
                    self.assertEqual(self.launch(base, combination, max_num_batched_tokens=8192)[:2], ('ok', plain))
        # --goal-rewriter-batch (its two keys) is independent of the forward.
        combination = ('full', GF.ALL3, True, K)
        status, value, _ = self.launch('goal_query-rewriter', combination, max_num_batched_tokens=FORWARD, **GF.BATCH)
        self.assertEqual(status, 'ok', value)
        self.assertEqual(value, as_forward4k(self.launched_batch(combination)[0], 8192))

    def test_refusals(self):
        for bad in (2048, 4095, 16384, '4096', True, 0):
            with self.subTest(bad=bad):
                status, message, _ = self.launch('full', max_num_batched_tokens=bad)
                self.assertEqual(status, 'refused')
                self.assertIn('the Gemma 4 profile takes --max-num-batched-tokens 8192 or 4096', message)
        # A Qwen release keeps its 16,384-token forwards and refuses the option.
        from test_gemma_install import gemma_fixture
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        directory = root / 'qwen-release'
        with redirect_stdout(io.StringIO()):
            package, site, model, _, _ = gemma_fixture(root)
            install.install(site, model, directory, package_root=package, apply=True)
        with redirect_stdout(io.StringIO()), patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            for value in (FORWARD, 8192):
                with self.subTest(qwen=value), self.assertRaisesRegex(
                        ValueError, '--max-num-batched-tokens applies to the Gemma 4 profile only'):
                    serve.prepare(directory, dict(GF.TASKS[0]), 'full', self.root / self.name('qwen'), 8000,
                                  max_num_batched_tokens=value)
            manifest, config, command, env = serve.prepare(directory, dict(GF.TASKS[0]), 'full',
                                                           self.root / self.name('qwen'), 8000)
        self.assertEqual(command[command.index('--max-num-batched-tokens') + 1], str(serve.MAX_BATCHED))

    def test_benchmark_settings(self):
        # The Gemma settings of the three benchmark docs (25 GiB; SWE-bench: 16 sessions at 131,072
        # tokens, Terminal-Bench as its dry runs ran: K = 17 at 65,536, MemoryAgentBench: 8 at 262,144)
        # with 4,096-token forwards: accepted, the same change, and the startup checks accept them.
        # Terminal-Bench's configuration at 262,144 (K = 20, 4,096-token forwards only):
        # test_terminal_bench_at_262144.
        shared = dict(kv_cache_gib=25, page_tokens=16, max_tokens=4096, connector_pool_gib=1, cuda_graphs='full',
                      decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, deferred_completion_io=True,
                      language_model_only=True, release_idle_pages=True)
        rewriter = dict(goal_rewriter_model=str(self.rewriter))
        benchmarks = {
            'swe': (dict(max_running=16, max_model_len=131072, dram_arena_gib=72), 8192,
                    dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=15360, admission_overlap=16)),
            'tb': (dict(protocol='chat-text', max_running=17, max_model_len=65536, dram_arena_gib=64), 8192,
                   dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=20480, admission_overlap=16)),
            'mab': (dict(protocol='kv-session', kv_session_thinking='off', max_running=8, max_model_len=262144,
                         dram_arena_gib=60), 32768, dict(goal_rewriter_batch=4, admission_overlap=8))}
        for name, (kw, budget, goal) in benchmarks.items():
            task = None if 'protocol' in kw else copy.deepcopy(GF.TASKS)
            for policy, extra, selection in (('observation_query', {}, {}), ('goal_query', goal, rewriter)):
                with self.subTest(benchmark=name, policy=policy):
                    values = []
                    for forward in (None, FORWARD):
                        out = self.name(name)
                        options = dict(shared, **kw, **extra)
                        if forward is not None:
                            options.update(max_num_batched_tokens=forward)
                        with redirect_stdout(io.StringIO()), \
                                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                            _, config, command, env = serve.prepare(
                                self.release, copy.deepcopy(task), policy, self.root / out, 8000,
                                dict(selection, active_token_budget=budget), **options)
                        env = {k: v for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'}
                        values.append((json.loads(json.dumps([config, command, env]).replace(out, 'OUT')), config))
                    (plain, _), (value, config) = values
                    self.assertEqual(value, as_forward4k(plain, budget, kv_bytes=25 << 30))
                    self.assertEqual(HC.kv_layout(self.load(config)[0]).forward_tokens, FORWARD)

    def test_terminal_bench_at_262144(self):
        # Terminal-Bench's Gemma servers (docs/TERMINALBENCH_BENCHMARK.md A.14, 2026-11-08): 25 GiB, K = 20,
        # 262,144 tokens, 4,096-token forwards and a 72-GiB arena for observation_query, random_fill and
        # goal_query: accepted, with the selective rule's request_pages (2,119; goal_query 2,106) and the
        # arena's 58,982 Pages, and the startup checks accept them. With 8,192-token forwards the same servers
        # are refused: K = 20 does not fit (at most 19 sessions; goal_query 17).
        shared = dict(protocol='chat-text', kv_cache_gib=25, page_tokens=16, max_tokens=4096, max_running=20,
                      max_model_len=262144, dram_arena_gib=72, connector_pool_gib=1, cuda_graphs='full',
                      decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True,
                      deferred_completion_io=True, language_model_only=True, release_idle_pages=True)
        goal = dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=20480, admission_overlap=16)
        rewriter = dict(goal_rewriter_model=str(self.rewriter))
        for policy, extra, selection, pages, most in (('observation_query', {}, {}, 2119, 19),
                                                      ('random_fill', {}, {}, 2119, 19),
                                                      ('goal_query', goal, rewriter, 2106, 17)):
            for forward in (FORWARD, None):
                with self.subTest(policy=policy, forward=forward):
                    options = dict(shared, **extra)
                    if forward is not None:
                        options.update(max_num_batched_tokens=forward)
                    out = self.root / self.name('tb262k')
                    args = (self.release, None, policy, out, 8000, dict(selection, active_token_budget=8192))
                    with redirect_stdout(io.StringIO()), \
                            patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                        if forward is None:
                            with self.assertRaisesRegex(ValueError, 'use at most %d sessions' % most):
                                serve.prepare(*args, **options)
                            continue
                        _, config, command, _ = serve.prepare(*args, **options)
                    layout = config['kv_layout']
                    self.assertEqual((layout['forward_tokens'], layout['request_pages'], layout['window_slots']),
                                     (FORWARD, pages, 20))
                    self.assertEqual([command[command.index(flag) + 1] for flag in (
                        '--max-num-seqs', '--max-model-len', '--max-num-batched-tokens', '--kv-cache-memory-bytes')],
                        ['20', '262144', '4096', str(25 << 30)])
                    self.assertEqual(config['dram_arena_pages'], 58982)
                    self.assertEqual(HC.kv_layout(self.load(config)[0]).forward_tokens, FORWARD)

    def test_other_protocols(self):
        # kv-session (16 MemAgent slots at 262,144 tokens under the selective rule) and chat-text
        # (docs/reference.md 4.14's chat-text example: K = 20 at 65,536 and 28 GiB; Terminal-Bench's servers:
        # test_terminal_bench_at_262144): the same change, and the startup checks accept it.
        launches = {
            'kv-session': ('observation_query', dict(active_token_budget=32768),
                           dict(protocol='kv-session', kv_session_thinking='off', max_running=16, max_model_len=262144,
                                release_idle_pages=True, dram_arena_gib=150, host_memory_shares=2, kv_cache_gib=28),
                           32768),
            'chat-text': ('observation_query', dict(active_token_budget=8192),
                          dict(protocol='chat-text', kv_cache_gib=28, max_model_len=65536, max_running=20,
                               dram_arena_gib=64, connector_pool_gib=1, release_idle_pages=True, cuda_graphs='full',
                               decode_fastpath=True), 8192)}
        for protocol, (policy, selection, kw, active) in launches.items():
            with self.subTest(protocol=protocol):
                values = []
                for forward in (None, FORWARD):
                    name = self.name('proto')
                    options = dict(GF.COMMON, **kw)
                    if forward is not None:
                        options.update(max_num_batched_tokens=forward)
                    with redirect_stdout(io.StringIO()), \
                            patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                        _, config, command, env = serve.prepare(self.release, None, policy, self.root / name, 8000,
                                                                copy.deepcopy(selection), **options)
                    env = {k: v for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'}
                    values.append((json.loads(json.dumps([config, command, env]).replace(name, 'OUT')), config))
                (plain, _), (value, config) = values
                self.assertEqual(value, as_forward4k(plain, active, kv_bytes=kw['kv_cache_gib'] << 30))
                loaded, _ = self.load(config)
                self.assertEqual(HC.kv_layout(loaded).forward_tokens, FORWARD)

    def test_dry_run_cli(self):
        task = self.root / 'forward4k-task.json'
        task.write_text(json.dumps(GF.TASKS[0]))

        def main(*extra):
            output = str(self.root / self.name('cli'))
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task), '--policy', 'full',
                    '--output', output, '--page-tokens', '16', '--kv-cache-gib', '28',
                    '--max-model-len', '131072', '--max-tokens', '4096', '--language-model-only', *extra]
            out = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(out), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return [a.replace(output, 'OUT') for a in json.loads(out.getvalue())['command']]
        default, explicit, forward = main(), main('--max-num-batched-tokens', '8192'), \
            main('--max-num-batched-tokens', '4096')
        at = default.index('--max-num-batched-tokens')
        self.assertEqual(default[at + 1], '8192')
        self.assertEqual(explicit, default)
        self.assertEqual(forward[at + 1], '4096')
        self.assertEqual(json.loads(forward[forward.index('--additional-config') + 1])[I0.LAYOUT_KEY]['forward_tokens'],
                         4096)
        with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            main('--max-num-batched-tokens', '2048')


class StartupTests(GF.Release, unittest.TestCase):
    maxDiff = None

    def test_load_and_argument_check(self):
        for base, combination in (('observation_query', ('full', GF.ALL3, True, K)), ('goal_query-rewriter', GF.PLAIN),
                                  ('full', ('off', (), True, 1))):
            with self.subTest(base=base):
                status, _, config = self.launch(base, combination, max_num_batched_tokens=FORWARD)
                self.assertEqual(status, 'ok')
                value, env = self.load(config)
                self.assertEqual(HC.kv_layout(value).forward_tokens, FORWARD)
                with patch.dict(os.environ, env):
                    effective = HC.effective_server_config(GF.parsed(config['effective_server_config']))
                self.assertEqual(effective['max_num_batched_tokens'], FORWARD)
                # vLLM parsed another --max-num-batched-tokens than the pin: refused.
                with patch.dict(os.environ, env), self.assertRaisesRegex(RuntimeError, 'differ from pinned'):
                    HC.effective_server_config(GF.parsed(dict(config['effective_server_config'],
                                                              max_num_batched_tokens=8192)))

    def test_load_refusals(self):
        _, _, config = self.launch('full', max_num_batched_tokens=FORWARD)

        def refused(pattern, edit):
            value = copy.deepcopy(config)
            edit(value)
            with self.assertRaisesRegex(RuntimeError, pattern):
                self.load(value)
        # The layout's forward and --max-num-batched-tokens must agree.
        refused('pinned vLLM arguments', lambda v: v['effective_server_config'].update(max_num_batched_tokens=8192))

        def forward(n):
            def edit(v):
                v['kv_layout']['forward_tokens'] = n
                additional = {I0.LAYOUT_KEY: v['kv_layout']}
                v['effective_server_config'].update(max_num_batched_tokens=n, additional_config=additional)
            return edit
        # Another forward than 8,192 or 4,096, with matching vLLM arguments: refused.
        for n in (2048, 16384):
            refused('pins one native forward of 8192 or 4096 tokens', forward(n))
        # 8,192 with matching vLLM arguments is the default layout: accepted.
        value = copy.deepcopy(config)
        forward(8192)(value)
        self.assertEqual(HC.kv_layout(self.load(value)[0]).forward_tokens, 8192)


if __name__ == '__main__':
    unittest.main()
