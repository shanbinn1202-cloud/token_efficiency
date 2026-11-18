"""--thinking-history strip on the window layout: the sliding window held at the cut (option G). CPU only.

The Gemma 4 strip path (remote_ra_live_v2.window_hold; the 11-09 design study, sections 3.2-3.4,
6.3 T3 / T4): the session commits at the cut X = e as on Qwen, but what crosses turns on the window
layout is the sliding window, kept_blocks(X) of five sliding groups, which vLLM frees block by block
while phase 2 decodes past X. A strip request holds it (one more reference) from its install, or
from its local turn 0 prefill past X, until its commit pins it for the session.

Nothing here is GPU evidence. The paths are driven by the window layout's real engine and worker
(test_gemma_e2e: vLLM's KV manager with its SlidingWindowManager, WindowPins, WindowPhase1Coordinator,
WindowAllocationOwner, WindowLifecycle and a runner that checks every sliding read against the last W
tokens of the stream) behind test_gemma_chat_text_e2e's API server, with the runtime's own Gemma 4 strip
transcripts (gemma4_renderer.Gemma4SourceRenderer.chat_transcripts with thinking and strip, at both ends)
over a stand-in of the pinned tokenizer whose thinking-on header ``<|turn>model\\n`` is the pinned
[105, 4368, 107] (K = |H| = 3) and the stand-in of the pinned template (test_gemma_thinking_on.STAND_IN_ON),
attached as compose_engine and attach_window_worker attach them (thinking_cut.attach_engine and
attach_worker, which hand the window layout to window_hold). Until the integration of gemma-thinking and
gemma-thinking-hold these tests ran on Qwen's chat-text turns as stand-ins.

1. The layout: ``hold_slots`` H, its additional-config pin (absent at 0), pool 1 with H more window
   slots (vLLM's planner too); the study's capacity table (capacity_hold.json) from this code; the
   launcher's sizing and refusals (serve.py --window-hold-slots, its default the most that fit; 0 and
   a line where none fits refused); the runtime config refuses a pinned H without the strip history
   and a Gemma strip config without one.
2. WindowPins over vLLM's SlidingWindowManager at W = 1,024: the hold at install and at local turn 0,
   replies shorter than 16 tokens and longer than 1,024, the commit from the hold, release on discard
   and on cancel, the refusals; --goal-overlap's trim composes.
3. End to end (T3): every completion commits at X with the held window (engine pin, worker
   WindowCarry at X, the next phase 1 borrows it and reads exactly the last W tokens), local turn 0
   with and without the split, replies shorter than a block and longer than the window, a cancel in
   phase 2, a rebuild after a cut, the gate with H slots (waits, never preemption), a full pool; the
   gate's figures (the hold term in a turn's need and ledger, its rest in a running request's headroom);
   the hold term at work (TightPoolTests): continuing turns in a pool 1 smaller than the launcher sizes it,
   and a cold prefill beside decoders in a tight pool; a hold whose cleanup failed (CleanupFailureTests): it
   keeps its slot until its admission's lease runs out, then the expiry retries the drain.
4. The flags (T4): the three decode fast paths, --deferred-completion-io, --admission-overlap,
   --release-idle-pages under pressure (--goal-overlap: test_gemma_strip_e2e).
5. Mutations: a broken hold fails (no hold, a hold without its reference, the worker's window at another
   end, the engine's hold at another end, no hold slot cap); without the gate's hold term (the review's
   mutation of 11-10, ``no_hold_term``) a running request of TightPoolTests finds no block.
6. The attachments and the config pin; identity: with H = 0 nothing is held, reserved, pinned or counted
   (the byte identity with the base commit is the identity probes', outside the suite).

The fake geometry is test_gemma_e2e's: P = 16, W = 64, one native forward N = 1,024; "longer than the
window" there is W + 2P tokens, which on the server (W = 1,024) is a reply longer than 1,024 tokens.
"""
import asyncio
import contextlib
import dataclasses
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

import serve  # noqa: E402
import test_gemma_engine as GEN  # noqa: E402  WindowPins over vLLM's managers (W = 1,024)
import test_gemma_kv_pools as GK  # noqa: E402

from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2 import window_capacity as WC  # noqa: E402
from remote_ra_live_v2 import window_hold as WH  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from remote_ra_live_v2.sliding_window import kept_blocks, phase1_row  # noqa: E402
from remote_ra_native_growing.contract import Abort  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

GiB, MiB = 1024 ** 3, 1024 ** 2
G = MP.GEMMA4_GEOMETRY
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'


# ---- 1. the layout, the capacity table, the launcher ----------------------------------------------

# The study's capacity_hold.json (11-09, pool 0 left at 25 GiB with H hold slots; P = 16, 4,096-token
# replies, the selective rule): (name, S, max_model_len, N, active budget, Goal tokens) and per H
# (pool 1 GiB to two places, pool-0 Pages, the rule's Pages, spare = Pages - 1 - needed). The study's
# rule had no term for the reply a strip turn re-prefills: with hold slots the rule now asks for
# REPREFILL more Pages (window_capacity.strip_reprefill_pages; the review of 11-10, finding 8).
STUDY = [
    ('TB obs K=20 262K N=4096', 20, 262144, 4096, 8192, 0,
     {0: (19.0, 4918, 2119, 2798), 1: (19.79, 4268, 2112, 2155), 2: (20.58, 3618, 2106, 1511),
      4: (22.17, 2318, 2093, 224), 8: (25.34, 0, 2070, -2071), 20: (34.87, 0, 2070, -2071)}),
    ('TB goal K=20 262K N=4096', 20, 262144, 4096, 8192, 2048,
     {0: (20.56, 3638, 2106, 1531), 1: (21.35, 2988, 2099, 888), 2: (22.15, 2338, 2093, 244),
      4: (23.73, 1038, 2080, -1043), 8: (26.91, 0, 2070, -2071), 20: (36.43, 0, 2070, -2071)}),
    ('SWE obs 16 131K N=8192', 16, 131072, 8192, 8192, 0,
     {0: (18.95, 4958, 2627, 2330), 1: (19.74, 4308, 2621, 1686), 2: (20.53, 3658, 2614, 1043),
      4: (22.12, 2358, 2601, -244), 8: (25.3, 0, 2578, -2579), 16: (31.64, 0, 2578, -2579)}),
    ('SWE goal 16 131K N=8192', 16, 131072, 8192, 8192, 2048,
     {0: (20.51, 3678, 2614, 1063), 1: (21.3, 3028, 2608, 419), 2: (22.1, 2378, 2601, -224),
      4: (23.68, 1078, 2588, -1511), 8: (26.86, 0, 2578, -2579), 16: (33.21, 0, 2578, -2579)}),
    ('SWE obs 16 131K N=4096', 16, 131072, 4096, 8192, 0,
     {0: (15.82, 7518, 2141, 5376), 1: (16.62, 6868, 2134, 4733), 2: (17.41, 6218, 2128, 4089),
      4: (19.0, 4918, 2115, 2802), 8: (22.17, 2318, 2089, 228), 16: (28.52, 0, 2066, -2067)}),
    ('MAB obs 8 262K N=4096 B=32768', 8, 262144, 4096, 32768, 0,
     {0: (9.48, 12718, 5257, 7460), 1: (10.27, 12068, 5250, 6817), 2: (11.06, 11418, 5244, 6173),
      4: (12.65, 10118, 5231, 4886), 8: (15.82, 7518, 5205, 2312)}),
]
# The most hold slots that fit each line (most_hold_slots, every H tried). The study's table lists
# H = 0, 1, 2, 4, 8 and S only; its risk 7 reads "2 (SWE obs at N = 8,192)": H = 3 fits there too.
MOST = {'TB obs K=20 262K N=4096': 3, 'TB goal K=20 262K N=4096': 1, 'SWE obs 16 131K N=8192': 3,
        'SWE goal 16 131K N=8192': 1, 'SWE obs 16 131K N=4096': 7, 'MAB obs 8 262K N=4096 B=32768': 8}
# The same by the study's rule (without the strip term): one more on three lines.
MOST_STUDY = {'TB obs K=20 262K N=4096': 4, 'TB goal K=20 262K N=4096': 2, 'SWE obs 16 131K N=8192': 3,
              'SWE goal 16 131K N=8192': 1, 'SWE obs 16 131K N=4096': 8, 'MAB obs 8 262K N=4096 B=32768': 8}
REPREFILL = 256         # ceil(4,096 / 16): one reply bound of re-prefilled content, in Pages


def study_row(S, N, B, goal, H, kv=25 * GiB):
    """One cell of the table from this code: (pool 1 GiB, pool-0 Pages, the rule's Pages, spare)."""
    groups, block, page = len(G.sliding_groups()), G.sliding_block_bytes(16), G.page_bytes(16)
    layout = I0.WindowLayout(16, G.sliding_window, S, N, goal, hold_slots=H)
    blocks = WC.pool_zero_blocks(kv, layout, groups, block, page)
    needed = WC.selective_pages(layout, B, 4096, blocks)
    return (round(WC.pool_one_bytes(layout, groups, block) / GiB, 2), blocks, needed, blocks - 1 - needed)


class LayoutTests(unittest.TestCase):
    def test_hold_slots_field_and_pin(self):
        plain = I0.for_window_server(16, 20, 1024, 4096)
        held = dataclasses.replace(plain, hold_slots=4)
        self.assertEqual((plain.hold_slots, plain.hold_blocks, held.hold_blocks), (0, 0, 4 * 65))
        # H = 0: no key (the record of every server today); H > 0: the key, and it parses back.
        self.assertNotIn(I0.HOLD_FIELD, I0.to_additional_config(plain)[I0.LAYOUT_KEY])
        record = I0.to_additional_config(held)
        self.assertEqual(record[I0.LAYOUT_KEY][I0.HOLD_FIELD], 4)
        self.assertEqual(I0.from_vllm_config(NS(additional_config=record)), held)
        self.assertEqual(I0.from_vllm_config(NS(additional_config=I0.to_additional_config(plain))), plain)
        # Pool 1: one window slot per hold slot and sliding group.
        self.assertEqual(I0.window_pool_blocks(held, 5) - I0.window_pool_blocks(plain, 5), 5 * 4 * 65)
        self.assertEqual(5 * 65 * G.sliding_block_bytes(16), 812.5 * MiB)
        # With the other fields.
        goal = I0.WindowLayout(16, 1024, 16, 8192, 2048, 900, 2)
        self.assertEqual(I0.from_vllm_config(NS(additional_config=I0.to_additional_config(goal))), goal)

    def test_refusals(self):
        record = I0.to_additional_config(I0.for_window_server(16, 4, 1024, 8192))[I0.LAYOUT_KEY]
        for bad, pattern in ((0, 'hold_slots must be positive'), (5, r'hold_slots must be in \[0, window_slots 4\]'),
                             (-1, 'hold_slots must be positive'), ('2', 'must be an int')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, pattern):
                I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: dict(record, hold_slots=bad)}))
        with self.assertRaisesRegex(ValueError, 'hold_slots must be in'):
            I0.WindowLayout(16, 1024, 4, 8192, hold_slots=5)
        with self.assertRaisesRegex(ValueError, 'exactly the keys'):
            I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: dict(record, hold=1)}))

    def test_vllm_planner_reserves_the_hold_slots(self):
        w = GK.gemma_world()
        for H in (0, 1, 4):
            with self.subTest(H=H):
                layout = dataclasses.replace(GK.window_layout(16, 10), hold_slots=H)
                _, config = GK.plan(w, layout, 28 * GiB)
                n1 = I0.window_pool_blocks(layout, 5)
                self.assertEqual(n1, 1 + 5 * (10 * 65 + 512 + H * 65))
                self.assertEqual(config.num_blocks_by_pool,
                                 ((28 * GiB - n1 * G.sliding_block_bytes(16)) // G.page_bytes(16), n1))


class CapacityTableTests(unittest.TestCase):
    def test_study_table(self):
        """capacity_hold.json of the study, recomputed from this code's sizing: equal in every cell without
        hold slots; with hold slots the pools are the study's and the rule asks for the strip term more."""
        for name, S, _, N, B, goal, rows in STUDY:
            for H, (pool1, pages, needed, spare) in rows.items():
                with self.subTest(case=name, H=H):
                    more = REPREFILL if H else 0
                    self.assertEqual(study_row(S, N, B, goal, H), (pool1, pages, needed + more, spare - more))

    def test_most_hold_slots(self):
        groups, block, page = len(G.sliding_groups()), G.sliding_block_bytes(16), G.page_bytes(16)
        for name, S, _, N, B, goal, rows in STUDY:
            with self.subTest(case=name):
                layout = I0.WindowLayout(16, G.sliding_window, S, N, goal)

                def needed(held, blocks, B=B):
                    return WC.selective_pages(held, B, 4096, blocks)
                most = WC.most_hold_slots(25 * GiB, layout, groups, block, page, needed)
                self.assertEqual(most, MOST[name])
                self.assertGreaterEqual(study_row(S, N, B, goal, most)[3], 0)
                if most < S:
                    self.assertLess(study_row(S, N, B, goal, most + 1)[3], 0)
                # The study's rule (no strip term) fitted at most one more.
                study = max(H for H in range(S + 1) if H == 0 or study_row(S, N, B, goal, H)[3] + REPREFILL >= 0)
                self.assertEqual(study, MOST_STUDY[name])
                self.assertIn(study - most, (0, 1))

    def test_design_table(self):
        """docs/thinking-gemma/DESIGN.md section 4.3: pool-0 Pages left beyond the selective rule at 25 GiB
        for H = 0 to 5 hold slots, and the most that fit (the launcher's default), from this code."""
        design = {'TB obs K=20 262K N=4096': ((2798, 1899, 1255, 612, -32, -675), 3),
                  'TB goal K=20 262K N=4096': ((1531, 632, -12, -655, -1299, -1942), 1),
                  'SWE obs 16 131K N=8192': ((2330, 1430, 787, 143, -500, -1144), 3),
                  'SWE goal 16 131K N=8192': ((1063, 163, -480, -1124, -1767, -2411), 1)}
        lines = {name: (S, N, B, goal) for name, S, _, N, B, goal, _ in STUDY}
        for name, (spare, most) in design.items():
            S, N, B, goal = lines[name]
            with self.subTest(case=name):
                self.assertEqual(tuple(study_row(S, N, B, goal, H)[3] for H in range(6)), spare)
                self.assertEqual((MOST[name], max(H for H in range(6) if spare[H] >= 0)), (most, most))
        # One hold slot: 325 block ids of pool 1 (65 per sliding group), 812.5 MiB, 650 global Pages.
        zero, one = study_row(20, 4096, 8192, 0, 0), study_row(20, 4096, 8192, 0, 1)
        self.assertEqual((zero[1] - one[1], 812.5 * MiB / G.page_bytes(16), 5 * 65), (650, 650, 325))
        # The Terminal-Bench line with 24 sessions (the most without a hold slot): none fits.
        self.assertEqual([study_row(24, 4096, 8192, 0, H)[3] for H in (0, 1)], [220, -679])
        # The Terminal-Bench lines' pool 0 for each H (the design's second table): Pages, the rule's Pages.
        self.assertEqual([study_row(20, 4096, 8192, 0, H)[1:3] for H in (1, 2, 3, 4)],
                         [(4268, 2368), (3618, 2362), (2968, 2355), (2318, 2349)])
        self.assertEqual([study_row(20, 4096, 8192, 2048, H)[1:3] for H in (1, 2)], [(2988, 2355), (2338, 2349)])
        # A session slot and a hold slot cost pool 0 the same: one session fewer buys one hold slot.
        self.assertEqual(study_row(19, 4096, 8192, 2048, 2)[1::2], (2988, 633))
        self.assertEqual(study_row(19, 4096, 8192, 0, 4)[1::2], (2968, 613))

    def test_hold_term(self):
        layout = I0.for_window_server(16, 4, 1024, 8192)
        # A whole window slot for any reply of 4,096 tokens; fewer when the reply bound is short.
        self.assertEqual(WC.hold_term(layout, 5000, 3, 4096), 65)
        self.assertEqual(WC.hold_term(layout, 5000, 3, 1536), 65)
        self.assertEqual(WC.hold_term(layout, 5000, 3, 13), 2)          # ceil(16 / 16) + 1
        self.assertEqual(WC.hold_term(layout, 5000, 3, 14), 3)
        self.assertEqual(WC.hold_term(layout, 40, 3, 4096), 3)          # the window at 40: blocks 0-2
        first, last = kept_blocks(5000, 1024, 16)
        self.assertEqual(last - first, 65)


class StripRuleTests(unittest.TestCase):
    """The selective rule's term for the reply a Gemma strip turn re-prefills (window_capacity.
    strip_reprefill_pages; the review of 11-10, finding 8), against the capacity gate's own count of a
    continuing turn's global Pages: the session's pins and its phase-2 need (capacity._need_by_pool), which
    idle_release.ceiling compares with pool 0 before it answers CAPACITY_EXCEEDED."""
    P, F, M, A = 16, 4096, 4096, 8192       # the Terminal-Bench lines: Pages, forwards, replies, the selection

    def test_the_term(self):
        for S, forward, goal, reply, budget, plain in ((20, 4096, 0, 4096, 8192, 2051), (16, 8192, 0, 4096, 8192, 2563),
                                                       (16, 8192, 0, 1536, 8192, 2243), (20, 4096, 2048, 4096, 8192, 2051),
                                                       (8, 4096, 0, 4096, 32768, 5123)):
            with self.subTest(S=S, forward=forward, goal=goal, reply=reply):
                layout = I0.WindowLayout(16, 1024, S, forward, goal)
                self.assertEqual((WC.strip_reprefill_pages(layout, reply), WC.selective_turn_pages(layout, budget, reply)),
                                 (0, plain))
                for H in (1, S):
                    held = dataclasses.replace(layout, hold_slots=H)
                    term = -(-reply // 16)
                    self.assertEqual((WC.strip_reprefill_pages(held, reply), WC.selective_turn_pages(held, budget, reply)),
                                     (term, plain + term))
                    self.assertEqual(WC.selective_pages(held, budget, reply, 3000),
                                     WC.selective_pages(layout, budget, reply, 3000) + term)
        # The figure the launcher pins for vLLM's startup check (request_pages) carries it.
        layout = dataclasses.replace(I0.WindowLayout(16, 1024, 20, 4096), hold_slots=3)
        # One turn, 19 other tails and the watermark of the 2,968 Pages left with three hold slots.
        self.assertEqual((WC.pinned_request_pages(layout, 8192, 4096, 25 * GiB), study_row(20, 4096, 8192, 0, 3)[2]),
                         (2307 + 19 + 29, 2355))
        # The end-to-end tests' geometry (test_gemma_release: 343 Pages) with and without hold slots.
        fake = I0.for_window_server(16, 16, 64, 1024)
        self.assertEqual((WC.selective_turn_pages(fake, 160, 1536),
                          WC.selective_turn_pages(dataclasses.replace(fake, hold_slots=2), 160, 1536)), (343, 343 + 96))

    def gate_pages(self, cut, sealed, row, header):
        """The global Pages of one continuing turn as the gate counts them, its own pins included: the
        session committed at ``cut`` (its tail there) and keeps one selection and the ``sealed`` tokens
        before the cut resident; the prompt adds ``row`` tokens and the ``header``; the reply bound is M."""
        from remote_ra_live_v2 import capacity as CAP, selection_contract as SC
        P, layout = self.P, I0.for_window_server(self.P, 20, 1024, self.F)
        contract = SC.default_contract(P, active_ratio=None, active_token_budget=self.A,
                                       vector_format=SC.vector_format_for(I0.to_additional_config(layout)[I0.LAYOUT_KEY]))
        Q = cut // P * P
        resident = tuple(range(self.A // P)) + tuple(range((cut - sealed) // P, Q // P))
        s = NS(local_turn=3, computed_end=cut, sealed_end=Q, tail=dict(page=Q // P) if cut > Q else None,
               resident=resident, policy='observation_query', namespace='ns', evicted=())
        transcript = NS(turn=3, extends=lambda messages: True, header=(0,) * header, initial_length=64)
        adapter = NS(window=None, layout=layout, sessions={'s': s}, page_tokens=P, decoupled=True)
        endpoint = NS(phase1=NS(groups=lambda: ((0,), (1, 2, 3, 4, 5), 6), max_tokens=self.F, handoff=None),
                      transcripts={'s': transcript}, selection=NS(contract=contract), claims={})
        core = NS(_remote_ra_growing_adapter=adapter, _remote_ra_private_v2_endpoint=endpoint)
        (phase1, _), (phase2, _) = CAP._need_by_pool(core, 's', [1] * (cut + row + header), None, self.M)
        own = len(resident) + (s.tail is not None)
        return max(own + phase1, own + phase2)

    def worst(self, sealed, row, header):
        return max(self.gate_pages(100000 + offset, sealed, row, header) for offset in range(2 * self.P))

    def test_the_worst_strip_turn(self):
        """Observations of one forward and replies at the bound. Without strip the last reply is resident
        and the rule holds the turn. A strip turn keeps the last run (a re-prefilled content c' and its
        observation) and computes c, the observation, the header and the reply: the rule without the term
        holds it while c' + c <= max_tokens, and is short of two contents at the bound by the term."""
        F, M = self.F, self.M
        layout = I0.for_window_server(self.P, 20, 1024, F)
        plain = WC.selective_turn_pages(layout, self.A, M)
        strip = WC.selective_turn_pages(dataclasses.replace(layout, hold_slots=3), self.A, M)
        self.assertEqual((plain, strip), (2051, 2307))
        # No strip (thinking off: the 7-token header): the run sealed last turn is its observation and reply.
        self.assertEqual(self.worst(F + M, F - 7, 7), plain - 2)
        for before, content in ((0, 0), (300, 300), (2048, 2048), (M, 0), (0, M)):
            with self.subTest(before=before, content=content):
                self.assertLessEqual(self.worst(before + F, content + F, 3), plain)
        self.assertEqual(self.worst(2048 + F, 2048 + F, 3), plain - 1)
        # Two contents at the bound: 255 Pages more than the rule without the term, one less than with it.
        worst = self.worst(M + F, M + F, 3)
        self.assertEqual((worst - plain, strip - worst), (255, 1))
        # In between, the excess is the contents beyond one reply bound: (3,000 + 3,000 - 4,096) / 16 Pages.
        self.assertEqual(self.worst(3000 + F, 3000 + F, 3) - self.worst(2048 + F, 2048 + F, 3), 119)


class LauncherTests(unittest.TestCase):
    """serve.py --window-hold-slots on a Gemma 4 strip server: H is sized by the server's rule and refused
    when pool 0 cannot hold it; left out, the most that fit; a server without the strip history reserves
    nothing and refuses the option."""
    maxDiff = None

    def setUp(self):
        import test_gemma_startup as GS
        self.gs = GS.GemmaStartupTests('test_load_accepts_the_gemma_startup_config')
        self.gs.setUp()
        self.addCleanup(self.gs.doCleanups)

    LINE = dict(protocol='chat-text', max_running=20, max_model_len=262144, kv_cache_gib=25,
                max_num_batched_tokens=4096, release_idle_pages=True, dram_arena_gib=72)
    STRIP = dict(enable_thinking=True, thinking_history='strip')

    def launch(self, hold=None, policy='observation_query', selection=None, strip=True, **kw):
        """The TB 262K line (K = 20, 4,096-token forwards, 25 GiB, the selective rule) with the strip
        history and --window-hold-slots ``hold`` (None: left out)."""
        selection = dict(active_token_budget=8192) if selection is None else selection
        options = dict(self.LINE, **(self.STRIP if strip else {}))
        if hold is not None:
            options.update(window_hold_slots=hold)
        return self.gs.prepare(None, policy, selection, **dict(options, **kw))

    @staticmethod
    def normal(config, command):
        out = json.dumps(config['output_directory'])[1:-1]
        return json.loads(json.dumps([config, command]).replace(out, 'OUT'))

    def test_without_the_strip_history_nothing_is_reserved(self):
        _, keep, command, _ = self.launch(strip=False, enable_thinking=True)
        self.assertNotIn(I0.HOLD_FIELD, keep['kv_layout'])
        self.assertNotIn('hold_slots', command[command.index('--additional-config') + 1])
        _, off, _, _ = self.launch(strip=False)
        self.assertEqual(off['kv_layout'], keep['kv_layout'])
        for options in (dict(strip=False), dict(strip=False, enable_thinking=True)):
            with self.subTest(options=options), self.assertRaisesRegex(
                    ValueError, 'applies to --thinking-history strip only'):
                self.launch(2, **options)

    def test_the_switch(self):
        self.assertIs(serve.GEMMA4_WINDOW_HOLD, True)
        with patch.object(serve, 'GEMMA4_WINDOW_HOLD', False):
            for hold in (None, 1, 4):
                with self.subTest(hold=hold), self.assertRaisesRegex(ValueError, 'switched off in this release'):
                    self.launch(hold)
            # Without the strip history the switch changes nothing.
            self.launch(strip=False, enable_thinking=True)

    def test_sizing_and_refusals(self):
        _, keep, keep_command, _ = self.launch(strip=False, enable_thinking=True)
        for hold, expect in ((None, 3), (1, 1), (3, 3)):
            with self.subTest(hold=hold):
                _, config, command, _ = self.launch(hold)
                self.assertEqual(config['kv_layout'][I0.HOLD_FIELD], expect)
                layout = I0.from_vllm_config(NS(additional_config=json.loads(
                    command[command.index('--additional-config') + 1])))
                self.assertEqual(layout.hold_slots, expect)
                # vLLM's startup check asks for the selective rule's Pages of the held pool.
                blocks = WC.pool_zero_blocks(25 * GiB, layout, 5, G.sliding_block_bytes(16), G.page_bytes(16))
                self.assertEqual(layout.request_pages, WC.selective_pages(layout, 8192, 4096, blocks))
                # Against the keep server of the same line: the layout record (config and vLLM's
                # --additional-config), the thinking record's history, and nothing else.
                a, b = self.normal(config, command), self.normal(keep, keep_command)
                differ = sorted(k for k in set(a[0]) | set(b[0]) if a[0].get(k) != b[0].get(k))
                self.assertEqual(differ, ['effective_server_config', 'kv_layout', 'server_config_sha256', 'thinking'])
                self.assertEqual([x for x, y in zip(a[1], b[1]) if x != y],
                                 [a[1][a[1].index('--additional-config') + 1]])
                self.assertEqual(len(a[1]), len(b[1]))
                loaded = self.gs.load(config)
                from remote_ra_http import config as HC
                self.assertEqual(HC.kv_layout(loaded).hold_slots, expect)
        _, blocks, needed, _ = study_row(20, 4096, 8192, 0, 5)
        with self.assertRaisesRegex(ValueError, r'--window-hold-slots 5 at --kv-cache-gib 25 with 20 sessions '
                                    r'leaves %d usable global Pages .* 812\.5 MiB each\); one selective turn of '
                                    r'a 8192-token selection and 19 other tails needs %d: use at most 3 hold '
                                    r'slots' % (blocks - 1, needed)):
            self.launch(5)
        # Four fitted the rule without the strip term (+224 Pages): short by 32 Pages with it.
        _, blocks, needed, spare = study_row(20, 4096, 8192, 0, 4)
        self.assertEqual((spare, spare + REPREFILL), (-32, 224))
        with self.assertRaisesRegex(ValueError, r'--window-hold-slots 4 at --kv-cache-gib 25 with 20 sessions '
                                    r'leaves %d usable global Pages .* needs %d: use at most 3 hold slots'
                                    % (blocks - 1, needed)):
            self.launch(4)
        for bad in (0, 21, -1):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    ValueError, r'must be 1 to 20 \(the running sessions\).*with 0 hold slots no reply could decode'):
                self.launch(bad)
        # goal_query on the same line: one hold slot (two are short by 12 Pages); with 19 sessions, two.
        goal = dict(active_token_budget=8192, no_goal_rewriter=True)
        _, config, _, _ = self.launch(None, 'goal_query', goal)
        self.assertEqual(config['kv_layout'][I0.HOLD_FIELD], 1)
        self.assertEqual(study_row(20, 4096, 8192, 2048, 2)[3], -12)
        with self.assertRaisesRegex(ValueError, r'--window-hold-slots 2 at --kv-cache-gib 25 with 20 sessions .*'
                                    r'use at most 1 hold slots'):
            self.launch(2, 'goal_query', goal)
        _, config, _, _ = self.launch(None, 'goal_query', goal, max_running=19)
        self.assertEqual(config['kv_layout'][I0.HOLD_FIELD], 2)

    def test_a_line_where_no_hold_slot_fits_is_refused(self):
        """K = 24 on the TB line: the selective rule holds without a hold slot (24 sessions is the most that
        fit there, the study's capacity.json) and with none of 1 to 24; the launcher names it instead of
        starting a strip server that could not decode."""
        self.launch(strip=False, enable_thinking=True, max_running=24)
        with self.assertRaisesRegex(ValueError, r'one hold slot \(the most that fit: none\) at --kv-cache-gib 25 '
                                    r'with 24 sessions .*: --thinking-history strip needs at least one hold slot: '
                                    r'use fewer sessions or a larger --kv-cache-gib'):
            self.launch(None, max_running=24)
        with self.assertRaisesRegex(ValueError, r'--window-hold-slots 1 at --kv-cache-gib 25 with 24 sessions .*'
                                    r'needs at least one hold slot'):
            self.launch(1, max_running=24)

    def test_the_runtime_config_and_the_strip_history(self):
        _, config, _, _ = self.launch(2)
        from remote_ra_http import config as HC
        layout = HC.kv_layout(self.gs.load(config))
        self.assertEqual((layout.hold_slots, I0.to_additional_config(layout)[I0.LAYOUT_KEY]), (2, config['kv_layout']))
        # A pinned H without the strip history (the keep record): refused.
        from remote_ra_live_v2 import thinking as TH
        keep = dict(config, thinking=TH.record(TH.CHAT, profile=MP.GEMMA4))
        with self.assertRaisesRegex(Exception, 'window hold slots need --thinking-history strip'):
            self.gs.load(keep)
        # The strip history without hold slots (H = 0): refused, by the layout check or by the strip pin.
        _, plain, _, _ = self.launch(strip=False, enable_thinking=True)
        bare = dict(config, kv_layout=plain['kv_layout'], effective_server_config=plain['effective_server_config'],
                    server_config_sha256=plain['server_config_sha256'])
        with self.assertRaisesRegex(Exception, 'needs hold slots for the sliding window at the cut'):
            self.gs.load(bare)
        # Another H than the pinned vLLM arguments: refused.
        other = dict(config, kv_layout=dict(config['kv_layout'], hold_slots=3))
        with self.assertRaisesRegex(Exception, 'KV layout differs'):
            self.gs.load(other)


# ---- 2. WindowPins: the hold over vLLM's SlidingWindowManager (W = 1,024, P = 16) ------------------

W, P, SLIDING = GEN.W, GEN.P, GEN.SLIDING
HEADER = 3              # Gemma's thinking-on generation header <|turn>model\n (study 1.2)


class PinsWorld(GEN.World):
    """test_gemma_engine's world: one overlay KV manager, WindowPins with H hold slots."""

    def __init__(self, hold=2, **kw):
        super().__init__(**kw)
        self.pins.hold_slots = hold

    def cold(self, rid, tokens):
        """A cold request's prefill in one allocation, the cursor after it (vLLM's rows)."""
        req, blocks = self.allocate(rid, tokens)
        assert blocks is not None
        req.num_computed_tokens = tokens
        return req

    def handover(self, rid, C, e, ns='ns'):
        """Phase 1 over [C, e) on the session's window: its fresh blocks, its drain, the handover."""
        head, (b0, b1), (f0, f1) = phase1_row(C, e, W, P)
        fresh = {g: self.pool.get_new_blocks(f1 - f0) for g in SLIDING}
        k0, _ = kept_blocks(e, W, P)
        self.pool.free_blocks([b for g in SLIDING for i, b in zip(range(f0, f1), fresh[g]) if i < k0])
        stay = {g: tuple((i, b) for i, b in zip(range(f0, f1), fresh[g]) if i >= k0) for g in SLIDING}
        return self.pins.handover(ns, rid, k0, stay)

    def phase2(self, rid, e, decode):
        """Phase 2 at e: the header [e, e + 3) then ``decode`` one-token steps (vLLM frees what leaves)."""
        req = GEN.KP.request(rid, e + HEADER, e)
        assert GEN.KP.alloc(self.manager, req, HEADER) is not None
        for _ in range(decode):
            self.step(req, 1)
        return req


class PinsHoldTests(unittest.TestCase):
    def turn0(self, w, N=3000, ns='ns'):
        """A local turn 0 committed at X = N - 3, held before the next allocation."""
        req = w.cold('r0', N)
        X = N - HEADER
        w.pins.hold(ns, 'r0', X, 'prefill', N)
        for _ in range(4):
            w.step(req, 1)
        w.pins.pin_after(ns, 'r0', X)
        w.free(req)
        w.pins.after_free('r0', ns)
        return X

    def check_window(self, w, end, ns='ns'):
        first, last = kept_blocks(end, W, P)
        window = w.pins.window(ns)
        self.assertEqual({g: [i for i, _ in v] for g, v in window.items()},
                         {g: list(range(first, last)) for g in SLIDING})
        self.assertTrue(all(b.ref_cnt == 1 for v in window.values() for _, b in v))
        self.assertEqual(w.pool.get_num_free_blocks(), w.baseline - 5 * (last - first))
        self.assertEqual((w.pins.holds, w.pins.held_blocks()), ({}, 0))

    def test_continuing_turn_short_and_long_replies(self):
        """The window at e, held at install: a reply shorter than 16 tokens leaves it all in the row, one
        longer than 1,024 tokens moves the whole window past it; the commit pins kept_blocks(e) either way."""
        for C0, e, decode, left in ((2997, 3300, 5, 0), (2997, 3300, 1100, 65),
                                   # About 1,000 tokens: all but three of the window's 65 blocks have left.
                                   (2997, 3300, 1000, 62),
                                   # (e - 1023) % 16 = 15: the header alone moves the window by a block.
                                   (2997, 3326, 2, 1), (2997, 3311, 2, 0)):
            with self.subTest(e=e, decode=decode):
                w = PinsWorld()
                self.turn0(w)
                self.check_window(w, 2997)
                head = w.handover('r1', C0, e)
                w.pins.install('r1')
                held = w.pins.hold('ns', 'r1', e, 'install', e)
                k0, k1 = kept_blocks(e, W, P)
                self.assertEqual(held, {g: tuple((k0 + j, b.block_id) for j, b in enumerate(head[g][1]))
                                        for g in SLIDING})
                self.assertTrue(all(b.ref_cnt == 2 for v in w.pins.holds['r1']['window'].values() for _, b in v))
                req = w.phase2('r1', e, decode)
                self.assertEqual(w.pins.held_left('r1'), left)
                # Every live block is the row's or the hold's: nothing the hold keeps was freed.
                live = {b.block_id for m in w.sliding for b in m.req_to_blocks['r1'] if not b.is_null}
                kept = {b.block_id for v in w.pins.holds['r1']['window'].values() for _, b in v}
                self.assertEqual(w.pool.get_num_free_blocks(), w.baseline - len(live | kept))
                w.pins.pin_after('ns', 'r1', e)
                w.free(req)
                w.pins.after_free('r1', 'ns')
                self.check_window(w, e)
                self.assertEqual(k1 - k0, 64 if (e - W + 1) % P == 0 else 65)
                w.pins.close('ns')
                self.assertEqual(w.pool.get_num_free_blocks(), w.baseline)

    def test_without_the_hold_a_long_reply_loses_the_window(self):
        """The mutation the hold exists for: phase 2 frees the window at e after about 16 tokens."""
        w = PinsWorld()
        self.turn0(w)
        w.handover('r1', 2997, 3300)
        w.pins.install('r1')
        w.phase2('r1', 3300, 40)
        with self.assertRaisesRegex(Abort, 'not live in the request'):
            w.pins.pin_after('ns', 'r1', 3300)
        # A short reply still has it in its row (the pin path without a hold).
        w2 = PinsWorld()
        self.turn0(w2)
        w2.handover('r1', 2997, 3300)
        w2.pins.install('r1')
        req2 = w2.phase2('r1', 3300, 2)
        w2.pins.pin_after('ns', 'r1', 3300)
        w2.free(req2)
        w2.pins.after_free('r1', 'ns')
        self.check_window(w2, 3300)

    def test_local_turn_zero(self):
        for N, decode in ((3000, 1100), (3000, 1), (1000, 30), (40, 3)):
            with self.subTest(N=N, decode=decode):
                w = PinsWorld()
                req = w.cold('r0', N)
                X = N - HEADER
                w.pins.hold('ns', 'r0', X, 'prefill', N)
                for _ in range(decode):
                    w.step(req, 1)
                w.pins.pin_after('ns', 'r0', X)
                w.free(req)
                w.pins.after_free('r0', 'ns')
                self.check_window(w, X)
        # A request that ended at the step that crossed X (no allocation after it): pinned from its row.
        w = PinsWorld()
        req = w.cold('r0', 3000)
        w.pins.pin_after('ns', 'r0', 2997)
        w.free(req)
        w.pins.after_free('r0', 'ns')
        self.check_window(w, 2997)

    def test_cancel_and_discard_release_the_hold(self):
        for where in ('phase 2', 'installed', 'local turn 0'):
            with self.subTest(where=where):
                w = PinsWorld()
                if where == 'local turn 0':
                    req = w.cold('r0', 3000)
                    w.pins.hold('ns', 'r0', 2997, 'prefill', 3000)
                    for _ in range(60):
                        w.step(req, 1)
                    w.free(req)                             # cancelled: vLLM frees it, then the hook
                    self.assertGreater(w.pins.held_blocks(), 0)
                    w.pins.after_free('r0', 'ns')
                    self.assertIsNone(w.pins.window('ns'))
                    self.assertEqual(w.pool.get_num_free_blocks(), w.baseline)
                    continue
                self.turn0(w)
                w.handover('r1', 2997, 3300)
                w.pins.install('r1')
                w.pins.hold('ns', 'r1', 3300, 'install', 3300)
                if where == 'installed':
                    w.pins.discard('r1')                    # before the worker saw it
                else:
                    req = w.phase2('r1', 3300, 1100)
                    w.free(req)
                    w.pins.after_free('r1', 'ns')
                self.assertEqual((w.pins.holds, w.pins.window('ns')), ({}, None))
                self.assertEqual(w.pool.get_num_free_blocks(), w.baseline)
                w.pins.close('ns')

    def test_trim_composes(self):
        """--goal-overlap: the drain's trim frees the pinned blocks before kept_blocks(e), the handover
        then drops nothing, and the hold takes the installed head."""
        w = PinsWorld()
        self.turn0(w)
        k0, k1 = kept_blocks(3300, W, P)
        trimmed = w.pins.trim('ns', k0)
        self.assertEqual(trimmed, 5 * sum(1 for i in range(*kept_blocks(2997, W, P)) if i < k0))
        w.handover('r1', 2997, 3300)
        w.pins.install('r1')
        w.pins.hold('ns', 'r1', 3300, 'install', 3300)
        self.assertEqual(w.pins.held_blocks(), 5 * (k1 - k0))
        req = w.phase2('r1', 3300, 1100)
        w.pins.pin_after('ns', 'r1', 3300)
        w.free(req)
        w.pins.after_free('r1', 'ns')
        self.check_window(w, 3300)

    def test_refusals(self):
        w = PinsWorld(hold=1)
        req = w.cold('r0', 3000)
        w.pins.hold_slots = 0
        with self.assertRaisesRegex(Abort, 'without a free hold slot'):
            w.pins.hold('ns', 'r0', 2997, 'prefill', 3000)          # no hold slots (H = 0)
        w.pins.hold_slots = 1
        with self.assertRaisesRegex(Abort, 'before the cut'):
            w.pins.hold('ns', 'r0', 2997, 'prefill', 2990)          # the cursor before X
        with self.assertRaisesRegex(Abort, 'installed head'):
            w.pins.hold('ns', 'r0', 2997, 'install', 3000)          # not installed
        w.pins.hold('ns', 'r0', 2997, 'prefill', 3000)
        with self.assertRaisesRegex(Abort, 'twice'):
            w.pins.hold('ns', 'r0', 2997, 'prefill', 3000)
        other = w.cold('r9', 500)
        with self.assertRaisesRegex(Abort, 'without a free hold slot'):
            w.pins.hold('ns9', 'r9', 497, 'prefill', 500)           # H = 1 holds already
        with self.assertRaisesRegex(Abort, 'while its request holds it'):
            w.pins.close('ns')
        with self.assertRaisesRegex(Abort, 'not the session'):
            w.pins.pin_after('ns', 'r0', 3000)                      # another end
        with self.assertRaisesRegex(Abort, 'not the session'):
            w.pins.pin_after('other', 'r0', 2997)                   # another namespace
        w.pins.pin_after('ns', 'r0', 2997)
        w.free(req)
        w.free(other)
        w.pins.after_free('r0', 'ns')
        self.check_window(w, 2997)


# ---- 3. end to end: the window layout's engine and worker behind the Gemma chat-text server ------------

import functools  # noqa: E402

import test_chat_text_e2e as CE  # noqa: E402  the scripted model
import test_chat_text_protocol as T  # noqa: E402  prompts and observations
import test_gemma_chat_text_e2e as GCE  # noqa: E402  the Gemma chat-text server on the window fakes
import test_gemma_e2e as GE  # noqa: E402  the window layout's engine and worker, its geometry
import test_gemma_thinking_on as GTO  # noqa: E402  the stand-in of the pinned template, the kits
import test_ms2_e2e as B2  # noqa: E402
import test_ms_e2e as E2E  # noqa: E402
import test_thinking_cut as TC  # noqa: E402  the clients (send, episode)

from remote_ra_live_v2 import admission_flags as AF  # noqa: E402
from remote_ra_live_v2 import admission_overlap as AO  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import goal_async as GA  # noqa: E402
from remote_ra_live_v2 import terminus2_goal as T2  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2 import thinking_cut as CUT  # noqa: E402
from remote_ra_live_v2.composition import compose_frontend  # noqa: E402
from remote_ra_live_v2.contract import closed_pages  # noqa: E402
from remote_ra_live_v2.window_lifecycle import WindowCarry  # noqa: E402
# Imported here, outside the tests' sys.modules patch, so the patch keeps them (test_gemma_fastpath).
from remote_ra_live_v2 import (decode_fastpath, decode_fastpath2, decode_fastpath3, deferred_completion,  # noqa: F401,E402
                               generation_guard, hooks, native_probe, page_compaction, positions, vector_space,
                               window_lifecycle, window_owner, worker_composition, worker_rows)

STRIP = TH.record(TH.CHAT, 'strip', profile=MP.GEMMA4)      # the Gemma 4 strip thinking record


class PinnedHeaderTokenizer:
    """Character-level stand-in of the pinned Gemma tokenizer (test_gemma_thinking_on.ThinkTokenizer: the
    special tokens keep their pinned ids, characters are ord + 1000) with three more pinned ids of plain
    tokens: ``model`` 4368, ``thought`` 45518 and the newline 107. The thinking-on header
    ``<|turn>model\\n`` is then the pinned [105, 4368, 107] (K = |H| = 3, as on the server) and a reply
    opens its channel with the pinned [100, 45518, 107]."""
    BASE = 1000
    WORDS = {'model': 4368, 'thought': 45518, '\n': 107}

    def __init__(self):
        self.table = dict(GTO.SPECIAL, **self.WORDS)
        self.order = sorted(self.table, key=len, reverse=True)
        self.special = {i: t for t, i in GTO.SPECIAL.items()}
        self.words = {i: t for t, i in self.WORDS.items()}
        assert not set(self.special) & set(self.words) and len(self.special) == len(GTO.SPECIAL)

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for token in self.order:
                if text.startswith(token, i):
                    ids.append(self.table[token])
                    i += len(token)
                    break
            else:
                ids.append(ord(text[i]) + self.BASE)
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for t in ids:
            if t in self.special:
                out.append('' if skip_special_tokens else self.special[t])
            else:
                out.append(self.words[t] if t in self.words else chr(t - self.BASE))
        return ''.join(out)


TOK = PinnedHeaderTokenizer()
# The stand-in kit (test_gemma_thinking_on.Kit): this tokenizer and the stand-in of the pinned template.
KIT = GTO.Kit('gtok3', TOK.encode, lambda ids: TOK.decode(ids, True), lambda ids: TOK.decode(ids, False),
              GTO.char_render, GTO.STAND_IN_ON)
HEADER_IDS = tuple(KIT.encode(GTO.HEADER))      # the strip transcripts' generation header: [105, 4368, 107]
assert HEADER_IDS == (MP.TURN_OPEN, 4368, 107) and tuple(KIT.encode(GTO.THOUGHT)) == (MP.CHANNEL_OPEN, 45518, 107)
HEAD = len(KIT.encode(C.GEMMA4_CHAT.reply_head))    # K: the reply head is the whole header
assert HEAD == len(HEADER_IDS) == HEADER == 3
FW, FP = GE.WINDOW, GE.P        # the fakes' window (64) and Page (16)


def reference(messages, kit=None):
    """Stock vLLM's thinking-on prompt of a request: the template's render of its messages with
    enable_thinking=True (every reasoning dropped), tokenized."""
    return (kit or KIT).stock(GTO.plain(messages))


def strip_transcripts(record, thinking=True, strip=True, kit=None):
    """The runtime's Gemma 4 strip transcript factory over the kit: Gemma4SourceRenderer.chat_transcripts
    with its startup checks (the header, the template's thinking-on render, the strip history)."""
    assert (record['header_text'], record['template_kwargs'], thinking, strip) == (
        GTO.HEADER, {'enable_thinking': True}, True, True)
    return GTO.renderer_of(kit or KIT).chat_transcripts(record, thinking=True, strip=True)


@contextlib.contextmanager
def strip_patches():
    """test_gemma_chat_text_e2e's server with the Gemma strip transcripts on both ends, the stand-in
    tokenizer with the pinned header, and the scripted thinking-on model (a thought channel with a draft
    action and JSON, then the original reply: test_gemma_thinking_on.thinking_model)."""
    class Renderer(GCE.S.Renderer):
        def chat_transcripts(self, record, thinking=False, strip=False):
            return strip_transcripts(record, thinking, strip)
    with patch.object(GCE, 'transcripts', strip_transcripts), patch.object(GCE, 'ChatRenderer', Renderer), \
            patch.object(GCE, 'GTOK', TOK), patch.object(CE, 'model', GTO.thinking_model(CE.model)):
        yield


def strip_layout(policy, limit, hold=None):
    return dataclasses.replace(GE.window_layout(policy, limit), hold_slots=limit if hold is None else hold)


def strip_config(policy, output, slots, limit, layout, overlap=None, release=False, **record):
    """A Gemma chat-text strip server's startup config on the window layout (test_gemma_e2e's geometry and
    selection contract; the Gemma strip records; the layout's hold slots in ``kv_layout``)."""
    value = GE.config(policy, output, C.slot_tasks(slots), layout)
    value.update(protocol=C.PROTOCOL, task=C.slot_tasks(1)[0], max_running_sessions=limit, model_profile=MP.GEMMA4)
    value['chat_text'] = C.record(slots=slots, profile=MP.GEMMA4, thinking=True, strip=True, **record)
    value['thinking'] = STRIP
    if policy == 'goal_query':
        value['chat_text_goal'] = T2.record()
    if overlap is not None:
        value['admission'] = AF.record(admission_overlap=overlap)
    if release:
        value['release_idle_pages'] = True
    assert CUT.check_config(value) is True          # the pin a real server's startup checks
    return value


def attach_fast(h, fast):
    """attach_window_worker's and compose_engine's decode fast paths 1-3 (test_gemma_fastpath(3))."""
    import test_gemma_fastpath as GF
    import test_gemma_fastpath3 as GF3
    if fast:
        (None, GF.attach_fast, GF.attach_fast2, GF3.attach_fast3)[fast](h)


def window_engine(fast=0, overlap=None, attached=None):
    """test_gemma_chat_text_e2e.GemmaEngine (the batched engine loop over the window layout's engine and
    worker) with the decode fast paths ``fast`` and, with ``overlap``, --admission-overlap K as
    compose_engine attaches it (test_thinking_cut.admission_overlap, test_gemma_chat_text_goal.goal_run)."""
    class Engine(GCE.GemmaEngine):
        def __init__(self, policy, output, server, **kwargs):
            if overlap is None:
                super().__init__(policy, output, server, **kwargs)
            else:
                with patch.object(GA, 'attach', functools.partial(GA.attach, overlap=overlap)):
                    super().__init__(policy, output, server, **kwargs)
                self.lifecycle.lease = AF.timeouts(overlap)[1]
                AO.attach_engine(self.core, self.lifecycle, AF.record(admission_overlap=overlap))
            attach_fast(self, fast)
            if attached is not None:
                attached.append(self)
    return Engine


class StripChatServer(GCE.GemmaChatServer):
    """test_gemma_chat_text_e2e's server with the prefill chunking seam of test_chat_text_e2e (``chunk``:
    'page', one Page per prefill step, or 'whole') and the replies encoded by the stand-in tokenizer."""
    chunk = staticmethod(lambda tokens: 'page')

    async def schedule(self, rid, external, prompt):
        h = self.engine
        info = h.infos[rid]
        info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
        messages = self.replies.pop(external)
        slot = h.sessions.index(info['sid'])
        self.served.append(dict(rid=rid, slot=slot, messages=messages, tokens=list(info['tokens'])))
        outputs = (TOK.encode(CE.model(messages)) + [MP.TURN_CLOSE])[:h.requests[rid].sampling_params.max_tokens]
        return await h.generate(rid, outputs, self.chunk(info['tokens']), self.holds.pop(slot, None))


def attach_strip(h, split=False):
    """compose_engine's and attach_window_worker's strip attachments on the fake engine and worker: the
    runtime's thinking_cut entry points, which hand the window layout to window_hold (``split``: window_hold's
    optional forward end at X for a local turn 0, which the runtime does not use)."""
    bridge = h.core._remote_ra_live_v2_bridge
    if split:
        WH.attach_engine(h.adapter, bridge, split=True)
    else:
        CUT.attach_engine(h.adapter, bridge)
        assert not getattr(bridge, 'commit_at_cut', False)
    CUT.attach_worker(h.owner, h.qwen, h.backend)
    assert h.adapter.commit_at_cut and h.adapter.window.hold_slots == h.layout.hold_slots >= 1
    assert h.owner.commit_at_cut and h.qwen.commit_at_cut and h.backend.commit_at_cut
    h.recorder = Recorder(h)
    return h


class Recorder:
    """What a strip engine and worker did with the window at the cut, checked as it happens: each hold
    (source, end, cursor) and release, the hold count against H, each completion's mirrors (engine,
    worker, the pinned window and the WindowCarry at X, its evidence record) and the drains."""

    def __init__(self, h):
        self.h, self.holds, self.releases, self.mismatches, self.completions = h, [], [], [], {}
        self.most, self.drains, self.cut_windows = 0, [], []
        pins, adapter, backend = h.adapter.window, h.adapter, h.backend
        hold, release, complete, drain, cut = pins.hold, pins.release, adapter.complete, backend.drain, \
            backend.hold_cut_window

        def recorded_hold(ns, rid, end, source, cursor):
            value = hold(ns, rid, end, source, cursor)
            self.holds.append((rid, source, end, cursor))
            self.most = max(self.most, len(pins.holds))
            return value

        def recorded_release(rid):
            count = release(rid)
            if count:
                self.releases.append((rid, count))
            return count

        def checked(binding):
            receipt = complete(binding)
            self.completed(binding)
            return receipt

        def recorded_drain(identity, operation, generation, **kwargs):
            self.drains.append((identity.request, operation, kwargs.get('positioned', False)))
            return drain(identity, operation, generation, **kwargs)

        def recorded_cut(identity, end, interval, generation):
            self.cut_windows.append((identity.request, end, interval))
            return cut(identity, end, interval, generation)
        pins.hold, pins.release, adapter.complete = recorded_hold, recorded_release, checked
        backend.drain, backend.hold_cut_window = recorded_drain, recorded_cut

    def completed(self, binding):
        h = self.h
        adapter, P, layout = h.adapter, h.adapter.page_tokens, h.layout
        s = adapter.sessions[binding.session]
        T_, X_ = binding.tokens, len(binding.tokens) - len(HEADER_IDS)
        Q = X_ // P * P
        w = h.owner.sessions[s.namespace]
        tail = adapter.owner.tails.get(s.namespace)
        carry = h.backend.carry[binding.session]
        first, last = kept_blocks(X_, layout.window_tokens, P)
        window = adapter.window.window(s.namespace) or {}
        hold = adapter.cut_records[binding.request].get('window_hold')
        checks = dict(
            engine=(s.computed_tokens == T_[:X_] and s.next_token == T_[X_] and s.sealed_tokens == T_[:Q] and
                    set(s.catalog) == set(range(Q // P)) and all(p < Q // P for p in s.resident)),
            tail=((tail is None) == (X_ == Q) and (tail is None or (tail.page, tail.start, tail.valid_end) ==
                                                    (Q // P, Q, X_))),
            worker=(w['computed_end'] == X_ and w['sealed_end'] == Q and tuple(w['resident']) == tuple(s.resident) and
                    set(w['catalog']) == set(s.catalog)),
            # The session pins the window at X (held through phase 2); the worker's carry names the same
            # blocks at X, and the engine echoes its digest.
            window=({g: [i for i, _ in v] for g, v in window.items()} ==
                    {g: list(range(first, last)) for g in GE.SLIDING} and
                    binding.request not in adapter.window.holds),
            carry=(type(carry) is WindowCarry and carry.computed_end == X_ and carry.digest == s.boundary ==
                   w['carry_digest'] and carry.blocks() == {g: tuple(b.block_id for _, b in v)
                                                            for g, v in window.items()}),
            evidence=(hold is not None and (hold['end'], hold['first'], hold['last']) == (X_, first, last) and
                      0 <= hold['left'] <= last - first))
        bad = sorted(k for k, ok in checks.items() if not ok)
        if bad:
            self.mismatches.append(('completion', binding.request, bad))
        self.completions[binding.request] = dict(X=X_, hold=hold, local=binding.local_turn)


def strip_window_server(test, policy='full', slots=8, limit=4, hold=None, blocks=None, split=False, fast=0,
                        release=False, goals=False, overlap=None, engine=None, attach=True, server=None, **record):
    """The Gemma chat-text server over the window layout's engine and worker with H = ``hold`` hold slots
    (``limit`` when None), the Gemma strip transcripts and the thinking model, attached as a strip server
    attaches (attach_strip). ``overlap``: --admission-overlap K; ``engine``: another engine class."""
    output = Path(test.enterContext(tempfile.TemporaryDirectory()))
    layout = strip_layout(policy, limit, hold)
    value = strip_config(policy, output, slots, limit, layout, overlap=overlap, release=release, **record)
    test.enterContext(patch('remote_ra_http.config.load', lambda: value))
    test.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
    if not getattr(test, '_strip_patched', False):
        # Once per test: the thinking model wraps test_chat_text_e2e's model (twice, it would think twice).
        test.enterContext(strip_patches())
        test._strip_patched = True
    engines = []
    Engine = engine or window_engine(fast, overlap, engines)
    frontend = (functools.partial(compose_frontend, admission_overlap=overlap) if overlap is not None
                else GCE.compose_frontend)
    with patch.object(GCE, 'GemmaEngine', Engine), patch.object(GCE, 'compose_frontend', frontend):
        srv = (server or StripChatServer)(policy, value, layout, output, test.router, test.Completion,
                                          fa_blocks=blocks or GE.FA_BLOCKS, release=release, goals=goals)
    test.addAsyncCleanup(srv.stop)
    h = srv.engine
    if overlap is not None:
        # compose_frontend pinned the ACK wait from K; the fake engine client keeps vLLM's former 300-s
        # bound (test_gemma_chat_text_goal.goal_run).
        client = srv.llm.engine_core
        add = client.registered_add_async
        client.registered_add_async = lambda frame, timeout: add(frame, min(timeout, 300))
        goals_table = h._remote_ra_goal_admissions

        def close():
            if goals_table is not None:
                for executor in list(getattr(goals_table, '_threads', {}).values()) + [goals_table._thread]:
                    executor.shutdown(wait=True, cancel_futures=True)
        test.addCleanup(close)
    if attach:
        attach_strip(h, split)
    return srv


def check_window_strip(test, srv, *, completions=None, split=False):
    """Every served request of a window strip server: the prompt is the stock render of its messages,
    the completion committed at X with the window held at X (the native file's plan, truncate and its
    window_hold, the worker frames and carry, the runtime row), the re-prefill of a continuing turn
    (phase 1 over [C, X), the hold from the install), local turn 0 (the hold from the prefill or, for a
    request that ended at the step that crossed X, the row), and the recorder's checks."""
    h = srv.engine
    P = h.adapter.page_tokens
    rows = {r['request_id']: r for r in srv.rows()}
    holds = {rid: (source, end, cursor) for rid, source, end, cursor in h.recorder.holds}
    checked = 0
    for entry in srv.served:
        rid, messages, tokens = entry['rid'], entry['messages'], entry['tokens']
        test.assertEqual(tokens, reference(messages), rid)
        if rid not in rows:
            continue          # cancelled or failed after scheduling
        native, row = srv.native(rid), rows[rid]
        test.assertEqual(native['prompt_token_ids'], tokens)
        N, g = len(tokens), row['chat_text']['completion_tokens']
        X_ = N - len(HEADER_IDS)
        Q = X_ // P * P
        first, last = kept_blocks(X_, FW, P)
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
        generated = native['truncate']['dropped']['generated_ids']
        hold = native['truncate']['window_hold']
        test.assertEqual(tokens[X_:], list(HEADER_IDS))
        # The dropped branch's end of reasoning: the first <channel|> id (one token), None when the reply
        # was cut before it.
        think_end = generated.index(MP.CHANNEL_CLOSE) if MP.CHANNEL_CLOSE in generated else None
        test.assertEqual(native['truncate'], dict(cut=X_, dropped=dict(start=X_, end=N + g, header_ids=tokens[X_:],
                                                                       generated_ids=generated, think_end=think_end),
                                                  window_hold=hold))
        test.assertEqual((hold['end'], hold['first'], hold['last']), (X_, first, last))
        stage = row['stage_b']
        test.assertEqual((stage['computed_end'], stage['sealed_end'], stage['phase2_end']), (X_, Q, N + g - 1))
        test.assertEqual(row['truncate'], dict(cut=X_, dropped_tokens=N + g - X_, header_tokens=len(HEADER_IDS),
                                               generated_tokens=g, think_end=think_end,
                                               reprefill=native.get('reprefill'), window_hold=hold))
        if local:
            # The re-prefill [C, X) on the window the session pinned at its last cut; the window at X is
            # the one phase 1 ended in, held from the install.
            C0 = plan['computed_start']
            piece = KIT.encode(CUT.gemma4_reply_piece(messages[-2]['content']))
            test.assertEqual(native['reprefill'], dict(start=C0, piece_end=C0 + len(piece)))
            test.assertEqual(tokens[C0:C0 + len(piece)], piece)
            # K = |H|: the re-prefilled piece starts with the whole header the cut dropped.
            test.assertEqual(tuple(tokens[C0:C0 + HEAD]), HEADER_IDS)
            test.assertEqual((native['stable_end'], native['observation_span']), (C0 + HEAD, [C0 + HEAD, X_]))
            phase1 = [f for f in native['worker_frames'] if f['operation'] == 'phase1']
            # Phase 1 over [C, X): one forward, or chunks when it is longer than one native forward.
            test.assertEqual((phase1[0]['start'], phase1[-1]['end']), (C0, X_))
            test.assertEqual([f['start'] for f in phase1[1:]], [f['end'] for f in phase1[:-1]])
            test.assertEqual((hold['source'], hold['held_at']), ('install', X_))
            test.assertEqual(holds.get(rid), ('install', X_, X_))
            test.assertEqual(row['thinking']['history'], CUT.CUT)
        else:
            test.assertNotIn('reprefill', native)
            test.assertIn(hold['source'], ('prefill', 'row'))
            if hold['source'] == 'prefill':
                test.assertEqual(holds.get(rid), ('prefill', X_, hold['held_at']))
                test.assertGreaterEqual(hold['held_at'], X_)
            else:
                test.assertIsNone(hold['held_at'])
                test.assertNotIn(rid, holds)
            ends = [s + c for step in h.runner.steps for r, s, c in step if r == rid]
            if split:
                test.assertIn(X_, ends)            # the prefill forward that holds X ends there
            test.assertIn(row['thinking']['history'], (CUT.FRESH, CUT.REBUILD))
        test.assertIn((rid, 'complete', True), h.recorder.drains)
        checked += 1
    test.assertEqual(h.recorder.mismatches, [])
    test.assertLessEqual(h.recorder.most, h.adapter.window.hold_slots)
    if completions is not None:
        test.assertEqual(checked, completions)
    return checked


def boundary(test, srv):
    """Nothing runs: pool 0 holds exactly the sessions' pins, pool 1 exactly their windows (each the
    window at its last cut); no hold is left, in the engine or in the worker."""
    h = srv.engine
    pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
    test.assertEqual(h.pool.get_num_free_blocks(), h.baseline - pins)
    windows = 0
    for s in h.adapter.sessions.values():
        window = h.adapter.window.window(s.namespace)
        if window is None:
            continue
        first, last = kept_blocks(s.computed_end, FW, FP)
        test.assertEqual({g: [i for i, _ in v] for g, v in window.items()},
                         {g: list(range(first, last)) for g in GE.SLIDING})
        test.assertTrue(all(b.ref_cnt == 1 for v in window.values() for _, b in v))
        windows += sum(len(v) for v in window.values())
    test.assertEqual(h.window_pool.get_num_free_blocks(), h.window_baseline - windows)
    test.assertEqual((h.adapter.window.holds, h.adapter.window.held_blocks()), ({}, 0))
    test.assertFalse(h.owner.window_holds)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    """T3: the strip commit of the window layout on the engine and worker fakes."""
    maxDiff = None
    send, episode = TC.EndToEndTests.send, TC.EndToEndTests.episode

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def test_sessions_commit_at_the_cut_with_the_held_window(self):
        """Three clients decode together on observation_query (selection from the second turn), one
        sending a reasoning back (dropped): every completion commits at X with the window held at X;
        every reply is longer than the window, so the whole window at X left the row before the commit;
        the next turn's phase 1 reads exactly the last W tokens from it."""
        srv = strip_window_server(self, 'observation_query', slots=8, limit=4)
        await asyncio.gather(*(self.episode(srv, k, turns=5, keep=k == 1, pad=k) for k in range(3)))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(check_window_strip(self, srv, completions=15), 15)
        holds = [c['hold'] for c in srv.engine.recorder.completions.values()]
        self.assertTrue(all(h['left'] == h['last'] - h['first'] for h in holds if h['source'] == 'install'))
        self.assertGreater(sum(srv.engine.runner.windows.values()), 0)
        self.assertGreater(srv.engine.recorder.most, 1)
        boundary(self, srv)

    async def test_short_replies_and_every_cut_offset(self):
        """Replies shorter than a block (a 5-token bound: the hold's blocks are all still in the row at
        the commit, ``left`` 0 unless the header moved the window) next to long ones, with cuts at every
        offset mod P; then the sessions continue on the windows pinned from those holds."""
        srv = strip_window_server(self, 'full', slots=24, limit=8)
        for k in range(8):
            await self.episode(srv, k, turns=4, pad=2 * k, sizes=lambda j, k=k: (j * 3 + k) % 5,
                               max_tokens=lambda j, k=k: 5 if (j + k) % 2 else None)
        self.assertEqual(check_window_strip(self, srv, completions=32), 32)
        done = srv.engine.recorder.completions.values()
        short = [c['hold'] for c in done if c['local'] and c['hold']['left'] < c['hold']['last'] - c['hold']['first']]
        self.assertTrue(short)
        self.assertGreaterEqual(len({c['X'] % FP for c in done}), 10)
        boundary(self, srv)

    async def test_local_turn_zero_with_and_without_the_split(self):
        """Fresh sessions of every cut offset mod P. Without the split (the runtime's attachment) the
        window at X is held at the first allocation past it: the next prefill chunk when the 3-token
        header spans two Pages of a one-Page-per-step prefill, else the first decode step; a reply of one
        token whose prefill ended at its crossing step ends there, and the commit pins the window from
        the row (no hold). With the split the prefill forward that holds X ends there and the next one
        is held at X."""
        for split, chunk, sources in ((False, 'page', {'prefill', 'row'}), (False, 'whole', {'prefill', 'row'}),
                                      (True, 'page', {'prefill'}), (True, 'whole', {'prefill'})):
            with self.subTest(split=split, chunk=chunk):
                srv = strip_window_server(self, 'full', slots=20, limit=16, split=split)
                srv.chunk = lambda tokens, chunk=chunk: chunk
                pads = range(FP) if SLOW else range(0, FP, 3)
                for k in pads:
                    await self.episode(srv, k, turns=2, pad=k, max_tokens=lambda j, k=k: 1 if k % 5 == 0 else None)
                self.assertEqual(check_window_strip(self, srv, completions=2 * len(pads), split=split), 2 * len(pads))
                done = [c['hold'] for c in srv.engine.recorder.completions.values() if not c['local']]
                self.assertEqual({h['source'] for h in done}, sources)
                if split:
                    self.assertTrue(all(h['held_at'] == h['end'] for h in done))
                boundary(self, srv)

    async def test_rebuild_after_a_cut_and_a_cancel_in_phase_2(self):
        srv = strip_window_server(self, 'observation_query', slots=4, limit=2)
        h = srv.engine
        messages = [dict(role='user', content=T.first_prompt('Rebuild.'))]
        status, body = await self.send(srv, messages)
        messages += [dict(role='assistant', content=body['choices'][0]['message']['content']),
                     dict(role='user', content=T.observation(0))]
        status, body = await self.send(srv, messages)
        # An unwound history: a rebuild frees the window at the last cut; local turn 0 holds again.
        unwound = messages[:1] + [dict(role='assistant', content='{"rewritten": true}'),
                                  dict(role='user', content=T.observation(1))]
        status, body = await self.send(srv, unwound)
        self.assertEqual(status, 200)
        row = srv.rows()[-1]
        self.assertEqual((row['stage_b']['rebuild'], row['stage_b']['local_turn'], row['thinking']['history']),
                         (True, 0, CUT.REBUILD))
        nxt = unwound + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                         dict(role='user', content=T.observation(2))]
        # A cancel while phase 2 decodes past X: its hold is released, the session keeps its committed
        # cut and is poisoned; the client's retry rebinds to a fresh slot.
        parked = srv.hold(0, 4)
        receive = asyncio.Queue()
        task = asyncio.create_task(srv.chat_request(nxt, receive=receive))
        await asyncio.wait_for(parked.parked.wait(), 10)
        s = h.adapter.sessions[srv.live.sessions[0].session_id]
        rid = next(r for r, *_ in reversed(h.recorder.holds))
        self.assertIn(rid, h.adapter.window.holds)
        committed = (s.computed_tokens, s.boundary, s.tail)
        receive.put_nowait(dict(type='http.disconnect'))
        await B2.settle(lambda: any(e[0] == 'abort' for e in h.events))
        parked.gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((s.computed_tokens, s.boundary, s.tail, s.poisoned), committed + (True,))
        self.assertNotIn(rid, h.adapter.window.holds)
        self.assertIn(rid, [r for r, _ in h.recorder.releases])
        status, body = await self.send(srv, nxt)
        self.assertEqual((status, srv.rows()[-1]['chat_text']['rebind_reason']), (200, 'failed'))
        check_window_strip(self, srv, completions=4)

    async def test_the_gate_with_one_hold_slot(self):
        """H = 1 of four window slots: three clients at once. A turn waits at the gate (CAPACITY_WAIT)
        while another strip reply holds the slot, and is admitted after its commit: every turn served, no
        preemption, never two holds."""
        waits = []
        real = WC.hold_wait

        def counted(lifecycle):
            try:
                return real(lifecycle)
            except BaseException as wait:
                waits.append(str(wait))
                raise
        self.enterContext(patch.object(WC, 'hold_wait', counted))
        srv = strip_window_server(self, 'observation_query', slots=8, limit=4, hold=1)
        await asyncio.gather(*(self.episode(srv, k, turns=3, pad=k) for k in range(3)))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(check_window_strip(self, srv, completions=9), 9)
        self.assertEqual(srv.engine.recorder.most, 1)
        self.assertTrue(waits and all('window hold slots: 1 of 1 in use' in w for w in waits), waits[:3])
        # Never two requests in phase 2 at once: no step decodes two requests.
        self.assertTrue(all(len(b.rows) == 1 for b in srv.engine.batches))
        boundary(self, srv)

    async def test_the_hold_terms_fit_a_full_pool(self):
        """Every window slot and every hold slot taken: four sessions of long replies on a layout of four
        window slots and four hold slots decode together; the gate's hold terms keep pool 1 from running
        out (a running request that finds no block is a fatal preemption here)."""
        srv = strip_window_server(self, 'full', slots=8, limit=4, hold=4)
        await asyncio.gather(*(self.episode(srv, k, turns=4, pad=k) for k in range(4)))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(check_window_strip(self, srv, completions=16), 16)
        self.assertGreater(max(len(b.rows) for b in srv.engine.batches), 1)
        boundary(self, srv)


def served(srv):
    """Each completed request in order: its prompt, its selection, its generated ids and its window hold."""
    rows = {r['request_id']: r for r in srv.rows()}
    out = []
    for entry in srv.served:
        if entry['rid'] in rows:
            native = srv.native(entry['rid'])
            out.append((tuple(entry['tokens']), tuple(rows[entry['rid']]['selected_logical_pages'] or ()),
                        tuple(native['truncate']['dropped']['generated_ids']), native['truncate']['window_hold']))
    return out


# ---- 4. the flags (T4) -------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class FlagTests(unittest.IsolatedAsyncioTestCase):
    """The strip commit of the window layout with the decode fast paths, --deferred-completion-io,
    --admission-overlap and --release-idle-pages: every check of section 3 holds, and what the requests
    computed and committed is the plain run's."""
    maxDiff = None
    send, episode = TC.EndToEndTests.send, TC.EndToEndTests.episode

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def workload(self, srv, clients=2, turns=4):
        for k in range(clients):
            await self.episode(srv, k, turns=turns, pad=k, max_tokens=lambda j, k=k: 5 if (j + k) % 3 == 2 else None)

    async def test_decode_fast_paths(self):
        runs = {}
        for fast in (0, 1, 2, 3):
            with self.subTest(fast=fast):
                srv = strip_window_server(self, 'observation_query', slots=6, limit=3, fast=fast)
                await self.workload(srv)
                self.assertEqual(check_window_strip(self, srv, completions=8), 8)
                boundary(self, srv)
                runs[fast] = served(srv)
                if fast >= 2:
                    self.assertTrue(srv.engine.owner.window_rows)
                if fast == 3:
                    self.assertTrue(srv.engine.owner.window_steady and srv.engine.backend.decode_fastpath3)
                await srv.stop()
        self.assertEqual(runs[1], runs[0])
        self.assertEqual(runs[2], runs[0])
        self.assertEqual(runs[3], runs[0])

    async def test_deferred_completion_io(self):
        import shutil
        import test_deferred_completion as TD
        engines = []
        root = Path(tempfile.mkdtemp(dir=os.environ.get('TMP')))
        self.addCleanup(shutil.rmtree, root, True)
        Engine = TD.batch_engine(window_engine(), engines, shadow_root=root)
        srv = strip_window_server(self, 'observation_query', slots=6, limit=3, engine=Engine)
        await self.workload(srv)
        self.assertEqual(check_window_strip(self, srv, completions=8), 8)
        await srv.stop()
        # Every native file the writer wrote equals the synchronous path's, the window hold included.
        self.assertEqual(len(engines), 1)
        self.assertEqual((engines[0].writer.mismatches, engines[0].writer.stats()['errors']), ([], 0))
        self.assertGreaterEqual(engines[0].writer.compared, 8)

    async def test_admission_overlap(self):
        """--admission-overlap 3: three observation_query sessions admitted while the others decode, the
        ledger counts the in-flight admissions' hold terms and the gate their hold slots (H = 2 of 3)."""
        srv = strip_window_server(self, 'observation_query', slots=6, limit=3, hold=2, blocks=6000, overlap=3)
        await asyncio.gather(*(self.episode(srv, k, turns=4, pad=k) for k in range(3)))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(check_window_strip(self, srv, completions=12), 12)
        h = srv.engine
        self.assertTrue(h.endpoint.phase1.overlap and h.lifecycle.reservations is not None)
        self.assertEqual((h.lifecycle.permits, h.recorder.most), (3, 2))
        boundary(self, srv)

    async def test_release_idle_pages_under_pressure(self):
        """Pool 0 at the selective rule's minimum: idle sessions' Pages are released and readmitted while
        strip replies hold their windows (the windows never leave pool 1); every turn served."""
        layout = strip_layout('observation_query', 6, hold=3)
        srv = strip_window_server(self, 'observation_query', slots=12, limit=6, hold=3, release=True,
                                  blocks=GCE.selective_blocks(layout, 6))
        await asyncio.gather(*(self.episode(srv, k, turns=3, pad=k, sizes=lambda j: 12) for k in range(5)))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(check_window_strip(self, srv, completions=15), 15)
        from remote_ra_live_v2 import idle_release as R
        path = srv.output / R.EVENTS
        events = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
        self.assertTrue({'evict', 'readmit'} <= {e['event'] for e in events}, events[:3])
        boundary(self, srv)


# ---- 5. mutations --------------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MutationTests(unittest.IsolatedAsyncioTestCase):
    """A broken hold fails: a request refused or failed closed, or check_window_strip's checks."""
    maxDiff = None
    send = TC.EndToEndTests.send

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def outcome(self, mutate=None, clients=1, turns=3, hold=None, limit=2, **kw):
        srv = strip_window_server(self, 'observation_query', slots=6, limit=limit, hold=hold, **kw)
        if mutate is not None:
            mutate(srv)

        async def client(k):
            messages = [dict(role='user', content=T.first_prompt('Mutation %d.' % k))]
            for j in range(turns):
                status, body = await self.send(srv, messages)
                if status != 200:
                    return 'status %d: %s' % (status, body['error']['message'])
                messages = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                                       dict(role='user', content=T.observation(j, size=j))]
            return 'ok'
        found = [f for f in await asyncio.gather(*(client(k) for k in range(clients))) if f != 'ok']
        failures = srv.rows('failures.jsonl')
        if found or failures:
            return (found + ['failure: %s' % json.dumps(f)[:600] for f in failures])[0]
        try:
            check_window_strip(self, srv)
            boundary(self, srv)
        except AssertionError as failure:
            return 'check: %s' % str(failure)[:300]
        return 'ok'

    async def test_unmutated_passes(self):
        self.assertEqual(await self.outcome(), 'ok')
        self.assertEqual(await self.outcome(clients=3, limit=4, hold=1), 'ok')

    async def test_no_hold(self):
        def mutate(srv):
            srv.engine.adapter.window.hold = lambda *args, **kwargs: {}
        found = await self.outcome(mutate)
        self.assertIn('a strip completion commits the window it held at the cut', found)

    async def test_a_hold_without_its_reference(self):
        """The hold records the blocks but takes no reference: vLLM frees them as they leave the row and
        hands them out again; the worker refuses the step that names one."""
        def mutate(srv):
            pins = srv.engine.adapter.window
            hold = pins.hold

            def unreferenced(*args):
                touch, pins.pool.touch = pins.pool.touch, lambda blocks: None
                try:
                    return hold(*args)
                finally:
                    pins.pool.touch = touch
            pins.hold = unreferenced
        found = await self.outcome(mutate, clients=2, limit=2)
        self.assertNotEqual(found, 'ok')
        self.assertTrue(any(text in found for text in ('aliases a held cut window block', 'lost their reference',
                                                       'not the session', 'native free left a window block with '
                                                       'another owner')), found)

    async def test_the_worker_window_at_another_end(self):
        def mutate(srv):
            owner = srv.engine.owner
            cut = owner.cut_window

            def late(identity, end, interval, generation):
                window = cut(identity, end, interval, generation)
                later = owner.final_window(identity, min(end + FP, interval[0] + interval[1]), generation)
                return later if later != window else window
            owner.cut_window = late
        found = await self.outcome(mutate, turns=4)
        self.assertNotEqual(found, 'ok')

    async def test_the_engine_hold_at_another_end(self):
        def mutate(srv):
            pins = srv.engine.adapter.window
            hold = pins.hold
            pins.hold = lambda ns, rid, end, source, cursor: hold(ns, rid, end - FP, source, cursor) \
                if end > FP else hold(ns, rid, end, source, cursor)
        found = await self.outcome(mutate)
        self.assertIn("the window at the cut is not the request's own live row", found)

    async def test_no_hold_slot_cap(self):
        """Without the gate's hold count the third concurrent strip reply finds no hold slot: the engine
        refuses its admission (fail-closed) instead of the gate making it wait."""
        self.enterContext(patch.object(WC, 'hold_wait', lambda lifecycle: None))
        found = await self.outcome(clients=3, limit=4, hold=1)
        self.assertNotEqual(found, 'ok')
        self.assertIn('without a free hold slot', found)


# ---- 6. identity: H = 0 holds, reserves, pins and counts nothing; the attachments ----------------------

class AttachmentTests(unittest.TestCase):
    def adapter(self, hold):
        case = GEN.AdapterLayoutTests('test_window_layout_gets_window_pins')
        return case.adapter(dataclasses.replace(I0.for_window_server(16, 2, 1024, 8192), hold_slots=hold))

    def test_engine_and_worker_attachments(self):
        bridge = NS()
        adapter = self.adapter(2)
        self.assertEqual((adapter.window.hold_slots, adapter.commit_at_cut), (0, False))   # until attached
        WH.attach_engine(adapter, bridge)
        self.assertEqual((adapter.window.hold_slots, adapter.commit_at_cut, adapter.cut_records), (2, True, {}))
        self.assertFalse(getattr(bridge, 'commit_at_cut', False))
        with self.assertRaisesRegex(LiveV2Error, 'collision'):
            WH.attach_engine(adapter, bridge)
        split = NS()
        WH.attach_engine(self.adapter(1), split, split=True)
        self.assertTrue(split.commit_at_cut)
        with self.assertRaisesRegex(LiveV2Error, 'needs hold slots'):
            WH.attach_engine(self.adapter(0), NS())
        with self.assertRaisesRegex(LiveV2Error, 'needs the window layout'):
            WH.attach_engine(NS(window=None, layout=None), NS())
        # The worker: the three window objects, once.
        from remote_ra_live_v2.window_owner import WindowAllocationOwner
        owner = WindowAllocationOwner.__new__(WindowAllocationOwner)
        qwen, backend = NS(state_kind='sliding'), NS(window=True)
        WH.attach_worker(owner, qwen, backend)
        self.assertTrue(owner.commit_at_cut and qwen.commit_at_cut and backend.commit_at_cut)
        with self.assertRaisesRegex(LiveV2Error, 'collision'):
            WH.attach_worker(owner, qwen, backend)
        with self.assertRaisesRegex(LiveV2Error, 'window worker'):
            WH.attach_worker(NS(), NS(state_kind='mamba'), NS())

    def test_thinking_cut_hands_the_window_layout_to_the_hold(self):
        """The runtime's entry points (compose_engine, attach_window_worker): thinking_cut.attach_engine and
        attach_worker attach the hold on the window layout, without the local-turn-0 split, and refuse a
        layout without hold slots; a Qwen adapter and worker attach as before."""
        bridge = NS()
        adapter = self.adapter(2)
        CUT.attach_engine(adapter, bridge)
        self.assertEqual((adapter.window.hold_slots, adapter.commit_at_cut, adapter.cut_records), (2, True, {}))
        self.assertFalse(getattr(bridge, 'commit_at_cut', False))
        with self.assertRaisesRegex(LiveV2Error, 'collision'):
            CUT.attach_engine(adapter, bridge)
        with self.assertRaisesRegex(LiveV2Error, 'needs hold slots'):
            CUT.attach_engine(self.adapter(0), NS())
        from remote_ra_live_v2.window_owner import WindowAllocationOwner
        owner = WindowAllocationOwner.__new__(WindowAllocationOwner)
        qwen, backend = NS(state_kind='sliding'), NS(window=True)
        CUT.attach_worker(owner, qwen, backend)
        self.assertTrue(owner.commit_at_cut and qwen.commit_at_cut and backend.commit_at_cut)
        with self.assertRaisesRegex(LiveV2Error, 'collision'):
            CUT.attach_worker(owner, qwen, backend)
        with self.assertRaisesRegex(LiveV2Error, 'window worker'):
            CUT.attach_worker(NS(), NS(state_kind='mamba'), NS(window=True))
        # Qwen: no window, the split at local turn 0 (unchanged).
        qwen_adapter, qwen_bridge, parts = NS(window=None), NS(), (NS(), NS(), NS())
        CUT.attach_engine(qwen_adapter, qwen_bridge)
        CUT.attach_worker(*parts)
        self.assertTrue(qwen_adapter.commit_at_cut and qwen_bridge.commit_at_cut and
                        all(p.commit_at_cut for p in parts))
        source = (ROOT / 'source-overlay' / 'remote_ra_live_v2' / 'thinking_cut.py').read_text(encoding='utf-8')
        self.assertNotIn('HOLD-PENDING', source)

    def test_pinned_hold_slots(self):
        layout = dataclasses.replace(I0.for_window_server(16, 2, 1024, 8192), hold_slots=2)
        self.assertEqual(WH.pinned_hold_slots({}, I0.for_window_server(16, 2, 1024, 8192)), 0)
        with self.assertRaisesRegex(LiveV2Error, 'need --thinking-history strip'):
            WH.pinned_hold_slots({}, layout)
        # A Gemma config with Qwen's strip record is refused (the merge with gemma-thinking: thinking_cut
        # takes the Gemma profile's own record); a config thinking_cut accepts pins H.
        with self.assertRaisesRegex(LiveV2Error, 'differs from the protocol pins'):
            WH.pinned_hold_slots(dict(thinking=TC.STRIP, model_profile=MP.GEMMA4, protocol=C.PROTOCOL), layout)
        gemma = dict(thinking=STRIP, model_profile=MP.GEMMA4, protocol=C.PROTOCOL,
                     kv_layout=I0.to_additional_config(layout)[I0.LAYOUT_KEY])
        self.assertEqual(WH.pinned_hold_slots(gemma, layout), 2)
        # The same config without hold slots in its record: the strip pin refuses it (H = 0).
        with self.assertRaisesRegex(LiveV2Error, 'needs hold slots for the sliding window at the cut'):
            WH.pinned_hold_slots(dict(gemma, kv_layout=I0.to_additional_config(
                I0.for_window_server(16, 2, 1024, 8192))[I0.LAYOUT_KEY]), layout)
        with patch.object(CUT, 'pinned', lambda config: True):
            self.assertEqual(WH.pinned_hold_slots({}, layout), 2)
            with self.assertRaisesRegex(LiveV2Error, 'refuses --async-scheduling'):
                WH.pinned_hold_slots(dict(async_scheduling=True, decode_fastpath2=True), layout)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class IdentityTests(unittest.TestCase):
    """With H = 0 (every server today, and a Gemma server without the strip history) none of the hold's
    code paths runs: test_gemma_e2e's sessions never hold, never take a cut window, never count a hold
    slot or a hold term, and every release finds nothing (the byte identity with the base commit is the
    identity probe's, outside this suite)."""

    def test_nothing_runs_without_hold_slots(self):
        from remote_ra_native_growing.window_pins import WindowPins
        from remote_ra_live_v2.window_owner import WindowAllocationOwner
        from remote_ra_live_v2 import capacity as CAP

        def never(*args, **kwargs):
            raise AssertionError('a hold path ran without hold slots')
        terms, gates, releases = [], [], []
        turn_hold, holding, release = WC._turn_hold, CAP.holding, WindowPins.release

        def term(*args):
            terms.append(turn_hold(*args))
            return terms[-1]

        def gate(lifecycle):
            gates.append(holding(lifecycle))
            return gates[-1]

        def released(pins, rid):
            releases.append(release(pins, rid))
            return releases[-1]
        with contextlib.ExitStack() as stack:
            for owner, name, value in ((WindowPins, 'hold', never), (WindowPins, 'release', released),
                                       (WindowAllocationOwner, 'cut_window', never), (WC, 'hold_wait', never),
                                       (WC, '_turn_hold', term), (CAP, 'holding', gate)):
                stack.enter_context(patch.object(owner, name, value))
            case = GE.GemmaEndToEndTests('test_sessions_decode_together')
            result = unittest.TestResult()
            case.run(result)
            self.assertEqual((result.testsRun, result.failures, result.errors), (1, [], []))
        self.assertTrue(terms and gates and releases)
        self.assertEqual((set(terms), set(gates), set(releases)), ({0}, {False}, {0}))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GateFigureTests(unittest.IsolatedAsyncioTestCase):
    """The gate's sliding figures with hold slots: a turn's phase-2 need and its ledger base and peak
    carry its hold term, a running request's headroom the rest of its hold term (WindowPins.held_left);
    with hold_slots 0 the figures are today's."""
    send = TC.EndToEndTests.send

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def test_need_ledger_and_headroom(self):
        from remote_ra_live_v2 import capacity as CAP
        srv = strip_window_server(self, 'observation_query', slots=4, limit=2)
        h = srv.engine
        messages = [dict(role='user', content=T.first_prompt('Gate figures.'))]
        status, body = await self.send(srv, messages)
        self.assertEqual(status, 200)
        sid = srv.live.sessions[0].session_id
        nxt = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                          dict(role='user', content=T.observation(0))]
        tokens = list(h.endpoint.transcripts[sid].plan(nxt)['tokens'])      # read only
        X_ = len(tokens) - len(HEADER_IDS)

        def figures(hold):
            h.adapter.window.hold_slots = hold
            try:
                return (CAP.turn_need_by_pool(h.core, sid, tokens, nxt, 1536),
                        WC.gate_ledger(h.core, sid, tokens, nxt, 1536))
            finally:
                h.adapter.window.hold_slots = 2
        (fa1, s1), (fa2, s2) = figures(2)[0]
        (ga1, t1), (ga2, t2) = figures(0)[0]
        term = WC.hold_term(h.layout, X_, len(HEADER_IDS), 1536)
        first, last = kept_blocks(X_, FW, FP)
        self.assertEqual(term, last - first)
        self.assertEqual(((fa1, s1), (fa2, s2)), ((ga1, t1), (ga2, t2 + 5 * term)))
        (base, peak), (base0, peak0) = figures(2)[1], figures(0)[1]
        self.assertEqual((base, peak), (base0 + 5 * term, peak0 + 5 * term))
        # A running request (parked mid-decode past X): its headroom's hold part is its term less the
        # blocks of its hold that have left its row.
        parked = srv.hold(0, 12)
        task = asyncio.create_task(self.send(srv, nxt))
        await asyncio.wait_for(parked.parked.wait(), 10)
        rid = next(r for r, *_ in reversed(h.recorder.holds))
        h.adapter.window.hold_slots = 0
        fa0, room0 = CAP.headroom_by_pool(h.lifecycle)
        h.adapter.window.hold_slots = 2
        fa, room = CAP.headroom_by_pool(h.lifecycle)
        request = h.requests[rid]
        geometry = h.endpoint.claims[rid]['geometry']
        running = WC.hold_term(h.layout, geometry.e, len(request.prompt_token_ids) - geometry.e,
                               CAP.max_tokens(request.sampling_params))
        left = h.adapter.window.held_left(rid)
        self.assertGreater(left, 0)
        self.assertEqual((fa, room), (fa0, room0 + 5 * (running - left)))
        parked.gate.set()
        status, body = await task
        self.assertEqual(status, 200)
        check_window_strip(self, srv, completions=2)
        boundary(self, srv)


@contextlib.contextmanager
def no_hold_term():
    """The review's mutation of 11-10 (review-1110 mut_hold_term.py): the gate without its hold term, i.e.
    window_capacity._turn_hold 0 in a turn's need and ledger, and every running request's hold reported as
    already out of its row in gate_headroom (WindowPins.held_left is read by nothing else in the gate). The
    hold slot count (hold_wait) stays."""
    from remote_ra_native_growing.window_pins import WindowPins
    headroom = WC.gate_headroom

    def without_hold(lifecycle):
        with patch.object(WindowPins, 'held_left', lambda self, rid: 10 ** 6):
            return headroom(lifecycle)
    with patch.object(WC, '_turn_hold', lambda *args, **kwargs: 0), patch.object(WC, 'gate_headroom', without_hold):
        yield


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class TightPoolTests(unittest.IsolatedAsyncioTestCase):
    """The gate's hold term at work (the review of 11-10, finding 5): with at most H turns in flight and pool 1
    sized with H hold slots the pool fits by construction, so the other tests pass without the term. Here pool 1
    is smaller than the launcher sizes it: once the sessions pinned their windows, the test takes blocks out of
    it (ballast, returned at the end). Continuing turns on the blocks one phase 1 needs, and a cold prefill beside
    two parked decoders on their headroom and a few blocks: with the term the gate makes the turns wait
    (``CAPACITY_WAIT: ... sliding pool``) and every turn is served; without it (``no_hold_term``) the gate
    admits a turn whose hold the pool cannot hold and a running request finds no block (Preempted; on the GPU
    a fatal preemption)."""
    maxDiff = None
    send = TC.EndToEndTests.send

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)
        from remote_ra_live_v2 import capacity as CAP
        self.waits = []
        gate = CAP.gate

        def counted(lifecycle, command):
            try:
                return gate(lifecycle, command)
            except CAP.CapacityWait as wait:
                self.waits.append(str(wait))
                raise
        self.enterContext(patch.object(CAP, 'gate', counted))

    async def opened(self, srv, count):
        """``count`` sessions after their first turn: (session id, its next request) each."""
        h, out = srv.engine, []
        for k in range(count):
            messages = [dict(role='user', content=T.first_prompt('Tight pool %d.' % k, host='t%04d' % k))]
            status, body = await self.send(srv, messages)
            self.assertEqual(status, 200, body)
            nxt = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                              dict(role='user', content=T.observation(0, host='t%04d' % k, size=k % 2))]
            sid = next(s for s, t in h.endpoint.transcripts.items() if t.turn > 0 and t.extends(nxt))
            out.append((sid, nxt))
        return out

    def ballast(self, srv, leave):
        """Take blocks out of pool 1 until ``leave`` are free; given back by ``finish``."""
        pool = srv.engine.window_pool
        self.assertGreaterEqual(pool.get_num_free_blocks(), leave)
        return pool.get_new_blocks(pool.get_num_free_blocks() - leave)

    def finish(self, srv, ballast, outcomes, completions):
        """Unmutated: every turn served, the strip checks, the ballast back, the boundary."""
        self.assertEqual([o if isinstance(o, BaseException) else o[0] for o in outcomes], [200] * len(outcomes))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertIsNone(srv.engine.dead)
        check_window_strip(self, srv, completions=completions)
        srv.engine.window_pool.free_blocks(ballast)
        boundary(self, srv)

    async def continuing(self, mutated=False):
        """Four sessions' continuing turns at once on four window slots and four hold slots, with pool 1 left
        with the largest phase-1 need of them and 10 blocks."""
        from remote_ra_live_v2 import capacity as CAP
        srv = strip_window_server(self, 'full', slots=8, limit=4, hold=4)
        h = srv.engine
        turns = await self.opened(srv, 4)
        phase1 = max(CAP.turn_need_by_pool(h.core, sid, list(h.endpoint.transcripts[sid].plan(nxt)['tokens']),
                                           nxt, 1536)[0][1] for sid, nxt in turns)
        ballast = self.ballast(srv, phase1 + 10)
        with no_hold_term() if mutated else contextlib.nullcontext():
            outcomes = await asyncio.gather(*(self.send(srv, nxt) for _, nxt in turns), return_exceptions=True)
        return srv, ballast, outcomes

    async def cold_beside_decoders(self, mutated=False):
        """Two sessions decode their continuing turns, parked three steps into phase 2 with their holds (whose
        blocks are all still in their rows); pool 1 is left with what they may still take (the gate's headroom,
        their hold terms included) and 5 blocks; then a new session's first turn, a cold prefill, arrives."""
        srv = strip_window_server(self, 'full', slots=8, limit=4, hold=4)
        h = srv.engine
        turns = await self.opened(srv, 2)
        parks = [srv.hold(k, 3) for k in range(2)]
        decoders = [asyncio.create_task(self.send(srv, nxt)) for _, nxt in turns]
        for park in parks:
            await asyncio.wait_for(park.parked.wait(), 20)
        self.assertEqual(len(h.adapter.window.holds), 2)
        self.assertEqual({h.adapter.window.held_left(rid) for rid in h.adapter.window.holds}, {0})
        ballast = self.ballast(srv, WC.gate_headroom(h.lifecycle) + 5)
        with no_hold_term() if mutated else contextlib.nullcontext():
            cold = asyncio.create_task(self.send(srv, [dict(role='user', content='Tiny task.\nroot@t:/app# ')]))
            await B2.settle(lambda: self.waits or len(h.adapter.window.holds) > 2 or cold.done() or h.dead, 20)
            admitted = len(h.adapter.window.holds) > 2 or cold.done()
            for park in parks:
                park.gate.set()
            outcomes = await asyncio.gather(*decoders, cold, return_exceptions=True)
        return srv, ballast, outcomes, admitted

    async def test_continuing_turns_in_a_pool_smaller_than_the_launchers(self):
        srv, ballast, outcomes = await self.continuing()
        self.finish(srv, ballast, outcomes, 8)
        # The gate made turns wait for the sliding pool: never four holds at once.
        self.assertTrue(self.waits and all('sliding pool' in w for w in self.waits), self.waits[:2])
        self.assertLess(srv.engine.recorder.most, 4)

    async def test_a_cold_prefill_beside_decoders_in_a_tight_pool(self):
        srv, ballast, outcomes, admitted = await self.cold_beside_decoders()
        self.finish(srv, ballast, outcomes, 5)
        # The cold turn waited while the two decoders held their windows (no third hold), then ran.
        self.assertFalse(admitted)
        self.assertTrue(self.waits and 'sliding pool' in self.waits[0], self.waits[:1])
        self.assertEqual(srv.engine.recorder.most, 2)

    async def test_without_the_hold_term_a_running_request_finds_no_block(self):
        """The review's mutation (``no_hold_term``) in both scenarios: no wait, and the engine fails."""
        for scenario in (self.continuing, self.cold_beside_decoders):
            with self.subTest(scenario=scenario.__name__):
                self.waits.clear()
                srv = (await scenario(mutated=True))[0]
                self.assertIsInstance(srv.engine.dead, B2.Preempted)
                self.assertEqual(self.waits, [])


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class CleanupFailureTests(unittest.IsolatedAsyncioTestCase):
    """A hold whose cleanup failed (the review of 11-10, finding 6). A cancel in phase 2 whose drain raises (the
    worker's cancel_and_drain) leaves the lifecycle entry drain_pending, the session poisoned and the hold in
    place, since after_drain (which aborts the request, and the native free releases the hold) runs only after a
    drain that succeeded; the error goes to the caller of the cancel. With H = 1 every other strip turn waits
    (``CAPACITY_WAIT: window hold slots: 1 of 1 in use``). drain_pending is expirable (lifecycle.EXPIRABLE): once
    the admission's lease has run out, the expiry (EngineLifecycle.expire, the engine loop's lifecycle sweep)
    retries the drain. A retry that fails raises ``lease expiry drain failed`` (out of the engine loop: the
    engine core fails, nothing silent); one that succeeds releases the hold and the waiting turn runs."""
    maxDiff = None
    send = TC.EndToEndTests.send

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    async def test_the_lease_expiry_ends_a_hold_whose_cleanup_failed(self):
        from remote_ra_live_v2 import capacity as CAP
        waits = []
        real_wait = WC.hold_wait

        def counted(lifecycle):
            try:
                return real_wait(lifecycle)
            except CAP.CapacityWait as wait:
                waits.append(str(wait))
                raise
        self.enterContext(patch.object(WC, 'hold_wait', counted))
        # Three open sessions: the poisoned one, the waiting one and the poisoned client's retry.
        srv = strip_window_server(self, 'observation_query', slots=6, limit=3, hold=1)
        h = srv.engine
        messages = [dict(role='user', content=T.first_prompt('Cleanup failure.'))]
        status, body = await self.send(srv, messages)
        self.assertEqual(status, 200)
        nxt = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                          dict(role='user', content=T.observation(0))]
        parked = srv.hold(0, 4)
        task = asyncio.create_task(self.send(srv, nxt))
        await asyncio.wait_for(parked.parked.wait(), 10)
        rid = next(r for r, *_ in reversed(h.recorder.holds))
        self.assertIn(rid, h.adapter.window.holds)
        entry = h.lifecycle.entries[rid]
        _, session, turn, _ = entry['identity']
        # The worker's drain fails (twice: at the cancel and at the first expiry retry).
        failures = [RuntimeError('injected: the expiry retry failed'), RuntimeError('injected: the drain failed')]
        drain = h.boundary.cancel_and_drain

        def flaky(binding, plan):
            if failures:
                raise failures.pop()
            return drain(binding, plan)
        h.boundary.cancel_and_drain = flaky
        with self.assertRaisesRegex(RuntimeError, 'injected: the drain failed'):
            h.lifecycle.cancel(entry['identity'])
        self.assertEqual((entry['state'], entry['cleanup_error']),
                         ('drain_pending', 'RuntimeError: injected: the drain failed'))
        self.assertEqual(h.lifecycle.sessions[session], (turn, 'poisoned'))
        self.assertIn(rid, h.adapter.window.holds)
        # The hold keeps its slot: another session's first turn waits at the gate.
        other = asyncio.create_task(self.send(srv, [dict(role='user', content=T.first_prompt('Waits.', host='w'))]))
        await B2.settle(lambda: waits, 10)
        self.assertIn('window hold slots: 1 of 1 in use', waits[0])
        self.assertFalse(other.done())
        # Before the lease runs out the expiry leaves it alone.
        h.lifecycle.expire()
        self.assertEqual(entry['state'], 'drain_pending')
        self.assertIn(rid, h.adapter.window.holds)
        clock = h.lifecycle.clock
        h.lifecycle.clock = lambda: clock() + h.lifecycle.lease + 1
        # The lease ran out: the expiry retries the drain. A retry that fails raises (the engine loop's sweep lets
        # it out); the hold is still there.
        with self.assertRaisesRegex(BaseExceptionGroup, 'lease expiry drain failed') as caught:
            h.lifecycle.expire()
        self.assertIn('injected: the expiry retry failed', str(caught.exception.exceptions[0]))
        self.assertEqual(entry['state'], 'drain_pending')
        self.assertIn(rid, h.adapter.window.holds)
        self.assertFalse(other.done())
        # The next sweep's retry succeeds: the request is aborted, its hold released, the waiting turn runs.
        h.lifecycle.expire()
        self.assertEqual((entry['state'], entry['cleanup_error']), ('cancelled', None))
        self.assertNotIn(rid, h.adapter.window.holds)
        self.assertIn(rid, [r for r, _ in h.recorder.releases])
        parked.gate.set()
        status, _ = await asyncio.wait_for(other, 60)
        self.assertEqual(status, 200)
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
        self.assertNotEqual(outcome if isinstance(outcome, BaseException) else outcome[0], 200)
        # The poisoned session's client retries on a fresh slot.
        h.lifecycle.clock = clock
        status, _ = await self.send(srv, nxt)
        self.assertEqual((status, srv.rows()[-1]['chat_text']['rebind_reason']), (200, 'failed'))
        check_window_strip(self, srv, completions=3)
        boundary(self, srv)
