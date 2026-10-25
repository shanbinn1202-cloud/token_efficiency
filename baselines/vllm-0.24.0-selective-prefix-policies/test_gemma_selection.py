"""Gemma 4 stage 1 (gemma-s1 WP3/WP7): selection on the global layers only, CPU only.

* The selection query: the pre-final-norm residual stream at the last decoder layer
  (59, a global layer), projected by the Selector V2 matrix of hidden size 5,376; its
  pin ``852b42e4...`` equals what the runtime's two builders compute (D-9).
* Selector V2 on a Gemma-shaped session (P = 16, budget 8,192, the stage-0 prompt of
  296 tokens): mandatory Pages cover the system + task prompt, at most 64 recent Pages,
  capacity 512 Pages; the vectorized and the reference planner agree on 5,376-wide
  vectors. The Pages are global-layer Pages; nothing in the plan depends on the window.
* The sliding window (``sliding_window``): the tokens the sliding layers read at the
  first phase-2 step, their overlap with candidate, selected, mandatory and recent
  Pages (the runtime row's evidence), and the window blocks' lifecycle arithmetic
  against vLLM's own SlidingWindowManager rule.
"""
import random
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_selection_fast as SF
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import selection_contract as sc
from remote_ra_live_v2 import sliding_window as SW
from remote_ra_live_v2.contract import LiveV2Error

try:
    import torch
except ImportError:      # the tokenizer interpreter has no torch
    torch = None

W, P = MP.GEMMA4_GEOMETRY.sliding_window, 16
LAYOUT = I0.for_window_server(16, 1, W, 8192)
SPACE = dict(hidden_size=MP.GEMMA4_GEOMETRY.hidden_size, decoder_layers=MP.GEMMA4_GEOMETRY.layers)
INITIAL = 296            # stage-0 system + task prompt (OpenAI parts render)


@unittest.skipIf(torch is None, 'torch required')
class QueryVectorTests(unittest.TestCase):
    def test_projection_pin_of_hidden_5376(self):
        contract = sc.default_contract(16)
        matrix, value = sc.projection(5376, contract)
        self.assertEqual(value, sc.PROJECTION_SHA256[5376])
        self.assertEqual(value, '852b42e43dc0beb2d2ea90deefc549a36acc0e08c088624bc0a4588658a1dfb7')
        self.assertEqual(tuple(matrix.shape), (128, 5376))
        # live_factory's builder (the producer provenance) gives the same hash.
        from policy_runtime.goal_runtime import build_xuanxue_projection
        self.assertEqual(build_xuanxue_projection(torch, 5376, 128, sc.PROJECTION_SEED)[1], value)
        # The Qwen pin is unchanged.
        self.assertEqual(sc.projection(5120, contract)[1],
                         '20c2e1cf4463ed9407a9f9af0a7794fe044c02c0da369128681270a7427c87aa')
        tampered = {**sc.PROJECTION_SHA256, 5376: '0' * 64}
        with patch.object(sc, 'PROJECTION_SHA256', tampered), \
                self.assertRaisesRegex(LiveV2Error, 'projection differs'):
            sc.projection(5376, contract)

    def test_vector_space_of_a_gemma_shaped_model(self):
        from remote_ra_live_v2 import vector_space

        class RMSNorm(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)

        class GemmaRMSNorm(RMSNorm):
            pass

        def model(norm):
            text = torch.nn.Module()
            text.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(60))
            text.embed_tokens = torch.nn.Identity()
            text.norm = norm
            outer = torch.nn.Module()
            outer.language_model = torch.nn.Module()
            outer.language_model.model = text
            return outer
        weight = torch.rand(5376) + 0.5
        info, gain = vector_space.resolve(model(RMSNorm(weight)))
        self.assertEqual((info['hidden_size'], info['decoder_layers']), (5376, 60))
        self.assertEqual(info['decoder_layers'] - 1, MP.GEMMA4_GEOMETRY.query_layer)
        self.assertIn(MP.GEMMA4_GEOMETRY.query_layer, MP.GEMMA4_GEOMETRY.global_layers)
        self.assertEqual(info['module'], 'language_model.model.norm')
        self.assertTrue(torch.equal(gain, weight))
        # vLLM's GemmaRMSNorm scales by (1 + w): either class inverts exactly.
        _, gain = vector_space.resolve(model(GemmaRMSNorm(weight)))
        self.assertTrue(torch.equal(gain, weight + 1.0))
        with self.assertRaisesRegex(LiveV2Error, 'cannot be inverted'):
            vector_space.resolve(model(RMSNorm(torch.zeros(5376))))
        # The recovered direction is the residual's.
        runner = NS(model=model(RMSNorm(weight)))
        vector_space.attach(runner)
        residual = torch.randn(3, 5376)
        normed = residual * torch.rsqrt(residual.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
        recovered = vector_space.pre_final_norm(runner, normed)
        cosine = torch.nn.functional.cosine_similarity(recovered, residual, dim=-1)
        self.assertTrue(bool((cosine > 0.99999).all()))

    def test_page_compactor_at_5376(self):
        from remote_ra_live_v2.page_compaction import PageCompactor
        compactor = PageCompactor(5376, 16)
        self.assertEqual(compactor.projection_sha256, sc.PROJECTION_SHA256[5376])
        hidden = torch.randn(48, 5376)
        pages = compactor.closing(hidden, 0, [0, 1, 2])
        self.assertEqual(len(pages), 3)
        matrix, _ = sc.projection(5376, sc.default_contract(16))
        for page, compact in zip((0, 1, 2), pages):
            row = hidden[(page + 1) * 16 - 1]
            values, scale = compact
            self.assertEqual(len(values), 128)
            reference = torch.nn.functional.normalize(torch.nn.functional.linear(row, matrix), dim=-1)
            mine = torch.tensor(values, dtype=torch.float32) * scale
            self.assertGreater(float(torch.nn.functional.cosine_similarity(mine, reference, dim=0)), 0.999)


def gemma_contract(policy='observation_query'):
    kw = dict(vector_format=sc.COMPACT_INT8_V1, active_ratio=None, active_token_budget=8192)
    if policy == 'random_fill':
        kw['random_fill'] = sc.random_fill_pin(42)
    return sc.default_contract(16, **kw)


@unittest.skipIf(SF.NP is None or torch is None, 'numpy, torch and the fast_selection self-check required')
class GlobalPageSelectionTests(unittest.TestCase):
    """Selector V2 on a Gemma-shaped session: 10 turns of 300- to 4,300-token observations."""

    def session(self, seed, policy='observation_query'):
        rng = random.Random(seed)
        gen = torch.Generator().manual_seed(seed)
        contract = gemma_contract(policy)
        fast = SF.authority(policy, contract, INITIAL, True)
        ref = SF.authority(policy, contract, INITIAL, False)
        matrix, _ = sc.projection(5376, contract)
        compact = lambda row: sc.compact(row.tolist(), matrix, fast.compact_type)
        o, ends, catalog, plans = INITIAL + 7, [], {}, []
        for turn in range(10):
            o += rng.randint(40, 120) + 2 + rng.choice((300, 900, 2600, 4300))
            n = o // P
            rows = torch.randn(n - len(catalog), 5376, generator=gen)
            for k, page in enumerate(range(len(catalog), n)):
                catalog[page] = compact(rows[k])
            query = compact(torch.randn(5376, generator=gen))
            if not sc.selection_active(n, len(ends), contract):
                query = (1.0,)          # plan_turn's placeholder below the switch threshold
            binding = NS(session='s', turn=turn + 1, model='m', renderer='r', stable_end=o)
            for auth in (fast, ref):
                auth.turn_ends['s'] = list(ends)
            got, want = SF.plan_both(None, (fast, ref), binding, n, catalog, query, len(ends), SPACE)
            self.assertEqual(SF.as_json(got), SF.as_json(want))
            plans.append((n, o, want))
            ranking, count, _ = want
            for auth in (fast, ref):
                auth.previous_active['s'] = (frozenset(ranking[:count]), n)
            ends.append(o)
        return plans

    def test_budget_mandatory_recent_on_global_pages(self):
        for policy in ('observation_query', 'random_fill'):
            for n, o, (ranking, count, evidence) in self.session(7, policy):
                if evidence['reason'] is not None:
                    self.assertLessEqual(n * P, 8192)
                    continue
                self.assertEqual(evidence['capacity_pages'], 8192 // P)
                self.assertEqual(count, 512)
                # Every Page touched by the 296-token system + task prompt: 19 Pages.
                self.assertEqual(evidence['mandatory_pages'], list(range(19)))
                self.assertLessEqual(len(evidence['recent_pages']), 64)
                self.assertEqual(sorted(ranking[:count]), evidence['selected_pages'])
                if policy == 'observation_query':
                    self.assertEqual(evidence['projection_sha256'], sc.PROJECTION_SHA256[5376])

    def test_window_evidence_next_to_the_selection(self):
        for n, o, (ranking, count, evidence) in self.session(11):
            if evidence['reason'] is not None:
                continue
            L = n * P
            for observation in (300, 4300):
                C, e = o, o + observation
                window = SW.window_evidence(L=L, C=C, e=e, window=W, page_tokens=P,
                                            selected=ranking[:count], mandatory=evidence['mandatory_pages'],
                                            recent=evidence['recent_pages'])
                self.assertEqual(window['span'], [e - W + 1, e + 1])
                if observation >= W - 1:
                    # The window lies inside the phase-1 run: no history, nothing unselected.
                    self.assertTrue(window['inside_run'])
                    self.assertEqual((window['history_tokens'], window['history_pages']), (0, []))
                else:
                    self.assertEqual(window['history_tokens'], L - (e - W + 1))
                    pages = window['history_pages']
                    self.assertEqual(pages, list(range((e - W + 1) // P, n)))
                    # The newest history Pages are recent (the two previous turns), so the
                    # sliding layers overlap the global layers' recent quota there.
                    self.assertTrue(set(window['recent_pages']) <= set(evidence['recent_pages']))
                    self.assertEqual(set(window['unselected_pages']),
                                     set(pages) - set(ranking[:count]))


class SlidingWindowArithmeticTests(unittest.TestCase):
    def vllm_live_blocks(self, computed, block=P):
        """vLLM's SlidingWindowManager: blocks wholly before the skipped tokens are null."""
        skipped = max(0, computed - W + 1)
        return list(range(skipped // block, -(-computed // block)))

    def test_kept_blocks_match_vllm_and_are_bounded(self):
        for computed in list(range(0, 3000, 7)) + [1023, 1024, 1025, 1039, 1040, 131071, 131072]:
            first, end = SW.kept_blocks(computed, W, P)
            self.assertEqual(list(range(first, end)), self.vllm_live_blocks(computed))
            self.assertLessEqual(end - first, SW.max_kept_blocks(W, P))
        self.assertEqual(SW.max_kept_blocks(W, 16), 65)
        self.assertEqual(SW.max_kept_blocks(W, 32), 33)
        self.assertEqual(max(e - f for f, e in (SW.kept_blocks(c, W, 16) for c in range(5000))), 65)

    def test_window_span(self):
        self.assertEqual(SW.window_span(5000, W), (3977, 5001))
        self.assertEqual(SW.window_span(10, W), (0, 11))
        self.assertEqual(SW.window_span(1023, W), (0, 1024))
        self.assertEqual(SW.window_span(1024, W), (1, 1025))

    def test_phase1_rows_and_drain(self):
        rng = random.Random(5)
        for _ in range(2000):
            C = rng.randint(1, 200000)
            e = C + rng.randint(1, 8192)
            head, (b0, b1), (f0, f1) = SW.phase1_row(C, e, W, P)
            # vLLM's row after computing C, allocated up to e: a null head of the skipped
            # blocks, the borrowed window, then the fresh blocks.
            self.assertEqual(head, max(0, C - W + 1) // P)
            self.assertEqual(list(range(b0, b1)), self.vllm_live_blocks(C))
            self.assertEqual((f0, f1), (b1, -(-e // P)))
            # One forward of N <= 8,192 tokens holds at most the slot plus the transient.
            self.assertLessEqual(f1 - head, LAYOUT.window_blocks + LAYOUT.transient_blocks)
            adopted, unpinned, freed = SW.phase1_drain(C, e, W, P)
            self.assertEqual(adopted, self.vllm_live_blocks(e))
            # Every block of the row is adopted, unpinned (old window) or freed (fresh).
            self.assertEqual(sorted(adopted + unpinned + freed), list(range(head, f1)))
            self.assertTrue(all(i < b1 for i in unpinned) and all(i >= f0 for i in freed))
            zeros, (a0, a1) = SW.phase2_head(e, W, P)
            self.assertEqual((zeros, list(range(a0, a1))), (a0, adopted))

    def test_short_run_keeps_part_of_the_old_window(self):
        # A 300-token run: the window after e still reaches into the pinned old window.
        self.assertEqual(SW.kept_blocks(50000, W, P), (3061, 3125))
        self.assertEqual(SW.kept_blocks(50300, W, P), (3079, 3144))
        adopted, unpinned, freed = SW.phase1_drain(50000, 50300, W, P)
        self.assertEqual((adopted, unpinned, freed), (list(range(3079, 3144)), list(range(3061, 3079)), []))
        # A 4,300-token run: the old window is unpinned entirely, fresh blocks before
        # the new window are freed.
        adopted, unpinned, freed = SW.phase1_drain(50000, 54300, W, P)
        self.assertEqual(unpinned, list(range(*SW.kept_blocks(50000, W, P))))
        self.assertEqual(adopted, list(range(*SW.kept_blocks(54300, W, P))))
        self.assertEqual(freed, list(range(SW.kept_blocks(50000, W, P)[1], SW.kept_blocks(54300, W, P)[0])))

    def test_refusals(self):
        for call in (lambda: SW.window_span(-1, W), lambda: SW.kept_blocks(10, W, 0),
                     lambda: SW.phase1_row(10, 10, W, P), lambda: SW.skipped_tokens(5, 0),
                     lambda: SW.window_evidence(L=17, C=20, e=30, window=W, page_tokens=P, selected=()),
                     lambda: SW.window_evidence(L=32, C=20, e=30, window=W, page_tokens=P, selected=())):
            with self.assertRaises(LiveV2Error):
                call()


if __name__ == '__main__':
    unittest.main()
