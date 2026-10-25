"""Integration of the decoupling work packages (exp-decouple-int). CPU only.

The interface items between WP2 (engine), WP3 (worker) and WP4 (selection):

- phase 1 and completion carry compact_int8_v1 Page vectors in the decoupled
  layout (WP3's worker, interface I3) while the observation/Goal query vectors
  stay float rows: ``contract.Phase1Result``, ``Phase1Coordinator._result`` and
  the adapter's completion accept exactly that, and refuse the other
  representation (no silent mixing); legacy keeps its float-only checks.
- the decoupled phase-1 probe keeps only the Page-closing rows (on their
  device, for the compactor) and the query row (host fp32) instead of copying
  every row of the forward to the host; legacy still copies them all.

The (P, M) end-to-end matrix over the real engine and worker is in
test_decouple_e2e.py.
"""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_decouple_wp2 as D2

from remote_ra_live_v2.contract import LiveV2Error, Phase1Result, closed_pages

try:
    import torch
except ImportError:
    torch = None

H = 'ab' * 32
FLOAT = 'pre_final_norm_last_token_l2'
COMPACT = 'compact_int8_v1'


def result(**changes):
    value = dict(binding=H, lease_id='phase1-' + '1' * 32, lease_digest=H, start=16, end=40,
                 page_vectors={1: [0.5, 0.5]}, observation_vector=(1.0, 0.25), goal_vector=None,
                 goal_payload_sha256=None, goal_token_count=None, state_digest=H, tail_capture_digest=None,
                 position_evidence=(), vector_space={}, worker_receipt={})
    value.update(changes)
    return Phase1Result(**value)


class Phase1ResultTests(unittest.TestCase):
    def test_compact_pages_with_float_queries(self):
        wire = D2.compact_vector(1)
        r = result(page_vectors={1: wire, 2: D2.compact_vector(2)}, representation=COMPACT,
                   goal_vector=(0.5, 1.0), goal_payload_sha256=H, goal_token_count=3)
        # Stored as plain (values, scale) tuples: QuerySelectionAuthority.page_vector parses them.
        self.assertEqual(r.page_vectors[1], (tuple(wire[0]), wire[1]))
        self.assertIs(type(r.page_vectors[1]), tuple)
        self.assertEqual((r.observation_vector, r.goal_vector), ((1.0, 0.25), (0.5, 1.0)))
        self.assertEqual(r.representation, COMPACT)
        # No Page closed: queries only.
        self.assertEqual(result(page_vectors={}, representation=COMPACT).page_vectors, {})

    def test_legacy_default_is_float_rows_of_one_width(self):
        r = result()
        self.assertEqual((r.representation, r.page_vectors), (FLOAT, {1: (0.5, 0.5)}))
        with self.assertRaisesRegex(LiveV2Error, 'dimensions differ'):
            result(page_vectors={1: [0.5, 0.5, 0.5]})

    def test_the_other_representation_is_refused(self):
        with self.assertRaisesRegex(LiveV2Error, 'Page vector: non-empty finite float vector'):
            result(page_vectors={1: D2.compact_vector(1)})                      # compact under legacy
        with self.assertRaisesRegex(LiveV2Error, 'invalid compact Page vector'):
            result(page_vectors={1: [0.5, 0.5]}, representation=COMPACT)      # floats under compact
        with self.assertRaisesRegex(LiveV2Error, 'invalid compact Page vector'):
            result(page_vectors={1: [[0] * 128, 0.1]}, representation=COMPACT)  # all-zero vector
        with self.assertRaisesRegex(LiveV2Error, 'invalid compact Page vector'):
            result(page_vectors={1: [[1] * 127, 0.1]}, representation=COMPACT)  # wrong width
        with self.assertRaisesRegex(LiveV2Error, 'vector representation'):
            result(representation='float_rows')
        # Query vectors still agree with each other in compact mode.
        with self.assertRaisesRegex(LiveV2Error, 'dimensions differ'):
            result(page_vectors={1: D2.compact_vector(1)}, representation=COMPACT,
                   goal_vector=(0.5, 1.0, 2.0), goal_payload_sha256=H, goal_token_count=3)


class ReceiptRepresentationTests(unittest.TestCase):
    """The engine's phase-1 and completion checks over the WP2 two-pool harness."""

    def run_case(self, layout, patch_worker=None, turns=3):
        with D2.matrix_case(layout):
            h = D2.Harness(layout, 'full')
            if patch_worker is not None:
                patch_worker(h.worker)
            D2.run_script(h, D2.script_for(layout.page_tokens)[:turns])
            return h

    def test_decoupled_turns_take_compact_page_vectors(self):
        layout = D2.layout_of(16, 784, True)
        h = self.run_case(layout)
        self.assertEqual(len(h.worker.parsed), 2)
        D2.check_boundary(self, h)

    def test_decoupled_engine_refuses_a_float_phase1_receipt(self):
        layout = D2.layout_of(16, 784, True)

        def legacy_phase1(worker):
            original = worker.phase1

            def phase1(p):
                receipt = original(p)
                return dict(receipt, page_vectors={k: [1.0, 0.5] for k in receipt['page_vectors']},
                            representation=FLOAT)
            worker.phase1 = phase1
        with self.assertRaisesRegex(LiveV2Error, 'foreign or partial phase-1 receipt'):
            self.run_case(layout, legacy_phase1)

    def test_decoupled_engine_refuses_compact_pages_without_float_queries(self):
        layout = D2.layout_of(16, 784, True)

        def no_query_representation(worker):
            original = worker.phase1

            def phase1(p):
                receipt = original(p)
                receipt.pop('query_representation')
                return receipt
            worker.phase1 = phase1
        with self.assertRaisesRegex(LiveV2Error, 'foreign or partial phase-1 receipt'):
            self.run_case(layout, no_query_representation)

    def test_legacy_engine_refuses_a_compact_phase1_receipt(self):
        layout = D2.layout_of(784, 784, False)

        def compact_phase1(worker):
            original = worker.phase1

            def phase1(p):
                receipt = original(p)
                return dict(receipt, page_vectors={k: D2.compact_vector(int(k)) for k in receipt['page_vectors']},
                            representation=COMPACT, query_representation=FLOAT)
            worker.phase1 = phase1
        with self.assertRaisesRegex(LiveV2Error, 'foreign or partial phase-1 receipt'):
            self.run_case(layout, compact_phase1)

    def test_decoupled_completion_refuses_float_page_vectors(self):
        layout = D2.layout_of(16, 784, True)

        def float_completion(worker):
            original = worker.complete

            def complete(p):
                receipt = original(p)
                return dict(receipt, page_vectors={k: [1.0, 0.5] for k in receipt['page_vectors']})
            worker.complete = complete
        with self.assertRaisesRegex(LiveV2Error, 'invalid compact Page vector'):
            self.run_case(layout, float_completion, turns=1)



@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ProbeRowsTests(unittest.TestCase):
    def test_closing_rows_last_row_and_finiteness(self):
        from remote_ra_live_v2.native_probe import ProbeRows
        P, start, count = 16, 37, 100                    # rows [37, 137): Pages 2..7 close
        hidden = torch.arange(120 * 5, dtype=torch.float32).reshape(120, 5).to(torch.bfloat16)
        rows = ProbeRows(hidden, start, count, P, True)
        pages = tuple(closed_pages(start, start + count, P))
        self.assertEqual(rows.pages, pages)
        self.assertEqual((rows.start, rows.count, rows.hidden_size, rows.finite), (start, count, 5, True))
        self.assertTrue(torch.equal(rows.closing, torch.stack([hidden[(p + 1) * P - 1 - start] for p in pages])))
        self.assertEqual(rows.closing.dtype, torch.bfloat16)          # device dtype, compacted there
        self.assertTrue(torch.equal(rows.last, hidden[count - 1].float()))
        self.assertEqual((rows.last.dtype, str(rows.last.device)), (torch.float32, 'cpu'))
        # A copy: a later forward reusing the buffer does not change them.
        hidden.zero_()
        self.assertTrue(bool(rows.closing.abs().sum() > 0) and bool(rows.last.abs().sum() > 0))
        # A non-finite row anywhere in the forward (not only a kept one) is seen.
        bad = torch.ones(120, 5)
        bad[40, 2] = float('nan')
        self.assertFalse(ProbeRows(bad, start, count, P, True).finite)
        # The Goal forward keeps its last row only; a forward closing no Page has no rows.
        goal = ProbeRows(torch.ones(9, 5), 0, 9, P, False)
        self.assertEqual((goal.pages, goal.closing), ((), None))
        self.assertEqual(ProbeRows(torch.ones(9, 5), 20, 9, P, True).closing, None)
        # Host bytes for a 6.5K-token observation at 5,120: one fp32 row, not 133 MB.
        self.assertEqual(6500 * 5120 * 4 // 10**6, 133)

    def phase1_outputs(self, compact):
        import test_stageb_worker as w
        from remote_ra_live_v2.page_compaction import PageCompactor, from_wire
        from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator
        self.enterContext(patch.dict(sys.modules, w.vllm_stubs()))
        h = w.Harness(); h.open('s')
        if compact:
            h.backend.compactor = PageCompactor(3, w.P)
        driver, outputs, receipts = h.runner._remote_ra_native_query_forward, [], []
        phase1 = driver.phase1

        def recorded(payload):
            outputs.append(phase1(payload))
            return outputs[-1]
        driver.phase1 = recorded
        original = Phase1Coordinator._result

        def engine_view(coordinator, result, binding, geo, session, goal_tokens):
            # This harness engine is the legacy one: hand it float rows of one width.
            receipts.append(result)
            if compact:
                result = dict(result, representation=result['query_representation'],
                              page_vectors={p: list(result['observation_vector'])
                                            for p in from_wire(result['page_vectors'])})
                result.pop('query_representation')
            return original(coordinator, result, binding, geo, session, goal_tokens)
        with patch.object(Phase1Coordinator, '_result', engine_view):
            h.run_turn('s', observation=w.words(5, 1), header=w.words(2, 2), outputs=w.words(3, 3))
            info = h.run_turn('s', boundary=w.words(1, 4), observation=w.words(11, 5), header=w.words(2, 6),
                              outputs=w.words(2, 7), selected=[0], goal=w.words(6, 8))
        return h, info, outputs, receipts

    def test_decoupled_probe_keeps_closing_rows_and_the_query_rows(self):
        import test_stageb_worker as w
        from remote_ra_live_v2.native_probe import ProbeRows
        from remote_ra_live_v2.page_compaction import from_wire
        h, info, (output,), (receipt,) = self.phase1_outputs(True)
        a = info['a']
        C, e = a.computed_start, a.observation_end
        rows, goal = output['rows'], output['goal_rows']
        pages = tuple(closed_pages(C, e, w.P))
        self.assertEqual(type(rows), ProbeRows)
        self.assertEqual((rows.start, rows.count, rows.pages, rows.hidden_size), (C, e - C, pages, 3))
        T = info['T']
        expect = torch.tensor([[float((p + 1) * w.P), T[(p + 1) * w.P - 1] + 1., 1.] for p in pages])
        self.assertTrue(torch.equal(rows.closing, expect))
        self.assertTrue(torch.equal(rows.last, torch.tensor([float(e), T[e - 1] + 1., 1.])))
        self.assertEqual(from_wire(receipt['page_vectors']),
                         dict(zip(pages, h.backend.compactor.compact(expect))))
        self.assertEqual((type(goal), goal.count, goal.closing), (ProbeRows, 6, None))
        self.assertEqual(receipt['goal_token_count'], 6)
        self.assertEqual(len(receipt['goal_vector']), 3)

    def test_legacy_probe_still_copies_every_row(self):
        h, info, (output,), (receipt,) = self.phase1_outputs(False)
        a = info['a']
        rows, goal = output['rows'], output['goal_rows']
        self.assertIsInstance(rows, torch.Tensor)
        self.assertEqual(tuple(rows.shape), (a.observation_end - a.computed_start, 3))
        self.assertEqual(tuple(goal.shape), (6, 3))
        self.assertEqual(receipt['representation'], FLOAT)
        self.assertNotIn('query_representation', receipt)


if __name__ == '__main__':
    unittest.main()
