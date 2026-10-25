"""Batched Page-vector compaction (H2-H4): decoupled only, legacy float path unchanged.

CPU torch (no CUDA). What is proved here and what is not:
- The torch compactor equals an independent numpy float64 reference bit for bit,
  for any batch shape (all rows at once, one by one, odd chunks). The CUDA run
  of the same torch code is NOT covered: it needs the GPU check (compare the
  GPU vectors with this numpy reference on the same rows).
- Today's path (worker ``normalized`` -> engine ``normalized`` ->
  ``selection_contract.compact``) and the batched path are compared on
  synthetic hidden states: they are not bit-identical (one element of a few
  vectors per thousand differs by 1), which is why legacy keeps its own path.
  Legacy receipts still carry exactly ``normalized(row)`` floats.
- Decoupled qwen/phase-1/publication bodies carry ``compact_int8_v1`` vectors
  equal to the reference compaction of the closing rows.
"""
from collections import namedtuple
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))
from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2.contract import Allocation, NativeIdentity, RenderAssignment, ZERO_DIGEST, closed_pages
from remote_ra_live_v2.query_selection import normalized
try:
    import torch
except ImportError:
    torch = None

CV = namedtuple('CV', 'values dtype scale')


def numpy_reference(rows, hidden_size):
    """Independent float64 compaction: project, L2 normalise, int8 per-vector scale."""
    matrix, _ = selection_contract.projection(hidden_size, selection_contract.default_contract())
    m = matrix.numpy().astype(np.float64)
    x = np.asarray(rows, dtype=np.float64)
    # einsum without BLAS: a different summation order than torch's GEMM, and
    # no second OpenMP runtime next to torch's in one process.
    v = np.einsum('ij,kj->ik', x, m, optimize=False)
    norm = np.sqrt(np.einsum('ij,ij->i', v, v, optimize=False))[:, None]
    v = v / np.maximum(norm, 1e-12)
    scale = np.abs(v).max(axis=1) / 127.0
    q = np.clip(np.round(v / scale[:, None]), -127, 127).astype(np.int8)   # half to even, as torch.round
    return [(tuple(int(a) for a in r), float(np.float32(s))) for r, s in zip(q, scale)]


def synthetic_rows(n, hidden=5120, seed=1):
    g = torch.Generator().manual_seed(seed)
    rows = torch.randn(n, hidden, generator=g) * 0.5
    rows[:, [7, 100, 2000, 4000] if hidden > 4000 else [0]] *= 80     # residual-stream outlier channels
    return rows.to(torch.bfloat16).to(torch.float32)                  # model-dtype values


@unittest.skipUnless(torch is not None, 'CPU torch required')
class CompactorTests(unittest.TestCase):
    def setUp(self):
        from remote_ra_live_v2.page_compaction import PageCompactor
        self.compactor = PageCompactor(5120, 16)

    def test_projection_is_the_upstream_vector_space(self):
        self.assertEqual(self.compactor.projection_sha256, selection_contract.PROJECTION_SHA256[5120])
        self.assertEqual(self.compactor.evidence()['accumulate'], 'float64')

    def test_equals_numpy_reference_for_any_batch_shape(self):
        rows = synthetic_rows(1200)
        reference = numpy_reference(rows.numpy(), 5120)
        batched = self.compactor.compact(rows)
        self.assertEqual([(c.values, c.scale) for c in batched], reference)
        ones = [self.compactor.compact(rows[i:i + 1])[0] for i in range(0, 1200, 7)]
        self.assertEqual(ones, [batched[i] for i in range(0, 1200, 7)])
        chunks = [c for i in range(0, 1200, 97) for c in self.compactor.compact(rows[i:i + 97])]
        self.assertEqual(chunks, list(batched))
        # bf16 input rows (the model dtype) give the same vectors as their fp32 copy.
        self.assertEqual(self.compactor.compact(rows[:50].to(torch.bfloat16)), batched[:50])

    def test_closing_rows_of_one_forward(self):
        from remote_ra_live_v2.page_compaction import PageCompactor
        compactor = PageCompactor(5120, 16)
        hidden = synthetic_rows(100, seed=3)
        start = 37                                   # rows [37, 137): Pages 2..7 close at 47, 63, ...
        pages = closed_pages(start, start + 100, 16)
        self.assertEqual(list(pages), [2, 3, 4, 5, 6, 7])
        expect = compactor.compact(torch.stack([hidden[(p + 1) * 16 - 1 - start] for p in pages]))
        self.assertEqual(compactor.closing(hidden, start, pages), expect)
        self.assertEqual(compactor.closing(hidden, start, ()), ())
        with self.assertRaisesRegex(RuntimeError, 'closing rows outside'):
            compactor.closing(hidden, start, range(2, 9))
        with self.assertRaisesRegex(RuntimeError, 'consecutive'):
            compactor.closing(hidden, start, (2, 4))

    def test_zero_or_wrong_rows_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid zero vector'):
            self.compactor.compact(torch.zeros(2, 5120))
        with self.assertRaisesRegex(RuntimeError, r'\[k, hidden\]'):
            self.compactor.compact(torch.ones(2, 64))

    def test_wire_round_trip_and_refusals(self):
        from remote_ra_live_v2.page_compaction import from_wire, to_wire, CompactPage
        pages = dict(zip((5, 3), self.compactor.compact(synthetic_rows(2))))
        wire = to_wire(pages)
        self.assertEqual(list(wire), ['3', '5'])
        self.assertEqual(from_wire(wire), pages)
        good = wire['3']
        for bad in ({'3': [good[0][:127], good[1]]}, {'3': [good[0][:127] + [128], good[1]]},
                    {'3': [good[0], 0.0]}, {'3': [good[0], 1]}, {'x': good},
                    {'3': [[float(v) for v in good[0]], good[1]]}, {'3': [[0] * 128, good[1]]}):
            with self.assertRaises(RuntimeError):
                from_wire(bad)
        with self.assertRaises(RuntimeError):
            to_wire({1: CompactPage((1,) * 127, 0.1)})


@unittest.skipUnless(torch is not None, 'CPU torch required')
class LegacyVersusBatchedTests(unittest.TestCase):
    """Both paths on the same synthetic hidden states (x86 CPU torch)."""

    def test_today_path_and_batched_path_differ_only_by_rare_unit_steps(self):
        from remote_ra_live_v2.page_compaction import PageCompactor
        rows = synthetic_rows(600, seed=5)
        matrix, _ = selection_contract.projection(5120, selection_contract.default_contract())
        today = [selection_contract.compact(normalized(normalized(r.tolist())), matrix, CV) for r in rows]
        batched = PageCompactor(5120, 784).compact(rows)
        differing = [(a.values, b.values) for a, b in zip(today, batched) if a.values != b.values]
        steps = [abs(x - y) for a, b in differing for x, y in zip(a, b) if x != y]
        self.assertLess(len(differing), len(rows) // 50, len(differing))   # observed ~0.35 %
        self.assertTrue(all(s == 1 for s in steps), steps)
        # Same ranking signal: cosines of every vector against a query agree closely.
        query = batched[0]
        def cos(a, b):
            dot = lambda x, y: sum(p * q for p, q in zip(x, y))
            return dot(a, b) / (dot(a, a) * dot(b, b)) ** 0.5
        worst = max(abs(cos(query.values, t.values) - cos(query.values, d.values))
                    for t, d in zip(today, batched))
        self.assertLess(worst, 0.02)


class SlotIO:
    evidence_kind = 'CUDA_PHYSICAL_TENSOR_OPERATIONS'
    def capture(self, groups, kind): return {'values': [0]}
    def restore(self, groups, kind, payload): pass
    def zero(self, groups, kind): pass
    def drain(self): pass
    def payload_digest(self, payload): return 'f' * 64


class Owner:
    def verify_identity(self): pass
    def verify_allocations(self, identity, slots, generation): pass
    def verify_zero_receipt(self, identity, slots, generation): pass
    def release_after_drain(self, identity, generation, operation): return ()


@unittest.skipUnless(torch is not None, 'CPU torch required')
class QwenPublicationTests(unittest.TestCase):
    P = 16

    def run_request(self, compactor, hidden=5120):
        from remote_ra_live_v2.qwen import QwenAlignLifecycle
        from remote_ra_live_v2.worker_backend import CudaWorkerBackend
        backend = CudaWorkerBackend(SlotIO(), Owner(), source='b' * 64)
        backend.compactor = compactor
        qwen = QwenAlignLifecycle(backend, page_tokens=self.P, mamba_tokens=784)
        identity = NativeIdentity('r0', 's', 0, 'g0', 'a' * 64, 'b' * 64, 'm')
        backend.begin(identity, 1, namespace='ns')
        a = RenderAssignment(identity, local_turn=0, computed_start=0, sealed_start=0, candidate_end=0,
                             observation_start=40, observation_end=40, phase2_start=0, prompt_tokens=50,
                             page_tokens=self.P, token_digest='c' * 64, carry_digest=ZERO_DIGEST)
        state = qwen.open(a, tuple(range(50)), 1)
        rows = synthetic_rows(60, hidden, seed=9)
        slots = (Allocation(1, 10, 1, 'mamba', 1), Allocation(2, 11, 1, 'mamba', 1))
        for count in (37, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1):     # prefill chunks, then decode
            start = state.cursor
            qwen.before_forward(identity, start=start, count=count, logical_positions=range(start, start + count),
                                attention_allocations=(Allocation(0, 1, 1, 'attention', 1),),
                                hybrid_allocations=slots, allocation_generation=1)
            qwen.after_forward(identity, rows[start:start + count])
        return backend, qwen, identity, state, rows

    def test_legacy_publication_is_the_float_path(self):
        backend, qwen, identity, state, rows = self.run_request(None, hidden=8)
        self.assertEqual(set(state.page_vectors), {0, 1, 2})
        for p in (0, 1, 2):
            self.assertEqual(state.page_vectors[p], normalized(rows[(p + 1) * 16 - 1].tolist()))
        body = backend.verify_receipt(identity, 'publish', qwen.publish_pages(identity))
        self.assertEqual(body['representation'], 'pre_final_norm_last_token_l2')
        self.assertEqual(body['page_vectors'], {str(p): list(state.page_vectors[p]) for p in (0, 1, 2)})

    def test_decoupled_publication_is_compact(self):
        from remote_ra_live_v2.page_compaction import PageCompactor, from_wire
        compactor = PageCompactor(5120, 16)
        backend, qwen, identity, state, rows = self.run_request(compactor)
        expect = numpy_reference(torch.stack([rows[(p + 1) * 16 - 1] for p in (0, 1, 2)]).numpy(), 5120)
        self.assertEqual([(state.page_vectors[p].values, state.page_vectors[p].scale) for p in (0, 1, 2)], expect)
        body = backend.verify_receipt(identity, 'publish', qwen.publish_pages(identity))
        self.assertEqual(body['representation'], 'compact_int8_v1')
        self.assertEqual(from_wire(body['page_vectors']), dict(state.page_vectors))


@unittest.skipUnless(torch is not None, 'CPU torch required for the worker harness')
class DecoupledHarnessTests(unittest.TestCase):
    """test_stageb_worker's real worker composition with the compactor attached (hidden size 3)."""

    def test_phase1_and_completion_carry_compact_vectors(self):
        import test_stageb_worker as w
        from remote_ra_live_v2.page_compaction import PageCompactor, from_wire
        from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator
        self.enterContext(patch.dict(sys.modules, w.vllm_stubs()))
        h = w.Harness(); h.open('s')
        compactor = h.backend.compactor = PageCompactor(3, w.P)
        receipts, original = [], Phase1Coordinator._result

        def legacy_view(coordinator, result, binding, geo, session, goal_tokens):
            # The engine side (WP2/WP4) consumes I3; this harness engine is still legacy.
            receipts.append(result)
            pages = from_wire(result['page_vectors'])
            # Keys only: the harness checks Page sets; Phase1Result wants one vector width.
            result = dict(result, representation=result['query_representation'],
                          page_vectors={str(p): list(result['observation_vector']) for p in pages})
            result.pop('query_representation')
            return original(coordinator, result, binding, geo, session, goal_tokens)

        def expected(T, pages):
            P = w.P
            return {p: compactor.compact(torch.tensor([[float((p + 1) * P), T[(p + 1) * P - 1] + 1., 1.]]))[0]
                    for p in pages}

        with patch.object(Phase1Coordinator, '_result', legacy_view):
            info = h.run_turn('s', observation=w.words(5, 1), header=w.words(2, 2), outputs=w.words(3, 3))
            done = info['complete']
            self.assertEqual(from_wire(done['page_vectors']), expected(info['T'] + tuple(w.words(3, 3)),
                                                                     closed_pages(0, 9, w.P)))
            info = h.run_turn('s', boundary=w.words(1, 4), observation=w.words(7, 5), header=w.words(2, 6),
                              outputs=w.words(2, 7), selected=[0])
        a, (receipt,) = info['a'], receipts
        self.assertEqual(receipt['representation'], 'compact_int8_v1')
        self.assertEqual(receipt['observation_vector'],
                         list(normalized([float(a.observation_end), info['T'][a.observation_end - 1] + 1., 1.])))
        self.assertEqual(from_wire(receipt['page_vectors']),
                         expected(info['T'], closed_pages(a.computed_start, a.observation_end, w.P)))
        body = h.backend.verify_receipt(a.identity, 'phase1', receipt['worker_receipt'])
        self.assertEqual(body['page_vectors'], receipt['page_vectors'])

    def test_decoupled_owner_spans_pages_and_passes_the_layout_to_the_lease(self):
        import importlib
        import test_stageb_worker as w
        from remote_ra_native_growing.contract import digest as native_digest
        self.enterContext(patch.dict(sys.modules, w.vllm_stubs()))
        # The module the owner imports now (another test's patch.dict may have
        # dropped an earlier copy from sys.modules).
        model_queries = importlib.import_module('remote_ra_live_v2.model_queries')
        h = w.Harness(); h.open('s')
        # The harness Page is 4 tokens (no KvLayout below 16): mark the owner decoupled directly.
        layout = NS(page_tokens=w.P, mamba_tokens=w.P, decoupled=True)
        h.owner.decoupled, h.owner.layout = True, layout
        h.owner.pool_of = {0: 0, 1: 1, 2: 1, 3: 1}
        seen, parse = [], model_queries.parse_lease

        def parse_with_layout(raw, *, attention_groups=None, layout=None):
            seen.append(layout)                       # WP2's parse_lease takes the layout
            return parse(raw, attention_groups=attention_groups)

        spans, rpc = [], h.rpc

        def engine_view(op, payload):
            receipt = rpc(op, payload)
            if op == 'complete':
                full = tuple(payload_tokens[payload['request']]) + tuple(payload['output_token_ids'])
                pages = {}
                for p, record in receipt['new_pages'].items():
                    self.assertEqual(set(record), {'token_span', 'namespace', 'binding', 'source', 'content'})
                    self.assertEqual(record['token_span'], [int(p) * w.P, (int(p) + 1) * w.P])
                    spans.append(int(p))
                    record = dict(record, token_sha256=native_digest(
                        full[int(p) * w.P:(int(p) + 1) * w.P]))
                    record.pop('token_span')
                    pages[p] = record
                receipt = dict(receipt, new_pages=pages)
            if op == 'prepare':
                payload_tokens[payload['request']] = payload['tokens']
            return receipt

        payload_tokens = {}
        h.rpc = engine_view
        with patch.object(model_queries, 'parse_lease', parse_with_layout):
            h.run_turn('s', observation=w.words(5, 1), header=w.words(2, 2), outputs=w.words(3, 3))
            h.run_turn('s', boundary=w.words(1, 4), observation=w.words(7, 5), header=w.words(2, 6),
                       outputs=w.words(2, 7), selected=[0])
        self.assertEqual(seen, [layout])
        self.assertTrue(spans)


if __name__ == '__main__':
    unittest.main()
