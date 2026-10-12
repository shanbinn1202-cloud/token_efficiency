"""Pinned Page arena and batched FA capture/restore/zero (H7), decoupled only.

CPU only. SlotTable is pure Python. The batched TensorIO paths run on real CPU
tensors (CpuTensorTestIO, unpinned arena) and are compared byte for byte with
the per-block legacy paths: captured Pages, restored device tensors, zeroed
blocks and the IO byte counters. Layouts cover block axis 0 and 1, kernel
ratios 1 and 3, several arena chunks, fragmented slots (host gather/scatter)
and staging slices. The allocation owner's batched wiring runs the whole
Stage B worker sequence suite (test_stageb_worker) with batched IO.

Not covered here (GPU check): pinned host memory, non_blocking D2H/H2D
ordering on a CUDA stream and the single device join per batch.
"""
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))
from remote_ra_growing_worker.tensors import SlotTable
try:
    import torch
except ImportError:
    torch = None


class SlotTableTests(unittest.TestCase):
    def table(self, chunk=4, limit=12):
        grown = []
        return SlotTable(chunk, limit, lambda: grown.append(1)), grown

    def test_lowest_slots_first_chunks_on_demand_and_exhaustion(self):
        t, grown = self.table()
        a = t.allocate(3)
        self.assertEqual([s for s, _ in a], [0, 1, 2]); self.assertEqual(len(grown), 1)
        b = t.allocate(3)
        self.assertEqual([s for s, _ in b], [3, 4, 5]); self.assertEqual(len(grown), 2)
        self.assertEqual(len({tok for _, tok in a + b}), 6)
        self.assertTrue(all(len(tok) == 64 and int(tok, 16) >= 0 for _, tok in a + b))
        with self.assertRaisesRegex(RuntimeError, 'DRAM_ARENA_EXHAUSTED'):
            t.allocate(7)
        t.allocate(6)
        self.assertEqual(len(grown), 3)
        with self.assertRaisesRegex(RuntimeError, 'DRAM_ARENA_EXHAUSTED'):
            t.allocate(1)

    def test_release_reuse_new_token_and_stale_release_ignored(self):
        t, _ = self.table()
        (s0, k0), (s1, k1) = t.allocate(2)
        t.release(s0, k0)
        self.assertFalse(t.holds(s0, k0))
        (s2, k2), = t.allocate(1)
        self.assertEqual(s2, s0); self.assertNotEqual(k2, k0)
        t.release(s0, k0)                      # stale: the slot now belongs to k2
        self.assertTrue(t.holds(s2, k2))
        self.assertEqual(t.allocate(1)[0][0], 2)

    def test_segments(self):
        t, _ = self.table(chunk=4, limit=16)
        t.allocate(12)
        self.assertEqual(t.segments([1, 2, 3, 4, 5, 9]),
                         [(0, 0, 3, range(1, 4)), (1, 3, 2, range(0, 2)), (2, 5, 1, range(1, 2))])
        self.assertEqual(t.segments([0, 2, 3, 8]), [(0, 0, 3, [0, 2, 3]), (2, 3, 1, range(0, 1))])


@unittest.skipUnless(torch is not None, 'CPU torch required')
class BatchedTensorIOTests(unittest.TestCase):
    BLOCKS = 20
    FA, GDN = 0, 1

    def world(self, ratio=1, axis1=False, seed=0, chunk_pages=3, max_pages=40):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO, LayerView, PageArena
        g = torch.Generator().manual_seed(seed)
        n = self.BLOCKS * ratio
        fa0 = torch.randn(n, 2, 4, 3, generator=g)
        fa1 = torch.randn(2, n, 4, 3, generator=g) if axis1 else torch.randn(n, 5, generator=g)
        conv, ssm = torch.randn(self.BLOCKS, 3, generator=g), torch.randn(self.BLOCKS, 2, 2, generator=g)
        views = [LayerView('fa0', self.FA, 'attention', (fa0,), (0,), ratio=ratio),
                 LayerView('fa1', self.FA, 'attention', (fa1,), (1 if axis1 else 0,), ratio=ratio),
                 LayerView('gdn', self.GDN, 'hybrid', (conv, ssm), (0, 0))]
        io = CpuTensorTestIO(views, decoupled=True)
        page_bytes = sum(ratio * (t.numel() // t.shape[a]) * 4 for v in views[:2]
                         for t, a in zip(v.tensors, v.axes))
        io.attach_arena(PageArena(io.views, max_pages=max_pages, pin=False,
                                  chunk_bytes=chunk_pages * page_bytes))
        legacy = CpuTensorTestIO([LayerView(v.name, v.group, v.kind, tuple(t.clone() for t in v.tensors),
                                            v.axes, ratio=v.ratio) for v in views])
        return io, legacy

    @staticmethod
    def tensors(io):
        return [t for v in io.views for t in v.tensors]

    def assert_same_tensors(self, io, legacy):
        for a, b in zip(self.tensors(io), self.tensors(legacy)):
            self.assertTrue(torch.equal(a, b))

    @staticmethod
    def arena_piece(io, payload, view, ti):
        arena, r = io.arena, view.ratio
        chunk, index = divmod(payload.slot, arena.slots.chunk_slots)
        return arena.host(chunk, view.name, ti).narrow(0, index * r, r)

    def check_capture(self, io, legacy, blocks):
        pages = [{self.FA: (b,)} for b in blocks]
        payloads = io.capture_pages(pages)
        old = [legacy.capture(p, 'attention') for p in pages]
        for payload, before in zip(payloads, old):
            for v in io.views[:2]:
                for ti, (t, a) in enumerate(zip(v.tensors, v.axes)):
                    piece = before[v.name][0][ti]
                    piece = piece.movedim(a, 0) if v.ratio > 1 else piece.unsqueeze(0)
                    self.assertTrue(torch.equal(self.arena_piece(io, payload, v, ti), piece))
        self.assertEqual(io._http_d2h_joined, legacy._http_d2h_joined)
        return payloads, old

    def round_trip(self, **kw):
        io, legacy = self.world(**kw)
        rng = random.Random(kw.get('seed', 0))
        blocks = rng.sample(range(1, self.BLOCKS), 9)
        payloads, old = self.check_capture(io, legacy, blocks)
        self.assertEqual(len({io.payload_digest(p) for p in payloads}), 9)
        # Restore into other blocks, in a shuffled order, after zeroing them.
        targets = rng.sample([b for b in range(1, self.BLOCKS) if b not in blocks[:2]], 7)
        chosen = rng.sample(range(9), 7)
        io.zero({self.FA: tuple(targets)}, 'attention'); legacy.zero({self.FA: tuple(targets)}, 'attention')
        self.assert_same_tensors(io, legacy)
        io.restore_pages([({self.FA: (t,)}, payloads[i]) for t, i in zip(targets, chosen)])
        for t, i in zip(targets, chosen):
            legacy.restore({self.FA: (t,)}, 'attention', old[i])
        self.assert_same_tensors(io, legacy)
        self.assertEqual((io._http_h2d_issued, io._http_h2d_joined),
                         (legacy._http_h2d_issued, legacy._http_h2d_joined))
        return io, legacy, payloads

    def test_round_trip_axis0_ratio1(self):
        self.round_trip()

    def test_round_trip_axis1_ratio3(self):
        self.round_trip(ratio=3, axis1=True, seed=4)

    def test_staging_slices(self):
        from remote_ra_growing_worker.tensors import TensorIO
        with patch.object(TensorIO, 'MAX_STAGING_BYTES', 1):     # one Page per slice
            self.round_trip(ratio=2, seed=6)

    def test_fragmented_slots_gather_and_scatter(self):
        io, legacy, payloads = self.round_trip(seed=8, chunk_pages=4)
        # Free every other slot, then capture into the holes: non-consecutive slots.
        kept = payloads[1::2]
        del payloads
        self.assertEqual(io.arena.stats()['pages_held'], len(kept))
        blocks = [2, 5, 7, 11, 13]
        fresh, old = self.check_capture(io, legacy, blocks)
        slots = sorted(p.slot for p in fresh)
        self.assertNotEqual(slots, list(range(slots[0], slots[0] + len(slots))))
        targets = [3, 4, 6, 8, 9]
        io.restore_pages([({self.FA: (t,)}, p) for t, p in zip(targets, reversed(fresh))])
        for t, p in zip(targets, reversed(old)):
            legacy.restore({self.FA: (t,)}, 'attention', p)
        self.assert_same_tensors(io, legacy)

    def test_mixed_payloads_and_refusals(self):
        io, legacy = self.world(seed=2)
        payloads = io.capture_pages([{self.FA: (1,)}, {self.FA: (2,)}])
        single = io.capture({self.FA: (3,)}, 'attention')      # e.g. the phase-1 tail capture
        legacy_payloads = [legacy.capture({self.FA: (b,)}, 'attention') for b in (1, 2, 3)]
        io.restore_pages([({self.FA: (10,)}, payloads[1]), ({self.FA: (11,)}, single), ({self.FA: (12,)}, payloads[0])])
        for t, p in zip((10, 11, 12), (legacy_payloads[1], legacy_payloads[2], legacy_payloads[0])):
            legacy.restore({self.FA: (t,)}, 'attention', p)
        self.assert_same_tensors(io, legacy)
        with self.assertRaisesRegex(RuntimeError, 'invalid physical block IDs'):
            io.restore_pages([({self.FA: (10,)}, payloads[0]), ({self.FA: (10,)}, payloads[1])])
        with self.assertRaisesRegex(RuntimeError, 'one block per FullAttention group'):
            io.capture_pages([{self.FA: (1, 2)}])
        other, _ = self.world(seed=3)
        foreign = other.capture_pages([{self.FA: (1,)}])[0]
        with self.assertRaisesRegex(RuntimeError, 'foreign or released'):
            io.restore_pages([({self.FA: (10,)}, foreign)])
        io.arena.slots.release(payloads[0].slot, payloads[0].token)   # as if it had been dropped
        with self.assertRaisesRegex(RuntimeError, 'foreign or released'):
            io.restore_pages([({self.FA: (10,)}, payloads[0])])
        with self.assertRaisesRegex(RuntimeError, 'no longer holds'):
            io.payload_digest(payloads[0])
        self.assertEqual(io.capture_pages([]), [])

    def test_dropped_payloads_return_their_slots(self):
        io, _ = self.world(seed=5, chunk_pages=2, max_pages=4)
        first = io.capture_pages([{self.FA: (b,)} for b in (1, 2, 3, 4)])
        with self.assertRaisesRegex(RuntimeError, 'DRAM_ARENA_EXHAUSTED'):
            io.capture_pages([{self.FA: (5,)}])
        del first
        again = io.capture_pages([{self.FA: (b,)} for b in (5, 6, 7, 8)])
        self.assertEqual(sorted(p.slot for p in again), [0, 1, 2, 3])
        self.assertEqual(io.arena.stats()['chunks'], 2)

    def test_batched_zero_equals_per_block_zero(self):
        io, legacy = self.world(ratio=3, axis1=True, seed=9)
        io.zero({self.FA: (1, 4, 19)}, 'attention'); legacy.zero({self.FA: (1, 4, 19)}, 'attention')
        io.zero({self.GDN: (2,)}, 'hybrid'); legacy.zero({self.GDN: (2,)}, 'hybrid')
        self.assert_same_tensors(io, legacy)
        self.assertEqual(io._http_zero_joins, legacy._http_zero_joins)

    def test_attach_requires_a_matching_decoupled_io(self):
        from remote_ra_growing_worker.tensors import PageArena
        io, legacy = self.world()
        with self.assertRaisesRegex(RuntimeError, 'Page arena must match'):
            legacy.attach_arena(PageArena(legacy.views, max_pages=4, pin=False))
        with self.assertRaisesRegex(RuntimeError, 'Page arena must match'):
            io.attach_arena(PageArena(io.views, max_pages=4, pin=False))      # already attached
        with self.assertRaisesRegex(RuntimeError, 'Page arena not attached'):
            legacy.capture_pages([{self.FA: (1,)}])


class MambaFirstBatchedTensorIOTests(BatchedTensorIOTests):
    """The same with the GDN group listed first (as the live model may): kind decides, not position."""
    FA, GDN = 3, 0


if torch is not None:
    import test_stageb_worker as w

    class BatchedFakeIO(w.FakeIO):
        """FakeIO with the arena entry points (store semantics), counting batched calls."""
        arena = object()

        def capture_pages(self, pages):
            self.log.append(('capture_pages', 'attention', len(pages)))
            return [self.capture(p, 'attention') for p in pages]

        def restore_pages(self, items):
            items = list(items)
            self.log.append(('restore_pages', 'attention', len(items)))
            for groups, payload in items:
                self.restore(groups, 'attention', payload)

    class BatchedOwnerSequenceTests(w.StageBSequenceTests):
        """The whole Stage B worker sequence suite with the owner's batched Page IO."""

        def setUp(self):
            self.enterContext(patch.dict(sys.modules, w.vllm_stubs()))
            self.enterContext(patch.object(w, 'FakeIO', BatchedFakeIO))
            self.h = w.Harness(); self.h.open('s')
            self.h.owner.batched_pages = True

        def test_batched_calls_replace_per_page_calls(self):
            h = self.h
            h.run_turn('s', observation=w.words(5, 1), header=w.words(2, 2), outputs=w.words(3, 3))
            h.run_turn('s', boundary=w.words(1, 4), observation=w.words(3, 5), header=w.words(2, 6),
                       outputs=w.words(2, 7), selected=[0])
            h.run_turn('s', boundary=w.words(1, 17), observation=w.words(1, 18), header=w.words(1, 19),
                       outputs=w.words(2, 20))
            ops = [e[0] for e in h.io.log]
            self.assertIn('capture_pages', ops)
            self.assertIn('restore_pages', ops)
            # Per-Page attention captures/restores now happen only inside the batched calls
            # (plus the single phase-1 tail capture of native_probe).
            batched = [e for e in h.io.log if e[0] in ('capture_pages', 'restore_pages')]
            self.assertTrue(all(n > 0 for _, _, n in batched))


if __name__ == '__main__':
    unittest.main()
