"""Gemma 4 stage 1 (gemma-s1 WP3): engine ownership of the sliding windows, CPU only.

``WindowPins`` over the overlay vLLM BlockPool and SlidingWindowManager (executed from
source with stub imports, test_kv_pools' world): a cold turn pins the window after its
computed end before vLLM frees the request; phase 1's window handover frees the pinned
blocks that left the window and moves the rest (with phase 1's fresh blocks) to the
phase-2 request; the head is installed before vLLM's first allocation, vLLM appends and
frees as the window slides, and the commit pins the next window. Reference counts and
both pools return to their baseline when the session closes. Also: the adapter's
window-layout check and the engine boundary's group kinds.
"""
import unittest
from types import SimpleNamespace as NS

import test_gemma_kv_pools as GK
import test_kv_pools as KP
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.sliding_window import kept_blocks, phase1_row
from remote_ra_native_growing.contract import Abort
from remote_ra_native_growing.window_pins import WindowPins

P, W = 16, 1024
SLIDING = (1, 2, 3, 4, 5)


class World:
    """One overlay KV manager over a small Gemma-shaped config: 1 FA group, 5 sliding."""

    def __init__(self, n_full=2000, n_swa=4000):
        self.w = GK.gemma_world()
        self.manager = KP.kv_manager(self.w, GK.small_gemma_config(self.w, P, n_full, n_swa),
                                     hash_block_size=P, scheduler_block_size=P)
        self.pool = self.manager.block_pools[1]
        managers = self.manager.coordinator.single_type_managers
        self.sliding = tuple(managers[g] for g in SLIDING)
        self.pins = WindowPins(self.pool, self.sliding, SLIDING, window_tokens=W, page_tokens=P)
        self.baseline = self.pool.get_num_free_blocks()

    def allocate(self, rid, tokens, computed=0):
        req = KP.request(rid, tokens, computed)
        return req, self.manager.allocate_slots(req, tokens - computed, skip_cache=True, has_scheduled_reqs=True)

    def step(self, req, count):
        """vLLM's own allocation of ``count`` more tokens (it frees what left the window)."""
        req.num_tokens = req.num_computed_tokens + count
        assert KP.alloc(self.manager, req, count) is not None
        return req

    def live(self, rid):
        return [[(i, b) for i, b in enumerate(m.req_to_blocks.get(rid, ())) if not b.is_null]
                for m in self.sliding]

    def free(self, req):
        self.manager.free(req)


class WindowPinsTests(unittest.TestCase):
    def cold_turn(self, w, tokens=3000):
        req, _ = w.allocate('r0', tokens)
        req.num_computed_tokens = tokens
        window = w.pins.pin_after('ns', 'r0', tokens)
        first, last = kept_blocks(tokens, W, P)
        self.assertEqual({g: [i for i, _ in v] for g, v in window.items()},
                         {g: list(range(first, last)) for g in SLIDING})
        w.free(req)
        w.pins.after_free('r0', 'ns')
        return first, last

    def test_cold_turn_pins_the_window_after_c_prime(self):
        w = World()
        first, last = self.cold_turn(w)
        self.assertEqual((first, last), (123, 188))          # [3000 - 1023, 3000): 65 blocks
        self.assertEqual(w.pins.pinned_blocks(), 5 * 65)
        self.assertEqual(w.pool.get_num_free_blocks(), w.baseline - 5 * 65)
        self.assertTrue(all(b.ref_cnt == 1 for v in w.pins.window('ns').values() for _, b in v))
        self.assertEqual(w.pins.close('ns'), 5 * 65)
        self.assertEqual(w.pool.get_num_free_blocks(), w.baseline)

    def test_continuing_turn_hands_over_installs_and_pins_again(self):
        w = World()
        self.cold_turn(w)                                    # C = 3000
        C, e = 3000, 3300
        head, (b0, b1), (f0, f1) = phase1_row(C, e, W, P)
        pinned = w.pins.window('ns')
        self.assertTrue(all([i for i, _ in pinned[g]] == list(range(b0, b1)) for g in SLIDING))
        # Phase 1 allocates its fresh blocks from pool 1; the drain frees the fresh ones
        # before the window after e and the rest wait for adoption.
        fresh = {g: w.pool.get_new_blocks(f1 - f0) for g in SLIDING}
        k0, k1 = kept_blocks(e, W, P)
        leave = [b for g in SLIDING for i, b in zip(range(f0, f1), fresh[g]) if i < k0]
        w.pool.free_blocks(leave)
        stay = {g: tuple((i, b) for i, b in zip(range(f0, f1), fresh[g]) if i >= k0) for g in SLIDING}
        before = w.pool.get_num_free_blocks()
        head = w.pins.handover('ns', 'r1', k0, stay)
        old_left = sum(1 for g in SLIDING for i, _ in pinned[g] if i < k0)
        self.assertEqual(w.pool.get_num_free_blocks(), before + old_left)
        self.assertIsNone(w.pins.window('ns'))
        self.assertEqual({g: (first, len(blocks)) for g, (first, blocks) in head.items()},
                         {g: (k0, k1 - k0) for g in SLIDING})
        w.pins.install('r1')
        for m in w.sliding:
            row = m.req_to_blocks['r1']
            self.assertEqual(len(row), k1)
            self.assertTrue(all(b.is_null for b in row[:k0]))
        # Phase 2 at e: vLLM appends after the head and frees what leaves the window.
        req = KP.request('r1', e + 7, e)
        self.assertIsNotNone(KP.alloc(w.manager, req, 7))
        for _ in range(40):
            w.step(req, 1)
        C2 = req.num_computed_tokens
        live = w.live('r1')
        self.assertTrue(all(len(v) <= 65 for v in live))
        window = w.pins.pin_after('ns', 'r1', C2)
        first, last = kept_blocks(C2, W, P)
        self.assertTrue(all([i for i, _ in v] == list(range(first, last)) for v in window.values()))
        w.free(req)
        w.pins.after_free('r1', 'ns')
        self.assertEqual(w.pool.get_num_free_blocks(), w.baseline - 5 * (last - first))
        w.pins.close('ns')
        self.assertEqual(w.pool.get_num_free_blocks(), w.baseline)
        self.assertEqual(w.manager.get_num_free_blocks_by_pool()[1], w.baseline)

    def test_discard_frees_pending_and_installed_heads(self):
        for installed in (False, True):
            with self.subTest(installed=installed):
                w = World()
                self.cold_turn(w)
                k0, _ = kept_blocks(3300, W, P)
                w.pins.handover('ns', 'r1', k0, {})
                if installed:
                    w.pins.install('r1')
                w.pins.discard('r1')
                self.assertEqual(w.pool.get_num_free_blocks(), w.baseline)
                self.assertFalse(any(m.req_to_blocks.get('r1') for m in w.sliding))
                w.pins.close('ns')                          # nothing pinned any more

    def test_refusals(self):
        w = World()
        with self.assertRaises(Abort):
            w.pins.handover('ns', 'r1', 0, {})              # no pinned window
        self.cold_turn(w)
        with self.assertRaises(Abort):
            w.pins.pin_after('ns', 'r0', 3000)              # already pinned
        req, _ = w.allocate('r5', 100)
        with self.assertRaises(Abort):
            w.pins.pin_after('other', 'r5', 3000)            # window not in the row
        w.free(req)
        k0, _ = kept_blocks(3300, W, P)
        w.pins.handover('ns', 'r1', k0, {})
        with self.assertRaises(Abort):
            w.pins.close('ns')                              # its request still holds it
        with self.assertRaises(Abort):
            WindowPins(w.pool, (KP.world().single.FullAttentionManager,), (1,), window_tokens=W, page_tokens=P)


class Proxy:
    def __init__(self):
        self.calls = []

    def call(self, operation, payload, **kw):
        self.calls.append(operation)
        assert operation == 'phase1_drain'
        return dict(operation='phase1_drain', drained=True, binding=payload['binding'],
                    lease_id=payload['lease_id'], lease_digest=payload['lease_digest'], worker_receipt={})


class WindowPhase1Tests(unittest.TestCase):
    """WindowPhase1Coordinator's lease, drain, adoption and abort on the overlay pools."""

    def fixture(self):
        from remote_ra_live_v2.window_phase1 import WindowPhase1Coordinator
        from remote_ra_native_growing.contract import TurnGeometry
        from remote_ra_native_growing.ownership import PrivateBlockOwner
        from remote_ra_live_v2.pool_incarnations import install
        w = World()
        pool0 = w.manager.block_pools[0]
        install(pool0), install(w.pool)                  # the engine boundary's startup
        req, _ = w.allocate('r0', 3008)
        w.pins.pin_after('ns', 'r0', 3008)               # the cold turn's window
        w.free(req)
        layout = I0.for_window_server(P, 1, W, 8192)
        adapter = NS(authority=NS(owner=lambda: None, sign=lambda value: 'f' * 64, source='c' * 64),
                     page_tokens=P, requests={}, owner=PrivateBlockOwner(pool0), kv_manager=w.manager,
                     managers=(w.manager.coordinator.single_type_managers[0],), layout=layout,
                     mamba_pool=w.pool, window=w.pins, decoupled=True,
                     sessions={'s': NS(poisoned=False, pending=None, local_turn=1, turn=1, boundary='3' * 64,
                                       namespace='ns', resident=(), policy='full')})
        core = NS(_remote_ra_growing_adapter=adapter,
                  scheduler=NS(connector=NS(has_pending_push_work=lambda: False), max_num_scheduled_tokens=8192))
        coordinator = WindowPhase1Coordinator(core, Proxy(), None)
        geo = TurnGeometry(1, 3008, 3008, 3010, 3300, 3310, 188, 3008, False, False, False, 3300, P)
        binding = NS(request='r1', session='s', turn=1, identity='1' * 64, epoch='e', generation='g',
                     model='m', tokens=tuple(range(3310)))
        return w, coordinator, geo, binding

    def test_lease_rows_drain_and_adoption(self):
        from remote_ra_live_v2.contract import parse_lease
        w, coordinator, geo, binding = self.fixture()
        pinned = w.pins.window('ns')
        request, _, _, _ = coordinator.run(binding, geo, allocation_generation=2, _lease_only=True)
        lease = request['lease']
        head, (b0, b1), (f0, f1) = phase1_row(3008, 3300, W, P)
        for g in SLIDING:
            row = lease['rows'][g]
            self.assertEqual(row[:head], [0] * head)
            self.assertEqual(row[b0:b1], [b.block_id for _, b in pinned[g]])
            self.assertEqual(len(row), f1)
        parsed = parse_lease(lease, attention_groups=(0,), layout=coordinator.layout)
        self.assertEqual(parsed.zero, tuple(lease['zero']))
        k0, k1 = kept_blocks(3300, W, P)
        record = coordinator.records['r1']
        self.assertEqual(len(record['transient']), 5 * sum(1 for i in range(f0, f1) if i < k0))
        self.assertEqual({g: [i for i, _ in v] for g, v in record['window']['fresh'].items()},
                         {g: [i for i in range(f0, f1) if i >= k0] for g in SLIDING})
        coordinator.recover('r1')
        self.assertEqual(coordinator.proxy.calls, ['phase1_drain'])
        fa = coordinator.adopt(binding)
        self.assertEqual(set(fa), set(geo.fresh_pages) if hasattr(geo, 'fresh_pages') else set(fa))
        self.assertIn('r1', w.pins.pending)
        head_installed = w.pins.install('r1')
        self.assertTrue(all(first == k0 and len(blocks) == k1 - k0 for first, blocks in head_installed.values()))
        # The session pins nothing until the commit; the pool holds the head and the FA run.
        self.assertIsNone(w.pins.window('ns'))

    def test_abort_frees_the_waiting_window_blocks(self):
        w, coordinator, geo, binding = self.fixture()
        coordinator.run(binding, geo, allocation_generation=2, _lease_only=True)
        free = w.pool.get_num_free_blocks()
        waiting = sum(len(v) for v in coordinator.records['r1']['window']['fresh'].values())
        transient = sum(1 for b in coordinator.records['r1']['transient'] if b.block_id < len(w.pool.blocks)
                        and w.pool.blocks[b.block_id] is b)
        coordinator.abort(binding)
        self.assertNotIn('r1', coordinator.records)
        self.assertEqual(w.pool.get_num_free_blocks(), free + waiting + transient)
        self.assertIsNotNone(w.pins.window('ns'))          # the session keeps its (poisoned) window
        w.pins.close('ns')
        self.assertEqual(w.pool.get_num_free_blocks(), w.baseline)

    def test_window_must_match_the_geometry(self):
        from remote_ra_native_growing.contract import TurnGeometry
        w, coordinator, geo, binding = self.fixture()
        late = TurnGeometry(1, 3024, 3024, 3026, 3300, 3310, 189, 3024, False, False, False, 3300, P)
        with self.assertRaisesRegex(LiveV2Error, 'differs from the session window pin'):
            coordinator.run(binding, late, allocation_generation=2, _lease_only=True)
        self.assertEqual(coordinator.groups(), ((0,), SLIDING, 6))


class AdapterLayoutTests(unittest.TestCase):
    def adapter(self, layout, n_swa=None):
        from remote_ra_native_growing.adapter import EngineCoreAdapter
        from remote_ra_native_growing.contract import Authority
        w = GK.gemma_world()
        if n_swa is None:
            n_swa = I0.window_pool_blocks(layout, 5)
        manager = KP.kv_manager(w, GK.small_gemma_config(w, P, 900, n_swa), hash_block_size=P,
                                scheduler_block_size=P)
        managers = tuple(m for m in manager.coordinator.single_type_managers
                         if type(m).__name__ == 'FullAttentionManager')
        authority = Authority(epoch='e' * 64, loaded_source='c' * 64)
        return EngineCoreAdapter(authority, manager, page_tokens=P, full_attention_managers=managers,
                                 layout=layout)

    def test_window_layout_gets_window_pins(self):
        layout = I0.for_window_server(P, 2, W, 8192)
        adapter = self.adapter(layout)
        self.assertIsNotNone(adapter.window)
        self.assertIs(adapter.owner.window, adapter.window)
        self.assertEqual(adapter.window.groups, SLIDING)
        self.assertIs(adapter.mamba_pool, adapter.pools[1])
        self.assertEqual(adapter.mamba_tokens, P)
        self.assertTrue(adapter.decoupled)

    def test_pool_one_must_be_the_reserved_window_pool(self):
        layout = I0.for_window_server(P, 2, W, 8192)
        with self.assertRaisesRegex(Abort, 'reserved sliding-window pool 1'):
            self.adapter(layout, n_swa=I0.window_pool_blocks(layout, 5) + 1)

    def test_engine_boundary_kinds(self):
        from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
        import inspect
        source = inspect.getsource(EngineRpcBoundary.dispatch_mixed_request)
        self.assertIn("'SlidingWindowSpec' if getattr(self.adapter,'window',None) is not None else 'MambaSpec'",
                      source)


if __name__ == '__main__':
    unittest.main()
