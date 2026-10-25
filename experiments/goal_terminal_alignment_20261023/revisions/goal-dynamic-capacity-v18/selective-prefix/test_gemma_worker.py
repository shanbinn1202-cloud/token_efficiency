"""Gemma 4 stage 1 (gemma-s1 WP5): the worker lifecycle of the window layout, CPU only.

* ``WindowLifecycle`` / ``WindowWorkerBackend``: no GDN state; the committed carry is
  the sliding window after the computed end (``WindowCarry``, its digest the session
  boundary); phase 1 must borrow exactly that window; phase 2's first step needs the
  owner's proof that its rows start with the window phase 1 ended in; the GDN entry
  points are refused.
* ``contract.parse_lease`` with the window layout: sliding rows are the null head, the
  borrowed window and the fresh blocks; a Goal forward's sliding rows are cold rows.
* ``WindowAllocationOwner``: the phase-1 window check, the phase-2 head check, every
  live sliding block leased per step, the bound on a row, the final window.
* The Qwen lifecycle's split (``_state_lease``, ``_after_state``, ``_final_state``)
  keeps its behaviour (test_stageb_worker and test_gdn_handoff run unchanged).
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))

from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2.contract import (Allocation, LiveV2Error, NativeIdentity, RenderAssignment,
                                        ZERO_DIGEST, digest, mamba_tokens, parse_lease, startup_layout)
from remote_ra_live_v2.qwen import QwenAlignLifecycle
from remote_ra_live_v2.sliding_window import kept_blocks, phase1_row
from remote_ra_live_v2.window_lifecycle import (WindowCarry, WindowLease, WindowLifecycle,
                                                WindowWorkerBackend, window_digest)

try:
    import torch
except ImportError:
    torch = None

P, W = 16, 64                   # a small window keeps the fixtures readable: 5 blocks per slot
LAYOUT = I0.WindowLayout(P, W, 2, 256)
SLIDING = (1, 2)                # group 0 is the global full-attention group


class IO:
    evidence_kind = 'CUDA_PHYSICAL_TENSOR_OPERATIONS'   # constructor shape only

    def drain(self):
        pass

    def payload_digest(self, payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Owner:
    """The allocation owner's lineage answers, set by the test."""

    def __init__(self):
        self.current, self.head, self.windows = {}, set(), {}

    def verify_identity(self):
        pass

    def lease(self, identity, generation, slots):
        self.current[identity.request] = (generation, {v.key for v in slots})

    def verify_allocations(self, identity, slots, generation):
        current = self.current[identity.request]
        assert current[0] == generation and all(v.key in current[1] for v in slots), 'foreign allocation'

    def window_head_verified(self, identity, generation):
        return identity.request in self.head

    def final_window(self, identity, computed_end, generation):
        return self.windows[identity.request]

    def release_after_drain(self, identity, generation, operation):
        return ()


def window(end, base=100):
    """A window after ``end`` tokens: ((group, ((index, block, incarnation), ...)), ...)."""
    first, last = kept_blocks(end, W, P)
    return tuple((g, tuple((i, base * g + i, 1) for i in range(first, last))) for g in SLIDING)


class Session:
    def __init__(self):
        self.io, self.owner = IO(), Owner()
        self.backend = WindowWorkerBackend(self.io, self.owner, source='b' * 64)
        self.life = WindowLifecycle(self.backend, page_tokens=P)
        self.turn = self.local = self.C = 0

    def identity(self):
        return NativeIdentity('r%d' % self.turn, 's', self.turn, 'g%d' % self.turn, 'a' * 64, 'b' * 64, 'm')

    def open(self, observation, header):
        identity, generation = self.identity(), self.turn + 1
        self.backend.begin(identity, generation, namespace='ns')
        if not self.local:
            a = RenderAssignment(identity, local_turn=0, computed_start=0, sealed_start=0, candidate_end=0,
                                 observation_start=observation, observation_end=observation, phase2_start=0,
                                 prompt_tokens=observation + header, page_tokens=P, token_digest='c' * 64,
                                 carry_digest=ZERO_DIGEST)
        else:
            C = self.C
            o, e = C + 2, C + 2 + observation
            carry = self.backend.carry['s']
            a = RenderAssignment(identity, local_turn=self.local, computed_start=C, sealed_start=C // P * P,
                                 candidate_end=o // P * P, observation_start=o, observation_end=e,
                                 phase2_start=e, prompt_tokens=e + header, page_tokens=P,
                                 token_digest='c' * 64, carry_digest=carry.digest)
            # Phase 1: borrow the committed window, end in the window after e.
            self.backend.restore_carry(identity, carry.blocks(), C, carry.digest, generation)
            first, last = kept_blocks(e, W, P)
            self.backend.capture_phase1_state(identity, 'phase1-' + '1' * 32, e,
                                              {g: tuple((i, 500 + i) for i in range(first, last))
                                               for g in SLIDING}, generation)
            self.owner.head.add(identity.request)
        return identity, self.life.open(a, tuple(range(a.prompt_tokens)), generation)

    def slots(self, generation, n=3):
        return tuple(Allocation(g, 10 * g + i, generation, 'sliding', 1) for g in SLIDING for i in range(n))

    def step(self, identity, state, count):
        start, gen = state.cursor, state.allocation_generation
        slots = self.slots(gen)
        self.owner.lease(identity, gen, slots)
        lease = self.life.before_forward(identity, start=start, count=count,
            logical_positions=range(start, start + count),
            attention_allocations=(Allocation(0, 1, gen, 'suffix', 1),), hybrid_allocations=slots,
            allocation_generation=gen)
        rows = torch.tensor([[float(x + 1), 1., .5] for x in range(start, start + count)])
        self.life.after_forward(identity, rows)
        return lease

    def complete(self, identity, state):
        self.owner.windows[identity.request] = window(state.cursor)
        receipt = self.life.finish(identity, 'complete')
        self.C = self.backend.carry['s'].computed_end
        self.local += 1
        self.turn += 1
        return receipt


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class WindowLifecycleTests(unittest.TestCase):
    def test_cold_then_continuing_turn(self):
        s = Session()
        identity, state = s.open(40, 3)
        lease = s.step(identity, state, 40)
        self.assertIsInstance(lease.hybrid, WindowLease)
        self.assertEqual((lease.phase, lease.hybrid.restored_checkpoint_seal), ('prefill', 0))
        self.assertFalse(state.phase1_restored)
        for _ in range(4):
            s.step(identity, state, 1)          # header [40, 43), then one decode token
        s.complete(identity, state)
        carry = s.backend.carry['s']
        self.assertIsInstance(carry, WindowCarry)
        self.assertEqual((carry.computed_end, carry.window), (44, window(44)))
        self.assertEqual(carry.digest, window_digest('s', 'ns', 44, window(44)))
        self.assertEqual(carry.first(), {1: 0, 2: 0})
        # Continuing turn: C = 44, o = 46, e = 76.
        identity, state = s.open(30, 2)
        self.assertEqual(state.cursor, 76)
        lease = s.step(identity, state, 1)
        self.assertTrue(state.phase1_restored)
        self.assertEqual((lease.phase, lease.hybrid.restored_checkpoint_seal), ('suffix', 76))
        s.step(identity, state, 1)
        s.step(identity, state, 1)              # decode
        s.complete(identity, state)
        self.assertEqual(s.backend.carry['s'].computed_end, 79)
        self.assertEqual(s.backend.carry['s'].owner, identity)

    def test_phase1_must_borrow_the_committed_window(self):
        s = Session()
        identity, state = s.open(40, 3)
        for count in (40, 1, 1, 1):
            s.step(identity, state, count)
        s.complete(identity, state)
        identity, generation = s.identity(), s.turn + 1
        s.backend.begin(identity, generation, namespace='ns')
        carry = s.backend.carry['s']
        wrong = {**carry.blocks(), 1: (999,) + carry.blocks()[1][1:]}
        with self.assertRaisesRegex(LiveV2Error, 'does not borrow the committed window'):
            s.backend.restore_carry(identity, wrong, carry.computed_end, carry.digest, generation)
        with self.assertRaisesRegex(LiveV2Error, 'committed window absent/stale'):
            s.backend.restore_carry(identity, carry.blocks(), carry.computed_end + 1, carry.digest, generation)
        with self.assertRaisesRegex(LiveV2Error, 'needs its lineage'):
            s.backend.capture_phase1_state(identity, 'phase1-' + '1' * 32, 80, {1: ((0, 1),)}, generation)

    def test_phase2_needs_the_owner_head_proof(self):
        s = Session()
        identity, state = s.open(40, 3)
        for count in (40, 1, 1, 1):
            s.step(identity, state, count)
        s.complete(identity, state)
        identity, state = s.open(30, 2)
        s.owner.head.discard(identity.request)
        gen = state.allocation_generation
        slots = s.slots(gen)
        s.owner.lease(identity, gen, slots)
        with self.assertRaisesRegex(LiveV2Error, 'does not continue the phase-1 window'):
            s.life.before_forward(identity, start=75, count=1, logical_positions=range(75, 76),
                attention_allocations=(Allocation(0, 1, gen, 'suffix', 1),), hybrid_allocations=slots,
                allocation_generation=gen)

    def test_illegal_window_reuse_and_gdn_entry_points(self):
        s = Session()
        identity, state = s.open(40, 3)
        for count in (40, 1, 1, 1):
            s.step(identity, state, count)
        s.complete(identity, state)
        older = NativeIdentity('r9', 's', 0, 'g9', 'a' * 64, 'b' * 64, 'm')   # turn not after the owner's
        with self.assertRaisesRegex(LiveV2Error, 'illegal cross-request window reuse'):
            s.backend.begin(older, 5, namespace='ns')
        with self.assertRaisesRegex(LiveV2Error, 'illegal cross-request window reuse'):
            s.backend.begin(s.identity(), 5, namespace='other')
        for name in ('restore_phase1_state', 'verify_zeroed_hybrid', 'relocate_latest_hybrid',
                     'capture_latest_hybrid', 'capture_final_hybrid'):
            with self.subTest(name), self.assertRaisesRegex(AssertionError, 'no Gated-DeltaNet state'):
                getattr(s.backend, name)()

    def test_cancel_keeps_the_committed_window(self):
        s = Session()
        identity, state = s.open(40, 3)
        for count in (40, 1, 1, 1):
            s.step(identity, state, count)
        s.complete(identity, state)
        carry = s.backend.carry['s']
        identity, state = s.open(30, 2)
        s.step(identity, state, 1)
        receipt = s.life.finish(identity, 'cancel')
        self.assertTrue(receipt.poisoned)
        self.assertIs(s.backend.carry['s'], carry)

    def test_wrong_kind_and_digest_refused(self):
        s = Session()
        identity, state = s.open(40, 3)
        gen = state.allocation_generation
        mamba = tuple(Allocation(g, 10 * g, gen, 'mamba', 1) for g in SLIDING)
        s.owner.lease(identity, gen, mamba)
        with self.assertRaisesRegex(LiveV2Error, 'allocation-generation leases required'):
            s.life.before_forward(identity, start=0, count=40, logical_positions=range(0, 40),
                attention_allocations=(Allocation(0, 1, gen, 'attention', 1),), hybrid_allocations=mamba,
                allocation_generation=gen)
        with self.assertRaisesRegex(LiveV2Error, 'digest differs'):
            WindowCarry('s', 'ns', 43, window(43), 'f' * 64, identity, 1)
        self.assertEqual(QwenAlignLifecycle.state_kind, 'mamba')
        with self.assertRaisesRegex(LiveV2Error, 'window worker backend required'):
            WindowLifecycle(NS(window=False), page_tokens=P)


# ---------------------------------------------------------------- the phase-1 lease

def window_lease(C=43, e=75, R=(), goal=False, shift=0):
    """A phase-1 lease of the window layout: FA group 0, sliding groups 1 and 2."""
    Q = C // P * P
    tail = C > Q
    L = (C + 2) // P * P
    fresh = range(Q // P + tail, -(-e // P))
    fa_row = [200 + p for p in R] + ([300] if tail else []) + [400 + p for p in fresh]
    head, (b0, b1), (f0, f1) = phase1_row(C, e, W, P)
    rows = [fa_row]
    for g in SLIDING:
        rows.append([0] * head + [100 * g + i + (shift if g == 2 else 0) for i in range(b0, b1)] +
                    [700 + 10 * g + i for i in range(f0, f1)])
    owned = [400 + p for p in fresh] + [b for g in SLIDING for b in rows[g][f0:]]
    where = [(0, b) for b in fa_row] + [(1, b) for g in SLIDING for b in rows[g] if b]
    raw = dict(lease_id='phase1-' + 'a' * 32, request='r', binding='1' * 64, session='s', namespace='ns',
               generation=2, epoch='e', source='2' * 64, local_turn=1, page_tokens=P, start=C, end=e,
               sealed_start=Q, candidate_end=L, resident_pages=list(R),
               tail=dict(page=Q // P, valid_end=C, block_ids=[[300]]) if tail else None,
               tail_candidate=False, rows=rows, borrowed=[200 + p for p in R] + ([300] if tail else []),
               owned=owned, zero=[400 + p for p in fresh],
               run_fresh={str(p): [[400 + p]] for p in fresh},
               incarnations={('%d:%d' % k if k[0] else str(k[1])): 1 for k in where},
               sparse=dict(logical_matched_tokens=Q, dropped_full_attention_tokens=Q - len(R) * P),
               carry_digest='3' * 64, goal=None, mac='4' * 64)
    if goal:
        g_rows = [[900, 901]] + [[950 + 10 * g, 951 + 10 * g] for g in SLIDING]
        ids = [(0, 900), (0, 901)] + [(1, b) for g in SLIDING for b in g_rows[g]]
        raw['goal'] = dict(lease_id='goal-' + 'b' * 32, tokens=list(range(20)), rows=g_rows,
                           owned=[b for _, b in ids], zero=[900, 901],
                           incarnations={('1:%d' % b if p else str(b)): 1 for p, b in ids},
                           payload_sha256='5' * 64)
    return raw


class WindowLeaseTests(unittest.TestCase):
    def test_parse_window_rows(self):
        lease = parse_lease(window_lease(), attention_groups=(0,), layout=LAYOUT)
        self.assertEqual(lease.attention_groups, (0,))
        self.assertEqual(lease.pools, (0, 1, 1))
        head, (b0, b1), (f0, f1) = phase1_row(43, 75, W, P)
        self.assertEqual((head, b0, b1, f0, f1), (0, 0, 3, 3, 5))
        self.assertEqual(lease.rows[1], (100, 101, 102, 713, 714))
        self.assertEqual(lease.zero, tuple(400 + p for p in range(3, 5)))
        self.assertIn((1, 713), lease.block_keys())
        # A late turn: a null head before the window.
        lease = parse_lease(window_lease(C=1000, e=1040), attention_groups=(0,), layout=LAYOUT)
        head, (b0, b1), (f0, f1) = phase1_row(1000, 1040, W, P)
        self.assertEqual(lease.rows[1][:head], (0,) * head)
        self.assertEqual(len(lease.rows[1]), f1)
        self.assertGreater(head, 0)

    def test_goal_rows_are_cold(self):
        # A goal_query layout reserves the Goal forward's transient (gemma-goal); the
        # 20-token Goal fits 32 tokens and no Goal fits a layout without one.
        goal = I0.WindowLayout(P, W, 2, 256, 32)
        lease = parse_lease(window_lease(goal=True), attention_groups=(0,), layout=goal)
        self.assertEqual(lease.goal.rows[1], (960, 961))
        for layout in (LAYOUT, I0.WindowLayout(P, W, 2, 256, 16)):
            with self.assertRaisesRegex(LiveV2Error, 'exceeds the Goal transient'):
                parse_lease(window_lease(goal=True), attention_groups=(0,), layout=layout)

    def test_refusals(self):
        def broken(edit, **kw):
            raw = window_lease(**kw)
            edit(raw)
            with self.assertRaises(LiveV2Error):
                parse_lease(raw, attention_groups=(0,), layout=LAYOUT)
        broken(lambda r: r['rows'][1].__setitem__(0, 0))           # a hole in the window
        broken(lambda r: r['rows'][1].append(0))                   # too long
        broken(lambda r: r['rows'][2].pop())                       # too short
        broken(lambda r: r['rows'][1].__setitem__(0, 5), C=1000, e=1040)   # no null head
        # The Mamba form of the same lease is refused by the window rule.
        with self.assertRaises(LiveV2Error):
            parse_lease(window_lease(), attention_groups=(0,), layout=I0.for_server(16, 1))

    def test_startup_layout_and_mamba_tokens(self):
        record = dict(page_tokens=16, kv_layout=I0.to_additional_config(LAYOUT)[I0.LAYOUT_KEY])
        self.assertEqual(startup_layout(record), LAYOUT)
        with self.assertRaises(LiveV2Error):
            startup_layout(dict(page_tokens=32, kv_layout=record['kv_layout']))
        with self.assertRaises(LiveV2Error):
            startup_layout(dict(page_tokens=16, kv_layout=dict(record['kv_layout'], state='other')))
        self.assertEqual(mamba_tokens(16, LAYOUT), 16)
        self.assertEqual(mamba_tokens(16, I0.for_server(16, 1)), 784)
        self.assertEqual(Allocation(1, 5, 1, 'sliding', 1).kind, 'sliding')


# ---------------------------------------------------------------- the window owner

class FakeTable:
    block_size = P
    blocks_per_kv_block = 1

    def __init__(self, ids):
        self.ids = ids

    def get_row(self, row):
        return self.ids


def owner_fixture():
    from remote_ra_live_v2.window_owner import WindowAllocationOwner
    groups = [NS(kv_cache_spec=type('FullAttentionSpec', (), {})()),
              NS(kv_cache_spec=type('SlidingWindowSpec', (), {})()),
              NS(kv_cache_spec=type('SlidingWindowSpec', (), {})())]
    runner = NS(kv_cache_config=NS(kv_cache_groups=groups, group_pool=(0, 1, 1)), requests={},
                input_batch=NS(block_table=NS(block_tables={})))
    io = NS(views=(NS(kind='attention', group=0),), arena=None)
    owner = WindowAllocationOwner(runner, io, dict(page_tokens=P, source='b' * 64), LAYOUT)
    return owner, runner


class WindowOwnerTests(unittest.TestCase):
    def test_groups_and_pools(self):
        owner, _ = owner_fixture()
        self.assertEqual((owner.attention_groups, owner.hybrid_groups), ((0,), (1, 2)))
        self.assertEqual(owner.pool_of, {0: 0, 1: 1, 2: 1})
        from remote_ra_live_v2.window_owner import WindowAllocationOwner
        with self.assertRaisesRegex(LiveV2Error, 'needs the window layout'):
            WindowAllocationOwner(NS(), NS(views=()), dict(page_tokens=P), I0.for_server(16, 1))

    def test_phase1_window_check_and_final_window(self):
        owner, runner = owner_fixture()
        identity = NativeIdentity('r1', 's', 1, 'g1', 'a' * 64, 'b' * 64, 'm')
        carry_window = tuple((g, tuple((i, 100 * g + i, 1) for i in range(0, 3))) for g in SLIDING)
        carry = WindowCarry('s', 'ns', 43, carry_window, window_digest('s', 'ns', 43, carry_window),
                            NativeIdentity('r0', 's', 0, 'g0', 'a' * 64, 'b' * 64, 'm'), 1)
        owner.backend = NS(carry={'s': carry})
        rec = dict(identity=identity, generation=2)
        lease = parse_lease(dict(window_lease(), carry_digest=carry.digest), attention_groups=(0,), layout=LAYOUT)
        head = owner.check_phase1_window(rec, lease, dict(carry_digest=carry.digest))
        first, last = kept_blocks(75, W, P)
        self.assertEqual(head, {g: (first, lease.rows[g][first:last]) for g in SLIDING})
        bad = parse_lease(dict(window_lease(shift=50), carry_digest=carry.digest),
                          attention_groups=(0,), layout=LAYOUT)
        with self.assertRaisesRegex(LiveV2Error, 'does not borrow the committed window'):
            owner.check_phase1_window(rec, bad, dict(carry_digest=carry.digest))
        with self.assertRaisesRegex(LiveV2Error, 'without its committed window'):
            owner.check_phase1_window(rec, lease, dict(carry_digest='0' * 64))
        # The final window from the last published rows.
        owner.records['r1'] = dict(identity=identity, generation=2,
                                   native_rows=[[(5, 1)] * 6, [(0, 0), (0, 0)] + [(40 + i, 2) for i in range(4)],
                                                [(0, 0), (0, 0)] + [(60 + i, 3) for i in range(4)]])
        self.assertEqual(kept_blocks(90, W, P), (1, 6))
        with self.assertRaisesRegex(LiveV2Error, 'not in the native rows'):
            owner.final_window(identity, 90, 2)
        self.assertEqual(kept_blocks(96, W, P), (2, 6))
        self.assertEqual(owner.final_window(identity, 96, 2),
                         ((1, tuple((2 + i, 40 + i, 2) for i in range(4))),
                          (2, tuple((2 + i, 60 + i, 3) for i in range(4)))))

    def test_observe_leases_live_sliding_blocks_and_checks_the_head(self):
        owner, runner = owner_fixture()
        from remote_ra_live_v2.sparse_mapping import RowLayout
        identity = NativeIdentity('r1', 's', 1, 'g1', 'a' * 64, 'b' * 64, 'm')
        a = RenderAssignment(identity, local_turn=1, computed_start=43, sealed_start=32, candidate_end=32,
                             observation_start=45, observation_end=75, phase2_start=75, prompt_tokens=77,
                             page_tokens=P, token_digest='c' * 64, carry_digest='3' * 64)
        layout = RowLayout(P, 32, (0, 1), {0: (11, 12)}, {0: (13, 14, 15)})
        head = {g: (1, (101 + 100 * (g - 1), 102 + 100 * (g - 1), 103 + 100 * (g - 1), 104 + 100 * (g - 1)))
                for g in SLIDING}
        owner.records['r1'] = dict(identity=identity, generation=2, assignment=a, layout=layout,
                                   window_head=head)
        fa = [11, 12, 13, 14, 15]
        sliding = {g: [0] + list(head[g][1]) for g in SLIDING}
        request = NS(num_computed_tokens=75, block_ids={0: fa, **sliding})
        runner.requests['r1'] = request
        runner.input_batch.block_table.block_tables[0] = FakeTable(fa)
        s = owner._observe(runner, 'r1', 0, 0, 1)
        self.assertEqual(len(s.hybrid), 8)
        self.assertTrue(all(v.kind == 'sliding' for v in s.hybrid))
        self.assertTrue(owner.window_head_verified(identity, 2))
        self.assertEqual(len(s.attention), 5)
        # A row that lost the window is refused at the first step.
        owner.records['r1'].pop('window_head_verified')
        request.block_ids[2] = [0, 999] + list(head[2][1][1:])
        with self.assertRaisesRegex(LiveV2Error, 'does not start with the phase-1 window'):
            owner._observe(runner, 'r1', 0, 0, 1)
        # A row with a hole or longer than one window is refused.
        owner.records['r1']['window_head_verified'] = True
        request.block_ids[2] = [0, 5, 0, 7]
        with self.assertRaisesRegex(LiveV2Error, 'one window of live blocks'):
            owner._observe(runner, 'r1', 0, 0, 1)
        request.block_ids[2] = list(range(1, 12))
        with self.assertRaisesRegex(LiveV2Error, 'one window of live blocks'):
            owner._observe(runner, 'r1', 0, 0, 1)


class AttachDispatchTests(unittest.TestCase):
    def test_window_layout_attaches_its_own_worker(self):
        from remote_ra_live_v2 import worker_composition as WC
        runner = NS(_remote_ra_growing_view_inputs=((), ()))
        worker = NS(model_runner=runner)
        with patch.object(WC, 'worker_layout', return_value=LAYOUT), \
                patch('remote_ra_live_v2.window_worker.attach_window_worker',
                      return_value='window evidence') as attach:
            self.assertEqual(WC.attach_loaded_worker(worker, dict(enabled=True)), 'window evidence')
        attach.assert_called_once()
        self.assertIs(attach.call_args.args[2], LAYOUT)

    def test_runtime_mode_needs_align_except_for_the_window_layout(self):
        from remote_ra_live_v2.positions import runtime_mode
        runner = NS(cache_config=NS(mamba_cache_mode='none'), num_spec_tokens=0, speculative_config=None,
                    use_async_scheduling=False, uses_xdrope_dim=0, uses_mrope=False)
        with self.assertRaisesRegex(LiveV2Error, 'mamba_cache_mode'):
            runtime_mode(runner, 'probe')
        runner._remote_ra_live_v2_allocation_owner = NS(layout=LAYOUT)
        runtime_mode(runner, 'probe')
        runner._remote_ra_live_v2_allocation_owner = NS(layout=I0.for_server(16, 1))
        with self.assertRaisesRegex(LiveV2Error, 'mamba_cache_mode'):
            runtime_mode(runner, 'probe')


if __name__ == '__main__':
    unittest.main()
