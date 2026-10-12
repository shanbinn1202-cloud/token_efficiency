"""WP3 worker gates for the decoupled KV layout (FA Pages of P, GDN align block M).

CPU only. Covers: the GDN latest-state capture cadence ``end % M`` with M != P,
the attach-time layout resolution and view geometry (FA == P, Mamba == M),
TensorIO's decoupled FA/GDN storage disjointness, and per-pool id uniqueness
in the allocation owner. Legacy (P == M == 784, no layout key) keeps today's
behaviour in every case.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))
from remote_ra_live_v2.contract import Allocation, NativeIdentity, RenderAssignment, ZERO_DIGEST
from remote_ra_live_v2 import kv_layout
from remote_ra_live_v2.qwen import QwenAlignLifecycle
from remote_ra_live_v2.worker_backend import CudaWorkerBackend
try:
    import torch
except ImportError:
    torch = None


class SlotIO:
    evidence_kind = 'CUDA_PHYSICAL_TENSOR_OPERATIONS'  # Constructor shape only, never READY.
    def __init__(self): self.slots = {}; self.log = []
    @staticmethod
    def keys(groups): return [(g, b) for g, ids in sorted(groups.items()) for b in ids]
    def capture(self, groups, kind):
        self.log.append(('capture', kind, self.keys(groups)))
        return {'values': [self.slots.get(k, 0) for k in self.keys(groups)]}
    def restore(self, groups, kind, payload):
        self.log.append(('restore', kind, self.keys(groups)))
        self.slots.update(zip(self.keys(groups), payload['values']))
    def zero(self, groups, kind): self.slots.update((k, 0) for k in self.keys(groups))
    def drain(self): pass
    def payload_digest(self, payload): return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    def count(self, op): return sum(1 for e in self.log if e[:2] == (op, 'hybrid'))


class Owner:
    def __init__(self, io): self.io = io; self.current = {}; self.zeroed = set()
    def verify_identity(self): pass
    def lease(self, identity, generation, slots):
        self.current[identity.request] = (generation, {v.key for v in slots})
        for v in slots:
            if v.key not in self.zeroed:
                self.io.zero({v.group: (v.block,)}, 'hybrid'); self.zeroed.add(v.key)
    def verify_allocations(self, identity, slots, generation):
        assert all(v.key in self.current[identity.request][1] for v in slots)
    def verify_zero_receipt(self, identity, slots, generation):
        assert all(v.key in self.zeroed for v in slots)
    def release_after_drain(self, identity, generation, operation): return ()


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MambaCadenceTests(unittest.TestCase):
    """P = 4, M = 8: Pages close every 4 tokens, the GDN slot moves only after an 8-aligned end."""
    P, M = 4, 8

    def open(self, prompt):
        io = SlotIO(); owner = Owner(io)
        backend = CudaWorkerBackend(io, owner, source='b' * 64)
        qwen = QwenAlignLifecycle(backend, page_tokens=self.P, mamba_tokens=self.M)
        identity = NativeIdentity('r0', 's', 0, 'g0', 'a' * 64, 'b' * 64, 'm')
        backend.begin(identity, 1, namespace='ns')
        a = RenderAssignment(identity, local_turn=0, computed_start=0, sealed_start=0, candidate_end=0,
                             observation_start=prompt, observation_end=prompt, phase2_start=0,
                             prompt_tokens=prompt, page_tokens=self.P, token_digest='c' * 64,
                             carry_digest=ZERO_DIGEST)
        return io, owner, backend, qwen, identity, qwen.open(a, tuple(range(prompt)), 1)

    def step(self, io, owner, qwen, identity, state, count, a, b):
        start = state.cursor; end = start + count
        slots = (Allocation(1, a, 1, 'mamba', 1), Allocation(2, b, 1, 'mamba', 1))
        owner.lease(identity, 1, slots)
        qwen.before_forward(identity, start=start, count=count, logical_positions=range(start, end),
                            attention_allocations=(Allocation(0, 1, 1, 'attention', 1),),
                            hybrid_allocations=slots, allocation_generation=1)
        self.assertTrue(all(io.slots[(v.group, v.block)] == start for v in slots), 'GDN discontinuous')
        for v in slots: io.slots[(v.group, v.block)] = end
        qwen.after_forward(identity, torch.tensor([[float(x + 1), 1., .5] for x in range(start, end)]))

    def test_latest_captured_only_at_m_aligned_ends(self):
        io, owner, backend, qwen, identity, state = self.open(32)
        captures = []
        for _ in range(8):                              # ends 2, 4, ..., 16
            self.step(io, owner, qwen, identity, state, 2, 10, 11)
            captures.append(io.count('capture'))
        self.assertEqual(captures, [0, 0, 0, 1, 1, 1, 1, 2])   # only at ends 8 and 16, not 4 or 12
        self.assertEqual(backend.latest['r0'][0], 16)
        self.assertEqual(set(state.page_vectors), {0, 1, 2, 3})  # a vector per P-token Page

    def test_slot_move_after_m_end_relocates_and_after_p_end_is_refused(self):
        io, owner, backend, qwen, identity, state = self.open(32)
        self.step(io, owner, qwen, identity, state, 8, 10, 11)       # end 8: latest captured
        self.step(io, owner, qwen, identity, state, 4, 12, 13)       # moved after end 8: relocated
        self.assertEqual(io.count('restore'), 1)
        moved = (Allocation(1, 14, 1, 'mamba', 1), Allocation(2, 15, 1, 'mamba', 1))
        owner.lease(identity, 1, moved)
        with self.assertRaisesRegex(RuntimeError, 'latest evolving Mamba state absent'):
            qwen.before_forward(identity, start=12, count=4, logical_positions=range(12, 16),
                                attention_allocations=(Allocation(0, 1, 1, 'attention', 1),),
                                hybrid_allocations=moved, allocation_generation=1)

    def test_legacy_default_keeps_page_cadence(self):
        qwen = QwenAlignLifecycle(NS(), page_tokens=784)
        self.assertEqual(qwen.mamba_tokens, 784)
        with self.assertRaisesRegex(RuntimeError, 'page size required'):
            QwenAlignLifecycle(NS(), page_tokens=16, mamba_tokens=24)   # M must be a multiple of P


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class QwenLeaseCheckTests(unittest.TestCase):
    """before_forward checks only the FA leases appended since the last accepted forward."""

    def test_appended_leases_checked_prefix_trusted_only_if_unchanged(self):
        io, owner = SlotIO(), Owner(SlotIO())
        backend = CudaWorkerBackend(io, owner, source='b' * 64)
        owner.io = io
        qwen = QwenAlignLifecycle(backend, page_tokens=4)
        identity = NativeIdentity('r0', 's', 0, 'g0', 'a' * 64, 'b' * 64, 'm')
        backend.begin(identity, 1, namespace='ns')
        a = RenderAssignment(identity, local_turn=0, computed_start=0, sealed_start=0, candidate_end=0,
                             observation_start=40, observation_end=40, phase2_start=0, prompt_tokens=40,
                             page_tokens=4, token_digest='c' * 64, carry_digest=ZERO_DIGEST)
        state = qwen.open(a, tuple(range(40)), 1)
        slots = (Allocation(1, 90, 1, 'mamba', 1), Allocation(2, 91, 1, 'mamba', 1))
        owner.lease(identity, 1, slots)
        attention = (Allocation(0, 1, 1, 'attention', 1),)

        def forward(attn, count=4):
            start = state.cursor
            qwen.before_forward(identity, start=start, count=count, logical_positions=range(start, start + count),
                                attention_allocations=attn, hybrid_allocations=slots, allocation_generation=1)
            for v in slots: io.slots[(v.group, v.block)] = start + count
            qwen.after_forward(identity, torch.ones(count, 3))

        forward(attention)
        self.assertIs(state.checked_attention, attention)
        attention += (Allocation(0, 2, 1, 'attention', 1),)
        forward(attention)
        for bad in (Allocation(0, 3, 2, 'attention', 1), Allocation(0, 3, 1, 'mamba', 1), 'x'):
            with self.assertRaisesRegex(RuntimeError, 'allocation-generation leases required'):
                forward(attention + (bad,))
        # A prefix that is not the checked tuple is checked whole.
        rebuilt = (Allocation(0, 1, 1, 'mamba', 1),) + attention[1:]
        with self.assertRaisesRegex(RuntimeError, 'allocation-generation leases required'):
            forward(rebuilt)
        forward(tuple(Allocation(*v.key) for v in attention))       # equal values: accepted


def views(fa_tokens, mamba_tokens, fa_group=0, mamba_groups=(1, 2, 3)):
    return [NS(name='fa', group=fa_group, kind='attention', allocation_tokens=fa_tokens)] + \
           [NS(name='gdn%d' % g, group=g, kind='hybrid', allocation_tokens=mamba_tokens) for g in mamba_groups]


class LayoutAndGeometryTests(unittest.TestCase):
    def runner(self, layout_value=None, group_pool=None, config=True):
        runner = NS(kv_cache_config=NS(group_pool=group_pool))
        if config:
            extra = {} if layout_value is None else {kv_layout.LAYOUT_KEY: layout_value}
            runner.vllm_config = NS(additional_config=extra)
        return runner

    def test_worker_layout_legacy_and_decoupled(self):
        from remote_ra_live_v2.worker_composition import worker_layout
        self.assertEqual(worker_layout(self.runner(), dict(page_tokens=784)), kv_layout.legacy())
        layout = kv_layout.for_server(16, 4)
        whole = kv_layout.to_additional_config(layout)
        value = whole[kv_layout.LAYOUT_KEY]
        self.assertEqual(value, dict(page_tokens=16, mamba_tokens=784, mamba_slots=9))
        for declared in (whole, value):            # the startup may carry either form
            startup = dict(page_tokens=16, kv_layout=declared)
            self.assertEqual(worker_layout(self.runner(value, (1, 1, 1, 0)), startup), layout)
            # Fakes without vllm_config resolve the startup declaration.
            self.assertEqual(worker_layout(self.runner(group_pool=(0, 1, 1, 1), config=False), startup),
                             layout)
        startup = dict(page_tokens=16, kv_layout=value)
        refusals = [(self.runner(value, (0, 1, 1, 1)), dict(page_tokens=784)),          # undeclared
                    (self.runner(), startup),                                          # vLLM legacy
                    (self.runner(value, (0, 1, 1, 1)), dict(startup, page_tokens=784)),
                    (self.runner(value, None), startup),                               # one pool
                    (self.runner(None, (0, 1, 1, 1)), dict(page_tokens=784)),
                    (self.runner(dict(value, mamba_slots=1), (0, 1, 1, 1)), startup)]  # invalid
        for runner, start in refusals:
            with self.assertRaises((RuntimeError, ValueError)):
                worker_layout(runner, start)

    def test_view_geometry(self):
        from remote_ra_live_v2.worker_composition import check_view_geometry
        legacy, decoupled = kv_layout.legacy(), kv_layout.for_server(16, 1)
        pools = (0, 1, 1, 1)
        check_view_geometry(NS(views=views(784, 784)), legacy)
        check_view_geometry(NS(views=views(16, 784)), decoupled, pools)
        check_view_geometry(NS(views=views(784, 784)), kv_layout.for_server(784, 1, decoupled=True), pools)
        check_view_geometry(NS(views=views(64, 832)), kv_layout.for_server(64, 2), pools)
        # The live model may list the Mamba groups first: groups are looked up by kind.
        check_view_geometry(NS(views=views(16, 784, fa_group=3, mamba_groups=(0, 1, 2))), decoupled,
                            (1, 1, 1, 0))
        for io, layout, group_pool, reason in (
                (views(16, 784), legacy, None, 'allocation_tokens == page_tokens'),
                (views(784, 16), decoupled, pools, 'FA allocation_tokens == P'),
                (views(16, 16), decoupled, pools, 'FA allocation_tokens == P'),
                (views(16, 784), decoupled, None, 'FA groups in pool 0'),
                (views(16, 784), decoupled, (1, 1, 1, 0), 'FA groups in pool 0'),
                (views(16, 784), decoupled, (0, 1, 1), 'FA groups in pool 0')):
            with self.assertRaisesRegex(RuntimeError, reason):
                check_view_geometry(NS(views=io), layout, group_pool)
        mixed = views(16, 784); mixed[2].allocation_tokens = 800
        with self.assertRaisesRegex(RuntimeError, 'common Mamba block width'):
            check_view_geometry(NS(views=mixed), decoupled, pools)


@unittest.skipUnless(torch is not None, 'CPU torch required for TensorIO')
class DecoupledAliasTests(unittest.TestCase):
    """One backing allocation shared by an FA view and a GDN view (the legacy hybrid cohort)."""
    def make(self, shared, decoupled):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO, LayerView
        fa_backing = torch.zeros(4 * 16)
        gdn_backing = fa_backing if shared else torch.zeros(4 * 16)
        fa = fa_backing.view(4, 16)
        conv = torch.as_strided(gdn_backing, (4, 2), (16, 1), 0)
        ssm = torch.as_strided(gdn_backing, (4, 3), (16, 1), 2)
        cohort = ('fa', 'gdn') if shared else None
        common = dict(page_stride=64, page_offset=0)
        return CpuTensorTestIO([
            LayerView('fa', 0, 'attention', (fa,), (0,), page_bytes=64,
                      cohort=cohort or ('fa',), **common),
            LayerView('gdn', 1, 'hybrid', (conv, ssm), (0, 0), page_bytes=20,
                      cohort=cohort or ('gdn',), **common)], decoupled=decoupled)

    def test_legacy_allows_the_declared_cohort_decoupled_refuses_it(self):
        self.assertFalse(self.make(True, False).decoupled)
        with self.assertRaisesRegex(RuntimeError, 'decoupled FA/GDN native storage alias'):
            self.make(True, True)
        self.assertTrue(self.make(False, True).decoupled)

    def test_validate_is_per_pool_since_each_call_names_one_kind(self):
        io = self.make(False, True)
        io.validate({0: (1, 2)}, 'attention')
        io.validate({1: (1, 2)}, 'hybrid')       # the same ids in the other pool
        with self.assertRaisesRegex(RuntimeError, 'invalid physical block IDs'):
            io.validate({0: (1, 1)}, 'attention')


class PerPoolOwnerTests(unittest.TestCase):
    def owner(self, layout=None, mamba_first=False, group_pool=None):
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        shape = dict(fa_group=3, mamba_groups=(0, 1, 2)) if mamba_first else {}
        io = NS(views=views(16 if layout else 784, 784, **shape))
        runner = NS(kv_cache_config=NS(group_pool=group_pool))
        return RunnerAllocationOwner(runner, io, dict(page_tokens=16 if layout else 784), layout)

    def update(self, owner, rows, start=0):
        identity = NativeIdentity('r', 's', 0, 'g', 'a' * 64, 'b' * 64, 'm')
        rec = dict(identity=identity, generation=1)
        state = NS(assignment=NS(identity=identity, phase2_start=start, prompt_tokens=64),
                   allocation_generation=1, poisoned=False, active=None, cursor=start)
        command = owner._allocation_update(rec, dict(rows=rows, start=start, count=1, binding='x'))
        return command(None, NS(requests={'r': state})), rec

    @staticmethod
    def rows(fa, mamba, mamba_first=False):
        fa_row, mamba_rows = [[b, 1] for b in fa], [[[0, 0], [b, 1]] for b in mamba]
        return mamba_rows + [fa_row] if mamba_first else [fa_row] + mamba_rows

    def test_allocation_update_ids_unique_per_pool(self):
        decoupled = kv_layout.for_server(16, 1)
        same_id = self.rows([5, 6], [5, 7, 8])       # FA 5 and Mamba 5: different pools
        with self.assertRaisesRegex(RuntimeError, 'alias across groups'):
            self.update(self.owner(), same_id)
        receipt, rec = self.update(self.owner(decoupled), same_id)
        self.assertEqual(receipt['operation'], 'allocation_update')
        self.assertEqual(rec['native_interval'], (0, 1))
        for rows in (self.rows([5, 5], [6, 7, 8]),      # FA alias in pool 0
                     self.rows([5], [7, 7, 8])):        # two Mamba groups share pool 1
            with self.assertRaisesRegex(RuntimeError, 'alias across groups'):
                self.update(self.owner(decoupled), rows)
        with self.assertRaisesRegex(RuntimeError, 'rows differ from the KV groups'):
            self.update(self.owner(decoupled), self.rows([5], [6, 7]))

    def test_mamba_groups_first_with_vllm_group_pools(self):
        decoupled = kv_layout.for_server(16, 1)
        owner = self.owner(decoupled, mamba_first=True, group_pool=(1, 1, 1, 0))
        self.assertEqual((owner.attention_groups, owner.hybrid_groups), ((3,), (0, 1, 2)))
        self.assertEqual(owner.pool_of, {0: 1, 1: 1, 2: 1, 3: 0})
        self.update(owner, self.rows([5, 6], [5, 7, 8], mamba_first=True))
        with self.assertRaisesRegex(RuntimeError, 'alias across groups'):
            self.update(owner, self.rows([5, 6], [6, 7, 7], mamba_first=True))
        with self.assertRaisesRegex(RuntimeError, 'FA groups in pool 0'):
            self.owner(decoupled, mamba_first=True, group_pool=(0, 1, 1, 1))

    def test_step_claims_ids_per_pool(self):
        from remote_ra_live_v2.allocation_owner import StepRow
        def run(owner, current):
            row = StepRow('r', 0, 0, 0, 1, (), (), 1, tuple(current))
            owner._observe = lambda runner, *args: row
            runner = NS(input_batch=NS(req_ids=['r']))
            return owner.step(runner, NS(num_scheduled_tokens={'r': 1}))
        current = [(0, 5, 1, 'attention', 1), (1, 5, 1, 'mamba', 1), (2, 6, 1, 'mamba', 1)]
        with self.assertRaisesRegex(RuntimeError, 'pool allocation alias'):
            run(self.owner(), current)
        self.assertEqual(len(run(self.owner(kv_layout.for_server(16, 1)), current)), 1)
        with self.assertRaisesRegex(RuntimeError, 'pool allocation alias'):
            run(self.owner(kv_layout.for_server(16, 1)), current + [(3, 6, 1, 'mamba', 1)])

    def test_layout_page_size_must_match_startup(self):
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        with self.assertRaisesRegex(RuntimeError, 'decoupled layout Page size'):
            RunnerAllocationOwner(NS(), NS(views=views(16, 784)), dict(page_tokens=32),
                                  kv_layout.for_server(16, 1))


if __name__ == '__main__':
    unittest.main()
