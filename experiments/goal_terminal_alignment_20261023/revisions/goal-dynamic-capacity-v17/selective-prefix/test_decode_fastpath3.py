"""serve.py --decode-fastpath3 (remote_ra_live_v2.decode_fastpath3). CPU only.

Off: no config key, worker record, evidence key or vLLM argument; the code paths are
those of --decode-fastpath2. On (with --decode-fastpath2): the same values, plans and
evidence as --decode-fastpath2 alone, checked piece by piece and end to end:

- the byte compare of a device row against np.array_equal, and the one-copy step
  positions against positions.step_positions, on random inputs;
- the owner's steady observation and materialize against the full per-step path on
  test_decouple_wp3_owner's randomized two-world sequences (Page and Mamba crossings,
  row moves, finishes with block reuse, corruptions), results, refusals and state;
- before_forward3 against before_forward on random lease sequences, values and refusals;
- the multi-session fakes (test_decouple_e2e's MultiHarness: the legacy 784 pool, the
  decoupled 784 and 16-token layouts; full and observation_query; goal_query with the
  GDN hand-off and with in-step phase 1), also on a vLLM-shaped runner (numpy block
  tables grown in place, MRoPE positions) where the steady paths run: identical replies,
  attended Pages, plans, computed positions, evidence files, per-request frames and RPCs.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import numpy as np

from remote_ra_live_v2 import decode_fastpath as F
from remote_ra_live_v2 import decode_fastpath2 as F2
from remote_ra_live_v2 import decode_fastpath3 as F3
from remote_ra_live_v2 import allocation_owner
from remote_ra_live_v2.contract import LiveV2Error
# Imported here so the end-to-end runs' sys.modules patch (enter_case) keeps them
# (test_decode_fastpath explains why).
from remote_ra_live_v2 import (generation_guard, hooks, loaded_sources, native_probe,  # noqa: F401
                               page_compaction, phase1_step, positions, vector_space, worker_composition,
                               worker_rows)
from remote_ra_growing_worker import tensors  # noqa: F401

try:
    import torch
except ImportError:
    torch = None

ALL = {'decode_fastpath': True, 'decode_fastpath2': True, 'decode_fastpath3': True}
TWO = {'decode_fastpath': True, 'decode_fastpath2': True}


class PinTests(unittest.TestCase):
    def test_pin_values(self):
        self.assertIsNone(F3.pinned({}))
        self.assertIsNone(F3.pinned(None))
        self.assertIsNone(F3.pinned(dict(TWO)))
        self.assertTrue(F3.pinned(dict(ALL)))
        for bad in (False, 1, 'true', None):
            with self.assertRaises(LiveV2Error):
                F3.pinned(dict(TWO, decode_fastpath3=bad))
        for bad in ({'decode_fastpath3': True}, {'decode_fastpath': True, 'decode_fastpath3': True},
                    {'decode_fastpath2': True, 'decode_fastpath3': True}):
            with self.assertRaises(LiveV2Error):
                F3.pinned(bad)                                   # needs both earlier fast paths

    def test_worker_flag(self):
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertIsNone(F3.worker_flag(dict(TWO)))
            self.assertTrue(F3.worker_flag(dict(ALL)))
            for bad in ({'decode_fastpath3': True}, dict(TWO, decode_fastpath3=False),
                        {'decode_fastpath': True, 'decode_fastpath3': True}):
                with self.assertRaises(LiveV2Error):
                    F3.worker_flag(bad)
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'):
            with patch('remote_ra_http.config.load', lambda: dict(ALL)):
                self.assertTrue(F3.worker_flag(dict(ALL)))
                with self.assertRaises(LiveV2Error):
                    F3.worker_flag(dict(TWO))
            with patch('remote_ra_http.config.load', lambda: dict(TWO)):
                self.assertIsNone(F3.worker_flag(dict(TWO)))
                with self.assertRaises(LiveV2Error):
                    F3.worker_flag(dict(ALL))

    def test_attach_needs_the_second_fast_path_first(self):
        owner, bridge, qwen = NS(decode_fastpath2=False), NS(decode_fastpath2=False), NS()
        with self.assertRaises(LiveV2Error):
            F3.attach_worker(owner, bridge, qwen)
        owner.decode_fastpath2 = bridge.decode_fastpath2 = True
        F3.attach_worker(owner, bridge, qwen)
        self.assertTrue(owner.decode_fastpath3 and bridge.decode_fastpath3 and qwen.decode_fastpath3)
        with self.assertRaises(LiveV2Error):
            F3.attach_worker(owner, bridge, qwen)                # once
        boundary = NS(batched=None)
        with self.assertRaises(LiveV2Error):
            F3.attach_engine(boundary)
        boundary = NS(batched={}, decode_fastpath3=None)
        F3.attach_engine(boundary)
        from remote_ra_live_v2 import allocation_guard, chunk_protocol, pool_incarnations
        self.assertEqual(boundary.decode_fastpath3, dict(guard=allocation_guard, chunk=chunk_protocol, fast2=F2,
                                                         incarnations=pool_incarnations))
        with self.assertRaises(LiveV2Error):
            F3.attach_engine(boundary)


class RowCompareTests(unittest.TestCase):
    def test_bytes_compare_equals_array_equal(self):
        rng = np.random.default_rng(3)
        same = allocation_owner._same_row
        checked = 0
        for _ in range(4000):
            n = int(rng.integers(0, 40))
            dtype = rng.choice([np.int32, np.int64, np.uint16, np.float32, np.float64])
            a = rng.integers(-3, 4, size=n).astype(dtype)
            kind = rng.integers(0, 7)
            if kind == 0:
                b = a.copy()
            elif kind == 1 and n:
                b = a.copy()
                b[int(rng.integers(0, n))] += 1
            elif kind == 2:
                b = a[:max(0, n - 1)].copy()
            elif kind == 3:
                b = a.astype(np.int64 if dtype != np.int64 else np.int32)   # another dtype, equal values
            elif kind == 4 and np.issubdtype(dtype, np.floating) and n:
                b = a.copy()
                b[0] = -0.0 if a[0] == 0 else np.nan                       # -0.0 == 0.0; NaN != NaN
                a = a.copy()
                a[0] = 0.0 if a[0] == 0 else np.nan
            elif kind == 5:
                table = np.zeros((3, n + 5), dtype=dtype)
                table[1, :n] = a
                b, a = a.copy(), table[1, :n]                                # a row view of a table
            else:
                b = a.reshape(1, n)                                          # another shape
            self.assertEqual(same(a, b), np.array_equal(a, b), (a, b))
            checked += 1
        self.assertEqual(checked, 4000)


@unittest.skipUnless(torch is not None, 'CPU torch required')
class StepPositionsTests(unittest.TestCase):
    def test_one_copy_reads_the_same_lists(self):
        rng = random.Random(5)
        for _ in range(400):
            n, end = rng.randrange(1, 40), rng.randrange(1, 45)
            mrope = rng.random() < .8
            rows = torch.randint(0, 9999, (3, n), dtype=torch.int64)
            flat = torch.randint(0, 9999, (n,), dtype=torch.int64)
            logical = torch.randint(0, 9999, (rng.choice((n, max(1, n - 3), n + 4)),),
                                    dtype=rng.choice((torch.int64, torch.int32)))
            runner = NS(uses_mrope=mrope, positions=logical)
            given = rows if mrope else flat
            if rng.random() < .1:
                given = rows if not mrope else flat                          # a wrong shape
            expected = positions.step_positions(runner, given, end)
            self.assertEqual(F3.step_positions(runner, given, end), expected)


class Fast3Driver:
    """test_decouple_wp3_owner's Driver with world A on --decode-fastpath2 and --decode-fastpath3
    (its materialize is the steady test, then materialize) and world B the full per-step path."""

    @staticmethod
    def make(seed, fa, mamba, layout, verified):
        import test_decouple_wp3_owner as O
        d = O.Driver(seed, fa, mamba, layout, verified)
        a = d.worlds[0]
        a.owner.decode_fastpath2 = a.owner.decode_fastpath3 = True
        counts = dict(steady=0, observed=0)
        full = a.owner.materialize

        def materialize(s):
            if a.owner.materialize_steady(s):
                counts['steady'] += 1
                return None
            return full(s)
        a.owner.materialize = materialize
        step = a.owner.step

        def counted(runner, so):
            steps = step(runner, so)
            for s in steps:
                o = a.owner.records[s.request].get('observed')
                if o is not None and o.base is not None and s.hybrid is o.base.hybrid:
                    counts['observed'] += 1            # the steady observation (its leases kept)
            return steps
        a.owner.step = counted
        return d, counts


class SteadyOwnerEquivalenceTests(unittest.TestCase):
    def drive(self, fa, mamba, layout, seeds, verified=True, events=250):
        totals = dict(steady=0, observed=0, ops=0)
        for seed in seeds:
            d, counts = Fast3Driver.make(seed, fa, mamba, layout, verified)
            d.run(self, events)
            totals['steady'] += counts['steady']
            totals['observed'] += counts['observed']
            totals['ops'] += d.ops
        return totals

    def test_legacy_single_pool(self):
        t = self.drive((0,), (1, 2, 3), None, range(16))
        self.assertGreater(t['observed'], 300, t)
        self.assertGreater(t['steady'], 300, t)

    def test_decoupled_two_pools(self):
        from remote_ra_live_v2 import kv_layout
        import test_decouple_wp3_owner as O
        t = self.drive((0,), (1, 2, 3), kv_layout.for_server(O.P, 3), range(16))
        self.assertGreater(t['observed'], 300, t)
        self.assertGreater(t['steady'], 300, t)

    def test_decoupled_mamba_groups_first(self):
        from remote_ra_live_v2 import kv_layout
        import test_decouple_wp3_owner as O
        t = self.drive((3,), (0, 1, 2), kv_layout.for_server(O.P, 3), range(10))
        self.assertGreater(t['observed'], 150, t)

    def test_fixtures_without_incarnation_rows(self):
        t = self.drive((0,), (1, 2, 3), None, range(6), verified=False)
        self.assertGreater(t['ops'], 100, t)


class LeaseSequenceTests(unittest.TestCase):
    """before_forward3 (world A) and before_forward (world B) over the same random lease
    sequences: equal results or the same refusal, equal Qwen states and backend calls."""

    def world(self):
        from remote_ra_live_v2.qwen import QwenAlignLifecycle
        calls = []
        backend = NS(restore_phase1_state=lambda *a: calls.append(('restore',) + a),
                     verify_zeroed_hybrid=lambda *a: calls.append(('zeroed',) + a),
                     relocate_latest_hybrid=lambda *a: calls.append(('relocate',) + a))
        q = QwenAlignLifecycle(backend, page_tokens=16, mamba_tokens=64)
        q.decode_fastpath = q.decode_fastpath2 = True
        return q, calls

    @staticmethod
    def outcome(fn):
        try:
            return ('ok', fn())
        except LiveV2Error as e:
            return ('error', str(e))

    def test_random_sequences(self):
        from remote_ra_live_v2.contract import (Allocation, NativeIdentity, RenderAssignment, ZERO_DIGEST)
        rng = random.Random(9)
        counts = dict(fallback=0, calls=0)
        for seq in range(60):
            (a, ca), (b, cb) = self.world(), self.world()
            local = rng.random() < .5
            prompt = rng.randrange(40, 90)
            identity = NativeIdentity('r%d' % seq, 's', 1, 'g', 'a' * 64, 'b' * 64, 'm')
            if local:
                ra = RenderAssignment(identity, 1, 20, 16, 32, 33, 36, 36, prompt, 16, 'c' * 64, 'd' * 64)
            else:
                ra = RenderAssignment(identity, 0, 0, 0, 0, 5, 5, 0, prompt, 16, 'c' * 64, ZERO_DIGEST)
            gen = 3
            tokens = tuple(range(prompt))
            for w in (a, b):
                w.open(ra, tokens, gen)
            fallback = a.before_forward

            def counted(*args, **kwargs):
                counts['fallback'] += 1
                return fallback(*args, **kwargs)
            a.before_forward = counted
            cursor = ra.phase2_start
            attn = tuple(Allocation(0, 100 + i, gen, 'attention') for i in range(3))
            hybrid = tuple(Allocation(g, 500 + g, gen, 'mamba') for g in (1, 2, 3))
            for step in range(60):
                x = rng.random()
                start, count = cursor, (1 if cursor >= prompt else min(prompt - cursor, rng.choice((1, 4, 16))))
                g = gen
                if x < .04:
                    start += 1                                        # stale cursor
                elif x < .07:
                    g = gen + 1                                       # stale generation
                if rng.random() < .1:
                    attn = attn + (Allocation(0, 200 + step, gen, 'decode' if cursor >= prompt else 'attention'),)
                if rng.random() < .05:
                    hybrid = tuple(Allocation(v.group, v.block + 50, gen, 'mamba') for v in hybrid)   # a move
                elif rng.random() < .05:
                    hybrid = tuple(hybrid)  # the same tuple object
                elif rng.random() < .05:
                    hybrid = tuple(list(hybrid))                      # an equal, new tuple
                if rng.random() < .03:
                    attn = attn + (Allocation(0, 900 + step, gen + 1, 'decode'),)   # a foreign lease
                logical = tuple(range(start, start + count))
                if rng.random() < .03:
                    logical = tuple(range(start + 1, start + count + 1))           # moved positions
                evidence = dict(request=identity.request, start=start, step=step)
                ra_ = self.outcome(lambda: a.before_forward3(identity, start, count, logical, evidence,
                                                             attn, hybrid, g))
                rb_ = self.outcome(lambda: b.before_forward(identity, start=start, count=count,
                                                            logical_positions=logical,
                                                            attention_allocations=attn,
                                                            hybrid_allocations=hybrid, allocation_generation=g,
                                                            position_evidence=evidence))
                counts['calls'] += 1
                self.assertEqual(ra_, rb_, (seq, step))
                self.assertEqual(vars(a.requests[identity.request]), vars(b.requests[identity.request]))
                self.assertEqual(ca, cb)
                if ra_[0] == 'ok':
                    if rng.random() < .05:
                        continue                                      # repeated callback next
                    for w in (a, b):                                  # after_forward's state change
                        st = w.requests[identity.request]
                        st.cursor, st.latest_state_step, st.active = start + count, start + count, None
                        st.prompt_complete = st.cursor >= prompt
                    cursor = start + count
                    if cursor >= prompt + 20:
                        break
                elif rng.random() < .3:
                    attn = tuple(v for v in attn if v.generation == gen)
            if rng.random() < .2:
                for w in (a, b):
                    w.requests[identity.request].terminal = True
                ra_ = self.outcome(lambda: a.before_forward3(identity, cursor, 1, (cursor,), None, attn, hybrid, gen))
                rb_ = self.outcome(lambda: b.before_forward(identity, start=cursor, count=1,
                                                            logical_positions=(cursor,), attention_allocations=attn,
                                                            hybrid_allocations=hybrid, allocation_generation=gen))
                self.assertEqual(ra_, rb_)
                self.assertEqual(ra_[0], 'error')
        self.assertGreater(counts['calls'] - counts['fallback'], 600, counts)
        self.assertGreater(counts['fallback'], 100, counts)


SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'


def first_difference(a, b, path=''):
    """Where two run outputs first differ (a path and the two values, shortened), or None."""
    if type(a) is not type(b):
        return path, type(a).__name__, type(b).__name__
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if a.get(k) != b.get(k):
                return first_difference(a.get(k), b.get(k), '%s/%s' % (path, k))
        return None
    if isinstance(a, (list, tuple)):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return first_difference(x, y, '%s[%d]' % (path, i))
        return None if len(a) == len(b) else (path, 'length', len(a), len(b))
    if isinstance(a, bytes) and a != b:
        try:
            return first_difference(json.loads(a), json.loads(b), path + '(json)')
        except ValueError:
            return path, a[:200], b[:200]
    return None if a == b else (path, repr(a)[:200], repr(b)[:200])


class NpTable:
    """vLLM 0.24's BlockTable shape: a numpy [rows, width] table and its row counts (no get_row)."""
    blocks_per_kv_block = 1

    def __init__(self, rows, width, block_size):
        self.block_size = block_size
        self.block_table = NS(np=np.zeros((rows, width), dtype=np.int32))
        self.num_blocks_per_row = np.zeros(rows, dtype=np.int32)

    def add_row(self, ids, row):
        n = len(ids)
        self.block_table.np[row, :n] = ids
        self.num_blocks_per_row[row] = n

    def append_row(self, ids, row):
        start = int(self.num_blocks_per_row[row])
        self.block_table.np[row, start:start + len(ids)] = ids
        self.num_blocks_per_row[row] = start + len(ids)

    def clear_row(self, row):
        self.num_blocks_per_row[row] = 0

    def move_row(self, src, tgt):
        n = int(self.num_blocks_per_row[src])
        self.block_table.np[tgt, :n] = self.block_table.np[src, :n]
        self.num_blocks_per_row[tgt] = n


def vllm_runner(mrope):
    """test_decouple_e2e's Runner with vLLM-shaped block tables: numpy rows, a running
    request's FullAttention id list extended in place (``_update_states``:
    ``block_ids[i].extend(new)``) and its device row appended (``append_row``); with
    ``mrope``, Qwen3.6's three text-only MRoPE position rows and the logical buffer."""
    import test_decouple_e2e as D

    class VllmRunner(D.Runner):
        def __init__(self, store, case):
            super().__init__(store, case)
            tables = self.input_batch.block_table.block_tables
            for g in list(tables):
                tables[g] = NpTable(8, case.blocks + 64, case.P if g == D.W.FA else case.M)
            if mrope:
                self.uses_mrope = True
                self.model_config = NS(hf_text_config=NS(rope_parameters=dict(
                    mrope_interleaved=True, mrope_section=[11, 11, 10])))

        def _remove(self, rid):
            batch = self.input_batch.req_ids
            row = batch.index(rid)
            batch[row] = None
            for t in self._tables():
                t.clear_row(row)

        def _condense(self):
            batch = self.input_batch.req_ids
            while batch and batch[-1] is None:
                batch.pop()
            while None in batch:
                hole, last = batch.index(None), len(batch) - 1
                batch[hole] = batch[last]
                batch.pop()
                for t in self._tables():
                    t.move_row(last, hole)
                    t.clear_row(last)
                while batch and batch[-1] is None:
                    batch.pop()

        def set_rows(self, rid, block_ids):
            req = self.requests[rid]
            row = self.input_batch.req_ids.index(rid) if rid in self.input_batch.req_ids else None
            for g, ids in zip(D.W.GROUPS, block_ids):
                cur, table = req.block_ids[g], self.input_batch.block_table.block_tables[g]
                if g == D.W.FA and len(ids) >= len(cur) and ids[:len(cur)] == cur:
                    new = list(ids[len(cur):])
                    cur.extend(new)
                    if row is not None and new:
                        table.append_row(new, row)
                else:
                    cur[:] = list(ids)
                    if row is not None:
                        table.add_row(list(ids), row)

        def execute_model(self, so):
            if not mrope:
                return super().execute_model(so)
            # D.Runner.execute_model with MRoPE positions (W.FakeRunner's form).
            before = hooks.before_forward

            def mrope_before(runner, scheduler_output, flat):
                runner.positions = flat
                return before(runner, scheduler_output, flat.repeat(3, 1))
            with patch.object(hooks, 'before_forward', mrope_before):
                return super().execute_model(so)
    return VllmRunner


def attach(h, fast3):
    """attach_loaded_worker's and compose_engine's fast-path attachments on a harness (both
    earlier fast paths, then the third); the queue goes out where the harness executes."""
    F.attach_worker(h.owner, h.bridge, h.qwen)
    F2.attach_worker(h.owner, h.bridge, h.qwen, h.io)
    h.boundary.batched = {}
    h.proxy.before_call = h.boundary.flush_updates
    if fast3:
        F3.attach_worker(h.owner, h.bridge, h.qwen)
        F3.attach_engine(h.boundary)
    execute = h.runner.execute_model

    def flushing(scheduler_output):
        h.boundary.flush_updates()
        return execute(scheduler_output)
    h.runner.execute_model = flushing


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    """--decode-fastpath3 vs --decode-fastpath2 alone over test_decouple_e2e's multi-session harness."""
    maxDiff = None

    def serve(self, case, policy, fast3, *, handoff=False, in_step=False, vllm=None):
        import test_decouple_e2e as D
        import test_host_trim as HT
        multi = D.Case(**dict(vars(case), sessions=len(D.TASKS)))
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = dict(D.config(policy, output, multi, D.TASKS), **TWO)
            flags = {}
            if in_step:
                flags['phase1_in_step'] = True
            if handoff:
                flags['gdn_device_handoff'] = True
            if flags:
                value['admission'] = flags
            if fast3:
                value['decode_fastpath3'] = True
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            if vllm is not None:
                stack.enter_context(patch.object(D, 'Runner', vllm_runner(vllm == 'mrope')))
            if in_step:
                import test_adm_phase1_step as A
                h = A.in_step_harness(handoff)(policy, output, multi)
            elif handoff:
                import test_gdn_handoff as G
                h = G.handoff_harness(D.MultiHarness)(policy, output, multi)
            else:
                h = D.MultiHarness(policy, output, multi)
            attach(h, fast3)
            counts = dict(steady=0, observed=0, rowless=0, lease=0)
            if fast3:
                steady = h.owner.materialize_steady
                h.owner.materialize_steady = lambda s: steady(s) and not counts.__setitem__(
                    'steady', counts['steady'] + 1)
                rowless = F3.rowless_entry

                def counted_rowless(*args):
                    receipt = rowless(*args)
                    counts['rowless'] += receipt is not None
                    return receipt
                stack.enter_context(patch.object(F3, 'rowless_entry', counted_rowless))
                lease = h.qwen.before_forward3
                fallback = h.qwen.before_forward

                def counted_lease(*args):
                    calls = []
                    h.qwen.before_forward = lambda *a, **k: calls.append(1) or fallback(*a, **k)
                    try:
                        return lease(*args)
                    finally:
                        h.qwen.before_forward = fallback
                        counts['lease'] += not calls
                h.qwen.before_forward3 = counted_lease
                step = h.owner.step

                def counted_step(runner, so):
                    steps = step(runner, so)
                    for s in steps:
                        o = h.owner.records[s.request].get('observed')
                        counts['observed'] += o is not None and o.base is not None and s.hybrid is o.base.hybrid
                    return steps
                h.owner.step = counted_step
            trace, rpc = [], h.collective_rpc

            def collective_rpc(method, args, timeout):
                operation, payload = args
                sent = HT.plain(payload)
                reply = rpc(method, args, timeout)
                trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
                return reply
            h.proxy.executor = NS(collective_rpc=collective_rpc)
            h.active = h.restores = 0
            h.serve()
            frames = {rid: HT.plain(items) for rid, items in h.proxy.http_receipts.items()}
            h.close()
            D.engine_closed(h.core)
            files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
            return dict(trace=trace, files=files, frames=frames, computed=list(h.runner.computed),
                        visible=dict(h.runner.visible), plans=dict(h.runner.plans),
                        done=[(i['rid'], list(i['request'].output_token_ids), list(i['plan']['selected']))
                              for i in h.done],
                        widths=list(h.widths), counts=counts,
                        forwards=[(r, s.cursor, s.receipts, s.position_evidence, s.latest_state_step)
                                  for r, s in sorted(h.qwen.requests.items())])

    def compare(self, two, three, rowless=True):
        for key in ('files', 'frames', 'computed', 'visible', 'plans', 'done', 'widths', 'trace', 'forwards'):
            # One located difference, not a diff of the whole run (unittest's diff of these takes minutes).
            self.assertTrue(three[key] == two[key], (key, first_difference(two[key], three[key])))
        self.assertGreater(max(two['widths']), 1)
        if rowless:
            self.assertGreater(three['counts']['rowless'], 0)

    def check(self, case, policy, vllm=None, **kw):
        two = self.serve(case, policy, False, vllm=vllm, **kw)
        three = self.serve(case, policy, True, vllm=vllm, **kw)
        # Rowless entries need the decoupled layout and no hand-off (decode_fastpath3.rowless_entry).
        self.compare(two, three, rowless=case.decoupled and not kw.get('handoff'))
        return three['counts']

    def test_vllm_shaped_16_token_pages(self):
        """The steady paths on the SWE-bench layout (P = 16, decoupled), MRoPE positions."""
        import test_decouple_e2e as D
        counts = self.check(D.CASES[2], 'full', vllm='mrope')
        for name in ('steady', 'observed', 'lease', 'rowless'):
            self.assertGreater(counts[name], 20, (name, counts))

    def test_legacy_784(self):
        """The legacy shared pool (784-token pages) on the vLLM-shaped runner: the steady paths run there
        too (its updates always carry rows: no rowless entry)."""
        import test_decouple_e2e as D
        counts = self.check(D.CASES[0], 'full', vllm='np')
        for name in ('steady', 'observed', 'lease'):
            self.assertGreater(counts[name], 20, (name, counts))

    @unittest.skipUnless(SLOW, 'about 2 minutes more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_every_case_and_policy(self):
        import test_decouple_e2e as D
        for case in D.CASES:
            for policy in ('full', 'observation_query'):
                for vllm in (None, 'np', 'mrope'):
                    if (case is D.CASES[2] and policy == 'full' and vllm == 'mrope') or \
                            (case is D.CASES[0] and policy == 'full' and vllm == 'np'):
                        continue                                       # in the default suite
                    with self.subTest(case=case.name, policy=policy, vllm=vllm):
                        counts = self.check(case, policy, vllm=vllm)
                        if vllm is not None and case.decoupled:
                            self.assertGreater(counts['observed'], 0, counts)

    def test_goal_query_with_the_gdn_hand_off(self):
        import test_decouple_e2e as D
        counts = self.check(D.CASES[2], 'goal_query', handoff=True, vllm='np')
        self.assertGreater(counts['observed'], 20, counts)
        self.assertEqual((counts['steady'], counts['rowless']), (0, 0))   # their IO joins differ: as before

    @unittest.skipUnless(SLOW, 'about a minute more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_goal_query_with_in_step_phase1(self):
        import test_decouple_e2e as D
        for handoff in (False, True):
            with self.subTest(handoff=handoff):
                self.check(D.CASES[2], 'goal_query', handoff=handoff, in_step=True)


class RowlessEntryTests(unittest.TestCase):
    """rowless_entry answers exactly where the command path would accept, else leaves the entry
    to it (None), so a refusal is the command path's own."""

    def setUp(self):
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        self.owner = owner = object.__new__(RunnerAllocationOwner)
        owner.decode_fastpath2 = owner.decode_fastpath3 = True
        owner.terminals = {}
        owner.claims = F.ClaimTable({(0, 5): ('a', 1), (0, 6): ('a', 1), (1, 9): ('a', 1), (0, 7): ('b', 1)})
        identity = NS(request='a')
        self.state = NS(assignment=NS(identity=identity, phase2_start=10, prompt_tokens=12),
                        allocation_generation=1, poisoned=False, active=None, cursor=14)
        owner.records = {'a': dict(identity=identity, generation=1, binding='bind-a', native_rows=((),),
                                   rows_seen=object(), native_interval=(13, 1))}
        self.qwen = NS(requests={'a': self.state})
        self.backend = NS(io=NS(handoff=False, group_bytes={0: 100, 1: 7}, _http_requests={'a': [3, 3, 1, 0]}),
                          owner=owner)
        self.entry = dict(request='a', binding='bind-a', start=14, count=1, rows=None)

    def command_path(self, entry, measured):
        from remote_ra_http.evidence import measured_command
        command = self.owner.resolve_worker_command('allocation_update', entry)
        if measured:
            return measured_command(self.backend, F2.BATCH, command, self.qwen, request=entry['request'])
        return command(self.backend, self.qwen)

    def outcome(self, fn):
        import itertools
        rec = dict(self.owner.records.get('a', {}))
        try:
            with patch('time.monotonic_ns', itertools.count(1).__next__):
                return ('ok', fn())
        except (LiveV2Error, KeyError) as e:
            return ('error', type(e).__name__, str(e))
        finally:
            self.owner.records.get('a', {}).update(rec)    # undo the record update for the next run

    def test_equal_to_the_command_path(self):
        for measured in (True, False):
            got = self.outcome(lambda: F3.rowless_entry(self.backend, self.owner, self.qwen, dict(self.entry),
                                                        measured))
            want = self.outcome(lambda: self.command_path(dict(self.entry), measured))
            self.assertEqual(got, want)
            self.assertEqual(got[0], 'ok')
            if measured:
                self.assertEqual(got[1]['http_measurement']['active_owned_kv_bytes'], 207)
                self.assertEqual(got[1]['http_measurement']['before'], dict(h2d_issued=3, h2d_joined=3, zero_joins=1))

    def test_refusals_are_the_command_paths(self):
        cases = [
            ('stale start', dict(start=13), None),
            ('a gap', dict(start=16), None),
            ('a count of two in decode', dict(count=2), None),
            ('no accepted rows', {}, lambda: self.owner.records['a'].update(native_rows=None)),
            ('an active lease', {}, lambda: setattr(self.state, 'active', object())),
            ('a poisoned request', {}, lambda: setattr(self.state, 'poisoned', True)),
        ]
        for name, change, prepare in cases:
            with self.subTest(name):
                self.setUp()
                if prepare is not None:
                    prepare()
                entry = dict(self.entry, **change)
                got = self.outcome(lambda: F3.rowless_entry(self.backend, self.owner, self.qwen, dict(entry), True))
                want = self.outcome(lambda: self.command_path(dict(entry), True))
                self.assertEqual(got[0], 'error', name)
                self.assertEqual(got, want, name)

    def test_left_to_the_command_path(self):
        cases = [('rows sent', dict(rows=[[]]), None),
                 ('a terminal replay', {}, lambda: self.owner.terminals.update(a=dict(operation='complete',
                                                                                    binding='bind-a'))),
                 ('an unknown request', dict(request='z'), None),
                 ('another binding', dict(binding='bind-x'), None),
                 ('the hand-off', {}, lambda: setattr(self.backend.io, 'handoff', True)),
                 ('no claim counts', {}, lambda: setattr(self.owner, 'claims', dict(self.owner.claims)))]
        for name, change, prepare in cases:
            with self.subTest(name):
                self.setUp()
                if prepare is not None:
                    prepare()
                self.assertIsNone(F3.rowless_entry(self.backend, self.owner, self.qwen, dict(self.entry, **change),
                                                   True))


class LauncherTests(unittest.TestCase):
    def test_switch_pins_one_more_config_key_and_changes_no_vllm_argument(self):
        import contextlib
        import io
        import install
        import serve
        import test_installer as INSTALL
        from remote_ra_http import config as http_config
        tasks = [dict(instance_id='repo__a-1', problem_statement='Fix a.'),
                 dict(instance_id='repo__b-2', problem_statement='Fix b.')]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            kw = dict(kv_cache_gib=24, page_tokens=16, cuda_graphs='full', max_model_len=131072)
            two = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'two', 8231, decode_fastpath=True,
                                decode_fastpath2=True, **kw)
            three = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'three', 8231,
                                  decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, **kw)
            off = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'off', 8231, decode_fastpath3=False,
                                **kw)
            for bad in (dict(decode_fastpath3=True), dict(decode_fastpath=True, decode_fastpath3=True),
                        dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=1)):
                with self.assertRaises(ValueError):
                    serve.prepare(release, [dict(t) for t in tasks], 'full', root/'bad', 8231, **bad, **kw)
            for name, config, expected in (('three', three[1], True), ('two', two[1], None)):
                path = root/(name + '.json')
                raw = serve.wire(config)
                path.write_bytes(raw)
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.digest(raw)):
                    self.assertEqual(http_config.load().get('decode_fastpath3'), expected)
                    self.assertEqual(F3.loaded(), expected)
            for bad in (dict(three[1], decode_fastpath3=False),
                        {k: v for k, v in three[1].items() if k != 'decode_fastpath2'}):
                path, raw = root/'bad.json', serve.wire(bad)
                path.write_bytes(raw)
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.digest(raw)):
                    with self.assertRaises(RuntimeError):
                        http_config.load()
        strip = lambda value, name: json.loads(json.dumps(value).replace(str(root/name).replace('\\', '\\\\'), '<out>'))
        self.assertEqual(strip(three[2], 'three'), strip(two[2], 'two'))                # vLLM argv unchanged
        self.assertEqual({k: v for k, v in strip(three[1], 'three').items() if k != 'decode_fastpath3'},
                         strip(two[1], 'two'))
        self.assertIs(three[1]['decode_fastpath3'], True)
        self.assertNotIn('decode_fastpath3', two[1])
        self.assertNotIn('decode_fastpath3', off[1])
        output = ('REMOTE_RA_HTTP_CONFIG', 'REMOTE_RA_HTTP_CONFIG_SHA256')
        self.assertEqual({k: v for k, v in three[3].items() if k not in output},
                         {k: v for k, v in two[3].items() if k not in output})

    def test_attach_loaded_worker_echoes_the_switch_and_attaches_it_after_the_second(self):
        import test_cudagraph as CG
        from remote_ra_live_v2 import worker_composition as WC

        class Owner(CG.FakeOwner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.claims = {}
        tests = CG.AttachLoadedWorkerTests()
        with patch.object(CG, 'FakeOwner', Owner), patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            two, _, _ = tests.attach(None, dict(TWO))
            three, runner, _ = tests.attach(None, dict(ALL))
            for bad in ({'decode_fastpath3': True}, {'decode_fastpath': True, 'decode_fastpath3': True}):
                with self.assertRaises(LiveV2Error):
                    tests.attach(None, bad)
        self.assertEqual(dict(three), dict(two, decode_fastpath3=True))
        owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
        self.assertTrue(owner.decode_fastpath3 and bridge.decode_fastpath3 and bridge.qwen.decode_fastpath3)
        self.assertTrue(owner.decode_fastpath2 and bridge.decode_fastpath2)
        self.assertIs(WC.WorkerBridge.decode_fastpath3, False)

    def test_compose_engine_sends_the_pin_once_and_checks_the_worker_echo(self):
        import contextlib
        import io
        import install
        import serve
        import test_installer as INSTALL
        from remote_ra_live_v2 import composition
        task = dict(instance_id='repo__a-1', problem_statement='Fix a.')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            config = serve.prepare(release, dict(task), 'full', root/'ce', 8231, decode_fastpath=True,
                                   decode_fastpath2=True, decode_fastpath3=True)[1]
            path = root/'ce.json'
            path.write_bytes(serve.wire(config))
            for echo in (dict(TWO), {}):
                sent = []

                def collective_rpc(method, args, timeout):
                    sent.append(args[0])
                    return [dict(echo)]
                core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                          _remote_ra_trusted_issue=dict(task), _remote_ra_trusted_issues=[dict(task)],
                          _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                          _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                    with self.assertRaises(LiveV2Error) as caught:
                        composition.compose_engine(core, object())
                startup, = sent
                self.assertEqual(startup, dict(enabled=True, source='m' * 64, page_tokens=784,
                                               loaded_sources_verified=True, **ALL))
                self.assertIn('decode_fastpath3' if echo else 'decode_fastpath', str(caught.exception))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ProfiledFastPath3Tests(unittest.TestCase):
    """The step profile on top of the three fast paths: the outputs of the run without it and
    the fast-path-3 steps under their names (rowless entries, steady leases and materialize)."""

    def serve(self, prof):
        import test_decouple_e2e as D
        import test_host_trim as HT
        import test_step_profile as TSP
        import remote_ra_step_profile as SP
        from remote_ra_live_v2.generation_guard import guarded_execute
        case = D.CASES[2]
        multi = D.Case(**dict(vars(case), sessions=len(D.TASKS)))
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = dict(D.config('full', output, multi, D.TASKS), **ALL)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            stack.enter_context(patch.object(SP, 'ENABLED', prof is not None))
            stack.enter_context(patch.object(SP, 'PROFILER', prof))
            stack.enter_context(patch.object(D, 'Runner', vllm_runner(True)))

            def guard(runner, execute):
                # attach_loaded_worker's order: the second and third fast paths, then the (profiled) guard.
                bridge = runner._remote_ra_live_v2_bridge
                owner = runner._remote_ra_live_v2_allocation_owner
                F2.attach_worker(owner, bridge, bridge.qwen, owner.io)
                F3.attach_worker(owner, bridge, bridge.qwen)
                if prof is None:
                    return guarded_execute(runner, execute)
                return SP.install_worker(runner, bridge, guarded_execute)(runner, execute)
            stack.enter_context(patch.object(D, 'guarded_execute', guard))
            if prof is not None:
                stack.callback(SP.uninstall, prof)
            h = (D.MultiHarness if prof is None else TSP.ProfiledMultiHarness)('full', output, multi)
            F.attach_worker(h.owner, h.bridge, h.qwen)
            h.boundary.batched = {}
            h.proxy.before_call = h.boundary.flush_updates
            F3.attach_engine(h.boundary)
            execute = h.runner.execute_model
            h.runner.execute_model = lambda so: (h.boundary.flush_updates(), execute(so))[1]
            if prof is not None:
                SP.install_engine(h.core, prof)
                SP.wrap(h, 'admit', 'input.registered', scope=True, profiler=prof)
                SP.wrap(h, 'finish', 'sched.complete', profiler=prof)
            h.serve()
            if prof is not None:
                prof.flush()
            h.close()
            D.engine_closed(h.core)
            files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
        return dict(files=files, done=[(i['rid'], list(i['request'].output_token_ids)) for i in h.done],
                    widths=list(h.widths))

    def test_profile_on_the_three_fast_paths(self):
        import test_step_profile as TSP
        import remote_ra_step_profile as SP
        off = self.serve(None)
        lines = []
        on = self.serve(SP.Profiler(every=10, emit=lines.append))
        self.assertTrue(on == off, first_difference(off, on))
        values = TSP.parse(lines)
        self.assertEqual([v['errors'] for v in values], [0] * len(values))
        total = values[-1]
        self.assertEqual(total['kind'], 'total')
        names = {name for g in total['groups'] for name in g['c']}
        for name in ('rpc.allocation_batch', 'rpc.allocation_batch.rowless', 'worker.before.qwen',
                     'worker.before.materialize', 'worker.finish', 'worker.finish.qwen'):
            self.assertIn(name, names)
        parent = dict(SP.TREE)
        for g in total['groups']:
            c = g['c']
            for name, stats in c.items():
                up = parent.get(name)
                if up is not None and up in c and name not in ('gpu.forward', 'gc'):
                    self.assertLessEqual(stats[2], c[up][2] + 1e-6, (name, up))
        decode = [g for g in total['groups'] if g['phase'] == 'decode' and 'rpc.allocation_batch' in g['c']]
        self.assertTrue(decode)
        for g in decode:
            # Every entry is tried as rowless once; the qwen lease once per request.
            self.assertEqual(g['c']['rpc.allocation_batch.rowless'][1], g['b'] * g['steps'])
            self.assertEqual(g['c']['worker.before.qwen'][1], g['b'] * g['steps'])


if __name__ == '__main__':
    unittest.main()
