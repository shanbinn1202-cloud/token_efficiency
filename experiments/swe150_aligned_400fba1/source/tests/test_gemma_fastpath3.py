"""Gemma 4 with --decode-fastpath3 (gemma-fastpath3): the third decode fast path on the window layout.
CPU only.

--decode-fastpath3 (docs/reference.md 4.11, test_decode_fastpath3) needs --decode-fastpath2. On the window layout
it adds to Qwen's cuts that apply there (rowless allocation entries, the byte compare of the global
group's device row, the steady materialize without its IO scope, the per-call imports bound once)
the window layout's own:

- engine: the window pool journals the blocks whose reference count or incarnation it changes
  (window_rows.Journal); a request's windows are tested block by block only when the journal names
  one of their blocks since their last test (window_rows.grow, ``fast3``);
- worker: a step whose rows did not change keeps the last observation's leases, key tuples and
  (empty) differences after the same checks (WindowAllocationOwner._observe_steady), its
  materialize only marks the step (materialize_steady); the window lifecycle builds the step lease
  of an unchanged window without re-testing its slots and tests a moved window's slots at C speed
  (WindowLifecycle.before_forward3); the backend does not re-test the generation of a slot tuple it
  tested at that generation (WindowWorkerBackend._verify); the owner looks a moved window's keys up
  in one pass (verify_allocations).

Checked here: the pins (launcher, startup check, worker attach, compose's engine attachment); the
journal path of grow against the full one over randomized engine steps on vLLM's KV manager with
its sliding managers (other requests allocating and freeing, double frees, freed blocks given to
another request, touched blocks, replaced row entries, journal restarts): equal results, refusals
and followed rows; the steady owner against the full and the other incremental worlds of
test_gemma_window_rows on its randomized sequences (row moves, finishes, corruptions); the lease
against before_forward and the backend's verification against CudaWorkerBackend's on random
sequences, refusals included; and test_gemma_e2e's engine and worker fakes with --decode-fastpath2
against --decode-fastpath3 (three sessions, every policy with REMOTE_RA_SLOW_TESTS=1, eager and with
the CUDA-graph record, one session, sixteen sessions with --release-idle-pages under pool pressure,
goal_query): identical RPC payloads and receipts, evidence files, per-request frames, computed
positions, attended Pages, plans, replies, sliding reads and per-step leases and claims.
"""
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
import test_gemma_e2e as GE  # noqa: E402
import test_gemma_fastpath as TF  # noqa: E402
import test_gemma_window_rows as WR  # noqa: E402
import test_stageb_worker as W  # noqa: E402
from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import decode_fastpath2 as F2  # noqa: E402
from remote_ra_live_v2 import decode_fastpath3 as F3  # noqa: E402
from remote_ra_live_v2 import window_owner as WO  # noqa: E402
from remote_ra_live_v2 import window_rows as WRows  # noqa: E402
from remote_ra_live_v2.contract import Allocation, LiveV2Error, NativeIdentity, RenderAssignment, ZERO_DIGEST  # noqa: E402
# Imported here, outside any sys.modules patch, so the patches the tests enter keep them
# (test_decode_fastpath explains why).
from remote_ra_live_v2 import (admission_flags, connector_guard, context_limits, decode_fastpath,  # noqa: F401,E402
                               decode_fastpath2, decode_fastpath3, generation_guard, hooks, idle_release,
                               language_model_only, loaded_sources, native_probe, page_compaction, positions,
                               vector_space, window_lifecycle, window_owner, window_worker, worker_composition,
                               worker_rows)

try:
    import torch
except ImportError:
    torch = None

SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
ALL = dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True)
TWO = dict(decode_fastpath=True, decode_fastpath2=True)


def attach_fast3(h):
    """attach_window_worker's and compose_engine's attachments of the three fast paths on a
    test_gemma_e2e harness (the first two as test_gemma_fastpath.attach_fast2 attaches them)."""
    TF.attach_fast2(h)
    F3.attach_worker(h.owner, h.bridge, h.qwen)
    F3.attach_engine(h.boundary)
    WO.attach_fastpath3(h.owner, h.qwen, h.backend)


# ---- the pins ------------------------------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    prepare, load = TF.LauncherTests.prepare, TF.LauncherTests.load

    def setUp(self):
        TF.LauncherTests.setUp(self)

    def test_switch_pins_one_more_config_key_and_changes_no_vllm_argument(self):
        _, two, two_argv, two_env = self.prepare(**TWO)
        _, three, three_argv, three_env = self.prepare(**ALL)
        self.assertEqual(three_argv, two_argv)
        self.assertNotIn('decode_fastpath3', two)
        self.assertEqual({k: v for k, v in three.items() if k != 'decode_fastpath3'}, two)
        self.assertIs(three['decode_fastpath3'], True)
        hashes = ('REMOTE_RA_HTTP_CONFIG_SHA256',)
        self.assertEqual({k: v for k, v in three_env.items() if k not in hashes},
                         {k: v for k, v in two_env.items() if k not in hashes})
        for kw, pattern in ((dict(decode_fastpath3=True), 'decode-fastpath3 requires --decode-fastpath2'),
                            (dict(decode_fastpath=True, decode_fastpath3=True),
                             'decode-fastpath3 requires --decode-fastpath2'),
                            (dict(TWO, decode_fastpath3=1), 'decode-fastpath3 is a switch')):
            with self.assertRaisesRegex(ValueError, pattern):
                self.prepare(**kw)
        # The startup loader takes the pin (the server's real config).
        _, config, _, _ = serve.prepare(self.release, dict(instance_id='a', problem_statement='fix a'), 'full',
                                        self.root / 'real3', 8000, None, kv_cache_gib=28, page_tokens=16,
                                        max_tokens=4096, max_model_len=131072, language_model_only=True, **ALL)
        self.assertIs(HC.decode_fastpath3(self.load(config)), True)
        with self.assertRaisesRegex(RuntimeError, 'requires decode_fastpath2'):
            self.load({k: v for k, v in config.items() if k != 'decode_fastpath2'})
        with self.assertRaisesRegex(RuntimeError, 'must be true'):
            self.load(dict(config, decode_fastpath3=False))

    def test_worker_attach_echoes_the_switch_and_attaches_the_window_paths(self):
        import test_gemma_graphs as TG
        attach = TG.WorkerAttachTests().attach
        two, two_runner = attach(dict(TWO), TG.compilation(None))
        three, runner = attach(dict(ALL, cuda_graphs='full'), TG.compilation('full'), (0,))
        self.assertEqual({k: v for k, v in three.items() if k != 'cuda_graphs'}, dict(two, decode_fastpath3=True))
        owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
        self.assertTrue(owner.window_steady and owner.decode_fastpath3 and bridge.decode_fastpath3 and
                        bridge.qwen.decode_fastpath3 and owner.backend.decode_fastpath3)
        self.assertTrue(owner.window_fast and owner.window_rows and owner.decode_fastpath2)
        off = two_runner._remote_ra_live_v2_allocation_owner
        self.assertFalse(off.window_steady or getattr(off, 'decode_fastpath3', False) or
                         getattr(off.backend, 'decode_fastpath3', False))
        for bad in ({'decode_fastpath3': True}, {'decode_fastpath': True, 'decode_fastpath3': True}):
            with self.assertRaisesRegex(LiveV2Error, 'decode_fastpath3 worker record needs decode_fastpath2'):
                attach(bad, TG.compilation(None))
        self.assertIs(WO.WindowAllocationOwner.window_steady, False)
        from remote_ra_live_v2.window_lifecycle import WindowWorkerBackend
        self.assertIs(WindowWorkerBackend.decode_fastpath3, False)

    def test_window_attachment_needs_the_earlier_ones_and_attaches_once(self):
        owner = NS(window_fast=True, window_rows=True, decode_fastpath3=True, window_steady=False)
        qwen = NS(state_kind='sliding', reuse_state_leases=True, decode_fastpath3=True)
        backend = NS(window=True)
        with self.assertRaisesRegex(LiveV2Error, 'first two window fast paths'):
            WO.attach_fastpath3(owner, qwen, backend)          # not a WindowAllocationOwner
        Owner = type('WindowAllocationOwner', (), {})
        for missing in ('window_fast', 'window_rows', 'decode_fastpath3'):
            o = Owner()
            o.__dict__.update(window_fast=True, window_rows=True, decode_fastpath3=True, window_steady=False)
            setattr(o, missing, False)
            with self.assertRaisesRegex(LiveV2Error, 'first two window fast paths'):
                WO.attach_fastpath3(o, qwen, backend)
        o = Owner()
        o.__dict__.update(window_fast=True, window_rows=True, decode_fastpath3=True, window_steady=False)
        for q in (NS(state_kind='mamba', reuse_state_leases=True, decode_fastpath3=True),
                  NS(state_kind='sliding', reuse_state_leases=False, decode_fastpath3=True)):
            with self.assertRaisesRegex(LiveV2Error, 'first two window fast paths'):
                WO.attach_fastpath3(o, q, backend)
        with self.assertRaisesRegex(LiveV2Error, 'first two window fast paths'):
            WO.attach_fastpath3(o, qwen, NS())                  # not the window backend
        WO.attach_fastpath3(o, qwen, backend)
        self.assertTrue(o.window_steady and backend.decode_fastpath3)
        with self.assertRaisesRegex(LiveV2Error, 'first two window fast paths'):
            WO.attach_fastpath3(o, qwen, backend)               # once

    def test_engine_attachment_binds_the_window_rows_and_their_journal(self):
        from remote_ra_live_v2 import allocation_guard, chunk_protocol, pool_incarnations
        qwen = dict(guard=allocation_guard, chunk=chunk_protocol, fast2=F2, incarnations=pool_incarnations)
        boundary = NS(batched={}, decode_fastpath3=None, adapter=NS(window=None))
        F3.attach_engine(boundary)
        self.assertEqual(boundary.decode_fastpath3, qwen)          # Qwen's adapter: no window rows
        pool = FakePool(8)
        boundary = NS(batched={}, decode_fastpath3=None, adapter=NS(window=NS(pool=pool)))
        F3.attach_engine(boundary)
        self.assertEqual(boundary.decode_fastpath3, dict(qwen, windows=WRows))
        journal = pool._remote_ra_window_journal
        self.assertIs(WRows.install_journal(pool), journal)        # once
        blocks = pool.get_new_blocks(2)
        pool.touch(iter(blocks[:1]))
        pool.free_blocks(reversed(blocks))
        pid = id(pool)
        self.assertEqual(journal.keys, [(pid, 1), (pid, 2), (pid, 1), (pid, 2), (pid, 1)])
        self.assertEqual([b.ref_cnt for b in blocks], [1, 0])
        self.assertEqual(pool._remote_ra_incarnations, {1: 1, 2: 1})


class FakePool:
    """The methods of vLLM 0.24's BlockPool the journal wraps, with its reference counts."""

    def __init__(self, n):
        self.blocks = [NS(block_id=i, ref_cnt=0, is_null=i == 0) for i in range(n)]
        self.free = list(self.blocks[1:])

    def get_new_blocks(self, num_blocks):
        out, self.free = self.free[:num_blocks], self.free[num_blocks:]
        for block in out:
            block.ref_cnt += 1
        return out

    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks):
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free.insert(0, block)


class JournalTests(unittest.TestCase):
    def test_marks_and_restarts(self):
        journal = WRows.Journal()
        live = {(1, 5), (1, 6)}
        mark = journal.mark()
        self.assertEqual(journal.touched(mark, live), set())
        journal.keys.extend([(1, 7), (2, 5)])
        self.assertEqual(journal.touched(mark, live), set())
        journal.keys.append((1, 6))
        self.assertEqual(journal.touched(mark, live), {(1, 6)})
        later = journal.mark()
        self.assertEqual(journal.touched(later, live), set())
        with patch.object(WRows.Journal, 'LIMIT', 2):
            restarted = journal.mark()
        self.assertEqual((journal.keys, restarted), ([], (1, 0)))
        self.assertIsNone(journal.touched(later, live))           # an older epoch names nothing
        self.assertEqual(journal.touched(restarted, live), set())


# ---- the engine: grow's journal path against its full path ---------------------------------------

P, WINDOW = GE.P, GE.WINDOW
SLIDING = GE.SLIDING


class EngineWorld:
    """vLLM's KV manager over test_gemma_e2e's Gemma-shaped config (one global group, five sliding
    groups in pool 1) and the engine's followed rows of each request, as EngineRpcBoundary keeps
    them under --decode-fastpath2: world A sends them through grow with --decode-fastpath3 (the
    window pool journaled), world B without."""

    def __init__(self, fast3, sessions=4, fa_blocks=600):
        import test_gemma_kv_pools as GK
        import test_kv_pools as KP
        from remote_ra_live_v2.pool_incarnations import install
        w = GK.gemma_world()
        self.layout = GE.window_layout('full', sessions)
        self.kv = KP.kv_manager(w, GE.kv_cache_config(w, self.layout, fa_blocks), hash_block_size=P,
                                scheduler_block_size=P)
        for pool in self.kv.block_pools:
            install(pool)
        self.window_pool = self.kv.block_pools[1]
        if fast3:
            WRows.install_journal(self.window_pool)
        self.fast3 = fast3
        self.groups = self.kv.coordinator.single_type_managers
        self.pools = tuple(self.kv.pool_of_group(g) for g in range(len(self.groups)))
        self.requests, self.records, self.freed = {}, {}, {}

    def call(self, fn, *args):
        try:
            return ('ok', fn(*args))
        except LiveV2Error as e:
            return ('error', str(e))

    def admit(self, rid, prompt, tokens):
        import test_kv_pools as KP
        from remote_ra_live_v2.pool_incarnations import rows
        req = KP.request(rid, tokens)
        self.requests[rid] = req
        assert self.kv._growing_original_allocate_slots(req, prompt, skip_cache=True, has_scheduled_reqs=True)
        req.num_computed_tokens += prompt
        native = rows(self.kv, rid, self.pools)
        fa = {0: list(native[0])}
        record = dict(native_tables={0: tuple(b for b, _ in native[0])},
                      incremental=dict(fa=fa, ids={b for b, _ in fa[0]}, groups=tuple(self.groups), pools=self.pools))
        record['incremental']['windows'] = WRows.engine_windows(self.groups, (0,), rid, self.pools)
        record['accepted_rows'] = native
        self.records[rid] = record

    def step(self, rid, count):
        req = self.requests[rid]
        if not self.kv._growing_original_allocate_slots(req, count, skip_cache=True, has_scheduled_reqs=True):
            return ('full',)
        req.num_computed_tokens += count
        record = self.records[rid]
        out = self.call(WRows.grow, record, record['incremental'], NS(request=rid), self.kv, self.pools,
                        record['accepted_rows'], self.fast3)
        if out[0] == 'ok':
            accepted = out[1] is record['accepted_rows']
            record['accepted_rows'] = out[1]
            return ('ok', 'accepted' if accepted else copy.deepcopy(out[1]))
        return out

    def finish(self, rid):
        # A block freed behind the request's back is referenced again first, so that the request's
        # own free leaves the pool consistent (another owner keeps it).
        self.window_pool.touch(self.freed.pop(rid, []))
        self.kv._growing_original_free(self.requests.pop(rid))
        self.records.pop(rid)

    def followed(self):
        """Every request's followed rows by block id (the pools' object ids differ per world)."""
        out = {}
        for rid, record in self.records.items():
            windows = record['incremental']['windows']
            out[rid] = (record['native_tables'], record['incremental']['fa'],
                        {g: (v[0], [b.block_id for b in v[1]], list(v[2])) for g, v in windows['groups'].items()},
                        sorted(b for _, b in windows['live']))
        return out

    def block(self, rid, g, index):
        return self.groups[g].req_to_blocks[rid][index]


class EngineDriver:
    """The same random engine steps on worlds A and B (requests decode and prefill in chunks, finish
    and are admitted; other requests' blocks come and go), with foreign events through the pool's
    own methods: a window block freed once more (its count reaches 0), freed and handed to another
    request (a new incarnation), touched and freed again; and a row entry inside a window replaced
    by another block object."""

    def __init__(self, test, seed):
        self.test, self.rng = test, random.Random(seed)
        self.worlds = (EngineWorld(True), EngineWorld(False))
        self.live, self.next = [], 0
        self.counts = dict(steps=0, refused=0, clean=0)
        self.events = dict(double_free=0, realloc=0, touch=0, replace=0)

    def both(self, fn):
        return [fn(w) for w in self.worlds]

    def compare(self, results):
        a, b = results
        self.test.assertEqual(a, b)
        self.test.assertEqual(self.worlds[0].followed(), self.worlds[1].followed())
        return a

    def admit(self):
        rid = 'q%d' % self.next
        self.next += 1
        prompt = self.rng.randrange(1, 6 * WINDOW)
        self.both(lambda w: w.admit(rid, prompt, prompt + 400))
        self.live.append(rid)

    def window_index(self, rid, last):
        """A window block of ``rid``: its last (no step of up to 40 tokens moves it out of the window,
        so the request's own allocation never frees it before the next check), or any."""
        w = self.worlds[0]
        g = self.rng.choice(SLIDING)
        first, blocks, _ = w.records[rid]['incremental']['windows']['groups'][g]
        if not blocks:
            return g, None
        return g, first + (len(blocks) - 1 if last else self.rng.randrange(len(blocks)))

    def foreign(self):
        rid = self.rng.choice(self.live)
        kind = self.rng.randrange(4)
        g, index = self.window_index(rid, kind < 3)
        if index is None:
            return

        def apply(w):
            block = w.block(rid, g, index)
            if block.block_id == 0 or block.ref_cnt != 1:
                return                                             # (each block of one owner, once)
            pool = w.window_pool
            if kind == 0:
                pool.free_blocks([block])                          # a double free: count 0
                w.freed.setdefault(rid, []).append(block)
            elif kind == 1:
                pool.free_blocks([block])
                w.freed.setdefault(rid, []).append(block)
                held = pool.get_new_blocks(1)                      # another owner, a new incarnation
                w.__dict__.setdefault('held', []).extend(held)
            elif kind == 2:
                pool.touch([block])                                # shared, then given back
                if self.rng_state % 2:
                    pool.free_blocks([block])
            else:
                other = pool.get_new_blocks(1)
                w.groups[g].req_to_blocks[rid][index] = other[0]   # the row names another block
                w.__dict__.setdefault('held', []).extend(other)
        self.rng_state = self.rng.randrange(2)
        self.both(apply)
        self.events[('double_free', 'realloc', 'touch', 'replace')[kind]] += 1

    def run(self, events):
        rng = self.rng
        with patch.object(WRows.Journal, 'LIMIT', 400), patch.object(WRows, "_unchanged", counted_unchanged(self)):
            for _ in range(events):
                if len(self.live) < 3 and (not self.live or rng.random() < .15):
                    self.admit()
                x = rng.random()
                if x < .85:
                    for rid in list(self.live):
                        r = self.worlds[0].requests[rid]
                        count = 1 if rng.random() < .8 else rng.choice((2, 7, 16, 40))
                        if r.num_computed_tokens + count > r.num_tokens:
                            count = 0
                        out = self.compare(self.both(lambda w: w.step(rid, count) if count else ('done',)))
                        self.counts['steps'] += 1
                        if out[0] != 'ok':
                            self.counts['refused'] += out[0] == 'error'
                            self.live.remove(rid)
                            self.both(lambda w: w.finish(rid))
                elif x < .93:
                    rid = rng.choice(self.live)
                    self.live.remove(rid)
                    self.both(lambda w: w.finish(rid))
                else:
                    self.foreign()
            for rid in list(self.live):
                self.both(lambda w: w.finish(rid))


def counted_unchanged(driver):
    unchanged = WRows._unchanged

    def counted(*args):
        result = unchanged(*args)
        driver.counts['clean'] += result is not None
        return result
    return counted


class EngineJournalTests(unittest.TestCase):
    def test_journal_path_equals_the_full_path(self):
        seeds = 16 if SLOW else 6
        counts = dict(steps=0, refused=0, clean=0)
        events = dict(double_free=0, realloc=0, touch=0, replace=0)
        for seed in range(seeds):
            d = EngineDriver(self, seed)
            d.run(220)
            counts = {k: v + d.counts[k] for k, v in counts.items()}
            events = {k: v + d.events[k] for k, v in events.items()}
        self.assertGreater(counts['clean'], 0.5 * counts['steps'], counts)
        self.assertGreater(counts['refused'], 2 * seeds // 3, counts)
        self.assertTrue(all(events.values()), events)


# ---- the worker owner: the steady observation against the other worlds ----------------------------

class SteadyDriver(WR.Driver):
    """test_gemma_window_rows' Driver with a fourth world D: --decode-fastpath2's rows (world C's
    deltas and rowless updates) and --decode-fastpath3's steady observation, materialize and
    verification. After every operation D must equal the full world B (and A and C)."""

    def __init__(self, seed):
        super().__init__(seed)
        d = WR.World(True, rows=True)
        d.owner.decode_fastpath3 = d.owner.window_steady = True
        self.steady = dict(observed=0, materialized=0)
        full, observe = d.owner.materialize, d.owner._observe_steady

        def materialize(s):
            if d.owner.materialize_steady(s):
                self.steady['materialized'] += 1
                return None
            return full(s)

        def observed(*args):
            s = observe(*args)
            self.steady['observed'] += s is not None
            return s
        d.owner.materialize, d.owner._observe_steady = materialize, observed
        self.worlds = self.worlds + (d,)

    def compare(self, test, results):
        b = results[1]
        for k, result in enumerate(results):
            test.assertEqual(result, b, k)
        state = self.worlds[1].state()
        for k, w in enumerate(self.worlds):
            test.assertEqual(w.state(), state, k)
        self.ops += 1


class SteadyWindowOwnerTests(unittest.TestCase):
    def test_steady_equals_full(self):
        seeds = 24 if SLOW else 8
        totals = dict(observed=0, materialized=0, ops=0)
        for seed in range(seeds):
            d = SteadyDriver(seed)
            d.run(self, 250)
            totals['observed'] += d.steady['observed']
            totals['materialized'] += d.steady['materialized']
            totals['ops'] += d.ops
        self.assertGreater(totals['observed'], 40 * seeds, totals)
        self.assertGreater(totals['materialized'], 30 * seeds, totals)


# ---- the window lifecycle: before_forward3 against before_forward --------------------------------

class WindowLeaseSequenceTests(unittest.TestCase):
    """before_forward3 (world A) and before_forward (world B) of WindowLifecycle with the first two
    fast paths over the same random lease sequences: windows that stay, move (a block leaves each
    group's front, one enters its end) or come as an equal new tuple, and corruptions (a slot of
    another kind or generation, an object that is no lease, a slot twice, a stale cursor or
    generation, moved positions, a foreign attention lease, a repeated callback, a terminal
    request): equal results or the same refusal, equal states and backend calls."""

    def world(self):
        from remote_ra_live_v2.window_lifecycle import WindowLifecycle
        calls = []
        backend = NS(window=True, verify_window_start=lambda *a: calls.append(('start',) + a))
        q = WindowLifecycle(backend, page_tokens=16)
        q.decode_fastpath = q.decode_fastpath2 = q.decode_fastpath3 = q.reuse_state_leases = True
        return q, calls

    @staticmethod
    def outcome(fn):
        try:
            lease = fn()
            return ('ok', lease, type(lease.hybrid).__name__)
        except LiveV2Error as e:
            return ('error', str(e))

    def test_random_sequences(self):
        rng = random.Random(11)
        counts = dict(fallback=0, calls=0, moved=0, same=0)
        groups = (1, 2, 3, 4, 5)
        for seq in range(80 if SLOW else 40):
            (a, ca), (b, cb) = self.world(), self.world()
            local = rng.random() < .5
            prompt = rng.randrange(40, 90)
            identity = NativeIdentity('r%d' % seq, 's', 1, 'g', 'a' * 64, 'b' * 64, 'm')
            if local:
                ra = RenderAssignment(identity, 1, 20, 16, 32, 33, 36, 36, prompt, 16, 'c' * 64, 'd' * 64)
            else:
                ra = RenderAssignment(identity, 0, 0, 0, 0, 5, 5, 0, prompt, 16, 'c' * 64, ZERO_DIGEST)
            gen = 3
            for w in (a, b):
                w.open(ra, tuple(range(prompt)), gen)
            fallback = a.before_forward

            def counted(*args, **kwargs):
                counts['fallback'] += 1
                return fallback(*args, **kwargs)
            a.before_forward = counted
            cursor, block = ra.phase2_start, 1000
            attn = tuple(Allocation(0, 100 + i, gen, 'attention') for i in range(3))
            good = tuple(Allocation(g, 10 * g + i, gen, 'sliding', 1) for g in groups for i in range(4))
            for step in range(60):
                x = rng.random()
                start, count = cursor, (1 if cursor >= prompt else min(prompt - cursor, rng.choice((1, 4, 16))))
                g = gen
                if x < .03:
                    start += 1                                        # stale cursor
                elif x < .05:
                    g = gen + 1                                       # stale generation
                if rng.random() < .1:
                    attn = attn + (Allocation(0, 200 + step, gen, 'decode' if cursor >= prompt else 'attention'),)
                y = rng.random()
                if y < .2:
                    # The window moves: each group's first slot leaves, a new one enters at its end.
                    block += 1
                    good = tuple(v for k in groups for v in
                                 [u for u in good if u.group == k][1:] + [Allocation(k, block + 50 * k, gen,
                                                                                     'sliding', 2)])
                    counts['moved'] += 1
                elif y < .25:
                    good = tuple(list(good))                          # an equal, new tuple
                else:
                    counts['same'] += 1                               # the same tuple object
                hybrid = good
                z = rng.random()
                if z < .03:
                    hybrid = good[:-1] + (Allocation(5, 999, gen, 'mamba'),)            # another kind
                elif z < .06:
                    hybrid = good[:2] + (Allocation(1, 998, gen + 1, 'sliding'),) + good[3:]  # another generation
                elif z < .08:
                    hybrid = good[:1] + ((1, 997),) + good[2:]       # not a lease
                elif z < .11:
                    hybrid = good + good[:1]                          # a slot twice
                if rng.random() < .03:
                    attn = attn + (Allocation(0, 900 + step, gen + 1, 'decode'),)   # a foreign lease
                logical = tuple(range(start, start + count))
                if rng.random() < .03:
                    logical = tuple(range(start + 1, start + count + 1))           # moved positions
                evidence = dict(request=identity.request, start=start, step=step)
                ra_ = self.outcome(lambda: a.before_forward3(identity, start, count, logical, evidence,
                                                             attn, hybrid, g))
                rb_ = self.outcome(lambda: b.before_forward(identity, start=start, count=count,
                                                            logical_positions=logical, attention_allocations=attn,
                                                            hybrid_allocations=hybrid, allocation_generation=g,
                                                            position_evidence=evidence))
                counts['calls'] += 1
                self.assertEqual(ra_, rb_, (seq, step))
                self.assertEqual(vars(a.requests[identity.request]), vars(b.requests[identity.request]))
                self.assertEqual(ca, cb)
                if ra_[0] == 'ok':
                    if rng.random() < .05:
                        continue                                      # a repeated callback next
                    for w, lease in ((a, ra_[1]), (b, rb_[1])):       # after_forward's state change
                        st = w.requests[identity.request]
                        st.latest_hybrid = lease.hybrid.slots
                        st.cursor, st.latest_state_step, st.active = start + count, start + count, None
                        st.prompt_complete = st.cursor >= prompt
                    cursor = start + count
                    if cursor >= prompt + 20:
                        break
                else:
                    if rng.random() < .5:
                        attn = tuple(v for v in attn if v.generation == gen)
                    if rng.random() < .5 and a.requests[identity.request].current_hybrid is not None:
                        current = a.requests[identity.request].current_hybrid
                        if all(type(v) is Allocation for v in current):
                            good = current
                    if rng.random() < .2:
                        for w in (a, b):
                            w.requests[identity.request].poisoned = True
            if rng.random() < .2:
                for w in (a, b):
                    w.requests[identity.request].terminal = True
                ra_ = self.outcome(lambda: a.before_forward3(identity, cursor, 1, (cursor,), None, attn, hybrid, gen))
                rb_ = self.outcome(lambda: b.before_forward(identity, start=cursor, count=1,
                                                            logical_positions=(cursor,), attention_allocations=attn,
                                                            hybrid_allocations=hybrid, allocation_generation=gen))
                self.assertEqual(ra_, rb_)
        self.assertGreater(counts['calls'] - counts['fallback'], 400, counts)
        self.assertGreater(counts['fallback'], 40, counts)
        self.assertGreater(counts['moved'], 100, counts)


class BackendVerifyTests(unittest.TestCase):
    """WindowWorkerBackend._verify with --decode-fastpath3 (world A) and without (world B) over the same
    random calls: the same tuple again, an equal new tuple, a list, a slot of another generation, an
    object without a generation, another generation, a request the backend does not hold: equal
    results or the same refusal or error, and the same owner calls."""

    def backend(self, fast3):
        from remote_ra_live_v2.window_lifecycle import WindowWorkerBackend
        calls = []

        def verify(identity, slots, generation):
            calls.append((identity.request, type(slots).__name__, list(slots), generation))
            if generation == 99:
                raise LiveV2Error('foreign/stale allocation lease')
        owner = NS(verify_allocations=verify, release_after_drain=lambda *a: (), verify_identity=lambda: None)
        backend = WindowWorkerBackend(NS(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS'), owner, source='s' * 64)
        backend.decode_fastpath3 = fast3
        return backend, calls

    @staticmethod
    def outcome(fn):
        try:
            return ('ok', fn())
        except (LiveV2Error, AttributeError) as e:
            return ('error', type(e).__name__, str(e))

    def test_random_calls(self):
        rng = random.Random(4)
        (a, ca), (b, cb) = self.backend(True), self.backend(False)
        identities = [NativeIdentity('v%d' % k, 's', 1, 'g', 'a' * 64, 'b' * 64, 'm') for k in range(3)]
        for w in (a, b):
            for identity in identities[:2]:
                w.requests[identity.request] = dict(identity=identity, generation=7)
        slots = {i.request: tuple(Allocation(g, 10 * g + k, 7, 'sliding', 1) for g in (1, 2) for k in range(5))
                 for i in identities}
        reused = 0
        for step in range(600):
            identity = rng.choice(identities)
            current = slots[identity.request]
            x, generation = rng.random(), 7
            if x < .5:
                given = current                                       # the same tuple again
                reused += 1
            elif x < .6:
                given = slots[identity.request] = tuple(list(current))  # an equal, new tuple
            elif x < .65:
                given = list(current)                                 # a list
            elif x < .7:
                given = current[:3] + (Allocation(2, 77, 8, 'sliding'),) + current[4:]
            elif x < .73:
                given = current[:2] + (NS(block=3),) + current[3:]    # no generation
            elif x < .78:
                given, generation = current, 8
            elif x < .8:
                given, generation = current, 99                       # the owner refuses
            elif x < .9:
                given = slots[identity.request] = tuple(Allocation(g, 10 * g + k + step, 7, 'sliding', 1)
                                                        for g in (1, 2) for k in range(5))
            else:
                for w in (a, b):                                      # drained, or begun again
                    if identity.request in w.requests:
                        del w.requests[identity.request]
                    else:
                        w.requests[identity.request] = dict(identity=identity, generation=7)
                continue
            self.assertEqual(self.outcome(lambda: a._verify(identity, given, generation)),
                             self.outcome(lambda: b._verify(identity, given, generation)), step)
            self.assertEqual(ca, cb)
        self.assertGreater(reused, 200)


# ---- end to end: --decode-fastpath3 against --decode-fastpath2 -----------------------------------

_RUNS = {}


def operations(h):
    """Record the name of every worker RPC of a harness in ``h.operations``."""
    h.operations, rpc = [], h.proxy.executor.collective_rpc

    def collective_rpc(method, args, timeout):
        h.operations.append(args[0])
        return rpc(method, args, timeout)
    h.proxy.executor = NS(collective_rpc=collective_rpc)
    return h


def counting(h, counts):
    """Count the fast-path-3 paths a harness takes (its owner, lifecycle and the rowless entries)."""
    observe, steady, lease, fallback = (h.owner._observe_steady, h.owner.materialize_steady,
                                        h.qwen.before_forward3, h.qwen.before_forward)

    def observed(*args):
        s = observe(*args)
        counts['observed'] += s is not None
        return s

    def materialized(s):
        done = steady(s)
        counts['materialized'] += done
        return done

    def leased(*args):
        calls = []
        h.qwen.before_forward = lambda *a, **k: calls.append(1) or fallback(*a, **k)
        try:
            return lease(*args)
        finally:
            h.qwen.before_forward = fallback
            counts['leased'] += not calls
    h.owner._observe_steady, h.owner.materialize_steady, h.qwen.before_forward3 = observed, materialized, leased


def script_one(run, rounds=5):
    """One session's turns (Page and forward chunks, decoding alone), then a rebuild."""
    h, infos = run.h, []
    lane = h.lanes[0]
    for k in range(rounds):
        observation = None if k == 0 else GE.OBSERVATIONS[(k - 1) % len(GE.OBSERVATIONS)]
        infos.append(h.turn(lane, observation, target=GE.TARGETS[k % len(GE.TARGETS)], eos=k % 3 == 2,
                            chunk='forward' if k % 2 else 'page'))
    retry = lane.messages[:-1] + [dict(role='user', content='Format error: expected exactly one action.')]
    infos.append(h.turn(lane, messages=retry, target=3))
    infos.append(h.turn(lane, GE.OBSERVATIONS[5], target=9))
    run.infos = infos
    return run


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    """test_gemma_fastpath's script over test_gemma_e2e's engine and worker with --decode-fastpath2
    and with --decode-fastpath3: everything compared, the batches included (their payloads do not
    change), and the per-step leases and claims."""
    maxDiff = None
    COMPARED = TF.EndToEndTests.COMPARED + ('extended',)
    setUp = TF.EndToEndTests.setUp

    def serve(self, policy, fast3, graphs=None, tasks=None, **script):
        """One deterministic run (shared by the tests): ``graphs`` the CUDA-graph record's mode (the
        runner forces probe passes eager), ``tasks`` (GE.A,) for one session."""
        key = (policy, fast3, graphs, None if tasks is None else tuple(t['instance_id'] for t in tasks),
               tuple(sorted(script.items())))
        if key not in _RUNS:
            flags = dict(ALL if fast3 else TWO, **({} if graphs is None else dict(cuda_graphs=graphs)))
            counts = dict(observed=0, materialized=0, leased=0, rowless=0, clean=0)

            def attach(h):
                if graphs is not None:
                    h.runner._remote_ra_cuda_graphs = dict(mode=graphs)
                if fast3:
                    attach_fast3(h)
                    counting(h, counts)
                else:
                    TF.attach_fast2(h)
            rowless, unchanged = F3.rowless_entry, WRows._unchanged

            def counted_rowless(*args):
                receipt = rowless(*args)
                counts['rowless'] += receipt is not None
                return receipt

            def counted_unchanged(*args):
                result = unchanged(*args)
                counts['clean'] += result is not None
                return result
            with ExitStack() as stack:
                stack.enter_context(patch.object(F3, 'rowless_entry', counted_rowless))
                stack.enter_context(patch.object(WRows, '_unchanged', counted_unchanged))
                if tasks is not None:
                    stack.enter_context(patch.object(TF, 'TASKS', tasks))
                with TF.Run(self, policy, flags, attach=attach) as run:
                    (script_one(run) if tasks is not None else run.script(**script))
                    _RUNS[key] = dict(run.result(), counts=counts)
        return _RUNS[key]

    def compare(self, policy, graphs=None, tasks=None, **script):
        two, three = (self.serve(policy, False, graphs, tasks, **script),
                      self.serve(policy, True, graphs, tasks, **script))
        for key in self.COMPARED:
            self.assertTrue(three[key] == two[key], key)
        counts = three['counts']
        for name in ('observed', 'materialized', 'leased', 'rowless', 'clean'):
            self.assertGreater(counts[name], 3, (name, counts))
        self.assertEqual(two['counts'], dict(observed=0, materialized=0, leased=0, rowless=0, clean=0))
        return three

    def test_fast3_equals_fast2(self):
        for policy in TF.POLICIES if SLOW else ('observation_query',):
            with self.subTest(policy=policy):
                three = self.compare(policy, **({} if SLOW else TF.SHORT))
                self.assertTrue(any(len(step) == 3 for step in three['batched']))
                self.assertTrue(any(n > 2 * GE.WINDOW for n in three['windows'].values()))

    def test_with_the_cuda_graph_record(self):
        # goal_query (its Goal forwards are probe passes) and full graphs; every policy with
        # REMOTE_RA_SLOW_TESTS=1. The turns are the eager run's.
        for policy in TF.POLICIES if SLOW else ('goal_query',):
            with self.subTest(policy=policy):
                script = {} if SLOW else TF.SHORT
                three = self.compare(policy, 'full', **script)
                eager = self.serve(policy, False, **script)
                for key in ('done', 'visible', 'windows', 'computed', 'plans', 'steps'):
                    self.assertEqual(three[key], eager[key], key)

    def test_one_session(self):
        three = self.compare('observation_query', tasks=(GE.A,))
        self.assertEqual({len(step) for step in three['batched']}, set())   # it decodes alone
        self.assertTrue(three['done'])


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ReleasePressureTests(unittest.TestCase):
    """test_gemma_release's pressure runs at the selective rule's minimum pool 0 (the release rule's
    for full) with --release-idle-pages: --decode-fastpath3 against --decode-fastpath2, the same
    turns, releases and readmissions. Sixteen sessions with REMOTE_RA_SLOW_TESTS=1; five in the
    archive suite."""
    maxDiff = None

    def setUp(self):
        import test_gemma_goal as GG
        import test_gemma_release as GR
        self.GG, self.GR = GG, GR
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.enterContext(patch.object(GE, 'GOAL', GG.SixteenGoalSessionsTests.TEST_GOAL))

    def check(self, h, info):
        return self.GR.SixteenSessionTests.check(self, h, info)

    def boundary(self, h):
        return self.GR.SixteenSessionTests.boundary(self, h)

    def window_of(self, h, sid):
        return self.GR.SixteenSessionTests.window_of(self, h, sid)

    def pressure(self, policy, sessions, turns, blocks=None):
        """``blocks``: pool 0's blocks (None: the rule's minimum, idle sessions released and readmitted)."""
        import test_gemma_full as GF
        GG, GR = self.GG, self.GR
        layout = GE.window_layout(policy, sessions)
        small = (blocks if blocks is not None else
                 GF.release_blocks(layout) if policy == 'full' else GR.selective_blocks(layout))
        counts = dict(observed=0, materialized=0, leased=0)

        def make(fast3):
            def build(*args):
                h = GR.Pressure(*args)
                if fast3:
                    attach_fast3(h)
                    counting(h, counts)
                else:
                    TF.attach_fast2(h)
                return operations(h)
            return build
        runs = [GG.pressure_run(self, policy, sessions, turns, small, make=make(fast3), extra=ALL if fast3 else TWO)
                for fast3 in (False, True)]
        (two, h2, events2, cleanup2), (three, h3, events3, cleanup3) = runs
        self.assertEqual(three, two)
        strip = lambda events: [{k: v for k, v in e.items() if k != 'monotonic_ns'} for e in events]
        self.assertEqual(strip(events3), strip(events2))
        if blocks is None:
            self.assertTrue(any(e['event'] == 'evict' for e in events3) and
                            any(e['event'] == 'readmit' for e in events3))
        self.assertEqual(cleanup3['idle_release'], cleanup2['idle_release'])
        self.assertEqual([i['native']['idle_release'] for i in h3.done], [i['native']['idle_release'] for i in h2.done])
        self.assertEqual(h3.operations, h2.operations)
        self.assertTrue(all(counts.values()), counts)
        self.assertGreater(max(h3.widths), 1)
        return h3

    def test_five_sessions(self):
        self.pressure('observation_query', 5, 3)

    @unittest.skipUnless(SLOW, 'about two minutes of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions(self):
        for policy in ('observation_query', 'goal_query', 'full'):
            with self.subTest(policy=policy):
                self.pressure(policy, self.GR.SESSIONS, 3)
        # A pool that never releases: the sixteen sessions decode together.
        h = self.pressure('observation_query', self.GR.SESSIONS, 3, blocks=self.GR.REFERENCE_BLOCKS)
        self.assertGreaterEqual(max(h.widths), 12)


if __name__ == '__main__':
    unittest.main()
