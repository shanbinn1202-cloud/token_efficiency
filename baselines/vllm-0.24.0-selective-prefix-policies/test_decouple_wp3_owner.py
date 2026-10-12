"""Incremental per-step ownership == the full per-step path, on randomized sequences.

CPU only (numpy, no torch). Two identical worlds, each a RunnerAllocationOwner
over a vLLM-shaped runner (FA BlockTables, request block-id lists, Mamba slot
index), receive the same random sequence of engine steps: allocation_update
with fresh payload lists (as after IPC), the owner's step (observation +
claims) and materialize, and verify_allocations of the Mamba slots. The
sequence has FA Page crossings, Mamba slot crossings (two live slots for one
step, then the old one freed), the prompt -> decode kind change, batches of
several requests, row moves, finishes with block reuse under new
incarnations, new admissions, and corruptions (list/device-row tampering,
changed incarnations, duplicate and foreign ids, malformed payload entries).
World A runs with ``INCREMENTAL = True``, world B with False (the original
code). After every operation both must return equal results, raise the same
error or none, and hold equal claims, rows and allocation sets. Layouts:
legacy (one pool), decoupled (FA pool 0, Mamba pool 1, coinciding ids) and
decoupled with the Mamba groups listed first.
"""
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))
from remote_ra_live_v2 import allocation_owner, kv_layout
from remote_ra_live_v2.contract import NativeIdentity
from remote_ra_live_v2.sparse_mapping import RowLayout
from test_decouple_wp3_rows import Table

P, M = 16, 64


class World:
    ROWS, WIDTH = 6, 2048

    def __init__(self, fa, mamba, layout, incremental, verified):
        self.fa, self.mamba, self.incremental = fa, mamba, incremental
        groups = len(fa) + len(mamba)
        pool = tuple(0 if g in fa else 1 for g in range(groups))
        self.runner = NS(requests={}, mamba_state_idx={}, kv_cache_config=NS(
            group_pool=pool if layout is not None else None), input_batch=NS(
            req_ids=[], block_table=NS(block_tables={g: Table(self.ROWS, self.WIDTH, P, 1) for g in fa})))
        views = [NS(group=g, kind='attention' if g in fa else 'hybrid') for g in range(groups)]
        self.zeros = []
        io = NS(views=views, zero=lambda groups, kind: self.zeros.append((kind, sorted(groups.items()))))
        self.owner = allocation_owner.RunnerAllocationOwner(
            self.runner, io, dict(page_tokens=P, source_closure_verified=verified), layout)
        self.runner._remote_ra_live_v2_allocation_owner = self.owner
        self.qwen = NS(requests={})

    def call(self, fn, *args):
        with patch.object(allocation_owner, 'INCREMENTAL', self.incremental):
            try:
                return ('ok', fn(*args))
            except Exception as e:
                return ('error', type(e).__name__, str(e))

    def state(self):
        recs = {rid: {k: (sorted(v) if isinstance(v, set) else v) for k, v in rec.items()
                      if k not in ('observed', 'rows_seen')}
                for rid, rec in self.owner.records.items()}
        return (dict(self.owner.claims), recs, list(self.runner.input_batch.req_ids))


class Driver:
    def __init__(self, seed, fa, mamba, layout, verified):
        self.rng = random.Random(seed)
        self.fa, self.mamba, self.layout = fa, mamba, layout
        self.worlds = (World(fa, mamba, layout, True, verified), World(fa, mamba, layout, False, verified))
        self.free = {0: list(range(1, 400)), 1: list(range(1, 400))}   # pool -> free ids
        self.incarnation = {}
        self.requests = {}
        self.next = 0
        self.fast = self.ops = 0

    def pool(self, g):
        return 0 if self.layout is None or g in self.fa else 1

    def alloc(self, g):
        free = self.free[self.pool(g)]
        b = free.pop(-1 - self.rng.randrange(min(len(free), 4)))  # recently freed ids come back
        key = (self.pool(g), b)
        self.incarnation[key] = self.incarnation.get(key, 0) + 1
        return b

    def both(self, fn):
        for w in self.worlds:
            fn(w)

    def compare(self, test, results):
        a, b = results
        test.assertEqual(a, b)
        test.assertEqual(self.worlds[0].state(), self.worlds[1].state())
        self.ops += 1

    # ---- engine events --------------------------------------------------
    def admit(self):
        rng = self.rng
        rid = 'r%d' % self.next; self.next += 1
        prompt = rng.randrange(20, 120)
        start = rng.randrange(0, prompt)
        n_fixed = rng.randrange(0, 4)
        fa_rows = {g: [self.alloc(g) for _ in range(n_fixed + 1 + start // P)] for g in self.fa}
        fixed = {g: tuple(fa_rows[g][:n_fixed]) for g in self.fa}
        layout = RowLayout(P, n_fixed * P, tuple(range(n_fixed)), fixed, {g: () for g in self.fa})
        idx = start // M
        mamba_rows = {g: [0] * idx + [self.alloc(g)] for g in self.mamba}
        r = dict(rid=rid, prompt=prompt, cursor=start, start=start, fa=fa_rows, mamba=mamba_rows,
                 idx=idx, layout=layout, generation=self.next, identity=NativeIdentity(
                     rid, 's', self.next, 'g', 'a' * 64, 'b' * 64, 'm'), stale=None)
        self.requests[rid] = r
        width = len(self.fa) + len(self.mamba)

        def apply(w):
            batch = w.runner.input_batch.req_ids
            row = len(batch); batch.append(rid)
            w.runner.requests[rid] = NS(block_ids=[list(fa_rows[g]) if g in self.fa else list(mamba_rows[g])
                                                   for g in range(width)], num_computed_tokens=start)
            w.runner.mamba_state_idx[rid] = idx
            for g in self.fa:
                w.runner.input_batch.block_table.block_tables[g].add_row(fa_rows[g], row)
            w.owner.records[rid] = dict(identity=r['identity'], generation=r['generation'], layout=layout,
                                        zeroed=set(), allocations=set(), attention_ready=True,
                                        assignment=NS(prompt_tokens=prompt, local_turn=1))
            w.qwen.requests[rid] = NS(assignment=NS(identity=r['identity'], phase2_start=start,
                                                    prompt_tokens=prompt),
                                      allocation_generation=r['generation'], poisoned=False, active=None,
                                      cursor=start)
        self.both(apply)
        return rid

    def finish(self, test, rid):
        r = self.requests.pop(rid)
        results = []
        for w in self.worlds:
            results.append(w.call(w.owner.release_after_drain, r['identity'], r['generation'], 'complete'))
        self.compare(test, results)

        def apply(w):
            batch = w.runner.input_batch.req_ids
            row = batch.index(rid); last = len(batch) - 1
            tables = w.runner.input_batch.block_table.block_tables.values()
            for t in tables:
                if row != last:
                    t.move_row(last, row)
                t.clear_row(last)
            batch[row] = batch[last]; batch.pop()
            del w.runner.requests[rid], w.runner.mamba_state_idx[rid], w.owner.records[rid]
        self.both(apply)
        live = {(self.pool(g), b) for o in self.requests.values() for g in self.fa for b in o['fa'][g]}
        for g in self.fa + self.mamba:
            free = self.free[self.pool(g)]
            for b in sorted(set(r['fa'][g] if g in self.fa else r['mamba'][g]) - {0}):
                if (self.pool(g), b) not in live and b not in free:
                    free.append(b)

    def swap_rows(self):
        def apply(w):
            batch = w.runner.input_batch.req_ids
            if len(batch) > 1:
                i, j = 0, len(batch) - 1
                for t in w.runner.input_batch.block_table.block_tables.values():
                    t.swap_row(i, j)
                batch[i], batch[j] = batch[j], batch[i]
        self.both(apply)

    def corrupt(self, rid):
        rng, r = self.rng, self.requests[rid]
        kind, g = rng.randrange(8), rng.choice(self.fa)
        others = [o for o in self.requests.values() if o is not r and o['fa'][g]]
        foreign = others[0]['fa'][g][-1] if kind == 7 and others else None
        pick = rng.random()
        def apply(w):
            req, row = w.runner.requests[rid], w.runner.input_batch.req_ids.index(rid)
            t = w.runner.input_batch.block_table.block_tables[g]
            ids = req.block_ids[g]
            if type(ids) is not list and kind in (0, 2, 5, 7):
                return
            if kind == 0 and len(ids) > 1:                      # list prefix reordered
                i = int(pick * (len(ids) - 1)); ids[i], ids[i + 1] = ids[i + 1], ids[i]
            elif kind == 1 and t.num_blocks_per_row[row]:       # device prefix tampered
                t.block_table.np[row, int(pick * t.num_blocks_per_row[row])] += 1000
            elif kind == 2:                                    # duplicate id appended everywhere
                ids.append(ids[0]); t.append_row([ids[0]], row)
            elif kind == 3:                                    # list replaced by a tuple
                req.block_ids[g] = tuple(ids)
            elif kind == 4:                                    # Mamba index past its row
                w.runner.mamba_state_idx[rid] += 3
            elif kind == 5:                                    # list grows alone
                ids.append(999)
            elif kind == 7 and foreign is not None:            # another live request's block
                ids.append(foreign); t.append_row([foreign], row)
        self.both(apply)
        if kind == 2 and type(self.worlds[0].runner.requests[rid].block_ids[g]) is list:
            r['fa'][g].append(r['fa'][g][0])
        if kind == 7 and foreign is not None and type(self.worlds[0].runner.requests[rid].block_ids[g]) is list:
            r['fa'][g].append(foreign)
        if kind == 6:                                          # engine reports a new incarnation
            r['stale'] = (g, int(pick * len(r['fa'][g])))

    def payload(self, r):
        rows = []
        for g in range(len(self.fa) + len(self.mamba)):
            if g in self.fa:
                row = [[b, self.incarnation[(self.pool(g), b)]] for b in r['fa'][g]]
                if r['stale'] is not None and r['stale'][0] == g and r['stale'][1] < len(row):
                    row[r['stale'][1]][1] += 7
            else:
                row = [[b, self.incarnation[(self.pool(g), b)] if b else 0] for b in r['mamba'][g]]
            rows.append(row)
        return rows

    def engine_step(self, test, scheduled):
        rng = self.rng
        plan = []
        for rid in scheduled:
            r = self.requests[rid]
            start = r['cursor']
            count = 1 if start >= r['prompt'] else min(r['prompt'] - start, rng.choice((1, 5, 16, 40)))
            end = start + count
            # FA Page crossing: the physical row covers [0, end).
            grow = {g: [] for g in self.fa}
            for g in self.fa:
                need = len(r['layout'].prefix_blocks[g]) + 1 + (end - 1) // P - len(r['fa'][g])
                for _ in range(max(0, need) + (1 if rng.random() < 0.05 else 0)):
                    grow[g].append(self.alloc(g))
                r['fa'][g].extend(grow[g])
            # Mamba crossing: the old slot lives one more step, then is freed.
            idx = (end - 1) // M
            freed = []
            for g in self.mamba:
                row = r['mamba'][g]
                for i, b in enumerate(row):
                    if b and i < r['idx']:
                        freed.append((g, b)); row[i] = 0
                if idx > r['idx']:
                    row.extend([0] * (idx + 1 - len(row))); row[idx] = self.alloc(g)
            old_idx, r['idx'] = r['idx'], max(r['idx'], idx)
            for g, b in freed:
                self.free[self.pool(g)].append(b)
            plan.append((rid, start, count, grow))

            def apply(w, rid=rid, grow=grow):
                req, row = w.runner.requests[rid], w.runner.input_batch.req_ids.index(rid)
                for g in self.fa:
                    if type(req.block_ids[g]) is list:
                        req.block_ids[g].extend(grow[g])
                    w.runner.input_batch.block_table.block_tables[g].append_row(grow[g], row)
                for g in self.mamba:
                    req.block_ids[g] = list(r['mamba'][g])
                w.runner.mamba_state_idx[rid] = r['idx']
            self.both(apply)
        # allocation_update per request (fresh lists per world, as after IPC).
        failed = set()
        for rid, start, count, _ in plan:
            r = self.requests[rid]
            results = []
            for w in self.worlds:
                body = dict(rows=copy.deepcopy(self.payload(r)), start=start, count=count, binding='b')
                rec = w.owner.records[rid]
                results.append(w.call(lambda: w.owner._allocation_update(rec, body)(None, w.qwen)))
            self.compare(test, results)
            if results[0][0] != 'ok':
                failed.add(rid)
        # The owner's step over the batch, then materialize and verify each.
        def scheduled_tokens(w):
            return {rid: count for rid, _, count, _ in plan}
        results = [w.call(w.owner.step, w.runner, NS(num_scheduled_tokens=scheduled_tokens(w)))
                   for w in self.worlds]
        self.compare(test, results)
        if results[0][0] != 'ok':
            failed.update(scheduled)
        if results[0][0] == 'ok':
            for w in self.worlds:
                o = w.owner.records[scheduled[0]].get('observed')
                if w.incremental and o is not None and o.base is not None:
                    self.fast += 1
            for i, s in enumerate(results[0][1]):
                outs = []
                for w, res in zip(self.worlds, results):
                    step = res[1][i]
                    out = w.call(w.owner.materialize, step)
                    rec = w.owner.records[step.request]
                    outs.append((out, w.call(w.owner.verify_allocations, rec['identity'], step.hybrid,
                                             rec['generation'])))
                self.compare(test, outs)
                if outs[0][0][0] != 'ok' or outs[0][1][0] != 'ok':
                    failed.add(s.request)
        # Advance cursors (both worlds and the driver).
        for rid, start, count, _ in plan:
            self.requests[rid]['cursor'] = start + count
            def apply(w, rid=rid, end=start + count):
                w.runner.requests[rid].num_computed_tokens = end
                w.qwen.requests[rid].cursor = end
            self.both(apply)
        return failed

    def run(self, test, events):
        live = []
        for _ in range(events):
            rng = self.rng
            if len(live) < 3 and (not live or rng.random() < 0.1):
                live.append(self.admit())
            x = rng.random()
            if x < 0.80:
                # The persistent batch holds exactly the scheduled requests: all live ones.
                for rid in self.engine_step(test, list(live)):
                    live.remove(rid); self.finish(test, rid)
            elif x < 0.85:
                self.swap_rows()
            elif x < 0.92:
                rid = rng.choice(live); live.remove(rid); self.finish(test, rid)
            else:
                self.corrupt(rng.choice(live))
        for rid in list(live):
            self.finish(test, rid)


class IncrementalOwnerEquivalenceTests(unittest.TestCase):
    def drive(self, fa, mamba, layout, seeds, verified=True, events=250):
        fast = ops = 0
        for seed in seeds:
            d = Driver(seed, fa, mamba, layout, verified)
            d.run(self, events)
            fast, ops = fast + d.fast, ops + d.ops
        return fast, ops

    def test_legacy_single_pool(self):
        fast, ops = self.drive((0,), (1, 2, 3), None, range(20))
        self.assertGreater(fast, 500, (fast, ops))

    def test_decoupled_two_pools(self):
        fast, ops = self.drive((0,), (1, 2, 3), kv_layout.for_server(P, 3), range(20))
        self.assertGreater(fast, 500, (fast, ops))

    def test_decoupled_mamba_groups_first(self):
        fast, ops = self.drive((3,), (0, 1, 2), kv_layout.for_server(P, 3), range(12))
        self.assertGreater(fast, 300, (fast, ops))

    def test_fixtures_without_incarnation_rows(self):
        # source_closure_verified False: incarnation 0, native_rows still sent here.
        fast, ops = self.drive((0,), (1, 2, 3), None, range(6), verified=False)
        self.assertGreater(fast, 100, (fast, ops))


if __name__ == '__main__':
    unittest.main()
