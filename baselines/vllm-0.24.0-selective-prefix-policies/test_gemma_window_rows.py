"""The window owner's incremental observation and rows == its full ones, on randomized sequences.

CPU only (numpy, no torch). The window layout's counterpart of test_decouple_wp3_owner: two
identical worlds, each a WindowAllocationOwner over a vLLM-shaped runner (numpy BlockTables
of the global group, request block-id lists that only grow, as vLLM's runner keeps them),
receive the same random sequence of engine steps: allocation_update with fresh payload lists
(as after IPC), the owner's step (observation and claims), materialize and
verify_allocations of the step's sliding leases. World A runs with --decode-fastpath
(``window_fast``: the incremental observation), world B without (the full observation).

EngineCore's sliding rows are modelled as vLLM's SlidingWindowManager keeps them: at each
step the blocks wholly before the step's window, [start - W + 1, ...), become null (and
free: recently freed ids come back under new incarnations, in the same step too) and new
blocks are appended for [.., start + count); the runner's rows only append. Steps are
prefill chunks (smaller and larger than the window) and decodes, in batches of several
requests, with row swaps, finishes and new admissions, and corruptions: runner rows
tampered inside and before the window or replaced by tuples, the FA list and device rows
tampered, a live entry before EngineCore's window, a changed incarnation, a duplicate and a
foreign block. After every operation both worlds must return equal results, raise the same
error or none, and hold equal claims, rows and allocation sets.

A third world (C) runs with --decode-fastpath2 as well (``window_rows``): after a request's
first (full) update it receives what window_rows.grow sends, a delta of the rows it accepted
last when the change is one (a row only grew, a window only lost blocks at its front), the
rows themselves (unchanged: no rows) otherwise; it keeps its sliding rows compact
(window_rows.WindowRow). It must equal the other two after every operation. (EngineCore never
sends full rows after a delta, and its own check covers the window and the entry before it:
vLLM writes nothing but nulls before a window. The full rows sent to world C when a
corruption makes the change something else check that the worker's full path still holds
after deltas.)
"""
import copy
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))
from remote_ra_live_v2 import allocation_owner, kv_layout
from remote_ra_live_v2.contract import NativeIdentity
from remote_ra_live_v2.sparse_mapping import RowLayout
from remote_ra_live_v2.window_owner import WindowAllocationOwner, attach_fastpath2
from remote_ra_live_v2.window_rows import WindowRow
from test_decouple_wp3_rows import Table

P, W = 16, 64
FA, SLIDING = (0,), (1, 2, 3)
LAYOUT = kv_layout.WindowLayout(P, W, 3, 256)
# Bookkeeping only the incremental world keeps (its observation, its verification cache).
FAST_ONLY = ('observed', 'rows_seen', 'verified_slots', 'allocations_epoch', 'window_ids')


def spec(name):
    return NS(kv_cache_spec=type(name, (), {})())


class World:
    ROWS, WIDTH = 6, 2048

    def __init__(self, fast, rows=False):
        self.fast, self.rows = fast, rows
        groups = len(FA) + len(SLIDING)
        self.runner = NS(requests={}, kv_cache_config=NS(
            kv_cache_groups=[spec('FullAttentionSpec' if g in FA else 'SlidingWindowSpec') for g in range(groups)],
            group_pool=tuple(0 if g in FA else 1 for g in range(groups))),
            input_batch=NS(req_ids=[], block_table=NS(block_tables={g: Table(self.ROWS, self.WIDTH, P, 1)
                                                                     for g in FA})))
        io = NS(views=[NS(group=g, kind='attention') for g in FA], arena=None)
        self.owner = WindowAllocationOwner(self.runner, io, dict(page_tokens=P, source_closure_verified=True),
                                           LAYOUT)
        self.owner.window_fast = fast
        if rows:
            self.owner.decode_fastpath2 = True
            attach_fastpath2(self.owner)
        self.runner._remote_ra_live_v2_allocation_owner = self.owner
        self.qwen = NS(requests={})

    def call(self, fn, *args):
        try:
            return ('ok', fn(*args))
        except Exception as e:
            return ('error', type(e).__name__, str(e))

    def state(self):
        recs = {rid: {k: (sorted(v) if isinstance(v, set) else v) for k, v in rec.items() if k not in FAST_ONLY}
                for rid, rec in self.owner.records.items()}
        return (dict(self.owner.claims), recs, list(self.runner.input_batch.req_ids))


class Driver:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.worlds = (World(True), World(False), World(True, rows=True))
        self.accepted = {}          # request -> the rows world C accepted last
        self.kinds = dict(full=0, delta=0, unchanged=0)
        self.free = {0: list(range(1, 400)), 1: list(range(1, 400))}   # pool -> free ids
        self.incarnation = {}
        self.requests = {}
        self.next = 0
        self.fast = self.ops = 0

    @staticmethod
    def pool(g):
        return 0 if g in FA else 1

    def alloc(self, g):
        free = self.free[self.pool(g)]
        b = free.pop(-1 - self.rng.randrange(min(len(free), 4)))  # recently freed ids come back
        key = (self.pool(g), b)
        self.incarnation[key] = self.incarnation.get(key, 0) + 1
        return b

    def release(self, g, b):
        """Free a block id unless a row still holds it (a corruption may have duplicated it)."""
        pool = self.pool(g)
        held = any(b in (o['fa'][h] if h in FA else o['sl'][h]) for o in self.requests.values()
                   for h in FA + SLIDING if self.pool(h) == pool)
        if b and not held and b not in self.free[pool]:
            self.free[pool].append(b)

    def both(self, fn):
        for w in self.worlds:
            fn(w)

    def compare(self, test, results):
        a, b, c = results
        test.assertEqual(a, b)
        test.assertEqual(c, b)
        state = self.worlds[1].state()
        test.assertEqual(self.worlds[0].state(), state)
        test.assertEqual(self.worlds[2].state(), state)
        self.ops += 1

    @staticmethod
    def one_window(row):
        first = next((i for i, v in enumerate(row) if v[0]), len(row))
        return first if first < len(row) and all(v[0] for v in row[first:]) else None

    def delta(self, prev, new):
        """window_rows.grow's form of the change from ``prev`` to ``new``: None (unchanged),
        {'window': [...]} when the change is one, else False (send the rows themselves)."""
        items = []
        for g, (old, row) in enumerate(zip(prev, new)):
            if old == row:
                items.append(None)
                continue
            if len(row) < len(old) or any(not v[0] for v in row[len(old):]):
                return False
            if g in FA:
                if row[:len(old)] != old:
                    return False
                items.append([0, len(row), row[len(old):]])
                continue
            first0, first = self.one_window(old), self.one_window(row)
            if first0 is None or first is None or not first0 <= first <= len(old) or \
                    row[first:len(old)] != old[first:]:
                return False
            items.append([first, len(row), row[len(old):]])
        return None if all(i is None for i in items) else dict(window=items)

    def body(self, w, rid, rows):
        """The allocation_update rows world ``w`` receives."""
        if not w.rows or rid not in self.accepted:
            return rows
        change = self.delta(self.accepted[rid], rows)
        if change is False:
            self.kinds['full'] += 1
            return rows
        self.kinds['unchanged' if change is None else 'delta'] += 1
        return change

    # ---- engine events --------------------------------------------------
    def admit(self):
        rng = self.rng
        rid = 'r%d' % self.next
        self.next += 1
        prompt = rng.randrange(20, 400)
        start = rng.randrange(0, prompt)
        n_fixed = rng.randrange(0, 4)
        fa_rows = {g: [self.alloc(g) for _ in range(n_fixed + 1 + start // P)] for g in FA}
        fixed = {g: tuple(fa_rows[g][:n_fixed]) for g in FA}
        layout = RowLayout(P, n_fixed * P, tuple(range(n_fixed)), fixed, {g: () for g in FA})
        head = max(0, start - W + 1) // P
        sliding = {g: [0] * head + [self.alloc(g) for _ in range(head, max(head + 1, -(-start // P)))]
                   for g in SLIDING}
        r = dict(rid=rid, prompt=prompt, cursor=start, fa=fa_rows, sl=sliding, layout=layout,
                 generation=self.next, identity=NativeIdentity(rid, 's', self.next, 'g', 'a' * 64, 'b' * 64, 'm'),
                 stale=None, prefix=None)
        self.requests[rid] = r

        def apply(w):
            batch = w.runner.input_batch.req_ids
            row = len(batch)
            batch.append(rid)
            w.runner.requests[rid] = NS(block_ids=[list(fa_rows[g]) if g in FA else list(sliding[g])
                                                   for g in range(len(FA) + len(SLIDING))], num_computed_tokens=start)
            for g in FA:
                w.runner.input_batch.block_table.block_tables[g].add_row(fa_rows[g], row)
            w.owner.records[rid] = dict(identity=r['identity'], generation=r['generation'], layout=layout,
                                        zeroed=set(), allocations=set(), attention_ready=True,
                                        assignment=NS(prompt_tokens=prompt, local_turn=0))
            w.qwen.requests[rid] = NS(assignment=NS(identity=r['identity'], phase2_start=start, prompt_tokens=prompt),
                                      allocation_generation=r['generation'], poisoned=False, active=None, cursor=start)
        self.both(apply)
        return rid

    def finish(self, test, rid):
        r = self.requests.pop(rid)
        results = [w.call(w.owner.release_after_drain, r['identity'], r['generation'], 'complete')
                   for w in self.worlds]
        self.compare(test, results)

        def apply(w):
            batch = w.runner.input_batch.req_ids
            row, last = batch.index(rid), len(batch) - 1
            for t in w.runner.input_batch.block_table.block_tables.values():
                if row != last:
                    t.move_row(last, row)
                t.clear_row(last)
            batch[row] = batch[last]
            batch.pop()
            del w.runner.requests[rid], w.owner.records[rid]
        self.both(apply)
        live = {(self.pool(g), b) for o in self.requests.values() for g in FA for b in o['fa'][g]} | \
            {(self.pool(g), b) for o in self.requests.values() for g in SLIDING for b in o['sl'][g] if b}
        for g in FA + SLIDING:
            for b in sorted(set(r['fa'][g] if g in FA else r['sl'][g]) - {0}):
                if (self.pool(g), b) not in live:
                    self.release(g, b)

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
        kind, pick = rng.randrange(9), rng.random()
        g = rng.choice(SLIDING)
        live = [i for i, b in enumerate(r['sl'][g]) if b]
        others = [b for o in self.requests.values() if o is not r for b in o['sl'][g] if b]

        def apply(w):
            req, row = w.runner.requests[rid], w.runner.input_batch.req_ids.index(rid)
            ids = req.block_ids[g]
            if type(ids) is not list:
                return
            if kind == 0 and live:                              # runner window entry changed
                ids[live[int(pick * len(live))]] = 999
            elif kind == 1 and live and live[0]:                # runner entry before the window
                ids[int(pick * live[0])] = 998
            elif kind == 2:                                    # runner row replaced by a tuple
                req.block_ids[g] = tuple(ids)
            elif kind == 3:                                    # runner row grows alone
                ids.append(997)
            elif kind == 4:                                    # FA list prefix reordered
                fa = req.block_ids[FA[0]]
                if len(fa) > 1:
                    i = int(pick * (len(fa) - 1))
                    fa[i], fa[i + 1] = fa[i + 1], fa[i]
            elif kind == 5:                                    # FA device prefix tampered
                t = w.runner.input_batch.block_table.block_tables[FA[0]]
                if t.num_blocks_per_row[row]:
                    t.block_table.np[row, int(pick * t.num_blocks_per_row[row])] += 1000
        self.both(apply)
        if kind == 6 and live:                                 # engine reports a new incarnation
            r['stale'] = (g, live[int(pick * len(live))])
        elif kind == 7 and live and live[0]:                   # a live entry before EngineCore's window
            r['prefix'] = (g, int(pick * live[0]))
        elif kind == 8 and (others or live):                   # a duplicate or foreign block appended
            b = others[0] if others and pick < .5 else r['sl'][g][live[0]]
            r['sl'][g].append(b)

            def grow(w):
                ids = w.runner.requests[rid].block_ids[g]
                if type(ids) is list:
                    ids.append(b)
            self.both(grow)

    def payload(self, r):
        rows = []
        for g in FA + SLIDING:
            if g in FA:
                row = [[b, self.incarnation[(0, b)]] for b in r['fa'][g]]
            else:
                row = [[b, self.incarnation[(1, b)]] if b else [0, 0] for b in r['sl'][g]]
                if r['stale'] is not None and r['stale'][0] == g and r['stale'][1] < len(row) and row[r['stale'][1]][0]:
                    row[r['stale'][1]][1] += 7
                if r['prefix'] is not None and r['prefix'][0] == g and r['prefix'][1] < len(row):
                    row[r['prefix'][1]] = [990, 1]
            rows.append(row)
        return rows

    def engine_step(self, test, scheduled):
        rng = self.rng
        plan = []
        for rid in scheduled:
            r = self.requests[rid]
            start = r['cursor']
            count = 1 if start >= r['prompt'] else min(r['prompt'] - start, rng.choice((1, 5, 16, 40, 90)))
            end = start + count
            grow = {g: [] for g in FA + SLIDING}
            for g in FA:
                need = len(r['layout'].prefix_blocks[g]) + 1 + (end - 1) // P - len(r['fa'][g])
                for _ in range(max(0, need)):
                    grow[g].append(self.alloc(g))
                r['fa'][g].extend(grow[g])
            # SlidingWindowManager: free the blocks before the step's window, then allocate.
            skip = max(0, start - W + 1) // P
            for g in SLIDING:
                row = r['sl'][g]
                for i in range(min(skip, len(row))):
                    if row[i]:
                        b, row[i] = row[i], 0
                        self.release(g, b)
            for g in SLIDING:
                row = r['sl'][g]
                while len(row) < -(-end // P):
                    b = self.alloc(g)
                    row.append(b)
                    grow[g].append(b)
            plan.append((rid, start, count))

            def apply(w, rid=rid, grow=grow):
                req, row = w.runner.requests[rid], w.runner.input_batch.req_ids.index(rid)
                for g in FA + SLIDING:
                    if type(req.block_ids[g]) is list:
                        req.block_ids[g].extend(grow[g])           # vLLM: new block ids appended
                for g in FA:
                    w.runner.input_batch.block_table.block_tables[g].append_row(grow[g], row)
            self.both(apply)
        failed = set()
        for rid, start, count in plan:
            r = self.requests[rid]
            results = []
            rows = self.payload(r)
            for w in self.worlds:
                body = dict(rows=copy.deepcopy(self.body(w, rid, rows)), start=start, count=count, binding='b')
                rec = w.owner.records[rid]
                results.append(w.call(lambda: w.owner._allocation_update(rec, body)(None, w.qwen)))
            self.compare(test, results)
            if results[2][0] == 'ok':
                self.accepted[rid] = rows
                compact = [type(v) is WindowRow for v in self.worlds[2].owner.records[rid]['native_rows']]
                test.assertEqual(compact[len(FA):], [self.one_window(rows[g]) is not None for g in SLIDING])
            if results[0][0] != 'ok':
                failed.add(rid)
        results = [w.call(w.owner.step, w.runner, NS(num_scheduled_tokens={rid: c for rid, _, c in plan}))
                   for w in self.worlds]
        self.compare(test, results)
        if results[0][0] != 'ok':
            failed.update(scheduled)
        else:
            o = self.worlds[0].owner.records[scheduled[0]].get('observed')
            if o is not None and o.base is not None:
                self.fast += 1
            for i, s in enumerate(results[0][1]):
                outs = []
                for w, res in zip(self.worlds, results):
                    step = res[1][i]
                    out = w.call(w.owner.materialize, step)
                    rec = w.owner.records[step.request]
                    verified = [w.call(w.owner.verify_allocations, rec['identity'], step.hybrid, rec['generation'])
                                for _ in range(2)]
                    outs.append((out, verified))
                self.compare(test, outs)
                if outs[0][0][0] != 'ok' or outs[0][1][0][0] != 'ok':
                    failed.add(s.request)
        for rid, start, count in plan:
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
            if x < 0.82:
                for rid in self.engine_step(test, list(live)):
                    live.remove(rid)
                    self.finish(test, rid)
            elif x < 0.86:
                self.swap_rows()
            elif x < 0.92:
                rid = rng.choice(live)
                live.remove(rid)
                self.finish(test, rid)
            else:
                self.corrupt(rng.choice(live))
        for rid in list(live):
            self.finish(test, rid)


class IncrementalWindowOwnerTests(unittest.TestCase):
    def test_incremental_equals_full(self):
        # 24 seeds with REMOTE_RA_SLOW_TESTS=1 (about 9 s), 8 in the archive suite.
        seeds = 24 if os.environ.get('REMOTE_RA_SLOW_TESTS') == '1' else 8
        fast = ops = 0
        kinds = dict(full=0, delta=0, unchanged=0)
        for seed in range(seeds):
            d = Driver(seed)
            d.run(self, 250)
            fast, ops = fast + d.fast, ops + d.ops
            kinds = {k: v + d.kinds[k] for k, v in kinds.items()}
        self.assertGreater(fast, 60 * seeds, (fast, ops))
        self.assertGreater(kinds['delta'], 40 * seeds, kinds)
        self.assertGreater(kinds['unchanged'], 10 * seeds, kinds)
        self.assertGreater(kinds['full'], 0, kinds)

    def test_incremental_off_observes_in_full(self):
        with patch.object(allocation_owner, 'INCREMENTAL', False):
            d = Driver(3)
            d.run(self, 120)
        self.assertEqual(d.fast, 0)


if __name__ == '__main__':
    unittest.main()
