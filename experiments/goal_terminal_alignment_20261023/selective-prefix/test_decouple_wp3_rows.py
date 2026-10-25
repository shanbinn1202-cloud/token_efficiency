"""Incremental FA row install (H1) == the full recomposition, on randomized sequences.

CPU only, numpy, no torch. Two identical worlds (a vLLM-shaped BlockTable per FA
group, the runner's request block-id lists, the owner records) receive the same
random sequence of native events: steps, Page crossings (native appends),
row moves and swaps (condense), re-adds from ``block_ids`` (restores after a
probe dropped the row), list replacement, releases and new admissions with new
layouts, plus corruptions (pinned/duplicate/null ids appended, device-row and
list-prefix tampering, count drift, list/device desync). World A runs
``install_row`` incrementally, world B with ``INCREMENTAL = False`` (the full
recomposition, i.e. the original code). After every event both worlds must
hold identical device rows, counts, id lists, sparse plans, and raise the same
error or none.
"""
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))
from remote_ra_live_v2 import worker_rows
from remote_ra_live_v2.sparse_mapping import RowLayout


class Table:
    """vLLM 0.24 BlockTable row semantics (block_table.py: append/add/clear/move/swap)."""
    def __init__(self, rows, width, kernel, ratio):
        self.block_size, self.blocks_per_kv_block = kernel, ratio
        self.block_table = NS(np=np.zeros((rows, width), dtype=np.int32))
        self.num_blocks_per_row = np.zeros(rows, dtype=np.int32)

    def append_row(self, ids, row):
        if not ids:
            return
        r = self.blocks_per_kv_block
        kernel = [b * r + i for b in ids for i in range(r)]
        start = self.num_blocks_per_row[row]
        self.num_blocks_per_row[row] += len(kernel)
        self.block_table.np[row, start:start + len(kernel)] = kernel

    def add_row(self, ids, row):
        self.num_blocks_per_row[row] = 0
        self.append_row(ids, row)

    def clear_row(self, row):
        n = self.num_blocks_per_row[row]
        if n > 0:
            self.block_table.np[row, :n] = 0
        self.num_blocks_per_row[row] = 0

    def move_row(self, src, tgt):
        n = self.num_blocks_per_row[src]
        self.block_table.np[tgt, :n] = self.block_table.np[src, :n]
        self.num_blocks_per_row[tgt] = n

    def swap_row(self, src, tgt):
        self.num_blocks_per_row[[src, tgt]] = self.num_blocks_per_row[[tgt, src]]
        self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]


class World:
    ROWS, WIDTH = 6, 4096

    def __init__(self, groups, kernel, ratio, incremental):
        self.groups, self.incremental = groups, incremental
        self.tables = {g: Table(self.ROWS, self.WIDTH, kernel, ratio) for g in groups}
        self.runner = NS(uses_mrope=False, requests={}, _selective_load_plans={},
                         input_batch=NS(block_table=NS(block_tables=self.tables)))
        self.owner = NS(records={}, attention_groups=groups)
        self.rows = {}

    def free_row(self):
        return min(set(range(self.ROWS)) - set(self.rows.values()))

    def ids(self, rid, g):
        return self.runner.requests[rid].block_ids[g]

    def install(self, rid):
        with patch.object(worker_rows, 'INCREMENTAL', self.incremental):
            try:
                worker_rows.install_row(self.runner, self.owner, rid, self.rows[rid])
                return None
            except Exception as e:
                return (type(e).__name__, str(e))

    def state(self):
        rec = {rid: {k: v for k, v in r.items() if k != 'row_cache'} for rid, r in self.owner.records.items()}
        return (tuple((g, t.block_table.np.tobytes(), t.num_blocks_per_row.tobytes())
                      for g, t in sorted(self.tables.items())),
                {rid: [list(v) for v in r.block_ids] for rid, r in self.runner.requests.items()},
                dict(self.runner._selective_load_plans), rec, dict(self.rows))


class Driver:
    """Applies one random event to both worlds identically."""

    def __init__(self, seed, groups, kernel, ratio):
        self.P = kernel * ratio                          # one FA allocation block per Page
        self.rng = random.Random(seed)
        self.worlds = (World(groups, kernel, ratio, True), World(groups, kernel, ratio, False))
        self.groups, self.next_id, self.next_rid = groups, 1, 0
        self.fixed, self.broken = {}, set()
        self.steps = self.compares = 0

    def fresh(self, n):
        ids = list(range(self.next_id, self.next_id + n))
        self.next_id += n
        return ids

    def both(self, fn):
        for w in self.worlds:
            fn(w)

    def admit(self):
        rng, P = self.rng, self.P
        rid = 'r%d' % self.next_rid
        self.next_rid += 1
        n_hist = rng.randrange(0, 12)
        prefix = sorted(rng.sample(range(n_hist), rng.randrange(0, n_hist + 1))) if n_hist else []
        run = rng.randrange(0, 4)
        prefix_blocks = {g: tuple(self.fresh(len(prefix))) for g in self.groups}
        run_blocks = {g: tuple(self.fresh(run)) for g in self.groups}
        layout = RowLayout(P, n_hist * P, tuple(prefix), prefix_blocks, run_blocks)
        native = {}
        for g in self.groups:
            ids = list(prefix_blocks[g] + run_blocks[g]) + self.fresh(rng.randrange(1, 5))
            if rng.random() < 0.3:
                rng.shuffle(ids)                         # compose reorders [fixed][rest]
            native[g] = ids
        self.fixed[rid] = {g: set(prefix_blocks[g] + run_blocks[g]) for g in self.groups}
        width = max(self.groups) + 1

        def apply(w):
            row = w.free_row()
            w.rows[rid] = row
            w.runner.requests[rid] = NS(block_ids=[list(native.get(g, [])) for g in range(width)])
            for g in self.groups:
                w.tables[g].add_row(native[g], row)
            w.owner.records[rid] = dict(layout=layout)
        self.both(apply)
        return rid

    def release(self, rid):
        def apply(w):
            row = w.rows.pop(rid)
            for t in w.tables.values():
                t.clear_row(row)
            del w.runner.requests[rid]
            del w.owner.records[rid]
        self.both(apply)
        self.broken.discard(rid)

    def grow(self, rid, n):
        new = {g: self.fresh(n) for g in self.groups}
        def apply(w):
            for g in self.groups:
                w.ids(rid, g).extend(new[g])
                w.tables[g].append_row(new[g], w.rows[rid])
        self.both(apply)

    def move(self, rid):
        def apply(w):
            src, tgt = w.rows[rid], w.free_row()
            for t in w.tables.values():
                t.move_row(src, tgt)
                t.clear_row(src)
            w.rows[rid] = tgt
        self.both(apply)

    def swap(self, a, b):
        def apply(w):
            for t in w.tables.values():
                t.swap_row(w.rows[a], w.rows[b])
            w.rows[a], w.rows[b] = w.rows[b], w.rows[a]
        self.both(apply)

    def readd(self, rid):
        def apply(w):
            old = w.rows.pop(rid)
            for t in w.tables.values():
                t.clear_row(old)
            row = w.free_row()
            w.rows[rid] = row
            for g in self.groups:
                w.tables[g].add_row(list(w.ids(rid, g)), row)
        self.both(apply)

    def replace_list(self, rid):
        def apply(w):
            r = w.runner.requests[rid]
            r.block_ids = [list(v) for v in r.block_ids]
        self.both(apply)

    def corrupt(self, rid):
        rng, g = self.rng, self.rng.choice(self.groups)
        kind = rng.randrange(8)
        pick = rng.random()
        def apply(w):
            ids, table, row = w.ids(rid, g), w.tables[g], w.rows[rid]
            r = table.blocks_per_kv_block
            if kind == 0 and self.fixed[rid][g]:            # a pinned block appended
                b = sorted(self.fixed[rid][g])[int(pick * len(self.fixed[rid][g]))]
                ids.append(b); table.append_row([b], row)
            elif kind == 1 and ids:                          # a duplicate id appended
                b = ids[int(pick * len(ids))]
                ids.append(b); table.append_row([b], row)
            elif kind == 2:                                  # null id appended
                ids.append(0); table.append_row([0], row)
            elif kind == 3 and table.num_blocks_per_row[row]:  # device prefix tampered
                j = int(pick * table.num_blocks_per_row[row])
                table.block_table.np[row, j] += 7 * r
            elif kind == 4 and table.num_blocks_per_row[row]:  # count drift
                table.num_blocks_per_row[row] -= 1
            elif kind == 5 and len(ids) > 1:                 # list prefix reordered in place
                i = int(pick * (len(ids) - 1))
                ids[i], ids[i + 1] = ids[i + 1], ids[i]
            elif kind == 6 and len(ids) > 1:                 # a block leaves list and row together
                del ids[int(pick * len(ids))]
                table.add_row(list(ids), row)
            else:                                            # list grows without the device row
                ids.append(10 ** 6 + int(pick * 1000))
        self.both(apply)
        self.broken.add(rid)

    def step(self, rid):
        a, b = self.worlds
        outcome = (a.install(rid), b.install(rid))
        self.steps += 1
        return outcome

    def check(self, test):
        a, b = self.worlds
        test.assertEqual(a.state(), b.state())
        self.compares += 1

    def run(self, test, events):
        live = []
        for _ in range(events):
            rng = self.rng
            if len(live) < 3 and (not live or rng.random() < 0.15):
                live.append(self.admit())
            rid = rng.choice(live)
            x = rng.random()
            if x < 0.45:
                outcome = self.step(rid)
                test.assertEqual(outcome[0], outcome[1], (rid, outcome))
                if outcome[0] is not None:
                    self.broken.add(rid)
            elif x < 0.65:
                self.grow(rid, rng.choice((1, 1, 1, 2, 3)))
            elif x < 0.71:
                self.move(rid) if len(live) < World.ROWS else None
            elif x < 0.76 and len(live) > 1:
                self.swap(rid, rng.choice([r for r in live if r != rid]))
            elif x < 0.81:
                self.readd(rid)
            elif x < 0.85:
                self.replace_list(rid)
            elif x < 0.90:
                self.release(rid); live.remove(rid)
            elif x < 0.93:
                self.corrupt(rid)
            else:
                outcome = self.step(rid)
                test.assertEqual(outcome[0], outcome[1], (rid, outcome))
            self.check(test)


class IncrementalRowEquivalenceTests(unittest.TestCase):
    def drive(self, groups, kernel, ratio, seeds, events=300):
        fast = full = 0
        compose = worker_rows._compose
        counter = {'full': 0}
        def counted(*args):
            counter['full'] += 1
            return compose(*args)
        for seed in seeds:
            d = Driver(seed, groups, kernel, ratio)
            with patch.object(worker_rows, '_compose', counted):
                before = counter['full']
                d.run(self, events)
                # World B calls _compose on every step, world A only on cache misses.
                a_full = counter['full'] - before - d.steps
            fast += d.steps - a_full
            full += a_full
        return fast, full

    def test_page16_single_fa_group(self):
        fast, full = self.drive((0,), 16, 1, range(40))
        self.assertGreater(fast, full, (fast, full))     # the fast path really runs

    def test_page784_legacy_geometry(self):
        # 784-token Pages over 16-token kernel blocks: 49 kernel ids per Page.
        fast, full = self.drive((0,), 16, 49, range(15))
        self.assertGreater(fast, full, (fast, full))

    def test_two_fa_groups_and_kernel_ratio(self):
        fast, full = self.drive((0, 4), 8, 2, range(25))
        self.assertGreater(fast, full, (fast, full))

    def test_steady_decode_never_recomposes(self):
        d = Driver(7, (0,), 16, 1)
        rid = d.admit()
        calls = []
        compose = worker_rows._compose
        with patch.object(worker_rows, '_compose', lambda *a: (calls.append(1), compose(*a))):
            self.assertEqual(d.worlds[0].install(rid), None)
            for i in range(200):
                if i % 16 == 0:
                    d.grow(rid, 1)                       # a Page crossing every 16 decode steps
                self.assertEqual(d.worlds[0].install(rid), None)
        self.assertEqual(len(calls), 1)
        a = d.worlds[0]
        self.assertEqual(a.ids(rid, 0), list(a.owner.records[rid]['row_cache'].groups[0].ids))

    def test_switch_off_is_the_original_code(self):
        d = Driver(3, (0,), 16, 1)
        rid = d.admit()
        b = d.worlds[1]
        self.assertIsNone(b.install(rid))
        self.assertNotIn('row_cache', b.owner.records[rid])


if __name__ == '__main__':
    unittest.main()
