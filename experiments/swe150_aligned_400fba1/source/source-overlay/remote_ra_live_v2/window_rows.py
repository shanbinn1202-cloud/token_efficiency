"""Incremental sliding rows of the window layout: ``--decode-fastpath2`` on Gemma 4 (gemma-perf).

A sliding group's native row (one (block id, incarnation) pair per block index) is, as vLLM's
SlidingWindowManager keeps it, null before the blocks the next query reads and live from
there to its end: ``[(0, 0)] * first + live``, where ``live`` holds at most one window plus
one step (65 entries at W = 1,024 and P = 16) while the row grows with the context
(8,192 entries per group at 131,072 tokens). Each step EngineCore's allocation nulls the
blocks that left the window at its front and appends blocks at its end.

Without this module every allocation update carries the whole rows of the five sliding
groups and of the global group: about 41K pairs per request per step at 131,072 tokens, built
by the engine and checked by the worker. With ``--decode-fastpath2`` on the window layout:

* the engine (``engine_windows`` at a request's first composed row, then ``grow``) follows each
  row incrementally: the global group's appended blocks are checked as ``EngineRpcBoundary._grow``
  checks them; each sliding group's live window is checked every step (its blocks are the same
  objects at the same indices, still live and at the same incarnation; the blocks that left it
  are null; the appended ones are live, tracked, the pool's own objects and unique across the
  groups), in O(window + new blocks);
* the update of a changed request carries only the changes (``delta``): per KV group either
  None (unchanged) or ``[first, length, appended]``, the row's first live index (0 for the
  global group), its new length and the (block, incarnation) pairs appended to it; an
  unchanged request is sent without rows, as before;
* the worker applies a delta to the rows it accepted last (``WindowDelta``), checking what a
  full row's check gives for the changed part: the appended pairs are live ints, the row only
  grows, the window only moves forward, live ids stay unique per pool; it keeps each sliding
  row compact (``WindowRow``), which reads as the full row it stands for.

A request's first allocation update still carries its full rows (checked in full by the
worker, which then keeps its sliding rows compact). The values the worker holds, observes and
leases are the full rows' values; only the payload and the host work differ.

With ``--decode-fastpath3`` (``grow(..., fast3=True)``) the window pool keeps a journal of the
blocks whose reference count or incarnation it changed (``Journal``, ``install_journal``), and a
request's windows are checked block by block (still live, the pool's own, at the same
incarnation: 5 x 65 blocks per request and step at W = 1,024 and P = 16) only when the
journal names one of their blocks since their last check; otherwise those values are the
ones that check found. The rest of each step's check (the row still holds the same block
objects, nulls before the window, the appended blocks) runs as before.
"""
from __future__ import annotations

from itertools import islice, repeat
from operator import attrgetter, is_ as _is

from .contract import require

NULL = (0, 0)
_BLOCK_ID = attrgetter('block_id')


class Journal:
    """--decode-fastpath3: the (pool, block id) keys whose reference count or incarnation a block
    pool changed, in order: each block ``get_new_blocks`` hands out (its incarnation is counted
    there, pool_incarnations), ``touch`` references and ``free_blocks`` releases. vLLM 0.24's
    BlockPool changes ``ref_cnt`` nowhere else, and never replaces an entry of ``blocks``.
    ``mark`` gives a position; past ``LIMIT`` keys the journal restarts under a new epoch, and an
    older position then names nothing (``touched`` is None: the windows are checked in full)."""
    __slots__ = ('keys', 'epoch')
    LIMIT = 1 << 16

    def __init__(self):
        self.keys, self.epoch = [], 0

    def mark(self):
        if len(self.keys) > self.LIMIT:
            del self.keys[:]
            self.epoch += 1
        return self.epoch, len(self.keys)

    def touched(self, mark, live):
        """The keys of ``live`` journaled since ``mark`` (a set), or None when ``mark`` is of an older
        epoch (every key may have changed)."""
        epoch, at = mark
        return live.intersection(self.keys[at:]) if epoch == self.epoch else None


def install_journal(pool):
    """The window pool's journal, its three methods wrapped once (startup, decode_fastpath3.attach_engine).
    Each wrapper calls the method it wraps with the same arguments (``touch`` and ``free_blocks``
    with their blocks as a list: they iterate them once) and returns its result."""
    journal = pool.__dict__.get('_remote_ra_window_journal')
    if journal is not None:
        return journal
    from .pool_incarnations import install
    install(pool)                       # the incarnation counter's wrapper stays inside ours
    journal = Journal()
    keys, pid = journal.keys, id(pool)
    get_new_blocks, touch, free_blocks = pool.get_new_blocks, pool.touch, pool.free_blocks

    def journaled_get_new_blocks(num_blocks):
        blocks = get_new_blocks(num_blocks)
        keys.extend(zip(repeat(pid), map(_BLOCK_ID, blocks)))
        return blocks

    def journaled_touch(blocks):
        blocks = list(blocks)
        keys.extend(zip(repeat(pid), map(_BLOCK_ID, blocks)))
        return touch(blocks)

    def journaled_free_blocks(ordered_blocks):
        blocks = list(ordered_blocks)
        keys.extend(zip(repeat(pid), map(_BLOCK_ID, blocks)))
        return free_blocks(blocks)
    pool.get_new_blocks, pool.touch, pool.free_blocks = (journaled_get_new_blocks, journaled_touch,
                                                         journaled_free_blocks)
    pool._remote_ra_window_journal = journal
    return journal


class WindowRow:
    """A sliding group's native row as ``(0, 0)`` before index ``first``, then ``live``
    (a tuple of (block, incarnation) pairs, every one live). It reads as the full row:
    ``len``, indexing and iteration give the full row's values."""
    __slots__ = ('first', 'live')

    def __init__(self, first, live):
        self.first, self.live = first, live

    def __len__(self):
        return self.first + len(self.live)

    def __getitem__(self, index):
        if type(index) is slice:
            return tuple(self)[index]
        n = self.first + len(self.live)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError('window row index out of range')
        return NULL if index < self.first else self.live[index - self.first]

    def __iter__(self):
        yield from repeat(NULL, self.first)
        yield from self.live

    def __eq__(self, other):
        if type(other) is WindowRow:
            return self.first == other.first and self.live == other.live
        return tuple(self) == tuple(other)

    __hash__ = None

    def __repr__(self):
        return 'WindowRow(%d, %r)' % (self.first, self.live)


def compact(row):
    """The ``WindowRow`` of a full sliding row (a tuple of pairs), or None when it is not one
    window of live blocks behind nulls (the observation then refuses it as before)."""
    n = len(row)
    first = 0
    while first < n and row[first][0] == 0:
        first += 1
    live = row[first:]
    if not live or not all(b > 0 for b, _ in live):
        return None
    return WindowRow(first, tuple(live))


# ---- the engine side -------------------------------------------------------------------------------

def engine_windows(groups, attention, request_id, pools):
    """At a request's first composed (published) row: each sliding group's window as the engine
    follows it, [first, blocks, pairs] (None for every group when a row is not one window
    behind nulls: that request's later updates are then sent whole), and the live ids."""
    from .pool_incarnations import install
    windows, live = {}, set()
    for g, (m, pool) in enumerate(zip(groups, pools)):
        if g in attention:
            continue
        state = install(pool)
        current = m.req_to_blocks[request_id]
        first = 0
        while first < len(current) and current[first].block_id == 0:
            first += 1
        blocks = list(current[first:])
        if not blocks or any(block.block_id == 0 for block in blocks):
            return None
        windows[g] = [first, blocks, [(block.block_id, state[block.block_id]) for block in blocks]]
        live.update((id(pool), block.block_id) for block in blocks)
    return dict(groups=windows, live=live)


def _unchanged(mark, windows, groups, pools, fa, live, rid):
    """--decode-fastpath3: the window pool when the journal names none of the request's kept window
    blocks since ``mark`` (each window's blocks behind the nulls now at its front left it: only
    they may be named), else None. Reads only; ``grow`` makes every test of the rows."""
    sliding = [g for g in range(len(groups)) if g not in fa]
    if not sliding or any(pools[g] is not pools[sliding[0]] for g in sliding):
        return None
    pool = pools[sliding[0]]
    journal = pool.__dict__.get('_remote_ra_window_journal')
    if journal is None or mark[0] != id(pool):
        return None
    touched = journal.touched(mark[1:], live)
    if touched is None:
        return None
    if touched:
        pid, left = id(pool), set()
        for g in sliding:
            first, blocks, pairs = windows['groups'][g]
            current = groups[g].req_to_blocks[rid]
            k, end = 0, min(first + len(blocks), len(current))
            while first + k < end and current[first + k].block_id == 0:
                k += 1
            left.update((pid, b) for b, _ in pairs[:k])
        if not touched <= left:
            return None
    return pool


def grow(record, fast, binding, manager, pools, accepted, fast3=False):
    """``EngineRpcBoundary._grow`` for the window layout under --decode-fastpath2: ``accepted``
    (the rows the worker acknowledged last) when no row changed, else the delta of every KV
    group (``{"window": [None or [first, length, appended], ...]}``).

    ``fast3`` (--decode-fastpath3): the kept blocks of a window whose pool journals its
    blocks (``install_journal``) are tested one by one only when the journal names one of them
    since the request's windows were last checked (``windows['mark']``): the blocks that left a
    window this step (now null in its row) are the only window blocks it may name otherwise, as
    the request's own allocation freed them. When it names no kept block, the pool changed none
    of their reference counts or incarnations, and the test gives that check's answer (it held).
    The step's other tests are unchanged."""
    from .pool_incarnations import install
    rid = binding.request
    groups = manager.coordinator.single_type_managers
    require(tuple(groups) == fast['groups'] and tuple(pools) == fast['pools'],
            'native KV groups changed after publication')
    tables, seen, windows = record['native_tables'], fast['ids'], fast['windows']
    live = windows['live']
    items = [None] * len(groups)
    # The global group(s): appended blocks only, as _grow checks them.
    fa_new, fresh = [], set()
    for g, (m, pool) in enumerate(zip(groups, pools)):
        row = fast['fa'].get(g)
        if row is None:
            continue
        state = install(pool)
        current = m.req_to_blocks[rid]
        prior = tables[g]
        n = len(prior)
        require(len(current) >= n and (n == 0 or current[n - 1].block_id == prior[-1]),
                'native sparse prefix moved after publication')
        new = current[n:]
        for block in new:
            b = block.block_id
            require(type(b) is int and b > 0 and b not in seen and b not in fresh, 'native FA row null/aliased')
            require(block.ref_cnt > 0 and b in state and pool.blocks[b] is block,
                    'native row owns untracked/free allocation')
            fresh.add(b)
        if new:
            fa_new.append((g, row, n, new, state))
    # The sliding groups: every window checked in place, the blocks that left it first (an id
    # freed at one window's front may be allocated at another's end in the same step).
    plans = []
    mark = windows.get('mark') if fast3 else None
    unchanged = None                 # --decode-fastpath3: no kept window block journaled since mark
    if mark is not None:
        unchanged = _unchanged(mark, windows, groups, pools, fast['fa'], live, rid)
    for g, (m, pool) in enumerate(zip(groups, pools)):
        if g in fast['fa']:
            continue
        state = install(pool)
        first, blocks, pairs = windows['groups'][g]
        current = m.req_to_blocks[rid]
        end = first + len(blocks)
        require(len(current) >= end and (first == 0 or current[first - 1].block_id == 0),
                'native sliding row moved after publication')
        k = 0
        while k < len(blocks) and current[first + k].block_id == 0:
            k += 1
        kept = blocks[k:]
        if not fast3:
            require(all(map(_is, islice(current, first + k, end), kept)) and
                    all(block.ref_cnt > 0 and pool.blocks[b] is block and state.get(b) == i
                        for block, (b, i) in zip(kept, pairs[k:])),
                    'native row owns untracked/free allocation')
            plans.append((g, pool, state, current, first, k, end))
            continue
        # --decode-fastpath3: the window's entries as a slice of the row (islice walked the row
        # from its start: O(context) per group and step); the same elements.
        require(all(map(_is, current[first + k:end], kept)) and
                (unchanged is pool or all(block.ref_cnt > 0 and pool.blocks[b] is block and state.get(b) == i
                                          for block, (b, i) in zip(kept, pairs[k:]))),
                'native row owns untracked/free allocation')
        plans.append((g, pool, state, current, first, k, end))
    for g, pool, state, current, first, k, end in plans:
        for b, _ in windows['groups'][g][2][:k]:
            live.discard((id(pool), b))
    deltas = []
    for g, pool, state, current, first, k, end in plans:
        new = current[end:]
        appended = []
        for block in new:
            b = block.block_id
            require(type(b) is int and b > 0 and block.ref_cnt > 0 and b in state and pool.blocks[b] is block,
                    'native row owns untracked/free allocation')
            key = (id(pool), b)
            require(key not in live, 'native live allocation IDs overlap across groups')
            live.add(key)
            appended.append((b, state[b]))
        deltas.append((g, first, k, end, new, appended))
    # Every check passed: advance the followed rows and build the delta.
    if fast3:
        # The windows' blocks as checked now: the journal's position (one window pool).
        journaled = {id(pool): pool for g, pool, *_ in plans}
        windows['mark'] = None
        if len(journaled) == 1:
            pool, = journaled.values()
            journal = pool.__dict__.get('_remote_ra_window_journal')
            if journal is not None:
                windows['mark'] = (id(pool),) + journal.mark()
    for g, row, n, new, state in fa_new:
        pairs = [(block.block_id, state[block.block_id]) for block in new]
        for b, _ in pairs:
            seen.add(b)
        row.extend(pairs)
        tables[g] = tables[g] + tuple(b for b, _ in pairs)
        items[g] = [0, len(row), [list(p) for p in pairs]]
    for g, first, k, end, new, appended in deltas:
        if not k and not new:
            continue
        window = windows['groups'][g]
        window[0] = first + k
        window[1] = window[1][k:] + list(new)
        window[2] = window[2][k:] + appended
        items[g] = [first + k, end + len(new), [list(p) for p in appended]]
    if all(item is None for item in items) and accepted is not None:
        return accepted
    return dict(window=items)


# ---- the worker side -------------------------------------------------------------------------------

def _pair(value):
    return (type(value) in (list, tuple) and len(value) == 2 and type(value[0]) is int and
            type(value[1]) is int and value[0] > 0 and value[1] > 0)


def apply_delta(owner, rec, rows):
    """The worker rows a delta stands for, and the commit of its id sets: (rows, commit).

    ``rows`` is ``{"window": [...]}`` (``grow``); the base is the rows the worker accepted
    last (``rec['native_rows']``, its sliding rows compact). Refused like a full row whose
    changed part fails the full check: a pair that is not two positive ints, a row that
    shrank or whose window moved back, an appended id live in its pool already."""
    items = rows.get('window') if type(rows) is dict else None
    old = rec.get('native_rows')
    ids = rec.get('window_ids')
    pools = owner.pool_of
    require(type(rows) is dict and set(rows) == {'window'} and type(items) in (list, tuple) and
            old is not None and ids is not None and len(items) == len(old) == len(pools),
            'window row delta without the accepted rows')
    out = list(old)
    gone, came = [], []
    for g, item in enumerate(items):
        if item is None:
            continue
        require(type(item) in (list, tuple) and len(item) == 3 and type(item[0]) is int and
                type(item[1]) is int and type(item[2]) in (list, tuple) and all(map(_pair, item[2])),
                'invalid native allocation incarnation row')
        first, length, appended = item
        base = old[g]
        pairs = tuple((b, i) for b, i in appended)
        if g in owner.attention_groups:
            require(first == 0 and pairs and length == len(base) + len(pairs),
                    'window row delta changes a full-attention row other than by appending')
            out[g] = base + pairs
            came.extend((pools[g], b) for b, _ in pairs)
            continue
        require(type(base) is WindowRow, 'window row delta on a row that is not one window')
        drop = first - base.first
        require(0 <= drop <= len(base.live) and length == len(base) + len(pairs) and first < length,
                'window row delta moves a sliding window back or shrinks its row')
        gone.extend((pools[g], b) for b, _ in base.live[:drop])
        came.extend((pools[g], b) for b, _ in pairs)
        out[g] = WindowRow(first, base.live[drop:] + pairs)
    remaining = ids - set(gone) if gone else ids
    require(len(set(came)) == len(came) and remaining.isdisjoint(came), 'native live pool IDs alias across groups')
    out = tuple(out)

    def commit():
        ids.difference_update(gone)
        ids.update(came)
    return out, commit


def compact_rows(owner, rec):
    """After a full update on the window layout under --decode-fastpath2: the accepted rows
    with every sliding row compact (one that is not one window stays whole; the observation
    refuses it), and the live ids per pool that deltas extend."""
    rows = rec['native_rows']
    pools = owner.pool_of
    out = []
    for g, row in enumerate(rows):
        if g in owner.hybrid_groups:
            row = compact(row) or row
        out.append(row)
    rec['native_rows'] = tuple(out)
    rec['window_ids'] = {(pools[g], b) for g, row in enumerate(rows) for b, _ in row if b}
