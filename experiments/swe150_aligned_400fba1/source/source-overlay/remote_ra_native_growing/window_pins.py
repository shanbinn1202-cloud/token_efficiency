"""Engine ownership of the sliding windows (the window layout, Gemma 4, gemma-s1 WP3).

The window layout's pool 1 holds the sliding-window groups. A session keeps its window
on the GPU between turns (W-GPU, design D-3): the blocks of every sliding group that
the next query reads, ``sliding_window.kept_blocks(C')`` after the computed end C'. This
module owns those pins, next to ``PrivateBlockOwner`` (pool 0, the global Pages):

* ``window(ns)``: the session's pinned window, {group: ((index, block), ...)}; phase 1
  borrows it (read-only, its last block may be partial and phase 1 appends into it);
* ``handover(ns, first)`` at the phase-2 admission: the pinned blocks from index
  ``first`` on (``kept_blocks(e)``) move to the request without a reference change;
  the older ones lose their pin and are freed (with --goal-overlap already by ``trim``
  after phase 1's drain, while the admission waits for its Goal);
* ``install(request, head)``: phase 2's sliding rows start as the null head and the
  window phase 1 ended in, before vLLM's first allocation (which would otherwise
  allocate every block of [0, e));
* ``pin_after(ns, request, computed_end)`` at the commit, before vLLM frees the
  request's blocks: the session pins the blocks of ``kept_blocks(C')`` (reference +1);
* ``close(ns)``: a closed or rebuilt session frees its window.

--thinking-history strip on this layout (option G, remote_ra_live_v2.window_hold): the session
commits at the cut X = e, but phase 2 decodes past it and vLLM frees each block that leaves the
window, so the window at X would be gone after about 1,024 decoded tokens. ``hold(ns, request,
X, source, cursor)`` takes one more reference on ``kept_blocks(X)`` of the request's row: at a
continuing turn's install (the head is exactly ``kept_blocks(e)``), at local turn 0 before the
first allocation at a cursor at or past X. ``pin_after(ns, request, X)`` then pins the held
blocks (the hold's references become the session's), ``release`` drops an uncommitted hold
(``discard``, ``after_free``). At most ``hold_slots`` holds exist at once (the layout's H; 0, and
nothing held, without the strip history).

vLLM's SlidingWindowManager frees a request's blocks as its window moves and at its
end; references taken here are the session's own. Every block is checked to be pool
1's own object before it is pinned or freed.
"""
from .contract import require


class WindowPins:
    # --thinking-history strip (window_hold.attach_engine): the layout's hold slots H; 0 holds nothing.
    hold_slots = 0

    def __init__(self, pool, managers, groups, *, window_tokens, page_tokens):
        require(type(groups) is tuple and groups and len(managers) == len(groups) and
                all(type(m).__name__ == "SlidingWindowManager" and m.block_pool is pool for m in managers),
                "sliding-window managers of pool 1 required")
        self.pool, self.managers, self.groups = pool, tuple(managers), groups
        self.window_tokens, self.page_tokens = window_tokens, page_tokens
        self.windows = {}       # namespace -> {group: ((index, block), ...)}
        self.pending = {}       # request -> {group: (first, (block, ...))}: handed over, not installed
        self.installed = {}     # request -> namespace: a head in the managers' rows
        # request -> {namespace, end, source, held_at, window: {group: ((index, block), ...)}}: the
        # window at the strip cut, one more reference each (``hold``); empty without the strip history.
        self.holds = {}

    def _own(self, block):
        return (type(block.block_id) is int and 0 < block.block_id < len(self.pool.blocks) and
                self.pool.blocks[block.block_id] is block and not block.is_null)

    def _kept(self, end):
        from remote_ra_live_v2.sliding_window import kept_blocks
        return kept_blocks(end, self.window_tokens, self.page_tokens)

    def window(self, namespace):
        return self.windows.get(namespace)

    def handover(self, namespace, request_id, first, fresh):
        """Phase-2 admission: the window phase 1 ended in becomes the request's head.

        ``fresh`` {group: ((index, block), ...)}: phase 1's fresh blocks from ``first`` on
        (their phase-1 reference becomes the request's). Pinned blocks before ``first``
        are freed; the session pins nothing until its commit."""
        pinned = self.windows.pop(namespace, None)
        require(pinned is not None and request_id not in self.pending and request_id not in self.installed,
                "window handover without a pinned window")
        drop = [b for g in self.groups for i, b in pinned[g] if i < first]
        keep = {g: [(i, b) for i, b in pinned[g] if i >= first] for g in self.groups}
        require(all(self._own(b) and b.ref_cnt == 1 for b in drop) and
                all(self._own(b) and b.ref_cnt == 1 for g in self.groups for _, b in keep[g]) and
                all(self._own(b) and b.ref_cnt == 1 for g in self.groups for _, b in fresh.get(g, ())),
                "window blocks changed owner before the handover")
        self.pool.free_blocks(reversed(drop))
        head = {}
        for g in self.groups:
            entries = keep[g] + list(fresh.get(g, ()))
            indices = [i for i, _ in entries]
            require(entries and indices == list(range(first, first + len(entries))),
                    "the handed-over window is not contiguous from its first index")
            head[g] = (first, tuple(b for _, b in entries))
        self.pending[request_id] = (namespace, head)
        return head

    def trim(self, namespace, first):
        """--goal-overlap (remote_ra_live_v2.goal_overlap, window layout): after a drained phase 1
        that waits for its Goal, the pinned blocks before index ``first`` (``kept_blocks(e)``,
        which no later query of the session reads) lose their pin and are freed now, as
        ``handover`` would free them at the phase-2 admission; the session keeps the rest
        pinned, and ``handover`` then drops nothing. Returns how many blocks it freed."""
        pinned = self.windows.get(namespace)
        require(pinned is not None and not any(ns == namespace for ns, _ in self.pending.values()),
                "window trim without a pinned window")
        drop = [b for g in self.groups for i, b in pinned[g] if i < first]
        require(all(self._own(b) and b.ref_cnt == 1 for b in drop), "window blocks changed owner before the trim")
        self.pool.free_blocks(reversed(drop))
        self.windows[namespace] = {g: tuple((i, b) for i, b in pinned[g] if i >= first) for g in self.groups}
        return len(drop)

    def install(self, request_id):
        """Install the handed-over head into the sliding managers' rows (before allocation)."""
        namespace, head = self.pending.pop(request_id)
        null = self.pool.null_block
        for g, manager in zip(self.groups, self.managers):
            require(not manager.req_to_blocks.get(request_id), "occupied sliding row")
            first, blocks = head[g]
            manager.req_to_blocks[request_id] = [null] * first + list(blocks)
            manager.num_cached_block[request_id] = 0
        self.installed[request_id] = namespace
        return head

    def hold(self, namespace, request_id, end, source, cursor):
        """--thinking-history strip: hold the window at the cut ``end`` = X (``kept_blocks(X)`` of
        every sliding group) in the request's row: one more reference each, so the blocks survive
        phase 2's window moves. ``source`` 'install' (a continuing turn, at its install: the head)
        or 'prefill' (local turn 0, before the first allocation at ``cursor`` >= X)."""
        require(self.hold_slots > 0 and source in ('install', 'prefill') and request_id not in self.holds and
                len(self.holds) < self.hold_slots and type(end) is int and end > 0 and
                type(cursor) is int and cursor >= end,
                "window hold without a free hold slot, twice, or before the cut")
        require(source != 'install' or self.installed.get(request_id) == namespace,
                "an install hold needs the request's installed head")
        first, last = self._kept(end)
        window = {}
        for g, manager in zip(self.groups, self.managers):
            row = manager.req_to_blocks.get(request_id, ())
            require(last <= len(row) and all(self._own(row[i]) and row[i].ref_cnt == 1 for i in range(first, last)),
                    "the window at the cut is not the request's own live row")
            window[g] = tuple((i, row[i]) for i in range(first, last))
        self.pool.touch([b for g in self.groups for _, b in window[g]])
        self.holds[request_id] = dict(namespace=namespace, end=end, source=source, held_at=cursor, window=window)
        return {g: tuple((i, b.block_id) for i, b in v) for g, v in window.items()}

    def release(self, request_id):
        """Drop an uncommitted hold (a failed admission, a cancel): its references go; a block
        that already left the request's row returns to the pool. Returns how many it released."""
        held = self.holds.get(request_id)
        if held is None:
            return 0
        blocks = [b for g in self.groups for _, b in held['window'][g]]
        require(all(self._own(b) and b.ref_cnt >= 1 for b in blocks), "held window blocks lost their reference")
        del self.holds[request_id]
        self.pool.free_blocks(reversed(blocks))
        return len(blocks)

    def held_left(self, request_id):
        """Blocks of the first sliding group's hold that have left the request's row (0 without a hold)."""
        held = self.holds.get(request_id)
        if held is None:
            return 0
        g, manager = self.groups[0], self.managers[0]
        row = manager.req_to_blocks.get(request_id, ())
        return sum(1 for i, b in held['window'][g] if i >= len(row) or row[i] is not b)

    def held_blocks(self):
        return sum(len(v) for held in self.holds.values() for v in held['window'].values())

    def discard(self, request_id):
        """An admission failed before the worker saw it: free the request's head."""
        self.release(request_id)
        pending = self.pending.pop(request_id, None)
        blocks = []
        if pending is not None:
            blocks = [b for first, row in pending[1].values() for b in row]
        elif self.installed.pop(request_id, None) is not None:
            for manager in self.managers:
                row = manager.req_to_blocks.pop(request_id, [])
                manager.num_cached_block.pop(request_id, None)
                blocks += [b for b in row if not b.is_null]
        require(all(self._own(b) and b.ref_cnt == 1 for b in blocks), "discarded window blocks have another owner")
        self.pool.free_blocks(reversed(blocks))

    def pin_after(self, namespace, request_id, computed_end):
        """Commit (before vLLM frees the request): the session pins the window after C'.

        --thinking-history strip: after the cut X (``computed_end``); a held window's references
        become the session's pin (no reference change: a block still in the row keeps the row's
        too until vLLM frees the request). Without a hold, from the row as before."""
        require(namespace not in self.windows, "a session window is already pinned")
        held = self.holds.get(request_id)
        if held is not None:
            require(held['namespace'] == namespace and held['end'] == computed_end and
                    all(self._own(b) and b.ref_cnt >= 1 for g in self.groups for _, b in held['window'][g]),
                    "the held window is not the session's window at the commit end")
            window = dict(held['window'])
            del self.holds[request_id]
            self.windows[namespace] = window
            self.installed.pop(request_id, None)
            return {g: tuple((i, b.block_id) for i, b in v) for g, v in window.items()}
        first, last = self._kept(computed_end)
        window = {}
        for g, manager in zip(self.groups, self.managers):
            row = manager.req_to_blocks.get(request_id, ())
            require(last <= len(row) and all(self._own(row[i]) and row[i].ref_cnt == 1 for i in range(first, last)),
                    "the window after the computed end is not live in the request's row")
            window[g] = tuple((i, row[i]) for i in range(first, last))
        self.pool.touch([b for g in self.groups for _, b in window[g]])
        self.windows[namespace] = window
        self.installed.pop(request_id, None)
        return {g: tuple((i, b.block_id) for i, b in v) for g, v in window.items()}

    def after_free(self, request_id, namespace):
        """After vLLM freed the request: its pinned window survives with the session's pin only.
        An uncommitted hold (a cancel in phase 2) is released."""
        self.installed.pop(request_id, None)
        self.release(request_id)
        window = self.windows.get(namespace)
        require(window is None or all(b.ref_cnt == 1 for g in self.groups for _, b in window[g]),
                "native free left a window block with another owner")

    def close(self, namespace):
        """Session close or rebuild: free the pinned window."""
        require(namespace not in self.installed.values() and
                not any(ns == namespace for ns, _ in self.pending.values()) and
                not any(held['namespace'] == namespace for held in self.holds.values()),
                "cannot close a window while its request holds it")
        window = self.windows.pop(namespace, None)
        if window is None:
            return 0
        blocks = [b for g in self.groups for _, b in window[g]]
        require(all(self._own(b) and b.ref_cnt == 1 for b in blocks), "unexpected window block owner at close")
        self.pool.free_blocks(reversed(blocks))
        return len(blocks)

    def pinned_blocks(self):
        return sum(len(v) for window in self.windows.values() for v in window.values())
