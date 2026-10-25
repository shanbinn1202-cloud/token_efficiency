"""Worker allocation owner of the window layout (Gemma 4 31B IT, gemma-s1 WP5).

``RunnerAllocationOwner`` observes and claims every owned request's native blocks per
step, zeroes and restores its full-attention Pages and keeps the session mirrors. The
window layout keeps all of that for the global full-attention group; its per-request
state groups are the sliding-window groups instead of the Mamba groups:

* each step leases every live sliding block of the request (allocation kind
  ``sliding``) at its incarnation; nothing is zeroed (vLLM does not zero sliding
  blocks, and the window blocks hold the KV the next query reads). The live blocks
  are EngineCore's (its published rows null the blocks that left the window); vLLM's
  runner only appends new block ids, so its row must equal EngineCore's on the live
  window, which holds everything the step reads, and keeps the older blocks unread;
* phase 1's sliding rows must borrow exactly the session's committed window
  (``check_phase1_window``); phase 2's rows must start with the window phase 1 ended
  in, behind its null head (checked at the request's first step);
* the drain commits the window after the computed end (``final_window``).

There is no ``mamba_state_idx``. Without --decode-fastpath every step is observed in full.
With it (``attach_fastpath``, gemma-perf) a step extends the last observation when the full
observation would accept the same rows: the full-attention rows only grew, and each sliding
group's live window lost blocks at its front and gained them at its end; the leases, claims
and allocation sets then move by those differences only (the same values).
"""
from __future__ import annotations

from . import allocation_owner as _owner
from .allocation_owner import RunnerAllocationOwner, StepRow, _FaRow, _Observed
from .contract import Allocation, require
from .sliding_window import kept_blocks, max_kept_blocks, phase1_row
from .window_rows import WindowRow


class WindowAllocationOwner(RunnerAllocationOwner):
    # --decode-fastpath (attach_fastpath): each step extends the last observation (full-attention
    # rows as RunnerAllocationOwner's incremental observation, sliding windows by the blocks that
    # left and entered them), the claims and allocation sets move by the differences, and an
    # unchanged sliding lease tuple is verified once per allocation epoch.
    window_fast = False
    # --decode-fastpath2 (attach_fastpath2): changed rows come as deltas (window_rows), sliding
    # rows are kept compact.
    window_rows = False

    def __init__(self, runner, io, startup, layout=None):
        from .kv_layout import is_window
        require(is_window(layout), "the window allocation owner needs the window layout")
        super().__init__(runner, io, startup, layout)
        self.window_tokens = layout.window_tokens

    @staticmethod
    def _state_groups(runner, io):
        """The sliding-window groups of the loaded KV cache config."""
        config = getattr(runner, "kv_cache_config", None)
        groups = tuple(g for g, group in enumerate(getattr(config, "kv_cache_groups", ()))
                       if type(group.kv_cache_spec).__name__ == "SlidingWindowSpec")
        require(groups, "sliding-window KV groups required")
        return groups

    # ---- phase 1 and phase 2 window lineage ----

    def check_phase1_window(self, rec, lease, session):
        """Phase 1's sliding rows: the null head, exactly the committed window, then fresh
        blocks; record the window phase 2 must start with."""
        P, W = self.page_tokens, self.window_tokens
        carry = self.backend.carry.get(rec["identity"].session)
        require(carry is not None and carry.digest == session["carry_digest"] and
                carry.computed_end == lease.start, "phase 1 without its committed window")
        borrowed = carry.blocks()
        # A chunked phase 1 (gemma-goal): ``rows`` are its first forward's, over
        # [C, end of that forward); the window phase 2 starts with is its last forward's.
        head, (b0, b1), (f0, f1) = phase1_row(lease.start, lease.forwards()[0][1], W, P)
        require(set(borrowed) == set(self.hybrid_groups) and carry.first() == {g: b0 for g in borrowed},
                "committed window differs from the phase-1 geometry")
        pinned = {(1, b) for blocks in borrowed.values() for b in blocks}
        k0, k1 = kept_blocks(lease.end, W, P)
        expected = {}
        for g in self.hybrid_groups:
            row = lease.rows[g]
            require(len(row) == f1 and row[:head] == (0,) * head and row[b0:b1] == borrowed[g] and
                    0 not in row[f0:] and not {(1, b) for b in row[f0:]} & pinned,
                    "phase-1 sliding row does not borrow the committed window")
            expected[g] = (k0, tuple(lease.final_rows[g][k0:k1]))
        rec["window_head"] = expected
        return expected

    def window_head_verified(self, identity, generation):
        rec = self.records.get(identity.request)
        return (rec is not None and rec["generation"] == generation and
                rec.get("window_head_verified") is True)

    def final_window(self, identity, computed_end, generation):
        """The window after ``computed_end``: ((group, ((index, block, incarnation), ...)), ...)
        from the engine's last published rows (the blocks the engine pins at commit)."""
        rec = self.records.get(identity.request)
        require(rec is not None and rec["identity"] == identity and rec["generation"] == generation,
                "final window of a foreign request")
        rows = rec.get("native_rows")
        require(rows is not None, "final window needs the published native rows")
        k0, k1 = kept_blocks(computed_end, self.window_tokens, self.page_tokens)
        window = []
        for g in self.hybrid_groups:
            row = rows[g]
            require(k1 <= len(row) and all(row[i][0] > 0 for i in range(k0, k1)),
                    "the window after the computed end is not in the native rows")
            window.append((g, tuple((i, row[i][0], row[i][1]) for i in range(k0, k1))))
        return tuple(window)

    # ---- per step ----

    def _observe(self, runner, rid, row, offset, count):
        """One request's native rows: the full-attention row as ``RunnerAllocationOwner``
        checks it (fixed [A_t][run] head, kernel expansion) and every live sliding block;
        no claims, no IO."""
        rec = self.records.get(rid)
        require(rec is not None and "layout" in rec and rid in runner.requests,
                "native runner request/allocation missing")
        request = runner.requests[rid]
        start = int(request.num_computed_tokens)
        generation = rec["generation"]
        a, layout = rec["assignment"], rec["layout"]
        native_rows = rec.get("native_rows")
        if self.startup.get("source_closure_verified"):
            require(native_rows is not None and rec["native_interval"] == (start, count),
                    "native allocation incarnation update missing/stale")
        fast = self.window_fast and _owner.INCREMENTAL
        if fast:
            s = self._observe_increment_window(runner, rec, request, rid, row, offset, start, count, native_rows)
            if s is not None:
                return s
        if native_rows is not None:
            require(all(tuple(request.block_ids[g]) == tuple(b for b, _ in native_rows[g])
                        for g in self.attention_groups),
                    "runner allocation rows differ from EngineCore ownership")
        incarnations = {}

        def incarnation(group, block):
            if native_rows is None:
                return 0      # explicit shaped CPU fixtures only
            if group not in incarnations:
                incarnations[group] = dict(native_rows[group])
            return incarnations[group][block]
        kind = ("decode" if start >= a.prompt_tokens else "suffix" if a.local_turn else "attention")
        attention, tables = [], {}
        from .geometry import AllocationId
        for group in self.attention_groups:
            table = runner.input_batch.block_table.block_tables[group]
            if hasattr(table, "get_row"):
                kernel = table.get_row(row)
            else:
                kernel = table.block_table.np[row, :int(table.num_blocks_per_row[row])]
            kernel_ids = tuple(int(v) for v in kernel if int(v) > 0)
            ids = tuple(AllocationId.from_wire(group, v).value for v in request.block_ids[group])
            ratio = int(getattr(table, "blocks_per_kv_block", 1))
            require(kernel_ids == tuple(b * ratio + i for b in ids for i in range(ratio)),
                    "native kernel row differs from allocation source expansion")
            fixed = layout.prefix_blocks[group] + layout.run_blocks[group]
            require(ids[:len(fixed)] == fixed, "phase-2 row lost its fixed [A_t][run] head")
            tables[group] = (ids, int(table.block_size) * ratio)
            for block in ids:
                if block > 0:
                    attention.append(Allocation(group, block, generation, kind, incarnation(group, block)))
        require(attention, "native attention allocations missing")
        # Sliding groups: logical rows; EngineCore's SlidingWindowManager nulls the blocks
        # that left the window. vLLM's runner only appends a request's new block ids
        # (CachedRequestData.new_block_ids), so its row keeps those blocks, which no query
        # reads: it must equal EngineCore's row on the live window, which holds everything
        # the step reads, [start - W + 1, start + count).
        head = rec.get("window_head") if a.local_turn and not rec.get("window_head_verified") else None
        if a.local_turn:
            require(rec.get("window_head_verified") or head is not None,
                    "continuing turn without its phase-1 window")
        P = self.page_tokens
        bound = max_kept_blocks(self.window_tokens, P) + -(-count // P)
        sliding = []
        windows = {} if fast else None
        for group in self.hybrid_groups:
            runner_ids = request.block_ids[group]
            ids = (tuple(int(b) for b in runner_ids) if native_rows is None else
                   tuple(b for b, _ in native_rows[group]))
            live = [(i, b) for i, b in enumerate(ids) if b > 0]
            require(live and len(live) <= bound and all(i >= live[0][0] for i, _ in live) and
                    [i for i, _ in live] == list(range(live[0][0], live[0][0] + len(live))),
                    "sliding row is not one window of live blocks")
            if native_rows is not None:
                first = live[0][0]
                require(len(runner_ids) == len(ids) and
                        tuple(int(b) for b in runner_ids[first:]) == ids[first:],
                        "runner sliding window differs from EngineCore ownership")
                require(first <= max(0, start - self.window_tokens + 1) // P and
                        len(ids) >= -(-(start + count) // P),
                        "the step reads outside its live sliding window")
            if head is not None:
                first, blocks = head[group]
                require(ids[:first] == (0,) * first and ids[first:first + len(blocks)] == blocks,
                        "phase-2 sliding row does not start with the phase-1 window")
            leases = tuple(Allocation(group, b, generation, "sliding", incarnation(group, b)) for _, b in live)
            sliding.extend(leases)
            if windows is not None:
                windows[group] = (live[0][0], leases)
        if head is not None:
            rec["window_head_verified"] = True
        rec["last_tables"] = tables
        current = tuple(v.key for v in tuple(attention) + tuple(sliding))
        s = StepRow(rid, row, offset, start, count, tuple(attention), tuple(sliding), generation, current)
        if fast:
            self._remember_window(runner, rec, request, row, s, kind, native_rows, windows)
        return s

    # ---- --decode-fastpath: the incremental observation of the window layout ----

    def _remember_window(self, runner, rec, request, row, s, kind, native_rows, windows):
        """Record a full observation so the next step can extend it: the full-attention rows as
        ``RunnerAllocationOwner._remember`` records them, and each sliding group's live window."""
        import numpy as np
        rec.pop("observed", None)
        if native_rows is None:
            return                            # explicit shaped CPU fixtures only
        o = _WindowObserved()
        o.layout, o.generation, o.kind, o.native = rec["layout"], s.generation, kind, True
        o.rows, o.fa_ids, pools, at, total = {}, set(), self.pool_of, 0, 0
        for group in self.attention_groups:
            table = runner.input_batch.block_table.block_tables[group]
            ids_tuple = rec["last_tables"][group][0]
            ratio = int(getattr(table, "blocks_per_kv_block", 1))
            if hasattr(table, "get_row") or int(table.num_blocks_per_row[row]) != len(ids_tuple) * ratio:
                return                        # fixtures without numpy rows, or null kernel entries
            fa = _FaRow()
            fa.ids, fa.ids_tuple = list(request.block_ids[group]), ids_tuple
            fa.kernel = np.array(table.block_table.np[row, :len(ids_tuple) * ratio])
            fa.pairs = native_rows[group]
            fa.allocs = s.attention[at:at + len(ids_tuple)]
            fa.keys = s.current[at:at + len(ids_tuple)]
            fa.ratio, fa.kernel_tokens = ratio, int(table.block_size)
            at += len(ids_tuple)
            o.rows[group] = fa
            o.fa_ids.update((pools[group], b) for b in ids_tuple)
            total += len(ids_tuple)
        o.fa_attention, o.fa_keys = s.attention, s.current[:at]
        o.windows, o.window_ids, slots = {}, set(), 0
        for group in self.hybrid_groups:
            first, leases = windows[group]
            o.windows[group] = (first, tuple((v.block, v.incarnation) for v in leases),
                                [v.block for v in leases], leases, tuple(v.key for v in leases))
            o.window_ids.update((pools[group], v.block) for v in leases)
            slots += len(leases)
        o.unique = len(o.fa_ids) == total and len(o.window_ids) == slots
        o.step, o.base = s, None
        o.added = o.removed = o.claim_added = o.claim_removed = None
        o.claimed = o.materialized = None
        rec["observed"] = o

    def _observe_increment_window(self, runner, rec, request, rid, row, offset, start, count, native_rows):
        """This step's StepRow from the last observation, or None (observe in full).

        Returns a row only when the full observation would accept and build the same one:
        the full-attention rows only grew (``RunnerAllocationOwner._observe_increment``'s
        checks: the native id list starts with the last observed list, the device row holds
        its kernel expansion plus that of the appended ids, the native incarnation rows
        start with the last observed ones), and every sliding group's EngineCore row is
        one window of live blocks behind nulls: its live part is the last window minus the
        blocks that left it at the front plus appended live blocks, it holds what the step
        reads and at most one window plus the step, and the runner's row equals it there.
        The leases of a sliding group are rebuilt only for the blocks that entered its
        window; its full row before the last window is compared with nulls at C speed, and
        a compact row of --decode-fastpath2 (``window_rows.WindowRow``) is null there by
        construction."""
        import numpy as np
        o = rec.get("observed")
        a = rec["assignment"]
        kind = ("decode" if start >= a.prompt_tokens else "suffix" if a.local_turn else "attention")
        if (type(o) is not _WindowObserved or native_rows is None or o.layout is not rec["layout"] or
                o.generation != rec["generation"] or o.kind != kind or
                (a.local_turn and not rec.get("window_head_verified"))):
            return None
        fast2 = self.decode_fastpath2
        tables = runner.input_batch.block_table.block_tables
        grown = []
        for group in self.attention_groups:
            fa, table = o.rows[group], tables[group]
            r = fa.ratio
            if (hasattr(table, "get_row") or int(getattr(table, "blocks_per_kv_block", 1)) != r or
                    int(table.block_size) != fa.kernel_tokens):
                return None
            ids, k = request.block_ids[group], len(fa.ids)
            if type(ids) is not list or len(ids) < k or (
                    ids != fa.ids if fast2 and len(ids) == k else ids[:k] != fa.ids):
                return None
            new = ids[k:]
            if new and not all(type(b) is int and b > 0 for b in new):
                return None
            n = int(table.num_blocks_per_row[row])
            device = table.block_table.np
            if n != len(ids) * r or not np.array_equal(device[row, :k * r], fa.kernel):
                return None
            tail = device[row, k * r:n]
            if new and [int(v) for v in tail] != [b * r + i for b in new for i in range(r)]:
                return None
            pairs = native_rows[group]
            if len(pairs) != len(ids) or (pairs is not fa.pairs and pairs[:k] != fa.pairs):
                return None
            if new and [b for b, _ in pairs[k:]] != new:
                return None
            grown.append((fa, group, new, tail, pairs, [i for _, i in pairs[k:]] if new else []))
        generation = rec["generation"]
        P, W = self.page_tokens, self.window_tokens
        reads, needs = max(0, start - W + 1) // P, -(-(start + count) // P)
        bound = max_kept_blocks(W, P) + -(-count // P)
        windows, left, entered = {}, [], []
        for group in self.hybrid_groups:
            first0, pairs0, blocks0, leases0, keys0 = o.windows[group]
            nat = native_rows[group]
            n = len(nat)
            if type(nat) is WindowRow:
                first, live = nat.first, nat.live      # --decode-fastpath2: null before its window
            else:
                if type(nat) is not tuple or n < first0 or nat[:first0] != _nulls(first0):
                    return None
                first = first0
                while first < n and nat[first][0] == 0:
                    first += 1
                live = nat[first:]
            drop = first - first0
            kept = len(pairs0) - drop
            if (drop < 0 or kept < 0 or not live or len(live) < kept or len(live) > bound or
                    first > reads or n < needs or (kept and live[:kept] != pairs0[drop:])):
                return None
            added = live[kept:]
            if added and not all(b > 0 for b, _ in added):
                return None
            window = blocks0[drop:] + [b for b, _ in added] if (drop or added) else blocks0
            runner_ids = request.block_ids[group]
            if type(runner_ids) is not list or len(runner_ids) != n or runner_ids[first:] != window:
                return None
            if drop or added:
                fresh = tuple(Allocation(group, b, generation, "sliding", i) for b, i in added)
                windows[group] = (first, pairs0[drop:] + tuple(added), window, leases0[drop:] + fresh,
                                  keys0[drop:] + tuple(v.key for v in fresh))
                left.extend(leases0[:drop])
                entered.extend(fresh)
            else:
                windows[group] = o.windows[group]
        # Every check passed: extend the record (nothing below refuses).
        pools = self.pool_of
        fresh_fa = [(pools[group], b) for _, group, new, _, _, _ in grown for b in new]
        gone = [(pools[v.group], v.block) for v in left]
        came = [(pools[v.group], v.block) for v in entered]
        o.window_ids.difference_update(gone)
        unique = (o.unique and len(set(fresh_fa)) == len(fresh_fa) and o.fa_ids.isdisjoint(fresh_fa) and
                  len(set(came)) == len(came) and o.window_ids.isdisjoint(came))
        o.window_ids.update(came)
        o.fa_ids.update(fresh_fa)
        added_keys = []
        for fa, group, new, tail, pairs, incarnations in grown:
            fa.pairs = pairs
            if not new:
                continue
            allocs = tuple(Allocation(group, b, generation, kind, i) for b, i in zip(new, incarnations))
            keys = tuple(v.key for v in allocs)
            fa.ids.extend(new)
            fa.ids_tuple += tuple(new)
            fa.kernel = np.concatenate([fa.kernel, np.asarray(tail, dtype=fa.kernel.dtype)])
            fa.allocs += allocs
            fa.keys += keys
            added_keys.extend(keys)
        base = o.step
        if added_keys:
            o.fa_attention = sum((o.rows[g].allocs for g in self.attention_groups), ())
            o.fa_keys = sum((o.rows[g].keys for g in self.attention_groups), ())
            rec["last_tables"] = {g: (o.rows[g].ids_tuple, o.rows[g].kernel_tokens * o.rows[g].ratio)
                                  for g in self.attention_groups}
        hybrid = (base.hybrid if not (left or entered) else
                  sum((windows[g][3] for g in self.hybrid_groups), ()))
        if added_keys or hybrid is not base.hybrid:
            attention = o.fa_attention if added_keys else base.attention
            current = o.fa_keys + sum((windows[g][4] for g in self.hybrid_groups), ())
        else:
            attention, current = base.attention, base.current
        o.windows, o.unique = windows, unique
        o.added = tuple(added_keys) + tuple(v.key for v in entered)
        o.removed = tuple(v.key for v in left)
        # A block that left a window at its front may come back at its end in the same step
        # (vLLM frees before it allocates): its claim key stays, as the full passes keep it.
        stays = {(v.group, v.block) for v in left} & {(v.group, v.block) for v in entered}
        o.claim_added = [(group, b) for _, group, new, _, _, _ in grown for b in new] + \
            [(v.group, v.block) for v in entered if (v.group, v.block) not in stays]
        o.claim_removed = [(v.group, v.block) for v in left if (v.group, v.block) not in stays]
        s = StepRow(rid, row, offset, start, count, attention, hybrid, generation, current)
        o.base, o.step = base, s
        return s

    def _allocation_update(self, rec, payload):
        """--decode-fastpath2 (``window_rows``): an update of a request's changed rows comes as a
        delta of the rows the worker accepted last (window_rows.apply_delta); a full update
        (a request's first) is checked as before and its sliding rows kept compact."""
        if not self.window_rows:
            return super()._allocation_update(rec, payload)
        rows = payload['rows']
        if type(rows) is not dict:
            command = super()._allocation_update(rec, payload)
            if rows is None:
                return command

            def full(backend, qwen):
                receipt = command(backend, qwen)
                from .window_rows import compact_rows
                compact_rows(self, rec)
                return receipt
            return full

        def update(backend, qwen):
            from .chunk_protocol import worker_interval
            from .window_rows import apply_delta
            native, commit = apply_delta(self, rec, rows)
            worker_interval(rec, qwen.requests[rec['identity'].request], payload['start'], payload['count'])
            rec['native_rows'] = native
            rec['native_interval'] = (payload['start'], payload['count'])
            commit()
            return dict(operation='allocation_update', binding=payload['binding'], drained=True)
        return update

    def verify_allocations(self, identity, slots, generation):
        """--decode-fastpath: a step's sliding leases are one tuple as long as the window does not
        move; the tuple verified at the same allocation epoch (no key left the request's current
        allocations since) is that verification's answer again."""
        rec = self.records.get(identity.request)
        if self.window_fast and rec is not None:
            cached = rec.get("verified_slots")
            if (cached is not None and cached[0] is slots and cached[1] == rec.get("allocations_epoch") and
                    rec["identity"] == identity and rec["generation"] == generation):
                return
        super().verify_allocations(identity, slots, generation)
        if self.window_fast and rec is not None and type(slots) is tuple:
            rec["verified_slots"] = (slots, rec.get("allocations_epoch"))

    def materialize(self, s):
        """The first step's full-attention zero and restores, as ``RunnerAllocationOwner``
        does them; sliding blocks are never zeroed."""
        rec = self.records[s.request]
        a, tables, current = rec["assignment"], rec["last_tables"], s.current
        if not rec.get("attention_ready", False):
            session = self.sessions[rec["namespace"]]
            keep = rec["retained_blocks"] | rec["handover_blocks"]
            fresh = {group: tuple(b for b in tables[group][0] if b not in keep)
                     for group in self.attention_groups}
            require(not {b for v in fresh.values() for b in v} & self._pinned(session),
                    "zero set touches a resident or tail block")
            if any(fresh.values()):
                require(all(fresh.values()), "zero set must cover every FullAttention group")
                self.io.zero(fresh, "attention")
            restore = rec["plan"]["restore"]
            require(all(page < a.candidate_end // a.page_tokens and page in session["catalog"]
                        for page in restore), "restored page missing from worker DRAM")
            if self.batched_pages:
                if restore:
                    self.io.restore_pages([(rec["target"][page], session["catalog"][page]) for page in restore])
            else:
                for page in restore:
                    self.io.restore(rec["target"][page], "attention", session["catalog"][page])
            rec["attention_ready"] = True
        o = rec.get("observed") if self.window_fast else None
        if o is not None and o.step is s and o.base is not None and o.materialized is o.base and \
                "current_allocations" in rec:
            # --decode-fastpath: the sets hold the last step's keys; apply this step's differences.
            rec["allocations"].update(o.added)
            if o.removed:
                rec["current_allocations"].difference_update(o.removed)
                rec["allocations_epoch"] = rec.get("allocations_epoch", 0) + 1
            rec["current_allocations"].update(o.added)
        else:
            rec["allocations"].update(current)
            rec["current_allocations"] = set(current)
            if self.window_fast:
                rec["allocations_epoch"] = rec.get("allocations_epoch", 0) + 1
        if o is not None and o.step is s:
            o.materialized = s


class _WindowObserved(_Observed):
    """``RunnerAllocationOwner``'s observation record with each sliding group's live window
    at the last observation: (first index, its (block, incarnation) pairs, its block ids,
    its leases, their keys); the ids of every window, and the full-attention leases and
    keys of the step (``fa_attention``, ``fa_keys``)."""
    __slots__ = ("windows", "window_ids", "fa_attention", "fa_keys")


_NULLS = {}


def _nulls(n):
    """((0, 0),) * n, kept per length: a native row's null entries before a window."""
    value = _NULLS.get(n)
    if value is None:
        if len(_NULLS) > 64:
            _NULLS.clear()
        value = _NULLS[n] = ((0, 0),) * n
    return value


def attach_fastpath(owner, qwen):
    """--decode-fastpath on the window layout (window_worker, after decode_fastpath.attach_worker):
    the incremental observation, claims and allocation sets of the window owner, and the
    window lifecycle's reuse of an unchanged step's sliding leases."""
    require(type(owner).__name__ == "WindowAllocationOwner" and not owner.window_fast and
            getattr(qwen, "state_kind", None) == "sliding", "window decode fast path attachment collision")
    owner.window_fast = True
    qwen.reuse_state_leases = True


def attach_fastpath2(owner):
    """--decode-fastpath2 on the window layout (window_worker, after decode_fastpath2.attach_worker
    and attach_fastpath): allocation updates of changed rows as deltas, compact sliding rows."""
    require(owner.window_fast and owner.decode_fastpath2 and not owner.window_rows,
            "window decode fast path 2 needs the window fast path and attaches once")
    owner.window_rows = True
