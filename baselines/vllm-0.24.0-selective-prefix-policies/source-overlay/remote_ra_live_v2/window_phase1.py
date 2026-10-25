"""Engine phase 1 of the window layout (Gemma 4 31B IT, gemma-s1 WP3).

``Phase1Coordinator`` runs phase 1 of [C, e) on the resident set with null-padded Mamba
rows that receive the committed GDN carry. The window layout replaces the Mamba rows by
sliding rows (``sliding_window.phase1_row``):

* ``[0] * head + window + fresh``: the null head, the session's pinned window after C
  (``WindowPins.window``; borrowed, never zeroed or freed here, its last block may be
  partial and phase 1 appends into it) and fresh pool-1 blocks for [ceil(C/P)*P, e);
* the Goal forward's sliding rows are cold rows of its own fresh blocks, at most the
  layout's Goal transient (``WindowLayout.goal_tokens``, gemma-goal): pool 1 reserves
  them next to phase 1's transient, since both are held until the joined drain;
* at the joined drain the Goal blocks and the fresh sliding blocks that left the window
  after e are freed; the rest of the fresh blocks wait for adoption;
* ``adopt`` hands the window phase 1 ended in (``kept_blocks(e)``: pinned blocks from
  its first index on plus the fresh ones) to the phase-2 reservation
  (``WindowPins.handover``; the adapter installs it as phase 2's sliding head) and
  frees the older pinned blocks; ``release`` (abort) frees the waiting fresh blocks
  (the session is poisoned and its window freed at close).

The full-attention part (resident Pages, tail, fresh run blocks, SparsePlan, the Goal's
FA rows) is ``Phase1Coordinator``'s, line for line.

Chunked phase 1 (gemma-goal, design WP10): a run [C, e) longer than one native forward
runs as forwards of at most ``chunk_tokens`` (model_profile.GEMMA4_PHASE1_CHUNK_TOKENS)
on one lease (``sliding_window.phase1_chunks``). The full-attention row holds the whole
run from the first forward on; each sliding row holds the window of its forward and its
new tokens, and the fresh blocks of tokens that left the window are reused for later
tokens, so phase 1 holds at most ``sliding_window.phase1_ring`` fresh blocks per group.
The lease names the later forwards in ``chunks`` (``contract.parse_lease``); the worker
runs them in order (``native_probe``).
"""
from __future__ import annotations

from hashlib import sha256
import secrets

from remote_ra_native_growing.contract import prompt_digest

from .contract import ZERO_DIGEST, block_key, digest, group_pools, hex64, parse_lease, plain_copy, require
from .phase1_coordinator import Phase1Coordinator
from .sliding_window import kept_blocks, phase1_chunks, phase1_ring, phase1_row


class WindowPhase1Coordinator(Phase1Coordinator):

    def __init__(self, core, proxy, renderer, *, max_tokens=None):
        super().__init__(core, proxy, renderer, max_tokens=max_tokens)
        self.window = getattr(self.adapter, 'window', None)
        require(self.window is not None and getattr(self.layout, 'state', None) == 'sliding',
                'window phase 1 needs the window layout and its window pins')
        # Pool 1 holds one forward's transient of N tokens beyond the window slots.
        require(self.max_tokens <= self.layout.forward_tokens,
                'phase-1 forward exceeds the forward tokens of the window layout')
        # goal_query (gemma-goal): the Goal forward's own transient of G tokens (0: no
        # Goal forward on this server); the capacity gate counts G for it.
        self.goal_tokens = self.layout.goal_tokens
        # Chunked phase 1 (gemma-goal): forwards of at most this many tokens when the run
        # exceeds one native forward.
        from .model_profile import GEMMA4_PHASE1_CHUNK_TOKENS
        self.chunk_tokens = min(GEMMA4_PHASE1_CHUNK_TOKENS, self.max_tokens)

    def groups(self):
        """(FullAttention group ids, sliding-window group ids, group count)."""
        P = self.adapter.page_tokens
        specs = [g.kv_cache_spec for g in self.adapter.kv_manager.kv_cache_config.kv_cache_groups]
        kinds = tuple(type(spec).__name__ for spec in specs)
        require(set(kinds) == {'FullAttentionSpec', 'SlidingWindowSpec'} and
                all(spec.block_size == P for spec in specs), 'unsupported phase-1 group layout')
        att = tuple(i for i, kind in enumerate(kinds) if kind == 'FullAttentionSpec')
        win = tuple(i for i, kind in enumerate(kinds) if kind == 'SlidingWindowSpec')
        require(len(att) == len(self.adapter.managers) and win == self.window.groups,
                'native attention or sliding manager order mismatch')
        return att, win, len(kinds)

    def run(self, binding, geo, *, allocation_generation, goal_payload=None, _lease_only=False):
        adapter = self.adapter
        adapter.authority.owner()
        P, W = adapter.page_tokens, self.layout.window_tokens
        s = adapter.sessions[binding.session]
        require(geo.local_turn > 0 and geo.page_tokens == P, 'phase 1 runs only after local turn 0')
        require(not s.poisoned and s.pending is None and s.local_turn == geo.local_turn and
                s.turn == binding.turn and binding.request not in adapter.requests,
                'phase 1 must precede the generation reserve')
        connector = self.core.scheduler.connector
        require(connector is not None and not connector.has_pending_push_work(),
                'native connector must be quiescent before phase-1 allocation')
        require(not self.records, 'prior phase-1 lease lacks a drain, adoption or abort')
        # A run longer than one native forward runs in chunks (gemma-goal).
        plan = phase1_chunks(geo.C, geo.e, self.max_tokens, self.chunk_tokens)
        require(type(allocation_generation) is int and allocation_generation > 0,
                'worker allocation generation required')
        require(hex64(s.boundary) and s.boundary != ZERO_DIGEST, 'phase 1 needs a committed window')
        owner, ns = adapter.owner, s.namespace
        resident = owner.sessions.get(ns, {})
        R = tuple(sorted(resident))
        pin = owner.tails.get(ns)
        require(R == tuple(s.resident) and all(p < geo.Q // P for p in R) and
                (pin is not None) == geo.tail and
                (pin is None or (pin.page, pin.start, pin.valid_end) == (geo.Q // P, geo.Q, geo.C)),
                'phase-1 geometry differs from the session pins')
        att, win, count = self.groups()
        head, (b0, b1), (f0, f1) = phase1_row(geo.C, geo.e, W, P)
        pinned = self.window.window(ns)
        require(pinned is not None and
                all([i for i, _ in pinned[g]] == list(range(b0, b1)) for g in win),
                'phase-1 window differs from the session window pin')
        goal_tokens = None
        if goal_payload is not None:
            require(s.policy == 'goal_query' and type(goal_payload) is str,
                    'a Goal forward belongs to goal_query only')
            from runtime_renderer import PLAIN_GOAL_SCHEMA
            goal_tokens = [int(t) for t in self.renderer(dict(schema=PLAIN_GOAL_SCHEMA,
                                                             text=goal_payload)).token_ids]
            require(0 < len(goal_tokens) <= self.max_tokens,
                    'PHASE1_TOO_LONG: Goal forward exceeds one native forward')
            # Its sliding blocks come out of the layout's Goal transient (gemma-goal).
            require(len(goal_tokens) <= self.goal_tokens,
                    'PHASE1_TOO_LONG: Goal forward of %d tokens exceeds the Goal transient of the '
                    'window layout (%d tokens)' % (len(goal_tokens), self.goal_tokens))
        fresh = list(geo.fresh_pages)
        width = -(-len(goal_tokens) // P) if goal_tokens else 0
        ring = phase1_ring(geo.C, plan, W, P)
        fa, sliding = len(att) * (len(fresh) + width), len(win) * (ring + width)
        require(fa <= self.pool.get_num_free_blocks() and sliding <= self.mamba_pool.get_num_free_blocks(),
                'insufficient native blocks for phase 1: FA %d of %d free, sliding %d of %d free' %
                (fa, self.pool.get_num_free_blocks(), sliding, self.mamba_pool.get_num_free_blocks()))
        from .pool_incarnations import install
        tables = (install(self.pool), install(self.mamba_pool))
        allocated = []

        def new(n, pool=self.pool):
            blocks = pool.get_new_blocks(n)
            allocated.extend(blocks)
            return list(blocks)
        try:
            fresh_fa = {g: new(len(fresh)) for g in att}
            ring_win = {g: new(ring, self.mamba_pool) for g in win}
            goal_fa = {g: new(width) for g in att} if goal_tokens else {}
            goal_win = {g: new(width, self.mamba_pool) for g in win} if goal_tokens else {}
        except BaseException:
            self._free(allocated)
            raise
        try:
            # Each forward's sliding rows: the window it reads and its new tokens; fresh
            # blocks of tokens before the window are reused, first in first out.
            borrowed_win = {g: dict(pinned[g]) for g in win}
            forwards, live = [], {g: {} for g in win}
            spare = {g: list(ring_win[g]) for g in win}
            for start, stop in plan:
                h, n = kept_blocks(start, W, P)[0], -(-stop // P)
                rows_now = {}
                for g in win:
                    for i in sorted(i for i in live[g] if i < h):
                        spare[g].append(live[g].pop(i))
                    row = [0] * h
                    for i in range(h, n):
                        if i < f0:
                            row.append(borrowed_win[g][i].block_id)
                            continue
                        if i not in live[g]:
                            live[g][i] = spare[g].pop(0)
                        row.append(live[g][i].block_id)
                    rows_now[g] = row
                forwards.append((start, stop, rows_now))
            tail_ids = [pin.groups[i][0].block_id for i in range(len(att))] if pin else []
            rows = []
            for g in range(count):
                if g in att:
                    i = att.index(g)
                    rows.append([resident[p][i][0].block_id for p in R] + tail_ids[i:i + 1] +
                                [b.block_id for b in fresh_fa[g]])
                else:
                    rows.append(forwards[0][2][g])
            pools = group_pools(att, count, self.layout)
            chunks = [[start, stop, [rows_now[g] for g in win]] for start, stop, rows_now in forwards[1:]]
            ids = [(pools[g], b) for g, row in enumerate(rows) for b in row if b]
            # Blocks a later forward uses first: owned after the first forward's, in order.
            seen, first_use = set(ids), []
            for start, _, rows_now in forwards[1:]:
                for g in win:
                    for b in rows_now[g][-(-start // P):]:
                        if (pools[g], b) not in seen:
                            seen.add((pools[g], b))
                            first_use.append(b)
                            ids.append((pools[g], b))
            require(all(b in tables[pool] for pool, b in ids), 'phase-1 block allocation incarnation missing')
            fa_owned = [b.block_id for g in att for b in fresh_fa[g]]
            owned = fa_owned + [b for g in win for b in rows[g][f0:]] + first_use
            goal = None
            if goal_tokens:
                grows = [[b.block_id for b in (goal_fa[g] if g in att else goal_win[g])] for g in range(count)]
                gids = [(pools[g], b) for g, row in enumerate(grows) for b in row]
                goal = dict(lease_id='goal-' + secrets.token_hex(16), tokens=goal_tokens, rows=grows,
                            owned=[b for _, b in gids], zero=[b for pool, b in gids if pool == 0],
                            incarnations={block_key(pool, b): tables[pool][b] for pool, b in gids},
                            payload_sha256=sha256(goal_payload.encode()).hexdigest())
            lease = dict(lease_id='phase1-' + secrets.token_hex(16), request=binding.request,
                binding=binding.identity, session=binding.session, namespace=ns,
                generation=allocation_generation, epoch=binding.epoch,
                source=adapter.authority.source, local_turn=geo.local_turn, page_tokens=P,
                start=geo.C, end=geo.e, sealed_start=geo.Q, candidate_end=geo.L,
                resident_pages=list(R),
                tail=(dict(page=pin.page, valid_end=pin.valid_end, block_ids=[[b] for b in tail_ids])
                      if pin else None),
                tail_candidate=geo.tail_candidate, rows=rows,
                borrowed=[resident[p][i][0].block_id for p in R for i in range(len(att))] + tail_ids,
                owned=owned, zero=fa_owned,
                run_fresh={str(p): [[fresh_fa[g][k].block_id] for g in att] for k, p in enumerate(fresh)},
                incarnations={block_key(pool, b): tables[pool][b] for pool, b in ids},
                sparse=dict(logical_matched_tokens=geo.Q,
                            dropped_full_attention_tokens=geo.Q - len(R) * P),
                carry_digest=s.boundary, goal=goal)
            if chunks:
                lease['chunks'] = chunks
            lease['mac'] = adapter.authority.sign(lease)
            parse_lease(lease, attention_groups=att, layout=self.layout)
            request = dict(request=binding.request, binding=binding.identity,
                session=binding.session, turn=binding.turn, local_turn=geo.local_turn,
                generation=binding.generation, allocation_generation=allocation_generation,
                epoch=binding.epoch, source=adapter.authority.source, model=binding.model,
                policy=s.policy, tokens=list(binding.tokens[:geo.e]),
                prompt_digest=prompt_digest(binding), lease=lease, lease_digest=digest(lease))
            request['request_digest'] = digest(request)
        except BaseException:
            self._free(allocated)
            raise
        # Fresh sliding blocks before the window after e leave at the drain; the
        # others wait for adoption (with the pinned ones from that index on).
        k0, _ = kept_blocks(geo.e, W, P)
        stay = {g: tuple((i, b) for i, b in sorted(live[g].items()) if i >= k0) for g in win}
        leave = {g: [b for b in ring_win[g] if all(b is not kept for _, kept in stay[g])] for g in win}
        self.records[binding.request] = dict(binding=binding.identity, lease=plain_copy(lease),
            request=plain_copy(request), drained=False,
            fa={p: tuple((fresh_fa[g][k],) for g in att) for k, p in enumerate(fresh)},
            transient=tuple(b for g in win for b in leave[g]) +
                      tuple(b for g in att for b in goal_fa.get(g, ())) +
                      tuple(b for g in win for b in goal_win.get(g, ())),
            window=dict(namespace=ns, first=k0, fresh=stay))
        query_tokens = (goal_tokens if goal_tokens else
                        list(binding.tokens[geo.o:geo.e]) if s.policy == 'observation_query' else None)
        if _lease_only:
            return request, query_tokens, s, goal_tokens
        try:
            result = self.proxy.call('phase1', request, query_tokens=query_tokens)
            return self._result(result, binding, geo, s, goal_tokens)
        finally:
            self.recover(binding.request)

    def held(self, request_id):
        """(FA, sliding) blocks ``request_id``'s lease holds now (capacity ledger)."""
        record = self.records.get(request_id)
        if record is None:
            return 0, 0
        fa = sum(len(groups) for groups in record['fa'].values())
        sliding = sum(len(v) for v in record['window']['fresh'].values())
        if not record['drained']:
            transient = list(record['transient'])
            sliding += sum(1 for b in transient if self._owns(self.mamba_pool, b))
            fa += sum(1 for b in transient if self._owns(self.pool, b))
        return fa, sliding

    @staticmethod
    def _owns(pool, b):
        return b.block_id < len(pool.blocks) and pool.blocks[b.block_id] is b

    def adopt(self, binding):
        """The FA run blocks (``Phase1Coordinator.adopt``) and the window phase 1 ended in,
        handed to the window pins for the phase-2 reservation."""
        record = self.records.get(binding.request)
        require(record is not None and record['binding'] == binding.identity and record['drained'],
                'adoption requires a drained phase-1 lease')
        window = record['window']
        self.window.handover(window['namespace'], binding.request, window['first'], window['fresh'])
        return super().adopt(binding)

    def release(self, request_id, binding=None):
        """Abort: the un-adopted fresh FA blocks (``Phase1Coordinator.release``) and the
        fresh sliding blocks that waited for adoption."""
        record = self.records.get(request_id)
        if record is None:
            return
        require(binding is None or record['binding'] == binding, 'foreign phase-1 abort')
        self.recover(request_id)
        blocks = [b for v in record['window']['fresh'].values() for _, b in v]
        require(all(b.ref_cnt == 1 for b in blocks), 'phase-1 window blocks changed owner before abort')
        self.mamba_pool.free_blocks(reversed(blocks))
        record['window']['fresh'] = {}
        super().release(request_id, binding)
