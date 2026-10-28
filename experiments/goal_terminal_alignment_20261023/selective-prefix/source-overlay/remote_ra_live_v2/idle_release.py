"""Route B capacity: release idle sessions' GPU Pages under pressure (serve.py --release-idle-pages).

Between turns a session pins its resident Pages R (the last turn's selected Pages
plus the Pages sealed in it) and its live tail; it holds no GDN slot (the carry is
on the host) and every sealed Page also has its DRAM copy in the worker's catalog
(the pinned arena). Sixteen sessions at a 32K budget pin more than a 25-32 GiB pool,
and with every session idle nothing would ever free a block. With this switch the
capacity gate (``capacity.admission_gate``) frees GPU Pages instead of waiting:

- **Victims.** When a turn does not fit, the FullAttention shortfall of the gate's
  inequalities is covered by releasing whole idle sessions' R, least recently used
  first (the ordinal of their last completion or readmission). Idle means open, not
  poisoned, no native request, no phase-1 record, no lifecycle request in flight
  (which excludes a running request, an admission parked at its Goal or in flight
  under --admission-overlap) and not the admitting session. The tail is never
  released: its [Q, C) KV has no DRAM copy and cannot be recomputed without the GDN
  state at Q. If the idle sessions together cannot cover the shortfall nothing is
  released and the turn waits (CAPACITY_WAIT) as before; a turn admitted while
  nothing runs (with --admission-overlap: and no admission is in flight) first
  releases what it needs.
- **Eviction** (engine thread, between steps): the worker drops the session's
  resident mirror (``evict``; the catalog, tail and carry stay), then the engine
  frees the blocks (``PrivateBlockOwner.evict``) and the session remembers the
  released Pages (``Session.evicted``).
- **Readmission.** The evicted session's next turn restores exactly those Pages
  before its phase 1 (``readmit`` in ``SplitPrivateEndpoint.reserve_steps``): fresh
  pinned blocks, one batched H2D from the catalog, the mirrors back to R. Phase 1,
  selection and phase 2 then run as if the Pages had never left; the capacity gate
  counts the readmitted blocks in both peaks (``capacity.turn_need_by_pool``).

Nothing waits inside an admission: a release is synchronous, a refusal is the
usual immediate CAPACITY_WAIT, so the 300 s ACK covers only an admitted turn's own
work. A turn needs at most one session's worth of blocks (the launcher refuses a
pool that cannot hold one session at the context limit), so releasing every idle
session always makes room once the running requests finished.

The window layout (Gemma 4, gemma-s1) releases the same global Pages (pool 0); the
sessions' sliding windows stay pinned in pool 1 (W-GPU) and phase 1 borrows them as
before. Its launcher may size pool 0 by the selective rule
(``window_capacity.selective_pages``: one selective turn, not one session at the
context limit), so a first turn or a rebuild may need more than pool 0 can ever
free: ``ceiling`` refuses such a turn (CAPACITY_EXCEEDED, not retried) instead of
letting it wait forever.

Off (the default): no startup config key, worker startup record key, evidence key
or vLLM argument, and none of this runs. On: startup.json pins
``release_idle_pages: true`` (decoupled KV layout only); ``compose_engine`` sends it
once in the worker startup record, the worker checks it against the pinned config,
accepts ``evict``/``readmit`` only then and echoes it in its evidence; ready.json
records it. Evidence: ``<output>/capacity.jsonl`` (one row per eviction and per
readmission), the native terminal file's ``idle_release`` record (the sessions
released to admit the request, its readmitted Pages) and engine-cleanup.json's
totals.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

from .contract import LiveV2Error, require

CONFIG_KEY = 'release_idle_pages'   # startup config, worker startup record, worker evidence, ready.json
EVENTS = 'capacity.jsonl'
SCHEMA = 'remote-ra-capacity-event-v1'
CAPACITY_EXCEEDED = 'CAPACITY_EXCEEDED'


class CapacityExceeded(LiveV2Error):
    """The window layout: a turn that no release or completion can ever make room for
    (``IdleRelease.ceiling``). Not a CAPACITY_WAIT: the frontend does not retry it."""


def pinned(config):
    """True when a startup config pins the switch, else None (the key is absent)."""
    if config is None or CONFIG_KEY not in config:
        return None
    require(config[CONFIG_KEY] is True, 'pinned release_idle_pages must be true; off is pinned by leaving it out')
    require(config.get('kv_layout') is not None, 'release_idle_pages needs the decoupled KV layout')
    return True


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def worker_flag(startup):
    """Worker startup: the record the engine sent, checked once against the pinned config."""
    sent = startup.get(CONFIG_KEY)
    require(sent is None or sent is True, 'release_idle_pages worker record must be true or absent')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker release_idle_pages differs from the pinned startup config')
    return sent


def attached(lifecycle):
    """The engine's IdleRelease, or None (off; also a lifecycle without an engine core)."""
    core = getattr(getattr(lifecycle, 'endpoint', None), 'core', None)
    return getattr(core, '_remote_ra_idle_release', None)


def shortfall(lifecycle, need):
    """(FA blocks missing, whether another pool is short) of an assessed turn under the
    gate's inequalities (``capacity.admission_gate``/``ledger_gate``): phase 1 counts the
    in-flight reservations and, when admissions decode on while one is parked, the
    running requests' headroom; phase 2 the headroom and the watermark. The window
    layout's sliding pool under the ledger: as ``capacity.ledger_gate`` (ledger_phase1)."""
    from .capacity import goal_parks, ledger_phase1, reserved
    ledger = getattr(lifecycle, 'reservations', None) is not None
    pools = 'pools' in need
    pair = lambda value: tuple(value) if pools else (value,)
    zero = (0, 0) if pools else (0,)
    transient = []
    ahead = pair(reserved(lifecycle, transient=transient)[0]) if ledger else zero
    free, phase2 = pair(need['free']), pair(need['phase2'])
    phase1 = pair(ledger_phase1(need, transient) if ledger else need['phase1'])
    room, watermark = pair(need['headroom']), pair(need['watermark'])
    soon = pair(need['releasing']) if 'releasing' in need else zero
    later = room if ledger or goal_parks(getattr(lifecycle, 'goals', None)) else zero
    shorts = [max(phase1[i] + ahead[i] + later[i] - free[i],
                  phase2[i] + ahead[i] + room[i] + watermark[i] - free[i] - soon[i])
              for i in range(len(free))]
    return max(0, shorts[0]), any(s > 0 for s in shorts[1:])


class IdleRelease:
    """The engine's releases and readmissions (compose_engine attaches it as
    ``core._remote_ra_idle_release``). Engine thread only."""

    def __init__(self, core, output=None):
        self.core = core
        self.used, self.clock = {}, 0           # session id -> ordinal of its last use
        self.requests = {}                       # request id -> its idle_release evidence
        self.totals = dict(evictions=0, evicted_pages=0, readmissions=0, readmitted_pages=0)
        self.events = None if output is None else Path(output) / EVENTS

    # ---- bookkeeping ---------------------------------------------------------------------
    def touch(self, session_id):
        """A completed turn or a readmission: the session is the most recently used."""
        self.clock += 1
        self.used[session_id] = self.clock

    def groups(self):
        """FullAttention groups: blocks per Page."""
        return len(self.core._remote_ra_private_v2_endpoint.phase1.groups()[0])

    def _write(self, value):
        if self.events is None:
            return
        from remote_ra_http.config import encoded
        with self.events.open('ab') as stream:
            stream.write(encoded(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _free(self):
        from .session_pool import free_blocks
        return free_blocks(self.core)

    def record(self, request_id):
        """The native terminal file's ``idle_release`` record of one request."""
        return dict(self.requests.get(request_id) or dict(evicted_sessions=[], evicted_pages=0,
                                                          readmitted_pages=[]))

    def _note(self, request_id):
        return self.requests.setdefault(request_id, dict(evicted_sessions=[], evicted_pages=0,
                                                         readmitted_pages=[]))

    def summary(self):
        return dict(self.totals)

    # ---- victims -------------------------------------------------------------------------
    def idle(self, lifecycle, exclude):
        """Idle sessions holding resident Pages, least recently used first."""
        core = self.core
        adapter, endpoint = core._remote_ra_growing_adapter, core._remote_ra_private_v2_endpoint
        authority = core._remote_ra_live_session_authority
        busy = {s.namespace for s in adapter.requests.values()}
        busy |= {lease.namespace for lease in adapter.owner.leases.values()}
        out = []
        for index, sid in enumerate(authority.ids):
            s = adapter.sessions.get(sid)
            state = lifecycle.sessions.get(sid, (0, None))[1]
            if (sid == exclude or sid not in authority.namespaces or s is None or s.poisoned or
                    s.pending is not None or not s.resident or s.namespace in busy or state is not None or
                    endpoint.phase1.pending(sid)):
                continue
            out.append((self.used.get(sid, 0), index, sid, s))
        return [(sid, s) for _, _, sid, s in sorted(out)]

    def victims(self, lifecycle, command, short):
        """The LRU prefix of idle sessions whose released blocks cover ``short``, or None."""
        A, freed, chosen = self.groups(), 0, []
        for sid, s in self.idle(lifecycle, command.session_id):
            if freed >= short:
                break
            chosen.append(sid)
            freed += A * len(s.resident)
        return chosen if freed >= short else None

    # ---- the gate --------------------------------------------------------------------------
    def admit(self, lifecycle, command, gate):
        """``gate`` (the capacity gate) with releases: a refused turn releases idle sessions
        covering its FA shortfall and is gated again; a turn admitted while nothing runs
        (the gate admits it unchecked) first releases what it needs, if it can.

        --admission-overlap: the ledger gate admits a turn unchecked when nothing runs and
        no admission is in flight, but returns its figures and records its reservation, so
        the release runs before that gate (integrate-1016: without it such a turn found its
        phase 1 short of blocks that idle sessions held)."""
        from .capacity import CapacityWait, assess, reserved
        if getattr(lifecycle, 'reservations', None) is not None and not (
                reserved(lifecycle)[1] or any(e['state'] == 'accepted' for e in lifecycle.entries.values())):
            self.make_room(lifecycle, command)
        try:
            need = gate(lifecycle, command)
        except CapacityWait:
            goals = getattr(lifecycle, 'goals', None)
            if getattr(lifecycle, 'reservations', None) is None and goals is not None and goals.parked:
                raise           # another admission waits for its Goal: no release helps
            assessed = assess(lifecycle, command)
            if assessed is None:
                raise
            self.ceiling(command, assessed)
            short, other = shortfall(lifecycle, assessed)
            chosen = None if other or not short else self.victims(lifecycle, command, short)
            if not chosen:
                raise
            self.release(chosen, command, short)
            return gate(lifecycle, command)
        if need is None:
            self.make_room(lifecycle, command)
        return need

    def make_room(self, lifecycle, command):
        """A turn the gate admits unchecked (nothing runs): release the idle sessions covering
        its FA shortfall, if they can; refuse it if no release ever could (``ceiling``)."""
        from .capacity import assess
        assessed = assess(lifecycle, command)
        if assessed is not None:
            self.ceiling(command, assessed)
            short, other = shortfall(lifecycle, assessed)
            chosen = None if other or not short else self.victims(lifecycle, command, short)
            if chosen:
                self.release(chosen, command, short)

    def ceiling(self, command, need):
        """The window layout (gemma-s1), whose launcher may size pool 0 by the selective
        rule (``window_capacity.selective_pages``) instead of one session at the context
        limit: refuse a turn that could never be admitted, i.e. whose FullAttention peak
        with its own pins (and the watermark) exceeds every block of pool 0 but the null
        block, the other sessions' tails and the pins of poisoned sessions (none of which
        a release or a completion frees). Waiting would never end. Qwen: nothing."""
        core = self.core
        adapter = core._remote_ra_growing_adapter
        if getattr(adapter, 'window', None) is None:
            return
        A, pool = self.groups(), adapter.owner.pool
        s = adapter.sessions.get(command.session_id)
        own = 0 if s is None else A * (len(s.resident) + (s.tail is not None))
        floor = sum(A * ((x.tail is not None) + (len(x.resident) if x.poisoned else 0))
                    for sid, x in adapter.sessions.items() if sid != command.session_id)
        pair = (lambda v: tuple(v)) if 'pools' in need else (lambda v: (v,))
        peak = max(pair(need['phase1'])[0], pair(need['phase2'])[0] + pair(need['watermark'])[0])
        most = pool.num_gpu_blocks - 1 - floor
        if own + peak > most:
            if own + peak <= pool.num_gpu_blocks - 1:
                # Another session's tail/poisoned Pages disappear on its explicit
                # close. Refuse this admission temporarily; do not poison another
                # otherwise feasible session while that cleanup is in flight.
                from .capacity import CapacityWait, CAPACITY_WAIT
                raise CapacityWait('%s: turn needs %d global Pages; %d other-session '
                                   'pins await session close (%d total usable)'
                                   % (CAPACITY_WAIT, own + peak, floor, pool.num_gpu_blocks - 1))
            raise CapacityExceeded('%s: the turn needs %d global Pages with its own %d pinned; at most %d can '
                                   'ever be free (pool 0 holds %d, other sessions pin %d that no release frees)'
                                   % (CAPACITY_EXCEEDED, own + peak, own, most, pool.num_gpu_blocks - 1, floor))

    def release(self, chosen, command, short):
        note = self._note(command.request_id)
        for sid in chosen:
            pages = self.evict(sid, command.request_id, short)
            note['evicted_sessions'].append(sid)
            note['evicted_pages'] += len(pages)

    # ---- one session -----------------------------------------------------------------------
    def evict(self, session_id, request_id, short):
        """Release one idle session's resident Pages (the worker mirror first)."""
        core = self.core
        adapter, authority = core._remote_ra_growing_adapter, core._remote_ra_live_session_authority
        s = adapter.sessions[session_id]
        pages = tuple(s.resident)
        require(pages and not s.evicted and tuple(sorted(adapter.owner.sessions.get(s.namespace, {}))) == pages,
                'evicted Pages differ from the session pins')
        before = self._free()
        proxy = core._remote_ra_live_v2_boundary.proxy
        receipt = proxy.call('evict', dict(session=session_id, namespace=s.namespace, epoch=authority.epoch,
                                           pages=list(pages)))
        require(receipt.get('operation') == 'evict' and receipt.get('drained') is True and
                receipt.get('namespace') == s.namespace and tuple(receipt.get('evicted', ())) == pages,
                'worker eviction receipt mismatch')
        released = adapter.owner.evict(s.namespace)
        require(tuple(sorted(released)) == pages, 'engine released other Pages than the worker')
        s.resident, s.evicted = (), pages
        self.totals['evictions'] += 1
        self.totals['evicted_pages'] += len(pages)
        self._write(dict(schema=SCHEMA, event='evict', session_id=session_id,
                         session_index=authority.index(session_id), pages=len(pages),
                         blocks=len(pages) * self.groups(), request_id=request_id, shortfall_blocks=short,
                         free_blocks_by_pool_before=before, free_blocks_by_pool=self._free(),
                         monotonic_ns=time.monotonic_ns()))
        return pages

    def readmit(self, binding, session):
        """Before an evicted session's phase 1: its released Pages back on the GPU."""
        core = self.core
        adapter, authority = core._remote_ra_growing_adapter, core._remote_ra_live_session_authority
        pages = tuple(session.evicted)
        require(pages and not session.resident and binding.local_turn > 0, 'nothing to readmit')
        before = self._free()
        try:
            pinned = adapter.owner.readmit(session.namespace, pages, self.groups())
            proxy = core._remote_ra_live_v2_boundary.proxy
            receipt = proxy.call('readmit', dict(request=binding.request, session=binding.session,
                turn=binding.turn, generation=binding.generation, local_turn=binding.local_turn,
                binding=binding.identity, namespace=session.namespace,
                pages={str(p): [[b.block_id for b in group] for group in groups] for p, groups in pinned.items()}))
            require(receipt.get('operation') == 'readmit' and receipt.get('binding') == binding.identity and
                    receipt.get('drained') is True and tuple(receipt.get('readmitted', ())) == pages,
                    'worker readmission receipt mismatch')
        except BaseException:
            # The worker may hold a partial restore: the session never serves again.
            session.poisoned = True
            raise
        session.resident, session.evicted = pages, ()
        self.touch(binding.session)
        self.totals['readmissions'] += 1
        self.totals['readmitted_pages'] += len(pages)
        self._note(binding.request)['readmitted_pages'] = list(pages)
        self._write(dict(schema=SCHEMA, event='readmit', session_id=binding.session,
                         session_index=authority.index(binding.session), pages=len(pages),
                         blocks=len(pages) * self.groups(), request_id=binding.request,
                         free_blocks_by_pool_before=before, free_blocks_by_pool=self._free(),
                         monotonic_ns=time.monotonic_ns()))
        return pages
