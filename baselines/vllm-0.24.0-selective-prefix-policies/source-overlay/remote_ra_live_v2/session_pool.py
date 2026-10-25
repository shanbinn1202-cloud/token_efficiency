"""Session pool (branch ms-pool): startup tasks opened lazily, closed explicitly.

A server started with ``--max-running-sessions K`` registers every startup task
(up to ``sessions.MAX_TASKS``, pinned in the startup config exactly like a
fixed task list) but keeps at most K sessions open. ``max_num_seqs`` is K, the
Mamba pool has 2K + 1 slots and the pinned Page arena is sized for K sessions.

- Open: the first command of a task's session (turn 0) opens it on the engine
  (``LiveSessionAuthority.open_session``) while fewer than K are open. The
  frontend never sends one beyond K: a request for an unopened task waits in
  the frontend's FIFO until a session closes (``controller.open_wait``).
- Close: the harness closes a task's session when its agent is done
  (``POST /v1/sessions/close``, loopback only). The frontend sends an
  authenticated ``CloseSession`` frame on the private channel and the engine
  runs ``close_session`` on its thread: the worker drops the session's CPU
  catalog (arena slots), committed GDN carry and Qwen state; the engine frees
  its resident Pages and live tail, its selection tables, Goal state and engine
  transcript, retires its namespace and lifecycle entry, and records a close
  receipt. A closed session never reopens: later requests for its task fail at
  once. A session with a request queued or in flight is refused (SESSION_BUSY).
- The capacity gate reads the pools' free blocks, so a closed session's blocks
  are available to the next admission; an unopened session's first turn is
  assessed as a turn 0 (``capacity.assess``).

A close also drops the host records its requests left behind (prompts, plans,
catalog copies, worker frames and terminal receipts, query receipts): they were
kept for the lifetime of the process (``release_requests``), so a long pool run
holds them for the open sessions only. Lifecycle entries and request tombstones
stay, slimmed, so request ids remain refused and the shutdown receipt lists
every admitted request.

Evidence: ``<output>/sessions.jsonl``, one row per open and per close, written
on the engine thread (pool counts are two integers read from the KV manager; no
hashing). Shutdown closes the sessions still open, as on any server, and
``engine-cleanup.json`` summarises the pool.
"""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import NamedTuple

from .contract import require

SCHEMA = 'remote-ra-session-event-v1'
TERMINAL = frozenset({'complete', 'cancelled', 'rejected'})


class ReleasedBinding(NamedTuple):
    """A closed session's lifecycle entry keeps its binding's identity, not its tokens."""
    request: str
    session: str
    turn: int
    generation: str
    identity: str


def configured(core):
    """K of the engine's session pool: ``core._remote_ra_session_pool`` as
    compose_engine set it, else the startup config's; None without a pool."""
    if hasattr(core, '_remote_ra_session_pool'):
        return core._remote_ra_session_pool
    from .composition import session_pool
    return session_pool()


def events_path():
    """Startup only: ``<output>/sessions.jsonl`` in the HTTP runtime, else None."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return Path(load()['output_directory']) / 'sessions.jsonl'


def free_blocks(core):
    """Free blocks per pool now: [FA, Mamba] (decoupled) or [shared] (legacy)."""
    adapter = core._remote_ra_growing_adapter
    manager = adapter.kv_manager
    if getattr(adapter, 'decoupled', False) is True:
        return list(manager.get_num_free_blocks_by_pool())
    return [manager.block_pool.get_num_free_blocks()]


def _write(authority, value):
    path = getattr(authority, 'events', None)
    if path is None:
        return
    from remote_ra_http.config import encoded
    with path.open('ab') as stream:
        stream.write(encoded(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def _event(core, authority, event, session_id, **fields):
    index = authority.index(session_id)
    return dict(schema=SCHEMA, event=event, session_id=session_id, session_index=index,
                instance_id=core._remote_ra_trusted_issues[index]['instance_id'],
                max_running_sessions=authority.limit, **fields)


def opened(core, session_id):
    """Evidence of one open (LiveSessionAuthority.open_session, engine thread)."""
    authority = core._remote_ra_live_session_authority
    value = _event(core, authority, 'open', session_id, open_sessions=len(authority.namespaces),
                   free_blocks_by_pool=free_blocks(core), monotonic_ns=time.monotonic_ns())
    _write(authority, value)
    return value


def close_session(lifecycle, session_id):
    """Close one registered session for good; returns its close receipt.

    Idempotent: closing a closed session returns the first receipt with
    ``repeated`` true. A never-opened session closes without touching the
    engine. An open one must be idle or poisoned with its native work drained.
    """
    core = lifecycle.endpoint.core
    authority = core._remote_ra_live_session_authority
    authority.verify()
    require(getattr(authority, 'limit', None) is not None,
            'SESSION_POOL_REQUIRED: session close needs a server started with --max-running-sessions')
    require(type(session_id) is str and session_id in authority.ids,
            'session not initialized by loaded server')
    prior = authority.ended.get(session_id)
    if prior is not None:
        return dict(prior, repeated=True)
    adapter = core._remote_ra_growing_adapter
    endpoint = core._remote_ra_private_v2_endpoint
    namespace = authority.namespaces.get(session_id)
    current = lifecycle.sessions.get(session_id)
    require(current is None or current[1] in (None, 'poisoned'),
            'SESSION_BUSY: session %s still owns a request' % session_id)
    fields = dict(reason='close', opened=namespace is not None)
    if namespace is not None:
        s = adapter.sessions[session_id]
        require(s.pending is None and not any(x is s for x in adapter.requests.values()) and
                not endpoint.phase1.pending(session_id),
                'SESSION_BUSY: session %s still owns native work' % session_id)
        before, pinned = free_blocks(core), len(adapter.owner.pinned(namespace))
        requests = [rid for rid, entry in lifecycle.entries.items() if entry['identity'][1] == session_id]
        require(all(lifecycle.entries[rid]['state'] in TERMINAL for rid in requests),
                'SESSION_BUSY: session %s has a request that is not terminal' % session_id)
        proxy = core._remote_ra_live_v2_boundary.proxy
        receipt = proxy.call('close_session', dict(session=session_id, namespace=namespace,
                                                   epoch=authority.epoch, reason='close',
                                                   requests=requests))
        require(receipt.get('operation') == 'close_session' and receipt.get('closed') is True and
                receipt.get('drained') is True and receipt.get('namespace') == namespace and
                receipt.get('reason') == 'close', 'worker session close not drained')
        adapter.owner.close_drained(namespace)      # frees resident Pages and the live tail
        adapter.sessions.pop(session_id)
        authority.namespaces.pop(session_id)
        goal = None
        if endpoint.selection is not None:
            goal = endpoint.selection.close_session(session_id, getattr(lifecycle, 'goals', None))
        transcript = endpoint.transcripts.get(session_id)
        if transcript is not None:
            transcript.release()
        release_requests(lifecycle, requests)
        poisoned = s.poisoned or (current is not None and current[1] == 'poisoned')
        fields.update(turns=s.turn, rebuilds=s.ns_generation, poisoned=poisoned, namespace=namespace,
                      released=dict(resident_pages=len(s.resident), tail=s.tail is not None,
                                    blocks=pinned, requests=len(requests), worker=receipt.get('released')),
                      free_blocks_by_pool_before=before, free_blocks_by_pool=free_blocks(core),
                      goal_state=goal)
    lifecycle.retire_session(session_id)
    authority.generations[session_id] = set()
    value = _event(core, authority, 'close', session_id, open_sessions=len(authority.namespaces),
                   monotonic_ns=time.monotonic_ns(), **fields)
    authority.ended[session_id] = value
    _write(authority, value)
    return value


def release_requests(lifecycle, requests):
    """Drop the per-request host records of a closed session's (terminal) requests.

    Each lifecycle entry keeps its identity, state, timings and the header of its
    drain receipt (operation, binding, drained), with its binding reduced to
    ``ReleasedBinding``; the request tombstones of the dispatcher stay as they are."""
    endpoint = lifecycle.endpoint
    core = endpoint.core
    private = core._remote_ra_private_v2_endpoint
    boundary = core._remote_ra_live_v2_boundary
    selection = private.selection
    tables = (private.claims, endpoint.claims, endpoint.prepared_payloads, endpoint.prepared_drains,
              boundary.prepared, boundary.proxy.http_receipts,
              selection.authorized if selection is not None else {})
    for rid in requests:
        entry = lifecycle.entries[rid]
        binding = entry['binding']
        if binding is not None and type(binding) is not ReleasedBinding:
            entry['binding'] = ReleasedBinding(binding.request, binding.session, binding.turn,
                                               binding.generation, binding.identity)
        entry['request'] = None
        receipt = entry.get('drain_receipt')
        if receipt is not None:
            entry['drain_receipt'] = {k: receipt[k] for k in ('operation', 'binding', 'drained') if k in receipt}
        for table in tables:
            table.pop(rid, None)
    gone = set(requests)
    for key in [key for key in boundary.receipts if key[0] in gone]:
        del boundary.receipts[key]


def summary(authority):
    """``engine-cleanup.json``'s pool record: what was opened and how it closed."""
    ended = authority.ended or {}
    return dict(max_running_sessions=authority.limit, registered=len(authority.ids),
                closed=[sid for sid in authority.ids if sid in ended and ended[sid]['opened']],
                closed_unopened=[sid for sid in authority.ids if sid in ended and not ended[sid]['opened']],
                open_at_shutdown=[sid for sid in authority.ids if sid in authority.namespaces],
                never_opened=[sid for sid in authority.ids
                              if sid not in ended and sid not in authority.namespaces])
