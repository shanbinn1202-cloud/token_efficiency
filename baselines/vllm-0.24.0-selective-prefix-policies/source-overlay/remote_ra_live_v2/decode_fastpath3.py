"""Third decode-step host fast path: ``serve.py --decode-fastpath3``, off by default, read once at startup.

It needs ``--decode-fastpath2`` (and so ``--decode-fastpath``). With both, most of the host
work this runtime still adds to a decode step before the forward is a fixed cost per
running request: the worker's step (observation, claims, materialize, positions, the Qwen
lease: ``worker.before``), the FA row check (``worker.rows``), the engine's per-request
allocation and the worker side of ``allocation_batch``. With the switch on, the same
values, plans and evidence are produced with less work (``test_decode_fastpath3``):

- ``RunnerAllocationOwner._observe_increment``: a request whose step appends no FA block
  and keeps the same Mamba row objects at the same slot index (15 of 16 steps at P = 16)
  takes the last step's Mamba leases, key tuples and claim differences (all empty) as they
  are, after the same FA list and device-row checks and the same Mamba row check. Its
  device-row compare reads the row's bytes once (the same answer as ``np.array_equal`` for
  one integer dtype; anything else compares as before), as does ``worker_rows._extend``.
- ``WorkerBridge.before_forward`` (``before_forward`` here): a request whose
  ``materialize`` would only mark the step materialized (its first step done, no fresh
  Mamba slot, nothing added or removed) skips it (``materialize_steady``), and skips its IO
  scope (``tensors.attributed``) when its lease also keeps the current Mamba slots: nothing
  in the scope can issue IO then, so it would attribute nothing. The step's position
  columns are read from the device in one copy instead of two (``step_positions``, the same
  lists).
- ``QwenAlignLifecycle.before_forward3``: the step lease of an unchanged Mamba tuple is
  built without re-checking slots that were checked when that tuple was accepted, with
  identity before equality where the objects are the same.
- ``EngineRpcBoundary.dispatch_mixed_request``: the per-call imports are bound once.
- ``decode_fastpath2.dispatch_batch``: an entry sent without rows runs its
  ``allocation_update`` command without building the command closure or an IO scope
  (``rowless_entry``): the same checks, the same record update and the same measured frame
  (the counters before and after, the active bytes, the elapsed time).

Off (the default) nothing here runs: no startup config key, no worker startup record key,
no evidence key and no vLLM argument. On, the startup config pins ``decode_fastpath3: true``
next to ``decode_fastpath2: true``; ``compose_engine`` sends it once in the worker startup
record; the worker checks it against its own reading of the pinned config and echoes it;
``write_ready`` records it. Integrity checks stay at startup; nothing here hashes per request
or step. With ``--gdn-device-handoff``, materialize and the rowless entries run as before
(their IO is issued and joined lazily); the steady observation and lease still apply.
"""
from __future__ import annotations

import os
import time

from .contract import require

CONFIG_KEY = 'decode_fastpath3'     # startup config, worker startup record, worker evidence, ready.json


# ---- the pin (startup only) ----------------------------------------------------------------------

def pinned(config):
    """True when a startup config pins the third fast path, else None (the key is absent)."""
    if config is None or CONFIG_KEY not in config:
        return None
    require(config[CONFIG_KEY] is True, 'pinned decode_fastpath3 must be true; off is pinned by leaving it out')
    require(config.get('decode_fastpath2') is True and config.get('decode_fastpath') is True,
            'decode_fastpath3 requires decode_fastpath2')
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
    require(sent is None or sent is True, 'decode_fastpath3 worker record must be true or absent')
    require(sent is None or (startup.get('decode_fastpath2') is True and startup.get('decode_fastpath') is True),
            'decode_fastpath3 worker record needs decode_fastpath2')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker decode_fastpath3 differs from the pinned startup config')
    return sent


# ---- attachment ------------------------------------------------------------------------------------

def attach_worker(owner, bridge, qwen):
    """Worker attachment with the switch on: after --decode-fastpath2's (before the step profile)."""
    require(getattr(owner, 'decode_fastpath2', False) is True and getattr(bridge, 'decode_fastpath2', False) is True,
            'decode fast path 3 needs decode fast path 2 attached first')
    require(not getattr(bridge, 'decode_fastpath3', False), 'decode fast path 3 attached twice')
    owner.decode_fastpath3 = True
    bridge.decode_fastpath3 = True
    qwen.decode_fastpath3 = True


def attach_engine(boundary):
    """Engine attachment: the modules the boundary's per-request allocation imports from, bound
    once; their functions are still looked up per call (the step profile wraps some). Startup only."""
    require(getattr(boundary, 'batched', None) is not None, 'decode fast path 3 needs decode fast path 2 attached first')
    require(getattr(boundary, 'decode_fastpath3', None) is None, 'decode fast path 3 engine attachment collision')
    from . import allocation_guard, chunk_protocol, decode_fastpath2, pool_incarnations
    boundary.decode_fastpath3 = dict(guard=allocation_guard, chunk=chunk_protocol, fast2=decode_fastpath2,
                                     incarnations=pool_incarnations)


# ---- the worker step ------------------------------------------------------------------------------

def step_positions(runner, positions, end):
    """``positions.step_positions`` with one device read: the same dict (the same lists), or None.

    MRoPE: the three position rows and the logical buffer's columns [0, end) are stacked on
    their device and read back together (one copy, one join) instead of read one after the
    other. Tensors of another shape, dtype or device are read as before."""
    if runner.uses_mrope:
        if not (positions.ndim == 2 and positions.shape[0] == 3 and positions.shape[1] >= end):
            return None
        logical = runner.positions[:end]
        if (getattr(logical, 'ndim', None) == 1 and logical.shape[0] == end and logical.dtype == positions.dtype and
                logical.device == positions.device):
            import torch
            rows = torch.cat((positions[:, :end], logical.unsqueeze(0))).tolist()
            return dict(rows=rows[:3], logical=rows[3])
        return dict(rows=positions[:, :end].tolist(), logical=logical.tolist())
    if not (positions.ndim == 1 and positions.shape[0] >= end):
        return None
    return dict(logical=positions[:end].tolist())


def before_forward(bridge, runner, scheduler_output, positions):
    """``WorkerBridge.before_forward`` (its owner branch) with --decode-fastpath3.

    The owner claims the whole step first; then each request in batch order, inside the
    request's IO scope (``tensors.attributed``): materialize, its positions, its Qwen lease.
    The scope is left out when nothing in it can issue IO: ``materialize`` would only mark the
    step materialized (``RunnerAllocationOwner.materialize_steady``, which then does that), the
    positions are host reads, and the lease keeps the request's current Mamba slots (no GDN
    restore or relocation); the scope would then attribute nothing."""
    from remote_ra_growing_worker.tensors import attributed
    from .positions import text_positions          # the module's function now (the step profile wraps it)
    owner, qwen = bridge.owner, bridge.qwen
    steps = owner.step(runner, scheduler_output)
    host = None
    if steps:
        host = step_positions(runner, positions, max(s.offset + s.count for s in steps))
    extra = {} if host is None else dict(host=host)
    assignments, io, states = bridge.assignments, owner.io, qwen.requests
    steady = owner.materialize_steady
    lease = qwen.before_forward3
    leases = []
    for s in steps:
        state = assignments.get(s.request)
        require(state is not None, "worker assignment absent")
        if steady(s):
            current = states.get(s.request)
            if current is not None and current.current_hybrid is not None and s.hybrid is current.current_hybrid:
                logical, evidence = text_positions(runner, positions, s.start, s.count, s.request, s.offset,
                                                   **extra)
                leases.append(lease(state.assignment.identity, s.start, s.count, logical, evidence,
                                    s.attention, s.hybrid, s.generation))
                continue
            with attributed(io, s.request):
                logical, evidence = text_positions(runner, positions, s.start, s.count, s.request, s.offset,
                                                   **extra)
                leases.append(lease(state.assignment.identity, s.start, s.count, logical, evidence,
                                    s.attention, s.hybrid, s.generation))
            continue
        with attributed(io, s.request):
            owner.materialize(s)
            logical, evidence = text_positions(runner, positions, s.start, s.count, s.request, s.offset, **extra)
            leases.append(lease(state.assignment.identity, s.start, s.count, logical, evidence,
                                s.attention, s.hybrid, s.generation))
    return leases[0] if len(leases) == 1 else tuple(leases)


# ---- the worker side of one allocation_batch entry sent without rows -------------------------------

def rowless_entry(backend, owner, qwen, entry, measured):
    """The measured receipt of an ``allocation_update`` entry with ``rows: None``, or None
    (``dispatch_batch`` then runs the entry as before).

    ``measured`` is True on the HTTP runtime (``evidence.measured_command`` measures each
    entry): the frame gets the same ``http_measurement`` (the request's counters before and
    after, its active owned KV bytes, the elapsed time). The update does no IO, so the IO
    scope ``measured_command`` opens around it would attribute nothing."""
    rid = entry['request']
    rec = owner.records.get(rid)
    if (entry['rows'] is not None or rid in owner.terminals or rec is None or
            rec.get('binding') != entry['binding'] or getattr(backend.io, 'handoff', False)):
        return None
    from remote_ra_http import evidence
    if measured:
        claims = owner.claims
        groups_of = getattr(claims, 'groups_of', None)
        per_group = getattr(backend.io, 'group_bytes', None)
        if groups_of is None or per_group is None:
            return None
        before = evidence.counters(backend.io, rid)
        start = time.monotonic_ns()
        active = sum(per_group.get(group, 0) * n for group, n in groups_of(rid).items())
    # RunnerAllocationOwner._allocation_update, rows None: the same checks and update.
    from .allocation_owner import INCREMENTAL
    require(owner.decode_fastpath2 and rec.get('native_rows') is not None and
            (rec.get('rows_seen') is not None or not INCREMENTAL),
            'allocation update without rows before any accepted rows')
    from .chunk_protocol import worker_interval
    worker_interval(rec, qwen.requests[rec['identity'].request], entry['start'], entry['count'])
    rec['native_interval'] = (entry['start'], entry['count'])
    receipt = dict(operation='allocation_update', binding=entry['binding'], drained=True)
    if not measured:
        return receipt
    after = evidence.counters(backend.io, rid)
    require(after['h2d_issued'] == after['h2d_joined'], 'tensor copies not joined')
    return dict(receipt, http_measurement=dict(before=before, after=after,
                                               elapsed_ns=time.monotonic_ns() - start,
                                               active_owned_kv_bytes=active))
