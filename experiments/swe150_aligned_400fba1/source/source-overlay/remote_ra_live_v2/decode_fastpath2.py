"""Second decode-step host fast path: ``serve.py --decode-fastpath2``, off by default, read once at startup.

It needs ``--decode-fastpath`` (decode_fastpath) and cuts more of the host work
that every running request costs in every decode step, with tokens, selections
and the worker frames of the evidence unchanged (``test_decode_fastpath2``):

- One ``allocation_batch`` worker RPC per engine step instead of one
  ``allocation_update`` RPC per scheduled request. ``EngineRpcBoundary`` queues
  each request's update during ``scheduler.schedule`` and sends the queue when
  ``schedule`` returns (``attach_engine`` wraps it), and before any other worker
  RPC. A request whose rows did not change since its last accepted update (no
  new FullAttention block, the same Mamba row) is sent without rows. The worker
  applies the entries in order, each through the same ``allocation_update``
  command and ``evidence.measured_command`` as before, and answers one receipt per
  entry; ``EngineWorkerProxy`` files each as that request's frame, so every
  ``native-*.json`` keeps one ``allocation_update`` frame per request per step
  with the same fields. The RPC payloads differ (one per step): that is why
  this is a pinned switch of its own.
- The per-request engine work of ``dispatch_mixed_request``: the native
  ``allocate_slots`` arguments are bound without ``inspect`` (``Binder``, the
  same values; anything unusual binds through ``inspect`` as before), and an
  unchanged Mamba row is not rebuilt (the same block objects at the same
  incarnations give the same values).
- ``measured_command`` sums the request's claimed blocks per KV group times the
  group's Page bytes (``group_bytes``, built once from the views) instead of
  visiting every layer's view (the same integer).
- Worker observation: a Mamba slot's incarnation is read at its row index (the
  ``dict`` of the whole row gave the same value: live ids are unique per row,
  as every accepted update checked), and an unchanged step reuses the previous
  claim-key tuple. The Qwen step lease takes an unchanged attention tuple by
  identity.
- After the forward, the step's finiteness is copied to the host without a wait
  and the per-request after-forward work (lease close, Page compaction, GDN
  capture) runs in the generation guard after the native ``execute_model``
  returns, i.e. after the logits GEMM is queued: the host work then overlaps it
  instead of running while the device is idle. The rows are still checked
  finite before any lease closes; a step whose rows are all finite checks them
  once. The step is joined by that finiteness read (the forward's completion on
  the stream), so ``leave_execute`` needs no second device drain; the next
  step's guard drains as before. With an in-step phase-1 probe the work runs in
  place as before.
- ``MarkerStop`` remembers per token id whether the token's own decode holds a
  '>' (the same answer: a token decodes to the same text every time).

Off (the default) nothing here runs: no startup config key, no worker startup
record key, no evidence key and no vLLM argument. On, the startup config pins
``decode_fastpath2: true`` next to ``decode_fastpath: true``; ``compose_engine``
sends it once in the worker startup record; the worker checks it against its own
reading of the pinned config and echoes it; ``write_ready`` records it.
Integrity checks stay at startup; nothing here hashes per request or step.
"""
from __future__ import annotations

import inspect
import os
from types import FunctionType, MethodType

from .contract import require

CONFIG_KEY = 'decode_fastpath2'     # startup config, worker startup record, worker evidence, ready.json
BATCH = 'allocation_batch'          # the one per-step worker RPC
ENTRY_KEYS = frozenset({'request', 'binding', 'start', 'count', 'rows'})


# ---- the pin (startup only) ----------------------------------------------------------------------

def pinned(config):
    """True when a startup config pins the second fast path, else None (the key is absent)."""
    if config is None or CONFIG_KEY not in config:
        return None
    require(config[CONFIG_KEY] is True, 'pinned decode_fastpath2 must be true; off is pinned by leaving it out')
    require(config.get('decode_fastpath') is True, 'decode_fastpath2 requires decode_fastpath')
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
    require(sent is None or sent is True, 'decode_fastpath2 worker record must be true or absent')
    require(sent is None or startup.get('decode_fastpath') is True, 'decode_fastpath2 worker record needs decode_fastpath')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker decode_fastpath2 differs from the pinned startup config')
    return sent


# ---- attachment ------------------------------------------------------------------------------------

def group_bytes(views):
    """{KV group: the Page bytes of all its views}: measured_command's active bytes per claimed block."""
    out = {}
    for v in views:
        out[v.group] = out.get(v.group, 0) + v.page_bytes
    return out


def attach_worker(owner, bridge, qwen, io):
    """Worker attachment with the switch on (before the step profile wraps the bridge)."""
    require(not getattr(bridge, 'decode_fastpath2', False), 'decode fast path 2 attached twice')
    owner.decode_fastpath2 = True
    bridge.decode_fastpath2 = True
    qwen.decode_fastpath2 = True
    io.group_bytes = group_bytes(io.views)


def attach_engine(core, boundary, proxy):
    """Engine attachment: the update queue, sent when ``scheduler.schedule`` returns and before any
    other worker RPC. Startup only."""
    require(getattr(boundary, 'batched', None) is None and getattr(proxy, 'before_call', None) is None,
            'decode fast path 2 engine attachment collision')
    boundary.batched = {}
    proxy.before_call = boundary.flush_updates
    scheduler = core.scheduler
    schedule = scheduler.schedule

    def flushing_schedule(*args, **kwargs):
        try:
            return schedule(*args, **kwargs)
        finally:
            # The updates of the requests allocated before any failure were sent at
            # once before; they are sent here, then the failure propagates.
            boundary.flush_updates()
    flushing_schedule.__wrapped__ = schedule
    scheduler.schedule = flushing_schedule


# ---- binding the native allocator's arguments ------------------------------------------------------

_BINDERS = {}


class Binder:
    """``inspect.signature(native).bind(...)`` + ``apply_defaults()`` for a bound method whose
    parameters are all positional-or-keyword: the same ``arguments`` (parameter order) and the
    same call. Anything else (a missing, unknown, repeated or surplus argument, another
    parameter kind) returns None and the caller binds through ``inspect`` as before."""
    __slots__ = ('state', 'names', 'index', 'defaults', 'required')

    def __init__(self, state, signature):
        params = tuple(signature.parameters.values())
        self.state = state
        self.names = tuple(p.name for p in params)
        self.index = frozenset(self.names)
        self.defaults = {p.name: p.default for p in params if p.default is not p.empty}
        self.required = tuple(p.name for p in params if p.default is p.empty)
        if any(p.kind is not p.POSITIONAL_OR_KEYWORD for p in params):
            self.names = None

    def bind(self, positional, kwargs):
        names = self.names
        if names is None or len(positional) > len(names):
            return None
        values = dict(zip(names, positional))
        for key, value in kwargs.items():
            if key not in self.index or key in values:
                return None
            values[key] = value
        for name in self.required:
            if name not in values:
                return None
        defaults = self.defaults
        return {name: values[name] if name in values else defaults[name] for name in names}


def binder(method):
    """The Binder of a plain bound function (rebuilt when its code or defaults change), else None."""
    function = getattr(method, '__func__', None)
    if (type(method) is not MethodType or type(function) is not FunctionType or
            '__wrapped__' in function.__dict__ or '__signature__' in function.__dict__):
        return None
    state = (function.__code__, function.__defaults__, function.__kwdefaults__)
    entry = _BINDERS.get(function)
    if entry is None or not all(map(lambda a, b: a is b, entry.state, state)):
        entry = _BINDERS[function] = Binder(state, inspect.signature(method))
    return entry


# ---- the worker side of one allocation_batch -------------------------------------------------------

def dispatch_batch(worker, backend, owner, payload):
    """Every entry through the ``allocation_update`` command, in order, each measured as before.

    Returns {operation: allocation_batch, receipts: [the per-entry receipt]}; an entry
    without rows reuses the request's last accepted rows (``_allocation_update``)."""
    require(getattr(owner, 'decode_fastpath2', False) is True, 'allocation batch without --decode-fastpath2')
    entries = payload.get('entries') if type(payload) is dict else None
    require(type(payload) is dict and set(payload) == {'entries'} and type(entries) is list and entries and
            all(type(e) is dict and set(e) == ENTRY_KEYS for e in entries),
            'allocation batch entries required')
    require(len({e['request'] for e in entries}) == len(entries), 'one allocation batch entry per request')
    qwen = worker._remote_ra_live_v2_qwen
    from remote_ra_http.evidence import measured_command
    runtime = os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1'
    handoff = getattr(backend.io, 'handoff', False) is True
    rowless = None
    if getattr(owner, 'decode_fastpath3', False):
        # --decode-fastpath3: entries sent without rows run their update directly, measured
        # the same way (decode_fastpath3.rowless_entry; None: as below).
        from . import decode_fastpath3
        rowless = decode_fastpath3.rowless_entry
    receipts = []
    for entry in entries:
        if rowless is not None:
            receipt = rowless(backend, owner, qwen, entry, runtime)
            if receipt is not None:
                require(receipt.get('binding') == entry['binding'], 'physical worker command receipt mismatch')
                receipts.append(receipt)
                continue
        command = owner.resolve_worker_command('allocation_update', entry)
        if runtime:
            # The frame of each request, measured as worker_dispatch measures one RPC.
            receipt = measured_command(backend, BATCH, command, qwen, request=entry['request'])
        elif handoff:
            backend.io.poll_joins()
            receipt = command(backend, qwen)
            backend.io.poll_joins()
        else:
            receipt = command(backend, qwen)
        require(type(receipt) is dict and receipt.get('operation') == 'allocation_update' and
                receipt.get('binding') == entry['binding'], 'physical worker command receipt mismatch')
        receipts.append(receipt)
    return dict(operation=BATCH, receipts=receipts)
