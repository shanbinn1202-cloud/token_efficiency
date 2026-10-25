"""Decode-step host fast path: ``serve.py --decode-fastpath``, off by default, read once at startup.

Steady decode at P = 16 spends most of its host time in work that grows with
(running requests) x (Pages per request) or that joins the device once per
request (decode-speed analysis, 2026-10-12). With the switch on, the same
values are produced with less work; tokens, RPC payloads, receipts and evidence
are unchanged (``test_decode_fastpath``):

- ``ClaimTable``: the worker owner's allocation claims keep a per-request count
  of claimed blocks per KV group, so ``measured_command`` (one per request per
  step, the ``allocation_update`` RPC) reads its request's counts instead of
  scanning every claim of every running request (O(B x Pages) per call, i.e.
  O(B^2 x Pages) per step).
- ``QwenAlignLifecycle.before_forward``: the step lease re-checks the type of
  the attention leases appended since the last accepted forward only; the
  prefix is the same tuple of the same objects, already checked (as the
  allocation-generation check there already does).
- ``WorkerBridge.before_forward``: the positions of every request of the step
  are read from the device once (one read of the step's columns instead of one
  or two per request); each request's checks and position evidence are the
  per-request ones (``positions.text_positions_batch``).
- ``WorkerBridge.after_forward``: one device drain (connector check,
  ``torch.cuda.synchronize``, connector check) for the step after the forward,
  instead of one per request in ``forward_complete``. Nothing is issued
  asynchronously between the requests' callbacks (Page compaction and latest-GDN
  capture read back synchronously), so the one drain joins the same work.
- ``MarkerStop`` (engine): the action-marker check decodes only the new output
  token when the request's previous token was checked without the marker and the
  new token has no '>' (the marker's last character); else the whole output, as
  before (``mswea_protocol.MarkerStop``, ``incremental``).

With the switch off nothing here runs: no startup config key, no worker
startup record key, no evidence key and no vLLM argument. With it on, the
startup config pins ``decode_fastpath: true``; ``compose_engine`` sends it once
in the worker startup record; the worker compares it with its own reading of the
pinned startup config and echoes it in its attach evidence; ``write_ready``
records it. Nothing here runs per request or step except the fast paths.
"""
from __future__ import annotations

import os

from .contract import require

CONFIG_KEY = 'decode_fastpath'     # startup config, worker startup record, worker evidence, ready.json


def pinned(config):
    """True when a startup config pins the fast path, else None (the key is absent)."""
    if config is None or CONFIG_KEY not in config:
        return None
    require(config[CONFIG_KEY] is True, 'pinned decode_fastpath must be true; off is pinned by leaving it out')
    return True


def loaded():
    """The fast-path pin of this process's startup config on the HTTP runtime, else None."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def worker_flag(startup):
    """Worker startup: the record the engine sent, checked once against the pinned config.

    With the HTTP runtime the worker reads the startup config itself and refuses a
    record that differs; without it (CPU fixtures) only the record's value is checked."""
    sent = startup.get(CONFIG_KEY)
    require(sent is None or sent is True, 'decode_fastpath worker record must be true or absent')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker decode_fastpath differs from the pinned startup config')
    return sent


class ClaimTable(dict):
    """``RunnerAllocationOwner.claims`` that also counts each owner request's claims per group.

    Keys are (KV group, block id); values are (request, generation) owners. Every
    mutation the owner uses goes through the overrides below; the counts equal what
    a scan of the items gives (``groups_of``, checked by the tests)."""
    __slots__ = ('_counts',)

    def __init__(self, items=()):
        super().__init__()
        self._counts = {}
        for key, owner in dict(items).items():
            self[key] = owner

    def _add(self, key, owner, step):
        per = self._counts.get(owner[0])
        if per is None:
            per = self._counts[owner[0]] = {}
        n = per.get(key[0], 0) + step
        if n:
            per[key[0]] = n
        else:
            del per[key[0]]
            if not per:
                del self._counts[owner[0]]

    def __setitem__(self, key, owner):
        old = dict.get(self, key)
        if old is not None:
            self._add(key, old, -1)
        dict.__setitem__(self, key, owner)
        self._add(key, owner, 1)

    def __delitem__(self, key):
        owner = dict.pop(self, key)
        self._add(key, owner, -1)

    _MISSING = object()

    def pop(self, key, default=_MISSING):
        if key in self:
            owner = dict.pop(self, key)
            self._add(key, owner, -1)
            return owner
        if default is ClaimTable._MISSING:
            raise KeyError(key)
        return default

    def popitem(self):
        key, owner = dict.popitem(self)
        self._add(key, owner, -1)
        return key, owner

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return dict.__getitem__(self, key)

    def update(self, *args, **kwargs):
        for key, owner in dict(*args, **kwargs).items():
            self[key] = owner

    def clear(self):
        dict.clear(self)
        self._counts.clear()

    def __ior__(self, other):
        self.update(other)
        return self

    def groups_of(self, request=None):
        """{group: claims} of one owner request (None: of every claim), as measured_command counts."""
        if request is not None:
            return dict(self._counts.get(request, ()))
        out = {}
        for per in self._counts.values():
            for group, n in per.items():
                out[group] = out.get(group, 0) + n
        return out


def attach_worker(owner, bridge, qwen):
    """Worker attachment with the switch on: the claim counts and the three step fast paths."""
    require(not owner.claims, 'decode fast path attaches before any claim')
    owner.claims = ClaimTable()
    bridge.decode_fastpath = True
    qwen.decode_fastpath = True
