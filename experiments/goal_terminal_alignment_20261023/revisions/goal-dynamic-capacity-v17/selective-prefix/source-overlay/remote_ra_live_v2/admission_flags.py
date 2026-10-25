"""Admission optimisations: one startup flag each, off by default, verified once at startup.

``serve.py --gdn-device-handoff`` (admission design (e)): the phase-1 GDN state
stays in its probe slot on the device and is copied device-to-device into the
phase-2 slot; Mamba-block crossings copy device-to-device as well; only the
committed carry keeps a host copy (``gdn_handoff``).

``serve.py --phase1-in-step`` (admission design (b)): an admission's phase 1
runs inside the next engine step, next to the running decodes, and the
admission parks until the step has collected it (``phase1_step``); the
separate probe-only pass stays when nothing else runs or it does not fit.

``serve.py --admission-overlap K`` (admission design (f)): up to K admissions
in flight at once. While one session's admission is parked (its goal_query
Goal on its own Goal thread, or its in-step phase 1), other sessions'
admissions proceed; each holds a capacity reservation until it is accepted
(``capacity``). K = 1 (the default) is off. The record is ``{"permits": K,
"admission_seconds": A, "lease_seconds": L}``: the frontend's wait for an
admission's ACK and the engine lease, pinned from K at startup (decision 8,
``timeouts``), never changed at run time.

With every flag off nothing changes: no startup config key, no worker startup
record key, no worker or READY evidence key and no vLLM argument; the server is
613a7d5's byte for byte. With a flag on:

- the startup config pins ``admission`` = ``{flag: True}`` for the switches
  that are on, plus ``admission_overlap`` = its record when K > 1 (the key is
  absent when every flag is off); ``remote_ra_http.config.load`` checks its
  shape;
- ``compose_engine`` reads that record once and sends it in the worker startup
  record (``admission``);
- the worker (``attach_loaded_worker``) compares the record with its own
  reading of the pinned startup config and returns it in its attach evidence;
  the engine compares that evidence with what it sent; READY records it. Any
  mismatch is refused at startup.

Nothing here runs per request or per step.
"""
from __future__ import annotations

import os

from .contract import require

CONFIG_KEY = 'admission'          # startup config, worker startup record, worker evidence, READY
FLAGS = ('gdn_device_handoff', 'phase1_in_step')   # serve.py: --gdn-device-handoff, --phase1-in-step
OVERLAP = 'admission_overlap'     # serve.py --admission-overlap K: its record when K > 1
MAX_PERMITS = 16                  # sessions.MAX_SESSIONS: at most one admission per running sequence

# Decision 8: the frontend's wait for an admission's ACK and the engine lease.
# With one admission in flight (registered_add_protocol.ADMISSION_SECONDS 300,
# LEASE_SECONDS 360 = 300 + the compensation Cancel's 60 s) the wait covers one
# rewriter call of up to its pinned 120 s behind at most one discarded Goal.
# With K in flight, the Goal rewriter is still one process serialised by its
# own lock, so an admission's rewrite may queue behind the other K - 1
# admissions' rewrites: each further permit adds one pinned rewriter call
# (selection_contract.rewriter_pin: timeout_seconds 120) to both bounds.
# (Measured on job 3321538, 5 sessions: a REWRITE takes 13.4 s at the median,
# so K queued rewrites at that pace fit well inside these bounds.)
ADMISSION_SECONDS, LEASE_SECONDS = 300, 360    # registered_add_protocol's (K = 1)
REWRITE_SECONDS = 120


def timeouts(permits):
    """(admission_seconds, lease_seconds) pinned for ``permits`` admissions in flight:
    300 + 120 (K - 1) and 360 + 120 (K - 1); K = 1 is today's 300 / 360."""
    require(type(permits) is int and 1 <= permits <= MAX_PERMITS, 'admission permits must be 1-%d' % MAX_PERMITS)
    extra = REWRITE_SECONDS * (permits - 1)
    return ADMISSION_SECONDS + extra, LEASE_SECONDS + extra


def overlap_record(permits):
    """The ``admission_overlap`` record of --admission-overlap K (K >= 2)."""
    require(type(permits) is int and 2 <= permits <= MAX_PERMITS,
            '--admission-overlap K must be an integer 2-%d (1 is off)' % MAX_PERMITS)
    admission, lease = timeouts(permits)
    return dict(permits=permits, admission_seconds=admission, lease_seconds=lease)


def record(**flags):
    """The ``admission`` record of a launcher whose flags are ``flags`` (None: all off).
    ``admission_overlap`` is the integer K (1: off); the others are switches."""
    permits = flags.pop(OVERLAP, 1)
    require(set(flags) <= set(FLAGS) and all(type(v) is bool for v in flags.values()),
            'admission flags must be booleans of %s' % (FLAGS,))
    require(type(permits) is int and 1 <= permits <= MAX_PERMITS,
            '--admission-overlap K must be an integer 1-%d' % MAX_PERMITS)
    value = {name: True for name in FLAGS if flags.get(name)}
    if permits > 1:
        value[OVERLAP] = overlap_record(permits)
    return value or None


def check(value):
    """A pinned ``admission`` record: a non-empty dict of known switches, each True,
    and at most the overlap record, exactly as ``overlap_record`` pins it."""
    require(type(value) is dict and value and set(value) <= set(FLAGS) | {OVERLAP} and
            all(v is True for k, v in value.items() if k != OVERLAP),
            'pinned admission flags must name %s, each true' % (FLAGS + (OVERLAP,),))
    if OVERLAP in value:
        overlap = value[OVERLAP]
        require(type(overlap) is dict and type(overlap.get('permits')) is int and
                2 <= overlap['permits'] <= MAX_PERMITS and overlap == overlap_record(overlap['permits']),
                'pinned admission_overlap must be {permits: K (2-%d), admission_seconds, lease_seconds} '
                'with the timeouts pinned from K' % MAX_PERMITS)
        return dict(value, **{OVERLAP: dict(overlap)})
    return dict(value)


def permits(flags):
    """K of a record (1: --admission-overlap off, or every flag off)."""
    if flags is None or OVERLAP not in flags:
        return 1
    return flags[OVERLAP]['permits']


def pinned_timeouts(flags):
    """(admission_seconds, lease_seconds) of a pinned record: today's 300 / 360 when off."""
    if flags is None or OVERLAP not in flags:
        return ADMISSION_SECONDS, LEASE_SECONDS
    return flags[OVERLAP]['admission_seconds'], flags[OVERLAP]['lease_seconds']


def allowed_timeouts():
    """Every (admission_seconds, lease_seconds) pair a startup may pin (K = 1-16)."""
    return frozenset(timeouts(k) for k in range(1, MAX_PERMITS + 1))


def pinned(config):
    """The startup config's admission record, or None (every flag off). Startup only."""
    if CONFIG_KEY not in config:
        return None
    return check(config[CONFIG_KEY])


def engine_flags():
    """The pinned record for the worker startup record (compose_engine), or None (off)."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def worker_flags(startup):
    """Worker startup: the record the engine sent, checked once against the pinned config.

    With the HTTP runtime the worker reads the startup config itself and refuses
    a record that differs (also an absent one when a flag is pinned, or the
    reverse). Without it (CPU fixtures) the record's shape is checked only."""
    sent = startup.get(CONFIG_KEY)
    if sent is not None:
        sent = check(sent)
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        from remote_ra_http.config import load
        require(pinned(load()) == sent, 'worker admission flags differ from the pinned startup config')
    return sent


def check_receipt(sent, receipt):
    """Engine startup: the worker attached with exactly the flags it was sent."""
    require(receipt.get(CONFIG_KEY) == sent, 'worker attached with other admission flags')


def on(flags, name):
    """Whether flag ``name`` is on in a record (None: every flag off)."""
    require(name in FLAGS, 'unknown admission flag %r' % (name,))
    return flags is not None and flags.get(name) is True
