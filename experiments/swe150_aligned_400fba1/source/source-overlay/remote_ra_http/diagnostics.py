"""Bounded startup diagnostics; diagnostic failures never mask the original."""
import json
import os
from pathlib import Path
import time
import traceback

def context(owner):
    now = time.monotonic()
    identity = getattr(owner, 'identity', None)
    return dict(pid=os.getpid(), phase=getattr(owner, 'diagnostic_phase', 'unavailable'),
        epoch=getattr(identity, 'epoch', None), policy=getattr(owner, 'policy', None),
        elapsed_seconds=round(now - getattr(owner, 'started_at', now), 3),
        deadline_remaining_seconds=round(getattr(owner, 'deadline', now) - now, 3))

def emit(value):
    try:
        print('REMOTE_RA_LIVE ' + json.dumps(value, sort_keys=True, ensure_ascii=True), flush=True)
    except BaseException:
        pass

def phase(owner, name):
    if owner is None: return  # feature-OFF remains a no-op
    owner.diagnostic_phase = name
    emit(dict(event='AUTHORITY_PHASE', **context(owner)))

def failure(event, exc, owner=None):
    value = dict(schema='remote-ra-startup-failure-v1', event=event, **context(owner),
        error=type(exc).__name__, reason=str(exc)[:4096],
        traceback=''.join(traceback.format_exception(exc))[-65536:])
    emit(value)  # before cleanup/pipe-close/reaping, even if configuration is broken
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1': return value
    try:
        from .config import load
        output = Path(load()['output_directory'])
        path = output / ('failure-' + event.lower().replace('_', '-') + '-' + str(os.getpid()) + '.json')
        raw = (json.dumps(value, sort_keys=True, ensure_ascii=True) + '\n').encode()
        # Preserve the first diagnostic at each stage; do not overwrite evidence.
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass
    except BaseException as diagnostic_error:
        emit(dict(event='DIAGNOSTIC_PERSIST_FAILED', error=type(diagnostic_error).__name__,
                  reason=str(diagnostic_error)[:1024], original_event=event, pid=os.getpid()))
    return value
