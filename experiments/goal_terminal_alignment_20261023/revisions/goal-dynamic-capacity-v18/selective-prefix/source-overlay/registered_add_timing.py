"""Content-free admission timing; diagnostic observations, never acceptance gates."""
from contextlib import contextmanager
import json
import os
import time


def event(phase, **fields):
    print("REMOTE_RA_REGISTERED_ADD " + json.dumps(
        dict(phase=phase, pid=os.getpid(), **fields), sort_keys=True), flush=True)


@contextmanager
def timed(phase, **fields):
    start = time.monotonic()
    event(phase + "_start", **fields)
    outcome = "ok"
    try:
        yield
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        event(phase + "_end", elapsed_ms=round((time.monotonic() - start) * 1000, 3),
              outcome=outcome, **fields)
