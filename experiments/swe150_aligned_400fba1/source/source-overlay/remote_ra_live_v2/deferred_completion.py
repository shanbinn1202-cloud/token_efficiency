"""Completion I/O off the engine thread: ``serve.py --deferred-completion-io``, off by default, read once at startup.

vLLM emits a request's last token only after ``scheduler.update_from_output`` ran its
completion (``lifecycle.before_success`` -> ``EngineLifecycle._drain``): the worker
``complete`` RPC, its checks, the transcript commit and the native terminal
(``remote_ra_http.evidence.native_terminal``), which reads the startup config again,
copies the drain receipt, encodes 2-5 MB of JSON and writes it with an fsync. Every other
running request waits for all of it.

With the switch, the native terminal keeps on the engine thread only what reads engine
state; its file writing goes to one writer thread that runs after the step's outputs are
in the output queue:

- The record is built by the same code (``evidence.terminal_value``). The request's worker
  frames are taken as the JSON texts ``EngineWorkerProxy.call`` filed (composition.WorkerFrames;
  host-trims-1111, perf 2.4): the writer parses each one (``copy_receipt``'s copy, as a read of
  the frames gives it) and encodes it, so the engine thread parses none of them. The containers other
  owners may still hold (the drain receipt, the plan, the query receipt, the idle-release
  record, the pool counts) are copied there and then with pickle (``snapshot``: an exact
  copy of every value and type, taken at the moment the JSON was taken before, about a
  third of the cost of encoding them); the writer unpickles and encodes them, and for the
  drain receipt does ``copy_receipt``'s round trip itself. The request's prompt token
  list (a fresh list) and the scalars belong to the record alone and are encoded on the
  writer as they are (so are frames a harness filed parsed, in a plain dict).
- goal_query's goals.jsonl row is encoded and named there (its SHA-256 is in the native
  record, as before); the row is appended on the writer.
- The startup config is not read again per request: the output directory and the layout
  are read once at startup (the config's own SHA-256 check stays at startup).

The job (the goals row, then the native file) is staged; ``post_step`` hands the staged
jobs to the writer in submission order. The writer makes the calls the engine thread made
before: the row with ``open('ab')``, one write, flush, fsync; the native file with
``open('xb')``, one write, flush, fsync. The native bytes are assembled from pieces
(``iter_encoded``): the same JSON encoder with the same options on sub-values, which equals
``encoded(value)`` byte for byte (test_deferred_completion checks it on the native files of
a GPU run and on random values). CPython's JSON encoder holds the GIL for a whole call, so
the writer encodes pieces of a few thousand leaves and sleeps briefly after about 0.5 ms
of work: the engine thread, which takes the GIL back after every GPU wait, never waits
long for it.

Readers see complete files:

- The frontend reads a native file (``FrontendEvidence.complete``) only after the engine
  acknowledged its authenticated Cancel with state complete
  (``ServingController.generation_closed``). While that request's files are still being
  written the acknowledgement is held (``dispatcher.handle_steps`` returns ``Held``) and the
  engine thread sends it once they are (``lifecycle.tick``; the writer wakes the loop). A
  write failure turns it into a failed acknowledgement: the request fails, as a failed
  write failed it before.
- A session's next admission (``RegisteredRouteBridge.add_steps``, which covers its
  readmission and a rebuild), the release of its Pages for another admission
  (``IdleRelease.evict``) and its close (``close_session``) first join its writes.
- A cancellation, an expiry, a disconnect and a terminal retry flush every write and run
  the synchronous path, so goals.jsonl keeps its row order.
- Shutdown (``remote_ra_http.shutdown.shutdown_engine``) flushes and stops the writer
  before the request drain; every later terminal is synchronous. A process exit without
  shutdown flushes at interpreter exit.

Failures: the first write failure is reported (``diagnostics.failure``
``COMPLETION_WRITE_FAILED``) and stops the writer; every later completion, join, held
acknowledgement and the shutdown raise it, and nothing more is written.

Off (the default) none of this runs: no startup config key, no evidence key, no vLLM
argument; ``native_terminal`` finds no writer and runs its previous code. On, the startup
config pins ``deferred_completion_io: true``; ``compose_engine`` attaches the writer once;
``ready.json`` records it. The worker is not involved. Nothing here hashes per request beyond
the native terminal's own hashes (the native file name, the goals.jsonl row).

The Gemma 4 profile's window layout (gemma-deferred-io) takes it as it is: what the layout
adds to a completion stays on the engine thread where it was, around the native terminal (the
worker's window carry in the ``complete`` RPC, ``WindowPins.pin_after`` before vLLM frees the
request, phase 1's leftover blocks in ``recover_probes``; after it vLLM's free and
``WindowPins.after_free``'s check of the pinned window); the record is the same code's, pool 1's free blocks
read with pool 0's at the same point; the native file's one reader on its three protocols is
the frontend's, after the held acknowledgement, and a session's close (a chat-text idle close
or eviction too), its admission and an idle release join its writes as for Qwen
(test_gemma_deferred_completion).
"""
from __future__ import annotations

import atexit
from collections import deque
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time

from .contract import require

CONFIG_KEY = 'deferred_completion_io'      # startup config, ready.json

# Bound once: tests that patch the evidence clocks (time.perf_counter_ns,
# time.monotonic_ns) keep their ticks; the writer never reads those clocks.
_clock = time.perf_counter
_sleep = time.sleep

PIECE = 2048            # leaves per encoded piece: a few to some tens of KB of JSON
SLICE_S = 0.0005        # the writer's own time between two pauses
PAUSE_S = 0.0001        # one pause; time.sleep lets a thread that waits take the GIL
MAX_DEPTH = 32
STOP_S = 60.0           # shutdown's and interpreter exit's wait for the last writes

# remote_ra_http.config.encoded: the same encoder and options (test_deferred_completion compares).
_ENCODER = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
_SCALARS = frozenset((int, float, str, bool, type(None)))
_STR = frozenset((str,))
# Private to the record: encoded on the writer as they are (see the module docstring).
PRIVATE = frozenset(('worker_frames', 'prompt_token_ids'))


# ---- the pin (startup only) ----------------------------------------------------------------------

def pinned(config):
    """True when a startup config pins the switch, else None (the key is absent)."""
    if config is None or CONFIG_KEY not in config:
        return None
    require(config[CONFIG_KEY] is True, 'pinned deferred_completion_io must be true; off is pinned by leaving it out')
    return True


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


# ---- exact JSON in pieces --------------------------------------------------------------------------

def _enc(value):
    """remote_ra_http.config.encoded(value)."""
    return _ENCODER.encode(value).encode()


def _copy_enc(value):
    """encoded(copy_receipt(value)): copy_receipt's JSON round trip, then the encoding."""
    return _enc(json.loads(_enc(value)))


def _weight(value, cap):
    """A cheap size of ``value`` in leaves (long strings count by length inside containers
    that hold containers); exact up to ``cap``, it stops counting above. It only sizes the
    pieces: the bytes never depend on it."""
    t = type(value)
    if t is dict:
        n, children = len(value), value.values()
    elif t is list or t is tuple:
        n, children = len(value), value
    elif t is str:
        return 1 + (len(value) >> 6)
    else:
        return 1
    if n > cap or _SCALARS.issuperset(map(type, children)):
        return n
    for child in children:
        c = type(child)
        if c is dict or c is list or c is tuple:
            n += _weight(child, cap - n)
        elif c is str and len(child) > 64:
            n += len(child) >> 6
        if n > cap:
            return n
    return n


def iter_encoded(value, cap=PIECE, depth=0, leaf=_enc):
    """Byte pieces whose concatenation is ``leaf(value)``: ``encoded(value)`` (``_enc``) or
    ``encoded(copy_receipt(value))`` (``_copy_enc``).

    A dict with string keys and a list or tuple larger than ``cap`` leaves are written
    member by member: the keys in sorted order (the encoder's own order), runs of small
    members as one ``leaf`` of the run without its brackets, larger members recursively.
    Anything else (another key type, a subclass, a scalar) is one ``leaf`` call, which also
    raises what the encoder raises for the whole value. The copy composes the same way:
    decoding a JSON object or array decodes each member on its own and keeps the key
    strings, and a run of scalars (int, float, str, bool, None) reads back unchanged."""
    t = type(value)
    if depth < MAX_DEPTH and value and (t is dict or t is list or t is tuple) and _weight(value, cap) > cap:
        if t is not dict:
            yield from _list_pieces(value, cap, depth, leaf)
            return
        if set(map(type, value)) <= _STR:
            yield from _dict_pieces(value, cap, depth, leaf)
            return
    yield leaf(value)


def _dict_pieces(value, cap, depth, leaf):
    keys = sorted(value)
    get = value.__getitem__
    yield b'{'
    sep = b''
    for start in range(0, len(keys), cap):
        block = keys[start:start + cap]
        children = list(map(get, block))
        if set(map(type, children)) <= _SCALARS:
            yield sep + _enc(dict(zip(block, children)))[1:-1]
            sep = b','
            continue
        group, weight = {}, 0
        for key, child in zip(block, children):
            w = _weight(child, cap)
            if w > cap:
                if group:
                    yield sep + leaf(group)[1:-1]
                    sep, group, weight = b',', {}, 0
                yield sep + _enc(key) + b':'
                sep = b','
                yield from iter_encoded(child, cap, depth + 1, leaf)
                continue
            if group and weight + w > cap:
                yield sep + leaf(group)[1:-1]
                sep, group, weight = b',', {}, 0
            group[key] = child
            weight += w
        if group:
            yield sep + leaf(group)[1:-1]
            sep = b','
    yield b'}'


def _list_pieces(value, cap, depth, leaf):
    yield b'['
    sep = b''
    for start in range(0, len(value), cap):
        block = value[start:start + cap]
        if set(map(type, block)) <= _SCALARS:
            yield sep + _enc(block)[1:-1]
            sep = b','
            continue
        group, weight = [], 0
        for child in block:
            w = _weight(child, cap)
            if w > cap:
                if group:
                    yield sep + leaf(group)[1:-1]
                    sep, group, weight = b',', [], 0
                if sep:
                    yield sep
                sep = b','
                yield from iter_encoded(child, cap, depth + 1, leaf)
                continue
            if group and weight + w > cap:
                yield sep + leaf(group)[1:-1]
                sep, group, weight = b',', [], 0
            group.append(child)
            weight += w
        if group:
            yield sep + leaf(group)[1:-1]
            sep = b','
    yield b']'


# ---- the record's snapshot (engine thread) and its bytes (writer) ------------------------------------

class _Private:
    """A record value nothing else holds (a scalar, the frames, the prompt list): encoded as it is."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def pieces(self, cap):
        return iter_encoded(self.value, cap)


class _Snapshot:
    """A container other owners may hold, copied now: its pickle (protocol 5: every value and
    type exactly), or its JSON when pickle cannot take it. ``copy``: the record holds
    ``copy_receipt`` of it (the drain receipt)."""
    __slots__ = ('pickled', 'raw', 'copy')

    def __init__(self, value, copy=False):
        self.copy = copy
        try:
            self.pickled, self.raw = pickle.dumps(value, protocol=5), None
        except Exception:
            self.pickled, self.raw = None, _enc(value)      # its bytes now; the encoder's errors raise here

    def pieces(self, cap):
        if self.pickled is not None:
            return iter_encoded(pickle.loads(self.pickled), cap, leaf=_copy_enc if self.copy else _enc)
        if self.copy:
            return iter_encoded(json.loads(self.raw), cap)  # copy_receipt's result is plain JSON
        return iter((self.raw,))


class _Frames:
    """A request's worker frames as filed (composition.WorkerFrames: each the JSON text the
    worker's receipt had when it answered, immutable bytes): on the writer each is parsed
    (``worker_frame``: copy_receipt's copy) and encoded. ``[`` + the frames' JSON joined by
    ``,`` + ``]`` is the JSON of the list of the parsed frames, as ``_Private`` wrote it."""
    __slots__ = ('items',)

    def __init__(self, items):
        self.items = tuple(items)

    def __len__(self):
        return len(self.items)

    def pieces(self, cap):
        from .composition import worker_frame
        yield b'['
        for i, item in enumerate(self.items):
            if i:
                yield b','
            yield from iter_encoded(worker_frame(item), cap)
        yield b']'


def filed_frames(receipts, request):
    """``terminal_value``'s ``frames``: the request's filed frames, not parsed here (a plain dict
    of parsed frames, a harness's: the frames as they are)."""
    raw = getattr(receipts, 'raw', None)
    if raw is None:
        return list(receipts.get(request, []))
    return _Frames(raw(request))


class _Members:
    """A dict with string keys snapshotted member by member: (key bytes, _Snapshot) in key order."""
    __slots__ = ('members',)

    def __init__(self, members):
        self.members = members

    def pieces(self, cap):
        yield b'{'
        for i, (key, member) in enumerate(self.members):
            yield (b',' + key if i else key) + b':'
            yield from member.pieces(cap)
        yield b'}'


def drain_snapshot(receipt):
    """``terminal_value``'s ``drain``: the drain receipt copied now, member by member when its
    keys are strings (copy_receipt then composes by member: ``iter_encoded``)."""
    if type(receipt) is dict and receipt and set(map(type, receipt)) <= _STR:
        return _Members(tuple((_enc(k), _Snapshot(receipt[k], True)) for k in sorted(receipt)))
    return _Snapshot(receipt, True)


def snapshot(value):
    """The native record as (key bytes, piece) in key order: the frames, the prompt list and
    the scalars as they are, every other container copied now (``_Snapshot``), the drain as
    ``drain_snapshot`` made it. ``assemble`` turns it into ``encoded(value)`` with
    ``value['drain']`` the receipt's copy, as native_terminal wrote it."""
    parts = []
    for key in sorted(value):
        v = value[key]
        t = type(v)
        if t is _Members or t is _Snapshot or t is _Frames:
            piece = v
        elif key in PRIVATE or t in _SCALARS:
            piece = _Private(v)
        else:
            piece = _Snapshot(v)
        parts.append((_enc(key), piece))
    return parts


def assemble(parts, cap=PIECE, tick=None):
    """The native file's bytes from ``snapshot`` parts; ``tick()`` runs after every piece."""
    out = [b'{']
    for i, (key, piece) in enumerate(parts):
        out.append((b',' + key if i else key) + b':')
        for chunk in piece.pieces(cap):
            out.append(chunk)
            if tick is not None:
                tick()
    out.append(b'}')
    return b''.join(out)


def goal_row(selection, output, value):
    """goal_text without its append: (reference, the goals.jsonl line or None).

    remote_ra_http.evidence.goal_text's steps on the engine thread (the pending record taken
    once, the row encoded, its SHA-256 named and remembered); the line is appended on the
    writer, before the native file, as goal_text appended it before save."""
    texts = getattr(selection, 'goal_texts', None)
    if texts is None:
        return None, None
    request = value['request_id']
    reference = texts.written.get(request)
    if reference is not None:
        return reference, None
    record = texts.pending.pop(request, None)
    if record is None:
        return None, None
    from .goal_text import SCHEMA, FILE, KEYS
    raw = _enc(dict({k: value[k] for k in KEYS}, schema=SCHEMA, **record))
    reference = texts.written[request] = dict(file=FILE, sha256=sha256(raw).hexdigest())
    return reference, (Path(output) / FILE, raw + b'\n')


# ---- the writer ----------------------------------------------------------------------------------

class Held:
    """A Cancel's acknowledgement that waits for its request's writes (engine thread only)."""
    __slots__ = ('request', 'reply')

    def __init__(self, request, reply):
        self.request, self.reply = request, reply


class _Job:
    __slots__ = ('request', 'session', 'row', 'path', 'parts', 'released')

    def __init__(self, request, session, row, path, parts):
        self.request, self.session, self.row, self.path, self.parts = request, session, row, path, parts
        self.released = None


class CompletionWriter:
    """The completion files of one engine, written by one thread in submission order.

    The engine thread stages jobs (``defer``), releases them after the step's outputs
    (``release``), joins a request, a session or everything (``join_request``,
    ``join_session``, ``flush``) and delivers held acknowledgements (``deliver``)."""

    def __init__(self, output, decoupled, *, wake=None, pause=None, slice_s=SLICE_S, cap=PIECE):
        require(type(decoupled) is bool and Path(output).is_absolute(), 'completion writer needs the evidence output')
        self.output, self.decoupled = Path(output), decoupled
        self.wake = wake
        self.pause = (lambda: _sleep(PAUSE_S)) if pause is None else pause
        self.slice_s, self.cap = slice_s, cap
        self.owner = threading.get_ident()
        self.cond = threading.Condition()
        self.staged, self.queue = [], deque()
        self.requests, self.sessions = {}, {}       # pending jobs (staged, queued, running)
        self.pending = 0
        self.waiters = {}                            # request -> held acknowledgements
        self.ready = deque()                         # acknowledgements the engine thread sends
        self.paths = set()
        self.error = None
        self.urgent = 0                              # engine-thread joins waiting now
        self.closed = self.stopping = False
        self.thread = None
        self.held = None                             # set by the Cancel wrapper inside handle_steps
        self.counters = dict(jobs=0, bytes=0, held=0, pauses=0, errors=0)
        self.times = dict(queue_s=0.0, encode_s=0.0, write_s=0.0, held_s=0.0, held_max_s=0.0)

    # ---- lifecycle ------------------------------------------------------------------------
    def start(self):
        require(self.thread is None, 'completion writer started twice')
        self.thread = threading.Thread(target=self._run, name='remote-ra-completion-writer', daemon=True)
        self.thread.start()
        atexit.register(self._at_exit)
        return self

    def close(self, timeout=STOP_S):
        """Shutdown (engine thread): every write done and the thread stopped; terminals after
        this are synchronous. Raises the first write failure."""
        if self.closed:
            self.check()
            return
        try:
            self._wait(lambda: not self.pending, timeout)
        finally:
            self.closed = True
            with self.cond:
                self.stopping = True
                self.cond.notify_all()
            thread = self.thread
            if thread is not None:
                thread.join(timeout)
            atexit.unregister(self._at_exit)
            self.deliver()
        require(thread is None or not thread.is_alive(), 'completion writer did not stop')
        self.check()

    def _at_exit(self):
        """Interpreter exit without shutdown: the staged and queued writes still land."""
        try:
            self.release()
            deadline = _clock() + STOP_S
            with self.cond:
                while self.pending and _clock() < deadline and self.thread is not None and self.thread.is_alive():
                    self.cond.wait(0.5)
        except BaseException as exc:          # never raise at exit; say it
            sys.stderr.write('remote-ra completion writer at exit: %s: %s\n' % (type(exc).__name__, exc))

    def check(self):
        """The first write failure, raised wherever the engine asks the writer for anything."""
        error = self.error
        if error is not None:
            raise RuntimeError('COMPLETION_WRITE_FAILED: %s: %s' % (type(error).__name__, error)) from error

    def _on_thread(self):
        require(threading.get_ident() == self.owner, 'completion writer: engine thread only')

    # ---- engine thread --------------------------------------------------------------------
    def defer(self, endpoint, entry, operation):
        """native_terminal with the writer attached: True when this request's files are staged;
        False when the caller runs the synchronous path (after every write is done): a
        cancellation, a retry, or a closed writer."""
        self._on_thread()
        if operation != 'complete' or self.closed:
            self.flush()
            return False
        binding = entry['binding']
        path = self.output / ('native-' + sha256(binding.request.encode()).hexdigest() + '.json')
        if path in self.paths:
            self.flush()
            return False
        self.check()
        from remote_ra_http import evidence
        core = endpoint.core
        value = evidence.terminal_value(endpoint, entry, operation, drain=drain_snapshot, frames=filed_frames)
        selection = core._remote_ra_private_v2_endpoint.selection
        reference, row = goal_row(selection, self.output, value)
        if reference is not None:
            value['goal_text'] = reference
        if self.decoupled:
            # Both pools after this request's drain, read now (other sessions may hold blocks).
            value['kv_pools'] = dict(free_blocks_by_pool=evidence.kv_pools(core)['free_blocks_by_pool'])
        job = _Job(binding.request, binding.session, row, path, self.snapshot(value))
        self.paths.add(path)
        with self.cond:
            self.staged.append(job)
            self.requests[job.request] = self.requests.get(job.request, 0) + 1
            self.sessions[job.session] = self.sessions.get(job.session, 0) + 1
            self.pending += 1
        return True

    def snapshot(self, value):
        return snapshot(value)

    def release(self):
        """After the step's outputs (post_step): the staged jobs go to the writer, in order."""
        if not self.staged:
            return
        now = _clock()
        with self.cond:
            for job in self.staged:
                job.released = now
            self.queue.extend(self.staged)
            self.staged = []
            self.cond.notify_all()

    def flush(self):
        self._wait(lambda: not self.pending)

    def join_request(self, request):
        self._wait(lambda: not self.requests.get(request))

    def join_session(self, session):
        self._wait(lambda: not self.sessions.get(session))

    def _wait(self, done, timeout=None):
        self.release()
        with self.cond:
            if not done():
                deadline = None if timeout is None else _clock() + timeout
                self.urgent += 1
                try:
                    while not done():
                        require(self.thread is not None and self.thread.is_alive(),
                                'completion writer stopped with writes pending')
                        require(deadline is None or _clock() < deadline, 'completion writes did not finish')
                        self.cond.wait(0.5)
                finally:
                    self.urgent -= 1
        if threading.get_ident() == self.owner:
            self.deliver()
        self.check()

    def hold(self, request):
        """A Cancel of a completed request: whether its acknowledgement waits for its writes."""
        self.check()
        self.release()
        with self.cond:
            return bool(self.requests.get(request))

    def after(self, request, callback):
        """``callback(error)`` once ``request``'s writes are done: now, on this thread, if they
        are; else on the writer when the last one ends."""
        with self.cond:
            if self.requests.get(request):
                self.waiters.setdefault(request, []).append(callback)
                return
        callback(self.error)

    def deliver(self):
        """Send the acknowledgements whose writes ended (engine thread: lifecycle tick, joins)."""
        while self.ready:
            send, call_id, reply, failure, error, held = self.ready.popleft()
            waited = _clock() - held
            self.times['held_s'] += waited
            self.times['held_max_s'] = max(self.times['held_max_s'], waited)
            if error is not None:
                reply, failure = None, RuntimeError('COMPLETION_WRITE_FAILED: %s: %s' % (type(error).__name__, error))
            send(call_id, reply, failure)

    def stats(self):
        """Counters and times (s) for the step profile's lines; read from any thread."""
        with self.cond:
            value = dict(self.counters, pending=self.pending, staged=len(self.staged), queued=len(self.queue))
        value.update({k: round(v * 1000, 3) for k, v in ((k.replace('_s', '_ms'), v) for k, v in self.times.items())})
        return value

    # ---- the writer thread ----------------------------------------------------------------
    def _run(self):
        while True:
            with self.cond:
                while not self.queue and not self.stopping:
                    self.cond.wait()
                if not self.queue:
                    return
                job = self.queue.popleft()
            error = self.error
            if error is None:
                try:
                    self._write(job)
                except BaseException as exc:
                    error = exc
                    self._failed(exc)
            self._done(job, error)

    def _write(self, job):
        """goal_text's append, then save's create: the same calls, the same bytes."""
        start = _clock()
        self.times['queue_s'] += start - job.released
        if job.row is not None:
            path, line = job.row
            with path.open('ab') as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        begin = _clock()
        mark = [begin]

        def tick():
            if self.urgent:
                return              # the engine thread waits for this job: no pause
            now = _clock()
            if now - mark[0] >= self.slice_s:
                self.counters['pauses'] += 1
                self.pause()
                mark[0] = _clock()
        raw = assemble(job.parts, self.cap, tick)
        encoded = _clock()
        with job.path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        end = _clock()
        self.times['encode_s'] += encoded - begin
        self.times['write_s'] += (begin - start) + (end - encoded)
        self.counters['jobs'] += 1
        self.counters['bytes'] += len(raw)

    def _failed(self, exc):
        with self.cond:
            if self.error is None:
                self.error = exc
            self.counters['errors'] += 1
        try:
            from remote_ra_http.diagnostics import failure
            failure('COMPLETION_WRITE_FAILED', exc)
        except BaseException:
            pass
        try:
            sys.stderr.write('remote-ra completion writer: %s: %s\n' % (type(exc).__name__, exc))
        except BaseException:
            pass

    def _done(self, job, error):
        with self.cond:
            self.pending -= 1
            left = self.requests[job.request] - 1
            if left:
                self.requests[job.request] = left
                callbacks = ()
            else:
                del self.requests[job.request]
                callbacks = self.waiters.pop(job.request, ())
            left = self.sessions[job.session] - 1
            if left:
                self.sessions[job.session] = left
            else:
                del self.sessions[job.session]
            self.cond.notify_all()
        for callback in callbacks:
            try:
                callback(error if error is not None else self.error)
            except BaseException as exc:
                self._failed(exc)

    # ---- held acknowledgements --------------------------------------------------------------
    def held_ack(self, send, call_id, held, failure):
        """``send(call_id, frame, failure)`` for a ``Held`` reply once its request's writes end."""
        now = _clock()
        self.counters['held'] += 1

        def ready(error):
            self.ready.append((send, call_id, held.reply, failure, error, now))
            if threading.get_ident() == self.owner:
                self.deliver()
            elif self.wake is not None:
                try:
                    self.wake()
                except BaseException:
                    pass                # the loop delivers at its next pass anyway
        self.after(held.request, ready)


# ---- attachment (compose_engine) -------------------------------------------------------------------

def attach_engine(core, lifecycle, dispatcher, *, ticker, wake, loop=None, config=None, factory=None, **writer):
    """--deferred-completion-io, once at startup: the writer and its engine touch points.

    ``core`` gets ``_remote_ra_completion_writer`` (native_terminal and shutdown_engine look
    for it); ``loop`` (the EngineCoreProc, ``core`` by default) releases staged jobs in its
    ``post_step`` and sends held acknowledgements through its ``_remote_ra_ack``;
    ``ticker.tick`` (the compat lifecycle's, called every input pass) delivers them;
    ``wake()`` wakes an idle loop. ``dispatcher`` is the RegisteredAddDispatcher with its
    RegisteredRouteBridge (``_remote_ra_live_v2``). ``factory`` and ``writer``: the writer's
    class (CompletionWriter) and options (tests)."""
    if config is None:
        from remote_ra_http.config import load
        config = load()
    require(pinned(config) is True, 'deferred completion I/O is not pinned')
    from remote_ra_http.evidence import decoupled
    require(getattr(core, '_remote_ra_completion_writer', None) is None, 'completion writer attached twice')
    loop = core if loop is None else loop
    bridge = dispatcher._remote_ra_live_v2
    require(bridge is not None and bridge.lifecycle is lifecycle, 'registered route bridge of another lifecycle')
    out = (factory or CompletionWriter)(config['output_directory'], decoupled(config), wake=wake, **writer)

    post_step = loop.post_step

    def released_post_step(*args, **kwargs):
        out.release()
        return post_step(*args, **kwargs)
    loop.post_step = released_post_step

    ack = loop._remote_ra_ack

    def held_ack(call_id, frame, failure):
        if type(frame) is not Held:
            return ack(call_id, frame, failure)
        out.held_ack(ack, call_id, frame, failure)
    loop._remote_ra_ack = held_ack

    tick = ticker.tick

    def delivering_tick(*args, **kwargs):
        if out.ready:
            out.deliver()
        return tick(*args, **kwargs)
    ticker.tick = delivering_tick

    handle_steps = dispatcher.handle_steps

    def held_handle_steps(frame):
        out.held = None
        try:
            reply = yield from handle_steps(frame)
        finally:
            request, out.held = out.held, None
        return reply if request is None else Held(request, reply)
    dispatcher.handle_steps = held_handle_steps

    cancel = bridge.cancel

    def holding_cancel(dispatcher, command):
        ack = cancel(dispatcher, command)
        if ack.state == 'complete' and out.hold(command.request_id):
            out.held = command.request_id
        return ack
    bridge.cancel = holding_cancel

    add_steps = bridge.add_steps

    def joined_add_steps(dispatcher, command, fingerprint):
        out.join_session(command.session_id)
        return add_steps(dispatcher, command, fingerprint)
    bridge.add_steps = joined_add_steps

    close_session = bridge.close_session

    def joined_close_session(dispatcher, command):
        out.join_session(command.session_id)
        return close_session(dispatcher, command)
    bridge.close_session = joined_close_session

    release = getattr(core, '_remote_ra_idle_release', None)
    if release is not None:
        evict = release.evict

        def joined_evict(session_id, request_id, short):
            out.join_session(session_id)
            return evict(session_id, request_id, short)
        release.evict = joined_evict
    core._remote_ra_completion_writer = out
    return out.start()
