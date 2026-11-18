"""The Goal rewriter on vLLM (``serve.py --goal-rewriter-engine vllm``, opt-in; off by default).

Without the flag (``hf``, the default) the goal_query Goal rewriter is the upstream
``PersistentCommandGoalRewriter`` and its Hugging Face child (or the batched child of
``--goal-rewriter-batch N``): nothing here runs and no key is pinned. With it, one child
(``goal_rewriter_vllm_child``) runs Qwen3-0.6B in a small vLLM ``AsyncLLM`` engine with
continuous batching, and several calls are in flight at once (docs/rewriter-vllm/DESIGN.md).

Only the model call changes. When the rewriter is called, what it is sent (the upstream
envelope line, byte for byte), the prompt, the JSON schema, greedy decoding with the pinned
max_new_tokens, the decode, ``parse_exact_candidate``, the parent's answer checks and
validation, the fallbacks, the 120-s call timeout and the answer line are the upstream ones:

- The constructor keeps the upstream ``PersistentCommandGoalRewriter`` instance (``inner``):
  its type, and with it the Goal control hash, is unchanged. Its ``rewrite`` and ``close``
  are replaced on the instance, as ``goal_rewriter_batch`` replaces them.
- Each call runs the upstream ``PersistentCommandGoalRewriter.rewrite`` on its own thread,
  against a ``_Ticket`` that stands in for the lock, the child process and the line reader:
  the upstream code numbers the request, encodes its envelope line, reads its answer line,
  checks the request id, parses the result and validates the candidate, as it does alone.
- One child, started at server startup (``install``: in the engine process, once the 27B
  engine has allocated its KV cache and before READY; its start time goes to ready.json).
  Its command puts only the clean environment's site-packages on ``sys.path`` (not the
  overlay, whose patched ``vllm`` would load), its environment has no ``REMOTE_RA_*``,
  ``VLLM_POLICY_*`` or ``SELECTIVE_*`` variable, and it runs in its own session: closing it
  ends its whole process group, so no EngineCore is left holding GPU memory.
- The child prints the upstream readiness record; its stdout carries protocol lines only.
  Each call's line is written under one write lock; a reader thread routes each answer
  line, by its ``request_id``, to its call, so answers may come in any order.
- A call's timeout (the pinned 120 s from its send) fails that call only ("persistent
  rewrite process timed out", as upstream) and sends ``{"abort": request_id}``: the child
  aborts that request in its engine; nothing is restarted (a vLLM restart costs a minute).
  A child exit fails every call in flight ("persistent rewrite process closed stdout
  (exit=N)") and the next call starts the child again (its start inside that call).
- ``close`` (server shutdown) fails the calls in flight, closes the child's input (it shuts
  its engine down), then ends the child's process group.

Evidence (goal_text): each call reports to the tap (``RewriterTap.stamp``, registered
through ``inner._remote_ra_timing``) ``locked`` when it may (re)start the child (after
another call's restart of the child, if one runs), ``sent`` when its line is written (after
its own restart, if any) and ``answered`` when its answer was routed, and its answer line.
So ``wait_ms`` is near 0 (the queueing behind other calls now happens inside the engine and
counts in ``call_ms``), ``start_ms`` is this call's restart of the child and the write of
its line. No batch keys. Nothing is checked per call; ``check`` verifies once, at startup,
that the authority's rewriter runs on vLLM with the pinned record.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import itertools
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import policy_registry
import threading
import time

from .contract import require

CONFIG_KEY = 'goal_rewriter_engine'     # startup.json and ready.json, only with --goal-rewriter-engine vllm
ENGINES = ('hf', 'vllm')                # serve.py --goal-rewriter-engine (hf: the default, no key)
ENGINE = 'vllm'
BACKENDS = ('lm-format-enforcer', 'xgrammar')   # --goal-rewriter-guided-backend
DEFAULT_BACKEND = 'lm-format-enforcer'          # the enforcer the Hugging Face child uses
# vLLM 0.24's xgrammar (and guidance, so also auto) refuse the schema's uniqueItems: the
# xgrammar arm passes the schema without it (a measurement arm; the parent still refuses
# duplicates).
WITHOUT_UNIQUE_ITEMS = ('xgrammar',)
KV_GIB = (2, 16)                        # --goal-rewriter-vllm-kv-gib G: an integer in range
DEFAULT_KV_GIB = 3                      # about 28K tokens at 112 KiB per token
GIB = 1 << 30
# vLLM's startup gate once the KV bytes are explicit: it needs this share of the GPU free.
GPU_MEMORY_UTILIZATION = 0.05
MAX_NUM_SEQS = 16
MAX_MODEL_LEN = 16384                   # the bounded request is at most ~7.5K prompt tokens (recorded calls: 7,544)
MAX_NUM_BATCHED_TOKENS = 8192
CUDAGRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16)
BATCH_INVARIANT_ENV = 'VLLM_BATCH_INVARIANT'    # --goal-rewriter-batch-invariant: set in the child only
SCRUBBED = ('REMOTE_RA_', 'VLLM_POLICY_', 'SELECTIVE_')
OFFLINE = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
CHILD = 'goal_rewriter_vllm_child.py'   # this directory: the vLLM child script
ABORT = 'abort'                         # {"abort": request_id}: abort one request in the child
STARTUP_KEY = 'startup_ms'              # worker-attached.json and ready.json: the eager start
CLOSE_SECONDS = 10.0                    # shutdown: the child's own engine shutdown, then the group
TERM_SECONDS = 5.0                      # SIGTERM to the group, then SIGKILL
EXIT_SECONDS = 5.0                      # a closed stdout: the wait for the child's exit code
PROGRAM = ("import sys,runpy;sys.path[:0]=[sys.argv[1]];script=sys.argv[2];"
           "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")


def canonical_text(value):
    """The upstream child's line encoding (``_write_json``)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def log(message):
    sys.stderr.write('goal rewriter vllm: %s\n' % message)
    sys.stderr.flush()


# ---- the pinned record --------------------------------------------------------------------------

def engine_args(model, backend, kv_bytes):
    """The child's ``AsyncEngineArgs`` (JSON; the child builds vLLM's config objects from the
    two nested dicts)."""
    return dict(model=str(model), tokenizer=str(model), dtype='bfloat16', distributed_executor_backend='uni',
                tensor_parallel_size=1, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, kv_cache_memory_bytes=kv_bytes,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION, enable_prefix_caching=False, enforce_eager=False,
                compilation_config=dict(cudagraph_capture_sizes=list(CUDAGRAPH_CAPTURE_SIZES)),
                structured_outputs_config=dict(backend=backend), generation_config='vllm', disable_log_stats=True)


def record(model, backend=DEFAULT_BACKEND, kv_gib=DEFAULT_KV_GIB, batch_invariant=False):
    """The ``goal_rewriter_engine`` record of --goal-rewriter-engine vllm for the pinned Goal
    rewriter checkpoint ``model``: engine, structured-output backend, whether the schema
    keeps uniqueItems, KV bytes, the utilization gate, batch invariance and the child's
    engine arguments."""
    require(backend in BACKENDS, '--goal-rewriter-guided-backend must be one of %s' % ', '.join(BACKENDS))
    require(type(kv_gib) is int and KV_GIB[0] <= kv_gib <= KV_GIB[1],
            '--goal-rewriter-vllm-kv-gib must be an integer %d-%d' % KV_GIB)
    require(type(batch_invariant) is bool, '--goal-rewriter-batch-invariant is a switch')
    require(type(model) is str and model, 'the vLLM Goal rewriter needs the pinned checkpoint path')
    kv = kv_gib * GIB
    return dict(engine=ENGINE, guided_backend=backend, schema_unique_items=backend not in WITHOUT_UNIQUE_ITEMS,
                kv_cache_memory_bytes=kv, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                batch_invariant=batch_invariant, engine_args=engine_args(model, backend, kv))


def pinned(config):
    """The pinned record, or None (hf: no key). Startup only: goal_query with a pinned Goal
    rewriter, --admission-overlap K >= 2, no goal_rewriter_batch, and exactly the record
    ``record`` builds from the pinned checkpoint and the record's own settings."""
    if CONFIG_KEY not in config:
        return None
    value = config[CONFIG_KEY]
    contract = config.get('selection_contract')
    require(policy_registry.goal_engine(config.get('policy')) and type(contract) is dict and
            type(contract.get('goal_rewriter')) is dict,
            'goal_rewriter_engine vllm needs goal_query with a pinned Goal rewriter (--goal-rewriter-model)')
    from .admission_flags import CONFIG_KEY as ADMISSION, permits
    require(permits(config.get(ADMISSION)) >= 2,
            'goal_rewriter_engine vllm needs --admission-overlap K >= 2: otherwise only one call is ever in flight')
    require('goal_rewriter_batch' not in config, 'goal_rewriter_engine vllm excludes goal_rewriter_batch')
    require(type(value) is dict and type(value.get('kv_cache_memory_bytes')) is int and
            value['kv_cache_memory_bytes'] % GIB == 0, 'goal_rewriter_engine must be the record of the vLLM rewriter')
    expected = record(contract['goal_rewriter']['model'], value.get('guided_backend'),
                      value['kv_cache_memory_bytes'] // GIB, value.get('batch_invariant'))
    require(value == expected, 'goal_rewriter_engine differs from the record of its settings and pinned checkpoint')
    return value


def ready_record(config, attached):
    """ready.json's ``goal_rewriter_engine``: the pinned record with the child's eager start
    time from worker-attached.json (``attached``); None without the pin (then the file
    carries none)."""
    if CONFIG_KEY not in config:
        require(CONFIG_KEY not in attached, 'the vLLM Goal rewriter started although it is not pinned')
        return None
    value = attached.get(CONFIG_KEY)
    require(type(value) is dict and set(value) == {STARTUP_KEY} and type(value[STARTUP_KEY]) in (int, float) and
            value[STARTUP_KEY] >= 0, 'the engine did not record the vLLM Goal rewriter start')
    return dict(config[CONFIG_KEY], **{STARTUP_KEY: value[STARTUP_KEY]})


# ---- the child's command and environment ---------------------------------------------------------

def child_argv(pin, value, source, site, executable=None):
    """The vLLM child's command: the isolated interpreter of the upstream rewriter command
    (-I -S -B -u) with only the clean ``site`` on sys.path, the child script of the sealed
    ``source`` tree, the upstream model arguments and the pinned record."""
    from hashlib import sha256
    model = Path(pin['model'])
    require((model/'config.json').is_file() and sha256((model/'config.json').read_bytes()).hexdigest()
            == pin['config_sha256'], 'Goal rewriter checkpoint differs from its startup pin')
    require(value['engine_args']['model'] == str(model), 'the vLLM Goal rewriter record names another checkpoint')
    script = Path(source)/'remote_ra_live_v2'/CHILD
    return (executable or sys.executable, '-I', '-S', '-B', '-u', '-c', PROGRAM, str(site), str(script),
            '--model', str(model), '--max-new-tokens', str(pin['max_new_tokens']), '--local-files-only', '--serve',
            '--engine', canonical_text(value))


def child_env(value, environ=None):
    """The child's environment: the server's without the runtime's variables (REMOTE_RA_*,
    VLLM_POLICY_*, SELECTIVE_*), offline, and VLLM_BATCH_INVARIANT=1 only with batch
    invariance."""
    environ = os.environ if environ is None else environ
    env = {k: v for k, v in environ.items() if not k.upper().startswith(SCRUBBED)}
    env.pop(BATCH_INVARIANT_ENV, None)
    env.update(OFFLINE)
    if value['batch_invariant']:
        env[BATCH_INVARIANT_ENV] = '1'
    return env


def stop(process, grace=TERM_SECONDS):
    """End ``process`` and its process group (its own session: the group id is its pid), so
    that its EngineCore ends too: SIGTERM to the group, SIGKILL to what is left after
    ``grace`` seconds. Without process groups (not Linux) the process alone."""
    killpg = getattr(os, 'killpg', None)
    if killpg is None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(grace)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return
    group = process.pid

    def alive():
        process.poll()                          # reap the leader: a zombie stays in the group
        try:
            killpg(group, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not alive():
            break
        try:
            killpg(group, sig)
        except (ProcessLookupError, PermissionError):
            break
        deadline = time.monotonic() + grace
        while alive() and time.monotonic() < deadline:
            time.sleep(0.05)
    process.poll()


# ---- the transport ----------------------------------------------------------------------------

def _fresh(error):
    """A copy of ``error`` to raise on one call's thread."""
    try:
        return copy.copy(error)
    except Exception:
        return error


class _Child:
    """One child process and its reader thread."""

    def __init__(self, transport, process):
        self.process = process
        self.first = queue.Queue()          # the readiness line (None: stdout closed before it)
        self.ended = threading.Event()
        self.thread = threading.Thread(target=transport._read, args=(self,), daemon=True,
                                       name='remote-ra-goal-rewriter-vllm-reader')


class _Ticket:
    """One call: what the upstream ``PersistentCommandGoalRewriter.rewrite`` uses of its
    instance (lock, ``_start``, process stdin, request number, timeout, ``_readline``,
    ``close``, pending startup latency), backed by the transport."""

    _lock = contextlib.nullcontext()

    def __init__(self, transport, number):
        self.transport, self._next_request_id = transport, number
        self.request_id = str(number)
        self.timeout_seconds = transport.timeout_seconds
        self._pending_startup_latency_ms = 0.0
        self._process = self.stdin = self       # write and flush land here
        self.text, self.answer, self.error = '', None, None
        self.locked = self.sent = self.answered = None
        self.child = None                       # the child this call was sent to
        self.done = threading.Event()

    def _start(self):
        self.transport.ensure(self)             # (re)start the child if it is not running

    def write(self, text):
        self.text += text

    def flush(self):
        self.transport.send(self)

    def _readline(self, timeout_seconds):
        """The call's answer line; or its failure (timeout: this request is aborted)."""
        if not self.done.wait(timeout_seconds):
            self.transport.timed_out(self)
            if not self.done.wait(EXIT_SECONDS):
                self.fail(self.transport.errors.RewriteCommandError('persistent rewrite process timed out'))
        stamp = self.transport.stamp
        if stamp is not None:
            try:
                stamp(response=None if self.answer is None else self.answer[:-1], locked=self.locked,
                      sent=self.sent, answered=self.answered)
            except Exception:
                pass                            # recording never fails a rewrite
        if self.error is not None:
            raise _fresh(self.error)
        return self.answer

    def close(self):
        self.transport.unroute(self)           # a failed call; the child keeps running

    def complete(self, line, answered):
        if not self.done.is_set():
            self.answer, self.answered = line, answered
            self.done.set()

    def fail(self, error):
        if not self.done.is_set():
            self.error = error
            self.done.set()


class VllmRewriter:
    """The vLLM child behind one upstream rewriter instance (``inner``)."""

    def __init__(self, inner, argv, value, env=None, stderr=None, popen=subprocess.Popen):
        require(isinstance(argv, tuple) and argv and all(type(a) is str and a for a in argv),
                'the vLLM Goal rewriter needs its child command')
        self.inner, self.argv, self.record, self.env, self.stderr, self.popen = inner, argv, value, env, stderr, popen
        self.timeout_seconds = inner.timeout_seconds
        self.startup_timeout_seconds = inner.startup_timeout_seconds
        self.errors = importlib.import_module(type(inner).__module__)
        self.upstream_rewrite = type(inner).rewrite
        self.stamp = None
        self.startup_ms = None                  # the eager start (install), for ready.json
        self.starts = 0
        self._numbers = itertools.count(1)
        self._start_lock = threading.Lock()     # one (re)start at a time
        self._write_lock = threading.Lock()     # one line at a time
        self._routes_lock = threading.Lock()
        self._routes = {}                       # request id -> the call in flight
        self._child = None
        self._closed = False
        self._pending_startup_latency_ms = 0.0

    # ---- startup -----------------------------------------------------------------------------
    def install(self):
        """Start the child (its readiness check) and route the calls of ``inner`` to it from now
        on. Startup only; a child that does not start fails the server's startup."""
        inner = self.inner
        require(getattr(inner, '_remote_ra_vllm', None) is None and getattr(inner, '_remote_ra_batch', None) is None
                and inner._process is None, 'the rewriter instance is already running, batched or on vLLM')
        with self._start_lock:
            self._child, latency_ms = self._spawn()
            self.starts = 1
        self.startup_ms = round(latency_ms, 3)
        self._pending_startup_latency_ms = latency_ms   # the first call reports it, as upstream
        inner.rewrite, inner.close = self.rewrite, self.close
        inner._remote_ra_timing, inner._remote_ra_vllm = self.timing, self
        return inner

    def timing(self, stamp):
        """goal_text.RewriterTap registers its ``stamp`` here once, at startup."""
        self.stamp = stamp

    def alive(self):
        child = self._child
        return not self._closed and child is not None and not child.ended.is_set()

    def _spawn(self):
        """A new child, ready: (child, its start in ms); the upstream errors otherwise."""
        Error = self.errors.RewriteCommandError
        started = time.perf_counter()
        try:
            process = self.popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                 text=True, encoding='utf-8', bufsize=1, env=self.env, start_new_session=True)
        except OSError as error:
            raise Error(f"persistent rewrite process failed to start: {error}") from error
        child = _Child(self, process)
        child.thread.start()
        try:
            line = child.first.get(timeout=self.startup_timeout_seconds)
        except queue.Empty:
            self._end(child)
            raise Error("persistent rewrite process timed out") from None
        if line is None:
            code = self._exit_code(child)
            self._end(child)
            raise Error(f"persistent rewrite process closed stdout (exit={code})")
        try:
            ready = json.loads(line)
        except json.JSONDecodeError as error:
            self._end(child)
            raise Error(f"persistent rewrite process failed to start: {error}") from error
        rewriter = self.errors
        expected = {"protocol": rewriter._PERSISTENT_PROTOCOL, "version": rewriter._PERSISTENT_PROTOCOL_VERSION,
                    "ready": True}
        if ready != expected:
            self._end(child)
            raise Error("persistent rewrite process returned an invalid readiness record")
        return child, (time.perf_counter() - started) * 1_000

    # ---- one call (its own thread) -----------------------------------------------------------
    def rewrite(self, request):
        return self.upstream_rewrite(_Ticket(self, next(self._numbers)), request)

    def ensure(self, ticket):
        """The call's turn (``locked``): the running child, or a new one started inside the call."""
        Error = self.errors.RewriteCommandError
        with self._start_lock:
            ticket.locked = time.perf_counter()
            if self._closed:
                raise Error('persistent rewrite process is not running')
            child = self._child
            if child is None or child.ended.is_set() or child.process.poll() is not None:
                if child is not None:
                    self._end(child)            # what is left of its group
                self._child = None
                child, latency_ms = self._spawn()
                if self._closed:                # closed while it started
                    self._end(child)
                    raise Error('persistent rewrite process is not running')
                self._child, self.starts = child, self.starts + 1
                ticket._pending_startup_latency_ms = latency_ms
            elif self._pending_startup_latency_ms:
                ticket._pending_startup_latency_ms, self._pending_startup_latency_ms = \
                    self._pending_startup_latency_ms, 0.0
            ticket.child = child

    def send(self, ticket):
        """Route the call's answer to it, then write its line (``sent``)."""
        child = ticket.child
        with self._routes_lock:
            require(ticket.request_id not in self._routes, 'Goal rewriter request id already in flight')
            self._routes[ticket.request_id] = ticket
        try:
            with self._write_lock:
                ticket.sent = time.perf_counter()
                if child is None or child.ended.is_set():
                    raise BrokenPipeError('the rewrite process ended')
                child.process.stdin.write(ticket.text)
                child.process.stdin.flush()
        except ValueError as error:             # its input was closed meanwhile (shutdown)
            self.unroute(ticket)
            raise BrokenPipeError(str(error)) from None
        except BaseException:
            self.unroute(ticket)
            raise

    def unroute(self, ticket):
        with self._routes_lock:
            if self._routes.get(ticket.request_id) is ticket:
                del self._routes[ticket.request_id]
                return True
        return False

    def timed_out(self, ticket):
        """The call's timeout: it fails alone, and its request is aborted in the child."""
        if not self.unroute(ticket):
            return                              # answered or failed meanwhile
        ticket.fail(self.errors.RewriteCommandError('persistent rewrite process timed out'))
        child = ticket.child
        try:
            with self._write_lock:
                if child is not None and not child.ended.is_set():
                    child.process.stdin.write(canonical_text({ABORT: ticket.request_id}) + '\n')
                    child.process.stdin.flush()
        except (OSError, ValueError):
            pass                                # the child ended: its reader fails the rest

    # ---- the reader (one thread per child) ---------------------------------------------------
    def _read(self, child):
        """The child's first line goes to ``_spawn`` (readiness); every later line to its call.
        No call is sent before ``_spawn`` accepted the readiness record."""
        stdout, first = child.process.stdout, True
        try:
            for line in iter(stdout.readline, ''):
                if first:
                    first = False
                    child.first.put(line)
                    continue
                self._route(child, line)
        except (OSError, ValueError):
            pass                                # its pipe closed (the child ended)
        finally:
            child.first.put(None)
            child.ended.set()
            self._ended(child)

    def _route(self, child, line):
        try:
            value = json.loads(line)
            request_id = value.get('request_id') if isinstance(value, dict) else None
        except ValueError:
            request_id = None
        ticket = None
        with self._routes_lock:
            candidate = self._routes.get(request_id) if type(request_id) is str else None
            if candidate is not None and candidate.child is child:
                ticket = self._routes.pop(request_id)
        if ticket is None:
            log('dropped an answer line without a call in flight: %s' % line[:200].rstrip())
            return
        ticket.complete(line if line.endswith('\n') else line + '\n', time.perf_counter())

    def _ended(self, child):
        """The child's stdout closed: every call in flight on it fails."""
        with self._routes_lock:
            failed = [t for t in self._routes.values() if t.child is child]
            for ticket in failed:
                del self._routes[ticket.request_id]
        if not failed:
            return
        error = self.errors.RewriteCommandError(
            f"persistent rewrite process closed stdout (exit={self._exit_code(child)})")
        for ticket in failed:
            ticket.fail(error)

    @staticmethod
    def _exit_code(child):
        try:
            return child.process.wait(EXIT_SECONDS)
        except subprocess.TimeoutExpired:
            return child.process.poll()

    def _end(self, child):
        """End a child that failed to start or died: its process group, its pipes."""
        stop(child.process)
        for stream in (child.process.stdin, child.process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        child.ended.set()

    # ---- shutdown ----------------------------------------------------------------------------
    def close(self):
        """Fail the calls in flight, close the child's input (its engine shuts down), end its
        process group. Idempotent."""
        self._closed = True
        child, self._child = self._child, None
        with self._routes_lock:
            pending = list(self._routes.values())
            self._routes.clear()
        error = self.errors.RewriteCommandError('persistent rewrite process is not running')
        for ticket in pending:
            ticket.fail(error)
        if child is None:
            return
        try:
            with self._write_lock:
                child.process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            child.process.wait(CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        self._end(child)
        if child.thread.is_alive() and child.thread is not threading.current_thread():
            child.thread.join(CLOSE_SECONDS)


def start(rewriter, pin, value, source, site, executable=None, stderr=None):
    """The vLLM Goal rewriter of ``serve.py --goal-rewriter-engine vllm``: the upstream
    instance the constructor holds, its calls sent to one vLLM child, started now (in the
    engine process, before READY). The child's stderr is the server's (``stderr`` None).
    Startup only."""
    argv = child_argv(pin, value, source, site, executable)
    inner = rewriter.PersistentCommandGoalRewriter(argv, timeout_seconds=pin['timeout_seconds'],
                                                   startup_timeout_seconds=pin['startup_timeout_seconds'])
    return VllmRewriter(inner, argv, value, env=child_env(value), stderr=stderr).install()


def check(rewriter, value):
    """Startup only: the authority's rewriter runs on vLLM with the pinned record (``value``
    None: not on vLLM)."""
    transport = getattr(rewriter, '_remote_ra_vllm', None)
    if value is None:
        require(transport is None, 'the Goal rewriter runs on vLLM although goal_rewriter_engine is not pinned')
        return None
    require(isinstance(transport, VllmRewriter) and transport.inner is rewriter and transport.record == value and
            transport.alive(), 'the Goal rewriter does not run on vLLM with the pinned record')
    return value


def attached(rewriter, value):
    """worker-attached.json's ``goal_rewriter_engine`` (startup only): the child's eager start."""
    transport = getattr(rewriter, '_remote_ra_vllm', None)
    require(isinstance(transport, VllmRewriter) and transport.record == value and
            type(transport.startup_ms) is float, 'the vLLM Goal rewriter did not start at startup')
    return {STARTUP_KEY: transport.startup_ms}
