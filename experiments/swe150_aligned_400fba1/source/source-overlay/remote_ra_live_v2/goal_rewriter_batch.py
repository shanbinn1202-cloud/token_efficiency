"""Batched Goal rewriter calls (``serve.py --goal-rewriter-batch N``, opt-in; off by default).

Without the flag the goal_query Goal rewriter is the upstream
``PersistentCommandGoalRewriter``: one child process (the vendored
``qwen_goal_rewriter.py --serve``) that serves one call at a time under the
parent's lock, so with ``--admission-overlap K`` the sessions' calls queue for
it (goals.jsonl ``wait_ms``). With ``--goal-rewriter-batch N`` (2 <= N <= K <= 16)
the calls that queue are sent together, up to N at a time, to a child that
runs them as one batched ``generate`` (``goal_rewriter_child``).

Only how the calls are executed changes. When the rewriter is called (the
gate), what it is sent (the upstream envelope line, byte for byte), how its
answer is parsed and validated, the fallbacks and the Goal constructor are the
upstream ones:

- The constructor keeps an upstream ``PersistentCommandGoalRewriter`` instance
  (``inner``): its type, and with it the Goal control hash, is unchanged. Its
  ``rewrite`` and ``close`` are replaced on the instance (as goal_text's tap
  replaces ``rewrite``).
- Each call runs the upstream ``PersistentCommandGoalRewriter.rewrite`` on its
  own thread, against a ``_Ticket`` that stands in for the lock, the child
  process and the line reader: the upstream code numbers the request, encodes
  its envelope line, reads its answer line, checks the request id, parses the
  result and validates the candidate, exactly as it does alone.
- One dispatcher thread owns the child through a second, plain upstream
  instance (``child``: its ``_start`` with the readiness check, its
  ``select`` line reader and its ``close``). Whenever the child is idle it
  takes every queued call, up to N, and writes one line ``{"batch": [envelope
  line, ...]}``; the child answers one line ``{"batch": [answer, ...]}``. Each
  call gets its own answer object, encoded canonically: the line the child
  would have printed for that call alone. There is no collection window; a
  lone call goes out at once as a batch of one.
- Timeouts and failures as upstream: the pinned per-call timeout runs from the
  send of the call's batch (time queued before it is not charged). A failure of
  the batch's exchange (the child's start, a broken pipe, a timeout, the child
  closing its output, an unreadable or mismatched answer line) closes the
  child, as upstream closes it after a failed call, and every call of that
  batch fails with that error, which the upstream code turns into its
  fallback_reason; the next batch starts the child again. A call's own answer
  (a child error, a candidate the parent refuses) concerns that call only.
- ``close`` (server shutdown) fails the queued calls ("persistent rewrite
  process is not running"), ends an in-flight batch and joins the dispatcher.
- Token cap (``--goal-rewriter-batch-tokens T``, pinned as goal_rewriter_batch_tokens;
  0: none). Each call's thread counts its prompt tokens before it queues, with the
  pinned checkpoint's tokenizer and the child's own prompt construction
  (``counter``: the child's ``prepare``, on the host, without the model). A batch takes
  the queued calls in order while its size times its longest prompt stays within T and
  stops at the first call that does not fit; a call over T alone (or one that could not
  be counted) is a batch of one, as unbatched. Without a cap, any N queued calls.

Evidence (goal_text): each call reports to the Goal-text tap (``RewriterTap.stamp``,
registered through ``inner._remote_ra_timing`` at startup) when its batch formed
(``locked``), was sent and answered, its answer line and its batch
(``batch_id``, the server's batch ordinal, and ``batch_size``). Nothing is
checked per call; ``check`` verifies once, at startup, that the authority's
rewriter runs batched with the pinned N and token cap.
"""
from __future__ import annotations

from collections import deque
import contextlib
import copy
import importlib
import itertools
import json
import threading
import time

from .contract import require
import policy_registry

CONFIG_KEY = 'goal_rewriter_batch'      # startup.json and ready.json, only with --goal-rewriter-batch N
TOKENS_KEY = 'goal_rewriter_batch_tokens'   # with it: the per-batch token cap T (0: none)
LIMITS = (2, 16)
TOKENS_LIMIT = 16 * 262144              # the largest batch size x the longest context
BATCH = 'batch'
CHILD = 'goal_rewriter_child.py'        # this directory: the batched child script


def canonical_text(value):
    """The upstream child's line encoding (``_write_json``)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def pinned(config):
    """The pinned --goal-rewriter-batch N, or None (off: no key). Startup only: an integer
    2-16, goal_query with a pinned Goal rewriter, and at most --admission-overlap K; its
    token cap pinned with it (``pinned_tokens``)."""
    if CONFIG_KEY not in config:
        require(TOKENS_KEY not in config, 'goal_rewriter_batch_tokens needs goal_rewriter_batch')
        return None
    value = config[CONFIG_KEY]
    require(type(value) is int and LIMITS[0] <= value <= LIMITS[1],
            'goal_rewriter_batch must be an integer %d-%d; off is pinned by leaving it out' % LIMITS)
    contract = config.get('selection_contract')
    require(policy_registry.goal_engine(config.get('policy')) and type(contract) is dict and
            contract.get('goal_rewriter') is not None,
            'goal_rewriter_batch needs goal_query with a pinned Goal rewriter (--goal-rewriter-model)')
    from .admission_flags import CONFIG_KEY as ADMISSION, permits
    require(value <= permits(config.get(ADMISSION)),
            'goal_rewriter_batch N needs --admission-overlap K >= N: only K Goals are built at once')
    pinned_tokens(config)
    return value


def pinned_tokens(config):
    """The pinned per-batch token cap T of a batched server (0: none), or None (not
    batched). Startup only."""
    if CONFIG_KEY not in config:
        require(TOKENS_KEY not in config, 'goal_rewriter_batch_tokens needs goal_rewriter_batch')
        return None
    value = config.get(TOKENS_KEY)
    require(type(value) is int and 0 <= value <= TOKENS_LIMIT,
            'goal_rewriter_batch_tokens must be an integer 0-%d (0: no cap)' % TOKENS_LIMIT)
    return value


class _HostTorch:
    """torch for counting prompt tokens in the server process: host tensors, no CUDA call."""
    cuda = type('NoCuda', (), {'is_available': staticmethod(lambda: False)})()

    def __init__(self, torch):
        self._torch = torch

    def __getattr__(self, name):
        return getattr(self._torch, name)


def counter(tokenizer, max_new_tokens, torch):
    """The prompt token count of an envelope line as the batched child builds its prompt:
    the child's ``prepare`` (the upstream generate up to its model call) on a generator
    with ``tokenizer``, on the host, without the model, the parser or the prefix function."""
    from . import goal_rewriter_child as C
    U = C.upstream()
    generator = U.QwenGoalGenerator.__new__(U.QwenGoalGenerator)
    generator.torch, generator.tokenizer, generator.device = _HostTorch(torch), tokenizer, 'cpu'
    generator.max_new_tokens, generator.model = max_new_tokens, None
    generator.JsonSchemaParser = lambda schema: None
    generator.build_prefix_fn = lambda tokenizer_, parser: None

    def count(line):
        return int(C.prepare(generator, json.loads(line)['request'])['input_ids'].shape[1])
    return count


def prompt_counter(pin):
    """Startup only (a token cap): the ``counter`` with the pinned checkpoint's tokenizer,
    loaded as the upstream child loads it; it must count a probe request."""
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(pin['model']), local_files_only=True, trust_remote_code=True)
    count = counter(tokenizer, pin['max_new_tokens'], torch)
    probe = count(canonical_text({'request_id': '1', 'request': {'root_goal': 'probe'}}))
    require(type(probe) is int and probe > 0, 'the Goal rewriter prompt counter does not count')
    return count


def _fresh(error):
    """A copy of ``error`` to raise on one call's thread (one batch failure, several calls)."""
    try:
        return copy.copy(error)
    except Exception:
        return error


class _Ticket:
    """One call: what the upstream ``PersistentCommandGoalRewriter.rewrite`` uses of its
    instance (lock, ``_start``, process stdin, request number, timeout, ``_readline``,
    ``close``, pending startup latency), backed by the batcher."""

    _lock = contextlib.nullcontext()

    def __init__(self, batcher, number):
        self.batcher, self._next_request_id = batcher, number
        self.timeout_seconds = batcher.child.timeout_seconds
        self._pending_startup_latency_ms = 0.0
        self._process = self.stdin = self       # write and flush land here
        self.text, self.answer, self.error = '', None, None
        self.locked = self.sent = self.answered = None
        self.batch = None                       # (batch id, batch size)
        self.tokens = None                      # its prompt tokens (a token cap only)
        self.done = threading.Event()

    def _start(self):
        pass                                    # the dispatcher starts the child for each batch

    def write(self, text):
        self.text += text

    def flush(self):
        self.tokens = self.batcher.measure(self.text)
        self.batcher.submit(self)

    def _readline(self, timeout_seconds):
        """The call's answer line, once its batch was answered; or its batch's failure."""
        while not self.done.wait(1.0):
            if not self.batcher.alive() and not self.done.is_set():
                self.fail(self.batcher.errors.RewriteCommandError('persistent rewrite process is not running'))
        stamp = self.batcher.stamp
        if stamp is not None:
            try:
                stamp(response=None if self.answer is None else self.answer[:-1], locked=self.locked,
                      sent=self.sent, answered=self.answered, batch=self.batch)
            except Exception:
                pass                            # recording never fails a rewrite
        if self.error is not None:
            raise _fresh(self.error)
        return self.answer

    def close(self):
        pass                                    # the dispatcher already closed the child on any failure

    def complete(self, line, answered):
        self.answer, self.answered = line, answered
        self.done.set()

    def fail(self, error):
        if not self.done.is_set():
            self.error = error
            self.done.set()


class BatchedRewriter:
    """The batcher behind one upstream rewriter instance (``inner``)."""

    def __init__(self, inner, child, limit, cap=0, count=None):
        require(type(limit) is int and LIMITS[0] <= limit <= LIMITS[1], 'goal rewriter batch must be 2-16')
        require(type(cap) is int and 0 <= cap <= TOKENS_LIMIT and (cap == 0 or callable(count)),
                'a goal rewriter token cap is an integer 0-%d and needs a prompt counter' % TOKENS_LIMIT)
        require(isinstance(child, type(inner)), 'the batched rewriter needs two upstream rewriter instances')
        self.inner, self.child, self.limit = inner, child, limit
        self.cap, self.count = cap, count
        self._counting = threading.Lock()       # one fast tokenizer, many Goal threads
        self.errors = importlib.import_module(type(inner).__module__)
        self.upstream_rewrite = type(inner).rewrite
        self.stamp = None
        self._cond = threading.Condition()
        self._queue = deque()
        self._closed = False
        self._numbers = itertools.count(1)
        self._batches = itertools.count(1)
        self._thread = threading.Thread(target=self._dispatch, name='remote-ra-goal-rewriter-batch', daemon=True)

    # ---- startup -----------------------------------------------------------------------------
    def install(self):
        """Start the child (its readiness check) and the dispatcher; the calls of ``inner``
        are batched from now on. Startup only."""
        inner = self.inner
        require(getattr(inner, '_remote_ra_batch', None) is None and inner._process is None,
                'the rewriter instance is already running or batched')
        self.child.__enter__()
        inner.rewrite, inner.close = self.rewrite, self.close
        inner._remote_ra_timing, inner._remote_ra_batch = self.timing, self
        self._thread.start()
        return inner

    def timing(self, stamp):
        """goal_text.RewriterTap registers its ``stamp`` here once, at startup."""
        self.stamp = stamp

    # ---- one call (its own thread) -----------------------------------------------------------
    def rewrite(self, request):
        return self.upstream_rewrite(_Ticket(self, next(self._numbers)), request)

    def measure(self, line):
        """With a token cap, the call's prompt tokens (its own thread, before it queues);
        None without a cap or when it cannot be counted: then the call runs alone."""
        if not self.cap:
            return None
        try:
            with self._counting:
                value = self.count(line)
        except Exception:
            return None
        return value if type(value) is int and value > 0 else None

    def submit(self, ticket):
        with self._cond:
            if self._closed or not self._thread.is_alive():
                raise self.errors.RewriteCommandError('persistent rewrite process is not running')
            self._queue.append(ticket)
            self._cond.notify()

    def alive(self):
        return self._thread.is_alive()

    # ---- the dispatcher ----------------------------------------------------------------------
    def _dispatch(self):
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if self._closed:
                    queued = list(self._queue)
                    self._queue.clear()
                    break
                batch = self._take()
            try:
                self._serve(batch)
            except BaseException as error:      # never leave a call waiting
                for ticket in batch:
                    ticket.fail(error)
        for ticket in queued:
            ticket.fail(self.errors.RewriteCommandError('persistent rewrite process is not running'))

    def _take(self):
        """The next batch, in queue order (under the lock): up to N calls; with a token cap,
        only while the batch's size times its longest prompt stays within it, stopping at
        the first call that does not fit. The first call always goes; one over the cap, or
        not counted, goes alone."""
        batch = [self._queue.popleft()]
        longest = batch[0].tokens
        if self.cap and (longest is None or longest > self.cap):
            return batch
        while self._queue and len(batch) < self.limit:
            if self.cap:
                tokens = self._queue[0].tokens
                if tokens is None or (len(batch) + 1) * max(longest, tokens) > self.cap:
                    break
                longest = max(longest, tokens)
            batch.append(self._queue.popleft())
        return batch

    def _serve(self, batch):
        child = self.child
        formed, number = time.perf_counter(), next(self._batches)
        for ticket in batch:
            ticket.locked, ticket.batch = formed, (number, len(batch))
        try:
            child._start()                      # upstream: (re)start and readiness check if not running
            batch[0]._pending_startup_latency_ms = child._pending_startup_latency_ms
            child._pending_startup_latency_ms = 0.0
            line = '{"%s":[%s]}\n' % (BATCH, ','.join(ticket.text.rstrip('\n') for ticket in batch))
            process = child._process
            sent = time.perf_counter()
            for ticket in batch:
                ticket.sent = sent
            process.stdin.write(line)
            process.stdin.flush()
            raw = child._readline(child.timeout_seconds)
            answered = time.perf_counter()
            value = json.loads(raw)
            if not (type(value) is dict and set(value) == {BATCH} and type(value[BATCH]) is list and
                    len(value[BATCH]) == len(batch)):
                raise self.errors.RewriteCommandError('persistent rewrite batch answer does not match its batch')
        except BaseException as error:
            failed = time.perf_counter()
            for ticket in batch:
                if ticket.sent is None:
                    ticket.sent = failed
            child.close()                       # as upstream after a failed call; the next batch restarts it
            for ticket in batch:
                ticket.fail(error)
            return
        for ticket, answer in zip(batch, value[BATCH]):
            ticket.complete(canonical_text(answer) + '\n', answered)

    # ---- shutdown ----------------------------------------------------------------------------
    def close(self):
        """Fail the queued calls, end an in-flight batch, join the dispatcher, close the child."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        process = self.child._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()             # an in-flight batch's reader sees the child end
            except OSError:
                pass
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(30)
        self.child.close()


def start(rewriter, argv, pin, limit, cap=0):
    """The batched Goal rewriter of ``serve.py --goal-rewriter-batch N``: the upstream
    instance the constructor holds, its calls batched by up to ``limit``, within the token
    ``cap`` (0: none; else counted with the pinned checkpoint's tokenizer). Startup only."""
    make = lambda: rewriter.PersistentCommandGoalRewriter(argv, timeout_seconds=pin['timeout_seconds'],
                                                          startup_timeout_seconds=pin['startup_timeout_seconds'])
    count = prompt_counter(pin) if cap else None
    return BatchedRewriter(make(), make(), limit, cap, count).install()


def check(rewriter, limit, cap=0):
    """Startup only: the authority's rewriter is batched with the pinned N and token cap
    (``limit`` None: not batched)."""
    batcher = getattr(rewriter, '_remote_ra_batch', None)
    if limit is None:
        require(batcher is None, 'the Goal rewriter is batched although goal_rewriter_batch is not pinned')
        return None
    require(isinstance(batcher, BatchedRewriter) and batcher.inner is rewriter and batcher.limit == limit and
            batcher.cap == cap and batcher.alive(), 'the Goal rewriter does not run batched with the pinned N '
            'and token cap')
    return limit
