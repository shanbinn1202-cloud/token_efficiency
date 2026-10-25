"""goal_query evidence: the texts behind each turn's Goal hashes (goals.jsonl).

Recording only: nothing here changes a Goal, a selection or the rewriter.

The runtime already hashes two texts of every goal_query turn: the Goal
(``goal_hash``, the GoalRecord's ``version_hash``: SHA-256 of the canonical
JSON of its selector payload, i.e. schema version, root goal, retrieval Goal
and policy hint) and the Goal query (``goal.payload_sha256``: SHA-256 of the
``<retrieval_query>`` text the Goal forward runs). ``record`` keeps both texts,
so their SHA-256 are those hashes, with the Goal decision and the rewriter
exchanges of the turn's construction. The engine appends one row per terminal
goal_query request to ``goals.jsonl`` (remote_ra_http.evidence.native_terminal)
and names it in the native file and the runtime row by its SHA-256.

Rewriter exchanges. ``RewriterTap`` sets three attributes on the one rewriter
instance; its type, lock, process and answers are unchanged, so is the
constructor's config and with it the Goal control hash (checked at startup).
``rewrite`` records the request payload exactly as the persistent rewriter
encodes it (``RewriteRequest.to_payload()``, sorted keys, compact) and how the
call ended (an exception as ``Type: message``); ``_start`` when the call got
the rewriter and was sent, ``_readline`` (PersistentCommandGoalRewriter) the
raw line the Qwen child answered with. A
rewriter without lines (the network-free fakes) gives the canonical JSON of the
returned ``raw_response`` instead. Only calls made inside ``capture()`` on the
same thread are recorded; every Goal construction (``GoalWork.run``, on the
engine thread or a Goal thread) opens its own.

The child's answer line is ``{"request_id":..,"result":{"candidate":{..},..}}``
(the candidate as the child parsed it from the generated text) or
``{"request_id":..,"error":"..","telemetry":{..}}``. The vendored child
(``dram_kv_scripts/qwen_goal_rewriter.py``, the upstream blob) does not send the
generated text itself, so a generation it could not parse is recorded by its
error and token counts only. A request the constructor refuses before calling
the rewriter (a RewriteRequest over its bounds) has no exchange; its
``fallback_reason`` says why.

Timing of each exchange (evidence only; ``time.perf_counter``, not the
``perf_counter_ns`` clock of the other evidence). ``wait_ms``: from the call's
entry until its turn came, i.e. the rewriter's lock acquired, the queueing
behind other sessions' calls (``_start`` is the first thing the upstream
``rewrite`` does under that lock); ``start_ms``: the child process (re)start
inside the call, 0 unless the child had died or was closed by an earlier
timeout; ``call_ms``: from the send to the answer line (``_readline``), or to
the failure when no line arrived. A rewriter without these hooks (the
network-free stand-ins of the tests) has ``wait_ms`` and ``start_ms`` null and
``call_ms`` the whole call. ``input_tokens``, ``output_tokens`` and
``generation_latency_ms`` are copied from the answer line (the child's own
prompt and generated token counts and its ``generate`` time; ``result`` or
``telemetry``), null where the line has none. ``totals`` sums one
construction's ``wait_ms`` and ``call_ms`` for its native file. Nothing here
changes when or how the rewriter is called, what it is sent or how its answer
is read.

Batched calls (``serve.py --goal-rewriter-batch N``, goal_rewriter_batch). The
batcher runs the calls itself and reports each call's times, answer line and
batch through ``RewriterTap.stamp`` (registered once, at startup, through the
rewriter's ``_remote_ra_timing``), on the call's own thread: ``wait_ms`` is
then the time queued until its batch formed, ``start_ms`` a child (re)start and
the batch line, ``call_ms`` the batch's send to its answer, and the exchange
adds ``batch_id`` (the server's batch ordinal) and ``batch_size`` (the calls
sent together).
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import json
import threading
import time

from .contract import require

SCHEMA = 'remote-ra-goal-text-v1'
FILE = 'goals.jsonl'
# Row keys taken from the native terminal record of the same request.
KEYS = ('request_id', 'session_id', 'session_index', 'epoch', 'turn', 'local_turn', 'terminal_state')
MAX_ERROR_CHARS = 4096
# Per exchange: the measured times (ms) and the child's own counts from its answer line.
TIMING = ('wait_ms', 'start_ms', 'call_ms')
TELEMETRY = ('input_tokens', 'output_tokens', 'generation_latency_ms')
BATCH = ('batch_id', 'batch_size')     # batched servers only (--goal-rewriter-batch N)
# Native-file totals of one construction's exchanges: (native key, exchange key).
TOTALS = (('goal_rewrite_wait_ms', 'wait_ms'), ('goal_rewrite_call_ms', 'call_ms'))


def canonical_text(value):
    """The upstream encoding of ``GoalRecord._stable_hash`` and of the persistent
    rewriter's request line."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def goal_json(goal):
    """The text whose SHA-256 is ``goal.version_hash`` (the runtime's ``goal_hash``)."""
    return canonical_text(goal.selector_payload)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _request_text(request):
    try:
        return canonical_text(request.to_payload())
    except Exception:           # recording never fails a rewrite
        return None


def _response_text(result):
    try:
        return canonical_text(_plain(result.raw_response))
    except Exception:
        return None


def _error_text(error):
    return ('%s: %s' % (type(error).__name__, error))[:MAX_ERROR_CHARS]


def _ms(start, end):
    return None if start is None or end is None else round((end - start) * 1000, 3)


def _telemetry(response):
    """The child's own counts in an answer line (its ``result``, else its error
    ``telemetry``, else, for a line-less rewriter, the raw response itself)."""
    try:
        value = json.loads(response)
        source = (value['result'] if isinstance(value.get('result'), dict) else
                  value['telemetry'] if isinstance(value.get('telemetry'), dict) else value)
        return {k: source[k] if type(source.get(k)) in (int, float) else None for k in TELEMETRY}
    except Exception:           # no line, or not an object: nothing reported
        return dict.fromkeys(TELEMETRY)


def _measured(entered, stamps, ended, response):
    """One exchange's timing (TIMING) and the child's counts (TELEMETRY)."""
    try:
        locked, sent, answered = (stamps.get(k) for k in ('locked', 'sent', 'answered'))
        value = dict(wait_ms=_ms(entered, locked), start_ms=_ms(locked, sent),
                     call_ms=_ms(entered if sent is None else sent, ended if answered is None else answered))
        batch = stamps.get('batch')
        if batch is not None:   # a batched server (goal_rewriter_batch)
            value.update(zip(BATCH, batch))
    except Exception:           # recording never fails a rewrite
        value = dict.fromkeys(TIMING)
    value.update(_telemetry(response))
    return value


def totals(rewrites):
    """The native-file totals (TOTALS) of one construction's exchanges; none without
    a rewriter call, null where an exchange has no such time."""
    if not rewrites:
        return {}
    out = {}
    for key, name in TOTALS:
        values = [exchange.get(name) for exchange in rewrites]
        out[key] = None if any(v is None for v in values) else round(sum(values), 3)
    return out


class RewriterTap:
    """Records the rewriter exchanges of the Goal construction running on this thread."""

    def __init__(self, rewriter):
        self.local = threading.local()
        local, rewrite = self.local, rewriter.rewrite

        def tapped_rewrite(request):
            exchanges = getattr(local, 'exchanges', None)
            if exchanges is None:
                return rewrite(request)
            entered = time.perf_counter()
            exchange = dict(request=_request_text(request), response=None, error=None)
            exchanges.append(exchange)
            local.exchange, local.stamps = exchange, {}
            result = None
            try:
                result = rewrite(request)
            except BaseException as error:
                exchange['error'] = _error_text(error)
                raise
            finally:
                ended = time.perf_counter()
                stamps, local.exchange, local.stamps = local.stamps, None, None
                if exchange['error'] is None and exchange['response'] is None:
                    exchange['response'] = _response_text(result)
                exchange.update(_measured(entered, stamps, ended, exchange['response']))
            return result
        rewriter.rewrite = tapped_rewrite
        if callable(getattr(type(rewriter), '_start', None)):
            start = rewriter._start

            def tapped_start():
                # PersistentCommandGoalRewriter.rewrite calls _start first under its lock:
                # its entry is the call's turn, its exit the send (after any restart).
                stamps = getattr(local, 'stamps', None)
                if stamps is None:
                    return start()
                stamps.setdefault('locked', time.perf_counter())
                try:
                    return start()
                finally:
                    stamps.setdefault('sent', time.perf_counter())
            rewriter._start = tapped_start
        if callable(getattr(type(rewriter), '_readline', None)):
            readline = rewriter._readline

            def tapped_readline(timeout_seconds):
                line = readline(timeout_seconds)
                exchange = getattr(local, 'exchange', None)
                if exchange is not None:
                    if 'sent' in local.stamps:      # not a restart's readiness line
                        local.stamps.setdefault('answered', time.perf_counter())
                    exchange['response'] = line[:-1] if line.endswith('\n') else line
                return line
            rewriter._readline = tapped_readline
        # An executor that runs the calls itself (goal_rewriter_batch) reports through stamp.
        register = getattr(rewriter, '_remote_ra_timing', None)
        if callable(register):
            register(self.stamp)

    def stamp(self, response=None, **marks):
        """A call's times (``locked``, ``sent``, ``answered``), answer line and ``batch``
        (id, size), reported on the call's own thread by an executor that runs the calls
        itself (goal_rewriter_batch); outside a capture nothing is recorded."""
        exchange = getattr(self.local, 'exchange', None)
        if exchange is None:
            return
        self.local.stamps.update({k: v for k, v in marks.items() if v is not None})
        if response is not None:
            exchange['response'] = response

    @contextmanager
    def capture(self):
        """The exchanges of the rewriter calls made on this thread inside the block."""
        previous = getattr(self.local, 'exchanges', None)
        exchanges = self.local.exchanges = []
        try:
            yield exchanges
        finally:
            self.local.exchanges = previous


def tap(constructor, control_hash):
    """Tap the constructor's rewriter (None without one); startup only: the
    constructor's control hash is unchanged by it."""
    rewriter = getattr(constructor, 'rewriter', None)
    if rewriter is None:
        return None
    before = control_hash(constructor)
    value = RewriterTap(rewriter)
    require(control_hash(constructor) == before, 'Goal text recording changed the Goal control hash')
    return value


def record(goal, variant, text, goal_hash, payload_sha256, rewrites):
    """One goal_query turn's texts; ``goal_hash`` and ``payload_sha256`` are the
    hashes the runtime computed (the SHA-256 of the two texts)."""
    return dict(
        goal=dict(sha256=goal_hash, text=goal_json(goal)),
        query=dict(variant=variant, sha256=payload_sha256, text=text),
        decision=dict(gate_decision=goal.gate_decision, applied_decision=goal.applied_decision,
                      triggers=list(goal.triggers), changed_fields=list(goal.changed_fields),
                      slm_attempted=goal.slm_attempted, slm_succeeded=goal.slm_succeeded,
                      fallback_reason=goal.fallback_reason),
        rewrites=[dict(exchange) for exchange in rewrites])


class GoalTexts:
    """The texts of this engine's goal_query turns from plan_turn until their
    terminal (``pending``), and the reference of every row written (``written``,
    so a retried terminal names the same row and writes none)."""

    def __init__(self):
        self.pending, self.written = {}, {}


def self_check(sha256, constructor_type, schema):
    """Startup only: the stored texts hash to the runtime's Goal hashes. One
    deterministic Goal of a throwaway constructor (no rewriter, no session of
    this server) must give ``sha256(goal_json(goal)) == goal.version_hash``."""
    context = schema.TurnContext('goal-text-self-check', 'goal-text-self-check', 1,
                                 'Fix the failing check in pkg/mod.py (ValueError: bad header).', None, None,
                                 'task', message_source='user',
                                 structured_state={'tool_name': 'bash', 'return_code': None})
    goal = constructor_type(None).construct(context)
    require(sha256(goal_json(goal).encode('utf-8')).hexdigest() == goal.version_hash,
            'Goal text encoding differs from the GoalRecord version hash')
    return True
