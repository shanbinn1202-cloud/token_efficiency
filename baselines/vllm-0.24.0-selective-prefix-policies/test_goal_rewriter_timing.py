"""goal_query: the Goal rewriter's per-call timing (remote_ra_live_v2.goal_text, goals.jsonl). CPU only.

The Goal rewriter is one process (the upstream PersistentCommandGoalRewriter)
that serves one call at a time under its lock. With --admission-overlap 16 the
sixteen sessions' Goal constructions run on their own threads and their
rewriter calls queue for that lock. Here the upstream persistent rewriter keeps
its own rewrite, lock and answer parsing; only its child is scripted: it
answers every request line with a valid candidate after a fixed delay.

- Sixteen sessions' INITIALIZE calls at once: the child serves one call at a
  time; each exchange's call_ms is the child's time; in the order the child
  served them, each call's wait_ms is the time the calls served before it took
  (the waits add up: the sixteenth waited for fifteen calls); wait + start +
  call is the call's time in rewrite() and the construction covers it.
- One session at a time: nothing waits.
- The --admission-overlap park table: each parked Goal that called the
  rewriter records its exchanges' totals (goal_rewrite_wait_ms,
  goal_rewrite_call_ms) next to goal_construct_ms, which covers them; a
  construction without a call (KEEP or PATCH) records no totals.
- A timeout's call_ms runs to the failure and it reports no counts; the child
  is then closed (upstream), and the next call's start_ms is the restart.
"""
import io
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

from remote_ra_live_v2 import admission_overlap as AO, goal_async as GA, goal_text, query_selection
from remote_ra_live_v2 import selection_contract

query_selection.upstream()      # the packaged dram_kv namespace
from dram_kv import schema      # noqa: E402
from dram_kv.goals import maintained, rewriter as R     # noqa: E402

DELAY = 0.04                    # the scripted child's time per call
SESSIONS = 16
SLACK_MS = 60.0                 # thread start-up and scheduling (Windows timer granularity)


def line(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def good(envelope):
    request = envelope['request']
    candidate = dict(description='Retrieve evidence for %s' % request['trigger'], phase_hint='hypothesis',
                     files=request['features']['files'][:2], evidence_needed=['diagnostics'])
    return line(dict(request_id=envelope['request_id'], result=dict(
        candidate=candidate, input_tokens=120, output_tokens=24, generation_latency_ms=1.5, model_cost=0.0)))


def timeout(envelope):
    return R.RewriteCommandError('persistent rewrite process timed out')


class Child:
    def __init__(self):
        self.stdin, self.stdout = io.StringIO(), None

    def poll(self):
        return None


class SlowRewriter(R.PersistentCommandGoalRewriter):
    """The upstream persistent rewriter whose child answers each request line after
    ``delay`` seconds (``answers[k]`` for call k, default ``good``); starting the child
    takes ``startup`` seconds. ``served``: (root goal, send, answer) per call, in the
    order the child served them."""

    def __init__(self, delay=DELAY, startup=0.0, answers=None):
        super().__init__(('scripted-child',), timeout_seconds=5.0, startup_timeout_seconds=5.0)
        self.delay, self.startup, self.answers = delay, startup, dict(answers or {})
        self.served, self.starts, self.active, self.overlapped = [], 0, 0, 0

    def _start(self):
        if self._process is None:
            time.sleep(self.startup)
            self._process = Child()
            self.starts += 1

    def _readline(self, timeout_seconds):
        sent = time.perf_counter()
        self.active += 1
        self.overlapped = max(self.overlapped, self.active)
        envelope = json.loads(self._process.stdin.getvalue().splitlines()[-1])
        k = len(self.served)
        time.sleep(self.delay)
        self.active -= 1
        self.served.append((envelope['request']['root_goal'], sent, time.perf_counter()))
        answer = self.answers.get(k, good)(envelope)
        if isinstance(answer, BaseException):
            raise answer
        return answer + '\n'

    def close(self):
        self._process = None


def task(k):
    return 'Fix the failing check %02d in module m%02d.' % (k, k)


def context(k, turn=1):
    """Session k's turn: 1 initializes the Goal (a rewriter call); 2 needs none."""
    if turn == 1:
        return schema.TurnContext('s%02d' % k, 'repo__m%02d' % k, 1, task(k), None, None, task(k),
                                  message_source='user', structured_state={'tool_name': 'bash', 'return_code': None})
    return schema.TurnContext('s%02d' % k, 'repo__m%02d' % k, turn, task(k), None, 'THOUGHT: look\nls',
                              '<returncode>0</returncode>\n<output>\nok\n</output>', message_source='tool',
                              structured_state={'tool_name': 'bash', 'return_code': 0})


def constructors(rewriter):
    """The --admission-overlap constructors (one upstream copy per session) and their tap."""
    own = AO.SessionConstructors(maintained.MaintainedGoalConstructor(
        rewriter, entity_drift_steps=3, initialize_with_rewriter=True))
    return own, goal_text.tap(own, selection_contract.constructor_control_hash)


def started(rewriter):
    rewriter._start()           # the server starts the child at startup (goal_rewriter)
    return rewriter


class ContentionTests(unittest.TestCase):
    def test_sixteen_sessions_queue_for_one_rewriter_and_their_waits_add_up(self):
        rewriter = started(SlowRewriter())
        own, tap = constructors(rewriter)
        barrier, out = threading.Barrier(SESSIONS), {}

        def session(k):
            work = GA.GoalWork('r%02d' % k, own, context(k), tap=tap)
            barrier.wait(10)
            begin = time.perf_counter()
            goal = work.run()
            out[k] = (goal, work.rewrites, (time.perf_counter() - begin) * 1000)
        threads = [threading.Thread(target=session, args=(k,)) for k in range(SESSIONS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(sorted(out), list(range(SESSIONS)))
        # One call per session, served one at a time by the one child.
        self.assertEqual((len(rewriter.served), rewriter.overlapped, rewriter.starts), (SESSIONS, 1, 1))
        spans = sorted((s, a) for _, s, a in rewriter.served)
        self.assertTrue(all(a0 <= s1 for (_, a0), (s1, _) in zip(spans, spans[1:])))
        exchange = {}
        for k, (goal, rewrites, elapsed) in out.items():
            self.assertTrue(goal.slm_attempted and goal.slm_succeeded)
            (x,) = rewrites
            exchange[k] = x
            self.assertEqual((x['error'], x['input_tokens'], x['output_tokens']), (None, 120, 24))
            self.assertGreaterEqual(x['call_ms'], DELAY * 1000 * 0.9)          # the child's time
            self.assertLess(x['start_ms'], SLACK_MS)                          # the child was running
            # wait + start + call: the call's time in rewrite(), inside the construction.
            self.assertLessEqual(x['wait_ms'] + x['start_ms'] + x['call_ms'], elapsed + 1)
            self.assertLess(elapsed - (x['wait_ms'] + x['start_ms'] + x['call_ms']), SLACK_MS)
        # In the child's order: each call waited for the calls served before it.
        order = {root: i for i, (root, _, _) in enumerate(rewriter.served)}
        ranked = sorted(exchange.values(), key=lambda x: order[json.loads(x['request'])['root_goal']])
        before, waits, held = 0.0, [], []
        for i, x in enumerate(ranked):
            self.assertGreaterEqual(x['wait_ms'], before - SLACK_MS, i)
            self.assertLessEqual(x['wait_ms'], before + SLACK_MS + 2 * i, i)
            waits.append(x['wait_ms'])
            held.append(before)
            before += x['start_ms'] + x['call_ms']
        self.assertLess(waits[0], SLACK_MS)                                    # the first one waited for nothing
        self.assertGreaterEqual(waits[-1], (SESSIONS - 1) * DELAY * 1000 * 0.9)  # the last for fifteen calls
        # The waits add up: in total, sixteen calls queued for 0 + 1 + ... + 15 calls.
        self.assertAlmostEqual(sum(waits), sum(held), delta=0.1 * sum(held) + SLACK_MS)
        self.assertGreaterEqual(sum(waits), SESSIONS * (SESSIONS - 1) / 2 * DELAY * 1000 * 0.9)

    def test_one_session_at_a_time_waits_for_nothing(self):
        rewriter = started(SlowRewriter(delay=0.005))
        own, tap = constructors(rewriter)
        for k in range(4):
            work = GA.GoalWork('r%d' % k, own, context(k), tap=tap)
            work.run()
            (x,) = work.rewrites
            self.assertLess(x['wait_ms'], 5)
            self.assertGreaterEqual(x['call_ms'], 5 * 0.9)

    def test_a_timeout_runs_to_its_failure_and_the_restart_is_the_next_calls_start(self):
        rewriter = started(SlowRewriter(delay=0.03, startup=0.05, answers={1: timeout}))
        own, tap = constructors(rewriter)
        exchanges = []
        for k in range(3):
            work = GA.GoalWork('r%d' % k, own, context(k), tap=tap)
            goal = work.run()
            self.assertEqual(goal.slm_succeeded, k != 1)
            exchanges += work.rewrites
        ok, failed, restarted = exchanges
        self.assertEqual(failed['error'], 'RewriteCommandError: persistent rewrite process timed out')
        self.assertIsNone(failed['response'])
        self.assertGreaterEqual(failed['call_ms'], 30 * 0.9)                   # to the failure
        self.assertEqual([failed[k] for k in goal_text.TELEMETRY], [None] * 3)
        self.assertLess(ok['start_ms'], SLACK_MS)
        self.assertEqual(rewriter.starts, 2)                                    # closed by the timeout
        self.assertGreaterEqual(restarted['start_ms'], 50 * 0.9)                # the restart, inside the call
        self.assertLess(restarted['wait_ms'], SLACK_MS)
        self.assertGreaterEqual(restarted['call_ms'], 30 * 0.9)


class ParkTableTests(unittest.TestCase):
    """OverlapGoalAdmissions (--admission-overlap 16): every session's Goal parked at
    once on its own thread; the native totals next to goal_construct_ms."""

    @staticmethod
    def admission(work):
        def run():
            value = yield work
            return value
        return run()

    def park_all(self, own, tap, turn):
        entries = {'r%02d-%d' % (k, turn): {} for k in range(SESSIONS)}
        table = GA.OverlapGoalAdmissions(NS(entries=entries), wake=lambda: None, goals=True, permits=SESSIONS)
        self.addCleanup(table.close)
        works, replies = {}, {}
        for k in range(SESSIONS):
            rid = 'r%02d-%d' % (k, turn)
            works[rid] = GA.GoalWork(rid, own, context(k, turn), tap=tap)
            self.assertIs(table.start(self.admission(works[rid]),
                                      lambda frame, failure, rid=rid: replies.__setitem__(rid, (frame, failure))),
                          table.PARKED)
        deadline = time.monotonic() + 30
        while table.parked and time.monotonic() < deadline:
            if table.ready():
                table.resume()
            else:
                time.sleep(0.002)
        self.assertEqual(table.parked, {})
        self.assertTrue(all(failure is None for _, failure in replies.values()))
        return entries, works, replies

    def test_parked_goals_record_their_calls_totals_next_to_goal_construct_ms(self):
        rewriter = started(SlowRewriter())
        own, tap = constructors(rewriter)
        entries, works, replies = self.park_all(own, tap, 1)
        self.assertEqual((len(rewriter.served), rewriter.overlapped), (SESSIONS, 1))
        waits = []
        for rid, entry in entries.items():
            (x,) = works[rid].rewrites
            self.assertTrue(replies[rid][0].slm_attempted)
            self.assertEqual(set(entry), {'goal_async_ms', 'goal_construct_ms', 'goal_rewrite_wait_ms',
                                          'goal_rewrite_call_ms'})
            self.assertEqual((entry['goal_rewrite_wait_ms'], entry['goal_rewrite_call_ms']), (x['wait_ms'], x['call_ms']))
            self.assertGreaterEqual(entry['goal_construct_ms'] + 1, x['wait_ms'] + x['start_ms'] + x['call_ms'])
            self.assertGreaterEqual(entry['goal_async_ms'], entry['goal_construct_ms'])
            waits.append(entry['goal_rewrite_wait_ms'])
        # Sixteen Goals parked at once, their calls served one at a time: the waits add up.
        self.assertGreaterEqual(sum(waits), SESSIONS * (SESSIONS - 1) / 2 * DELAY * 1000 * 0.9)
        self.assertGreaterEqual(max(waits), (SESSIONS - 1) * DELAY * 1000 * 0.9)
        # The next turn needs no rewriter call (KEEP or PATCH): no totals.
        calls = len(rewriter.served)
        entries, works, replies = self.park_all(own, tap, 2)
        self.assertEqual(len(rewriter.served), calls)
        for rid, entry in entries.items():
            self.assertFalse(replies[rid][0].slm_attempted)
            self.assertEqual(works[rid].rewrites, ())
            self.assertEqual(set(entry), {'goal_async_ms', 'goal_construct_ms'})

    def test_totals(self):
        self.assertEqual(goal_text.totals(()), {})
        self.assertEqual(goal_text.totals([dict(wait_ms=1.5, call_ms=2.25), dict(wait_ms=0.5, call_ms=1.0)]),
                         dict(goal_rewrite_wait_ms=2.0, goal_rewrite_call_ms=3.25))
        self.assertEqual(goal_text.totals([dict(wait_ms=None, call_ms=2.0)]),
                         dict(goal_rewrite_wait_ms=None, goal_rewrite_call_ms=2.0))


if __name__ == '__main__':
    unittest.main()
