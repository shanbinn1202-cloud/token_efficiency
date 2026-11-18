"""A one-session goal_query server answers the heartbeat while its Goal is built (review 1110, runtime 1.5).

On a one-session server (``max_num_seqs <= 1``) goal_async attaches no park table: the Goal and its rewriter
call run in place on the engine thread, the only thread that answers the manager's heartbeat (NativeEngine.poll;
90 s on both sides, remote_ra_linux/native_authority.py). The rewriter's pinned timeout is 120 s (a child start
300 s), after which the construction falls back to the deterministic fields: the server was killed at about
90 s, before the documented fallback.

``goal_async.InlineGoals`` runs the same construction on a thread of its own while the engine thread waits and
answers the heartbeat; the admission stays synchronous. CPU only; the times here are the real ones scaled down
(a 0.5-s heartbeat deadline, a rewriter call of 1 s; the engine thread answers every 0.02 s).

* InlineGoalsTests: the construction's value or failure comes back unchanged (also a TimeoutError of its own);
  the heartbeat is answered throughout a slow call and not at all before or after; a failing heartbeat is
  raised only once the construction has ended; run_inline uses it exactly for a work that carries it.
* AttachTests: goal_async.attach arms it only on a one-session server whose Goals may call the rewriter and
  only with a heartbeat (compose_engine passes the engine authority's poll); every other server is unchanged.
* ServerTests: test_ms_e2e's one-session server (packaged router, frontend, real engine and worker) with a slow
  rewriter stand-in and a model of the two 90-s watchdogs: without the heartbeat the model kills the server
  during the first rewriter call; with it the server lives through slow calls and through a call that times
  out, whose Goal is the documented fallback; Goals, selections, tokens and evidence equal the unguarded run's.
"""
import asyncio
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_ms2_goal_async as TG  # noqa: E402
import test_ms_e2e as E2E  # noqa: E402
from remote_ra_live_v2 import goal_async, query_selection  # noqa: E402
from remote_ra_live_v2.goal_async import GoalWork, InlineGoals  # noqa: E402

DEADLINE = 0.5          # the model's heartbeat deadline (90 s on a server)
SLOW = 1.0              # one rewriter call (up to 120 s on a server)


class Heart:
    """The engine side of the heartbeat and both watchdogs: ``poll`` answers a PING (the deadline moves);
    a gap longer than the deadline between two answers kills the server."""

    def __init__(self, deadline=DEADLINE):
        self.deadline, self.beats, self.thread = deadline, [], threading.current_thread()

    def poll(self):
        assert threading.current_thread() is self.thread, 'the heartbeat is answered by the engine thread only'
        self.beats.append(time.monotonic())

    def killed(self, start, end):
        """Whether a watchdog fired in [start, end]: a stretch longer than the deadline without an answer."""
        marks = [start] + [b for b in self.beats if start <= b <= end] + [end]
        return max(b - a for a, b in zip(marks, marks[1:])) > self.deadline


def work(construct, context=None):
    return GoalWork('r', NS(construct=construct), context)


class InlineGoalsTests(unittest.TestCase):
    def test_a_slow_construction_keeps_the_heartbeat_answered(self):
        heart, seen = Heart(), {}

        def construct(context):
            seen['thread'] = threading.current_thread()
            time.sleep(SLOW)
            return context * 2
        guard = InlineGoals(heart.poll, interval=0.02)
        start = time.monotonic()
        self.assertEqual(guard.run(work(construct, 21)), 42)
        end = time.monotonic()
        self.assertGreater(end - start, SLOW - 0.1)                 # the engine thread waited for it
        self.assertIsNot(seen['thread'], threading.current_thread())
        self.assertEqual(seen['thread'].name, 'remote-ra-goal-inline')
        self.assertFalse(seen['thread'].is_alive())
        self.assertGreater(len(heart.beats), 5)
        self.assertFalse(heart.killed(start, end))
        # In place, the same call starves the heartbeat for its whole length.
        plain, start = Heart(), time.monotonic()
        self.assertEqual(work(construct, 21).run(), 42)
        self.assertTrue(plain.killed(start, time.monotonic()))

    def test_a_fast_construction_and_its_failures_come_back_unchanged(self):
        heart = Heart()
        guard = InlineGoals(heart.poll, interval=5)
        self.assertEqual(guard.run(work(lambda c: ('goal', c), 7)), ('goal', 7))
        self.assertEqual(heart.beats, [])
        for failure in (ValueError('bad'), TimeoutError('rewrite process timed out'), KeyboardInterrupt()):
            def construct(context, failure=failure):
                raise failure
            with self.subTest(failure=type(failure).__name__), self.assertRaises(type(failure)) as caught:
                InlineGoals(heart.poll, interval=0.01).run(work(construct))
            self.assertIs(caught.exception, failure)
        for bad in (dict(heartbeat=None), dict(heartbeat=heart.poll, interval=0), dict(heartbeat=heart.poll, interval='1')):
            with self.subTest(bad=sorted(bad)), self.assertRaises(Exception):
                InlineGoals(**bad)

    def test_a_failing_heartbeat_is_raised_after_the_construction_ended(self):
        events, calls = [], []

        def construct(context):
            time.sleep(0.3)
            events.append('constructed')
            return 'goal'

        def heartbeat():
            calls.append(time.monotonic())
            raise RuntimeError('native authority revoked')
        with self.assertRaisesRegex(RuntimeError, 'native authority revoked'):
            InlineGoals(heartbeat, interval=0.02).run(work(construct))
            events.append('returned')
        # One thread at a time in the constructor: the failure waited for the construction, and the
        # heartbeat was not tried again.
        self.assertEqual((events, len(calls)), (['constructed'], 1))

    def test_run_inline_uses_it_for_a_work_that_carries_it(self):
        heart, threads = Heart(), []

        def construct(context):
            threads.append(threading.current_thread().name)
            time.sleep(0.15)
            return context + 1

        def steps(inline):
            first = yield GoalWork('r', NS(construct=construct), 1, inline=inline)
            second = yield GoalWork('r', NS(construct=construct), first, inline=inline)
            return second
        guard = InlineGoals(heart.poll, interval=0.02)
        self.assertEqual(goal_async.run_inline(steps(guard)), 3)
        self.assertEqual(threads, ['remote-ra-goal-inline'] * 2)
        self.assertGreater(len(heart.beats), 4)
        del threads[:], heart.beats[:]
        self.assertEqual(goal_async.run_inline(steps(None)), 3)
        self.assertEqual((threads, heart.beats), ([threading.current_thread().name] * 2, []))
        # A failure is thrown into the admission where construct was, as before.
        seen = []

        def failing():
            try:
                yield GoalWork('r', NS(construct=lambda c: (_ for _ in ()).throw(ValueError('bad'))), 1, inline=guard)
            except ValueError as exc:
                seen.append(str(exc))
                raise
        with self.assertRaisesRegex(ValueError, 'bad'):
            goal_async.run_inline(failing())
        self.assertEqual(seen, ['bad'])


class AttachTests(unittest.TestCase):
    def core(self, rewriter):
        selection = None if rewriter is NotImplemented else NS(rewriter=rewriter, inline_goals=None)
        return NS(_remote_ra_private_v2_endpoint=NS(selection=selection)), NS(goals=None)

    def test_only_a_one_session_server_with_a_rewriter_and_a_heartbeat(self):
        beat = lambda: None                                          # noqa: E731
        core, lifecycle = self.core(object())
        self.assertIsNone(goal_async.attach(core, lifecycle, max_num_seqs=1, wake=lambda: None, heartbeat=beat))
        guard = core._remote_ra_private_v2_endpoint.selection.inline_goals
        self.assertIs(type(guard), InlineGoals)
        self.assertIs(guard.heartbeat, beat)
        self.assertEqual(guard.interval, 1.0)
        self.assertIsNone(lifecycle.goals)                           # no park table: still synchronous
        with self.assertRaisesRegex(Exception, 'attached twice'):
            goal_async.attach(core, lifecycle, max_num_seqs=1, wake=lambda: None, heartbeat=beat)
        # Without a heartbeat (no engine authority), without a rewriter, for another policy: nothing.
        for rewriter, kwargs in ((object(), {}), (None, dict(heartbeat=beat)), (NotImplemented, dict(heartbeat=beat))):
            core, lifecycle = self.core(rewriter)
            self.assertIsNone(goal_async.attach(core, lifecycle, max_num_seqs=1, wake=lambda: None, **kwargs))
            selection = core._remote_ra_private_v2_endpoint.selection
            self.assertIsNone(getattr(selection, 'inline_goals', None))
        # Sessions decoding together: the park table, as before, and no inline guard.
        core, lifecycle = self.core(object())
        table = goal_async.attach(core, lifecycle, max_num_seqs=2, wake=lambda: None, heartbeat=beat)
        self.addCleanup(table.close)
        self.assertIs(type(table), goal_async.GoalAdmissions)
        self.assertIsNone(core._remote_ra_private_v2_endpoint.selection.inline_goals)

    def test_compose_engine_passes_the_engine_authoritys_poll(self):
        text = (ROOT / 'source-overlay' / 'remote_ra_live_v2' / 'composition.py').read_text(encoding='utf-8')
        self.assertIn('heartbeat=None if authority is None else authority.poll', text)
        self.assertIn('authority = getattr(core, "_remote_ra_authority", None)', text)

    def test_the_authority_yields_works_that_carry_the_guard(self):
        import test_selection_stageb as SEL
        rewriter = TG.PayloadRewriter(0)
        with patch.object(query_selection, 'goal_rewriter', lambda pin: rewriter):
            auth, transcript = SEL.authority('goal_query', policy='goal_query')
        self.assertIsNone(auth.inline_goals)
        g0 = SEL.geometry(0, 0, 1000, 1000, 1007)
        transcript.pending, transcript.turn = True, 0
        heart = Heart()
        auth.inline_goals = InlineGoals(heart.poll, interval=0.01)
        steps = auth.begin_turn_steps(SEL.binding(0, g0, 'r0'), g0, SEL.conversation(0))
        first = next(steps)
        self.assertIs(first.inline, auth.inline_goals)
        query = goal_async.run_inline(steps, first)
        self.assertEqual((query.goal.gate_decision, query.goal.slm_attempted, rewriter.calls),
                         ('INITIALIZE', True, ['initialization']))
        self.assertEqual(rewriter.threads, ['remote-ra-goal-inline'])


class ServerTests(unittest.IsolatedAsyncioTestCase):
    """The one-session stage-1 server (the Goal built during the synchronous admission)."""
    check_engine = TG.GoalAsyncTests.check_engine
    goal_view = TG.GoalAsyncTests.goal_view
    comparable = staticmethod(TG.GoalAsyncTests.comparable)

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)
        self.rewriting, self.rewriters = dict(delay=SLOW, fail=None), []

        def start(pin):
            rewriter = TG.PayloadRewriter(**self.rewriting)
            self.rewriters.append(rewriter)
            return rewriter
        self.enterContext(patch.object(query_selection, 'goal_rewriter', start))

    def server(self, heart):
        srv = E2E.MultiSessionEndToEndTests.server(self, 'goal_query', [TG.A])
        h = srv.engine
        # compose_engine's attachment on a one-session server (max_num_seqs 1).
        self.assertIsNone(goal_async.attach(h.core, h.lifecycle, max_num_seqs=1, wake=lambda: None,
                                            **({} if heart is None else dict(heartbeat=heart.poll))))
        guard = h.selection.inline_goals
        if guard is not None:
            guard.interval = 0.02
        return srv

    async def run_turns(self, heart, turns):
        srv = self.server(heart)
        seen = TG.record(srv)
        views, spans = [], []
        for i in turns:
            start = time.monotonic()
            info = await TG.play(srv, 0, i)
            spans.append((start, time.monotonic()))
            views.append(self.goal_view(srv, seen, info))
        return srv, views, spans

    async def test_the_server_lives_through_slow_rewriter_calls(self):
        turns = range(3)                         # turn 0 initializes with the rewriter, turn 2 rewrites
        plain_heart = Heart()
        _, plain, plain_spans = await self.run_turns(None, turns)
        plain_rewriter = self.rewriters[-1]
        heart = Heart()
        srv, guarded, spans = await self.run_turns(heart, turns)
        rewriter = self.rewriters[-1]
        self.assertEqual(rewriter.calls, plain_rewriter.calls)
        self.assertGreaterEqual(len(rewriter.calls), 2)
        # Without the heartbeat the engine thread built the Goal and answered nothing: the model's
        # watchdog fires during the first rewriter call.
        self.assertTrue(all(name == threading.current_thread().name for name in plain_rewriter.threads))
        self.assertTrue(plain_heart.killed(*plain_spans[0]))
        # With it the construction ran on its own thread and no stretch of any turn exceeds the deadline.
        self.assertEqual(set(rewriter.threads), {'remote-ra-goal-inline'})
        for start, end in spans:
            self.assertFalse(heart.killed(start, end))
        self.assertGreater(spans[0][1] - spans[0][0], DEADLINE + 0.3)
        # Goals, selections, prompts, outputs and evidence are the synchronous ones.
        self.assertEqual(guarded, plain)
        # Still no park table and no Goal-thread timing evidence: the admission did not park.
        self.assertIsNone(srv.engine.lifecycle.goals)
        natives = [srv.engine.native_file(rid) for rid in srv.engine.lifecycle.entries]
        self.assertFalse(any(key in native for native in natives
                             for key in ('goal_async_ms', 'goal_construct_ms', 'goal_overlap_ms')))

    async def test_a_rewriter_timeout_ends_in_the_fallback_not_in_a_dead_server(self):
        from dram_kv.goals.rewriter import RewriteCommandError
        self.rewriting['fail'] = dict(hypothesis_invalidated=RewriteCommandError('persistent rewrite process timed out'))
        turns = range(3)
        heart = Heart()
        srv, views, spans = await self.run_turns(heart, turns)
        # Turn 2's REWRITE call ran its whole (scaled) timeout and failed: the documented fallback.
        self.assertEqual((views[2]['gate'], views[2]['applied'], views[2]['slm'], views[2]['fallback']),
                         ('REWRITE', 'PATCH', (True, False), 'RewriteCommandError: persistent rewrite process timed out'))
        self.assertGreater(spans[2][1] - spans[2][0], DEADLINE + 0.3)
        for start, end in spans:
            self.assertFalse(heart.killed(start, end))
        # The server goes on: the next turn is admitted and completes.
        info = await TG.play(srv, 0, 3)
        self.assertTrue(info['text'])
        self.assertEqual(srv.engine.lifecycle.sessions[srv.lanes[0].sid], (4, None))


if __name__ == '__main__':
    unittest.main()
