"""goal-rollback-1116: the Goal of a goal_query session goes on across a format-error rollback (CPU only; not GPU
evidence).

Mini-SWE-Agent drops a reply with a format error and sends the previous request again with one more user message
(the format-error message). The engine serves that request as a rebuild at local turn 0 (keep-rebuild-1116: with
thinking on the exact ids of the previous request, a "rollback"; off, the replies re-encoded from text). Before this
branch every rebuild released the session's Goal (QuerySelectionAuthority.reset_session), so the Goal maintained over
many turns started again from the task (INITIALIZE, the format-error message as the observation). Now a rebuild whose
request is the previous request + one user message (``mswea_protocol.Transcript.drops_reply``) of a policy whose
query is the Goal keeps the session's Goal state; the rollback turn reuses the Goal of the turn whose reply was
dropped (a KEEP without the rewriter, ``reused_previous``), the Goal cache stays, and later Goal turns go on from the
kept turn's id. Every other rebuild, and every other policy, is unchanged.

1. The transcript: ``drops_reply`` (thinking on and off) and the pending render's ``drops_reply``.
2. The selection authority over a real XML transcript (thinking on and off; the deterministic Goal and a stand-in
   rewriter): every Goal of a session with dropped replies (one, two in a row, the first reply) equals the Goal of
   the same session in which the dropped replies never existed; the rollback turn reuses the kept Goal; the
   rewriter is called with the same requests; the constructor's state at the end is the same; the rollback row's
   record (``query.goal.carry``, goals.jsonl ``carry``); a non-rollback rebuild, a rollback that does not repeat the
   kept turn, a Goal state changed since, a close and other policies behave as before. Also: goal_query_markers
   (the marker-pages plug-in, its query hook over the real token stream) carries the same way; Gemma 4's mswea-xml
   transcripts (thinking off) carry; --admission-overlap's per-session constructors carry, and a session pool's
   close after a carry releases the Goal on the Goal thread; a resend after a carry starts the Goal again.
3. End to end, thinking on: test_goal_overlap_thinking's batched goal_query server (HTTP, frontend, signed frames,
   the engine loop, Goals parked on the Goal thread, the stand-in rewriter), --goal-overlap off and on: per session
   and logical turn the same Goals as the sessions without the dropped replies.
4. End to end, thinking off: test_goal_overlap_vllm's three goal_query sessions, the Hugging Face child's in-process
   fake (all flags off) and the vLLM child on its fake site with --admission-overlap 3 and --goal-overlap: the same,
   with a CAPACITY_WAIT before rollbacks (the engine untouched; the retry is the same rollback).
"""
import asyncio
import copy
import dataclasses
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_mswea_protocol as X  # noqa: E402  the harness tokenizer
import test_selection_stageb as SEL  # noqa: E402

from remote_ra_live_v2 import goal_text, mswea_protocol as XML, query_selection, selection_contract  # noqa: E402
from remote_ra_live_v2.query_selection import QuerySelectionAuthority, digest  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

P = SEL.P
TASK = SEL.TASK
SYSTEM = 'You are a careful engineer.'
ISSUE = '<pr_description>\nFix the bug\n</pr_description>'
FORMAT_ERROR = ('Format error:\n\n<error>\nExpected exactly 1 action, found %d.\n</error>\n\n'
                'Please always provide EXACTLY ONE action in triple backticks.')
TRACE = 'Traceback (most recent call last):\n  File "%s", line %d, in %s\n%s'
OUT = '<returncode>%d</returncode>\n<output>\n%s\n</output>'
# The logical turns' observations (None: the task): the first turn initializes the Goal, turns 2 and 4 raise a new
# error type (the gate REWRITEs), the others PATCH or KEEP (test_ms2_goal_async's script).
OBSERVATIONS = (None, OUT % (0, 'src/app.py'), OUT % (1, TRACE % ('pkg/mod.py', 3, 'parse', 'ValueError: bad header')),
                OUT % (0, 'all tests pass now, good'), OUT % (1, TRACE % ('pkg/io.py', 9, 'load', 'KeyError: missing')),
                OUT % (0, 'ok ok'), OUT % (0, 'pkg/mod.py pkg/io.py'))
COMMANDS = ('ls src', 'python -m pytest tests/test_app.py', 'cat pkg/mod.py', 'sed -i s/a/b/ pkg/mod.py',
            'cat pkg/io.py', 'python -m pytest', 'git diff')


def action(i):
    return 'THOUGHT: step %d\n<mswea_bash_command>%s</mswea_bash_command>' % (i, COMMANDS[i % len(COMMANDS)])


DROPPED = 'THOUGHT: two at once\n<mswea_bash_command>ls</mswea_bash_command>\n<mswea_bash_command>pwd</mswea_bash_command>'


def thought(i, d=0):
    return 'Step %d (try %d): read the trace, then act.' % (i, d)


class StandInRewriter:
    """A greedy rewriter's stand-in (test_ms2_goal_async.PayloadRewriter without its delay): the upstream
    DeterministicFakeRewriter's candidate with a description tagged by the request payload's digest. ``calls``: the
    payload of every call, in order."""

    def __init__(self):
        query_selection.upstream()
        from dram_kv.goals.rewriter import DeterministicFakeRewriter
        self.inner, self.calls, self.closed = DeterministicFakeRewriter(), [], False

    def rewrite(self, request):
        from dram_kv.goals.rewriter import RewriteResult, validate_rewrite_candidate
        payload = json.dumps(request.to_payload(), sort_keys=True, separators=(',', ':'))
        self.calls.append(payload)
        tag = sha256(payload.encode()).hexdigest()[:12]
        base = self.inner.rewrite(request).candidate
        candidate = validate_rewrite_candidate(
            dataclasses.asdict(dataclasses.replace(base, description='%s [%s]' % (base.description, tag))),
            request.current_goal, trigger=request.trigger, allow_constraint_updates=request.allow_constraint_updates)
        return RewriteResult(candidate=candidate, latency_ms=1.0, input_tokens=len(payload) // 4, output_tokens=12)

    def close(self):
        self.closed = True


def goal_view(goal):
    """Everything the Goal decides (latency aside)."""
    return dict(selector_payload=json.loads(json.dumps(goal.selector_payload)), version_hash=goal.version_hash,
                retrieval_goal_hash=goal.retrieval_goal_hash, gate=goal.gate_decision, applied=goal.applied_decision,
                triggers=goal.triggers, changed=goal.changed_fields, slm=(goal.slm_attempted, goal.slm_succeeded),
                fallback=goal.fallback_reason, reused=goal.reused_previous,
                tokens=(goal.input_tokens, goal.output_tokens))


def state_view(constructor, session='s'):
    """The constructor's state of one session (the upstream _SessionState and its PhaseTracker)."""
    state = constructor._sessions.get(session)
    if state is None:
        return None
    tracker = {k: (dict(v) if isinstance(v, dict) else list(v) if not isinstance(v, (str, int, float, bool,
                                                                                         type(None))) else v)
               for k, v in vars(state.phase_tracker).items() if k != 'config'}
    return dict(retrieval_goal=dataclasses.asdict(state.retrieval_goal), root_goal=state.root_goal,
                last_top_frame=state.last_top_frame, drift_streak=state.drift_streak, tracker=repr(tracker))


class Driver:
    """One goal_query (or other) session on a QuerySelectionAuthority over a real XML transcript, served as the
    engine serves it: the transcript renders the request; a rebuild resets the session (SplitPrivateEndpoint.rebuild:
    reset_session); begin_turn, phase 1 (a Goal forward when asked), plan_turn; the completion publishes the sealed
    Pages and commits the reply. Geometry is synthetic and grows per local turn."""

    def __init__(self, test, policy='goal_query', thinking=True, rewriter=None, transcript=None, per_session=False,
                 real=False):
        """``transcript``: a factory of the engine transcript (default: Qwen's grammar over the harness tokenizer);
        ``per_session``: the constructor as --admission-overlap wraps it (admission_overlap.SessionConstructors);
        ``real``: the geometry of the transcript's own token stream (a query-hook plug-in reads it), else synthetic."""
        self.test, self.real = test, real
        if transcript is None:
            self.t = XML.Transcript(encode=X.TOK.encode, render_initial=X.render_initial, decode=X.TOK.decode,
                                    stop_ids=frozenset({X.EOS, X.PAD}), allow_rebuild=True, thinking=thinking)
        else:
            self.t = transcript()
        self.thinking = self.t.thinking

        def verify(value, mac):
            if digest(value) != mac:
                raise RuntimeError('mac')
        adapter = NS(page_tokens=P, sessions={'s': NS(policy=policy)},
                     authority=NS(source='x' * 64, sign=digest, verify=verify))
        contract = None
        if rewriter is not None:
            directory = test.enterContext(tempfile.TemporaryDirectory())
            contract = SEL.pinned_contract(directory)
        with patch.object(query_selection, 'goal_rewriter', lambda pin: rewriter):
            self.auth = QuerySelectionAuthority(adapter, task=TASK, contract=contract, transcript=self.t)
        if per_session:
            from remote_ra_live_v2.admission_overlap import SessionConstructors
            self.auth.constructor = SessionConstructors(self.auth.constructor)
        self.ordinal, self.C, self.rows = 0, 0, []

    def call(self, messages, text, reasoning=None):
        """One request: returns its row (the query, the receipt body, the goals.jsonl record, the render)."""
        auth, t = self.auth, self.t
        tokens, stable, span = t.render([dict(m) for m in messages])
        pending = dict(t.pending)
        if pending['rebuild']:
            auth.reset_session('s')
            self.C = 0
        L = t.turn
        if self.real:
            geo = SEL.geometry(L, self.C if L else 0, stable, span[1] if L else stable, len(tokens))
        elif L == 0:
            geo = SEL.geometry(0, 0, 1000, 1000, 1007)
        else:
            o = self.C + 6000 + 300 * L
            geo = SEL.geometry(L, self.C, o, o + 100, o + 107)
        b = SEL.binding(self.ordinal, geo, 'r%d' % self.ordinal)
        conversation = [dict(role=m['role'], content=m['content']) for m in messages]
        q = auth.begin_turn(b, geo, conversation)
        p1 = None
        if L:
            forward = q.goal_payload is not None
            vector = torch.zeros(64)
            vector[0] = 1.0
            p1 = SEL.phase1(b, geo, (), vector.tolist() if forward else None, q.text_sha256 if forward else None)
        ranking, count, receipt = auth.plan_turn(b, geo, 1, q, p1)
        generated = t.encode((reasoning + '\n</think>\n\n' if self.thinking else '') + text)
        computed = len(tokens) + len(generated) - 1 if self.real else geo.e + 400 if L else 1100
        sealed = computed // P * P
        done = set(auth.vector_catalog.get('s', {})) | set(p1.page_vectors if p1 else ())
        later = {page: SEL.vec(page) for page in range(sealed // P) if page not in done}
        auth.publish_completion(b, geo, p1.page_vectors if p1 else {}, later, sealed, computed)
        committed = t.commit(tokens, generated)
        record = auth.goal_texts.pending.pop(b.request, None)
        row = dict(query=q, body=receipt['body'], record=record, pending=pending, committed=committed,
                   local_turn=L, turn=self.ordinal)
        self.rows.append(row)
        self.ordinal += 1
        self.C = computed
        return row

    def reply(self, committed):
        """The assistant message Mini-SWE-Agent sends back for a kept reply."""
        if self.thinking:
            return dict(role='assistant', content=committed['content'].strip(), reasoning=committed['reasoning'])
        return dict(role='assistant', content=committed['text'].strip())


def play(test, drops, turns=len(OBSERVATIONS), **kwargs):
    """A Mini-SWE-Agent session over the logical turns: ``drops[i]`` replies of logical turn i are rejected (format
    errors) before one is kept. Returns the driver and, per logical turn, its rows (the dropped calls, then the kept
    one)."""
    d = Driver(test, **kwargs)
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=ISSUE)]
    logical = []
    for i in range(turns):
        if OBSERVATIONS[i] is not None:
            messages = messages + [dict(role='user', content=OBSERVATIONS[i])]
        rows = []
        for k in range(drops.get(i, 0)):
            rows.append(d.call(messages, DROPPED, thought(i, k + 1)))
            messages = messages + [dict(role='user', content=FORMAT_ERROR % (k + 2))]
        rows.append(d.call(messages, action(i), thought(i)))
        messages = messages + [d.reply(rows[-1]['committed'])]
        logical.append(rows)
    d.messages = messages
    return d, logical


# ---- 1. the transcript ----------------------------------------------------------------------------------------

class DropsReplyTests(unittest.TestCase):
    def test_the_pending_render_says_whether_the_previous_reply_was_dropped(self):
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                d, logical = play(self, {1: 2, 3: 1}, turns=5, thinking=thinking, policy='observation_query')
                labels = [(row['pending']['rebuild'], row['pending']['drops_reply']) for rows in logical for row in rows]
                self.assertEqual(labels, [(False, False), (False, False), (True, True), (True, True), (False, False),
                                          (False, False), (True, True), (False, False)])
                if thinking:
                    self.assertEqual([row['pending']['history'] for rows in logical for row in rows],
                                     ['fresh', 'append', 'rollback', 'rollback', 'append', 'append', 'rollback',
                                      'append'])
                # A resend of the same request and an edited history are rebuilds that dropped no reply.
                t = d.t
                last = d.rows[-1]
                messages = [dict(m) for m in t.history]
                t.render(messages)
                self.assertEqual((t.pending['rebuild'], t.pending['drops_reply']), (True, False))
                t.abandon()
                edited = messages[:2] + [dict(role='user', content='something else')]
                t.render(edited)
                self.assertEqual((t.pending['rebuild'], t.pending['drops_reply']), (True, False))
                t.abandon()
                self.assertFalse(t.drops_reply(messages + [dict(role='assistant', content='x')]))
                self.assertTrue(last['committed'])


# ---- 2. the selection authority -------------------------------------------------------------------------------

SCENARIOS = {'one rollback': {3: 1}, 'two in a row': {4: 2}, 'the first reply': {0: 1},
             'the first reply twice and later ones': {0: 2, 2: 1, 5: 1}}


class AuthorityTests(unittest.TestCase):
    maxDiff = None

    def compare(self, drops, thinking, rewriter, **kwargs):
        made = []

        def make():
            made.append(StandInRewriter() if rewriter else None)
            return made[-1]
        ref, reference = play(self, {}, thinking=thinking, rewriter=make(), **kwargs)
        d, logical = play(self, drops, thinking=thinking, rewriter=make(), **kwargs)
        self.assertEqual(len(logical), len(reference))
        for i, (rows, (expected,)) in enumerate(zip(logical, reference)):
            want = goal_view(expected['query'].goal)
            # The first request of a logical turn is the reference's (its reply may be dropped later).
            self.assertEqual(goal_view(rows[0]['query'].goal), want, i)
            self.assertNotIn('carry', rows[0]['body']['goal'])
            self.assertNotIn('carry', rows[0]['record'] or {})
            for k, row in enumerate(rows[1:]):
                # A rollback turn: the kept Goal (its hashes and payload), recorded as a KEEP without the rewriter.
                goal = row['query'].goal
                self.assertEqual((row['local_turn'], row['pending']['drops_reply']), (0, True))
                self.assertEqual({key: goal_view(goal)[key] for key in ('selector_payload', 'version_hash',
                                                                       'retrieval_goal_hash')},
                                 {key: want[key] for key in ('selector_payload', 'version_hash',
                                                             'retrieval_goal_hash')}, (i, k))
                self.assertEqual((goal.gate_decision, goal.applied_decision, goal.triggers, goal.changed_fields,
                                  goal.slm_attempted, goal.reused_previous, goal.fallback_reason),
                                 ('KEEP', 'KEEP', ('rollback',), (), False, True, None))
                self.assertEqual(goal.diagnostics, {'kept_from_turn': expected['query'].goal.turn_id})
                carry = dict(carried=True, reason=None, from_turn=rows[0]['turn'],
                             goal_turn=expected['query'].goal.turn_id)
                self.assertEqual(row['body']['goal']['carry'], carry)
                self.assertEqual(row['body']['goal_hash'], expected['body']['goal_hash'])
                self.assertEqual(row['body']['goal']['payload_sha256'], expected['body']['goal']['payload_sha256'])
                if rewriter:
                    self.assertEqual((row['record']['carry'], row['record']['rewrites']), (carry, []))
                    self.assertEqual(row['record']['decision']['applied_decision'], 'KEEP')
            # The Goal's turn ids are the reference's (the constructor's turns stay increasing).
            self.assertEqual(rows[-1]['query'].goal.turn_id, expected['query'].goal.turn_id)
        # The same rewriter calls, and the same constructor state at the end.
        if rewriter:
            self.assertEqual(made[1].calls, made[0].calls)
            self.assertTrue(made[0].calls)
        self.assertEqual(state_view(d.auth.constructor), state_view(ref.auth.constructor))
        decisions = [r['query'].goal.applied_decision for rows in reference for r in rows]
        return decisions

    def test_goals_after_a_rollback_equal_the_session_without_the_dropped_replies(self):
        seen = set()
        for name, drops in SCENARIOS.items():
            for thinking in (True, False):
                for rewriter in (True, False):
                    with self.subTest(name, thinking=thinking, rewriter=rewriter):
                        seen.update(self.compare(drops, thinking, rewriter))
        # The script exercised the Goal's decisions (initialization, kept, patched and rewritten Goals).
        self.assertTrue({'INITIALIZE', 'REWRITE'} <= seen and seen & {'KEEP', 'PATCH'}, seen)

    def test_before_this_branch_a_rollback_started_the_goal_again(self):
        # Without the carry (the old reset_session), the rollback turn initializes from the format-error message
        # and the later Goals differ from the reference's.
        def old(self, session_id):
            self._drop_tables(session_id)
            self.constructor.release_session(session_id)
        with patch.object(QuerySelectionAuthority, 'reset_session', old):
            d, logical = play(self, {3: 1}, thinking=True, rewriter=StandInRewriter())
        ref, reference = play(self, {}, thinking=True, rewriter=StandInRewriter())
        rolled = logical[3][1]['query'].goal
        self.assertEqual((rolled.gate_decision, rolled.reused_previous), ('INITIALIZE', False))
        self.assertNotIn('carry', logical[3][1]['body']['goal'])
        self.assertNotEqual([goal_view(rows[-1]['query'].goal) for rows in logical[4:]],
                            [goal_view(rows[-1]['query'].goal) for rows in reference[4:]])

    def test_the_goal_cache_and_the_turn_offset(self):
        d, logical = play(self, {}, turns=6, rewriter=StandInRewriter())
        auth = d.auth
        cache = dict(auth.goal_cache['s'])
        self.assertTrue(cache)
        kept = logical[5][0]['query'].goal.turn_id
        self.assertEqual(kept, 6)
        dropped = d.messages[:-1]
        messages = dropped + [dict(role='user', content=FORMAT_ERROR % 2)]
        row = d.call(messages, action(5), thought(5))
        # The rollback kept the Goal cache (the Goal forward is a cold forward of the Goal text: valid in any
        # namespace) and set the Goal turn offset: the selector's context keeps the session-local turn.
        self.assertEqual(auth.goal_cache['s'], cache)
        self.assertEqual((row['local_turn'], row['query'].context.turn_id, row['query'].goal.turn_id), (0, 1, kept))
        self.assertEqual(auth.goal_base['s'], kept - 1)
        messages = messages + [d.reply(row['committed']), dict(role='user', content=OBSERVATIONS[6])]
        last = d.call(messages, action(6), thought(6))
        self.assertEqual((last['query'].context.turn_id, last['query'].goal.turn_id), (2, kept + 1))
        self.assertEqual(auth.constructor._sessions['s'].last_turn, kept + 1)
        self.assertEqual(auth.goal_carry['s']['goal_turn'], kept + 1)
        # A close drops the kept Goal, the offset and the Goal state.
        auth.close_session('s')
        self.assertEqual((auth.goal_carry, auth.goal_base, auth.__dict__.get('goal_resume', {})), ({}, {}, {}))
        self.assertIsNone(state_view(auth.constructor))

    def test_other_rebuilds_start_the_goal_again(self):
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                d, logical = play(self, {}, turns=4, thinking=thinking, rewriter=StandInRewriter())
                auth, t = d.auth, d.t
                self.assertTrue(auth.goal_cache['s'])
                # A resend of the same request: a rebuild that dropped no reply.
                messages = [dict(m) for m in t.history]
                with patch.object(auth.constructor, 'release_session',
                                  wraps=auth.constructor.release_session) as release:
                    row = d.call(messages, action(9), thought(9))
                    release.assert_called_once_with('s')
                self.assertEqual((row['pending']['rebuild'], row['pending']['drops_reply']), (True, False))
                self.assertEqual((row['query'].goal.gate_decision, row['query'].goal.turn_id), ('INITIALIZE', 1))
                self.assertNotIn('carry', row['body']['goal'])
                self.assertNotIn('carry', row['record'])
                self.assertNotIn('s', auth.__dict__.get('goal_base', {}))
                self.assertNotIn('s', auth.goal_cache)
                self.assertEqual(auth.goal_carry['s']['goal_turn'], 1)

    def test_a_rollback_that_cannot_carry_starts_the_goal_again_and_says_why(self):
        def broken(change):
            d = Driver(self, rewriter=StandInRewriter())
            messages = [dict(role='system', content=SYSTEM), dict(role='user', content=ISSUE)]
            for i in range(3):
                if OBSERVATIONS[i] is not None:
                    messages = messages + [dict(role='user', content=OBSERVATIONS[i])]
                row = d.call(messages, action(i), thought(i))
                messages = messages + [d.reply(row['committed'])]
            messages = messages + [dict(role='user', content=OBSERVATIONS[3])]
            d.call(messages, DROPPED, thought(3, 1))
            change(d)
            with patch.object(d.auth.constructor, 'release_session', wraps=d.auth.constructor.release_session) as release:
                row = d.call(messages + [dict(role='user', content=FORMAT_ERROR % 2)], action(3), thought(3))
                release.assert_called_once_with('s')
            return row

        def advanced(d):
            # A Goal built after the kept turn (no served request does this: a failed admission poisons its session).
            context = d.auth.constructor._sessions['s']
            context.last_turn += 1

        def other(d):
            d.auth.goal_carry['s'] = dict(d.auth.goal_carry['s'], messages=1)
        for change, reason in ((lambda d: d.auth.goal_carry.pop('s'), 'no Goal of an earlier turn'),
                               (other, 'the rollback does not repeat the turn of the kept Goal'),
                               (advanced, 'the Goal state changed after the kept turn')):
            with self.subTest(reason):
                row = broken(change)
                goal = row['query'].goal
                self.assertEqual((goal.gate_decision, goal.turn_id, goal.reused_previous), ('INITIALIZE', 1, False))
                carry = dict(carried=False, reason=reason, from_turn=None, goal_turn=None)
                self.assertEqual((row['body']['goal']['carry'], row['record']['carry']), (carry, carry))

    def test_other_policies_are_unchanged(self):
        # No Goal carry, no record, and the Goal started again at the rollback, as before this branch.
        for policy in ('observation_query', 'random_fill', 'recent_fill', 'recent_tools', 'akv_rule'):
            for thinking in (True, False):
                with self.subTest(policy, thinking=thinking):
                    d, logical = play(self, {2: 1}, turns=4, thinking=thinking, policy=policy)
                    rows = [row for rows in logical for row in rows]
                    self.assertTrue(all('carry' not in row['body']['goal'] for row in rows))
                    self.assertEqual(logical[2][1]['query'].goal.gate_decision, 'INITIALIZE')
                    self.assertEqual({k: d.auth.__dict__.get(k) for k in ('goal_carry', 'goal_base')},
                                     dict(goal_carry=None, goal_base=None))
                    self.assertEqual(d.auth.__dict__.get('goal_resume', {}), {})

    def test_the_rollback_turn_never_calls_the_constructor(self):
        rewriter = StandInRewriter()
        d = Driver(self, rewriter=rewriter)
        messages = [dict(role='system', content=SYSTEM), dict(role='user', content=ISSUE)]
        row = d.call(messages, DROPPED, thought(0, 1))
        calls = len(rewriter.calls)
        with patch.object(d.auth.constructor, 'construct', side_effect=AssertionError('constructed')):
            row = d.call(messages + [dict(role='user', content=FORMAT_ERROR % 2)], action(0), thought(0))
        self.assertEqual((len(rewriter.calls), row['query'].goal.reused_previous), (calls, True))
        self.assertEqual(row['body']['goal']['carry'], dict(carried=True, reason=None, from_turn=0, goal_turn=1))

    def test_goal_query_markers_carries_the_goal_too(self):
        # The marker-pages plug-in's goal_query_markers ranks by the Goal (policy_registry query_vector 'goal').
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                self.compare({2: 1, 4: 2}, thinking, False, policy='goal_query_markers', real=True)
        # Its observation-query twin keeps today's rebuild.
        d, logical = play(self, {2: 1}, turns=4, policy='observation_query_markers', real=True)
        self.assertTrue(all('carry' not in row['body']['goal'] for rows in logical for row in rows))
        self.assertEqual(logical[2][1]['query'].goal.gate_decision, 'INITIALIZE')

    def test_gemma_xml_goal_query_carries_the_goal_with_thinking_off(self):
        # Gemma 4's mswea-xml transcripts (the engine's Gemma4SourceRenderer over the character kit, thinking off, as
        # the Gemma SWE runs) share the Transcript class: a format-error retry is a rebuild that dropped the reply.
        import test_gemma_thinking_on as G
        kit = G.char_kit()

        def gemma():
            made = G.renderer_of(kit).transcript()
            made.allow_rebuild = True
            self.assertIs(made.grammar, XML.GEMMA4_XML)
            self.assertFalse(made.thinking)
            return made
        for name, drops in SCENARIOS.items():
            for rewriter in (True, False):
                with self.subTest(name, rewriter=rewriter):
                    self.compare(drops, False, rewriter, transcript=gemma)

    def test_per_session_constructors_carry_and_close_on_the_goal_thread(self):
        # --admission-overlap's per-session constructor copies (the carry reads their merged snapshot), and a session
        # pool's close after a carry: the Goal state is released on the session's Goal thread, every carry table goes.
        from remote_ra_live_v2 import goal_async
        self.compare({3: 1, 5: 2}, True, True, per_session=True)
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                d, logical = play(self, {3: 1}, rewriter=StandInRewriter(), per_session=True)
                auth = d.auth
                self.assertTrue(logical[3][1]['body']['goal']['carry']['carried'])
                self.assertEqual(auth.constructor.session_count(), 1)
                goals = (goal_async.OverlapGoalAdmissions(NS(), wake=lambda: None, goals=True, permits=2) if overlap
                         else goal_async.GoalAdmissions(NS(), wake=lambda: None, goals=True))
                self.assertEqual(auth.close_session('s', goals), 'goal_thread')
                goals.close()
                self.assertEqual((auth.constructor.session_count(), auth.constructor._sessions), (0, {}))
                self.assertEqual({k: auth.__dict__.get(k, {}) for k in ('goal_carry', 'goal_base', 'goal_resume')},
                                 dict(goal_carry={}, goal_base={}, goal_resume={}))
                self.assertNotIn('s', auth.goal_cache)

    def test_a_resend_after_a_carry_starts_the_goal_again(self):
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                d, logical = play(self, {3: 1}, turns=5, thinking=thinking, rewriter=StandInRewriter())
                auth = d.auth
                self.assertTrue(logical[3][1]['body']['goal']['carry']['carried'])
                self.assertTrue(auth.goal_base['s'])
                # The same request again (a resend): a rebuild that dropped no reply.
                row = d.call([dict(m) for m in d.t.history], action(9), thought(9))
                self.assertEqual((row['pending']['rebuild'], row['pending']['drops_reply']), (True, False))
                self.assertEqual((row['query'].goal.gate_decision, row['query'].goal.turn_id), ('INITIALIZE', 1))
                self.assertNotIn('carry', row['body']['goal'])
                self.assertNotIn('s', auth.goal_base)
                self.assertNotIn('s', auth.goal_cache)
                messages = [dict(m) for m in d.t.history] + [d.reply(row['committed']),
                                                            dict(role='user', content=OBSERVATIONS[5])]
                after = d.call(messages, action(10), thought(10))
                self.assertEqual((after['query'].context.turn_id, after['query'].goal.turn_id), (2, 2))


# ---- end to end: what both servers compare -------------------------------------------------------------------

# Per session (lane): logical turn -> how many of its replies are dropped before one is kept. Session 0: one
# rollback; session 1: its first reply dropped twice (two rollbacks in a row on the first turn); session 2: two
# rollbacks in a row later and one more.
E2E_DROPS = {0: {2: 1}, 1: {0: 2}, 2: {1: 1, 3: 2}}


def compare_servers(test, reference, rolled, drops):
    """``reference`` / ``rolled``: per session, per logical turn, the calls' views (the dropped calls, then the
    kept one) of the server without and with dropped replies. Every first call of a logical turn and every call
    after a rollback equals the reference's turn; a rollback call holds the kept Goal as a KEEP and its record."""
    rollbacks = 0
    test.assertEqual(set(rolled), set(reference))
    for k in reference:
        test.assertEqual(len(rolled[k]), len(reference[k]), k)
        for i, (calls, (expected,)) in enumerate(zip(rolled[k], reference[k])):
            test.assertEqual(len(calls), drops.get(k, {}).get(i, 0) + 1, (k, i))
            test.assertEqual(calls[0]['goal'], expected['goal'], (k, i))
            test.assertIsNone(calls[0]['carry'], (k, i))
            for call in calls[1:]:
                rollbacks += 1
                test.assertEqual((call['rebuild'], call['local_turn']), (True, 0), (k, i))
                test.assertEqual(call['carry'], dict(carried=True, reason=None, from_turn=calls[0]['turn'],
                                                     goal_turn=call['carry']['goal_turn']), (k, i))
                test.assertEqual(call['record_carry'], call['carry'], (k, i))
                goal = call['goal']
                test.assertEqual((goal['goal_hash'], goal['payload_sha256'], goal['text']),
                                 (expected['goal']['goal_hash'], expected['goal']['payload_sha256'],
                                  expected['goal']['text']), (k, i))
                test.assertEqual((goal['decision']['gate_decision'], goal['decision']['applied_decision'],
                                  goal['decision']['triggers'], goal['decision']['slm_attempted'], goal['rewrites']),
                                 ('KEEP', 'KEEP', ['rollback'], False, []), (k, i))
    test.assertEqual(rollbacks, sum(sum(d.values()) for d in drops.values()))
    decisions = {c['goal']['decision']['applied_decision'] for calls in reference.values() for t in calls for c in t}
    test.assertTrue({'INITIALIZE', 'REWRITE'} <= decisions, decisions)
    test.assertTrue(any(c['goal']['rewrites'] for calls in reference.values() for t in calls for c in t))


def call_view(body, row, record, *, rebuild, local_turn, turn):
    """One call: its Goal (the signed hash and payload hash, the goals.jsonl texts, decision and rewriter
    exchanges without ids and times) and its rollback record."""
    rewrites = []
    for x in record['rewrites']:
        answer = None if x['response'] is None else json.loads(x['response'])
        if isinstance(answer, dict):
            answer.pop('request_id', None)
            for key in ('latency_ms', 'startup_latency_ms', 'generation_latency_ms'):
                (answer.get('result') or {}).pop(key, None)
                (answer.get('telemetry') or {}).pop(key, None)
        rewrites.append(dict(request=x['request'], answer=answer, error=x['error']))
    goal = dict(goal_hash=body['goal_hash'], payload_sha256=body['goal']['payload_sha256'],
                text=record['goal']['text'], query=record['query']['text'], decision=record['decision'],
                rewrites=rewrites)
    return dict(goal=goal, carry=body['goal'].get('carry'), record_carry=record.get('carry'), rebuild=rebuild,
                local_turn=local_turn, turn=turn, row=row)


# The native file's Goal timing fields (goal_async, goal_overlap): absent on a carried rollback turn.
GOAL_TIMING = ('goal_async_ms', 'goal_construct_ms', 'goal_overlap_ms', 'goal_forward_ms', 'goal_rewrite_wait_ms',
               'goal_rewrite_call_ms')


# ---- 3. end to end, thinking on ------------------------------------------------------------------------------

def _thinking_harness():
    import test_goal_overlap_thinking as TOT
    return TOT


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ThinkingOnEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """test_goal_overlap_thinking's batched goal_query keep-mode server (Goals parked on the Goal thread, the
    stand-in rewriter), its three sessions playing test_ms2_goal_async's Goal script with Mini-SWE-Agent's
    format-error retries, --goal-overlap off and on."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        TOT = _thinking_harness()
        TOT.XmlKeepHarness.setUpClass.__func__(cls)
        cls.TOT = TOT

    def server(self, overlap):
        srv = self.TOT.XmlKeepHarness.server(self, overlap)
        h = srv.engine
        from remote_ra_live_v2 import thinking as TH

        async def schedule(rid, external, prompt):
            # The reply of the call: its reasoning and content as the agent planned them, whatever the prompt's
            # length (the rolled and the reference sessions' prompts differ; their kept replies do not).
            info = srv.admitted[external] = h.infos[rid]
            info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
            target, eos, chunk = srv.replies.pop(external)
            k = h.sessions.index(info['sid'])
            text, content, stop = srv.thoughts.pop(k)
            outputs = X.TOK.encode(text + '\n' + TH.THINK_END + '\n\n') + X.reply(content, stop)
            return await h.generate(rid, outputs, chunk, srv.holds.pop(k, None))
        srv.schedule = schedule
        return srv

    async def call(self, srv, k, i, messages, text, content, stop):
        h = srv.engine
        srv.thoughts[k] = (text, content, stop)
        wire = json.dumps(dict(model='m', messages=messages)).encode()

        async def body():
            return wire
        raw = NS(app=NS(state=srv.state), headers={}, receive=asyncio.Queue().get, body=body)
        request = NS(messages=[{n: v for n, v in m.items() if n != 'reasoning_content'} for m in messages],
                     tools=None, stream=False, n=1, use_beam_search=False, max_tokens=None, temperature=None,
                     return_token_ids=None, stop_token_ids=[])
        srv.plans[id(request)] = (0, stop, 'page')
        try:
            response = await asyncio.wait_for(srv.router.create_chat_completion(request, raw), 30)
        finally:
            srv.plans.pop(id(request), None)
        external = response.content['id']
        srv.results.pop(external)
        info = srv.admitted.pop(external)
        rid = info['rid']
        row = next(r for r in srv.rows() if r['request_id'] == rid)
        (record,) = [r for r in srv.rows(goal_text.FILE) if r['request_id'] == rid]
        body = h.selection.authorized[rid]['body']
        return dict(call_view(body, row, record, rebuild=row['stage_b']['rebuild'],
                              local_turn=row['stage_b']['local_turn'], turn=row['turn']),
                    message=None, history=row['thinking']['history'], rid=rid, native=h.native_file(rid))

    async def agent(self, srv, k, drops):
        import test_ms2_goal_async as GT
        task = srv.lanes[k].task
        messages = [dict(role='system', content=SYSTEM), dict(role='user', content=task['problem_statement'])]
        turns = []
        for i in range(len(GT.GOAL_SCRIPT)):
            observation = GT.turn(k, i)[0]
            if observation:
                messages = messages + [dict(role='user', content=observation)]
            calls = []
            for d in range(drops.get(i, 0)):
                calls.append(await self.call(srv, k, i, messages, thought(i, d + 1), 'THOUGHT: no action yet.',
                                             True))
                messages = messages + [dict(role='user', content=FORMAT_ERROR % 0)]
            content = 'THOUGHT: lane %d step %d\n<mswea_bash_command>%s</mswea_bash_command>' % (
                k, i, COMMANDS[i % len(COMMANDS)])
            calls.append(await self.call(srv, k, i, messages, thought(i), content, False))
            h = srv.engine
            # Mini-SWE-Agent sends the kept reply back with its reasoning (an append).
            committed = h.endpoint.transcripts[srv.lanes[k].sid].turns[-1]
            messages = messages + [dict(role='assistant', content=committed['content'].strip(),
                                        reasoning_content=committed['reasoning'])]
            turns.append(calls)
        return turns

    async def play(self, overlap, drops):
        srv = self.server(overlap)
        played = await asyncio.gather(*(self.agent(srv, k, drops.get(k, {})) for k in range(len(srv.lanes))),
                                      return_exceptions=True)
        failures = [value for value in played if isinstance(value, BaseException)]
        if failures:
            raise ExceptionGroup('sessions failed', [f if isinstance(f, Exception) else RuntimeError(repr(f))
                                                     for f in failures])
        self.assertFalse(srv.rows('failures.jsonl'))
        h = srv.engine
        self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
        return dict(enumerate(played))

    async def test_the_goal_goes_on_across_rollbacks_with_and_without_the_goal_overlap(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                reference = await self.play(overlap, {})
                rolled = await self.play(overlap, E2E_DROPS)
                compare_servers(self, reference, rolled, E2E_DROPS)
                labels = [c['history'] for turns in rolled.values() for calls in turns for c in calls[1:]]
                self.assertEqual(set(labels), {'rollback'})
                for turns in rolled.values():
                    for calls in turns:
                        for call in calls[1:]:
                            # The rollback row: its Goal record says it carried and from which request turn.
                            self.assertEqual(call['row']['query']['goal']['carry'], call['carry'])
                            # A carried turn builds no Goal: it did not park at a Goal and has no Goal timing.
                            self.assertFalse(set(GOAL_TIMING) & set(call['native']), call['rid'])
                        # A turn that built its Goal on this parking server has its timing.
                        first = calls[0]['native']
                        self.assertIn('goal_async_ms', first)
                        self.assertEqual('goal_overlap_ms' in first, overlap)


# ---- 4. end to end, thinking off -----------------------------------------------------------------------------

def thinking_off_plan(drops, turns):
    """One lane's calls: (observation entry of test_decouple_e2e's script without its padding target, the reply's
    logical turn, whether the call rolls back the previous one, whether its reply is kept)."""
    plan = []
    for i, (text, fill, target, eos) in enumerate(turns):
        for d in range(drops.get(i, 0)):
            plan.append(((text, fill, None, False), i, False, False) if d == 0 else
                        (('Format error %d' % d, 0, None, False), i, True, False))
        plan.append(((text, fill, None, eos), i, False, True) if not drops.get(i) else
                    (('Format error %d' % drops[i], 0, None, eos), i, True, True))
    return plan


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows and the fake rewriter model')
class ThinkingOffEndToEndTests(unittest.TestCase):
    """test_goal_overlap_vllm's three goal_query sessions at 16-token Pages (test_decouple_e2e's fakes), the
    rewriter pinned: all flags off with the Hugging Face child's in-process fake, and --admission-overlap 3,
    --goal-overlap and the vLLM rewriter child on its fake site. Mini-SWE-Agent's format-error retries (the previous
    request + one user message) are rebuilds that re-encode the replies; the Goal goes on across them as with
    thinking on."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_goal_overlap_vllm as TGV
        TGV.setUpModule()
        cls.addClassCleanup(TGV.tearDownModule)
        cls.TGV = TGV

    def setUp(self):
        TGV = self.TGV
        self.log = TGV.TV.fresh('rollback-vllm').with_suffix('.jsonl')
        self.rewriters = TGV.Rewriters(self, self.log, TGV.TV.pin())

    harness = None      # set below: test_goal_overlap_vllm's harness

    def serve(self, *, combined, drops):
        from contextlib import ExitStack
        import test_decouple_e2e as DE
        with ExitStack() as stack:
            h = type(self).harness(self, stack, combined=combined)
            original = h.admit
            plans = {}
            for lane in h.lanes:
                plans[lane.index] = thinking_off_plan(drops.get(lane.index, {}), list(lane.script))
                lane.script = [entry for entry, _, _, _ in plans[lane.index]]
                lane.calls = list(plans[lane.index])

            from remote_ra_live_v2 import capacity
            gate, waited = capacity.admission_gate, []

            def admit(lane):
                entry, i, rollback, kept = lane.calls[0]
                saved = lane.messages
                if rollback:
                    lane.messages = lane.messages[:-1]       # the client dropped the previous reply
                text, fill, target, eos = entry
                reply = ('Done %d-%d.' % (lane.index, i) if eos else
                         'THOUGHT: lane %d step %d\n<mswea_bash_command>ls</mswea_bash_command>' % (lane.index, i)
                         if kept else 'THOUGHT: lane %d no action yet' % lane.index)

                def refuse(lifecycle, command):
                    # A rollback's first attempt while other requests run waits for capacity (CAPACITY_WAIT: the
                    # engine unchanged, the frontend's render abandoned); its retry is the same rollback.
                    waited.append(key)
                    raise capacity.CapacityWait('%s: forced before a rollback' % capacity.CAPACITY_WAIT)
                key = (lane.index, len(lane.calls))
                first = rollback and key not in waited and bool(h.running)
                with patch.object(X, 'reply', lambda _text, stop=False: X.TOK.encode(reply) + ([X.EOS] if stop
                                                                                                else [])), \
                        patch.object(capacity, 'admission_gate', refuse if first else gate):
                    out = original(lane)
                if out is None:
                    lane.messages = saved               # CAPACITY_WAIT: the same call again later
                else:
                    lane.calls.pop(0)
                return out
            h.admit = admit
            h.serve()
            self.assertEqual(len(h.done), sum(len(p) for p in plans.values()))
            self.assertGreaterEqual(len(waited), 1 if drops else 0)
            self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
            records = {r['request_id']: r for r in
                       (json.loads(line) for line in (h.output / goal_text.FILE).read_bytes().splitlines())}
            played = {}
            for lane in h.lanes:
                infos = sorted((info for info in h.done if info['lane'] is lane),
                               key=lambda info: int(info['rid'][1:].split('-')[0]))
                turns = []
                for info, (_, i, rollback, kept) in zip(infos, plans[lane.index]):
                    rid = info['rid']
                    body = h.selection.authorized[rid]['body']
                    view = call_view(body, None, records[rid], rebuild=info['binding'].rebuild,
                                     local_turn=info['geo'].local_turn, turn=info['binding'].turn)
                    if len(turns) <= i:
                        turns.append([])
                    turns[i].append(view)
                played[lane.index] = turns
            h.close()
            DE.engine_closed(h.core)
            return played

    def test_the_goal_goes_on_across_format_error_rebuilds(self):
        for combined in (False, True):
            with self.subTest(combined=combined):
                reference = self.serve(combined=combined, drops={})
                rolled = self.serve(combined=combined, drops=E2E_DROPS)
                compare_servers(self, reference, rolled, E2E_DROPS)
        self.assertEqual(len(self.rewriters.children), 2)
        self.assertEqual(len(self.rewriters.engines), 2)


def _borrow():
    import test_goal_overlap_vllm as TGV
    ThinkingOffEndToEndTests.harness = TGV.VllmGoalOverlapEndToEndTests.harness


_borrow()


if __name__ == '__main__':
    unittest.main()
