"""goal_query Goal text recording (remote_ra_live_v2.goal_text, goals.jsonl). CPU only.

- RewriterTap on the upstream PersistentCommandGoalRewriter (its child replaced
  by a scripted answer per request line): every exchange of a construction is
  recorded with the exact request payload, the child's raw answer line, the
  error of a fallback (a candidate the parent refuses, an error from the child,
  a timeout), its measured times and the child's token counts; calls outside a
  capture, on another thread or with no tap record nothing; the GoalRecords are
  those of the untapped rewriter and the Goal control hash is unchanged.
- record: the stored texts hash to the runtime's goal_hash (GoalRecord version
  hash) and goal.payload_sha256; the startup self-check refuses another encoding.
- End to end (test_stageb_e2e's script through test_host_trim.play, with the
  scripted rewriter): one goals.jsonl row per terminal goal_query request, named
  by SHA-256 in its native file and runtime row; texts, decisions and rewriter
  exchanges agree with the runtime row; with the recording switched off every
  RPC, receipt, output and evidence file is the same apart from the reference;
  full and observation_query write neither goals.jsonl nor a reference.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import dataclasses
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

from remote_ra_http import evidence
from remote_ra_live_v2 import goal_async, goal_text, query_selection, selection_contract

try:
    import torch
except ImportError:
    torch = None

query_selection.upstream()      # the packaged dram_kv namespace
from dram_kv import schema      # noqa: E402
from dram_kv.goals import maintained, rewriter as R     # noqa: E402

ROOT_GOAL = 'Fix the parser bug in pkg/mod.py'
ERROR = ('<returncode>1</returncode>\n<output>\nTraceback (most recent call last):\n'
         '  File "pkg/mod.py", line 3, in parse\nValueError: bad header\n</output>')


def line(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def good(envelope):
    """A valid child answer: a candidate from the request's own features."""
    request = envelope['request']
    candidate = dict(description='Retrieve evidence for %s' % request['trigger'], phase_hint='hypothesis',
                     files=request['features']['files'][:2], evidence_needed=['diagnostics'])
    return line(dict(request_id=envelope['request_id'], result=dict(
        candidate=candidate, input_tokens=120, output_tokens=24, generation_latency_ms=1.5, model_cost=0.0)))


def duplicate(envelope):
    """A candidate the parent refuses (the most frequent fallback on the GPU)."""
    return line(dict(request_id=envelope['request_id'], result=dict(
        candidate=dict(description='Look at parse', symbols=['parse', 'parse']), input_tokens=90, output_tokens=30)))


def child_error(envelope):
    """The child could not parse its own generation (truncated JSON)."""
    return line(dict(request_id=envelope['request_id'],
                     error="GenerationFailed: JSONDecodeError: Unterminated string starting at: line 1 column 300",
                     telemetry=dict(input_tokens=1500, output_tokens=384, generation_latency_ms=9000.0)))


def timeout(envelope):
    return R.RewriteCommandError('persistent rewrite process timed out')


# goal_text.TELEMETRY of each answer: (input_tokens, output_tokens, generation_latency_ms).
COUNTS = {good: (120, 24, 1.5), duplicate: (90, 30, None), child_error: (1500, 384, 9000.0),
          timeout: (None, None, None)}


class ScriptedProcess:
    def __init__(self):
        self.stdin, self.stdout = io.StringIO(), None

    def poll(self):
        return None


class ScriptedRewriter(R.PersistentCommandGoalRewriter):
    """The upstream persistent rewriter whose child answers each request line with
    ``answers[k](envelope)`` (cycling): a line, or an exception _readline raises."""

    def __init__(self, *answers):
        super().__init__(('scripted-child',), timeout_seconds=5.0, startup_timeout_seconds=5.0)
        self.answers, self.envelopes = answers or (good,), []

    def _start(self):
        if self._process is None:
            self._process = ScriptedProcess()

    def _readline(self, timeout_seconds):
        envelope = json.loads(self._process.stdin.getvalue().splitlines()[-1])
        self.envelopes.append(envelope)
        answer = self.answers[(len(self.envelopes) - 1) % len(self.answers)](envelope)
        if isinstance(answer, BaseException):
            raise answer
        return answer + '\n'

    def close(self):
        self._process = None


def context(turn, observation, action=None, rc=None):
    return schema.TurnContext('s', 't', turn, ROOT_GOAL, None, action, observation,
                              message_source='user' if turn == 1 else 'tool',
                              structured_state={'tool_name': 'bash', 'return_code': rc})


CONTEXTS = (context(1, 'task: ' + ROOT_GOAL), context(2, ERROR, 'reply 0', 1),
            context(3, '<returncode>0</returncode>\n<output>\nok\n</output>', 'reply 1', 0))


def constructor(rewriter):
    return maintained.MaintainedGoalConstructor(rewriter, entity_drift_steps=3, initialize_with_rewriter=True)


def plain(goal):
    """A GoalRecord without its wall-clock fields."""
    value = dataclasses.asdict(goal)
    value['latency_ms'] = None
    value['diagnostics'] = {k: v for k, v in value['diagnostics'].items() if 'latency' not in k}
    return value


class TapTests(unittest.TestCase):
    def build(self, *answers, tapped=True):
        rewriter = ScriptedRewriter(*answers)
        c = constructor(rewriter)
        tap = goal_text.tap(c, selection_contract.constructor_control_hash) if tapped else None
        goals, rewrites = [], []
        for ctx in CONTEXTS:
            work = goal_async.GoalWork('r%d' % ctx.turn_id, c, ctx, tap=tap)
            goals.append(work.run())
            rewrites.append(work.rewrites)
        return rewriter, c, tap, goals, rewrites

    def test_exchanges_are_recorded_and_the_goals_are_unchanged(self):
        scripts = [(good, good), (good, duplicate), (duplicate, child_error), (child_error, timeout), (timeout, good)]
        for answers in scripts:
            with self.subTest(answers=[a.__name__ for a in answers]):
                rewriter, c, tap, goals, rewrites = self.build(*answers)
                _, c0, none, goals0, rewrites0 = self.build(*answers, tapped=False)
                self.assertIsNone(none)
                self.assertEqual([plain(g) for g in goals], [plain(g) for g in goals0])
                self.assertEqual(rewrites0, [()] * len(CONTEXTS))
                self.assertIs(type(rewriter), ScriptedRewriter)
                self.assertEqual(selection_contract.constructor_control_hash(c),
                                 selection_contract.constructor_control_hash(c0))
                # Initialization and the REWRITE of the new error call the rewriter; the
                # third turn is a PATCH.
                self.assertEqual([g.slm_attempted for g in goals], [True, True, False])
                self.assertEqual([len(r) for r in rewrites], [1, 1, 0])
                for goal, (exchange,), envelope, answer in zip(goals, rewrites, rewriter.envelopes, answers):
                    self.assertEqual(set(exchange), {'request', 'response', 'error', *goal_text.TIMING,
                                                     *goal_text.TELEMETRY})
                    # Measured on the upstream rewriter's own hooks (_start under its lock,
                    # _readline for the answer); nobody else held the lock.
                    self.assertTrue(all(type(exchange[k]) is float and exchange[k] >= 0 for k in goal_text.TIMING))
                    self.assertLess(exchange['wait_ms'], 1000)
                    # The child's own counts, copied from its answer line.
                    self.assertEqual(tuple(exchange[k] for k in goal_text.TELEMETRY), COUNTS[answer])
                    # The request is the payload line the child received, byte for byte.
                    self.assertEqual(exchange['request'], line(envelope['request']))
                    self.assertEqual(json.loads(exchange['request'])['root_goal'], ROOT_GOAL)
                    sent = answer(envelope)
                    if isinstance(sent, BaseException):
                        self.assertIsNone(exchange['response'])
                    else:
                        self.assertEqual(exchange['response'], sent)
                    self.assertEqual(goal.slm_succeeded, answer is good)
                    self.assertEqual(exchange['error'], goal.fallback_reason)
                    if answer is duplicate:
                        self.assertEqual(goal.fallback_reason,
                                         'RewriteValidationError: retrieval_goal.symbols must be unique')
                        self.assertIn('"symbols":["parse","parse"]', exchange['response'])
                    if answer is child_error:
                        self.assertTrue(exchange['error'].startswith('RewriteCommandError: persistent rewrite failed: '
                                                                     'GenerationFailed: JSONDecodeError'))
                        self.assertEqual(json.loads(exchange['response'])['telemetry']['output_tokens'], 384)
                self.assertEqual([json.loads(r[0]['request'])['trigger'] for r in rewrites[:2]],
                                 ['initialization', 'hypothesis_invalidated'])

    def test_nothing_is_recorded_outside_a_capture_or_on_another_thread(self):
        rewriter = ScriptedRewriter(good)
        c = constructor(rewriter)
        tap = goal_text.tap(c, selection_contract.constructor_control_hash)
        c.construct(CONTEXTS[0])                      # no capture: passes through
        self.assertEqual(len(rewriter.envelopes), 1)
        with tap.capture() as mine, ThreadPoolExecutor(1) as other:
            done = other.submit(c.construct, CONTEXTS[1]).result()      # another thread
        self.assertTrue(done.slm_attempted and done.slm_succeeded)
        self.assertEqual((mine, len(rewriter.envelopes)), ([], 2))
        # A parked Goal (GoalAdmissions._build on the Goal thread) records its own.
        work = goal_async.GoalWork('r3', c, CONTEXTS[2], tap=tap)
        with ThreadPoolExecutor(1, thread_name_prefix='remote-ra-goal') as thread:
            thread.submit(goal_async.GoalAdmissions._build, work).result()
        self.assertEqual(work.rewrites, ())          # a PATCH turn
        c.release_session('s')
        work = goal_async.GoalWork('r4', c, CONTEXTS[0], tap=tap)
        with ThreadPoolExecutor(1, thread_name_prefix='remote-ra-goal') as thread:
            goal = thread.submit(goal_async.GoalAdmissions._build, work).result()
        self.assertTrue(goal.slm_succeeded)
        self.assertEqual(len(work.rewrites), 1)
        self.assertEqual(work.rewrites[0]['request'], line(rewriter.envelopes[-1]['request']))

    def test_concurrent_constructions_keep_their_own_exchanges(self):
        barrier = threading.Barrier(2)

        class Inner:
            """A line-less rewriter: both calls are inside rewrite at once."""
            def rewrite(self, request):
                barrier.wait(10)
                return R.RewriteResult(candidate=dataclasses.replace(request.current_goal,
                                       description='for ' + request.latest_observation[:12]),
                                       raw_response={'seen': request.latest_observation[:12]})

            def close(self):
                pass
        inner = Inner()
        tap = goal_text.RewriterTap(inner)
        copies = [constructor(inner), constructor(inner)]

        def build(k):
            ctx = schema.TurnContext('s%d' % k, 't', 1, ROOT_GOAL, None, None, 'obs number %d' % k,
                                     message_source='tool', structured_state={'tool_name': 'bash'})
            work = goal_async.GoalWork('r%d' % k, copies[k], ctx, tap=tap)
            return work.run(), work.rewrites
        with ThreadPoolExecutor(2) as pool:
            (g0, r0), (g1, r1) = pool.map(build, (0, 1))
        self.assertEqual([len(r0), len(r1)], [1, 1])
        self.assertEqual(json.loads(r0[0]['request'])['latest_observation'], 'obs number 0')
        self.assertEqual(json.loads(r1[0]['request'])['latest_observation'], 'obs number 1')
        # A rewriter without lines: the canonical JSON of the returned raw_response.
        self.assertEqual((r0[0]['response'], r1[0]['response']),
                         ('{"seen":"obs number 0"}', '{"seen":"obs number 1"}'))
        # No lock hook: no wait or start measured; the call is the whole rewrite().
        for exchange in r0 + r1:
            self.assertEqual((exchange['wait_ms'], exchange['start_ms']), (None, None))
            self.assertGreaterEqual(exchange['call_ms'], 0)
            self.assertEqual([exchange[k] for k in goal_text.TELEMETRY], [None] * 3)
        self.assertEqual((g0.text, g1.text), ('for obs number 0', 'for obs number 1'))

    def test_no_rewriter_no_tap(self):
        self.assertIsNone(goal_text.tap(maintained.MaintainedGoalConstructor(None),
                                        selection_contract.constructor_control_hash))


class RecordTests(unittest.TestCase):
    def test_texts_hash_to_the_runtime_hashes(self):
        c = maintained.MaintainedGoalConstructor(None)
        goals = [c.construct(ctx) for ctx in CONTEXTS]
        for variant in ('full_goal', 'candidate_entity_cap8'):
            for goal in goals:
                text = selection_contract.goal_query_text(goal, variant)
                value = goal_text.record(goal, variant, text, goal.version_hash,
                                         sha256(text.encode()).hexdigest(), ())
                self.assertEqual(sha256(value['goal']['text'].encode('utf-8')).hexdigest(), goal.version_hash)
                self.assertEqual(value['goal']['sha256'], goal.version_hash)
                self.assertEqual(sha256(value['query']['text'].encode('utf-8')).hexdigest(), value['query']['sha256'])
                selector = json.loads(value['goal']['text'])
                self.assertEqual(selector['root_goal'], ROOT_GOAL)
                self.assertEqual(selector['retrieval_goal'], json.loads(json.dumps(
                    dataclasses.asdict(goal.retrieval_goal))))
                self.assertEqual(value['decision'], dict(
                    gate_decision=goal.gate_decision, applied_decision=goal.applied_decision,
                    triggers=list(goal.triggers), changed_fields=list(goal.changed_fields),
                    slm_attempted=False, slm_succeeded=False, fallback_reason=goal.fallback_reason))
                self.assertEqual((value['query']['variant'], value['rewrites']), (variant, []))

    def test_startup_self_check(self):
        self.assertTrue(goal_text.self_check(sha256, maintained.MaintainedGoalConstructor, schema))
        unsorted = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        with patch.object(goal_text, 'canonical_text', unsorted):
            with self.assertRaisesRegex(Exception, 'version hash'):
                goal_text.self_check(sha256, maintained.MaintainedGoalConstructor, schema)

    def test_the_authority_taps_only_the_goal_query_rewriter_at_startup(self):
        import test_selection_stageb as SEL
        rewriter = ScriptedRewriter(good)
        with patch.object(query_selection, 'goal_rewriter', lambda pin: rewriter), \
                patch.object(goal_text, 'self_check', wraps=goal_text.self_check) as checked:
            auth, _ = SEL.authority('goal_query', policy='goal_query')
            self.assertIs(auth.rewriter, rewriter)
            self.assertIsNotNone(auth.tap)
            self.assertIn('_readline', vars(rewriter))
            self.assertEqual(auth.goal_control_hash, selection_contract.constructor_control_hash(
                constructor(ScriptedRewriter())))
            checked.assert_called_once()
            for policy in ('observation_query', 'full'):
                auth, _ = SEL.authority(policy, policy=policy)
                self.assertIsNone(auth.tap)
            checked.assert_called_once()


# ---- end to end -------------------------------------------------------------------------------

def rows(raw):
    return [json.loads(x) for x in raw.splitlines()]


def without_reference(files):
    """test_host_trim.without_goal_text: the files without goals.jsonl and the references."""
    import test_host_trim as HT
    return HT.without_goal_text(files)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    maxDiff = None
    ANSWERS = (good, duplicate, child_error, timeout)

    def play(self, policy, recording=True):
        import test_host_trim as HT
        made = []

        def start(pin):
            made.append(ScriptedRewriter(*self.ANSWERS))
            return made[-1]
        with ExitStack() as stack:
            if policy == 'goal_query':
                stack.enter_context(patch.object(query_selection, 'goal_rewriter', start))
            if not recording:
                stack.enter_context(patch.object(goal_text, 'tap', lambda constructor, control: None))
                stack.enter_context(patch.object(evidence, 'goal_text', lambda selection, output, value: None))
            played = HT.play(policy)
        played['rewriters'] = made
        return played

    def test_one_row_per_terminal_goal_query_request(self):
        played = self.play('goal_query')
        files, (rewriter,) = played['files'], played['rewriters']
        goals = rows(files[goal_text.FILE])
        lines = files[goal_text.FILE].splitlines()
        by_sha = {sha256(raw).hexdigest(): row for raw, row in zip(lines, goals)}
        natives = {json.loads(raw)['request_id']: json.loads(raw) for name, raw in files.items()
                   if name.startswith('native-')}
        runtime = rows(files['runtime.jsonl'])
        # Every native terminal (completed and cancelled) names its own row, once.
        self.assertEqual(len(goals), len(natives))
        self.assertEqual(sorted(g['request_id'] for g in goals), sorted(natives))
        for rid, native in natives.items():
            self.assertEqual(native['goal_text']['file'], goal_text.FILE)
            row = by_sha[native['goal_text']['sha256']]
            self.assertEqual({k: row[k] for k in goal_text.KEYS}, {k: native[k] for k in goal_text.KEYS})
            self.assertEqual(row['schema'], goal_text.SCHEMA)
        self.assertEqual([g['terminal_state'] for g in goals], ['complete'] * len(runtime) + ['cancelled'])
        self.assertEqual(len(runtime), len(played['outputs']) - 1)
        for r in runtime:
            q = r['query']
            self.assertEqual(q['goal_text'], natives[r['request_id']]['goal_text'])
            g = by_sha[q['goal_text']['sha256']]
            self.assertEqual((g['request_id'], g['session_id'], g['turn']), (r['request_id'], r['session_id'], r['turn']))
            # The stored texts hash to the runtime row's hashes.
            self.assertEqual(sha256(g['goal']['text'].encode()).hexdigest(), q['goal_hash'])
            self.assertEqual(g['goal']['sha256'], q['goal_hash'])
            self.assertEqual(sha256(g['query']['text'].encode()).hexdigest(), q['goal']['payload_sha256'])
            self.assertEqual(g['query']['variant'], q['goal']['variant'])
            self.assertEqual({k: g['decision'][k] for k in ('gate_decision', 'applied_decision', 'slm_attempted',
                                                            'slm_succeeded', 'fallback_reason')},
                             {k: q['goal'][k] for k in ('gate_decision', 'applied_decision', 'slm_attempted',
                                                        'slm_succeeded', 'fallback_reason')})
            self.assertTrue(g['query']['text'].startswith('<retrieval_query>\n'))
        # One exchange per rewriter call, in call order, with its answer or failure.
        exchanges = [x for g in goals for x in g['rewrites']]
        self.assertEqual([len(g['rewrites']) for g in goals], [int(g['decision']['slm_attempted']) for g in goals])
        self.assertEqual(len(exchanges), len(rewriter.envelopes))
        self.assertGreaterEqual(len(exchanges), 2)      # the two initializations (start, rebuild)
        for k, (exchange, envelope) in enumerate(zip(exchanges, rewriter.envelopes)):
            answer = self.ANSWERS[k % len(self.ANSWERS)]
            sent = answer(envelope)
            self.assertEqual(exchange['request'], line(envelope['request']))
            self.assertEqual(exchange['response'], None if isinstance(sent, BaseException) else sent)
            self.assertEqual(exchange['error'] is None, answer is good)
            self.assertTrue(all(type(exchange[name]) is float for name in goal_text.TIMING))
            self.assertEqual(tuple(exchange[name] for name in goal_text.TELEMETRY), COUNTS[answer])
        # One session: its Goals are built in place, not parked, so no native totals.
        self.assertTrue(all('goal_rewrite_call_ms' not in n and 'goal_rewrite_wait_ms' not in n
                            for n in natives.values()))
        for g in goals:
            for exchange in g['rewrites']:
                self.assertEqual(exchange['error'], g['decision']['fallback_reason'])
        self.assertIn('RewriteValidationError: retrieval_goal.symbols must be unique',
                      [g['decision']['fallback_reason'] for g in goals])

    def test_recording_changes_nothing_else(self):
        on, off = self.play('goal_query'), self.play('goal_query', recording=False)
        for key in ('outputs', 'identities', 'trace', 'prepared', 'receipts', 'allocations', 'views', 'forwards'):
            self.assertEqual(on[key], off[key], key)
        self.assertEqual([r.envelopes for r in on['rewriters']], [r.envelopes for r in off['rewriters']])
        self.assertNotIn(goal_text.FILE, off['files'])
        self.assertIn(goal_text.FILE, on['files'])
        self.assertEqual(without_reference(on['files']), off['files'])

    def test_other_policies_write_no_goal_text(self):
        for policy in ('full', 'observation_query'):
            with self.subTest(policy=policy):
                files = self.play(policy)['files']
                self.assertNotIn(goal_text.FILE, files)
                self.assertEqual(without_reference(files), files)
                for name, raw in files.items():
                    self.assertNotIn(b'goal_text', raw, name)


if __name__ == '__main__':
    unittest.main()
