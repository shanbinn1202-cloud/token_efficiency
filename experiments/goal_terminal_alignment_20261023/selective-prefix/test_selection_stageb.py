"""Stage B selection API: begin_turn / plan_turn / publish_completion / reset_session; CPU only."""
from contextlib import nullcontext
import dataclasses
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
import torch
from remote_ra_live_v2 import query_selection, selection_contract
from remote_ra_live_v2.query_selection import QuerySelectionAuthority, digest

P = 784
TASK = dict(instance_id='t', problem_statement='Fix the bug')


class Transcript:
    """Only what the authority reads from the engine XML transcript."""
    def __init__(self):
        self.turn, self.pending, self.initial_length, self.replies = 0, None, 1000, []

    def previous_action(self):
        return self.replies[-1] if self.replies else None


def authority(session_policy, contract=None, **kwargs):
    session = NS(policy=session_policy)
    def verify(value, mac):
        if digest(value) != mac: raise RuntimeError('mac')
    adapter = NS(page_tokens=P, sessions={'s': session}, authority=NS(source='x'*64, sign=digest, verify=verify))
    transcript = Transcript()
    return QuerySelectionAuthority(adapter, task=TASK, contract=contract, transcript=transcript,
                                   **kwargs), transcript


def conversation(turn):
    messages = [dict(role='system', content='sys'),
                dict(role='user', content='<pr_description>\nFix the bug\n</pr_description>')]
    for k in range(turn):
        messages += [dict(role='assistant', content=f'reply {k}'),
                     dict(role='user', content=f'<returncode>{k % 2}</returncode>\n<output>\nout {k}\n</output>')]
    return messages


def geometry(local_turn, C, o, e, N):
    # As TurnGeometry: a local turn 0 has no candidates (n = L = 0, spec 0).
    Q, n = C // P * P, o // P if local_turn else 0
    L = n * P
    return NS(local_turn=local_turn, C=C, Q=Q, o=o, e=e, N=N, n=n, L=L, tail=C > Q,
              tail_candidate=C > Q and L == Q + P, tail_live=C > Q and L == Q,
              phase2_start=e if local_turn else 0, page_tokens=P)


def binding(turn, geo, request):
    return NS(request=request, session='s', turn=turn, generation='g', epoch='e', model='m', renderer='r',
              identity='id-' + request, stable_end=geo.o, observation_span=(geo.o, geo.e) if geo.local_turn else None)


def vec(seed, hot=False):
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(64); base[0] = 1.0
    return ((base if hot else torch.eye(64)[1 + seed % 60]) + torch.randn(64, generator=g) * .05).tolist()


def phase1(b, geo, relevant, goal_vector=None, goal_sha=None):
    closed = {x // P: vec(x // P, x // P in relevant) for x in range(geo.C, geo.e) if (x + 1) % P == 0}
    query = torch.zeros(64); query[0] = 1.0
    return NS(binding=b.identity, lease_id='lease', lease_digest='d', start=geo.C, end=geo.e,
              page_vectors=closed, observation_vector=query.tolist(), goal_vector=goal_vector,
              goal_payload_sha256=goal_sha, goal_token_count=None if goal_vector is None else 5,
              state_digest='s', tail_capture_digest=None,
              vector_space=dict(decoder_layers=64))


def run_turn(auth, transcript, ordinal, geo, relevant=(), goal_vector=True, messages=conversation):
    b = binding(ordinal, geo, f'r{ordinal}')
    transcript.pending, transcript.turn = True, geo.local_turn
    q = auth.begin_turn(b, geo, messages(geo.local_turn))
    p1 = None
    if geo.local_turn:
        forward = q.goal_payload is not None
        query = torch.zeros(64); query[0] = 1.0
        p1 = phase1(b, geo, relevant, query.tolist() if forward and goal_vector else None,
                    q.text_sha256 if forward else None)
    ranking, count, receipt = auth.plan_turn(b, geo, 1, q, p1)
    return b, q, p1, ranking, count, receipt


def complete(auth, transcript, b, geo, p1, computed_end, relevant=()):
    sealed = computed_end // P * P
    done = set(auth.vector_catalog.get('s', {})) | set(p1.page_vectors if p1 else ())
    later = {page: vec(page, page in relevant) for page in range(sealed // P) if page not in done}
    auth.publish_completion(b, geo, p1.page_vectors if p1 else {}, later, sealed, computed_end)
    transcript.pending = None; transcript.replies.append(f'reply {geo.local_turn}'); transcript.turn += 1


class StageBSelectionTests(unittest.TestCase):
    def test_goal_query_turns_threshold_selection_and_goal_cache(self):
        auth, transcript = authority('goal_query')
        relevant = {3, 12, 15, 17, 19, 21}
        g0 = geometry(0, 0, 1000, 1000, 1007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 0, g0)
        self.assertEqual((q.goal_payload, count, receipt['body']['selection']['reason']), (None, 0, 'cold_start'))
        self.assertEqual(receipt['body']['candidate_pages'], 0)
        complete(auth, transcript, b, g0, None, 1100, relevant)
        # Turn 1: below the threshold, the Goal is built but no Goal forward.
        g1 = geometry(1, 1100, 5000, 5100, 5107)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 1, g1, relevant)
        self.assertEqual((q.goal_payload, count, receipt['body']['selection']['reason']),
                         (None, 6, 'below_switch_threshold'))
        self.assertEqual(receipt['body']['local_turn'], 1)
        complete(auth, transcript, b, g1, p1, 9000, relevant)
        # Turn 2: selection active, a Goal forward is required on a cache miss.
        g2 = geometry(2, 9000, 20000, 20300, 20307)
        fixed = []
        original = auth.constructor.construct
        def construct(context):
            if not fixed: fixed.append(original(context))
            return fixed[0]
        with patch.object(auth.constructor, 'construct', side_effect=construct):
            b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 2, g2, relevant)
            selection = receipt['body']['selection']
            self.assertIsNotNone(q.goal_payload)
            self.assertEqual((selection['reason'], selection['capacity_pages'], count), (None, 10, 10))
            self.assertEqual(selection['mandatory_pages'], [0, 1])
            self.assertEqual(selection['recent_pages'], [23, 24])
            self.assertEqual(set(selection['retrieved_pages']), relevant)
            self.assertEqual(set(ranking[:count]), {0, 1, 23, 24} | relevant)
            self.assertEqual(sorted(ranking), list(range(25)))
            self.assertEqual(selection['projection_sha256'],
                             selection_contract.projection(64, auth.contract)[1])
            self.assertEqual(receipt['body']['goal']['forward'], True)
            complete(auth, transcript, b, g2, p1, 21000, relevant)
            # Turn 3: same retrieval Goal -> cache hit, no Goal forward, same vector.
            g3 = geometry(3, 21000, 24000, 24100, 24107)
            b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 3, g3, relevant)
            self.assertIsNone(q.goal_payload)
            self.assertEqual((receipt['body']['goal']['cache_hit'], receipt['body']['goal']['forward']), (True, False))
            self.assertTrue(relevant <= set(ranking[:count]))
        auth.verify_reservation(b, ranking, count, receipt)
        with self.assertRaisesRegex(Exception, 'replay'):
            auth.verify_reservation(b, ranking, count, receipt)

    def test_xml_context_carries_previous_reply_and_returncode(self):
        # Upstream _context: previous decoded reply, <returncode> of the observation.
        auth, transcript = authority('goal_query')
        g0 = geometry(0, 0, 1000, 1000, 1007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 0, g0)
        self.assertEqual((q.context.previous_action, q.context.structured_state),
                         (None, {'tool_name': 'bash', 'return_code': None}))
        complete(auth, transcript, b, g0, None, 9500)
        g1 = geometry(1, 9500, 20000, 20100, 20107)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 1, g1)
        self.assertEqual((q.context.turn_id, q.context.previous_action, q.context.structured_state),
                         (2, 'reply 0', {'tool_name': 'bash', 'return_code': 0}))
        self.assertEqual(q.context.message_source, 'tool')
        head, body, tail = q.goal_payload.split('\n')
        self.assertEqual((head, tail), ('<retrieval_query>', '</retrieval_query>'))
        self.assertEqual(receipt['body']['goal']['payload_sha256'], q.text_sha256)

    def test_turn_boundaries_must_grow_once_per_local_turn(self):
        auth, transcript = authority('goal_query')
        g1 = geometry(1, 1100, 5000, 5100, 5107)
        transcript.pending, transcript.turn = True, 1
        with self.assertRaisesRegex(Exception, 'boundaries'):
            auth.begin_turn(binding(1, g1, 'r1'), g1, conversation(1))

    def test_observation_query_uses_phase1_row_and_phase1_pages_are_candidates(self):
        auth, transcript = authority('observation_query')
        g0 = geometry(0, 0, 1000, 1000, 1007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 0, g0)
        complete(auth, transcript, b, g0, None, 9500)
        # C=9500 -> Q=9408, tail [9408, 9500); o=10193 completes page 12 in phase 1 (tail candidate).
        g1 = geometry(1, 9500, 10193, 12000, 12007)
        self.assertTrue(g1.tail_candidate)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 1, g1, {12})
        self.assertIsNone(q.goal_payload)
        self.assertIn(12, p1.page_vectors)                  # closed by the last token/boundary
        self.assertEqual(sorted(ranking), list(range(13)))  # candidates [0, n=13)
        self.assertEqual(receipt['body']['candidate_pages'], 13)
        # Pages closed by the observation (>= n) are not candidates but are published at completion.
        self.assertTrue(any(page >= 13 for page in p1.page_vectors))
        complete(auth, transcript, b, g1, p1, 12100)
        self.assertEqual(set(auth.vector_catalog['s']), set(range(12100 // P)))

    def test_phase1_must_publish_exactly_its_closed_pages(self):
        auth, transcript = authority('observation_query')
        g0 = geometry(0, 0, 1000, 1000, 1007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 0, g0)
        complete(auth, transcript, b, g0, None, 9500)
        g1 = geometry(1, 9500, 10193, 12000, 12007)
        b = binding(1, g1, 'r1'); transcript.pending, transcript.turn = True, 1
        q = auth.begin_turn(b, g1, conversation(1))
        bad = phase1(b, g1, ())
        bad.page_vectors = dict(bad.page_vectors); bad.page_vectors.pop(12)
        with self.assertRaisesRegex(Exception, 'exactly the pages'):
            auth.plan_turn(b, g1, 1, q, bad)

    def test_compact_cache_projects_only_new_pages_and_matches_a_fresh_plan(self):
        auth, transcript = authority('observation_query')
        g0 = geometry(0, 0, 1000, 1000, 1007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 0, g0)
        complete(auth, transcript, b, g0, None, 9500)
        g1 = geometry(1, 9500, 10193, 12000, 12007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 1, g1, {3})
        complete(auth, transcript, b, g1, p1, 12100)
        g2 = geometry(2, 12100, 14000, 15000, 15007)
        original, compared = auth.plan, []
        def plan(*args):
            first = original(*args)
            saved, auth.compact_cache = auth.compact_cache, {}   # the same plan without the cache
            again = original(*args)
            auth.compact_cache = saved
            compared.append((first[0], first[1], first[2]['scores']) == (again[0], again[1], again[2]['scores']))
            return first
        with patch.object(selection_contract, 'compact', wraps=selection_contract.compact) as compact, \
                patch.object(auth, 'plan', side_effect=plan):
            b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 2, g2, {3})
        self.assertEqual(compared, [True])
        n = receipt['body']['candidate_pages']
        # Cached pass: only the pages closed since the last plan, plus the query; uncached pass: n + 1.
        cached_calls = compact.call_count - (n + 1)
        self.assertLess(cached_calls, n + 1)
        self.assertEqual(set(auth.compact_cache['s']), set(range(n)))

    def test_catalog_version_detects_a_publication_between_plan_and_reserve(self):
        auth, transcript = authority('observation_query')
        g0 = geometry(0, 0, 1000, 1000, 1007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 0, g0)
        complete(auth, transcript, b, g0, None, 9500)
        g1 = geometry(1, 9500, 10193, 12000, 12007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 1, g1)
        self.assertEqual(receipt['body']['catalog_version'], auth.catalog_token(b))
        auth.catalog_version['s'] += 1      # another publication after this plan
        with self.assertRaisesRegex(Exception, 'stale catalog'):
            auth.verify_reservation(b, ranking, count, receipt)

    def test_rebuild_resets_session_state(self):
        auth, transcript = authority('goal_query')
        g0 = geometry(0, 0, 1000, 1000, 1007)
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 0, g0)
        complete(auth, transcript, b, g0, None, 2000)
        with patch.object(auth.constructor, 'release_session', wraps=auth.constructor.release_session) as release:
            auth.reset_session('s')
            release.assert_called_once_with('s')
        self.assertNotIn('s', auth.vector_catalog); self.assertNotIn('s', auth.turn_ends)
        self.assertNotIn('s', auth.compact_cache); self.assertNotIn('s', auth.catalog_version)
        transcript.turn, transcript.replies = 0, []
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, 1, geometry(0, 0, 13000, 13000, 13007))
        self.assertEqual((receipt['body']['local_turn'], receipt['body']['turn'],
                          receipt['body']['selection']['reason']), (0, 1, 'cold_start'))
        self.assertEqual(q.context.turn_id, 1)


# ---- observation_query and the Goal rewriter ---------------------------------
# Selector V2 ranks observation_query Pages by the observation vector alone, so
# its Goal is deterministic and the rewriter is never started or called.

UNRELATED = 'Port the YAML parser to Rust and drop Python 2 support'
SELECTION_KEYS = ('reason', 'capacity_pages', 'selected_pages', 'mandatory_pages', 'recent_pages',
                  'retrieved_pages')
GOAL_KEYS = {'variant', 'payload_sha256', 'gate_decision', 'applied_decision', 'slm_attempted',
             'slm_succeeded', 'fallback_reason', 'cache_hit', 'forward'}
# Cold start, below the threshold, then two turns of active selection.
TURNS = ((geometry(0, 0, 1000, 1000, 1007), 1100), (geometry(1, 1100, 5000, 5100, 5107), 9000),
         (geometry(2, 9000, 20000, 20300, 20307), 21000), (geometry(3, 21000, 24000, 24100, 24107), 24500))
# A new error type on local turn 2 gates a Goal REWRITE (hypothesis_invalidated).
ERROR = ('<returncode>1</returncode>\n<output>\nTraceback (most recent call last):\n'
         '  File "pkg/mod.py", line 3, in parse\nValueError: bad header\n</output>')


def error_conversation(turn):
    messages = conversation(turn)
    if turn == 2:
        messages[-1] = dict(role='user', content=ERROR)
    return messages


def unrelated(goal):
    """The same Goal record with unrelated text, entities and retrieval description."""
    retrieval = goal.retrieval_goal
    if retrieval is not None:
        retrieval = dataclasses.replace(retrieval, description=UNRELATED, files=('zzz/unrelated.rs',),
                                        symbols=('nothing_here',), tests=('tests/test_unrelated.py',),
                                        errors=('ImaginaryError',))
    return dataclasses.replace(goal, text=UNRELATED if retrieval is not None else 'An unrelated goal: ' + UNRELATED,
                               files=('zzz/unrelated.rs',), symbols=('nothing_here',), errors=('ImaginaryError',),
                               retrieval_goal=retrieval, slm_attempted=True, slm_succeeded=True)


def run_turns(auth, transcript, relevant=(), messages=conversation):
    rows = []
    for ordinal, (geo, end) in enumerate(TURNS):
        b, q, p1, ranking, count, receipt = run_turn(auth, transcript, ordinal, geo, relevant, messages=messages)
        rows.append(dict(ranking=tuple(ranking), count=count, query=q, body=receipt['body']))
        complete(auth, transcript, b, geo, p1, end, relevant)
    return rows


def pinned_contract(directory):
    model = Path(directory)/'qwen3-0.6b'; model.mkdir(); (model/'config.json').write_bytes(b'{}')
    return selection_contract.default_contract(P, goal_rewriter=selection_contract.rewriter_pin(model))


class FakeRewriter:
    """The upstream network-free rewriter plus the close() of the persistent one."""
    def __init__(self):
        query_selection.upstream()      # the packaged dram_kv namespace, as goal_rewriter() needs it
        from dram_kv.goals.rewriter import DeterministicFakeRewriter
        self.inner, self.closed = DeterministicFakeRewriter(), False

    def rewrite(self, request):
        return self.inner.rewrite(request)

    def close(self):
        self.closed = True


class ObservationQueryGoalTests(unittest.TestCase):
    def test_selection_is_identical_with_an_unrelated_goal(self):
        relevant = {3, 12, 15, 17, 19, 21}
        def run(swap):
            auth, transcript = authority('observation_query')
            original = auth.constructor.construct
            with (patch.object(auth.constructor, 'construct', side_effect=lambda c: unrelated(original(c)))
                  if swap else nullcontext()):
                return run_turns(auth, transcript, relevant)
        base, swapped = run(False), run(True)
        self.assertEqual([r['body']['selection']['reason'] for r in base],
                         ['cold_start', 'below_switch_threshold', None, None])
        for turn, (a, b) in enumerate(zip(base, swapped)):
            self.assertNotEqual(a['query'].goal.text, b['query'].goal.text, turn)   # the Goal really changed
            self.assertNotEqual(a['body']['goal_hash'], b['body']['goal_hash'], turn)
            self.assertEqual((a['ranking'], a['count'], {k: a['body']['selection'][k] for k in SELECTION_KEYS}),
                             (b['ranking'], b['count'], {k: b['body']['selection'][k] for k in SELECTION_KEYS}), turn)
        self.assertTrue(relevant <= set(base[2]['ranking'][:base[2]['count']]))

    def test_observation_query_never_starts_the_rewriter_and_refuses_a_pinned_one(self):
        with TemporaryDirectory() as directory:
            pinned = pinned_contract(directory)
        with patch.object(query_selection, 'goal_rewriter') as start:
            # Fail closed: an observation_query contract that pins a rewriter is refused.
            with self.assertRaisesRegex(Exception, 'observation_query does not use the Goal rewriter'):
                authority('observation_query', pinned)
            with self.assertRaisesRegex(Exception, 'observation_query does not use the Goal rewriter'):
                authority('observation_query', pinned, policy='observation_query')
            for contract in (None, dict(pinned, goal_rewriter=None)):
                auth, transcript = authority('observation_query', contract)
                self.assertEqual((auth.policy, auth.rewriter, auth.constructor.rewriter,
                                  auth.constructor.initialize_with_rewriter), ('observation_query', None, None, False))
            run_turns(auth, transcript, messages=error_conversation)
            auth.close()
        start.assert_not_called()

    def test_goal_query_still_starts_and_uses_its_rewriter(self):
        with TemporaryDirectory() as directory:
            pinned = pinned_contract(directory)
        fake = FakeRewriter()
        with patch.object(query_selection, 'goal_rewriter', return_value=fake) as start:
            auth, transcript = authority('goal_query', pinned, policy='goal_query')
        start.assert_called_once_with(pinned['goal_rewriter'])
        self.assertEqual((auth.policy, auth.rewriter, auth.constructor.rewriter, auth.constructor.initialize_with_rewriter),
                         ('goal_query', fake, fake, True))
        rows = run_turns(auth, transcript, messages=error_conversation)
        # Initialization on turn 0 and the REWRITE gated by the new error on turn 2.
        self.assertEqual([request.trigger for request in fake.inner.calls], ['initialization', 'hypothesis_invalidated'])
        self.assertEqual([(r['body']['goal']['slm_attempted'], r['body']['goal']['slm_succeeded']) for r in rows],
                         [(True, True), (False, False), (True, True), (False, False)])
        auth.close()
        self.assertTrue(fake.closed)

    def test_goal_evidence_stays_well_formed_without_the_rewriter(self):
        auth, transcript = authority('observation_query')
        rows = run_turns(auth, transcript, messages=error_conversation)
        goals = [r['body']['goal'] for r in rows]
        for goal, row in zip(goals, rows):
            self.assertEqual(set(goal), GOAL_KEYS)
            self.assertEqual((goal['variant'], goal['slm_attempted'], goal['slm_succeeded'], goal['cache_hit'],
                              goal['forward']), ('full_goal', False, False, False, False))
            self.assertEqual(goal['payload_sha256'], row['query'].text_sha256)
            self.assertEqual(len(row['body']['goal_hash']), 64)
            self.assertIsNone(row['query'].goal_payload)
        self.assertEqual([(g['gate_decision'], g['applied_decision'], g['fallback_reason']) for g in goals],
                         [('INITIALIZE', 'INITIALIZE', None), ('KEEP', 'KEEP', None),
                          ('REWRITE', 'PATCH', 'rewriter_disabled'), ('PATCH', 'PATCH', None)])

    def test_the_authority_policy_is_the_server_policy(self):
        with self.assertRaisesRegex(Exception, 'differs from the server policy'):
            authority('goal_query', policy='observation_query')
        with self.assertRaisesRegex(Exception, 'differs from the server policy'):
            authority('observation_query', policy='nonsense')
        auth, transcript = authority('observation_query', policy='observation_query')
        auth.adapter.sessions['s'].policy = 'goal_query'
        g0 = geometry(0, 0, 1000, 1000, 1007)
        transcript.pending, transcript.turn = True, 0
        with self.assertRaisesRegex(Exception, 'session policy differs'):
            auth.begin_turn(binding(0, g0, 'r0'), g0, conversation(0))
        # full/random50 servers hold an inert authority that never starts a rewriter.
        with patch.object(query_selection, 'goal_rewriter') as start:
            self.assertIsNone(authority('full')[0].rewriter)
        start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
