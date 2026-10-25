"""Vectorized Selector V2 (fast_selection) against the reference path; CPU only.

The reference path is QuerySelectionAuthority's upstream one (Page records, the
upstream index, PrefillPageEmbeddingSelectorV2 or the random-fill selector, and
compact_cosine in Python). Every case here runs the same inputs through an
authority with the fast path and one without (``fast = None``) and requires the
same ranking, count and evidence, compared as JSON text (key order, float repr,
sign of zero) on randomized catalogs of up to 16,384 Pages: exact ties (repeated
and power-of-two-scaled vectors), zero cosines, mandatory prompts covering none,
some or all of the catalog, recent quotas of 0, 1, 64 and more than the catalog,
budgets by active ratio and by active token budget, Page sizes 16 and 64,
multi-turn sequences (the Page column cache across turns, equal vectors in new
objects) and a refused overwrite. Also: the cosine floats against compact_cosine,
the random fill against selection_contract.fill_order and the startup self-check.
"""
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
# Random catalogs per parity test: the first 16 of the same seeded sequence in the default
# suite, all 48 in the slow pass (REMOTE_RA_SLOW_TESTS=1, README section 2.3).
CATALOGS = 48 if os.environ.get('REMOTE_RA_SLOW_TESTS') == '1' else 16
sys.path.insert(0, str(ROOT/'source-overlay'))
from remote_ra_live_v2 import fast_selection, query_selection, selection_contract as sc
from remote_ra_live_v2.query_selection import QuerySelectionAuthority, digest

VECTORS, SCHEMA, _, _ = query_selection.upstream()
NP = fast_selection.available(VECTORS)
TASK = dict(instance_id='t', problem_statement='Fix the bug')
D = sc.PROJECTION_DIMENSION
SPACE = dict(hidden_size=5120, decoder_layers=64)       # pinned projection: no torch needed


class Transcript:
    def __init__(self, initial):
        self.initial_length = initial


def authority(policy, contract, initial, fast):
    def verify(value, mac):
        if digest(value) != mac:
            raise RuntimeError('mac')
    adapter = NS(page_tokens=contract['page_tokens'], sessions={'s': NS(policy=policy)},
                 authority=NS(source='x' * 64, sign=digest, verify=verify))
    auth = QuerySelectionAuthority(adapter, task=TASK, contract=contract, transcript=Transcript(initial))
    if not fast:
        auth.fast = None
    return auth


def contract(P, rng, policy, *, ratio=None, budget=None, quota=None, recent_turns=None):
    kw = dict(vector_format=sc.COMPACT_INT8_V1)
    if policy == 'random_fill':
        kw['random_fill'] = sc.random_fill_pin(rng.randrange(1 << 32))
    if budget is not None:
        kw.update(active_ratio=None, active_token_budget=budget)
    elif ratio is not None:
        kw['active_ratio'] = ratio
    value = sc.default_contract(P, **kw)
    value['switch_threshold_tokens'] = 0
    if quota is not None:
        value['recent_quota_pages'] = quota
    if recent_turns is not None:
        value['recent_turns'] = recent_turns
    return sc.validate(value)


class Pool:
    """Compact vectors with many exact ties: a few distinct directions, each also at
    power-of-two scales (the same unit vector), sparse and extreme ones."""

    def __init__(self, rng, compact_type, distinct):
        self.rng, self.type, self.base = rng, compact_type, []
        for k in range(distinct):
            kind = k % 4
            if kind == 0:
                values = [rng.randint(-127, 127) for _ in range(D)]
            elif kind == 1:
                values = [rng.choice((-1, 0, 0, 1)) for _ in range(D)]
            elif kind == 2:
                values = [rng.choice((-127, 127)) for _ in range(D)]
            else:
                values = [0] * D
                values[rng.randrange(D)] = rng.choice((-127, 1, 127))
            if not any(values):
                values[0] = 1
            self.base.append((tuple(values), rng.uniform(1e-4, 1e-2)))

    def draw(self):
        values, scale = self.rng.choice(self.base)
        if self.rng.random() < 0.5:
            values = tuple(self.rng.randint(-127, 127) for _ in range(D)) if self.rng.random() < 0.3 else values
        if not any(values):
            values = (1,) + values[1:]
        return self.type(values, 'int8', scale * 2.0 ** self.rng.randint(-3, 3))

    def copy(self, vector):
        return self.type(tuple(vector.values), vector.dtype, vector.scale)


def as_json(planned):
    ranking, count, evidence = planned
    return json.dumps([list(ranking), count, evidence])


def plan_both(case, auths, binding, n, entries, query, local_turn, space=SPACE):
    out = []
    for auth in auths:
        goal = SCHEMA.GoalRecord(session_id='s', turn_id=local_turn + 1, constructor='test',
                                 constructor_config={}, text='find the bug')
        context = SCHEMA.TurnContext('s', 'inst', local_turn + 1, 'objective', None, None, 'observation')
        out.append(auth.plan(binding, goal, context, dict(entries), query, n, space, local_turn))
    return out


def scenario(rng, *, P=16, n_final=None, turns=None, policy='observation_query', distinct=None, steps=1,
             budget_mode=None, quota=None, recent_turns=None, initial=None):
    """One random session of ``steps`` selective plans; yields (fast, reference) results per plan."""
    n_final = n_final or rng.randint(2, 3000)
    turns = turns or rng.randint(1, 12)
    budget_mode = budget_mode or rng.choice(('ratio', 'budget'))
    ratio = rng.choice((0.05, 0.25, 0.4, 0.75, 0.99)) if budget_mode == 'ratio' else None
    budget = P * rng.randint(1, max(1, n_final)) if budget_mode == 'budget' else None
    quota = rng.choice((0, 1, 2, 64, 10 ** 6)) if quota is None else quota
    recent_turns = rng.choice((0, 1, 2, 3, 5)) if recent_turns is None else recent_turns
    value = contract(P, rng, policy, ratio=ratio, budget=budget, quota=quota, recent_turns=recent_turns)
    # initial prompt: within the first Page, part of the catalog, or all of it
    initial = initial or rng.choice((1, rng.randint(1, max(1, n_final * P // 3)), n_final * P + rng.randint(0, P)))
    fast = authority(policy, value, initial, True)
    ref = authority(policy, value, initial, False)
    assert fast.fast is not None
    pool = Pool(rng, fast.compact_type, distinct or rng.choice((1, 2, 5, 40)))
    catalog = {}
    # turn ends before the selective plans: strictly increasing, the last below the first stable end
    first_n = max(1, n_final - rng.randint(0, n_final // 2)) if steps > 1 else n_final
    first_o = first_n * P + rng.randint(0, P - 1)
    ends = sorted(rng.sample(range(1, first_o), min(turns, first_o - 1)))
    previous = None
    if rng.random() < 0.7:
        prev_n = rng.randint(0, first_n)
        previous = (frozenset(rng.sample(range(prev_n), rng.randint(0, prev_n))), prev_n)
    for auth in (fast, ref):
        auth.turn_ends['s'] = list(ends)
        if previous is not None:
            auth.previous_active['s'] = previous
    local_turn = len(ends)
    o = first_o
    for step in range(steps):
        n = o // P
        for page in range(n):
            if page not in catalog:
                catalog[page] = pool.draw()
        # the newest Pages arrive as fresh objects (phase 1), equal in value
        fresh = {p: pool.copy(catalog[p]) for p in range(max(0, n - rng.randint(0, 20)), n)}
        entries = {p: fresh.get(p, catalog[p]) for p in range(n)}
        query = rng.choice((pool.draw(), catalog[rng.randrange(n)]))
        binding = NS(session='s', turn=local_turn + 3, model='m', renderer='r', stable_end=o)
        got, want = plan_both(None, (fast, ref), binding, n, entries, query, local_turn)
        yield got, want, dict(n=n, P=P, turn=local_turn, policy=policy, quota=quota, recent_turns=recent_turns,
                              ratio=ratio, budget=budget, initial=initial)
        ranking, count, _ = want
        for auth in (fast, ref):
            auth.turn_ends['s'].append(o)
            auth.previous_active['s'] = (frozenset(ranking[:count]), n)
        local_turn += 1
        o += rng.randint(1, 600) * 1 + P * rng.randint(0, 40)


@unittest.skipIf(NP is None, 'numpy and a passing fast_selection self-check required')
class PlanParityTests(unittest.TestCase):
    maxDiff = None

    def check(self, results):
        count = 0
        for got, want, info in results:
            self.assertEqual(as_json(got), as_json(want), info)
            count += 1
        return count

    def test_random_catalogs_observation_and_goal_query(self):
        rng = random.Random(1)
        for k in range(CATALOGS):
            policy = ('observation_query', 'goal_query')[k % 2]
            self.check(scenario(rng, P=(16, 16, 64)[k % 3], policy=policy, steps=rng.choice((1, 3))))

    def test_random_catalogs_random_fill(self):
        rng = random.Random(2)
        for k in range(CATALOGS):
            self.check(scenario(rng, P=(16, 64)[k % 2], policy='random_fill', steps=rng.choice((1, 3))))

    def test_edges_of_the_partition(self):
        rng = random.Random(3)
        for policy in ('observation_query', 'random_fill'):
            # every Page mandatory; none mandatory but the first; quota 0; quota above the catalog;
            # recent_turns 0 (every Page recent-eligible); budget of one Page; budget of every Page
            for kw in (dict(initial=10 ** 6), dict(initial=1), dict(quota=0), dict(quota=10 ** 6),
                       dict(recent_turns=0), dict(budget_mode='budget', n_final=500), dict(turns=1)):
                self.check(scenario(rng, policy=policy, steps=2, **kw))

    def test_ties_everywhere(self):
        rng = random.Random(4)
        for policy in ('observation_query', 'goal_query'):
            for distinct in (1, 2, 3):
                self.check(scenario(rng, policy=policy, distinct=distinct, n_final=2000, steps=2))

    def test_sixteen_thousand_pages(self):
        rng = random.Random(5)
        for policy, budget in (('observation_query', 'ratio'), ('goal_query', 'budget'), ('random_fill', 'ratio')):
            self.assertEqual(self.check(scenario(rng, policy=policy, n_final=16384, turns=40, distinct=50,
                                                 budget_mode=budget, quota=64, recent_turns=2, steps=2)), 2)

    def test_float_query_is_projected_alike(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('CPU torch required for the projection')
        rng = random.Random(6)
        value = contract(16, rng, 'observation_query', ratio=0.4)
        fast, ref = (authority('observation_query', value, 100, f) for f in (True, False))
        pool = Pool(rng, fast.compact_type, 7)
        entries = {p: pool.draw() for p in range(900)}
        for auth in (fast, ref):
            auth.turn_ends['s'] = [500, 7000]
        query = tuple(rng.gauss(0, 1) for _ in range(64))
        binding = NS(session='s', turn=2, model='m', renderer='r', stable_end=14400)
        got, want = plan_both(None, (fast, ref), binding, 900, entries, query, 2, space=dict(decoder_layers=64))
        self.assertEqual(as_json(got), as_json(want))

    def test_a_changed_sealed_vector_is_refused_like_the_index_does(self):
        rng = random.Random(7)
        value = contract(16, rng, 'observation_query', ratio=0.4)
        for fast in (True, False):
            auth = authority('observation_query', value, 100, fast)
            pool = Pool(rng, auth.compact_type, 9)
            entries = {p: pool.draw() for p in range(600)}
            auth.turn_ends['s'] = [300]
            binding = NS(session='s', turn=1, model='m', renderer='r', stable_end=9600)
            plan_both(None, (auth,), binding, 600, entries, pool.draw(), 1)
            entries[17] = auth.compact_type(tuple(-v for v in entries[17].values), 'int8', entries[17].scale)
            auth.turn_ends['s'].append(9600)
            binding = NS(session='s', turn=2, model='m', renderer='r', stable_end=9700)
            with self.assertRaisesRegex(ValueError, 'cannot be overwritten'):
                plan_both(None, (auth,), binding, 600, entries, pool.draw(), 2)

    def test_previous_set_past_the_catalog_takes_the_reference_path(self):
        rng = random.Random(8)
        value = contract(16, rng, 'observation_query', ratio=0.4)
        fast, ref = (authority('observation_query', value, 100, f) for f in (True, False))
        pool = Pool(rng, fast.compact_type, 9)
        entries = {p: pool.draw() for p in range(700)}
        for auth in (fast, ref):
            auth.turn_ends['s'] = [300, 5000]
            auth.previous_active['s'] = (frozenset({3, 699, 900}), 650)
        binding = NS(session='s', turn=2, model='m', renderer='r', stable_end=11200)
        with patch.object(fast, 'plan_reference', wraps=fast.plan_reference) as reference:
            got, want = plan_both(None, (fast, ref), binding, 700, entries, pool.draw(), 2)
        self.assertEqual(reference.call_count, 1)
        self.assertEqual(as_json(got), as_json(want))

    def test_rebuild_drops_the_columns(self):
        rng = random.Random(9)
        value = contract(16, rng, 'observation_query', ratio=0.4)
        auth = authority('observation_query', value, 100, True)
        pool = Pool(rng, auth.compact_type, 9)
        entries = {p: pool.draw() for p in range(600)}
        auth.turn_ends['s'] = [300]
        binding = NS(session='s', turn=1, model='m', renderer='r', stable_end=9600)
        plan_both(None, (auth,), binding, 600, entries, pool.draw(), 1)
        self.assertEqual(len(auth.page_columns['s'].objects), 600)
        auth.reset_session('s')
        self.assertNotIn('s', auth.page_columns)


@unittest.skipIf(NP is None, 'numpy and a passing fast_selection self-check required')
class PieceTests(unittest.TestCase):
    def test_cosines_are_compact_cosine_floats(self):
        rng = random.Random(10)
        compact = query_selection.memo_types(VECTORS)[0]
        pool = Pool(rng, compact, 30)
        vectors = [pool.draw() for _ in range(3000)]
        unit = VECTORS.CompactVector.unit_values
        columns = NP.array([unit(v) for v in vectors]).T.copy()
        for query in [pool.draw() for _ in range(6)] + vectors[:3]:
            got = fast_selection.cosines(NP, columns, unit(query)).tolist()
            want = [VECTORS.compact_cosine(query, v) for v in vectors]
            self.assertEqual([repr(x) for x in got], [repr(x) for x in want])
        with self.assertRaisesRegex(ValueError, 'dimensionality'):
            fast_selection.cosines(NP, columns, (1.0,) * (D - 1))

    def test_fill_order_is_the_reference_permutation(self):
        rng = random.Random(11)
        for m in [0, 1, 2, 3, 4, 5, 16, 17, 100, 1023, 1024, 1025, 4097, 16384]:
            for _ in range(3):
                seed, session, turn = rng.randrange(1 << 32), rng.randrange(1 << 16), rng.randrange(256)
                eligible = sorted(rng.sample(range(3 * m + 5), m))
                self.assertEqual(fast_selection.fill_order(NP, eligible, seed, session, turn, sc.fill_order),
                                 sc.fill_order(eligible, seed, session, turn), (m, seed, session, turn))
        with self.assertRaisesRegex(Exception, 'out of range'):
            fast_selection.fill_order(NP, [1, 2, 3], 1 << 32, 0, 0, sc.fill_order)

    def test_splitmix64_outputs_and_rejection_bound(self):
        state, outputs = 12345, []
        for _ in range(50):
            state, z = sc.splitmix64(state)
            outputs.append(z)
        self.assertEqual(fast_selection.splitmix64_outputs(NP, 12345, 50).tolist(), outputs)
        rng = random.Random(12)
        bounds = [2, 3, 5, 7, 1 << 20, (1 << 20) + 1, (1 << 63) + 1, (1 << 64) - 1, 3 << 61]
        z = [rng.randrange(1 << 64) for _ in bounds] + [(1 << 64) - 1] * len(bounds)
        b = bounds * 2
        got = fast_selection.rejected(NP, NP.array(z, dtype=NP.uint64), NP.array(b, dtype=NP.uint64)).tolist()
        self.assertEqual(got, [zz >= (1 << 64) - (1 << 64) % bb for zz, bb in zip(z, b)])
        self.assertTrue(any(got))

    def test_a_rejected_draw_falls_back_to_the_reference(self):
        calls = []

        def reference(eligible, seed, session, turn):
            calls.append(1)
            return sc.fill_order(eligible, seed, session, turn)
        with patch.object(fast_selection, 'rejected', lambda np, z, bound: NP.ones(len(z), dtype=bool)):
            self.assertEqual(fast_selection.fill_order(NP, list(range(50)), 1, 2, 3, reference),
                             sc.fill_order(list(range(50)), 1, 2, 3))
        self.assertEqual(calls, [1])

    def test_startup_self_check_gates_the_fast_path(self):
        self.assertIs(fast_selection.available(VECTORS), NP)
        with patch.dict(fast_selection._STATE, clear=True), \
                patch.object(fast_selection, '_check_vectors', return_value=False):
            self.assertIsNone(fast_selection.available(VECTORS))
            auth = authority('observation_query', contract(16, random.Random(0), 'observation_query'), 1, True)
            self.assertIsNone(auth.fast)
        with patch.dict(fast_selection._STATE, clear=True), patch.object(fast_selection, 'numpy', return_value=None):
            self.assertIsNone(fast_selection.available(VECTORS))



if __name__ == '__main__':
    unittest.main()
