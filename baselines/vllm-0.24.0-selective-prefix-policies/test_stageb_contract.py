"""Stage B shared contract (M0): RenderAssignment v2, TurnGeometry, phase-1 lease/result,
compose_rows golden layouts for every spec 3.10 row. Pure CPU; no vLLM import."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))

from remote_ra_live_v2.contract import (LiveV2Error, NativeIdentity, Phase1Result, RenderAssignment,
                                        ZERO_DIGEST, closed_pages, digest, parse_lease)
from remote_ra_live_v2.chunk_protocol import worker_interval
from remote_ra_live_v2.sparse_mapping import RowLayout, SparsePlan, compose_rows
from remote_ra_native_growing.contract import (Abort, Authority, RenderBinding, TurnGeometry,
                                               select_pages, turn_geometry)

P = 784
ATT = (0,)                                  # 1 FullAttention group, 3 Mamba groups
H = lambda c: c * 64                        # hex64 test digests
FULL = tuple(range(7, 7 + 8000))            # the session's full token stream


def bid(group, kind, page=0):
    """Readable, pool-unique ids: group 0 resident Page p is 100+p, the tail 200,
    fresh FA 300+p, fresh Mamba 1000g+400, restore target 500+p, transient 601..."""
    return 1000 * group + dict(R=100, T=200, F=300, M=400, S=500, N=600, GF=700, GM=800)[kind] + page


# Every spec 3.10 row (plus a later-turn restore). R = committed resident Pages,
# sel = A_t. Goldens are explicit ids so layout arithmetic regressions show.
CASES = [
    dict(name='local_turn_zero', local=0, turn=0, C=0, R=(), o=900, e=900, N=1000, sel=(),
         n=0, L=0, flags=(False, False, False), run=(), fresh=(), closed=(),
         p1=None, plan=((), (), ()), p2=((601, 602), (0, 0))),
    dict(name='c_equals_q', local=1, turn=1, C=1568, R=(0, 1), o=1571, e=2000, N=2010, sel=(1,),
         n=2, L=1568, flags=(False, False, False), run=(2,), fresh=(2,), closed=(),
         p1=((100, 101, 302), (1568, 0)), plan=((1,), (), (0,)), p2=((101, 302, 601), (1568, 784))),
    dict(name='tail_live', local=1, turn=1, C=1668, R=(0, 1), o=1671, e=2400, N=2410, sel=(0, 1),
         n=2, L=1568, flags=(True, False, True), run=(2, 3), fresh=(3,), closed=(2,),
         p1=((100, 101, 200, 303), (1568, 0)), plan=((0, 1), (), ()),
         p2=((100, 101, 200, 303, 601), (1568, 0))),
    dict(name='tail_live_short_observation', local=1, turn=1, C=1668, R=(0, 1), o=1671, e=1700,
         N=1710, sel=(1,), n=2, L=1568, flags=(True, False, True), run=(2,), fresh=(), closed=(),
         p1=((100, 101, 200), (1568, 0)), plan=((1,), (), (0,)), p2=((101, 200, 601), (1568, 784))),
    dict(name='tail_candidate_selected', local=1, turn=2, C=2350, R=(0, 1), o=2353, e=3000, N=3010,
         sel=(0, 2), n=3, L=2352, flags=(True, True, False), run=(3,), fresh=(3,), closed=(2,),
         p1=((100, 101, 200, 303), (1568, 0)), plan=((0, 2), (), (1,)),
         p2=((100, 200, 303, 601), (2352, 784))),          # tail block = last prefix entry
    dict(name='tail_candidate_unselected', local=1, turn=2, C=2350, R=(0, 1), o=2353, e=3000,
         N=3010, sel=(0, 1), n=3, L=2352, flags=(True, True, False), run=(3,), fresh=(3,),
         closed=(2,), p1=((100, 101, 200, 303), (1568, 0)), plan=((0, 1), (), (2,)),
         p2=((100, 101, 303, 601), (2352, 784))),          # tail block not in the row
    dict(name='below_threshold', local=1, turn=1, C=1568, R=(0, 1), o=1571, e=2000, N=2010,
         sel=(0, 1), n=2, L=1568, flags=(False, False, False), run=(2,), fresh=(2,), closed=(),
         p1=((100, 101, 302), (1568, 0)), plan=((0, 1), (), ()),
         p2=((100, 101, 302, 601), (1568, 0))),            # physical == logical
    dict(name='first_selective_turn', local=1, turn=3, C=2350, R=(0, 1), o=2353, e=3000, N=3010,
         sel=(1, 2), n=3, L=2352, flags=(True, True, False), run=(3,), fresh=(3,), closed=(2,),
         p1=((100, 101, 200, 303), (1568, 0)), plan=((1, 2), (), (0,)),   # R' = [0, n)
         p2=((101, 200, 303, 601), (2352, 784))),
    dict(name='later_turn_restore', local=2, turn=4, C=1568, R=(1,), o=1571, e=2000, N=2010,
         sel=(0, 1), n=2, L=1568, flags=(False, False, False), run=(2,), fresh=(2,), closed=(),
         p1=((101, 302), (1568, 784)), plan=((1,), (0,), ()), p2=((500, 101, 302, 601), (1568, 0))),
    dict(name='rebuild', local=0, turn=5, rebuild=True, C=0, R=(), o=900, e=900, N=1000, sel=(),
         n=0, L=0, flags=(False, False, False), run=(), fresh=(), closed=(),
         p1=None, plan=((), (), ()), p2=((601, 602), (0, 0))),
]


def session(case):
    C, Q = case['C'], case['C'] // P * P
    if case['local'] == 0:
        return NS(local_turn=0, computed_tokens=(), sealed_tokens=(), tail=None, next_token=None,
                  resident=(), boundary='')
    tail = dict(page=Q // P, start=Q, valid_end=C, block_ids=[[bid(0, 'T')]]) if C > Q else None
    return NS(local_turn=case['local'], computed_tokens=FULL[:C], sealed_tokens=FULL[:Q], tail=tail,
              next_token=FULL[C], resident=case['R'], boundary=H('d'))


def binding(case, **over):
    values = dict(epoch='e', session='s', request='r%d' % case['turn'], turn=case['turn'],
                  local_turn=case['local'], rebuild=case.get('rebuild', False), generation='g',
                  model='m', renderer='r', tokens=FULL[:case['N']], stable_end=case['o'],
                  observation_span=(case['o'], case['e']) if case['local'] else None,
                  native_prefix='p', nonce='n')
    values.update(over)
    return RenderBinding(**values)


def identity(turn=1):
    return NativeIdentity('r1', 's', turn, 'g', H('a'), H('c'), 'm')


def assignment(geo, turn=1, **over):
    values = dict(identity=identity(turn), local_turn=geo.local_turn, computed_start=geo.C,
                  sealed_start=geo.Q, candidate_end=geo.L, observation_start=geo.o,
                  observation_end=geo.e, phase2_start=geo.phase2_start, prompt_tokens=geo.N,
                  page_tokens=P, token_digest=digest(list(FULL[:geo.N])),
                  carry_digest=H('d') if geo.local_turn else ZERO_DIGEST)
    values.update(over)
    return RenderAssignment(**values)


def plan_sets(geo, R, selected):
    """Spec 3.4 retain/restore/release with R' = R + tail-candidate page."""
    Rp = set(R) | ({geo.Q // P} if geo.tail_candidate else set())
    return (tuple(sorted(set(selected) & Rp)), tuple(sorted(set(selected) - Rp)),
            tuple(sorted(Rp - set(selected))))


def build_lease(geo, R, *, att=ATT, groups=4, goal_tokens=None):
    """Independent oracle of the spec 3.2 engine lease composition."""
    M, fresh = -(-geo.e // P), list(geo.fresh_pages)
    rows = []
    for g in range(groups):
        if g in att:
            rows.append([bid(g, 'R', p) for p in R] + ([bid(g, 'T')] if geo.tail else [])
                        + [bid(g, 'F', p) for p in fresh])
        else:
            rows.append([0] * (M - 1) + [bid(g, 'M')])
    borrowed = [bid(g, 'R', p) for g in att for p in R] + ([bid(g, 'T') for g in att] if geo.tail else [])
    owned = [bid(g, 'F', p) for p in fresh for g in att] + [bid(g, 'M') for g in range(groups) if g not in att]
    goal = None
    if goal_tokens:
        width = -(-goal_tokens // P)
        grows = [[bid(g, 'GF', i) for i in range(width)] if g in att else
                 [0] * (width - 1) + [bid(g, 'GM')] for g in range(groups)]
        gids = [b for r in grows for b in r if b]
        goal = dict(lease_id='goal-' + 'b' * 32, tokens=list(range(goal_tokens)), rows=grows,
                    owned=gids, zero=list(gids), incarnations={str(b): 1 for b in gids},
                    payload_sha256=H('f'))
    return dict(lease_id='phase1-' + 'a' * 32, request='r1', binding=H('b'), session='s',
                namespace='sparse/x', generation=3, epoch=H('e'), source=H('c'),
                local_turn=geo.local_turn, page_tokens=P, start=geo.C, end=geo.e, sealed_start=geo.Q,
                candidate_end=geo.L, resident_pages=list(R),
                tail=dict(page=geo.Q // P, valid_end=geo.C, block_ids=[[bid(g, 'T')] for g in att])
                if geo.tail else None,
                tail_candidate=geo.tail_candidate, rows=rows, borrowed=borrowed, owned=owned,
                zero=list(owned), run_fresh={str(p): [[bid(g, 'F', p)] for g in att] for p in fresh},
                incarnations={str(b): 2 for r in rows for b in r if b},
                sparse=dict(logical_matched_tokens=geo.Q,
                            dropped_full_attention_tokens=geo.Q - len(R) * P),
                carry_digest=H('d'), goal=goal, mac=H('9'))


def phase2_layout(case, geo):
    """[A_t][run]: retain -> resident/tail block, restore -> fresh target; run from phase 1."""
    retain, restore, _ = plan_sets(geo, case['R'], case['sel'])
    block = lambda p: (bid(0, 'S', p) if p in restore else
                       bid(0, 'T') if geo.tail and p == geo.Q // P else bid(0, 'R', p))
    run = tuple(bid(0, 'T') if geo.tail_live and p == geo.Q // P else bid(0, 'F', p)
                for p in geo.run_pages)
    return RowLayout(P, geo.L, case['sel'], {0: tuple(block(p) for p in case['sel'])}, {0: run})


def physical_block(row, plan, x):
    return row[(x - plan.dropped_full_attention_tokens) // P]


class GoldenLayoutTests(unittest.TestCase):
    def test_every_spec_edge_case(self):
        for case in CASES:
            with self.subTest(case=case['name']):
                geo = turn_geometry(binding(case), P, session(case))
                Q = case['C'] // P * P
                self.assertEqual((geo.local_turn, geo.C, geo.Q, geo.o, geo.e, geo.N, geo.n, geo.L),
                                 (case['local'], case['C'], Q, case['o'], case['e'], case['N'],
                                  case['n'], case['L']))
                self.assertEqual((geo.tail, geo.tail_candidate, geo.tail_live), case['flags'])
                self.assertEqual(geo.phase2_start, case['e'] if case['local'] else 0)
                self.assertEqual((tuple(geo.run_pages), tuple(geo.fresh_pages), tuple(geo.closed_pages)),
                                 (case['run'], case['fresh'], case['closed']))
                # The worker assignment carries the same geometry.
                a = assignment(geo, case['turn'])
                self.assertEqual((a.tail, a.tail_candidate, a.tail_live, a.candidate_pages,
                                  tuple(a.run_pages)), (*case['flags'], case['n'], case['run']))
                self.assertFalse(hasattr(a, 'new_seal'))
                self.assertEqual(plan_sets(geo, case['R'], case['sel']), case['plan'])
                if case['p1'] is None:
                    self.assertEqual(geo.phase2_start, 0)
                else:
                    self._phase1(case, geo)
                self._phase2(case, geo)

    def _phase1(self, case, geo):
        lease = parse_lease(build_lease(geo, case['R']), attention_groups=ATT)
        row, (start, dropped) = case['p1']
        self.assertEqual(lease.rows[0], row)
        self.assertEqual(dict(lease.sparse), dict(logical_matched_tokens=start,
                                                   dropped_full_attention_tokens=dropped))
        self.assertEqual(lease.mamba_index, -(-geo.e // P) - 1)
        self.assertEqual(tuple(lease.fresh_pages), case['fresh'])
        self.assertEqual(closed_pages(geo.C, geo.e, P), geo.closed_pages)
        self.assertFalse(set(lease.zero) & set(lease.borrowed))
        # Phase 1 is compose_rows with run_start=Q, prefix R and run [tail][fresh].
        run = row[len(case['R']):]
        layout = RowLayout(P, geo.Q, case['R'], {0: row[:len(case['R'])]}, {0: run})
        rows, plan = compose_rows(layout, {0: (row, P)})
        self.assertEqual((rows[0][0], plan), (row, SparsePlan(start, dropped, ATT)))
        for x in (geo.C, geo.o, geo.e - 1):          # logical x -> |R| + (x - Q)//P
            want = bid(0, 'T') if geo.tail and x < geo.Q + P else bid(0, 'F', x // P)
            self.assertEqual(physical_block(row, plan, x), want)

    def _phase2(self, case, geo):
        layout = phase2_layout(case, geo)
        golden, (start, dropped) = case['p2']
        fixed = layout.prefix_blocks[0] + layout.run_blocks[0]
        transients = tuple(b for b in golden if b not in fixed)
        # Native order: engine-installed head, then allocation order. A scrambled
        # head proves compose_rows orders fixed blocks by the layout and keeps
        # transients in native allocation order.
        for raw in (fixed + transients, transients + tuple(reversed(fixed))):
            rows, plan = compose_rows(layout, {0: (raw, P)})
            self.assertEqual(rows[0], (golden, P))
            self.assertEqual(plan, SparsePlan(start, dropped, ATT))
        self.assertEqual(dropped, geo.L - len(case['sel']) * P)
        for x in {geo.L, geo.e, geo.N - 1} - {0} if geo.local_turn else ():
            page = x // P
            want = (bid(0, 'T') if geo.tail_live and page == geo.Q // P else
                    bid(0, 'F', page) if page in geo.run_pages else None)
            if want is not None:
                self.assertEqual(physical_block(golden, plan, x), want)
        if geo.tail_candidate:
            self.assertEqual(bid(0, 'T') in golden, geo.Q // P in case['sel'])


class RenderAssignmentTests(unittest.TestCase):
    def geo(self, name):
        case = next(c for c in CASES if c['name'] == name)
        return turn_geometry(binding(case), P, session(case))

    def test_rejects_every_broken_invariant(self):
        g1, g0 = self.geo('tail_live'), self.geo('local_turn_zero')
        bad = [
            (g0, dict(computed_start=784, sealed_start=784)), (g0, dict(observation_end=901)),
            (g0, dict(observation_start=1001, observation_end=1001)),
            (g0, dict(carry_digest=H('d'))), (g0, dict(phase2_start=900)), (g0, dict(candidate_end=784)),
            (g1, dict(sealed_start=1500)), (g1, dict(computed_start=1568 + 784)),
            (g1, dict(computed_start=1500)), (g1, dict(observation_start=1668)),
            (g1, dict(observation_end=2410)), (g1, dict(observation_end=1671)),
            (g1, dict(candidate_end=784)), (g1, dict(candidate_end=2352)),
            (g1, dict(phase2_start=0)), (g1, dict(carry_digest=ZERO_DIGEST)),
            (g1, dict(token_digest='x' * 64)), (g1, dict(local_turn=2)),
            (g1, dict(page_tokens=0)), (g1, dict(prompt_tokens=True)), (g1, dict(identity='r1')),
            # C == Q with L = Q + P: a candidate tail page without a tail block.
            (g1, dict(computed_start=1568, observation_start=2360, candidate_end=2352)),
        ]
        for geo, override in bad:
            with self.subTest(override=override), self.assertRaises((LiveV2Error, TypeError)):
                assignment(geo, 1, **override)
        # Positional v1 construction (old_seal/stable_end) is gone.
        with self.assertRaises(TypeError):
            RenderAssignment(identity(), 0, 900, 1000, 900, 1000, P, H('c'))

    def test_rebuild_keeps_global_ordinal(self):
        geo = self.geo('rebuild')
        a = assignment(geo, 5)
        self.assertEqual((a.identity.turn, a.local_turn, a.carry_digest), (5, 0, ZERO_DIGEST))
        self.assertEqual(tuple(a.run_pages), ())


class TurnGeometryTests(unittest.TestCase):
    def case(self, name, **over):
        return dict(next(c for c in CASES if c['name'] == name), **over)

    def reject(self, b, s, message):
        with self.assertRaisesRegex(Abort, message):
            turn_geometry(b, P, s)

    def test_prefix_next_token_and_local_turn(self):
        c = self.case('tail_live')
        s = session(c); s.computed_tokens = s.computed_tokens[:-1] + (0,)
        self.reject(binding(c), s, 'SESSION_PREFIX_MISMATCH')
        s = session(c); s.next_token = FULL[c['C']] + 1
        self.reject(binding(c), s, 'SESSION_PREFIX_MISMATCH')
        s = session(c); s.sealed_tokens = FULL[:784]           # Q != floor(C/P)*P
        self.reject(binding(c), s, 'SESSION_PREFIX_MISMATCH')
        s = session(c); s.local_turn = 2
        self.reject(binding(c), s, 'LOCAL_TURN_MISMATCH')

    def test_observation_and_candidate_range(self):
        c = self.case('c_equals_q')
        self.reject(binding(c, stable_end=1568, observation_span=(1568, 2000)), session(c), 'OBSERVATION')
        self.reject(binding(c, observation_span=(1571, 2010)), session(c), 'OBSERVATION')   # e == N
        c = self.case('c_equals_q', o=3200, e=3300, N=3310)                                  # n = 4
        self.reject(binding(c), session(c), 'CANDIDATE_RANGE')
        c = self.case('c_equals_q', o=2355, e=2400, N=2410)          # candidate page, no tail
        self.reject(binding(c), session(c), 'CANDIDATE_RANGE')
        # stable_end must equal the Observation start at render shape already.
        with self.assertRaisesRegex(Abort, 'Observation boundary'):
            turn_geometry(binding(self.case('tail_live'), stable_end=1670), P, session(self.case('tail_live')))

    def test_tail_flag_and_mirror(self):
        c = self.case('tail_live')
        s = session(c); s.tail = None
        self.reject(binding(c), s, 'TAIL_MIRROR_MISMATCH')
        s = session(c); s.tail = dict(s.tail, valid_end=1667)
        self.reject(binding(c), s, 'TAIL_MIRROR_MISMATCH')
        c = self.case('c_equals_q')
        s = session(c); s.tail = dict(page=2, start=1568, valid_end=1568)
        self.reject(binding(c), s, 'TAIL_MIRROR_MISMATCH')

    def test_local_turn_zero(self):
        c = self.case('local_turn_zero')
        self.assertEqual(turn_geometry(binding(c), P, session(c)),
                         TurnGeometry(0, 0, 0, 900, 900, 1000, 0, 0, False, False, False, 0, P))
        s = session(c); s.computed_tokens = FULL[:10]
        self.reject(binding(c), s, 'LOCAL_TURN_ZERO')
        for field, value in (('tail', {}), ('next_token', 7), ('resident', (0,)), ('boundary', H('d')),
                             ('local_turn', 1)):
            s = session(c); setattr(s, field, value)
            self.reject(binding(c), s, 'LOCAL_TURN_ZERO')
        self.reject(binding(c, stable_end=1000), session(c), 'LOCAL_TURN_ZERO')   # empty header
        self.reject(binding(c, stable_end=0), session(c), 'LOCAL_TURN_ZERO')
        self.reject(binding(c, observation_span=(900, 950)), session(c), 'Observation')

    def test_geometry_dataclass_is_self_consistent(self):
        good = TurnGeometry(1, 1668, 1568, 1671, 2400, 2410, 2, 1568, True, False, True, 2400, P)
        self.assertEqual(tuple(good.run_pages), (2, 3))
        for override in (dict(tail=False), dict(tail_candidate=True), dict(Q=784), dict(n=3),
                         dict(phase2_start=0), dict(L=2352), dict(o=1668), dict(e=2410),
                         dict(local_turn=0), dict(C=True)):
            values = dict(vars(good), **override)
            with self.subTest(override=override), self.assertRaises(Abort):
                TurnGeometry(**values)


class RenderBindingTests(unittest.TestCase):
    def authority(self):
        return Authority(epoch='e', loaded_source=H('c'))

    def render(self, authority, rid, **over):
        T = FULL[:1000]
        values = dict(session='s', turn=0, local_turn=0, rebuild=False, generation='g', model='m',
                      renderer='r', tokens=T, stable_prefix_tokens=T[:900], observation_span=None,
                      native_prefix='p')
        values.update(over)
        return authority.render(NS(request_id=rid, prompt_token_ids=values['tokens']), **values)

    def test_render_binds_local_turn_and_rebuild(self):
        a = self.authority()
        b, mac = self.render(a, 'r0')
        self.assertEqual((b.turn, b.local_turn, b.rebuild), (0, 0, False))
        a.authenticate(a._requests['r0'][0], b, mac)
        rebuilt, mac2 = self.render(a, 'r5', turn=5, rebuild=True)
        self.assertEqual((rebuilt.turn, rebuilt.local_turn, rebuilt.rebuild), (5, 0, True))
        a.authenticate(a._requests['r5'][0], rebuilt, mac2)
        T = FULL[:1000]
        b1, _ = self.render(a, 'r1', turn=3, local_turn=1, stable_prefix_tokens=T[:900],
                            observation_span=(900, 990))
        self.assertEqual((b1.stable_end, b1.observation_span), (900, (900, 990)))

    def test_render_rejects_inconsistent_turns(self):
        T = FULL[:1000]
        bad = [dict(rebuild=True),                                    # rebuild needs a prior request
               dict(turn=3, local_turn=1, rebuild=True, observation_span=(900, 990)),
               dict(turn=1, local_turn=2, observation_span=(900, 990)),
               dict(turn=1, local_turn=0, observation_span=(900, 990)),
               dict(turn=1, local_turn=1),                            # no Observation
               dict(turn=1, local_turn=1, observation_span=(901, 990)),  # o != stable_end
               dict(turn=1, local_turn=1, observation_span=[900, 990]),
               dict(rebuild=1)]
        for i, override in enumerate(bad):
            with self.subTest(override=override), self.assertRaises(Abort):
                self.render(self.authority(), 'x%d' % i, **override)
        with self.assertRaises(TypeError):                           # no silent local_turn default
            self.authority().render(NS(request_id='y', prompt_token_ids=T), session='s', turn=0,
                                    generation='g', model='m', renderer='r', tokens=T,
                                    stable_prefix_tokens=T[:900], observation_span=None,
                                    native_prefix='p')

    def test_identity_covers_new_fields(self):
        c = CASES[0]
        self.assertNotEqual(binding(c).identity, binding(c, rebuild=True, turn=2).identity)
        self.assertNotEqual(binding(c, turn=2).identity, binding(c, rebuild=True, turn=2).identity)

    def test_select_pages_uses_local_turn(self):
        with self.assertRaises(Abort):
            select_pages('goal_query', 5, local_turn=0, native_prefix='p', ranking=(0, 1, 2, 3, 4),
                         selected_count=2)
        self.assertEqual(select_pages('goal_query', 5, local_turn=1, native_prefix='p',
                                      ranking=(4, 1, 0, 2, 3), selected_count=2), ((1, 4), None))


class LeaseTests(unittest.TestCase):
    def geo(self, name):
        case = next(c for c in CASES if c['name'] == name)
        return case, turn_geometry(binding(case), P, session(case))

    def test_parse_all_tail_cases(self):
        for name in ('c_equals_q', 'tail_live', 'tail_live_short_observation', 'tail_candidate_selected'):
            with self.subTest(case=name):
                case, geo = self.geo(name)
                raw = build_lease(geo, case['R'])
                lease = parse_lease(raw)
                self.assertEqual(lease.attention_groups, ATT)
                self.assertEqual(lease.lease_digest, digest(raw))
                self.assertEqual((lease.tail is not None, lease.tail_candidate),
                                 (geo.tail, geo.tail_candidate))
                self.assertEqual(set(lease.borrowed), {bid(0, 'R', p) for p in case['R']} |
                                 ({bid(0, 'T')} if geo.tail else set()))
                self.assertEqual(lease.zero, lease.owned)
                self.assertEqual(set(lease.run_fresh), set(geo.fresh_pages))
                self.assertTrue(all(type(k) is int for k in lease.incarnations))
                for g in (1, 2, 3):
                    self.assertEqual(lease.rows[g][:-1], (0,) * lease.mamba_index)
        case, geo = self.geo('tail_live')
        tuples = build_lease(geo, case['R'])
        tuples['rows'] = tuple(tuple(r) for r in tuples['rows'])           # in-process RPC shape
        self.assertEqual(parse_lease(tuples).rows, parse_lease(build_lease(geo, case['R'])).rows)

    def test_attention_groups_are_inferred_and_pinned(self):
        case, geo = self.geo('tail_candidate_selected')
        lease = parse_lease(build_lease(geo, case['R'], att=(2,)))
        self.assertEqual(lease.attention_groups, (2,))
        two = parse_lease(build_lease(geo, case['R'], att=(0, 3), groups=5), attention_groups=(0, 3))
        self.assertEqual((two.attention_groups, two.tail.block_ids), ((0, 3), ((200,), (3200,))))
        with self.assertRaisesRegex(LiveV2Error, 'group layout'):
            parse_lease(build_lease(geo, case['R']), attention_groups=(1,))

    def test_goal_lease_uses_its_own_blocks(self):
        case, geo = self.geo('tail_live')
        lease = parse_lease(build_lease(geo, case['R'], goal_tokens=900))
        self.assertEqual((lease.goal.rows[0], lease.goal.rows[1]), ((700, 701), (0, 1800)))
        raw = build_lease(geo, case['R'], goal_tokens=900)
        raw['goal']['rows'][0][0] = bid(0, 'F', 3)                    # reuses a phase-1 block
        raw['goal']['owned'][0] = raw['goal']['zero'][0] = bid(0, 'F', 3)
        raw['goal']['incarnations'] = {str(b): 1 for r in raw['goal']['rows'] for b in r if b}
        with self.assertRaisesRegex(LiveV2Error, 'alias'):
            parse_lease(raw)
        raw = build_lease(geo, case['R'], goal_tokens=900)
        raw['goal']['rows'][1] = [1800, 0]
        with self.assertRaisesRegex(LiveV2Error, 'null-padded'):
            parse_lease(raw)
        raw = build_lease(geo, case['R'], goal_tokens=900)
        raw['goal']['lease_id'] = raw['lease_id']
        with self.assertRaisesRegex(LiveV2Error, 'goal lease identity'):
            parse_lease(raw)

    def test_rejects_malformed_leases(self):
        case, geo = self.geo('tail_live')
        T, F3, M1 = bid(0, 'T'), bid(0, 'F', 3), bid(1, 'M')
        def mutate(fn):
            raw = build_lease(geo, case['R']); fn(raw); return raw
        bad = {
            'key set': lambda r: r.update(extra=1),
            'lease id': lambda r: r.update(lease_id='probe-' + 'a' * 32),
            'identity': lambda r: r.update(carry_digest=ZERO_DIGEST),
            'integers': lambda r: r.update(generation=True),
            'local turn': lambda r: r.update(local_turn=0),
            'geometry': lambda r: r.update(tail_candidate=True),
            'resident Pages': lambda r: r.update(resident_pages=[1, 0]),
            'tail mismatch': lambda r: r['tail'].update(valid_end=1667),
            'fresh Page set': lambda r: r['run_fresh'].pop('3'),
            'FullAttention row': lambda r: r['rows'][0].__setitem__(slice(2, 4), [F3, T]),
            'null-padded': lambda r: r['rows'].__setitem__(1, [M1, 0, 0, 0]),
            'aliased': lambda r: r['rows'][2].__setitem__(-1, M1),
            'zero != owned': lambda r: r.update(zero=r['owned'] + [T]),
            'fresh block borrowed': lambda r: r.update(borrowed=r['borrowed'] + [F3]),
            'incarnation': lambda r: r['incarnations'].pop(str(T)),
            'SparsePlan': lambda r: r['sparse'].update(dropped_full_attention_tokens=784),
            'tail without': lambda r: r.update(start=1568, tail=r['tail']),
        }
        for message, fn in bad.items():
            with self.subTest(message=message), self.assertRaises(LiveV2Error):
                parse_lease(mutate(fn))
        with self.assertRaises(LiveV2Error):
            parse_lease({k: v for k, v in build_lease(geo, case['R']).items() if k != 'mac'})


class Phase1ResultTests(unittest.TestCase):
    def values(self, **over):
        v = dict(binding=H('b'), lease_id='phase1-' + 'a' * 32, lease_digest=H('c'), start=2350, end=3000,
                 page_vectors={2: [0.6, 0.8]}, observation_vector=[1.0, 0.0], goal_vector=None,
                 goal_payload_sha256=None, goal_token_count=None, state_digest=H('d'),
                 tail_capture_digest=H('e'), position_evidence=[dict(start=2350)],
                 vector_space=dict(decoder_layers=64), worker_receipt=dict(body={}, mac=H('f')))
        v.update(over)
        return v

    def test_normalizes_and_validates(self):
        r = Phase1Result(**self.values())
        self.assertEqual((r.page_vectors, r.observation_vector, r.position_evidence),
                         ({2: (0.6, 0.8)}, (1.0, 0.0), (dict(start=2350),)))
        self.assertEqual(set(r.page_vectors), set(closed_pages(2350, 3000, P)))
        g = Phase1Result(**self.values(goal_vector=[0.0, 1.0], goal_payload_sha256=H('a'),
                                       goal_token_count=12, page_vectors={}, tail_capture_digest=None))
        self.assertEqual(g.goal_vector, (0.0, 1.0))
        bad = [dict(page_vectors={'2': [1.0, 0.0]}), dict(page_vectors={2: [1.0, 0.0], 4: [1.0, 0.0]}),
               dict(page_vectors={2: [float('nan'), 1.0]}), dict(page_vectors={2: [1, 0]}),
               dict(page_vectors={2: [1.0, 0.0, 0.0]}), dict(goal_vector=[1.0, 0.0]),
               dict(goal_vector=[1.0, 0.0], goal_payload_sha256=H('a'), goal_token_count=0),
               dict(lease_id='phase1-xyz'), dict(start=3000), dict(binding='id-r1'),
               dict(state_digest=None), dict(tail_capture_digest='x'), dict(observation_vector=[]),
               dict(vector_space=None)]
        for override in bad:
            with self.subTest(override=override), self.assertRaises(LiveV2Error):
                Phase1Result(**self.values(**override))


class ComposeRowsTests(unittest.TestCase):
    def layout(self, prefix_pages=(0,), prefix=(100,), run=(200, 303), start=1568, groups=None):
        groups = groups or {0: (prefix, run)}
        return RowLayout(P, start, prefix_pages, {g: v[0] for g, v in groups.items()},
                         {g: v[1] for g, v in groups.items()})

    def test_empty_prefix_only_and_mixed(self):
        rows, plan = compose_rows(RowLayout.empty(P, ATT), {0: ((601, 602), P)})
        self.assertEqual((rows, plan), ({0: ((601, 602), P)}, SparsePlan(0, 0, ATT)))
        rows, plan = compose_rows(self.layout((0, 1), (100, 101), ()), {0: ((601, 101, 100), P)})
        self.assertEqual((rows[0][0], plan), ((100, 101, 601), SparsePlan(1568, 0, ATT)))
        rows, plan = compose_rows(self.layout(), {0: ((601, 303, 100, 200, 602), P)})
        self.assertEqual((rows[0][0], plan), ((100, 200, 303, 601, 602), SparsePlan(1568, 784, ATT)))

    def test_stable_across_native_steps(self):
        layout, prior = self.layout(), ()
        for raw in ((100, 200, 303), (100, 200, 303, 601), (100, 200, 303, 601, 602)):
            rows, plan = compose_rows(layout, {0: (raw, P)})
            self.assertEqual(rows[0][0][:len(prior)], prior)
            self.assertEqual(plan, SparsePlan(1568, 784, ATT))
            prior = rows[0][0]

    def test_multiple_groups(self):
        layout = self.layout(groups={0: ((100,), (200,)), 4: ((4100,), (4200,))})
        rows, plan = compose_rows(layout, {4: ((4600, 4200, 4100), P), 0: ((100, 200), P)})
        self.assertEqual(rows, {0: ((100, 200), P), 4: ((4100, 4200, 4600), P)})
        self.assertEqual(plan.full_attention_group_ids, (0, 4))

    def test_rejections(self):
        layout = self.layout()
        for raw, message in (({0: ((100, 200), P)}, 'missing'), ({0: ((100, 200, 303, 601, 601), P)}, 'aliased'),
                             ({0: ((100, 200, 303, 0), P)}, 'null'), ({0: ((100, 200, 303), 392)}, 'one FA block'),
                             ({0: ((100, 200, 303), P), 1: ((5,), P)}, 'groups differ'), ({}, 'groups differ')):
            with self.subTest(message=message), self.assertRaisesRegex(LiveV2Error, message):
                compose_rows(layout, raw)
        for args, message in (
                (((0,), (100,), (100, 303)), 'aliased'),                  # prefix block reused in the run
                (((1, 0), (101, 100), (303,)), 'ascending'),
                (((2,), (102,), (303,)), 'before the run'),               # prefix Page >= L/P
                (((0,), (100, 101), (303,)), 'width'),
                (((0,), (100,), (303,), 1500), 'geometry')):
            with self.subTest(message=message), self.assertRaisesRegex(LiveV2Error, message):
                self.layout(*args)
        with self.assertRaisesRegex(LiveV2Error, 'width'):
            self.layout(groups={0: ((100,), (200,)), 4: ((4100,), ())})
        with self.assertRaisesRegex(LiveV2Error, 'group set'):
            RowLayout(P, 0, (), {}, {})

    def test_sparse_plan_is_fail_closed(self):
        for args in ((0, 0, ()), (784, 1568, ATT), (True, 0, ATT), (0, 0, [0]), (0, 0, (0, 0))):
            with self.subTest(args=args), self.assertRaises(LiveV2Error):
                SparsePlan(*args)


class WorkerIntervalTests(unittest.TestCase):
    def test_phase2_starts_at_e_without_seal_split(self):
        case = next(c for c in CASES if c['name'] == 'tail_live')
        a = assignment(turn_geometry(binding(case), P, session(case)))
        state = NS(assignment=a, allocation_generation=3, poisoned=False, active=None, cursor=2400)
        record = dict(identity=a.identity, generation=3)
        worker_interval(record, state, 2400, 10)                    # the header [e, N) in one chunk
        with self.assertRaisesRegex(LiveV2Error, 'gap'):
            state.cursor = 1568; worker_interval(record, state, 1568, 10)
        state.cursor = 2400
        with self.assertRaisesRegex(LiveV2Error, 'crosses prompt'):
            worker_interval(record, state, 2400, 11)
        record['native_interval'] = (2400, 10); state.cursor = 2410
        worker_interval(record, state, 2410, 1)                     # decode
        with self.assertRaisesRegex(LiveV2Error, 'speculative'):
            worker_interval(record, state, 2410, 2)
        a0 = assignment(TurnGeometry(0, 0, 0, 900, 900, 1000, 0, 0, False, False, False, 0, P), 0)
        worker_interval(dict(identity=a0.identity, generation=3),
                        NS(assignment=a0, allocation_generation=3, poisoned=False, active=None, cursor=0), 0, 784)


if __name__ == '__main__':
    unittest.main()
