"""A turn-geometry refusal cleans up like any other refused admission (review 1110, runtime 1.4). CPU only.

``SplitPrivateEndpoint.worker_payload`` is where the turn geometry is first evaluated
(``remote_ra_native_growing.contract.turn_geometry``: CANDIDATE_RANGE, OBSERVATION_GEOMETRY,
SESSION_PREFIX_MISMATCH, TAIL_MIRROR_MISMATCH). ``LiveEndpoint.reserve_worker_steps`` called it before it set
``prepared_payloads`` and outside its ``try``: a refusal skipped the admission's own cleanup, the lifecycle's
cancel hit ``KeyError`` on the missing payload, the entry stayed ``drain_pending`` and every lease sweep past
the 360-s deadline raised on the engine thread (no containment: the engine died with every session).

* LifecycleTests: the reviewer's script (work/F-engine-turn/geometry_refusal_cleanup.py) as a test: the real
  EngineLifecycle, LiveEndpoint, turn_geometry and Session; the private endpoint and the worker are stand-ins.
* EngineTests: test_ms_engine's three-session engine with the real private endpoint, adapter and worker
  owner: a continuing admission of one session is refused by the geometry; the request is cancelled and
  tombstoned, its session poisoned on the engine and the worker, nothing leaks, the lease sweep stays quiet
  and the other sessions keep serving.
* BoundaryTests: kv-session's client-supplied ``boundary_tokens`` (o = C + boundary_tokens) is held to the
  same geometry at the protocol check (``BOUNDARY_RANGE``, a 400 that changes nothing): the limit equals
  what turn_geometry accepts for every C and P tried, and the renderers hand the transcripts their Page size.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_ms_engine as ME  # noqa: E402
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402
from remote_ra_live_v2.composition import LiveEndpoint  # noqa: E402
from remote_ra_live_v2.lifecycle import EngineLifecycle  # noqa: E402
from remote_ra_native_growing import contract as native  # noqa: E402
from remote_ra_native_growing.adapter import Session  # noqa: E402
from remote_ra_native_growing.contract import Abort, RenderBinding, turn_geometry  # noqa: E402

LEASE = 360.0


def mirror(C, P, *, extra=100, boundary=2):
    """A committed session at C computed tokens and a continuing binding with o = C + boundary."""
    Q = C // P * P
    full = tuple(range(100, 101 + C))                 # the previous prompt and reply: full[C] is the last sample
    session = Session('ns', 'full', 'm', 'r', turn=1, local_turn=1, sealed_tokens=full[:Q],
                      computed_tokens=full[:C], next_token=full[C], boundary='a' * 64,
                      tail=dict(page=Q // P, start=Q, valid_end=C, block_ids=[[5]]) if C > Q else None,
                      resident=tuple(range(Q // P)))
    tokens = full + tuple(range(50000, 50000 + boundary + extra))
    o = C + boundary
    binding = RenderBinding(epoch='e', session='s0', request='r1', turn=1, local_turn=1, rebuild=False,
                            generation='g1', model='m', renderer='r', tokens=tokens, stable_end=o,
                            observation_span=(o, len(tokens) - 7), native_prefix='p', nonce='n')
    return session, binding


class LifecycleTests(unittest.TestCase):
    """The reviewer's script: a kv-session continuing request with boundary_tokens 40 at C = 40, P = 16."""

    def world(self, *, refuse=True, cancel=None):
        P = 16
        session, binding = mirror(40, P, boundary=40 if refuse else 2)
        calls = []

        class Private:
            phase1 = None
            adapter = NS(sessions={'s0': session})

            def render_claim(self, body):
                return NS(request_id='r1'), binding

            def worker_payload(self, b):
                geo = turn_geometry(b, P, session)    # SplitPrivateEndpoint.worker_payload's first check
                return dict(local_turn=geo.local_turn, namespace=session.namespace,
                            ns_generation=session.ns_generation)

        def call(operation, payload):
            calls.append((operation, dict(payload)))
            if operation == 'cancel' and cancel is not None:
                raise cancel
            return dict(operation=operation, binding=payload['binding'], drained=True, allocation_generation=1)
        core = NS(_remote_ra_private_v2_endpoint=Private(),
                  _remote_ra_growing_owner=NS(owner_check=lambda: None, verify_identity=lambda: None),
                  _remote_ra_live_v2_bridge=NS(assignments={}),
                  _remote_ra_growing_adapter=NS(owns=lambda rid: False))
        core._remote_ra_assembly = NS(state='active')
        endpoint = LiveEndpoint(core, core._remote_ra_assembly, NS(call=call))
        clock = [0.0]
        life = EngineLifecycle(endpoint, clock=lambda: clock[0])
        life.sessions['s0'] = (1, None)
        return NS(life=life, endpoint=endpoint, calls=calls, clock=clock, binding=binding, session=session)

    def admit(self, w):
        return w.life.admit(dict(request='r1', session='s0', turn=1, generation='g1'))

    def test_the_refusal_is_cleaned_up_and_the_lease_sweep_stays_quiet(self):
        w = self.world()
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            with self.assertRaisesRegex(Abort, 'CANDIDATE_RANGE') as caught:
                self.admit(w)
            # The refusal itself, not a group with a cleanup failure.
            self.assertNotIsInstance(caught.exception, BaseExceptionGroup)
            entry = w.life.entries['r1']
            self.assertEqual((entry['state'], entry['cleanup_error']), ('cancelled', None))
            self.assertEqual(w.life.sessions['s0'], (1, 'poisoned'))
            # The worker never saw a prepare: one cancel that names the request, its binding and its
            # worker session (so the worker tombstones the request and poisons that session).
            self.assertEqual([op for op, _ in w.calls], ['cancel'])
            payload = w.calls[0][1]
            self.assertEqual({k: payload[k] for k in ('request', 'session', 'turn', 'generation', 'binding')},
                             dict(request='r1', session='s0', turn=1, generation='g1', binding=w.binding.identity))
            self.assertEqual((payload['namespace'], payload['ns_generation']), ('ns', 0))
            self.assertNotIn('tokens', payload)
            self.assertNotIn('r1', w.endpoint.prepared_payloads)
            self.assertIn('r1', w.endpoint.prepared_drains)
            # The lease sweep: before and after the deadline, any number of times.
            for now in (100.0, LEASE + 1, LEASE + 2, 10 * LEASE):
                w.clock[0] = now
                w.life.expire()
            self.assertEqual(w.life.entries['r1']['state'], 'cancelled')
            # A retried cancel (the frontend's compensation) answers from the kept receipt.
            self.assertEqual(w.life.cancel(('r1', 's0', 1, 'g1'))['state'], 'cancelled')
            self.assertEqual([op for op, _ in w.calls], ['cancel'])

    def test_an_accepted_geometry_is_unchanged(self):
        w = self.world(refuse=False)
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'), \
                patch.object(LiveEndpoint, 'enqueue', lambda self, request, binding: None), \
                patch.object(w.endpoint.core, '_remote_ra_live_v2_bridge',
                             NS(assignments={}, publish_assignment=lambda a, g: None), create=True), \
                patch('remote_ra_live_v2.composition.RunnerAllocationOwner._assignment', lambda raw: raw), \
                patch.object(w.endpoint.core._remote_ra_private_v2_endpoint, 'claims', {'r1': {}}, create=True), \
                patch.object(w.endpoint.core._remote_ra_private_v2_endpoint, 'reserve',
                             lambda request, binding: None, create=True), \
                patch.object(w.endpoint.core, '_remote_ra_live_v2_boundary', NS(submit=lambda b: None), create=True):
            # The payload the worker gets is base fields then worker_payload's, in that order, as before.
            w.endpoint.core._remote_ra_private_v2_endpoint.worker_payload = lambda b: dict(
                local_turn=1, namespace='ns', ns_generation=0, identity={}, assignment={}, tokens=[1])
            self.assertEqual(self.admit(w)['state'], 'accepted')
        self.assertEqual([op for op, _ in w.calls], ['prepare'])
        self.assertEqual(list(w.calls[0][1]), ['request', 'session', 'turn', 'generation', 'binding', 'local_turn',
                                               'namespace', 'ns_generation', 'identity', 'assignment', 'tokens'])
        self.assertIs(w.endpoint.prepared_payloads['r1']['tokens'], w.calls[0][1]['tokens'])

    def test_a_failed_cleanup_is_still_reported_and_retried(self):
        """The cleanup's own failure keeps today's handling: both errors, drain_pending, and the next
        sweep past the deadline retries the cancel with the kept payload (no KeyError)."""
        w = self.world(cancel=RuntimeError('worker gone'))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            with self.assertRaises(BaseExceptionGroup) as caught:
                self.admit(w)
            flat = []

            def leaves(error):
                for inner in getattr(error, 'exceptions', ()) or [None]:
                    if inner is None:
                        flat.append(str(error))
                    else:
                        leaves(inner)
            leaves(caught.exception)
            self.assertTrue(any('CANDIDATE_RANGE' in text for text in flat) and
                            any('worker gone' in text for text in flat), flat)
            self.assertFalse(any('KeyError' in text or text == "'r1'" for text in flat), flat)
            self.assertEqual(w.life.entries['r1']['state'], 'drain_pending')
            self.assertIn('r1', w.endpoint.prepared_payloads)        # the handle for the retry
            w.clock[0] = LEASE + 1
            with self.assertRaises(BaseExceptionGroup) as again:
                w.life.expire()
            self.assertIn('worker gone', str(again.exception.exceptions[0]))
            self.assertGreaterEqual(len(w.calls), 2)


class EngineTests(unittest.TestCase):
    """test_ms_engine's engine (real private endpoint, adapter, worker owner, lifecycle, route)."""
    setUp = ME.MultiSessionEngineTests.setUp
    harness = ME.MultiSessionEngineTests.harness
    check = ME.MultiSessionEngineTests.check
    isolated = ME.MultiSessionEngineTests.isolated

    def refuse_next(self, lane):
        """The geometry refuses the next continuing admission of ``lane`` (as a client-supplied or an
        internal mismatch would), at its first evaluation."""
        real, seen = native.turn_geometry, []

        def geometry(binding, page_tokens, session):
            if binding.session == lane.sid and not seen:
                seen.append(binding.request)
                raise Abort('CANDIDATE_RANGE: n outside [Q/P, Q/P + 1]')
            return real(binding, page_tokens, session)
        return patch.object(native, 'turn_geometry', geometry), seen

    def test_one_sessions_refusal_leaves_the_engine_and_the_others_serving(self):
        from remote_ra_live_v2.contract import LiveV2Error
        h = self.harness('observation_query', [ME.A, ME.B, ME.C])
        a, b, c = h.lanes
        for k in range(2):
            for lane in h.lanes:
                self.check(h, h.script(lane, k))
        keys = ('namespace', 'turn', 'resident', 'tail', 'catalog', 'boundary', 'pins', 'carry', 'transcript')
        committed = lambda state: {k: state[k] for k in keys}        # noqa: E731
        before, others, free = committed(h.state(a)), [h.state(b), h.state(c)], h.pool.get_num_free_blocks()
        patcher, seen = self.refuse_next(a)
        with patcher:
            with self.assertRaisesRegex(Abort, 'CANDIDATE_RANGE') as caught:
                h.script(a, 2)
            self.assertNotIsInstance(caught.exception, BaseExceptionGroup)
        (rid,) = seen
        entry = h.lifecycle.entries[rid]
        self.assertEqual((entry['state'], entry['cleanup_error']), ('cancelled', None))
        self.assertEqual(h.lifecycle.sessions[a.sid], (2, 'poisoned'))
        # The worker: no record of the request, a cancel tombstone, the session poisoned as for a
        # prepare it refused; the engine's and the worker's committed state of A is unchanged.
        s = h.adapter.sessions[a.sid]
        self.assertNotIn(rid, h.owner.records)
        self.assertEqual(h.owner.terminals[rid]['operation'], 'cancel')
        self.assertTrue(h.owner.sessions[s.namespace]['poisoned'])
        self.assertEqual(committed(h.state(a)), before)
        self.assertEqual([h.state(b), h.state(c)], others)
        self.assertEqual(h.pool.get_num_free_blocks(), free)
        self.assertNotIn(rid, h.live.prepared_payloads)
        self.assertNotIn(rid, h.core._remote_ra_live_v2_bridge.assignments)
        self.isolated(h)
        # The lease sweep past every deadline: nothing to drain, nothing raised.
        clock = h.lifecycle.clock
        try:
            h.lifecycle.clock = lambda: clock() + 10 * LEASE
            h.lifecycle.expire()
            h.lifecycle.expire()
        finally:
            h.lifecycle.clock = clock
        # A refuses its next request at once; B and C keep serving.
        with self.assertRaisesRegex(LiveV2Error, 'poisoned'):
            h.script(a, 2)
        for k in (2, 3):
            for lane in (b, c):
                self.check(h, h.script(lane, k))


class BoundaryTests(unittest.TestCase):
    def transcript(self, C, P):
        t = K.KvTranscript(header=(7, 8, 9), im_start=3, vocab=1000, decode=str, stop_ids=frozenset({1}),
                           page_tokens=P)
        t.stream = tuple(100 + k % 800 for k in range(C + 1))
        t.turns = [dict(text='x', count=1)]
        t.initial_length = 5
        return t

    def request(self, t, boundary, extra=40):
        ids = [t.stream[-1]] + [11] * (boundary - 1) + [3] + [12] * extra + list(t.header)
        return K.KvRequest('0' * 32, tuple(ids), 3, t.initial_length, boundary, 16, 42, ())

    def test_the_limit_is_what_the_turn_geometry_accepts(self):
        for P in (4, 16, 32, 64):
            for C in sorted({P - 1, P, P + 1, 2 * P - 1, 2 * P, 3 * P + P // 2, 5 * P - 1, 5 * P, 5 * P + 1, 40}):
                t = self.transcript(C, P)
                limit = t.boundary_limit()
                accepted = []
                for boundary in range(1, 3 * P + 2):
                    session, binding = mirror(C, P, boundary=boundary, extra=3 * P)
                    try:
                        turn_geometry(binding, P, session)
                        accepted.append(boundary)
                    except Abort as refusal:
                        self.assertIn('CANDIDATE_RANGE', str(refusal), (P, C, boundary))
                self.assertEqual(accepted, list(range(1, limit + 1)), (P, C))
                self.assertGreaterEqual(limit, min(P - 1, 3), (P, C))    # the clients' 2-3 tokens always fit
                # The protocol check agrees, with a clear refusal that changes nothing.
                t.check(self.request(t, limit))
                stream = t.stream
                with self.assertRaises(K.KvSessionError) as caught:
                    t.render(self.request(t, limit + 1))
                self.assertEqual((caught.exception.kind, caught.exception.status), ('BOUNDARY_RANGE', 400))
                self.assertIn('boundary_tokens %d is out of contract: at most %d' % (limit + 1, limit),
                              str(caught.exception))
                self.assertEqual((t.stream, t.pending, t.staged, t.turn), (stream, None, None, 1))

    def test_the_reviewers_request_is_refused_at_the_protocol_check(self):
        t = self.transcript(40, 16)
        self.assertEqual(t.boundary_limit(), 23)                  # C % P = 8: o < Q + 2P
        t.check(self.request(t, 3))
        with self.assertRaises(K.KvSessionError) as caught:
            t.check(self.request(t, 40))
        self.assertEqual(caught.exception.kind, 'BOUNDARY_RANGE')

    def test_without_a_page_size_or_at_turn_zero_nothing_is_checked(self):
        t = self.transcript(40, 16)
        t.page_tokens = None
        self.assertIsNone(t.boundary_limit())
        t.check(self.request(t, 40))
        fresh = K.KvTranscript(header=(7, 8, 9), im_start=3, vocab=1000, decode=str, stop_ids=frozenset({1}),
                               page_tokens=16)
        self.assertIsNone(fresh.boundary_limit())
        for bad in (0, -1, 1.5, '16', True):
            with self.subTest(page_tokens=bad), self.assertRaises(Exception):
                K.KvTranscript(header=(7, 8, 9), im_start=3, vocab=1000, decode=str, stop_ids=frozenset({1}),
                               page_tokens=bad)

    def test_the_renderers_pin_their_page_size(self):
        """Both profiles' kv_transcripts factories hand the renderer's Page size to the transcripts."""
        import ast
        for name in ('runtime_renderer.py', 'gemma4_renderer.py'):
            tree = ast.parse((ROOT / 'source-overlay' / name).read_text(encoding='utf-8'))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and getattr(node.func, 'attr', None) == 'KvTranscript']
            self.assertEqual(len(calls), 1, name)
            self.assertIn('page_tokens', [k.arg for k in calls[0].keywords], name)


if __name__ == '__main__':
    unittest.main()
