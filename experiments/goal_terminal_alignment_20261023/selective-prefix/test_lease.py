"""The registered-add lease bounds admission only; it never cancels a running request.

v10 job 3276233 (random50, turn 6): admission plus about 1,190 generated tokens took
longer than the 120-second lease, the engine sweep cancelled the running request, and
the frontend never received a terminal output. CPU only, fake clock.
"""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source-overlay'))

from registered_add_protocol import Cancel, RegisteredAddDispatcher
from remote_ra_live_v2.lifecycle import EngineLifecycle
from remote_ra_live_v2.registered_route import bind_registered_dispatcher

LEASE = 120


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Endpoint:
    """Records every drain; the native side of EngineLifecycle."""
    def __init__(self, fail_drains=0):
        self.core, self.drains, self.fail_drains = NS(), [], fail_drains

    def verify_identity(self):
        pass

    def frame_identity(self, body):
        return tuple(body[k] for k in ('request', 'session', 'turn', 'generation'))

    def binding_identity(self, binding):
        return binding.ident

    def render_claim(self, body):
        return object(), NS(ident=self.frame_identity(body), identity=body['request'])

    def reserve_worker(self, request, binding):
        pass

    def enqueue(self, request, binding):
        pass

    def drain(self, binding, operation):
        if self.fail_drains:
            self.fail_drains -= 1
            raise RuntimeError('drain failed')
        self.drains.append((binding.identity, operation))
        return dict(drained=True, operation=operation)

    def verify_drain(self, binding, operation, receipt):
        if receipt.get('drained') is not True:
            raise RuntimeError('unknown drain')


def command(rid, turn=0):
    return NS(request_id=rid, command_id='add-' + rid, session_id='s1', lease_seconds=LEASE,
              metadata=dict(turn=turn, generation='g1'), core_b64='', source='src',
              external_id='x-' + rid, prompt_hash='h')


def build(clock, endpoint):
    dispatcher = RegisteredAddDispatcher(codec=None, boundary=None, decode_core=lambda raw: raw,
                                         preprocess=lambda core: core, clock=clock)
    lifecycle = EngineLifecycle(endpoint, clock=clock, lease_seconds=LEASE)
    bridge = bind_registered_dispatcher(dispatcher, lifecycle)
    return dispatcher, lifecycle, bridge


@patch('remote_ra_http.evidence.native_terminal', lambda *a: None)
class RunningRequestLeaseTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.endpoint = Clock(), Endpoint()
        self.dispatcher, self.lifecycle, self.bridge = build(self.clock, self.endpoint)

    def admit(self, rid='r1', turn=0):
        ack = self.bridge.add(self.dispatcher, command(rid, turn), fingerprint='f-' + rid)
        self.assertEqual(ack.state, 'accepted')

    def tick(self):
        # The engine's periodic lifecycle tick (remote_ra_live_v2.compat.tick).
        self.dispatcher.sweep()
        self.lifecycle.expire()

    def test_generation_longer_than_lease_is_not_cancelled(self):
        self.admit()
        for seconds in (60, 119, 121, 154):     # 1536 tokens at about 100 ms each is 154 s
            self.clock.now = 1000.0 + seconds
            self.tick()
            self.assertEqual(self.dispatcher._records['r1']['state'], 'accepted')
            self.assertEqual(self.lifecycle.entries['r1']['state'], 'accepted')
        self.assertEqual(self.endpoint.drains, [])
        self.dispatcher.finalize('r1', successful=True)
        self.assertEqual(self.dispatcher._records['r1']['state'], 'complete')
        self.assertEqual(self.endpoint.drains, [('r1', 'complete')])
        # The frontend's generation_closed Cancel must then report "complete".
        ack = self.bridge.cancel(self.dispatcher, Cancel('cancel-r1', 'r1'))
        self.assertEqual(ack.state, 'complete')
        self.assertEqual(self.lifecycle.sessions['s1'], (1, None))

    def test_next_turn_admits_after_a_long_turn(self):
        self.admit('r1', 0)
        self.clock.now += 3 * LEASE
        self.tick()
        self.dispatcher.finalize('r1', successful=True)
        self.admit('r2', 1)
        self.assertEqual(self.lifecycle.entries['r2']['state'], 'accepted')

    def test_client_cancel_still_drains_a_running_request(self):
        self.admit()
        self.clock.now += 2 * LEASE
        ack = self.bridge.cancel(self.dispatcher, Cancel('cancel-r1', 'r1'))
        self.assertEqual(ack.state, 'cancelled')
        self.assertEqual(self.endpoint.drains, [('r1', 'cancel')])
        self.assertEqual(self.lifecycle.sessions['s1'], (0, 'poisoned'))

    def test_disconnect_still_drains_a_running_request(self):
        self.admit()
        self.clock.now += 2 * LEASE
        self.lifecycle.disconnect()
        self.assertEqual(self.endpoint.drains, [('r1', 'disconnect')])

    def test_failed_admission_drain_is_still_retried_after_the_lease(self):
        endpoint = Endpoint(fail_drains=1)
        _, lifecycle, _ = build(self.clock, endpoint)
        body = dict(request='r9', session='s9', turn=0, generation='g1')
        with patch.object(endpoint, 'enqueue', side_effect=RuntimeError('enqueue failed')):
            with self.assertRaises(BaseExceptionGroup):
                lifecycle.admit(body)
        self.assertEqual(lifecycle.entries['r9']['state'], 'drain_pending')
        self.clock.now += LEASE - 1
        lifecycle.expire()
        self.assertEqual(endpoint.drains, [])
        self.clock.now += 2
        lifecycle.expire()
        self.assertEqual(endpoint.drains, [('r9', 'timeout')])
        self.assertEqual(lifecycle.entries['r9']['state'], 'cancelled')

    def test_legacy_accepted_record_keeps_its_lease(self):
        # Only live-v2 records changed; the legacy prototype path still expires.
        self.dispatcher._records['old'] = dict(state='accepted', session_id='s0',
                                               deadline=self.clock.now + LEASE)
        with patch.object(self.dispatcher, '_cleanup') as cleanup:
            self.clock.now += LEASE
            self.dispatcher.sweep()
        cleanup.assert_called_once_with('old')


if __name__ == '__main__':
    unittest.main()
