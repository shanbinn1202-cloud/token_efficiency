"""CPU fault injection for the real Goal lifecycle and private byte protocol."""
import ast
import asyncio
import copy
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'selective-prefix/source-overlay'
sys.path.insert(0, str(SOURCE))
from registered_add_protocol import Ack, Cancel, Status, ProtocolCodec, RegisteredAddDispatcher
from remote_ra_live_v2.idle_release import IdleRelease, CapacityExceeded
from remote_ra_live_v2.capacity import CapacityWait
from remote_ra_linux.native_authority import monitor_grace_seconds, NativeEngine
from registered_add_bridge import FrontendBridge


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def completion_method():
    # Run the production method without importing a vLLM/GPU serving application.
    tree = ast.parse((SOURCE/'serving_controller.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ServingController')
    method = next(x for x in cls.body if isinstance(x, ast.AsyncFunctionDef) and x.name == 'generation_closed')
    env = dict(asyncio=asyncio, time=time, Ack=Ack, Status=Status,
               sha256=hashlib.sha256, require=require)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE/'serving_controller.py'), 'exec'), env)
    return env['generation_closed']


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.codec = ProtocolCodec(key=b'k'*32, epoch='test-epoch', identity='a'*64)
        self.dispatcher = RegisteredAddDispatcher(codec=self.codec, boundary=NS(),
            decode_core=lambda x:x, preprocess=lambda x:x)
        self.dispatcher.sweep = lambda: None

    def test_status_does_not_cancel_or_mutate_admission(self):
        records = self.dispatcher._records
        records['request'] = dict(state='accepted', session_id='session')
        before = copy.deepcopy(records)
        frame = self.codec.encode(Status('status-request', 'request'))
        for _ in range(3):
            ack = self.codec.decode(self.dispatcher.handle(frame))
            self.assertEqual(ack.state, 'accepted')
            self.assertEqual(records, before)
            self.assertEqual(self.dispatcher._command_ids, {})
        records['request']['state'] = 'complete'
        self.assertEqual(self.codec.decode(self.dispatcher.handle(frame)).state, 'complete')

    def test_cancelled_is_reported_as_cancelled(self):
        self.dispatcher._records['request'] = dict(state='cancelled')
        ack = self.codec.decode(self.dispatcher.handle(self.codec.encode(Status('s', 'request'))))
        self.assertEqual(ack.state, 'cancelled')

    def test_foreign_status_rejected_before_mutation(self):
        foreign = ProtocolCodec(key=b'x'*32, epoch='test-epoch', identity='a'*64)
        with self.assertRaises(Exception):
            self.dispatcher.handle(foreign.encode(Status('s', 'request')))
        self.assertEqual(self.dispatcher._records, {})


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, states):
        codec = ProtocolCodec(key=b'k'*32, epoch='test', identity='a'*64)
        commands = []
        submitted = {'request': {'assigned': NS(external_id='external')}}
        call = NS(internal_id='request', external_id='external', terminal=True, engine_complete=False)
        async def exchange(frame, timeout):
            command = codec.decode(frame)
            commands.append(command)
            self.assertIs(type(command), Status)
            state = states.pop(0)
            return codec.encode(Ack(command.command_id, command.request_id, state, {}, ''))
        llm = NS(engine_core=NS(registered_add_async=exchange))
        registry = NS(llm=llm, codec=codec, bridge=NS(timeout=1), _submitted=submitted)
        controller = NS(registry=registry, current=lambda:call)
        await completion_method()(controller, llm, NS(request_id='request'), True)
        return call, submitted, commands

    async def test_delayed_native_completion_remains_successful(self):
        call, submitted, commands = await self.exercise(['accepted', 'accepted', 'complete'])
        self.assertTrue(call.engine_complete)
        self.assertEqual(submitted, {})
        self.assertEqual(len(commands), 3)

    async def test_real_cancellation_cannot_become_success(self):
        with self.assertRaisesRegex(RuntimeError, 'cancelled is not success'):
            await self.exercise(['cancelled'])

    async def test_expired_confirmation_is_bounded(self):
        codec = ProtocolCodec(key=b'k'*32, epoch='test', identity='a'*64)
        call = NS(internal_id='request', external_id='external', terminal=True, engine_complete=False)
        async def exchange(frame, timeout):
            command = codec.decode(frame)
            await asyncio.sleep(.01)
            return codec.encode(Ack(command.command_id, command.request_id, 'accepted', {}, ''))
        llm = NS(engine_core=NS(registered_add_async=exchange))
        c = NS(current=lambda:call, registry=NS(llm=llm, codec=codec,
            bridge=NS(timeout=.02), _submitted={'request':{'assigned':NS(external_id='external')}}))
        with self.assertRaisesRegex(RuntimeError, 'confirmation timed out'):
            await completion_method()(c, llm, NS(request_id='request'), True)
        self.assertFalse(call.engine_complete)
        self.assertIn('request', c.registry._submitted)


class CapacityTests(unittest.TestCase):
    def release(self):
        own = NS(resident=tuple(range(2304)), tail=None, poisoned=False)
        other = NS(resident=tuple(range(2323)), tail=None, poisoned=True)
        adapter = NS(window=object(), owner=NS(pool=NS(num_gpu_blocks=6088)),
                     sessions={'own':own, 'other':other})
        release = IdleRelease(NS(_remote_ra_growing_adapter=adapter))
        release.groups = lambda:1
        return release, adapter

    def test_other_session_close_unblocks_feasible_request(self):
        release, adapter = self.release()
        need = dict(phase1=2553, phase2=2553, watermark=0)
        with self.assertRaises(CapacityWait):
            release.ceiling(NS(session_id='own'), need)
        self.assertFalse(adapter.sessions['own'].poisoned)
        del adapter.sessions['other']
        release.ceiling(NS(session_id='own'), need)

    def test_intrinsically_oversized_turn_still_rejected(self):
        release, _ = self.release()
        with self.assertRaises(CapacityExceeded):
            release.ceiling(NS(session_id='own'), dict(phase1=7000, phase2=7000, watermark=0))


class HeartbeatTests(unittest.TestCase):
    def test_control_receipts_cover_overlapping_admission(self):
        bridge = FrontendBridge.__new__(FrontendBridge)
        bridge.timeout, bridge.admission_timeout = 60, 2100
        self.assertEqual(bridge.control_timeout, 2100)
        bridge.admission_timeout = 300
        self.assertEqual(bridge.control_timeout, 60)

    def test_grace_covers_existing_admission_lease(self):
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), patch(
                'remote_ra_http.config.load', return_value={'admission':{
                    'admission_overlap':{'lease_seconds':2160}}}):
            self.assertEqual(monitor_grace_seconds(), 2220)

    def test_stalled_engine_still_revokes(self):
        authority = NativeEngine.__new__(NativeEngine)
        authority.parent = NS(is_alive=lambda:True)
        authority.deadline = 2220
        authority.revoked = threading.Event()
        authority.stop = NS(wait=lambda delay:delay == 5)
        with patch('remote_ra_linux.native_authority.time.monotonic', return_value=2221), patch(
                'remote_ra_linux.native_authority.failure') as failure:
            authority._watch()
        self.assertTrue(authority.revoked.is_set())
        failure.assert_called_once()


if __name__ == '__main__':
    unittest.main()
