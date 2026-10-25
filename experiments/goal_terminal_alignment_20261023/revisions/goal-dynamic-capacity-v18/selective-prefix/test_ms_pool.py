"""Session pool (branch ms-pool): many registered tasks, at most K open sessions. CPU only.

A server started with ``serve.py --max-running-sessions K`` registers up to 64
startup tasks (pinned exactly like a fixed task list) and keeps at most K
sessions open; a session opens at its task's first request and the harness
closes it when its agent is done (POST /v1/sessions/close), so one server runs
e.g. 32 SWE-bench tasks with 16 concurrent sessions, refilled as agents finish.
Nothing here is CUDA evidence. Covered:

- startup: the bounds (64 registered tasks, 1 <= K <= min(16, tasks)), the
  launcher (max_num_seqs = K, a Mamba pool of 2K + 1 slots, the attention-pool
  check for K sessions, the pinned ``max_running_sessions``), the startup config
  check, the request bounds sized from the registered tasks; without the option
  nothing changes;
- the private protocol's CloseSession frame and the dispatcher;
- the frontend controller: FIFO open wait, refill order, cancelled waiters,
  refused closes (busy, unknown task, no pool), idempotent closes, shutdown;
- the HTTP close route: loopback only, status codes;
- the worker: a close drops the CPU catalog (its arena slots come back, and
  refilled sessions reuse them without growing the pinned chunks), the carry,
  and refuses the closed session for good;
- the Goal state of a closed session released behind a Goal still on the Goal thread;
- end to end on the engine (test_decouple_e2e's batched engine and worker): 32
  registered tasks with K = 16 (and 10 tasks with K = 4 at 16-token Pages, and 6
  with K = 2), sessions finishing at different times, refill in registration
  order, the pools returning by exactly each closed session's pins and to their
  baselines at the end, at most K sessions open and decoding, per-session
  results equal to a fixed-session server's, refusals of closed, unregistered
  and over-K sessions, a busy close refused, a poisoned session closed and its
  slot refilled, the capacity gate counting only open sessions' pins, the
  evidence (sessions.jsonl, engine-cleanup.json);
- end to end through HTTP (test_ms2_e2e's API server, frontend, signed frames,
  engine loop): 32 tasks served by a pool of 16 agents that close their
  sessions as they finish, an open wait and its FIFO order, refusals, failure
  isolation and shutdown with sessions open.
"""
import asyncio
from collections import deque
from contextlib import ExitStack
import contextlib
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL

from registered_add_protocol import (CLOSE_PREFIX, Ack, CloseSession, ProtocolCodec,
                                     RegisteredAddDispatcher)
from remote_ra_http import config as http_config
from remote_ra_live_v2 import capacity
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import sessions
from remote_ra_live_v2.capacity import CapacityWait
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.lifecycle import EngineLifecycle

try:
    import torch
except ImportError:
    torch = None

GIB = 1024 ** 3
K16, T32 = 16, 32


def tasks(count, start=0):
    return [dict(instance_id='repo__p%02d-%d' % (k, 200 + k),
                 problem_statement='Repair the flaky step %02d of pipeline p%02d.' % (k, k))
            for k in range(start, start + count)]


def close_command(sid):
    return CloseSession('close-' + sid, CLOSE_PREFIX + sid, sid)


# ---- startup: bounds, launcher, config -------------------------------------------------------

class BoundTests(unittest.TestCase):
    def test_registered_tasks_running_sessions_and_ids(self):
        self.assertEqual((sessions.MAX_SESSIONS, sessions.MAX_TASKS), (16, 64))
        self.assertEqual(sessions.validate_tasks(tasks(64), sessions.MAX_TASKS), tasks(64))
        with self.assertRaisesRegex(RuntimeError, r'startup task list of 1\.\.64 tasks required'):
            sessions.validate_tasks(tasks(65), sessions.MAX_TASKS)
        # The default is unchanged: 16 tasks, ids and request bounds of 16 sessions.
        with self.assertRaisesRegex(RuntimeError, r'1\.\.16'):
            sessions.validate_tasks(tasks(17))
        for call in (lambda: sessions.session_ids('s', 17), lambda: sessions.request_limit(17)):
            with self.assertRaises(RuntimeError):
                call()
        self.assertEqual((sessions.task_limit(False), sessions.task_limit(True)), (16, 64))
        ids = sessions.session_ids('http-x', T32, sessions.MAX_TASKS)
        self.assertEqual((ids[0], ids[31], len(set(ids))), ('http-x', 'http-x-s31', 32))
        self.assertEqual(sessions.session_index('http-x', T32, 'http-x-s20', sessions.MAX_TASKS), 20)
        self.assertEqual([sessions.request_limit(n, sessions.MAX_TASKS) for n in (8, 16, 32, 64)],
                         [2048, 4096, 8192, 16384])
        self.assertIsNone(sessions.running_limit(None, tasks(3)))
        self.assertEqual([sessions.running_limit(k, tasks(T32)) for k in (1, 16)], [1, 16])
        self.assertEqual(sessions.running_limit(3, tasks(3)), 3)
        for bad, listed in ((0, tasks(3)), (17, tasks(T32)), (4, tasks(3)), (True, tasks(3)), (2.0, tasks(3))):
            with self.assertRaisesRegex(RuntimeError, 'max_running_sessions must be 1..16'):
                sessions.running_limit(bad, listed)
        # Routing is by the task message, whatever the number of registered tasks.
        listed = tasks(64)
        messages = [dict(role='system', content='s'), dict(role='user', content=listed[47]['problem_statement'])]
        self.assertEqual(sessions.route(listed, messages), 47)

    def test_lifetime_request_bounds_from_the_registered_tasks(self):
        from remote_ra_live_v2.composition import engine_lifecycle
        from remote_ra_live_v2.registered_route import bind_registered_dispatcher
        for pool, count, limit, most in ((16, 32, 8192, 32), (4, 10, 2560, 10), (None, 16, 4096, 16),
                                         (None, 3, 1024, 16)):
            with self.subTest(pool=pool, tasks=count):
                core = NS(_remote_ra_trusted_issues=tasks(count), _remote_ra_session_pool=pool)
                lifecycle = engine_lifecycle(core, NS())
                self.assertEqual((lifecycle.max_requests, lifecycle.max_sessions), (limit, most))
                dispatcher = RegisteredAddDispatcher(codec=None, boundary=None, decode_core=lambda raw: raw,
                                                     preprocess=lambda request: request)
                bind_registered_dispatcher(dispatcher, lifecycle)
                self.assertEqual(dispatcher._max_records, limit)

    def test_lifecycle_retires_closed_sessions(self):
        lifecycle = EngineLifecycle(NS(), max_sessions=4)
        lifecycle.sessions.update(a=(3, None), b=(1, 'poisoned'), c=(2, 'r9'))
        lifecycle.retire_session('a')
        lifecycle.retire_session('b')
        lifecycle.retire_session('never')
        with self.assertRaisesRegex(RuntimeError, 'SESSION_BUSY'):
            lifecycle.retire_session('c')
        self.assertEqual(lifecycle.sessions, dict(a=(3, 'closed'), b=(1, 'closed'), c=(2, 'r9'),
                                                  never=(0, 'closed')))


class LauncherTests(unittest.TestCase):
    """serve.py's dry run (the release fixture of test_installer) and the config check."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def prepare(self, task, policy='full', kv_cache_gib=8, page_tokens=784, layout=None, selection=None,
                **pool):
        type(self).count += 1
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.count), 8231,
                                 selection, 'mswea-xml', kv_cache_gib, page_tokens, layout, **pool)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                return http_config.load()

    def effective(self, config):
        with patch.object(http_config, 'load', return_value=config):
            return http_config.effective_server_config(NS(**config['effective_server_config']))

    def test_32_tasks_16_running_in_the_shared_pool(self):
        listed = tasks(T32)
        _, config, argv, _ = self.prepare(listed, max_running=K16)
        effective = config['effective_server_config']
        self.assertEqual((effective['max_num_seqs'], argv[argv.index('--max-num-seqs') + 1]), (16, '16'))
        self.assertEqual((config['tasks'], config['task'], config['max_running_sessions']),
                         (listed, listed[0], 16))
        self.assertEqual(config['tasks_sha256'], serve.pin(listed))
        self.assertNotIn('kv_layout', config)
        loaded = self.load(config)
        self.assertEqual((loaded['tasks'], loaded['max_running_sessions'], http_config.running_sessions(loaded)),
                         (listed, 16, 16))
        self.assertEqual(self.effective(config), effective)
        # One running sequence per open session: 32 (one per task) is refused.
        pinned = dict(config, effective_server_config=dict(effective, max_num_seqs=T32))
        pinned['server_config_sha256'] = serve.pin(pinned['effective_server_config'])
        with self.assertRaisesRegex(RuntimeError, 'one sequence per startup session'):
            self.effective(pinned)

    def test_32_tasks_16_running_decoupled(self):
        _, config, argv, _ = self.prepare(tasks(T32), 'goal_query', 24, 16, selection=dict(no_goal_rewriter=True),
                                          max_running=K16)
        record = dict(page_tokens=16, mamba_tokens=784, mamba_slots=33)     # K = 2 x 16 + 1
        self.assertEqual(config['kv_layout'], record)
        self.assertEqual(argv[-6:], ['--block-size', '16', '--mamba-block-size', '784', '--additional-config',
                                     json.dumps({'remote_ra_kv_layout': record}, sort_keys=True,
                                                separators=(',', ':'))])
        layout = I0.for_server(16, K16, True)
        self.assertEqual(http_config.kv_layout(config), layout)
        self.assertEqual(self.load(config)['kv_layout'], record)
        # The attention pool is the one of a 16-session server: the Mamba pool follows K.
        self.assertEqual(serve.attention_blocks(24 * GIB, layout), 19682)
        self.effective(config)
        # The attention-pool refusal counts K sessions, not the registered tasks.
        self.prepare(tasks(T32), kv_cache_gib=8, page_tokens=16, max_running=9)
        with self.assertRaisesRegex(ValueError, 'with 10 sessions leaves 5059 usable attention blocks'):
            self.prepare(tasks(T32), kv_cache_gib=8, page_tokens=16, max_running=10)

    def test_refusals(self):
        for listed, running, error in ((tasks(T32), 17, RuntimeError), (tasks(3), 4, RuntimeError),
                                       (tasks(65), 16, RuntimeError), (tasks(1)[0], 1, ValueError),
                                       (tasks(3), 0, RuntimeError)):
            with self.subTest(tasks=len(listed) if type(listed) is list else 'one', running=running):
                with self.assertRaises(error):
                    self.prepare(listed, max_running=running)
        # Without the option the fixed bound is unchanged: 17 tasks are refused.
        with self.assertRaisesRegex(RuntimeError, r'1\.\.16'):
            self.prepare(tasks(17))
        # The config check refuses a pool without a list, beyond 16 or above the tasks.
        _, config, _, _ = self.prepare(tasks(4), max_running=2)
        for change in (dict(max_running_sessions=5), dict(max_running_sessions=0),
                       dict(max_running_sessions=None)):
            with self.subTest(change=change):
                with self.assertRaises(RuntimeError):
                    self.load(dict(config, **change))
        bare = {k: v for k, v in config.items() if k not in ('tasks', 'tasks_sha256')}
        with self.assertRaisesRegex(RuntimeError, 'a session pool needs a pinned task list'):
            self.load(bare)
        listed = tasks(17)
        with self.assertRaisesRegex(RuntimeError, r'1\.\.16'):    # 17 tasks need the pool option
            self.load(dict({k: v for k, v in config.items() if k != 'max_running_sessions'},
                           tasks=listed, task=listed[0], tasks_sha256=http_config.digest(listed)))

    def test_without_the_option_nothing_changes(self):
        for listed in (tasks(1)[0], tasks(3), tasks(16)):
            for P in (784, 16):
                with self.subTest(tasks=len(listed) if type(listed) is list else 1, P=P):
                    _, config, argv, env = self.prepare(listed, kv_cache_gib=16, page_tokens=P)
                    _, again, argv2, env2 = self.prepare(listed, kv_cache_gib=16, page_tokens=P, max_running=None)
                    self.assertNotIn('max_running_sessions', config)
                    self.assertEqual(('tasks' in config), type(listed) is list and len(listed) > 1)
                    swap = lambda value: json.loads(json.dumps(value).replace('out%d' % self.count,
                                                                              'out%d' % (self.count - 1)))
                    self.assertEqual(swap(again), config)
                    self.assertEqual(swap(argv2), argv)
                    self.assertEqual(http_config.running_sessions(self.load(config)),
                                     1 if type(listed) is dict else len(listed))
        # A one-task pool pins its one-task list and K = 1.
        _, config, _, _ = self.prepare(tasks(1), max_running=1)
        self.assertEqual((config['tasks'], config['max_running_sessions']), (tasks(1), 1))
        self.assertEqual(self.load(config)['tasks'], tasks(1))


# ---- the private protocol ------------------------------------------------------------------

class ProtocolTests(unittest.TestCase):
    def codec(self):
        return ProtocolCodec(key=b'k' * 32, epoch='e' * 64, identity='c' * 64)

    def test_close_frames_and_the_dispatcher(self):
        codec = self.codec()
        command = close_command('http-x-s3')
        self.assertEqual(codec.decode(codec.encode(command)), command)
        with self.assertRaisesRegex(Exception, 'session close correlation id required'):
            codec.encode(CloseSession('close-x', 'close:other', 'http-x-s3'))
        closes = []

        class Live:
            def close_session(self, dispatcher, command):
                closes.append(command.session_id)
                if command.session_id == 'busy':
                    raise LiveV2Error('SESSION_BUSY: session busy still owns a request')
                return Ack(command.command_id, command.request_id, 'complete', dict(closed=command.session_id), '')
        dispatcher = RegisteredAddDispatcher(codec=codec, boundary=None, decode_core=lambda raw: raw,
                                             preprocess=lambda request: request)
        with patch('registered_add_protocol.RegisteredAddDispatcher._log_rejection', lambda *a: None):
            # Without the live route a close is an authenticated refusal.
            ack = codec.decode(dispatcher.handle(codec.encode(command)))
            self.assertEqual((ack.state, ack.error), ('rejected', 'PolicyRuntimeAbort: session close requires '
                                                      'the live runtime'))
            dispatcher._remote_ra_live_v2 = Live()
            for _ in range(2):      # a retry of the same command reaches the live route again
                ack = codec.decode(dispatcher.handle(codec.encode(command)))
                self.assertEqual((ack.command_id, ack.request_id, ack.state, ack.receipt),
                                 (command.command_id, command.request_id, 'complete', dict(closed='http-x-s3')))
            busy = close_command('busy')
            ack = codec.decode(dispatcher.handle(codec.encode(busy)))
            self.assertEqual(ack.state, 'rejected')
            self.assertIn('SESSION_BUSY', ack.error)
        self.assertEqual(closes, ['http-x-s3', 'http-x-s3', 'busy'])
        # A close takes a command tombstone, never a request tombstone.
        self.assertEqual((dispatcher._records, set(dispatcher._command_ids)), ({}, {command.command_id, busy.command_id}))


# ---- end to end on the engine ----------------------------------------------------------------

def pool_harness_class():
    import test_decouple_e2e as DE

    class PoolHarness(DE.MultiHarness):
        """test_decouple_e2e's batched engine and worker on a session-pool server:
        every task registered, at most K sessions open. ``serve`` keeps K lanes
        active (the frontend's open slots), each opening at its first admission;
        a lane whose script is done is closed with an authenticated CloseSession
        through the production dispatcher, and the next registered task refills
        its slot. Each close is recorded with both pools before and after it."""

        def __init__(self, policy, output, case, tasks, limit, scripts):
            super().__init__(policy, output, case, tasks=tasks)
            from remote_ra_live_v2.registered_route import bind_registered_dispatcher
            self.route = bind_registered_dispatcher(self.dispatcher, self.lifecycle)
            self.limit = limit
            for lane, script in zip(self.lanes, scripts):
                lane.script = list(script)
                lane.length = len(script)
            self.opened, self.closes, self.max_open, self.open_lanes = [], [], 0, []

        def close_lane(self, lane):
            sid = lane.sid
            s = self.adapter.sessions.get(sid)
            namespace = None if s is None else s.namespace
            pins = 0 if s is None else len(self.adapter.owner.pinned(namespace))
            catalog = 0 if s is None else len(s.catalog)
            before = list(self.kv.get_num_free_blocks_by_pool())
            ack = self.dispatcher._close_session(close_command(sid))
            self.closes.append(NS(lane=lane, receipt=ack.receipt, before=before, pins=pins, namespace=namespace,
                                  catalog=catalog,
                                  after=list(self.kv.get_num_free_blocks_by_pool()), running=len(self.running),
                                  open=[l.sid for l in self.open_lanes if l is not lane]))
            return ack.receipt

        def serve(self, check=None, on_close=None, queue=None):
            queue, k = deque(self.lanes) if queue is None else queue, 0
            while queue or self.open_lanes or self.running:
                while queue and len(self.open_lanes) < self.limit:      # refill, in registration order
                    lane = queue.popleft()
                    self.open_lanes.append(lane)
                    self.opened.append(lane.index)
                busy = {info['lane'].index for info in self.running}
                for lane in [l for l in self.open_lanes if not l.script and l.index not in busy]:
                    self.flush()
                    self.close_lane(lane)
                    self.open_lanes.remove(lane)
                    if on_close is not None:
                        on_close(self, self.closes[-1])
                ready = [l for l in self.open_lanes if l.script and l.index not in busy]
                if ready:
                    self.admit(ready[k % len(ready)])
                    k += 1
                self.max_open = max(self.max_open, len(self.sessions.namespaces))
                if self.running:
                    before = len(self.done)
                    self.batch_step()
                    for info in self.done[before:]:
                        if check is not None:
                            check(self, info)
            self.flush()

    return PoolHarness


def scripts_for(count, long_lanes=3, turns=5):
    """Lane k's script: the first ``turns`` script turns for the first ``long_lanes``
    lanes (compared with a fixed-session server), else 1-5 turns, so that sessions
    finish at different times."""
    import test_decouple_e2e as DE
    return [list(DE.TURNS[:turns if k < long_lanes else 1 + (3 * k) % 5]) for k in range(count)]


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EnginePoolTests(unittest.TestCase):
    maxDiff = None

    def enter(self, stack, case, policy, listed, limit, pooled=True):
        import test_decouple_e2e as DE
        multi = DE.Case(**dict(vars(case), sessions=limit))
        DE.enter_case(stack, multi)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = DE.config(policy, output, multi, listed)
        if pooled:
            value['max_running_sessions'] = limit
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        return multi, output

    def check_close(self, h, record):
        """One close: exactly that session's pins come back, nothing else moves, and
        every engine and worker mirror of the session is gone."""
        r, lane = record.receipt, record.lane
        self.assertEqual((r['event'], r['reason'], r['session_id'], r['session_index'], r['instance_id']),
                         ('close', 'close', lane.sid, lane.index, lane.task['instance_id']))
        self.assertEqual((r['opened'], r['turns'], r['max_running_sessions']), (True, lane.length, h.limit))
        self.assertEqual(record.after[0] - record.before[0], record.pins)
        self.assertEqual(record.after[1:], record.before[1:])          # idle sessions pin no GDN slot
        self.assertEqual((r['released']['blocks'], r['free_blocks_by_pool_before'], r['free_blocks_by_pool']),
                         (record.pins, record.before, record.after))
        # The worker's CPU catalog is the engine's (every sealed Page of the session);
        # the worker drops each request's terminal receipt and phase-1 lease.
        rids = [rid for rid, entry in h.lifecycle.entries.items() if entry['identity'][1] == lane.sid]
        self.assertEqual(len(rids), lane.length)
        self.assertEqual(r['released']['worker'], dict(catalog_pages=record.catalog, carry=True,
                                                       terminals=lane.length, phase1_leases=lane.length - 1))
        self.assertEqual(r['released']['requests'], lane.length)
        # The engine keeps a slim tombstone per request and nothing else of them.
        from remote_ra_live_v2.session_pool import ReleasedBinding
        for rid in rids:
            entry = h.lifecycle.entries[rid]
            self.assertEqual((entry['state'], entry['request'], type(entry['binding'])),
                             ('complete', None, ReleasedBinding))
            self.assertEqual(set(entry['drain_receipt']), {'operation', 'binding', 'drained'})
            for table in (h.endpoint.claims, h.live.claims, h.live.prepared_payloads, h.boundary.prepared,
                          h.proxy.http_receipts, h.selection.authorized, h.owner.terminals):
                self.assertNotIn(rid, table)
            self.assertFalse(any(key[0] == rid for key in h.boundary.receipts))
            self.assertFalse(any(lease['payload']['request'] == rid for lease in h.driver.leases.values()))
            self.assertEqual(h.dispatcher._records[rid]['state'], 'complete')
        self.assertEqual(r['open_sessions'], len(h.sessions.namespaces))
        for table in (h.adapter.sessions, h.sessions.namespaces, h.backend.carry, h.owner.session_namespaces):
            self.assertNotIn(lane.sid, table)
        self.assertNotIn(record.namespace, h.owner.sessions)
        self.assertNotIn(record.namespace, h.adapter.owner.sessions)
        self.assertIn(lane.sid, h.owner.closed_sessions)
        self.assertEqual(h.lifecycle.sessions[lane.sid], (lane.length, 'closed'))
        self.assertEqual((h.endpoint.transcripts[lane.sid].stream, h.endpoint.transcripts[lane.sid].turns), ((), []))
        if h.policy in ('goal_query', 'observation_query'):
            for table in (h.selection.vector_catalog, h.selection.turn_ends, h.selection.page_index):
                self.assertNotIn(lane.sid, table)
            self.assertEqual(r['goal_state'], 'inline')
            self.assertNotIn(lane.sid, h.selection.constructor._sessions)
        if not record.running:
            # Nothing in flight: every open session's pins are all the pool lacks.
            pinned = sum(len(h.adapter.owner.pinned(h.adapter.sessions[sid].namespace))
                         for sid in record.open if sid in h.adapter.sessions)
            self.assertEqual(record.after[0], h.baseline - pinned)
            if h.case.decoupled:
                self.assertEqual(record.after[1], h.mamba_baseline)

    def run_pool(self, case, policy, listed, limit, blocks, scripts, check_turn=True):
        import test_decouple_e2e as DE
        PoolHarness = pool_harness_class()
        with ExitStack() as stack:
            multi, output = self.enter(stack, DE.Case(**dict(vars(case), blocks=blocks)), policy, listed, limit)
            h = PoolHarness(policy, output, multi, listed, limit, scripts)
            h.active = h.restores = 0           # MultiSessionMatrixTests.check counts these
            # Startup: every task registered, nothing open, the pools at their baselines.
            self.assertEqual((h.sessions.ids, h.sessions.limit, h.sessions.namespaces, h.adapter.sessions),
                             (tuple(lane.sid for lane in h.lanes), limit, {}, {}))
            limit_requests = sessions.request_limit(len(listed), sessions.MAX_TASKS)
            self.assertEqual((h.lifecycle.max_requests, h.lifecycle.max_sessions, h.authority.max_attempts,
                              h.dispatcher._max_records), (limit_requests, len(listed), limit_requests,
                                                           limit_requests))
            if case.decoupled:
                self.assertEqual(len(h.mamba_pool.blocks), 1 + 3 * (2 * limit + 1))
            check = (lambda h, info: DE.MultiSessionMatrixTests.check(self, h, info)) if check_turn else None
            h.serve(check, self.check_close)
            # Every turn of every task ran, no preemption (batch_step asserts it), never
            # more than K sessions open or decoding; refill in registration order while
            # sessions finished out of order.
            self.assertEqual(len(h.done), sum(len(s) for s in scripts))
            self.assertEqual(h.opened, list(range(len(listed))))
            self.assertEqual((h.max_open, max(h.widths)), (limit, limit))
            order = [r.lane.index for r in h.closes]
            self.assertEqual(sorted(order), list(range(len(listed))))
            self.assertNotEqual(order, sorted(order))
            # Everything released: both pools at their baselines before shutdown.
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))
            self.assertEqual((h.adapter.sessions, h.sessions.namespaces, h.owner.sessions, h.backend.carry,
                              h.adapter.owner.sessions, h.adapter.owner.tails), ({}, {}, {}, {}, {}, {}))
            self.assertEqual(set(h.lifecycle.sessions.values()),
                             {(lane.length, 'closed') for lane in h.lanes})
            # Evidence: one open and one close row per task, in the order they happened.
            rows = [json.loads(line) for line in (output / 'sessions.jsonl').read_bytes().splitlines()]
            self.assertEqual([(r['event'], r['session_id']) for r in rows if r['event'] == 'close'],
                             [('close', r.lane.sid) for r in h.closes])
            opens = [r for r in rows if r['event'] == 'open']
            # The engine opens a session at its first admission (the harness admits the
            # open lanes round robin), so within the open slots the order may differ.
            self.assertEqual(sorted(r['session_index'] for r in opens), list(range(len(listed))))
            self.assertTrue(all(1 <= r['open_sessions'] <= limit for r in opens))
            self.assertEqual({r['schema'] for r in rows}, {'remote-ra-session-event-v1'})
            # A closed task fails at once and changes nothing.
            lane = h.lanes[0]
            free = list(h.kv.get_num_free_blocks_by_pool())
            with self.assertRaisesRegex(LiveV2Error, 'SESSION_CLOSED'):
                h.route.add(h.dispatcher, h.command(lane, [1, 2, 3], lane.messages), 'fp-closed')
            self.assertEqual((list(h.kv.get_num_free_blocks_by_pool()), h.sessions.namespaces), (free, {}))
            # Closing it again returns its receipt.
            again = h.dispatcher._close_session(close_command(lane.sid)).receipt
            self.assertEqual(again, dict(h.closes[0].receipt if h.closes[0].lane is lane else
                                         next(r.receipt for r in h.closes if r.lane is lane), repeated=True))
            h.close()
            DE.engine_closed(h.core)
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            ids = [lane.sid for lane in h.lanes]
            self.assertEqual(cleanup['sessions'], [])
            self.assertEqual(cleanup['session_pool'], dict(max_running_sessions=limit, registered=len(listed),
                                                           closed=ids, closed_unopened=[], open_at_shutdown=[],
                                                           never_opened=[]))
            return {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'])
                                 for i in h.done if i['lane'] is lane] for lane in h.lanes}, h

    def run_fixed(self, case, policy, listed, blocks, scripts):
        """The same tasks on a fixed-session server (every session open at startup)."""
        import test_decouple_e2e as DE
        with ExitStack() as stack:
            multi, output = self.enter(stack, DE.Case(**dict(vars(case), blocks=blocks)), policy, listed,
                                       len(listed), pooled=False)
            h = DE.MultiHarness(policy, output, multi, tasks=listed)
            self.assertIsNone(h.sessions.limit)
            self.assertEqual(len(h.adapter.sessions), len(listed))     # all open at startup, as before
            for lane, script in zip(h.lanes, scripts):
                lane.script = list(script)
            h.serve()
            h.close()
            return {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'])
                                 for i in h.done if i['lane'] is lane] for lane in h.lanes}

    # The slow pass (README section 2.3); test_10_tasks_4_running_at_16_token_pages runs in
    # the default suite.
    @unittest.skipUnless(os.environ.get('REMOTE_RA_SLOW_TESTS') == '1',
                         'about 20 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_32_tasks_16_running_sessions(self):
        import test_decouple_e2e as DE
        listed = list(DE.TASKS) + tasks(T32 - len(DE.TASKS))
        scripts = scripts_for(T32)
        pool, h = self.run_pool(DE.CASES[0], 'full', listed, K16, 16 * 12, scripts)
        fixed = self.run_fixed(DE.CASES[0], 'full', list(DE.TASKS), 16 * 12, scripts[:3])
        for k in range(len(DE.TASKS)):
            self.assertEqual(pool[k], fixed[k])

    def test_10_tasks_4_running_at_16_token_pages(self):
        import test_decouple_e2e as DE
        listed = list(DE.TASKS) + tasks(10 - len(DE.TASKS))
        scripts = scripts_for(10)
        pool, h = self.run_pool(DE.CASES[2], 'goal_query', listed, 4, 4 * 400, scripts)
        self.assertGreater(h.active, 0)          # selection was active (MultiSessionMatrixTests.check)
        fixed = self.run_fixed(DE.CASES[2], 'goal_query', list(DE.TASKS), 4 * 400, scripts[:3])
        for k in range(len(DE.TASKS)):
            self.assertEqual(pool[k], fixed[k])

    def test_six_tasks_two_running_refusals_and_isolation(self):
        """K = 2 of 6 tasks: a session beyond K, a busy close, an unregistered id and a
        closed task are refused without changing anything; a never-opened task closes
        without touching the engine; a session poisoned by a cancel mid-decode is closed
        (its pins come back) and its slot refills, while the others run as on a
        fixed-session server."""
        import test_decouple_e2e as DE
        PoolHarness = pool_harness_class()
        listed = list(DE.TASKS) + tasks(3)
        scripts = [list(DE.TURNS[:3]) for _ in range(6)]
        with ExitStack() as stack:
            case = DE.Case(**dict(vars(DE.CASES[0]), blocks=40))
            multi, output = self.enter(stack, case, 'random50', listed, 2)
            h = PoolHarness('random50', output, multi, listed, 2, scripts)
            h.active = h.restores = 0
            a, b, c, d, e, f = h.lanes
            first_a, first_b = h.admit(a), h.admit(b)
            self.assertEqual(list(h.sessions.namespaces), [a.sid, b.sid])

            def unchanged(action, error):
                state = (list(h.kv.get_num_free_blocks_by_pool()), dict(h.sessions.namespaces),
                         set(h.adapter.sessions), dict(h.lifecycle.sessions))
                with self.assertRaisesRegex(LiveV2Error, error):
                    action()
                self.assertEqual(state, (list(h.kv.get_num_free_blocks_by_pool()), dict(h.sessions.namespaces),
                                         set(h.adapter.sessions), dict(h.lifecycle.sessions)))
            # A third session is refused on the engine (the frontend never sends it).
            tokens = c.frontend.render(c.messages)[0]
            command = h.command(c, tokens, c.messages)
            unchanged(lambda: h.route.add(h.dispatcher, command, 'fp-c'), 'SESSION_LIMIT: 2 of 2 sessions open')
            c.frontend.abandon()
            self.assertEqual(h.dispatcher._records[command.request_id]['state'], 'cancelled')
            unchanged(lambda: h.dispatcher._close_session(close_command(a.sid)),
                      'SESSION_BUSY: session %s still owns a request' % a.sid)
            unchanged(lambda: h.dispatcher._close_session(close_command(a.sid + '-x')),
                      'session not initialized by loaded server')
            # A task that never opened closes without touching the engine, for good.
            receipt = h.dispatcher._close_session(close_command(f.sid)).receipt
            self.assertEqual({k: receipt[k] for k in ('opened', 'open_sessions', 'session_index')},
                             dict(opened=False, open_sessions=2, session_index=5))
            self.assertNotIn('released', receipt)
            f.script = []
            # Session b is cancelled mid-decode: poisoned, its native work drained.
            h.batch_step()
            self.assertIn(first_b, h.running)
            h.route.cleanup(h.dispatcher, first_b['rid'])
            h.running.remove(first_b)
            self.assertTrue(h.adapter.sessions[b.sid].poisoned)
            self.assertEqual(h.lifecycle.sessions[b.sid], (0, 'poisoned'))
            while first_a in h.running:                     # a goes on alone
                h.batch_step()
            DE.MultiSessionMatrixTests.check(self, h, first_a)
            h.flush()
            pins = len(h.adapter.owner.pinned(h.adapter.sessions[b.sid].namespace))
            free = h.pool.get_num_free_blocks()
            receipt = h.dispatcher._close_session(close_command(b.sid)).receipt
            self.assertEqual((receipt['opened'], receipt['poisoned'], receipt['turns'], receipt['open_sessions']),
                             (True, True, 0, 1))
            self.assertEqual(h.pool.get_num_free_blocks() - free, pins)
            b.script = []
            unchanged(lambda: h.route.add(h.dispatcher, h.command(b, [1, 2, 3], b.messages), 'fp-b'),
                      'SESSION_CLOSED')
            # The slot refills: c, then d and e, in registration order.
            h.open_lanes = [a]
            h.serve(lambda h, info: DE.MultiSessionMatrixTests.check(self, h, info), self.check_close,
                    queue=deque([c, d, e]))
            self.assertEqual(h.opened, [c.index, d.index, e.index])
            self.assertEqual((h.pool.get_num_free_blocks(), h.max_open), (h.baseline, 2))
            pool = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'])
                                 for i in h.done if i['lane'] is lane]
                    for lane in (a, c)}
            h.close()
            DE.engine_closed(h.core)
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual(cleanup['session_pool'], dict(
                max_running_sessions=2, registered=6, closed=[a.sid, b.sid, c.sid, d.sid, e.sid],
                closed_unopened=[f.sid], open_at_shutdown=[], never_opened=[]))
            rows = [json.loads(line) for line in (output / 'sessions.jsonl').read_bytes().splitlines()]
            events = [(r['event'], r['session_index']) for r in rows]
            self.assertEqual(events[:4], [('open', 0), ('open', 1), ('close', 5), ('close', 1)])
            self.assertEqual(sorted(events), sorted([('open', k) for k in range(5)] +
                                                    [('close', k) for k in range(6)]))
        fixed = self.run_fixed(DE.CASES[0], 'random50', list(DE.TASKS), 40, scripts[:3])
        self.assertEqual((pool[0], pool[2]), (fixed[0], fixed[2]))

    def test_capacity_gate_counts_only_open_sessions(self):
        """With a request running, an opening session's turn 0 that the free blocks
        cannot cover waits (CAPACITY_WAIT); closing an idle session returns its pins
        and the same turn is admitted."""
        import test_decouple_e2e as DE
        PoolHarness = pool_harness_class()
        listed = list(DE.TASKS) + tasks(1)
        scripts = [list(DE.TURNS[:2])] + [list(DE.TURNS[:1])] * 3
        with ExitStack() as stack:
            case = DE.Case(**dict(vars(DE.CASES[2]), blocks=3000))
            multi, output = self.enter(stack, case, 'full', listed, 3)
            h = PoolHarness('full', output, multi, listed, 3, scripts)
            a, b, c, _ = h.lanes
            for _ in range(2):
                info = h.admit(a)
                while info in h.running:
                    h.batch_step()
            h.flush()
            pins = len(h.adapter.owner.pinned(h.adapter.sessions[a.sid].namespace))
            self.assertGreater(pins, 0)
            running = h.admit(b)
            h.batch_step()
            tokens = c.frontend.render(c.messages)[0]
            command = h.command(c, tokens, c.messages)
            h.kv.watermark_blocks = 3
            self.assertTrue(capacity.opening(h.core, c.sid))
            need = capacity.assess(h.lifecycle, command)
            att, mam, _ = h.endpoint.phase1.groups()
            end = -(-(len(tokens) + 1536) // case.P)
            self.assertEqual((need['phase1'], need['phase2']), ((0, 0), (len(att) * end, 2 * len(mam))))
            exact = need['phase2'][0] + need['headroom'][0] + need['watermark'][0]
            hog = h.pool.get_new_blocks(h.pool.get_num_free_blocks() - (exact - 1))
            with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: FA pool: .* %d free' % (exact - 1)):
                capacity.admission_gate(h.lifecycle, command)
            receipt = h.dispatcher._close_session(close_command(a.sid)).receipt
            self.assertEqual(receipt['released']['blocks'], pins)
            self.assertEqual(capacity.admission_gate(h.lifecycle, command)['free'][0], exact - 1 + pins)
            self.assertFalse(capacity.opening(h.core, a.sid))     # a closed session never opens again
            h.pool.free_blocks(hog)
            c.frontend.abandon()
            while running in h.running:
                h.batch_step()
            h.flush()
            h.dispatcher._close_session(close_command(b.sid))
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))
            h.close()
            DE.engine_closed(h.core)
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual(cleanup['session_pool']['never_opened'], [c.sid, h.lanes[3].sid])
            self.assertEqual(cleanup['kv_pools']['free_blocks_by_pool'], [h.baseline, h.mamba_baseline])


# ---- the frontend controller and the HTTP close route -------------------------------------------

def first_messages(task):
    return [dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content='<pr_description>\n%s\n</pr_description>' % task['problem_statement'])]


class FakeEngineClose:
    """The serving controller's close seam: the engine's receipt, or its refusal."""

    def __init__(self):
        self.closes, self.refuse = [], {}

    async def close_session(self, session_id):
        self.closes.append(session_id)
        await asyncio.sleep(0)
        if session_id in self.refuse:
            raise RuntimeError('session close rejected: ' + self.refuse.pop(session_id))
        return dict(event='close', session_id=session_id, opened=True)

    async def shutdown_close(self, timeout=None):
        pass


def pool_controller(count=4, limit=2, pooled=True):
    import test_mswea_protocol as X
    from remote_ra_live_v2.controller import LiveFourPolicyController
    listed = tasks(count)
    return LiveFourPolicyController(FakeEngineClose(), startup_policy='full', instance_id='http-x',
        readiness=NS(require_native=lambda: None), session_id='http-x', tasks=listed,
        transcripts=tuple(X.transcript(allow_rebuild=True) for _ in listed), max_num_seqs=limit,
        max_running=limit if pooled else None), listed


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_open_wait_fifo_refill_and_closes(self):
        live, listed = pool_controller(5, 2)
        self.assertEqual((live.pooled, live.limit, live.open_count()), (True, 2, 0))
        self.assertTrue(all(not s.opened for s in live.sessions))
        requests = [NS(name=k) for k in range(5)]
        for k in range(5):
            live.reserve(requests[k], first_messages(listed[k]))
        self.assertEqual([await live.open_wait(requests[k]) for k in (0, 1)], [0.0, 0.0])
        waiting = [asyncio.create_task(live.open_wait(requests[k])) for k in (2, 3, 4)]
        await asyncio.sleep(0)
        self.assertEqual(([w.done() for w in waiting], len(live._opening), live.open_count()),
                         ([False] * 3, 3, 2))
        # A request waiting to open that goes away takes no slot and leaves its task
        # registered: nothing reached the engine.
        waiting[2].cancel()
        await asyncio.gather(waiting[2], return_exceptions=True)
        live.failed(requests[4])
        live.release(requests[4])
        self.assertEqual((live.sessions[4].opened, live.sessions[4].closed, len(live._opening)), (False, False, 2))
        # Closing task 1 opens the oldest waiter (task 2), not task 3.
        live.release(requests[1])
        receipt = await live.close_session(listed[1]['instance_id'])
        await asyncio.sleep(0)
        self.assertEqual(receipt, dict(event='close', session_id='http-x-s1', opened=True))
        self.assertEqual(([w.done() for w in waiting[:2]], live.open_count()), ([True, False], 2))
        self.assertIsInstance(waiting[0].result(), float)
        s1 = live.sessions[1]
        self.assertEqual((s1.closed, s1.ended, s1.owner, s1.transcript.stream), (True, True, None, ()))
        with self.assertRaisesRegex(RuntimeError, 'session http-x-s1 closed by the harness'):
            live.reserve(NS(), first_messages(listed[1]))
        # Closing again returns the receipt without another engine close.
        self.assertEqual(await live.close_session(listed[1]['instance_id']), dict(receipt, repeated=True))
        self.assertEqual(live.serving.closes, ['http-x-s1'])
        # Refused closes change nothing: busy (task 0 still holds its slot), unknown,
        # and an engine refusal.
        for instance, error in ((listed[0]['instance_id'], 'SESSION_BUSY: session http-x has a request'),
                                ('repo__nope', 'SESSION_UNKNOWN')):
            with self.assertRaisesRegex(RuntimeError, error):
                await live.close_session(instance)
        live.release(requests[0])
        live.serving.refuse['http-x'] = 'SESSION_BUSY: session http-x still owns native work'
        with self.assertRaisesRegex(RuntimeError, 'session close rejected: SESSION_BUSY'):
            await live.close_session(listed[0]['instance_id'])
        s0 = live.sessions[0]
        self.assertEqual((s0.closed, s0.ended, s0.owner, s0.opened, live.open_count()), (False, False, None, True, 2))
        # Granted, then cancelled before it ran: the open slot passes on.
        await live.close_session(listed[0]['instance_id'])
        self.assertTrue(live.sessions[3].opened)
        waiting[1].cancel()
        await asyncio.gather(waiting[1], return_exceptions=True)
        self.assertFalse(live.sessions[3].opened)
        live.release(requests[3])
        # Task 4 (whose earlier request went away) opens now, task 3 again later.
        again = NS(name='again')
        live.reserve(again, first_messages(listed[4]))
        self.assertEqual((await live.open_wait(again), live.open_count()), (0.0, 2))
        late = NS(name='late')
        live.reserve(late, first_messages(listed[3]))
        pending = asyncio.create_task(live.open_wait(late))
        await asyncio.sleep(0)
        await live.close()          # shutdown: nothing opens any more
        with self.assertRaisesRegex(RuntimeError, 'server shutting down'):
            await pending

    async def test_without_a_pool(self):
        from remote_ra_live_v2.controller import open_request
        live, listed = pool_controller(3, 3, pooled=False)
        self.assertEqual((live.pooled, live.open_count()), (False, 3))
        self.assertTrue(all(s.opened and not s.ended for s in live.sessions))
        request = NS()
        live.reserve(request, first_messages(listed[1]))
        self.assertEqual(await live.open_wait(request), 0.0)
        self.assertIsNone(await open_request(NS(_remote_ra_live_four_policy=live), request))
        with self.assertRaisesRegex(RuntimeError, 'SESSION_POOL_REQUIRED'):
            await live.close_session(listed[2]['instance_id'])
        # A failure still closes its session, as before.
        live.failed(request)
        self.assertTrue(live.sessions[1].closed)
        with self.assertRaisesRegex(RuntimeError, 'one sequence per open session'):
            from remote_ra_live_v2.controller import LiveFourPolicyController
            LiveFourPolicyController(FakeEngineClose(), startup_policy='full', instance_id='http-x',
                                     readiness=NS(require_native=lambda: None), session_id='http-x',
                                     tasks=tasks(4), max_num_seqs=4, max_running=2)

    async def test_close_route(self):
        import route_runtime as route
        import test_http as H
        live, listed = pool_controller(3, 2)
        fixed, _ = pool_controller(3, 3, pooled=False)
        with H.actual_router() as (router, _), patch.object(route, 'require_installable', lambda: None):
            async def close(host='127.0.0.1', body=None, controller=live, dynamic=True):
                async def payload():
                    if body == 'garbage':
                        raise ValueError('not JSON')
                    return dict(instance_id=listed[1]['instance_id']) if body is None else body
                state = NS(engine_client=NS(_remote_ra_dynamic_mode=dynamic),
                           openai_serving_chat=NS(_remote_ra_live_four_policy=controller))
                response = await router.close_session(NS(app=NS(state=state), client=NS(host=host), json=payload))
                return getattr(response, 'status_code', 200), response.content
            status, content = await close()
            self.assertEqual((status, content), (200, dict(event='close', session_id='http-x-s1', opened=True,
                                                           object='session.close')))
            self.assertEqual((await close())[1]['repeated'], True)
            busy = NS()
            live.reserve(busy, first_messages(listed[0]))
            for kwargs, code in ((dict(host='10.1.2.3'), 403), (dict(host='::1', body=dict(instance_id='x')), 404),
                                 (dict(body=dict(instance_id=listed[0]['instance_id'])), 409),
                                 (dict(body=dict(task='x')), 400), (dict(body='garbage'), 400),
                                 (dict(body=dict(instance_id=7)), 400), (dict(controller=fixed), 400),
                                 (dict(dynamic=False), 400)):
                with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                    status, content = await close(**kwargs)
                    self.assertEqual((status, content['error']['code']), (code, code))
            self.assertEqual(live.serving.closes, ['http-x-s1'])
            self.assertEqual(route.session_close_status(RuntimeError('boom')), 500)


# ---- the worker and the Goal thread -----------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for the Page arena')
class WorkerCloseTests(unittest.TestCase):
    def test_close_returns_arena_slots_and_refills_reuse_them(self):
        """Twelve sessions, two open at a time, each sealing six Pages into a real
        (unpinned, CPU) Page arena: each close returns its slots, so the arena never
        holds more than two sessions' Pages and never allocates a chunk beyond them."""
        import test_decouple_wp3_arena as WA
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        world = WA.BatchedTensorIOTests('test_round_trip_axis0_ratio1')
        io, _ = world.world(seed=3, chunk_pages=3, max_pages=12)
        owner = RunnerAllocationOwner.__new__(RunnerAllocationOwner)
        owner.io, owner.runner, owner.records = io, NS(), {}
        owner.sessions, owner.session_namespaces = {}, {}
        backend, qwen = NS(verify_identity=lambda: None, carry={}), NS(requests={})
        held, chunks = [], set()
        for k in range(12):
            sid, ns = 's%d' % k, 'ns%d' % k
            payloads = io.capture_pages([{world.FA: (b,)} for b in range(1, 7)])
            owner.sessions[ns] = dict(catalog=dict(enumerate(payloads)))
            owner.session_namespaces[sid] = ns
            backend.carry[sid] = object()
            del payloads
            held.append(io.arena.stats()['pages_held'])
            if k:
                chunks.add(io.arena.stats()['chunks'])
            if k >= 1:
                receipt = owner._close_session(dict(session='s%d' % (k - 1), namespace='ns%d' % (k - 1),
                                                    reason='close'))(backend, qwen)
                self.assertEqual(receipt['released']['catalog_pages'], 6)
                self.assertEqual((receipt['released']['carry'], receipt['released']['arena']['pages_held']),
                                 (True, 6))
                self.assertIn('s%d' % (k - 1), owner.closed_sessions)
        self.assertEqual((held, chunks), ([6] + [12] * 11, {4}))
        # Shutdown and rebuild closes report nothing extra and close nothing for good.
        receipt = owner._close_session(dict(session='s11', namespace='ns11', reason='shutdown'))(backend, qwen)
        self.assertEqual((set(receipt), io.arena.stats()['pages_held']),
                         ({'operation', 'closed', 'drained', 'namespace', 'reason'}, 0))
        self.assertNotIn('s11', owner.closed_sessions)
        with self.assertRaisesRegex(RuntimeError, 'close_session namespace/reason mismatch'):
            owner._close_session(dict(session='s1', namespace='ns1', reason='other'))


class GoalStateTests(unittest.TestCase):
    def test_closed_session_goal_state_is_released_behind_the_goal_thread(self):
        from remote_ra_live_v2.goal_async import GoalAdmissions, GoalWork
        from remote_ra_live_v2.query_selection import QuerySelectionAuthority
        gate, order = threading.Event(), []

        class Constructor:
            def construct(self, context):
                gate.wait(5)
                order.append(('construct', context))

            def release_session(self, session_id):
                order.append(('release', session_id))
        selection = QuerySelectionAuthority.__new__(QuerySelectionAuthority)
        for name in ('vector_catalog', 'turn_ends', 'previous_active', 'ever_selected', 'goal_cache',
                     'compact_cache', 'catalog_version', 'page_index'):
            setattr(selection, name, {'s1': object(), 's2': object()})
        selection.constructor = Constructor()
        goals = GoalAdmissions(NS(entries={}), wake=lambda: None)
        # A discarded Goal of s1 still builds on the Goal thread.
        goals._thread.submit(GoalAdmissions._build, GoalWork('r1', selection.constructor, 's1'))
        self.assertEqual(selection.close_session('s1', goals), 'goal_thread')
        self.assertTrue(all(set(getattr(selection, n)) == {'s2'} for n in ('vector_catalog', 'page_index')))
        self.assertEqual(order, [])                     # the engine thread did not wait
        gate.set()
        goals._thread.submit(lambda: None).result()     # the Goal thread's queue is FIFO
        goals.close()
        self.assertEqual(order, [('construct', 's1'), ('release', 's1')])
        # No Goal thread (or a closed one): released in place.
        self.assertEqual(selection.close_session('s2', goals), 'inline')
        self.assertEqual(selection.close_session('s3', None), 'inline')
        self.assertEqual(order[2:], [('release', 's2'), ('release', 's3')])


# ---- end to end through HTTP ---------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class HttpPoolTests(unittest.IsolatedAsyncioTestCase):
    """test_ms2_e2e's API server (packaged router, invoke_contract, the real
    LiveFourPolicyController, ServingController, FrontendRegistry and FrontendBridge
    with signed private frames, RegisteredAddDispatcher, the real engine and worker
    behind a stub of vLLM's engine loop) on a session-pool server; the harness
    closes a session with the packaged POST /v1/sessions/close route."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_ms2_e2e as B2
        import test_ms_e2e as E2E
        import test_ms_engine as ME
        cls.B2, cls.E2E, cls.ME = B2, E2E, ME
        for name in ('check', 'isolated', 'attributed', 'settled', 'quiescent', 'alone'):
            setattr(cls, name, getattr(B2.BatchedEndToEndTests, name))
        cls.check_engine = ME.MultiSessionEngineTests.check

    async def asyncSetUp(self):
        await self.E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    @staticmethod
    def comparable(info):
        import test_ms2_e2e as B2
        value = B2.BatchedEndToEndTests.comparable(info)
        value['row'].pop('open_wait_s', None)
        return value

    def pool_server(self, policy, listed, limit, blocks=4000):
        import functools
        from remote_ra_http import mswea_route
        from remote_ra_live_v2.composition import compose_frontend
        B2, ME = self.B2, self.ME
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = ME.config(policy, output, listed)
        value['max_running_sessions'] = limit
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        with patch.object(B2, 'compose_frontend', functools.partial(compose_frontend, max_running=limit)):
            srv = B2.BatchServer(policy, listed, output, self.router, self.Completion, B2.SERVER, seqs=limit,
                                 blocks=blocks, mrope=True)
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        self.addAsyncCleanup(srv.stop)
        srv.opens = []
        wait = srv.live.open_wait

        async def watched(request):
            session = srv.live.session_of(request)
            opened = session.opened
            waited = await wait(request)
            if not opened:
                srv.opens.append((session.index, srv.live.open_count(), waited))
            return waited
        srv.live.open_wait = watched
        return srv

    async def http_close(self, srv, k, host='127.0.0.1', instance=None):
        lane = srv.lanes[k]

        async def payload():
            return dict(instance_id=lane.task['instance_id'] if instance is None else instance)
        raw = NS(app=NS(state=srv.state), client=NS(host=host), json=payload, headers={})
        response = await asyncio.wait_for(self.router.close_session(raw), 10)
        return getattr(response, 'status_code', 200), response.content

    async def shutdown(self, srv):
        from entrypoint_runtime import ShutdownOwner
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        await asyncio.wait_for(srv.loop, 1)
        return json.loads((srv.output / 'engine-cleanup.json').read_bytes())

    # The slow pass (README section 2.3); the default suite keeps the HTTP pool's other
    # test and ControllerTests' close route.
    @unittest.skipUnless(os.environ.get('REMOTE_RA_SLOW_TESTS') == '1',
                         'about 15 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_32_tasks_served_by_16_agents_that_close_as_they_finish(self):
        A = self.ME.A
        listed = [A] + tasks(T32 - 1)
        turns = [1 + k % 4 for k in range(T32)]
        alone = await self.alone('full', listed[:3], {k: range(turns[k]) for k in range(3)})
        srv = self.pool_server('full', listed, K16)
        h = srv.engine
        queue, results, closes = deque(range(T32)), {}, []

        async def agent():
            while queue:
                k = queue.popleft()
                results[k] = [self.check(srv, await srv.play(k, i)) for i in range(turns[k])]
                closes.append((k, await self.http_close(srv, k)))
        await asyncio.gather(*(agent() for _ in range(K16)))
        # Every turn of every task; at most 16 sessions open (16 at once), all
        # opened without waiting, the slots refilled as agents closed sessions.
        self.assertEqual(sorted(results), list(range(T32)))
        self.assertEqual(sorted(index for index, _, _ in srv.opens), list(range(T32)))
        self.assertEqual(max(count for _, count, _ in srv.opens), K16)
        self.assertTrue(all(1 <= count <= K16 for _, count, _ in srv.opens))
        widths = [len(b.rows) for b in h.batches]
        self.assertTrue(2 <= max(widths) <= K16)
        self.assertNotEqual([k for k, _ in closes], sorted(k for k, _ in closes))
        for k, (status, content) in closes:
            lane = srv.lanes[k]
            self.assertEqual((status, content['object'], content['session_id'], content['instance_id'],
                              content['opened'], content['turns']),
                             (200, 'session.close', lane.sid, lane.task['instance_id'], True, turns[k]))
        # Each of the first three tasks saw exactly what it saw alone on a fixed-session server.
        for k in range(3):
            self.assertEqual([self.comparable(i) for i in results[k]], alone[k], k)
        rows = srv.rows()
        self.assertEqual(len(rows), sum(turns))
        self.assertTrue(all(type(r['open_wait_s']) is float and r['open_wait_s'] >= 0 for r in rows))
        # Everything released: pools at their baseline, nothing open anywhere.
        self.settled(srv)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertEqual((h.adapter.sessions, h.sessions.namespaces, h.owner.sessions, h.backend.carry),
                         ({}, {}, {}, {}))
        self.assertEqual((srv.live.open_count(), [s.ended for s in srv.live.sessions]), (0, [True] * T32))
        events = [json.loads(line) for line in (srv.output / 'sessions.jsonl').read_bytes().splitlines()]
        self.assertEqual(sorted((r['event'], r['session_index']) for r in events),
                         sorted([('open', k) for k in range(T32)] + [('close', k) for k in range(T32)]))
        # A closed task fails at once, before admission.
        with self.assertRaisesRegex(RuntimeError, 'closed by the harness'):
            await srv.play(5, 0)
        self.assertEqual(srv.rows('failures.jsonl')[-1]['session_index'], None)
        cleanup = await self.shutdown(srv)
        self.assertEqual(cleanup['sessions'], [])
        self.assertEqual(cleanup['session_pool'], dict(max_running_sessions=K16, registered=T32,
                                                       closed=[lane.sid for lane in srv.lanes],
                                                       closed_unopened=[], open_at_shutdown=[], never_opened=[]))

    async def test_open_wait_refusals_isolation_and_shutdown(self):
        from remote_ra_live_v2 import capacity as gate
        self.enterContext(patch.object(gate, 'BACKOFF_SECONDS', 0.02))
        A, B, C = self.ME.A, self.ME.B, self.ME.C
        listed = [A, B, C] + tasks(3)
        srv = self.pool_server('full', listed, 2, blocks=2000)
        h = srv.engine
        await srv.play(0, 0)
        await srv.play(1, 0)
        # K sessions are open: tasks 2 and 3 wait, FIFO, before admission.
        t2 = asyncio.create_task(srv.play(2, 0))
        t3 = asyncio.create_task(srv.play(3, 0))
        await self.B2.settle(lambda: len(srv.live._opening) == 2)
        self.assertEqual(set(h.sessions.namespaces), {srv.lanes[0].sid, srv.lanes[1].sid})
        # Refused closes change nothing: not loopback, unknown, busy (session 0's request held).
        hold = srv.hold(0, 1)
        t0 = asyncio.create_task(srv.play(0, 1))
        await asyncio.wait_for(hold.parked.wait(), 5)
        for kwargs, code in ((dict(host='192.0.2.7'), 403), (dict(instance='repo__nope'), 404), ({}, 409)):
            status, content = await self.http_close(srv, 0, **kwargs)
            self.assertEqual((status, content['error']['code']), (code, code), kwargs)
        hold.gate.set()
        await t0
        self.assertEqual(len(srv.live._opening), 2)
        # Closing task 1 opens task 2 (the oldest waiter); closing task 0 opens task 3.
        status, receipt = await self.http_close(srv, 1)
        self.assertEqual((status, receipt['turns'], receipt['open_sessions']), (200, 1, 1))
        info2 = await asyncio.wait_for(t2, 10)
        self.assertFalse(t3.done())
        await self.http_close(srv, 0)
        info3 = await asyncio.wait_for(t3, 10)
        self.assertEqual([index for index, _, _ in srv.opens], [0, 1, 2, 3])
        self.assertGreaterEqual(info3['row']['open_wait_s'], info2['row']['open_wait_s'])
        self.assertGreaterEqual(info2['row']['open_wait_s'], 0.0)
        self.assertEqual((await self.http_close(srv, 1))[1]['repeated'], True)
        # Task 4's request goes away while it waits: its task is not closed and opens later.
        t4 = asyncio.create_task(srv.play(4, 0))
        await self.B2.settle(lambda: len(srv.live._opening) == 1)
        srv.disconnect(4)
        with self.assertRaises(asyncio.CancelledError):
            await t4
        self.assertEqual((srv.live.sessions[4].opened, srv.live.sessions[4].closed, srv.live._opening),
                         (False, False, deque()))
        failure = srv.rows('failures.jsonl')[-1]
        self.assertEqual((failure['session_index'], failure['error']), (4, 'CancelledError'))
        # A failure inside one session closes only that session; the harness closes
        # it (its pins come back) and its slot refills.
        h.tamper['prepare'] = lambda payload: dict(payload, ns_generation=-1)
        with self.assertRaisesRegex(RuntimeError, 'registered-add rejected: .*prepare identity/assignment mismatch'):
            await srv.play(2, 1)
        self.assertTrue(srv.live.sessions[2].closed and not srv.live.sessions[2].ended)
        await srv.play(3, 1)                                      # session 3 goes on
        status, receipt = await self.http_close(srv, 2)
        self.assertEqual((status, receipt['poisoned'], receipt['opened']), (200, True, True))
        await srv.play(4, 0)                                      # task 4 opens in the freed slot
        with self.assertRaisesRegex(RuntimeError, 'request matches no startup task'):
            await srv.call(4, [dict(role='system', content='s'), dict(role='user', content='Something else.')])
        with self.assertRaisesRegex(RuntimeError, 'session %s closed by the harness' % srv.lanes[0].sid):
            await srv.play(0, 2)
        self.isolated(h)
        # Shutdown with sessions 3 and 4 open: they close there; task 5 never opened.
        cleanup = await self.shutdown(srv)
        lanes = srv.lanes
        self.assertEqual(cleanup['sessions'], [dict(session_id=lanes[k].sid, session_index=k,
                                                    namespace=cleanup['sessions'][i]['namespace'])
                                               for i, k in enumerate((3, 4))])
        self.assertEqual(cleanup['session_pool'], dict(
            max_running_sessions=2, registered=6, closed=[lanes[k].sid for k in (0, 1, 2)], closed_unopened=[],
            open_at_shutdown=[lanes[3].sid, lanes[4].sid], never_opened=[lanes[5].sid]))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)


if __name__ == '__main__':
    unittest.main()
