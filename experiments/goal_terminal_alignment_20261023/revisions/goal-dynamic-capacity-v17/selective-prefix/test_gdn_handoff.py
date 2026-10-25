"""GDN hand-off on the device, admission design (e): ``serve.py --gdn-device-handoff``. CPU only.

Flag plumbing (design §7, step 1):

- off (the default) is byte-identical: serve.prepare and the dry-run main give
  the same manifest, startup config, argv, environment and refusals with the
  switch absent or False (test_cudagraph also runs b37044f's prepare, main,
  load, compose_engine and attach_loaded_worker next to the current ones, and
  test_decouple_evidence pins the evidence files byte for byte);
- on: the startup config gains exactly ``admission = {gdn_device_handoff:
  true}``; the vLLM argv and every other config field are unchanged;
  remote_ra_http.config.load checks the record; compose_engine sends it in the
  worker startup record and refuses a worker that attached with other flags;
  the worker compares it with the pinned startup config; the worker evidence
  and READY gain the record and nothing else.

TensorIO issue variants (step 2), on real CPU tensors (CpuTensorTestIO, an
unpinned Page arena):

- issue_restore / issue_copy / issue_zero / issue_restore_pages /
  issue_capture_into write exactly what restore / copy / zero / restore_pages /
  capture write, and the counters end equal once joined;
- H2D bytes are issued at once and joined at poll_joins, a drain or a join, in
  issue order, and each join is credited to the request whose IO scope issued
  it (never to a scope that is open at the join), so every frame can end with
  issued == joined;
- a pinned source (payload, arena Page, host bounce) stays referenced until its
  event is joined; a wait on an event that has not completed synchronizes it;
- off: no issue variant runs, nothing is pending and a request's counters keep
  their three keys.

The hand-off (step 3) end to end, over test_decouple_e2e's fakes (the real
engine, worker, selector and evidence writers; a store-backed IO with the
issue variants; the fake runner asserts at every forward that each GDN slot it
reads holds the dense chain state at that position):

- every (P, M) case: the per-turn checks of test_decouple_e2e hold, with
  ``restore.h2d_bytes`` = restored Pages and ``restore.d2d_bytes`` = GDN x
  (hand-off + crossings), the carry still restored by phase 1, every held slot
  freed exactly once (FakePool refuses a double free), the pools back at their
  baselines, no claim left;
- on == off: prompts, geometry, plans, rankings, attended Pages, computed
  positions, outputs, committed text, carries, Page vectors (int8 compact at
  P = 16) and selections are equal; the evidence differs only in the
  documented hand-off fields (IO counters, the phase-1 state identity and the
  digests and MACs over it); the SHA-256/HMAC calls are the off path's;
- three sessions decoding together, with the 3-slot gate;
- the held slot: an alias claim (another request's step, another phase-1
  lease) is refused, the claim ends when the copy is issued, a cancelled
  hand-off frees it after the drain; the gate counts three GDN slots;
- a carry buffer is never written before its last copy completed.
"""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import admission_flags as AF
from remote_ra_live_v2.contract import LiveV2Error

try:
    import torch
except ImportError:
    torch = None

ON = dict(gdn_device_handoff=True)
POLICIES = ('full', 'random50', 'goal_query', 'observation_query', 'random_fill')


# ---- launcher and startup config ---------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    A = dict(instance_id='a', problem_statement='fix a')
    B = dict(instance_id='b', problem_statement='fix b')

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path))

    def outcome(self, *args, **kwargs):
        try:
            return ('ok', serve.prepare(self.release, *args, **kwargs))
        except Exception as error:
            return (type(error).__name__, str(error))

    def cases(self):
        out = self.root/'out'
        cases = []
        for policy in POLICIES:
            for task in (dict(self.A), [dict(self.A), dict(self.B)]):
                for page_tokens, layout in ((784, None), (784, 'decoupled'), (16, None)):
                    selection = dict(no_goal_rewriter=True) if policy == 'goal_query' else None
                    cases.append(((task, policy, out, 8231, selection, 'mswea-xml', 24),
                                  dict(page_tokens=page_tokens, layout=layout)))
        cases += [
            (([dict(self.A), dict(self.B)], 'full', out, 8231), dict(page_tokens=16, max_running=1)),
            ((dict(self.A), 'full', out, 8231), dict(page_tokens=16, cuda_graphs='piecewise')),
            ((dict(self.A), 'observation_query', out, 8231), dict(page_tokens=16, cuda_graphs='full-decode')),
            # refusals
            ((dict(self.A), 'goal_query', out, 8231), {}),
            ((dict(self.A), 'full', out, 8231), dict(kv_cache_gib=12)),
            ((dict(instance_id='a'), 'full', out, 8231), {}),
        ]
        return cases

    def test_off_is_byte_identical_and_refuses_alike(self):
        kinds = set()
        for args, kwargs in self.cases():
            with self.subTest(args=args[1:], kwargs=kwargs):
                old = self.outcome(*args, **kwargs)
                self.assertEqual(self.outcome(*args, gdn_device_handoff=False, **kwargs), old)
                kinds.add(old[0])
                if old[0] == 'ok':
                    self.assertNotIn('admission', old[1][1])
        self.assertEqual(kinds, {'ok', 'ValueError'})

    def test_on_pins_only_the_admission_record(self):
        for args, kwargs in self.cases():
            with self.subTest(args=args[1:], kwargs=kwargs):
                off = self.outcome(*args, **kwargs)
                on = self.outcome(*args, gdn_device_handoff=True, **kwargs)
                if off[0] != 'ok':
                    self.assertEqual(on, off)
                    continue
                (m0, c0, a0, e0), (m1, c1, a1, e1) = off[1], on[1]
                self.assertEqual((m1, a1), (m0, a0))        # no vLLM argument changes
                self.assertEqual(c1, dict(c0, admission=ON))
                self.assertEqual({k: v for k, v in e1.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'},
                                 {k: v for k, v in e0.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'})
                self.assertEqual(e1['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(c1))

    def test_prepare_refuses_a_non_switch(self):
        for value in (1, 0, 'on', None):
            with self.subTest(value=value):
                self.assertEqual(self.outcome(dict(self.A), 'full', self.root/'out', 8231, gdn_device_handoff=value),
                                 ('ValueError', '--gdn-device-handoff is a switch'))

    def test_dry_run_main(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                    '--output', str(self.root/'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                serve.main()
            return stream.getvalue()
        for extra in ((), ('--page-tokens', '16'), ('--cuda-graphs', 'piecewise')):
            with self.subTest(extra=extra):
                # The dry run prints the vLLM command, which the flag does not change.
                self.assertEqual(main(*extra, '--gdn-device-handoff'), main(*extra))
        captured = []
        with patch.object(serve, 'prepare', lambda *args: captured.append(args) or (None, dict(goal_rewriter=None),
                                                                                   [], {})):
            main('--gdn-device-handoff')
            main()
        # (gdn_device_handoff, phase1_in_step): the last two arguments.
        self.assertEqual([args[-2:] for args in captured], [(True, False), (False, False)])

    # -- the startup config check --

    def startup(self, value):
        path = self.root/'startup.json'
        path.write_bytes(serve.wire(value))
        return patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                          REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(value))

    def config(self, **kwargs):
        name = 'cfg-%d' % len(os.listdir(self.root))
        return serve.prepare(self.release, dict(self.A), 'full', self.root/name, 8231, **kwargs)[1]

    def test_load_checks_the_record(self):
        off, on = self.config(), self.config(gdn_device_handoff=True)
        with self.startup(off):
            self.assertNotIn('admission', http_config.load())
            self.assertIsNone(AF.engine_flags())
        with self.startup(on):
            self.assertEqual(http_config.load()['admission'], ON)
            self.assertEqual(AF.engine_flags(), ON)
        for bad in ({}, dict(gdn_device_handoff=False), dict(gdn_device_handoff=1), dict(other=True),
                    [['gdn_device_handoff', True]], True):
            with self.subTest(bad=bad), self.startup(dict(off, admission=bad)):
                with self.assertRaises(RuntimeError):
                    http_config.load()

    def test_worker_compares_its_record_with_the_pinned_config(self):
        off, on = self.config(), self.config(gdn_device_handoff=True)
        for pinned, sent, ok in ((off, None, True), (on, ON, True), (on, None, False), (off, ON, False)):
            with self.subTest(pinned='admission' in pinned, sent=sent), self.startup(pinned):
                startup = dict(enabled=True) if sent is None else dict(enabled=True, admission=sent)
                if ok:
                    self.assertEqual(AF.worker_flags(startup), sent)
                else:
                    with self.assertRaises(LiveV2Error):
                        AF.worker_flags(startup)
        # Without the HTTP runtime (CPU fixtures) only the shape is checked.
        self.assertEqual(AF.worker_flags(dict(admission=ON)), ON)
        self.assertIsNone(AF.worker_flags({}))
        with self.assertRaises(LiveV2Error):
            AF.worker_flags(dict(admission=dict(gdn_device_handoff=False)))

    def test_compose_engine_sends_the_record_once_and_checks_the_reply(self):
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        for pinned in (self.config(), self.config(gdn_device_handoff=True)):
            for reply in ({}, dict(admission=ON)):
                with self.subTest(pinned='admission' in pinned, reply=reply):
                    sent = []

                    def collective_rpc(method, args, timeout):
                        sent.append(args[0])
                        return [dict(reply, evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS',
                                     loaded_sources_verified=True)]

                    core = NS(_remote_ra_live_v2={},
                              _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                              _remote_ra_trusted_issue=dict(self.A), _remote_ra_trusted_issues=[dict(self.A)],
                              _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                              _remote_ra_growing_adapter=NS(layout=None),
                              model_executor=NS(collective_rpc=collective_rpc))
                    with self.startup(pinned), patch.object(composition, 'EngineWorkerProxy',
                                                            side_effect=Stop()):
                        expected = pinned.get('admission')
                        if reply.get('admission') == expected:
                            with self.assertRaises(Stop):     # past the check, at the proxy
                                composition.compose_engine(core, object())
                        else:
                            with self.assertRaises(LiveV2Error):
                                composition.compose_engine(core, object())
                    (startup,) = sent
                    self.assertEqual(startup.get('admission'), expected)
                    self.assertEqual(set(startup) - {'admission'},
                                     {'enabled', 'source', 'page_tokens', 'loaded_sources_verified'})


# ---- worker attach evidence and READY ----------------------------------------------------------

def synced(*names):
    """remote_ra_live_v2's submodule attributes equal to sys.modules (a test that patched
    sys.modules may have dropped a module that a later relative import reloads)."""
    import importlib
    import remote_ra_live_v2
    for name in names:
        setattr(remote_ra_live_v2, name, importlib.import_module('remote_ra_live_v2.' + name))


class EvidenceTests(unittest.TestCase):
    def test_worker_evidence_gains_only_the_record(self):
        import test_cudagraph as CGT
        synced('worker_composition', 'loaded_sources', 'native_probe', 'vector_space', 'gdn_handoff',
               'admission_flags', 'cuda_graphs')
        from remote_ra_live_v2 import gdn_handoff
        attach = CGT.AttachLoadedWorkerTests('test_off_evidence_is_unchanged').attach
        off = attach(CGT.resolved_config(None))[0]
        calls = []
        with patch.object(gdn_handoff, 'attach_worker', lambda flags, **parts: calls.append((flags, parts))):
            on, runner, worker = attach(CGT.resolved_config(None), dict(admission=ON))
        self.assertEqual(on, dict(off, admission=ON))
        (flags, parts), = calls
        self.assertEqual((flags, sorted(parts)), (ON, ['backend', 'io', 'owner', 'qwen']))
        self.assertIs(parts['backend'], worker._remote_ra_live_v2_backend)
        with self.assertRaises(LiveV2Error):
            attach(CGT.resolved_config(None), dict(admission=dict(gdn_device_handoff=False)))

    def test_ready_records_the_flags_and_checks_the_worker(self):
        import test_decouple_evidence as DEV
        from remote_ra_http import evidence
        config, attached, frames = DEV.config, evidence.worker_attached, DEV.frames

        def counted():
            # The worker's frames carry d2d_bytes when the hand-off is on.
            out = frames()
            for frame in out:
                for side in ('before', 'after'):
                    frame['http_measurement'][side]['d2d_bytes'] = 7 if side == 'after' else 0
            return out
        for policy in ('full', 'observation_query'):
            with tempfile.TemporaryDirectory() as base:
                plain = json.loads(DEV.scenario(evidence, Path(base), policy, 784, False)['ready.json'])
            self.assertNotIn('admission', plain)
            plain.pop('worker_receipt_sha256')
            for pinned, receipt, ok in ((ON, ON, True), (ON, None, False), (None, ON, False)):
                with self.subTest(policy=policy, pinned=pinned, receipt=receipt), \
                        tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                    stack.enter_context(patch.object(DEV, 'config', lambda *args: dict(
                        config(*args), **({} if pinned is None else dict(admission=pinned)))))
                    stack.enter_context(patch.object(DEV, 'frames', counted if pinned else frames))
                    stack.enter_context(patch.object(evidence, 'worker_attached', lambda core, w, b, l, r: attached(
                        core, w, b, l, dict(r, **({} if receipt is None else dict(admission=receipt))))))
                    if not ok:
                        with self.assertRaises(RuntimeError):
                            DEV.scenario(evidence, Path(tmp), policy, 784, False)
                        continue
                    files = DEV.scenario(evidence, Path(tmp), policy, 784, False)
                    ready = json.loads(files['ready.json'])
                    self.assertEqual(ready['admission'], ON)
                    row = json.loads(files['runtime.jsonl'])
                    self.assertEqual(row['restore'], dict(joined=True, h2d_bytes=0, d2d_bytes=7))
                    # Everything else is the pinned legacy evidence.
                    self.assertEqual({k: v for k, v in ready.items() if k not in ('admission', 'worker_receipt_sha256')},
                                     plain)



# ---- TensorIO issue variants (step 2) ----------------------------------------------------------

class FakeEvent:
    """A CUDA event stand-in that completes only when synchronized (or set done)."""
    def __init__(self, log, name):
        self.log, self.name, self.done = log, name, False

    def query(self):
        return self.done

    def synchronize(self):
        self.log.append(('synchronize', self.name))
        self.done = True


@unittest.skipUnless(torch is not None, 'CPU torch required')
class TensorIssueTests(unittest.TestCase):
    BLOCKS = 12
    FA, GDN = 0, 1

    def world(self, handoff=True, seed=0, chunk_pages=2):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO, LayerView, PageArena
        g = torch.Generator().manual_seed(seed)
        n = self.BLOCKS
        views = [LayerView('fa0', self.FA, 'attention', (torch.randn(n, 2, 3, generator=g),), (0,)),
                 LayerView('fa1', self.FA, 'attention', (torch.randn(2, n, 3, generator=g),), (1,)),
                 LayerView('gdn0', self.GDN, 'hybrid', (torch.randn(n, 3, generator=g),
                                                         torch.randn(n, 2, 2, generator=g)), (0, 0)),
                 LayerView('gdn1', self.GDN, 'hybrid', (torch.randn(n, 4, generator=g),
                                                         torch.randn(n, 2, generator=g)), (0, 0))]
        io = CpuTensorTestIO(views, decoupled=True)
        page_bytes = sum((t.numel() // t.shape[a]) * 4 for v in views[:2] for t, a in zip(v.tensors, v.axes))
        io.attach_arena(PageArena(io.views, max_pages=20, pin=False, chunk_bytes=chunk_pages * page_bytes))
        if handoff:
            io.enable_handoff()
        return io

    def pair(self):
        """(handoff io, plain io) over equal tensors and equal arena contents."""
        return self.world(True), self.world(False)

    @staticmethod
    def tensors(io):
        return [t for v in io.views for t in v.tensors]

    def assert_same(self, a, b):
        for x, y in zip(self.tensors(a), self.tensors(b)):
            self.assertTrue(torch.equal(x, y))

    def test_off_issues_nothing(self):
        from remote_ra_growing_worker.tensors import attributed, request_counters
        io = self.world(False)
        self.assertEqual((io.handoff, io._pending), (False, ()))
        host = io.capture({self.GDN: (3,)}, 'hybrid')
        for call in (lambda: io.issue_restore({self.GDN: (4,)}, 'hybrid', host),
                     lambda: io.issue_copy({self.GDN: (3,)}, {self.GDN: (4,)}, 'hybrid'),
                     lambda: io.issue_zero({self.GDN: (4,)}, 'hybrid'),
                     lambda: io.issue_restore_pages([]),
                     lambda: io.host_buffer({self.GDN: (4,)}, 'hybrid'),
                     lambda: io.issue_capture_into({self.GDN: (4,)}, 'hybrid', host),
                     lambda: io.record_step()):
            before = [t.clone() for t in self.tensors(io)]
            with self.assertRaisesRegex(RuntimeError, 'issue variants need --gdn-device-handoff'):
                call()
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, self.tensors(io))))
        with attributed(io, 'r'):
            io.restore({self.GDN: (4,)}, 'hybrid', host)
        size = io.byte_count({self.GDN: (4,)}, 'hybrid')
        self.assertEqual(request_counters(io, 'r'), dict(h2d_issued=size, h2d_joined=size, zero_joins=0))
        self.assertEqual(io._pending, ())

    def test_issue_variants_write_what_the_synchronous_ones_write(self):
        a, b = self.pair()
        host_a, host_b = a.capture({self.GDN: (2, 3)}, 'hybrid'), b.capture({self.GDN: (2, 3)}, 'hybrid')
        a.issue_restore({self.GDN: (7, 8)}, 'hybrid', host_a)
        b.restore({self.GDN: (7, 8)}, 'hybrid', host_b)
        a.issue_copy({self.GDN: (5,)}, {self.GDN: (9,)}, 'hybrid')
        b.copy({self.GDN: (5,)}, {self.GDN: (9,)}, 'hybrid')
        a.issue_zero({self.GDN: (1,)}, 'hybrid')
        b.zero({self.GDN: (1,)}, 'hybrid')
        a.issue_zero({self.FA: (2, 4)}, 'attention')
        b.zero({self.FA: (2, 4)}, 'attention')
        pages = [{self.FA: (blk,)} for blk in (3, 6, 10, 11)]
        pa, pb = a.capture_pages(pages), b.capture_pages(pages)
        a.capture_pages([{self.FA: (1,)}])                      # fragment the arena alike
        b.capture_pages([{self.FA: (1,)}])
        tail_a, tail_b = a.capture({self.FA: (5,)}, 'attention'), b.capture({self.FA: (5,)}, 'attention')
        targets = [{self.FA: (blk,)} for blk in (7, 8, 9, 2)]
        a.issue_restore_pages(list(zip(targets, pa[::-1])) + [({self.FA: (4,)}, tail_a)])
        b.restore_pages(list(zip(targets, pb[::-1])) + [({self.FA: (4,)}, tail_b)])
        buffer = a.host_buffer({self.GDN: (9,)}, 'hybrid')
        a.issue_capture_into({self.GDN: (9,)}, 'hybrid', buffer)
        captured = b.capture({self.GDN: (9,)}, 'hybrid')
        self.assertGreater(len(a._pending), 0)
        a.poll_joins()
        self.assert_same(a, b)
        self.assertEqual(sorted(buffer), sorted(captured))
        for name in buffer:
            for x, y in zip(buffer[name][0], captured[name][0]):
                self.assertTrue(torch.equal(x, y))
        for name in ('_http_h2d_issued', '_http_h2d_joined', '_http_zero_joins', '_http_d2h_joined'):
            self.assertEqual(getattr(a, name), getattr(b, name), name)
        self.assertEqual(a._http_d2d_bytes, a.byte_count({self.GDN: (5,)}, 'hybrid'))
        self.assertEqual(a._pending, [])

    def test_copy_refuses_like_the_synchronous_copy(self):
        io = self.world()
        for source, target in (({self.GDN: (3,)}, {self.GDN: (3,)}),          # alias
                               ({self.GDN: (3,)}, {self.GDN: (4, 5)}),        # count
                               ({self.GDN: (3,)}, {self.FA: (4,)})):          # coverage
            with self.subTest(source=source, target=target):
                with self.assertRaises(RuntimeError):
                    io.issue_copy(source, target, 'hybrid')
        self.assertEqual(io._pending, [])

    def test_joins_are_credited_to_the_issuing_request(self):
        from remote_ra_growing_worker.tensors import attributed, request_counters
        io = self.world()
        host = io.capture({self.GDN: (2,)}, 'hybrid')
        size = io.byte_count({self.GDN: (2,)}, 'hybrid')
        with attributed(io, 'A'):
            io.issue_restore({self.GDN: (6,)}, 'hybrid', host)
            io.issue_zero({self.GDN: (7,)}, 'hybrid')
            io.issue_copy({self.GDN: (6,)}, {self.GDN: (8,)}, 'hybrid')
        self.assertEqual(request_counters(io, 'A'), dict(h2d_issued=size, h2d_joined=0, zero_joins=0,
                                                         d2d_bytes=size))
        with attributed(io, 'B'):
            with attributed(io, 'C'):
                io.poll_joins()                  # joined inside other requests' scopes
            io.issue_restore({self.GDN: (9,)}, 'hybrid', host)
            io.drain()                           # a drain joins it too
        self.assertEqual(request_counters(io, 'A'), dict(h2d_issued=size, h2d_joined=size, zero_joins=1,
                                                         d2d_bytes=size))
        self.assertEqual(request_counters(io, 'B'), dict(h2d_issued=size, h2d_joined=size, zero_joins=0,
                                                         d2d_bytes=0))
        self.assertEqual(request_counters(io, 'C'), dict(h2d_issued=0, h2d_joined=0, zero_joins=0, d2d_bytes=0))
        self.assertEqual((io._http_h2d_issued, io._http_h2d_joined, io._http_zero_joins), (2 * size, 2 * size, 1))

    def test_events_join_in_order_and_sources_live_until_joined(self):
        import gc
        io = self.world()
        log = []
        names = iter('abcdefgh')
        io._event = lambda: FakeEvent(log, next(names))
        pages = io.capture_pages([{self.FA: (blk,)} for blk in (1, 2, 3)])
        held = {p.slot: p.token for p in pages}
        host = io.capture({self.GDN: (2,)}, 'hybrid')       # (a capture drains: joins all pending)
        io.issue_restore_pages([({self.FA: (blk,)}, p) for blk, p in zip((7, 8, 9), pages)])   # event a
        first = io.issue_restore({self.GDN: (6,)}, 'hybrid', host)                             # event b
        last = io.issue_copy({self.GDN: (6,)}, {self.GDN: (8,)}, 'hybrid')                     # event c
        del pages
        gc.collect()
        # The arena Pages are referenced by their pending restore: their slots stay held.
        self.assertTrue(all(io.arena.slots.holds(slot, token) for slot, token in held.items()))
        self.assertIs(io._pending[1].holds[0], host)
        self.assertFalse(io.joined(first))
        io.join(first)                           # joins a and b, in order; c stays pending
        self.assertEqual(log, [('synchronize', 'b')])
        self.assertEqual([e.event.name for e in io._pending], ['c'])
        gc.collect()
        self.assertFalse(any(io.arena.slots.holds(slot, token) for slot, token in held.items()))
        self.assertTrue(io.joined(first))
        io.wait(first)                           # joined: a completed event, synchronized again
        io.poll_joins()
        self.assertEqual(log, [('synchronize', 'b'), ('synchronize', 'b'), ('synchronize', 'c')])
        self.assertTrue(io.joined(last) and io._pending == [])

    def test_wait_synchronizes_an_event_that_has_not_completed(self):
        io = self.world()
        log = []
        event = FakeEvent(log, 'reader')
        self.assertFalse(event.query())
        io.wait(event)                           # not pending here: synchronized directly
        self.assertEqual(log, [('synchronize', 'reader')])
        io.wait(None)
        self.assertEqual(len(log), 1)

    def test_capture_into_gives_a_fresh_identity_each_time(self):
        io = self.world()
        buffer = io.host_buffer({self.GDN: (3,)}, 'hybrid')
        io.issue_capture_into({self.GDN: (3,)}, 'hybrid', buffer)
        first = io.payload_digest(buffer)
        io.issue_capture_into({self.GDN: (4,)}, 'hybrid', buffer)
        second = io.payload_digest(buffer)
        self.assertNotEqual(first, second)
        self.assertEqual(len(second), 64)
        with self.assertRaises(RuntimeError):
            io.issue_capture_into({self.GDN: (3,)}, 'hybrid', io.host_buffer({self.FA: (3,)}, 'attention'))
        io.poll_joins()
        self.assertEqual(io._http_d2h_joined, 2 * io.byte_count({self.GDN: (3,)}, 'hybrid'))



# ---- the hand-off end to end (step 3) ----------------------------------------------------------

def handoff_io(base):
    """test_decouple_e2e's store IO with the issue variants; joins borrowed from TensorIO."""
    from remote_ra_growing_worker import tensors as TS
    import test_stageb_e2e as S

    class HandoffIO(base):
        enable_handoff = TS.TensorIO.enable_handoff
        _issued, _credit = TS.TensorIO._issued, TS.TensorIO._credit
        poll_joins, join, joined, wait = (TS.TensorIO.poll_joins, TS.TensorIO.join, TS.TensorIO.joined,
                                          TS.TensorIO.wait)
        check_connector, record_step = TS.TensorIO.check_connector, TS.TensorIO.record_step
        handoff, _pending = False, ()

        def _event(self):
            return TS._DoneEvent()

        def drain(self):
            super().drain()
            if self._pending:
                self.poll_joins()

        def issue_restore(self, groups, kind, payload):
            keys = self.keys(groups, kind)
            assert payload['kind'] == kind and len(payload['blocks']) == len(keys)
            self.log.append(('restore', kind, keys))
            for (g, b), value in zip(keys, payload['blocks']):
                self.store.write(kind, g, b, value)
            size = len(keys) * S.BYTES[kind]
            self._http_h2d_issued += size
            return self._issued(h2d=size, holds=(payload,))

        def issue_restore_pages(self, items):
            items = list(items)
            self.log.append(('restore_pages', 'attention', len(items)))
            events = [self.issue_restore(g, 'attention', p) for g, p in items]
            return events[-1] if events else None

        def issue_copy(self, source, target, kind):
            src, dst = self.keys(source, kind), self.keys(target, kind)
            assert len(src) == len(dst) and not {b for _, b in src} & {b for _, b in dst}, 'copy alias'
            self.log.append(('copy', kind, list(zip(src, dst))))
            for (g, a), (h, b) in zip(src, dst):
                assert g == h
                self.store.write(kind, g, b, self.store.read(kind, g, a))
            self._http_d2d_bytes = getattr(self, '_http_d2d_bytes', 0) + len(dst) * S.BYTES[kind]
            return self._issued()

        def issue_zero(self, groups, kind):
            self.zero(groups, kind)
            return self._issued()        # the store IO's zero counts no join either

        def host_buffer(self, groups, kind):
            return dict(kind=kind, blocks=[None] * len(self.keys(groups, kind)))

        def issue_capture_into(self, groups, kind, payload):
            keys = self.keys(groups, kind)
            assert payload['kind'] == kind and len(payload['blocks']) == len(keys)
            self.log.append(('capture_into', kind, keys))
            payload['blocks'][:] = [self.store.read(kind, g, b) for g, b in keys]
            return self._issued(d2h=len(keys) * S.BYTES[kind], holds=(payload,))
    return HandoffIO


def handoff_harness(base):
    """A test_decouple_e2e harness (or MultiHarness) wired with the hand-off on, as
    attach_loaded_worker and compose_engine wire it."""
    from remote_ra_live_v2 import gdn_handoff

    class HandoffHarness(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.io.__class__ = handoff_io(type(self.io))
            gdn_handoff.attach_worker(ON, io=self.io, owner=self.owner, backend=self.backend, qwen=self.qwen)
            self.held = gdn_handoff.attach_engine(self.core, ON)
            free = self.held.release
            self.releases = []

            def release(request):
                done = free(request)
                self.releases.append((request, done, list(self.rpc_log)))
                return done
            self.held.release = release
    return HandoffHarness


def handoff_config(config):
    return lambda *args: dict(config(*args), admission=dict(ON))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class HandoffEndToEndTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_decouple_e2e as DE
        cls.DE = DE

    def run_on(self, policy, case, check=None, deterministic=False):
        DE = self.DE
        with ExitStack() as stack:
            stack.enter_context(patch.object(DE, 'Harness', handoff_harness(DE.Harness)))
            stack.enter_context(patch.object(DE, 'config', handoff_config(DE.config)))
            return DE.run(policy, case, check=check, deterministic=deterministic)

    def check_turn(self, h, info):
        """test_decouple_e2e's per-turn checks, with the hand-off's byte identities first."""
        DE = self.DE
        rid, geo, row = info['rid'], info['geo'], info['row']
        plan = info['plan']
        moves = sum(1 for a, c in zip(h.slots[rid], h.slots[rid][1:]) if a != c)
        restored = DE.BYTES['attention'] * len(plan['restore'])
        self.assertEqual(row['prepare']['h2d_bytes'], DE.HYBRID * bool(geo.local_turn))   # the carry
        self.assertEqual(row['restore']['h2d_bytes'], restored)                            # Pages only
        self.assertEqual(row['restore']['d2d_bytes'], DE.HYBRID * (bool(geo.local_turn) + moves))
        for frame in info['frames']:
            measured = frame['http_measurement']
            for side in ('before', 'after'):
                self.assertEqual(measured[side]['h2d_issued'], measured[side]['h2d_joined'])
                self.assertIn('d2d_bytes', measured[side])
        # The held slot went exactly once, and no claim or held state survives the turn.
        releases = [done for request, done, _ in h.releases if request == rid and done]
        self.assertEqual(len(releases), 1 if geo.local_turn else 0)
        self.assertEqual((h.held.slots, h.owner.held_claims, h.backend.final_events), ({}, {}, {}))
        self.assertFalse(any(op == 'capture' and kind == 'hybrid' for op, kind, *_ in
                             h.io.log[info['mark'][1]:]), 'a host capture of GDN state under the hand-off')
        row = dict(row, restore=dict(row['restore'], h2d_bytes=restored + DE.HYBRID *
                                     (bool(geo.local_turn) + moves)))
        DE.MatrixChecks.check_turn(self, h, dict(info, row=row))

    def test_every_case_with_the_hand_off(self):
        DE = self.DE
        for case, policy in zip(DE.CASES, ('random50', 'goal_query', 'goal_query')):
            with self.subTest(case=case.name, policy=policy):
                def check(h, info):
                    if not hasattr(h, 'geometries'):
                        h.geometries, h.active_turns, h.restores = [], [], 0
                    self.check_turn(h, info)
                h, infos, cancel, files = self.run_on(policy, case, check=check)
                self.assertEqual(cancel['receipt_cancel']['state'], 'cancelled')
                self.assertEqual(cancel['committed_before'], cancel['committed_after'])
                # The cancelled turn's held slot went after its drain, once.
                self.assertEqual([done for r, done, _ in h.releases if r == cancel['rid'] and done], [True])
                self.assertEqual(h.held.freed, sum(1 for i in infos + [cancel] if i['geo'].local_turn))
                self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                                 (h.baseline, h.mamba_baseline))
                self.assertTrue(all(b.ref_cnt == 0 for pool in h.kv.block_pools for b in pool.blocks[1:]))
                self.assertFalse(h.owner.held_claims or h.backend.carry_buffers or h.io._pending)
                copies = [e for e in h.io.log if e[0] == 'copy']
                self.assertGreater(len(copies), sum(1 for i in infos if i['geo'].local_turn))
                self.assertEqual(h.kv.decoupled, case.decoupled)

    def test_a_hand_off_that_never_ran_frees_after_the_drain(self):
        DE = self.DE
        case = DE.CASES[2]
        with ExitStack() as stack:
            DE.enter_case(stack, case)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = dict(DE.config('full', output, case), admission=dict(ON))
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = handoff_harness(DE.Harness)('full', output, case)
            h.play_turn(*DE.TURNS[0])
            h.play_turn(*DE.TURNS[1])
            info = h.submit(h.messages + [dict(role='user', content=DE.OUT % (0, 'never runs'))])
            rid = info['rid']
            self.assertEqual(set(h.held.slots), {rid})
            (session, blocks), = h.held.slots.values()
            self.assertEqual(set(h.owner.held_claims.values()), {(rid, h.owner.records[rid]['generation'])})
            self.assertEqual(sorted(h.owner.held_claims), sorted((g, b.block_id) for g, b in zip(DE.E.MAMBA, blocks)))
            free = h.mamba_pool.get_num_free_blocks()
            h.lifecycle.cancel(info['ident'])
            (request, done, rpcs), = [r for r in h.releases if r[1] and r[0] == rid]
            self.assertEqual((request, done), (rid, True))
            self.assertEqual(rpcs[-1], ('cancel', rid))         # freed after the worker's drain
            self.assertEqual(h.mamba_pool.get_num_free_blocks(), free + len(blocks))
            self.assertEqual((h.held.slots, h.owner.held_claims, h.held.freed), ({}, {}, 2))
            h.evidence.failed(NS(internal_id=rid), RuntimeError('client disconnected'))
            h.close()
            DE.engine_closed(h.core)
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))

    def test_on_equals_off(self):
        DE = self.DE
        import test_host_trim as HT
        for case, policy in ((DE.CASES[2], 'goal_query'), (DE.CASES[0], 'observation_query')):
            with self.subTest(case=case.name, policy=policy):
                hashing = [HT.Hashing(), HT.Hashing()]

                def counted(k, run):
                    # SHA-256/HMAC calls from runtime sources over the whole run (decoupled case).
                    if not case.decoupled:
                        return run()
                    sys.setprofile(hashing[k])
                    hashing[k].on = True
                    try:
                        return run()
                    finally:
                        hashing[k].on = False
                        sys.setprofile(None)
                off = counted(0, lambda: DE.run(policy, case, deterministic=True))
                on = counted(1, lambda: self.run_on(policy, case, deterministic=True))
                self.assertEqual(hashing[1].calls, hashing[0].calls)
                self.assertEqual(bool(hashing[0].calls), case.decoupled)
                (h0, infos0, cancel0, files0), (h1, infos1, cancel1, files1) = off, on
                self.assertEqual(DE.semantics(h1, infos1, cancel1), DE.semantics(h0, infos0, cancel0))
                self.assertEqual(HT.plain(h1.kv.calls), HT.plain(h0.kv.calls))
                self.assertEqual([t['operation'] for t in h1.trace], [t['operation'] for t in h0.trace])
                for a, b in zip(infos0, infos1):
                    self.assertEqual(a['receipt']['carry'], b['receipt']['carry'])
                    self.assertEqual(a['receipt']['page_vectors'], b['receipt']['page_vectors'])
                    self.assertEqual(a['receipt']['new_pages'], b['receipt']['new_pages'])
                if h0.selection is not None:
                    vectors = lambda h: {sid: {p: (tuple(getattr(v, 'values', v)), getattr(v, 'scale', None))
                                               for p, v in pages.items()}
                                         for sid, pages in h.selection.vector_catalog.items()}
                    self.assertEqual(vectors(h1), vectors(h0))
                self.assertEqual(sorted(files1), sorted(files0))
                if not case.decoupled:
                    # One shared pool: the held slot is freed one allocation later, so the
                    # pool hands out other FA ids too; the decisions above are what count.
                    continue
                seen = set()
                for name, raw in files0.items():
                    load = lambda b: ([json.loads(line) for line in b.splitlines()] if name.endswith('.jsonl')
                                      else json.loads(b))
                    for path, kind in DE.json_differences(load(raw), load(files1[name]), ''):
                        path = __import__('re').sub(r'/\d+(?=/|\[|$)', '/<n>', path)
                        self.assertTrue(any(__import__('re').fullmatch(p, path) for p in HANDOFF_FIELDS),
                                        (name, path, kind))
                        seen.add(path)
                self.assertIn(r'[]/restore/d2d_bytes', seen)
                for a, b in zip(h0.trace, h1.trace):
                    self.assertEqual(fa_only(a['payload']), fa_only(b['payload']), a['operation'])

    def test_three_sessions_with_the_three_slot_gate(self):
        DE = self.DE
        import test_decouple_e2e as DEM
        for case in (DE.CASES[2],):
            with self.subTest(case=case.name):
                multi = DE.Case(**dict(vars(case), sessions=len(DE.TASKS)))
                with ExitStack() as stack:
                    DE.enter_case(stack, multi)
                    output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    value = dict(DE.config('goal_query', output, multi, DE.TASKS), admission=dict(ON))
                    stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
                    stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
                    h = handoff_harness(DE.MultiHarness)('goal_query', output, multi)
                    h.active = h.restores = 0
                    checks = DEM.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')
                    h.serve(checks.check)
                    self.assertEqual(len(h.done), 5 * len(DE.TASKS))
                    self.assertEqual(max(h.widths), len(DE.TASKS))
                    self.assertGreater(h.probes_while_running, 0)
                    self.assertEqual(h.held.freed, sum(1 for i in h.done if i['geo'].local_turn))
                    h.close()
                    DE.engine_closed(h.core)
                    self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                                     (h.baseline, h.mamba_baseline))
                    # Three slots per hand-off, a held slot free once its first step ran:
                    # K = 2S + 1 still admits every turn without a CAPACITY_WAIT here.
                    self.assertEqual(h.waits, 0)


def fa_only(payload):
    """An RPC payload without its Mamba block ids (and the digests and MACs over them):
    the held probe slot is freed one allocation later, so the Mamba pool hands out
    other ids. FullAttention ids, tokens and plans stay."""
    payload = json.loads(json.dumps(payload))
    if 'rows' in payload:                                  # allocation_update: FA group 0
        payload['rows'] = payload['rows'][:1]
    request = payload.get('phase1_request')
    for holder in (payload, request or {}):
        lease = holder.get('lease')
        for item in (lease, (lease or {}).get('goal')):
            if type(item) is dict:
                item['rows'] = item['rows'][:1]
                for key in ('owned', 'incarnations', 'mac'):
                    item.pop(key, None)
        for key in ('lease_digest', 'request_digest'):
            holder.pop(key, None)
    phase1 = (payload.get('plan') or {}).get('phase1')
    if type(phase1) is dict:
        phase1.pop('lease_digest'), phase1.pop('receipt_digest')
    return payload


# Evidence fields the hand-off changes (on vs off, same deterministic run, decoupled):
# the IO counters of every frame (H2D now Pages only, d2d_bytes added); the phase-1
# state's identity token (the off path hashes its host copy); the Mamba block ids
# (the held probe slot is freed one allocation later, so the Mamba pool hands out
# other ids: allocation releases, phase-1 lease rows) and the digests and MACs over
# them; the restore/d2d fields of the runtime rows and their file digests.
HANDOFF_FIELDS = [
    r'(/drain|/worker_frames\[\])/http_measurement/(before|after)/(h2d_issued|h2d_joined|d2d_bytes)',
    r'(/drain|/worker_frames\[\])/http_measurement/active_owned_kv_bytes',
    r'/worker_frames\[\](/worker_receipt/body)?/(state_digest|lease_digest|request_digest)',
    r'/worker_frames\[\]/worker_receipt/mac',
    r'(/drain|/worker_frames\[\])/(plan|lease_digest)',
    r'/plan/phase1/(receipt_digest|lease_digest)',
    r'/query/body/phase1/lease_digest',
    r'\[\]/restore/(h2d_bytes|d2d_bytes)',
    r'\[\]/native_evidence_sha256',
    r'/query/body/phase1/state_digest',
    r'/query/mac',
    # Mamba block ids: the held probe slot is freed one allocation later, so the
    # Mamba pool hands the phase-2 slots other ids (FA ids are unchanged).
    r'(/drain|/worker_frames\[\])/release\[\]\[\]',
]


# ---- the held slot, the gate and the carry buffers (unit level) --------------------------------

class HeldSlotTests(unittest.TestCase):
    def owner(self):
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        from remote_ra_live_v2.contract import NativeIdentity
        owner = RunnerAllocationOwner.__new__(RunnerAllocationOwner)
        owner.hybrid_groups, owner.attention_groups = (1, 2), (0,)
        owner.claims, owner.held_claims, owner.records = {}, {}, {}
        owner.pool_of = None
        ids = {}
        for k, rid in enumerate(('a', 'b')):
            ids[rid] = NativeIdentity(rid, 's%d' % k, 1, 'g', 'a' * 64, 'b' * 64, 'model')
            owner.records[rid] = dict(identity=ids[rid], generation=k + 1, allocations=set())
        return owner, ids

    def test_held_slot_claims_refuse_aliases_and_end_once(self):
        from remote_ra_live_v2.allocation_owner import StepRow
        from remote_ra_live_v2.contract import Allocation
        owner, ids = self.owner()
        owner.hold_probe(ids['a'], {1: (40,), 2: (41,)}, 1)
        self.assertEqual(owner.held_claims, {(1, 40): ('a', 1), (2, 41): ('a', 1)})
        with self.assertRaises(LiveV2Error):        # held twice
            owner.hold_probe(ids['a'], {1: (42,), 2: (43,)}, 1)
        with self.assertRaises(LiveV2Error):        # another request's probe on the same slot
            owner.hold_probe(ids['b'], {1: (40,), 2: (44,)}, 2)
        alias = Allocation(1, 40, 2, 'mamba', 1)
        step = StepRow('b', 0, 0, 5, 1, (), (alias,), 2, (alias.key,))
        with self.assertRaisesRegex(LiveV2Error, 'held phase-1 GDN slot'):
            owner._refuse_held((step,))
        owner.verify_held(ids['a'], {1: (40,), 2: (41,)}, 1)
        owner.release_probe(ids['a'], 1)
        self.assertEqual(owner.held_claims, {})
        owner._refuse_held((step,))                # the slot may now be anyone's
        with self.assertRaises(LiveV2Error):
            owner.verify_held(ids['a'], {1: (40,), 2: (41,)}, 1)
        owner.release_probe(ids['a'], 1)           # idempotent

    def test_engine_ledger_frees_exactly_once(self):
        import test_stageb_engine as E
        from remote_ra_live_v2.gdn_handoff import HeldSlots
        pool = E.FakePool(10)
        freed = []
        coordinator = NS(_free=lambda blocks: (freed.append(blocks), pool.free_blocks(reversed(blocks))))
        held = HeldSlots(coordinator)
        blocks = pool.get_new_blocks(3)
        held.hold('r', 's', blocks)
        with self.assertRaises(LiveV2Error):
            held.hold('r', 's', blocks)
        self.assertTrue(held.holds('s') and not held.holds('t'))
        self.assertTrue(held.release('r'))
        self.assertFalse(held.release('r'))
        self.assertEqual((len(freed), held.freed, pool.get_num_free_blocks()), (1, 1, 9))

    def test_gate_counts_three_slots_in_phase_two(self):
        from remote_ra_live_v2 import capacity
        endpoint = NS(phase1=NS(groups=lambda: ((0,), (1, 2, 3), 4), max_tokens=4096, handoff=None),
                      transcripts={}, selection=None)
        session = NS(local_turn=2, sealed_end=1568, resident=(0, 1), tail=None, policy='full')
        core = NS(_remote_ra_growing_adapter=NS(sessions={'s': session}, page_tokens=784),
                  _remote_ra_private_v2_endpoint=endpoint)
        off = capacity.turn_need_by_pool(core, 's', list(range(2000)), None, 1536)
        endpoint.phase1.handoff = object()
        on = capacity.turn_need_by_pool(core, 's', list(range(2000)), None, 1536)
        self.assertEqual(off[1][1], 2 * 3)
        self.assertEqual(on, (off[0], (off[1][0], 3 * 3)))
        session.local_turn = 0                     # turn 0: no phase 1, nothing held
        self.assertEqual(capacity.turn_need_by_pool(core, 's', list(range(2000)), None, 1536)[1][1], 2 * 3)

    def test_gate_counts_a_held_slot_free_once_its_first_step_ran(self):
        from remote_ra_live_v2 import capacity
        from remote_ra_live_v2.gdn_handoff import HeldSlots
        held = HeldSlots(NS(_free=lambda blocks: None))
        held.hold('a', 's', [NS(ref_cnt=1) for _ in range(3)])
        request = NS(num_computed_tokens=100)
        endpoint = NS(phase1=NS(handoff=held), claims={'a': dict(geometry=NS(phase2_start=100))})
        lifecycle = NS(endpoint=NS(core=NS(_remote_ra_private_v2_endpoint=endpoint)),
                       entries={'a': dict(state='accepted', request=request)})
        self.assertEqual(capacity.releasing(lifecycle), 0)       # its first step has not run
        request.num_computed_tokens = 101
        self.assertEqual(capacity.releasing(lifecycle), 3)
        lifecycle.entries['a']['state'] = 'reserving'
        self.assertEqual(capacity.releasing(lifecycle), 0)
        endpoint.phase1.handoff = None
        self.assertEqual(capacity.releasing(lifecycle), 0)
        # Phase 2 may count it as free; phase 1 (allocated at once) may not.
        need = dict(pools=capacity.POOLS, free=(50, 4), phase1=(5, 3), phase2=(10, 9), headroom=(0, 0),
                    watermark=(1, 0))
        with self.assertRaises(capacity.CapacityWait):
            capacity._two_pool_gate(need, False)
        self.assertIs(capacity._two_pool_gate(dict(need, releasing=(0, 5)), False)['releasing'], (0, 5))
        with self.assertRaises(capacity.CapacityWait):
            capacity._two_pool_gate(dict(need, phase1=(5, 5), releasing=(0, 5)), False)

    def test_gdn_headroom_counts_the_slots_a_request_holds(self):
        """Under the hand-off a running request's GDN headroom is its peak (running +
        crossing) less the slots it holds: vLLM frees a crossing's old slot before
        it allocates the next, so a request holding both needs none more."""
        from remote_ra_live_v2 import capacity
        null, live = NS(is_null=True), NS(is_null=False)
        mamba = NS(req_to_blocks={'a': [null, live], 'b': [null, live, live], 'c': []})
        adapter = NS(page_tokens=784, decoupled=True, mamba_tokens=784,
                     kv_manager=NS(coordinator=NS(single_type_managers=[NS(), mamba, NS(), NS()])))
        requests = {rid: NS(num_computed_tokens=done, prompt_token_ids=list(range(1000)),
                            sampling_params=NS(max_tokens=1536)) for rid, done in (('a', 1200), ('b', 1600), ('c', 1000))}
        claims = {rid: dict(geometry=NS(phase2_start=1000)) for rid in requests}
        endpoint = NS(phase1=NS(groups=lambda: ((0,), (1, 2, 3), 4), handoff=None), claims=claims)
        lifecycle = NS(endpoint=NS(core=NS(_remote_ra_private_v2_endpoint=endpoint, _remote_ra_growing_adapter=adapter)),
                       entries={rid: dict(state='accepted', request=r) for rid, r in requests.items()})
        off = capacity.headroom_by_pool(lifecycle)
        self.assertEqual(off[1], 3 * (1 + 1 + (1 + 1)))     # a, b may cross; c has not run yet
        endpoint.phase1.handoff = object()
        on = capacity.headroom_by_pool(lifecycle)
        self.assertEqual(on, (off[0], 3 * (1 + 0 + 2)))      # b holds its crossing slot already

    @unittest.skipUnless(torch is not None, 'CPU torch required')
    def test_a_carry_buffer_is_written_only_after_its_last_copy(self):
        from remote_ra_live_v2.worker_backend import CudaWorkerBackend
        from remote_ra_live_v2.contract import Allocation, NativeIdentity
        io = TensorIssueTests('test_off_issues_nothing').world()
        log = []
        names = iter(['e%d' % k for k in range(20)])
        io._event = lambda: FakeEvent(log, next(names))
        io.issue_capture_into = (lambda real: lambda *args: (log.append(('capture', args[2] is backend.carry_buffers['s'][0][0])),
                                                             real(*args))[1])(io.issue_capture_into)
        backend = CudaWorkerBackend.__new__(CudaWorkerBackend)
        backend.io, backend.handoff, backend.carry_buffers, backend.final_events = io, True, {}, {}
        backend.carry, backend.latest = {}, {}
        backend._verify = lambda *args: None
        identity = NativeIdentity('r1', 's', 1, 'g', 'a' * 64, 'b' * 64, 'model')
        slot = (Allocation(1, 3, 1, 'mamba', 1),)
        backend._capture_carry(identity, slot, 10, 1)
        buffers = backend.carry_buffers['s']
        first = buffers[0][1]
        self.assertFalse(first.done)
        # Commit buffer 0 as the carry; phase 1 of the next turn restores from it (event e1).
        backend.carry['s'] = NS(payload=buffers[0][0])
        backend._buffer_event('s', buffers[0][0], io.issue_restore({1: (5,)}, 'hybrid', buffers[0][0]))
        backend._capture_carry(identity, slot, 20, 1)     # into buffer 1 (never used: no wait)
        self.assertEqual(log[-1], ('capture', False))
        self.assertNotIn(('synchronize', 'e0'), log)
        # Commit buffer 1; the next capture reuses buffer 0 only after its restore completed.
        backend.carry['s'] = NS(payload=buffers[1][0])
        reader = buffers[0][1]
        self.assertFalse(reader.done)
        backend._capture_carry(identity, slot, 30, 1)
        self.assertLess(log.index(('synchronize', reader.name)), len(log) - 1)
        self.assertEqual(log[-1], ('capture', True))
        self.assertTrue(reader.done)
        io.poll_joins()


if __name__ == '__main__':
    unittest.main()
