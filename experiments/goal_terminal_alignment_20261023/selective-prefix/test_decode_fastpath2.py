"""serve.py --decode-fastpath2 (remote_ra_live_v2.decode_fastpath2). CPU only.

Off: no config key, worker record, evidence key or vLLM argument; the code paths are
the former ones. On (with --decode-fastpath): the same tokens, selections and
per-request evidence as --decode-fastpath alone, checked piece by piece and end to
end over the decoupled multi-session fakes (test_decouple_e2e's MultiHarness at
P = 16 and 784, the legacy pool, the GDN hand-off and in-step phase 1): identical
replies, attended Pages, plans, computed positions and evidence files (every
native-*.json keeps one allocation_update frame per request per step), and the
allocation_batch RPCs, each entry expanded with the rows it stands for when it
was sent without rows, are exactly the per-request allocation_update RPCs of the
first fast path, payloads and receipts.
"""
from contextlib import ExitStack
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

from remote_ra_live_v2 import decode_fastpath as F
from remote_ra_live_v2 import decode_fastpath2 as F2
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.mswea_protocol import MarkerStop
# Imported here so the end-to-end runs' sys.modules patch (enter_case) keeps them
# (test_decode_fastpath explains why).
from remote_ra_live_v2 import (generation_guard, hooks, loaded_sources, native_probe,  # noqa: F401
                               page_compaction, phase1_step, positions, vector_space, worker_composition,
                               worker_rows)
from remote_ra_growing_worker import tensors  # noqa: F401

try:
    import torch
except ImportError:
    torch = None

BOTH = {'decode_fastpath': True, 'decode_fastpath2': True}


def attach_fast2(h):
    """attach_loaded_worker's and compose_engine's --decode-fastpath2 attachments on a harness: the
    harness's batch step allocates every request, then executes, so the queue is sent there (where
    scheduler.schedule returns)."""
    F2.attach_worker(h.owner, h.bridge, h.qwen, h.io)
    h.boundary.batched = {}
    h.proxy.before_call = h.boundary.flush_updates
    execute = h.runner.execute_model

    def flushing(scheduler_output):
        h.boundary.flush_updates()
        return execute(scheduler_output)
    h.runner.execute_model = flushing


def expand(trace):
    """The trace with every allocation_batch entry as the allocation_update RPC it stands for:
    its payload with the rows last sent for that request when it came without rows."""
    out, last = [], {}
    for item in trace:
        if item['operation'] != F2.BATCH:
            out.append(item)
            continue
        receipts = item['receipt']['receipts']
        assert item['receipt']['operation'] == F2.BATCH and len(receipts) == len(item['payload']['entries'])
        for entry, receipt in zip(item['payload']['entries'], receipts):
            payload = dict(entry)
            if payload['rows'] is None:
                payload['rows'] = last[payload['request']]
            last[payload['request']] = payload['rows']
            out.append(dict(operation='allocation_update', payload=payload, receipt=receipt))
    return out


class PinTests(unittest.TestCase):
    def test_pin_values(self):
        self.assertIsNone(F2.pinned({}))
        self.assertIsNone(F2.pinned(None))
        self.assertIsNone(F2.pinned({'decode_fastpath': True}))
        self.assertTrue(F2.pinned(dict(BOTH)))
        for bad in (False, 1, 'true', None):
            with self.assertRaises(LiveV2Error):
                F2.pinned({'decode_fastpath': True, 'decode_fastpath2': bad})
        with self.assertRaises(LiveV2Error):
            F2.pinned({'decode_fastpath2': True})          # needs the first fast path

    def test_worker_flag(self):
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertIsNone(F2.worker_flag({}))
            self.assertTrue(F2.worker_flag(dict(BOTH)))
            for bad in ({'decode_fastpath2': True}, {'decode_fastpath': True, 'decode_fastpath2': False}):
                with self.assertRaises(LiveV2Error):
                    F2.worker_flag(bad)
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'):
            with patch('remote_ra_http.config.load', lambda: dict(BOTH)):
                self.assertTrue(F2.worker_flag(dict(BOTH)))
                with self.assertRaises(LiveV2Error):
                    F2.worker_flag({'decode_fastpath': True})
            with patch('remote_ra_http.config.load', lambda: {'decode_fastpath': True}):
                self.assertIsNone(F2.worker_flag({'decode_fastpath': True}))
                with self.assertRaises(LiveV2Error):
                    F2.worker_flag(dict(BOTH))


class Allocator:
    """The overlay KVCacheManager.allocate_slots signature (vllm/v1/core/kv_cache_manager.py)."""

    def allocate(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                 num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                 num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0, has_scheduled_reqs=True,
                 external_computed_tokens_by_group=None, computed_tokens_by_group=None, skip_cache=False):
        return dict(locals())

    def starred(self, request, *args, **kwargs):
        return args, kwargs

    def keyword_only(self, request, *, count=1):
        return count


class BinderTests(unittest.TestCase):
    def test_binds_like_inspect(self):
        import inspect
        native = Allocator().allocate
        binder = F2.binder(native)
        names = binder.names
        rng = random.Random(4)
        for _ in range(3000):
            n = rng.randrange(0, len(names) + 2)
            positional = tuple(rng.randrange(9) for _ in range(n))
            keys = rng.sample(names + ('bogus',), rng.randrange(0, 4))
            kwargs = {k: rng.randrange(9) for k in keys}
            try:
                call = inspect.signature(native).bind(*positional, **kwargs)
                call.apply_defaults()
                expected = call.arguments
            except TypeError:
                expected = None
            got = binder.bind(positional, kwargs)
            if expected is None:
                self.assertIsNone(got)
                continue
            if got is None:                       # only inspect-valid calls may be bound
                self.fail((positional, kwargs))
            self.assertEqual(list(got.items()), list(expected.items()))
            self.assertEqual(native(*[got[k] for k in names]), native(*call.args, **call.kwargs))
        self.assertIs(F2.binder(native), binder)               # built once per function

    def test_unusual_callables_bind_through_inspect(self):
        allocator = Allocator()
        for method in (allocator.starred, allocator.keyword_only):
            binder = F2.binder(method)
            self.assertIsNone(binder.bind((1,), {}))
        self.assertIsNone(F2.binder(lambda request: None))       # not a bound method
        wrapped = Allocator()
        wrapped.allocate.__func__.__dict__['__wrapped__'] = None
        try:
            self.assertIsNone(F2.binder(wrapped.allocate))
        finally:
            del Allocator.allocate.__dict__['__wrapped__']


class MarkerStopCacheTests(unittest.TestCase):
    def test_cached_decisions_equal_the_full_decode(self):
        import test_decode_fastpath as T
        vocab, special = T.vocabulary()
        rng = random.Random(12)
        streams = [[rng.randrange(len(vocab)) for _ in range(rng.randrange(1, 60))] for _ in range(2000)]
        for tokens in streams[:150]:
            tokens[rng.randrange(len(tokens)):] = [7, 2, 3, 4, 5, 6]
        full, fast, cached = T.Decoder(vocab, special), T.Decoder(vocab, special), T.Decoder(vocab, special)
        play = T.MarkerStopTests.play
        with patch.dict(sys.modules, {'vllm.v1.request': NS(RequestStatus=NS(FINISHED_STOPPED='stopped'))}):
            for tokens in streams:
                expected = play(None, MarkerStop(full), tokens)
                self.assertEqual(play(None, MarkerStop(fast, incremental=True), tokens), expected)
                self.assertEqual(play(None, MarkerStop(cached, incremental=True, token_cache=True), tokens),
                                 expected)
        self.assertLess(cached.calls, fast.calls)

    def test_the_cache_answers_and_starts_over(self):
        decode = lambda ids: ''.join('>' if i % 3 == 0 else 'a' for i in ids)
        stop = MarkerStop(decode, incremental=True, token_cache=True)
        stop.TOKEN_CACHE = 4
        self.assertEqual([stop._closes(t) for t in range(10)], [t % 3 == 0 for t in range(10)])
        self.assertLessEqual(len(stop._gt), 4)
        with self.assertRaises(LiveV2Error):
            MarkerStop(decode, token_cache=True)              # a switch of the incremental check
        with self.assertRaises(LiveV2Error):
            MarkerStop(decode, incremental=True, token_cache=1)


def real_tokenizer():
    import test_decode_fastpath as T
    return T.real_tokenizer()


@unittest.skipUnless(real_tokenizer() is not None, 'transformers and the pinned tokenizer files required')
class RealTokenizerMarkerStopCacheTests(unittest.TestCase):
    def test_real_qwen_tokens(self):
        import test_decode_fastpath as T
        tok = real_tokenizer()
        decode = lambda ids: tok.decode(list(ids), skip_special_tokens=True)
        texts = ['THOUGHT: check the file.\n<mswea_bash_command>ls -la src/ && cat a->b >> c</mswea_bash_command>',
                 'Résumé 中文 → done: </mswea_bash_command> trailing',
                 'if a > b: print("<mswea_bash_command>")\n</mswea_bash_command',
                 'no marker here >>> >= -> <|im_end|> </mswea_bash_comman d>']
        rng = random.Random(6)
        cached = MarkerStop(decode, incremental=True, token_cache=True)     # one cache across streams
        with patch.dict(sys.modules, {'vllm.v1.request': NS(RequestStatus=NS(FINISHED_STOPPED='stopped'))}):
            for text in texts:
                ids = tok.encode(text, add_special_tokens=False)
                streams = [ids, ids + [tok.convert_tokens_to_ids('<|im_end|>')]]
                for _ in range(20):
                    cut = rng.randrange(1, len(ids))
                    streams.append(ids[cut:] + ids[:cut])
                for k, tokens in enumerate(streams):
                    expected = T.MarkerStopTests.play(None, MarkerStop(decode), tokens)
                    got = T.MarkerStopTests.play(None, cached, tokens, rid='s%d-%d' % (len(text), k))
                    self.assertEqual(got, expected, text)
        self.assertTrue(cached._gt)


class MeasuredBytesTests(unittest.TestCase):
    def test_group_bytes_give_the_same_active_bytes(self):
        from remote_ra_http.evidence import measured_command
        rng = random.Random(8)
        claims = {(g, b): ('r%d' % rng.randrange(4), 1) for g in range(4) for b in range(1, 300)
                  if rng.random() < .5}
        views = [NS(group=g, page_bytes=1000 + 7 * g + i) for i, g in enumerate((0, 0, 1, 2, 3, 3, 3))]
        out = []
        for per_group in (None, F2.group_bytes(views)):
            io = NS(views=views, handoff=False)
            if per_group is not None:
                io.group_bytes = per_group
            backend = NS(io=io, owner=NS(claims=F.ClaimTable(claims), records={}))
            for request in ('r0', 'r3', 'absent', None):
                with patch('time.monotonic_ns', itertools.count(1).__next__):
                    out.append(measured_command(backend, 'allocation_update',
                                                lambda b, q: dict(operation='allocation_update'), None,
                                                request=request))
        self.assertEqual(out[:4], out[4:])
        self.assertTrue(all(r['http_measurement']['active_owned_kv_bytes'] for r in out[:2]))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    """--decode-fastpath2 vs --decode-fastpath alone over test_decouple_e2e's multi-session harness."""
    maxDiff = None

    def serve(self, case, policy, fast2, *, handoff=False, in_step=False):
        import test_decouple_e2e as D
        import test_host_trim as HT
        multi = D.Case(**dict(vars(case), sessions=len(D.TASKS)))
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = dict(D.config(policy, output, multi, D.TASKS), decode_fastpath=True)
            flags = {}
            if in_step:
                flags['phase1_in_step'] = True
            if handoff:
                flags['gdn_device_handoff'] = True
            if flags:
                value['admission'] = flags
            if fast2:
                value['decode_fastpath2'] = True
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            if in_step:
                import test_adm_phase1_step as A
                h = A.in_step_harness(handoff)(policy, output, multi)
            elif handoff:
                import test_gdn_handoff as G
                h = G.handoff_harness(D.MultiHarness)(policy, output, multi)
            else:
                h = D.MultiHarness(policy, output, multi)
            F.attach_worker(h.owner, h.bridge, h.qwen)
            finishes = []
            if fast2:
                attach_fast2(h)
                finish = h.bridge.finish_step
                h.bridge.finish_step = lambda: finishes.append(1) or finish()
            drains, drain = [], h.io.drain
            h.io.drain = lambda: (drains.append(1), drain())[1]
            trace, rpc = [], h.collective_rpc

            def collective_rpc(method, args, timeout):
                operation, payload = args
                sent = HT.plain(payload)
                reply = rpc(method, args, timeout)
                trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
                return reply
            h.proxy.executor = NS(collective_rpc=collective_rpc)
            h.active = h.restores = 0
            h.serve()
            frames = {rid: HT.plain(items) for rid, items in h.proxy.http_receipts.items()}
            h.close()
            D.engine_closed(h.core)
            files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
            return dict(trace=trace, files=files, frames=frames, computed=list(h.runner.computed),
                        visible=dict(h.runner.visible), plans=dict(h.runner.plans),
                        done=[(i['rid'], list(i['request'].output_token_ids), list(i['plan']['selected']))
                              for i in h.done],
                        widths=list(h.widths), drains=len(drains), finishes=len(finishes))

    def compare(self, fast, fast2, decoupled):
        for key in ('files', 'frames', 'computed', 'visible', 'plans', 'done', 'widths'):
            self.assertEqual(fast2[key], fast[key], key)
        self.assertEqual(expand(fast2['trace']), fast['trace'])
        batches = [t for t in fast2['trace'] if t['operation'] == F2.BATCH]
        updates = [t for t in fast['trace'] if t['operation'] == 'allocation_update']
        self.assertTrue(batches and not any(t['operation'] == 'allocation_update' for t in fast2['trace']))
        self.assertEqual(sum(len(t['payload']['entries']) for t in batches), len(updates))
        self.assertLess(len(batches), len(updates))                 # several requests per batch
        unchanged = sum(e['rows'] is None for t in batches for e in t['payload']['entries'])
        if decoupled:
            self.assertGreater(unchanged, len(updates) // 2)         # most steps add no block
        self.assertGreater(max(fast['widths']), 1)
        self.assertGreater(fast2['finishes'], 0)
        self.assertEqual(fast['finishes'], 0)

    def test_fast2_equals_fast(self):
        import test_decouple_e2e as D
        # Default suite: full on legacy-784 and observation_query at P = 16; the slow pass
        # (README section 2.3) every case with both.
        for case in D.CASES:
            for policy in ('full', 'observation_query'):
                if not (D.SLOW or (case.name, policy) in (('legacy-784', 'full'), ('16-784', 'observation_query'))):
                    continue
                with self.subTest(case=case.name, policy=policy):
                    fast = self.serve(case, policy, False)
                    fast2 = self.serve(case, policy, True)
                    self.compare(fast, fast2, case.decoupled)
                    self.assertLess(fast2['drains'], fast['drains'])  # leave_execute joined by the read

    def test_with_the_gdn_hand_off_and_in_step_phase1(self):
        import test_decouple_e2e as D
        for handoff, in_step in ((True, False), (False, True), (True, True)):
            with self.subTest(handoff=handoff, in_step=in_step):
                fast = self.serve(D.CASES[2], 'goal_query', False, handoff=handoff, in_step=in_step)
                fast2 = self.serve(D.CASES[2], 'goal_query', True, handoff=handoff, in_step=in_step)
                self.compare(fast, fast2, True)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class FailureTests(unittest.TestCase):
    def test_a_refused_batch_poisons_its_sessions_and_raises(self):
        from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
        sessions = {'a': NS(poisoned=False), 'b': NS(poisoned=False)}
        records = {r: dict(native_allocation={}) for r in sessions}

        def boundary(reply):
            b = object.__new__(EngineRpcBoundary)
            b.adapter = NS(requests=sessions)
            b.batched = {r: (records[r], NS(request=r, identity='bind-' + r), 5, 1, [[]], dict(end=6))
                         for r in ('a', 'b')}
            sent = []

            def call(operation, payload):
                sent.append((operation, payload))
                if isinstance(reply, Exception):
                    raise reply
                return reply
            b.proxy = NS(call=call)
            return b, sent
        for reply in (dict(operation=F2.BATCH, receipts=[dict(operation='allocation_update', binding='bind-a')]),
                      dict(operation='other', receipts=[]), RuntimeError('worker refused')):
            for s in sessions.values():
                s.poisoned = False
            b, _ = boundary(reply)
            with self.assertRaises((LiveV2Error, RuntimeError)):
                b.flush_updates()
            self.assertTrue(all(s.poisoned for s in sessions.values()))
            self.assertNotIn('scheduled_cursor', records['a'])         # nothing advanced
            self.assertEqual(b.batched, {})
        good = dict(operation=F2.BATCH, receipts=[dict(operation='allocation_update', binding='bind-' + r)
                                                  for r in ('a', 'b')])
        for s in sessions.values():
            s.poisoned = False
        b, sent = boundary(good)
        self.assertIs(b.flush_updates(), good)
        self.assertEqual(sent, [(F2.BATCH, dict(entries=[
            dict(request=r, binding='bind-' + r, start=5, count=1, rows=[[]]) for r in ('a', 'b')]))])
        self.assertEqual([records[r]['scheduled_cursor'] for r in ('a', 'b')], [6, 6])
        self.assertEqual([records[r]['accepted_rows'] for r in ('a', 'b')], [[[]], [[]]])
        self.assertFalse(any(s.poisoned for s in sessions.values()))
        self.assertIsNone(b.flush_updates())                           # nothing queued: no RPC
        self.assertEqual(len(sent), 1)

    def test_a_second_allocation_of_a_request_sends_the_queue_first(self):
        from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
        boundary = object.__new__(EngineRpcBoundary)
        boundary.batched = {'r': ('queued',)}
        flushed = []

        def flush():
            flushed.append(dict(boundary.batched))
            raise RuntimeError('stop here')
        boundary.flush_updates = flush
        with self.assertRaises(RuntimeError):
            boundary.dispatch_mixed_request(NS(request='r'), {}, None, None, (), {})
        self.assertEqual(flushed, [{'r': ('queued',)}])
        boundary.batched = {'other': ('queued',)}
        boundary.prepared = {}
        with self.assertRaises(KeyError):                              # not flushed: on to the record
            boundary.dispatch_mixed_request(NS(request='r'), {}, None, None, (), {})
        self.assertEqual(len(flushed), 1)

    def test_the_worker_refuses_batches_and_rowless_updates_without_the_switch(self):
        owner = NS(decode_fastpath2=False)
        with self.assertRaises(LiveV2Error):
            F2.dispatch_batch(NS(_remote_ra_live_v2_qwen=None), NS(io=NS()), owner, dict(entries=[]))
        owner = NS(decode_fastpath2=True)
        for payload in ({}, dict(entries=[]), dict(entries=[dict(request='a')]),
                        dict(entries=[dict(request='a', binding='x', start=0, count=1, rows=None)] * 2),
                        dict(entries=[], extra=1)):
            with self.assertRaises(LiveV2Error):
                F2.dispatch_batch(NS(_remote_ra_live_v2_qwen=None), NS(io=NS()), owner, payload)
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        plain = object.__new__(RunnerAllocationOwner)
        rec = dict(native_rows=((),), rows_seen=object(), identity=NS(request='a'))
        update = plain._allocation_update(rec, dict(rows=None, start=0, count=1, binding='x'))
        with self.assertRaises(LiveV2Error):
            update(None, NS(requests={}))

    def test_the_schedule_wrapper_sends_the_queue_even_when_schedule_raises(self):
        sent = []
        boundary = NS(batched=None, flush_updates=lambda: sent.append(1))
        proxy = NS(before_call=None)

        def schedule(throttle=False):
            if throttle:
                raise RuntimeError('refused')
            return 'output'
        core = NS(scheduler=NS(schedule=schedule))
        F2.attach_engine(core, boundary, proxy)
        self.assertEqual((boundary.batched, proxy.before_call), ({}, boundary.flush_updates))
        self.assertEqual(core.scheduler.schedule(), 'output')
        with self.assertRaises(RuntimeError):
            core.scheduler.schedule(True)
        self.assertEqual(sent, [1, 1])
        with self.assertRaises(LiveV2Error):
            F2.attach_engine(core, boundary, proxy)                     # once

    def test_a_non_finite_row_is_refused_as_before_and_the_step_is_dropped(self):
        from remote_ra_live_v2.worker_composition import WorkerBridge
        closed = []

        class Qwen:
            backend = NS(handoff=False, io=NS())
            step_joined = False

            def after_forward(self, identity, rows, finite=None, drained=False, finite_ok=False):
                closed.append((identity, finite_ok, None if finite is None else finite.tolist()))
                if not finite_ok and not bool(finite.all()):
                    raise LiveV2Error('exact finite unpadded hidden rows required')
                return identity
        bridge = WorkerBridge(Qwen(), owner=NS(scheduled_rows=lambda runner, so: (('a', 0, 0, 1), ('b', 1, 1, 2))))
        bridge.assignments = {r: NS(assignment=NS(identity=r)) for r in 'ab'}
        bridge.decode_fastpath2 = True
        hidden = torch.ones(3, 4)
        with patch('remote_ra_live_v2.native_probe.mixed_probe', lambda runner: None):
            self.assertIsNone(bridge.after_forward(NS(), NS(), hidden))
            self.assertEqual(bridge.finish_step(), ('a', 'b'))
            self.assertEqual(closed, [('a', True, None), ('b', True, None)])
            self.assertTrue(bridge.qwen.step_joined)
            hidden[2, 1] = float('nan')
            closed.clear()
            bridge.qwen.step_joined = False
            bridge.after_forward(NS(), NS(), hidden)
            with self.assertRaises(LiveV2Error):
                bridge.finish_step()
            self.assertEqual(closed, [('a', False, [True]), ('b', False, [True, False])])
            self.assertFalse(bridge.qwen.step_joined)
            self.assertIsNone(bridge.finish_step())                    # taken once
            bridge.after_forward(NS(), NS(), hidden)
            bridge.discard_step()
            self.assertIsNone(bridge.take_deferred())


class LauncherTests(unittest.TestCase):
    def test_switch_pins_one_more_config_key_and_changes_no_vllm_argument(self):
        import contextlib
        import io
        import install
        import serve
        import test_installer as INSTALL
        from remote_ra_http import config as http_config
        tasks = [dict(instance_id='repo__a-1', problem_statement='Fix a.'),
                 dict(instance_id='repo__b-2', problem_statement='Fix b.')]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            kw = dict(kv_cache_gib=24, page_tokens=16, cuda_graphs='full', max_model_len=131072)
            one = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'one', 8231, decode_fastpath=True, **kw)
            two = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'two', 8231, decode_fastpath=True,
                                decode_fastpath2=True, **kw)
            off = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'off', 8231, decode_fastpath2=False, **kw)
            with self.assertRaises(ValueError):
                serve.prepare(release, [dict(t) for t in tasks], 'full', root/'bad', 8231, decode_fastpath2=True, **kw)
            with self.assertRaises(ValueError):
                serve.prepare(release, [dict(t) for t in tasks], 'full', root/'bad', 8231, decode_fastpath=True,
                              decode_fastpath2=1, **kw)
            for name, config, expected in (('two', two[1], True), ('one', one[1], None)):
                path = root/(name + '.json')
                raw = serve.wire(config)
                path.write_bytes(raw)
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.digest(raw)):
                    self.assertEqual(http_config.load().get('decode_fastpath2'), expected)
                    self.assertEqual(F2.loaded(), expected)
            for bad in (dict(two[1], decode_fastpath2=False), {k: v for k, v in two[1].items() if k != 'decode_fastpath'}):
                path, raw = root/'bad.json', serve.wire(bad)
                path.write_bytes(raw)
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.digest(raw)):
                    with self.assertRaises(RuntimeError):
                        http_config.load()
        strip = lambda value, name: json.loads(json.dumps(value).replace(str(root/name).replace('\\', '\\\\'), '<out>'))
        self.assertEqual(strip(two[2], 'two'), strip(one[2], 'one'))                  # vLLM argv unchanged
        self.assertEqual({k: v for k, v in strip(two[1], 'two').items() if k != 'decode_fastpath2'},
                         strip(one[1], 'one'))
        self.assertIs(two[1]['decode_fastpath2'], True)
        self.assertNotIn('decode_fastpath2', one[1])
        self.assertNotIn('decode_fastpath2', off[1])
        output = ('REMOTE_RA_HTTP_CONFIG', 'REMOTE_RA_HTTP_CONFIG_SHA256')
        self.assertEqual({k: v for k, v in two[3].items() if k not in output},
                         {k: v for k, v in one[3].items() if k not in output})

    def test_attach_loaded_worker_echoes_the_switch_and_attaches_it_first(self):
        import test_cudagraph as CG
        from remote_ra_live_v2 import worker_composition as WC

        class Owner(CG.FakeOwner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.claims = {}
        tests = CG.AttachLoadedWorkerTests()
        with patch.object(CG, 'FakeOwner', Owner), patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            one, _, _ = tests.attach(None, {'decode_fastpath': True})
            two, runner, _ = tests.attach(None, dict(BOTH))
            with self.assertRaises(LiveV2Error):
                tests.attach(None, {'decode_fastpath2': True})
        self.assertEqual(dict(two), dict(one, decode_fastpath2=True))
        owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
        self.assertTrue(owner.decode_fastpath2 and bridge.decode_fastpath2 and bridge.qwen.decode_fastpath2)
        self.assertIsInstance(owner.claims, F.ClaimTable)
        self.assertEqual(owner.io.group_bytes, F2.group_bytes(owner.io.views))
        self.assertIs(WC.WorkerBridge.decode_fastpath2, False)

    def test_compose_engine_sends_the_pin_once_and_checks_the_worker_echo(self):
        import contextlib
        import io
        import install
        import serve
        import test_installer as INSTALL
        from remote_ra_live_v2 import composition
        task = dict(instance_id='repo__a-1', problem_statement='Fix a.')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            config = serve.prepare(release, dict(task), 'full', root/'ce', 8231, decode_fastpath=True,
                                   decode_fastpath2=True)[1]
            path = root/'ce.json'
            path.write_bytes(serve.wire(config))
            for echo in ({}, {'decode_fastpath': True}):
                sent = []

                def collective_rpc(method, args, timeout):
                    sent.append(args[0])
                    return [dict(echo)]
                core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                          _remote_ra_trusted_issue=dict(task), _remote_ra_trusted_issues=[dict(task)],
                          _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                          _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                    with self.assertRaises(LiveV2Error) as caught:
                        composition.compose_engine(core, object())
                startup, = sent
                self.assertEqual(startup, dict(enabled=True, source='m' * 64, page_tokens=784,
                                               loaded_sources_verified=True, **BOTH))
                self.assertIn('decode_fastpath2' if echo else 'decode_fastpath', str(caught.exception))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ProfiledFastPath2Tests(unittest.TestCase):
    """The step profile on top of both fast paths: the outputs of the run without it, the
    per-step batch RPC and the deferred after-forward work under their own names."""

    def serve(self, case, prof):
        import test_decouple_e2e as D
        import test_host_trim as HT
        import test_step_profile as TSP
        import remote_ra_step_profile as SP
        from remote_ra_live_v2.generation_guard import guarded_execute
        multi = D.Case(**dict(vars(case), sessions=len(D.TASKS)))
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = dict(D.config('full', output, multi, D.TASKS), **BOTH)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            stack.enter_context(patch.object(SP, 'ENABLED', prof is not None))
            stack.enter_context(patch.object(SP, 'PROFILER', prof))

            def guard(runner, execute):
                # attach_loaded_worker's order: the switch first, then the (profiled) guard.
                bridge = runner._remote_ra_live_v2_bridge
                F2.attach_worker(runner._remote_ra_live_v2_allocation_owner, bridge, bridge.qwen,
                                 runner._remote_ra_live_v2_allocation_owner.io)
                if prof is None:
                    return guarded_execute(runner, execute)
                return SP.install_worker(runner, bridge, guarded_execute)(runner, execute)
            stack.enter_context(patch.object(D, 'guarded_execute', guard))
            if prof is not None:
                stack.callback(SP.uninstall, prof)
            h = (D.MultiHarness if prof is None else TSP.ProfiledMultiHarness)('full', output, multi)
            F.attach_worker(h.owner, h.bridge, h.qwen)
            h.boundary.batched = {}
            h.proxy.before_call = h.boundary.flush_updates
            execute = h.runner.execute_model
            h.runner.execute_model = lambda so: (h.boundary.flush_updates(), execute(so))[1]
            if prof is not None:
                SP.install_engine(h.core, prof)
                SP.wrap(h, 'admit', 'input.registered', scope=True, profiler=prof)
                SP.wrap(h, 'finish', 'sched.complete', profiler=prof)
            h.serve()
            if prof is not None:
                prof.flush()
            h.close()
            D.engine_closed(h.core)
            files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
        return dict(files=files, done=[(i['rid'], list(i['request'].output_token_ids)) for i in h.done],
                    widths=list(h.widths))

    def test_profile_on_both_fast_paths(self):
        import test_decouple_e2e as D
        import test_step_profile as TSP
        import remote_ra_step_profile as SP
        off = self.serve(D.CASES[2], None)
        lines = []
        on = self.serve(D.CASES[2], SP.Profiler(every=10, emit=lines.append))
        self.assertEqual(on, off)
        values = TSP.parse(lines)
        self.assertEqual([v['errors'] for v in values], [0] * len(values))
        total = values[-1]
        self.assertEqual(total['kind'], 'total')
        names = {name for g in total['groups'] for name in g['c']}
        for name in ('rpc.allocation_batch', 'rpc.allocation_batch.worker', 'rpc.allocation_batch.worker.command',
                     'worker.finish', 'worker.finish.finite', 'worker.finish.qwen', 'worker.finish.qwen.drain',
                     'worker.finish.qwen.compact'):
            self.assertIn(name, names)
        for name in ('rpc.allocation_update', 'worker.after.qwen', 'worker.after.finite'):
            self.assertNotIn(name, names)
        parent = dict(SP.TREE)
        for g in total['groups']:
            c = g['c']
            for name, stats in c.items():
                up = parent.get(name)
                if up is not None and up in c and name not in ('gpu.forward', 'gc'):
                    self.assertLessEqual(stats[2], c[up][2] + 1e-6, (name, up))
        decode = [g for g in total['groups'] if g['phase'] == 'decode']
        self.assertTrue(decode)
        for g in decode:
            if 'rpc.allocation_batch' in g['c']:
                self.assertEqual(g['c']['rpc.allocation_batch'][1], g['c']['rpc.allocation_batch'][0])
                self.assertEqual(g['c']['rpc.allocation_batch.worker'][1], g['b'] * g['steps'])


if __name__ == '__main__':
    unittest.main()
