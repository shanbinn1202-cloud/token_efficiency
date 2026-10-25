"""Gemma 4 with the decode fast paths (gemma-perf): --decode-fastpath and --decode-fastpath2 on the window
layout. CPU only.

The switch is Qwen's (README 4.11, test_decode_fastpath): claim counts per request, one
position read and one device drain per step, the incremental marker check. On the window
layout it also turns on the window owner's incremental observation
(window_owner.attach_fastpath): a step extends the last observation when the full one
would accept the same rows, so the full-attention leases grow by the appended blocks, each
sliding group's leases by the blocks that entered its window minus those that left it, the
claims and allocation sets by those differences, and an unchanged sliding lease tuple is
neither rebuilt nor re-checked.

Checked over test_gemma_e2e's engine and worker (vLLM's KV manager with the sliding
managers, a runner that keeps its rows as vLLM's does and checks every full-attention and
sliding read): three sessions, every policy (REMOTE_RA_SLOW_TESTS=1; observation_query on a
shorter script in the archive suite), Page-chunked and forward-chunked prefill, batched
decode (goal_query's too: its layout reserves the Goal forward's transient) and a rebuild,
with the switch off and on: identical RPC payloads and receipts, evidence files,
per-request frames, computed positions, attended Pages, plans, replies and sliding reads,
and per step identical leases (full-attention, sliding and their keys) and claims. With the
switch most observations are extended, not rebuilt. A runner row that differs inside the
window is refused with the switch too; its entries before the window are not read and not
checked. Launcher, startup check and worker attach: the pin as
for Qwen, nothing else changes.

--decode-fastpath2 (README 4.11, test_decode_fastpath2) adds Qwen's one allocation_batch RPC per
step and the after-forward work behind the logits, and on the window layout the incremental
sliding rows (window_rows): a request's changed rows go as a delta of the rows the worker
accepted last, the worker keeps its sliding rows compact. Against --decode-fastpath alone on
the same script: identical evidence files, frames, positions, attended Pages, plans, replies,
sliding reads, per-step leases and claims, and the batches, each entry expanded with the rows
it stands for (unchanged: the last rows; a delta: applied to them), equal the per-request
allocation_update RPCs of the first fast path, payloads and receipts; the deltas carry a
small part of the pairs the full rows carry.
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
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_gemma_e2e as GE
import test_host_trim as HT
import test_stageb_worker as W
from test_gemma_install import gemma_fixture
from remote_ra_http import config as HC
from remote_ra_http.evidence import engine_closed
from remote_ra_live_v2 import decode_fastpath as F
from remote_ra_live_v2 import decode_fastpath2 as F2
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import window_owner as WO
from remote_ra_live_v2.contract import LiveV2Error
# Imported here, outside any sys.modules patch, so the patches the tests enter keep them
# (a module first imported inside patch.dict(sys.modules) leaves sys.modules at its exit
# while its package attribute stays; test_decode_fastpath explains it).
from remote_ra_live_v2 import (admission_flags, connector_guard, context_limits, decode_fastpath,  # noqa: F401,E402
                               decode_fastpath2, generation_guard, idle_release, language_model_only,
                               loaded_sources, native_probe, page_compaction, vector_space, window_lifecycle,
                               window_owner, window_worker, worker_composition)
# Imported here so the runs' sys.modules patch keeps them (test_decode_fastpath explains why).
from remote_ra_live_v2 import (generation_guard, hooks, native_probe, page_compaction, positions,  # noqa: F401
                               vector_space, worker_composition, worker_rows)

try:
    import torch
except ImportError:
    torch = None

P, WINDOW, N, SLIDING = GE.P, GE.WINDOW, GE.N, GE.SLIDING
TASKS = (GE.A, GE.B, GE.C)
POLICIES = ('full', 'random50', 'observation_query', 'random_fill', 'goal_query')
# Every policy with the whole script: REMOTE_RA_SLOW_TESTS=1 (about 40 s); the archive suite
# runs observation_query on a shorter script.
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
SHORT = dict(rounds=2, together=1)        # the archive suite's script
_RUNS = {}                                # (policy, switches, rounds, together) -> a run's result


def attach_fast(h):
    """attach_window_worker's --decode-fastpath attachments on a harness."""
    F.attach_worker(h.owner, h.bridge, h.qwen)
    WO.attach_fastpath(h.owner, h.qwen)


def attach_fast2(h):
    """attach_window_worker's --decode-fastpath2 attachments and compose_engine's (the queue is sent
    when the harness executes, where scheduler.schedule returns)."""
    attach_fast(h)
    F2.attach_worker(h.owner, h.bridge, h.qwen, h.io)
    WO.attach_fastpath2(h.owner)
    h.boundary.batched = {}
    h.proxy.before_call = h.boundary.flush_updates
    execute = h.runner.execute_model

    def flushing(scheduler_output):
        h.boundary.flush_updates()
        return execute(scheduler_output)
    h.runner.execute_model = flushing


def apply_delta(rows, items):
    """The full rows a window delta stands for (window_rows.grow's form)."""
    out = []
    for row, item in zip(rows, items):
        if item is None:
            out.append(row)
            continue
        first, length, appended = item
        new = [[0, 0]] * first + row[first:] + appended
        assert len(new) == length, (len(new), length)
        out.append(new)
    return out


def expand(trace):
    """The trace with every allocation_batch entry as the allocation_update RPC it stands for, and
    the (full, delta, unchanged) entries and the pairs every entry carried."""
    out, last, kinds, pairs = [], {}, dict(full=0, delta=0, unchanged=0), 0
    for item in trace:
        if item['operation'] != F2.BATCH:
            out.append(item)
            continue
        for entry, receipt in zip(item['payload']['entries'], item['receipt']['receipts']):
            payload = dict(entry)
            rows = payload['rows']
            if rows is None:
                kinds['unchanged'] += 1
                rows = last[payload['request']]
            elif type(rows) is dict:
                kinds['delta'] += 1
                pairs += sum(len(i[2]) for i in rows['window'] if i is not None)
                rows = apply_delta(last[payload['request']], rows['window'])
            else:
                kinds['full'] += 1
                pairs += sum(len(row) for row in rows)
            payload['rows'] = last[payload['request']] = rows
            out.append(dict(operation='allocation_update', payload=payload, receipt=receipt))
    return out, kinds, pairs


def keys(allocations):
    return [list(v.key) for v in allocations]


class Run:
    """One server run of a script over test_gemma_e2e's Harness; everything it sent, kept and wrote."""

    def __init__(self, test, policy, flags, *, attach=None, tamper=None):
        self.test, self.policy, self.flags, self.attach, self.tamper = test, policy, flags, attach, tamper

    def __enter__(self):
        stack = self.stack = ExitStack()
        HT.deterministic(stack)
        output = self.output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        # The harness's layout: goal_query's also reserves the Goal forward's transient.
        layout = GE.window_layout(self.policy, len(TASKS))
        value = dict(GE.config(self.policy, output, list(TASKS), layout), **self.flags)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        if self.tamper is not None:
            stack.enter_context(self.tamper)
        h = self.h = GE.Harness(self.policy, output, list(TASKS))
        if self.attach is not None:
            self.attach(h)
        self.trace, rpc = [], h.collective_rpc

        def collective_rpc(method, args, timeout):
            operation, payload = args
            sent = HT.plain(payload)
            reply = rpc(method, args, timeout)
            self.trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        self.steps, self.extended, step = [], [], h.owner.step

        def observed_step(runner, scheduler_output):
            rows = step(runner, scheduler_output)
            self.steps.append(([(s.request, s.start, s.count, keys(s.attention), keys(s.hybrid),
                                 [list(k) for k in s.current]) for s in rows],
                               sorted((list(k), list(v)) for k, v in h.owner.claims.items())))
            return rows
        h.owner.step = observed_step
        increment = getattr(h.owner, '_observe_increment_window', None)
        if increment is not None:
            def counted(*args):
                s = increment(*args)
                self.extended.append(s is not None)
                return s
            h.owner._observe_increment_window = counted
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def script(self, rounds=4, together=3):
        """Turns one session at a time (Page and forward chunks), then rounds decoded together, a rebuild."""
        h, infos = self.h, []
        for k in range(rounds):
            for lane in h.lanes:
                observation = None if k == 0 else GE.OBSERVATIONS[(k - 1 + lane.index) % len(GE.OBSERVATIONS)]
                infos.append(h.turn(lane, observation, target=GE.TARGETS[(k + lane.index) % len(GE.TARGETS)],
                                    eos=(k + lane.index) % 3 == 2, chunk='forward' if (k + lane.index) % 2 else 'page'))
        for k in range(together):
            batch = []
            for lane in h.lanes:
                observation = GE.OBSERVATIONS[(k + 2 * lane.index) % len(GE.OBSERVATIONS)]
                messages = lane.messages + [dict(role='user', content=observation)]
                info = h.submit(lane, messages)
                info['outputs'] = h.reply(info, GE.TARGETS[(k + lane.index + 3) % len(GE.TARGETS)], False)
                request = info['request']
                h.runner.samples[request.request_id] = iter(info['outputs'])
                first = True
                while request.num_computed_tokens < len(request.prompt_token_ids):
                    start = request.num_computed_tokens
                    h.step(request, min(len(request.prompt_token_ids), (start // P + 1) * P) - start, first)
                    first = False
                batch.append(info)
            h.decode_together(batch)
            for info in batch:
                h.complete(info)
                info['lane'].messages = info['messages'] + [dict(role='assistant',
                                                                 content=' ' + info['committed']['text'] + '\n')]
                infos.append(info)
        lane = h.lanes[1]
        retry = lane.messages[:-1] + [dict(role='user', content='Format error: expected exactly one action.')]
        infos.append(h.turn(lane, messages=retry, target=3))
        infos.append(h.turn(lane, GE.OBSERVATIONS[5], target=9))
        self.infos = infos
        return self

    def result(self):
        h = self.h
        frames = {rid: HT.plain(items) for rid, items in h.proxy.http_receipts.items()}
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        files = {p.name: p.read_bytes() for p in sorted(self.output.iterdir())}
        windows = {r: n for r, n in h.runner.windows.items()}
        return dict(trace=self.trace, files=files, frames=frames, computed=list(h.runner.computed),
                    visible=dict(h.runner.visible), plans=dict(h.runner.plans), windows=windows,
                    done=[(i['rid'], list(i['request'].output_token_ids), list(i['plan']['selected']))
                          for i in self.infos],
                    batched=list(h.batched), steps=self.steps, extended=self.extended)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    maxDiff = None
    COMPARED = ('trace', 'files', 'frames', 'computed', 'visible', 'plans', 'windows', 'done', 'batched', 'steps')

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def serve(self, policy, fast, rounds=4, together=3):
        """One run of the script with no switch (0), --decode-fastpath (1) or both (2); deterministic
        (test_host_trim.deterministic), so the tests share each run."""
        key = (policy, int(fast), rounds, together)
        if key not in _RUNS:
            flags = ({}, dict(decode_fastpath=True), dict(decode_fastpath=True, decode_fastpath2=True))[int(fast)]
            attach = (None, attach_fast, attach_fast2)[int(fast)]
            with Run(self, policy, flags, attach=attach) as run:
                _RUNS[key] = run.script(rounds=rounds, together=together).result()
        return _RUNS[key]

    def compare(self, off, on):
        for key in self.COMPARED:
            self.assertEqual(on[key], off[key], key)

    def test_on_equals_off(self):
        for policy in POLICIES if SLOW else ('observation_query',):
            with self.subTest(policy=policy):
                script = {} if SLOW else SHORT
                off, on = self.serve(policy, False, **script), self.serve(policy, True, **script)
                self.compare(off, on)
                self.assertFalse(off['extended'])
                # Most observations extend the last one; the first step of a request and a
                # change of phase (prefill to decode) observe in full.
                self.assertGreater(sum(on['extended']), 0.6 * len(on['extended']))
                # Every policy decodes the three sessions together (goal_query too: its layout
                # reserves the Goal forward's own transient).
                self.assertTrue(any(len(step) == 3 for step in on['batched']))
                self.assertTrue(any(n > 2 * WINDOW for n in on['windows'].values()))

    def test_a_runner_row_is_checked_on_its_window_with_the_switch(self):
        fired = []

        def tampered(where, count=40):
            calls, real = [0], GE.Runner.set_rows

            def edit(runner, rid, block_ids):
                real(runner, rid, block_ids)
                calls[0] += 1
                req = runner.requests[rid]
                row = req.block_ids[SLIDING[1]]
                first = max(0, req.num_computed_tokens - WINDOW + 1) // P
                if calls[0] >= count and (first > 0 or where == 'window'):
                    row[first - 1 if where == 'stale' else len(row) - 1] = 99999 + calls[0]
                    fired.append(where)
                    if rid in runner.input_batch.req_ids:
                        t = runner.input_batch.block_table.block_tables[SLIDING[1]]
                        t.add_row(list(row), runner.input_batch.req_ids.index(rid))
            return patch.object(GE.Runner, 'set_rows', edit)
        with Run(self, 'observation_query', dict(decode_fastpath=True), attach=attach_fast,
                 tamper=tampered('stale')) as run:
            stale = run.script(**SHORT).result()
        reference = self.serve('observation_query', False, **SHORT)
        self.assertGreater(fired.count('stale'), 10)
        for key in ('done', 'visible', 'windows', 'computed'):
            self.assertEqual(stale[key], reference[key], key)
        with Run(self, 'observation_query', dict(decode_fastpath=True), attach=attach_fast,
                 tamper=tampered('window')) as run:
            with self.assertRaisesRegex(Exception, 'runner sliding window differs'):
                run.script(**SHORT)
        self.assertEqual(fired.count('window'), 1)

    def test_fast2_equals_fast(self):
        policies = ('full', 'observation_query', 'random_fill') if SLOW else ('observation_query',)
        for policy in policies:
            with self.subTest(policy=policy):
                script = {} if SLOW else SHORT
                fast, fast2 = self.serve(policy, 1, **script), self.serve(policy, 2, **script)
                for key in self.COMPARED:
                    if key != 'trace':
                        self.assertEqual(fast2[key], fast[key], key)
                trace, kinds, pairs = expand(fast2['trace'])
                self.assertEqual(trace, fast['trace'])
                updates = [t for t in fast['trace'] if t['operation'] == 'allocation_update']
                self.assertFalse(any(t['operation'] == 'allocation_update' for t in fast2['trace']))
                self.assertEqual(sum(kinds.values()), len(updates))
                self.assertTrue(kinds['delta'] and kinds['unchanged'] and kinds['full'])
                # A full update is a request's first; every later changed one is a delta.
                self.assertEqual(kinds['full'], len({t['payload']['request'] for t in updates}))
                full_pairs = sum(len(row) for t in updates for row in t['payload']['rows'])
                self.assertLess(5 * pairs, full_pairs)

    @unittest.skipUnless(SLOW, 'about 4 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_full_observation_when_incremental_is_off(self):
        # allocation_owner.INCREMENTAL False (the A/B switch) observes every step in full.
        from remote_ra_live_v2 import allocation_owner
        with patch.object(allocation_owner, 'INCREMENTAL', False):
            with Run(self, 'observation_query', dict(decode_fastpath=True), attach=attach_fast) as run:
                full = run.script(rounds=2, together=1).result()
        with Run(self, 'observation_query', dict(decode_fastpath=True), attach=attach_fast) as run:
            extended = run.script(rounds=2, together=1).result()
        self.assertFalse(full['extended'])
        self.assertTrue(any(extended['extended']))
        for key in self.COMPARED:
            self.assertEqual(extended[key], full[key], key)


class LeaseReuseTests(unittest.TestCase):
    def test_extending_builds_the_same_window_lease(self):
        from remote_ra_live_v2.contract import Allocation, NativeIdentity
        from remote_ra_live_v2.window_lifecycle import WindowLease
        identity = NativeIdentity('r', 's', 1, 'g', 'a' * 64, 'b' * 64, 'm')
        slots = tuple(Allocation(g, 10 * g + i, 2, 'sliding', 1) for g in (1, 2) for i in range(3))
        self.assertEqual(WindowLease.extending(identity, slots, 75, 80), WindowLease(identity, slots, 75, 80))
        with self.assertRaisesRegex(LiveV2Error, 'invalid window lineage'):
            WindowLease.extending(identity, slots, 81, 80)

    def test_qwen_never_reuses(self):
        from remote_ra_live_v2.qwen import QwenAlignLifecycle
        from remote_ra_live_v2.window_lifecycle import WindowLifecycle
        self.assertIs(QwenAlignLifecycle.reuse_state_leases, False)
        self.assertIs(WindowLifecycle.reuse_state_leases, False)
        self.assertIs(WO.WindowAllocationOwner.window_fast, False)
        with self.assertRaisesRegex(LiveV2Error, 'attachment collision'):
            WO.attach_fastpath(NS(window_fast=False), NS(state_kind='sliding'))


class LauncherTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, self.release = gemma_fixture(self.root)
            install.install(site, gemma, self.release, package_root=package, apply=True, model_profile=MP.GEMMA4)
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))
        self.n = 0

    def prepare(self, **kw):
        kw = dict(dict(kv_cache_gib=24, page_tokens=16, max_tokens=4096, max_model_len=131072,
                       language_model_only=True, release_idle_pages=True, dram_arena_gib=79,
                       cuda_graphs='full'), **kw)
        self.n += 1
        out = 'out%d' % self.n
        tasks = [dict(instance_id='g%02d' % k, problem_statement='fix %02d' % k) for k in range(16)]
        result = serve.prepare(self.release, tasks, 'observation_query', self.root / out, 8000,
                               dict(active_token_budget=8192), **kw)
        return [json.loads(json.dumps(v).replace(out, 'out')) if i else v for i, v in enumerate(result)]

    def load(self, config):
        self.n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
            return HC.load()

    def test_switch_pins_one_config_key_and_changes_no_vllm_argument(self):
        _, off, off_argv, off_env = self.prepare()
        _, on, on_argv, on_env = self.prepare(decode_fastpath=True)
        self.assertEqual(on_argv, off_argv)
        self.assertNotIn('decode_fastpath', off)
        self.assertEqual({k: v for k, v in on.items() if k != 'decode_fastpath'}, off)
        self.assertIs(on['decode_fastpath'], True)
        hashes = ('REMOTE_RA_HTTP_CONFIG_SHA256',)
        self.assertEqual({k: v for k, v in on_env.items() if k not in hashes},
                         {k: v for k, v in off_env.items() if k not in hashes})
        # The startup loader takes the pin (the server's real config, not the stripped copy).
        _, config, _, _ = serve.prepare(self.release, dict(instance_id='a', problem_statement='fix a'), 'full',
                                        self.root / 'real', 8000, None, kv_cache_gib=28, page_tokens=16,
                                        max_tokens=4096, max_model_len=131072, language_model_only=True,
                                        decode_fastpath=True)
        self.assertIs(HC.decode_fastpath(self.load(config)), True)
        with self.assertRaisesRegex(RuntimeError, 'must be true'):
            self.load(dict(config, decode_fastpath=False))
        for kw, pattern in ((dict(decode_fastpath=1), 'decode-fastpath is a switch'),
                            (dict(decode_fastpath2=True), 'requires --decode-fastpath'),
                            (dict(decode_fastpath=True, decode_fastpath2=1), 'decode-fastpath2 is a switch')):
            with self.assertRaisesRegex(ValueError, pattern):
                self.prepare(**kw)
        # --decode-fastpath2: one more key, no vLLM argument.
        _, two, two_argv, _ = self.prepare(decode_fastpath=True, decode_fastpath2=True)
        self.assertEqual(two_argv, off_argv)
        self.assertEqual({k: v for k, v in two.items() if k != 'decode_fastpath2'}, on)
        _, config, _, _ = serve.prepare(self.release, dict(instance_id='a', problem_statement='fix a'), 'full',
                                        self.root / 'real2', 8000, None, kv_cache_gib=28, page_tokens=16,
                                        max_tokens=4096, max_model_len=131072, language_model_only=True,
                                        decode_fastpath=True, decode_fastpath2=True)
        self.assertIs(HC.decode_fastpath2(self.load(config)), True)
        with self.assertRaisesRegex(RuntimeError, 'requires decode_fastpath'):
            self.load({k: v for k, v in config.items() if k != 'decode_fastpath'})

    def test_worker_attach_echoes_the_switch_and_attaches_the_fast_paths(self):
        import test_gemma_graphs as TG
        attach = TG.WorkerAttachTests().attach
        off, off_runner = attach({}, TG.compilation(None))
        on, runner = attach({'decode_fastpath': True}, TG.compilation(None))
        self.assertEqual(dict(on), dict(off, decode_fastpath=True))
        owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
        self.assertIsInstance(owner.claims, F.ClaimTable)
        self.assertTrue(owner.window_fast and bridge.decode_fastpath and bridge.qwen.decode_fastpath and
                        bridge.qwen.reuse_state_leases)
        off_owner = off_runner._remote_ra_live_v2_allocation_owner
        self.assertIs(type(off_owner.claims), dict)
        self.assertFalse(off_owner.window_fast)
        two, runner = attach({'decode_fastpath': True, 'decode_fastpath2': True}, TG.compilation(None))
        self.assertEqual(dict(two), dict(on, decode_fastpath2=True))
        owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
        self.assertTrue(owner.window_rows and owner.decode_fastpath2 and bridge.decode_fastpath2 and
                        bridge.qwen.decode_fastpath2 and owner.window_fast)
        self.assertFalse(off_owner.window_rows)
        with self.assertRaisesRegex(LiveV2Error, 'decode_fastpath2'):
            attach({'decode_fastpath2': True}, TG.compilation(None))


if __name__ == '__main__':
    unittest.main()
