"""The legacy KV layout is unchanged by WP2 (exp-decouple-pages). CPU only.

Legacy = P = 784 without --page-tokens/--kv-layout in production, and every
existing fixture (one shared pool, Mamba block == Page, no layout). The 8586dbd
source of every function WP2 changed is kept at the end of this file, verbatim
(FORMER, generated with ast from ``git show 8586dbd:<file>``), and compiled into
its own module's namespace, so one process runs the former and the current code.
FORMER also holds the functions the integration (exp-decouple-int) changed in
these files and in model_queries.py: Phase1Result.__post_init__,
Phase1Coordinator._result, NativeQueryForward.after_forward, model_queries.phase1
(compact phase-1 vectors and the decoupled probe rows):

- test_stageb_e2e's script for all four policies (seven turns, a rebuild, a
  continuing turn and a cancel, through test_host_trim.play): identical generated
  tokens, worker RPC payloads and receipts, plans, scheduled intervals, native
  allocations, lifecycle views, worker forward records and evidence files (byte
  for byte);
- test_ms2_worker's interleaved three-session batched script (the worker's
  phase-1 driver): identical receipts, frames, IO and forward records;
- test_ms_engine's interleaved two-session script with a rebuild: identical RPCs,
  receipts, allocations, session states and evidence files, and the same
  capacity turn_need on every command;
- the capacity gate's headroom/assess/admission_gate on random states and needs;
- serve.prepare and the dry-run main on legacy inputs: identical manifest,
  startup config, argv, environment and pins, and the same refusals;
  remote_ra_http.config load/legacy_geometry/effective_server_config: the same
  results and refusals;
- compose_engine: the same worker startup record; growing_startup: the same
  adapter apart from the added layout attributes.
"""
import __future__
from contextlib import ExitStack, redirect_stdout
import base64
from collections import Counter
import functools
import importlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_host_trim as HT        # play(): test_stageb_e2e's script, deterministic
import test_installer as INSTALL
import test_ms_engine as ME
import test_ms2_worker as MW
import test_stageb_e2e as S
import test_stageb_worker as W

from remote_ra_http import config as http_config
from remote_ra_live_v2 import capacity
from remote_ra_live_v2 import kv_layout as I0
# The IMPORTERS bind parse_lease when first imported: import them here, so that a
# former run patches and restores the current function (a first import during a
# former run kept the 8586dbd parse_lease in model_queries for the rest of the process).
from remote_ra_live_v2 import model_queries, phase1_coordinator  # noqa: F401
from remote_ra_live_v2.capacity import CapacityWait

try:
    import torch
except ImportError:
    torch = None

FLAGS = __future__.annotations.compiler_flag
LAUNCH = ('serve', 'remote_ra_http.config')
# Modules that bind a changed function by name at import time.
IMPORTERS = {('remote_ra_live_v2.contract', 'parse_lease'): ('remote_ra_live_v2.phase1_coordinator',
                                                             'remote_ra_live_v2.model_queries')}


def compiled(module_name, decorators, source):
    """The former function, defined against the live module namespace."""
    module = importlib.import_module(module_name)
    scope = {}
    exec(compile(source, '<8586dbd %s>' % module_name, 'exec', FLAGS, dont_inherit=True),
         module.__dict__, scope)
    (function,) = scope.values()
    for decorator in reversed(decorators):
        function = dict(property=property)[decorator](function)
    return module, function


def counted(function, target, calls):
    """The function, counting its calls in ``calls[target]``."""
    if isinstance(function, property):
        return property(counted(function.fget, target, calls))

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        calls[target] += 1
        return function(*args, **kwargs)
    return wrapper


def former(stack, skip=(), calls=None):
    """Every function WP2 changed, as it was at 8586dbd (modules in ``skip`` excepted)."""
    for module_name, target, decorators, source in FORMER:
        if module_name in skip:
            continue
        module, function = compiled(module_name, decorators, source)
        if calls is not None:
            function = counted(function, target, calls)
        if '.' in target:
            owner, name = target.split('.')
            stack.enter_context(patch.object(getattr(module, owner), name, function))
            continue
        stack.enter_context(patch.object(module, target, function))
        for importer in IMPORTERS.get((module_name, target), ()):
            stack.enter_context(patch.object(importlib.import_module(importer), target, function))


def run(fn, old):
    with ExitStack() as stack:
        if old:
            former(stack)
        return fn()


class FormerTableTests(unittest.TestCase):
    def test_former_sources_compile_and_cover_every_changed_function(self):
        targets = {(m, t) for m, t, _, _ in FORMER}
        self.assertEqual(len(targets), len(FORMER))
        with ExitStack() as stack:
            former(stack)
            from remote_ra_live_v2 import contract, phase1_coordinator
            self.assertEqual(contract.parse_lease.__code__.co_filename, '<8586dbd remote_ra_live_v2.contract>')
            self.assertIs(phase1_coordinator.parse_lease, contract.parse_lease)
        self.assertEqual(importlib.import_module('remote_ra_live_v2.contract').parse_lease.__code__.co_filename,
                         str(ROOT/'source-overlay'/'remote_ra_live_v2'/'contract.py'))


# ---- stage-B end to end ----------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class StageBEquivalenceTests(unittest.TestCase):
    maxDiff = None

    def test_former_and_current_code_send_and_write_the_same(self):
        for policy in ('full', 'random50', 'goal_query', 'observation_query'):
            with self.subTest(policy=policy):
                # play() stubs the startup config itself (load); the launcher and the
                # config checks have their own comparisons below.
                calls = Counter()
                with patch.object(HT, 'former', lambda stack: former(stack, skip=LAUNCH, calls=calls)):
                    before = HT.play(policy, old=True)
                after = HT.play(policy)
                self.assertGreater(len(before['trace']), 20)
                # The former run did run the former engine and worker code.
                self.assertLessEqual(
                    {'parse_lease', '_incarnations', 'Phase1Lease.mamba_index', 'Phase1Coordinator.__init__',
                     'Phase1Coordinator.groups', 'Phase1Coordinator.run', 'Phase1Coordinator.recover',
                     'NativeQueryForward.__init__', 'NativeQueryForward._register', 'rows',
                     'EngineRpcBoundary.__init__', 'EngineRpcBoundary.dispatch_mixed_request',
                     'EngineRpcBoundary.complete_and_drain',
                     'EngineCoreAdapter.__init__', 'EngineCoreAdapter.reserve', 'EngineCoreAdapter.complete',
                     'PrivateBlockOwner._validate', 'PrivateBlockOwner.begin',
                     'Phase1Result.__post_init__', 'Phase1Coordinator._result',
                     'NativeQueryForward.after_forward', 'phase1'} |
                    ({'_parse_goal'} if policy == 'goal_query' else set()), set(calls))
                for key in ('outputs', 'identities', 'trace', 'prepared', 'receipts', 'allocations', 'views',
                            'forwards'):
                    self.assertEqual(before[key], after[key], key)
                self.assertEqual(sorted(after['files']), sorted(before['files']))
                for name, raw in after['files'].items():
                    self.assertEqual(raw, before['files'][name], name)

    def test_batched_worker_steps_are_unchanged(self):
        sids = ('sa', 'sb', 'sc')
        turns = MW.interleave(*[(sid, MW.script(k)) for k, sid in enumerate(sids)])

        def play():
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, W.vllm_stubs()))
                HT.deterministic(stack)
                h = MW.BatchHarness(mrope=True)
                for sid in sids:
                    h.open(sid)
                h.drive(list(turns))
                return dict(summary=dict(MW.by_session(h)), receipts=HT.plain(h.receipts), frames=list(h.frames),
                            io=HT.plain(h.io.log), forwards=[[dict(r['body']) for r in info['receipts_raw']]
                                                             for info in h.done],
                            widths=max(len(s) for s in h.runner.steps))
        with patch.object(MW.BatchHarness, 'finish', HT.finish_keeping_records):
            before, after = run(play, True), run(play, False)
        self.assertEqual(before['widths'], 3)
        for key in ('summary', 'receipts', 'frames', 'io', 'forwards'):
            self.assertEqual(before[key], after[key], key)


# ---- multi-session engine and the capacity gate ----------------------------------------------

def play_multi(policy, needs):
    """test_ms_engine's two sessions, interleaved turn by turn, a rebuild and a continuing
    turn; capacity.turn_need of every command appended to ``needs``."""
    with ExitStack() as stack:
        stubs = W.vllm_stubs()
        request = stubs['vllm.v1.request'] = ModuleType('vllm.v1.request')
        request.RequestStatus = NS(FINISHED_STOPPED='FINISHED_STOPPED')
        stack.enter_context(patch.dict(sys.modules, stubs))
        stack.enter_context(patch.object(W, 'P', ME.P))
        stack.enter_context(patch.object(W.Table, 'block_size', ME.P))
        stack.enter_context(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = ME.config(policy, output, [ME.A, ME.B])
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        HT.deterministic(stack)
        h = ME.MultiHarness(policy, output)
        trace = []

        def collective_rpc(method, args, timeout):
            operation, payload = args
            sent = HT.plain(payload)
            reply = h.collective_rpc(method, args, timeout)
            trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        send = h.send

        def bounded(command):
            tokens = list(h.natives[command.request_id.encode()].prompt_token_ids)
            messages = [dict(role=m['role'], content=m['content']) for m in command.source['messages']]
            needs.append(capacity.turn_need(h.core, command.session_id, tokens, messages, 1536))
            needs.append(capacity.assess(NS(endpoint=h.lifecycle.endpoint, entries={}), command))
            return send(command)
        h.send = bounded
        a, b = h.lanes
        infos = []
        for i in range(len(S.SCRIPT)):
            for lane in (a, b):
                infos.append(h.script(lane, i))
                headroom_states(h, needs)
        retry = a.messages[:-1] + [dict(role='user', content=S.REBUILD[0])]
        infos.append(h.turn(a, messages=retry, target=S.REBUILD[1]))
        infos.append(h.turn(a, S.AFTER[0], target=S.AFTER[1]))
        states = [{k: v for k, v in h.state(lane).items()} for lane in (a, b)]
        prepared = {rid: {k: v for k, v in record.items()
                          if k in ('plan', 'ranking', 'selected_count', 'submitted', 'native_tables',
                                   'native_allocation', 'scheduled_cursor', 'scheduled_intervals')}
                    for rid, record in h.boundary.prepared.items()}
        receipts = {'%s/%s' % key: value for key, value in h.boundary.receipts.items()}
        outputs = {i['rid']: list(i['request'].output_token_ids) for i in infos}
        h.sessions.close()
        files = {path.name: path.read_bytes() for path in sorted(output.iterdir())}
        return dict(trace=trace, prepared=HT.plain(prepared), receipts=HT.plain(receipts),
                    allocations=HT.plain(h.kv.calls), states=HT.plain(states), outputs=outputs, files=files,
                    free=h.pool.get_num_free_blocks(), baseline=h.baseline)


def headroom_states(h, out, count=6):
    """headroom over synthetic running requests on the harness's real endpoint."""
    rng = random.Random(len(out))
    claims = h.endpoint.claims
    for _ in range(count):
        entries = {}
        for k in range(rng.randint(0, 3)):
            rid = 'synthetic-%d' % k
            prompt = rng.randint(1, 3000)
            entries[rid] = dict(state=rng.choice(('accepted', 'accepted', 'draining')),
                                request=NS(prompt_token_ids=range(prompt), num_computed_tokens=rng.randint(0, prompt + 40),
                                           sampling_params=NS(max_tokens=rng.choice((None, 1536, 4000)))))
            if rng.random() < 0.5:
                claims[rid] = dict(geometry=NS(phase2_start=rng.randint(0, prompt)))
        out.append(capacity.headroom(NS(endpoint=h.lifecycle.endpoint, entries=entries)))
        for rid in entries:
            claims.pop(rid, None)


class MultiSessionEquivalenceTests(unittest.TestCase):
    maxDiff = None

    @unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
    def test_two_interleaved_sessions_and_their_capacity_needs(self):
        for policy in ('random50', 'goal_query'):
            with self.subTest(policy=policy):
                old_needs, new_needs = [], []
                before = run(lambda: play_multi(policy, old_needs), True)
                after = run(lambda: play_multi(policy, new_needs), False)
                self.assertGreater(len(old_needs), 20)
                self.assertEqual(old_needs, new_needs)
                self.assertTrue(all(type(n) is tuple or type(n) is int or type(n) is dict for n in new_needs))
                for key in ('trace', 'prepared', 'receipts', 'allocations', 'states', 'outputs', 'free', 'baseline'):
                    self.assertEqual(before[key], after[key], key)
                self.assertEqual(sorted(after['files']), sorted(before['files']))
                for name, raw in after['files'].items():
                    self.assertEqual(raw, before['files'][name], name)

    def test_admission_gate_on_random_needs(self):
        rng = random.Random(784)

        def outcome(need, goals):
            lifecycle = NS(entries={'r': dict(state=rng_state)}, goals=goals)
            with patch.object(capacity, 'assess', lambda lifecycle, command: need):
                try:
                    return ('ok', capacity.admission_gate(lifecycle, NS()))
                except CapacityWait as wait:
                    return ('wait', str(wait))
        seen = set()
        for _ in range(400):
            need = dict(free=rng.randint(0, 60), phase1=rng.randint(0, 30), phase2=rng.randint(0, 30),
                        headroom=rng.randint(0, 30), watermark=rng.randint(0, 3))
            goals = rng.choice((None, NS(parked=None), NS(parked=object())))
            rng_state = rng.choice(('accepted', 'draining'))
            old, new = run(lambda: outcome(need, goals), True), run(lambda: outcome(need, goals), False)
            self.assertEqual(old, new)
            seen.add(old[0])
        self.assertEqual(seen, {'ok', 'wait'})


# ---- launcher, startup config, startup wiring ---------------------------------------------------

class LaunchEquivalenceTests(unittest.TestCase):
    maxDiff = None
    A = dict(instance_id='a', problem_statement='fix a')
    B = dict(instance_id='b', problem_statement='fix b')

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        with redirect_stdout(io.StringIO()):
            package, site, model, self.release = INSTALL.fixture(self.root)
            install.install(site, model, self.release, package_root=package, apply=True)
        self.enterContext(patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path))
        self.rewriter = self.root/'qwen3-0.6b'
        self.rewriter.mkdir()
        (self.rewriter/'config.json').write_bytes(b'{}')

    def outcome(self, *args, **kwargs):
        try:
            return ('ok', serve.prepare(self.release, *args, **kwargs))
        except Exception as error:
            return (type(error).__name__, str(error))

    def test_prepare_is_byte_identical_and_refuses_alike(self):
        cases = []
        # The policies 8586dbd served (random_fill came later: test_random_fill_legacy).
        for policy in ('full', 'random50', 'goal_query', 'observation_query'):
            for task in (dict(self.A), [dict(self.A)], [dict(self.A), dict(self.B), dict(self.A, instance_id='c')]):
                for gib in (8, 24):
                    selection = None
                    if policy == 'goal_query':
                        selection = dict(no_goal_rewriter=True)
                    cases.append(((task, policy, self.root/'out', 8231, selection, 'mswea-xml', gib), {}))
        query = dict(active_token_budget=4096, goal_query_variant='candidate_entity_cap8')
        cases += [
            ((dict(self.A), 'observation_query', self.root/'out', 8231, query), {}),
            ((dict(self.A), 'goal_query', self.root/'out', 8231, dict(goal_rewriter_model=str(self.rewriter),
                                                                       active_ratio=0.25)), {}),
            ((dict(self.A), 'observation_query', self.root/'out', 8231, dict(goal_rewriter_model=str(self.rewriter))), {}),
            # refusals
            ((dict(self.A), 'goal_query', self.root/'out', 8231), {}),
            ((dict(self.A), 'full', self.root/'out', 8231, dict(active_ratio=.5)), {}),
            ((dict(self.A), 'full', self.root/'out', 8231), dict(kv_cache_gib=12)),
            ((dict(self.A), 'full', self.root/'out', 0), {}),
            ((dict(self.A), 'nope', self.root/'out', 8231), {}),
            ((dict(instance_id='a'), 'full', self.root/'out', 8231), {}),
            ((dict(self.A), 'full', self.root, 8231), {}),
            ((dict(self.A), 'full', self.root/'out', 8231, None, 'json'), {}),
        ]
        kinds = set()
        for args, kwargs in cases:
            with self.subTest(args=args[1:], kwargs=kwargs):
                old = run(lambda: self.outcome(*args, **kwargs), True)
                new = run(lambda: self.outcome(*args, **kwargs), False)
                self.assertEqual(old, new)
                kinds.add(old[0])
                if old[0] == 'ok':
                    manifest, config, command, env = new[1]
                    self.assertNotIn('kv_layout', config)
                    self.assertEqual(config['page_tokens'], 784)
        self.assertIn('ok', kinds)
        self.assertIn('ValueError', kinds)

    def test_dry_run_main_prints_the_same(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        tasks = self.root/'tasks.json'
        tasks.write_text(json.dumps([self.A, self.B]))
        variants = [['--policy', 'full'], ['--policy', 'random50', '--kv-cache-gib', '16'],
                    ['--policy', 'goal_query', '--no-goal-rewriter', '--active-ratio', '0.3'],
                    ['--policy', 'observation_query', '--goal-rewriter-model', str(self.rewriter)],
                    ['--policy', 'goal_query', '--goal-rewriter-model', str(self.rewriter), '--task', str(tasks)]]
        for extra in variants:
            with self.subTest(extra=extra):
                argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                        '--output', str(self.root/'dry'), '--port', '8231', *extra]

                def main():
                    stream = io.StringIO()
                    with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                        serve.main()
                    return stream.getvalue()
                old, new = run(main, True), run(main, False)
                self.assertEqual(old, new)
                self.assertNotIn('--block-size', new)
                self.assertNotIn('--additional-config', new)

    def test_config_checks_agree(self):
        manifest, config, command, env = serve.prepare(self.release, [dict(self.A), dict(self.B)], 'goal_query',
                                                       self.root/'cfg', 8231, dict(no_goal_rewriter=True))
        single = serve.prepare(self.release, dict(self.A), 'full', self.root/'cfg1', 8231)[1]
        variants = [config, single, dict(single, page_tokens=16), dict(single, page_tokens=None),
                    dict(single, selection_seed=7), dict(single, warmup_requests=1),
                    {k: v for k, v in single.items() if k != 'page_tokens'}]

        def load(value):
            path = self.root/'startup.json'
            path.write_bytes(serve.wire(value))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(value)):
                try:
                    loaded = http_config.load()
                    geometry = http_config.legacy_geometry()
                except RuntimeError as error:
                    return ('refused', str(error))
                checks = []
                for args in (value['effective_server_config'], dict(value['effective_server_config'], seed=1),
                             dict(value['effective_server_config'], block_size=784)):
                    try:
                        checks.append(http_config.effective_server_config(NS(**args)))
                    except RuntimeError as error:
                        checks.append(str(error))
                return ('ok', loaded, geometry, checks)
        kinds = set()
        for value in variants:
            old, new = run(lambda: load(value), True), run(lambda: load(value), False)
            self.assertEqual(old, new)
            kinds.add(old[0])
        self.assertEqual(kinds, {'ok', 'refused'})

    def test_compose_engine_sends_the_same_startup_record(self):
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        for layout in (None, I0.legacy()):
            sent = []

            def collective_rpc(method, args, timeout):
                sent.append((method, args, timeout))
                raise Stop()

            def start():
                core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                          _remote_ra_trusted_issue=dict(self.A), _remote_ra_trusted_issues=[dict(self.A)],
                          _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                          _remote_ra_growing_adapter=NS(layout=layout),
                          model_executor=NS(collective_rpc=collective_rpc))
                with self.assertRaises(Stop):
                    composition.compose_engine(core, object())
                return sent.pop()
            self.assertEqual(run(start, True), run(start, False))

    def test_growing_startup_builds_the_same_adapter(self):
        import test_decouple_wp2 as D
        legacy = I0.legacy()
        attach = D.GrowingStartupTests('test_layout_from_config_must_match_vllm')

        def adapter():
            with patch.object(attach, 'assertIn', lambda *args: None):
                a = attach.attach(legacy, legacy, legacy)
            return dict(page_tokens=a.page_tokens, managers=[id(m) for m in a.managers],
                        capabilities=a.capabilities(), pool=id(a.owner.pool),
                        sessions=a.sessions, requests=a.requests, source_test_only=a.source_test_only,
                        backend=type(a.backend).__name__)
        old, new = run(adapter, True), run(adapter, False)
        self.assertEqual({k: v for k, v in old.items() if k not in ('managers', 'pool')},
                         {k: v for k, v in new.items() if k not in ('managers', 'pool')})


# ---- the 8586dbd source of every function WP2 changed (generated, verbatim) -------------------

FORMER = (   # (module, target, decorators, 8586dbd source), verbatim
    ('serve', 'selection_contract', (), r'''
def selection_contract(release,policy,selection):
    """Upstream Online Selector V2 contract for query policies; None otherwise.

    Only goal_query pins the Goal rewriter. observation_query ranks Pages by the
    observation vector alone: it needs no rewriter choice, and a given
    --goal-rewriter-model is accepted but never pinned (goal_rewriter_use records it).
    """
    if policy not in ('goal_query','observation_query'):
        need(not selection,'selection options apply to goal_query/observation_query only');return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import selection_contract as contract
    selection=dict(selection or {})
    rewriter=selection.pop('goal_rewriter_model',None);disabled=selection.pop('no_goal_rewriter',False)
    need(rewriter is None or not disabled,'--goal-rewriter-model and --no-goal-rewriter exclude each other')
    used=contract.uses_goal_rewriter(policy)
    need(not used or rewriter is not None or disabled,'goal_query requires --goal-rewriter-model PATH or explicit --no-goal-rewriter')
    budget=selection.pop('active_token_budget',None);ratio=selection.pop('active_ratio',None)
    variant=selection.pop('goal_query_variant','full_goal');need(not selection,'unknown selection option')
    return contract.for_policy(contract.default_contract(784,
        active_ratio=None if budget is not None else (.4 if ratio is None else float(ratio)),
        active_token_budget=budget,goal_rewriter=contract.rewriter_pin(rewriter) if used and rewriter is not None else None,
        goal_query_variant=variant),policy)
'''),

    ('serve', 'prepare', (), r'''
def prepare(release,task,policy,output,port,selection=None,protocol='mswea-xml',kv_cache_gib=8):
    """task: one task, or a list of tasks: one session each, decoded in one batch."""
    release=Path(release).resolve(strict=True);output=Path(output).absolute()
    manifest=installed_check(release)
    need(policy in POLICIES and type(port) is int and 0<port<65536,'policy/port invalid')
    need(type(kv_cache_gib) is int and kv_cache_gib in KV_CACHE_GIB,'--kv-cache-gib must be one of 8, 16, 24, 32')
    kv_bytes=kv_cache_gib*1024**3
    contract=selection_contract(release,policy,selection)
    need(protocol in PROTOCOLS,'this runtime serves the Mini-SWE-Agent XML protocol only (--protocol mswea-xml)')
    tasks=None
    if type(task) is list:
        sys.path.insert(0,str(release/'source'))
        from remote_ra_live_v2.sessions import validate_tasks
        tasks=validate_tasks(task);task=tasks[0]
    # Multi-session stage 2: one running sequence per startup session (batched
    # decode across sessions); a one-task server keeps 1.
    seqs=1 if tasks is None else len(tasks)
    need(type(task) is dict and set(task)=={'instance_id','problem_statement'} and
         all(type(v) is str and v.strip() for v in task.values()),'task must contain instance_id and problem_statement')
    need(not output.exists() and not output.is_symlink() and output.parent.is_dir(),'new output directory under existing parent required')
    need(not output.resolve().is_relative_to(release),'output must be outside sealed release')
    model=manifest['model_root'];revision=manifest['model_revision']
    template=output/'chat_template.jinja';cp=output/'startup.json'
    common=dict(max_tokens=16,temperature=0,seed=42,n=1,stream=False)
    effective=dict(model=model,revision=revision,host='127.0.0.1',distributed_executor_backend='uni',
        tensor_parallel_size=1,pipeline_parallel_size=1,max_num_seqs=seqs,max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_BATCHED,kv_cache_memory_bytes=kv_bytes,seed=42,enable_prefix_caching=True,
        enforce_eager=True,mamba_cache_mode='align',gdn_prefill_backend='triton',enable_auto_tool_choice=True,
        tool_call_parser='qwen3_xml',chat_template_content_format='string')
    config=dict(schema='remote-ra-http-config-v2',policy=policy,task=task,output_directory=str(output),
        page_tokens=784,selection_seed=42,selection_fraction=.5,selection_contract=contract,
        goal_rewriter=goal_rewriter_use(policy,selection,contract),protocol=PROTOCOLS[protocol],warmup_requests=0,
        input_pins_sha256=pin(dict(task=task,common=common)),task_sha256=pin(task),
        issue_sha256=digest(task['problem_statement'].encode()),common_config_sha256=pin(common),
        model_revision=revision,tokenizer_revision=revision,effective_server_config=effective,server_config_sha256=pin(effective))
    if tasks is not None and len(tasks)>1:
        config.update(tasks=tasks,tasks_sha256=pin(tasks))
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',kv_connector_extra_config={'cpu_bytes_to_use':32*1024**3})
    argv=['--model',model,'--served-model-name','selective-prefix','--revision',revision,
        '--host','127.0.0.1','--port',str(port),'--distributed-executor-backend','uni',
        '--no-async-scheduling','--tensor-parallel-size','1','--pipeline-parallel-size','1',
        '--max-num-seqs',str(seqs),'--max-model-len',str(MAX_MODEL_LEN),'--max-num-batched-tokens',str(MAX_BATCHED),
        '--kv-cache-memory-bytes',str(kv_bytes),'--seed','42','--enable-prefix-caching',
        '--prefix-caching-hash-algo','sha256','--enforce-eager','--mamba-cache-mode','align',
        '--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_xml',
        '--chat-template',str(template),'--chat-template-content-format','string',
        '--kv-transfer-config',json.dumps(transfer,separators=(',',':')),'--gdn-prefill-backend','triton']
    program=("import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];"
        "sys.argv=['vllm.entrypoints.openai.api_server']+sys.argv[3:];"
        "runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')")
    command=[sys.executable,'-I','-S','-B','-c',program,str(release/'source'),manifest['site'],*argv]
    env={k:v for k,v in os.environ.items() if not k.startswith('REMOTE_RA_')}
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST',None)
    env.update(REMOTE_RA_HTTP_RUNTIME='1',REMOTE_RA_HTTP_CONFIG=str(cp),REMOTE_RA_HTTP_CONFIG_SHA256=pin(config),
        REMOTE_RA_DYNAMIC_STARTUP='1',VLLM_POLICY_RUNTIME_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USE_V2_MODEL_RUNNER='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    return manifest,config,command,env
'''),

    ('serve', 'main', (), r'''
def main():
    parser=argparse.ArgumentParser(description=__doc__,prog='selective-prefix serve')
    parser.add_argument('--release-dir',required=True,type=Path)
    parser.add_argument('--task',required=True,type=Path,
        help='JSON task {instance_id, problem_statement}, or a list of tasks: one session each')
    parser.add_argument('--policy',required=True,choices=POLICIES)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--protocol',choices=tuple(PROTOCOLS),default='mswea-xml',
        help='agent protocol: Mini-SWE-Agent XML actions (the only protocol of this runtime)')
    group=parser.add_argument_group('Selector V2 (goal_query/observation_query only)')
    budget=group.add_mutually_exclusive_group()
    budget.add_argument('--active-ratio',type=float,help='share of sealed history kept active (default 0.40)')
    budget.add_argument('--active-token-budget',type=int,help='absolute active history budget in tokens (E1)')
    rewriter=group.add_mutually_exclusive_group()
    rewriter.add_argument('--goal-rewriter-model',type=Path,
        help='local Qwen3-0.6B checkpoint for the goal_query Goal rewriter (observation_query: accepted, never used)')
    rewriter.add_argument('--no-goal-rewriter',action='store_true',help='goal_query: deterministic Goal only (differs from the PoC)')
    group.add_argument('--goal-query-variant',choices=('full_goal','candidate_entity_cap8'))
    parser.add_argument('--kv-cache-gib',type=int,choices=(8,16,24,32),default=8,
        help='GPU KV cache size in GiB (default 8, the qualified profile)')
    parser.add_argument('--start',action='store_true');args=parser.parse_args()
    selection={k:v for k,v in dict(active_ratio=args.active_ratio,active_token_budget=args.active_token_budget,
        goal_rewriter_model=args.goal_rewriter_model and str(args.goal_rewriter_model),
        no_goal_rewriter=args.no_goal_rewriter or None,goal_query_variant=args.goal_query_variant).items() if v is not None}
    manifest,config,command,env=prepare(args.release_dir,json.loads(args.task.read_bytes()),args.policy,args.output,args.port,
        selection,args.protocol,args.kv_cache_gib)
    if not args.start:
        print(json.dumps(dict(status='SERVICE_DRY_RUN',command=command,starts_model=False,
            goal_rewriter=config['goal_rewriter']),sort_keys=True));return
    need(sys.platform=='linux','loaded GPU service requires Linux')
    output=Path(config['output_directory']);output.mkdir(mode=0o700)
    save(output/'startup.json',wire(config))
    template=json.loads((Path(manifest['model_root'])/'tokenizer_config.json').read_bytes())['chat_template']
    save(output/'chat_template.jinja',template.encode())
    save(output/'launch.json',wire(dict(command=command,policy=args.policy,port=args.port,
        runtime_manifest=digest((args.release_dir/'runtime-manifest.json').read_bytes()))))
    # Replace this launcher so the shell's PID is the owned APIServer. No job,
    # tunnel, scheduler, retry, or daemon is created by this helper.
    os.execve(sys.executable,command,env)
'''),

    ('remote_ra_http.config', 'load', (), r'''
def load():
    require(os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')=='1','HTTP runtime default OFF')
    p=Path(os.environ['REMOTE_RA_HTTP_CONFIG'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(),'regular absolute startup config required')
    raw=p.read_bytes()
    require(sha256(raw).hexdigest()==os.environ['REMOTE_RA_HTTP_CONFIG_SHA256'],'startup config drift')
    value=json.loads(raw)
    require(value.get('schema')=='remote-ra-http-config-v2','startup schema mismatch')
    require(value.get('policy') in POLICIES,'startup policy required')
    # Stage B (two-phase turns, live-tail carry) exists for the XML protocol only;
    # the tool-call protocol stays with the last qualified release (4d90616).
    require(value.get('protocol')=='mswea_xml_v1','this runtime serves the Mini-SWE-Agent XML protocol only')
    contract=value.get('selection_contract')
    if value['policy'] in ('goal_query','observation_query'):
        from remote_ra_live_v2.selection_contract import for_policy
        # Fail closed: a config that pins the Goal rewriter for observation_query
        # (which never uses it) is refused, not run or silently ignored.
        for_policy(contract,value['policy'])
        require(contract['page_tokens']==value.get('page_tokens'),'selection contract Page size mismatch')
    else:
        require(contract is None,'selection contract applies to query policies only')
    # The launcher's record of the rewriter decision (absent in older configs).
    record=value.get('goal_rewriter')
    require(record is None or (type(record) is dict and record.get('used') is
            (contract is not None and contract['goal_rewriter'] is not None)),
            'Goal rewriter record differs from the selection contract')
    task=value.get('task')
    require(type(task) is dict and set(task)=={'instance_id','problem_statement'}
            and all(type(v) is str and v.strip() for v in task.values()),'frozen issue required')
    require(value.get('page_tokens')==784 and value.get('selection_seed')==42
            and value.get('selection_fraction')==.5,'pinned geometry/selection config required')
    require(value.get('warmup_requests')==0,'no warmup requests permitted')
    require(type(value.get('output_directory')) is str and Path(value['output_directory']).is_absolute(),
            'absolute owned evidence output required')
    for key in ('input_pins_sha256','task_sha256','common_config_sha256','issue_sha256'):
        require(type(value.get(key)) is str and len(value[key])==64,'missing startup '+key)
    require(sha256(task['problem_statement'].encode()).hexdigest()==value['issue_sha256'],
            'frozen issue hash mismatch')
    # Multi-session stage 1: an optional pinned task list, one session per task;
    # task stays tasks[0] (session 0). A one-task config is normalized to [task].
    from remote_ra_live_v2.sessions import validate_tasks
    if 'tasks' in value:
        tasks=validate_tasks(value['tasks'])
        require(tasks[0]==task and value.get('tasks_sha256')==digest(tasks),'pinned task list required')
    else:
        value['tasks']=[task]
    return value
'''),

    ('remote_ra_http.config', 'legacy_geometry', (), r'''
def legacy_geometry():
    """Only geometry/task compatibility; contains no fixture messages/stages."""
    config=load()
    return dict(page_tokens=config['page_tokens'],task=config['task'],
                selected_count=2,projection_seed=42)
'''),

    ('remote_ra_http.config', 'effective_server_config', (), r'''
def effective_server_config(args):
    names=('model','revision','host','distributed_executor_backend','tensor_parallel_size',
        'pipeline_parallel_size','max_num_seqs','max_model_len','max_num_batched_tokens',
        'kv_cache_memory_bytes','seed','enable_prefix_caching','enforce_eager','mamba_cache_mode',
        'gdn_prefill_backend','enable_auto_tool_choice','tool_call_parser','chat_template_content_format')
    actual={name:getattr(args,name,None) for name in names}
    config=load();expected=config['effective_server_config']
    require(actual==expected,'observed API server arguments differ from pinned common config')
    # Multi-session stage 2: one running sequence per startup session; one task keeps 1.
    sessions=len(config.get('tasks') or [config['task']])
    require(actual['host']=='127.0.0.1' and actual['max_num_seqs']==sessions,
            'exclusive loopback server with one sequence per startup session required')
    require(digest(actual)==config['server_config_sha256'],'effective server config hash mismatch')
    return actual
'''),

    ('remote_ra_live_v2.contract', '_incarnations', (), r'''
def _incarnations(value, ids, what):
    require(type(value) is dict and set(value) == {str(b) for b in ids} and
            all(_int(v, 1) for v in value.values()),
            what + ": allocation incarnation proof missing")
    return {int(k): v for k, v in value.items()}
'''),

    ('remote_ra_live_v2.contract', 'Phase1Lease.mamba_index', ('property',), r'''
def mamba_index(self) -> int:
    """Native Mamba slot index at num_computed=C, scheduled e-C: ceil(e/P)-1."""
    return -(-self.end // self.page_tokens) - 1
'''),

    ('remote_ra_live_v2.contract', '_parse_goal', (), r'''
def _parse_goal(raw, attention, groups, page_tokens, taken):
    require(type(raw) is dict and set(raw) == GOAL_LEASE_KEYS and
            type(raw["lease_id"]) is str and _GOAL_ID.fullmatch(raw["lease_id"]) is not None
            and hex64(raw["payload_sha256"]), "goal lease identity")
    tokens = raw["tokens"]
    require(_seq(tokens) and len(tokens) > 0 and all(_int(x) for x in tokens),
            "goal tokens required")
    width = -(-len(tokens) // page_tokens)
    rows = _rows(raw["rows"], groups)
    ids = []
    for g, row in enumerate(rows):
        if g in attention:
            require(len(row) == width and 0 not in row, "goal FullAttention row layout")
        else:
            require(len(row) == width and row[:-1] == (0,) * (width - 1) and row[-1] > 0,
                    "goal Mamba row must be null-padded")
        ids += [b for b in row if b]
    require(len(set(ids)) == len(ids) and not taken.intersection(ids),
            "goal blocks alias each other or the phase-1 lease")
    require(_ids(raw["owned"]) and tuple(raw["zero"]) == tuple(raw["owned"]) and
            set(raw["owned"]) == set(ids), "goal owned/zero blocks")
    return GoalLease(raw["lease_id"], tuple(tokens), rows, tuple(raw["owned"]),
                     tuple(raw["zero"]), _incarnations(raw["incarnations"], ids, "goal"),
                     raw["payload_sha256"])
'''),

    ('remote_ra_live_v2.contract', 'parse_lease', (), r'''
def parse_lease(raw, *, attention_groups=None) -> Phase1Lease:
    """Structural, pure check of an engine-built phase-1 lease; fail closed.

    Worker-state checks (borrowed == resident_blocks + tail, claims, undrained
    leases, carry digest, max tokens) stay with the caller. FullAttention
    groups are inferred from the tail/fresh blocks; pass attention_groups to
    pin them to the worker's own layout.
    """
    require(type(raw) is dict and set(raw) == LEASE_KEYS, "phase-1 lease key set")
    require(type(raw["lease_id"]) is str and _PHASE1_ID.fullmatch(raw["lease_id"]) is not None,
            "phase-1 lease id")
    require(all(type(raw[k]) is str and 0 < len(raw[k]) <= 256
                for k in ("request", "session", "namespace", "epoch")) and
            all(hex64(raw[k]) for k in ("binding", "source", "carry_digest", "mac")) and
            raw["carry_digest"] != ZERO_DIGEST, "phase-1 lease identity")
    P, C, e, Q, L = (raw[k] for k in
                     ("page_tokens", "start", "end", "sealed_start", "candidate_end"))
    tc = raw["tail_candidate"]
    require(_int(raw["generation"]) and _int(raw["local_turn"], 1) and _int(P, 1) and
            all(_int(v) for v in (C, e, Q, L)) and type(tc) is bool,
            "phase-1 lease integers")
    tail = C > Q
    # L = Q + P exactly when the tail page is a candidate; that needs a tail.
    require(Q % P == 0 and Q <= C < Q + P and C < e and L < e and
            L == Q + P * tc and (tail or not tc), "phase-1 lease geometry")
    R = raw["resident_pages"]
    require(_seq(R) and all(_int(p) for p in R) and list(R) == sorted(set(R)) and
            all(p < Q // P for p in R), "resident Pages must be ascending below Q")
    R = tuple(R)
    M = -(-e // P)
    fresh = range(Q // P + tail, M)
    rows = _rows(raw["rows"])
    where = {}
    for g, row in enumerate(rows):
        for b in row:
            if b:
                require(b not in where, "phase-1 row block aliased")
                where[b] = g
    t = raw["tail"]
    if tail:
        require(type(t) is dict and set(t) == {"page", "valid_end", "block_ids"} and
                t["page"] == Q // P and t["valid_end"] == C and
                _int(t["page"]) and _int(t["valid_end"]), "phase-1 tail mismatch")
        tail_ids = _one_block_groups(t["block_ids"], "tail")
    else:
        require(t is None, "phase-1 tail without a live tail")
        tail_ids = ()
    rf = raw["run_fresh"]
    require(type(rf) is dict and set(rf) == {str(p) for p in fresh},
            "phase-1 fresh Page set")
    run_fresh = {p: _one_block_groups(rf[str(p)], "fresh") for p in fresh}
    # e > C >= Q: either a tail exists or the fresh range is non-empty.
    anchor = tail_ids or run_fresh[fresh[0]]
    require(all(len(v) == len(anchor) for v in run_fresh.values()) and
            all(b in where for (b,) in anchor), "fresh/tail blocks outside the rows")
    att = tuple(where[b] for (b,) in anchor)
    require(list(att) == sorted(set(att)) and len(att) < len(rows) and
            (attention_groups is None or tuple(attention_groups) == att),
            "FullAttention group layout mismatch")
    borrowed_fa, owned_all = [], []
    for fi, g in enumerate(att):
        row = rows[g]
        tail_part = (tail_ids[fi][0],) if tail else ()
        fresh_part = tuple(run_fresh[p][fi][0] for p in fresh)
        # Physical x in [C, e) sits at |R|*P + (x - Q): [R][tail][fresh].
        require(len(row) == len(R) + len(tail_part) + len(fresh_part) and 0 not in row and
                row[len(R):] == tail_part + fresh_part, "phase-1 FullAttention row layout")
        borrowed_fa += row[:len(R)] + tail_part
        owned_all += fresh_part
    for g, row in enumerate(rows):
        if g not in att:
            require(len(row) == M and row[:-1] == (0,) * (M - 1) and row[-1] > 0,
                    "phase-1 Mamba row must be null-padded")
            owned_all.append(row[-1])
    borrowed, owned, zero = raw["borrowed"], raw["owned"], raw["zero"]
    require(_ids(borrowed) and _ids(owned) and _seq(zero) and tuple(zero) == tuple(owned) and
            set(borrowed) == set(borrowed_fa) and set(owned) == set(owned_all),
            "phase-1 borrowed/owned/zero blocks")
    sparse = raw["sparse"]
    require(type(sparse) is dict and all(type(v) is int for v in sparse.values()) and
            sparse == {"logical_matched_tokens": Q,
                       "dropped_full_attention_tokens": Q - len(R) * P},
            "phase-1 SparsePlan mismatch")
    goal = raw["goal"]
    if goal is not None:
        goal = _parse_goal(goal, att, len(rows), P, set(where))
    return Phase1Lease(
        raw["lease_id"], raw["request"], raw["binding"], raw["session"], raw["namespace"],
        raw["generation"], raw["epoch"], raw["source"], raw["local_turn"], P, C, e, Q, L, R,
        LeaseTail(Q // P, C, tail_ids) if tail else None, tc, rows, tuple(borrowed),
        tuple(owned), tuple(zero), run_fresh,
        _incarnations(raw["incarnations"], where, "phase-1"), dict(sparse),
        raw["carry_digest"], goal, raw["mac"], att, digest(raw))
'''),

    ('remote_ra_live_v2.contract', 'Phase1Result.__post_init__', (), r'''
def __post_init__(self):
    require(hex64(self.binding) and hex64(self.lease_digest) and
            type(self.lease_id) is str and _PHASE1_ID.fullmatch(self.lease_id) is not None,
            "phase-1 result identity")
    require(_int(self.start) and _int(self.end) and self.start < self.end,
            "phase-1 result interval")
    pv = self.page_vectors
    require(isinstance(pv, Mapping) and all(_int(p) for p in pv) and
            (not pv or set(pv) == set(range(min(pv), max(pv) + 1))),
            "phase-1 Page vectors must cover a contiguous int Page range")
    pages = {p: _vector(pv[p], "Page vector") for p in sorted(pv)}
    obs = _vector(self.observation_vector, "observation vector")
    goal = (self.goal_vector, self.goal_payload_sha256, self.goal_token_count)
    goal_vector = None
    if any(v is not None for v in goal):
        require(hex64(self.goal_payload_sha256) and _int(self.goal_token_count, 1),
                "Goal vector, payload digest and token count travel together")
        goal_vector = _vector(self.goal_vector, "Goal vector")
    dims = {len(v) for v in (obs, *pages.values())} | (
        {len(goal_vector)} if goal_vector else set())
    require(len(dims) == 1, "phase-1 vector dimensions differ")
    require(hex64(self.state_digest) and
            (self.tail_capture_digest is None or hex64(self.tail_capture_digest)),
            "phase-1 state/tail digests")
    require(_seq(self.position_evidence) and isinstance(self.vector_space, Mapping)
            and isinstance(self.worker_receipt, Mapping), "phase-1 evidence")
    for name, value in (("page_vectors", pages), ("observation_vector", obs),
                        ("goal_vector", goal_vector),
                        ("position_evidence", tuple(self.position_evidence)),
                        ("vector_space", dict(self.vector_space)),
                        ("worker_receipt", dict(self.worker_receipt))):
        object.__setattr__(self, name, value)
'''),

    ('remote_ra_live_v2.phase1_coordinator', 'Phase1Coordinator.__init__', (), r'''
def __init__(self, core, proxy, renderer, *, max_tokens=None):
    self.core, self.proxy, self.renderer = core, proxy, renderer
    self.adapter = core._remote_ra_growing_adapter
    self.pool = self.adapter.kv_manager.block_pool
    limit = (max_tokens if max_tokens is not None else
             getattr(core.scheduler, 'max_num_scheduled_tokens', None))
    # One native forward: runner.max_num_tokens == max_num_batched_tokens.
    require(type(limit) is int and limit > 0, 'phase-1 single-forward token limit required')
    self.max_tokens = limit
    self.records = {}  # request -> lease record until adopt/abort
'''),

    ('remote_ra_live_v2.phase1_coordinator', 'Phase1Coordinator.groups', (), r'''
def groups(self):
    """(FullAttention group ids, Mamba group ids, group count), group-id order."""
    P = self.adapter.page_tokens
    specs = [g.kv_cache_spec for g in self.adapter.kv_manager.kv_cache_config.kv_cache_groups]
    kinds = tuple(type(spec).__name__ for spec in specs)
    # Mamba align block == Page: the live slot of [0, e) is index ceil(e/P) - 1.
    require(set(kinds) == {'FullAttentionSpec', 'MambaSpec'} and
            all(spec.block_size == P for spec in specs), 'unsupported phase-1 group layout')
    att = tuple(i for i, kind in enumerate(kinds) if kind == 'FullAttentionSpec')
    mam = tuple(i for i, kind in enumerate(kinds) if kind == 'MambaSpec')
    require(len(att) == len(self.adapter.managers), 'native attention manager order mismatch')
    return att, mam, len(kinds)
'''),

    ('remote_ra_live_v2.phase1_coordinator', 'Phase1Coordinator.run', (), r'''
def run(self, binding, geo, *, allocation_generation, goal_payload=None):
    """Lease, run and drain phase 1 for local turn > 0; return its Phase1Result."""
    adapter = self.adapter
    adapter.authority.owner()
    P = adapter.page_tokens
    s = adapter.sessions[binding.session]
    require(geo.local_turn > 0 and geo.page_tokens == P, 'phase 1 runs only after local turn 0')
    require(not s.poisoned and s.pending is None and s.local_turn == geo.local_turn and
            s.turn == binding.turn and binding.request not in adapter.requests,
            'phase 1 must precede the generation reserve')
    connector = self.core.scheduler.connector
    require(connector is not None and not connector.has_pending_push_work(),
            'native connector must be quiescent before phase-1 allocation')
    # Process-wide, like the worker: no lease starts while any session's
    # lease lacks its joined drain (native_probe refuses the alias).
    require(not self.records, 'prior phase-1 lease lacks a drain, adoption or abort')
    require(geo.e - geo.C <= self.max_tokens, 'PHASE1_TOO_LONG: e - C exceeds one native forward')
    require(type(allocation_generation) is int and allocation_generation > 0,
            'worker allocation generation required')
    require(hex64(s.boundary) and s.boundary != ZERO_DIGEST, 'phase 1 needs a committed carry')
    owner, ns = adapter.owner, s.namespace
    resident = owner.sessions.get(ns, {})
    R = tuple(sorted(resident))
    pin = owner.tails.get(ns)
    require(R == tuple(s.resident) and all(p < geo.Q // P for p in R) and
            (pin is not None) == geo.tail and
            (pin is None or (pin.page, pin.start, pin.valid_end) == (geo.Q // P, geo.Q, geo.C)),
            'phase-1 geometry differs from the session pins')
    att, mam, count = self.groups()
    goal_tokens = None
    if goal_payload is not None:
        require(s.policy == 'goal_query' and type(goal_payload) is str,
                'a Goal forward belongs to goal_query only')
        # Upstream _goal_query_text, tokenized raw without special tokens.
        from runtime_renderer import PLAIN_GOAL_SCHEMA
        goal_tokens = [int(t) for t in self.renderer(dict(schema=PLAIN_GOAL_SCHEMA,
                                                         text=goal_payload)).token_ids]
        require(0 < len(goal_tokens) <= self.max_tokens,
                'PHASE1_TOO_LONG: Goal forward exceeds one native forward')
    fresh = list(geo.fresh_pages)
    M = -(-geo.e // P)
    width = -(-len(goal_tokens) // P) if goal_tokens else 0
    needed = len(att) * len(fresh) + len(mam) + (len(att) * width + len(mam) if goal_tokens else 0)
    require(needed <= self.pool.get_num_free_blocks(), 'insufficient native blocks for phase 1')
    from .pool_incarnations import install
    versions = install(self.pool)
    allocated = []
    def new(n):
        blocks = self.pool.get_new_blocks(n)
        allocated.extend(blocks)
        return list(blocks)
    try:
        fresh_fa = {g: new(len(fresh)) for g in att}
        fresh_mam = {g: new(1)[0] for g in mam}
        goal_fa = {g: new(width) for g in att} if goal_tokens else {}
        goal_mam = {g: new(1)[0] for g in mam} if goal_tokens else {}
    except BaseException:
        self.pool.free_blocks(reversed(allocated))
        raise
    try:
        tail_ids = [pin.groups[i][0].block_id for i in range(len(att))] if pin else []
        rows = []
        for g in range(count):
            if g in att:
                i = att.index(g)
                rows.append([resident[p][i][0].block_id for p in R] + tail_ids[i:i + 1] +
                            [b.block_id for b in fresh_fa[g]])
            else:
                rows.append([0] * (M - 1) + [fresh_mam[g].block_id])
        ids = [b for row in rows for b in row if b]
        require(all(b in versions for b in ids), 'phase-1 block allocation incarnation missing')
        owned = [b.block_id for g in att for b in fresh_fa[g]] + [fresh_mam[g].block_id for g in mam]
        goal = None
        if goal_tokens:
            grows = [[b.block_id for b in goal_fa[g]] if g in att else
                     [0] * (width - 1) + [goal_mam[g].block_id] for g in range(count)]
            gids = [b for row in grows for b in row if b]
            goal = dict(lease_id='goal-' + secrets.token_hex(16), tokens=goal_tokens, rows=grows,
                        owned=gids, zero=list(gids),
                        incarnations={str(b): versions[b] for b in gids},
                        payload_sha256=sha256(goal_payload.encode()).hexdigest())
        lease = dict(lease_id='phase1-' + secrets.token_hex(16), request=binding.request,
            binding=binding.identity, session=binding.session, namespace=ns,
            generation=allocation_generation, epoch=binding.epoch,
            source=adapter.authority.source, local_turn=geo.local_turn, page_tokens=P,
            start=geo.C, end=geo.e, sealed_start=geo.Q, candidate_end=geo.L,
            resident_pages=list(R),
            tail=(dict(page=pin.page, valid_end=pin.valid_end, block_ids=[[b] for b in tail_ids])
                  if pin else None),
            tail_candidate=geo.tail_candidate, rows=rows,
            borrowed=[resident[p][i][0].block_id for p in R for i in range(len(att))] + tail_ids,
            owned=owned, zero=list(owned),
            run_fresh={str(p): [[fresh_fa[g][k].block_id] for g in att] for k, p in enumerate(fresh)},
            incarnations={str(b): versions[b] for b in ids},
            sparse=dict(logical_matched_tokens=geo.Q,
                        dropped_full_attention_tokens=geo.Q - len(R) * P),
            carry_digest=s.boundary, goal=goal)
        lease['mac'] = adapter.authority.sign(lease)
        parse_lease(lease, attention_groups=att)  # the worker parses the same shape
        request = dict(request=binding.request, binding=binding.identity,
            session=binding.session, turn=binding.turn, local_turn=geo.local_turn,
            generation=binding.generation, allocation_generation=allocation_generation,
            epoch=binding.epoch, source=adapter.authority.source, model=binding.model,
            policy=s.policy, tokens=list(binding.tokens[:geo.e]),
            prompt_digest=digest(list(binding.tokens)), lease=lease, lease_digest=digest(lease))
        request['request_digest'] = digest(request)
    except BaseException:
        self.pool.free_blocks(reversed(allocated))
        raise
    self.records[binding.request] = dict(binding=binding.identity, lease=deepcopy(lease),
        request=deepcopy(request), drained=False,
        fa={p: tuple((fresh_fa[g][k],) for g in att) for k, p in enumerate(fresh)},
        transient=tuple(fresh_mam.values()) + tuple(b for g in att for b in goal_fa.get(g, ()))
                  + tuple(goal_mam.values()))
    # Evidence: the query tokens are the Goal tokens when a Goal forward ran,
    # else the observation T[o:e] for observation_query.
    query_tokens = (goal_tokens if goal_tokens else
                    list(binding.tokens[geo.o:geo.e]) if s.policy == 'observation_query' else None)
    try:
        result = self.proxy.call('phase1', request, query_tokens=query_tokens)
        return self._result(result, binding, geo, s, goal_tokens)
    finally:
        # A failed forward must expose its own signed drain through the
        # separate idempotent RPC; never infer release from an error.
        self.recover(binding.request)
'''),

    ('remote_ra_live_v2.phase1_coordinator', 'Phase1Coordinator._result', (), r'''
def _result(self, result, binding, geo, session, goal_tokens):
    record = self.records[binding.request]
    lease, request = record['lease'], record['request']
    require(type(result) is dict and result.get('operation') == 'phase1' and
            result.get('binding') == binding.identity and
            result.get('request_digest') == request['request_digest'] and
            result.get('lease_digest') == request['lease_digest'] and
            result.get('allocation_generation') == request['allocation_generation'] and
            result.get('start') == geo.C and result.get('end') == geo.e and
            result.get('forward_tokens') == geo.e - geo.C and result.get('drained') is True and
            result.get('representation') == 'pre_final_norm_last_token_l2',
            'foreign or partial phase-1 receipt')
    require(result.get('carry_restored') == dict(start=geo.C, digest=session.boundary),
            'phase 1 did not start from the committed carry')
    vectors = _pages(result.get('page_vectors'))
    require(set(vectors) == set(closed_pages(geo.C, geo.e, geo.page_tokens)),
            'phase 1 must return exactly the Pages whose last token it computed')
    goal = goal_tokens is not None
    require((result.get('goal_vector') is not None) == goal and
            (not goal or (result.get('goal_token_count') == len(goal_tokens) and
                          result.get('goal_tokens_sha256') == digest(goal_tokens))),
            'Goal forward result present iff requested')
    require((result.get('tail_capture_digest') is not None) == geo.tail_candidate,
            'tail capture iff the tail Page is a candidate')
    require(type(result.get('worker_receipt')) is dict and
            type(result.get('vector_space')) is dict, 'phase-1 worker evidence missing')
    return Phase1Result(binding=binding.identity, lease_id=lease['lease_id'],
        lease_digest=request['lease_digest'], start=geo.C, end=geo.e,
        page_vectors={p: tuple(v) for p, v in vectors.items()},
        observation_vector=tuple(result['observation_vector']),
        goal_vector=tuple(result['goal_vector']) if goal else None,
        goal_payload_sha256=lease['goal']['payload_sha256'] if goal else None,
        goal_token_count=len(goal_tokens) if goal else None,
        state_digest=result['state_digest'], tail_capture_digest=result['tail_capture_digest'],
        position_evidence=tuple(result.get('position_evidence') or ()),
        vector_space=result['vector_space'], worker_receipt=result['worker_receipt'])
'''),

    ('remote_ra_live_v2.phase1_coordinator', 'Phase1Coordinator.recover', (), r'''
def recover(self, request_id=None):
    """Idempotent joined drain. Lost replies never imply ownership release."""
    self.adapter.authority.owner()
    for rid, record in tuple(self.records.items()):
        if (request_id is not None and rid != request_id) or record['drained']:
            continue
        lease, request = record['lease'], record['request']
        status = self.proxy.call('phase1_drain', dict(request=rid, binding=record['binding'],
            lease_id=lease['lease_id'], lease_digest=request['lease_digest'],
            phase1_request=request))
        require(type(status) is dict and status.get('operation') == 'phase1_drain' and
                status.get('drained') is True and status.get('binding') == record['binding'] and
                status.get('lease_id') == lease['lease_id'] and
                status.get('lease_digest') == request['lease_digest'] and
                type(status.get('worker_receipt')) is dict,
                'phase-1 drain unknown/foreign: native references retained')
        # Free exactly once after the joined drain: Mamba and Goal blocks.
        # Fresh FA blocks stay for adoption (they hold [Q or L, e) KV).
        require(all(b.ref_cnt == 1 for b in record['transient']),
                'phase-1 transient blocks acquired another owner before cleanup')
        self.pool.free_blocks(reversed(record['transient']))
        record['drained'] = True
'''),

    ('remote_ra_live_v2.native_probe', 'NativeQueryForward.__init__', (), r'''
def __init__(self,runner,owner,backend):
    self.runner,self.owner,self.backend=runner,owner,backend
    self.active=None;self.drains={};self.leases={};self.executing=False
'''),

    ('remote_ra_live_v2.native_probe', 'NativeQueryForward._register', (), r'''
def _register(self,payload):
    from .contract import parse_lease
    lease=parse_lease(payload['lease'],attention_groups=self.owner.attention_groups)
    rid=lease.lease_id
    record=self.owner.records[payload['request']]
    identity=record['identity']
    require(record['binding']==payload['binding']
            and record['generation']==payload['allocation_generation']
            and lease.lease_digest==payload['lease_digest']
            and lease.request==payload['request']
            and lease.session==identity.session
            and lease.generation==record['generation']
            and lease.source==identity.source
            and lease.epoch==payload['epoch']
            and sha256(lease.epoch.encode()).hexdigest()==identity.epoch,
            'foreign native phase-1 lease')
    prior=self.leases.get(rid)
    if prior is not None:
        require(prior['digest']==digest(payload),'phase-1 replay changed payload')
        return prior
    owned=set(lease.owned)|set(lease.goal.owned if lease.goal else ())
    blocks=owned|set(lease.borrowed)
    # Borrowed pins are read-only; owned blocks are zeroed. Neither may be
    # claimed by a generation or held by another undrained lease.
    require(not any(key not in self.drains for key in self.leases) and
            not any(b in blocks for _,b in self.owner.claims),
            'phase-1 physical allocation alias')
    prior=dict(payload=deepcopy(payload),digest=digest(payload),lease=lease,
               identity=identity,generation=record['generation'])
    self.leases[rid]=prior
    return prior
'''),

    ('remote_ra_live_v2.native_probe', 'NativeQueryForward.after_forward', (), r'''
def after_forward(self,hidden):
    require(self.active is not None and self.active['restored'],'probe forward before private restore')
    a=self.active
    self.backend.io.drain()
    rows=hidden[:len(a['tokens'])-a['start']].detach().to(device='cpu').clone()
    if a['phase']=='phase1':
        lease=a['lease']
        a['state']=self.backend.capture_phase1_state(a['identity'],lease.lease_id,lease.end,
                                                     self._mamba_slot(),a['generation'])
        if lease.tail_candidate:
            # Page Q/P closed in phase 1: its CPU copy must exist before its
            # block can leave the rows (SPEC!=DESIGN 1).
            a['tail']=self.backend.io.capture(dict(zip(lease.attention_groups,
                                                       lease.tail.block_ids)),'attention')
    a['rows']=rows
    return rows
'''),

    ('remote_ra_live_v2.pool_incarnations', 'rows', (), r'''
def rows(manager,request):
    state=install(manager.block_pool)
    result=[]
    for group in manager.coordinator.single_type_managers:
        values=[]
        for block in group.req_to_blocks[request]:
            if block.block_id==0:values.append((0,0));continue
            require(block.ref_cnt>0 and block.block_id in state,'native row owns untracked/free allocation')
            values.append((block.block_id,state[block.block_id]))
        result.append(values)
    ids=[b for group in result for b,_ in group if b]
    require(len(ids)==len(set(ids)),'native live allocation IDs overlap across groups')
    return result
'''),

    ('remote_ra_live_v2.engine_boundary', 'EngineRpcBoundary.__init__', (), r'''
def __init__(self, proxy, adapter):
    self.proxy, self.adapter = proxy, adapter
    self.prepared, self.receipts = {}, {}
    from .pool_incarnations import install
    install(adapter.kv_manager.block_pool)
'''),

    ('remote_ra_live_v2.engine_boundary', 'EngineRpcBoundary.dispatch_mixed_request', (), r'''
def dispatch_mixed_request(self, binding, plan, manager, request, args, kwargs):
    record=self.prepared[binding.request]
    # Runs per request per engine step. The adapter passes the binding it
    # reserved, so ``is`` decides at once; another object is compared field
    # by field as before (a binding always equals itself).
    require(record["submitted"] and (record['binding'] is binding or record['binding']==binding)
            and record['plan']==plan and manager is self.adapter.kv_manager,
            "native allocation before worker reserve")
    from .sparse_mapping import RowLayout, SparsePlan, compose_rows
    native=manager._growing_original_allocate_slots
    call=_signature(native).bind(request,*args,**kwargs);call.apply_defaults()
    values=call.arguments
    start=plan['phase2_start']
    from .allocation_guard import validate_native_allocation
    validate_native_allocation(values, request, binding, plan, start)
    from .chunk_protocol import validate_interval
    interval=validate_interval(record,request,binding,plan,values['num_new_tokens'])
    groups=manager.kv_cache_config.kv_cache_groups
    kinds=tuple(type(g.kv_cache_spec).__name__ for g in groups)
    require(set(kinds)=={'FullAttentionSpec','MambaSpec'},'unsupported sparse native group layout')
    attention=tuple(i for i,kind in enumerate(kinds) if kind=='FullAttentionSpec')
    require(len(attention)==len(self.adapter.managers),'native attention manager order mismatch')
    logical=request.num_computed_tokens;width=self.adapter.page_tokens
    selected,run,L=tuple(plan['selected']),tuple(plan['run_pages']),plan['candidate_end']
    dropped=L-len(selected)*width
    require(plan['sparse']=={'logical_matched_tokens':L,'dropped_full_attention_tokens':dropped},
            'plan SparsePlan differs from its selected Pages')
    require(type(logical) is int and logical>=start and request.request_id==binding.request,
            'native logical cursor precedes the phase-2 start')
    count=values['num_new_tokens']
    require(type(count) is int and count>0,'positive native scheduled interval required')
    # FA groups: physical = logical - (L - |A_t|P); Mamba stays logical.
    cursors=tuple(logical-dropped if i in attention else logical for i in range(len(groups)))
    supplied=values.get('computed_tokens_by_group')
    if supplied is not None and tuple(supplied)!=cursors:
        from .allocation_guard import reject
        reject('computed_tokens_by_group', supplied, 'foreign native compact counters')
    external=values.get('external_computed_tokens_by_group')
    if external is not None and any(external):
        from .allocation_guard import reject
        reject('external_computed_tokens_by_group', external, 'external selective prefix forbidden')
    values['computed_tokens_by_group']=cursors;values['skip_cache']=True
    # The native method remains the allocator. The fixed head [A_t][run] is
    # installed at admission, so allocation only appends transients: no
    # capacity floor (_remote_ra_chunk_capacity stays unset).
    result=native(*call.args,**call.kwargs)
    require(request.num_computed_tokens==logical,'native allocator changed logical/RoPE cursor')
    if result is None:return None
    raw={group:(tuple(b.block_id for b in m.req_to_blocks[binding.request]),m.block_size)
         for group,m in zip(attention,self.adapter.managers)}
    target,handover=record['target'],record['run']
    layout=RowLayout(width,L,selected,
        {g:tuple(target[p][i][0].block_id for p in selected) for i,g in enumerate(attention)},
        {g:tuple(handover[p][i][0].block_id for p in run) for i,g in enumerate(attention)})
    tables,sparse=compose_rows(layout,raw)
    require(sparse==SparsePlan(L,dropped,attention),'composed SparsePlan differs from the plan')
    for group,m in zip(attention,self.adapter.managers):
        current=m.req_to_blocks[binding.request];by_id={b.block_id:b for b in current}
        ids=tables[group][0]
        prior=record.get('native_tables',{}).get(group,())
        require(tuple(ids[:len(prior)])==prior,'native sparse prefix moved after publication')
        current[:]=[by_id[b] for b in ids]
    record['native_tables']={g:ids for g,(ids,size) in tables.items()}
    record['native_allocation']=dict(binding=binding.identity,generation=binding.generation,
        logical_cursor=logical,computed_tokens_by_group=cursors,
        dropped_full_attention_tokens=dropped)
    from .pool_incarnations import rows
    native_rows=rows(manager,binding.request)
    update=self.proxy.call('allocation_update',dict(request=binding.request,binding=binding.identity,
        start=logical,count=count,rows=native_rows))
    require(update.get('operation')=='allocation_update' and update.get('binding')==binding.identity,
            'worker allocation incarnation update not acknowledged')
    record['native_allocation']['rows']=native_rows
    record['scheduled_cursor']=interval['end']
    record.setdefault('scheduled_intervals',[]).append(interval)
    return result
'''),

    ('remote_ra_live_v2.engine_boundary', 'EngineRpcBoundary.complete_and_drain', (), r'''
def complete_and_drain(self, binding, plan, output_token_ids, computed_end, sealed_end):
    worker = self.proxy.call("complete", dict(request=binding.request,
        session=binding.session, turn=binding.turn, local_turn=binding.local_turn,
        generation=binding.generation, binding=binding.identity,
        output_token_ids=list(output_token_ids), computed_end=computed_end,
        sealed_end=sealed_end))
    receipt = self._receipt(binding, plan, "complete", worker)
    self.receipts[(binding.request, "complete")] = receipt
    return receipt
'''),

    ('remote_ra_live_v2.engine_boundary', 'EngineRpcBoundary.cancel_and_drain', (), r'''
def cancel_and_drain(self, binding, plan):
    worker = self.proxy.call("cancel", dict(request=binding.request,
        session=binding.session, turn=binding.turn,
        generation=binding.generation, binding=binding.identity))
    receipt = self._receipt(binding, plan, "cancel", worker)
    self.receipts[(binding.request, "cancel")] = receipt
    return receipt
'''),

    ('remote_ra_live_v2.capacity', 'turn_need', (), r'''
def turn_need(core, session_id, tokens, messages, generation):
    """(phase-1 peak, phase-2 peak) in blocks beyond the session's current pins."""
    adapter = core._remote_ra_growing_adapter
    endpoint = core._remote_ra_private_v2_endpoint
    s, P = adapter.sessions[session_id], adapter.page_tokens
    att, mam, _ = endpoint.phase1.groups()
    A, M = len(att), len(mam)
    end = pages(len(tokens) + generation, P)
    if s.local_turn == 0:
        return 0, A * end + 2 * M
    transcript = endpoint.transcripts.get(session_id)
    if transcript is not None and transcript.turn > 0 and not transcript.extends(messages):
        # Rebuild: render closes the namespace (frees R and the tail) first.
        return 0, A * (end - len(s.resident) - (s.tail is not None)) + 2 * M
    Q = s.sealed_end
    base = Q // P + (s.tail is not None)          # first Page past the pinned tail
    candidates = {Q // P, base}                   # n = floor(o/P) is one of them
    contract = getattr(endpoint.selection, 'contract', None)
    initial = transcript.initial_length if transcript is not None else 0
    largest = max(selected_max(s.policy, n, s.local_turn, contract, initial) for n in candidates)
    restores = min(Q // P - len(s.resident), largest)
    goal = 0
    if s.policy == 'goal_query' and contract is not None:
        from .selection_contract import selection_active
        if any(selection_active(n, s.local_turn, contract) for n in candidates):
            goal = pages(endpoint.phase1.max_tokens, P)   # phase 1 refuses a longer Goal
    fresh = max(0, pages(len(tokens), P) - base)
    phase1 = A * (fresh + goal) + M * (1 + (goal > 0))
    phase2 = A * (end - base + restores) + 2 * M
    return phase1, phase2
'''),

    ('remote_ra_live_v2.capacity', 'headroom', (), r'''
def headroom(lifecycle):
    """Blocks the running (accepted, undrained) requests may still allocate."""
    endpoint = lifecycle.endpoint.core._remote_ra_private_v2_endpoint
    adapter = lifecycle.endpoint.core._remote_ra_growing_adapter
    att, mam, _ = endpoint.phase1.groups()
    A, M, P = len(att), len(mam), adapter.page_tokens
    total = 0
    for rid, entry in lifecycle.entries.items():
        request = entry['request']
        if entry['state'] != 'accepted' or request is None:
            continue
        geometry = endpoint.claims.get(rid, {}).get('geometry')
        start = 0 if geometry is None else geometry.phase2_start
        done = request.num_computed_tokens
        left = max(0, pages(len(request.prompt_token_ids) + max_tokens(request.sampling_params), P)
                   - pages(done, P))
        total += A * left + M * (left > 0) + M * (done <= start)
    return total
'''),

    ('remote_ra_live_v2.capacity', 'assess', (), r'''
def assess(lifecycle, command):
    """The gate's numbers for one authenticated add, or None when the request's
    own checks will refuse it anyway (unknown, busy, poisoned or stale session,
    undecodable core or source)."""
    core = lifecycle.endpoint.core
    adapter = core._remote_ra_growing_adapter
    meta = command.metadata
    s = adapter.sessions.get(command.session_id)
    if (s is None or s.poisoned or s.pending is not None or type(meta) is not dict
            or meta.get('turn') != s.turn):
        return None
    try:
        native = core._remote_ra_assembly.dispatcher._decode_core(
            base64.b64decode(command.core_b64, validate=True))
        tokens = list(native.prompt_token_ids)
        messages = None
        if core._remote_ra_private_v2_endpoint.transcripts.get(command.session_id) is not None:
            # XML protocol: the messages the engine transcript renders (rebuild or not).
            from runtime_renderer import validate_xml_source
            messages = [dict(role=m['role'], content=m['content'])
                        for m in validate_xml_source(command.source)['messages']]
    except Exception:
        return None
    phase1, phase2 = turn_need(core, command.session_id, tokens, messages,
                               max_tokens(getattr(native, 'sampling_params', None)))
    manager = adapter.kv_manager
    watermark = getattr(manager, 'watermark_blocks', 0)
    return dict(free=manager.block_pool.get_num_free_blocks(), phase1=phase1, phase2=phase2,
                headroom=headroom(lifecycle), watermark=watermark if type(watermark) is int else 0)
'''),

    ('remote_ra_live_v2.capacity', 'admission_gate', (), r'''
def admission_gate(lifecycle, command):
    """Raise CapacityWait unless free blocks cover this turn and every running request.

    On a server that parks goal_query admissions at their Goal (goal_async), the
    running requests keep decoding between this gate and phase 1, so phase 1 must
    fit even after they have taken all their headroom (a refusal only delays the
    turn: the retry is the same turn). And no add starts while one is parked: it
    has reserved nothing yet, so the pool does not show it (the frontend's
    admission lock, held until acceptance, already keeps that from happening)."""
    goals = getattr(lifecycle, 'goals', None)
    if goals is not None and goals.parked:
        raise CapacityWait('%s: another admission waits for its Goal' % CAPACITY_WAIT)
    if not any(e['state'] == 'accepted' for e in lifecycle.entries.values()):
        return None
    need = assess(lifecycle, command)
    if need is None:
        return None
    later = need['headroom'] if goals is not None else 0    # decode while the Goal is built
    if (need['phase1'] + later > need['free'] or
            need['phase2'] + need['headroom'] + need['watermark'] > need['free']):
        raise CapacityWait('%s: turn needs %d (phase 1) / %d (phase 2) blocks, running requests '
                           '%d more, watermark %d; %d free' % (CAPACITY_WAIT, need['phase1'],
                           need['phase2'], need['headroom'], need['watermark'], need['free']))
    return need
'''),

    ('remote_ra_live_v2.composition', 'compose_engine', (), r'''
def compose_engine(core, ticket):
    """Attach the active protocol-v2 lifecycle before B0/IO threads."""
    if ticket is None:
        return None
    state = core._remote_ra_live_v2
    assembly = core._remote_ra_assembly
    if not hasattr(core,'_remote_ra_trusted_issue'):
        task=dict(core._remote_ra_growing_owner.smoke_config['task'])
        require(set(task)=={'instance_id','problem_statement'},'trusted startup task record required')
        core._remote_ra_trusted_issue=task
    if not hasattr(core,'_remote_ra_trusted_issues'):
        # Indexed like the sessions; _remote_ra_trusted_issue stays tasks[0].
        core._remote_ra_trusted_issues=trusted_tasks(core._remote_ra_trusted_issue)
    startup = {"enabled": True,
               "source": core._remote_ra_growing_owner.identity.manifest,
               "page_tokens": assembly.provider.provenance.page_tokens,
               "loaded_sources_verified": True}
    replies = core.model_executor.collective_rpc(
        "remote_ra_live_v2_attach", args=(startup,), timeout=30)
    require(type(replies) is list and len(replies) == 1,
            "one loaded worker attachment required")
    proxy = EngineWorkerProxy(core.model_executor, replies[0])
    from .engine_boundary import EngineRpcBoundary
    boundary = EngineRpcBoundary(proxy, core._remote_ra_growing_adapter)
    core._remote_ra_growing_adapter.backend = boundary
    core._remote_ra_live_v2_boundary = boundary
    from .session_authority import LiveSessionAuthority
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, proxy)
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        from .private_endpoint import SplitPrivateEndpoint
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
    transcript = core._remote_ra_private_v2_endpoint.transcript
    if transcript is not None:
        # XML protocol: the engine stops on the action marker itself, so the
        # request finishes as complete instead of a frontend stop-string abort.
        # One policy stops every session's request: it decodes only the
        # request's own output, through the decoder all transcripts share.
        from .mswea_protocol import MarkerStop
        require(all(t.decode == transcript.decode
                    for t in core._remote_ra_private_v2_endpoint.transcripts.values()),
                "engine transcripts must share one decoder")
        require(getattr(core.scheduler, "_remote_ra_output_stop", None) is None,
                "output stop policy attachment collision")
        core.scheduler._remote_ra_output_stop = MarkerStop(transcript.decode)
    endpoint = LiveEndpoint(core, assembly, proxy)
    lifecycle = EngineLifecycle(endpoint)
    bridge = EngineBridge()
    core._remote_ra_live_v2_endpoint = endpoint
    core._remote_ra_live_v2_lifecycle = lifecycle
    core._remote_ra_live_v2_bridge = bridge
    core.scheduler._remote_ra_live_v2_bridge = bridge
    # EngineAssembly itself is constructed with the v3 lifecycle owner (not
    # lifecycle_hooks.EngineLifecycle).  Bind its protocol-v2 half now, after
    # the loaded worker receipt and before B0/IO threads.
    assembly.lifecycle.bind_native(lifecycle)
    from .registered_route import bind_registered_dispatcher
    bind_registered_dispatcher(assembly.dispatcher, lifecycle)
    # Multi-session stage 2: when the sessions decode together, a goal_query
    # admission waits for its Goal off the engine thread (goal_async); a parked
    # Goal wakes the idle engine loop through its input queue.
    from vllm.v1.engine import EngineCoreRequestType
    from .goal_async import attach
    attach(core, lifecycle, max_num_seqs=core.vllm_config.scheduler_config.max_num_seqs,
           wake=lambda: core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None)))
    bind = __import__("remote_ra_live_v2.startup", fromlist=["bind_loaded_worker"])
    from remote_ra_http.connector import isolate_connector
    import os
    if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1":
        isolate_connector(core)
    bind.bind_loaded_worker(core, worker_backend=proxy, qwen_lifecycle=bridge,
        engine_lifecycle=lifecycle, cuda_evidence=replies[0])
    return state
'''),

    ('remote_ra_native_growing.adapter', 'EngineCoreAdapter.__init__', (), r'''
def __init__(self, authority, kv_manager, *, page_tokens, full_attention_managers,
             source_test_backend=None):
    require(type(authority) is Authority, "native private authority required")
    self.authority, self.kv_manager = authority, kv_manager
    self.page_tokens, self.managers = page_tokens, tuple(full_attention_managers)
    # Stage B (#28): one FullAttention block per Page per group, w = 1.
    require(type(page_tokens) is int and page_tokens > 0 and self.managers and
            all(m.block_size == page_tokens for m in self.managers),
            "Stage B requires one FullAttention block per Page (w = 1)")
    self.backend = source_test_backend or CompiledWorkerBoundary()
    self.source_test_only = source_test_backend is not None
    self.owner = PrivateBlockOwner(kv_manager.block_pool)
    self.sessions, self.requests = {}, {}
    self._unsubmitted_cancelled = {}
'''),

    ('remote_ra_native_growing.adapter', 'EngineCoreAdapter.reserve', (), r'''
def reserve(self, request, binding, mac, *, geometry, ranking=(), selected_count=None,
            query_receipt=None, phase1=None, handover=None):
    """Admit phase 2 at e (spec 3.4). Ownership of ``handover`` (the phase-1
    fresh FA blocks, Page -> groups) moves to this call; failures free it."""
    handover = {} if handover is None else handover
    s = pending = None
    adopted = False
    try:
        self.authority.authenticate(request, binding, mac)
        s = self.sessions[binding.session]
        require(not s.poisoned and s.pending is None and s.turn == binding.turn and
                s.local_turn == binding.local_turn and binding.model == s.model and
                binding.renderer == s.renderer,
                "session lineage or immutable identity mismatch")
        require(binding.request not in self.requests and s.turn < 256,
                "native replay/turn limit")
        P = self.page_tokens
        geo = turn_geometry(binding, P, s)
        require(geo == geometry, "turn geometry changed after phase 1")
        require((phase1 is None) == (geo.local_turn == 0) and
                (phase1 is None or (phase1.binding == binding.identity and
                                    phase1.start == geo.C and phase1.end == geo.e)),
                "phase-1 result does not belong to this turn")
        require(geo.n <= 4096, "Page catalog limit")
        selected, bootstrap = select_pages(s.policy, geo.n, geo.local_turn,
                                           binding.native_prefix, ranking, selected_count)
        if s.policy in QUERY:
            require(query_receipt is not None, "authenticated model query receipt required")
        config = getattr(self.kv_manager, "kv_cache_config", None)
        self.backend.verify_zero_before_restore_config(config)
        # Stop before any ownership mutation when physical worker support is absent.
        self.backend.require_ready()
        self.backend.validate_layout(binding, self.managers, query_receipt, ranking,
                                     selected_count)
        ns = s.namespace
        R = tuple(sorted(self.owner.sessions.get(ns, {})))
        require(R == tuple(s.resident) and
                self.tail_mirror(self.owner.tails.get(ns)) == s.tail,
                "resident/tail mirrors differ from the native session pins")
        # R' = R + the tail page when phase 1 completed it (SPEC!=DESIGN 1).
        candidate = {geo.Q // P} if geo.tail_candidate else set()
        Rp = set(R) | candidate
        retain = sorted(set(selected) & Rp)
        restore = sorted(set(selected) - Rp)
        release = sorted(Rp - set(selected))
        require(all(p in s.catalog for p in restore), "restored Page missing from the catalog")
        require(all(p in s.catalog or p in candidate for p in release),
                "released Page has no CPU copy")
        run = list(geo.run_pages)
        plan = {
            "binding": binding.identity, "namespace": ns, "ns_generation": s.ns_generation,
            "policy": s.policy, "local_turn": geo.local_turn, "page_tokens": P,
            "computed_start": geo.C, "sealed_start": geo.Q, "candidate_end": geo.L,
            "observation": list(binding.observation_span) if geo.local_turn else None,
            "phase2_start": geo.phase2_start, "prompt_end": geo.N,
            "selected": list(selected), "resident_before": list(R),
            "tail_before": ({"page": geo.Q // P, "valid_end": geo.C} if geo.tail else None),
            "tail_candidate": geo.tail_candidate, "tail_live": geo.tail_live,
            "retain": retain, "restore": restore, "release": release,
            "run_pages": run, "run_tail_page": geo.Q // P if geo.tail_live else None,
            "sparse": {"logical_matched_tokens": geo.L,
                       "dropped_full_attention_tokens": geo.L - len(selected) * P},
            "seal_at_completion": {"from_page": geo.L // P, "rule": "floor(C'/P)",
                                   "phase1_sealed": sorted(candidate)},
            "phase1": None if phase1 is None else {
                "lease_id": phase1.lease_id, "lease_digest": phase1.lease_digest,
                "receipt_digest": digest(dict(phase1.worker_receipt))},
            "bootstrap": bootstrap,
        }
        pending = {"binding": binding, "plan": plan, "request": request, "geometry": geo,
                   "phase1": phase1, "phase": "allocating", "worker_started": False}
        s.pending = pending
        self.requests[binding.request] = s
        adopted = True  # PrivateBlockOwner.begin frees the handover on any failure.
        lease = self.owner.begin(ns, binding.request, geo, tuple(selected),
                                 len(self.managers), handover)
        self.owner.install_tables(binding.request, self.managers)
        pending["phase"] = "installed"
        # Once the worker may have seen a command, every exception requires
        # authenticated cancellation/drain, even before staging returns.
        self.owner.mark_submitted(binding.request)
        pending["worker_started"] = True
        self.backend.stage_reservation(binding, plan, lease, s.catalog)
        pending["phase"] = "reserved"
        return plan
    except BaseException as failure:
        if s is not None:
            s.poisoned = True
        cleanup = []
        try:
            if not adopted:
                self.owner.discard_handover(handover)
            if pending is not None and not pending["worker_started"]:
                self._cleanup_unsubmitted(binding, s)
        except BaseException as error:
            cleanup.append(error)
        if cleanup:
            raise BaseExceptionGroup("admission and handover cleanup failed",
                                     [failure, *cleanup]) from None
        raise
'''),

    ('remote_ra_native_growing.adapter', 'EngineCoreAdapter.complete', (), r'''
def complete(self, binding):
    """Terminal commit (spec 3.8): seal floor(C'/P), pin A_t + [L/P, Q'/P) and
    the new tail, commit the carry, then publish the turn's vectors."""
    s = self.requests[binding.request]
    p = s.pending
    require(not s.poisoned and p and p["binding"] == binding and
            p["phase"] in {"running", "reserved"}, "completion lineage mismatch")
    try:
        geo, plan, P, rid = p["geometry"], p["plan"], self.page_tokens, binding.request
        output = tuple(p["request"].output_token_ids)
        require(output and all(type(t) is int and t >= 0 for t in output),
                "generated token IDs required")
        # vLLM never computes the last sampled token: C' = N + g - 1.
        full = binding.tokens + output
        C2 = len(full) - 1
        Q2 = C2 // P * P
        r = self.backend.complete_and_drain(binding, plan, list(output), C2, Q2)
        self._verify_receipt(binding, plan, r, "complete")
        require(r["computed_end"] == C2 and r["sealed_end"] == Q2,
                "worker computed/sealed end differs from prompt + output")
        base = geo.L // P
        new = _pages(r["new_pages"])
        expected = set(range(base, Q2 // P)) | ({geo.Q // P} if geo.tail_candidate else set())
        require(set(new) == expected and not set(new) & set(s.catalog) and
                set(s.catalog) | set(new) == set(range(Q2 // P)),
                "worker must seal exactly [L/P, Q'/P) plus a phase-1 tail candidate")
        for page, record in new.items():
            require(type(record) is dict and
                    set(record) == {"token_sha256", "namespace", "binding", "source", "content"}
                    and record["token_sha256"] == digest(full[page*P:(page+1)*P]) and
                    record["namespace"] == s.namespace and
                    record["binding"] == binding.identity and
                    record["source"] == self.authority.source and _hex64(record["content"]),
                    "foreign/stale Page publication")
        resident = tuple(sorted(set(plan["selected"]) | set(range(base, Q2 // P))))
        require(tuple(r["resident"]) == resident and tuple(r["dram"]) == tuple(range(Q2 // P)),
                "native working set/catalog receipt mismatch")
        blocks = {page: _block_ids(self.owner.row_groups(rid, page)) for page in resident}
        require({k: _wire_ids(v) for k, v in _pages(r["resident_blocks"]).items()} == blocks,
                "worker resident blocks differ from the engine row")
        tail = None
        if C2 > Q2:
            tail = {"page": Q2 // P, "start": Q2, "valid_end": C2,
                    "block_ids": _block_ids(self.owner.row_groups(rid, Q2 // P))}
        wire_tail = r["tail"]
        require((wire_tail is None and tail is None) or
                (type(wire_tail) is dict and tail is not None and
                 dict(wire_tail, block_ids=_wire_ids(wire_tail.get("block_ids"))) == tail),
                "worker tail differs from the engine row")
        carry = r["carry"]
        require(type(carry) is dict and set(carry) == {"computed_end", "digest"} and
                carry["computed_end"] == C2 and _hex64(carry["digest"]) and
                carry["digest"] != ZERO_DIGEST and r["boundary"] == carry["digest"],
                "carry must be committed at the computed end")
        vectors = {page: tuple(v) for page, v in _pages(r["page_vectors"]).items()}
        require(set(vectors) == set(range(geo.phase2_start // P, C2 // P)),
                "phase-2 vectors must cover exactly the Pages closed in [S2, C')")
        self.owner.commit(rid, keep_pages=resident,
                          tail=(Q2 // P, Q2, C2) if tail is not None else None)
        s.catalog.update(new)
        s.sealed_tokens, s.computed_tokens, s.next_token = full[:Q2], full[:C2], full[C2]
        s.boundary, s.tail, s.resident = carry["digest"], tail, resident
        s.turn += 1
        s.local_turn += 1
        p["phase"] = "committed"
        if s.policy in QUERY:
            selection = getattr(self.backend, "selection", None)
            require(selection is not None, "query selection authority missing at completion")
            phase1 = p["phase1"]
            selection.publish_completion(binding, geo,
                dict(phase1.page_vectors) if phase1 is not None else {}, vectors, Q2, C2)
        return r
    except BaseException:
        s.poisoned = True
        raise
'''),

    ('remote_ra_native_growing.ownership', 'PrivateBlockOwner._validate', (), r'''
def _validate(self, blocks, namespace, *, may_be_new=False):
    ids = [b.block_id for b in blocks]
    require(len(ids) == len(set(ids)), "duplicate block aliases in private table")
    for b in blocks:
        require(type(b.block_id) is int and 0 <= b.block_id < len(self.pool.blocks)
                and self.pool.blocks[b.block_id] is b and not b.is_null and
                b.ref_cnt > 0 and b.block_hash is None,
                "not an exclusively private allocated native block")
        owner = self.block_namespaces.get(b.block_id)
        require(owner == namespace or (may_be_new and owner is None),
                "foreign sparse namespace block")
'''),

    ('remote_ra_native_growing.ownership', 'PrivateBlockOwner.discard_handover', (), r'''
def discard_handover(self, handover):
    """Free adopted phase-1 blocks that never reached a lease (F: 1 -> 0)."""
    blocks = self._blocks(handover)
    require(len({b.block_id for b in blocks}) == len(blocks) and
            all(b.ref_cnt == 1 and self.block_namespaces.get(b.block_id) is None
                for b in blocks), "handover blocks changed owner before adoption")
    self.pool.free_blocks(reversed(blocks))
'''),

    ('remote_ra_native_growing.ownership', 'PrivateBlockOwner.begin', (), r'''
def begin(self, namespace, request_id, geometry, selected, groups, handover):
    """Admission (spec 3.4/3.7): borrow A_t & R' and a live tail, allocate
    restore targets, adopt the phase-1 run blocks.

    Ownership of ``handover`` moves to this call: every failure frees it.
    """
    geo = geometry
    if (request_id in self.leases or request_id in self.finished or
            any(x.namespace == namespace for x in self.leases.values())):
        self.discard_handover(handover)
        raise Abort("duplicate native request ownership or session already has native work")
    lease = Lease(namespace, request_id, dict(self.sessions.get(namespace, {})),
                  self.tails.get(namespace), geo, groups, tuple(selected),
                  adopted=list(self._blocks(handover)))
    self.leases[request_id] = lease
    try:
        P, old, tail = geo.page_tokens, lease.old, lease.old_tail
        require(type(groups) is int and groups > 0 and
                all(self._page_groups(v, groups) for v in old.values()) and
                (tail is None or self._page_groups(tail.groups, groups)),
                "native Page layout changed")
        self._validate(self.pinned(namespace), namespace)
        require(all(p < geo.Q // P for p in old),
                "resident Page at or after the sealed end")
        require((tail is not None) == geo.tail and
                (tail is None or (tail.page, tail.start, tail.valid_end) ==
                 (geo.Q // P, geo.Q, geo.C)), "session tail pin differs from turn geometry")
        require(type(selected) is tuple and selected == tuple(sorted(set(selected))) and
                all(type(p) is int and 0 <= p < geo.n for p in selected),
                "invalid selected Page table")
        require(type(handover) is dict and set(handover) == set(geo.fresh_pages) and
                all(self._page_groups(v, groups) for v in handover.values()),
                "handover must be exactly the phase-1 fresh run Pages")
        ids = [b.block_id for b in lease.adopted]
        require(len(ids) == len(set(ids)) and
                all(b.ref_cnt == 1 and b.block_hash is None and not b.is_null and
                    self.block_namespaces.get(b.block_id) is None for b in lease.adopted),
                "handover block is not an exclusive phase-1 allocation")
        candidate = geo.Q // P if geo.tail_candidate else None
        restore = [p for p in selected if p not in old and p != candidate]
        require(len(restore) * groups <= self.pool.get_num_free_blocks(),
                "insufficient native blocks")
        for page in selected:
            if page in old or page == candidate:
                pinned = old[page] if page in old else tail.groups
                blocks = self._blocks({page: pinned})
                self.pool.touch(blocks)  # request borrow: S -> S+Q
                lease.acquired.extend(blocks)
                lease.target[page] = pinned
            else:
                fresh = []
                for _ in range(groups):
                    block = self.pool.get_new_blocks(1)[0]
                    lease.acquired.append(block)
                    fresh.append((block,))
                self._validate(tuple(b for (b,) in fresh), namespace, may_be_new=True)
                for (b,) in fresh:
                    self.block_namespaces[b.block_id] = namespace
                lease.restore_blocks.extend(b for (b,) in fresh)
                lease.target[page] = tuple(fresh)
        for page in geo.run_pages:
            if geo.tail_live and page == geo.Q // P:
                blocks = self._blocks({page: tail.groups})
                self.pool.touch(blocks)  # the live tail continues: S -> S+Q
                lease.acquired.extend(blocks)
                lease.run[page] = tail.groups
            else:
                lease.run[page] = handover[page]
                for (b,) in handover[page]:
                    self.block_namespaces[b.block_id] = namespace
        self._validate(self._blocks(lease.target) + self._blocks(lease.run), namespace)
        lease.phase = "allocated"
        return lease
    except BaseException:
        # These references have not reached a worker yet; synchronous undo.
        self.abort_unsubmitted(request_id)
        raise
'''),

    ('growing_startup', 'attach_dormant', (), r'''
def attach_dormant(core, ticket):
    # Normal vLLM construction never imports native authority or changes state.
    if ticket is None:
        return None
    from remote_ra_linux.native_authority import current, NativeEngine
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    owner = current()
    require(type(owner) is NativeEngine and core._remote_ra_authority is owner
            and owner.core is core, 'canonical engine-owned authority required')
    owner.owner_check(); owner.verify_identity()
    for name in ('growing_startup', 'remote_ra_native_growing.adapter',
                 'remote_ra_native_growing.contract', 'remote_ra_native_growing.ownership',
                 'vllm.v1.core.single_type_kv_cache_manager'):
        owner.loaded.verify_module(name)
    require(ticket.core is core and ticket.phase == 'active'
            and core._remote_ra_startup_ticket is ticket,
            'growing attachment requires pre-B0 startup ticket')
    require(not hasattr(core, 'output_thread') and not getattr(core, 'engines_running', False),
            'growing attachment after engine serving')
    scheduler = core.scheduler
    require(all(len(getattr(scheduler, n)) == 0 for n in ('requests','waiting','running')),
            'growing attachment requires empty scheduler')
    manager = scheduler.kv_cache_manager
    offload = getattr(getattr(scheduler, 'connector', None), 'connector_scheduler', None)
    require(offload is not None, 'native offloading scheduler required')
    targets = (core, scheduler, manager, offload)
    require(all(not hasattr(t, '_remote_ra_growing_adapter') for t in targets),
            'growing attachment collision')
    managers = tuple(m for m in manager.coordinator.single_type_managers
                     if type(m) is FullAttentionManager)
    authority = Authority(epoch=owner.identity.epoch, loaded_source=owner.loaded.manifest_hash)
    adapter = EngineCoreAdapter(authority, manager,
        page_tokens=owner.smoke_config['page_tokens'], full_attention_managers=managers)
    require(adapter.capabilities()['native_ready'] is False, 'unexpected live backend')
    for target in targets:
        target._remote_ra_growing_adapter = adapter
    core._remote_ra_growing_port = NativePort(adapter)
    core._remote_ra_growing_owner = owner
    return adapter
'''),

    ('remote_ra_live_v2.model_queries', 'phase1', (), r'''
def phase1(owner, backend, rec, payload):
    """Check the engine phase-1 request against the prepared turn, run it, sign it."""
    a, identity = rec['assignment'], rec['identity']
    C, e, P = a.computed_start, a.observation_end, a.page_tokens
    require(a.local_turn > 0 and rec['phase1'] is None and 'plan' not in rec,
            'phase 1 runs once, after prepare and before reserve')
    require(payload['allocation_generation'] == rec['generation'] and
            payload['binding'] == rec['binding'] and payload['source'] == identity.source and
            payload['model'] == identity.model and
            (payload['session'], payload['turn'], payload['generation'], payload['local_turn'])
            == (identity.session, identity.turn, identity.generation, a.local_turn) and
            list(payload['tokens']) == list(rec['tokens'][:e]) and
            payload['prompt_digest'] == digest(list(rec['tokens'])) and
            hex64(payload['request_digest']), 'stale phase-1 allocation/render')
    lease = parse_lease(payload['lease'], attention_groups=owner.attention_groups)
    s = owner.sessions[rec['namespace']]
    tail = s['tail']
    require(lease.lease_digest == payload['lease_digest'] and
            (lease.request, lease.binding, lease.session, lease.namespace, lease.generation,
             lease.source, lease.epoch) ==
            (identity.request, rec['binding'], identity.session, rec['namespace'],
             rec['generation'], identity.source, payload['epoch']) and
            sha256(lease.epoch.encode()).hexdigest() == identity.epoch and
            (lease.local_turn, lease.page_tokens, lease.start, lease.end, lease.sealed_start,
             lease.candidate_end, lease.tail_candidate) ==
            (a.local_turn, P, C, e, a.sealed_start, a.candidate_end, a.tail_candidate) and
            lease.carry_digest == a.carry_digest == s['carry_digest'] and
            lease.resident_pages == tuple(s['resident']) and
            (lease.tail is None) == (tail is None), 'phase-1 lease differs from the prepared turn')
    # Borrowed = the session's resident Page blocks in R order, then the tail
    # block: exactly what phase 1 reads, never zeroed or freed by the lease.
    for fi, g in enumerate(owner.attention_groups):
        expect = tuple(s['resident_blocks'][p][g][0] for p in lease.resident_pages)
        if tail is not None:
            require(lease.tail.block_ids[fi] == tail['blocks'][g], 'phase-1 tail block changed')
            expect += tail['blocks'][g]
        require(lease.rows[g][:len(expect)] == expect, 'phase-1 borrowed blocks differ from pins')
    driver = getattr(owner.runner, '_remote_ra_native_query_forward', None)
    require(callable(getattr(driver, 'phase1', None)),
            'LOADED_QUERY_FORWARD_UNAVAILABLE: native phase-1 driver is not attached')
    start = time.perf_counter_ns()
    output = driver.phase1(payload)
    import torch
    rows, goal_rows = output['rows'], output['goal_rows']
    require(isinstance(rows, torch.Tensor) and rows.ndim == 2 and rows.shape[0] == e - C and
            bool(torch.isfinite(rows).all()), 'exact phase-1 forward rows required')
    rows = rows.detach().to(device='cpu', dtype=torch.float32)
    from .query_selection import normalized
    # Pages whose last token lies in [C, e): first computation, used now and
    # committed at completion; the observation query is row e-1.
    vectors = {p: normalized(rows[(p + 1) * P - 1 - C].tolist()) for p in closed_pages(C, e, P)}
    observation = normalized(rows[e - 1 - C].tolist())
    goal = None
    if lease.goal is not None:
        require(isinstance(goal_rows, torch.Tensor) and goal_rows.ndim == 2 and
                goal_rows.shape[0] == len(lease.goal.tokens) and
                bool(torch.isfinite(goal_rows).all()), 'exact Goal forward rows required')
        goal = normalized(goal_rows[-1].detach().to(device='cpu', dtype=torch.float32).tolist())
    else:
        require(goal_rows is None, 'Goal rows without a Goal lease')
    require(len(observation) == rows.shape[1] and (goal is None or len(goal) == len(observation)),
            'query/model hidden dimension mismatch')
    state, captured = output['state'], output['tail_capture']
    require(state.request == identity.request and state.lease_id == lease.lease_id and
            state.end == e and (captured is not None) == lease.tail_candidate,
            'phase-1 state/tail capture incomplete')
    tail_digest = backend.io.payload_digest(captured) if captured is not None else None
    backend.io.drain()
    body = dict(operation='phase1', binding=rec['binding'], request_digest=payload['request_digest'],
        lease_id=lease.lease_id, lease_digest=lease.lease_digest,
        allocation_generation=rec['generation'], start=C, end=e, forward_tokens=e - C,
        page_vectors={str(p): list(v) for p, v in vectors.items()},
        observation_vector=list(observation), goal_vector=None if goal is None else list(goal),
        goal_token_count=None if goal is None else len(lease.goal.tokens),
        goal_tokens_sha256=None if goal is None else digest(list(lease.goal.tokens)),
        goal_payload_sha256=None if goal is None else lease.goal.payload_sha256,
        state_digest=state.digest, tail_capture_digest=tail_digest,
        carry_restored=dict(start=C, digest=lease.carry_digest),
        position_evidence=output['position_evidence'],
        vector_space=dict(getattr(owner.runner, '_remote_ra_vector_space', None) or {}),
        representation='pre_final_norm_last_token_l2', phase1_ns=time.perf_counter_ns() - start,
        drained=True)
    receipt = backend.sign_receipt(identity, 'phase1', body)
    backend.verify_receipt(identity, 'phase1', receipt)
    rec['phase1'] = dict(lease=lease, vectors=vectors, obs=observation, goal=goal,
        tail_capture=captured, tail_capture_digest=tail_digest, state_digest=state.digest,
        receipt=receipt)
    return dict(body, worker_receipt=receipt)
'''),

)


if __name__ == '__main__':
    unittest.main()
