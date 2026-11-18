"""--async-scheduling on the Gemma 4 profile (async-gemma; docs/async-sched/GEMMA.md). CPU only; not GPU evidence.

Option C as async-c built it for Qwen (remote_ra_live_v2.async_sched; test_async_sched), on the window layout:
no Gated-DeltaNet state, six KV groups (the global group in pool 0, five sliding-window groups in pool 1), the
engine's window follower (fast paths 2 and 3) and WindowPins, the --goal-overlap park after phase 1.

1. The pin: the Gemma config check; serve.prepare_gemma4 with and without the switch (one config key, one effective
   key, --async-scheduling in place of --no-async-scheduling) on mswea-xml, chat-text and kv-session, with
   --goal-overlap, CUDA graphs and 4,096-token forwards; the loader, the frontend's parsed-flag check and the
   gates; the refusals; the window worker's record check, attachment and echo.
2. The void on window-layout metadata (eager and persistent buffers): slot -1 in the global and every sliding
   group, no GDN state index; its refusals.
3. End to end on test_gemma_e2e's engine and worker (vLLM's KV manager with the sliding managers, WindowPins,
   WindowPhase1Coordinator, WindowAllocationOwner, WindowLifecycle, a runner that checks every full-attention and
   sliding read) in vLLM's asynchronous order (schedule and execute N+1, then update N; the worker resolves N in
   before_forward; the batch in flight drained before every admission and Goal resume; vLLM's max_tokens skip:
   a lane's turns stop alternately on an EOS, voided, and on max_tokens, not scheduled past and taken out of the
   runner's batch and prev_req_id_to_index, GPU job 3568311), against the synchronous server with the same fast
   paths (2, or 2 and 3):
   - mswea-xml (test_gemma_release's scheduler) at b = 1 and b > 1, the stop positions chosen so that both
     Gemma-specific speculative steps happen (C' % 16 == 0: a fresh global and sliding block for C';
     (C' - W + 1) % 16 == 0: the window's front block freed one step early);
   - --goal-overlap (test_gemma_goal_overlap's harness: Goals parked on their threads, phase 1 before the Goal
     join, the Goal forward as its own lease, --admission-overlap);
   - kv-session (test_gemma_kv_session_e2e's MemAgent lanes, Gemma's stop ids);
   - chat-text (test_gemma_chat_text_e2e's API server; its engine loop runs core.py's own _process_input_queue,
     whose drains are the shipped ones).
   Per turn: the replies, prompts, selections, attended Pages, Goal vectors, the positions computed (exactly once,
   none at or past C'), the committed window (its block indices; the engine pin equals the worker carry) and every
   native file under test_async_sched's relaxed contract, the window carry's digest (over renamed blocks and
   incarnations) set aside. Mutations: a missing void, a sliding group left out of it, a missed stop and a wrong
   stop fail.
4. Speed defaults: test_speed_defaults (a bare --async-scheduling on the Gemma profile; --no-speed-defaults needs
   --decode-fastpath --decode-fastpath2).
"""
import asyncio
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_async_sched as TA  # noqa: E402
import test_gemma_e2e as GE  # noqa: E402
import test_gemma_fastpath as TF  # noqa: E402
import test_gemma_fastpath3 as F3T  # noqa: E402
import test_gemma_release as GR  # noqa: E402
import test_host_trim as HT  # noqa: E402
import test_stageb_e2e as S  # noqa: E402
import test_stageb_worker as W  # noqa: E402
from test_gemma_install import gemma_fixture  # noqa: E402

from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import admission_flags as AF  # noqa: E402
from remote_ra_live_v2 import async_sched as A  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from remote_ra_live_v2.sliding_window import kept_blocks  # noqa: E402
# Imported here, outside any sys.modules patch, so the patches the tests enter keep them
# (test_decode_fastpath explains why).
from remote_ra_live_v2 import (admission_flags, admission_overlap, capacity, connector_guard,  # noqa: F401,E402
                               context_limits, decode_fastpath, decode_fastpath2, decode_fastpath3,
                               generation_guard, goal_async, goal_overlap, hooks, idle_release,
                               language_model_only, loaded_sources, native_probe, page_compaction, positions,
                               query_selection, vector_space, window_capacity, window_lifecycle, window_owner,
                               window_phase1, window_rows, window_worker, worker_composition, worker_rows)
from remote_ra_growing_worker import tensors  # noqa: F401,E402

try:
    import torch
except ImportError:
    torch = None

P, WINDOW, N, FA, SLIDING = GE.P, GE.WINDOW, GE.N, GE.FA, GE.SLIDING
GROUPS = (FA,) + tuple(SLIDING)
FAST = dict(decode_fastpath=True, decode_fastpath2=True)
DUMMY = 0           # the token a voided row samples (never used)
GEMMA = MP.GEMMA4


def live(name):
    """The module sys.modules holds now (an earlier test's patch.dict may have dropped the package
    attribute's copy): the one the engine and worker import (test_gemma_overlap.live)."""
    return importlib.import_module('remote_ra_live_v2.' + name)


# ---- 1. the pin -------------------------------------------------------------------------------------

class PinTests(unittest.TestCase):
    def test_the_gemma_config_takes_the_switch_with_qwens_refusals(self):
        base = dict(FAST, model_profile=GEMMA)
        self.assertIs(A.check_config(dict(base, async_scheduling=True)), True)
        self.assertIsNone(A.check_config(base))
        for bad, why in ((dict(base, decode_fastpath2=False, async_scheduling=True), 'decode-fastpath2'),
                         (dict(base, async_scheduling='yes'), 'true or absent'),
                         (dict(base, async_scheduling=True, admission=dict(gdn_device_handoff=True)), 'refuses'),
                         (dict(base, async_scheduling=True, admission=dict(phase1_in_step=True)), 'refuses')):
            with self.subTest(why=why), self.assertRaisesRegex(LiveV2Error, why):
                A.check_config(bad)


class LauncherTests(unittest.TestCase):
    """serve.prepare on a Gemma 4 release: the switch pins what it pins for Qwen and nothing else."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, cls.release = gemma_fixture(cls.root)
            install.install(site, gemma, cls.release, package_root=package, apply=True, model_profile=GEMMA)
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))

    def prepare(self, protocol='mswea-xml', policy='full', selection=None, raw=False, **kw):
        kw = dict(dict(kv_cache_gib=25, page_tokens=16, max_model_len=65536, language_model_only=True,
                       max_running=4, **FAST), **kw)
        task = [dict(instance_id='repo__g-%d' % j, problem_statement='Fix g%d.' % j) for j in range(4)]
        if protocol == 'chat-text':
            task = None
        if protocol == 'kv-session':
            task, kw = None, dict(kw, kv_session_thinking='off')
        if selection is None and policy == 'goal_query':
            selection = dict(active_token_budget=8192, goal_rewriter_model=str(self.rewriter))
        type(self).n += 1
        output = self.root / ('out%d' % self.n)
        with redirect_stdout(io.StringIO()):
            manifest, config, command, env = serve.prepare(self.release, task, policy, output, 8231, selection,
                                                           protocol, **kw)
        if raw:
            return config, command
        named = json.loads(json.dumps([config, command]).replace(json.dumps(str(output))[1:-1], 'OUT'))
        return named[0], named[1]

    def load(self, config):
        type(self).n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                   REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
        with patch.dict(os.environ, env):
            return HC.load(), A.loaded(), A.allowed(), A.batches()

    CASES = (('mswea-xml', 'full', {}), ('mswea-xml', 'observation_query', dict(selection=dict(active_token_budget=8192))),
             ('chat-text', 'full', {}), ('kv-session', 'observation_query', dict(selection=dict(active_token_budget=8192))),
             ('mswea-xml', 'goal_query', dict(goal_overlap=True, admission_overlap=4)),
             ('chat-text', 'goal_query', dict(goal_overlap=True)),
             ('mswea-xml', 'full', dict(cuda_graphs='full', decode_fastpath3=True, deferred_completion_io=True)),
             ('mswea-xml', 'full', dict(max_num_batched_tokens=4096)),
             ('kv-session', 'full', dict(release_idle_pages=True, cuda_graphs='full')))

    def test_the_switch_pins_one_key_and_swaps_the_scheduling_argument(self):
        for protocol, policy, extra in self.CASES:
            with self.subTest(protocol=protocol, policy=policy, extra=sorted(extra)):
                sync_config, sync_argv = self.prepare(protocol, policy, **extra)
                on_config, on_argv = self.prepare(protocol, policy, async_scheduling=True, **extra)
                self.assertEqual(sync_config['model_profile'], GEMMA)
                self.assertNotIn('async_scheduling', sync_config)
                self.assertNotIn('async_scheduling', sync_config['effective_server_config'])
                self.assertEqual(sync_argv.count('--no-async-scheduling'), 1)
                self.assertNotIn('--async-scheduling', sync_argv)
                self.assertEqual([a if a != '--async-scheduling' else '--no-async-scheduling' for a in on_argv],
                                 sync_argv)
                self.assertIs(on_config.pop('async_scheduling'), True)
                self.assertIs(on_config['effective_server_config'].pop('async_scheduling'), True)
                on_config.pop('server_config_sha256')
                sync_config.pop('server_config_sha256')
                self.assertEqual(on_config, sync_config)
                # async_scheduling=False is the former launch, byte for byte.
                self.assertEqual(self.prepare(protocol, policy, async_scheduling=False, **extra)[1], sync_argv)

    def test_the_loader_the_frontend_check_and_the_gates(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                config, _ = self.prepare('chat-text', async_scheduling=asynchronous, raw=True)
                loaded, pin, allowed, batches = self.load(config)
                self.assertEqual(loaded.get('async_scheduling'), True if asynchronous else None)
                self.assertEqual((pin, allowed, batches), (True, True, 2) if asynchronous else (None, False, 1))
                path = self.root / ('startup%d.json' % self.n)
                env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                           REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
                with patch.dict(os.environ, env):
                    effective = dict(config['effective_server_config'])
                    self.assertEqual(HC.effective_server_config(NS(**effective)), effective)
                    flipped = dict(effective, async_scheduling=not asynchronous)
                    with self.assertRaises(RuntimeError):
                        HC.effective_server_config(NS(**flipped))
                    self.assertIs(A.worker_flag({A.WORKER_KEY: True} if asynchronous else {}),
                                  True if asynchronous else None)
                    with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned'):
                        A.worker_flag({} if asynchronous else {A.WORKER_KEY: True})
                # The startup gates read the same pin (test_async_sched's gate test on this config).
                import bootstrap_contract
                from policy_runtime import bootstrap as policy_bootstrap
                with patch.dict(os.environ, env):
                    vllm = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                                                 decode_context_parallel_size=1, prefill_context_parallel_size=1,
                                                 distributed_executor_backend='uni'),
                              scheduler_config=NS(async_scheduling=asynchronous),
                              model_config=NS(runner_type='generate', is_encoder_decoder=False,
                                              enable_prompt_embeds=False),
                              speculative_config=None, max_concurrent_batches=2 if asynchronous else 1,
                              use_v2_model_runner=False)
                    bootstrap_contract.validate_topology(vllm)
                    policy_bootstrap.validate_supported_config(vllm)
                    other = NS(**dict(vars(vllm), scheduler_config=NS(async_scheduling=not asynchronous),
                                      max_concurrent_batches=1 if asynchronous else 2))
                    with self.assertRaises(Exception):
                        bootstrap_contract.validate_topology(other)

    def test_the_refusals_stay(self):
        for extra, pattern in ((dict(decode_fastpath2=False), 'requires --decode-fastpath2'),
                               (dict(decode_fastpath=False, decode_fastpath2=False), 'requires --decode-fastpath2'),
                               (dict(gdn_device_handoff=True), 'refuses --gdn-device-handoff'),
                               (dict(phase1_in_step=True), 'refuses --gdn-device-handoff and --phase1-in-step'),
                               # --enable-thinking (keep) is taken since gemma-thinking (test_gemma_thinking_on);
                               # --thinking-history strip serves chat-text only.
                               (dict(enable_thinking=True, thinking_history='strip'),
                                'strip serves --protocol chat-text only'),
                               (dict(async_scheduling=1), 'is a switch')):
            with self.subTest(extra=sorted(extra)), self.assertRaisesRegex(ValueError, pattern):
                self.prepare(**dict(dict(async_scheduling=True), **extra))
        # The loader refuses a Gemma config that pins the switch without the second fast path.
        config, _ = self.prepare(async_scheduling=True, raw=True)
        bad = {k: v for k, v in config.items() if k not in ('decode_fastpath2', 'decode_fastpath3')}
        with self.assertRaisesRegex(RuntimeError, 'needs --decode-fastpath2'):
            self.load(bad)


class WindowWorkerAttachTests(unittest.TestCase):
    """attach_window_worker with the loaded runner's seams faked (test_gemma_release.WorkerAttachTests):
    the async record checked once, async_sched.attach_worker called after the fast paths, the echo."""

    def attach(self, startup):
        WW = importlib.import_module('remote_ra_live_v2.window_worker')
        layout = GE.I0.for_window_server(16, 2, 1024, 8192)
        io_ = NS(views=[], evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', drain=lambda: None,
                 attach_arena=lambda arena: setattr(io_, 'arena', arena))
        fake_tensors = ModuleType('remote_ra_growing_worker.tensors')
        fake_tensors.TensorIO = lambda views, decoupled: io_
        fake_tensors.views_from_runner = lambda runner, *inputs, skip_sliding: []
        fake_tensors.PageArena = lambda views, max_pages: NS(stats=lambda: dict(max_pages=max_pages))
        modules = W.vllm_stubs()
        modules['vllm.distributed.kv_transfer'].get_kv_transfer_group = lambda: NS()
        modules['remote_ra_growing_worker.tensors'] = fake_tensors
        calls = []

        class Owner:
            release_idle = False

            def __init__(self, runner, io, startup, layout):
                self.startup = startup

            def bind_backend(self, backend):
                self.backend = backend

            def bind_bridge(self, bridge):
                self.bridge = bridge

        class Backend:
            def __init__(self, io, owner, source):
                self.io, self.owner = io, owner

        runner = NS(kv_cache_config=NS(group_pool=(0, 1)), max_model_len=131072, execute_model=lambda so: None)
        worker = NS(model_runner=runner)
        startup = dict(dict(source='s' * 64, page_tokens=16, loaded_sources_verified=True), **startup)

        def record(name):
            return lambda *args: calls.append((name, args))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(WW, 'check_view_geometry', lambda *a: None))
            for target, value in (('remote_ra_live_v2.cuda_graphs.attach', lambda *a: None),
                                  ('remote_ra_live_v2.context_limits.check_worker', lambda *a: None),
                                  ('remote_ra_live_v2.language_model_only.worker_check', lambda *a: None),
                                  ('remote_ra_live_v2.loaded_sources.verify_closure', lambda runner: {}),
                                  ('remote_ra_live_v2.connector_guard.check_worker_pool', lambda c: None),
                                  ('remote_ra_live_v2.vector_space.attach', lambda runner: dict(hidden_size=3)),
                                  ('remote_ra_live_v2.window_owner.WindowAllocationOwner', Owner),
                                  ('remote_ra_live_v2.window_lifecycle.WindowWorkerBackend', Backend),
                                  ('remote_ra_live_v2.window_lifecycle.WindowLifecycle', lambda b, page_tokens: NS()),
                                  ('remote_ra_live_v2.page_compaction.PageCompactor',
                                   lambda hidden, page: NS(evidence=lambda: {})),
                                  ('remote_ra_live_v2.native_probe.NativeQueryForward', lambda *a: NS()),
                                  ('remote_ra_live_v2.worker_composition.WorkerBridge', lambda q, o: NS()),
                                  ('remote_ra_live_v2.generation_guard.guarded_execute',
                                   lambda r, f: calls.append(('guard', ())) or f),
                                  ('remote_ra_live_v2.decode_fastpath.attach_worker', record('fast')),
                                  ('remote_ra_live_v2.decode_fastpath2.attach_worker', record('fast2')),
                                  ('remote_ra_live_v2.decode_fastpath3.attach_worker', record('fast3')),
                                  ('remote_ra_live_v2.window_owner.attach_fastpath', record('window_fast')),
                                  ('remote_ra_live_v2.window_owner.attach_fastpath2', record('window_fast2')),
                                  ('remote_ra_live_v2.window_owner.attach_fastpath3', record('window_fast3')),
                                  ('remote_ra_live_v2.async_sched.attach_worker', record('async'))):
                stack.enter_context(patch(target, value))
            evidence = WW.attach_window_worker(worker, startup, layout, ((), ()))
        return evidence, [name for name, _ in calls], calls

    def test_the_record_attach_and_echo(self):
        fast2 = {'decode_fastpath': True, 'decode_fastpath2': True}
        evidence, order, calls = self.attach(dict(fast2, **{A.WORKER_KEY: True}))
        self.assertIs(evidence[A.WORKER_KEY], True)
        # After fast path 2 (it defers its finish_step), before the generation guard, as for Qwen.
        self.assertEqual(order.count('async'), 1)
        self.assertLess(order.index('fast2'), order.index('async'))
        self.assertLess(order.index('async'), order.index('guard'))
        owner, bridge, qwen, runner = dict(calls)['async']
        self.assertEqual(type(owner).__name__, 'Owner')
        evidence, order, _ = self.attach(dict(fast2, decode_fastpath3=True, **{A.WORKER_KEY: True}))
        self.assertLess(order.index('fast3'), order.index('async'))
        evidence, order, _ = self.attach(fast2)
        self.assertNotIn(A.WORKER_KEY, evidence)
        self.assertNotIn('async', order)
        for startup, pattern in (({A.WORKER_KEY: True}, 'needs decode_fastpath2'),
                                 (dict(decode_fastpath=True, **{A.WORKER_KEY: True}), 'needs decode_fastpath2'),
                                 (dict(fast2, **{A.WORKER_KEY: 'yes'}), 'true or absent')):
            with self.subTest(startup=sorted(startup)), self.assertRaisesRegex(LiveV2Error, pattern):
                self.attach(startup)
        # The HTTP runtime compares the record with the pinned startup config once.
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                patch('remote_ra_http.config.load', lambda: dict(fast2, model_profile=GEMMA)), \
                self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
            self.attach(dict(fast2, **{A.WORKER_KEY: True}))

    @unittest.skipUnless(torch is not None, 'CPU torch required')
    def test_attach_worker_on_the_window_classes(self):
        """async_sched.attach_worker on test_gemma_e2e's real window owner, lifecycle and backend: the
        sliding groups are the void's; a Qwen-shaped lifecycle next to the window owner is refused."""
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, W.vllm_stubs()))
            stack.enter_context(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
            h = window_harness(stack, GR.Pressure, 'full', 1, turns=1)
            TF.attach_fast2(h)
            h.runner.use_async_scheduling = True
            saved = h.qwen.state_kind
            h.qwen.state_kind = 'mamba'
            with self.assertRaisesRegex(LiveV2Error, 'window owner, lifecycle and backend together'):
                A.attach_worker(h.owner, h.bridge, h.qwen, h.runner)
            h.qwen.state_kind = saved
            A.attach_worker(h.owner, h.bridge, h.qwen, h.runner)
            self.assertEqual(h.owner.async_sliding, tuple(SLIDING))
            self.assertEqual(h.owner.attention_groups, (FA,))
            self.assertTrue(h.owner.async_scheduling and h.bridge.async_scheduling and h.qwen.async_scheduling)


# ---- 2. the void on the window layout -----------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required')
class WindowVoidTests(unittest.TestCase):
    def metadata(self, persistent):
        """Rows p (a prefill of 6), a, b, c (decodes): 9 tokens. Every KV group has its slot mapping (a view of a
        padded persistent buffer under FULL graphs); two layers per group share one metadata object, the second
        layer's forward-context mapping is its own tensor."""
        tokens = 9
        buffers = {g: torch.arange(100 * (g + 1), 100 * (g + 1) + 32) for g in GROUPS}
        groups = {g: (buffers[g][:tokens] if persistent else buffers[g][:tokens].clone()) for g in GROUPS}
        attn, layers = {}, {}
        for g in GROUPS:
            md = NS(slot_mapping=groups[g])
            attn['l%d.0' % g] = attn['l%d.1' % g] = md
            layers['l%d.0' % g], layers['l%d.1' % g] = groups[g], groups[g].clone()
        runner = NS(input_batch=NS(req_ids=['p', 'a', 'b', 'c']), _remote_ra_step_metadata=(attn, layers, groups))
        so = NS(num_scheduled_tokens=dict(p=6, a=1, b=1, c=1))
        return runner, so, attn, layers, groups, buffers

    def test_every_group_is_voided_and_nothing_else(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                runner, so, attn, layers, groups, buffers = self.metadata(persistent)
                before = {g: groups[g].clone() for g in GROUPS}
                states, slots = A.void_rows(runner, so, frozenset({'a', 'c'}), (FA,), tuple(SLIDING))
                self.assertEqual(states, 0)                      # no Gated-DeltaNet state on this layout
                self.assertEqual(slots, 2 * len(GROUPS))         # each group's mapping and each clone, once
                for g in GROUPS:
                    expect = before[g].tolist()
                    expect[6] = expect[8] = A.NULL_SLOT
                    self.assertEqual(groups[g].tolist(), expect)
                    self.assertEqual(layers['l%d.1' % g].tolist(), expect)
                    if persistent:
                        self.assertEqual(buffers[g][:9].tolist(), expect)
                        self.assertEqual(buffers[g][9:].tolist(), list(range(100 * (g + 1) + 9, 100 * (g + 1) + 32)))

    def test_the_qwen_void_refuses_this_layout(self):
        """async-c's void (no sliding groups) on window metadata fails closed: it requires a Gated-DeltaNet
        state index, which the window layout has none of (the void every Gemma stop needs was refused)."""
        runner, so, attn, layers, groups, _ = self.metadata(False)
        with self.assertRaisesRegex(LiveV2Error, 'no GDN state index'):
            A.void_rows(runner, so, frozenset({'a'}), (FA,))

    def test_refusals(self):
        runner, so, attn, layers, groups, _ = self.metadata(False)
        attn['gdn.0'] = NS(non_spec_state_indices_tensor=torch.arange(1, 5))
        with self.assertRaisesRegex(LiveV2Error, 'Gated-DeltaNet state index on the window layout'):
            A.void_rows(runner, so, frozenset({'a'}), (FA,), tuple(SLIDING))
        runner, so, attn, layers, groups, _ = self.metadata(False)
        del groups[SLIDING[-1]]
        with self.assertRaisesRegex(LiveV2Error, 'slot mapping is missing'):
            A.void_rows(runner, so, frozenset({'a'}), (FA,), tuple(SLIDING))
        runner, so, *_ = self.metadata(False)
        with self.assertRaisesRegex(LiveV2Error, 'more than one token'):
            A.void_rows(runner, so, frozenset({'p'}), (FA,), tuple(SLIDING))


# ---- 3. end to end in vLLM's asynchronous order ---------------------------------------------------------

class AsyncRunner(GE.Runner):
    """test_gemma_e2e's window runner under vLLM's asynchronous order: the step's slot mappings (one per KV group,
    views of persistent buffers; two layers per group, the second layer's forward-context mapping its own copy) are
    stashed for the void and honoured by the forward (a row whose tokens are slot -1 in every group writes no KV
    and reads nothing; a partly voided row fails), and the sampled ids stay on the device
    (prev_sampled_token_ids) until the next step. ``extra_rows``: rows computed past their request's last token
    (a stop the device ran past)."""

    def __init__(self, store):
        super().__init__(store)
        self.use_async_scheduling = True
        self.input_batch.prev_sampled_token_ids = None
        self.input_batch.prev_req_id_to_index = None
        self.sampled, self.voided_rows, self.extra_rows = [], [], []
        self.buffers = {g: torch.full((8192,), -9, dtype=torch.int64) for g in GROUPS}

    def _remove(self, rid):
        """InputBatch.remove_request (gpu_input_batch.py:556-557): the request also leaves the previous step's
        sampled-token index (test_stageb_worker.FakeRunner._remove)."""
        super()._remove(rid)
        if self.input_batch.prev_req_id_to_index is not None:
            self.input_batch.prev_req_id_to_index.pop(rid, None)

    def execute_model(self, so):
        from remote_ra_live_v2.native_probe import active_probe
        if active_probe(self) is not None:
            return super().execute_model(so)
        assert self.execute_model_state is None
        TA.require_token_ids(self, so)
        self._update_states(so)
        batch = list(self.input_batch.req_ids)
        assert batch and len(batch) == len(so.num_scheduled_tokens) and set(batch) == set(so.num_scheduled_tokens)
        from remote_ra_live_v2.worker_rows import install_before_prepare
        install_before_prepare(self, so)
        spans = [(rid, self.requests[rid], self.requests[rid].num_computed_tokens, so.num_scheduled_tokens[rid])
                 for rid in batch]
        live = [b for rid in batch for g in SLIDING
                for b in self.requests[rid].block_ids[g][max(0, self.requests[rid].num_computed_tokens -
                                                             WINDOW + 1) // P:] if b]
        assert len(live) == len(set(live)), 'a pool-1 block is in two sliding windows of one step'
        self.steps.append(tuple((rid, start, count) for rid, _, start, count in spans))
        slots = {g: [] for g in GROUPS}
        for rid, req, start, count in spans:
            plan = self._selective_load_plans.get(rid)
            dropped = plan.dropped_full_attention_tokens if plan else 0
            for x in range(start, start + count):
                phys = x - dropped
                slots[FA].append(req.block_ids[FA][phys // P] * P + phys % P)
                for g in SLIDING:
                    slots[g].append(req.block_ids[g][x // P] * P + x % P)
        total = sum(count for *_, count in spans)
        groups, attn, layers = {}, {}, {}
        for g in GROUPS:
            self.buffers[g][:total] = torch.tensor(slots[g])
            groups[g] = self.buffers[g][:total]
            md = NS(slot_mapping=groups[g])
            attn['l%d.0' % g] = attn['l%d.1' % g] = md
            layers['l%d.0' % g], layers['l%d.1' % g] = groups[g], groups[g].clone()
        self._remote_ra_step_metadata = (attn, layers, groups)
        from remote_ra_live_v2 import hooks
        flat = torch.cat([torch.arange(start, start + count) for _, _, start, count in spans])
        hooks.before_forward(self, so, flat)
        hidden, offset, void = [], 0, []
        for rid, req, start, count in spans:
            marks = [int(m[offset + i]) == A.NULL_SLOT for g in GROUPS
                     for m in (groups[g], layers['l%d.1' % g]) for i in range(count)]
            assert all(marks) or not any(marks), 'partial void of %s' % rid
            void.append(all(marks))
            if all(marks):
                self.voided_rows.append((rid, start))
                hidden.append(torch.zeros(count, GE.HIDDEN))
            else:
                if start >= len(req.prompt_token_ids) + len(req.output_token_ids):
                    self.extra_rows.append((rid, start))
                hidden.append(self.forward(rid, req, start, count))
            offset += count
        result = hooks.after_forward(self, so, torch.cat(hidden))
        ids, index, sampled = [], {}, {}
        for k, (rid, req, start, count) in enumerate(spans):
            req.num_computed_tokens = start + count
            token = DUMMY
            if start + count >= len(req.prompt_token_ids):
                index[rid] = k
                if not void[k] and start + count == len(req.prompt_token_ids) + len(req.output_token_ids):
                    token = next(self.samples[rid], None)
                    if token is None:
                        token = DUMMY
                    else:
                        req.output_token_ids.append(token)
                        sampled[rid] = token
            ids.append([token])
        self.input_batch.prev_sampled_token_ids = torch.tensor(ids)
        self.input_batch.prev_req_id_to_index = index
        self.sampled.append(sampled)
        return result

    def empty_step(self, so):
        """vLLM's execute_model with nothing scheduled (TA.length_skip): the persistent batch is updated, then it
        returns before any forward, hook or sampling (test_async_sched's AsyncRunner.empty_step)."""
        assert self.execute_model_state is None and not so.num_scheduled_tokens
        TA.require_token_ids(self, so)
        self._update_states(so)
        self.sampled.append({})


def async_harness(Base):
    """``Base`` (test_gemma_release.Pressure or a subclass: the XML lanes, test_gemma_goal_overlap's
    GoalOverlapPressure, test_gemma_kv_session_e2e's Sessions) in vLLM's asynchronous order
    (EngineCore.step_with_batch_queue at two batches, test_async_sched.async_harness's form): schedule N+1
    (computed counts advance, a sampling step holds a placeholder), execute it (the worker resolves N in
    before_forward), then update N (tokens appended, stops finished, the link's predictions checked). Every
    admission and Goal resume is a pass boundary: the batch in flight is drained first. A request stops on its
    last scripted output, which the host learns only from the update (TA.scripted_limits): alternately an EOS,
    scheduled once more and voided, and its max_tokens-th token, which vLLM's schedule does not
    schedule past (TA.length_skip; GPU job 3568311). ``speculative``: per stopped request scheduled once more
    (rid, C', its rows before and after its speculative allocation). ``length_skips``: per schedule that
    skipped a request, (the skipped requests, the requests it scheduled)."""

    class AsyncHarness(Base):
        def __init__(self, *args, **kwargs):
            self.finished_rows = set()
            super().__init__(*args, **kwargs)
            self.inflight, self.async_drains, self.resume_drains, self.speculative = None, 0, 0, []
            self.on_done, self.reused, self.length_skips = None, [], []

        def free(self, request):
            """test_gemma_e2e.Harness.free (KVCacheManager.free with the growing hooks) without its runner
            cleanup: the finished row leaves the runner with the next step's finished_req_ids, as vLLM's
            scheduler sends it. The synchronous fakes' shortcut, an empty out-of-band update, would also take
            every running request out of the runner's batch, which no production path does but a phase-1 pass
            (async-fix2: then the next step must send their token ids, test_async_sched.require_token_ids)."""
            rid = request.request_id
            self.frees[rid] += 1
            self.enqueued.discard(rid)
            self.adapter.before_free(request)
            self.kv._growing_original_free(request)
            self.adapter.after_free(request)
            self.finished_rows.add(rid)

        def scripted(self):
            if all(hasattr(lane, 'agent') for lane in self.lanes):
                return any(lane.agent.answer is None for lane in self.lanes)
            return any(lane.script for lane in self.lanes)

        def ready_lanes(self):
            busy = {info['lane'].index for info in self.running} | set(getattr(self, 'busy', ()))
            if all(hasattr(lane, 'agent') for lane in self.lanes):
                return [lane for lane in self.lanes if lane.agent.answer is None and lane.index not in busy]
            return [lane for lane in self.lanes if lane.script and lane.index not in busy]

        def admit(self, lane):
            info = super().admit(lane)
            return TA.scripted_limits(info) if type(info) is dict else info

        def accepted(self, info, ack, before):
            return TA.scripted_limits(super().accepted(info, ack, before))

        def rows(self, rid):
            return tuple(tuple(b.block_id for b in m.req_to_blocks[rid]) for m in self.kv.coordinator.single_type_managers)

        def schedule(self):
            """``reused``: per schedule, the blocks a stopped request's speculative allocation freed that a
            request allocated after it in the same schedule took ((pool, block); vLLM puts a freed unhashed
            block at the head of the free queue)."""
            from vllm.v1.core.sched.output import NewRequestData
            counts, news, freed, gained, skipped = {}, [], set(), set(), []
            pool = lambda g: 0 if g == FA else 1
            for info in self.running:
                request, rid = info['request'], info['rid']
                if TA.length_skip(request):
                    skipped.append(rid)
                    continue
                n, start = len(request.prompt_token_ids), request.num_computed_tokens
                waiting = request.status.name == 'WAITING'
                count = min(n, (start // P + 1) * P) - start if start < n else 1
                stopped = start >= n and len(request.output_token_ids) + request.num_output_placeholders >= \
                    len(info['outputs'])
                before = self.rows(rid) if rid in self.kv.coordinator.single_type_managers[0].req_to_blocks \
                    else tuple(() for _ in GROUPS)
                result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
                    new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                    delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=waiting,
                    reserved_blocks=0, has_scheduled_reqs=waiting and bool(counts))
                assert result is not None, 'running request %s found no free block (preemption)' % rid
                rows = self.rows(rid)
                old = {(pool(g), b) for g, row in enumerate(before) for b in row if b}
                new = {(pool(g), b) for g, row in enumerate(rows) for b in row if b}
                if stopped:
                    self.speculative.append((rid, start, before, rows))
                    freed |= old - new
                else:
                    gained |= (new - old) & freed
                if info['first']:
                    news.append(NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None,
                                               tuple(list(r) for r in rows), start, None))
                    info['first'] = False
                else:
                    self.runner.set_rows(rid, tuple(list(r) for r in rows))
                counts[rid] = count
            if gained:
                self.reused.append(sorted(gained))
            if skipped:
                self.length_skips.append((tuple(skipped), tuple(counts)))
            for info in self.running:          # vLLM's _update_after_schedule
                request = info['request']
                if info['rid'] not in counts:
                    continue
                request.status.name = 'RUNNING'
                if request.num_computed_tokens + counts[info['rid']] >= len(request.prompt_token_ids):
                    request.num_output_placeholders += 1
                request.num_computed_tokens += counts[info['rid']]
            return counts, news

        def batch_step(self):
            counts, news = self.schedule()
            finished, self.finished_rows = self.finished_rows, set()
            # Nothing scheduled (TA.length_skip): vLLM's execute_model returns before the forward (empty_step).
            (self.runner.execute_model if counts else self.runner.empty_step)(NS(num_scheduled_tokens=dict(counts),
                total_num_scheduled_tokens=sum(counts.values()), scheduled_new_reqs=news, finished_req_ids=finished,
                all_token_ids=TA.cached_token_ids(A.link(), counts, news),
                new_block_ids_to_zero=self.kv.take_new_block_ids(),
                kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
            self.widths.append(len(counts))
            previous, self.inflight = self.inflight, (counts, self.runner.sampled[-1])
            if previous is not None:
                self.update(previous)

        def update(self, step):
            counts, sampled = step
            for info in list(self.running):
                rid, request = info['rid'], info['request']
                if rid not in counts:
                    continue
                token = sampled.get(rid)
                if token is not None:
                    request.output_token_ids.append(token)
                    request.num_output_placeholders -= 1
                if len(request.output_token_ids) == len(info['outputs']):
                    self.running.remove(info)
                    self.complete(info)
                    lane = info['lane']
                    if hasattr(lane, 'agent'):
                        lane.messages = []          # a kv lane's agent keeps its own (Sessions.batch_step)
                    else:
                        lane.messages = info['messages'] + [dict(role='assistant',
                                                                 content=' ' + info['committed']['text'] + '\n')]
                    self.done.append(info)
                    if self.on_done is not None:
                        self.on_done(self, info)    # at once, as the synchronous loops check a finished turn
            running = {info['rid'] for info in self.running}
            A.link().check([rid for rid in counts if rid not in running])

        def drain(self):
            if self.inflight is not None:
                step, self.inflight = self.inflight, None
                self.async_drains += 1
                self.update(step)

        def serve(self, check):
            table = getattr(self, 'table', None)
            self.on_done = check
            k = passes = 0
            while (self.scripted() or self.running or self.inflight is not None or
                   (table is not None and table.parked)):
                passes += 1
                assert passes < 400000, 'the asynchronous loop does not finish'
                started = None
                if table is not None and table.ready():
                    self.resume_drains += self.inflight is not None
                    self.drain()                      # a pass boundary (core._remote_ra_resume)
                    self.resume()
                    started = 'resumed'
                else:
                    ready = self.ready_lanes()
                    if ready and (table is None or len(self.busy) < self.permits):
                        self.drain()                  # a pass boundary (a client request)
                        ready = self.ready_lanes()
                        if ready:
                            started = self.admit(ready[k % len(ready)])
                            k += 1
                if table is not None:
                    self.parked_peak = max(self.parked_peak, len(table.parked))
                if self.running:
                    self.batch_step()
                elif self.inflight is not None:
                    self.drain()
                elif table is not None:
                    if started is None:
                        time.sleep(0.002)             # only a Goal thread can move now
                else:
                    assert started is not None, 'no request runs and the gate refused every lane'
            self.drain()
            if table is not None:
                table.close()
    AsyncHarness.__name__ = 'Async' + Base.__name__
    return AsyncHarness


def attach_async(h, fast3=False):
    """attach_window_worker's and compose_engine's attachments on a window harness: fast paths 1 and 2 (and 3) as
    test_gemma_fastpath(3) attach them but with the updates held (no flush where schedule returns), then
    async_sched.attach_worker; the link reads the harness's requests (the scripted stop)."""
    F2, F3, WO = live('decode_fastpath2'), live('decode_fastpath3'), live('window_owner')
    TF.attach_fast(h)
    F2.attach_worker(h.owner, h.bridge, h.qwen, h.io)
    WO.attach_fastpath2(h.owner)
    h.boundary.batched, h.boundary.async_hold = {}, True
    h.proxy.before_call = h.boundary.flush_updates
    if fast3:
        F3.attach_worker(h.owner, h.bridge, h.qwen)
        F3.attach_engine(h.boundary)
        WO.attach_fastpath3(h.owner, h.qwen, h.backend)
    A.attach_worker(h.owner, h.bridge, h.qwen, h.runner)
    scheduler = NS(requests=h.requests, max_model_len=1 << 20, _remote_ra_output_stop=None,
                   prev_step_scheduled_req_ids=set())
    A.detach_engine()
    A._LINK = A.EngineLink(scheduler, h.boundary, check_stop=TA.scripted_stop, core=TA.InFlight(h))
    return A._LINK


def attach_sync(h, fast3=False):
    (F3T.attach_fast3 if fast3 else TF.attach_fast2)(h)


# Targets of C' mod P (test_stageb_e2e.Harness.reply puts C' = N + g - 1 there): 0 (a fresh global and sliding
# block for C'), 15 ((C' - W + 1) % P == 0 at W = 64: the window's front block freed one step early), and others.
TARGETS = (0, 15, 7, 0, 15, 3, 11, 0, 15, 9)


def script(index, turns):
    """test_gemma_release.script with TARGETS (both Gemma-specific speculative steps, every lane)."""
    return [(observation, TARGETS[(k + 2 * index) % len(TARGETS)], eos)
            for k, (observation, _, eos) in enumerate(GR.script(index, turns))]


def window_harness(stack, Base, policy, sessions, *, turns, release=False, flags=None, asynchronous=False,
                   fast3=False, blocks=GR.REFERENCE_BLOCKS, extra=None):
    """A Gemma window harness of ``Base`` (its constructor's test_gemma_release arguments) in a fresh output
    directory, the startup config patched in; asynchronous: the async order and attachments."""
    output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    tasks = GR.tasks(sessions)
    layout = GE.window_layout(policy, sessions)
    value = dict(GR.config(policy, output, tasks, layout, release), **(flags or {}))
    stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
    stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
    cls = Base
    if asynchronous:
        stack.enter_context(patch.object(GE, 'Runner', AsyncRunner))
        cls = async_harness(Base)
    h = cls(policy, output, tasks, blocks, release, turns, **(extra or {}))
    h.output = output
    return h


def window_state(h, info):
    """A finished turn's window: the engine pin and the worker carry per sliding group (block indices, and the
    blocks with their incarnations), the turn's computed positions, its committed end."""
    lane, rid = info['lane'], info['rid']
    s = h.adapter.sessions[lane.sid]
    pin = h.adapter.window.window(s.namespace)
    carry = h.backend.carry[lane.sid]
    return dict(computed_end=s.computed_end, carry_end=carry.computed_end,
                pin={g: [i for i, _ in v] for g, v in pin.items()},
                carry={g: [i for i, _, _ in entries] for g, entries in carry.window},
                same=(carry.blocks() == {g: tuple(b.block_id for _, b in v) for g, v in pin.items()}),
                computed=[x for r, x in h.runner.computed if r == rid])


def turns(h):
    """(lane, turn) of every finished turn, by its lane's order."""
    for info in h.done:
        info['turn'] = sum(1 for other in h.done[:h.done.index(info)] if other['lane'] is info['lane'])
    return {(info['lane'].index, info['turn']): info for info in h.done}


# The window layout's digests over block ids and incarnations are over renamed values, as the other digests of
# test_async_sched's relaxed contract: the window carry's (window_lifecycle.window_digest; ``digest`` of drain.carry
# and of a frame's carry_restored), its echo as the session boundary (``boundary``) and phase 1's window state
# (``state_digest``, WindowWorkerBackend.capture_phase1_state). The windows' indices are compared exactly
# (window_state).
DIGESTS = TA.DIGEST_KEYS | {'digest', 'boundary', 'state_digest'}


def slots(value):
    """A native file with every released slot [group, block, generation, kind, incarnation] named by its
    (block, incarnation) pair: on the window layout a sliding block that left a request's window can come back
    to the same request under a new incarnation, depending on the free queues' order (H1), so a consistent
    renaming is one of (block, incarnation) slots and test_async_sched's shape of a release (the (group, kind)
    multiset, the distinct blocks per group) counts distinct slots per group."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if (k == 'release' and isinstance(v, list) and v and
                    all(isinstance(e, list) and len(e) == 5 for e in v)):
                names = {}
                out[k] = [[e[0], names.setdefault((e[0], e[1], e[4]), len(names) + 1), e[2], e[3], e[4]] for e in v]
            else:
                out[k] = slots(v)
        return out
    if isinstance(value, list):
        return [slots(v) for v in value]
    return value


def relaxed(sync, asyn):
    with patch.object(TA, 'DIGEST_KEYS', DIGESTS):
        return TA.relaxed_differences(slots(sync), slots(asyn))


def native_digests(native):
    """The carry digests of a native file (drain.carry, the frames' carry_restored)."""
    found = []

    def walk(x, keys):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == 'digest' and keys and keys[-1] in ('carry', 'carry_restored'):
                    found.append(v)
                walk(v, keys + (k,))
        elif isinstance(x, list):
            for v in x:
                walk(v, keys)
    walk(native, ())
    return found


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class AsyncWindowEndToEndTests(unittest.TestCase):
    """Synchronous fast path 2 (or 3) vs --async-scheduling over the Gemma window harnesses."""
    maxDiff = None
    window_of = GR.SixteenSessionTests.window_of

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def check(self, h, info):
        """test_gemma_release's per-turn check (exactly-once computation of [e, C'), phase 1's [C, e), the
        committed window kept_blocks(C') pinned by the engine and named by the worker carry, the attended Pages)
        and the turn's window state."""
        GR.SixteenSessionTests.check(self, h, info)
        info['window'] = window_state(h, info)

    def serve(self, policy, sessions, asynchronous, *, turns=3, fast3=False, release=False, mutate=None,
              blocks=GR.REFERENCE_BLOCKS):
        with ExitStack() as stack:
            HT.deterministic(stack)
            flags = dict(FAST, **(dict(decode_fastpath3=True) if fast3 else {}))
            if asynchronous:
                flags['async_scheduling'] = True
            h = window_harness(stack, GR.Pressure, policy, sessions, turns=turns, release=release, flags=flags,
                               asynchronous=asynchronous, blocks=blocks)
            for lane in h.lanes:
                lane.script = script(lane.index, turns)
            link = None
            if asynchronous:
                link = attach_async(h, fast3)
                stack.callback(A.detach_engine)
                if mutate is not None:
                    mutate(stack, h)
            else:
                attach_sync(h, fast3)
            h.serve(self.check)
            self.assertEqual(len(h.done), sessions * turns)
            GR.SixteenSessionTests.boundary(self, h)
            return self.finish(h, link)

    def finish(self, h, link):
        from remote_ra_http.evidence import engine_closed
        by_turn = turns(h)
        result = dict(
            turns={key: dict(tokens=list(i['tokens']), outputs=list(i['request'].output_token_ids),
                             text=i['committed']['text'], selected=list(i['plan']['selected']),
                             visible=h.runner.visible[i['rid']], probe=i['probe'], window=i['window'])
                   for key, i in by_turn.items()},
            natives={key: dict(native=i['native'], receipt=i['receipt']) for key, i in by_turn.items()},
            widths=list(h.widths), h=h)
        if link is not None:
            result.update(voided=list(h.runner.voided_rows), extra=list(h.runner.extra_rows), voids=link.voids,
                          resolves=link.resolves, drains=h.async_drains, speculative=list(h.speculative),
                          reused=list(h.reused), unbatched=link.unbatched_requests,
                          length_skips=list(h.length_skips))
        path = h.output / live('idle_release').EVENTS
        result['events'] = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        self.assertEqual(h.adapter.window.pinned_blocks(), 0)
        self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                         (h.baseline, h.window_baseline))
        result['files'] = sorted(p.name for p in h.output.iterdir())
        return result

    def same(self, sync, asyn):
        """Every turn equals the synchronous server's; native files under the relaxed contract (the window
        carry's digest over renamed values); no row ran past its stop; every void is counted."""
        self.assertEqual(sorted(asyn['turns']), sorted(sync['turns']))
        for key in sorted(sync['turns']):
            self.assertEqual(asyn['turns'][key], sync['turns'][key], key)
            window = asyn['turns'][key]['window']
            self.assertTrue(window['same'], key)                               # engine pin == worker carry
            first, last = kept_blocks(window['computed_end'], WINDOW, P)
            self.assertEqual(window['carry'], {g: list(range(first, last)) for g in SLIDING})
            self.assertTrue(all(x < window['computed_end'] for x in window['computed']), key)
        named = lambda files: ([n for n in files if not n.startswith('native-')],
                               sum(n.startswith('native-') for n in files))
        self.assertEqual(named(asyn['files']), named(sync['files']))
        for key in sorted(sync['natives']):
            self.assertEqual(relaxed(sync['natives'][key], asyn['natives'][key]), [], key)
        self.assertEqual(asyn['extra'], [])
        self.assertEqual(len(asyn['voided']), asyn['voids'])
        self.assertEqual(sorted(asyn['voided']), sorted((rid, c) for rid, c, _, _ in asyn['speculative']))

    def speculative_steps(self, asyn):
        """The Gemma-specific speculative steps of the stopped requests, read from their rows before and after
        their allocation of N+1: (fresh: C' % P == 0 took a new global and sliding block for C'; front:
        (C' - W + 1) % P == 0 freed the front block of every sliding window one step early)."""
        fresh = front = 0
        for rid, c, before, after in asyn['speculative']:
            if c % P == 0:
                self.assertTrue(all(len(after[g]) == len(before[g]) + 1 and after[g][-1] > 0 for g in GROUPS
                                    if g in SLIDING) and len(after[FA]) > len(before[FA]), rid)
                fresh += 1
            if c >= WINDOW and (c - WINDOW + 1) % P == 0:
                i = (c - WINDOW + 1) // P - 1
                self.assertTrue(all(before[g][i] > 0 and after[g][i] == 0 for g in SLIDING), rid)
                front += 1
            else:
                self.assertTrue(all(sum(1 for b in before[g] if b) <= sum(1 for b in after[g] if b)
                                    for g in SLIDING), rid)
        return fresh, front

    def test_one_session_equals_sync(self):
        """b = 1: every turn of one XML session equals the synchronous server's (fast paths 1-2 and 1-3), and
        both Gemma-specific speculative steps happened and were voided. Turns 0, 2, 4 stop on an EOS (voided),
        turns 1, 3, 5 on max_tokens: vLLM schedules nothing after those (an empty step, TA.length_skip)."""
        for fast3 in (False, True):
            with self.subTest(fast3=fast3):
                sync = self.serve('observation_query', 1, False, turns=6, fast3=fast3)
                asyn = self.serve('observation_query', 1, True, turns=6, fast3=fast3)
                self.same(sync, asyn)
                self.assertEqual(max(asyn['widths']), 1)
                self.assertEqual(asyn['voids'], 3)                  # every EOS stop voided in a later step
                self.assertEqual([scheduled for _, scheduled in asyn['length_skips']], [()] * 3)
                fresh, front = self.speculative_steps(asyn)
                self.assertGreater(fresh, 0)
                self.assertGreater(front, 0)

    def test_three_sessions_equal_sync(self):
        """b > 1 on the fakes (no batch variance): every turn equal, several requests per step, voided rows
        next to live ones; the structural checks of section 4 on every turn (same())."""
        for policy, fast3 in (('full', True), ('random_fill', False), ('goal_query', True)):
            with self.subTest(policy=policy, fast3=fast3):
                sync = self.serve(policy, 3, False, turns=3, fast3=fast3)
                asyn = self.serve(policy, 3, True, turns=3, fast3=fast3)
                self.same(sync, asyn)
                self.assertGreater(max(asyn['widths']), 1)
                self.assertGreater(asyn['drains'], 0)
                self.assertGreater(asyn['voids'], 0)        # a stop updated by a drain needs no void
                fresh, front = self.speculative_steps(asyn)
                self.assertGreater(fresh + front, 0)
                # Phase-1 passes took running requests out of the batch (async-fix2).
                self.assertGreater(asyn['unbatched'], 0)
                # max_tokens stops vLLM did not schedule past, next to requests it did schedule: the resolve
                # of that step met a request of the step before outside it (GPU job 3568311).
                self.assertTrue(any(scheduled for _, scheduled in asyn['length_skips']), asyn['length_skips'])

    def test_idle_releases_under_pressure(self):
        """--release-idle-pages at the selective rule's minimum pool 0: the gate's evictions and readmissions run
        at pass boundaries (after the drain), so every turn reads what the synchronous server's does."""
        small = GR.selective_blocks(GE.window_layout('observation_query', 4))
        sync = self.serve('observation_query', 4, False, turns=3, fast3=True, release=True, blocks=small)
        asyn = self.serve('observation_query', 4, True, turns=3, fast3=True, release=True, blocks=small)
        self.same(sync, asyn)
        kinds = Counter(e['event'] for e in asyn['events'])
        self.assertTrue(kinds['evict'] and kinds['readmit'], kinds)
        self.assertEqual(kinds, Counter(e['event'] for e in sync['events']))

    def reuse(self, asynchronous, mutate=None):
        """Two XML sessions: R's first turn stops on C' % 16 == 15 (its speculative step frees the front block of
        every sliding window), and T is admitted three tokens before R's stop, so that T's first prefill chunks
        (whose windows free nothing yet) take new sliding blocks in the very schedule that frees R's: vLLM puts a
        freed unhashed block at the head of the free queue, so T takes R's blocks while R's row of that step is
        voided and R still holds the worker's claims (released at its completion, later in that iteration)."""
        with ExitStack() as stack:
            HT.deterministic(stack)
            flags = dict(FAST, decode_fastpath3=True, **(dict(async_scheduling=True) if asynchronous else {}))
            h = window_harness(stack, GR.Pressure, 'full', 2, turns=1, flags=flags, asynchronous=asynchronous)
            h.lanes[0].script, h.lanes[1].script = [(None, 15, False)], [(None, 7, False)]
            if asynchronous:
                attach_async(h, True)
                stack.callback(A.detach_engine)
                if mutate is not None:
                    mutate(stack, h)
            else:
                attach_sync(h, True)
            first = h.admit(h.lanes[0])
            request = first['request']
            while len(first['outputs']) - len(request.output_token_ids) > 3:
                h.batch_step()
                if asynchronous:
                    h.drain()                  # the host's count exact: R is admitted here, three before its stop
            second = h.admit(h.lanes[1])
            while h.running:
                h.batch_step()
            if asynchronous:
                h.drain()
            for info in h.done:
                self.check(h, info)
            self.assertEqual(len(h.done), 2)
            result = self.finish(h, A.link() if asynchronous else None)
            result['first'] = (first['request'].output_token_ids[-1:], len(first['tokens']))
            return result

    def test_a_freed_window_block_taken_in_the_same_schedule(self):
        sync, asyn = self.reuse(False), self.reuse(True)
        self.same(sync, asyn)
        rid, c, before, after = asyn['speculative'][0]
        self.assertEqual((c % P, (c - WINDOW + 1) % P), (15, 0))         # R's front block freed one step early
        self.assertTrue(asyn['reused'])                                    # ... and taken by T in that schedule
        freed = {b for g in SLIDING for b, a in zip(before[g], after[g]) if b and not a}
        self.assertTrue({b for _, b in asyn['reused'][0]} <= freed)

        def strict(stack, h):
            stack.enter_context(patch.object(type(h.owner), '_voided_holder', lambda self, held: False))
        with self.assertRaisesRegex(LiveV2Error, 'allocation owned by another generation'):
            self.reuse(True, strict)

    # ---- mutations ----------------------------------------------------------------------------------
    def mutated(self, mutate):
        try:
            return self.serve('observation_query', 2, True, turns=3, mutate=mutate)
        except (AssertionError, LiveV2Error, RuntimeError, ExceptionGroup) as failure:
            return failure

    def test_a_missing_void_is_caught(self):
        """The resolve decides the stops but writes nothing on the device: the stopped request's extra row runs,
        writes KV at C' in the global tail and the window, and the turn's exactly-once check refuses it."""
        def no_void(stack, h):
            stack.enter_context(patch.object(A, 'void_rows', lambda *args, **kwargs: (0, 1)))
        outcome = self.mutated(no_void)
        self.assertIsInstance(outcome, BaseException)
        self.assertRegex(str(outcome), r'\[|computed|range|!=')

    def test_a_sliding_group_left_out_of_the_void_is_caught(self):
        """async-c's void (the global group only) on the window layout: the runner sees a partly voided row."""
        def global_only(runner, so, voided, attention_groups, sliding_groups=None):
            rows, tokens = A.void_positions(runner, so, voided)
            attn, layers, groups = runner._remote_ra_step_metadata
            for name, md in attn.items():
                if name.startswith('l%d.' % FA):
                    md.slot_mapping[tokens] = A.NULL_SLOT
                    layers[name][tokens] = A.NULL_SLOT
            groups[FA][tokens] = A.NULL_SLOT
            return 0, 1

        def mutate(stack, h):
            stack.enter_context(patch.object(A, 'void_rows', global_only))
        outcome = self.mutated(mutate)
        self.assertIsInstance(outcome, AssertionError)
        self.assertIn('partial void', str(outcome))

    def test_a_missed_stop_fails_closed(self):
        def blind(stack, h):
            stack.enter_context(patch.object(A._LINK, 'stops', lambda rid, token: False))
        self.assertIsInstance(self.mutated(blind), BaseException)

    def test_a_wrong_stop_fails_closed(self):
        def eager(stack, h):
            stack.enter_context(patch.object(A._LINK, 'stops', lambda rid, token: True))
        self.assertIsInstance(self.mutated(eager), BaseException)

    def test_without_the_applied_check_a_drained_step_is_decided_twice(self):
        """Mutation of EngineLink.applied (async-gemma's fix, shared with Qwen): under pressure a lane's
        registered add is refused at every pass (CAPACITY_WAIT, no worker RPC) after the drain, so the next
        resolve meets a step whose tokens the engine already applied; deciding its stops again counts a token
        twice, a continuing request with one token left is voided and the step is lost (fail closed)."""
        small = GR.selective_blocks(GE.window_layout('observation_query', 4))

        def twice(stack, h):
            stack.enter_context(patch.object(A.EngineLink, 'applied', lambda self, rid, cursor: False))
        with self.assertRaisesRegex(LiveV2Error, 'predicted stop did not stop|still holds a step lease|cursor'):
            self.serve('observation_query', 4, True, turns=3, fast3=True, release=True, blocks=small, mutate=twice)

    def test_without_the_unbatched_report_a_phase1_pass_loses_the_token_ids(self):
        """async-fix2 (GPU job 3519531), on the window layout: an admission's phase 1 (its separate pass) takes the
        running requests out of the runner's batch; without EngineLink.unbatched the next step sends no token ids
        for them and vLLM's async runner raises KeyError adding one back (test_async_sched.require_token_ids)."""
        def no_report(stack, h):
            stack.enter_context(patch.object(A.EngineLink, 'unbatched', lambda self, rids: None))
        with self.assertRaises((KeyError, ExceptionGroup)) as caught:
            self.serve('observation_query', 2, True, turns=3, mutate=no_report)
        error = caught.exception
        if isinstance(error, ExceptionGroup):
            self.assertTrue(error.subgroup(KeyError), error)


class AppliedTests(unittest.TestCase):
    """EngineLink.applied: whether a pass boundary already applied the issued step's token."""

    def test_the_count_against_the_cursor(self):
        requests = dict(a=TA.Request('a', [1] * 10, [5, 6], 99, -1), b=TA.Request('b', [1] * 10, [5, 6, 7], 99, -1))
        link = A.EngineLink(NS(requests=requests), None, check_stop=TA.check_stop)
        self.assertIs(link.applied('a', 12), False)        # 12 tokens, the lease ended at 12: not yet updated
        self.assertIs(link.applied('b', 12), True)         # 13 tokens: the drain appended the token at 12
        self.assertIs(link.applied('gone', 12), True)      # updated and finished
        for cursor in (10, 14):
            with self.assertRaisesRegex(LiveV2Error, 'disagree'):
                link.applied('a', cursor)


def untimed(value):
    """A native file without its clock fields (``*_ms`` / ``*_ns``) and its goal_text reference (the SHA-256 of
    its goals.jsonl row, which holds the request id and those times): test_async_goal_overlap.untimed."""
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items() if not k.endswith(('_ms', '_ns')) and k != 'goal_text'}
    if isinstance(value, list):
        return [untimed(v) for v in value]
    return value


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class AsyncGoalOverlapTests(unittest.TestCase):
    """--async-scheduling with --goal-overlap on the window layout (test_gemma_goal_overlap's harness: the
    rewriter stand-in, Goals parked on their sessions' threads, phase 1 before the Goal join, the trim of the
    old window at phase 1's drain, the Goal forward as its own lease): every turn, Goal vector and native file
    equal the synchronous server's with the same overlap; Goal resumes met a batch in flight."""
    maxDiff = None
    import test_gemma_overlap as GOV
    TEST_GOAL = GOV.OverlapEndToEndTests.TEST_GOAL
    setUp = GOV.OverlapEndToEndTests.setUp
    window_of = GR.SixteenSessionTests.window_of
    check = AsyncWindowEndToEndTests.check
    del GOV

    def serve(self, asynchronous, permits, *, sessions=4, turns=3, fast3=False, mutate=None):
        import test_gemma_goal_overlap as GGO
        with ExitStack() as stack:
            HT.deterministic(stack)
            flags = dict(FAST, admission=dict(AF.record(admission_overlap=permits) or {}, goal_overlap=True),
                         **(dict(decode_fastpath3=True) if fast3 else {}))
            if asynchronous:
                flags['async_scheduling'] = True
            h = window_harness(stack, GGO.GoalOverlapPressure, 'goal_query', sessions, turns=turns, flags=flags,
                               asynchronous=asynchronous, extra=dict(permits=permits))
            stack.callback(lambda: [e.shutdown(wait=True, cancel_futures=True)
                                    for e in list(getattr(h.table, '_threads', {}).values())])
            self.assertIsNotNone(h.selection.rewriter)
            for lane in h.lanes:
                lane.script = script(lane.index, turns)
            link = None
            if asynchronous:
                link = attach_async(h, fast3)
                stack.callback(A.detach_engine)
                if mutate is not None:
                    mutate(stack, h)
            else:
                attach_sync(h, fast3)
            h.serve(self.check)
            self.assertEqual(len(h.done), sessions * turns)
            GR.SixteenSessionTests.boundary(self, h)
            goals = GGO.goals_of(h)
            result = AsyncWindowEndToEndTests.finish(self, h, link)
            result['goals'] = {key: goals[key] for key in sorted(goals)}
            result['forwards'] = {(i['lane'].index, i['turn']): [f['operation'] for f in h.proxy.http_receipts[i['rid']]]
                                  for i in h.done}
            if asynchronous:
                result.update(resume_drains=h.resume_drains, parked_peak=h.parked_peak)
            return result

    def test_async_equals_sync_with_the_goal_overlap(self):
        for permits, fast3 in ((4, True), (1, False)):
            with self.subTest(admission_overlap=permits, fast3=fast3):
                sync = self.serve(False, permits, fast3=fast3)
                asyn = self.serve(True, permits, fast3=fast3)
                self.assertEqual(sorted(asyn['turns']), sorted(sync['turns']))
                for key in sorted(sync['turns']):
                    self.assertEqual(asyn['turns'][key], sync['turns'][key], key)
                self.assertEqual(asyn['goals'], sync['goals'])
                self.assertTrue(any(g is not None for g in sync['goals'].values()))
                self.assertEqual(asyn['forwards'], sync['forwards'])
                self.assertTrue(any('goal_forward' in ops for ops in asyn['forwards'].values()))
                with patch.object(TA, 'IDENTITY_KEYS', TA.IDENTITY_KEYS | {'phase1_lease_id'}):
                    for key in sorted(sync['natives']):
                        self.assertEqual(relaxed(untimed(sync['natives'][key]), untimed(asyn['natives'][key])), [],
                                         key)
                self.assertEqual(asyn['extra'], [])
                self.assertGreater(asyn['voids'], 0)
                self.assertEqual(len(asyn['voided']), asyn['voids'])
                self.assertGreater(asyn['parked_peak'], 0)          # Goals parked while others decoded
                self.assertGreater(asyn['drains'], 0)
                self.assertGreater(asyn['resume_drains'], 0)        # Goal resumes met a batch in flight


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class AsyncKvSessionTests(unittest.TestCase):
    """--async-scheduling on the kv-session protocol (test_gemma_kv_session_e2e's MemAgent lanes of the Gemma
    client: reads that stop on <|tool_response>, continuations without a header, phase 1 longer than one
    forward, idle releases at the selective pool): every turn equals the synchronous server's."""
    maxDiff = None

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def window_of(self, h, sid):
        s = h.adapter.sessions[sid]
        window = h.adapter.window.window(s.namespace)
        return None if window is None else {g: tuple(b.block_id for _, b in v) for g, v in window.items()}

    def check(self, h, info):
        import test_gemma_kv_session_e2e as GKS
        GKS.GemmaMemAgentSessionTests.check(self, h, info)
        info['window'] = window_state(h, info)

    def serve(self, policy, sessions, asynchronous, *, release, small, fast3=True):
        import test_gemma_kv_session_e2e as GKS
        from remote_ra_http.evidence import engine_closed
        with ExitStack() as stack:
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            layout = GE.window_layout(policy, sessions)
            blocks = GKS.GemmaMemAgentSessionTests.small(self, layout) if small else GKS.REFERENCE_BLOCKS
            value = dict(GKS.config(policy, output, layout, release, sessions), **FAST,
                         **(dict(decode_fastpath3=True) if fast3 else {}),
                         **(dict(async_scheduling=True) if asynchronous else {}))
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            cls = GKS.Sessions
            if asynchronous:
                stack.enter_context(patch.object(GE, 'Runner', AsyncRunner))
                cls = async_harness(cls)
            h = cls(policy, output, blocks, release, sessions, False)
            h.output = output
            link = None
            if asynchronous:
                link = attach_async(h, fast3)
                stack.callback(A.detach_engine)
            else:
                attach_sync(h, fast3)
            h.serve(self.check)
            for lane in h.lanes:
                self.assertEqual(lane.agent.answer, 'Answer: chunk 2 names the harbour city.')
            GKS.GemmaMemAgentSessionTests.boundary(self, h)
            by_turn = turns(h)
            result = dict(turns={key: dict(tokens=list(i['tokens']), outputs=list(i['request'].output_token_ids),
                                           text=i['committed']['text'], selected=list(i['plan']['selected']),
                                           visible=h.runner.visible[i['rid']], probe=i['probe'],
                                           finish=i['response']['finish_reason'], window=i['window'])
                                 for key, i in by_turn.items()},
                          natives={key: dict(native=i['native'], receipt=i['receipt']) for key, i in by_turn.items()},
                          answers=[lane.agent.answer for lane in h.lanes])
            if link is not None:
                result.update(voided=list(h.runner.voided_rows), voids=link.voids, extra=list(h.runner.extra_rows),
                              widths=list(h.widths), length_skips=list(h.length_skips))
            path = output / live('idle_release').EVENTS
            result['events'] = Counter(json.loads(line)['event'] for line in path.read_bytes().splitlines()) \
                if path.exists() else Counter()
            h.lifecycle.disconnect()
            h.sessions.close()
            engine_closed(h.core)
            self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                             (h.baseline, h.window_baseline))
            return result

    def test_async_equals_sync(self):
        for policy, sessions, release, small in (('observation_query', 1, False, False),
                                                 ('observation_query', 4, True, True),
                                                 ('goal_query', 3, False, False)):
            with self.subTest(policy=policy, sessions=sessions, release=release):
                sync = self.serve(policy, sessions, False, release=release, small=small)
                asyn = self.serve(policy, sessions, True, release=release, small=small)
                self.assertEqual(sorted(asyn['turns']), sorted(sync['turns']))
                for key in sorted(sync['turns']):
                    self.assertEqual(asyn['turns'][key], sync['turns'][key], key)
                for key in sorted(sync['natives']):
                    self.assertEqual(relaxed(untimed(sync['natives'][key]), untimed(asyn['natives'][key])), [], key)
                self.assertEqual(asyn['answers'], sync['answers'])
                self.assertEqual(asyn['events'], sync['events'])
                self.assertEqual(asyn['extra'], [])
                self.assertGreater(asyn['voids'], 0)
                self.assertEqual(len(asyn['voided']), asyn['voids'])
                if sessions == 1:
                    self.assertEqual(max(asyn['widths']), 1)
                elif not release:
                    # A max_tokens stop not scheduled past, next to scheduled requests (GPU job 3568311's case;
                    # under the release pressure nearly every pass drains, so a stop rarely meets a next step).
                    self.assertTrue(any(scheduled for _, scheduled in asyn['length_skips']), asyn['length_skips'])
                if release:
                    self.assertTrue(asyn['events']['evict'] and asyn['events']['readmit'], asyn['events'])
                # Gemma's stop ids ended the replies (a read on <|tool_response>, a turn on <turn|>).
                self.assertTrue({t['finish'] for t in asyn['turns'].values()} <= {'stop', 'length'})


def async_engine(Base):
    """test_gemma_chat_text_e2e's GemmaEngine (test_ms2_e2e's stub of EngineCoreProc over the window engine and
    worker) in vLLM's asynchronous order. Its loop runs core.py's own _process_input_queue, _remote_ra_hold and
    _remote_ra_resume (test_ms2_e2e.engine_input_loop), whose drains before a client request and a Goal resume
    are the shipped ones: they call ``_remote_ra_async_drain`` here, which updates the batch in flight as
    core.py's does. ``step`` is step_with_batch_queue at two batches: schedule N+1 (vLLM's max_tokens skip of a
    request whose last token is in flight, scheduler.py:489-503; the placeholders), execute it, then update N;
    with nothing to schedule it pops and updates the batch in flight."""

    class AsyncEngine(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.inflight, self.async_drains, self.length_skips = None, 0, 0

        def has_work(self):
            return super().has_work() or self.inflight is not None

        def _remote_ra_async_drain(self):
            if self.inflight is not None:
                step, self.inflight = self.inflight, None
                self.async_drains += 1
                self.update(step)

        def step(self):
            from vllm.v1.core.sched.output import NewRequestData
            order, counts, starts, news = [], {}, {}, []
            for rid in list(self.running) + [r for r in self.waiting if r in self.plans]:
                if self.held(rid):
                    continue
                request = self.requests[rid]
                n, start = len(request.prompt_token_ids), request.num_computed_tokens
                waiting = request.status.name == 'WAITING'
                if (not waiting and request.num_output_placeholders > 0 and
                        start + 2 - request.num_output_placeholders >= n + request.sampling_params.max_tokens):
                    self.length_skips += 1          # the step in flight reaches max_tokens: no extra step
                    continue
                if start < n:
                    count = min(n, (start // P + 1) * P) - start if self.plans[rid].chunk == 'page' else n - start
                    count = hooks.scheduler_split(self.core.scheduler, request, start, count)
                else:
                    count = 1
                result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
                    new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                    delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=waiting,
                    reserved_blocks=0, has_scheduled_reqs=waiting and bool(self.running))
                if result is None:
                    if waiting:
                        break
                    raise AssertionError('running request %s found no free block (preemption)' % rid)
                rows = tuple([b.block_id for b in m.req_to_blocks[rid]]
                             for m in self.kv.coordinator.single_type_managers)
                if waiting:
                    news.append(NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None,
                                               rows, start, None))
                    self.waiting.remove(rid)
                    self.running.append(rid)
                else:
                    self.runner.set_rows(rid, rows)
                order.append(rid)
                counts[rid], starts[rid] = count, start
            if not order:
                if self.inflight is None:
                    return False
                self._remote_ra_async_drain()       # nothing to schedule: the batch in flight is popped
                return True
            for rid in order:                       # vLLM's _update_after_schedule
                request = self.requests[rid]
                request.status.name = 'RUNNING'
                if request.num_computed_tokens + counts[rid] >= len(request.prompt_token_ids):
                    request.num_output_placeholders += 1
                request.num_computed_tokens += counts[rid]
            finished, self.finished = self.finished, set()
            mark = len(self.io.log)
            self.runner.execute_model(NS(num_scheduled_tokens=dict(counts),
                total_num_scheduled_tokens=sum(counts.values()), scheduled_new_reqs=news,
                all_token_ids=TA.cached_token_ids(A.link(), counts, news),
                finished_req_ids=finished, new_block_ids_to_zero=self.kv.take_new_block_ids(),
                kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
            self.batches.append(NS(rows=tuple((rid, starts[rid], counts[rid]) for rid in order),
                                   batch=tuple(self.runner.input_batch.req_ids), io=self.io.log[mark:],
                                   before=None, after=None))
            previous, self.inflight = self.inflight, (order, self.runner.sampled[-1])
            if previous is not None:
                self.update(previous)
            return True

        def update(self, step):
            """update_from_output of a step, in schedule order: the sampled tokens appended, stops finished
            (before_success, then the native free), then the link's predicted stops checked."""
            order, sampled = step
            for rid in order:
                if rid not in self.running:
                    continue
                request = self.requests[rid]
                self.stepped[rid] += 1
                token = sampled.get(rid)
                if token is None:
                    continue
                request.output_token_ids.append(token)
                request.num_output_placeholders -= 1
                if token in self.endpoint.transcript.stop_ids or hooks.output_stop(self.core.scheduler, request):
                    self.finish(rid)
            A.link().check([rid for rid in order if rid not in self.running])

        def shutdown(self):
            self._remote_ra_async_drain()           # the batch in flight finishes before teardown (core.py)
            return super().shutdown()
    AsyncEngine.__name__ = 'Async' + Base.__name__
    return AsyncEngine


def chat_stop(request, max_model_len):
    """The chat-text engine's stop (test_ms2_e2e.BatchEngine's update: a stop id, then vLLM's max_tokens)."""
    return (request.output_token_ids[-1] in MP.GEMMA4_STOP_IDS or
            len(request.output_token_ids) >= request.sampling_params.max_tokens)


def chat_turns(srv):
    """(agent, turn) -> what a chat-text turn served: its prompt, the engine's reply, the row's selection and
    finish, the native file (the agent named by its instruction, 'Task k: ...')."""
    import re
    h, rows, out, seen = srv.engine, {r['request_id']: r for r in srv.rows()}, {}, Counter()
    for entry in srv.served:
        named = re.match(r'Task (\d+):', entry['messages'][0]['content'])
        agent = int(named.group(1)) if named else -1
        key = (agent, seen[agent])
        seen[agent] += 1
        row = rows[entry['rid']]
        out[key] = dict(tokens=list(entry['tokens']), outputs=list(h.requests[entry['rid']].output_token_ids),
                        selected=row['selected_logical_pages'], finish=row['chat_text']['finish_reason'],
                        kind=row['chat_text']['kind'], rebuild=row['stage_b']['rebuild'],
                        native=srv.native(entry['rid']))
    return out


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class AsyncChatTextTests(unittest.IsolatedAsyncioTestCase):
    """--async-scheduling on the chat-text protocol: test_gemma_chat_text_e2e's API server (the packaged router,
    the chat-text route, the real controller and frontend, Terminus 2 clients) over the window engine in vLLM's
    asynchronous order (``async_engine``), against the same server synchronous; fast paths 1-3."""
    maxDiff = None
    import test_gemma_chat_text_e2e as GCE
    asyncSetUp = GCE.GemmaChatTextEndToEndTests.asyncSetUp
    agent, play = GCE.GemmaChatTextEndToEndTests.agent, GCE.GemmaChatTextEndToEndTests.play
    check_prompts, windows = GCE.GemmaChatTextEndToEndTests.check_prompts, GCE.GemmaChatTextEndToEndTests.windows
    window_blocks, boundary = GCE.GemmaChatTextEndToEndTests.window_blocks, GCE.GemmaChatTextEndToEndTests.boundary
    quiescent = GCE.GemmaChatTextEndToEndTests.quiescent
    del GCE

    def server(self, asynchronous, policy='full', fast3=True, **kw):
        import test_gemma_chat_text_e2e as GCE
        stack = ExitStack()
        self.addCleanup(stack.close)
        HT.deterministic(stack)
        config = GCE.config
        flags = dict(FAST, **(dict(decode_fastpath3=True) if fast3 else {}),
                     **(dict(async_scheduling=True) if asynchronous else {}))
        with ExitStack() as building:
            building.enter_context(patch.object(GCE, 'config', lambda *a, **k: (lambda v, l: (dict(v, **flags), l))(
                *config(*a, **k))))
            if asynchronous:
                building.enter_context(patch.object(GE, 'Runner', AsyncRunner))
                building.enter_context(patch.object(GCE, 'GemmaEngine', async_engine(GCE.GemmaEngine)))
            srv = GCE.GemmaChatTextEndToEndTests.server(self, policy, **kw)
        h = srv.engine
        if asynchronous:
            link = attach_async(h, fast3)
            link.check_stop = chat_stop
            h._remote_ra_async_link = link
            self.addCleanup(A.detach_engine)
        else:
            attach_sync(h, fast3)
        return srv

    async def run_agents(self, asynchronous, count, episodes, policy, **kw):
        srv = self.server(asynchronous, policy, slots=count * (episodes + 1), limit=max(4, count), **kw)
        agents = [self.agent(srv, k) for k in range(count)]
        await asyncio.gather(*(self.play(a, episodes) for a in agents))
        h = srv.engine
        self.check_prompts(srv)
        self.assertFalse(srv.rows('failures.jsonl'))
        for agent in agents:
            self.assertTrue(all(status == 200 for _, status, _ in agent.requests))
        result = dict(turns=chat_turns(srv), widths=[len(b.rows) for b in h.batches])
        if asynchronous:
            result.update(voided=list(h.runner.voided_rows), voids=A.link().voids, extra=list(h.runner.extra_rows),
                          drains=h.async_drains)
        return srv, result

    def same(self, sync, asyn):
        self.assertEqual(sorted(asyn['turns']), sorted(sync['turns']))
        for key in sorted(sync['turns']):
            a, b = dict(sync['turns'][key]), dict(asyn['turns'][key])
            native_a, native_b = a.pop('native'), b.pop('native')
            self.assertEqual(b, a, key)
            self.assertEqual(relaxed(untimed(native_a), untimed(native_b)), [], key)
        self.assertEqual(asyn['extra'], [])
        self.assertGreater(asyn['voids'], 0)
        self.assertEqual(len(asyn['voided']), asyn['voids'])

    async def test_one_client_equals_sync(self):
        """b = 1: one Terminus client, four episodes, observation_query; then a turn cut at its bound (a length
        stop vLLM does not schedule past) and one more turn."""
        import test_chat_text_protocol as T
        import test_gemma_chat_text_e2e as GCE
        results = []
        for asynchronous in (False, True):
            srv, result = await self.run_agents(asynchronous, 1, 4, 'observation_query')
            agent_messages = [dict(role='user', content=T.first_prompt('Count.'))]
            status, body = await srv.chat_request(agent_messages, max_tokens=10)
            self.assertEqual((status, body['choices'][0]['finish_reason']), (200, 'length'))
            result['length'] = (body['choices'][0]['message']['content'], body['usage'])
            if asynchronous:
                result['length_skips'] = srv.engine.length_skips
            self.boundary(srv)
            self.quiescent(srv)
            await srv.stop()
            results.append(result)
        sync, asyn = results
        self.same(sync, asyn)
        self.assertEqual(asyn['length'], sync['length'])
        self.assertGreater(asyn['length_skips'], 0)
        self.assertEqual(max(asyn['widths']), 1)

    async def test_four_clients_equal_sync(self):
        """b > 1: four clients decode together (the fakes have no batch variance: every turn equal), voided rows
        next to live ones, admissions between steps (each drained first by core.py's input loop)."""
        results = []
        for asynchronous in (False, True):
            srv, result = await self.run_agents(asynchronous, 4, 2, 'full')
            self.boundary(srv)
            self.quiescent(srv)
            await srv.stop()
            results.append(result)
        sync, asyn = results
        self.same(sync, asyn)
        self.assertGreater(max(asyn['widths']), 1)
        self.assertGreater(asyn['drains'], 0)


if __name__ == '__main__':
    unittest.main()
