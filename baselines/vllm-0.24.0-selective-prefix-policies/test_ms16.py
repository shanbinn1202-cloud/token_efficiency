"""Sixteen sessions per server (branch ms16). CPU only; nothing here is CUDA evidence.

The cap is one constant, ``remote_ra_live_v2.sessions.MAX_SESSIONS`` = 16 (stage 1
set it to 8). Covered at 9-16 sessions:

- routing: startup task lists of 9-16 tasks (17 refused), session ids and their
  round trip, routing by the task message, ambiguous statements still refused;
- the per-process lifetime request bound (lifecycle entries, request tombstones,
  minted bindings): 256 per session, at least 1,024 (branch align-ctx; 128 per
  session above 8 sessions before);
- startup validation: serve.py's dry run with 16 tasks (max_num_seqs 16, the
  decoupled layout with K = 33 GDN slots) accepted by remote_ra_http.config, 17
  refused; the launcher's refusal of a KV budget whose attention pool cannot hold
  one session at the context limit once the Mamba pool has grown with S (every
  1-8 session configuration passes, so nothing changes there);
- the Mamba pool, 1 + 3(2S + 1) blocks (100 at S = 16), in the layout, in the vLLM
  planner and in the engine's startup cross-check; the worker's pinned Page arena;
- the capacity gate with 15 requests running beside an admitting turn;
- 16 sessions decoding in one batch end to end (decoupled 16-token Pages and the
  legacy 784-token pool), no preemption, both pools back at their baselines, every
  turn's prompt and reply equal to the upstream reference and, for the sessions a
  3-session server also serves, equal to that server's.

Sessions above 5 per server have not run on a GPU.
"""
import contextlib
from contextlib import ExitStack
from hashlib import sha256
import importlib.util
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
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import capacity
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import sessions
from remote_ra_live_v2.capacity import CapacityWait
from remote_ra_live_v2.lifecycle import EngineLifecycle

GIB = 1024 ** 3
S16 = 16


def tasks(count, start=0):
    return [dict(instance_id='repo__m%02d-%d' % (k, 100 + k),
                 problem_statement='Fix the failing check %02d in module m%02d.' % (k, k))
            for k in range(start, start + count)]


def chat(task, extra=()):
    return [dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content='<pr_description>\n%s\n</pr_description>' % task['problem_statement']),
            *extra]


def pages(tokens, page_tokens):
    return -(-tokens // page_tokens)


class CapTests(unittest.TestCase):
    def test_one_constant_caps_every_layer(self):
        from remote_ra_native_growing.adapter import MAX_SESSIONS as adapter_cap
        self.assertEqual(sessions.MAX_SESSIONS, S16)
        self.assertEqual(adapter_cap, S16)
        self.assertEqual(EngineLifecycle(NS()).max_sessions, S16)
        spec = importlib.util.spec_from_file_location('build_tasks', ROOT / 'examples/swebench/build_tasks.py')
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        self.assertEqual(build.MAX_SESSIONS, S16)
        self.assertEqual([len(g) for g in build.session_groups(['i%d' % k for k in range(20)], 16)], [10, 10])
        self.assertEqual([len(g) for g in build.session_groups(['i%d' % k for k in range(16)], 16)], [16])
        with self.assertRaises(SystemExit):
            build.session_groups(['i0'], 17)

    def test_task_lists_of_9_to_16_and_17_refused(self):
        for count in range(9, S16 + 1):
            self.assertEqual(sessions.validate_tasks(tasks(count)), tasks(count))
        with self.assertRaisesRegex(RuntimeError, r'startup task list of 1\.\.16 tasks required'):
            sessions.validate_tasks(tasks(S16 + 1))
        # Containment is still refused among sixteen tasks.
        bad = tasks(15) + [dict(instance_id='x', problem_statement=tasks(1)[0]['problem_statement'] + ' Also docs.')]
        with self.assertRaisesRegex(RuntimeError, 'no problem statement may contain another'):
            sessions.validate_tasks(bad)

    def test_ids_index_and_routing_at_16(self):
        ids = sessions.session_ids('http-x', S16)
        self.assertEqual((ids[0], ids[9], ids[15], len(set(ids))), ('http-x', 'http-x-s9', 'http-x-s15', 16))
        self.assertEqual([sessions.session_index('http-x', S16, s) for s in ids], list(range(S16)))
        for foreign in ('http-x-s16', 'http-x-s0', 'http-y-s3'):
            with self.assertRaises(RuntimeError):
                sessions.session_index('http-x', S16, foreign)
        with self.assertRaises(RuntimeError):
            sessions.session_ids('http-x', S16 + 1)
        listed = tasks(S16)
        for k, task in enumerate(listed):
            # Only messages[1] decides, even when a later message quotes another task.
            later = chat(task, [dict(role='assistant', content='ls'),
                                dict(role='user', content=listed[(k + 1) % S16]['problem_statement'])])
            self.assertEqual((sessions.route(listed, chat(task)), sessions.route(listed, later)), (k, k))
        with self.assertRaisesRegex(RuntimeError, 'request matches no startup task'):
            sessions.route(listed, chat(tasks(1, start=S16)[0]))

    def test_request_limit(self):
        self.assertEqual([sessions.request_limit(n) for n in range(1, 9)], [1024] * 4 + [1280, 1536, 1792, 2048])
        self.assertEqual([sessions.request_limit(n) for n in (9, 12, 16)], [2304, 3072, 4096])
        for bad in (0, 17, 8.0, True):
            with self.assertRaises(RuntimeError):
                sessions.request_limit(bad)

    def test_engine_lifecycle_and_bindings_are_sized_from_the_task_list(self):
        from registered_add_protocol import RegisteredAddDispatcher
        from remote_ra_live_v2.composition import engine_lifecycle
        from remote_ra_live_v2.registered_route import bind_registered_dispatcher
        for count, limit in ((1, 1024), (4, 1024), (8, 2048), (9, 2304), (16, 4096)):
            with self.subTest(sessions=count):
                core = NS(_remote_ra_trusted_issues=tasks(count))
                lifecycle = engine_lifecycle(core, NS())
                self.assertEqual((lifecycle.max_requests, lifecycle.max_sessions), (limit, S16))
                dispatcher = RegisteredAddDispatcher(codec=None, boundary=None, decode_core=lambda raw: raw,
                                                     preprocess=lambda request: request)
                self.assertEqual(dispatcher._max_records, 1024)
                bind_registered_dispatcher(dispatcher, lifecycle)
                self.assertEqual(dispatcher._max_records, limit)
        # A lifecycle built with the default keeps the dispatcher's own 1,024.
        dispatcher = RegisteredAddDispatcher(codec=None, boundary=None, decode_core=lambda raw: raw,
                                             preprocess=lambda request: request)
        bind_registered_dispatcher(dispatcher, EngineLifecycle(NS()))
        self.assertEqual(dispatcher._max_records, 1024)

    def config_load(self, **changes):
        task = tasks(1)[0]
        value = dict(schema='remote-ra-http-config-v2', policy='full', protocol='mswea_xml_v1',
                     selection_contract=None, task=task, page_tokens=784, selection_seed=42,
                     selection_fraction=.5, warmup_requests=0,
                     output_directory=str(Path(tempfile.gettempdir()).resolve()),
                     input_pins_sha256='0' * 64, task_sha256='0' * 64, common_config_sha256='0' * 64,
                     issue_sha256=sha256(task['problem_statement'].encode()).hexdigest())
        value.update(changes)
        raw = json.dumps(value).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'startup.json'
            path.write_bytes(raw)
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=sha256(raw).hexdigest()):
                return http_config.load()

    def test_startup_config_with_16_tasks(self):
        listed = tasks(S16)
        value = self.config_load(tasks=listed, tasks_sha256=http_config.digest(listed))
        self.assertEqual(value['tasks'], listed)
        listed = tasks(S16 + 1)
        with self.assertRaisesRegex(RuntimeError, r'1\.\.16'):
            self.config_load(tasks=listed, tasks_sha256=http_config.digest(listed))


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

    def prepare(self, task, policy='full', kv_cache_gib=8, page_tokens=784, layout=None, selection=None):
        type(self).count += 1
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.count), 8231,
                                 selection, 'mswea-xml', kv_cache_gib, page_tokens, layout)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                return http_config.load()

    def test_sixteen_tasks_in_the_shared_pool(self):
        _, config, argv, _ = self.prepare(tasks(S16))
        effective = config['effective_server_config']
        self.assertEqual((effective['max_num_seqs'], argv[argv.index('--max-num-seqs') + 1]), (16, '16'))
        self.assertEqual((config['tasks'], config['task']), (tasks(S16), tasks(1)[0]))
        self.assertNotIn('kv_layout', config)
        self.assertEqual(self.load(config)['tasks'], tasks(S16))
        with patch.object(http_config, 'load', return_value=config):
            self.assertEqual(http_config.effective_server_config(NS(**effective)), effective)
        # One running sequence per startup session: 15 is refused for 16 tasks.
        pinned = dict(config, effective_server_config=dict(effective, max_num_seqs=15))
        pinned['server_config_sha256'] = serve.pin(pinned['effective_server_config'])
        with patch.object(http_config, 'load', return_value=pinned):
            with self.assertRaisesRegex(RuntimeError, 'one sequence per startup session'):
                http_config.effective_server_config(NS(**pinned['effective_server_config']))

    def test_seventeen_tasks_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, r'1\.\.16'):
            self.prepare(tasks(S16 + 1))

    def test_sixteen_tasks_decoupled(self):
        for gib, blocks in ((16, 11490), (24, 19682), (32, 27874)):
            with self.subTest(gib=gib):
                _, config, argv, _ = self.prepare(tasks(S16), 'goal_query', gib, 16,
                                                  selection=dict(no_goal_rewriter=True))
                record = dict(page_tokens=16, mamba_tokens=784, mamba_slots=33)
                self.assertEqual(config['kv_layout'], record)
                self.assertEqual(argv[-6:], ['--block-size', '16', '--mamba-block-size', '784', '--additional-config',
                                             json.dumps({'remote_ra_kv_layout': record}, sort_keys=True,
                                                        separators=(',', ':'))])
                layout = I0.for_server(16, S16, True)
                self.assertEqual(http_config.kv_layout(dict(config, tasks=config['tasks'])), layout)
                self.assertEqual(self.load(config)['kv_layout'], record)
                self.assertEqual(serve.attention_blocks(gib * GIB, layout), blocks)
                with patch.object(http_config, 'load', return_value=config):
                    http_config.effective_server_config(NS(**config['effective_server_config']))

    def test_the_mamba_pool_can_leave_too_few_attention_blocks(self):
        # 8 GiB at P = 16: 9 sessions leave 5,352 usable attention blocks (one session at
        # 64K needs 5,121 + 97 + 53 = 5,271), 10 sessions 5,059 (needs 5,268).
        self.prepare(tasks(9), kv_cache_gib=8, page_tokens=16)
        with self.assertRaisesRegex(ValueError, r'--kv-cache-gib 8 with 10 sessions leaves 5059 usable attention '
                                                r'blocks of 16 tokens .* needs 5268'):
            self.prepare(tasks(10), kv_cache_gib=8, page_tokens=16)
        with self.assertRaisesRegex(ValueError, 'with 16 sessions leaves 3297 usable attention blocks'):
            self.prepare(tasks(S16), kv_cache_gib=8, page_tokens=16)
        with self.assertRaisesRegex(ValueError, 'with 9 sessions leaves'):
            self.prepare(tasks(9), kv_cache_gib=8, page_tokens=784, layout='decoupled')
        self.prepare(tasks(S16), kv_cache_gib=16, page_tokens=16)
        # The shared 784-token pool has no Mamba pool: sixteen sessions start at 8 GiB.
        self.prepare(tasks(S16), kv_cache_gib=8)

    def test_every_1_to_8_session_configuration_passes(self):
        # The refusal changes nothing up to 8 sessions: every launcher choice passes.
        for P in (16, 32, 64, 784):
            for gib in serve.KV_CACHE_GIB:
                for count in range(1, 9):
                    layout = I0.for_server(P, count, True)
                    blocks = serve.attention_blocks(gib * GIB, layout)
                    self.assertGreaterEqual(blocks - 1, serve.one_session_blocks(P, blocks),
                                            (P, gib, count))
        self.prepare(tasks(8), kv_cache_gib=8, page_tokens=16)
        self.prepare(tasks(8), kv_cache_gib=8, page_tokens=784, layout='decoupled')
        # The formula gives the pool sizes of the validated decoupled runs (README 4.4).
        for gib, count, attention, mamba in ((8, 1, 7702, 10), (24, 4, 23205, 28), (24, 5, 22912, 34)):
            layout = I0.for_server(16, count, True)
            self.assertEqual((serve.attention_blocks(gib * GIB, layout), I0.mamba_pool_blocks(layout, 3)),
                             (attention, mamba))


class MambaPoolTests(unittest.TestCase):
    def test_layout_slots_and_pool_blocks_9_to_16(self):
        for P, M in ((16, 784), (32, 800), (64, 832), (784, 784)):
            for count in range(9, S16 + 1):
                layout = I0.for_server(P, count, True)
                self.assertEqual((layout.mamba_tokens, layout.mamba_slots), (M, 2 * count + 1))
                self.assertEqual(I0.mamba_pool_blocks(layout, 3), 1 + 3 * (2 * count + 1))
        layout = I0.for_server(16, S16, True)
        self.assertEqual((layout.mamba_slots, I0.mamba_pool_blocks(layout, 3)), (33, 100))
        # 100 blocks of 51,314,688 B: 4.78 GiB of the KV budget.
        self.assertEqual(100 * serve.MAMBA_BLOCK_BYTES, 5131468800)

    def test_vllm_planner_at_9_to_16_sessions(self):
        import test_kv_pools as KP
        w = KP.world()
        for P in (16, 784):
            for count in (9, 12, 16):
                for gib in (16, 24, 32):
                    with self.subTest(P=P, sessions=count, gib=gib):
                        layout = I0.for_server(P, count, True)
                        _, config = KP.plan(w, layout, gib * GIB)
                        n_fa = KP.fa_pool_blocks(layout, gib * GIB, count)
                        self.assertEqual(config.num_blocks_by_pool, (n_fa, 1 + 3 * (2 * count + 1)))
                        self.assertEqual(n_fa, serve.attention_blocks(gib * GIB, layout))
                        self.assertLessEqual(sum(t.size for t in config.kv_cache_tensors), gib * GIB)

    def test_worker_page_arena_at_16_sessions(self):
        from remote_ra_live_v2.worker_composition import arena_pages
        runner = NS(max_model_len=65536)
        self.assertEqual(arena_pages(runner, {}, I0.for_server(16, S16, True)), 16 * 4096 + 16)
        self.assertEqual(arena_pages(runner, {}, I0.for_server(784, S16, True)), 16 * 84 + 16)
        # Up to 65,552 Pages of 1 MiB pinned at P = 16 (allocated on demand, 256 MiB chunks).
        self.assertEqual(arena_pages(runner, {}, I0.for_server(16, S16, True)) * 16 * 65536 / GIB, 64.015625)

    def test_engine_startup_accepts_exactly_the_16_session_pool(self):
        import test_decouple_wp2 as WP2
        import test_stageb_engine as E
        from remote_ra_native_growing.adapter import EngineCoreAdapter
        from remote_ra_native_growing.contract import Authority
        layout = I0.for_server(16, S16, True)
        with WP2.matrix_case(layout):
            kv = WP2.KVManager(layout, 2000)
            self.assertEqual(kv.kv_cache_config.num_blocks_by_pool, (2000, 100))
            adapter = EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), kv, page_tokens=16,
                                        full_attention_managers=(kv.fa,), layout=layout)
            self.assertIs(adapter.mamba_pool, kv.mamba_pool)
            for count in range(S16):
                adapter.open('x%d' % count, 'full', 'm', 'r')
            with self.assertRaisesRegex(Exception, 'session collision/limit'):
                adapter.open('x16', 'full', 'm', 'r')
            # A Mamba pool sized for 15 sessions is refused by a 16-session layout.
            small = WP2.KVManager(I0.for_server(16, 15, True), 2000)
            with self.assertRaisesRegex(Exception, 'Mamba pool 1 of 1 \\+ groups x slots blocks'):
                EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), small, page_tokens=16,
                                  full_attention_managers=(small.fa,), layout=layout)


class CapacityGateTests(unittest.TestCase):
    """The two-pool gate of a 16-session server: 15 requests running at their phase-2
    start (each may still allocate its row, a running and a crossing GDN slot) beside
    session s's admitting turn."""

    def test_fifteen_running_requests_and_one_admission(self):
        import test_decouple_wp2 as WP2
        import test_stageb_engine as E
        layout = I0.for_server(16, S16, True)
        with WP2.matrix_case(layout):
            h = WP2.Harness(layout, 'full', fa_blocks=8000)
            WP2.run_script(h, WP2.script_for(16)[:3])
            self.assertEqual((len(h.mamba_pool.blocks), h.mamba_pool.get_num_free_blocks()), (100, 99))
            entries, fa_left = {}, 0
            for k in range(15):
                prompt = 1200 + 97 * k
                request = NS(prompt_token_ids=list(range(prompt)), num_computed_tokens=prompt,
                             sampling_params=NS(max_tokens=1536), output_token_ids=[])
                rid = 'run%d' % k
                h.endpoint.claims[rid] = dict(geometry=NS(phase2_start=prompt))   # before its first step
                entries[rid] = dict(state='accepted', request=request)
                fa_left += pages(prompt + 1536, 16) - pages(prompt, 16)
            lifecycle = NS(entries=entries, endpoint=NS(core=h.core), goals=None)
            # Every running request may still cross an M-token block: running + crossing slot each.
            self.assertEqual(capacity.headroom_by_pool(lifecycle), (fa_left, 15 * 6))
            request, binding = h.render(E.obs(5))
            h.kv.watermark_blocks = 7
            command = NS(session_id='s', metadata=dict(turn=h.session.turn), core_b64='', source=None)
            h.core._remote_ra_assembly = NS(dispatcher=NS(_decode_core=lambda raw: request))
            with patch('base64.b64decode', lambda raw, validate: b''):
                need = capacity.assess(lifecycle, command)
                self.assertEqual((need['headroom'], need['watermark']), ((fa_left, 90), (7, 0)))
                # K = 2S + 1: 15 running requests' slots plus this turn's never exhaust the Mamba pool.
                self.assertLessEqual(need['phase2'][1] + need['headroom'][1], need['free'][1])
                self.assertEqual(need['phase2'][1], 6)
                exact = need['phase2'][0] + need['headroom'][0] + need['watermark'][0]
                hog = h.pool.get_new_blocks(h.pool.get_num_free_blocks() - exact)
                self.assertEqual(capacity.admission_gate(lifecycle, command)['free'], (exact, 99))
                hog += h.pool.get_new_blocks(1)
                with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: FA pool: .* %d free \(FA %d free, '
                                            r'Mamba 99 free\)$' % (exact - 1, exact - 1)):
                    capacity.admission_gate(lifecycle, command)
                h.pool.free_blocks(hog)
            h.endpoint.claims.pop(binding.request)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class BatchedSixteenSessionsTests(unittest.TestCase):
    """test_decouple_e2e's multi-session engine loop with sixteen tasks: admissions
    between steps, one execute_model for every running request, the per-pool gate."""
    maxDiff = None

    def run_server(self, case, policy, listed, blocks, tight=False):
        import test_decouple_e2e as DE
        multi = DE.Case(**dict(vars(case), sessions=len(listed), blocks=blocks))
        with ExitStack() as stack:
            DE.enter_case(stack, multi)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = DE.config(policy, output, multi, listed)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = DE.MultiHarness(policy, output, multi, tasks=listed)
            h.active = h.restores = 0
            self.assertEqual((h.lifecycle.max_requests, h.authority.max_attempts),
                             (sessions.request_limit(len(listed)),) * 2)
            if case.decoupled:
                self.assertEqual(len(h.mamba_pool.blocks), 1 + 3 * (2 * len(listed) + 1))
            h.serve(lambda h, info: DE.MultiSessionMatrixTests.check(self, h, info))
            self.assertEqual(len(h.done), 5 * len(listed))       # no preemption (batch_step asserts it)
            self.assertGreater(h.probes_while_running, 0)        # phase 1 while others decode
            if tight:
                self.assertGreater(h.waits, 0)                   # CAPACITY_WAIT, then admitted
            else:
                self.assertEqual(max(h.widths), len(listed))     # every session in one step
                self.assertEqual(h.waits, 0)
            h.close()
            DE.engine_closed(h.core)
            self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
            self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)
            self.assertTrue(all(block.ref_cnt == 0 for pool in h.kv.block_pools for block in pool.blocks[1:]))
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual(len(cleanup['sessions']), len(listed))
            return {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'])
                                 for i in h.done if i['lane'] is lane] for lane in h.lanes}, h

    def test_sixteen_sessions_decode_in_one_batch(self):
        import test_decouple_e2e as DE
        listed = list(DE.TASKS) + tasks(S16 - len(DE.TASKS))
        # Default suite: full on the shared 784 pool; the slow pass (README section 2.3) adds
        # goal_query at P = 16 with its tight-pool run.
        matrix = ((DE.CASES[2], 'goal_query', 16 * 400), (DE.CASES[0], 'full', 16 * 12))
        for case, policy, blocks in matrix if os.environ.get('REMOTE_RA_SLOW_TESTS') == '1' else matrix[1:]:
            with self.subTest(case=case.name, policy=policy):
                sixteen, h = self.run_server(case, policy, listed, blocks)
                if policy == 'goal_query':
                    self.assertGreater(h.active, 0)              # selection active
                # Token-exact against a 3-session server for the sessions both serve.
                three, _ = self.run_server(case, policy, list(DE.TASKS), blocks)
                for k in range(len(DE.TASKS)):
                    self.assertEqual(sixteen[k], three[k])
                if case.decoupled:
                    # A pool too small for every session at once: turns wait at the gate
                    # and nothing else changes. 3,200 blocks: at 3,000 the idle sessions' pins
                    # leave a lone turn less than its worst case, which the gate admitted
                    # unchecked while nothing ran until tb16 job 3398129's fix; now it waits
                    # there, and with every session idle the run stalls.
                    tight, h = self.run_server(case, policy, listed, 3200, tight=True)
                    self.assertEqual(tight, sixteen)


if __name__ == '__main__':
    unittest.main()
