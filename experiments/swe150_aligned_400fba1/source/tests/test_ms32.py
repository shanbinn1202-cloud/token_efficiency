"""Up to 32 sessions per server (branch sessions-32-1116). CPU only; nothing here is CUDA evidence.

``remote_ra_live_v2.sessions.MAX_SESSIONS`` and ``admission_flags.MAX_PERMITS`` go from 16 to
32, for the concurrency experiments of a session pool (K = 20, 24, 32). A chat-text pool keeps
its own cap (``MAX_CHAT_TEXT_RUNNING`` 24). Nothing else is keyed by the cap: the Mamba pool
(2K + 1 slots), the attention-pool refusal, the pinned-host check, the lifetime request bound
and vLLM's batch sizes follow K as before, so every configuration of up to 16 sessions is the
one of the 16-session cap (test_ms16, test_ms_pool, test_int_flags unchanged apart from the
bound refusals). Covered above 16 sessions:

- the bounds: task lists and session ids of 17-32 (33 refused), K of a pool 17-32 over up to
  64 tasks (33 refused; chat-text still 1-24; the registration cap is 256 since integrate-1117,
  test_tasks256), the request bounds, the admission permits and their pinned timeouts up to
  K = 32 (4,020 s / 4,080 s);
- the launcher at K = 20, 24, 32 (serve.py's dry run on the release fixture of test_installer):
  max_num_seqs = K, the decoupled layout with 2K + 1 GDN slots, the startup config and the
  frontend's argument check; the attention-pool refusal at 262,144 tokens (25 GiB holds K <= 25
  with 4,096-token replies; 32 GiB every K up to 32), the pinned-host check and the
  --dram-arena-gib range, --admission-overlap K, --phase1-graph-tokens' capture sizes;
- the layout numbers: the attention and Mamba pools from the launcher's formula against
  vLLM's planner (test_kv_pools), the worker's pinned Page arena;
- vLLM 0.24's capture sizes at K = 20, 24, 32 (its own sizing code, as test_int_flags);
- the engine's startup cross-check of a 24-session pool, and the capacity gate with 23
  requests running beside an admitting turn;
- end to end on the engine (test_decouple_e2e's batched engine and worker): 24 sessions
  decoding in one batch (no preemption, the pools back at their baselines, the sessions a
  3-session server also serves token-exact; with goal_query at 16-token Pages, in the slow
  pass, a turn may wait once for the Mamba pool: BatchedTwentyFourSessionsTests), and a session
  pool of K = 24 over 32 tasks (test_ms_pool's pool harness: never more than K open or
  decoding, refill, closes).

Not GPU-validated above 16 sessions (chat-text: 20 in job 3461661).
"""
import contextlib
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import admission_flags as AF
from remote_ra_live_v2 import capacity
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import sessions
from remote_ra_live_v2.capacity import CapacityWait
from remote_ra_live_v2.lifecycle import EngineLifecycle

GIB = 1024 ** 3
CAP = 32
L256 = 262144
# 25 GiB at 16-token Pages (the group's KV size): attention blocks and Mamba blocks by K.
POOLS_25 = {16: (20706, 100), 20: (19531, 124), 24: (18357, 148), 32: (16008, 196)}

try:
    import torch
except ImportError:
    torch = None


def tasks(count, start=0):
    return [dict(instance_id='repo__q%02d-%d' % (k, 300 + k),
                 problem_statement='Mend the broken stage %02d of workflow w%02d.' % (k, k))
            for k in range(start, start + count)]


def pages(tokens, page_tokens):
    return -(-tokens // page_tokens)


class BoundTests(unittest.TestCase):
    def test_the_caps(self):
        from remote_ra_native_growing.adapter import EngineCoreAdapter
        self.assertEqual((sessions.MAX_SESSIONS, AF.MAX_PERMITS, sessions.MAX_TASKS, sessions.MAX_CHAT_TEXT_RUNNING),
                         (CAP, CAP, 256, 24))
        self.assertEqual((EngineCoreAdapter.max_open, EngineLifecycle(NS()).max_sessions), (CAP, CAP))
        self.assertEqual((sessions.running_cap(), sessions.running_cap('mswea_xml_v1'),
                          sessions.running_cap(sessions.CHAT_TEXT_PROTOCOL)), (CAP, CAP, 24))
        self.assertEqual((sessions.task_limit(False), sessions.task_limit(True)), (CAP, 256))

    def test_task_lists_ids_and_routing_up_to_32(self):
        for count in (17, 20, 24, CAP):
            self.assertEqual(sessions.validate_tasks(tasks(count)), tasks(count))
        with self.assertRaisesRegex(RuntimeError, r'startup task list of 1\.\.32 tasks required'):
            sessions.validate_tasks(tasks(CAP + 1))
        ids = sessions.session_ids('http-x', CAP)
        self.assertEqual((ids[0], ids[16], ids[31], len(set(ids))), ('http-x', 'http-x-s16', 'http-x-s31', CAP))
        self.assertEqual([sessions.session_index('http-x', CAP, s) for s in ids], list(range(CAP)))
        for foreign in ('http-x-s32', 'http-x-s0'):
            with self.assertRaises(RuntimeError):
                sessions.session_index('http-x', CAP, foreign)
        with self.assertRaises(RuntimeError):
            sessions.session_ids('http-x', CAP + 1)
        listed = tasks(CAP)
        for k, task in enumerate(listed):
            messages = [dict(role='system', content='s'), dict(role='user', content=task['problem_statement'])]
            self.assertEqual(sessions.route(listed, messages), k)

    def test_running_sessions_of_a_pool(self):
        listed = tasks(64)
        self.assertEqual([sessions.running_limit(k, listed) for k in (17, 20, 24, CAP)], [17, 20, 24, CAP])
        for bad, registered in ((CAP + 1, listed), (24, tasks(20))):
            with self.assertRaisesRegex(RuntimeError, 'max_running_sessions must be 1..32 and at most the number'):
                sessions.running_limit(bad, registered)
        # A chat-text pool keeps its own cap.
        self.assertEqual(sessions.running_limit(24, listed, sessions.CHAT_TEXT_PROTOCOL), 24)
        with self.assertRaisesRegex(RuntimeError, 'max_running_sessions must be 1..24'):
            sessions.running_limit(25, listed, sessions.CHAT_TEXT_PROTOCOL)

    def test_request_bounds(self):
        # 256 per session, at least 1,024: 5,120 / 6,144 / 8,192 at 20 / 24 / 32 fixed sessions;
        # a pool sizes them from its registered tasks (16,384 at 64).
        self.assertEqual([sessions.request_limit(n) for n in (17, 20, 24, CAP)], [4352, 5120, 6144, 8192])
        self.assertEqual(sessions.request_limit(64, sessions.MAX_TASKS), 16384)
        with self.assertRaises(RuntimeError):
            sessions.request_limit(CAP + 1)
        from registered_add_protocol import RegisteredAddDispatcher
        from remote_ra_live_v2.composition import engine_lifecycle
        from remote_ra_live_v2.registered_route import bind_registered_dispatcher
        for pool, count, limit, most in ((None, 24, 6144, CAP), (24, 64, 16384, 64), (20, 32, 8192, 32)):
            with self.subTest(pool=pool, tasks=count):
                core = NS(_remote_ra_trusted_issues=tasks(count), _remote_ra_session_pool=pool)
                lifecycle = engine_lifecycle(core, NS())
                self.assertEqual((lifecycle.max_requests, lifecycle.max_sessions), (limit, most))
                dispatcher = RegisteredAddDispatcher(codec=None, boundary=None, decode_core=lambda raw: raw,
                                                     preprocess=lambda request: request)
                bind_registered_dispatcher(dispatcher, lifecycle)
                self.assertEqual(dispatcher._max_records, limit)


class AdmissionPermitTests(unittest.TestCase):
    def test_permits_and_their_pinned_timeouts_up_to_32(self):
        # 300 + 120 (K - 1) and 360 + 120 (K - 1): one more pinned 120-s rewriter call per permit.
        for K, pair in ((16, (2100, 2160)), (20, (2580, 2640)), (24, (3060, 3120)), (CAP, (4020, 4080))):
            self.assertEqual(AF.timeouts(K), pair)
            record = AF.record(admission_overlap=K)
            self.assertEqual(record, dict(admission_overlap=dict(permits=K, admission_seconds=pair[0],
                                                                 lease_seconds=pair[1])))
            self.assertEqual((AF.check(record), AF.permits(record), AF.pinned_timeouts(record)),
                             (record, K, pair))
            self.assertEqual(EngineLifecycle(NS(), lease_seconds=pair[1]).lease, pair[1])
        self.assertEqual(len(AF.allowed_timeouts()), CAP)
        for call in (lambda: AF.timeouts(CAP + 1), lambda: AF.overlap_record(CAP + 1),
                     lambda: AF.record(admission_overlap=CAP + 1),
                     lambda: AF.check(dict(admission_overlap=dict(permits=33, admission_seconds=4140,
                                                                  lease_seconds=4200))),
                     lambda: EngineLifecycle(NS(), lease_seconds=4200)):
            with self.assertRaises(RuntimeError):
                call()


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

    def prepare(self, task, policy='full', kv_cache_gib=25, page_tokens=16, layout=None, selection=None, **kw):
        type(self).count += 1
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.count), 8231,
                                 selection, 'mswea-xml', kv_cache_gib, page_tokens, layout, **kw)

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

    def test_pools_of_20_24_and_32_over_64_tasks(self):
        listed = tasks(64)
        for K in (20, 24, CAP):
            with self.subTest(K=K):
                # 32 running sessions pin 133.8 GiB at the limit: they need an arena (70 GiB as the SWE runs).
                arena = dict(dram_arena_gib=70) if K == CAP else {}
                _, config, argv, _ = self.prepare(listed, max_running=K, connector_pool_gib=1, **arena)
                effective = config['effective_server_config']
                self.assertEqual((effective['max_num_seqs'], argv[argv.index('--max-num-seqs') + 1]), (K, str(K)))
                record = dict(page_tokens=16, mamba_tokens=784, mamba_slots=2 * K + 1)
                self.assertEqual((config['kv_layout'], config['max_running_sessions'], config['tasks']),
                                 (record, K, listed))
                self.assertEqual(argv[argv.index('--additional-config') + 1],
                                 json.dumps({'remote_ra_kv_layout': record}, sort_keys=True, separators=(',', ':')))
                layout = I0.for_server(16, K, True)
                self.assertEqual(http_config.kv_layout(config), layout)
                self.assertEqual((serve.attention_blocks(25 * GIB, layout), I0.mamba_pool_blocks(layout, 3)),
                                 POOLS_25[K])
                loaded = self.load(config)
                self.assertEqual((loaded['max_running_sessions'], http_config.running_sessions(loaded)), (K, K))
                self.assertEqual(self.effective(config), effective)
                # One running sequence per open session: 16 (the old cap) or 64 (one per task) are refused.
                for seqs in (16, 64):
                    pinned = dict(config, effective_server_config=dict(effective, max_num_seqs=seqs))
                    pinned['server_config_sha256'] = serve.pin(pinned['effective_server_config'])
                    with self.assertRaisesRegex(RuntimeError, 'one sequence per startup session'):
                        self.effective(pinned)
        with self.assertRaisesRegex(RuntimeError, r'max_running_sessions must be 1\.\.32'):
            self.prepare(listed, max_running=CAP + 1, connector_pool_gib=1)

    def test_fixed_24_sessions_and_the_pinned_host_budget(self):
        # 24 fixed sessions at 65,536 tokens pin 131.7 GiB with vLLM's 32-GiB connector pool (over the
        # 128 GiB budget) and 100.7 GiB with the 1-GiB pool (the command line's speed default).
        with self.assertRaisesRegex(ValueError, r'24 sessions at the 65536-token limit may pin 131\.7 GiB'):
            self.prepare(tasks(24))
        _, config, argv, _ = self.prepare(tasks(24), connector_pool_gib=1)
        self.assertEqual((config['effective_server_config']['max_num_seqs'], config['kv_layout']['mamba_slots']),
                         (24, 49))
        self.assertNotIn('max_running_sessions', config)
        self.assertEqual(self.load(config)['tasks'], tasks(24))
        self.effective(config)
        # 32 sessions need an arena even at 65,536 tokens: at most 122 GiB.
        with self.assertRaisesRegex(ValueError, r'32 sessions at the 65536-token limit may pin 133\.8 GiB'):
            self.prepare(tasks(CAP), connector_pool_gib=1)
        self.prepare(tasks(CAP), connector_pool_gib=1, dram_arena_gib=122)
        with self.assertRaisesRegex(ValueError, r'or a smaller --dram-arena-gib \(at most 122 GiB\)'):
            self.prepare(tasks(CAP), connector_pool_gib=1, dram_arena_gib=123)

    def test_the_attention_pool_refusal_at_262144_tokens(self):
        # One session's turn at the limit with 4,096-token replies: 16,385 + 1 + 1,024 + 257 + 1 blocks plus the
        # 1 % watermark. 25 GiB holds it up to K = 25 (18,062 usable of 17,846 needed) and not at 26.
        listed = tasks(64)
        long = dict(max_model_len=L256, max_tokens=4096, dram_arena_gib=70, connector_pool_gib=1)
        for K in (20, 24, 25):
            _, config, _, _ = self.prepare(listed, max_running=K, **long)
            self.assertEqual((config['dram_arena_pages'], config['kv_layout']['mamba_slots']), (71680, 2 * K + 1))
            self.load(config)
        for K, usable, needed in ((26, 17769, 17843), (CAP, 16007, 17826)):
            with self.assertRaisesRegex(ValueError, r'--kv-cache-gib 25 with %d sessions leaves %d usable attention '
                                                    r'blocks of 16 tokens after the Mamba pool; one session at the '
                                                    r'262144-token limit with 4096-token replies needs %d'
                                        % (K, usable, needed)):
                self.prepare(listed, max_running=K, **long)
        # 24 GiB holds K <= 20 there; 32 GiB every K up to 32.
        self.prepare(listed, 'full', 24, max_running=20, **long)
        with self.assertRaisesRegex(ValueError, 'with 24 sessions leaves 17332 usable'):
            self.prepare(listed, 'full', 24, max_running=24, **long)
        _, config, _, _ = self.prepare(listed, 'full', 32, max_running=CAP, **long)
        self.assertEqual(serve.attention_blocks(32 * GIB, I0.for_server(16, CAP, True)), 23176)
        # At 65,536 and 131,072 tokens every K up to 32 passes at 25 GiB.
        for length in (65536, 131072):
            self.prepare(listed, max_running=CAP, max_model_len=length, max_tokens=4096, dram_arena_gib=70,
                         connector_pool_gib=1)

    def test_the_arena_range_at_262144_tokens(self):
        listed = tasks(64)
        long = dict(max_model_len=L256, max_tokens=4096, connector_pool_gib=1)
        # Without an arena 24 sessions at the limit would pin 388.7 GiB.
        with self.assertRaisesRegex(ValueError, r'24 sessions at the 262144-token limit may pin 388\.7 GiB'):
            self.prepare(listed, max_running=24, **long)
        # One session at the limit is 16,385 Pages: 16 GiB holds 16,384.
        with self.assertRaisesRegex(ValueError, r'--dram-arena-gib 16 holds 16384 Pages of 16 tokens; it must '
                                                r'hold one session at the 262144-token limit \(16385 Pages\)'):
            self.prepare(listed, max_running=24, dram_arena_gib=16, **long)
        self.prepare(listed, max_running=24, dram_arena_gib=17, **long)
        # The budget: 1 GiB connector pool + 24 GDN carries (3.44 GiB) + the arena: at most 123 GiB.
        self.assertEqual(round(serve.pinned_host_bytes(24, 16, True, L256, 70 * 1024, 1, GIB) / GIB, 2), 74.44)
        _, config, _, _ = self.prepare(listed, max_running=24, dram_arena_gib=123, **long)
        self.assertEqual(config['dram_arena_pages'], 123 * 1024)
        with self.assertRaisesRegex(ValueError, r'or a smaller --dram-arena-gib \(at most 123 GiB\)'):
            self.prepare(listed, max_running=24, dram_arena_gib=124, **long)
        # 32 sessions: at most 122 GiB (4.59 GiB of carries).
        self.prepare(listed, 'full', 32, max_running=CAP, dram_arena_gib=122, **long)
        with self.assertRaisesRegex(ValueError, r'at most 122 GiB'):
            self.prepare(listed, 'full', 32, max_running=CAP, dram_arena_gib=123, **long)

    def test_admission_overlap_up_to_k(self):
        listed = tasks(64)
        for K in (20, 24, CAP):
            _, config, _, _ = self.prepare(listed, max_running=K, connector_pool_gib=1, dram_arena_gib=70,
                                           admission_overlap=K)
            admission, lease = AF.timeouts(K)
            self.assertEqual(config['admission'], dict(admission_overlap=dict(
                permits=K, admission_seconds=admission, lease_seconds=lease)))
            self.assertEqual(self.load(config)['admission'], config['admission'])
        with self.assertRaisesRegex(ValueError, '--admission-overlap 25 needs at least 25 sessions running together '
                                                r'\(max_num_seqs is 24\)'):
            self.prepare(listed, max_running=24, connector_pool_gib=1, admission_overlap=25)
        with self.assertRaisesRegex(ValueError, '--admission-overlap K must be an integer 1-32'):
            self.prepare(listed, max_running=CAP, connector_pool_gib=1, dram_arena_gib=70, admission_overlap=CAP + 1)

    def test_cuda_graphs_and_phase1_graph_tokens(self):
        from remote_ra_live_v2 import phase1_graphs
        listed = tasks(64)
        for K in (20, 24, CAP):
            _, config, argv, _ = self.prepare(listed, max_running=K, connector_pool_gib=1, dram_arena_gib=70,
                                              cuda_graphs='full')
            # vLLM derives the sizes from max_num_seqs: nothing pinned.
            self.assertEqual(config['effective_server_config']['compilation_config'],
                             {'cudagraph_mode': 'FULL_AND_PIECEWISE', 'mode': 'VLLM_COMPILE'})
            _, config, _, _ = self.prepare(listed, max_running=K, connector_pool_gib=1, dram_arena_gib=70,
                                           cuda_graphs='full', phase1_graph_tokens=256)
            base = [1, 2, 4] + list(range(8, 2 * K + 1, 8))
            self.assertEqual(phase1_graphs.base_sizes(K, serve.MAX_BATCHED), base)
            sizes = base + [s for s in (64, 96, 128, 192, 256) if s > base[-1]]
            self.assertEqual(config['phase1_graphs'], dict(capture_sizes=sizes, tokens=256))
            self.assertEqual(config['effective_server_config']['compilation_config']['cudagraph_capture_sizes'],
                             sizes)
            self.load(config)


class GraphSizeTests(unittest.TestCase):
    def test_vllm_capture_sizes_at_20_24_32(self):
        """vLLM 0.24's own sizing code (the overlay copy of VllmConfig._set_cudagraph_sizes, as
        test_int_flags): sizes up to 2K. Its FULL uniform-decode graphs stop at max_num_seqs
        (vllm/v1/cudagraph_dispatcher.py, not in the overlay): K = 24 and 32 are capture sizes, so a
        decode step of up to K rows replays a FULL graph; K = 20 is not, so a step of 17-20 rows is
        padded to 24 and replays the piecewise graph (as 9-12 rows do at K = 12)."""
        import test_int_flags as IF
        for K, sizes in ((20, [1, 2, 4, 8, 16, 24, 32, 40]), (24, [1, 2, 4, 8, 16, 24, 32, 40, 48]),
                         (CAP, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64])):
            for mode in ('FULL_AND_PIECEWISE', 'PIECEWISE', 'FULL_DECODE_ONLY'):
                with self.subTest(K=K, mode=mode):
                    self.assertEqual(IF.capture_sizes(K, mode), (sizes, sizes[-1]))
            full_decode = [s for s in sizes if s <= K]
            self.assertEqual(full_decode[-1] == K, K in (24, CAP))
        self.assertEqual(IF.capture_sizes(16, 'FULL_AND_PIECEWISE'), ([1, 2, 4, 8, 16, 24, 32], 32))


class LayoutTests(unittest.TestCase):
    def test_pools_by_the_formula_and_vllms_planner(self):
        import test_kv_pools as KP
        w = KP.world()
        for K in (17, 20, 24, CAP):
            layout = I0.for_server(16, K, True)
            self.assertEqual((layout.mamba_slots, I0.mamba_pool_blocks(layout, 3)), (2 * K + 1, 1 + 3 * (2 * K + 1)))
            for gib in (24, 25, 32):
                with self.subTest(K=K, gib=gib):
                    _, config = KP.plan(w, layout, gib * GIB)
                    n_fa = KP.fa_pool_blocks(layout, gib * GIB, K)
                    self.assertEqual(config.num_blocks_by_pool, (n_fa, 1 + 3 * (2 * K + 1)))
                    self.assertEqual(n_fa, serve.attention_blocks(gib * GIB, layout))
                    self.assertLessEqual(sum(t.size for t in config.kv_cache_tensors), gib * GIB)
        for K, pools in POOLS_25.items():
            layout = I0.for_server(16, K, True)
            self.assertEqual((serve.attention_blocks(25 * GIB, layout), I0.mamba_pool_blocks(layout, 3)), pools)
        # 196 Mamba blocks of 51,314,688 B at K = 32: 9.37 GiB of the KV budget.
        self.assertEqual(196 * serve.MAMBA_BLOCK_BYTES, 10057678848)

    def test_worker_page_arena(self):
        from remote_ra_live_v2.worker_composition import arena_pages
        for K in (20, 24, CAP):
            layout = I0.for_server(16, K, True)
            self.assertEqual(arena_pages(NS(max_model_len=L256), {}, layout), K * 16384 + K)
            self.assertEqual(arena_pages(NS(max_model_len=65536), {}, layout), K * 4096 + K)
            self.assertEqual(arena_pages(NS(max_model_len=L256), dict(dram_arena_pages=71680), layout), 71680)


class EngineStartupTests(unittest.TestCase):
    def test_engine_startup_accepts_exactly_the_24_session_pool(self):
        import test_decouple_wp2 as WP2
        import test_stageb_engine as E
        from remote_ra_native_growing.adapter import EngineCoreAdapter
        from remote_ra_native_growing.contract import Authority
        layout = I0.for_server(16, 24, True)
        with WP2.matrix_case(layout):
            kv = WP2.KVManager(layout, 2000)
            self.assertEqual(kv.kv_cache_config.num_blocks_by_pool, (2000, 148))
            adapter = EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), kv, page_tokens=16,
                                        full_attention_managers=(kv.fa,), layout=layout)
            for count in range(24):
                adapter.open('x%d' % count, 'full', 'm', 'r')
            self.assertEqual(len(adapter.sessions), 24)
            # A Mamba pool sized for 23 sessions (or the old 16) is refused by a 24-session layout.
            for fewer in (23, 16):
                small = WP2.KVManager(I0.for_server(16, fewer, True), 2000)
                with self.assertRaisesRegex(Exception, 'Mamba pool 1 of 1 \\+ groups x slots blocks'):
                    EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), small, page_tokens=16,
                                      full_attention_managers=(small.fa,), layout=layout)


class CapacityGateTests(unittest.TestCase):
    """The two-pool gate of a 24-session server: 23 requests running at their phase-2 start
    (each may still allocate its row, a running and a crossing GDN slot) beside session s's
    admitting turn. K = 2S + 1 slots: 23 x 2 + 2 never exhaust the Mamba pool."""

    def test_twenty_three_running_requests_and_one_admission(self):
        import test_decouple_wp2 as WP2
        import test_stageb_engine as E
        layout = I0.for_server(16, 24, True)
        with WP2.matrix_case(layout):
            h = WP2.Harness(layout, 'full', fa_blocks=8000)
            WP2.run_script(h, WP2.script_for(16)[:3])
            self.assertEqual((len(h.mamba_pool.blocks), h.mamba_pool.get_num_free_blocks()), (148, 147))
            entries, fa_left = {}, 0
            for k in range(23):
                prompt = 1200 + 97 * k
                request = NS(prompt_token_ids=list(range(prompt)), num_computed_tokens=prompt,
                             sampling_params=NS(max_tokens=1536), output_token_ids=[])
                rid = 'run%d' % k
                h.endpoint.claims[rid] = dict(geometry=NS(phase2_start=prompt))   # before its first step
                entries[rid] = dict(state='accepted', request=request)
                fa_left += pages(prompt + 1536, 16) - pages(prompt, 16)
            lifecycle = NS(entries=entries, endpoint=NS(core=h.core), goals=None)
            self.assertEqual(capacity.headroom_by_pool(lifecycle), (fa_left, 23 * 6))
            request, binding = h.render(E.obs(5))
            h.kv.watermark_blocks = 7
            command = NS(session_id='s', metadata=dict(turn=h.session.turn), core_b64='', source=None)
            h.core._remote_ra_assembly = NS(dispatcher=NS(_decode_core=lambda raw: request))
            with patch('base64.b64decode', lambda raw, validate: b''):
                need = capacity.assess(lifecycle, command)
                self.assertEqual((need['headroom'], need['watermark']), ((fa_left, 138), (7, 0)))
                self.assertEqual(need['phase2'][1], 6)
                self.assertLessEqual(need['phase2'][1] + need['headroom'][1], need['free'][1])
                exact = need['phase2'][0] + need['headroom'][0] + need['watermark'][0]
                hog = h.pool.get_new_blocks(h.pool.get_num_free_blocks() - exact)
                self.assertEqual(capacity.admission_gate(lifecycle, command)['free'], (exact, 147))
                hog += h.pool.get_new_blocks(1)
                with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: FA pool: .* %d free \(FA %d free, '
                                            r'Mamba 147 free\)$' % (exact - 1, exact - 1)):
                    capacity.admission_gate(lifecycle, command)
                h.pool.free_blocks(hog)
            h.endpoint.claims.pop(binding.request)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class BatchedTwentyFourSessionsTests(unittest.TestCase):
    """test_decouple_e2e's multi-session engine loop with 24 tasks (test_ms16's harness).

    The pool is sized as test_ms16's (12 or 400 blocks per session), so no turn waits for the
    attention pool. At 20-24 sessions (not at 16 or 17 in this script) one goal_query turn waits once
    for the Mamba pool: the gate counts a running request that has just crossed an M-token block
    with both slots it still holds (vLLM frees the old one a step later) plus a further crossing,
    one slot more than it can hold at once (the --gdn-device-handoff headroom subtracts what is
    held), so 2S + 1 slots no longer cover every case of the count. The turn retries and nothing
    else changes (the outputs below are those of a 3-session server)."""
    maxDiff = None

    def run_server(self, case, policy, listed, blocks, tight=False):
        """test_ms16's run_server, with every refusal's reason recorded: without a tight pool a
        turn may wait for the Mamba pool only (see above), never for the attention pool."""
        import test_decouple_e2e as DE
        reasons = []
        gate = capacity.admission_gate

        def recorded(lifecycle, command, *args, **kw):
            try:
                return gate(lifecycle, command, *args, **kw)
            except CapacityWait as error:
                reasons.append(str(error))
                raise
        multi = DE.Case(**dict(vars(case), sessions=len(listed), blocks=blocks))
        with ExitStack() as stack:
            DE.enter_case(stack, multi)
            stack.enter_context(patch.object(capacity, 'admission_gate', recorded))
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
            self.assertEqual(h.waits, len(reasons))
            if tight:
                self.assertGreater(h.waits, 0)                   # CAPACITY_WAIT, then admitted
            else:
                self.assertEqual(max(h.widths), len(listed))     # every session in one step
                self.assertTrue(all(r.startswith('CAPACITY_WAIT: Mamba pool:') for r in reasons), reasons)
                self.assertLessEqual(len(reasons), 2)
                if not case.decoupled or policy != 'goal_query':
                    self.assertEqual(reasons, [])
            h.close()
            DE.engine_closed(h.core)
            self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
            self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)
            self.assertTrue(all(block.ref_cnt == 0 for pool in h.kv.block_pools for block in pool.blocks[1:]))
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual(len(cleanup['sessions']), len(listed))
            return {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'])
                                 for i in h.done if i['lane'] is lane] for lane in h.lanes}, h

    def test_twenty_four_sessions_decode_in_one_batch(self):
        import test_decouple_e2e as DE
        listed = list(DE.TASKS) + tasks(24 - len(DE.TASKS))
        # Default suite: full on the shared 784 pool; the slow pass adds goal_query at P = 16 with
        # its tight-pool run (turns wait at the gate, nothing else changes).
        matrix = ((DE.CASES[2], 'goal_query', 24 * 400), (DE.CASES[0], 'full', 24 * 12))
        for case, policy, blocks in matrix if os.environ.get('REMOTE_RA_SLOW_TESTS') == '1' else matrix[1:]:
            with self.subTest(case=case.name, policy=policy):
                many, h = self.run_server(case, policy, listed, blocks)
                self.assertEqual((h.lifecycle.max_sessions, len(h.lanes)), (CAP, 24))
                if policy == 'goal_query':
                    self.assertGreater(h.active, 0)
                three, _ = self.run_server(case, policy, list(DE.TASKS), blocks)
                for k in range(len(DE.TASKS)):
                    self.assertEqual(many[k], three[k])
                if case.decoupled:
                    # test_ms16's 3,200 blocks scaled to 24 sessions (at 3,200 the 24 idle sessions'
                    # pins leave no turn room: every lane waits and the run stalls).
                    tight, h = self.run_server(case, policy, listed, 4800, tight=True)
                    self.assertEqual(tight, many)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PoolOfTwentyFourTests(unittest.TestCase):
    """A session pool of K = 24 over 32 tasks on the batched engine (test_ms_pool's harness):
    at most 24 open and decoding, every task opened in registration order as sessions finish,
    the pools back at their baselines, the sessions a fixed 3-session server serves token-exact."""
    maxDiff = None

    def setUp(self):
        import test_ms_pool as MP
        self.MP = MP

    def enter(self, *args, **kw):
        return self.MP.EnginePoolTests.enter(self, *args, **kw)

    def check_close(self, *args, **kw):
        return self.MP.EnginePoolTests.check_close(self, *args, **kw)

    def run_fixed(self, *args, **kw):
        return self.MP.EnginePoolTests.run_fixed(self, *args, **kw)

    def run_pool(self, *args, **kw):
        return self.MP.EnginePoolTests.run_pool(self, *args, **kw)

    def test_32_tasks_24_running_sessions(self):
        import test_decouple_e2e as DE
        listed = list(DE.TASKS) + tasks(32 - len(DE.TASKS))
        scripts = self.MP.scripts_for(32)
        pool, h = self.run_pool(DE.CASES[0], 'full', listed, 24, 24 * 12, scripts)
        # The adapter's open cap of a pool is max(K, 16) (session_authority.POOL_OPEN_FLOOR).
        self.assertEqual((h.max_open, max(h.widths), h.sessions.limit, h.adapter.max_open), (24, 24, 24, 24))
        fixed = self.run_fixed(DE.CASES[0], 'full', list(DE.TASKS), 24 * 12, scripts[:3])
        for k in range(len(DE.TASKS)):
            self.assertEqual(pool[k], fixed[k])


if __name__ == '__main__':
    unittest.main()
