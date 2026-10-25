"""Gemma 4 stage 1 (gemma-s1 WP1, WP4): launcher and capacity, CPU only.

* serve.py on a Gemma release (dry runs, and --start up to the execve of vLLM): the vLLM command (TRITON_ATTN, --language-model-only, the window
  layout at --block-size 16, 8,192-token forwards, Gemma's parsers and content format,
  no Mamba or Gated-DeltaNet flag), the startup config and its refusals.
* Capacity: pool 1 = one window slot per session plus one forward's transient, pool 0
  must hold one session's worst turn at the context limit (13 sessions fit 28 GiB at
  131,072 tokens with 4,096-token replies, 14 do not); with --release-idle-pages and a
  fixed --active-token-budget the selective rule instead (one selective turn and the
  other sessions' tails: 16 sessions at 131,072 tokens from 22.13 GiB, so at 24 GiB
  and at the stock 26.41 GiB; 18 at 24 GiB, 23 at 28 GiB); the host budget of a
  --mem=113560M job (pinned + 30 GiB with a 1 GiB connector pool: 7 sessions without
  an arena cap at 131,072, --dram-arena-gib 79 at most); the sliding pool's per-turn
  needs and headroom.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import install
import serve
import test_gemma_kv_pools as GK
from test_gemma_install import gemma_fixture
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import window_capacity as WC
from remote_ra_live_v2.contract import LiveV2Error

ROOT = Path(__file__).resolve().parent
GiB = 1024 ** 3
G = MP.GEMMA4_GEOMETRY
A = dict(instance_id='a', problem_statement='fix a')


def tasks(n):
    return [dict(instance_id='t%02d' % k, problem_statement='fix check %02d' % k) for k in range(n)]


class LauncherTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, self.release = gemma_fixture(self.root)
            install.install(site, gemma, self.release, package_root=package, apply=True, model_profile=MP.GEMMA4)
            self.qwen = self.root / 'qwen-release'
            install.install(site, model, self.qwen, package_root=package, apply=True)
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))
        self.n = 0

    def prepare(self, task=A, policy='full', selection=None, **kw):
        kw = dict(dict(kv_cache_gib=28, page_tokens=16, max_tokens=4096, max_model_len=131072,
                       language_model_only=True), **kw)
        self.n += 1
        release = kw.pop('release', self.release)
        return serve.prepare(release, task, policy, self.root / ('out%d' % self.n), 8000, selection, **kw)

    def refused(self, pattern, **kw):
        with self.assertRaisesRegex(ValueError, pattern):
            self.prepare(**kw)

    def test_vllm_command(self):
        manifest, config, command, env = self.prepare(tasks(4), 'observation_query', dict(active_token_budget=8192))
        argv = command[command.index('--model'):]
        value = lambda flag: argv[argv.index(flag) + 1]
        self.assertEqual(value('--attention-backend'), 'TRITON_ATTN')
        self.assertEqual(value('--block-size'), '16')
        self.assertEqual(value('--max-num-batched-tokens'), '8192')
        self.assertEqual(value('--max-model-len'), '131072')
        self.assertEqual(value('--max-num-seqs'), '4')
        self.assertEqual(value('--kv-cache-memory-bytes'), str(28 * GiB))
        self.assertEqual(value('--chat-template-content-format'), 'openai')
        self.assertEqual((value('--reasoning-parser'), value('--tool-call-parser')), ('gemma4', 'gemma4'))
        self.assertEqual(json.loads(value('--additional-config')), {I0.LAYOUT_KEY: dict(
            page_tokens=16, state='sliding', window_tokens=1024, window_slots=4, forward_tokens=8192)})
        self.assertEqual(argv[-1], '--language-model-only')
        self.assertIn('--enforce-eager', argv)
        for absent in ('--mamba-cache-mode', '--mamba-block-size', '--gdn-prefill-backend', 'qwen3', 'qwen3_xml'):
            self.assertNotIn(absent, argv)
        # The idle connector pool is 1 GiB (the host-pool line's --connector-pool-gib 1).
        transfer = json.loads(value('--kv-transfer-config'))
        self.assertEqual(transfer['kv_connector_extra_config'], {'cpu_bytes_to_use': GiB})
        self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
        # The same additional_config parses back to the window layout.
        layout = I0.from_vllm_config(type('VC', (), dict(additional_config=json.loads(value('--additional-config')))))
        self.assertEqual(layout, I0.for_window_server(16, 4, 1024, 8192))

    def test_startup_config(self):
        _, config, _, _ = self.prepare(tasks(3), 'observation_query', dict(active_token_budget=8192))
        self.assertEqual(config['model_profile'], MP.GEMMA4)
        self.assertEqual(config['protocol'], 'mswea_xml_gemma4_v1')
        self.assertEqual(config['protocol'], MP.GEMMA4_PROTOCOL)
        self.assertEqual(config['kv_layout'], dict(page_tokens=16, state='sliding', window_tokens=1024,
                                                   window_slots=3, forward_tokens=8192))
        self.assertIs(config['language_model_only'], True)
        self.assertEqual(config['connector_pool_bytes'], GiB)
        self.assertEqual(config['max_tokens'], 4096)
        effective = config['effective_server_config']
        self.assertEqual(effective['attention_backend'], 'TRITON_ATTN')
        self.assertEqual(effective['max_num_batched_tokens'], 8192)
        self.assertEqual(effective['chat_template_content_format'], 'openai')
        self.assertNotIn('mamba_cache_mode', effective)
        self.assertEqual(config['server_config_sha256'], serve.pin(effective))
        contract = config['selection_contract']
        self.assertEqual((contract['page_tokens'], contract['active_token_budget'], contract['recent_quota_pages'],
                          contract['vector_format']), (16, 8192, 64, 'compact_int8_v1'))
        self.assertEqual(config['goal_rewriter']['used'], False)
        _, full, _, _ = self.prepare(A, 'full')
        self.assertIsNone(full['selection_contract'])
        _, p32, _, _ = self.prepare(A, 'random_fill', dict(active_token_budget=8192), page_tokens=32)
        self.assertEqual((p32['selection_contract']['page_tokens'], p32['selection_contract']['recent_quota_pages']),
                         (32, 32))
        self.assertEqual(p32['kv_layout']['page_tokens'], 32)

    def test_refusals(self):
        self.refused('requires --language-model-only', language_model_only=False)
        self.refused('Gated-DeltaNet', gdn_device_handoff=True)
        self.refused('phase1-in-step', phase1_in_step=True)
        # --admission-overlap runs since gemma-goal (test_gemma_overlap).
        self.refused('--page-tokens 16', page_tokens=784)
        self.refused('--page-tokens 16', page_tokens=64)
        self.refused('kv-cache-gib 24, 28 or 32', kv_cache_gib=16)
        # 262144 (gemma-mab): the strict rule needs one session at the limit in pool 0.
        self.refused('one session at the 262144-token limit needs', max_model_len=262144)
        self.refused('decoupled only', layout='shared')
        self.refused('selection options apply', selection=dict(active_token_budget=8192))
        # The Qwen launcher keeps its choices: 28 GiB is the Gemma profile's only.
        with self.assertRaisesRegex(ValueError, 'must be one of 8, 16, 24, 32'):
            self.prepare(release=self.qwen, kv_cache_gib=28, language_model_only=False, page_tokens=16,
                         max_tokens=1536, max_model_len=65536)

    def test_gpu_session_limit(self):
        # 28 GiB, P = 16, 131,072 tokens, 4,096-token replies: pool 1 at 13 windows is 6,786
        # block ids (16.57 GiB), pool 0 then holds 9,365 Pages against the 9,055 one turn
        # needs; 14 windows leave 8,715 < 9,049.
        self.prepare(tasks(13), dram_arena_gib=64)
        with self.assertRaisesRegex(ValueError, 'use at most 13 sessions'):
            self.prepare(tasks(14), dram_arena_gib=64)
        self.assertEqual(I0.window_pool_blocks(I0.for_window_server(16, 13, 1024, 8192), 5), 6786)
        common = dict(page_tokens=16, window_tokens=1024, forward_tokens=8192, groups=5,
                      block_bytes=G.sliding_block_bytes(16), page_bytes=G.page_bytes(16))
        table = {(gib, L, reply): WC.max_sessions(gib * GiB, max_model_len=L, max_tokens=reply, **common)
                 for gib in (24, 28, 32) for L in (65536, 131072) for reply in (1536, 4096)}
        self.assertEqual(table[(28, 131072, 4096)], 13)
        self.assertEqual(table[(24, 131072, 4096)], 8)
        self.assertEqual(table[(32, 131072, 4096)], 18)
        self.assertGreater(table[(28, 65536, 4096)], table[(28, 131072, 4096)])
        for (gib, L, reply), most in table.items():
            layout = I0.for_window_server(16, most, 1024, 8192)
            blocks = WC.pool_zero_blocks(gib * GiB, layout, 5, G.sliding_block_bytes(16), G.page_bytes(16))
            self.assertGreaterEqual(blocks - 1, WC.one_session_pages(layout, L, reply, blocks))
            # vLLM's own startup check (one request at max_model_len in pool 0 plus pool 1) passes too.
            self.assertLessEqual(-(-L // 16) * G.page_bytes(16) + I0.window_pool_blocks(layout, 5)
                                 * G.sliding_block_bytes(16), gib * GiB)

    def test_selective_rule_for_sixteen_sessions(self):
        # 16 sessions at 131,072 tokens, an 8,192-token budget, 4,096-token replies, P = 16:
        # pool 1 is 7,761 block ids (18.95 GiB); one selective turn is 2 x (512 + 769) + 1
        # = 2,563 Pages, plus 15 tails and the watermark: 22.13 GiB in all.
        sixteen = dict(task=tasks(16), policy='observation_query', selection=dict(active_token_budget=8192),
                       dram_arena_gib=79)
        _, config, command, _ = self.prepare(release_idle_pages=True, **dict(sixteen, kv_cache_gib=24))
        self.assertIs(config['release_idle_pages'], True)
        self.assertEqual(command[command.index('--max-num-seqs') + 1], '16')
        self.prepare(release_idle_pages=True, page_tokens=32, **dict(sixteen, policy='random_fill'))
        # Without the switch, with full or random50 or with --active-ratio the strict rule holds.
        with self.assertRaisesRegex(ValueError, 'one session at the 131072-token limit needs 9003: use at most '
                                                '8 sessions.*sizes it by one selective turn instead'):
            self.prepare(**dict(sixteen, kv_cache_gib=24))
        with self.assertRaisesRegex(ValueError, 'one session at the 131072-token limit'):
            self.prepare(release_idle_pages=True, **dict(sixteen, policy='full', selection=None))
        with self.assertRaisesRegex(ValueError, 'one session at the 131072-token limit'):
            self.prepare(release_idle_pages=True, **dict(sixteen, selection=dict(active_ratio=.4)))
        layout = I0.for_window_server(16, 16, 1024, 8192)
        common = dict(page_tokens=16, window_tokens=1024, forward_tokens=8192, groups=5,
                      block_bytes=G.sliding_block_bytes(16), page_bytes=G.page_bytes(16), max_model_len=131072,
                      max_tokens=4096)
        # A pool below the rule (20 GiB, not a launcher choice) names the sessions that fit.
        most = WC.max_sessions(20 * GiB, active_tokens=8192, **common)
        self.assertLess(most, 16)
        with patch.object(serve, 'GEMMA4_KV_CACHE_GIB', (20, 24, 28, 32)):
            with self.assertRaisesRegex(ValueError, 'one selective turn of a 8192-token selection and 15 '
                                                    'other tails needs 2586: use at most %d sessions' % most):
                self.prepare(release_idle_pages=True, **dict(sixteen, kv_cache_gib=20))
        self.assertEqual(I0.window_pool_blocks(layout, 5), 7761)
        self.assertEqual((WC.selective_turn_pages(layout, 8192, 4096), WC.selective_turn_pages(layout, 8192, 1536)),
                         (2563, 2243))
        self.assertEqual({gib: WC.max_sessions(gib * GiB, active_tokens=8192, **common) for gib in (24, 28, 32)},
                         {24: 18, 28: 23, 32: 28})
        self.assertEqual(WC.max_sessions(int(26.41 * GiB), active_tokens=8192, **common), 21)
        smallest = lambda kv: WC.pool_zero_blocks(kv, layout, 5, G.sliding_block_bytes(16), G.page_bytes(16))
        enough = lambda kv: smallest(kv) - 1 >= WC.selective_pages(layout, 8192, 4096, smallest(kv))
        self.assertTrue(enough(int(22.13 * GiB)))
        self.assertFalse(enough(int(22.12 * GiB)))

    def test_host_budget_of_a_one_gpu_job(self):
        # --mem=113560M less 30 GiB: 80.9 GiB pinned. At 131,072 tokens one session's arena
        # is 8,193 Pages of 1.25 MiB (10 GiB): 7 sessions fit with the 1 GiB connector pool.
        self.assertAlmostEqual(WC.PINNED_BUDGET_BYTES / GiB, 80.9, places=1)
        self.assertEqual(WC.CONNECTOR_BYTES, GiB)
        self.prepare(tasks(7))
        with self.assertRaisesRegex(ValueError, 'the 1 GiB connector pool.*dram-arena-gib 79 or less'):
            self.prepare(tasks(8))
        self.prepare(tasks(12), dram_arena_gib=79)
        with self.assertRaisesRegex(ValueError, 'dram-arena-gib 79 or less'):
            self.prepare(tasks(12), dram_arena_gib=80)
        self.assertEqual(WC.largest_arena_gib(16, G.page_bytes(16), 131072), 79)
        _, config, _, _ = self.prepare(tasks(12), dram_arena_gib=79)
        self.assertEqual(config['dram_arena_pages'], 79 * GiB // G.page_bytes(16))
        pinned = WC.pinned_host_bytes(12, 16, G.page_bytes(16), 131072, config['dram_arena_pages'])
        self.assertTrue(WC.host_fits(pinned))
        self.assertLessEqual(pinned + 30 * GiB, 113560 * 1024 ** 2)

    def test_dry_run_main_and_start_refused(self):
        task = self.root / 'task.json'
        task.write_text(json.dumps(A))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--policy', 'full',
                    '--output', str(self.root / ('main%d' % len(extra))), '--page-tokens', '16',
                    '--kv-cache-gib', '28', '--max-model-len', '131072', '--max-tokens', '4096',
                    '--language-model-only', *extra]
            out = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(out), \
                    patch.object(os, 'execve', side_effect=AssertionError('execve')):
                serve.main()
            return out.getvalue()
        dry = json.loads(main())
        self.assertEqual(dry['status'], 'SERVICE_DRY_RUN')
        self.assertIn('--attention-backend', dry['command'])
        # --start writes the server's files and replaces the launcher with vLLM (execve).
        with patch.object(sys, 'platform', 'linux'), self.assertRaisesRegex(AssertionError, 'execve'):
            main('--start')
        out = self.root / 'main1'
        config = json.loads((out / 'startup.json').read_bytes())
        self.assertEqual((config['model_profile'], config['protocol']), (MP.GEMMA4, MP.GEMMA4_PROTOCOL))
        launch = json.loads((out / 'launch.json').read_bytes())
        self.assertEqual(launch['command'][launch['command'].index('--attention-backend') + 1], 'TRITON_ATTN')
        template = out / 'chat_template.jinja'
        self.assertTrue(template.is_file() and template.stat().st_size > 0)
        with patch.object(serve, 'GEMMA4_SERVABLE', False),                 self.assertRaisesRegex(ValueError, 'switched to dry runs only'):
            main('--start', '--port', '8001')
        self.assertFalse((self.root / 'main3').exists())


class SlidingNeedTests(unittest.TestCase):
    layout = I0.for_window_server(16, 12, 1024, 8192)

    def test_cold_turns(self):
        self.assertEqual(WC.sliding_turn_need(self.layout, 5, cold=True, prompt_tokens=300), (0, 5 * 19))
        # A long cold prefill holds a slot plus one forward's transient per group:
        # 5 x 577 = 2,885 block ids, stock vLLM's own per-request bound.
        self.assertEqual(WC.sliding_turn_need(self.layout, 5, cold=True, prompt_tokens=60000), (0, 2885))
        self.assertEqual(WC.cold_peak(self.layout, 60000), 65 + 512)

    def test_continuing_turns(self):
        # 4,300-token observation: 269 fresh blocks per group; phase 2 adopts a full window.
        self.assertEqual(WC.sliding_turn_need(self.layout, 5, cold=False, C=50000, e=54300), (5 * 269, 0))
        # 300 tokens: 19 fresh blocks; the adopted window is full (65 blocks).
        self.assertEqual(WC.sliding_turn_need(self.layout, 5, cold=False, C=50000, e=50300), (5 * 19, 0))
        # Early in a session the window is not full yet: phase 2 may grow it to 65.
        self.assertEqual(WC.sliding_turn_need(self.layout, 5, cold=False, C=100, e=400), (5 * 18, 5 * 40))
        # A goal_query Goal forward of 300 tokens is a separate cold forward, in the Goal
        # transient of a goal_query layout (test_gemma_goal).
        goal = I0.for_window_server(16, 12, 1024, 8192, MP.GEMMA4_GOAL_TOKENS)
        self.assertEqual(WC.sliding_turn_need(goal, 5, cold=False, C=50000, e=50300, goal_tokens=300),
                         (5 * 38, 0))
        # A run longer than one native forward runs in 4,096-token chunks that reuse the fresh
        # blocks which left the window (gemma-goal): at most ceil((1,023 + 4,096) / 16) + 1.
        self.assertEqual(WC.sliding_turn_need(self.layout, 5, cold=False, C=100, e=100 + 8193)[0], 5 * 321)
        with self.assertRaises(LiveV2Error):
            WC.sliding_turn_need(I0.for_server(16, 1), 5, cold=True)

    def test_headroom(self):
        self.assertEqual(WC.sliding_headroom(self.layout, 5, [65, 30, 0]), 5 * (0 + 35 + 65))
        with self.assertRaises(LiveV2Error):
            WC.sliding_headroom(self.layout, 5, [-1])

    def test_cold_peak_against_vllm_managers(self):
        # A cold prefill in native forwards of at most 8,192 tokens on the overlay KV
        # manager never holds more than cold_peak blocks per sliding group, and reaches
        # it when a forward starts one token past a block boundary.
        w = GK.gemma_world()
        for chunks in ((300,), (5000,), (8192, 8192, 3616), (1041, 8192, 8192)):
            manager = GK.KP.kv_manager(w, GK.small_gemma_config(w, 16, 4000, 4000), hash_block_size=16,
                                       scheduler_block_size=16)
            tokens = sum(chunks)
            req = GK.KP.request('c', tokens)
            peak = 0
            for chunk in chunks:
                GK.KP.alloc(manager, req, chunk)
                live = [sum(1 for b in row if not b.is_null) for row in manager.coordinator.get_blocks('c')[1:]]
                peak = max(peak, max(live))
            bound = WC.cold_peak(self.layout, tokens)
            self.assertLessEqual(peak, bound)
            self.assertEqual(peak, bound if chunks[0] in (300, 5000, 1041) else bound - 1)


class GateTests(unittest.TestCase):
    """capacity.turn_need_by_pool / headroom_by_pool / assess on the window layout:
    the FA figures of the Qwen rules, the sliding ones from window_capacity."""
    layout = I0.for_window_server(16, 12, 1024, 8192)
    SLIDING = (1, 2, 3, 4, 5)

    def core(self, sessions=None, transcripts=None, claims=None, rows=None, windows=None):
        """``windows``: namespace -> the pinned window {group: [(index, block)]} (WindowPins.window)."""
        from types import SimpleNamespace as NS
        manager = NS(req_to_blocks=rows or {})
        managers = [NS(req_to_blocks={})] + [manager] * 5
        pins = NS(window=lambda namespace: (windows or {}).get(namespace))
        adapter = NS(window=pins, layout=self.layout, sessions=sessions or {}, page_tokens=16,
                     decoupled=True, mamba_tokens=16,
                     kv_manager=NS(coordinator=NS(single_type_managers=managers)))
        endpoint = NS(phase1=NS(groups=lambda: ((0,), self.SLIDING, 6), max_tokens=8192, handoff=None),
                      transcripts=transcripts or {}, selection=NS(contract=None), claims=claims or {})
        return NS(_remote_ra_growing_adapter=adapter, _remote_ra_private_v2_endpoint=endpoint)

    @staticmethod
    def session(C, local_turn=3, policy='full', resident=()):
        from types import SimpleNamespace as NS
        Q = C // 16 * 16
        return NS(local_turn=local_turn, computed_end=C, sealed_end=Q, tail=dict(page=Q // 16) if C > Q else None,
                  resident=tuple(resident), policy=policy, namespace='ns-s')

    @staticmethod
    def transcript(extends=True, turn=3, header=7):
        from types import SimpleNamespace as NS
        return NS(turn=turn, extends=lambda messages: extends, header=(0,) * header, initial_length=3000)

    def test_cold_turns(self):
        from remote_ra_live_v2 import capacity as CAP
        core = self.core()
        # A session's first turn of 3,000 tokens with 4,096-token replies: FA as for Qwen
        # (444 Pages), sliding the cold prefill's 188 blocks per group.
        (fa1, s1), (fa2, s2) = CAP.turn_need_by_pool(core, 's', [1] * 3000, None, 4096)
        self.assertEqual(((fa1, s1), (fa2, s2)), ((0, 0), (444, 5 * 188)))
        # A short prompt grows to a slot while it decodes.
        self.assertEqual(CAP.turn_need_by_pool(core, 's', [1] * 300, None, 4096)[1], (275, 5 * 65))
        # A rebuild (the request does not extend the transcript) is cold too.
        core = self.core({'s': self.session(50000)}, {'s': self.transcript(extends=False)})
        self.assertEqual(CAP.turn_need_by_pool(core, 's', [1] * 60000, None, 4096)[1][1], 5 * 577)
        # Its render frees the session's pinned window first: the rebuild needs those blocks again,
        # so they are credited (a session holding a full window: 65 blocks per group).
        window = {g: [(i, object()) for i in range(65)] for g in self.SLIDING}
        core = self.core({'s': self.session(50000)}, {'s': self.transcript(extends=False)},
                         windows={'ns-s': window})
        self.assertEqual(CAP.turn_need_by_pool(core, 's', [1] * 60000, None, 4096)[1][1], 5 * (577 - 65))
        self.assertEqual(CAP.turn_need_by_pool(core, 's', [1] * 300, None, 4096)[1][1], 5 * (65 - 65))

    def test_continuing_turns(self):
        from remote_ra_live_v2 import capacity as CAP
        core = self.core({'s': self.session(50000, resident=range(100))}, {'s': self.transcript()})
        need = CAP.turn_need_by_pool(core, 's', [1] * 54307, None, 4096)
        fa = CAP._need_by_pool(core, 's', [1] * 54307, None, 4096)
        # Phase 1 [50000, 54300): 269 fresh blocks per group; phase 2 computes the 7-token
        # header on the adopted window (65 blocks): its peak is 66.
        self.assertEqual(need, ((fa[0][0], 5 * 269), (fa[1][0], 5 * 1)))
        self.assertEqual(WC.phase2_peak(self.layout, 7), 66)
        self.assertEqual(WC.phase2_peak(self.layout, 1), 65)
        # A phase 1 longer than one forward runs in chunks (gemma-goal): their ring of 320
        # fresh blocks per group, not the run's 894.
        core = self.core({'s': self.session(40000)}, {'s': self.transcript()})
        self.assertEqual(CAP.turn_need_by_pool(core, 's', [1] * 54307, None, 4096)[0][1], 5 * 320)

    def test_running_headroom(self):
        from types import SimpleNamespace as NS
        from remote_ra_live_v2 import capacity as CAP
        live, null = NS(is_null=False), NS(is_null=True)
        geo = lambda turn, e: NS(local_turn=turn, phase2_start=e if turn else 0, e=e)
        request = lambda n, done: NS(prompt_token_ids=[1] * n, num_computed_tokens=done,
                                     sampling_params=NS(max_tokens=4096))
        entries = dict(r1=dict(state='accepted', request=request(54307, 54400)),     # decoding: a full slot
                       r2=dict(state='accepted', request=request(54307, 54300)),     # before its header step
                       r3=dict(state='accepted', request=request(20000, 8192)),      # a cold prefill
                       r4=dict(state='accepted', request=request(300, 0)),           # a short cold turn
                       r5=dict(state='rendered', request=None))
        claims = dict(r1=dict(geometry=geo(3, 54300)), r2=dict(geometry=geo(3, 54300)),
                      r3=dict(geometry=geo(0, 0)), r4=dict(geometry=geo(0, 0)))
        rows = dict(r1=[null] * 3329 + [live] * 65, r2=[null] * 3329 + [live] * 65,
                    r3=[null] * 100 + [live] * 413)
        core = self.core(claims=claims, rows=rows)
        lifecycle = NS(endpoint=NS(core=core), entries=entries)
        fa, sliding = CAP.headroom_by_pool(lifecycle)
        self.assertEqual(fa, CAP._headroom_by_pool(lifecycle)[0])
        self.assertEqual(sliding, 5 * (0 + 1 + (577 - 413) + 65))
        self.assertEqual(WC.sliding_headroom(self.layout, 5, [65, 30], [66, 65]), 5 * (1 + 35))
        with self.assertRaises(LiveV2Error):
            WC.sliding_headroom(self.layout, 5, [65], [65 + 513])

    def test_two_pool_gate_names_the_sliding_pool(self):
        from remote_ra_live_v2 import capacity as CAP
        need = dict(pools=CAP.WINDOW_POOLS, free=(5000, 20), phase1=(10, 5 * 19), phase2=(30, 5),
                    headroom=(0, 0), watermark=(0, 0))
        with self.assertRaisesRegex(CAP.CapacityWait, 'CAPACITY_WAIT: sliding pool: turn needs 95'):
            CAP._two_pool_gate(need, False)


if __name__ == '__main__':
    unittest.main()
