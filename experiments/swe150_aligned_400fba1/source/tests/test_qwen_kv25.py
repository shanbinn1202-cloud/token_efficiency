"""--kv-cache-gib 25 for Qwen3.6-27B (integrate-1026; the group's runs use 25 GiB for both models). CPU only; not GPU
evidence (Qwen at 25 GiB passed on the GPU in jobs 3461661 and 3461703 at cf2b28d, docs/reference.md 4.9).

The Qwen launcher took 8, 16, 24 and 32 GiB and refused 25 (`--kv-cache-gib must be one of 8, 16, 24, 32`); now 25 is a
choice of both profiles. Nothing is keyed by the size: the KV bytes, vLLM's two pools (16 sessions at P = 16: 20,706
attention + 100 Mamba blocks, the planner's own figures), the launcher's refusal of one session's turn at the context
limit, the pinned-host check (the arena), the startup config and its pins, the frontend's argument check and the Goal
rewriter's batch cap (16,384 by default, sized next to the 32-GiB server) all follow it. Checked here at the group's
settings: 16 sessions, 131,072 and 262,144 tokens, 4,096-token replies, observation_query and goal_query at the
8,192-token budget, with and without --release-idle-pages, CUDA graphs full, the three decode fast paths, the 1-GiB
connector pool, --language-model-only and a 70-GiB arena; and the docs/reference.md 4.9 capacity rule's figures at 25 GiB.
"""
import contextlib
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

import serve
import test_align_ctx as AC

from remote_ra_http import config as http_config
from remote_ra_live_v2 import kv_layout as I0

GIB = 1024 ** 3
L128, L256 = 131072, 262144
tasks = AC.tasks
GROUP = dict(cuda_graphs='full', max_running=16, max_tokens=4096, dram_arena_gib=70, connector_pool_gib=1,
             decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, deferred_completion_io=True,
             language_model_only=True)
BUDGET = dict(active_token_budget=8192)


def pages(tokens):
    return -(-tokens // 16)


class QwenKv25(AC.Release, unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')

    def group(self, policy, max_model_len, gib=25, release_idle_pages=False, **extra):
        selection = dict(BUDGET, goal_rewriter_model=str(self.rewriter)) if policy == 'goal_query' else dict(BUDGET)
        flags = dict(GROUP, max_model_len=max_model_len, **extra)
        if release_idle_pages:
            flags['release_idle_pages'] = True
        if policy == 'goal_query':
            flags.update(admission_overlap=16, goal_rewriter_batch=4)
        return self.outcome(tasks(32), policy, gib, 16, 'decoupled', selection, out='kv25', **flags)

    def test_the_choice(self):
        self.assertEqual(serve.KV_CACHE_GIB, (8, 16, 24, 25, 32))
        for bad in (12, 28, 25.0, '25'):
            with self.subTest(bad=bad):
                self.assertEqual(self.outcome(tasks(1)[0], 'full', bad),
                                 ('ValueError', '--kv-cache-gib must be one of 8, 16, 24, 25, 32'))
        ok = self.outcome(tasks(1)[0], 'full', 25)
        self.assertEqual(ok[0], 'ok')
        manifest, config, command, env = ok[1]
        self.assertEqual(config['effective_server_config']['kv_cache_memory_bytes'], 25 * GIB)
        self.assertEqual(command[command.index('--kv-cache-memory-bytes') + 1], str(25 * GIB))
        # the frontend accepts exactly the pinned bytes
        with self.startup(config):
            effective = config['effective_server_config']
            self.assertEqual(http_config.effective_server_config(NS(**effective)), effective)
            with self.assertRaisesRegex(RuntimeError, 'differ from pinned common config'):
                http_config.effective_server_config(NS(**dict(effective, kv_cache_memory_bytes=24 * GIB)))

    def test_group_settings_at_25_gib(self):
        """16 sessions at 131,072 and 262,144 tokens, budget 8,192, with and without --release-idle-pages: every
        launch passes; it differs from the 24-GiB launch in the KV bytes only."""
        for policy in ('observation_query', 'goal_query'):
            for L in (L128, L256):
                for release in (False, True):
                    with self.subTest(policy=policy, L=L, release=release):
                        kind, value = self.group(policy, L, 25, release)
                        self.assertEqual(kind, 'ok', value)
                        manifest, config, command, env = value
                        effective = config['effective_server_config']
                        self.assertEqual((effective['kv_cache_memory_bytes'], effective['max_model_len'],
                                          effective['max_num_seqs'], config.get('release_idle_pages')),
                                         (25 * GIB, L, 16, True if release else None))
                        self.assertEqual(config['dram_arena_pages'], 70 * GIB // (16 * 65536))
                        if policy == 'goal_query':   # the rewriter's batch cap keeps its default at 25 GiB
                            self.assertEqual(config['goal_rewriter_batch_tokens'], serve.GOAL_REWRITER_BATCH_TOKENS)
                        kind24, value24 = self.group(policy, L, 24, release)
                        self.assertEqual(kind24, 'ok')
                        c24 = value24[1]
                        self.assertEqual({k: v for k, v in config.items() if k not in (
                            'effective_server_config', 'server_config_sha256')},
                            {k: v for k, v in c24.items() if k not in ('effective_server_config',
                                                                        'server_config_sha256')})
                        self.assertEqual(dict(c24['effective_server_config'], kv_cache_memory_bytes=25 * GIB),
                                         effective)
                        self.assertEqual([a.replace(str(25 * GIB), str(24 * GIB)) for a in command], value24[2])
                        self.assertEqual(self.load(config)['effective_server_config'], effective)

    def test_pools_and_the_launcher_refusal(self):
        """vLLM's planner gives the pools serve.attention_blocks counts; one session's turn at 262,144 tokens fits 25
        GiB at 1 to 16 sessions (8 and 16 GiB stay refused); a full server of 16 sessions starts too (its turns then
        wait for capacity: the gate, docs/reference.md 4.9)."""
        import test_kv_pools as KP
        w = KP.world()
        for S, attention in ((16, 20706), (20, 19531), (1, 25110)):
            layout = I0.for_server(16, S, True)
            self.assertEqual(serve.attention_blocks(25 * GIB, layout), attention, S)
            _, config = KP.plan(w, layout, 25 * GIB)
            self.assertEqual(config.num_blocks_by_pool, (attention, 1 + 3 * (2 * S + 1)), S)
            self.assertLessEqual(sum(t.size for t in config.kv_cache_tensors), 25 * GIB)
        for S in range(1, 17):
            layout = I0.for_server(16, S, True)
            blocks = serve.attention_blocks(25 * GIB, layout)
            for T in serve.MAX_TOKENS_CHOICES:
                self.assertGreaterEqual(blocks - 1, serve.one_session_blocks(16, blocks, T, L256), (S, T))
        self.assertEqual(serve.one_session_blocks(16, 20706, 4096, L256), 17873)
        kind, value = self.outcome(tasks(32), 'full', 25, 16, 'decoupled', None, **dict(GROUP, max_model_len=L256))
        self.assertEqual(kind, 'ok', value)
        for gib in (8, 16):
            self.assertEqual(self.outcome(tasks(32), 'full', gib, 16, 'decoupled', None,
                                          **dict(GROUP, max_model_len=L256))[0], 'ValueError')

    def test_capacity_rule_at_25_gib(self):
        """docs/reference.md 4.9's rule (S - 1 idle sessions + one admitting turn at the full-context peak + one reply and the
        1 % watermark within the usable attention blocks): the most sessions that fit at 24 / 25 / 32 GiB."""
        def most(gib, L, idle):
            peak = pages(L) + 1 + pages(serve.MAX_BATCHED)
            found = 0
            for n in range(1, 25):
                fa = serve.attention_blocks(gib * GIB, I0.for_server(16, n, True))
                if (n - 1) * idle + peak + pages(4096) + 1 + int(serve.WATERMARK * fa) <= fa - 1:
                    found = n
            return found
        rows = {L256: ((16385, 1, 1, 1), (6554, 1, 2, 3), (6992, 1, 1, 2), (7834, 1, 1, 2), (513, 8, 9, 18),
                       (951, 5, 6, 12), (1793, 3, 4, 7)),
                L128: ((8193, 2, 2, 3), (3277, 5, 5, 7), (3715, 4, 4, 6), (4557, 3, 4, 5), (513, 18, 20, 24),
                       (951, 12, 13, 19), (1793, 7, 8, 11))}
        for L, table in rows.items():
            for idle, at24, at25, at32 in table:
                with self.subTest(L=L, idle=idle):
                    self.assertEqual((most(24, L, idle), most(25, L, idle), most(32, L, idle)), (at24, at25, at32))

    def test_dry_run_main(self):
        task = self.root / 'kv25-tasks.json'
        task.write_text(json.dumps(tasks(16)))
        argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task), '--output', str(self.root / 'dry'),
                '--port', '8231', '--policy', 'observation_query', '--active-token-budget', '8192', '--kv-cache-gib', '25',
                '--page-tokens', '16', '--kv-layout', 'decoupled', '--cuda-graphs', 'full', '--max-running-sessions',
                '16', '--max-tokens', '4096', '--max-model-len', '262144', '--dram-arena-gib', '70',
                '--connector-pool-gib', '1', '--release-idle-pages', '--language-model-only']
        stream = io.StringIO()
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            serve.main()
        value = json.loads(stream.getvalue())
        self.assertEqual(value['status'], 'SERVICE_DRY_RUN')
        command = value['command']
        self.assertEqual(command[command.index('--kv-cache-memory-bytes') + 1], str(25 * GIB))


if __name__ == '__main__':
    unittest.main()
