"""Up to 256 registered tasks per session-pool server (branch integrate-1117). CPU only; nothing here is CUDA evidence.

``remote_ra_live_v2.sessions.MAX_TASKS`` goes from 64 to 256, so that one server (``serve.py
--max-running-sessions 24``) runs the first 150 SWE-bench Verified instances at 24 concurrent sessions. A registered
task holds only metadata (its task record, session id, an empty transcript and generation set); the KV pools, the
Mamba slots and the arena are sized for the K open sessions, and the lifetime request bound (256 per task, a bound,
nothing reserved) grows with the requests made. The kv-session protocol keeps its own cap of 64 slots and a
chat-text server its 1,024. Covered:

- the bounds: task lists, session ids and request bounds up to 256 (257 refused), the other caps unchanged;
- the launcher (serve.py's dry run on the release fixture of test_installer) with 150 tasks at K = 24 in the
  README's 24-session configuration (full and ledger_query; 25 GiB, 262,144 tokens, 4,096-token replies, arena 68
  GiB, connector pool 1 GiB, thinking on, async scheduling, idle release): max_num_seqs 24, 2K + 1 Mamba slots, the
  pinned task list, the startup config check; 257 tasks refused by the launcher and by the config check;
- end to end on the engine (test_ms_pool's pool harness on test_decouple_e2e's batched engine and worker): a pool
  of K = 24 over 150 tasks opens every task in registration order, never more than 24 open or decoding, closes
  each one with exactly its own pins, ends with the pools at their baselines, and serves the three long sessions
  token-exact as a fixed 3-session server does.
"""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import chat_text_protocol as CT
from remote_ra_live_v2 import kv_session_protocol as KV
from remote_ra_live_v2 import sessions

try:
    import torch
except ImportError:
    torch = None

CAP = 256
T150, K24 = 150, 24
L256 = 262144


def tasks(count, start=0):
    return [dict(instance_id='repo__v%03d-%d' % (k, 500 + k),
                 problem_statement='Restore the dropped column %03d of ledger v%03d.' % (k, k))
            for k in range(start, start + count)]


class BoundTests(unittest.TestCase):
    def test_the_caps(self):
        self.assertEqual((sessions.MAX_TASKS, sessions.MAX_SESSIONS, sessions.MAX_CHAT_TEXT_SLOTS,
                          sessions.MAX_CHAT_TEXT_RUNNING), (CAP, 32, 1024, 24))
        self.assertEqual((sessions.task_limit(False), sessions.task_limit(True),
                          sessions.task_limit(True, sessions.CHAT_TEXT_PROTOCOL)), (32, CAP, 1024))
        # The kv-session protocol keeps 64 slots (one MemoryAgentBench allocation opens about 50).
        self.assertEqual((KV.MAX_SLOTS, KV.DEFAULT_SLOTS, CT.MAX_SLOTS), (64, 64, 1024))
        for bad in (65, CAP):
            with self.assertRaises(Exception):
                KV.slot_tasks(bad)

    def test_task_lists_ids_routing_and_request_bounds_up_to_256(self):
        for count in (65, T150, CAP):
            listed = tasks(count)
            self.assertEqual(sessions.validate_tasks(listed, sessions.MAX_TASKS), listed)
        with self.assertRaisesRegex(RuntimeError, r'startup task list of 1\.\.256 tasks required'):
            sessions.validate_tasks(tasks(CAP + 1), sessions.MAX_TASKS)
        # A fixed-session server keeps 32.
        with self.assertRaisesRegex(RuntimeError, r'startup task list of 1\.\.32 tasks required'):
            sessions.validate_tasks(tasks(33))
        ids = sessions.session_ids('http-x', T150, sessions.MAX_TASKS)
        self.assertEqual((ids[0], ids[64], ids[149], len(set(ids))), ('http-x', 'http-x-s64', 'http-x-s149', T150))
        self.assertEqual(sessions.session_index('http-x', T150, 'http-x-s149', sessions.MAX_TASKS), 149)
        with self.assertRaises(RuntimeError):
            sessions.session_index('http-x', T150, 'http-x-s150', sessions.MAX_TASKS)
        with self.assertRaises(RuntimeError):
            sessions.session_ids('http-x', CAP + 1, sessions.MAX_TASKS)
        listed = tasks(T150)
        for k in (0, 9, 10, 99, 100, 149):
            messages = [dict(role='system', content='s'), dict(role='user', content=listed[k]['problem_statement'])]
            self.assertEqual(sessions.route(listed, messages), k)
        # 256 per registered task: 38,400 at 150 (150 x 120 steps = 18,000 requests at most), 65,536 at 256.
        self.assertEqual([sessions.request_limit(n, sessions.MAX_TASKS) for n in (64, T150, CAP)],
                         [16384, 38400, 65536])
        with self.assertRaises(RuntimeError):
            sessions.request_limit(CAP + 1, sessions.MAX_TASKS)
        self.assertLessEqual(T150 * 120, sessions.request_limit(T150, sessions.MAX_TASKS))
        # K stays 1-32 (1-24 on chat-text) and at most the number of tasks.
        self.assertEqual(sessions.running_limit(K24, listed), K24)
        with self.assertRaisesRegex(RuntimeError, 'max_running_sessions must be 1..32'):
            sessions.running_limit(33, listed)


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

    def prepare(self, task, policy='full', selection=None, **kw):
        type(self).count += 1
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.count), 8231,
                                 selection, 'mswea-xml', 25, 16, 'decoupled', **kw)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                return http_config.load()

    # The README's 24-session configuration (the speed defaults spelled out as serve.py's main applies them).
    SERVER = dict(max_running=K24, max_model_len=L256, max_tokens=4096, dram_arena_gib=68, connector_pool_gib=1,
                  cuda_graphs='full', decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True,
                  deferred_completion_io=True, language_model_only=True, enable_thinking=True,
                  async_scheduling=True, release_idle_pages=True)

    def test_150_tasks_at_24_sessions(self):
        listed = tasks(T150)
        for policy, selection in (('full', None), ('ledger_query', dict(active_token_budget=8192))):
            with self.subTest(policy=policy):
                _, config, argv, _ = self.prepare(listed, policy, selection, **self.SERVER)
                effective = config['effective_server_config']
                self.assertEqual((effective['max_num_seqs'], argv[argv.index('--max-num-seqs') + 1]), (K24, '24'))
                self.assertEqual((config['max_running_sessions'], config['tasks'], config['task']),
                                 (K24, listed, listed[0]))
                self.assertEqual(config['kv_layout'], dict(page_tokens=16, mamba_tokens=784, mamba_slots=2 * K24 + 1))
                self.assertEqual(config['dram_arena_pages'], 68 * 1024)
                loaded = self.load(config)
                self.assertEqual((len(loaded['tasks']), http_config.running_sessions(loaded)), (T150, K24))
                if policy == 'ledger_query':
                    self.assertEqual(config['selection_contract']['active_token_budget'], 8192)
        # 256 tasks start; 257 are refused by the launcher and by the config check.
        _, config, _, _ = self.prepare(tasks(CAP), **self.SERVER)
        self.assertEqual(len(self.load(config)['tasks']), CAP)
        with self.assertRaisesRegex(Exception, r'startup task list of 1\.\.256 tasks required'):
            self.prepare(tasks(CAP + 1), **self.SERVER)
        over = dict(config, tasks=tasks(CAP + 1))
        over['tasks_sha256'] = serve.pin(over['tasks'])
        with self.assertRaisesRegex(RuntimeError, r'startup task list of 1\.\.256 tasks required'):
            self.load(over)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PoolOf150Tests(unittest.TestCase):
    """A session pool of K = 24 over 150 tasks on the batched engine (test_ms_pool's harness): at most 24 open
    and decoding, every task opened in registration order as sessions finish (and closed out of order), each
    close returning exactly its pins, the pools back at their baselines, the request bounds sized from the 150
    registered tasks (38,400), and the three long sessions token-exact against a fixed 3-session server."""
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

    def test_150_tasks_24_running_sessions(self):
        import test_decouple_e2e as DE
        listed = list(DE.TASKS) + tasks(T150 - len(DE.TASKS))
        # Lanes 0-2: five turns (compared below); the others 1-2 turns, so that sessions finish at different
        # times while the run stays short.
        scripts = [list(DE.TURNS[:5 if k < 3 else 1 + k % 2]) for k in range(T150)]
        pool, h = self.run_pool(DE.CASES[0], 'full', listed, K24, K24 * 12, scripts)
        self.assertEqual((h.max_open, max(h.widths), h.sessions.limit, h.adapter.max_open), (K24,) * 4)
        self.assertEqual((h.lifecycle.max_requests, len(h.sessions.ids)), (38400, T150))
        fixed = self.run_fixed(DE.CASES[0], 'full', list(DE.TASKS), K24 * 12, scripts[:3])
        for k in range(len(DE.TASKS)):
            self.assertEqual(pool[k], fixed[k])


if __name__ == '__main__':
    unittest.main()
