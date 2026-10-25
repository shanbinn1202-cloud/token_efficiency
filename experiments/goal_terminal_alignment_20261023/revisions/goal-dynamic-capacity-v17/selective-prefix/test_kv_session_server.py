"""kv-session protocol on the server (serve.py --protocol kv-session). CPU only.

- the launcher: a session pool whose registered tasks are the protocol's slots,
  no --task, the pinned kv_session record; refusals; the XML launch unchanged;
- the startup config check (remote_ra_http.config.load) and /v1/kv-session/info.

Nothing here is CUDA evidence.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_installer as INSTALL  # noqa: E402

from remote_ra_http import config as http_config  # noqa: E402
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402


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

    def prepare(self, task=None, policy='goal_query', selection=None, protocol='kv-session', kv_cache_gib=32,
                page_tokens=16, **options):
        type(self).count += 1
        if policy == 'goal_query' and selection is None:
            selection = dict(no_goal_rewriter=True, active_token_budget=32768)
        options.setdefault('max_running', 16)
        options.setdefault('max_tokens', 4096)
        options.setdefault('max_model_len', 131072)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.count), 8231,
                                 selection, protocol, kv_cache_gib, page_tokens, None, **options)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                return http_config.load()

    def test_kv_session_server(self):
        # 16 sessions at 131072 tokens: the pinned host budget needs one Page arena.
        _, config, argv, env = self.prepare(dram_arena_gib=90)
        slots = K.slot_tasks(64)
        self.assertEqual((config['protocol'], config['tasks'], config['task'], config['max_running_sessions']),
                         ('kv_session_v1', slots, slots[0], 16))
        self.assertEqual(config['kv_session'], dict(method='dram_backed_kv', budget=32768, thinking='on',
                                                    header_text='<|im_start|>assistant\n<think>\n', slots=64))
        self.assertEqual((config['max_tokens'], config['effective_server_config']['max_model_len'],
                          config['effective_server_config']['max_num_seqs']), (4096, 131072, 16))
        self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '16')
        self.assertEqual(config['tasks_sha256'], serve.pin(slots))
        loaded = self.load(config)
        self.assertEqual(loaded, config)
        info = K.info(loaded)
        self.assertEqual((info['method'], info['budget'], info['options']['policy'],
                          info['options']['budget_mode'], info['options']['active_token_budget'],
                          info['options']['max_running_sessions'], info['options']['slots'],
                          info['options']['page_tokens']), ('dram_backed_kv', 32768, 'goal_query', 'fixed', 32768,
                                                            16, 64, 16))
        # Other policies and choices: full has no budget; thinking off pins the other header.
        _, full, _, _ = self.prepare(policy='full', max_running=4, kv_session_slots=8, kv_session_thinking='off')
        self.assertEqual(full['kv_session'], dict(method='dram_backed_kv', budget=None, thinking='off',
                                                  header_text='<|im_start|>assistant\n<think>\n\n</think>\n\n',
                                                  slots=8))
        self.assertEqual((full['tasks'], self.load(full)['max_running_sessions']), (K.slot_tasks(8), 4))
        _, obs, _, _ = self.prepare(policy='observation_query', selection={}, max_running=2, kv_session_slots=2)
        self.assertEqual((obs['kv_session']['budget'], K.info(obs)['options']['budget_mode']), (None, 'ratio'))
        _, fill, _, _ = self.prepare(policy='random_fill', selection=dict(active_token_budget=32768, fill_seed=5),
                                     max_running=8)
        self.assertEqual((fill['kv_session']['budget'], K.info(fill)['options']['fill_seed']), (32768, 5))

    def test_262144_token_context(self):
        # The 262144-token profile (MemoryAgentBench's answer turns may pass 131072):
        # 16 sessions, a 90 GiB Page arena, 4096-token replies.
        _, config, argv, _ = self.prepare(max_model_len=262144, dram_arena_gib=90)
        self.assertEqual((argv[argv.index('--max-model-len') + 1], config['effective_server_config']['max_model_len']),
                         ('262144', 262144))
        loaded = self.load(config)
        self.assertEqual(K.info(loaded)['options']['max_model_len'], 262144)
        from remote_ra_http.kv_session_route import frontend
        self.assertEqual(frontend(loaded).max_model_len, 262144)
        # The source cap follows the request caps the 262144-token server pins.
        import runtime_renderer
        from remote_ra_live_v2.context_limits import request_caps
        with patch.object(runtime_renderer, 'MAX_SOURCE_BYTES', request_caps(262144)['source_bytes']):
            self.assertEqual(K.source_bytes(), 2 << 20)
        self.assertEqual(K.source_bytes(), 1 << 20)

    def test_refusals(self):
        task = dict(instance_id='x', problem_statement='y')
        for kwargs, error in ((dict(task=task, max_running=4), 'takes no --task'),
                              (dict(max_running=None), 'needs --max-running'),
                              (dict(kv_session_slots=65, max_running=4), r'1\.\.64'),
                              (dict(kv_session_slots=0, max_running=4), r'1\.\.64'),
                              (dict(kv_session_thinking='maybe', max_running=4), 'on or off'),
                              (dict(max_running=17), 'max_running_sessions must be 1..16'),
                              (dict(max_running=4, kv_session_slots=3), 'at most the number of tasks')):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(Exception, error):
                    self.prepare(**kwargs)
        # kv-session options on the XML protocol.
        for kwargs in (dict(kv_session_slots=8), dict(kv_session_thinking='off')):
            with self.assertRaisesRegex(ValueError, 'apply to --protocol kv-session only'):
                self.prepare(task=task, protocol='mswea-xml', max_running=None, **kwargs)
        # The config check: the record follows the contract, the slots, the pool.
        _, config, _, _ = self.prepare(max_running=2, kv_session_slots=4)
        record = config['kv_session']
        for change in (dict(kv_session=dict(record, budget=16384)), dict(kv_session=dict(record, slots=5)),
                       dict(kv_session=dict(record, header_text='x')), dict(kv_session=None),
                       dict(tasks=K.slot_tasks(5), tasks_sha256=http_config.digest(K.slot_tasks(5))),
                       dict(protocol='openai_tool_calls_v1')):
            with self.subTest(change=sorted(change)):
                with self.assertRaises(RuntimeError):
                    self.load(dict(config, **change))
        no_pool = {k: v for k, v in config.items() if k != 'max_running_sessions'}
        with self.assertRaises(RuntimeError):
            self.load(no_pool)
        # An XML config cannot carry a kv-session record.
        _, xml, _, _ = self.prepare(task=task, protocol='mswea-xml', max_running=None)
        with self.assertRaisesRegex(RuntimeError, 'kv-session record on another protocol'):
            self.load(dict(xml, kv_session=record))

    def test_xml_launch_is_unchanged(self):
        task = dict(instance_id='repo__a-1', problem_statement='Fix the failing check.')
        _, config, argv, env = self.prepare(task=task, protocol='mswea-xml', max_running=None)
        _, again, argv2, env2 = self.prepare(task=task, protocol='mswea-xml', max_running=None,
                                              kv_session_thinking='on', kv_session_slots=None)
        self.assertNotIn('kv_session', config)
        self.assertEqual(config['protocol'], 'mswea_xml_v1')
        swap = lambda value: json.loads(json.dumps(value).replace('out%d' % self.count, 'out%d' % (self.count - 1)))
        self.assertEqual((swap(again), swap(argv2)), (config, argv))

    def test_command_line(self):
        task = self.root / 'task.json'
        task.write_text(json.dumps(dict(instance_id='repo__a-1', problem_statement='Fix it.')))

        def main(*argv):
            out, err = io.StringIO(), io.StringIO()
            with patch.object(sys, 'argv', ['serve.py', '--release-dir', str(self.release), '--policy', 'full',
                                            '--output', str(self.root / ('cli%d' % len(argv)))] + list(argv)), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    serve.main()
                    code = 0
                except SystemExit as exit:
                    code = exit.code
            return code, out.getvalue(), err.getvalue()
        code, out, _ = main('--protocol', 'kv-session', '--max-running-sessions', '4', '--kv-session-slots', '8',
                            '--page-tokens', '16', '--kv-cache-gib', '24')
        self.assertEqual(code, 0)
        dry = json.loads(out)
        self.assertEqual(dry['status'], 'SERVICE_DRY_RUN')
        self.assertEqual(dry['command'][dry['command'].index('--max-num-seqs') + 1], '4')
        # The XML protocol still requires --task, with argparse's own message.
        code, _, err = main()
        self.assertEqual(code, 2)
        self.assertIn('the following arguments are required: --task', err)
        code, _, err = main('--task', str(task), '--kv-session-slots', '8')
        self.assertEqual(code, 2)
        self.assertIn('apply to --protocol kv-session only', err)
        code, out, _ = main('--task', str(task))
        self.assertEqual((code, json.loads(out)['status']), (0, 'SERVICE_DRY_RUN'))

    def test_readme_commands(self):
        # The kv-session launch of README section 6.6, in both languages, as a dry run.
        import re
        import shlex
        commands = []
        for name in ('README.md', 'README.zh-CN.md'):
            text = (ROOT / name).read_text(encoding='utf-8')
            # Qwen's (the Gemma profile's kv-session launch of section 4.14: test_gemma_kv_session).
            found = [b for b in re.findall(r'```sh\n(.*?)```', text, re.S)
                     if '--protocol kv-session' in b and '$RELEASE-gemma4' not in b]
            self.assertEqual(len(found), 1, name)
            commands.append(found[0])
        self.assertEqual(commands[0], commands[1])
        type(self).count += 1
        output = self.root / ('readme%d' % self.count)
        values = {'$RELEASE': str(self.release), '$RUNS/mab-obs': str(output)}
        argv = [values.get(v, v) for v in shlex.split(commands[0].replace('\\\n', ' '))]
        self.assertEqual(argv[:3], ['python', '-B', 'serve.py'])
        self.assertEqual(argv[-1], '--start')
        out = io.StringIO()
        with patch.object(sys, 'argv', ['serve.py'] + argv[3:-1]), contextlib.redirect_stdout(out), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            serve.main()
        dry = json.loads(out.getvalue())
        command = dry['command']
        self.assertEqual(dry['status'], 'SERVICE_DRY_RUN')
        for flag, value in (('--max-num-seqs', '16'), ('--max-model-len', '262144'), ('--block-size', '16')):
            self.assertEqual(command[command.index(flag) + 1], value)


if __name__ == '__main__':
    unittest.main()
