"""REMOTE_RA_TORCH_PROFILE (serve.py): vLLM's torch profiler as a diagnostic. CPU only; not GPU evidence.

Unset (the default), the launcher's command, environment and startup config are those of a
launcher without the variable; the identity probe covers those bytes. Set to an absolute
directory, the vLLM command gains exactly vLLM's own ``--profiler-config
{"profiler":"torch","torch_profiler_dir":<directory>}`` (which turns on vLLM's
/start_profile and /stop_profile routes), placed before ``--language-model-only`` so that
flag stays last; the startup config, effective_server_config, their pins and the server's
environment are unchanged (the variable itself never reaches the server), and the
frontend's argument check accepts vLLM's parsed profiler_config. A relative or malformed
directory and REMOTE_RA_STEP_PROFILE=1 at the same time are refused.
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

import install
import serve
import test_align_ctx as AC

from remote_ra_http import config as http_config
from remote_ra_live_v2 import model_profile as MP

tasks = AC.tasks
ENV = 'REMOTE_RA_TORCH_PROFILE'
TRACES = '/u/someone/stage/job-output/o16-prof/torch-trace'
CONFIG = ['--profiler-config', '{"profiler":"torch","torch_profiler_dir":"%s"}' % TRACES]
LMO = '--language-model-only'
CASES = AC.CASES + (  # task, policy, GiB, P, layout, selection, extra flags
    (tasks(16), 'full', 32, 16, 'decoupled', None,
     dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, cuda_graphs='full',
          language_model_only=True, max_tokens=4096, max_model_len=131072, dram_arena_gib=40,
          connector_pool_gib=1, max_running=16)),
    (tasks(16), 'observation_query', 24, 16, 'decoupled', dict(active_token_budget=8192),
     dict(decode_fastpath=True, cuda_graphs='full', language_model_only=True)),
    (tasks(1)[0], 'full', 8, 784, None, None, dict(max_model_len=131072)),            # refused: shared pool
    (tasks(16), 'full', 24, 16, None, None, dict(max_model_len=131072)),               # refused: host budget
)


def outside():
    """The caller's environment without any REMOTE_RA_ variable."""
    return {k: v for k, v in os.environ.items() if not k.startswith('REMOTE_RA_')}


class LauncherTests(AC.Release, unittest.TestCase):
    maxDiff = None

    def run_with(self, env, *args, **kwargs):
        with patch.dict(os.environ, dict(outside(), **env), clear=True):
            return self.outcome(*args, **kwargs)

    def test_off_adds_nothing(self):
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                plain = self.run_with({}, task, policy, gib, P, layout, selection, out='off', **extra)
                other = self.run_with(dict(REMOTE_RA_OTHER='x'), task, policy, gib, P, layout, selection,
                                      out='off', **extra)
                self.assertEqual(other, plain)
                if plain[0] == 'ok':
                    manifest, config, command, env = plain[1]
                    self.assertNotIn('--profiler-config', command)
                    self.assertFalse([k for k in env if k.startswith('REMOTE_RA_TORCH')])

    def test_on_inserts_one_vllm_argument_and_nothing_else(self):
        kinds = set()
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                off = self.run_with({}, task, policy, gib, P, layout, selection, out='tp', **extra)
                on = self.run_with({ENV: TRACES}, task, policy, gib, P, layout, selection, out='tp', **extra)
                kinds.add(off[0])
                if off[0] != 'ok':
                    self.assertEqual(on, off)               # the same KV / host refusals
                    continue
                (m0, c0, a0, e0), (m1, c1, a1, e1) = off[1], on[1]
                self.assertEqual((m1, c1, e1), (m0, c0, e0))  # manifest, startup config and pins, environment
                if a0[-1] == LMO:
                    self.assertEqual(a1, a0[:-1] + CONFIG + [LMO])
                else:
                    self.assertNotIn(LMO, a0)
                    self.assertEqual(a1, a0 + CONFIG)
                self.assertEqual(a1.count('--profiler-config'), 1)
                self.assertEqual(json.loads(a1[a1.index('--profiler-config') + 1]),
                                 dict(profiler='torch', torch_profiler_dir=TRACES))
                self.assertNotIn(ENV, e1)
        self.assertEqual(kinds, {'ok', 'ValueError'})

    def test_refusals(self):
        task = tasks(1)[0]
        message = 'REMOTE_RA_TORCH_PROFILE must name an absolute directory for the torch traces'
        for bad in ('', '/', 'relative/traces', ' /x', '/x ', '/x\n', '/x\ty', 'C:/traces'):
            with self.subTest(bad=bad):
                self.assertEqual(self.run_with({ENV: bad}, task, out='bad'), ('ValueError', message))
        stacked = ('ValueError', 'REMOTE_RA_TORCH_PROFILE and REMOTE_RA_STEP_PROFILE=1 do not stack: run one '
                   'profiler at a time')
        self.assertEqual(self.run_with({ENV: TRACES, 'REMOTE_RA_STEP_PROFILE': '1'}, task, out='bad'), stacked)
        self.assertEqual(self.run_with({ENV: TRACES, 'REMOTE_RA_STEP_PROFILE': '1',
                                        'REMOTE_RA_STEP_PROFILE_EVERY': '50'}, task, out='bad'), stacked)
        self.assertEqual(self.run_with({ENV: TRACES, 'REMOTE_RA_STEP_PROFILE': '0'}, task, out='ok')[0], 'ok')
        self.assertEqual(serve.torch_profile_arguments({}), [])
        self.assertEqual(serve.torch_profile_arguments({ENV: TRACES}), CONFIG)

    def test_dry_run_main(self):
        task = self.root / 'dry-task.json'
        task.write_text(json.dumps(tasks(1)[0]))

        def main(env, *extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--output',
                    str(self.root / 'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.dict(os.environ, dict(outside(), **env), clear=True), patch.object(sys, 'argv', argv), \
                    contextlib.redirect_stdout(stream):
                serve.main()
            return json.loads(stream.getvalue())
        for extra in ((), ('--page-tokens', '16', '--kv-cache-gib', '24', '--cuda-graphs', 'full',
                           '--decode-fastpath', '--max-model-len', '131072', LMO)):
            with self.subTest(extra=extra):
                plain, traced = main({}, *extra), main({ENV: TRACES}, *extra)
                command = plain['command']
                expected = command[:-1] + CONFIG + [LMO] if command[-1] == LMO else command + CONFIG
                self.assertEqual(traced, dict(plain, command=expected))
                self.assertNotIn('--profiler-config', command)

    def test_frontend_check_accepts_the_parsed_profiler_config(self):
        on = self.run_with({ENV: TRACES}, tasks(1)[0], out='front', language_model_only=True)
        self.assertEqual(on[0], 'ok')
        config = on[1][1]
        effective = config['effective_server_config']
        profiler = NS(profiler='torch', torch_profiler_dir=TRACES)   # what vLLM's parser makes of the JSON
        with self.startup(config):
            self.assertEqual(http_config.effective_server_config(NS(profiler_config=profiler, **effective)),
                             effective)


class GemmaLauncherTests(unittest.TestCase):
    """The Gemma 4 launcher path: the same argument, before its final --language-model-only."""
    maxDiff = None

    def test_gemma_command(self):
        from test_gemma_install import gemma_fixture
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            root = Path(directory)
            package, site, model, gemma, release = gemma_fixture(root)
            install.install(site, gemma, release, package_root=package, apply=True, model_profile=MP.GEMMA4)
            kw = dict(kv_cache_gib=28, page_tokens=16, max_tokens=4096, max_model_len=131072,
                      language_model_only=True)
            task = dict(instance_id='a', problem_statement='fix a')
            with patch.dict(os.environ, outside(), clear=True):
                off = serve.prepare(release, task, 'full', root / 'off', 8000, None, **kw)
            with patch.dict(os.environ, dict(outside(), **{ENV: TRACES}), clear=True):
                on = serve.prepare(release, task, 'full', root / 'off', 8000, None, **kw)
            with patch.dict(os.environ, dict(outside(), **{ENV: 'traces'}), clear=True):
                with self.assertRaisesRegex(ValueError, 'absolute directory'):
                    serve.prepare(release, task, 'full', root / 'bad', 8000, None, **kw)
        (m0, c0, a0, e0), (m1, c1, a1, e1) = off, on
        self.assertEqual((m1, c1, e1), (m0, c0, e0))
        self.assertEqual(a0[-1], LMO)
        self.assertEqual(a1, a0[:-1] + CONFIG + [LMO])


if __name__ == '__main__':
    unittest.main()
