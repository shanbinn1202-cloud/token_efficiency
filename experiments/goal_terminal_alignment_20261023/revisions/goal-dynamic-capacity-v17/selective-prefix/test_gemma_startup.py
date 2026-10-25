"""Gemma 4 stage 1 (gemma-s1 WP1): the runtime's startup checks of a Gemma server, CPU only.

* ``remote_ra_http.config.load`` accepts the startup config serve.py writes for a Gemma
  release (Gemma's XML grammar, the window KV layout, TRITON_ATTN, --language-model-only,
  the 1 GiB connector pool) and refuses any change of those pins; a Qwen config keeps
  Qwen's grammar.
* ``effective_server_config`` compares vLLM's parsed arguments of a Gemma server: no
  Mamba cache or GDN arguments, the attention backend by name, --block-size and the
  window layout's --additional-config.
* ``validate_wire`` checks a Gemma request like any XML protocol request, and
  ``mswea_protocol.xml_transcripts`` (frontend and engine) builds transcripts of the
  pinned protocol's grammar only.
* serve.py refuses what the Gemma stage 1 does not run (kv-session with thinking on
  (gemma-mab serves it with thinking off: test_gemma_kv_session), chat-text with a --task
  (gemma-tb serves it with slot tasks: test_gemma_chat_text),
  --cuda-graphs values other than vLLM's modes, --decode-fastpath2 alone); goal_query runs
  since gemma-goal (the Goal forward's own sliding transient: test_gemma_goal), CUDA graphs
  and both decode fast paths since gemma-perf (test_gemma_graphs, test_gemma_fastpath).
"""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import install
import serve
from test_gemma_install import gemma_fixture
from remote_ra_http import config as HC
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP

ROOT = Path(__file__).resolve().parent
A = dict(instance_id='a', problem_statement='fix a')


def tasks(n):
    return [dict(instance_id='t%02d' % k, problem_statement='fix check %02d' % k) for k in range(n)]


class Backend:
    """vLLM's parsed --attention-backend: an enum member."""
    name = 'TRITON_ATTN'


class GemmaStartupTests(unittest.TestCase):
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

    def load(self, config):
        """config.load() of a startup config written as serve.py writes it."""
        self.n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                   REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
        with patch.dict(os.environ, env):
            return HC.load()

    def args(self, config, **changes):
        """vLLM's parsed arguments of the server the config pins."""
        effective = dict(config['effective_server_config'], attention_backend=Backend(),
                         mamba_cache_mode='none', mamba_block_size=None, gdn_prefill_backend=None)
        effective.update(changes)
        return NS(**effective)

    def test_load_accepts_the_gemma_startup_config(self):
        _, config, _, _ = self.prepare(tasks(3), 'observation_query', dict(active_token_budget=8192))
        value = self.load(config)
        self.assertEqual(value['protocol'], MP.GEMMA4_PROTOCOL)
        self.assertEqual(HC.model_profile(value), MP.GEMMA4)
        self.assertEqual(HC.kv_layout(value), I0.for_window_server(16, 3, 1024, 8192))
        self.assertEqual(HC.running_sessions(value), 3)
        _, full, _, _ = self.prepare(A, 'random_fill', dict(active_token_budget=8192), page_tokens=32)
        self.assertEqual(HC.kv_layout(self.load(full)), I0.for_window_server(32, 1, 1024, 8192))

    def test_load_refuses_changed_pins(self):
        _, config, _, _ = self.prepare(tasks(2), 'full')

        def refused(pattern, edit):
            value = copy.deepcopy(config)
            edit(value)
            with self.assertRaisesRegex(RuntimeError, pattern):
                self.load(value)
        refused('unknown pinned model profile', lambda v: v.update(model_profile='qwen3.6-27b'))
        refused('pins its XML grammar', lambda v: v.update(protocol='mswea_xml_v1'))
        refused('connector_pool_bytes must be 1-31 GiB', lambda v: v.update(connector_pool_bytes=32 << 30))
        refused('pins its XML grammar', lambda v: v.pop('connector_pool_bytes'))
        refused('pins its XML grammar', lambda v: v['effective_server_config'].update(attention_backend='FLASH_ATTN'))
        # --decode-fastpath (gemma-perf) is pinned as for Qwen: true or absent.
        self.assertIs(HC.decode_fastpath(self.load(dict(config, decode_fastpath=True))), True)
        refused('must be true', lambda v: v.update(decode_fastpath=False))
        # CUDA graphs (gemma-perf): a mode needs its vLLM arguments pinned, as for Qwen.
        refused('differs from the pinned vLLM arguments', lambda v: v.update(cuda_graphs='full'))
        self.assertIs(HC.decode_fastpath2(self.load(dict(config, decode_fastpath=True, decode_fastpath2=True))), True)
        refused('requires decode_fastpath', lambda v: v.update(decode_fastpath2=True))
        refused('takes --admission-overlap only', lambda v: v.update(admission=dict(phase1_in_step=True)))
        # --decode-fastpath3 is Qwen's (fastpath3).
        refused('not in the Gemma 4 stage 1',
                lambda v: v.update(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True))
        # --release-idle-pages is part of the Gemma profile (its pin as for Qwen: true or absent).
        self.assertIs(HC.release_idle_pages(self.load(dict(config, release_idle_pages=True))), True)
        refused('must be true', lambda v: v.update(release_idle_pages=False))
        refused('KV layout differs from the server layout', lambda v: v['kv_layout'].update(window_slots=3))
        refused('pinned KV layout invalid', lambda v: v['kv_layout'].update(window_tokens=1000))
        refused('pinned KV layout required', lambda v: v.update(kv_layout=dict(page_tokens=16, mamba_tokens=784,
                                                                               mamba_slots=5)))
        refused('pinned vLLM arguments', lambda v: v['effective_server_config'].update(block_size=32))
        refused('pinned vLLM arguments', lambda v: v['effective_server_config'].update(mamba_block_size=784))
        refused('pinned vLLM arguments', lambda v: v['effective_server_config'].update(max_num_batched_tokens=4096))
        # A config without the profile key is Qwen's: Gemma's grammar is refused there.
        refused('XML protocol only', lambda v: [v.pop('model_profile')])

    def test_qwen_config_keeps_qwen_grammar(self):
        _, config, _, _ = self.prepare(release=self.qwen, kv_cache_gib=8, page_tokens=784, max_tokens=1536,
                                       max_model_len=65536, language_model_only=False)
        self.assertNotIn('model_profile', config)
        value = self.load(config)
        self.assertIsNone(HC.model_profile(value))
        self.assertEqual(value['protocol'], 'mswea_xml_v1')
        changed = dict(config, protocol=MP.GEMMA4_PROTOCOL)
        with self.assertRaisesRegex(RuntimeError, 'XML protocol only'):
            self.load(changed)

    def test_effective_server_config(self):
        _, config, _, _ = self.prepare(tasks(2), 'full')
        path = self.root / 'startup-effective.json'
        path.write_bytes(serve.wire(config))
        env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                   REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
        with patch.dict(os.environ, env):
            actual = HC.effective_server_config(self.args(config))
            self.assertEqual(actual, config['effective_server_config'])
            self.assertEqual(HC.digest(actual), config['server_config_sha256'])
            # vLLM writing a None --gdn-prefill-backend into additional_config is accepted.
            layout = dict(config['effective_server_config']['additional_config'], gdn_prefill_backend=None)
            HC.effective_server_config(self.args(config, additional_config=layout))
            # A plain string backend compares too.
            HC.effective_server_config(self.args(config, attention_backend='TRITON_ATTN'))
            for changes, pattern in ((dict(mamba_cache_mode='align'), 'no Mamba cache arguments'),
                                     (dict(mamba_block_size=784), 'no Mamba cache arguments'),
                                     (dict(attention_backend='FLASH_ATTN'), 'differ from pinned'),
                                     (dict(block_size=32), 'differ from pinned'),
                                     (dict(max_num_seqs=3), 'differ from pinned'),
                                     (dict(language_model_only=False), 'differ from pinned')):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    HC.effective_server_config(self.args(config, **changes))

    def test_validate_wire_treats_gemma_as_xml(self):
        task = dict(A)
        body = dict(messages=[dict(role='system', content='sys'), dict(role='user', content='please fix a')])
        self.assertEqual(HC.validate_wire(body, task, 0, MP.GEMMA4_PROTOCOL), body['messages'])
        with self.assertRaisesRegex(RuntimeError, 'XML protocol forbids tools'):
            HC.validate_wire(dict(body, tools=[dict(type='function')]), task, 0, MP.GEMMA4_PROTOCOL)
        with self.assertRaisesRegex(RuntimeError, 'unknown agent protocol'):
            HC.validate_wire(body, task, 0, 'mswea_xml_gemma4_v2')

    def test_xml_transcripts_follow_the_pinned_grammar(self):
        from remote_ra_live_v2 import mswea_protocol as X
        renderer = lambda grammar: NS(transcript=lambda: NS(grammar=grammar))
        made = X.xml_transcripts(renderer(X.GEMMA4_XML), MP.GEMMA4_PROTOCOL, 3)
        self.assertEqual([t.grammar for t in made], [X.GEMMA4_XML] * 3)
        self.assertEqual(len(X.xml_transcripts(renderer(X.QWEN_XML), X.PROTOCOL, 2)), 2)
        # A renderer without a grammar attribute renders Qwen's (the test fakes).
        self.assertEqual(len(X.xml_transcripts(NS(transcript=lambda: NS()), X.PROTOCOL, 1)), 1)
        self.assertEqual(X.xml_transcripts(renderer(X.QWEN_XML), None, 2), ())
        self.assertEqual(X.xml_transcripts(renderer(X.QWEN_XML), 'openai_tool_calls_v1', 2), ())
        for grammar, protocol in ((X.QWEN_XML, MP.GEMMA4_PROTOCOL), (X.GEMMA4_XML, X.PROTOCOL)):
            with self.assertRaisesRegex(Exception, 'turn grammar differs from the pinned protocol'):
                X.xml_transcripts(renderer(grammar), protocol, 1)

    def test_connector_pool(self):
        # --connector-pool-gib applies to Gemma; left out, the Gemma pool is 1 GiB.
        for gib, expected in ((None, 1), (4, 4), (1, 1)):
            kw = {} if gib is None else dict(connector_pool_gib=gib)
            _, config, command, _ = self.prepare(tasks(2), 'full', **kw)
            self.assertEqual(config['connector_pool_bytes'], expected << 30)
            transfer = json.loads(command[command.index('--kv-transfer-config') + 1])
            self.assertEqual(transfer['kv_connector_extra_config'], {'cpu_bytes_to_use': expected << 30})
            self.assertEqual(HC.connector_pool(self.load(config)), expected << 30)
        # The host budget counts the pool: 7 sessions at 131072 without an arena with 1 GiB, 6 with 16 GiB.
        self.prepare(tasks(7), 'full')
        with self.assertRaisesRegex(ValueError, 'the 16 GiB connector pool'):
            self.prepare(tasks(7), 'full', connector_pool_gib=16)
        self.prepare(tasks(6), 'full', connector_pool_gib=16)

    def test_launcher_refuses_what_stage_one_does_not_run(self):
        for kw, pattern in ((dict(decode_fastpath2=True), 'requires --decode-fastpath'),
                            (dict(decode_fastpath=True, decode_fastpath2=1), 'decode-fastpath2 is a switch'),
                            (dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True),
                             'decode-fastpath3 is not available for the Gemma 4 profile'),
                            (dict(release_idle_pages=1), 'release-idle-pages is a switch'),
                            (dict(protocol='kv-session', max_running=1), 'kv-session-thinking off only'),
                            (dict(protocol='chat-text', max_running=1), 'chat-text takes no --task'),
                            (dict(connector_pool_gib=32), 'connector-pool-gib must be an integer 1-31'),
                            (dict(cuda_graphs='eager'), 'cuda-graphs must be one of')):
            with self.assertRaisesRegex(ValueError, pattern):
                self.prepare(**kw)
        # goal_query runs since gemma-goal (test_gemma_goal).


if __name__ == '__main__':
    unittest.main()
