"""Alignment settings of branch align-ctx. CPU only; nothing here is GPU evidence.

Reply bound (``serve.py --max-tokens {1536,4096}``, remote_ra_live_v2.context_limits):
a server setting, not part of the XML protocol. The default 1536 changes nothing
(launcher outputs equal those without the option, no startup key, no worker record);
4096 adds only startup.json ``max_tokens``. Covered: the launcher (pins, refusals,
the one-session attention-pool refusal with 4096-token replies), the startup config,
the XML protocol's enforcement (route and EngineCore), the capacity gate's reply
reservation, and the startup checks of the frontend (live controller), the engine
(endpoint, worker startup record and its echo) and the worker (runner
max_model_len); end to end, a 4096 server serves the same turns as a 1536 server
when replies stay short, and EngineCore refuses a request with another bound.

Context limit (``serve.py --max-model-len {65536,131072}``, ``--dram-arena-gib``):
the default changes nothing; 131072 changes only vLLM's --max-model-len and
effective_server_config.max_model_len. Covered: the launcher's refusals (attention
pool and shared pool at the limit, the pinned-host budget, the arena cap), the
startup config, the frontend's check of vLLM's arguments, the engine and worker
checks, the worker's arena capacity, the Page catalog bound, and the token and
frame caps on Mini-SWE-Agent conversations of about 127K tokens with the real
tokenizer (skipped unless REMOTE_RA_TOKENIZER_DIR names a directory with the
pinned tokenizer.json and the tokenizers package is importable).
"""
import asyncio
import contextlib
from contextlib import ExitStack
from hashlib import sha256
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
from remote_ra_live_v2 import context_limits as CL
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import mswea_protocol as xml
from remote_ra_live_v2.contract import LiveV2Error
# Imported here, before any test patches sys.modules: a module first imported
# inside test_decouple_e2e's patch.dict(sys.modules) would be dropped again and
# re-imported later as a second module object, which test_cudagraph's
# attach_loaded_worker fakes (patched by module) would then miss.
from remote_ra_live_v2 import loaded_sources, native_probe, vector_space, worker_composition  # noqa: F401

GIB = 1024 ** 3


def tasks(count, start=0):
    return [dict(instance_id='repo__m%02d-%d' % (k, 100 + k),
                 problem_statement='Fix the failing check %02d in module m%02d.' % (k, k))
            for k in range(start, start + count)]


def pages(tokens, page_tokens):
    return -(-tokens // page_tokens)


class Release:
    """The release fixture of test_installer, installed once per test class."""

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

    def prepare(self, task, policy='full', kv_cache_gib=8, page_tokens=784, layout=None, selection=None,
                out=None, **kwargs):
        if out is None:
            type(self).count += 1
            out = 'out%d' % self.count
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / out, 8231, selection, 'mswea-xml',
                                 kv_cache_gib, page_tokens, layout, **kwargs)

    def outcome(self, *args, **kwargs):
        try:
            return ('ok', self.prepare(*args, **kwargs))
        except Exception as error:
            return (type(error).__name__, str(error))

    @contextlib.contextmanager
    def startup(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                yield

    def load(self, config):
        with self.startup(config):
            return http_config.load()


CASES = (  # task, policy, GiB, P, layout, selection, extra flags
    (tasks(1)[0], 'full', 8, 784, None, None, {}),
    (tasks(3), 'goal_query', 24, 16, 'decoupled', dict(no_goal_rewriter=True), {}),
    (tasks(8), 'observation_query', 24, 784, 'decoupled', dict(active_token_budget=8192), {}),
    (tasks(16), 'random_fill', 32, 16, None, None, dict(cuda_graphs='full')),
    (tasks(24), 'full', 24, 16, None, None, dict(max_running=8)),
)


# ---- 1. reply bound (--max-tokens) -------------------------------------------------------------

class ReplyBoundLauncherTests(Release, unittest.TestCase):
    maxDiff = None

    def test_default_is_byte_identical(self):
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                out = 'same%d' % P
                without = self.prepare(task, policy, gib, P, layout, selection, out=out, **extra)
                default = self.prepare(task, policy, gib, P, layout, selection, out=out, max_tokens=1536,
                                       **extra)
                self.assertEqual(default, without)
                self.assertNotIn('max_tokens', without[1])

    def test_4096_adds_only_the_startup_key(self):
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                out = 'pin%d' % P
                m0, c0, a0, e0 = self.prepare(task, policy, gib, P, layout, selection, out=out, **extra)
                m1, c1, a1, e1 = self.prepare(task, policy, gib, P, layout, selection, out=out,
                                              max_tokens=4096, **extra)
                self.assertEqual((m1, a1), (m0, a0))                 # the vLLM command is unchanged
                self.assertEqual(c1, dict(c0, max_tokens=4096))
                self.assertEqual(c1['effective_server_config'], c0['effective_server_config'])
                self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                self.assertEqual(self.load(c1)['max_tokens'], 4096)
                self.assertEqual(CL.record(self.load(c1)), dict(max_model_len=65536, max_tokens=4096))
                self.assertIsNone(CL.record(self.load(c0)))

    def test_other_values_are_refused(self):
        for value in (0, 1024, 2048, 8192, '4096', 4096.0, True):
            with self.subTest(value=value):
                self.assertEqual(self.outcome(tasks(1)[0], max_tokens=value),
                                 ('ValueError', '--max-tokens must be one of 1536, 4096'))

    def test_one_session_refusal_reserves_the_reply(self):
        # 8 GiB at P = 16: nine sessions leave 5,352 usable attention blocks. One session
        # at 64K needs 4,097 + 1,024 + ceil(1536/16) + 1 + 53 = 5,271 with 1536-token
        # replies and 4,097 + 1,024 + 256 + 1 + 53 = 5,431 with 4096-token replies.
        layout = I0.for_server(16, 9, True)
        blocks = serve.attention_blocks(8 * GIB, layout)
        self.assertEqual((blocks - 1, serve.one_session_blocks(16, blocks),
                          serve.one_session_blocks(16, blocks, 4096)), (5352, 5271, 5431))
        self.prepare(tasks(9), kv_cache_gib=8, page_tokens=16)
        with self.assertRaisesRegex(ValueError, r'^--kv-cache-gib 8 with 9 sessions leaves 5352 usable attention '
                                                r'blocks of 16 tokens after the Mamba pool; one session at the '
                                                r'65536-token limit with 4096-token replies needs 5431: '):
            self.prepare(tasks(9), kv_cache_gib=8, page_tokens=16, max_tokens=4096)
        self.prepare(tasks(8), kv_cache_gib=8, page_tokens=16, max_tokens=4096)
        # Every 1-8 session configuration still passes with 4096-token replies.
        for P in (16, 32, 64, 784):
            for gib in serve.KV_CACHE_GIB:
                for count in range(1, 9):
                    layout = I0.for_server(P, count, True)
                    blocks = serve.attention_blocks(gib * GIB, layout)
                    self.assertGreaterEqual(blocks - 1, serve.one_session_blocks(P, blocks, 4096), (P, gib, count))

    def test_dry_run_flag(self):
        task = self.root / 'task.json'
        task.write_text(json.dumps(tasks(1)[0]))
        base = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--policy', 'full']

        def main(*argv):
            out = io.StringIO()
            with patch.object(sys, 'argv', base + list(argv)), contextlib.redirect_stdout(out), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(out.getvalue())
        output = str(self.root / 'dry')
        self.assertEqual(main('--output', output, '--max-tokens', '4096'), main('--output', output))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main('--output', output, '--max-tokens', '2048')


class ReplyBoundConfigTests(Release, unittest.TestCase):
    def test_load_refuses_a_non_canonical_or_unknown_bound(self):
        config = self.prepare(tasks(1)[0], max_tokens=4096)[1]
        self.assertEqual(CL.check_config(self.load(config)), (4096, 65536))
        for value in (1536, 2048, '4096', 4096.0, True, None):
            with self.subTest(value=value):
                bad = dict(config, max_tokens=value)
                with self.assertRaisesRegex(RuntimeError, 'pinned max_tokens must be one of 4096; the default '
                                                          '1536 is pinned by leaving the key out'):
                    self.load(bad)

    def test_record_and_values(self):
        self.assertEqual((CL.max_tokens(None), CL.max_model_len(None), CL.record({})), (1536, 65536, None))
        self.assertEqual(CL.record(dict(max_tokens=4096)), dict(max_model_len=65536, max_tokens=4096))
        self.assertEqual(xml.MAX_TOKENS, CL.DEFAULT_MAX_TOKENS)


class ReplyBoundProtocolTests(unittest.TestCase):
    """The XML protocol enforces the server's bound; it does not define it."""

    def good(self, bound):
        return NS(n=1, temperature=0.0, max_tokens=bound, min_tokens=0, stop=[], ignore_eos=False,
                  stop_token_ids=[2, 3], eos_token_id=2, repetition_penalty=1.0, presence_penalty=0.0,
                  frequency_penalty=0.0, logit_bias=None, allowed_token_ids=None, bad_words=[],
                  structured_outputs=None, logits_processors=None)

    def test_engine_check_sampling(self):
        stops = frozenset({2, 3})
        self.assertTrue(xml.check_sampling(self.good(1536), stops))
        self.assertTrue(xml.check_sampling(self.good(4096), stops, 4096))
        with self.assertRaisesRegex(LiveV2Error, r'differs: max_tokens=1536$'):
            xml.check_sampling(self.good(1536), stops, 4096)
        with self.assertRaisesRegex(LiveV2Error, r'differs: max_tokens=4096$'):
            xml.check_sampling(self.good(4096), stops)
        for bound in (2048, '4096', None):
            with self.assertRaisesRegex(LiveV2Error, 'pinned reply bound required'):
                xml.check_sampling(self.good(4096), stops, bound)

    def test_route_fixes_the_server_bound(self):
        from remote_ra_http.mswea_route import prepare_request

        def request():
            return NS(messages=[], tools=None, stream=False, n=1, max_tokens=99, temperature=0.7,
                      return_token_ids=None, stop_token_ids=[])
        self.assertEqual(prepare_request(request(), frozenset({2, 3})).max_tokens, 1536)
        self.assertEqual(prepare_request(request(), frozenset({2, 3}), 4096).max_tokens, 4096)
        with self.assertRaisesRegex(RuntimeError, 'pinned reply bound required'):
            prepare_request(request(), frozenset({2, 3}), 2048)

    def test_capacity_reads_the_requests_bound(self):
        # The gate (unchanged) reserves each request's max_tokens, floored at 1536; the
        # route sets the pinned bound and check_sampling refuses any other before allocation.
        self.assertEqual(capacity.max_tokens(None), 1536)
        self.assertEqual(capacity.max_tokens(NS(max_tokens=1536)), 1536)
        self.assertEqual(capacity.max_tokens(NS(max_tokens=4096)), 4096)
        from remote_ra_live_v2.private_endpoint import SplitPrivateEndpoint
        self.assertEqual(SplitPrivateEndpoint.max_tokens, 1536)

    def test_headroom_of_running_requests_with_4096(self):
        """test_ms16's two-pool gate with the pinned 4096 bound: every running request
        may still allocate its row to ceil((N + 4096) / P)."""
        import test_decouple_wp2 as WP2
        import test_stageb_engine as E
        layout = I0.for_server(16, 16, True)
        with WP2.matrix_case(layout):
            h = WP2.Harness(layout, 'full', fa_blocks=8000)
            WP2.run_script(h, WP2.script_for(16)[:3])
            request, binding = h.render(E.obs(5))
            tokens = len(request.prompt_token_ids)
            h.core._remote_ra_assembly = NS(dispatcher=NS(_decode_core=lambda raw: request))
            command = NS(session_id='s', metadata=dict(turn=h.session.turn), core_b64='', source=None)
            rows = {}
            for bound in (1536, 4096):
                with self.subTest(bound=bound):
                    h.endpoint.max_tokens = bound
                    request.sampling_params = NS(max_tokens=bound)
                    entries, fa_left = {}, 0
                    for k in range(15):
                        prompt = 1200 + 97 * k
                        running = NS(prompt_token_ids=list(range(prompt)), num_computed_tokens=prompt,
                                     sampling_params=NS(max_tokens=bound), output_token_ids=[])
                        h.endpoint.claims['run%d' % k] = dict(geometry=NS(phase2_start=prompt))
                        entries['run%d' % k] = dict(state='accepted', request=running)
                        fa_left += pages(prompt + bound, 16) - pages(prompt, 16)
                    lifecycle = NS(entries=entries, endpoint=NS(core=h.core), goals=None)
                    self.assertEqual(capacity.headroom_by_pool(lifecycle), (fa_left, 15 * 6))
                    with patch('base64.b64decode', lambda raw, validate: b''):
                        need = capacity.assess(lifecycle, command)
                    self.assertEqual(need['headroom'], (fa_left, 90))
                    rows[bound] = need['phase2'][0]
                    for k in range(15):
                        h.endpoint.claims.pop('run%d' % k)
            # Phase 2's row runs to ceil((N + bound) / P).
            self.assertEqual(rows[4096] - rows[1536], pages(tokens + 4096, 16) - pages(tokens + 1536, 16))
            self.assertGreater(rows[4096], rows[1536])
            h.endpoint.claims.pop(binding.request)


class ReplyBoundStartupTests(Release, unittest.TestCase):
    """Startup checks of the frontend, the engine and the worker; nothing per request."""

    def test_frontend_controller_holds_the_bound(self):
        import test_ms_frontend as F
        from remote_ra_live_v2.composition import compose_frontend
        from remote_ra_live_v2.controller import LiveFourPolicyController
        config = self.prepare(tasks(1)[0], max_tokens=4096)[1]
        for value, bound in ((None, 1536), (config, 4096)):
            with self.subTest(bound=bound), patch.object(CL, 'loaded', lambda: value):
                live = compose_frontend(F.Serving(), F.Serving(), policy='full', instance_id='inst',
                                        readiness=F.READY)
                self.assertEqual(live.max_tokens, bound)
        self.assertEqual(compose_frontend(F.Serving(), F.Serving(), policy='full', instance_id='inst',
                                          readiness=F.READY, max_tokens=4096).max_tokens, 4096)
        for bound in (2048, '4096', None):
            with self.assertRaisesRegex(LiveV2Error, 'reply bound must be one of 1536, 4096'):
                LiveFourPolicyController(F.Serving(), startup_policy='full', instance_id='inst',
                                         readiness=F.READY, max_tokens=bound)

    def startup_record(self, config, echo=None, vllm_len=None):
        """compose_engine's worker startup record; the worker's reply is ``echo``
        (None: the RPC stops) and the engine's vLLM max_model_len ``vllm_len``."""
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        sent = []

        def collective_rpc(method, args, timeout):
            sent.append(args[0])
            if echo is None:
                raise Stop()
            return [echo]
        core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                  _remote_ra_trusted_issue=dict(config['task']), _remote_ra_trusted_issues=[dict(config['task'])],
                  _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                  _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
        with self.startup(config), self.assertRaises(Stop if echo is None else LiveV2Error) as raised:
            composition.compose_engine(core, object())
        return sent.pop(), raised.exception

    def test_engine_sends_the_record_and_checks_the_echo(self):
        default = self.prepare(tasks(1)[0])[1]
        pinned = self.prepare(tasks(1)[0], max_tokens=4096)[1]
        record, _ = self.startup_record(default)
        self.assertEqual(record, dict(enabled=True, source='m' * 64, page_tokens=784, loaded_sources_verified=True))
        record, _ = self.startup_record(pinned)
        self.assertEqual(record['context'], dict(max_model_len=65536, max_tokens=4096))
        receipt = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True)
        for echo in (receipt, dict(receipt, context=dict(max_model_len=65536, max_tokens=1536))):
            with self.subTest(echo=echo):
                _, error = self.startup_record(pinned, echo)
                self.assertEqual(str(error), 'worker context record differs from the pinned one')
        _, error = self.startup_record(default, dict(receipt, context=dict(max_model_len=65536, max_tokens=4096)))
        self.assertEqual(str(error), 'worker context record differs from the pinned one')

    def test_engine_checks_vllm_max_model_len(self):
        pinned = self.load(self.prepare(tasks(1)[0], max_tokens=4096)[1])
        self.assertEqual(CL.check_engine(NS(model_config=NS(max_model_len=65536)), pinned), 65536)
        self.assertIsNone(CL.check_engine(NS(), None))
        for vllm_config in (NS(model_config=NS(max_model_len=131072)), NS()):
            with self.assertRaisesRegex(LiveV2Error, 'engine max_model_len .* differs from the pinned 65536'):
                CL.check_engine(vllm_config, pinned)

    def test_worker_checks_the_record(self):
        record = dict(max_model_len=65536, max_tokens=4096)
        self.assertIsNone(CL.check_worker(NS(), None))
        self.assertEqual(CL.check_worker(NS(max_model_len=65536), record), record)
        self.assertEqual(CL.check_worker(NS(model_config=NS(max_model_len=65536)), record), record)
        with self.assertRaisesRegex(LiveV2Error, 'worker max_model_len 131072 differs from the pinned 65536'):
            CL.check_worker(NS(max_model_len=131072), record)
        for bad in (dict(record, max_tokens=1536), dict(record, max_tokens=2048), dict(record, extra=1),
                    dict(max_tokens=4096), [record]):
            with self.subTest(bad=bad):
                with self.assertRaises(LiveV2Error):
                    CL.check_worker(NS(max_model_len=65536), bad)

    def test_attach_loaded_worker_echoes_the_record(self):
        import test_cudagraph as CGT
        attach = CGT.AttachLoadedWorkerTests().attach
        default = attach(CGT.resolved_config(None))[0]
        record = dict(max_model_len=65536, max_tokens=4096)
        with patch.object(CL, 'runner_max_model_len', lambda runner: 65536):
            evidence = attach(CGT.resolved_config(None), dict(context=record))[0]
        self.assertEqual(evidence, dict(default, context=record))
        with patch.object(CL, 'runner_max_model_len', lambda runner: 131072), self.assertRaises(LiveV2Error):
            attach(CGT.resolved_config(None), dict(context=record))


class ReplyBoundEndToEndTests(unittest.TestCase):
    """test_decouple_e2e's engine, worker, frontend transcript and evidence on a server
    pinned to 4096-token replies (requests sampled with that bound)."""
    maxDiff = None

    @staticmethod
    def sampling(bound):
        import test_stageb_e2e as S
        former = S.sampling

        def sampling():
            value = former()
            value.max_tokens = bound
            return value
        return patch.object(S, 'sampling', sampling)

    def run_case(self, policy, case, bound, sampled):
        import test_decouple_e2e as DE
        former = DE.config

        def config(*args, **kwargs):
            value = former(*args, **kwargs)
            if bound != 1536:
                value['max_tokens'] = bound
            return value
        with patch.object(DE, 'config', config), self.sampling(sampled):
            return DE.play(policy, case)

    def test_a_4096_server_serves_the_same_turns(self):
        import test_decouple_e2e as DE
        # Default suite: full on legacy-784 and goal_query at P = 16; the slow pass
        # (README section 2.3) both on both.
        for case in (DE.CASES[0], DE.CASES[2]):
            for policy in ('full', 'goal_query'):
                if not (DE.SLOW or (case.name, policy) in (('legacy-784', 'full'), ('16-784', 'goal_query'))):
                    continue
                with self.subTest(case=case.name, policy=policy):
                    default = self.run_case(policy, case, 1536, 1536)
                    pinned = self.run_case(policy, case, 4096, 4096)
                    # Short replies: every prompt, plan, forward, output and evidence file is the same.
                    self.assertEqual(pinned['semantics'], default['semantics'])
                    self.assertEqual(pinned['files'], default['files'])

    def test_engine_refuses_another_bound(self):
        import test_decouple_e2e as DE
        for bound, sampled in ((4096, 1536), (1536, 4096)):
            with self.subTest(bound=bound, sampled=sampled):
                with self.assertRaisesRegex(LiveV2Error, r'differs: max_tokens=%d$' % sampled):
                    self.run_case('full', DE.CASES[2], bound, sampled)


# ---- 2. context limit (--max-model-len, --dram-arena-gib) --------------------------------------

L128 = 131072
CASES_128 = (  # configurations that fit at 131,072 tokens (KV budget and pinned host memory)
    (tasks(1)[0], 'full', 16, 784, None, None, {}),
    (tasks(3), 'goal_query', 24, 16, 'decoupled', dict(no_goal_rewriter=True), {}),
    (tasks(8), 'observation_query', 24, 784, 'decoupled', dict(active_token_budget=8192), {}),
    (tasks(11), 'random_fill', 32, 16, None, None, dict(cuda_graphs='full')),
    (tasks(24), 'full', 24, 16, None, None, dict(max_running=8)),
)


class ContextLauncherTests(Release, unittest.TestCase):
    maxDiff = None

    def test_default_is_byte_identical(self):
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                out = 'ctx%d' % P
                without = self.prepare(task, policy, gib, P, layout, selection, out=out, **extra)
                default = self.prepare(task, policy, gib, P, layout, selection, out=out, max_model_len=65536,
                                       dram_arena_gib=None, **extra)
                self.assertEqual(default, without)
                self.assertNotIn('dram_arena_pages', without[1])

    def test_131072_changes_only_the_context_limit(self):
        for task, policy, gib, P, layout, selection, extra in CASES_128:
            with self.subTest(policy=policy, P=P, extra=extra):
                out = 'l%d' % P
                m0, c0, a0, e0 = self.prepare(task, policy, gib, P, layout, selection, out=out, **extra)
                m1, c1, a1, e1 = self.prepare(task, policy, gib, P, layout, selection, out=out,
                                              max_model_len=L128, **extra)
                i = a0.index('--max-model-len') + 1
                self.assertEqual((a0[i], a1[i]), ('65536', '131072'))
                self.assertEqual(a1, a0[:i] + ['131072'] + a0[i + 1:])
                effective = dict(c0['effective_server_config'], max_model_len=L128)
                self.assertEqual(c1, dict(c0, effective_server_config=effective,
                                          server_config_sha256=serve.pin(effective)))
                self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                loaded = self.load(c1)
                self.assertEqual(CL.record(loaded), dict(max_model_len=L128, max_tokens=1536))
                # The frontend's check of vLLM's parsed arguments: 131072, not 65536.
                with patch.object(http_config, 'load', return_value=c1):
                    args = dict(effective)
                    if 'compilation_config' in args:
                        from remote_ra_live_v2 import cuda_graphs as CG
                        args['compilation_config'] = CG.observed(args['compilation_config'])
                    self.assertEqual(http_config.effective_server_config(NS(**args))['max_model_len'], L128)
                    with self.assertRaisesRegex(RuntimeError, 'observed API server arguments differ'):
                        http_config.effective_server_config(NS(**dict(args, max_model_len=65536)))

    def test_other_values_are_refused(self):
        # 262144 is a choice too (test_ctx_256k).
        for value in (32768, 100000, 524288, '131072', 131072.0, True):
            with self.subTest(value=value):
                self.assertEqual(self.outcome(tasks(1)[0], max_model_len=value),
                                 ('ValueError', '--max-model-len must be one of 65536, 131072, 262144'))
        for value in (0, -8, 8.0, '8', True):
            with self.subTest(arena=value):
                self.assertEqual(self.outcome(tasks(1)[0], page_tokens=16, dram_arena_gib=value),
                                 ('ValueError', '--dram-arena-gib must be a positive integer'))

    def test_one_session_must_fit_at_the_limit(self):
        # Decoupled, P = 16, one session at 8 GiB: 7,701 usable attention blocks, while one
        # session at 131,072 tokens needs 8,193 + 1,024 + 96 + 1 + 77 = 9,391 (9,551 with 4096).
        with self.assertRaisesRegex(ValueError, r'^--kv-cache-gib 8 with 1 sessions leaves 7701 usable attention '
                                                r'blocks of 16 tokens after the Mamba pool; one session at the '
                                                r'131072-token limit needs 9391: '):
            self.prepare(tasks(1)[0], kv_cache_gib=8, page_tokens=16, max_model_len=L128)
        with self.assertRaisesRegex(ValueError, r'limit with 4096-token replies needs 9551'):
            self.prepare(tasks(1)[0], kv_cache_gib=8, page_tokens=16, max_model_len=L128, max_tokens=4096)
        self.prepare(tasks(1)[0], kv_cache_gib=16, page_tokens=16, max_model_len=L128, max_tokens=4096)
        # The shared 784-token pool: 166 usable blocks at 8 GiB; one session at 131,072 tokens
        # needs 168 + 1 + 6 + 2 + 1 + 1 = 179 (what vLLM's startup check refuses too).
        with self.assertRaisesRegex(ValueError, r'^--kv-cache-gib 8 leaves 166 usable blocks of 784 tokens in the '
                                                r'shared pool; one session at the 131072-token limit needs 179: '):
            self.prepare(tasks(1)[0], max_model_len=L128)
        self.prepare(tasks(1)[0], kv_cache_gib=16, max_model_len=L128, max_tokens=4096)
        # Nothing changes at 65,536 tokens: the shared pool's check passes everywhere.
        for gib in serve.KV_CACHE_GIB:
            for T in serve.MAX_TOKENS_CHOICES:
                blocks = gib * GIB // serve.LEGACY_BLOCK_BYTES
                self.assertGreaterEqual(blocks - 1, serve.legacy_session_blocks(blocks, T, 65536))
        self.assertEqual(serve.legacy_session_blocks(167, 1536, 65536), 95)   # 85 + 6 + 3 + watermark 1

    def test_pinned_host_budget(self):
        # Every 65,536-token configuration fits the budget (at most 99.4 GiB), so nothing
        # the launcher accepted before is refused.
        worst = max(serve.pinned_host_bytes(S, P, P != 784 or decoupled, 65536)
                    for P in serve.PAGE_TOKENS for S in range(1, 17) for decoupled in (False, True))
        self.assertEqual(round(worst / GIB, 2), 99.37)
        self.assertLess(worst, serve.PINNED_HOST_GIB * GIB)
        # 16 sessions at 131,072 tokens, P = 16: 513 arena chunks of 256 MiB + 32 GiB + 16 GDN
        # carries = 162.5 GiB; 11 sessions fit (121.8 GiB), 12 do not (130.0 GiB).
        self.assertEqual(round(serve.pinned_host_bytes(16, 16, True, L128) / GIB, 2), 162.54)
        with self.assertRaisesRegex(ValueError, r'^16 sessions at the 131072-token limit may pin 162\.5 GiB of '
                                                r'host memory .* more than the 128 GiB budget: use fewer '
                                                r'sessions or --dram-arena-gib$'):
            self.prepare(tasks(16), kv_cache_gib=24, page_tokens=16, max_model_len=L128)
        self.prepare(tasks(11), kv_cache_gib=24, page_tokens=16, max_model_len=L128)
        with self.assertRaisesRegex(ValueError, r'^12 sessions .* 130\.0 GiB'):
            self.prepare(tasks(12), kv_cache_gib=24, page_tokens=16, max_model_len=L128)
        # The shared pool pins a host copy per Page too, and has no arena cap.
        self.prepare(tasks(11), kv_cache_gib=24, max_model_len=L128)
        with self.assertRaisesRegex(ValueError, r'^12 sessions .* use fewer sessions$'):
            self.prepare(tasks(12), kv_cache_gib=24, max_model_len=L128)
        # A session pool counts its K running sessions, not its registered tasks.
        self.prepare(tasks(32), kv_cache_gib=24, page_tokens=16, max_model_len=L128, max_running=11)

    def test_gdn_handoff_counts_two_carries(self):
        # --gdn-device-handoff keeps two pinned carries per session: one more GDN carry each.
        one = serve.pinned_host_bytes(16, 16, True, L128, 88 * 1024)
        two = serve.pinned_host_bytes(16, 16, True, L128, 88 * 1024, 2)
        self.assertEqual(two - one, 16 * serve.GDN_CARRY_BYTES)
        self.assertEqual((round(one / GIB, 2), round(two / GIB, 2)), (122.29, 124.59))
        # An arena of 93 GiB fits 16 sessions with one carry each (127.3 GiB), not with two (129.6 GiB).
        kwargs = dict(max_model_len=L128, dram_arena_gib=93, max_tokens=4096)
        self.prepare(tasks(16), 'full', 24, 16, **kwargs)
        with self.assertRaisesRegex(ValueError, r'^16 sessions at the 131072-token limit may pin 129\.6 GiB of '
                                                r'host memory \(the 32 GiB connector pool, GDN carries \(two per '
                                                r'session with --gdn-device-handoff\)'):
            self.prepare(tasks(16), 'full', 24, 16, gdn_device_handoff=True, **kwargs)

    def test_dram_arena_cap(self):
        _, config, argv, _ = self.prepare(tasks(16), 'observation_query', 24, 16, max_model_len=L128,
                                          dram_arena_gib=88, max_tokens=4096,
                                          selection=dict(active_token_budget=8192), cuda_graphs='full')
        self.assertEqual(config['dram_arena_pages'], 88 * 1024)       # 1-MiB Pages
        self.assertEqual(round(serve.pinned_host_bytes(16, 16, True, L128, 88 * 1024) / GIB, 2), 122.29)
        loaded = self.load(config)
        self.assertEqual(CL.arena_pages(loaded, 16), 90112)
        for gib, message in ((8, r'holds 8192 Pages of 16 tokens; it must hold one session at the 131072-token '
                                 r'limit \(8193 Pages\) and less than every session \(131088 Pages\)'),
                             (200, r'holds 204800 Pages')):
            with self.subTest(gib=gib), self.assertRaisesRegex(ValueError, message):
                self.prepare(tasks(16), kv_cache_gib=24, page_tokens=16, max_model_len=L128, dram_arena_gib=gib)
        with self.assertRaisesRegex(ValueError, '--dram-arena-gib needs the decoupled KV layout'):
            self.prepare(tasks(4), kv_cache_gib=24, max_model_len=L128, dram_arena_gib=40)
        # At 65,536 tokens the option caps the arena too (fewer sessions' worth).
        config = self.prepare(tasks(16), kv_cache_gib=24, page_tokens=16, dram_arena_gib=32)[1]
        self.assertEqual(self.load(config)['dram_arena_pages'], 32768)


class ContextConfigTests(Release, unittest.TestCase):
    def test_load_refusals(self):
        config = self.prepare(tasks(4), kv_cache_gib=24, page_tokens=16, max_model_len=L128)[1]
        self.assertEqual(CL.check_config(self.load(config)), (1536, L128))
        effective = dict(config['effective_server_config'], max_model_len=100000)
        with self.assertRaisesRegex(RuntimeError, 'pinned max_model_len must be one of 65536, 131072, 262144'):
            self.load(dict(config, effective_server_config=effective, server_config_sha256=serve.pin(effective)))
        one = CL.session_pages(16, L128)
        self.assertEqual(self.load(dict(config, dram_arena_pages=one))['dram_arena_pages'], one)
        for pages_, message in ((one - 1, 'dram_arena_pages must hold one session'),
                                (4 * one, 'less than 4 sessions'), ('9000', 'dram_arena_pages must hold'),
                                (9000.0, 'dram_arena_pages must hold')):
            with self.subTest(pages=pages_), self.assertRaisesRegex(RuntimeError, message):
                self.load(dict(config, dram_arena_pages=pages_))
        legacy = self.prepare(tasks(4), kv_cache_gib=24, max_model_len=L128)[1]
        with self.assertRaisesRegex(RuntimeError, 'dram_arena_pages needs the decoupled KV layout'):
            self.load(dict(legacy, dram_arena_pages=200))


class ContextStartupTests(Release, unittest.TestCase):
    def startup_record(self, config):
        return ReplyBoundStartupTests.startup_record(self, config)[0]

    def test_engine_sends_the_context_record_and_the_arena(self):
        config = self.prepare(tasks(1)[0], kv_cache_gib=16, max_model_len=L128)[1]
        record = self.startup_record(config)
        self.assertEqual(record['context'], dict(max_model_len=L128, max_tokens=1536))
        self.assertNotIn('dram_arena_pages', record)
        config = self.prepare(tasks(3), kv_cache_gib=24, page_tokens=16, max_model_len=L128, dram_arena_gib=12)[1]
        self.assertEqual(self.startup_record(config)['dram_arena_pages'], 12 * 1024)
        config = self.prepare(tasks(3), kv_cache_gib=24, page_tokens=16, dram_arena_gib=6)[1]
        record = self.startup_record(config)
        self.assertEqual((record['dram_arena_pages'], 'context' in record), (6 * 1024, False))
        self.assertEqual(CL.check_engine(NS(model_config=NS(max_model_len=L128)), self.load(
            self.prepare(tasks(1)[0], kv_cache_gib=16, max_model_len=L128)[1])), L128)

    def test_worker_arena_and_record(self):
        from remote_ra_live_v2.worker_composition import arena_pages
        runner = NS(max_model_len=L128)
        layout = I0.for_server(16, 16, True)
        self.assertEqual(arena_pages(runner, {}, layout), 16 * 8192 + 16)
        self.assertEqual(arena_pages(runner, {}, I0.for_server(784, 16, True)), 16 * 168 + 16)
        self.assertEqual(arena_pages(runner, dict(dram_arena_pages=90112), layout), 90112)
        record = dict(max_model_len=L128, max_tokens=4096)
        self.assertEqual(CL.check_worker(runner, record), record)
        self.assertEqual(CL.check_worker(runner, dict(record, max_tokens=1536)), dict(record, max_tokens=1536))
        with self.assertRaisesRegex(LiveV2Error, 'worker max_model_len 65536 differs from the pinned 131072'):
            CL.check_worker(NS(max_model_len=65536), record)

    def test_page_catalog_bound_follows_the_context_limit(self):
        import test_decouple_wp2 as WP2
        import test_stageb_engine as E
        from remote_ra_native_growing.adapter import EngineCoreAdapter
        from remote_ra_native_growing.contract import Authority
        layout = I0.for_server(16, 16, True)
        with WP2.matrix_case(layout):
            for length, limit in ((65536, 4096), (L128, 8192)):
                kv = WP2.KVManager(layout, 2000)
                kv.max_model_len = length
                adapter = EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), kv, page_tokens=16,
                                            full_attention_managers=(kv.fa,), layout=layout)
                self.assertEqual(adapter.page_limit, limit)


# ---- 3. 120 steps: the lifetime request bound and the SWE-bench examples ------------------------

class RequestBoundTests(unittest.TestCase):
    def test_the_lifetime_bound_sums_the_per_session_bound(self):
        from remote_ra_live_v2 import sessions
        from remote_ra_live_v2.controller import LiveFourPolicyController
        from remote_ra_live_v2.session_authority import MAX_GENERATIONS
        import inspect
        # One per-session bound everywhere: the frontend's max_turns, the engine's generations.
        self.assertEqual(sessions.MAX_TURNS, 256)
        self.assertEqual(inspect.signature(LiveFourPolicyController).parameters['max_turns'].default, 256)
        self.assertEqual(MAX_GENERATIONS, 256)
        self.assertEqual(sessions.REQUESTS_PER_SESSION, sessions.MAX_TURNS)
        for count in range(1, sessions.MAX_TASKS + 1):
            limit = sessions.request_limit(count, sessions.MAX_TASKS)
            self.assertEqual(limit, max(1024, 256 * count))
            self.assertGreaterEqual(limit, count * sessions.MAX_TURNS)     # never binds first
        # 32 tasks of 120-step agents: 3,840 requests against 8,192 (4,096 before).
        self.assertEqual(sessions.request_limit(32, sessions.MAX_TASKS), 8192)
        self.assertLessEqual(32 * 120, sessions.request_limit(32, sessions.MAX_TASKS) // 2)

    def test_a_capacity_wait_takes_no_lifetime_record(self):
        """The gate refuses before the request tombstone, command id, lifecycle entry and
        render binding: retries of a waiting turn never count against the bound."""
        from registered_add_protocol import RegisteredAdd, RegisteredAddDispatcher
        from remote_ra_live_v2 import capacity as gate
        from remote_ra_live_v2.registered_route import RegisteredRouteBridge
        dispatcher = RegisteredAddDispatcher(codec=None, boundary=None, decode_core=lambda raw: raw,
                                             preprocess=lambda request: request)
        lifecycle = NS(entries={}, max_requests=1024, endpoint=NS(core=NS()))
        bridge = RegisteredRouteBridge(lifecycle)
        command = RegisteredAdd('c1', 'r1', 'x1', 'a' * 64, 's', '', {}, dict(turn=3), 300)

        def wait(lifecycle, command):
            raise gate.CapacityWait('CAPACITY_WAIT: test')
        with patch.object(gate, 'admission_gate', wait):
            for retry in range(5):
                with self.assertRaises(gate.CapacityWait):
                    bridge.add(dispatcher, command, 'f')
        self.assertEqual((dispatcher._records, dispatcher._command_ids, lifecycle.entries), ({}, {}, {}))


class SweBenchExampleTests(unittest.TestCase):
    """examples/swebench: STEP_LIMIT and MAX_TOKENS for run_agents.sh, pilot bytes by default."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location('render_override',
                                                      ROOT / 'examples/swebench/render_override.py')
        cls.R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.R)
        cls.template = (ROOT / 'examples/swebench/model_override.yaml').read_bytes().decode('utf-8')

    def test_defaults_are_the_pilot_file(self):
        pilot = self.template.replace('<PORT>', '20000')        # run_agents.sh's former sed "s/<PORT>/.../"
        self.assertEqual(self.R.render(self.template, '20000'), pilot)
        self.assertEqual(self.R.render(self.template, '20000', '1536', '90'), pilot)
        self.assertIn('    max_tokens: 1536\n', pilot)
        self.assertIn('  step_limit: 90\n', pilot)

    def test_selected_values_change_two_lines(self):
        pilot = self.template.replace('<PORT>', '20000').split('\n')
        chosen = self.R.render(self.template, '20000', '4096', '120').split('\n')
        changed = [(a, b) for a, b in zip(pilot, chosen) if a != b]
        self.assertEqual(len(chosen), len(pilot))
        self.assertEqual(changed, [('    max_tokens: 1536', '    max_tokens: 4096'),
                                   ('  step_limit: 90', '  step_limit: 120')])

    def test_refusals(self):
        for args in (('20000', '2048', '120'), ('20000', '4096', '0'), ('20000', '4096', '257'),
                     ('20000', '4096', '090'), ('20000', '4096', 'x'), ('200', '1536', '90')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.R.render(self.template, *args)
        with self.assertRaises(ValueError):
            self.R.render(self.template.replace('  step_limit: 90', '  step_limit: 100'), '20000')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.R.main(['render_override.py', str(ROOT / 'examples/swebench/model_override.yaml'),
                                          '20000', '2048', '90']), 2)

    def test_scripts_use_the_renderer(self):
        script = (ROOT / 'examples/swebench/run_agents.sh').read_text(encoding='utf-8')
        self.assertNotIn('sed "s/<PORT>/', script)
        self.assertEqual(script.count('render_override.py" "$HERE/model_override.yaml"'), 2)
        self.assertIn('STEP_LIMIT=${STEP_LIMIT:-90}; MAX_TOKENS=${MAX_TOKENS:-1536}', script)
        slurm = (ROOT / 'examples/slurm/serve_gh200.sh').read_text(encoding='utf-8')
        for flag in ('--max-tokens "$MAX_TOKENS"', '--max-model-len "$MAX_MODEL_LEN"',
                     '--dram-arena-gib "$DRAM_ARENA_GIB"', '${STEP_LIMIT:+STEP_LIMIT=$STEP_LIMIT}'):
            self.assertIn(flag, slurm)
        # The PoC files stay pinned.
        sums = (ROOT / 'examples/swebench/upstream/SHA256SUMS').read_text(encoding='utf-8')
        for name in ('swebench_runner.py', 'swebench_prompt_override.yaml'):
            digest = sha256((ROOT / 'examples/swebench/upstream' / name).read_bytes()).hexdigest()
            self.assertIn('%s  %s' % (digest, name), sums)
            self.assertIn(digest, script)


# ---- token and frame caps at about 127K tokens, real tokenizer ----------------------------------

try:
    import tokenizers as _tokenizers
except ImportError:
    _tokenizers = None
TOKENIZER_DIR = os.environ.get('REMOTE_RA_TOKENIZER_DIR')


def _msgpack_ints(ids):
    """msgpack of a list of non-negative ints (as vLLM's encoder writes prompt_token_ids)."""
    n = len(ids)
    out = bytearray(bytes([0x90 | n]) if n < 16 else b'\xdc' + n.to_bytes(2, 'big') if n < 65536
                    else b'\xdd' + n.to_bytes(4, 'big'))
    for t in ids:
        out += (bytes([t]) if t < 128 else b'\xcc' + bytes([t]) if t < 256 else
                b'\xcd' + t.to_bytes(2, 'big') if t < 65536 else b'\xce' + t.to_bytes(4, 'big'))
    return bytes(out)


@unittest.skipUnless(_tokenizers is not None and TOKENIZER_DIR, 'real tokenizer not available '
                     '(set REMOTE_RA_TOKENIZER_DIR to the pinned Qwen3.6-27B tokenizer directory)')
class TokenFrameCapTests(unittest.TestCase):
    """Mini-SWE-Agent XML requests of about 127K tokens (131,072 - 4,096: the longest
    prompt a 131072/4096 server accepts) through the runtime's real caps: the XML
    source snapshot (runtime_renderer.MAX_SOURCE_BYTES, frontend and engine), the
    per-message encode bound, the dispatcher's token bound and the authenticated
    frame (registered_add_protocol MAX_TOKENS, MAX_CORE_BYTES, MAX_WIRE_BYTES).
    Observations are cut at 10,000 characters as the PoC's observation template does."""

    @classmethod
    def setUpClass(cls):
        path = Path(TOKENIZER_DIR) / 'tokenizer.json'
        pins = json.loads((ROOT / 'INSTALL_RELEASE.json').read_bytes())['model_pins']
        if sha256(path.read_bytes()).hexdigest() != pins['tokenizer.json']:
            raise unittest.SkipTest('tokenizer.json differs from the pinned model metadata')
        cls.tokenizer = _tokenizers.Tokenizer.from_file(str(path))
        import random
        cls.rng = random.Random(20261009)
        code = []
        for source in sorted((ROOT / 'source-overlay' / 'vllm' / 'v1').rglob('*.py')):
            code.append(source.read_text(encoding='utf-8'))
        cls.code = '\n'.join(code)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    # SWE-like observations, at most 10,000 characters each.
    def cat_n(self):
        i = self.rng.randrange(0, len(self.code) - 12000)
        lines = self.code[i:i + 12000].splitlines()
        first = self.rng.randrange(1, 2000)
        return '\n'.join('%6d\t%s' % (first + k, line) for k, line in enumerate(lines))[:9800]

    def pytest_v(self):
        rows = ['=' * 29 + ' test session starts ' + '=' * 30, 'collecting ... collected 412 items', '']
        k = 0
        while sum(len(r) + 1 for r in rows) < 9800:
            k += 1
            left = 'astropy/io/fits/tests/test_header.py::TestHeaderFunctions::test_case_%d %s' % (
                k, self.rng.choice(('PASSED', 'PASSED', 'SKIPPED', 'FAILED')))
            rows.append(left + ' ' * max(1, 113 - len(left)) + '[%3d%%]' % min(99, k // 4))
        return '\n'.join(rows)[:9800]

    def fits_cards(self):
        rows = []
        while sum(len(r) + 1 for r in rows) < 9800:
            key = self.rng.choice(('SIMPLE', 'BITPIX', 'NAXIS1', 'OBJECT', 'CRVAL1', 'CTYPE2', 'EXPTIME'))
            value = self.rng.choice(('T', str(self.rng.randrange(-64, 4096)), "'RA---TAN'", '%.6E' % self.rng.uniform(-1e3, 1e3)))
            rows.append(('%-8s= %20s / %s' % (key, value, self.rng.choice(('', 'reference pixel', 'number of axes'))))[:80].ljust(80))
        return '\n'.join(rows)[:9800]

    def banners(self):
        rows = []
        while sum(len(r) + 1 for r in rows) < 9800:
            rows += [self.rng.choice('=-_*#') * 80, self.rng.choice(('PASSED', 'warnings summary', ''))]
        return '\n'.join(rows)[:9800]

    def conversation(self, makers, target):
        system = 'You are a helpful assistant that can interact with a computer.'
        task = '<pr_description>\nConsider the following PR description:\n%s\n</pr_description>' % (
            'Header.fromstring does not accept Python 3 bytes. ' * 40)
        messages = [dict(role='system', content=system), dict(role='user', content=task)]
        ids = self.encode('<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n' % (system, task))
        header = self.encode(xml.GENERATION_HEADER)
        k = 0
        while True:
            k += 1
            reply = ('THOUGHT: inspect step %d.\n\n<mswea_bash_command>sed -n \'%d,%dp\' '
                     'astropy/io/fits/header.py</mswea_bash_command>' % (k, 40 * k, 40 * k + 120))
            observation = '<returncode>0</returncode>\n<output>\n%s</output>' % makers[k % len(makers)]()
            a = self.encode(xml.message_text('assistant', reply))
            o = self.encode(xml.message_text('user', observation))
            if len(ids) + len(a) + len(o) + len(header) > target:
                lo, hi, room = 0, len(observation), target - len(ids) - len(a) - len(header)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if len(self.encode(xml.message_text('user', observation[:mid]))) <= room:
                        lo = mid
                    else:
                        hi = mid - 1
                messages += [dict(role='assistant', content=reply), dict(role='user', content=observation[:lo])]
                ids += a + self.encode(xml.message_text('user', observation[:lo]))
                break
            messages += [dict(role='assistant', content=reply), dict(role='user', content=observation)]
            ids += a + o
        return messages, ids + header

    def through_the_caps(self, messages, ids):
        """Every cap a request of these messages and prompt ids meets; returns the sizes."""
        import registered_add_protocol as R
        import runtime_renderer as RR
        xml.validate_messages(messages)
        source = RR.validate_xml_source(dict(schema=RR.XML_SOURCE_SCHEMA, messages=messages))
        for message in messages:
            self.assertLessEqual(len(xml.message_text(message['role'], message['content']).encode()),
                                 RR.MAX_SOURCE_BYTES)
        self.assertLessEqual(len(ids), R.MAX_TOKENS)
        core = _msgpack_ints(ids) + b'\x00' * 4096        # the prompt ids plus a bound for the other fields
        self.assertLessEqual(len(core), R.MAX_CORE_BYTES)
        import base64
        codec = R.ProtocolCodec(key=b'k' * 32, epoch='e' * 64, identity='f' * 64)
        command = R.RegisteredAdd(command_id='c' * 64, request_id='r' * 80, external_id='x' * 80,
                                  prompt_hash='a' * 64, session_id='s' * 40,
                                  core_b64=base64.b64encode(core).decode(), source=source,
                                  metadata=dict(instance_id='s' * 40, turn=119, generation='g' * 64),
                                  lease_seconds=300)
        frame = codec.encode(command)
        self.assertEqual(codec.decode(frame).source, source)
        snapshot = len(json.dumps(source, ensure_ascii=False, separators=(',', ':')).encode())
        return dict(tokens=len(ids), source_bytes=snapshot, core_bytes=len(core), frame_bytes=len(frame))

    def test_swe_like_requests_at_127k_fit_every_cap(self):
        import runtime_renderer as RR
        for name, makers in (('code', [self.cat_n]), ('padded', [self.pytest_v, self.fits_cards, self.cat_n]),
                             ('fits', [self.fits_cards])):
            with self.subTest(mix=name):
                messages, ids = self.conversation(makers, L128 - 4096)
                self.assertGreaterEqual(len(ids), L128 - 4096 - 16)
                sizes = self.through_the_caps(messages, ids)
                # 3.3 (code) to 5.3 (FITS cards) source bytes per token: under two thirds of the
                # 1 MiB source cap; the core stays under 0.3 MB and the frame under 1.5 MB.
                self.assertLess(sizes['source_bytes'], 0.66 * RR.MAX_SOURCE_BYTES, sizes)

    def test_banner_only_logs_hit_the_source_cap(self):
        """The 1 MiB source cap binds above 8.26 bytes per token at 126,976 tokens (16.4 at
        64,000): banner-only logs, one 80-column rule per token, exceed it at 127K tokens.
        The cap is unchanged; such histories are refused before admission."""
        import runtime_renderer as RR
        messages, ids = self.conversation([self.banners], 20000)
        sizes = self.through_the_caps(messages, ids)
        self.assertGreater(sizes['source_bytes'] / sizes['tokens'], RR.MAX_SOURCE_BYTES / (L128 - 4096))
        messages, ids = self.conversation([self.banners], L128 - 4096)
        with self.assertRaisesRegex(Exception, 'normalized source exceeds limit'):
            RR.validate_xml_source(dict(schema=RR.XML_SOURCE_SCHEMA, messages=messages))


if __name__ == '__main__':
    unittest.main()
