"""256K context, stage 1 (branch ctx-256k). CPU only; nothing here is GPU evidence.

``serve.py --max-model-len 262144`` is the third context limit: like 131072 it changes
only vLLM's --max-model-len and effective_server_config.max_model_len (262144 is the
model's max_position_embeddings with its default rope; no YaRN, no other pin). One
request's byte and token caps scale with the pinned limit
(remote_ra_live_v2.context_limits.request_caps): at 65536 and 131072 they are the
values the sources carry and nothing is set; at 262144 the frontend and the engine
set them once at startup (pin_request_caps). Covered:

- the launcher: 262144 changes only the context limit (prepare and the dry-run
  main), other values are refused, the defaults and 131072 are unchanged;
- refusals at 262144: the attention pool (8 and 16 GiB in every layout and session
  count; 24 and 32 GiB pass), the pinned-host budget (5 running sessions without an
  arena; 6 and 16 refused), the arena range for 16 sessions (17 to 93 GiB, 91 with
  --gdn-device-handoff, the largest fitting arena named in the refusal), a session
  pool counting its K running sessions;
- startup: load, the frontend's argument check, the engine's and the worker's
  checks, the worker startup record and its echo, the worker's arena (262,160 Pages
  for 16 sessions at P = 16) and the Page catalog bound (16,384 Pages);
- the request caps: the derived values, the startup pin in compose_engine and
  compose_frontend (nothing at 65536 and 131072), EngineCore's registered-add message
  bound (the shipped _decode_registered_add, extracted from core.py);
- Mini-SWE-Agent requests of 258,048 tokens (262,144 - 4,096) through the scaled
  caps with the pinned tokenizer (skipped unless REMOTE_RA_TOKENIZER_DIR names the
  pinned tokenizer directory and the tokenizers package is importable);
- 262144 together with ``--decode-fastpath`` (main 43fd9c3), on the configurations of
  the GPU servers of job 3360375: the launcher's config holds both pins and the vLLM
  command is the 262144 one; load, the frontend, the engine's startup record and echo
  checks, the worker's echo and ready.json carry and check both, each on its own.
"""
import ast
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

import serve
import test_align_ctx as AC

from remote_ra_http import config as http_config
from remote_ra_live_v2 import context_limits as CL
from remote_ra_live_v2 import decode_fastpath as F
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import mswea_protocol as xml
from remote_ra_live_v2.contract import LiveV2Error

GIB = 1024 ** 3
MIB = 1024 ** 2
L256 = 262144
L128 = 131072
tasks = AC.tasks

CAPS_256 = dict(source_bytes=2 * MIB, core_bytes=2 * MIB, wire_bytes=8 * MIB, input_bytes=9 * MIB,
                prompt_tokens=L256)
CASES_256 = (  # configurations that fit at 262,144 tokens (KV budget and pinned host memory)
    (tasks(1)[0], 'full', 24, 784, None, None, {}),
    (tasks(3), 'goal_query', 24, 16, 'decoupled', dict(no_goal_rewriter=True), {}),
    (tasks(5), 'observation_query', 32, 784, 'decoupled', dict(active_token_budget=8192), {}),
    (tasks(5), 'random_fill', 32, 16, None, None, dict(cuda_graphs='full')),
    (tasks(24), 'full', 24, 16, None, None, dict(max_running=5)),
)


def caps_now():
    """The caps the modules hold now, keyed like CL.CAP_TARGETS (module.attribute)."""
    import importlib
    return {'%s.%s' % (name, attribute): getattr(importlib.import_module(name), attribute)
            for name, attribute, _ in CL.CAP_TARGETS}


def module_caps(caps):
    """What caps_now() reads once ``caps`` (a request_caps value) are pinned."""
    return {'%s.%s' % (name, attribute): caps[key] for name, attribute, key in CL.CAP_TARGETS}


@contextlib.contextmanager
def caps_restored():
    """Leave the module caps as they were, whatever a test pins."""
    import importlib
    saved = [(importlib.import_module(name), attribute) for name, attribute, _ in CL.CAP_TARGETS]
    values = [getattr(module, attribute) for module, attribute in saved]
    try:
        yield
    finally:
        for (module, attribute), value in zip(saved, values):
            setattr(module, attribute, value)


def shipped_decode():
    """EngineCoreProc._decode_registered_add, extracted from the shipped core.py."""
    tree = ast.parse((ROOT / 'source-overlay/vllm/v1/engine/core.py').read_text('utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCoreProc')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_registered_add')
    namespace = dict(RegisteredAddWire='wire',
                     MsgpackDecoder=lambda kind: NS(decode=lambda frames: (kind, len(frames[0]))))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'core.py', 'exec'), namespace)
    return namespace['_decode_registered_add']


# ---- 1. launcher ---------------------------------------------------------------------------------

class LauncherTests(AC.Release, unittest.TestCase):
    maxDiff = None

    def test_262144_changes_only_the_context_limit(self):
        for task, policy, gib, P, layout, selection, extra in CASES_256:
            with self.subTest(policy=policy, P=P, extra=extra):
                out = 'l%d' % P
                m0, c0, a0, e0 = self.prepare(task, policy, gib, P, layout, selection, out=out, **extra)
                m1, c1, a1, e1 = self.prepare(task, policy, gib, P, layout, selection, out=out,
                                              max_model_len=L128, **extra)
                m2, c2, a2, e2 = self.prepare(task, policy, gib, P, layout, selection, out=out,
                                              max_model_len=L256, **extra)
                i = a0.index('--max-model-len') + 1
                self.assertEqual((a0[i], a1[i], a2[i]), ('65536', '131072', '262144'))
                self.assertEqual(a2, a0[:i] + ['262144'] + a0[i + 1:])
                self.assertEqual(m2, m0)
                effective = dict(c0['effective_server_config'], max_model_len=L256)
                self.assertEqual(c2, dict(c0, effective_server_config=effective,
                                          server_config_sha256=serve.pin(effective)))
                self.assertEqual(c2, dict(c1, effective_server_config=effective,
                                          server_config_sha256=serve.pin(effective)))
                self.assertEqual(e2, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c2)))
                loaded = self.load(c2)
                self.assertEqual(CL.check_config(loaded), (1536, L256))
                self.assertEqual(CL.record(loaded), dict(max_model_len=L256, max_tokens=1536))
                # The frontend's check of vLLM's parsed arguments: 262144, not 131072.
                with patch.object(http_config, 'load', return_value=c2):
                    args = dict(effective)
                    if 'compilation_config' in args:
                        from remote_ra_live_v2 import cuda_graphs as CG
                        args['compilation_config'] = CG.observed(args['compilation_config'])
                    self.assertEqual(http_config.effective_server_config(NS(**args))['max_model_len'], L256)
                    with self.assertRaisesRegex(RuntimeError, 'observed API server arguments differ'):
                        http_config.effective_server_config(NS(**dict(args, max_model_len=L128)))

    def test_dry_run_main(self):
        task = self.root / 'task256.json'
        task.write_text(json.dumps(tasks(1)[0]))
        base = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--policy', 'full',
                '--output', str(self.root / 'dry256'), '--page-tokens', '16', '--kv-cache-gib', '24']

        def main(*argv):
            out = io.StringIO()
            with patch.object(sys, 'argv', base + list(argv)), contextlib.redirect_stdout(out), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(out.getvalue())
        plain, long_ = main(), main('--max-model-len', '262144', '--max-tokens', '4096')
        self.assertEqual((long_['status'], long_['starts_model']), ('SERVICE_DRY_RUN', False))
        i = plain['command'].index('--max-model-len') + 1
        self.assertEqual(long_['command'], plain['command'][:i] + ['262144'] + plain['command'][i + 1:])
        self.assertEqual(main('--max-model-len', '131072')['command'][i], '131072')
        self.assertEqual(main('--max-model-len', '65536'), plain)
        for value in ('524288', '1010000', '262143'):
            with self.subTest(value=value), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                main('--max-model-len', value)

    def test_other_values_are_refused(self):
        for value in (32768, 100000, 262143, 524288, 1010000, '262144', 262144.0, True):
            with self.subTest(value=value):
                self.assertEqual(self.outcome(tasks(1)[0], max_model_len=value),
                                 ('ValueError', '--max-model-len must be one of 65536, 131072, 262144'))

    def test_one_session_must_fit_at_262144(self):
        # 16 GiB at P = 16, one session: 15,893 usable attention blocks; one session at
        # 262,144 tokens needs 16,385 + 1,024 + 256 + 1 + 158 = 17,824 with 4096-token replies
        # (17,664 with 1536). 24 GiB with 16 sessions: 19,681 usable, 17,862 needed.
        with self.assertRaisesRegex(ValueError, r'^--kv-cache-gib 16 with 1 sessions leaves 15893 usable attention '
                                                r'blocks of 16 tokens after the Mamba pool; one session at the '
                                                r'262144-token limit with 4096-token replies needs 17824: use a '
                                                r'larger --kv-cache-gib or fewer sessions$'):
            self.prepare(tasks(1)[0], kv_cache_gib=16, page_tokens=16, max_model_len=L256, max_tokens=4096)
        with self.assertRaisesRegex(ValueError, r'262144-token limit needs 17664: '):
            self.prepare(tasks(1)[0], kv_cache_gib=16, page_tokens=16, max_model_len=L256)
        layout = I0.for_server(16, 16, True)
        blocks = serve.attention_blocks(24 * GIB, layout)
        self.assertEqual((blocks - 1, serve.one_session_blocks(16, blocks, 4096, L256)), (19681, 17862))
        self.prepare(tasks(16), kv_cache_gib=24, page_tokens=16, max_model_len=L256, max_tokens=4096,
                     dram_arena_gib=93)
        # The shared 784-token pool: 333 usable blocks at 16 GiB; one session needs
        # 335 + 1 + 6 + 6 + 1 + 3 = 352 with 4096-token replies. 24 GiB (500 usable) passes.
        with self.assertRaisesRegex(ValueError, r'^--kv-cache-gib 16 leaves 333 usable blocks of 784 tokens in the '
                                                r'shared pool; one session at the 262144-token limit needs 352: '
                                                r'use a larger --kv-cache-gib$'):
            self.prepare(tasks(1)[0], kv_cache_gib=16, max_model_len=L256, max_tokens=4096)
        self.prepare(tasks(1)[0], kv_cache_gib=24, max_model_len=L256, max_tokens=4096)

    def test_kv_refusals_in_every_layout(self):
        """8 and 16 GiB are refused at every session count and Page size; 24 and 32 GiB pass."""
        for P in (16, 32, 64, 784):
            for count in range(1, 17):
                layout = I0.for_server(P, count, True)
                for T in serve.MAX_TOKENS_CHOICES:
                    for gib in serve.KV_CACHE_GIB:
                        blocks = serve.attention_blocks(gib * GIB, layout)
                        fits = blocks - 1 >= serve.one_session_blocks(P, blocks, T, L256)
                        self.assertEqual(fits, gib >= 24, (P, count, T, gib))
        for gib in serve.KV_CACHE_GIB:
            blocks = gib * GIB // serve.LEGACY_BLOCK_BYTES
            for T in serve.MAX_TOKENS_CHOICES:
                self.assertEqual(blocks - 1 >= serve.legacy_session_blocks(blocks, T, L256), gib >= 24, (gib, T))
        # Through prepare: the refusal names the attention pool, whatever the host budget.
        for P, layout in ((16, None), (32, None), (64, None), (784, 'decoupled'), (784, None)):
            for count in (1, 5, 16):
                for gib in (8, 16):
                    with self.subTest(P=P, layout=layout, count=count, gib=gib):
                        task = tasks(1)[0] if count == 1 else tasks(count)
                        kind, message = self.outcome(task, kv_cache_gib=gib, page_tokens=P, layout=layout,
                                                     max_model_len=L256)
                        self.assertEqual(kind, 'ValueError')
                        self.assertRegex(message, r'^--kv-cache-gib %d .* one session at the 262144-token limit '
                                                  r'needs \d+: use a larger --kv-cache-gib' % gib)

    def test_pinned_host_budget_at_262144(self):
        # P = 16: 32 GiB + S GDN carries + the arena in 256 MiB chunks of S x 16,385 Pages.
        host = {S: round(serve.pinned_host_bytes(S, 16, True, L256) / GIB, 2) for S in (1, 5, 6, 16)}
        self.assertEqual(host, {1: 48.39, 5: 112.97, 6: 129.11, 16: 290.54})
        self.prepare(tasks(5), kv_cache_gib=24, page_tokens=16, max_model_len=L256, max_tokens=4096)
        for count, gib in ((6, '129\\.1'), (16, '290\\.5')):
            with self.subTest(count=count), \
                    self.assertRaisesRegex(ValueError, r'^%d sessions at the 262144-token limit may pin %s GiB of host '
                                                       r'memory .* more than the 128 GiB budget: use fewer sessions '
                                                       r'or --dram-arena-gib$' % (count, gib)):
                self.prepare(tasks(count), kv_cache_gib=24, page_tokens=16, max_model_len=L256)
        # The shared pool pins a host copy per Page too, and has no arena cap.
        self.prepare(tasks(5), kv_cache_gib=24, max_model_len=L256)
        with self.assertRaisesRegex(ValueError, r'^6 sessions at the 262144-token limit may pin 129\.3 GiB .* '
                                                r'use fewer sessions$'):
            self.prepare(tasks(6), kv_cache_gib=24, max_model_len=L256)
        # A session pool counts its K running sessions, not its registered tasks.
        self.prepare(tasks(32), kv_cache_gib=24, page_tokens=16, max_model_len=L256, max_running=5)
        with self.assertRaisesRegex(ValueError, r'^6 sessions at the 262144-token limit'):
            self.prepare(tasks(32), kv_cache_gib=24, page_tokens=16, max_model_len=L256, max_running=6)

    def test_sixteen_sessions_need_the_arena(self):
        kwargs = dict(policy='observation_query', kv_cache_gib=32, page_tokens=16, max_model_len=L256,
                      max_tokens=4096, selection=dict(active_token_budget=8192))
        for gib in (17, 88, 93):
            with self.subTest(gib=gib):
                config = self.prepare(tasks(16), dram_arena_gib=gib, **kwargs)[1]
                self.assertEqual(config['dram_arena_pages'], gib * 1024)       # 1-MiB Pages
                self.assertEqual(CL.arena_pages(self.load(config), 16), gib * 1024)
        self.assertEqual([round(serve.pinned_host_bytes(16, 16, True, L256, g * 1024) / GIB, 2) for g in (17, 93, 94)],
                         [51.29, 127.29, 128.29])
        # Below one session at the limit (16,385 Pages), or more than the budget allows.
        with self.assertRaisesRegex(ValueError, r'^--dram-arena-gib 16 holds 16384 Pages of 16 tokens; it must hold '
                                                r'one session at the 262144-token limit \(16385 Pages\) and less '
                                                r'than every session \(262160 Pages\)$'):
            self.prepare(tasks(16), dram_arena_gib=16, **kwargs)
        for gib in (94, 120, 255):
            with self.subTest(gib=gib), \
                    self.assertRaisesRegex(ValueError, r'^16 sessions at the 262144-token limit may pin [\d.]+ GiB .* '
                                                       r'use fewer sessions or a smaller --dram-arena-gib \(at most '
                                                       r'93 GiB\)$'):
                self.prepare(tasks(16), dram_arena_gib=gib, **kwargs)
        with self.assertRaisesRegex(ValueError, r'holds 263168 Pages'):
            self.prepare(tasks(16), dram_arena_gib=257, **kwargs)
        # --gdn-device-handoff pins a second GDN carry per session: at most 91 GiB.
        self.prepare(tasks(16), dram_arena_gib=91, gdn_device_handoff=True, **kwargs)
        with self.assertRaisesRegex(ValueError, r'^16 sessions at the 262144-token limit may pin 128\.6 GiB .* '
                                                r'\(at most 91 GiB\)$'):
            self.prepare(tasks(16), dram_arena_gib=92, gdn_device_handoff=True, **kwargs)
        # The same budget at 131072 tokens: the hint names the largest arena there too.
        with self.assertRaisesRegex(ValueError, r'^16 sessions at the 131072-token limit .* \(at most 93 GiB\)$'):
            self.prepare(tasks(16), 'full', 24, 16, max_model_len=L128, dram_arena_gib=100)


# ---- 2. startup ----------------------------------------------------------------------------------

class StartupTests(AC.Release, unittest.TestCase):
    def compose_engine(self, config):
        """compose_engine up to the worker RPC: (worker startup record, the EngineCore stub)."""
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        sent = []

        def collective_rpc(method, args, timeout):
            sent.append(args[0])
            raise Stop()
        listed = config.get('tasks') or [config['task']]
        core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                  _remote_ra_trusted_issue=dict(listed[0]), _remote_ra_trusted_issues=[dict(t) for t in listed],
                  _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                  _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
        with self.startup(config), self.assertRaises(Stop):
            composition.compose_engine(core, object())
        return sent.pop(), core

    def test_engine_record_checks_and_caps(self):
        config = self.prepare(tasks(1)[0], kv_cache_gib=24, max_model_len=L256, max_tokens=4096)[1]
        with caps_restored():
            record, core = self.compose_engine(config)
            self.assertEqual(record['context'], dict(max_model_len=L256, max_tokens=4096))
            self.assertNotIn('dram_arena_pages', record)
            self.assertEqual(caps_now(), module_caps(CAPS_256))
            self.assertEqual(getattr(core, CL.ENGINE_INPUT_ATTRIBUTE), 9 * MIB)
        self.assertEqual(CL.check_engine(NS(model_config=NS(max_model_len=L256)), self.load(config)), L256)
        with self.assertRaisesRegex(LiveV2Error, 'engine max_model_len 131072 differs from the pinned 262144'):
            CL.check_engine(NS(model_config=NS(max_model_len=L128)), self.load(config))
        config = self.prepare(tasks(16), kv_cache_gib=32, page_tokens=16, max_model_len=L256, dram_arena_gib=93)[1]
        with caps_restored():
            record, _ = self.compose_engine(config)
        self.assertEqual((record['context'], record['dram_arena_pages']),
                         (dict(max_model_len=L256, max_tokens=1536), 93 * 1024))

    def test_defaults_and_131072_set_nothing(self):
        before = caps_now()
        self.assertEqual(before, {'runtime_renderer.MAX_SOURCE_BYTES': MIB,
                                  'runtime_renderer.MAX_PROMPT_TOKENS': 131072,
                                  'registered_add_protocol.MAX_CORE_BYTES': MIB,
                                  'registered_add_protocol.MAX_WIRE_BYTES': 4 * MIB,
                                  'registered_add_protocol.MAX_TOKENS': 131072})
        for length in (65536, L128):
            config = self.prepare(tasks(1)[0], kv_cache_gib=16, max_model_len=length)[1]
            with caps_restored():
                record, core = self.compose_engine(config)
                self.assertEqual(caps_now(), before)
                self.assertFalse(hasattr(core, CL.ENGINE_INPUT_ATTRIBUTE))
                self.assertIsNone(CL.pin_request_caps(self.load(config), core))
            self.assertEqual('context' in record, length != 65536)

    def test_worker_checks_echo_arena_and_catalog(self):
        record = dict(max_model_len=L256, max_tokens=4096)
        runner = NS(max_model_len=L256)
        self.assertEqual(CL.check_worker(runner, record), record)
        self.assertEqual(CL.check_worker(runner, dict(record, max_tokens=1536)), dict(record, max_tokens=1536))
        with self.assertRaisesRegex(LiveV2Error, 'worker max_model_len 131072 differs from the pinned 262144'):
            CL.check_worker(NS(max_model_len=L128), record)
        for bad in (dict(record, max_model_len=524288), dict(record, max_model_len=262143)):
            with self.assertRaisesRegex(LiveV2Error, 'context record must hold pinned, non-default values'):
                CL.check_worker(runner, bad)
        import test_cudagraph as CGT
        attach = CGT.AttachLoadedWorkerTests().attach
        default = attach(CGT.resolved_config(None))[0]
        with patch.object(CL, 'runner_max_model_len', lambda runner: L256):
            evidence = attach(CGT.resolved_config(None), dict(context=record))[0]
        self.assertEqual(evidence, dict(default, context=record))
        from remote_ra_live_v2.worker_composition import arena_pages
        layout = I0.for_server(16, 16, True)
        self.assertEqual(arena_pages(runner, {}, layout), 16 * 16384 + 16)
        self.assertEqual(arena_pages(runner, dict(dram_arena_pages=93 * 1024), layout), 93 * 1024)
        import test_decouple_wp2 as WP2
        import test_stageb_engine as E
        from remote_ra_native_growing.adapter import EngineCoreAdapter
        from remote_ra_native_growing.contract import Authority
        with WP2.matrix_case(layout):
            kv = WP2.KVManager(layout, 2000)
            kv.max_model_len = L256
            adapter = EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), kv, page_tokens=16,
                                        full_attention_managers=(kv.fa,), layout=layout)
            self.assertEqual(adapter.page_limit, 16384)

    def test_load_refusals(self):
        config = self.prepare(tasks(4), kv_cache_gib=24, page_tokens=16, max_model_len=L256)[1]
        self.assertEqual(CL.check_config(self.load(config)), (1536, L256))
        for value in (524288, 262143, '262144'):
            effective = dict(config['effective_server_config'], max_model_len=value)
            with self.subTest(value=value), \
                    self.assertRaisesRegex(RuntimeError, '^pinned max_model_len must be one of 65536, 131072, '
                                                         '262144$'):
                self.load(dict(config, effective_server_config=effective, server_config_sha256=serve.pin(effective)))
        one = CL.session_pages(16, L256)
        self.assertEqual(one, 16385)
        self.assertEqual(self.load(dict(config, dram_arena_pages=one))['dram_arena_pages'], one)
        for pages_, message in ((one - 1, 'dram_arena_pages must hold one session at the 262144-token limit'),
                                (4 * one, 'less than 4 sessions \\(65540 Pages\\)')):
            with self.subTest(pages=pages_), self.assertRaisesRegex(RuntimeError, message):
                self.load(dict(config, dram_arena_pages=pages_))


# ---- 3. request caps -----------------------------------------------------------------------------

class RequestCapsTests(unittest.TestCase):
    def test_derived_values(self):
        import registered_add_protocol as R
        import runtime_renderer as RR
        # The sources' own values (before any pin) are SOURCE_CAPS ...
        self.assertEqual((RR.MAX_SOURCE_BYTES, R.MAX_CORE_BYTES, R.MAX_WIRE_BYTES, R.MAX_TOKENS, RR.MAX_PROMPT_TOKENS),
                         (MIB, MIB, 4 * MIB, 131072, 131072))
        self.assertEqual(CL.SOURCE_CAPS, dict(source_bytes=MIB, core_bytes=MIB, wire_bytes=4 * MIB, input_bytes=5 * MIB,
                                              prompt_tokens=131072))
        # ... and the caps of 65536 and 131072; 262144 doubles every one.
        self.assertEqual(CL.request_caps(65536), CL.SOURCE_CAPS)
        self.assertEqual(CL.request_caps(L128), CL.SOURCE_CAPS)
        self.assertEqual(CL.request_caps(L256), CAPS_256)
        for value in (32768, 262143, 524288, '262144', None):
            with self.subTest(value=value), self.assertRaisesRegex(LiveV2Error, 'context limit must be one of'):
                CL.request_caps(value)
        # Room by construction at 262144: every prompt id takes at most 5 msgpack bytes, and the
        # frame (the base64 of the base64 core plus the source) stays within 2 x (source + core).
        caps = CL.request_caps(L256)
        self.assertLess(5 * caps['prompt_tokens'] + 4096, caps['core_bytes'])
        self.assertLess(4 / 3 * (4 / 3 * caps['core_bytes'] + caps['source_bytes']) + 4096, caps['wire_bytes'])
        self.assertEqual(caps['input_bytes'] - caps['wire_bytes'], MIB)

    def test_pin_sets_once_and_agrees(self):
        config = dict(effective_server_config=dict(max_model_len=L256))
        with caps_restored():
            core = NS()
            self.assertEqual(CL.pin_request_caps(config, core), CAPS_256)
            self.assertEqual(caps_now(), module_caps(CAPS_256))
            self.assertEqual(getattr(core, CL.ENGINE_INPUT_ATTRIBUTE), 9 * MIB)
            self.assertEqual(CL.pin_request_caps(config, core), CAPS_256)       # a repeat agrees
            self.assertEqual(CL.pin_request_caps(config), CAPS_256)
            import registered_add_protocol as R
            R.MAX_WIRE_BYTES = 6 * MIB
            with self.assertRaisesRegex(LiveV2Error, 'registered_add_protocol.MAX_WIRE_BYTES is pinned for another '
                                                     'context limit'):
                CL.pin_request_caps(config)
            R.MAX_WIRE_BYTES = 8 * MIB
            with self.assertRaisesRegex(LiveV2Error, 'engine registered-add bound is pinned for another'):
                CL.pin_request_caps(config, NS(**{CL.ENGINE_INPUT_ATTRIBUTE: 5 * MIB}))
        for config in (None, {}, dict(effective_server_config=dict(max_model_len=65536)),
                       dict(effective_server_config=dict(max_model_len=L128))):
            core = NS()
            self.assertIsNone(CL.pin_request_caps(config, core))
            self.assertEqual(vars(core), {})
        self.assertEqual(caps_now(), module_caps(CL.SOURCE_CAPS))

    def test_frontend_pins_the_caps(self):
        import test_ms_frontend as F
        from remote_ra_live_v2.composition import compose_frontend
        config = dict(effective_server_config=dict(max_model_len=L256), max_tokens=4096)
        with caps_restored(), patch.object(CL, 'loaded', lambda: config):
            live = compose_frontend(F.Serving(), F.Serving(), policy='full', instance_id='inst', readiness=F.READY)
            self.assertEqual(live.max_tokens, 4096)
            self.assertEqual(caps_now(), module_caps(CAPS_256))
        for length in (65536, L128):
            config = dict(effective_server_config=dict(max_model_len=length))
            with patch.object(CL, 'loaded', lambda: config):
                compose_frontend(F.Serving(), F.Serving(), policy='full', instance_id='inst', readiness=F.READY)
            self.assertEqual(caps_now(), module_caps(CL.SOURCE_CAPS))

    def test_engine_registered_add_message_bound(self):
        decode = shipped_decode()
        with contextlib.redirect_stdout(io.StringIO()):
            # 5 MiB without the attribute: the 65536- and 131072-token servers, unchanged.
            self.assertEqual(decode(NS(), [b'x' * (5 * MIB)]), ('wire', 5 * MIB))
            with self.assertRaisesRegex(ValueError, 'bounded single registered-add frame required'):
                decode(NS(), [b'x' * (5 * MIB + 1)])
            with self.assertRaisesRegex(ValueError, 'bounded single registered-add frame required'):
                decode(NS(), [b'x', b'y'])
            # A 262144-token engine (compose_engine set the attribute): 9 MiB.
            engine = NS(**{CL.ENGINE_INPUT_ATTRIBUTE: CL.request_caps(L256)['input_bytes']})
            self.assertEqual(decode(engine, [b'x' * (9 * MIB)]), ('wire', 9 * MIB))
            with self.assertRaisesRegex(ValueError, 'bounded single registered-add frame required'):
                decode(engine, [b'x' * (9 * MIB + 1)])

    def test_codec_and_renderer_follow_the_pin(self):
        import base64
        import registered_add_protocol as R
        import runtime_renderer as RR
        codec = R.ProtocolCodec(key=b'k' * 32, epoch='e' * 64, identity='f' * 64)

        def command(core_bytes, text_bytes):
            source = dict(schema=RR.XML_SOURCE_SCHEMA, messages=[dict(role='system', content='s'),
                                                                  dict(role='user', content='x' * text_bytes)])
            return R.RegisteredAdd('c' * 64, 'r' * 80, 'x' * 80, 'a' * 64, 's' * 40,
                                   base64.b64encode(b'\x00' * core_bytes).decode(), source, {}, 300)
        big = command(int(1.5 * MIB), int(1.5 * MIB))
        source = big.source
        with self.assertRaisesRegex(Exception, 'normalized source exceeds limit'):
            RR.validate_xml_source(source)
        with self.assertRaisesRegex(Exception, 'core request byte limit exceeded'):
            codec.encode(big)
        with caps_restored():
            CL.pin_request_caps(dict(effective_server_config=dict(max_model_len=L256)))
            self.assertEqual(RR.validate_xml_source(source), source)
            frame = codec.encode(big)
            self.assertGreater(len(frame), 4 * MIB)
            self.assertEqual(codec.decode(frame), big)
            # The largest request the caps admit still fits the frame.
            full = command(2 * MIB, 2 * MIB - 200)
            self.assertLessEqual(len(codec.encode(full)), 8 * MIB)
            with self.assertRaisesRegex(Exception, 'normalized source exceeds limit'):
                RR.validate_xml_source(command(0, 2 * MIB).source)
        with self.assertRaisesRegex(Exception, 'bounded byte frame required'):
            codec.decode(frame)


# ---- 4. requests of 258,048 tokens, real tokenizer -----------------------------------------------

@unittest.skipUnless(AC._tokenizers is not None and AC.TOKENIZER_DIR, 'real tokenizer not available '
                     '(set REMOTE_RA_TOKENIZER_DIR to the pinned Qwen3.6-27B tokenizer directory)')
class TokenFrameCapTests256(unittest.TestCase):
    """Mini-SWE-Agent XML requests of 258,048 tokens (262,144 - 4,096: the longest prompt a
    262144/4096 server accepts) through the caps a 262144-token server pins: the XML
    source snapshot and the per-message encode bound (runtime_renderer), the token bound,
    the core request and the authenticated frame (registered_add_protocol) and EngineCore's
    registered-add message bound. The observation makers are test_align_ctx's (at most
    10,000 characters each), plus excerpts of this README (English text with commands and tables)."""

    @classmethod
    def setUpClass(cls):
        from hashlib import sha256
        import random
        path = Path(AC.TOKENIZER_DIR) / 'tokenizer.json'
        pins = json.loads((ROOT / 'INSTALL_RELEASE.json').read_bytes())['model_pins']
        if sha256(path.read_bytes()).hexdigest() != pins['tokenizer.json']:
            raise unittest.SkipTest('tokenizer.json differs from the pinned model metadata')
        cls.tokenizer = AC._tokenizers.Tokenizer.from_file(str(path))
        cls.rng = random.Random(20261010)
        cls.code = '\n'.join(source.read_text(encoding='utf-8')
                             for source in sorted((ROOT / 'source-overlay' / 'vllm' / 'v1').rglob('*.py')))
        cls.readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        cls.sizes = {}

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('REMOTE_RA_CAPS_REPORT'):
            Path(os.environ['REMOTE_RA_CAPS_REPORT']).write_text(json.dumps(cls.sizes, indent=1, sort_keys=True))

    encode = AC.TokenFrameCapTests.encode
    cat_n = AC.TokenFrameCapTests.cat_n
    pytest_v = AC.TokenFrameCapTests.pytest_v
    fits_cards = AC.TokenFrameCapTests.fits_cards
    banners = AC.TokenFrameCapTests.banners
    conversation = AC.TokenFrameCapTests.conversation

    def readme_text(self):
        i = self.rng.randrange(0, len(self.readme) - 12000)
        return self.readme[i:i + 9800]

    def through_the_caps(self, messages, ids):
        """Every cap a request of these messages and ids meets at the pinned caps; the sizes."""
        import base64
        import registered_add_protocol as R
        import runtime_renderer as RR
        xml.validate_messages(messages)
        source = RR.validate_xml_source(dict(schema=RR.XML_SOURCE_SCHEMA, messages=messages))
        for message in messages:
            self.assertLessEqual(len(xml.message_text(message['role'], message['content']).encode()),
                                 RR.MAX_SOURCE_BYTES)
        self.assertLessEqual(len(ids), R.MAX_TOKENS)
        self.assertLessEqual(len(ids), RR.MAX_PROMPT_TOKENS)
        core = AC._msgpack_ints(ids) + b'\x00' * 4096        # the prompt ids plus a bound for the other fields
        self.assertLessEqual(len(core), R.MAX_CORE_BYTES)
        codec = R.ProtocolCodec(key=b'k' * 32, epoch='e' * 64, identity='f' * 64)
        command = R.RegisteredAdd(command_id='c' * 64, request_id='r' * 80, external_id='x' * 80,
                                  prompt_hash='a' * 64, session_id='s' * 40,
                                  core_b64=base64.b64encode(core).decode(), source=source,
                                  metadata=dict(instance_id='s' * 40, turn=119, generation='g' * 64),
                                  lease_seconds=300)
        frame = codec.encode(command)
        self.assertEqual(codec.decode(frame).source, source)
        # EngineCore's message: the frame plus msgpack's struct header (well under 64 bytes).
        self.assertLessEqual(len(frame) + 64, CL.request_caps(L256)['input_bytes'])
        snapshot = len(json.dumps(source, ensure_ascii=False, separators=(',', ':')).encode())
        return dict(tokens=len(ids), source_bytes=snapshot, core_bytes=len(core), frame_bytes=len(frame),
                    source_bytes_per_token=round(snapshot / len(ids), 3))

    def test_swe_like_requests_at_258k_fit_every_cap(self):
        import runtime_renderer as RR
        mixes = (('code', [self.cat_n]), ('padded', [self.pytest_v, self.fits_cards, self.cat_n]),
                 ('fits', [self.fits_cards]), ('readme', [self.readme_text]))
        for name, makers in mixes:
            with self.subTest(mix=name):
                messages, ids = self.conversation(makers, L256 - 4096)
                self.assertGreaterEqual(len(ids), L256 - 4096 - 16)
                with caps_restored():
                    CL.pin_request_caps(dict(effective_server_config=dict(max_model_len=L256)))
                    sizes = self.through_the_caps(messages, ids)
                self.sizes[name] = sizes
                self.assertLess(sizes['source_bytes'], 0.75 * CAPS_256['source_bytes'], sizes)
                self.assertLess(sizes['core_bytes'], 0.5 * CAPS_256['core_bytes'], sizes)
                self.assertLess(sizes['frame_bytes'], 0.5 * CAPS_256['wire_bytes'], sizes)
                if sizes['source_bytes'] > CL.SOURCE_CAPS['source_bytes']:
                    # Such a history needs the scaled cap: the 1 MiB source cap refuses it.
                    with self.assertRaisesRegex(Exception, 'normalized source exceeds limit'):
                        RR.validate_xml_source(dict(schema=RR.XML_SOURCE_SCHEMA, messages=messages))
        # FITS header cards (about 5.3 bytes per token) pass 1 MiB at 258K tokens.
        self.assertGreater(self.sizes['fits']['source_bytes'], CL.SOURCE_CAPS['source_bytes'])

    def test_banner_only_logs_still_hit_the_source_cap(self):
        """The 2 MiB source cap binds above 8.13 bytes per token at 258,048 tokens: banner-only
        logs (about 19 bytes per token) are still refused before admission."""
        import runtime_renderer as RR
        messages, ids = self.conversation([self.banners], L256 - 4096)
        with caps_restored():
            CL.pin_request_caps(dict(effective_server_config=dict(max_model_len=L256)))
            with self.assertRaisesRegex(Exception, 'normalized source exceeds limit'):
                RR.validate_xml_source(dict(schema=RR.XML_SOURCE_SCHEMA, messages=messages))


# ---- 5. with --decode-fastpath -------------------------------------------------------------------

# The launcher options of job 3360375's servers at 262144 tokens with 4096-token replies:
# k-L1-full (one session, `full`, eager, 24 GiB, P = 16) and k-B16-obs-full (16 sessions,
# observation_query with an 8192-token budget, --cuda-graphs full, 32 GiB, P = 16, an
# 88-GiB arena); `legacy` is one session in the shared 784-token pool.
SERVERS_256 = {
    'k-L1-full': (tasks(1)[0], 'full', 24, 16, None, None, {}),
    'k-B16-obs-full': (tasks(16), 'observation_query', 32, 16, None, dict(active_token_budget=8192),
                       dict(cuda_graphs='full', dram_arena_gib=88)),
    'legacy': (tasks(1)[0], 'full', 24, 784, None, None, {}),
}
CONTEXT_256 = dict(max_model_len=L256, max_tokens=4096)


class FastPathTests(AC.Release, unittest.TestCase):
    """--max-model-len 262144 with --decode-fastpath: each pin travels and is checked as
    without the other (the fast path adds its one key; the context limit its record, caps
    and arena)."""
    maxDiff = None

    def server(self, name, fast, out):
        task, policy, gib, P, layout, selection, extra = SERVERS_256[name]
        kwargs = dict(extra, max_model_len=L256, max_tokens=4096, **({'decode_fastpath': True} if fast else {}))
        return self.prepare(task, policy, gib, P, layout, selection, out=out, **kwargs)

    def test_launcher_config_holds_both_pins(self):
        for name in SERVERS_256:
            with self.subTest(server=name):
                m0, c0, a0, e0 = self.server(name, False, 'fp-' + name)
                m1, c1, a1, e1 = self.server(name, True, 'fp-' + name)
                self.assertEqual(m1, m0)
                self.assertEqual(a1, a0)                         # vLLM's command: the 262144 one
                self.assertEqual(a1[a1.index('--max-model-len') + 1], '262144')
                self.assertEqual(c1, dict(c0, decode_fastpath=True))
                self.assertEqual((c1['effective_server_config']['max_model_len'], c1['max_tokens']), (L256, 4096))
                self.assertEqual(c1.get('dram_arena_pages'), 88 * 1024 if name == 'k-B16-obs-full' else None)
                self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                loaded = self.load(c1)
                self.assertEqual(CL.check_config(loaded), (4096, L256))
                self.assertEqual(CL.record(loaded), CONTEXT_256)
                self.assertEqual(CL.request_caps(CL.max_model_len(loaded)), CAPS_256)
                self.assertIs(F.pinned(loaded), True)
                with self.startup(c1):
                    self.assertIs(F.loaded(), True)
                    self.assertEqual(CL.loaded(), loaded)

    def test_refusals_at_262144_are_unchanged_by_the_switch(self):
        for task, kwargs in ((tasks(1)[0], dict(kv_cache_gib=16, page_tokens=16)),           # attention pool
                             (tasks(16), dict(kv_cache_gib=24, page_tokens=16)),             # host, no arena
                             (tasks(16), dict(kv_cache_gib=32, page_tokens=16, dram_arena_gib=94)),
                             (tasks(6), dict(kv_cache_gib=24))):                             # shared pool, host
            for T in serve.MAX_TOKENS_CHOICES:
                with self.subTest(kwargs=kwargs, T=T):
                    off = self.outcome(task, max_model_len=L256, max_tokens=T, **kwargs)
                    on = self.outcome(task, max_model_len=L256, max_tokens=T, decode_fastpath=True, **kwargs)
                    self.assertEqual(off[0], 'ValueError')
                    self.assertEqual(on, off)

    def test_dry_run_main_with_both(self):
        task = self.root / 'task-fp16.json'
        task.write_text(json.dumps(tasks(16)))
        base = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--policy', 'observation_query',
                '--output', str(self.root / 'dry-fp'), '--page-tokens', '16', '--kv-cache-gib', '32',
                '--active-token-budget', '8192', '--cuda-graphs', 'full', '--dram-arena-gib', '88',
                '--max-model-len', '262144', '--max-tokens', '4096']
        prepare, configs = serve.prepare, []

        def recording(*args, **kwargs):
            configs.append(prepare(*args, **kwargs))
            return configs[-1]

        def main(*argv):
            out = io.StringIO()
            with patch.object(sys, 'argv', base + list(argv)), contextlib.redirect_stdout(out), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path), \
                    patch.object(serve, 'prepare', recording):
                serve.main()
            return json.loads(out.getvalue()), configs[-1][1]
        (plain, off), (fast, on) = main(), main('--decode-fastpath')
        self.assertEqual(fast, plain)
        self.assertEqual((plain['status'], plain['starts_model']), ('SERVICE_DRY_RUN', False))
        self.assertEqual(plain['command'][plain['command'].index('--max-model-len') + 1], '262144')
        self.assertEqual(on, dict(off, decode_fastpath=True))
        self.assertEqual((on['effective_server_config']['max_model_len'], on['max_tokens'], on['dram_arena_pages'],
                          len(on['tasks'])), (L256, 4096, 88 * 1024, 16))

    def compose_engine(self, config, reply):
        """compose_engine with a worker that answers ``reply``: (its startup record, the
        EngineCore stub, the refusal). The stub cannot attach, so every run is refused."""
        from remote_ra_live_v2 import composition
        sent = []

        def collective_rpc(method, args, timeout):
            sent.append(args[0])
            return [reply]
        listed = config.get('tasks') or [config['task']]
        core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                  _remote_ra_trusted_issue=dict(listed[0]), _remote_ra_trusted_issues=[dict(t) for t in listed],
                  _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                  _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
        with self.startup(config), self.assertRaises(LiveV2Error) as caught:
            composition.compose_engine(core, object())
        startup, = sent
        return startup, core, str(caught.exception)

    def test_engine_sends_and_checks_both(self):
        echo = dict(context=CONTEXT_256, decode_fastpath=True)
        for name in ('k-L1-full', 'k-B16-obs-full'):
            config = self.server(name, True, 'fp-engine-' + name)[1]
            off = self.server(name, False, 'fp-engine-' + name)[1]
            with self.subTest(server=name), caps_restored():
                startup, core, refusal = self.compose_engine(config, echo)
                # Both echoes pass; the next check (the attach receipt) refuses the stub's reply.
                self.assertIn('loaded CUDA worker attachment receipt', refusal)
                self.assertEqual((startup['context'], startup['decode_fastpath']), (CONTEXT_256, True))
                self.assertEqual(startup.get('dram_arena_pages'), config.get('dram_arena_pages'))
                self.assertEqual(caps_now(), module_caps(CAPS_256))
                self.assertEqual(getattr(core, CL.ENGINE_INPUT_ATTRIBUTE), 9 * MIB)
                self.assertEqual(startup, dict(self.compose_engine(off, dict(context=CONTEXT_256))[0],
                                               decode_fastpath=True))
                for reply, message in ((dict(context=CONTEXT_256), 'worker attached with another decode_fastpath'),
                                       (dict(decode_fastpath=True), 'worker context record differs'),
                                       (dict(echo, context=dict(CONTEXT_256, max_model_len=L128)),
                                        'worker context record differs')):
                    self.assertIn(message, self.compose_engine(config, reply)[2])

    def test_worker_echoes_both(self):
        import test_cudagraph as CGT

        class Owner(CGT.FakeOwner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.claims = {}
        tests = CGT.AttachLoadedWorkerTests()
        with patch.object(CGT, 'FakeOwner', Owner), patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'), \
                patch.object(CL, 'runner_max_model_len', lambda runner: L256):
            for compilation, graphs, bound in ((CGT.resolved_config(None), {}, ()),
                                               (CGT.resolved_config('FULL_AND_PIECEWISE'), dict(cuda_graphs='full'),
                                                (0,))):
                with self.subTest(graphs=graphs):
                    plain = tests.attach(compilation, dict(graphs, context=CONTEXT_256), bound)[0]
                    evidence, runner, _ = tests.attach(compilation, dict(graphs, context=CONTEXT_256,
                                                                         decode_fastpath=True), bound)
                    self.assertEqual(evidence, dict(plain, decode_fastpath=True))
                    self.assertEqual(evidence['context'], CONTEXT_256)
                    owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
                    self.assertIsInstance(owner.claims, F.ClaimTable)
                    self.assertTrue(bridge.decode_fastpath and bridge.qwen.decode_fastpath)
            with patch.object(CL, 'runner_max_model_len', lambda runner: L128), \
                    self.assertRaisesRegex(LiveV2Error, 'worker max_model_len 131072 differs from the pinned 262144'):
                tests.attach(CGT.resolved_config(None), dict(context=CONTEXT_256, decode_fastpath=True))
        # On the HTTP runtime the worker compares the record with its own reading of the pin.
        with self.startup(self.server('k-B16-obs-full', True, 'fp-worker')[1]):
            self.assertIs(F.worker_flag(dict(decode_fastpath=True)), True)
            with self.assertRaisesRegex(LiveV2Error, 'worker decode_fastpath differs from the pinned startup config'):
                F.worker_flag({})

    def test_frontend_and_ready_record_both(self):
        import pathlib
        import test_ms_frontend as MF
        from remote_ra_http.evidence import FrontendEvidence
        from remote_ra_live_v2.composition import compose_frontend
        config = self.server('legacy', True, 'fp-ready')[1]
        loaded = self.load(config)
        with caps_restored(), patch.object(CL, 'loaded', lambda: loaded):
            live = compose_frontend(MF.Serving(), MF.Serving(), policy='full', instance_id='inst', readiness=MF.READY)
            self.assertEqual(live.max_tokens, 4096)
            self.assertEqual(caps_now(), module_caps(CAPS_256))
        effective = dict(config['effective_server_config'])
        with patch.object(http_config, 'load', return_value=loaded):
            self.assertEqual(http_config.effective_server_config(NS(**effective))['max_model_len'], L256)
        readiness = dict(native_ready=True)
        authority = NS(identity=NS(epoch=MF.EPOCH, manifest='m' * 64), policy='full', session_id=MF.SERVER,
                       verify_admission=lambda: True, live_v2_readiness=readiness,
                       loaded=NS(manifest=dict(qualification_parent_manifest='q' * 64)))
        read_text = pathlib.Path.read_text

        def proc(path, *args, **kwargs):
            name = str(path).replace('\\', '/')
            if name == '/proc/sys/kernel/random/boot_id':
                return 'boot\n'
            if name == '/proc/self/stat':
                return '1 (python) ' + ' '.join(['0'] * 30)
            return read_text(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            value = dict(loaded, output_directory=str(output))

            def ready(receipt):
                (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                    epoch=MF.EPOCH, report=readiness,
                    receipt=dict(tensor_geometry={}, source_closure={}, **receipt))).encode())
                with patch('remote_ra_http.evidence.load', lambda: value), \
                        patch.object(pathlib.Path, 'read_text', proc):
                    evidence = FrontendEvidence(authority, None, effective)
                    evidence.write_ready(NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]),
                                         NS(started=True))
                return json.loads((output / 'ready.json').read_bytes())
            for receipt, message in ((dict(context=CONTEXT_256), 'worker/parent decode_fastpath mismatch'),
                                     (dict(decode_fastpath=True), 'worker/parent context record mismatch')):
                with self.subTest(receipt=receipt), self.assertRaisesRegex(Exception, message):
                    ready(receipt)
            self.assertFalse((output / 'ready.json').exists())
            written = ready(dict(context=CONTEXT_256, decode_fastpath=True))
        self.assertEqual((written['context'], written['decode_fastpath']), (CONTEXT_256, True))
        self.assertEqual(written['effective_server_config']['max_model_len'], L256)


if __name__ == '__main__':
    unittest.main()
