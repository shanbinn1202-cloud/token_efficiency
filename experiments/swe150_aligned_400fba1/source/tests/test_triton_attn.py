"""serve.py --triton-attn (remote_ra_live_v2.attention_backend). CPU only; not GPU evidence.

Off (the default) adds no vLLM argument, startup config key, worker record or evidence
key, and the explicit False equals the default. On (the decoupled layout at 16-, 32- or
64-token Pages):

- launcher: the vLLM command gains ``--attention-backend TRITON_ATTN`` and nothing else
  (``--language-model-only`` stays last); startup.json pins ``attention_backend`` and
  ``effective_server_config.attention_backend``; the KV and pinned-host arithmetic accepts
  and refuses the same launches; the shared pool and the decoupled layout's 784-token
  validation size are refused; on a Gemma 4 release (always TRITON_ATTN) the switch changes
  nothing;
- startup checks: ``remote_ra_http.config.load`` refuses a pin without its vLLM argument
  and the reverse; the frontend compares vLLM's parsed backend by name (and refuses a parsed
  TRITON_ATTN when the pin is off); compose_engine sends the pin once and checks the
  worker's echo; the engine's and the worker's resolved backend must be TRITON_ATTN and
  every FullAttention group must have loaded vLLM's Triton backend (and a Qwen vLLM that
  resolved TRITON_ATTN without the pin is refused); ready.json records the worker's record;
- the pinned vLLM 0.24.0 sources behind the documentation (overlay copies, the stock
  wheel's bytes): FlashAttention's cache shape and its hybrid kernel block sizes, the
  decoupled block-size check, the runner's metadata rewrite before any backend's builder.
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

import install
import serve
import test_align_ctx as AC
import test_cudagraph as TCG
import test_language_model_only as LM

from remote_ra_http import config as http_config
from remote_ra_live_v2 import attention_backend as T
from remote_ra_live_v2.contract import LiveV2Error

tasks = AC.tasks
PAIR = ['--attention-backend', 'TRITON_ATTN']
L128 = 131072
CASES = (  # task, policy, GiB, P, layout, selection, extra flags (all decoupled at 16/32/64)
    (tasks(3), 'goal_query', 24, 16, 'decoupled', dict(no_goal_rewriter=True), {}),
    (tasks(16), 'random_fill', 32, 16, None, None, dict(cuda_graphs='full')),
    (tasks(24), 'full', 24, 16, None, None, dict(max_running=8)),
    (tasks(16), 'observation_query', 24, 16, 'decoupled', None,
     dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, cuda_graphs='full',
          language_model_only=True, release_idle_pages=True, connector_pool_gib=1)),
    (tasks(4), 'full', 16, 32, None, None, dict(cuda_graphs='full-decode')),
    (tasks(2), 'observation_query', 16, 64, None, dict(active_token_budget=8192), dict(max_model_len=L128)),
    (tasks(16), 'full', 24, 16, None, None, dict(max_model_len=L128)),               # refused: host budget
)
REFUSED_LAYOUTS = (  # the shared pool and the decoupled validation size: FlashAttention's 49 x 16 kernel blocks
    (tasks(1)[0], 'full', 8, 784, None, None, {}),
    (tasks(8), 'observation_query', 24, 784, 'decoupled', dict(active_token_budget=8192), {}),
)


class Enum:
    """vLLM's AttentionBackendEnum member as the checks read it (by name)."""

    def __init__(self, name):
        self.name = name


def vllm_config(backend):
    return NS(attention_config=NS(backend=backend))


class FullAttentionSpec:
    pass


class MambaSpec:
    pass


def backend_class(name, module):
    cls = type('Backend', (), {'get_name': staticmethod(lambda: name)})
    cls.__module__ = module
    return cls


TRITON_BACKEND = backend_class('TRITON_ATTN', T.MODULE)
FLASH_BACKEND = backend_class('FLASH_ATTN', 'vllm.v1.attention.backends.flash_attn')
GDN_BACKEND = backend_class('GDN_ATTN', 'vllm.v1.attention.backends.gdn_attn')


def groups(*attention, mamba=3):
    """A Qwen-shaped runner's attention groups: the FullAttention groups' backends, then GDN groups."""
    found = [NS(kv_cache_group_id=g, kv_cache_spec=FullAttentionSpec(), backend=b) for g, b in enumerate(attention)]
    found += [NS(kv_cache_group_id=len(attention) + k, kv_cache_spec=MambaSpec(), backend=GDN_BACKEND)
              for k in range(mamba)]
    return lambda: iter(found)


def triton_runner(**extra):
    """Runner attributes of a vLLM worker on TRITON_ATTN: resolved backend, Triton FullAttention group."""
    return dict(dict(vllm_config=vllm_config(Enum('TRITON_ATTN')),
                     _kv_cache_spec_attn_group_iterator=groups(TRITON_BACKEND)), **extra)


ON = {'attention_backend': 'TRITON_ATTN', 'effective_server_config': {'attention_backend': 'TRITON_ATTN'},
      'page_tokens': 16, 'kv_layout': {'state': 'mamba'}}


# ---- 1. the pin and the resolved-backend checks --------------------------------------------------

class PinTests(unittest.TestCase):
    def test_pin_values(self):
        self.assertIsNone(T.pinned(None))
        self.assertIsNone(T.pinned({}))
        self.assertIsNone(T.pinned({'effective_server_config': {'seed': 42}}))
        # A Gemma 4 config pins its backend in effective_server_config, never this key.
        gemma = {'model_profile': 'gemma4-31b-it', 'effective_server_config': {'attention_backend': 'TRITON_ATTN'}}
        self.assertIsNone(T.pinned(gemma))
        self.assertEqual(T.pinned(ON), 'TRITON_ATTN')
        for P in (32, 64):
            self.assertEqual(T.pinned(dict(ON, page_tokens=P)), 'TRITON_ATTN')
        refused = [dict(ON, attention_backend=bad) for bad in ('FLASH_ATTN', 'triton', True, None, '')]
        refused += [dict(ON, effective_server_config={}), dict(ON, effective_server_config=None),
                    dict(ON, effective_server_config={'attention_backend': 'FLASH_ATTN'}),
                    {'effective_server_config': {'attention_backend': 'TRITON_ATTN'}},
                    {'effective_server_config': {'attention_backend': 'FLASH_ATTN'}},
                    dict(ON, page_tokens=784), dict(ON, kv_layout=None), {k: v for k, v in ON.items() if k != 'kv_layout'},
                    dict(gemma, attention_backend='TRITON_ATTN')]
        for value in refused:
            with self.subTest(value=value), self.assertRaises(LiveV2Error):
                T.pinned(value)

    def test_record_is_the_checked_facts(self):
        self.assertEqual(T.RECORD, {'backend': 'TRITON_ATTN', 'module': 'vllm.v1.attention.backends.triton_attn'})
        self.assertIsNone(T.record({}))
        record = T.record(ON)
        self.assertEqual(record, T.RECORD)
        record['backend'] = 'x'                                # a copy: the constant is untouched
        self.assertEqual(T.RECORD['backend'], 'TRITON_ATTN')

    def test_engine_check(self):
        for off_ok in (None, NS(), NS(attention_config=None), vllm_config(None), vllm_config(Enum('FLASH_ATTN')),
                       vllm_config('FLASH_ATTN')):
            with self.subTest(off=off_ok):
                self.assertIsNone(T.check_engine(off_ok, {}))
                self.assertIsNone(T.check_engine(off_ok, None))
        for triton in (vllm_config(Enum('TRITON_ATTN')), vllm_config('TRITON_ATTN')):
            with self.assertRaisesRegex(LiveV2Error, "pins vLLM's own choice"):
                T.check_engine(triton, {})
            self.assertEqual(T.check_engine(triton, ON), 'TRITON_ATTN')
            # A Gemma 4 engine always resolves TRITON_ATTN; its profile's checks cover it.
            self.assertIsNone(T.check_engine(triton, {'model_profile': 'gemma4-31b-it'}))
        for bad in (vllm_config(None), vllm_config(Enum('FLASH_ATTN')), NS()):
            with self.subTest(bad=bad), self.assertRaisesRegex(LiveV2Error, 'pinned TRITON_ATTN'):
                T.check_engine(bad, ON)

    def test_worker_check_reads_every_full_attention_group(self):
        runner = NS(**triton_runner())
        self.assertEqual(T.worker_check(runner, {'attention_backend': 'TRITON_ATTN'}), T.RECORD)
        self.assertEqual(T.full_attention_backends(runner), [(0, 'TRITON_ATTN', T.MODULE)])
        two = NS(**triton_runner(_kv_cache_spec_attn_group_iterator=groups(TRITON_BACKEND, TRITON_BACKEND)))
        self.assertEqual(T.worker_check(two, {'attention_backend': 'TRITON_ATTN'}), T.RECORD)
        other_module = backend_class('TRITON_ATTN', 'somewhere.else')
        for runner, message in (
                (NS(**triton_runner(vllm_config=vllm_config(None))), 'resolved attention backend None'),
                (NS(**triton_runner(vllm_config=vllm_config(Enum('FLASH_ATTN')))), "resolved attention backend 'FLASH"),
                (NS(**triton_runner(_kv_cache_spec_attn_group_iterator=groups(FLASH_BACKEND))), 'another attention backend'),
                (NS(**triton_runner(_kv_cache_spec_attn_group_iterator=groups(TRITON_BACKEND, FLASH_BACKEND))),
                 'another attention backend'),
                (NS(**triton_runner(_kv_cache_spec_attn_group_iterator=groups(other_module))), 'another attention backend'),
                (NS(**triton_runner(_kv_cache_spec_attn_group_iterator=groups())), 'no FullAttention group'),
                (NS(vllm_config=vllm_config(Enum('TRITON_ATTN'))), "needs the loaded runner's attention groups")):
            with self.subTest(message=message), self.assertRaisesRegex(LiveV2Error, message):
                T.worker_check(runner, {'attention_backend': 'TRITON_ATTN'})
        for bad in ('FLASH_ATTN', True, T.RECORD):
            with self.subTest(record=bad), self.assertRaisesRegex(LiveV2Error, 'must be TRITON_ATTN or absent'):
                T.worker_check(NS(**triton_runner()), {'attention_backend': bad})
        # Off: nothing is read but the resolved backend, which must not be TRITON_ATTN.
        self.assertIsNone(T.worker_check(NS(), {}))
        self.assertIsNone(T.worker_check(NS(vllm_config=vllm_config(Enum('FLASH_ATTN'))), {}))
        with self.assertRaisesRegex(LiveV2Error, "pins vLLM's own choice"):
            T.worker_check(NS(**triton_runner()), {})


# ---- 2. launcher, startup config, frontend, engine ------------------------------------------------

class LauncherTests(AC.Release, unittest.TestCase):
    maxDiff = None

    def test_off_is_the_default(self):
        for task, policy, gib, P, layout, selection, extra in CASES + REFUSED_LAYOUTS + AC.CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                plain = self.outcome(task, policy, gib, P, layout, selection, out='off', **extra)
                explicit = self.outcome(task, policy, gib, P, layout, selection, out='off', triton_attn=False,
                                        **extra)
                self.assertEqual(explicit, plain)
                if plain[0] == 'ok':
                    manifest, config, command, env = plain[1]
                    self.assertNotIn('--attention-backend', command)
                    self.assertNotIn('attention_backend', config)
                    self.assertNotIn('attention_backend', config['effective_server_config'])

    def test_on_adds_one_vllm_argument_and_pins_it(self):
        kinds = set()
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                off = self.outcome(task, policy, gib, P, layout, selection, out='tr', **extra)
                on = self.outcome(task, policy, gib, P, layout, selection, out='tr', triton_attn=True, **extra)
                kinds.add(off[0])
                if off[0] != 'ok':
                    self.assertEqual(on, off)               # the same KV / host refusals
                    continue
                (m0, c0, a0, e0), (m1, c1, a1, e1) = off[1], on[1]
                self.assertEqual(m1, m0)
                tail = 1 if extra.get('language_model_only') else 0      # --language-model-only stays last
                cut = len(a0) - tail
                self.assertEqual(a1, a0[:cut] + PAIR + a0[cut:])
                self.assertEqual(a1[-1] == '--language-model-only', bool(tail))
                effective = dict(c0['effective_server_config'], attention_backend='TRITON_ATTN')
                self.assertEqual(c1['effective_server_config'], effective)
                self.assertEqual(c1, dict(c0, attention_backend='TRITON_ATTN', effective_server_config=effective,
                                          server_config_sha256=serve.pin(effective)))
                self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                self.assertEqual(effective['kv_cache_memory_bytes'], gib * 1024 ** 3)
                self.assertEqual(effective['block_size'], P)
        self.assertEqual(kinds, {'ok', 'ValueError'})

    def test_refused_outside_the_decoupled_pages(self):
        for task, policy, gib, P, layout, selection, extra in REFUSED_LAYOUTS:
            with self.subTest(P=P, layout=layout):
                self.assertEqual(self.outcome(task, policy, gib, P, layout, selection, triton_attn=True, **extra),
                                 ('ValueError', '--triton-attn needs the decoupled KV layout at 16-, 32- or 64-token '
                                                'Pages (--page-tokens 16, 32 or 64)'))
                self.assertEqual(self.outcome(task, policy, gib, P, layout, selection, **extra)[0], 'ok')

    def test_refuses_anything_but_a_switch(self):
        for bad in (1, 'yes', None, 0, 'TRITON_ATTN'):
            with self.subTest(bad=bad):
                self.assertEqual(self.outcome(tasks(1)[0], 'full', 24, 16, triton_attn=bad),
                                 ('ValueError', '--triton-attn is a switch'))

    def test_dry_run_main(self):
        task = self.root / 'dry-task.json'
        task.write_text(json.dumps(tasks(1)[0]))

        def main(*extra):
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task), '--output',
                    str(self.root / 'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream):
                serve.main()
            return json.loads(stream.getvalue())
        for extra in (('--page-tokens', '16', '--kv-cache-gib', '24'),
                      ('--page-tokens', '16', '--kv-cache-gib', '24', '--cuda-graphs', 'full', '--decode-fastpath',
                       '--max-model-len', '131072')):
            with self.subTest(extra=extra):
                plain, flagged = main(*extra), main(*extra, '--triton-attn')
                self.assertEqual(flagged, dict(plain, command=plain['command'] + PAIR))
                self.assertNotIn('--attention-backend', plain['command'])
        with self.assertRaises(ValueError):
            main('--triton-attn')                       # the shared pool

    def test_load_accepts_the_pin_and_refuses_mismatches(self):
        off = self.prepare(tasks(3), 'observation_query', 24, 16, 'decoupled', out='load')[1]
        on = self.prepare(tasks(3), 'observation_query', 24, 16, 'decoupled', out='load', triton_attn=True)[1]
        self.assertEqual(self.load(on)['attention_backend'], 'TRITON_ATTN')
        self.assertNotIn('attention_backend', self.load(off))
        with self.startup(on):
            self.assertEqual(T.loaded(), 'TRITON_ATTN')
        with self.startup(off):
            self.assertIsNone(T.loaded())
        self.assertIsNone(T.loaded())                    # no HTTP runtime
        effective_on, effective_off = on['effective_server_config'], off['effective_server_config']
        legacy = self.prepare(tasks(1)[0], out='load-legacy')[1]
        bad = {'pin without the vLLM argument': dict(on, effective_server_config=effective_off),
               'vLLM argument without the pin': dict(off, effective_server_config=effective_on),
               'another backend': dict(on, attention_backend='FLASH_ATTN'),
               'argument another backend': dict(on, effective_server_config=dict(effective_on,
                                                                                 attention_backend='FLASH_ATTN')),
               'shared pool': dict(legacy, attention_backend='TRITON_ATTN', effective_server_config=dict(
                   legacy['effective_server_config'], attention_backend='TRITON_ATTN'))}
        for name, value in bad.items():
            with self.subTest(name):
                value = dict(value, server_config_sha256=serve.pin(value['effective_server_config']))
                with self.assertRaisesRegex(RuntimeError, 'attention_backend'):
                    self.load(value)

    def test_frontend_compares_the_parsed_backend(self):
        off = self.prepare(tasks(1)[0], 'full', 24, 16, out='front')[1]
        on = self.prepare(tasks(1)[0], 'full', 24, 16, out='front', triton_attn=True)[1]

        def frontend(config, args):
            with self.startup(config):
                try:
                    return http_config.effective_server_config(NS(**args))
                except RuntimeError as error:
                    return ('refused', str(error))
        effective = on['effective_server_config']
        self.assertEqual(frontend(on, effective), effective)
        self.assertEqual(frontend(on, dict(effective, attention_backend=Enum('TRITON_ATTN'))), effective)  # vLLM's enum
        for args in (dict(effective, attention_backend=Enum('FLASH_ATTN')), dict(effective, attention_backend=None),
                     {k: v for k, v in effective.items() if k != 'attention_backend'}):
            with self.subTest(args=args.get('attention_backend', '<absent>')):
                outcome = frontend(on, args)
                self.assertEqual(outcome[0], 'refused')
                self.assertIn('observed API server arguments differ', outcome[1])
        effective = off['effective_server_config']
        self.assertEqual(frontend(off, effective), effective)
        for vllm_choice in (None, Enum('FLASH_ATTN'), 'FLASH_ATTN'):            # never compared when unpinned
            self.assertEqual(frontend(off, dict(effective, attention_backend=vllm_choice)), effective)
        for parsed in ('TRITON_ATTN', Enum('TRITON_ATTN')):
            self.assertEqual(frontend(off, dict(effective, attention_backend=parsed)),
                             ('refused', "vLLM parsed --attention-backend TRITON_ATTN but the startup config pins "
                                         "vLLM's own choice"))

    def startup_record(self, config, echo):
        """compose_engine's worker startup record and its refusal of the worker's reply ``echo``."""
        from remote_ra_live_v2 import composition
        sent = []

        def collective_rpc(method, args, timeout):
            sent.append(args[0])
            return [echo]
        task = config['task']
        core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=16))),
                  _remote_ra_trusted_issue=dict(task), _remote_ra_trusted_issues=[dict(task)],
                  _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                  _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
        with self.startup(config), self.assertRaises(LiveV2Error) as caught:
            composition.compose_engine(core, object())
        startup, = sent
        return startup, str(caught.exception)

    def test_compose_engine_sends_the_pin_once_and_checks_the_echo(self):
        base = dict(enabled=True, source='m' * 64, page_tokens=16, loaded_sources_verified=True)
        next_check = 'loaded CUDA worker attachment receipt'       # the check after the echoes
        off = self.prepare(tasks(1)[0], 'full', 24, 16, out='ce')[1]
        on = self.prepare(tasks(1)[0], 'full', 24, 16, out='ce', triton_attn=True)[1]
        pinned = dict(base, attention_backend='TRITON_ATTN')
        for config, echo, record, message in (
                (off, {}, base, next_check),
                (off, {'attention_backend': T.RECORD}, base, 'another attention_backend'),
                (on, {}, pinned, 'another attention_backend'),
                (on, {'attention_backend': 'TRITON_ATTN'}, pinned, 'another attention_backend'),
                (on, {'attention_backend': T.RECORD}, pinned, next_check)):
            with self.subTest(on='attention_backend' in config, echo=echo):
                startup, error = self.startup_record(config, echo)
                self.assertEqual(startup, record)
                self.assertIn(message, error)


class GemmaLauncherTests(unittest.TestCase):
    """The Gemma 4 profile always runs TRITON_ATTN: the switch changes nothing there."""

    def test_switch_is_a_no_op_on_gemma(self):
        import test_gemma_capacity as GC
        case = GC.LauncherTests('test_vllm_command')
        case.setUp()
        try:
            plain = case.prepare(GC.tasks(4), 'observation_query', dict(active_token_budget=8192))
            case.n -= 1                                            # the same output directory
            flagged = case.prepare(GC.tasks(4), 'observation_query', dict(active_token_budget=8192),
                                   triton_attn=True)
        finally:
            case.doCleanups()
        self.assertEqual(flagged, plain)
        manifest, config, command, env = plain
        self.assertEqual(command[command.index('--attention-backend') + 1], 'TRITON_ATTN')
        self.assertNotIn('attention_backend', config)
        self.assertEqual(config['effective_server_config']['attention_backend'], 'TRITON_ATTN')


# ---- 3. worker ------------------------------------------------------------------------------------

class WorkerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        patcher = patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_off_evidence_is_unchanged(self):
        plain = TCG.AttachLoadedWorkerTests().attach(TCG.resolved_config(None))[0]
        for runner in ({}, dict(vllm_config=vllm_config(None)), dict(vllm_config=vllm_config(Enum('FLASH_ATTN')),
                                                                      _kv_cache_spec_attn_group_iterator=None)):
            with self.subTest(runner=sorted(runner)):
                self.assertEqual(LM.attach(None, **runner)[0], plain)
        self.assertNotIn('attention_backend', plain)

    def test_off_refuses_a_triton_vllm(self):
        with self.assertRaisesRegex(LiveV2Error, "pins vLLM's own choice"):
            LM.attach(None, **triton_runner())

    def test_on_evidence_gains_only_the_record(self):
        plain = LM.attach(None)[0]
        evidence, runner = LM.attach(dict(attention_backend='TRITON_ATTN'), **triton_runner())
        self.assertEqual(evidence, dict(plain, attention_backend=T.RECORD))
        both = LM.attach(dict(attention_backend='TRITON_ATTN', decode_fastpath=True), **triton_runner())[0]
        self.assertEqual(both, dict(plain, attention_backend=T.RECORD, decode_fastpath=True))

    def test_on_refuses_before_building_the_worker(self):
        for runner, message in (
                (triton_runner(vllm_config=vllm_config(None)), 'resolved attention backend None'),
                (triton_runner(_kv_cache_spec_attn_group_iterator=groups(FLASH_BACKEND)), 'another attention backend'),
                (triton_runner(_kv_cache_spec_attn_group_iterator=groups()), 'no FullAttention group')):
            with self.subTest(message=message), self.assertRaisesRegex(LiveV2Error, message):
                LM.attach(dict(attention_backend='TRITON_ATTN'), **runner)

    def test_http_runtime_record_must_equal_the_pin(self):
        runner = NS(**triton_runner())
        off = {'effective_server_config': {}}
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'):
            with patch('remote_ra_http.config.load', lambda: ON):
                self.assertEqual(T.worker_check(runner, {'attention_backend': 'TRITON_ATTN'}), T.RECORD)
                with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
                    T.worker_check(runner, {})
            with patch('remote_ra_http.config.load', lambda: off):
                self.assertIsNone(T.worker_check(NS(), {}))
                with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
                    T.worker_check(runner, {'attention_backend': 'TRITON_ATTN'})


# ---- 4. ready.json --------------------------------------------------------------------------------

class ReadyTests(AC.Release, unittest.TestCase):
    def test_ready_records_the_worker_record(self):
        import pathlib
        import test_ms_frontend as MF
        from remote_ra_http.evidence import FrontendEvidence
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
        written = {}
        for flag in (False, True):
            loaded = self.load(self.prepare(tasks(1)[0], 'full', 24, 16, out='ready', triton_attn=flag)[1])
            effective = dict(loaded['effective_server_config'])
            pools = dict(kv_layout=loaded['kv_layout'], num_blocks_by_pool=[10, 20], free_blocks_by_pool=[9, 19])
            for receipt in ({}, {'attention_backend': T.RECORD}):
                with self.subTest(flag=flag, receipt=receipt), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    value = dict(loaded, output_directory=str(output))
                    (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                        epoch=MF.EPOCH, report=readiness, kv_pools=pools,
                        receipt=dict(tensor_geometry={}, source_closure={}, **receipt))).encode())
                    with patch('remote_ra_http.evidence.load', lambda: value), \
                            patch.object(pathlib.Path, 'read_text', proc):
                        evidence = FrontendEvidence(authority, None, effective)
                        app = NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})])
                        if bool(receipt) != flag:
                            with self.assertRaisesRegex(Exception, 'worker/parent attention_backend mismatch'):
                                evidence.write_ready(app, NS(started=True))
                            self.assertFalse((output / 'ready.json').exists())
                            continue
                        evidence.write_ready(app, NS(started=True))
                    written[flag] = json.loads((output / 'ready.json').read_bytes())
        self.assertNotIn('attention_backend', written[False])
        self.assertNotIn('attention_backend', written[False]['effective_server_config'])
        self.assertEqual(written[True]['attention_backend'], T.RECORD)
        self.assertEqual(written[True]['effective_server_config']['attention_backend'], 'TRITON_ATTN')
        differ = ('effective_server_config', 'server_config_sha256', 'server_run_id', 'worker_receipt_sha256')
        self.assertEqual({k: v for k, v in written[True].items() if k not in differ + ('attention_backend',)},
                         {k: v for k, v in written[False].items() if k not in differ})


# ---- 5. the pinned vLLM 0.24.0 sources the docs cite ------------------------------------------------

class PinnedVllmSourceTests(unittest.TestCase):
    CLAIMS = {
        # FlashAttention's per-layer cache shape (the same as TRITON_ATTN's) and its kernel block
        # sizes on a hybrid model with fp32 Mamba state (Qwen3.6's mamba_ssm_dtype).
        'vllm/v1/attention/backends/flash_attn.py': (
            '        return (num_blocks, 2, block_size, num_kv_heads, head_size)',
            '            return [16, 32, 64]\n        return [MultipleOf(16)]',
            '        elif cache_layout == "NHD":\n            stride_order = (0, 1, 2, 3, 4)'),
        # The decoupled layout keeps --block-size P when it is a multiple of a kernel block size.
        'vllm/platforms/interface.py': (
            '    if not any(P % size == 0 for size in kernel_sizes):',
            '            _check_decoupled_block_sizes(vllm_config, backend_cls, layout)'),
        # The runner rewrites the common metadata (physical seq lens) before any builder runs.
        'vllm/v1/worker/gpu_model_runner.py': (
            '            if kv_cache_gid in self._selective_full_attention_group_ids:\n'
            '                cm.seq_lens = self.selective_physical_seq_lens[',
            '                attn_metadata_i = builder.build(\n'
            '                    common_prefix_len=cascade_attn_prefix_len,\n'
            '                    common_attn_metadata=common_attn_metadata,',
            '                if isinstance(group.kv_cache_spec, FullAttentionSpec)'),
        # Every backend runs behind the same attention op (a splitting op of the compiled modes).
        'vllm/model_executor/layers/attention/attention.py': (
            'torch.ops.vllm.unified_attention_with_output(',),
    }

    def test_claims_hold_in_the_stock_files(self):
        definition = json.loads((ROOT / 'INSTALL_RELEASE.json').read_bytes())
        sources = install.inventory(ROOT / 'source-overlay')
        for name, snippets in self.CLAIMS.items():
            with self.subTest(name=name):
                text = (ROOT / 'source-overlay' / name).read_text(encoding='utf-8')
                for snippet in snippets:
                    self.assertIn(snippet, text)
        # FlashAttention and the attention layer are the stock wheel's bytes.
        for name in ('vllm/v1/attention/backends/flash_attn.py', 'vllm/model_executor/layers/attention/attention.py'):
            self.assertEqual(sources[name], definition['expected_live'][name])
            self.assertNotIn(name, definition['payload'])
        # TRITON_ATTN is not in the overlay: the installed wheel's file, hashed by the release index.
        self.assertNotIn('vllm/v1/attention/backends/triton_attn.py', sources)

    def test_remote_ra_modules_name_no_flash_attention_api(self):
        """No runtime module imports or reads FlashAttention's metadata or kernels."""
        hits = []
        for path in sorted((ROOT / 'source-overlay').rglob('*.py')):
            relative = path.relative_to(ROOT / 'source-overlay').as_posix()
            if relative.startswith('vllm/'):
                continue
            text = path.read_text(encoding='utf-8')
            for word in ('FlashAttentionMetadata', 'flash_attn_varlen', 'scheduler_metadata', 'FlashAttentionBackend',
                         'get_flash_attn_version', 'max_num_splits'):
                if word in text:
                    hits.append((relative, word))
        # connector_guard's empty_scheduler_metadata is vLLM's SchedulerOutput connector metadata
        # (called by the generation guard and, with --async-scheduling, async_sched's guard).
        self.assertEqual(hits, [('remote_ra_live_v2/async_sched.py', 'scheduler_metadata'),
                                ('remote_ra_live_v2/connector_guard.py', 'scheduler_metadata'),
                                ('remote_ra_live_v2/generation_guard.py', 'scheduler_metadata')])


if __name__ == '__main__':
    unittest.main()
