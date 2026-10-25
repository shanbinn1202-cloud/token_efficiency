"""serve.py --language-model-only (remote_ra_live_v2.language_model_only). CPU only; not GPU evidence.

Off (the default) adds no vLLM argument, startup config key, worker record or evidence
key; the identity probe covers those bytes, and here the explicit False equals the
default. On:

- launcher: the vLLM command gains vLLM's ``--language-model-only`` at its end and
  nothing else; startup.json pins ``language_model_only`` and
  ``effective_server_config.language_model_only``; every acceptance and refusal of the
  launcher's KV and pinned-host arithmetic is the same as without the flag (the KV
  cache stays ``--kv-cache-memory-bytes``, so no weight accounting changes); the chat
  template, parser and renderer pins are the same bytes;
- startup checks: ``remote_ra_http.config.load`` refuses a pin without its vLLM
  argument and the reverse; the frontend compares vLLM's parsed flag (and refuses it
  when the pin is off); compose_engine sends the pin once and checks the worker's
  echo; the engine's and the worker's resolved multimodal config must be text-only
  with no vision-tower tensor loaded (and must not be when the pin is off);
  ready.json records the worker's record;
- the per-request paths: the multi-session fakes serve the same RPC trace, evidence
  files and tokens with the pin as without it, and no renderer, tokenizer or
  request module reads it;
- the pinned vLLM 0.24.0 sources behind the documentation: byte-identical to the
  stock wheel (expected_live), with the tower placeholder, the multimodal processor
  and warmup gate and the text-only fused attention path where the docs say.
"""
import contextlib
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_align_ctx as AC
import test_cudagraph as TCG

from remote_ra_http import config as http_config
from remote_ra_live_v2 import language_model_only as T
from remote_ra_live_v2.contract import LiveV2Error
# Imported before any test patches sys.modules (see test_align_ctx).
from remote_ra_live_v2 import loaded_sources, native_probe, vector_space, worker_composition as WC  # noqa: F401

try:
    import torch
except ImportError:
    torch = None

tasks = AC.tasks
FLAG = '--language-model-only'
L128 = 131072
CASES = AC.CASES + (  # task, policy, GiB, P, layout, selection, extra flags
    (tasks(16), 'observation_query', 24, 16, 'decoupled', None, dict(decode_fastpath=True, cuda_graphs='full')),
    (tasks(1)[0], 'full', 16, 784, None, None, dict(max_model_len=L128, max_tokens=4096)),
    (tasks(1)[0], 'full', 8, 784, None, None, dict(max_model_len=L128)),            # refused: shared pool
    (tasks(16), 'full', 24, 16, None, None, dict(max_model_len=L128)),               # refused: host budget
)


class MM:
    """vLLM's MultiModalConfig as the checks read it."""

    def __init__(self, language_model_only, limit=0):
        self.language_model_only, self.limit = language_model_only, limit

    def get_limit_per_prompt(self, modality):
        return self.limit


class Tensor:
    def __init__(self, numel, device='cuda', element_size=2):
        self._numel, self._size, self.device = numel, element_size, NS(type=device)

    def numel(self):
        return self._numel

    def element_size(self):
        return self._size


class Model:
    def __init__(self, parameters, buffers=()):
        self.parameters, self.buffers = list(parameters), list(buffers)

    def named_parameters(self):
        return iter(self.parameters)

    def named_buffers(self):
        return iter(self.buffers)


LANGUAGE = [('language_model.model.embed_tokens.weight', Tensor(40)),
            ('language_model.model.layers.3.self_attn.qkv_proj.weight', Tensor(24)),
            ('language_model.lm_head.weight', Tensor(40))]
TOWER = [('visual.blocks.0.attn.qkv.weight', Tensor(12)), ('visual.merger.linear_fc1.weight', Tensor(8))]


def text_runner(**extra):
    """Runner attributes of a text-only vLLM worker: resolved flag, limits 0, no tower."""
    return dict(dict(model_config=NS(multimodal_config=MM(True)), model=Model(LANGUAGE)), **extra)


class FakeOwner(TCG.FakeOwner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.claims = {}                 # --decode-fastpath replaces the empty claims


def attach(startup_extra=None, **runner_extra):
    """test_cudagraph's attach_loaded_worker fixture (legacy layout, eager), with runner attributes."""
    from remote_ra_growing_worker import tensors
    views = TCG.AttachLoadedWorkerTests().views()
    runner = NS(_remote_ra_growing_view_inputs=('kv', (784,)), kv_cache_config=NS(group_pool=None),
                execute_model=lambda *args: None, compilation_config=TCG.resolved_config(None),
                _selective_cudagraph_groups=frozenset(), **runner_extra)
    worker = NS(model_runner=runner)
    startup = dict(enabled=True, source='m' * 64, page_tokens=784, loaded_sources_verified=True,
                   **(startup_extra or {}))
    transfer = ModuleType('vllm.distributed.kv_transfer')
    transfer.get_kv_transfer_group = lambda: 'connector'
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {'vllm.distributed.kv_transfer': transfer}))
        stack.enter_context(patch.object(tensors, 'TensorIO', TCG.FakeIO))
        stack.enter_context(patch.object(tensors, 'views_from_runner', lambda runner, kv, sizes: views))
        stack.enter_context(patch.object(WC, 'worker_layout', lambda runner, startup: AC.I0.legacy()))
        stack.enter_context(patch.object(WC, 'check_view_geometry', lambda *args: None))
        stack.enter_context(patch.object(WC, 'RunnerAllocationOwner', FakeOwner))
        stack.enter_context(patch.object(WC, 'CudaWorkerBackend', TCG.FakeBackend))
        stack.enter_context(patch.object(WC, 'QwenAlignLifecycle', TCG.FakeQwen))
        stack.enter_context(patch.object(loaded_sources, 'verify_closure', lambda runner: {'closure': 'ok'}))
        stack.enter_context(patch.object(vector_space, 'attach', lambda runner: {'hidden_size': 8}))
        stack.enter_context(patch.object(native_probe, 'NativeQueryForward', lambda *args: 'probe'))
        evidence = WC.attach_loaded_worker(worker, startup)
    return evidence, runner


ON = {'language_model_only': True, 'effective_server_config': {'language_model_only': True}}


# ---- 1. the pin and the resolved-config checks ---------------------------------------------------

class PinTests(unittest.TestCase):
    def test_pin_values(self):
        self.assertIsNone(T.pinned(None))
        self.assertIsNone(T.pinned({}))
        self.assertIsNone(T.pinned({'effective_server_config': {'seed': 42}}))
        self.assertIs(T.pinned(ON), True)
        refused = [dict(ON, language_model_only=bad) for bad in (False, 1, 'true', None)]
        refused += [dict(ON, effective_server_config={}), dict(ON, effective_server_config={'language_model_only': 1}),
                    dict(ON, effective_server_config=None), {'effective_server_config': {'language_model_only': True}},
                    {'effective_server_config': {'language_model_only': False}}]
        for value in refused:
            with self.subTest(value=value), self.assertRaises(LiveV2Error):
                T.pinned(value)

    def test_record_is_the_checked_facts(self):
        self.assertEqual(T.RECORD, {'mm_limits': {'image': 0, 'video': 0}, 'vision_tower_tensors': 0})
        self.assertEqual(set(T.RECORD['mm_limits']), set(T.TOWER_MODALITIES))
        self.assertIsNone(T.record({}))
        record = T.record(ON)
        self.assertEqual(record, T.RECORD)
        record['mm_limits']['image'] = 1                      # a copy: the constant is untouched
        self.assertEqual(T.RECORD['mm_limits']['image'], 0)

    def test_engine_check(self):
        def config(mm):
            return NS(model_config=NS(multimodal_config=mm))
        for off_ok in (None, NS(), NS(model_config=None), NS(model_config=NS()), config(None), config(MM(False)),
                       config(MM(False, 999))):
            with self.subTest(off=off_ok):
                self.assertIsNone(T.check_engine(off_ok, None))
        with self.assertRaisesRegex(LiveV2Error, 'pins it off'):
            T.check_engine(config(MM(True)), None)
        self.assertIs(T.check_engine(config(MM(True)), True), True)
        for bad, message in ((config(MM(False)), 'did not resolve'), (config(None), 'did not resolve'),
                             (NS(), 'did not resolve'), (config(MM(True, 999)), 'multimodal limit 0'),
                             (config(MM(True, False)), 'multimodal limit 0'),
                             (config(MM(1)), 'did not resolve')):
            with self.subTest(bad=message), self.assertRaisesRegex(LiveV2Error, message):
                T.check_engine(bad, True)

    def test_tower_tensors_count_loaded_tower_names_only(self):
        self.assertEqual(T.tower_tensors(Model(LANGUAGE)), (0, 0))
        self.assertEqual(T.tower_tensors(Model(LANGUAGE + TOWER)), (2, 40))
        self.assertEqual(T.tower_tensors(Model(LANGUAGE, [('model.visual.pos_embed', Tensor(4, 'cpu', 4))])), (1, 16))
        meta = [(name, Tensor(t.numel(), 'meta')) for name, t in TOWER]
        self.assertEqual(T.tower_tensors(Model(LANGUAGE + meta)), (0, 0))       # created, never allocated
        lookalike = [('language_model.visual_norm.weight', Tensor(4)), ('visualize.weight', Tensor(4))]
        self.assertEqual(T.tower_tensors(Model(lookalike)), (0, 0))
        self.assertEqual(T.tower_tensors(NS()), (0, 0))

    @unittest.skipUnless(torch is not None, 'CPU torch required')
    def test_tower_tensors_on_torch_modules(self):
        nn = torch.nn

        class Placeholder(nn.Module):
            """A skipped tower kept outside the module tree, as a stage placeholder may hold it."""
            def __init__(self, module):
                super().__init__()
                self.__dict__['module'] = module

        def composite(tower):
            model = nn.Module()
            model.language_model = nn.Linear(4, 4)
            if tower == 'loaded':
                model.visual = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
            elif tower == 'meta':
                with torch.device('meta'):
                    model.visual = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
            else:
                with torch.device('meta'):
                    model.visual = Placeholder(nn.Sequential(nn.Linear(4, 4)))
            return model
        self.assertEqual(T.tower_tensors(composite('loaded')), (4, 4 * (16 + 4 + 4 + 4)))
        self.assertEqual(T.tower_tensors(composite('meta')), (0, 0))
        self.assertEqual(T.tower_tensors(composite('placeholder')), (0, 0))


# ---- 2. launcher, startup config, frontend, engine ------------------------------------------------

class LauncherTests(AC.Release, unittest.TestCase):
    maxDiff = None

    def test_off_is_the_default(self):
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                plain = self.outcome(task, policy, gib, P, layout, selection, out='off', **extra)
                explicit = self.outcome(task, policy, gib, P, layout, selection, out='off',
                                        language_model_only=False, **extra)
                self.assertEqual(explicit, plain)
                if plain[0] == 'ok':
                    manifest, config, command, env = plain[1]
                    self.assertNotIn(FLAG, command)
                    self.assertNotIn('language_model_only', config)
                    self.assertNotIn('language_model_only', config['effective_server_config'])

    def test_on_appends_one_vllm_argument_and_pins_it(self):
        kinds = set()
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                off = self.outcome(task, policy, gib, P, layout, selection, out='lm', **extra)
                on = self.outcome(task, policy, gib, P, layout, selection, out='lm', language_model_only=True,
                                  **extra)
                kinds.add(off[0])
                if off[0] != 'ok':
                    self.assertEqual(on, off)               # the same KV / host refusals
                    continue
                (m0, c0, a0, e0), (m1, c1, a1, e1) = off[1], on[1]
                self.assertEqual(m1, m0)
                self.assertEqual(a1, a0 + [FLAG])          # chat template, parsers, KV bytes: the same argv
                effective = dict(c0['effective_server_config'], language_model_only=True)
                self.assertEqual(c1['effective_server_config'], effective)
                self.assertEqual(c1, dict(c0, language_model_only=True, effective_server_config=effective,
                                          server_config_sha256=serve.pin(effective)))
                self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                self.assertEqual(effective['kv_cache_memory_bytes'], gib * 1024 ** 3)
        self.assertEqual(kinds, {'ok', 'ValueError'})

    def test_refuses_anything_but_a_switch(self):
        for bad in (1, 'yes', None, 0):
            with self.subTest(bad=bad):
                self.assertEqual(self.outcome(tasks(1)[0], language_model_only=bad),
                                 ('ValueError', '--language-model-only is a switch'))

    def test_dry_run_main(self):
        task = self.root / 'dry-task.json'
        task.write_text(json.dumps(tasks(1)[0]))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--output',
                    str(self.root / 'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream):
                serve.main()
            return json.loads(stream.getvalue())
        for extra in ((), ('--page-tokens', '16', '--kv-cache-gib', '24', '--cuda-graphs', 'full',
                           '--decode-fastpath', '--max-model-len', '131072')):
            with self.subTest(extra=extra):
                plain, flagged = main(*extra), main(*extra, FLAG)
                self.assertEqual(flagged, dict(plain, command=plain['command'] + [FLAG]))
                self.assertNotIn(FLAG, plain['command'])

    def test_load_accepts_the_pin_and_refuses_mismatches(self):
        off = self.prepare(tasks(3), 'observation_query', 24, 16, 'decoupled', out='load')[1]
        on = self.prepare(tasks(3), 'observation_query', 24, 16, 'decoupled', out='load',
                          language_model_only=True)[1]
        self.assertIs(self.load(on)['language_model_only'], True)
        self.assertNotIn('language_model_only', self.load(off))
        with self.startup(on):
            self.assertIs(T.loaded(), True)
        with self.startup(off):
            self.assertIsNone(T.loaded())
        effective_on, effective_off = on['effective_server_config'], off['effective_server_config']
        bad = {'pin without the vLLM argument': dict(on, effective_server_config=effective_off),
               'vLLM argument without the pin': dict(off, effective_server_config=effective_on),
               'pin false': dict(on, language_model_only=False),
               'argument false': dict(on, effective_server_config=dict(effective_on, language_model_only=False))}
        for name, value in bad.items():
            with self.subTest(name):
                value = dict(value, server_config_sha256=serve.pin(value['effective_server_config']))
                with self.assertRaisesRegex(RuntimeError, 'language_model_only'):
                    self.load(value)

    def test_frontend_compares_the_parsed_flag(self):
        off = self.prepare(tasks(1)[0], out='front')[1]
        on = self.prepare(tasks(1)[0], out='front', language_model_only=True)[1]

        def frontend(config, args):
            with self.startup(config):
                try:
                    return http_config.effective_server_config(NS(**args))
                except RuntimeError as error:
                    return ('refused', str(error))
        effective = on['effective_server_config']
        self.assertEqual(frontend(on, effective), effective)
        for args in (dict(effective, language_model_only=False), dict(effective, language_model_only=None),
                     {k: v for k, v in effective.items() if k != 'language_model_only'}):
            with self.subTest(args=args.get('language_model_only', '<absent>')):
                outcome = frontend(on, args)
                self.assertEqual(outcome[0], 'refused')
                self.assertIn('observed API server arguments differ', outcome[1])
        effective = off['effective_server_config']
        self.assertEqual(frontend(off, effective), effective)
        self.assertEqual(frontend(off, dict(effective, language_model_only=False)), effective)  # vLLM's default
        self.assertEqual(frontend(off, dict(effective, language_model_only=True)),
                         ('refused', 'vLLM parsed --language-model-only but the startup config pins it off'))

    def startup_record(self, config, echo):
        """compose_engine's worker startup record and its refusal of the worker's reply ``echo``."""
        from remote_ra_live_v2 import composition
        sent = []

        def collective_rpc(method, args, timeout):
            sent.append(args[0])
            return [echo]
        task = config['task']
        core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                  _remote_ra_trusted_issue=dict(task), _remote_ra_trusted_issues=[dict(task)],
                  _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                  _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
        with self.startup(config), self.assertRaises(LiveV2Error) as caught:
            composition.compose_engine(core, object())
        startup, = sent
        return startup, str(caught.exception)

    def test_compose_engine_sends_the_pin_once_and_checks_the_echo(self):
        base = dict(enabled=True, source='m' * 64, page_tokens=784, loaded_sources_verified=True)
        next_check = 'loaded CUDA worker attachment receipt'       # the check after the echoes
        off = self.prepare(tasks(1)[0], out='ce')[1]
        on = self.prepare(tasks(1)[0], out='ce', language_model_only=True)[1]
        for config, echo, record, message in (
                (off, {}, base, next_check),
                (off, {'language_model_only': T.RECORD}, base, 'another language_model_only'),
                (on, {}, dict(base, language_model_only=True), 'another language_model_only'),
                (on, {'language_model_only': True}, dict(base, language_model_only=True), 'another language_model_only'),
                (on, {'language_model_only': T.RECORD}, dict(base, language_model_only=True), next_check)):
            with self.subTest(on='language_model_only' in config, echo=echo):
                startup, error = self.startup_record(config, echo)
                self.assertEqual(startup, record)
                self.assertIn(message, error)


# ---- 3. worker ------------------------------------------------------------------------------------

class WorkerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        patcher = patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_off_evidence_is_unchanged(self):
        plain = TCG.AttachLoadedWorkerTests().attach(TCG.resolved_config(None))[0]
        for runner in ({}, dict(model_config=NS(multimodal_config=MM(False, 999)), model=Model(LANGUAGE + TOWER)),
                       dict(model_config=NS(multimodal_config=None))):
            with self.subTest(runner=sorted(runner)):
                self.assertEqual(attach(None, **runner)[0], plain)
        self.assertNotIn('language_model_only', plain)

    def test_off_refuses_a_text_only_vllm(self):
        with self.assertRaisesRegex(LiveV2Error, 'pins it off'):
            attach(None, **text_runner())

    def test_on_evidence_gains_only_the_record(self):
        plain = attach(None)[0]
        evidence, runner = attach(dict(language_model_only=True), **text_runner())
        self.assertEqual(evidence, dict(plain, language_model_only=T.RECORD))
        both = attach(dict(language_model_only=True, decode_fastpath=True), **text_runner())[0]
        self.assertEqual(both, dict(plain, language_model_only=T.RECORD, decode_fastpath=True))
        # get_model() (vLLM's runner unwraps its CUDA-graph wrapper) is preferred to .model.
        wrapped = text_runner(model=NS(), get_model=lambda: Model(LANGUAGE))
        self.assertEqual(attach(dict(language_model_only=True), **wrapped)[0], evidence)

    def test_on_refuses_before_building_the_worker(self):
        for runner, startup, message in (
                (text_runner(model_config=NS(multimodal_config=MM(False))), None, 'did not resolve'),
                (text_runner(model_config=NS(multimodal_config=None)), None, 'did not resolve'),
                ({'model': Model(LANGUAGE)}, None, 'did not resolve'),
                (text_runner(model_config=NS(multimodal_config=MM(True, 999))), None, 'multimodal limit 0'),
                (text_runner(model=Model(LANGUAGE + TOWER)), None, 'loaded 2 vision-tower tensors'),
                (text_runner(model=NS(), get_model=lambda: Model(TOWER)), None, 'loaded 2 vision-tower tensors'),
                (text_runner(model=None), None, 'needs the loaded model'),
                (text_runner(), dict(language_model_only=False), 'must be true or absent'),
                (text_runner(), dict(language_model_only=T.RECORD), 'must be true or absent')):
            with self.subTest(message=message, startup=startup):
                with self.assertRaisesRegex(LiveV2Error, message):
                    attach(startup or dict(language_model_only=True), **runner)

    def test_http_runtime_record_must_equal_the_pin(self):
        runner = NS(**text_runner())
        off = {'effective_server_config': {}}
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'):
            with patch('remote_ra_http.config.load', lambda: ON):
                self.assertEqual(T.worker_check(runner, {'language_model_only': True}), T.RECORD)
                with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
                    T.worker_check(runner, {})
            with patch('remote_ra_http.config.load', lambda: off):
                self.assertIsNone(T.worker_check(NS(), {}))
                with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
                    T.worker_check(runner, {'language_model_only': True})


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
            loaded = self.load(self.prepare(tasks(1)[0], out='ready', language_model_only=flag)[1])
            effective = dict(loaded['effective_server_config'])
            for receipt in ({}, {'language_model_only': T.RECORD}):
                with self.subTest(flag=flag, receipt=receipt), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    value = dict(loaded, output_directory=str(output))
                    (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                        epoch=MF.EPOCH, report=readiness,
                        receipt=dict(tensor_geometry={}, source_closure={}, **receipt))).encode())
                    with patch('remote_ra_http.evidence.load', lambda: value), \
                            patch.object(pathlib.Path, 'read_text', proc):
                        evidence = FrontendEvidence(authority, None, effective)
                        app = NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})])
                        if bool(receipt) != flag:
                            with self.assertRaisesRegex(Exception, 'worker/parent language_model_only mismatch'):
                                evidence.write_ready(app, NS(started=True))
                            self.assertFalse((output / 'ready.json').exists())
                            continue
                        evidence.write_ready(app, NS(started=True))
                    written[flag] = json.loads((output / 'ready.json').read_bytes())
        self.assertNotIn('language_model_only', written[False])
        self.assertNotIn('language_model_only', written[False]['effective_server_config'])
        self.assertEqual(written[True]['language_model_only'], T.RECORD)
        self.assertIs(written[True]['effective_server_config']['language_model_only'], True)
        # The run id holds the start time; the receipt hash covers the worker's record.
        differ = ('effective_server_config', 'server_config_sha256', 'server_run_id', 'worker_receipt_sha256')
        self.assertEqual({k: v for k, v in written[True].items() if k not in differ + ('language_model_only',)},
                         {k: v for k, v in written[False].items() if k not in differ})


# ---- 5. the per-request paths -----------------------------------------------------------------------

class RequestPathTests(unittest.TestCase):
    """The pin is read at startup only: requests, renderer and tokenizer never see it."""

    def serve(self, case, policy, text_only):
        import test_decouple_e2e as D
        import test_host_trim as HT
        multi = D.Case(**dict(vars(case), sessions=len(D.TASKS)))
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = D.config(policy, output, multi, D.TASKS)
            if text_only:
                value.update(ON)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = D.MultiHarness(policy, output, multi)
            trace, rpc = [], h.collective_rpc

            def collective_rpc(method, args, timeout):
                operation, payload = args
                reply = rpc(method, args, timeout)
                trace.append(dict(operation=operation, payload=HT.plain(payload), receipt=HT.plain(reply[0])))
                return reply
            h.proxy.executor = NS(collective_rpc=collective_rpc)
            h.serve()
            h.close()
            D.engine_closed(h.core)
            files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
            return dict(trace=trace, files=files, computed=list(h.runner.computed),
                        done=[(i['rid'], list(i['request'].output_token_ids)) for i in h.done])

    def test_multi_session_fakes_serve_the_same_bytes(self):
        import test_decouple_e2e as D
        for case, policy in ((D.CASES[0], 'full'), (D.CASES[2], 'observation_query')):
            with self.subTest(case=case.name, policy=policy):
                off, on = self.serve(case, policy, False), self.serve(case, policy, True)
                for key in ('trace', 'files', 'computed', 'done'):
                    self.assertEqual(on[key], off[key], key)
                self.assertTrue(on['done'] and on['files'])

    def test_only_the_startup_modules_read_the_pin(self):
        readers = sorted(str(path.relative_to(ROOT / 'source-overlay')).replace('\\', '/')
                         for path in (ROOT / 'source-overlay').rglob('*.py')
                         if 'language_model_only' in path.read_text(encoding='utf-8'))
        self.assertEqual(readers, ['remote_ra_http/config.py', 'remote_ra_http/evidence.py',
                                   'remote_ra_live_v2/composition.py', 'remote_ra_live_v2/language_model_only.py',
                                   'remote_ra_live_v2/window_worker.py',    # the Gemma worker's startup (gemma-s1)
                                   'remote_ra_live_v2/worker_composition.py',
                                   'vllm/model_executor/models/qwen3_next.py'])


# ---- 6. the pinned vLLM 0.24.0 sources the docs cite ------------------------------------------------

class PinnedVllmSourceTests(unittest.TestCase):
    CLAIMS = {
        # A tower whose modalities all have limit 0 is replaced by a placeholder, never built.
        'vllm/model_executor/models/interfaces.py': (
            'lambda mod: StageMissingLayer(stage_name, mod),',
            'if all(mm_config.get_limit_per_prompt(m) == 0 for m in modalities)'),
        # Qwen3.5/3.6 build the vision tower (TOWER, TOWER_MODALITIES) inside it.
        'vllm/model_executor/models/qwen3_5.py': (
            'with self._mark_tower_model(vllm_config, {"image", "video"}):\n'
            '            self.visual = Qwen3_VisionTransformer(',),
        # No multimodal processor without multimodal inputs, and its warmups need one.
        'vllm/renderers/base.py': (
            '        if mm_registry.supports_multimodal_inputs(config.model_config):\n'
            '            mm_processor_cache = mm_registry.processor_cache_from_config(config)',
            '        if self.mm_processor:\n            try:\n'
            '                logger.debug("Warming up multi-modal processing...")',
            '        if self._readonly_mm_processor is not None:'),
        # No encoder cache budget without multimodal inputs.
        'vllm/v1/engine/input_processor.py': (
            'self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)',),
        # Text-only mode switches the full-attention layers to the fused QK-norm/RoPE/gate kernel.
        'vllm/model_executor/models/qwen3_next.py': (
            'text_only = mm_config is None or mm_config.language_model_only',
            '            and text_only\n        )'),
    }

    def test_claims_hold_in_the_stock_files(self):
        definition = json.loads((ROOT / 'INSTALL_RELEASE.json').read_bytes())
        sources = install.inventory(ROOT / 'source-overlay')
        for name, snippets in self.CLAIMS.items():
            with self.subTest(name=name):
                self.assertEqual(sources[name], definition['expected_live'][name])   # the stock wheel's bytes
                self.assertNotIn(name, definition['payload'])
                text = (ROOT / 'source-overlay' / name).read_text(encoding='utf-8')
                for snippet in snippets:
                    self.assertIn(snippet, text)


if __name__ == '__main__':
    unittest.main()
