"""Gemma 4 startup (gemma-assembly-fix): the engine assembly, its readiness gate and the frontend
capture take the Gemma 4 renderer, and the native startup path of a Gemma server runs through. CPU only.

GPU jobs 3398198 (runtime 0d37346) and 3398614 (runtime 4eb9f9d): all nine Gemma 4 runtime servers
(full, observation_query, goal_query; eager and CUDA graphs) stopped in the engine's phase
engine-dormant-assembly. ``EngineAssembly`` required ``type(renderer) is EngineSourceRenderer`` and
refused the ``Gemma4SourceRenderer`` that remote_ra_linux.live_renderer binds for the Gemma profile
(engine_assembly.py line 69, from NativeEngine.build); ``bootstrap_contract.EngineReadiness`` had the
same check. gemma-mab had relaxed only the frontend capture (frontend_registry._trusted_renderer).
One predicate, ``runtime_renderer.trusted_source_renderer``, now serves the three: exactly
``EngineSourceRenderer`` or exactly the loaded ``gemma4_renderer.Gemma4SourceRenderer``.

Why the Gemma end-to-end tests did not see it: they wire the engine as compose_engine wires it, from
an ``assembly`` namespace with a stand-in renderer (test_ms_engine.MultiHarness, test_gemma_e2e.Harness),
and the frontend from ``Gemma4SourceRenderer.__new__`` (test_gemma_kv_session_e2e,
test_gemma_chat_text_e2e). No test built EngineAssembly, EngineReadiness or NativeEngine.build, for any
model.

* PredicateTests: the predicate; every site uses it; no exact-type renderer check is left on the
  startup modules.
* AssemblyTests (every interpreter): the real EngineAssembly (construction, activate, require_active and
  its EngineReadiness gate) over the real provider, codec, selector bridge, private Channel and
  EngineFence, with a Gemma4SourceRenderer (built without its tokenizer contract, so that it runs
  without transformers; the pinned one with REMOTE_RA_GEMMA4_METADATA), an EngineSourceRenderer, and a
  subclass of each (refused).
* StartupPathTests (REMOTE_RA_GEMMA4_METADATA and transformers): for a Gemma release's startup config
  (serve.prepare) of each protocol (mswea-xml with full and goal_query; kv-session with thinking off
  and chat-text, observation_query; chat-text with goal_query and --admission-overlap 16, gemma-tb-goal),
  the engine thread runs the real startup: startup_runtime.begin and
  activate with NativeEngine.build (build_engine_components, ActualRendererBinding and the pinned
  Gemma4SourceRenderer, register_uniproc with the real worker probe, EngineAssembly),
  growing_startup.attach_dormant on vLLM's KV manager with the window layout (test_gemma_kv_pools'
  world), attach_engine_dormant, compose_engine and release_b0 (PREPARED_LIVE on the real private
  Channel to the real NativeManager reader, its size against MAX_FRAME). The frontend runs the real
  live_frontend._bind (the pinned template check, its own Gemma4SourceRenderer, TrustedFrontendCapture,
  FrontendRegistry, ServingController, compose_frontend with the protocol's transcripts, F1 through
  NativeManager.bind_frontend and the engine's poll). Then the first request: its frontend capture and
  the engine's render_claim agree on every token.
  Stand-ins only where vLLM (not installed here) or the GPU would be: the vLLM classes the path checks
  by type and the HfRenderer's async methods, the loaded worker (its probe is the real
  worker_readiness; its attach evidence an echo of the startup record: the window worker's own attach
  is test_gemma_graphs'), the overlay KV manager's import world, and, without torch, the projection's
  random draw. Per-thread native_authority.current stands for the two processes.
"""
import ast
import asyncio
import base64
import contextlib
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import multiprocessing
import os
from pathlib import Path
import queue
import secrets
import sys
import tempfile
import threading
import time
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import gemma4_testing as G  # noqa: E402
import gemma4_renderer as GR  # noqa: E402
import runtime_renderer as RR  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402

HF = G.tokenizer() if G.metadata_dir() is not None else None
NEEDS_PINNED = (HF is not None, 'pinned Gemma 4 tokenizer (%s) and transformers required' % G.ENV)
MANIFEST = 'a1' * 32
P = 16


def hex64():
    return secrets.token_hex(32)


def hf_render(tokenizer):
    """vLLM's safe_apply_chat_template in miniature: the tokenizer's own template call, no tokenization."""
    def render(template, conversation, kwargs):
        return tokenizer.apply_chat_template(conversation, chat_template=template, tokenize=False, **kwargs)
    return render


# ---- renderers ------------------------------------------------------------------------------------

def bare(cls, template='{{ bos_token }}template'):
    """An instance of ``cls`` with the attributes its _check_identity reads, without the tokenizer
    contract (which needs the pinned files and transformers)."""
    renderer = cls.__new__(cls)
    backend = NS(to_str=lambda: '{}', normalizer=None)
    renderer.tokenizer = NS(is_fast=True, chat_template=template, backend_tokenizer=backend)
    renderer.template, renderer.page_tokens = template, P
    renderer.verify_identity = lambda: True
    renderer._semantic = NS(verify=lambda: True)
    return renderer


def pinned_gemma():
    """The Gemma4SourceRenderer of the pinned metadata (the live renderer's construction)."""
    return GR.Gemma4SourceRenderer(model_dir=G.metadata_dir(), tokenizer=HF, render_template=hf_render(HF),
                                   verify_identity=lambda: True, page_tokens=P)


class QwenSub(RR.EngineSourceRenderer):
    pass


class GemmaSub(GR.Gemma4SourceRenderer):
    pass


# ---- the engine's objects ---------------------------------------------------------------------------

def topology(model=None, *, max_model_len=131072, seqs=16, additional=None, pool=1 << 30):
    """A vLLM config namespace that bootstrap_contract.validate_topology and compose_engine's checks
    read (uniprocess, synchronous, one batch, no speculation; text only)."""
    parallel = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                  decode_context_parallel_size=1, prefill_context_parallel_size=1,
                  distributed_executor_backend='uni', worker_extension_cls='')
    pinned = G.config()['text_config']
    text = NS(hidden_size=pinned['hidden_size'], num_hidden_layers=pinned['num_hidden_layers'],
              _commit_hash=MP.GEMMA4_REVISION)
    multimodal = NS(language_model_only=True, get_limit_per_prompt=lambda modality: 0)
    model_config = NS(model=str(model or 'gemma-4-31b-it'), revision=MP.GEMMA4_REVISION, runner_type='generate',
                      is_encoder_decoder=False, enable_prompt_embeds=False, hf_text_config=text,
                      max_model_len=max_model_len, multimodal_config=multimodal)
    return NS(parallel_config=parallel, model_config=model_config, speculative_config=None,
              scheduler_config=NS(async_scheduling=False, max_num_seqs=seqs, max_num_batched_tokens=8192),
              max_concurrent_batches=1, use_v2_model_runner=False, additional_config=additional or {},
              kv_transfer_config=NS(kv_connector='OffloadingConnector',
                                    kv_connector_extra_config={'cpu_bytes_to_use': pool}))


class Offloader:
    """The overlay OffloadingConnectorScheduler's runtime seams: configure_policy_runtime (as in
    vllm/.../offloading/scheduler.py) and the five callbacks remote_ra_http.connector isolates."""

    def __init__(self):
        self._req_status = {}

    def configure_policy_runtime(self, coordinator, context_provider):
        if self._req_status:
            raise RuntimeError('configure policy runtime before admitting requests')
        self._policy_runtime = coordinator
        self._policy_runtime_context = context_provider

    def has_pending_push_work(self):
        return False

    def on_new_request(self, request):
        raise AssertionError('isolated')

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        raise AssertionError('isolated')

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        raise AssertionError('isolated')

    def request_finished(self, request, block_ids):
        raise AssertionError('isolated')

    def build_connector_meta(self, output):
        raise AssertionError('isolated')


class Core:
    """EngineCore's attributes and bound callbacks that the startup path checks."""

    def __init__(self, vllm_config, scheduler, model_executor=None):
        self.vllm_config, self.scheduler, self.model_executor = vllm_config, scheduler, model_executor
        self.input_queue, self.output_queue, self.aborts_queue = queue.Queue(), queue.Queue(), queue.Queue()
        self.batch_queue, self.engines_running = None, False
        self.preprocessed, self.added, self.aborted = {}, [], []

    def preprocess_add_request(self, request):
        """vLLM's Request of the decoded EngineCoreRequest (the fields the runtime reads)."""
        native = NS(request_id=request.request_id, prompt_token_ids=request.prompt_token_ids,
                    sampling_params=request.sampling_params, kv_transfer_params=None, block_hashes=[],
                    output_token_ids=[], num_computed_tokens=0)
        self.preprocessed[request.request_id] = native
        return native, 0

    def add_request(self, request, request_wave=0):
        self.added.append(request)

    def abort_requests(self, request_ids):
        self.aborted.extend(request_ids)


def scheduler(kv_cache_manager=None, max_tokens=8192):
    return NS(requests={}, waiting=[], running=[], kv_cache_manager=kv_cache_manager,
              connector=NS(connector_scheduler=Offloader()), max_num_scheduled_tokens=max_tokens)


def private_pair(manifest=MANIFEST):
    """One launch identity and key; the two ends of its private pipe."""
    from private_authority.protocol import Identity
    frontend, engine = multiprocessing.Pipe(duplex=True)
    identity = Identity(hex64(), hex64(), hex64(), manifest, os.getpid(), os.getpid() + 1)
    identity.validate()
    return identity, secrets.token_bytes(32), frontend, engine


def engine_parts(renderer, channel, fence, *, identity=MANIFEST, model='gemma-4-31b-it'):
    """remote_ra_linux.live_factory.build_engine_components without vLLM: the provider of the
    renderer and the private selector bridge."""
    from contracts import ProducerProvenance, RuntimeCoordinator
    from dynamic_runtime import IdempotentGoalRuntimeAdapter
    from policy_runtime.live_composition import build_approved_policy_router
    from policy_runtime.tagged_selector import TaggedSelectionBridge
    from remote_ra_http.provider import RequestContextProvider, reject_legacy_selection
    from remote_ra_linux.live_factory import _static_only
    from two_span_observation import TwoSpanObservationRuntime
    provenance = ProducerProvenance(model=model, revision=MP.GEMMA4_REVISION, hidden_layer=59,
                                    projection_hash='5' * 64, dimension=128, page_tokens=P)
    coordinator = RuntimeCoordinator(emit=lambda marker: None)
    coordinator.register_adapter('goal_query', IdempotentGoalRuntimeAdapter(provenance))
    coordinator.register_adapter('observation_query', TwoSpanObservationRuntime(
        dimension=128, provenance_hash=provenance.provenance_hash, capture_point=provenance.vector_source,
        selected_count=2))
    provider = RequestContextProvider(provenance=provenance, selected_count=2, projection_seed=42,
                                      coordinator=coordinator, renderer=renderer, renderer_identity=identity)
    router = build_approved_policy_router(full_handler=_static_only, random50_handler=_static_only)
    bridge = TaggedSelectionBridge(channel=channel, fence=fence, router=router, producer_provenance=provenance,
                                   input_factory=reject_legacy_selection)
    return provider, bridge


class Decoder:
    """vllm.v1.serial_utils.MsgpackDecoder: the frame's request, registered by the test."""
    frames = {}

    def __init__(self, request_type):
        self.request_type = request_type

    def decode(self, frames):
        (raw,) = frames
        return self.frames[bytes(raw)]


class EngineCoreRequest:
    pass


def assemble(test, renderer, *, attach=True):
    """The real EngineAssembly of ``renderer`` on the engine's objects, dormant, and (``attach``)
    activated; its owner authority installed for require_installable."""
    from engine_assembly import AssemblyConfig, EngineAssembly
    from private_authority.protocol import Channel, EngineFence
    from registered_add_protocol import ProtocolCodec
    from remote_ra_linux import native_authority as NA
    identity, key, _frontend, engine = private_pair()
    test.addCleanup(_frontend.close)
    test.addCleanup(engine.close)
    owner = NA.NativeEngine.__new__(NA.NativeEngine)
    owner.owner, owner.revoked = threading.get_ident(), threading.Event()
    owner.loaded = NS(verify_loaded=lambda: True)
    test.enterContext(patch.object(NA, '_authority', owner))
    channel = Channel(engine, identity, key, 'engine')
    fence = EngineFence(owner.revoked)
    provider, bridge = engine_parts(renderer, channel, fence, identity=identity.manifest)
    codec = ProtocolCodec(key=secrets.token_bytes(32), identity=identity.manifest, epoch=identity.epoch)
    core = Core(topology(), scheduler())
    quiescent = lambda: True                                    # noqa: E731
    assembly = EngineAssembly(config=AssemblyConfig(enabled=True, identity=identity.manifest, epoch=identity.epoch),
                              core=core, provider=provider, renderer=renderer, codec=codec,
                              core_type=EngineCoreRequest, decoder_type=Decoder, verify_source=lambda: True,
                              verify_worker=lambda: True, verify_quiescent=quiescent, selector_bridge=bridge)
    if attach:
        assembly.activate()
    return assembly, core


# ---- 1. the predicate ---------------------------------------------------------------------------------

STARTUP_MODULES = ('engine_assembly.py', 'bootstrap_contract.py', 'frontend_registry.py', 'runtime_renderer.py',
                   'startup_runtime.py', 'growing_startup.py', 'gemma4_renderer.py', 'remote_ra_linux',
                   'remote_ra_live_v2/composition.py', 'remote_ra_live_v2/private_endpoint.py',
                   'remote_ra_live_v2/startup.py', 'remote_ra_live_v2/session_authority.py',
                   'remote_ra_live_v2/phase1_coordinator.py', 'remote_ra_live_v2/window_phase1.py',
                   'remote_ra_live_v2/query_selection.py', 'remote_ra_live_v2/controller.py',
                   'serving_controller.py', 'lifecycle_binding.py', 'entrypoint_runtime.py', 'route_runtime.py')
RENDERER_CLASSES = {'EngineSourceRenderer', 'Gemma4SourceRenderer'}


def renderer_type_checks(path):
    """``type(x) is C`` / ``is not C`` / ``in (C, ...)`` comparisons naming a renderer class, outside
    runtime_renderer.trusted_source_renderer."""
    tree = ast.parse(path.read_text('utf-8'))
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'trusted_source_renderer':
            allowed |= {id(n) for n in ast.walk(node)}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or id(node) in allowed:
            continue
        sides = [node.left] + list(node.comparators)
        if not any(isinstance(s, ast.Call) and getattr(s.func, 'id', None) == 'type' for s in sides):
            continue
        names = {n.id if isinstance(n, ast.Name) else n.attr for s in sides for n in ast.walk(s)
                 if isinstance(n, (ast.Name, ast.Attribute))}
        if names & RENDERER_CLASSES:
            found.append((path.name, node.lineno))
    return found


class PredicateTests(unittest.TestCase):
    def test_exactly_the_two_profile_renderers(self):
        self.assertIs(RR.trusted_source_renderer(bare(RR.EngineSourceRenderer)), True)
        self.assertIs(RR.trusted_source_renderer(bare(GR.Gemma4SourceRenderer)), True)
        for other in (bare(QwenSub), bare(GemmaSub), object(), NS(template='t'), None,
                      RR.EngineSourceRenderer, GR.Gemma4SourceRenderer):
            with self.subTest(other=type(other).__name__):
                self.assertIs(RR.trusted_source_renderer(other), False)

    def test_the_gemma_class_is_the_loaded_modules_own(self):
        import importlib.util
        gemma = bare(GR.Gemma4SourceRenderer)
        # The module is looked up, never imported: without it only the Qwen class passes.
        loaded = sys.modules.pop('gemma4_renderer')
        try:
            self.assertIs(RR.trusted_source_renderer(gemma), False)
            self.assertIs(RR.trusted_source_renderer(bare(RR.EngineSourceRenderer)), True)
            self.assertNotIn('gemma4_renderer', sys.modules)
        finally:
            sys.modules['gemma4_renderer'] = loaded
        # A second copy of the module (another file object) is not the loaded module's class.
        spec = importlib.util.spec_from_file_location('gemma4_renderer_copy', GR.__file__)
        copy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy_module)
        self.assertIs(RR.trusted_source_renderer(bare(copy_module.Gemma4SourceRenderer)), False)

    def test_one_predicate_at_every_site(self):
        import bootstrap_contract
        import engine_assembly
        import frontend_registry
        for module in (bootstrap_contract, engine_assembly, frontend_registry):
            with self.subTest(module=module.__name__):
                self.assertIs(module.trusted_source_renderer, RR.trusted_source_renderer)
        self.assertFalse(hasattr(frontend_registry, '_trusted_renderer'))
        found = []
        for name in STARTUP_MODULES:
            path = ROOT / 'source-overlay' / name
            for file in (sorted(path.glob('*.py')) if path.is_dir() else (path,)):
                found += renderer_type_checks(file)
        self.assertEqual(found, [])


# ---- 2. the real assembly -------------------------------------------------------------------------

class AssemblyTests(unittest.TestCase):
    def check_active(self, renderer):
        assembly, core = assemble(self, renderer)
        self.assertEqual(assembly.state, 'active')
        self.assertIs(assembly.renderer, renderer)
        self.assertIs(assembly.provider._render, renderer)
        self.assertIs(assembly.require_active(), True)
        self.assertIs(assembly.readiness(), True)
        self.assertIs(core._remote_ra_assembly, assembly)
        self.assertIs(core.scheduler.connector.connector_scheduler._policy_runtime_context, assembly.provider)
        return assembly

    def test_the_gemma_renderer_assembles(self):
        """Before the fix: PolicyRuntimeAbort 'engine renderer and dynamic provider required' (the GPU
        failure of jobs 3398198 and 3398614)."""
        self.check_active(bare(GR.Gemma4SourceRenderer))

    @unittest.skipUnless(*NEEDS_PINNED)
    def test_the_pinned_gemma_renderer_assembles(self):
        renderer = pinned_gemma()
        assembly = self.check_active(renderer)
        # The readiness gate rechecks the Gemma renderer's own identity (here: the pinned template).
        renderer.tokenizer = NS(**{k: getattr(HF, k) for k in ('is_fast', 'backend_tokenizer')},
                                chat_template='{{ other }}')
        with self.assertRaisesRegex(Exception, 'pinned Gemma template'):
            assembly.readiness()

    def test_the_qwen_renderer_assembles(self):
        self.check_active(bare(RR.EngineSourceRenderer))

    def test_subclasses_are_refused_at_construction(self):
        for cls in (QwenSub, GemmaSub):
            with self.subTest(cls=cls.__name__), \
                    self.assertRaisesRegex(Exception, 'engine renderer and dynamic provider required'):
                assemble(self, bare(cls), attach=False)

    def test_the_readiness_gate_refuses_another_renderer(self):
        """Before the fix the gate refused the Gemma renderer itself ('dynamic provider must own the
        engine source renderer'); now it refuses a subclass swapped in after construction."""
        assembly = self.check_active(bare(GR.Gemma4SourceRenderer))
        other = bare(GemmaSub)
        assembly.readiness.renderer = other
        with self.assertRaisesRegex(Exception, 'dynamic provider must own the engine source renderer'):
            assembly.readiness()
        assembly.readiness.renderer = assembly.renderer
        self.assertIs(assembly.readiness(), True)

    def test_the_frontend_capture_takes_the_same_two(self):
        from frontend_registry import TrustedFrontendCapture
        renderer = NS(tokenizer=None, model_config=NS(is_encoder_decoder=False, enable_prompt_embeds=False))
        for cls, ok in ((RR.EngineSourceRenderer, True), (GR.Gemma4SourceRenderer, True), (QwenSub, False),
                        (GemmaSub, False)):
            with self.subTest(cls=cls.__name__):
                make = lambda: TrustedFrontendCapture(renderer=renderer, source_renderer=bare(cls),  # noqa: E731
                                                      verify_identity=lambda: True)
                if ok:
                    make()
                else:
                    with self.assertRaisesRegex(Exception, 'trusted source renderer'):
                        make()


# ---- 3. the native startup path ---------------------------------------------------------------------

class HfRenderer:
    """vllm.renderers.hf.HfRenderer's surface that the runtime binds: its tokenizer and model config
    and the four methods ActualRendererBinding pins (the protocols render on their transcripts)."""

    def __init__(self, tokenizer, model_config):
        self.tokenizer, self.model_config = tokenizer, model_config

    async def render_messages_async(self, messages, chat_params):
        raise AssertionError('the protocols render on their transcripts')

    async def tokenize_prompts_async(self, prompts, params):
        return [dict(prompts[0])]

    async def process_for_engine_async(self, prompt, arrival_time, skip_mm_cache=False):
        return dict(type='token', prompt=prompt['prompt'], prompt_token_ids=list(prompt['prompt_token_ids']))

    def _apply_prompt_extras(self, tokenized, extras):
        return None


def safe_apply_chat_template(model_config, tokenizer, conversation, *, chat_template, tokenize, return_dict,
                             **kwargs):
    """vLLM's text render of a conversation: the tokenizer's own template call."""
    assert tokenize is False and return_dict is False
    return tokenizer.apply_chat_template(conversation, chat_template=chat_template, tokenize=False, **kwargs)


class UniProcExecutor:
    """vllm.v1.executor.uniproc_executor.UniProcExecutor: one driver worker; collective_rpc runs the
    named method on it (the wrapper forwards to the worker)."""

    def __init__(self, wrapper, parallel_config, scheduler_config):
        self.driver_worker, self.parallel_config, self.scheduler_config = wrapper, parallel_config, scheduler_config
        self.methods = []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.methods.append(method)
        return [getattr(self.driver_worker, method)(*args, **(kwargs or {}))]


class WorkerWrapperBase:
    """vllm.v1.worker.worker_base.WorkerWrapperBase with the overlay's registration method."""

    def __init__(self, worker):
        self.worker = worker

    def remote_ra_register_dynamic(self, request):
        gate = self.__dict__.get('_remote_ra_dynamic_gate')
        if gate is None:
            raise RuntimeError('dynamic worker registration not armed')
        return gate.register(request)

    def __getattr__(self, name):
        return getattr(self.__dict__['worker'], name)


class ModelRunnerOutput:
    def __init__(self, req_ids, req_id_to_index):
        self.req_ids, self.req_id_to_index = req_ids, req_id_to_index
        self.policy_runtime_publications = ()


class Worker:
    """The loaded GPUWorker: the overlay's readiness probe (worker_readiness.worker_probe, real) and
    the live-v2 attach, whose evidence echoes the startup record."""

    def __init__(self, model_root, layout):
        from policy_runtime.worker_transport import RuntimeWorkerOutbox
        self.layout = layout
        text = NS(num_hidden_layers=60, _commit_hash=MP.GEMMA4_REVISION)
        self.model_runner = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
                               model_config=NS(model=str(model_root), hf_text_config=text, revision=MP.GEMMA4_REVISION),
                               _policy_runtime_publication_sink=RuntimeWorkerOutbox())
        self.startups = []

    def execute_model(self, scheduler_output):
        raise AssertionError('no model forward at startup')

    def remote_ra_worker_readiness(self, request):
        from worker_readiness import worker_probe
        return worker_probe(self, request)

    def remote_ra_live_v2_attach(self, startup):
        from remote_ra_live_v2 import context_limits as CL
        from remote_ra_live_v2 import language_model_only as T
        from remote_ra_live_v2.kv_layout import to_additional_config
        self.startups.append(copy.deepcopy(startup))
        evidence = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS',
                        loaded_sources_verified=startup['loaded_sources_verified'], copies_drained=True,
                        source=startup['source'], source_closure=dict(stand_in='worker'),
                        vector_space=dict(stand_in='worker'), tensor_geometry=[dict(stand_in='worker')],
                        kv_layout=to_additional_config(self.layout))
        for key in (CL.RECORD_KEY, 'admission', 'decode_fastpath', 'decode_fastpath2', 'release_idle_pages',
                    'cuda_graphs'):
            if key in startup:
                evidence[key] = copy.deepcopy(startup[key])
        if startup.get(T.CONFIG_KEY):
            evidence[T.CONFIG_KEY] = T.record(dict(language_model_only=True,
                                                   effective_server_config=dict(language_model_only=True)))
        return evidence


class ProjectionTorch:
    """build_xuanxue_projection's torch calls (a CPU generator, randint, the int8/float32 casts and the
    in-place affine), for an interpreter without torch: the projection only names the provenance here
    (the worker's vectors are the GPU's)."""
    int8, float32 = 'int8', 'float32'

    class Generator:
        def __init__(self, device):
            assert device == 'cpu'

        def manual_seed(self, seed):
            import numpy
            self.rng = numpy.random.default_rng(seed)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def to(self, dtype):
            return ProjectionTorch.Tensor(self.value.astype('float32'))

        def mul_(self, x):
            self.value *= x
            return self

        def sub_(self, x):
            self.value -= x
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self.value

    @classmethod
    def randint(cls, low, high, size, generator, dtype):
        return cls.Tensor(generator.rng.integers(low, high, size=size).astype('int8'))


@contextlib.contextmanager
def stand_in_modules(values):
    """``values`` in sys.modules for the run, then the previous entries back; every other module the
    run imports stays (patch.dict would drop them while the packages that imported them keep their
    attributes). The runtime imports vLLM inside functions only, so nothing keeps a stand-in."""
    saved = {name: sys.modules[name] for name in values if name in sys.modules}
    sys.modules.update(values)
    try:
        yield
    finally:
        for name in values:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def vllm_modules(world, renderer_box):
    """The vLLM modules the startup path imports, with the classes it checks by type."""
    def module(name, **values):
        value = types.ModuleType(name)
        value.__dict__.update(values)
        value.__path__ = []
        return value
    from enum import Enum

    class EngineCoreRequestType(Enum):
        ADD, ABORT, START_DP_WAVE, UTILITY, EXECUTOR_FAILED, WAKEUP = (b'\x00', b'\x01', b'\x02', b'\x03',
                                                                      b'\x04', b'\x05')
    hf = module('vllm.renderers.hf', HfRenderer=HfRenderer, safe_apply_chat_template=safe_apply_chat_template)
    return {
        'vllm': module('vllm'), 'vllm.renderers': module('vllm.renderers',
                                                         renderer_from_config=lambda config: renderer_box[0]),
        'vllm.renderers.hf': hf, 'vllm.renderers.base': module('vllm.renderers.base'),
        'vllm.renderers.params': module('vllm.renderers.params'), 'vllm.entrypoints': module('vllm.entrypoints'),
        'vllm.entrypoints.chat_utils': module('vllm.entrypoints.chat_utils'), 'vllm.v1': module('vllm.v1'),
        'vllm.v1.engine': module('vllm.v1.engine', EngineCoreRequest=EngineCoreRequest,
                                 EngineCoreRequestType=EngineCoreRequestType),
        'vllm.v1.serial_utils': module('vllm.v1.serial_utils', MsgpackDecoder=Decoder),
        'vllm.v1.executor': module('vllm.v1.executor'),
        'vllm.v1.executor.uniproc_executor': module('vllm.v1.executor.uniproc_executor',
                                                    UniProcExecutor=UniProcExecutor),
        'vllm.v1.worker': module('vllm.v1.worker'),
        'vllm.v1.worker.worker_base': module('vllm.v1.worker.worker_base', WorkerWrapperBase=WorkerWrapperBase),
        'vllm.v1.outputs': module('vllm.v1.outputs', ModelRunnerOutput=ModelRunnerOutput),
        'vllm.v1.core': module('vllm.v1.core'),
        'vllm.v1.core.single_type_kv_cache_manager': module(
            'vllm.v1.core.single_type_kv_cache_manager', FullAttentionManager=world.single.FullAttentionManager),
    }


class Loaded:
    """LiveIdentity of the release (remote_ra_linux.live_identity) for the in-process run: the installed
    Gemma release's runtime manifest and its digest, with the pinned model directory as its model root;
    the loaded modules are recorded (their digests against the release are check_git_release's)."""

    def __init__(self, release, model_root):
        from hashlib import sha256
        raw = (Path(release) / 'runtime-manifest.json').read_bytes()
        self.manifest = dict(json.loads(raw), model_root=str(model_root))
        self.manifest_hash, self.model_root, self.modules = sha256(raw).hexdigest(), Path(model_root), set()

    def verify_loaded(self):
        return True

    def verify_module(self, name):
        if sys.modules.get(name) is None:
            raise RuntimeError('required live module missing: ' + name)
        self.modules.add(name)
        return True


class EngineThread:
    """The engine process's owner thread: runs the submitted startup steps in order and, once B0 is
    released, polls the private channel between them (EngineCore's busy loop)."""

    def __init__(self):
        self.tasks, self.authority, self.polling, self.failure = queue.Queue(), None, False, None
        self.thread = threading.Thread(target=self._run, name='engine-owner', daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                fn, box = self.tasks.get(timeout=0.002)
            except queue.Empty:
                if self.polling:
                    try:
                        self.authority.poll()
                    except BaseException as failure:
                        self.failure, self.polling = failure, False
                continue
            if fn is None:
                return
            try:
                box['value'] = fn()
            except BaseException as failure:
                box['error'] = failure
            box['done'].set()

    def call(self, fn, timeout=120):
        box = dict(done=threading.Event())
        self.tasks.put((fn, box))
        if not box['done'].wait(timeout):
            import faulthandler
            faulthandler.dump_traceback(sys.__stderr__, all_threads=True)
            raise AssertionError('engine thread step timed out')
        if 'error' in box:
            raise box['error']
        return box['value']

    def stop(self):
        self.polling = False
        self.tasks.put((None, None))
        self.thread.join(10)


class Recorder:
    """A private pipe end that records the length of every frame it sends."""

    def __init__(self, pipe):
        self.pipe, self.sent = pipe, []

    def send_bytes(self, raw):
        self.sent.append(len(raw))
        self.pipe.send_bytes(raw)

    def recv_bytes(self, limit):
        return self.pipe.recv_bytes(limit)

    def poll(self, timeout=0):
        return self.pipe.poll(timeout)


SELECTION = dict(active_token_budget=8192)
LAUNCHES = {
    'mswea-xml-full': ('full', None, dict(kv_cache_gib=32, max_tokens=4096, max_model_len=131072,
                                          language_model_only=True)),
    'mswea-xml-goal': ('goal_query', dict(SELECTION, no_goal_rewriter=True),
                       dict(kv_cache_gib=32, max_tokens=4096, max_model_len=131072, language_model_only=True,
                            cuda_graphs='full', decode_fastpath=True)),
    'kv-session': ('observation_query', dict(active_token_budget=32768),
                   dict(protocol='kv-session', kv_session_thinking='off', max_running=16, max_model_len=262144,
                        release_idle_pages=True, dram_arena_gib=150, host_memory_shares=2, kv_cache_gib=28,
                        language_model_only=True)),
    'chat-text': ('observation_query', SELECTION,
                  dict(protocol='chat-text', kv_cache_gib=28, max_tokens=4096, max_model_len=65536,
                       language_model_only=True, max_running=20, dram_arena_gib=64, connector_pool_gib=1,
                       release_idle_pages=True, cuda_graphs='full', decode_fastpath=True, chat_text_slots=512,
                       chat_text_idle_timeout=600, chat_text_evict_after=30)),
}
# goal_query on chat-text (gemma-tb-goal): the Terminal-Bench server above with goal_query (no rewriter
# process on the CPU) and --admission-overlap 16.
LAUNCHES['chat-text-goal'] = ('goal_query', dict(SELECTION, no_goal_rewriter=True),
                              dict(LAUNCHES['chat-text'][2], admission_overlap=16))
TASKS = [dict(instance_id='g%02d' % k, problem_statement='fix check %02d of the window sums' % k)
         for k in range(2)]


def parsed(effective):
    """vLLM's parsed arguments of a server pinning ``effective`` (test_gemma_flags.parsed)."""
    value = dict(effective, attention_backend=NS(name=MP.GEMMA4_ATTENTION_BACKEND), mamba_cache_mode='none',
                 mamba_block_size=None, gdn_prefill_backend=None)
    if 'compilation_config' in effective:
        pinned = effective['compilation_config']
        value['compilation_config'] = NS(mode=NS(name=pinned['mode']),
                                         cudagraph_mode=NS(name=pinned['cudagraph_mode']))
    return NS(trust_request_chat_template=False, **value)


@unittest.skipUnless(*NEEDS_PINNED)
class StartupPathTests(unittest.TestCase):
    """The engine's and the frontend's native startup of a Gemma server, then its first request."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import install
        from test_gemma_install import gemma_fixture
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, cls.release = gemma_fixture(cls.root)
            install.install(site, gemma, cls.release, package_root=package, apply=True, model_profile=MP.GEMMA4)

    def launch(self, name):
        import serve
        policy, selection, kw = LAUNCHES[name]
        tasks = copy.deepcopy(TASKS) if 'protocol' not in kw else None
        output = self.root / name
        with redirect_stdout(io.StringIO()), patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            _, config, command, env = serve.prepare(self.release, tasks, policy, output, 8000, selection,
                                                    page_tokens=P, **kw)
        # serve.main's startup files (the API server reads startup.json through its pin).
        output.mkdir()
        (output / 'startup.json').write_bytes(serve.wire(config))
        self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
        return config, command, env

    def startup(self, name):
        """The engine's and the frontend's startup of launch ``name``; returns the run's objects."""
        from frontend_registry import FrontendRegistry
        from private_authority.protocol import Channel, EngineFence, MAX_FRAME
        from remote_ra_http import config as HC
        from remote_ra_linux import native_authority as NA
        from remote_ra_live_v2 import kv_layout as I0
        import startup_runtime
        import test_gemma_kv_pools as GK
        import test_kv_pools as KP
        config, command, env = self.launch(name)
        stack = self.enterContext(ExitStack())
        # A 262144-token server pins the module request caps at startup (context_limits): restore them.
        from remote_ra_live_v2 import context_limits as CL
        import importlib
        caps = [(importlib.import_module(n), a) for n, a, _ in CL.CAP_TARGETS]
        caps = [(m, a, getattr(m, a)) for m, a in caps]
        stack.callback(lambda: [setattr(m, a, v) for m, a, v in caps])
        stack.enter_context(patch.dict(os.environ, {k: v for k, v in env.items()
                                                    if k.startswith(('REMOTE_RA_', 'VLLM_POLICY'))}))
        os.environ.pop('VLLM_POLICY_RUNTIME_MANIFEST', None)
        value = HC.load()
        self.assertEqual(value['model_profile'], MP.GEMMA4)
        model_root = G.metadata_dir()
        layout = I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: value['kv_layout']}))
        # vLLM's KV manager over the window layout: one global group in pool 0, five sliding groups.
        world = GK.gemma_world()
        window = MP.GEMMA4_GEOMETRY.sliding_window
        groups = [world.iface.KVCacheGroupSpec(['g'], GK.full_spec(world, P))]
        groups += [world.iface.KVCacheGroupSpec(['s%d' % g], GK.sliding_spec(world, P, window=window))
                   for g in range(1, 6)]
        kv_config = world.iface.KVCacheConfig(num_blocks=4096, kv_cache_tensors=[], kv_cache_groups=groups,
                                              group_pool=(0,) + (1,) * 5,
                                              num_blocks_by_pool=(4096, I0.window_pool_blocks(layout, 5)))
        kv = KP.kv_manager(world, kv_config, hash_block_size=P, scheduler_block_size=P)
        effective = value['effective_server_config']
        vllm_config = topology(model_root, max_model_len=effective['max_model_len'], seqs=effective['max_num_seqs'],
                               additional=json.loads(command[command.index('--additional-config') + 1]),
                               pool=value['connector_pool_bytes'])
        engine_renderer = HfRenderer(HF, vllm_config.model_config)
        frontend_renderer = HfRenderer(HF, vllm_config.model_config)
        box = [engine_renderer]
        stack.enter_context(stand_in_modules(vllm_modules(world, box)))
        try:
            import torch  # noqa: F401
            projection = {}
        except ImportError:
            projection = {'torch': ProjectionTorch}         # the engine startup only
        stack.enter_context(redirect_stdout(io.StringIO()))
        # The private launch: one identity and key, the pipe's two ends (the frontend's records frames).
        loaded = Loaded(self.release, model_root)
        self.assertEqual(MP.from_manifest(loaded.manifest), MP.GEMMA4)
        identity, key, frontend_pipe, engine_pipe = private_pair(loaded.manifest_hash)
        self.addCleanup(frontend_pipe.close)
        self.addCleanup(engine_pipe.close)
        sent = Recorder(engine_pipe)
        authorities = {}

        def current():
            authority = authorities.get(threading.get_ident())
            NA.require(type(authority) in (NA.NativeManager, NA.NativeEngine),
                       'native authority is not installed in this thread')
            authority.owner_check()
            NA.require(not authority.revoked.is_set(), 'native authority revoked')
            return authority
        stack.enter_context(patch.object(NA, 'current', current))
        stack.enter_context(patch.object(startup_runtime, '_factory', None))
        # ---- the API server's half: the NativeManager reader (its _run, real) --------------------
        manager = NA.NativeManager.__new__(NA.NativeManager)
        manager.owner, manager.policy, manager.loaded = threading.get_ident(), value['policy'], loaded
        manager.revoked, manager.done, manager.b0, manager.f1 = (threading.Event() for _ in range(4))
        manager.commands, manager.lock, manager.close_lock = queue.Queue(maxsize=1), threading.RLock(), threading.Lock()
        manager.channel = manager.identity = manager.reader = manager.watchdog = None
        manager.process, manager.pipe = NS(pid=None), None     # the provisioning stands in below
        manager._codec, manager.closed, manager.taken, manager.live_v2_readiness = None, False, True, None
        manager.error, manager.started_at = None, time.monotonic()
        manager.deadline, manager.diagnostic_phase = manager.started_at + 600, 'parent-created'
        manager.closing, manager.closed_receipt, manager.close_state = False, None, 'open'
        authorities[manager.owner] = manager

        def parent_provision(pipe, *, child_pid, manifest, deadline):
            self.assertEqual(manifest, loaded.manifest_hash)
            return Channel(frontend_pipe, identity, key, 'frontend')
        stack.enter_context(patch.object(NA, 'parent_provision', parent_provision))
        reader = threading.Thread(target=manager._run, name='native-private-authority', daemon=True)
        # ---- the engine's half --------------------------------------------------------------------
        engine = EngineThread()
        worker = Worker(model_root, layout)
        wrapper = WorkerWrapperBase(worker)
        core = Core(vllm_config, scheduler(kv, max_tokens=effective['max_num_batched_tokens']),
                    UniProcExecutor(wrapper, vllm_config.parallel_config, vllm_config.scheduler_config))

        def stop():
            manager.revoked.set()
            manager.done.wait(10)
            engine.stop()
        self.addCleanup(stop)

        def engine_startup():
            authority = NA.NativeEngine.__new__(NA.NativeEngine)
            authority.owner, authority.revoked, authority.stop = threading.get_ident(), threading.Event(), threading.Event()
            authority.policy, authority._codec, authority.loaded = value['policy'], None, loaded
            authority.core = authority.gate = authority.assembly = None
            authority.started_at = time.monotonic()
            authority.deadline, authority.diagnostic_phase = authority.started_at + 600, 'engine-provision'
            authority.channel = Channel(sent, identity, key, 'engine')
            authority.identity, authority.fence = identity, EngineFence(authority.revoked)
            authorities[authority.owner] = authority
            engine.authority = authority
            core._remote_ra_authority = authority
            startup_runtime.install_server_factory(authority.build)
            ticket = startup_runtime.begin(core)
            startup_runtime.activate(core, ticket)              # NativeEngine.build -> EngineAssembly
            from growing_startup import attach_dormant
            from remote_ra_live_v2.startup import attach_engine_dormant
            from remote_ra_live_v2.composition import compose_engine
            from policy_runtime.live_composition import engine_b0_after_worker, engine_native_ready
            attach_dormant(core, ticket)
            attach_engine_dormant(core, ticket)
            compose_engine(core, ticket)
            engine_b0_after_worker(core, authority)             # release_b0: PREPARED_LIVE
            engine_native_ready(authority, 'INPUT_READY')
            engine_native_ready(authority, 'HANDSHAKE_READY')
            engine.polling = True
            return authority
        reader.start()
        try:
            with stand_in_modules(projection):
                authority = engine.call(engine_startup)
        except BaseException as failure:
            raise AssertionError('engine startup failed; the API server reader: %r' % (manager.error,)) from failure
        self.assertTrue(manager.b0.wait(30), manager.error)
        prepared = max(sent.sent)
        self.assertLessEqual(prepared, MAX_FRAME)
        # ---- the frontend: live_frontend._bind (real), F1 ---------------------------------------
        from remote_ra_linux import live_frontend
        engine_core = NS(client_index=0, client_count=1, encoder=NS(encode=lambda value: [b'frame']))
        llm = NS(renderer=frontend_renderer, engine_core=engine_core)
        engine_core._remote_ra_authority = manager
        render = NS(renderer=frontend_renderer, chat_template=(model_root / 'chat_template.jinja').read_text('utf-8'))
        serving = NS(engine_client=llm, renderer=frontend_renderer, openai_serving_render=render)
        state = NS(engine_client=llm, openai_serving_chat=serving)
        binding = asyncio.run(live_frontend._bind(llm, state, parsed(effective), manager))
        self.assertTrue(manager.f1.is_set())
        # The API server's ready file (evidence.write_ready, after uvicorn starts): the worker's
        # attach record against the parent's PREPARED_LIVE readiness; Linux's /proc ids stand in.
        app = NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})])
        read_text = Path.read_text

        def proc(path, *args, **kwargs):
            if path.as_posix() == '/proc/sys/kernel/random/boot_id':
                return 'boot\n'
            if path.as_posix() == '/proc/self/stat':
                return '1 (python) ' + ' '.join(['0'] * 50)
            return read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', proc):
            state._remote_ra_http_evidence.write_ready(app, NS(started=True))
        ready = json.loads((Path(value['output_directory']) / 'ready.json').read_bytes())
        self.assertEqual((ready['protocol'], ready['epoch']), (value['protocol'], identity.epoch))
        self.assertIs(type(llm._remote_ra_frontend_registry), FrontendRegistry)
        self.assertEqual(engine.call(lambda: authority.fence.state), 'ADMITTED')
        return NS(config=value, core=core, authority=authority, manager=manager, engine=engine, llm=llm,
                  state=state, binding=binding, worker=worker, wrapper=wrapper, prepared=prepared, loaded=loaded,
                  frontend=llm._remote_ra_loaded_renderer.renderer)

    def check_startup(self, run):
        """What every protocol's startup must have reached."""
        from private_authority.protocol import MAX_FRAME
        from remote_ra_live_v2.composition import EngineWorkerProxy
        assembly = run.core._remote_ra_assembly
        self.assertEqual(run.engine.call(lambda: (assembly.state, assembly.require_active())), ('active', True))
        engine_renderer = assembly.renderer
        self.assertIs(type(engine_renderer), GR.Gemma4SourceRenderer)
        self.assertIs(type(run.frontend), GR.Gemma4SourceRenderer)
        self.assertIsNot(run.frontend, engine_renderer)          # each process binds its own
        self.assertEqual(run.core.model_executor.methods, ['remote_ra_register_dynamic', 'remote_ra_live_v2_attach'])
        self.assertIs(type(run.core._remote_ra_live_v2_boundary.proxy), EngineWorkerProxy)
        self.assertEqual(run.manager.live_v2_readiness['native_ready'], True)
        # PREPARED_LIVE, the largest startup frame (1,039 bytes here; its schema is fixed): a GPU run's
        # PIDs (7 digits) add at most 12 bytes.
        self.assertLess(run.prepared + 12, MAX_FRAME)
        self.assertIn('gemma4_renderer', run.loaded.modules)

    async def capture(self, run, messages, transcript):
        from trusted_renderer_adapter import TOKEN_OPTION_NAMES
        capture = run.llm._remote_ra_frontend_registry.capture_factory
        options = dict(truncate_prompt_tokens=None, pad_prompt_tokens=None, truncation_side=None,
                       do_lower_case=False, add_special_tokens=False)
        self.assertEqual(set(options), set(TOKEN_OPTION_NAMES))
        return await capture.capture(request=NS(stream=False, n=1, use_beam_search=False), messages=messages,
                                     chat_params=None, tok_params=NS(**options), transcript=transcript)

    def first_request(self, run, captured, session, transcript, max_tokens):
        """The engine's render of the first request (registered_route's order: the session's
        command authentication, then private_endpoint.render_claim) on the frontend's capture."""
        from contracts import canonical_prompt_hash
        rid, external = 'r-' + secrets.token_hex(8), 'x-' + secrets.token_hex(8)
        tokens = list(captured.tokens)
        params = NS(n=1, temperature=0, max_tokens=max_tokens, stop_token_ids=sorted(transcript.stop_ids),
                    eos_token_id=None)
        raw = secrets.token_bytes(16)
        Decoder.frames[raw] = NS(request_id=rid, external_req_id=external, prompt_token_ids=tokens,
                                 sampling_params=params, client_index=0, current_wave=0, data_parallel_rank=None,
                                 mm_features=None, prompt_embeds=None, prompt_is_token_ids=None, lora_request=None,
                                 pooling_params=None)
        generation = 'g-' + rid
        body = dict(request=rid, session=session, turn=0, generation=generation,
                    core=base64.b64encode(raw).decode(), source=json.loads(captured.source_json),
                    external_id=external, prompt_hash=canonical_prompt_hash(tokens))
        command = NS(session_id=session, metadata=dict(instance_id=session, turn=0, generation=generation))
        core = run.core

        def render():
            core._remote_ra_live_session_authority.authenticate_command(command)
            request, binding = core._remote_ra_private_v2_endpoint.render_claim(body)
            return request, binding
        request, binding = run.engine.call(render)
        self.assertIs(request, core.preprocessed[rid])
        self.assertEqual(list(binding.tokens), tokens)
        return binding

    def test_mswea_xml(self):
        from remote_ra_live_v2.mswea_protocol import GEMMA4_XML
        for name in ('mswea-xml-full', 'mswea-xml-goal'):
            with self.subTest(launch=name):
                run = self.startup(name)
                self.check_startup(run)
                endpoint = run.core._remote_ra_private_v2_endpoint
                self.assertEqual({type(t).__name__ for t in endpoint.transcripts.values()}, {'Transcript'})
                self.assertIs(endpoint.transcript.grammar, GEMMA4_XML)
                self.assertIsNotNone(run.core.scheduler._remote_ra_output_stop)     # MarkerStop
                live = run.llm._remote_ra_live_four_policy
                lane = live.sessions[1]
                messages = [dict(role='system', content=G.SYSTEM),
                            dict(role='user', content='<pr_description>\n%s\n</pr_description>'
                                 % TASKS[1]['problem_statement'])]
                captured = asyncio.run(self.capture(run, messages, lane.transcript))
                self.assertEqual(captured.tokens[0], MP.BOS)
                sid = run.core._remote_ra_live_session_authority.ids[1]
                binding = self.first_request(run, captured, sid, endpoint.transcripts[sid], run.config['max_tokens'])
                self.assertEqual(binding.local_turn, 0)
                self.doCleanups()

    def test_kv_session_thinking_off(self):
        from remote_ra_live_v2 import kv_session_protocol as K
        run = self.startup('kv-session')
        self.check_startup(run)
        endpoint = run.core._remote_ra_private_v2_endpoint
        self.assertTrue(all(K.kv(t) for t in endpoint.transcripts.values()))
        self.assertIsNone(getattr(run.core.scheduler, '_remote_ra_output_stop', None))
        live = run.llm._remote_ra_live_four_policy
        self.assertIs(live.kv.empty_header, True)
        messages = [dict(role='system', content='You are a memory agent.'),
                    dict(role='user', content='Read the chunks, then answer: which gate?')]
        text = HF.apply_chat_template(GR.openai_parts(messages), chat_template=run.frontend.template, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        ids = HF.encode(text, add_special_tokens=False)
        header = len(HF.encode(K.GEMMA4_HEADER, add_special_tokens=False))
        record = run.config['kv_session']
        body = dict(method=K.METHOD, session_id=secrets.token_hex(16), input_ids=ids, header_tokens=header,
                    prefix_tokens=len(ids) - header, boundary_tokens=0, budget=record['budget'], max_tokens=256,
                    temperature=0, top_p=1, top_k=-1, repetition_penalty=1, seed=None, messages=messages)
        request = K.parse_complete(body, budget=record['budget'], bound=live.max_tokens,
                                   max_model_len=run.config['effective_server_config']['max_model_len'],
                                   empty_header=True)
        lane = live.sessions[0]
        lane.transcript.stage(request)
        captured = asyncio.run(self.capture(run, messages, lane.transcript))
        self.assertEqual(list(captured.tokens), ids)
        sid = run.core._remote_ra_live_session_authority.ids[0]
        self.first_request(run, captured, sid, endpoint.transcripts[sid], 256)

    def test_chat_text(self):
        from remote_ra_live_v2 import chat_text_protocol as C
        run = self.startup('chat-text')
        self.check_startup(run)
        endpoint = run.core._remote_ra_private_v2_endpoint
        self.assertTrue(all(C.chat(t) and t.grammar is C.GEMMA4_CHAT for t in endpoint.transcripts.values()))
        live = run.llm._remote_ra_live_four_policy
        self.assertIsNotNone(live.chat)
        messages = [dict(role='system', content='You are a terminal agent.'),
                    dict(role='user', content='List the files of the repository.')]
        lane = live.sessions[0]
        captured = asyncio.run(self.capture(run, messages, lane.transcript))
        self.assertEqual(captured.tokens[0], MP.BOS)
        sid = run.core._remote_ra_live_session_authority.ids[0]
        self.first_request(run, captured, sid, endpoint.transcripts[sid], 256)

    def test_chat_text_goal_query(self):
        """gemma-tb-goal: the Terminal-Bench server with goal_query starts as observation_query's does,
        with the provisional Terminus 2 Goal inputs pinned, the frontend classifying them and the
        engine's selection authority on goal_query; a Terminus 2 first prompt renders on both sides."""
        import test_terminus2_goal as TG
        from remote_ra_live_v2 import chat_text_protocol as C
        from remote_ra_live_v2 import terminus2_goal as T2
        run = self.startup('chat-text-goal')
        self.check_startup(run)
        self.assertEqual((run.config['policy'], run.config['chat_text_goal'], run.config['kv_layout']['goal_tokens']),
                         ('goal_query', T2.record(), MP.GEMMA4_GOAL_TOKENS))
        self.assertEqual(run.config['admission']['admission_overlap']['permits'], 16)
        endpoint = run.core._remote_ra_private_v2_endpoint
        self.assertTrue(all(C.chat(t) and t.grammar is C.GEMMA4_CHAT for t in endpoint.transcripts.values()))
        self.assertEqual((endpoint.selection.policy, endpoint.selection.rewriter), ('goal_query', None))
        self.assertEqual(endpoint.phase1.goal_tokens, MP.GEMMA4_GOAL_TOKENS)
        live = run.llm._remote_ra_live_four_policy
        self.assertTrue(live.chat.goal_inputs)
        self.assertEqual(live.chat.info['goal_inputs'], T2.record())
        messages = [dict(role='user', content=TG.P0)]
        self.assertEqual(T2.classify(messages).request, 'main')
        lane = live.sessions[0]
        captured = asyncio.run(self.capture(run, messages, lane.transcript))
        self.assertEqual(captured.tokens[0], MP.BOS)
        sid = run.core._remote_ra_live_session_authority.ids[0]
        self.first_request(run, captured, sid, endpoint.transcripts[sid], 256)


if __name__ == '__main__':
    unittest.main()
