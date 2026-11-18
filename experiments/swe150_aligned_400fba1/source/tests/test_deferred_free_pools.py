"""Deferred block frees on the decoupled KV layouts (async-fix). CPU only, no torch.

GPU job 3499729 (perf-async-g1, runtime f162120): every runtime server with --async-scheduling died while
EngineCore constructed the Scheduler. vLLM sets ``defer_block_free`` when batches overlap (async scheduling:
max_concurrent_batches 2) and the KV connector is a consumer (our OffloadingConnector is kv_both), and the
overlay refused that on every decoupled layout (``kv_cache_config.group_pool``: the Qwen layout's FA pool 0 and
Mamba pool 1, Gemma's global pool 0 and sliding pool 1) because its deferred-free list mixed blocks of several
pools and returned them all to pool 0. The async-c CPU tests used fake engines that never constructed the real
Scheduler; test_kv_pools asserted the refusal itself.

The fix: a deferred entry holds (block pool, block) pairs (KVCacheManager.pop_blocks_for_free_by_pool), and
_drain_deferred_frees returns each block to its own pool, in the shared pool's order (the entry reversed, one
free_blocks call per pool) and behind the same fence; the shared pool's path is unchanged.

Here the REAL overlay vllm/v1/core/sched/scheduler.py is executed from its source (test_kv_pools' loader: the
KV cache interface, utils, block pool, single-type managers, coordinator and manager are the overlay's own;
the rest of vLLM is stubbed) and its __init__ runs:

- construction with defer_block_free (two batches, a consumer connector) on the Qwen decoupled layout, the
  Gemma window layout and the shared pool, and the constructor path of every server kind of perf-async-g1's
  plan (serve.prepare's launch of each, its layout, sessions, scheduling flag and connector role, the overlay
  planner's KV cache config) and of the synchronous servers: the test that would have caught the GPU failure;
  on async-gemma also every Gemma 4 server kind with and without --async-scheduling (the window layout's pools);
- deferred frees: nothing returns before its fence, every block returns to its own pool exactly once and in
  the shared pool's order, entries drain FIFO with non-decreasing fences; requests owned by the runtime's
  adapter still free at once (their terminal drain is the fence);
- the shared pool: the path is byte-identical to ea18256's (frozen copies of its two methods).
"""
import ast
from contextlib import redirect_stdout
import enum
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_kv_pools as KP  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402

GiB = 1024 ** 3


# ---- the real Scheduler from its source, with the rest of vLLM stubbed ---------------------------------

class Policy(enum.Enum):
    FCFS = 'fcfs'
    PRIORITY = 'priority'


class PauseState(enum.Enum):
    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class Interface:
    pass


class Connector:
    """The scheduler-side KV connector: only what __init__ calls."""

    def __init__(self):
        self.bound = None

    def bind_gpu_block_pool(self, pool):
        self.bound = pool


class Stub:
    """A stubbed vLLM module: the names given, else a fresh placeholder class per name."""

    def __init__(self, **names):
        self.__dict__.update(names)

    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


_SCHEDULERS = {}


def scheduler_module(w=None):
    """The overlay's scheduler.py executed with test_kv_pools' world (``w``) for the KV cache modules."""
    w = w or KP.world()
    if id(w) in _SCHEDULERS:
        return _SCHEDULERS[id(w)]
    connectors = []

    def create_connector(config, role, kv_cache_config):
        connectors.append(Connector())
        return connectors[-1]
    stubs = dict(w.stubs)
    for name in ('vllm.compilation.cuda_graph', 'vllm.distributed.ec_transfer.ec_connector.base',
                 'vllm.distributed.kv_transfer.kv_connector.v1.base', 'vllm.distributed.kv_transfer.kv_connector.v1.metrics',
                 'vllm.distributed.kv_transfer.kv_connector.v1.offloading.common',
                 'vllm.model_executor.layers.fused_moe.routed_experts_capturer', 'vllm.multimodal.encoder_budget',
                 'vllm.multimodal.utils', 'vllm.v1.core.sched.output', 'vllm.v1.core.sched.utils', 'vllm.v1.engine',
                 'vllm.v1.metrics.perf', 'vllm.v1.outputs', 'vllm.v1.spec_decode.dynamic.utils',
                 'vllm.v1.spec_decode.metrics', 'vllm.v1.structured_output'):
        stubs[name] = Stub()
    stubs.update({
        'vllm.config': Stub(VllmConfig=object),
        'vllm.distributed.ec_transfer.ec_connector.factory': Stub(ECConnectorFactory=None),
        'vllm.distributed.kv_events': Stub(EventPublisherFactory=NS(create=lambda config, index: NS()),
                                           KVEventBatch=object),
        'vllm.distributed.kv_transfer.kv_connector.factory': Stub(
            KVConnectorFactory=NS(create_connector=create_connector)),
        'vllm.distributed.kv_transfer.kv_connector.v1': Stub(KVConnectorRole=NS(SCHEDULER='scheduler', WORKER='worker'),
                                                            KVConnectorBase_V1=object, SupportsHMA=object),
        'vllm.multimodal': Stub(MULTIMODAL_REGISTRY=NS(supports_multimodal_inputs=lambda config: False),
                                MultiModalRegistry=object),
        'vllm.v1.core.encoder_cache_manager': Stub(EncoderCacheManager=lambda cache_size: NS(cache_size=cache_size),
                                                   EncoderDecoderCacheManager=lambda cache_size: NS()),
        'vllm.v1.core.sched.interface': Stub(PauseState=PauseState, SchedulerInterface=Interface),
        'vllm.v1.core.sched.request_queue': Stub(SchedulingPolicy=Policy, RequestQueue=object,
                                                 create_request_queue=lambda policy: []),
        'vllm.v1.metrics.stats': Stub(PrefixCacheStats=KP.Record, SchedulerStats=KP.Record),
        'vllm.v1.request': Stub(Request=object, RequestStatus=KP.Status, StreamingUpdate=object),
        'vllm.v1.utils': Stub(record_function_or_nullcontext=None, tensor_data=None),
    })
    text = (ROOT / 'source-overlay/vllm/v1/core/sched/scheduler.py').read_text('utf-8')
    module = KP.load(text, '_async_fix_scheduler_%d' % len(_SCHEDULERS), stubs, 'overlay:vllm/v1/core/sched/scheduler.py')
    module._test_connectors, module._test_stubs = connectors, stubs
    _SCHEDULERS[id(w)] = module
    return module


def vllm_config(*, asynchronous, consumer=True, seqs=16, max_len=131072, batched=16384, block_size=16,
                caching=True, connector=True):
    """The fields Scheduler.__init__ reads: max_concurrent_batches 2 with async scheduling (pp 1, the V1
    runner), the KV transfer config of serve.py (kv_both: a consumer)."""
    return NS(
        scheduler_config=NS(max_num_seqs=seqs, max_num_scheduled_tokens=None, max_num_batched_tokens=batched,
                            policy='fcfs', watermark=0.0, scheduler_reserve_full_isl=False,
                            async_scheduling=asynchronous, long_prefill_token_threshold=0),
        cache_config=NS(enable_prefix_caching=caching, mamba_cache_mode='align', num_gpu_blocks=1000,
                        block_size=block_size),
        lora_config=None, kv_events_config=None,
        parallel_config=NS(decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1,
                           data_parallel_index=0),
        observability_config=NS(kv_cache_metrics=False, enable_mfu_metrics=False),
        model_config=NS(is_encoder_decoder=False, max_model_len=max_len, is_diffusion=False,
                        enable_return_routed_experts=False),
        kv_transfer_config=(NS(is_kv_consumer=consumer, kv_load_failure_policy='recompute') if connector else None),
        ec_transfer_config=None, speculative_config=None, num_speculative_tokens=0,
        max_concurrent_batches=2 if asynchronous else 1, use_v2_model_runner=False)


def construct(kv_cache_config, *, asynchronous, w=None, **kw):
    """Scheduler(...) as EngineCore builds it: the scheduling block is the LCM of the groups' blocks (784 on the
    Qwen decoupled layout, P on the window layout), the hash block P."""
    import math
    module = scheduler_module(w)
    block = math.lcm(*(g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups))
    hashed = min(g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups)
    return module.Scheduler(vllm_config(asynchronous=asynchronous, block_size=hashed, **kw), kv_cache_config, None,
                            block_size=block, hash_block_size=hashed)



def qwen_config(w, sessions, budget=25 * GiB, max_len=131072):
    """The overlay planner's KV cache config of a Qwen server on the decoupled layout at P = 16."""
    layout = I0.for_server(16, sessions, True)
    return KP.plan(w, layout, budget, max_len=max_len)[1]


def gemma_config(w, sessions=16, budget=25 * GiB, forward=8192):
    """The overlay planner's KV cache config of a Gemma 4 server on the window layout (W = 1,024, P = 16)."""
    import test_gemma_kv_pools as GKP
    w = GKP.gemma_world()
    return w, GKP.plan(w, GKP.window_layout(16, sessions, forward), budget)[1]


def legacy_config(w, budget=8 * GiB):
    return KP.plan(w, I0.legacy(), budget)[1]


def pools_of(sched):
    """(block pool, its index) of every KV group, in group order."""
    pools = sched.kv_cache_manager.block_pools
    return [(m.block_pool, pools.index(m.block_pool)) for m in sched.kv_cache_manager.coordinator.single_type_managers]


def held(sched, rid):
    """The request's blocks per group (allocation order) with their pool index, before any free."""
    out = []
    for (pool, index), m in zip(pools_of(sched), sched.kv_cache_manager.coordinator.single_type_managers):
        out.extend((index, block) for block in m.req_to_blocks.get(rid, ()))
    return out


def free_lists(sched):
    """Each pool's free queue, head first (block ids)."""
    return [[b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
            for pool in sched.kv_cache_manager.block_pools]


def allocate(sched, rid, tokens):
    req = KP.request(rid, tokens)
    assert KP.alloc(sched.kv_cache_manager, req, tokens) is not None, rid
    return req


# ---- 1. construction ------------------------------------------------------------------------------------

class ConstructionTests(unittest.TestCase):
    """The REAL overlay Scheduler.__init__ (its source executed, the rest of vLLM stubbed)."""

    def test_defer_block_free_on_every_layout(self):
        w = KP.world()
        cases = (('qwen decoupled', w, qwen_config(w, 4)), ('gemma window',) + gemma_config(w, 4),
                 ('shared pool', w, legacy_config(w)))
        for name, world, config in cases:
            for asynchronous, consumer in ((True, True), (True, False), (False, True)):
                with self.subTest(layout=name, asynchronous=asynchronous, consumer=consumer):
                    sched = construct(config, asynchronous=asynchronous, consumer=consumer, w=world)
                    self.assertIs(sched.defer_block_free, asynchronous and consumer)
                    decoupled = config.group_pool is not None
                    self.assertIs(sched._remote_ra_deferred_by_pool, decoupled)
                    self.assertEqual(len(sched.kv_cache_manager.block_pools), 2 if decoupled else 1)
                    module = scheduler_module(world)
                    self.assertIs(module._test_connectors[-1].bound, sched.kv_cache_manager.block_pools[0])
                    self.assertEqual((sched.sched_step_seq, sched.processed_step_seq, len(sched.deferred_frees)),
                                     (0, 0, 0))

    def test_the_base_refused_it(self):
        """ea18256's scheduler (from Git; skipped in a Git archive) refused exactly this construction: the GPU
        failure of job 3499729."""
        source = KP.base_source.__globals__['subprocess']
        try:
            text = source.run(['git', '-c', 'safe.directory=*', '-C', str(ROOT), 'show',
                               'ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46:source-overlay/vllm/v1/core/sched/scheduler.py'],
                              capture_output=True, check=True, timeout=60).stdout.decode('utf-8')
        except (OSError, source.SubprocessError):
            self.skipTest('base commit not available (Git archive)')
        w = KP.world()
        module = scheduler_module(w)
        base = KP.load(text, '_async_fix_base_scheduler', module._test_stubs, 'base:scheduler.py')
        config = qwen_config(w, 4)
        with self.assertRaisesRegex(ValueError, 'does not support deferred block frees'):
            base.Scheduler(vllm_config(asynchronous=True), config, None, block_size=784, hash_block_size=16)
        base.Scheduler(vllm_config(asynchronous=False), config, None, block_size=784, hash_block_size=16)


# The server kinds of perf-async-g1's plan (development/exp-decouple-pages/perf-async-g1, plan g1): every pair
# is one launch with and without --async-scheduling. Shared: --policy full --kv-cache-gib 25 --page-tokens 16
# --max-model-len 131072 --connector-pool-gib 1, the three fast paths, --deferred-completion-io,
# --language-model-only; chat-text adds 16 running sessions, 96 slots, --dram-arena-gib 16, --max-tokens 4096;
# XML five tasks at 1,536.
PLAN = (('fe', 'chat-text', {}), ('fg', 'chat-text', dict(cuda_graphs='full')),
        ('te', 'chat-text', dict(triton_attn=True)), ('xe', 'mswea-xml', {}),
        ('se', 'chat-text', dict(enable_thinking=True, thinking_history='strip')),
        ('tg', 'chat-text', dict(cuda_graphs='full', triton_attn=True)))
SHARED = dict(kv_cache_gib=25, page_tokens=16, max_model_len=131072, connector_pool_gib=1, decode_fastpath=True,
              decode_fastpath2=True, decode_fastpath3=True, deferred_completion_io=True, language_model_only=True)


def server_kinds(root):
    """serve.prepare's launch of every server of the plan: (label, the KV layout, running sessions, the
    scheduling flag, the connector's role, the KV bytes, the context limit, the forward)."""
    import install
    import serve
    import test_installer as INSTALL
    with redirect_stdout(io.StringIO()):
        package, site, model, release = INSTALL.fixture(root)
        install.install(site, model, release, package_root=package, apply=True)
    tasks = [dict(instance_id='repo__x-%d' % k, problem_statement='Fix x%d.' % k) for k in range(5)]
    out = []
    for pair, protocol, extra in PLAN:
        for asynchronous in (False, True):
            label = pair + ('a' if asynchronous else 's')
            options = dict(SHARED, **extra, **(dict(async_scheduling=True) if asynchronous else {}))
            if protocol == 'chat-text':
                options.update(max_running=16, chat_text_slots=96, dram_arena_gib=16, max_tokens=4096)
                task = None
            else:
                options.update(max_tokens=1536)
                task = [dict(t) for t in tasks]
            with redirect_stdout(io.StringIO()):
                _, config, command, _ = serve.prepare(release, task, 'full', root / label, 8231, None, protocol,
                                                      **options)
            effective = config['effective_server_config']
            raw = config['kv_layout']
            layout = I0.KvLayout(raw['page_tokens'], raw['mamba_tokens'], raw['mamba_slots'], True)
            transfer = json.loads(command[command.index('--kv-transfer-config') + 1])
            out.append(dict(label=label, layout=layout, seqs=effective['max_num_seqs'],
                            asynchronous=effective.get('async_scheduling', False) is True,
                            vllm_async='--async-scheduling' in command, role=transfer['kv_role'],
                            kv_bytes=effective['kv_cache_memory_bytes'], max_len=effective['max_model_len'],
                            batched=effective['max_num_batched_tokens'], caching=effective['enable_prefix_caching']))
    return out


class ServerKindTests(unittest.TestCase):
    """The constructor path of every server kind of perf-async-g1's plan: its KV cache config from the overlay
    planner on serve.prepare's layout and KV bytes, and the Scheduler as EngineCore constructs it, with vLLM's
    max_concurrent_batches (2 with --async-scheduling) and the connector's role (kv_both: a consumer). This is the
    construction every async server of job 3499729 failed in."""

    def test_every_server_constructs(self):
        w = KP.world()
        with tempfile.TemporaryDirectory() as directory:
            kinds = server_kinds(Path(directory))
        self.assertEqual(sorted(k['label'] for k in kinds),
                         sorted(p + s for p, _, _ in PLAN for s in 'as'))
        configs = {}
        for kind in kinds:
            with self.subTest(server=kind['label']):
                self.assertEqual(kind['asynchronous'], kind['vllm_async'])
                self.assertEqual(kind['role'], 'kv_both')
                key = (kind['layout'], kind['kv_bytes'], kind['max_len'])
                if key not in configs:
                    configs[key] = KP.plan(w, kind['layout'], kind['kv_bytes'], max_len=kind['max_len'])[1]
                config = configs[key]
                self.assertIsNotNone(config.group_pool)
                sched = construct(config, asynchronous=kind['asynchronous'], consumer=kind['role'] in
                                  ('kv_both', 'kv_consumer'), seqs=kind['seqs'], max_len=kind['max_len'],
                                  batched=kind['batched'], caching=kind['caching'])
                self.assertIs(sched.defer_block_free, kind['asynchronous'])
                self.assertIs(sched._remote_ra_deferred_by_pool, True)
                self.assertEqual(sched.max_num_running_reqs, kind['seqs'])


# The Gemma 4 servers (async-gemma: the profile takes --async-scheduling): the protocols of its benchmarks, eager
# and with FULL graphs, 8,192- and 4,096-token forwards, each with and without the switch (docs/async-sched/GEMMA.md
# section 8's pairs).
GEMMA_PLAN = (('ge', 'chat-text', {}), ('gg', 'chat-text', dict(cuda_graphs='full')), ('gx', 'mswea-xml', {}),
              ('gk', 'kv-session', {}), ('g4', 'chat-text', dict(max_num_batched_tokens=4096)))
GEMMA_SHARED = dict(kv_cache_gib=25, page_tokens=16, max_model_len=65536, connector_pool_gib=1, decode_fastpath=True,
                    decode_fastpath2=True, decode_fastpath3=True, deferred_completion_io=True, language_model_only=True)


def gemma_server_kinds(root):
    """serve.prepare's launch of every Gemma server kind on a Gemma 4 release: (label, the window layout, running
    sessions, the scheduling flag, the connector's role, the KV bytes, the context limit, the forward)."""
    import install
    import serve
    from test_gemma_install import gemma_fixture
    from remote_ra_live_v2 import model_profile as MP
    with redirect_stdout(io.StringIO()):
        package, site, model, gemma, release = gemma_fixture(root)
        install.install(site, gemma, release, package_root=package, apply=True, model_profile=MP.GEMMA4)
    tasks = [dict(instance_id='repo__g-%d' % k, problem_statement='Fix g%d.' % k) for k in range(4)]
    out = []
    for pair, protocol, extra in GEMMA_PLAN:
        for asynchronous in (False, True):
            label = pair + ('a' if asynchronous else 's')
            options = dict(GEMMA_SHARED, **extra, **(dict(async_scheduling=True) if asynchronous else {}))
            task = None
            if protocol == 'chat-text':
                options.update(max_running=16, chat_text_slots=96, dram_arena_gib=64)   # Terminal-Bench's arena
            elif protocol == 'kv-session':
                options.update(max_running=8, kv_session_thinking='off')
            else:
                task = [dict(t) for t in tasks]
            with redirect_stdout(io.StringIO()):
                _, config, command, _ = serve.prepare(release, task, 'full', root / label, 8231, None, protocol,
                                                      **options)
            effective = config['effective_server_config']
            layout = I0.from_vllm_config(NS(additional_config=effective['additional_config']))
            transfer = json.loads(command[command.index('--kv-transfer-config') + 1])
            out.append(dict(label=label, layout=layout, seqs=effective['max_num_seqs'],
                            asynchronous=effective.get('async_scheduling', False) is True,
                            vllm_async='--async-scheduling' in command, role=transfer['kv_role'],
                            kv_bytes=effective['kv_cache_memory_bytes'], max_len=effective['max_model_len'],
                            batched=effective['max_num_batched_tokens'], caching=effective['enable_prefix_caching']))
    return out


class GemmaServerKindTests(unittest.TestCase):
    """The constructor path of every Gemma 4 server kind (async-gemma): the window layout's KV cache config from the
    overlay planner (the global pool 0 and the sliding pool 1) and the Scheduler with two batches on the async
    servers and the kv_both connector: defer_block_free by pool."""

    def test_every_gemma_server_constructs(self):
        import test_gemma_kv_pools as GKP
        w = GKP.gemma_world()
        with tempfile.TemporaryDirectory() as directory:
            kinds = gemma_server_kinds(Path(directory))
        self.assertEqual(sorted(k['label'] for k in kinds), sorted(p + s for p, _, _ in GEMMA_PLAN for s in 'as'))
        for kind in kinds:
            with self.subTest(server=kind['label']):
                self.assertEqual(kind['asynchronous'], kind['vllm_async'])
                self.assertEqual(kind['role'], 'kv_both')
                self.assertEqual(kind['layout'].state, 'sliding')
                config = GKP.plan(w, kind['layout'], kind['kv_bytes'], max_len=kind['max_len'],
                                  batched=kind['batched'])[1]
                self.assertEqual(set(config.group_pool), {0, 1})
                sched = construct(config, asynchronous=kind['asynchronous'], w=w, seqs=kind['seqs'],
                                  max_len=kind['max_len'], batched=kind['batched'], caching=kind['caching'])
                self.assertIs(sched.defer_block_free, kind['asynchronous'])
                self.assertIs(sched._remote_ra_deferred_by_pool, True)
                self.assertIsNone(sched._remote_ra_mamba_align_tokens)
                # A deferred free on the planned pools returns each block to its own pool after its fence.
                pools = sched.kv_cache_manager.block_pools
                baseline = [pool.get_num_free_blocks() for pool in pools]
                req = allocate(sched, 'probe', min(kind['batched'], 2000))
                if kind['asynchronous']:
                    deferred(sched, req, 2, 1)
                    self.assertEqual(len(sched.deferred_frees), 1)
                    sched.processed_step_seq = 2
                    sched._drain_deferred_frees()
                else:
                    sched.kv_cache_manager.free(req)
                self.assertEqual([pool.get_num_free_blocks() for pool in pools], baseline)


# ---- 2. deferred frees on the decoupled layouts ------------------------------------------------------------

def deferred(sched, req, fence, processed):
    """The request freed while its last scheduled step (``fence``) is still in flight."""
    sched.sched_step_seq, sched.processed_step_seq = fence, processed
    req.last_sched_seq = fence
    sched._free_request_blocks(req)


class DeferredFreeTests(unittest.TestCase):
    def schedulers(self):
        w = KP.world()
        yield 'qwen decoupled', construct(KP.small_config(w, I0.for_server(16, 2, True), 400, 40),
                                          asynchronous=True, w=w)
        import test_gemma_e2e as GE
        import test_gemma_kv_pools as GKP
        gw = GKP.gemma_world()
        layout = GE.window_layout('full', 2)
        yield 'gemma window', construct(GE.kv_cache_config(gw, layout, 400), asynchronous=True, w=gw)

    def test_each_block_returns_to_its_own_pool_after_its_fence(self):
        for name, sched in self.schedulers():
            with self.subTest(layout=name):
                self.assertTrue(sched.defer_block_free and sched._remote_ra_deferred_by_pool)
                pools = sched.kv_cache_manager.block_pools
                baseline = [p.get_num_free_blocks() for p in pools]
                req = allocate(sched, 'a', 1000)            # several FA blocks, Mamba or window blocks
                sched.kv_cache_manager.coordinator.remove_skipped_blocks('a', 1000)   # window: null front blocks
                blocks = held(sched, 'a')
                self.assertEqual({i for i, _ in blocks}, {0, 1})
                live = [(i, b) for i, b in blocks if not b.is_null]
                before = [p.get_num_free_blocks() for p in pools]
                counts = {id(b): b.ref_cnt for _, b in blocks}
                deferred(sched, req, 3, 1)
                self.assertEqual(len(sched.deferred_frees), 1)
                self.assertEqual(sched.deferred_frees[0][0], 3)
                self.assertTrue(all(type(e) is tuple and e[0] in pools for e in sched.deferred_frees[0][1]))
                self.assertEqual([(pools.index(p), b) for p, b in sched.deferred_frees[0][1]], blocks)
                self.assertEqual(held(sched, 'a'), [])                     # bookkeeping popped
                for processed in (1, 2):                                   # before the fence: nothing returns
                    sched.processed_step_seq = processed
                    sched._drain_deferred_frees()
                    self.assertEqual([p.get_num_free_blocks() for p in pools], before)
                    self.assertTrue(all(b.ref_cnt == counts[id(b)] for _, b in blocks))
                sched.processed_step_seq = 3                               # the fence: every block, its own pool
                sched._drain_deferred_frees()
                self.assertEqual(len(sched.deferred_frees), 0)
                self.assertEqual([p.get_num_free_blocks() for p in pools], baseline)
                lists = free_lists(sched)
                for index, pool in enumerate(pools):
                    mine = [b for i, b in live if i == index]
                    # The shared pool's order: the entry reversed, unhashed blocks at the queue's head.
                    expected = [b.block_id for i, b in reversed(live) if i == index]
                    self.assertEqual(lists[index][:len(expected)], expected)
                    self.assertTrue(all(b.ref_cnt == 0 for b in mine))
                    queue = pool.free_block_queue.get_all_free_blocks()
                    self.assertTrue(all(any(q is b for q in queue) for b in mine))
                    other = pools[1 - index].free_block_queue.get_all_free_blocks()
                    self.assertFalse(any(q is b for q in other for b in mine))
                sched._drain_deferred_frees()                              # nothing twice
                self.assertEqual([p.get_num_free_blocks() for p in pools], baseline)
                self.assertTrue(all(b.ref_cnt >= 0 for _, b in blocks if not b.is_null))

    def test_the_same_pools_as_an_immediate_free(self):
        """Deferred and drained, every pool holds what KVCacheManager.free would have returned to it."""
        for (name, deferring), (_, at_once) in zip(self.schedulers(), self.schedulers()):
            with self.subTest(layout=name):
                for sched in (deferring, at_once):
                    for k, tokens in enumerate((600, 300, 40)):
                        allocate(sched, 'r%d' % k, tokens)
                reqs = [KP.request('r%d' % k, 0) for k in range(3)]
                for req in reqs:
                    deferred(deferring, req, 5, 4)
                    at_once.kv_cache_manager.free(req)
                deferring.processed_step_seq = 5
                deferring._drain_deferred_frees()
                self.assertEqual([p.get_num_free_blocks() for p in deferring.kv_cache_manager.block_pools],
                                 [p.get_num_free_blocks() for p in at_once.kv_cache_manager.block_pools])
                self.assertEqual([sorted(x) for x in free_lists(deferring)], [sorted(x) for x in free_lists(at_once)])
                self.assertEqual([[b.ref_cnt for b in p.blocks] for p in deferring.kv_cache_manager.block_pools],
                                 [[b.ref_cnt for b in p.blocks] for p in at_once.kv_cache_manager.block_pools])

    def test_fifo_with_non_decreasing_fences(self):
        for name, sched in self.schedulers():
            with self.subTest(layout=name):
                pools = sched.kv_cache_manager.block_pools
                baseline = [p.get_num_free_blocks() for p in pools]
                reqs = [allocate(sched, 'f%d' % k, 200 + 50 * k) for k in range(3)]
                sizes = [[sum(1 for i, b in held(sched, r.request_id) if i == p and not b.is_null) for p in (0, 1)]
                         for r in reqs]
                for req, fence in zip(reqs, (2, 2, 4)):
                    deferred(sched, req, fence, 1)
                self.assertEqual([f for f, _ in sched.deferred_frees], [2, 2, 4])
                sched.processed_step_seq = 2
                sched._drain_deferred_frees()
                self.assertEqual([f for f, _ in sched.deferred_frees], [4])
                self.assertEqual([p.get_num_free_blocks() for p in pools],
                                 [baseline[p] - sizes[2][p] for p in (0, 1)])
                sched.processed_step_seq = 3
                sched._drain_deferred_frees()
                self.assertEqual(len(sched.deferred_frees), 1)
                sched.processed_step_seq = 4
                sched._drain_deferred_frees()
                self.assertEqual(([p.get_num_free_blocks() for p in pools], len(sched.deferred_frees)), (baseline, 0))

    def test_a_processed_step_frees_at_once_and_owned_requests_bypass(self):
        for name, sched in self.schedulers():
            with self.subTest(layout=name):
                pools = sched.kv_cache_manager.block_pools
                baseline = [p.get_num_free_blocks() for p in pools]
                deferred(sched, allocate(sched, 'done', 300), 2, 2)       # its last step processed: at once
                self.assertEqual(([p.get_num_free_blocks() for p in pools], len(sched.deferred_frees)), (baseline, 0))
                # The runtime's own requests: the adapter's terminal drain is their fence (KVCacheManager.free).
                sched._remote_ra_growing_adapter = NS(owns=lambda rid: True)
                deferred(sched, allocate(sched, 'owned', 300), 3, 1)
                self.assertEqual(([p.get_num_free_blocks() for p in pools], len(sched.deferred_frees)), (baseline, 0))
                del sched._remote_ra_growing_adapter


# ---- 3. the shared pool: byte-identical to ea18256 ---------------------------------------------------------

# ea18256's two methods, frozen (vllm/v1/core/sched/scheduler.py).
FROZEN = """
def _growing_original__free_request_blocks(self, request):
    if not self.defer_block_free or (
        request.last_sched_seq <= self.processed_step_seq
    ):
        self.kv_cache_manager.free(request)
        return
    blocks = self.kv_cache_manager.pop_blocks_for_free(request)
    if blocks:
        self.deferred_frees.append((self.sched_step_seq, blocks))


def _drain_deferred_frees(self):
    while self.deferred_frees:
        fence, _ = self.deferred_frees[0]
        if fence > self.processed_step_seq:
            break
        _, blocks = self.deferred_frees.popleft()
        self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
"""


class SharedPoolTests(unittest.TestCase):
    def run_case(self, frozen):
        import types
        w = KP.world()
        sched = construct(legacy_config(w), asynchronous=True, w=w)
        self.assertTrue(sched.defer_block_free)
        self.assertIs(sched._remote_ra_deferred_by_pool, False)
        if frozen:
            namespace = {}
            exec(FROZEN, namespace)
            for name in ('_growing_original__free_request_blocks', '_drain_deferred_frees'):
                setattr(sched, name, types.MethodType(namespace[name], sched))
        else:
            def refused(request):
                raise AssertionError('the shared pool took the by-pool path')
            sched.kv_cache_manager.pop_blocks_for_free_by_pool = refused
        trace = []
        reqs = [allocate(sched, 's%d' % k, 800 * (k + 1)) for k in range(4)]
        for req, fence, processed in zip(reqs, (2, 2, 3, 5), (1, 1, 2, 2)):
            deferred(sched, req, fence, processed)
            trace.append(([(f, [b.block_id for b in blocks]) for f, blocks in sched.deferred_frees], free_lists(sched)))
        for processed in (2, 3, 4, 5, 5):
            sched.processed_step_seq = processed
            sched._drain_deferred_frees()
            trace.append(([(f, [b.block_id for b in blocks]) for f, blocks in sched.deferred_frees], free_lists(sched),
                          [b.ref_cnt for b in sched.kv_cache_manager.block_pool.blocks]))
        return trace

    def test_the_shared_pool_path_is_ea18256s(self):
        self.assertEqual(self.run_case(False), self.run_case(True))
        self.assertTrue(self.run_case(False)[0][0])                       # entries were deferred


if __name__ == '__main__':
    unittest.main()
