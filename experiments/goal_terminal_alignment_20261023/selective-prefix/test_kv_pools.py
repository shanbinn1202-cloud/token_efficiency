"""WP1 (exp-decouple-pages): vLLM dual block pools, CPU only, no torch.

The overlay vLLM modules are executed from their source with stub imports
(torch and the rest of vLLM are not installed here). Legacy results are also
compared with the same functions taken from the exp-decouple-pages base commit
(the unpatched overlay) when the Git history is available.
"""
import ast
import contextlib
import enum
from pathlib import Path
import subprocess
import sys
import types
import typing
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'source-overlay'
sys.path.insert(0, str(SOURCE))

from remote_ra_live_v2 import kv_layout as I0

BASE = '8586dbd'          # exp-decouple-pages base: the unpatched overlay
GiB = 1024**3
STUBBED = ('vllm', 'torch', 'typing_extensions')
FA_PAGE_PER_TOKEN = 4096  # 2 * 4 KV heads * 256 * bf16
MAMBA_PAGE = 3_207_168    # 61,440 conv (bf16) + 3,145,728 ssm (fp32)
LEGACY_PAGE = 3_211_264   # 784 * 4096: the padded shared page


class DType(str):
    pass


BF16, FP32, INT8 = DType('bfloat16'), DType('float32'), DType('int8')
SIZES = {'bfloat16': 2, 'float32': 4, 'int8': 1}
TORCH = NS(dtype=DType, float32=FP32, bfloat16=BF16, int8=INT8)


class Log:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class MambaEnum(enum.Enum):
    MAMBA2 = 'mamba2'
    GDN_ATTN = 'gdn_attn'


class Status(enum.Enum):
    WAITING = 'WAITING'
    RUNNING = 'RUNNING'
    PREEMPTED = 'PREEMPTED'


class Record:
    def __init__(self, *args, **kwargs):
        self.args, self.__dict__ = args, {**self.__dict__, **kwargs}


def cdiv(a, b):
    return -(-a // b)


def base_source(rel):
    try:
        return subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(HERE), 'show',
                               f'{BASE}:source-overlay/{rel}'], capture_output=True,
                              check=True, timeout=60).stdout.decode('utf-8')
    except (OSError, subprocess.SubprocessError):
        return None


def load(text, name, stubs, filename):
    """Execute a module source with vLLM/torch imports taken from `stubs`."""
    tree = ast.parse(text)
    body = []
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.module
                and node.module.split('.')[0] in STUBBED):
            for alias in node.names:
                body += ast.parse(f'{alias.asname or alias.name} = '
                                  f'__stub__({node.module!r}, {alias.name!r})').body
        elif (isinstance(node, ast.Import)
              and any(a.name.split('.')[0] in STUBBED for a in node.names)):
            for alias in node.names:
                body += ast.parse(f'{alias.asname or alias.name} = '
                                  f'__stub__({alias.name!r}, None)').body
        else:
            body.append(node)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__stub__ = lambda mod, attr: stubs[mod] if attr is None else getattr(stubs[mod], attr)
    sys.modules[name] = module  # dataclasses resolve string annotations here
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, 'exec'), module.__dict__)
    return module


def make_registry():
    class Registry:
        managers = {}

        @classmethod
        def get_manager_class(cls, spec):
            return cls.managers.get(type(spec))

        @classmethod
        def get_uniform_type_base_spec(cls, spec):
            return cls.full if isinstance(spec, cls.full) else type(spec)

        @classmethod
        def check_kv_cache_spec_registry(cls, specs):
            pass
    return Registry


_WORLDS = {}


def world(version='overlay'):
    """The vLLM KV planner/pool modules of this branch, or of the base commit."""
    if version in _WORLDS:
        return _WORLDS[version]
    if version == 'overlay':
        read = lambda rel: (SOURCE/rel).read_text('utf-8')
    else:
        if base_source('vllm/v1/kv_cache_interface.py') is None:
            _WORLDS[version] = None
            return None
        read = base_source
    registry = make_registry()
    stubs = {
        'torch': TORCH,
        'typing_extensions': NS(Self=typing.Self, override=lambda f: f),
        'vllm': NS(envs=NS(VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=False,
                           VLLM_PREFIX_CACHE_RETENTION_INTERVAL=None)),
        'vllm.logger': NS(init_logger=lambda name: Log()),
        'vllm.config': NS(VllmConfig=object),
        'vllm.utils.math_utils': NS(cdiv=cdiv, round_up=lambda x, y: cdiv(x, y) * y),
        'vllm.utils.torch_utils': NS(get_dtype_size=lambda d: SIZES[d],
                                     nvfp4_kv_cache_full_dim=lambda h: h),
        'vllm.utils.hashing': NS(sha256_cbor=len, xxhash_cbor=hash),
        'vllm.utils.mem_utils': NS(format_gib=lambda b: round(b / GiB, 2)),
        'vllm.v1.attention.backends.registry': NS(MambaAttentionBackendEnum=MambaEnum),
        'vllm.v1.kv_cache_spec_registry': NS(KVCacheSpecRegistry=registry),
        'vllm.v1.request': NS(Request=object, RequestStatus=Status),
        'vllm.v1.utils': NS(tensor_data=None),
        'vllm.distributed.kv_events': NS(MEDIUM_GPU='GPU', AllBlocksCleared=Record,
                                         BlockRemoved=Record, BlockStored=Record,
                                         KVCacheEvent=object),
        'vllm.v1.core.kv_cache_metrics': NS(KVCacheMetricsCollector=object),
        'vllm.v1.metrics.stats': NS(PrefixCacheStats=Record),
    }
    w = NS(version=version, read=read, stubs=stubs)

    def add(rel, key, attr):
        module = load(read(rel), f'_wp1_{version}_{attr}', stubs, f'{version}:{rel}')
        stubs[key] = module
        setattr(w, attr, module)

    add('vllm/v1/kv_cache_interface.py', 'vllm.v1.kv_cache_interface', 'iface')
    registry.full = w.iface.FullAttentionSpec
    add('vllm/v1/core/kv_cache_utils.py', 'vllm.v1.core.kv_cache_utils', 'utils')
    add('vllm/v1/core/block_pool.py', 'vllm.v1.core.block_pool', 'pool')
    add('vllm/v1/core/single_type_kv_cache_manager.py',
        'vllm.v1.core.single_type_kv_cache_manager', 'single')
    registry.managers.update({w.iface.FullAttentionSpec: w.single.FullAttentionManager,
                              w.iface.MambaSpec: w.single.MambaManager})
    add('vllm/v1/core/kv_cache_coordinator.py', 'vllm.v1.core.kv_cache_coordinator', 'coord')
    add('vllm/v1/core/kv_cache_manager.py', 'vllm.v1.core.kv_cache_manager', 'manager')
    _WORLDS[version] = w
    return w


def outcome(fn, *args):
    """The result of fn, or the error it raises (compared like a result)."""
    try:
        return fn(*args)
    except (ValueError, AssertionError, NotImplementedError) as exc:
        return ('raised', type(exc).__name__, str(exc))


def base_or_skip(test):
    base = world('base')
    if base is None:
        test.skipTest('base commit not available (Git archive)')
    return base


# ---------------------------------------------------------------- fixtures

def fa_spec(w, P):
    return w.iface.FullAttentionSpec(block_size=P, num_kv_heads=4, head_size=256, dtype=BF16)


def mamba_spec(w, M, padded):
    return w.iface.MambaSpec(block_size=M, shapes=((3, 10240), (48, 128, 128)),
                             dtypes=(BF16, FP32), page_size_padded=padded,
                             mamba_type=MambaEnum.GDN_ATTN, mamba_cache_mode='align')


def fa_name(i):
    return f'model.layers.{i}.self_attn.attn'


def gdn_name(i):
    return f'model.layers.{i}.linear_attn'


def qwen_specs(w, layout, fa_first=False):
    """Qwen3.6-27B: 64 layers, every 4th full attention (16 FA, 48 GDN)."""
    if layout.decoupled:
        fa, gdn = fa_spec(w, layout.page_tokens), mamba_spec(w, layout.mamba_tokens, None)
    else:
        fa, gdn = fa_spec(w, 784), mamba_spec(w, 784, LEGACY_PAGE)
    order = range(64)
    if fa_first:
        order = [i for i in order if i % 4 == 3] + [i for i in order if i % 4 != 3]
    return {(fa_name(i) if i % 4 == 3 else gdn_name(i)): (fa if i % 4 == 3 else gdn)
            for i in order}


def vconfig(layout, override=None, prefix=True, connector=True, hash_block_size=None,
            extra=None, max_len=65536):
    additional = dict(extra or {})
    additional.update(I0.to_additional_config(layout) or {})
    return NS(
        additional_config=additional,
        cache_config=NS(num_gpu_blocks_override=override, enable_prefix_caching=prefix,
                        hash_block_size=hash_block_size, block_size=layout.page_tokens,
                        mamba_cache_mode='align'),
        model_config=NS(max_model_len=max_len, original_max_model_len=max_len),
        parallel_config=NS(decode_context_parallel_size=1, prefill_context_parallel_size=1,
                           world_size=1),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=(NS(kv_connector_extra_config={'cpu_bytes_to_use': 32 * GiB})
                            if connector else None),
        speculative_config=None,
        kv_events_config=None)


def plan(w, layout, budget, **kw):
    vc = vconfig(layout, **kw)
    (config,) = w.utils.get_kv_cache_configs(vc, [qwen_specs(w, layout)], [budget])
    return vc, config


def spec_key(spec):
    return (type(spec).__name__, tuple(sorted(
        (k, v.name if isinstance(v, enum.Enum) else v) for k, v in vars(spec).items())))


def normal(config):
    """World-independent view of a KVCacheConfig."""
    return (config.num_blocks,
            [(t.size, list(t.shared_by), t.offset, t.block_stride)
             for t in config.kv_cache_tensors],
            [(list(g.layer_names), spec_key(g.kv_cache_spec), g.is_eagle_group)
             for g in config.kv_cache_groups],
            getattr(config, 'group_pool', None), getattr(config, 'num_blocks_by_pool', None))


def fa_pool_blocks(layout, budget, sessions):
    mamba = I0.mamba_pool_blocks(layout, 3) * MAMBA_PAGE * 16
    return (budget - mamba) // (layout.page_tokens * FA_PAGE_PER_TOKEN * 16)


# ---------------------------------------------------------------- planner

class LegacyPlannerTests(unittest.TestCase):
    def test_golden_qwen36_8gib(self):
        w = world()
        layout = I0.legacy()
        vc, config = plan(w, layout, 8 * GiB)
        self.assertEqual(config.num_blocks, 167)
        self.assertEqual(len(config.kv_cache_tensors), 16)
        self.assertEqual({t.size for t in config.kv_cache_tensors}, {167 * LEGACY_PAGE})
        self.assertIsNone(config.group_pool)
        self.assertIsNone(config.num_blocks_by_pool)
        groups = config.kv_cache_groups
        self.assertEqual(len(groups), 4)
        kinds = [type(g.kv_cache_spec).__name__ for g in groups]
        self.assertEqual(sorted(kinds), ['FullAttentionSpec'] + ['MambaSpec'] * 3)
        self.assertEqual({g.kv_cache_spec.block_size for g in groups}, {784})
        self.assertEqual({g.kv_cache_spec.page_size_bytes for g in groups}, {LEGACY_PAGE})
        gdn = [gdn_name(i) for i in range(64) if i % 4 != 3]
        mamba = [g.layer_names for g in groups if isinstance(g.kv_cache_spec, w.iface.MambaSpec)]
        self.assertEqual(mamba, [gdn[i::3] for i in range(3)])
        for i, tensor in enumerate(config.kv_cache_tensors):
            self.assertEqual(tensor.shared_by, [g.layer_names[i] for g in groups])
        self.assertEqual(w.utils.resolve_kv_cache_block_sizes(config, vc), (784, 784))
        # A legacy object compares equal to one built without the new fields.
        rebuilt = w.iface.KVCacheConfig(num_blocks=config.num_blocks,
            kv_cache_tensors=config.kv_cache_tensors, kv_cache_groups=config.kv_cache_groups)
        self.assertEqual(config, rebuilt)
        self.assertEqual(config.tensor_pools(), [0] * 16)

    def test_unrelated_additional_config_stays_legacy(self):
        w = world()
        _, a = plan(w, I0.legacy(), 8 * GiB)
        _, b = plan(w, I0.legacy(), 8 * GiB, extra={'other': 1})
        self.assertEqual(normal(a), normal(b))

    def test_legacy_matches_base_commit_field_by_field(self):
        base = base_or_skip(self)
        w = world()
        layout = I0.legacy()
        for budget in (8 * GiB, 16 * GiB, 24 * GiB, 32 * GiB):
            for override in (None, 1, 100, 200):
                for prefix, connector in ((True, True), (False, False), (True, False)):
                    for fa_first in (False, True):
                        with self.subTest(budget=budget, override=override, prefix=prefix,
                                          connector=connector, fa_first=fa_first):
                            kw = dict(override=override, prefix=prefix, connector=connector)
                            out = []
                            for v in (w, base):
                                vc = vconfig(layout, **kw)

                                def run():
                                    (config,) = v.utils.get_kv_cache_configs(
                                        vc, [qwen_specs(v, layout, fa_first)], [budget])
                                    groups = config.kv_cache_groups
                                    return (
                                        normal(config)[:3],
                                        v.utils.resolve_kv_cache_block_sizes(config, vc),
                                        v.utils.get_kv_cache_capacity(vc, config),
                                        v.utils._max_memory_usage_bytes_from_groups(vc, groups),
                                        normal(v.utils.get_kv_cache_config_from_groups(
                                            vc, groups, budget))[:3])
                                out.append(outcome(run))
                            self.assertEqual(out[0], out[1])
                            self.assertEqual(out[0][0] == 'raised', override == 1)
        self.assertEqual(normal(plan(w, layout, 8 * GiB)[1])[3:], (None, None))


class DecoupledPlannerTests(unittest.TestCase):
    LAYOUTS = [(16, 784), (32, 800), (64, 832), (784, 784)]

    def test_groups_identical_to_legacy(self):
        w = world()
        _, legacy = plan(w, I0.legacy(), 8 * GiB)
        for P, M in self.LAYOUTS:
            with self.subTest(P=P):
                layout = I0.for_server(P, 1, True)
                _, config = plan(w, layout, 8 * GiB)
                self.assertEqual([g.layer_names for g in config.kv_cache_groups],
                                 [g.layer_names for g in legacy.kv_cache_groups])
                for group in config.kv_cache_groups:
                    spec = group.kv_cache_spec
                    if isinstance(spec, w.iface.MambaSpec):
                        self.assertEqual((spec.block_size, spec.page_size_padded,
                                          spec.page_size_bytes), (M, None, MAMBA_PAGE))
                    else:
                        self.assertEqual((spec.block_size, spec.page_size_bytes),
                                         (P, P * FA_PAGE_PER_TOKEN))
                self.assertEqual(config.group_pool, tuple(
                    1 if isinstance(g.kv_cache_spec, w.iface.MambaSpec) else 0
                    for g in config.kv_cache_groups))

    def test_pool_sizes_and_tensors(self):
        w = world()
        for P, M in self.LAYOUTS:
            for sessions in (1, 4, 8):
                for budget in (8 * GiB, 24 * GiB, 32 * GiB):
                    with self.subTest(P=P, sessions=sessions, budget=budget):
                        layout = I0.for_server(P, sessions, True)
                        _, config = plan(w, layout, budget)
                        n_m = 1 + 3 * (2 * sessions + 1)
                        n_fa = fa_pool_blocks(layout, budget, sessions)
                        self.assertEqual(config.num_blocks_by_pool, (n_fa, n_m))
                        self.assertEqual(config.num_blocks, n_fa)
                        groups = config.kv_cache_groups
                        fa = [g for g in groups if not isinstance(g.kv_cache_spec, w.iface.MambaSpec)]
                        gdn = [g for g in groups if isinstance(g.kv_cache_spec, w.iface.MambaSpec)]
                        tensors = config.kv_cache_tensors
                        self.assertEqual(len(tensors), 32)
                        for i, tensor in enumerate(tensors[:16]):
                            self.assertEqual(tensor.shared_by, [fa[0].layer_names[i]])
                            self.assertEqual(tensor.size, n_fa * P * FA_PAGE_PER_TOKEN)
                        for i, tensor in enumerate(tensors[16:]):
                            self.assertEqual(tensor.shared_by, [g.layer_names[i] for g in gdn])
                            self.assertEqual(tensor.size, n_m * MAMBA_PAGE)
                        self.assertEqual(config.tensor_pools(), [0] * 16 + [1] * 16)
                        self.assertLessEqual(sum(t.size for t in tensors), budget)

    def test_design_capacity_table(self):
        # DESIGN.md section 4.4 (P = 16): Mamba blocks and FA tokens.
        w = world()
        for budget, sessions, n_m, tokens in ((8, 1, 10, 123_232), (8, 8, 52, 90_352),
                                              (24, 4, 28, 371_280), (32, 8, 52, 483_568)):
            with self.subTest(budget=budget, sessions=sessions):
                _, config = plan(w, I0.for_server(16, sessions), budget * GiB)
                self.assertEqual(config.num_blocks_by_pool[1], n_m)
                self.assertEqual(config.num_blocks_by_pool[0] * 16, tokens)

    def test_profiling_and_override(self):
        w = world()
        layout = I0.for_server(16, 1)
        vc = vconfig(layout, override=4)
        groups = w.utils.get_kv_cache_groups(vc, qwen_specs(w, layout))
        config = w.utils.get_kv_cache_config_from_groups(vc, groups, available_memory=0)
        self.assertEqual(config.num_blocks_by_pool, (4, 10))
        self.assertEqual([t.size for t in config.kv_cache_tensors],
                         [4 * 16 * FA_PAGE_PER_TOKEN] * 16 + [10 * MAMBA_PAGE] * 16)
        _, config = plan(w, layout, 8 * GiB, override=5000)
        self.assertEqual(config.num_blocks_by_pool, (5000, 10))
        self.assertEqual(config.kv_cache_tensors[0].size, 5000 * 16 * FA_PAGE_PER_TOKEN)
        self.assertEqual(config.kv_cache_tensors[-1].size, 10 * MAMBA_PAGE)

    def test_resolve_block_sizes(self):
        w = world()
        for (P, M), expected in zip(self.LAYOUTS, ((784, 16), (800, 32), (832, 64), (784, 784))):
            with self.subTest(P=P):
                vc, config = plan(w, I0.for_server(P, 1, True), 8 * GiB)
                self.assertEqual(w.utils.resolve_kv_cache_block_sizes(config, vc), expected)
                vc.cache_config.hash_block_size = P
                self.assertEqual(w.utils.resolve_kv_cache_block_sizes(config, vc), expected)
        vc, config = plan(w, I0.for_server(16, 1), 8 * GiB)
        vc.cache_config.hash_block_size = 48
        with self.assertRaises(ValueError):
            w.utils.resolve_kv_cache_block_sizes(config, vc)
        vc, config = plan(w, I0.for_server(16, 1), 8 * GiB, prefix=False, connector=False)
        self.assertEqual(w.utils.resolve_kv_cache_block_sizes(config, vc), (784, 784))

    def test_memory_check_counts_both_pools(self):
        w = world()
        layout = I0.for_server(16, 1)
        # One 64K request needs 4 GiB of FA blocks plus the 0.51 GB Mamba pool.
        with self.assertRaisesRegex(ValueError, 'max seq len'):
            plan(w, layout, 4 * GiB)
        _, config = plan(w, layout, 5 * GiB)
        self.assertEqual(config.num_blocks_by_pool[1], 10)
        vc = vconfig(layout)
        groups = w.utils.get_kv_cache_groups(vc, qwen_specs(w, layout))
        self.assertEqual(w.utils._max_memory_usage_bytes_from_groups(vc, groups),
                         4096 * 16 * 16 * FA_PAGE_PER_TOKEN + 10 * MAMBA_PAGE * 16)

    def test_capacity_is_the_smaller_pool(self):
        w = world()
        vc, config = plan(w, I0.for_server(16, 1), 8 * GiB)
        fa = config.num_blocks_by_pool[0] / 4096       # 64K tokens / 16
        mamba = config.num_blocks_by_pool[1] / 6       # 2 align blocks * 3 groups
        tokens, concurrency = w.utils.get_kv_cache_capacity(vc, config)
        self.assertAlmostEqual(concurrency, min(fa, mamba))
        self.assertEqual(tokens, int(min(fa, mamba) * 65536))

    def test_validation(self):
        w = world()
        layout = I0.for_server(16, 1)
        vc = vconfig(layout)
        good = qwen_specs(w, layout)
        groups = w.utils.get_kv_cache_groups(vc, good)
        cases = {
            'padded Mamba page': {n: (s if isinstance(s, w.iface.FullAttentionSpec)
                                      else mamba_spec(w, 784, LEGACY_PAGE)) for n, s in good.items()},
            'FA block != P': {n: (fa_spec(w, 32) if isinstance(s, w.iface.FullAttentionSpec) else s)
                              for n, s in good.items()},
            'Mamba block != M': {n: (s if isinstance(s, w.iface.FullAttentionSpec)
                                     else mamba_spec(w, 800, None)) for n, s in good.items()},
        }
        for name, specs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                w.utils.get_kv_cache_config_from_groups(
                    vc, w.utils.get_kv_cache_groups(vc, specs), 8 * GiB)
        only_fa = [g for g in groups if not isinstance(g.kv_cache_spec, w.iface.MambaSpec)]
        with self.assertRaisesRegex(ValueError, 'attention and Mamba'):
            w.utils.get_kv_cache_config_from_groups(vc, only_fa, 8 * GiB)

    def test_decoupled_layout_helper(self):
        w = world()
        self.assertIsNone(w.iface.decoupled_kv_layout(NS(additional_config={})))
        self.assertIsNone(w.iface.decoupled_kv_layout(NS()))
        layout = I0.for_server(16, 2)
        self.assertEqual(w.iface.decoupled_kv_layout(vconfig(layout)), layout)
        with self.assertRaises(ValueError):
            w.iface.decoupled_kv_layout(NS(additional_config={I0.LAYOUT_KEY: {}}))


# ---------------------------------------------------------------- pools

def small_config(w, layout, n_fa, n_m=None, fa_first=True, layers=2):
    """A Qwen-shaped config with few layers: 1 FA group, 3 Mamba groups."""
    fa = fa_spec(w, layout.page_tokens if layout.decoupled else 784)
    gdn = mamba_spec(w, layout.mamba_tokens, None if layout.decoupled else LEGACY_PAGE)
    groups = [w.iface.KVCacheGroupSpec([f'fa{i}' for i in range(layers)], fa)]
    groups += [w.iface.KVCacheGroupSpec([f'm{g}.{i}' for i in range(layers)], gdn)
               for g in range(3)]
    if not fa_first:
        groups = groups[1:] + groups[:1]
    if not layout.decoupled:
        return w.iface.KVCacheConfig(num_blocks=n_fa, kv_cache_tensors=[], kv_cache_groups=groups)
    pools = tuple(1 if isinstance(g.kv_cache_spec, w.iface.MambaSpec) else 0 for g in groups)
    return w.iface.KVCacheConfig(num_blocks=n_fa, kv_cache_tensors=[], kv_cache_groups=groups,
                                 group_pool=pools, num_blocks_by_pool=(n_fa, n_m))


def kv_manager(w, config, caching=True, watermark=0.0, hash_block_size=16,
               scheduler_block_size=784):
    return w.manager.KVCacheManager(
        kv_cache_config=config, max_model_len=65536, scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size, max_num_batched_tokens=16384, enable_caching=caching,
        watermark=watermark)


def request(rid, tokens, computed=0, status=Status.WAITING):
    return NS(request_id=rid, num_tokens=tokens, num_computed_tokens=computed, status=status,
              block_hashes=[], num_preemptions=0)


def alloc(manager, req, new):
    blocks = manager.allocate_slots(req, new, skip_cache=True, has_scheduled_reqs=True)
    if blocks is not None:
        req.num_computed_tokens += new
    return blocks


class PoolTests(unittest.TestCase):
    def test_legacy_single_pool_interface(self):
        w = world()
        manager = kv_manager(w, small_config(w, I0.legacy(), 30), hash_block_size=784)
        self.assertEqual(manager.block_pools, (manager.block_pool,))
        self.assertEqual(manager.coordinator.group_pool, (0, 0, 0, 0))
        for g in range(4):
            self.assertIs(manager.pool_of_group(g), manager.block_pool)
            self.assertIs(manager.coordinator.single_type_managers[g].block_pool,
                          manager.block_pool)
        self.assertEqual(manager.get_num_free_blocks_by_pool(), (29,))
        self.assertEqual(manager.usage_by_pool, (manager.usage,))

    def test_legacy_allocation_matches_base_commit(self):
        base = base_or_skip(self)
        traces = []
        for v in (world(), base):
            manager = kv_manager(v, small_config(v, I0.legacy(), 12), hash_block_size=784,
                                 watermark=0.1)
            trace, live = [], []
            for step, (tokens, new) in enumerate(((1700, 1568), (900, 784), (2400, 784),
                                                  (800, 784), (5000, 2352))):
                req = request(f'r{step}', tokens)
                blocks = alloc(manager, req, new)
                trace.append((None if blocks is None else blocks.get_block_ids(),
                              manager.block_pool.get_num_free_blocks()))
                if blocks is not None:
                    live.append(req)
                    blocks = alloc(manager, req, tokens - new)
                    trace.append((None if blocks is None else blocks.get_block_ids(),
                                  manager.block_pool.get_num_free_blocks()))
            for req in live:
                manager.free(req)
                trace.append(manager.block_pool.get_num_free_blocks())
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_two_pools_allocate_and_free(self):
        w = world()
        for fa_first in (True, False):
            with self.subTest(fa_first=fa_first):
                layout = I0.for_server(16, 1)
                manager = kv_manager(w, small_config(w, layout, 40, 10, fa_first))
                fa_gid = 0 if fa_first else 3
                self.assertEqual(len(manager.block_pools), 2)
                self.assertEqual([p.num_gpu_blocks for p in manager.block_pools], [40, 10])
                self.assertEqual(manager.get_num_free_blocks_by_pool(), (39, 9))
                for g, sub in enumerate(manager.coordinator.single_type_managers):
                    pool = manager.block_pools[0 if g == fa_gid else 1]
                    self.assertIs(sub.block_pool, pool)
                    self.assertIs(manager.pool_of_group(g), pool)
                self.assertIs(manager.block_pool, manager.block_pools[0])
                req = request('a', 300)
                blocks = alloc(manager, req, 300)
                ids = blocks.get_block_ids()
                # FA: ceil(300 / 16) = 19 blocks; Mamba: one align block per group.
                self.assertEqual(len(ids[fa_gid]), 19)
                mamba_ids = [ids[g] for g in range(4) if g != fa_gid]
                self.assertEqual(sorted(i for row in mamba_ids for i in row), [1, 2, 3])
                # Ids are per pool: the same numbers appear in both pools.
                self.assertTrue({1, 2, 3} <= set(ids[fa_gid]))
                self.assertEqual(manager.get_num_free_blocks_by_pool(), (20, 6))
                for usage, expected in zip(manager.usage_by_pool, (19 / 39, 3 / 9)):
                    self.assertAlmostEqual(usage, expected)
                self.assertAlmostEqual(manager.usage, 19 / 39)
                manager.free(req)
                self.assertEqual(manager.get_num_free_blocks_by_pool(), (39, 9))

    def test_each_pool_is_checked_on_its_own(self):
        w = world()
        layout = I0.for_server(16, 1)
        # Mamba pool: 9 usable blocks = 3 requests; the FA pool has room for more.
        manager = kv_manager(w, small_config(w, layout, 200, 10))
        reqs = [request(f'r{i}', 100) for i in range(4)]
        self.assertTrue(all(alloc(manager, r, 100) is not None for r in reqs[:3]))
        self.assertEqual(manager.get_num_free_blocks_by_pool(), (199 - 21, 0))
        self.assertIsNone(alloc(manager, reqs[3], 100))
        self.assertEqual(manager.get_num_free_blocks_by_pool(), (199 - 21, 0))
        manager.free(reqs[0])
        self.assertIsNotNone(alloc(manager, reqs[3], 100))
        # FA pool: 160 tokens need 10 FA blocks; only 9 usable.
        manager = kv_manager(w, small_config(w, layout, 10, 100))
        self.assertIsNone(alloc(manager, request('x', 160), 160))
        self.assertIsNotNone(alloc(manager, request('y', 144), 144))
        # The full-sequence admission gate is per pool too.
        manager = kv_manager(w, small_config(w, layout, 200, 10))
        req = request('z', 1600)
        self.assertIsNotNone(manager.allocate_slots(req, 784, full_sequence_must_fit=True,
                                                    skip_cache=True))
        manager = kv_manager(w, small_config(w, layout, 60, 10))
        self.assertIsNone(manager.allocate_slots(request('z', 1600), 784,
                                                 full_sequence_must_fit=True, skip_cache=True))

    def test_watermark_and_reserved_blocks_apply_to_pool_zero(self):
        w = world()
        layout = I0.for_server(16, 1)
        # watermark 0.5 of 20 FA blocks = 10; Mamba (9 free) is not watermarked.
        manager = kv_manager(w, small_config(w, layout, 20, 10), watermark=0.5)
        self.assertEqual(manager.watermark_blocks, 10)
        self.assertIsNone(alloc(manager, request('a', 160), 160))      # 10 + 10 > 19
        self.assertIsNotNone(alloc(manager, request('b', 144), 144))   # 9 + 10 <= 19
        manager = kv_manager(w, small_config(w, layout, 20, 10))
        req = request('c', 144)
        self.assertIsNone(manager.allocate_slots(req, 144, reserved_blocks=11, skip_cache=True))
        self.assertIsNotNone(manager.allocate_slots(req, 144, reserved_blocks=10,
                                                    skip_cache=True))

    def test_blocks_to_allocate_by_pool_sums_to_scalar(self):
        w = world()
        manager = kv_manager(w, small_config(w, I0.for_server(16, 1), 400, 10))
        coordinator = manager.coordinator
        empty = manager.empty_kv_cache_blocks.blocks
        for tokens in (1, 16, 17, 784, 785, 3000):
            kw = dict(request_id='q', num_tokens=tokens, new_computed_blocks=empty,
                      num_encoder_tokens=0, total_computed_tokens=0,
                      num_tokens_main_model=tokens)
            by_pool = coordinator.get_num_blocks_to_allocate_by_pool(**kw)
            self.assertEqual(by_pool, (cdiv(tokens, 16), 3))
            self.assertEqual(sum(by_pool), coordinator.get_num_blocks_to_allocate(**kw))

    def test_mamba_align_sentinel_uses_its_own_pool(self):
        w = world()
        layout = I0.for_server(16, 1)
        manager = kv_manager(w, small_config(w, layout, 400, 10))
        coordinator = manager.coordinator
        mamba = coordinator.single_type_managers[1]
        self.assertIs(mamba.block_pool, manager.block_pools[1])
        hit = NS(block_hash=b'h', ref_cnt=1, is_null=False)
        mamba.cached_blocks_this_step.add(b'h')
        self.assertEqual(mamba.get_num_blocks_to_allocate('q', 784, [hit], 0, 784), 11)
        computed = ((), (hit,), (), ())
        by_pool = coordinator.get_num_blocks_to_allocate_by_pool(
            request_id='q', num_tokens=1568, new_computed_blocks=computed,
            num_encoder_tokens=0, total_computed_tokens=784, num_tokens_main_model=1568)
        self.assertEqual(by_pool[1], 11 + 1 + 1)
        self.assertFalse(manager._pools_can_allocate(by_pool, 0))

    def test_cache_hit_lookup_uses_the_group_pool(self):
        w = world()
        manager = kv_manager(w, small_config(w, I0.for_server(16, 1), 400, 10, fa_first=False))
        coordinator = manager.coordinator
        seen = []

        def recorder(cls):
            def find(*args, block_pool, kv_cache_group_ids, **kwargs):
                seen.append((tuple(kv_cache_group_ids), block_pool))
                return tuple([] for _ in kv_cache_group_ids)
            return classmethod(lambda c, *a, **k: find(*a, **k))
        with patch.object(w.single.MambaManager, 'find_longest_cache_hit',
                          recorder(w.single.MambaManager)), \
                patch.object(w.single.FullAttentionManager, 'find_longest_cache_hit',
                             recorder(w.single.FullAttentionManager)):
            coordinator.find_longest_cache_hit([], 1000)
            coordinator.find_longest_cache_hit_per_group([], 1000)
        pools = manager.block_pools
        self.assertEqual(seen, [((3,), pools[0]), ((0, 1, 2), pools[1])] * 2)

    def test_reset_events_and_eviction(self):
        w = world()
        manager = kv_manager(w, small_config(w, I0.for_server(16, 1), 40, 10))
        self.assertTrue(manager.reset_prefix_cache())
        req = request('a', 100)
        alloc(manager, req, 100)
        self.assertFalse(manager.reset_prefix_cache())
        manager.free(req)
        self.assertTrue(manager.reset_prefix_cache())
        fake = NS(block_pool=NS(take_events=lambda: ['e0']),
                  block_pools=(None, NS(take_events=lambda: ['e1'])),
                  kv_cache_event_metadata=())
        self.assertEqual(w.manager.KVCacheManager.take_events(fake), ['e0', 'e1'])
        evicted = []
        fake = NS(block_pool=NS(evict_blocks=evicted.append))
        w.manager.KVCacheManager.evict_blocks(fake, {1})
        self.assertEqual(evicted, [{1}])

    def test_coordinator_refuses_inconsistent_pools(self):
        w = world()
        layout = I0.for_server(16, 1)
        for bad in (dict(group_pool=(0, 1, 1)), dict(num_blocks_by_pool=(41, 10)),
                    dict(group_pool=(0, 2, 2, 2))):
            config = small_config(w, layout, 40, 10)
            for key, value in bad.items():
                setattr(config, key, value)
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                kv_manager(w, config)


# ---------------------------------------------------------------- scheduler

def scheduler_node(name):
    text = (SOURCE/'vllm/v1/core/sched/scheduler.py').read_text('utf-8')
    cls = next(n for n in ast.parse(text).body
               if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def scheduler_function(name, namespace):
    node = scheduler_node(name)
    exec('from __future__ import annotations\n'
         + ast.unparse(ast.Module(body=[node], type_ignores=[])), namespace)
    return namespace[name]


class SchedulerTests(unittest.TestCase):
    def test_mamba_split_uses_M_not_the_attention_block(self):
        split = scheduler_function('_mamba_block_aligned_split', {})
        legacy = NS(cache_config=NS(block_size=784), use_eagle=False,
                    _remote_ra_mamba_align_tokens=None)
        decoupled = NS(cache_config=NS(block_size=16), use_eagle=False,
                       _remote_ra_mamba_align_tokens=784)
        broken = NS(cache_config=NS(block_size=16), use_eagle=False,
                    _remote_ra_mamba_align_tokens=None)
        differs = False
        for prompt in (100, 783, 784, 1700, 6500, 16000):
            for computed in (0, 784, 1568):
                for new in (1, 16, 500, 784, 1000, 2000, 16384):
                    for common in (0, 800, 2000):
                        if computed >= prompt:
                            continue
                        req = NS(num_computed_tokens=computed, num_prompt_tokens=prompt,
                                 num_tokens=prompt)
                        args = (req, min(new, prompt - computed), 0, 0, common)
                        self.assertEqual(split(decoupled, *args), split(legacy, *args))
                        differs |= split(broken, *args) != split(legacy, *args)
        self.assertTrue(differs)  # upstream would split at P = 16 instead

    def test_init_reads_M_and_refuses_deferred_frees(self):
        node = scheduler_node('__init__')
        block = next(n for n in ast.walk(node) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'kv_cache_config.group_pool is not None')
        code = compile(ast.Module(body=[block], type_ignores=[]), 'scheduler-init', 'exec')
        w = world()
        layout = I0.for_server(16, 1)
        config = small_config(w, layout, 40, 10)
        me = NS(defer_block_free=False, _remote_ra_mamba_align_tokens=None)
        exec(code, {'self': me, 'kv_cache_config': config, 'MambaSpec': w.iface.MambaSpec})
        self.assertEqual(me._remote_ra_mamba_align_tokens, 784)
        me = NS(defer_block_free=True, _remote_ra_mamba_align_tokens=None)
        with self.assertRaisesRegex(ValueError, 'deferred block frees'):
            exec(code, {'self': me, 'kv_cache_config': config, 'MambaSpec': w.iface.MambaSpec})
        legacy = NS(group_pool=None, kv_cache_groups=[])
        me = NS(defer_block_free=True, _remote_ra_mamba_align_tokens=None)
        exec(code, {'self': me, 'kv_cache_config': legacy, 'MambaSpec': w.iface.MambaSpec})
        self.assertIsNone(me._remote_ra_mamba_align_tokens)

    def test_selective_purge_evicts_in_each_group_pool(self):
        free = scheduler_function('_remote_ra_native_free_request', {'logger': Log()})
        evicted = {0: [], 1: []}
        pools = {0: NS(evict_blocks=evicted[0].append), 1: NS(evict_blocks=evicted[1].append)}
        block = lambda i, h=b'h', null=False: NS(block_id=i, block_hash=h, is_null=null)
        rows = ([block(1), block(2, None)], [block(1)], [block(0, None, True)], [block(3)])
        group_pool = (0, 1, 1, 1)
        for pool_split in (True, False):
            evicted[0].clear(); evicted[1].clear()
            req = type('Req', (), dict(request_id='r', client_index=0,
                                       is_finished=lambda self: True))()
            me = NS(_inflight_prefills=set(), _connector_finished=lambda r: (True, None),
                    encoder_cache_manager=NS(free=lambda r: None),
                    _selective_load_plans={'r': object()}, finished_req_ids=set(),
                    finished_req_ids_dict=None,
                    kv_cache_config=NS(group_pool=group_pool if pool_split else None),
                    kv_cache_manager=NS(get_blocks=lambda rid: NS(blocks=rows),
                                        pool_of_group=lambda g: pools[group_pool[g]],
                                        block_pool=pools[0]))
            free(me, req)
            if pool_split:
                self.assertEqual(evicted, {0: [{1}], 1: [{1}, {3}]})
            else:
                self.assertEqual(evicted, {0: [{1, 3}], 1: []})


# ---------------------------------------------------------------- platform

class Backend:
    sizes = [16, 32, 64]

    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return cls.sizes

    @classmethod
    def get_name(cls):
        return 'FLASH_ATTN'


def platform_namespace(w):
    shapes, dtypes = ((3, 10240), (48, 128, 128)), (BF16, FP32)
    model_cls = NS(get_mamba_state_shape_from_config=lambda vc: shapes,
                   get_mamba_state_dtype_from_config=lambda vc: dtypes)
    modules = {
        'vllm': types.ModuleType('vllm'),
        'vllm.config': types.ModuleType('vllm.config'),
        'vllm.config.vllm': NS(set_current_vllm_config=lambda vc: contextlib.nullcontext()),
        'vllm.model_executor': types.ModuleType('vllm.model_executor'),
        'vllm.model_executor.models': NS(ModelRegistry=NS(
            resolve_model_cls=lambda arch, model_config: (model_cls, arch))),
        'vllm.utils': types.ModuleType('vllm.utils'),
        'vllm.utils.math_utils': NS(cdiv=cdiv),
        'vllm.utils.torch_utils': NS(STR_DTYPE_TO_TORCH_DTYPE={}),
        'vllm.v1': types.ModuleType('vllm.v1'),
        'vllm.v1.attention': types.ModuleType('vllm.v1.attention'),
        'vllm.v1.attention.backend': NS(MultipleOf=type('MultipleOf', (), {})),
        'vllm.v1.kv_cache_interface': w.iface,
    }
    return modules


def platform_functions(version='overlay'):
    text = (SOURCE/'vllm/platforms/interface.py').read_text('utf-8') if version == 'overlay' \
        else base_source('vllm/platforms/interface.py')
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Platform')
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name == '_align_hybrid_block_size']
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_check_decoupled_block_sizes']
    namespace = {'logger': Log(), 'Any': typing.Any}
    exec('from __future__ import annotations\n'
         + ast.unparse(ast.Module(body=nodes, type_ignores=[])), namespace)
    return namespace['_align_hybrid_block_size'].__func__


def platform_config(layout, block_size=16, user_block=False, mamba_block=None,
                    mode='align', padded=None):
    return NS(
        additional_config=I0.to_additional_config(layout) or {},
        cache_config=NS(cache_dtype='auto', block_size=block_size,
                        user_specified_block_size=user_block, mamba_block_size=mamba_block,
                        user_specified_mamba_block_size=mamba_block is not None,
                        mamba_cache_mode=mode, mamba_page_size_padded=padded,
                        kv_cache_dtype_skip_layers=None),
        model_config=NS(dtype=BF16, use_mla=False, get_num_kv_heads=lambda pc: 4,
                        get_head_size=lambda: 256, architecture='Qwen3_5'),
        parallel_config=NS())


class PlatformTests(unittest.TestCase):
    def run_align(self, vc, version='overlay', backend=Backend):
        w = world() if version == 'overlay' else world('base')
        with patch.dict(sys.modules, platform_namespace(w)):
            platform_functions(version)(None, vc, backend)
        c = vc.cache_config
        return c.block_size, c.mamba_block_size, c.mamba_page_size_padded

    def test_legacy_raises_block_size_and_pads(self):
        vc = platform_config(I0.legacy())
        self.assertEqual(self.run_align(vc), (784, 784, LEGACY_PAGE))
        # Legacy keeps overriding a user --block-size, as upstream does.
        vc = platform_config(I0.legacy(), block_size=16, user_block=True)
        self.assertEqual(self.run_align(vc), (784, 784, LEGACY_PAGE))

    def test_legacy_matches_base_commit(self):
        base_or_skip(self)
        for kw in (dict(), dict(block_size=16, user_block=True), dict(block_size=1568),
                   dict(mamba_block=784)):
            with self.subTest(kw=kw):
                self.assertEqual(self.run_align(platform_config(I0.legacy(), **kw)),
                                 self.run_align(platform_config(I0.legacy(), **kw), 'base'))

    def test_decoupled_keeps_P_and_M_unpadded(self):
        for P in (16, 32, 64, 784):
            with self.subTest(P=P):
                layout = I0.for_server(P, 2, True)
                vc = platform_config(layout, P, True, layout.mamba_tokens, padded=LEGACY_PAGE)
                self.assertEqual(self.run_align(vc), (P, layout.mamba_tokens, None))

    def test_decoupled_requires_user_sizes_and_align(self):
        layout = I0.for_server(16, 1)
        for kw in (dict(block_size=16, user_block=False, mamba_block=784),
                   dict(block_size=32, user_block=True, mamba_block=784),
                   dict(block_size=16, user_block=True, mamba_block=None),
                   dict(block_size=16, user_block=True, mamba_block=800),
                   dict(block_size=16, user_block=True, mamba_block=784, mode='all')):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.run_align(platform_config(layout, **kw))

        class Coarse(Backend):
            sizes = [32]
        with self.assertRaisesRegex(ValueError, 'kernel block size'):
            self.run_align(platform_config(layout, 16, True, 784), backend=Coarse)


# ---------------------------------------------------------------- offloading

def cpu_spec_module(w):
    class Spec:
        def __init__(self, vllm_config, kv_cache_config):
            self.vllm_config, self.kv_cache_config = vllm_config, kv_cache_config
            self.extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
            self.block_size_factor = 1
            self.kv_events_config = NS(enable_kv_cache_events=False)
    stubs = dict(w.stubs)
    stubs.update({
        'vllm.platforms': NS(current_platform=None),
        'vllm.v1.kv_offload.base': NS(CanonicalKVCaches=object, GPULoadStoreSpec=object,
            LoadStoreSpec=object, OffloadingCounterMetadata=Record,
            OffloadingGaugeMetadata=Record, OffloadingManager=object,
            OffloadingMetricMetadata=object, OffloadingSpec=Spec),
        'vllm.v1.kv_offload.cpu.common': NS(CPULoadStoreSpec=object,
                                            CPUOffloadingMetrics=NS(CPU_CACHE_USAGE_PERC='u')),
        'vllm.v1.kv_offload.cpu.gpu_worker': NS(CpuGpuOffloadingHandlers=object),
        'vllm.v1.kv_offload.cpu.manager': NS(CPUOffloadingManager=object),
        'vllm.v1.kv_offload.worker.worker': NS(OffloadingHandler=object),
    })
    read = w.read
    return load(read('vllm/v1/kv_offload/cpu/spec.py'), f'_wp1_{w.version}_cpu_spec', stubs,
                f'{w.version}:spec.py')


class FakeTensor:
    def __init__(self, name, ptr=0, element_size=2):
        self.name, self.ptr, self._element_size, self.device = name, ptr, element_size, 'cuda'
        self.view_shape = None

    def element_size(self):
        return self._element_size

    def storage_offset(self):
        return 0

    def stride(self, dim=None):
        return (self.ptr, 1) if dim is None else self.ptr

    def data_ptr(self):
        return self.ptr

    def untyped_storage(self):
        return self

    def set_(self, storage, offset=0, shape=None, stride=None):
        out = FakeTensor('view:' + storage.name, storage.ptr)
        out.view_shape = shape
        return out

    def view(self, shape):
        self.view_shape = shape
        return self


class Untouchable:
    def __getattr__(self, name):
        raise AssertionError('Mamba KV cache must not be registered: ' + name)


def worker_register(version='overlay'):
    text = (SOURCE/'vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py').read_text(
        'utf-8') if version == 'overlay' else base_source(
        'vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py')
    cls = next(n for n in ast.parse(text).body
               if isinstance(n, ast.ClassDef) and n.name == 'OffloadingConnectorWorker')
    node = next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == 'register_kv_caches')
    return node


class OffloadingTests(unittest.TestCase):
    def test_cpu_pool_sized_from_the_attention_pool(self):
        w = world()
        module = cpu_spec_module(w)
        vc, config = plan(w, I0.legacy(), 8 * GiB)
        spec = module.CPUOffloadingSpec(vc, config)
        self.assertEqual(spec.num_blocks, 32 * GiB // (16 * LEGACY_PAGE))
        self.assertEqual(spec.num_blocks, 668)
        vc, config = plan(w, I0.for_server(16, 1), 8 * GiB)
        spec = module.CPUOffloadingSpec(vc, config)
        self.assertEqual(spec.cpu_page_size_per_worker, 16 * 16 * FA_PAGE_PER_TOKEN)
        self.assertEqual(spec.num_blocks, 32768)   # 32 GiB of 1 MiB FA blocks

    def test_cpu_pool_legacy_matches_base_commit(self):
        base = base_or_skip(self)
        out = []
        for v in (world(), base):
            vc, config = plan(v, I0.legacy(), 24 * GiB)
            spec = cpu_spec_module(v).CPUOffloadingSpec(vc, config)
            out.append((spec.num_blocks, spec.kv_bytes_per_offloaded_block,
                        spec.cpu_page_size_per_worker))
        self.assertEqual(out[0], out[1])

    def register(self, w, config, kv_caches, version='overlay'):
        captured = []
        namespace = {
            'torch': NS(tensor=lambda data, dtype, device: FakeTensor('new'), int8=INT8,
                        Tensor=FakeTensor),
            'UniformTypeKVCacheSpecs': w.iface.UniformTypeKVCacheSpecs,
            'AttentionSpec': w.iface.AttentionSpec, 'MambaSpec': w.iface.MambaSpec,
            'replace': __import__('dataclasses').replace,
            'defaultdict': __import__('collections').defaultdict,
            'CanonicalKVCacheTensor': Record, 'CanonicalKVCacheRef': Record,
            'CanonicalKVCaches': Record}
        node = worker_register(version)
        exec('from __future__ import annotations\n'
             + ast.unparse(ast.Module(body=[node], type_ignores=[])), namespace)
        me = NS(spec=NS(kv_cache_config=config), _register_handlers=captured.append)
        namespace['register_kv_caches'](me, kv_caches)
        (caches,) = captured
        return caches

    def test_worker_registers_only_the_attention_pool(self):
        w = world()
        _, config = plan(w, I0.for_server(16, 1), 8 * GiB)
        kv_caches = {}
        for t, tensor in enumerate(config.kv_cache_tensors):
            for name in tensor.shared_by:
                kv_caches[name] = (FakeTensor(name, 1000 + t) if '.self_attn.' in name
                                   else Untouchable())
        caches = self.register(w, config, kv_caches)
        self.assertEqual(len(caches.tensors), 16)
        self.assertEqual({t.page_size_bytes for t in caches.tensors}, {16 * FA_PAGE_PER_TOKEN})
        self.assertEqual({t.tensor.view_shape for t in caches.tensors},
                         {(config.num_blocks, 16 * FA_PAGE_PER_TOKEN)})
        refs = caches.group_data_refs
        self.assertEqual(len(refs), 4)
        for group, group_refs in zip(config.kv_cache_groups, refs):
            if isinstance(group.kv_cache_spec, w.iface.MambaSpec):
                self.assertEqual(group_refs, [])
            else:
                self.assertEqual(len(group_refs), 16)

    def test_worker_legacy_matches_base_commit(self):
        base = base_or_skip(self)
        out = []
        for version, v in (('overlay', world()), ('base', base)):
            _, config = plan(v, I0.legacy(), 8 * GiB)
            kv_caches = {}
            for t, tensor in enumerate(config.kv_cache_tensors):
                for name in tensor.shared_by:
                    kv_caches[name] = (FakeTensor(name, 1000 + t) if '.self_attn.' in name
                                       else [FakeTensor(name, 1000 + t)])
            caches = self.register(v, config, kv_caches, version)
            out.append(([(t.page_size_bytes, t.tensor.view_shape, t.tensor.ptr)
                         for t in caches.tensors],
                        [[(r.tensor_idx, r.page_size_bytes) for r in g]
                         for g in caches.group_data_refs]))
        self.assertEqual(out[0], out[1])
        self.assertEqual(len(out[0][0]), 16)

    def test_scheduler_refuses_jobs_in_decoupled_mode(self):
        text = (SOURCE/'vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py'
                ).read_text('utf-8')
        cls = next(n for n in ast.parse(text).body
                   if isinstance(n, ast.ClassDef) and n.name == 'OffloadingConnectorScheduler')
        meta_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'build_connector_meta')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        flag = next(n for n in init.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == 'self._remote_ra_decoupled')
        namespace = {'OffloadingConnectorMetadata': Record, 'KVConnectorMetadata': object,
                     'SchedulerOutput': object}
        exec(ast.unparse(ast.Module(body=[meta_fn], type_ignores=[])), namespace)
        for pools, decoupled in ((None, False), ((0, 1, 1, 1), True)):
            me = NS()
            exec(ast.unparse(flag), {'self': me,
                                     'spec': NS(kv_cache_config=NS(group_pool=pools))})
            self.assertIs(me._remote_ra_decoupled, decoupled)

        def scheduler(decoupled, load_jobs):
            return NS(_update_req_states=lambda out: None, manager=NS(on_schedule_end=lambda: None),
                      _req_status={}, _jobs={}, _block_id_to_pending_jobs={},
                      _current_batch_allocated_block_ids=set(),
                      _current_batch_jobs_to_flush=set(), _current_batch_load_jobs=load_jobs,
                      _build_store_jobs=lambda out: {}, _remote_ra_decoupled=decoupled)
        output = NS(preempted_req_ids=None, num_scheduled_tokens={})
        build = namespace['build_connector_meta']
        self.assertEqual(build(scheduler(True, {}), output).load_jobs, {})
        self.assertEqual(build(scheduler(False, {1: 'job'}), output).load_jobs, {1: 'job'})
        with self.assertRaisesRegex(RuntimeError, 'isolated'):
            build(scheduler(True, {1: 'job'}), output)


if __name__ == '__main__':
    unittest.main()
