"""Gemma 4 stage 1 (gemma-s1 WP2): the two-pool window layout in vLLM's KV planner, CPU only.

Pool 0 holds the global full-attention group in blocks of P tokens (one block per
runtime Page, 1.25 MiB at P = 16); pool 1 holds the five sliding-window groups in
blocks of P tokens, statically reserved: a null block plus per sliding group S window
slots of ceil(1024 / P) + 1 blocks and one forward's transient (design 3.1). The
overlay vLLM modules run from source with stub imports, as in test_kv_pools.
"""
import ast
import contextlib
import sys
import typing
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_kv_pools as KP
from test_kv_pools import BF16, GiB, I0, Log, SOURCE, cdiv, world
from remote_ra_live_v2 import model_profile as MP

GEOMETRY = MP.GEMMA4_GEOMETRY
MiB = 1024 ** 2


def gemma_world():
    """The overlay world of test_kv_pools with vLLM's SlidingWindowManager registered."""
    w = world()
    registry = w.stubs['vllm.v1.kv_cache_spec_registry'].KVCacheSpecRegistry
    registry.managers.setdefault(w.iface.SlidingWindowSpec, w.single.SlidingWindowManager)
    return w


def full_spec(w, P):
    return w.iface.FullAttentionSpec(block_size=P, num_kv_heads=4, head_size=512, dtype=BF16)


def sliding_spec(w, P, window=1024):
    return w.iface.SlidingWindowSpec(block_size=P, num_kv_heads=16, head_size=256, dtype=BF16,
                                     sliding_window=window)


def layer(i):
    return f'language_model.model.layers.{i}.self_attn.attn'


def gemma_specs(w, P, window=1024):
    """Gemma 4 31B IT: 60 layers, every sixth global (5, 11, ..., 59)."""
    return {layer(i): (full_spec(w, P) if i in GEOMETRY.global_layers else sliding_spec(w, P, window))
            for i in range(GEOMETRY.layers)}


def vconfig(layout, override=None, max_len=131072, batched=8192):
    vc = KP.vconfig(layout, override=override, max_len=max_len)
    vc.scheduler_config = NS(disable_hybrid_kv_cache_manager=False, max_num_batched_tokens=batched)
    return vc


def plan(w, layout, budget, **kw):
    vc = vconfig(layout, **kw)
    (config,) = w.utils.get_kv_cache_configs(vc, [gemma_specs(w, layout.page_tokens)], [budget])
    return vc, config


def window_layout(P=16, sessions=16, forward=8192):
    return I0.for_window_server(P, sessions, 1024, forward)


class WindowLayoutTests(unittest.TestCase):
    def test_slot_and_transient_blocks(self):
        # Design 3.1: 65 blocks per window slot and group at P = 16 (33 at P = 32);
        # one forward of N tokens adds ceil((1023 + N) / P) + 1 - 65 per group.
        layout = window_layout()
        self.assertEqual((layout.window_blocks, layout.transient_blocks), (65, 512))
        self.assertEqual(window_layout(forward=4096).transient_blocks, 256)
        self.assertEqual(window_layout(forward=2048).transient_blocks, 128)
        self.assertEqual(window_layout(forward=16384).transient_blocks, 1024)
        p32 = window_layout(P=32)
        self.assertEqual((p32.window_blocks, p32.transient_blocks), (33, 256))

    def test_pool_one_sizes_of_the_design_table(self):
        # Design 3.1 table: 16 windows + T at N = 2,048 / 4,096 / 8,192 / 16,384.
        block = GEOMETRY.sliding_block_bytes(16)
        for forward, gib in ((2048, 14.26), (4096, 15.82), (8192, 18.95), (16384, 25.20)):
            layout = window_layout(forward=forward)
            blocks = I0.window_pool_blocks(layout, 5)
            self.assertEqual(blocks, 1 + 5 * (16 * 65 + layout.transient_blocks))
            self.assertAlmostEqual(blocks * block / GiB, gib, places=2)
        # One window slot: 325 block ids = 812.5 MiB.
        self.assertEqual(5 * 65 * block, 812.5 * MiB)

    def test_additional_config_round_trip(self):
        layout = window_layout()
        record = I0.to_additional_config(layout)
        self.assertEqual(record, {I0.LAYOUT_KEY: {'page_tokens': 16, 'state': 'sliding', 'window_tokens': 1024,
                                                  'window_slots': 16, 'forward_tokens': 8192}})
        self.assertEqual(I0.from_vllm_config(NS(additional_config=record)), layout)
        self.assertTrue(I0.is_window(layout))
        self.assertTrue(layout.decoupled)
        # The Mamba form is unchanged and is never a window layout.
        qwen = I0.for_server(16, 3)
        self.assertEqual(I0.to_additional_config(qwen), {I0.LAYOUT_KEY: {
            'page_tokens': 16, 'mamba_tokens': 784, 'mamba_slots': 7}})
        self.assertFalse(I0.is_window(qwen))
        with self.assertRaisesRegex(ValueError, 'no Mamba pool'):
            I0.mamba_pool_blocks(layout, 3)
        with self.assertRaisesRegex(ValueError, 'window pool'):
            I0.window_pool_blocks(qwen, 5)

    def test_refusals(self):
        record = I0.to_additional_config(window_layout())[I0.LAYOUT_KEY]
        for bad in (dict(record, state='mamba'), dict(record, extra=1), {k: v for k, v in record.items()
                                                                         if k != 'window_slots'},
                    dict(record, page_tokens=24), dict(record, window_tokens=1000),
                    dict(record, window_slots=0), dict(record, window_slots=65),
                    dict(record, forward_tokens=8200), dict(record, forward_tokens=8),
                    dict(record, window_slots=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: bad}))
        with self.assertRaises(ValueError):
            I0.for_window_server(16, 0, 1024, 8192)


class WindowPlannerTests(unittest.TestCase):
    def test_groups_pools_and_tensors(self):
        w = gemma_world()
        for P in (16, 32):
            for sessions, budget in ((1, 28 * GiB), (10, 28 * GiB), (16, 32 * GiB)):
                with self.subTest(P=P, sessions=sessions, budget=budget):
                    layout = window_layout(P, sessions)
                    _, config = plan(w, layout, budget)
                    groups = config.kv_cache_groups
                    self.assertEqual(len(groups), 6)
                    kinds = [type(g.kv_cache_spec).__name__ for g in groups]
                    self.assertEqual(sorted(kinds), ['FullAttentionSpec'] + ['SlidingWindowSpec'] * 5)
                    self.assertEqual(config.group_pool, tuple(int(k == 'SlidingWindowSpec') for k in kinds))
                    full = [g for g in groups if type(g.kv_cache_spec).__name__ == 'FullAttentionSpec']
                    sliding = [g for g in groups if type(g.kv_cache_spec).__name__ == 'SlidingWindowSpec']
                    self.assertEqual(full[0].layer_names, [layer(i) for i in GEOMETRY.global_layers])
                    self.assertEqual([g.layer_names for g in sliding],
                                     [[layer(i) for i in group] for group in GEOMETRY.sliding_groups()])
                    # Every group keeps P-token blocks: no page-size unification.
                    self.assertEqual({g.kv_cache_spec.block_size for g in groups}, {P})
                    page = full[0].kv_cache_spec.page_size_bytes
                    self.assertEqual(page * 10, GEOMETRY.page_bytes(P))
                    swa = sliding[0].kv_cache_spec.page_size_bytes
                    self.assertEqual(swa * 10, GEOMETRY.sliding_block_bytes(P))
                    n1 = I0.window_pool_blocks(layout, 5)
                    n0 = (budget - n1 * swa * 10) // (page * 10)
                    self.assertEqual(config.num_blocks_by_pool, (n0, n1))
                    self.assertEqual(config.num_blocks, n0)
                    tensors = config.kv_cache_tensors
                    self.assertEqual(len(tensors), 20)
                    self.assertEqual(config.tensor_pools(), [0] * 10 + [1] * 10)
                    for i, tensor in enumerate(tensors[:10]):
                        self.assertEqual(tensor.shared_by, [full[0].layer_names[i]])
                        self.assertEqual(tensor.size, n0 * page)
                    for i, tensor in enumerate(tensors[10:]):
                        self.assertEqual(tensor.shared_by, [g.layer_names[i] for g in sliding])
                        self.assertEqual(tensor.size, n1 * swa)
                    self.assertLessEqual(sum(t.size for t in tensors), budget)

    def test_design_pool_sizes(self):
        # Design 3.1/3.4: P = 16, N = 8,192, 16 windows: pool 1 = 7,761 block ids (18.95 GiB),
        # pool 0 the rest in 1.25 MiB Pages.
        w = gemma_world()
        _, config = plan(w, window_layout(16, 16), 32 * GiB)
        self.assertEqual(config.num_blocks_by_pool[1], 7761)
        self.assertEqual(config.num_blocks_by_pool[0], (32 * GiB - 7761 * 5 * 512 * 1024) // (1280 * 1024))
        _, config = plan(w, window_layout(16, 10), 28 * GiB)
        self.assertEqual(config.num_blocks_by_pool[1], 1 + 5 * (10 * 65 + 512))

    def test_startup_check_at_28_gib(self):
        # vLLM's own startup check: one request at max_model_len in pool 0 plus the
        # static pool 1. At 28 GiB, N = 8,192 and 131,072 tokens: 16 windows need
        # 10 + 18.95 GiB and are refused; 12 fit (10 + 15.77 GiB). Sessions are
        # limited to 10-12 in stage 1 (D-5).
        w = gemma_world()
        with self.assertRaisesRegex(ValueError, 'max seq len'):
            plan(w, window_layout(16, 16), 28 * GiB)
        _, config = plan(w, window_layout(16, 12), 28 * GiB)
        self.assertEqual(config.num_blocks_by_pool[1], 1 + 5 * (12 * 65 + 512))
        with self.assertRaisesRegex(ValueError, 'max seq len'):
            plan(w, window_layout(16, 12), 28 * GiB, max_len=262144)

    def test_block_sizes_are_one_for_every_group(self):
        w = gemma_world()
        for P in (16, 32):
            vc, config = plan(w, window_layout(P), 32 * GiB)
            self.assertEqual(w.utils.resolve_kv_cache_block_sizes(config, vc), (P, P))

    def test_memory_check_and_override(self):
        w = gemma_world()
        layout = window_layout(16, 16)
        vc = vconfig(layout)
        groups = w.utils.get_kv_cache_groups(vc, gemma_specs(w, 16))
        need = w.utils._max_memory_usage_bytes_from_groups(vc, groups)
        # One request at 131,072 tokens in pool 0 (8,192 Pages) plus the static pool 1.
        self.assertEqual(need, 8192 * GEOMETRY.page_bytes(16) + 7761 * GEOMETRY.sliding_block_bytes(16))
        _, config = plan(w, layout, 32 * GiB, override=9000)
        self.assertEqual(config.num_blocks_by_pool, (9000, 7761))
        self.assertEqual(config.kv_cache_tensors[0].size, 9000 * GEOMETRY.page_bytes(16) // 10)
        self.assertEqual(config.kv_cache_tensors[-1].size, 7761 * GEOMETRY.sliding_block_bytes(16) // 10)

    def test_capacity_is_per_pool(self):
        w = gemma_world()
        vc, config = plan(w, window_layout(16, 4), 28 * GiB)
        tokens, concurrency = w.utils.get_kv_cache_capacity(vc, config)
        full = config.num_blocks_by_pool[0] / (131072 // 16)
        swa = config.num_blocks_by_pool[1] / (5 * (cdiv(1023 + 8192, 16) + 1))
        self.assertAlmostEqual(concurrency, min(full, swa))

    def test_refusals(self):
        w = gemma_world()
        layout = window_layout()
        vc = vconfig(layout)
        good = gemma_specs(w, 16)
        cases = {
            'full block != P': {n: (full_spec(w, 32) if type(s).__name__ == 'FullAttentionSpec' else s)
                                for n, s in good.items()},
            'sliding block != P': {n: (sliding_spec(w, 32) if type(s).__name__ == 'SlidingWindowSpec' else s)
                                   for n, s in good.items()},
            'window != W': gemma_specs(w, 16, window=512),
        }
        for name, specs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                w.utils.get_kv_cache_config_from_groups(vc, w.utils.get_kv_cache_groups(vc, specs), 28 * GiB)
        groups = w.utils.get_kv_cache_groups(vc, good)
        only_full = [g for g in groups if type(g.kv_cache_spec).__name__ == 'FullAttentionSpec']
        with self.assertRaisesRegex(ValueError, 'full-attention and sliding-window'):
            w.utils.get_kv_cache_config_from_groups(vc, only_full, 28 * GiB)
        # The window layout on Qwen's Mamba groups, and the Mamba layout on Gemma's groups.
        qwen = I0.for_server(16, 1)
        with self.assertRaises(ValueError):
            w.utils.get_kv_cache_config_from_groups(
                vc, w.utils.get_kv_cache_groups(KP.vconfig(qwen), KP.qwen_specs(w, qwen)), 28 * GiB)
        with self.assertRaisesRegex(ValueError, 'attention and Mamba'):
            w.utils.get_kv_cache_config_from_groups(KP.vconfig(qwen), groups, 28 * GiB)

    def test_qwen_planner_unchanged(self):
        # The Mamba form plans exactly as before (test_kv_pools compares it with the
        # base commit field by field); spot-check one configuration here.
        w = gemma_world()
        _, config = KP.plan(w, I0.for_server(16, 4), 24 * GiB)
        self.assertEqual(config.num_blocks_by_pool, (KP.fa_pool_blocks(I0.for_server(16, 4), 24 * GiB, 4), 28))


def small_gemma_config(w, P, n_full, n_swa, layers=2):
    """A Gemma-shaped config with few layers: 1 full-attention group, 5 sliding groups."""
    groups = [w.iface.KVCacheGroupSpec([f'g{i}' for i in range(layers)], full_spec(w, P))]
    groups += [w.iface.KVCacheGroupSpec([f's{g}.{i}' for i in range(layers)], sliding_spec(w, P))
               for g in range(5)]
    return w.iface.KVCacheConfig(num_blocks=n_full, kv_cache_tensors=[], kv_cache_groups=groups,
                                 group_pool=(0, 1, 1, 1, 1, 1), num_blocks_by_pool=(n_full, n_swa))


class WindowPoolTests(unittest.TestCase):
    def manager(self, n_full=900, n_swa=1000):
        w = gemma_world()
        return w, KP.kv_manager(w, small_gemma_config(w, 16, n_full, n_swa), hash_block_size=16,
                                scheduler_block_size=16)

    def test_groups_draw_from_their_pools(self):
        w, manager = self.manager()
        self.assertEqual(manager.coordinator.group_pool, (0, 1, 1, 1, 1, 1))
        self.assertEqual([p.num_gpu_blocks for p in manager.block_pools], [900, 1000])
        req = KP.request('a', 3000)
        ids = KP.alloc(manager, req, 3000).get_block_ids()
        # Full attention: every token's block; sliding: the same count before any skip.
        self.assertEqual(len(ids[0]), cdiv(3000, 16))
        self.assertEqual(manager.get_num_free_blocks_by_pool(), (899 - 188, 999 - 5 * 188))
        manager.free(req)
        self.assertEqual(manager.get_num_free_blocks_by_pool(), (899, 999))

    def test_sliding_blocks_outside_the_window_go_back_to_pool_one(self):
        w, manager = self.manager()
        req = KP.request('a', 4096)
        KP.alloc(manager, req, 2048)
        held = manager.get_num_free_blocks_by_pool()
        # The next chunk first frees the sliding blocks that left the window
        # [n - 1023, n] (remove_skipped_blocks), then allocates.
        req.num_tokens = 4096
        KP.alloc(manager, req, 2048)
        full_free, swa_free = manager.get_num_free_blocks_by_pool()
        self.assertEqual(full_free, held[0] - 128)
        live = [sum(1 for b in row if not b.is_null) for row in manager.coordinator.get_blocks('a')[1:]]
        # Skipped = max(0, 2048 - 1024 + 1) = 1025 tokens -> 64 whole blocks freed.
        self.assertEqual(live, [256 - 64] * 5)
        self.assertEqual(swa_free, 999 - 5 * (256 - 64))

    def test_window_null_head_and_bound(self):
        # A decoding request holds at most ceil(1024 / 16) + 1 = 65 live blocks per group.
        w, manager = self.manager()
        req = KP.request('d', 5000)
        KP.alloc(manager, req, 5000)
        for step in range(40):
            req.num_tokens += 1
            KP.alloc(manager, req, 1)
            live = [sum(1 for b in row if not b.is_null) for row in manager.coordinator.get_blocks('d')[1:]]
            self.assertLessEqual(max(live), 65)


class ScannerTests(unittest.TestCase):
    def test_scheduler_init_has_no_mamba_alignment_for_the_window_layout(self):
        node = KP.scheduler_node('__init__')
        block = next(n for n in ast.walk(node) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'kv_cache_config.group_pool is not None')
        code = compile(ast.Module(body=[block], type_ignores=[]), 'scheduler-init', 'exec')
        w = gemma_world()
        me = NS(defer_block_free=False, _remote_ra_mamba_align_tokens=None)
        exec(code, {'self': me, 'kv_cache_config': small_gemma_config(w, 16, 40, 40),
                    'MambaSpec': w.iface.MambaSpec})
        self.assertIsNone(me._remote_ra_mamba_align_tokens)
        me = NS(defer_block_free=True, _remote_ra_mamba_align_tokens=None)
        with self.assertRaisesRegex(ValueError, 'deferred block frees'):
            exec(code, {'self': me, 'kv_cache_config': small_gemma_config(w, 16, 40, 40),
                        'MambaSpec': w.iface.MambaSpec})


def window_check(w):
    """interface._check_decoupled_window_block_size, run from the overlay source."""
    text = (SOURCE / 'vllm/platforms/interface.py').read_text('utf-8')
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_check_decoupled_window_block_size']
    namespace = {'logger': Log(), 'Any': typing.Any}
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[])),
         namespace)
    return namespace['_check_decoupled_window_block_size']


def platform_vc(layout, block_size=16, user_block=True, mamba_block=None):
    return NS(additional_config=I0.to_additional_config(layout) or {},
              cache_config=NS(block_size=block_size, user_specified_block_size=user_block,
                              mamba_block_size=mamba_block,
                              user_specified_mamba_block_size=mamba_block is not None))


class PlatformTests(unittest.TestCase):
    def run_check(self, vc, backend=KP.Backend):
        w = gemma_world()
        with patch.dict(sys.modules, KP.platform_namespace(w)):
            window_check(w)(vc, backend)

    def test_window_layout_needs_the_user_block_size(self):
        layout = window_layout()
        self.run_check(platform_vc(layout))
        for kw in (dict(user_block=False), dict(block_size=32), dict(mamba_block=784)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.run_check(platform_vc(layout, **kw))

        class Coarse(KP.Backend):
            sizes = [32]
        with self.assertRaisesRegex(ValueError, 'kernel block size'):
            self.run_check(platform_vc(layout), backend=Coarse)
        self.run_check(platform_vc(window_layout(P=32), block_size=32), backend=Coarse)

    def test_no_key_is_the_upstream_path_and_mamba_form_refused(self):
        self.run_check(NS(additional_config={}, cache_config=NS()))
        with self.assertRaisesRegex(ValueError, 'hybrid'):
            self.run_check(platform_vc(I0.for_server(16, 1)))

    def test_hybrid_check_refuses_the_window_form(self):
        layout = window_layout()
        vc = KP.platform_config(I0.for_server(16, 1), 16, True, 784)
        vc.additional_config = I0.to_additional_config(layout)
        with self.assertRaisesRegex(ValueError, 'window layout is for sliding-window models'):
            PlatformRun().align(vc)


class PlatformRun:
    def align(self, vc):
        w = gemma_world()
        with patch.dict(sys.modules, KP.platform_namespace(w)):
            KP.platform_functions()(None, vc, KP.Backend)


if __name__ == '__main__':
    unittest.main()
