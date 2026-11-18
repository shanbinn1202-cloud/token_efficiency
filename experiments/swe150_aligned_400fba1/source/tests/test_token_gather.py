"""Token gather (token-gather-1114): the KV layout, the gather copy, the attention it gives, the picks, the fill and
the pin.

CPU only (CPU torch for the tensor tests). Covered:

- the KV layout (2026-11-14, after GPU job 3602601 refused at startup: its worker found the FlashAttention views
  [blocks, 2, P, heads, head size], block axis 0, where the check and the first fixtures assumed [2, blocks, P, ...],
  block axis 1): the views are built exactly as the server builds them, with vLLM 0.24.0's own code compiled from the
  pinned source closure (FlashAttentionBackend.get_kv_cache_shape and get_kv_cache_stride_order, the GPU runner's
  _reshape_attention_kv_cache) and the runtime's views_from_runner, in both stride orders (NHD, and HND, which vLLM
  uses here because the runtime's OffloadingConnector asks for it, and a padded block stride); check_views accepts
  exactly that layout, reads the block and token axes from the loaded backend, refuses the old layout and other
  stride orders; the old check and the old indexing fail on the real views (what the job hit); the worker's attach
  path attaches the axes;
- TensorIO.gather_tokens on those views (bf16, as on the GPU): every gathered slot is bit-identical to its source
  row (device rows, arena Pages across chunks, a captured single Page), nothing else changes, the H2D bytes count
  only the host rows, and every refusal fires before the first copy, in both stride orders;
- the attention a gathered row gives (both stride orders): a toy RoPE attention layer whose keys are rotated at their
  logical positions before they are written (as Qwen3.6 stores K), read through a block table up to the physical
  sequence length as FlashAttention does, equals dense attention over exactly the recorded token set (selected
  Pages, picks and pad, the run); positions stay logical; the order of prefix keys does not matter; a zero or
  empty pad slot would change the output (why the pad holds real tokens);
- the pickers (markers newest first with multi-token markers and the incremental stream index, separators
  with a quota), the shared-blocks fill (Selector V2's fill without picks; never over the room), the pad rule,
  the record, the receipt-to-plan check and the contract pin (absent: every contract keeps its bytes).

Not covered here (GPU validation, docs/token-gather/DESIGN.md): the CUDA index kernels, pinned host memory and
non_blocking ordering, and the real model's logits against a full-Page restore.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import os
import random
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'source-overlay'))

from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2 import token_gather as TG
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_growing_worker.contract import WorkerAbort
REFUSED = (LiveV2Error, WorkerAbort)

try:
    import torch
except ImportError:
    torch = None


# ---- the server's KV views: vLLM 0.24.0's own code and the runtime's views_from_runner ---------------------------

FLASH_ATTN = 'vllm/v1/attention/backends/flash_attn.py'
ATTN_UTILS = 'vllm/v1/worker/gpu/attn_utils.py'
CONNECTOR = 'vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py'
ORDERS = ('NHD', 'HND')
_VLLM = {}


def vllm_code():
    """vLLM 0.24.0's FlashAttentionBackend.get_kv_cache_shape and get_kv_cache_stride_order and the GPU runner's
    _reshape_attention_kv_cache, compiled from the text of the pinned source closure (importing vLLM needs CUDA):
    (backend class, reshape function, state); ``state['layout']`` is what get_kv_cache_layout() answers."""
    if not _VLLM:
        nodes = []
        text = (HERE / 'source-overlay' / FLASH_ATTN).read_text(encoding='utf-8')
        backend = next(n for n in ast.parse(text).body
                       if isinstance(n, ast.ClassDef) and n.name == 'FlashAttentionBackend')
        for node in backend.body:
            if isinstance(node, ast.FunctionDef) and node.name in ('get_kv_cache_shape', 'get_kv_cache_stride_order'):
                node.decorator_list = []            # @staticmethod, applied again below
                nodes.append(node)
        text = (HERE / 'source-overlay' / ATTN_UTILS).read_text(encoding='utf-8')
        nodes += [n for n in ast.parse(text).body
                  if isinstance(n, ast.FunctionDef) and n.name == '_reshape_attention_kv_cache']
        assert [n.name for n in nodes] == ['get_kv_cache_shape', 'get_kv_cache_stride_order',
                                           '_reshape_attention_kv_cache'], [n.name for n in nodes]
        state = dict(layout='NHD')
        scope = dict(torch=torch, prod=math.prod, AttentionSpec=object, get_kv_cache_layout=lambda: state['layout'],
                     get_dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'vllm-0.24.0 %s + %s' % (FLASH_ATTN, ATTN_UTILS),
                     'exec'), scope)
        flash = type('FlashAttentionBackend', (), dict(
            get_kv_cache_shape=staticmethod(scope['get_kv_cache_shape']),
            get_kv_cache_stride_order=staticmethod(scope['get_kv_cache_stride_order'])))
        _VLLM.update(backend=flash, reshape=scope['_reshape_attention_kv_cache'], state=state)
    return _VLLM['backend'], _VLLM['reshape'], _VLLM['state']


class FullAttentionSpec(types.SimpleNamespace):
    """The fields of vLLM 0.24.0's FullAttentionSpec that views_from_runner and _reshape_attention_kv_cache read."""


class MambaSpec(types.SimpleNamespace):
    """The fields of vLLM 0.24.0's MambaSpec that views_from_runner reads."""


class OldLayoutBackend:
    """The layout the first token gather check and fixtures assumed, [2, blocks, P, heads, head size] (block axis 1),
    made block-major as vLLM 0.24.0's _update_hybrid_attention_mamba_layout does for a hybrid model with such a
    backend (``block_major``). Not vLLM 0.24.0's FlashAttention."""

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (0, 1, 2, 3, 4)

    @staticmethod
    def block_major(cache):
        hidden = cache.shape[2:].numel()
        cache.as_strided_(size=cache.shape, stride=(hidden, 2 * hidden, *cache.stride()[2:]))


class FlashRunner:
    """A loaded runner's KV caches as vLLM 0.24.0 builds them for Qwen3.6's decoupled layout, small: FA group 0
    (``layers`` FlashAttention layers fa0, fa1, ... of ``blocks`` P-token blocks: raw int8 storage reshaped by
    _reshape_attention_kv_cache with the backend's shape and the stride order of ``order``, as get_kv_cache_layout
    answers) and GDN group 1 (one Mamba layer, its two states strided over a raw page as GPUModelRunner does), and
    the runtime's own views_from_runner over them. ``padded``: a larger FA page (vLLM's padded strided view)."""
    MAMBA = ((3,), (2, 2))

    def __init__(self, order, blocks, P, heads, size, layers=2, dtype=None, backend=None, padded=0):
        from remote_ra_growing_worker.tensors import views_from_runner
        flash, reshape, state = vllm_code()
        backend = backend or flash
        dtype = dtype or torch.bfloat16
        state['layout'] = order
        item = torch.empty((), dtype=dtype).element_size()
        page = 2 * P * heads * size * item + padded
        spec = FullAttentionSpec(block_size=P, storage_block_size=P, num_kv_heads=heads, head_size=size, dtype=dtype,
                                 page_size_padded=page if padded else None, page_size_bytes=page)
        self.names = ['fa%d' % i for i in range(layers)]
        caches, configs = {}, []
        for name in self.names:
            raw = torch.zeros(blocks * page, dtype=torch.int8)
            shape = backend.get_kv_cache_shape(blocks, P, heads, size, cache_dtype_str='auto')
            caches[name] = reshape(raw, spec, shape, backend.get_kv_cache_stride_order(), blocks, None)
            if hasattr(backend, 'block_major'):
                backend.block_major(caches[name])
            configs.append(types.SimpleNamespace(shared_by=[name], block_stride=0, offset=0))
        dtypes = (torch.float64, torch.float64)
        mamba_page = sum(math.prod(s) * 8 for s in self.MAMBA)
        mamba = MambaSpec(block_size=784, shapes=self.MAMBA, dtypes=dtypes, page_size_bytes=mamba_page)
        raw, states, offset = torch.zeros(blocks * mamba_page, dtype=torch.int8), [], 0
        for shape, dt in zip(self.MAMBA, dtypes):
            target = (blocks,) + shape
            stride = torch.empty(target).stride()
            states.append(torch.as_strided(raw.view(dt), size=target, stride=(mamba_page // 8,) + stride[1:],
                                           storage_offset=offset // 8))
            offset += stride[0] * 8
        caches['gdn'] = states
        configs.append(types.SimpleNamespace(shared_by=['gdn'], block_stride=0, offset=0))
        groups = [types.SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=list(self.names),
                                        backend=backend),
                  types.SimpleNamespace(kv_cache_group_id=1, kv_cache_spec=mamba, layer_names=['gdn'], backend=None)]
        self.runner = types.SimpleNamespace(
            kv_cache_config=types.SimpleNamespace(kv_cache_tensors=configs),
            _kv_cache_spec_attn_group_iterator=lambda: iter(groups), runner_only_attn_layers=set(),
            cache_config=types.SimpleNamespace(cache_dtype='auto'))
        self.caches = caches
        self.views = views_from_runner(self.runner, caches, [P, 784])
        self.fa = [caches[n] for n in self.names]
        self.gdn = caches['gdn']

    def fill(self, seed):
        g = torch.Generator().manual_seed(seed)
        for t in self.fa:
            t.copy_(torch.randn(tuple(t.shape), generator=g, dtype=torch.float64).to(t.dtype))
        for t in self.gdn:
            t.copy_(torch.randn(tuple(t.shape), generator=g, dtype=t.dtype))
        return self


def old_check_views(views, page_tokens):
    """token_gather.check_views of 58bc6a6 / 325cdc5 (GPU job 3602601's worker refused its startup on it)."""
    attention = [v for v in views if v.kind == 'attention']
    return bool(attention) and all(v.ratio == 1 and v.kernel_tokens in (0, page_tokens) and
                                   all(t.ndim >= 4 and a == 1 and t.shape[0] == 2 and t.shape[2] == page_tokens
                                       for t, a in zip(v.tensors, v.axes)) for v in attention)


def bits(row):
    """A K/V row as integers of its element size (bit-exact comparison)."""
    return row.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[row.element_size()])


# ---- the KV layout -----------------------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required')
class LayoutTests(unittest.TestCase):
    """check_views on the server's views; what GPU job 3602601 hit."""
    BLOCKS, P, HEADS, SIZE = 24, 16, 2, 4

    def flash(self, order, **kwargs):
        return FlashRunner(order, self.BLOCKS, self.P, self.HEADS, self.SIZE, **kwargs).fill(1)

    def test_the_fixture_is_vllm_0_24_0s_own_code(self):
        release = json.loads((HERE / 'INSTALL_RELEASE.json').read_text(encoding='utf-8'))
        closure = release['source_closure']['files']
        for name in (FLASH_ATTN, ATTN_UTILS):
            self.assertEqual(hashlib.sha256((HERE / 'source-overlay' / name).read_bytes()).hexdigest(), closure[name])
            self.assertNotIn(name, release['payload'])          # stock vLLM 0.24.0: the runtime does not patch it
        flash, _, state = vllm_code()
        self.assertEqual(flash.get_kv_cache_shape(17, 16, 4, 256), (17, 2, 16, 4, 256))     # Qwen3.6-27B, P = 16
        for layout, order in (('NHD', (0, 1, 2, 3, 4)), ('HND', (0, 1, 3, 2, 4))):
            state['layout'] = layout
            self.assertEqual(flash.get_kv_cache_stride_order(), order)
            self.assertEqual(TG.STRIDE_ORDERS[order], layout)
        # The runtime's OffloadingConnector asks for HND (why the server's views are HND).
        text = (HERE / 'source-overlay' / CONNECTOR).read_text(encoding='utf-8')
        self.assertIn('def get_required_kvcache_layout', text)
        self.assertIn('return "HND"', text)

    def test_views_are_flash_attention_blocks_first_in_both_stride_orders(self):
        H, D, P, N = self.HEADS, self.SIZE, self.P, self.BLOCKS
        strides = dict(NHD=(2 * P * H * D, P * H * D, H * D, D, 1), HND=(2 * P * H * D, P * H * D, D, P * D, 1))
        for order in ORDERS:
            with self.subTest(order=order):
                flash = self.flash(order)
                fa = [v for v in flash.views if v.kind == 'attention']
                self.assertEqual([(tuple(v.tensors[0].shape), v.axes, tuple(v.tensors[0].stride())) for v in fa],
                                 [((N, 2, P, H, D), (0,), strides[order])] * 2)
                layouts = TG.runner_layouts(flash.runner, P)
                self.assertEqual(layouts, {name: dict(shape=(17, 2, P, H, D), block_axis=0, token_axis=2, heads=H,
                                                      head_size=D) for name in flash.names})
                self.assertEqual(TG.check_views(flash.views, P, layouts), {(n, 0): (0, 2) for n in flash.names})
                self.assertEqual([TG.STRIDE_ORDERS[TG.stride_order(v.tensors[0], 0)] for v in fa], [order] * 2)
        padded = self.flash('HND', padded=64)          # a padded block stride: still dense within a block
        t = padded.fa[0]
        self.assertEqual(t.stride()[0] * t.element_size(), 2 * P * H * D * 2 + 64)
        self.assertEqual(TG.check_views(padded.views, P, TG.runner_layouts(padded.runner, P))[('fa0', 0)], (0, 2))

    def test_the_old_check_and_indexing_fail_on_the_server_views(self):
        # GPU job 3602601 (2026-11-14): the worker refused at startup ("token gather needs FullAttention views
        # [2, blocks, 16, heads, head size] (block axis 1)"); past the check, the old gather_tokens indexing
        # tensor[:, blocks, offsets] would have indexed the K/V axis by block id.
        for order in ORDERS:
            with self.subTest(order=order):
                flash = self.flash(order)
                self.assertFalse(old_check_views(flash.views, self.P))
                t = flash.fa[0]
                with self.assertRaises(IndexError):
                    t[:, torch.tensor([5, 7]), torch.tensor([3, 4])]       # block ids beyond the K/V axis
                wrong = t[:, torch.tensor([1]), torch.tensor([3])].movedim(1, 0)
                self.assertNotEqual(tuple(wrong.shape), (1, 2, self.HEADS, self.SIZE))   # not one token's row
        # The new check refuses the old layout, by its backend or as the first fixtures built it by hand.
        old = FlashRunner('NHD', self.BLOCKS, self.P, self.HEADS, self.SIZE, backend=OldLayoutBackend)
        self.assertEqual([v.axes for v in old.views if v.kind == 'attention'], [(1,), (1,)])
        self.assertTrue(old_check_views(old.views, self.P))                       # what the old check wanted
        with self.assertRaisesRegex(LiveV2Error, r'\[blocks, 2, 16, heads, head size\] \(block axis 0, token axis 2'):
            TG.check_views(old.views, self.P, TG.runner_layouts(old.runner, self.P))
        from remote_ra_growing_worker.tensors import LayerView
        server = self.flash('HND')
        by_hand = [LayerView(name, 0, 'attention', (torch.zeros(2, self.BLOCKS, self.P, self.HEADS, self.SIZE),),
                             (1,), allocation_tokens=self.P, kernel_tokens=self.P) for name in server.names]
        with self.assertRaisesRegex(LiveV2Error, 'block axis 0, token axis 2'):
            TG.check_views(by_hand, self.P, TG.runner_layouts(server.runner, self.P))

    def test_check_views_refusals(self):
        flash = self.flash('HND')
        layouts = TG.runner_layouts(flash.runner, self.P)
        with self.assertRaisesRegex(LiveV2Error, 'no attention backend layout for fa1'):
            TG.check_views(flash.views, self.P, {'fa0': layouts['fa0']})
        with self.assertRaisesRegex(LiveV2Error, 'one 32-token kernel block'):
            TG.check_views(flash.views, 32, layouts)
        with self.assertRaisesRegex(LiveV2Error, 'FullAttention views and their layouts'):
            TG.check_views([v for v in flash.views if v.kind != 'attention'], self.P, layouts)

        class HeadsOutermost:          # the logical layout of FlashAttention in a third stride order
            get_kv_cache_shape = staticmethod(vllm_code()[0].get_kv_cache_shape)

            @staticmethod
            def get_kv_cache_stride_order(include_num_layers_dimension=False):
                return (0, 3, 1, 2, 4)
        odd = FlashRunner('HND', self.BLOCKS, self.P, self.HEADS, self.SIZE, backend=HeadsOutermost)
        self.assertEqual(tuple(odd.fa[0].shape), (self.BLOCKS, 2, self.P, self.HEADS, self.SIZE))
        with self.assertRaisesRegex(LiveV2Error, 'NHD or HND stride order'):
            TG.check_views(odd.views, self.P, TG.runner_layouts(odd.runner, self.P))

        class Ambiguous:               # a block size that moves two axes
            @staticmethod
            def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
                return (num_blocks, 2, block_size, block_size, head_size)
        with self.assertRaisesRegex(LiveV2Error, 'not uniquely identifiable'):
            TG.backend_layout(Ambiguous, self.P, self.HEADS, self.SIZE)

    def test_attach_token_gather_refusals(self):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO
        flash = self.flash('NHD')
        good = TG.check_views(flash.views, self.P, TG.runner_layouts(flash.runner, self.P))
        for bad in ({('fa0', 0): (0, 2)}, {**good, ('fa0', 0): (1, 2)}, {**good, ('fa0', 0): (0, 0)},
                    {**good, ('fa0', 0): (0, 5)}, {**good, ('gdn', 0): (0, 1)}, [], None):
            io = CpuTensorTestIO(flash.views, decoupled=True)
            with self.assertRaisesRegex(REFUSED, 'block and token axis'):
                io.attach_token_gather(bad)
            self.assertIsNone(io.gather_axes)
        io = CpuTensorTestIO(flash.views, decoupled=True)
        io.attach_token_gather(good)
        with self.assertRaisesRegex(REFUSED, 'block and token axis'):
            io.attach_token_gather(good)                              # once

    def test_the_worker_attach_checks_the_views_and_attaches_the_axes(self):
        import test_cudagraph as TCG
        import test_integrate_flags as IF
        from remote_ra_growing_worker.tensors import CpuTensorTestIO

        class IO(CpuTensorTestIO):
            def attach_arena(self, arena):                # IF.attach's arena is a stand-in
                self.arena = arena

        def attach(flash):
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'), patch.object(IF, 'ArenaIO', IO), \
                    patch.object(TCG.AttachLoadedWorkerTests, 'views', lambda self: flash.views):
                return IF.attach(dict(token_gather=True), decoupled=True,
                                 _kv_cache_spec_attn_group_iterator=flash.runner._kv_cache_spec_attn_group_iterator,
                                 cache_config=flash.runner.cache_config)
        for order in ORDERS:
            with self.subTest(order=order):
                flash = self.flash(order)
                evidence, runner = attach(flash)
                owner = runner._remote_ra_live_v2_allocation_owner
                self.assertIs(evidence[TG.WORKER_KEY], True)
                self.assertIs(owner.token_gather, True)
                self.assertEqual(owner.io.gather_axes, {(n, 0): (0, 2) for n in flash.names})
                self.assertEqual([g['strides'][0] for g in evidence['tensor_geometry'] if g['kind'] == 'attention'],
                                 [list(t.stride()) for t in flash.fa])
        old = FlashRunner('NHD', self.BLOCKS, self.P, self.HEADS, self.SIZE, backend=OldLayoutBackend)
        with self.assertRaisesRegex(LiveV2Error, r'token gather needs FullAttention views \[blocks, 2, 16'):
            attach(old)


# ---- the gather copy -------------------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required')
class GatherCopyTests(unittest.TestCase):
    """TensorIO.gather_tokens on the server's views (bf16; NHD and HND strides)."""
    BLOCKS, P, HEADS, SIZE = 24, 16, 2, 4
    FA, GDN = 0, 1

    def world(self, order, seed=0, chunk_pages=3, attach=True):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO, PageArena
        flash = FlashRunner(order, self.BLOCKS, self.P, self.HEADS, self.SIZE).fill(seed)
        io = CpuTensorTestIO(flash.views, decoupled=True)
        page_bytes = sum(t.numel() // t.shape[a] * t.element_size() for v in io.views if v.kind == 'attention'
                         for t, a in zip(v.tensors, v.axes))
        io.attach_arena(PageArena(io.views, max_pages=40, pin=False, chunk_bytes=chunk_pages * page_bytes))
        if attach:
            io.attach_token_gather(TG.check_views(io.views, self.P, TG.runner_layouts(flash.runner, self.P)))
        return io, flash.fa, flash.gdn

    def block(self, b):
        return {self.FA: (b,)}

    @staticmethod
    def row(t, b, o):
        """Token slot o of block b, read as vLLM 0.24.0 FlashAttention lays it out: [blocks, 2, P, heads, size]."""
        return bits(t[b, :, o])

    def test_every_slot_equals_its_source_row_bit_exact_and_nothing_else_moves(self):
        P = self.P
        for order in ORDERS:
            with self.subTest(order=order):
                io, fa, gdn = self.world(order)
                # DRAM Pages: blocks 1..7 captured into the arena (three chunks), then overwritten on the device.
                dram = dict(zip(range(1, 8), io.capture_pages([self.block(b) for b in range(1, 8)])))
                self.assertEqual(len({p.slot // io.arena.slots.chunk_slots for p in dram.values()}), 3)
                single = io.capture(self.block(8), 'attention')          # a one-Page capture (the phase-1 tail)
                original = [t.clone() for t in fa]
                for t in fa:
                    t[1:9] = -7.0
                rng = random.Random(order)
                wanted = []                                                # 32 slots: 2 gathered blocks
                for k in range(2 * P):
                    kind = ('device', 'arena', 'capture', 'arena')[k % 4]
                    block = dict(device=10 + k % 3, arena=1 + (5 * k) % 7, capture=8)[kind]
                    wanted.append(('device' if kind == 'device' else 'host', block, rng.randrange(P)))
                sources = [(kind, self.block(b) if kind == 'device' else (single if b == 8 else dram[b]), o)
                           for kind, b, o in wanted]
                targets = [self.block(20), self.block(21)]
                before = [t.clone() for t in fa]
                gdn_before = [t.clone() for t in gdn]
                io._http_h2d_issued = io._http_h2d_joined = 0
                io.gather_tokens(targets, sources)
                for t, ref, old in zip(fa, original, before):
                    for k, (kind, b, o) in enumerate(wanted):
                        dst = targets[k // P][self.FA][0]
                        self.assertTrue(torch.equal(self.row(t, dst, k % P), self.row(ref, b, o)), (k, kind, b, o))
                    untouched = [b for b in range(self.BLOCKS) if b not in (20, 21)]
                    self.assertTrue(torch.equal(bits(t[untouched]), bits(old[untouched])))
                for t, old in zip(gdn, gdn_before):
                    self.assertTrue(torch.equal(t, old))
                host_rows = sum(kind == 'host' for kind, _, _ in wanted)
                row_bytes = 2 * self.HEADS * self.SIZE * fa[0].element_size()
                self.assertEqual(io._http_h2d_issued, host_rows * row_bytes * len(fa))
                self.assertEqual(io._http_h2d_joined, io._http_h2d_issued)

    def test_device_only_gather_moves_no_host_bytes(self):
        for order in ORDERS:
            with self.subTest(order=order):
                io, fa, _ = self.world(order)
                original = [t.clone() for t in fa]
                sources = [('device', self.block(2 + k % 3), (5 * k) % self.P) for k in range(self.P)]
                io._http_h2d_issued = 0
                io.gather_tokens([self.block(15)], sources)
                self.assertEqual(io._http_h2d_issued, 0)
                for t, ref in zip(fa, original):
                    for k in range(self.P):
                        self.assertTrue(torch.equal(self.row(t, 15, k), self.row(ref, 2 + k % 3, (5 * k) % self.P)))

    def test_refusals_before_any_copy(self):
        for order in ORDERS:
            with self.subTest(order=order):
                io, fa, _ = self.world(order)
                payload = io.capture_pages([self.block(1)])[0]
                snapshot = [t.clone() for t in fa]
                good = [('device', self.block(2), 0)] * (self.P - 1) + [('host', payload, 1)]
                cases = [
                    ([self.block(20)], good[:-1]),                                       # one source short
                    ([self.block(20)], good[:-1] + [('host', payload, self.P)]),         # offset outside the Page
                    ([self.block(2)], good),                                             # target aliases a source
                    ([self.block(20), self.block(20)], good + good),                     # aliased targets
                    ([self.block(20)], good[:-1] + [('disk', payload, 0)]),              # unknown source kind
                    ([self.block(20)], good[:-1] + [('host', {'fa0': ()}, 0)]),          # incomplete payload
                    ([], []),                                                            # nothing to gather
                ]
                for targets, sources in cases:
                    with self.assertRaises(REFUSED, msg=repr(targets)):
                        io.gather_tokens(targets, sources)
                released = io.capture_pages([self.block(3)])[0]
                io.arena.slots.release(released.slot, released.token)
                with self.assertRaisesRegex(REFUSED, 'foreign or released'):
                    io.gather_tokens([self.block(20)], [('host', released, 0)] * self.P)
                unchecked, _, _ = self.world(order, attach=False)                  # no axes attached at startup
                with self.assertRaisesRegex(REFUSED, 'attach_token_gather'):
                    unchecked.gather_tokens([self.block(20)], [('device', self.block(2), 0)] * self.P)
                for t, ref in zip(fa, snapshot):
                    self.assertTrue(torch.equal(bits(t), bits(ref)))


# ---- the attention a gathered row gives --------------------------------------------------------------------

def rope(x, positions, base=10000.0):
    """Rotary embedding of x [n, d] at integer positions (the even/odd pairs rotate)."""
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, dtype=torch.float64) / d)
    angle = positions.to(torch.float64)[:, None] * inv[None, :]
    cos, sin = torch.cos(angle), torch.sin(angle)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


@unittest.skipUnless(torch is not None, 'CPU torch required')
class GatheredAttentionTests(unittest.TestCase):
    """One attention layer (two KV heads, head size 8) over a 16-token-Page cache built by the runtime's own
    gather, on the server's view ([blocks, 2, P, heads, size], ``ORDER`` strides). Keys are rotated at their
    logical positions before they are written (K stored after RoPE)."""
    P, HEADS, SIZE, BLOCKS = 16, 2, 8, 64
    L, RUN = 10 * 16, 21          # candidate end and the run (observation + decode) after it
    ORDER = 'HND'

    def setUp(self):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO, PageArena
        g = torch.Generator().manual_seed(7)
        self.total = self.L + self.RUN
        x = torch.randn(self.total, 12, generator=g, dtype=torch.float64)
        wq, wk, wv = (torch.randn(12, self.HEADS * self.SIZE, generator=g, dtype=torch.float64) for _ in range(3))
        pos = torch.arange(self.total)
        split = lambda m: m.view(self.total, self.HEADS, self.SIZE)
        self.q = torch.stack([rope(split(x @ wq)[:, h], pos) for h in range(self.HEADS)], 1)
        self.k = torch.stack([rope(split(x @ wk)[:, h], pos) for h in range(self.HEADS)], 1)
        self.v = split(x @ wv)
        flash = FlashRunner(self.ORDER, self.BLOCKS, self.P, self.HEADS, self.SIZE, layers=1, dtype=torch.float64)
        self.cache = flash.fa[0]                  # [blocks, 2, P, heads, size]
        self.assertEqual(TG.STRIDE_ORDERS[TG.stride_order(self.cache, 0)], self.ORDER)
        self.io = CpuTensorTestIO(flash.views, decoupled=True)
        self.io.attach_arena(PageArena(self.io.views, max_pages=32, pin=False))
        self.io.attach_token_gather(TG.check_views(self.io.views, self.P, TG.runner_layouts(flash.runner, self.P)))
        # The session's history: Page p of [0, L) written once at blocks 1 + p, captured to DRAM, then released.
        n = self.L // self.P
        for p in range(n):
            self.write(1 + p, range(p * self.P, (p + 1) * self.P))
        self.dram = dict(enumerate(self.io.capture_pages([{0: (1 + p,)} for p in range(n)])))
        self.next = 1 + n

    def write(self, block, positions):
        for slot, t in enumerate(positions):
            self.cache[block, 0, slot] = self.k[t]
            self.cache[block, 1, slot] = self.v[t]

    def row(self, selected, tokens, resident=()):
        """The phase-2 row [selected Pages][gathered blocks][run]: block ids and the dropped token count."""
        P, table = self.P, []
        for p in selected:
            if p in resident:
                table.append(1 + p)                        # retained: the Page's own block
            else:
                b, self.next = self.next, self.next + 1   # restored from DRAM into a fresh block
                self.cache[b] = 0
                self.io.restore_pages([({0: (b,)}, self.dram[p])])
                table.append(b)
        gathered = []
        for _ in range(len(tokens) // P):
            b, self.next = self.next, self.next + 1
            gathered.append(b)
        if gathered:
            res = {p: {0: (1 + p,)} for p in resident}
            sources = TG.sources(tokens, P, res, None, self.dram)
            self.io.gather_tokens([{0: (b,)} for b in gathered], sources)
        table += gathered
        for k in range(-(-self.RUN // P)):                 # the run: logical [L, L + RUN)
            b, self.next = self.next, self.next + 1
            self.write(b, range(self.L + k * P, min(self.total, self.L + (k + 1) * P)))
            table.append(b)
        dropped = self.L - len(selected) * P - len(tokens)
        return table, dropped

    def flash(self, table, dropped, x):
        """FlashAttention's view of query x: every slot of the table up to physical x - dropped (causal)."""
        phys = x - dropped
        keys = torch.stack([self.cache[table[j // self.P], 0, j % self.P] for j in range(phys + 1)])
        values = torch.stack([self.cache[table[j // self.P], 1, j % self.P] for j in range(phys + 1)])
        return self.attend(self.q[x], keys, values)

    @staticmethod
    def attend(q, keys, values):
        scores = torch.einsum('hd,nhd->hn', q, keys) / q.shape[-1] ** 0.5
        return torch.einsum('hn,nhd->hd', torch.softmax(scores, -1), values)

    def dense(self, positions, x):
        positions = sorted(positions)
        return self.attend(self.q[x], self.k[positions], self.v[positions])

    def attended(self, selected, tokens, x):
        return [t for p in selected for t in range(p * self.P, (p + 1) * self.P)] + list(tokens) + \
            list(range(self.L, x + 1))

    def test_gathered_row_equals_dense_attention_over_the_recorded_tokens(self):
        P, n = self.P, self.L // self.P
        selected = [0, 2, 7]
        picks = [20, 21, 50, 63, 64, 99, 140, 141, 142]          # outside the selected Pages
        attended = set(selected)
        padding = TG.pad((-len(picks)) % P, self.L, P, attended, picks)
        tokens = sorted(picks + padding)
        self.assertEqual(len(tokens) % P, 0)
        self.assertTrue(all(t // P not in attended for t in tokens))
        table, dropped = self.row(selected, tokens, resident={2})
        for x in range(self.L, self.total):
            got = self.flash(table, dropped, x)
            want = self.dense(self.attended(selected, tokens, x), x)
            self.assertTrue(torch.allclose(got, want, rtol=0, atol=1e-12), x)
        # Positions stay logical: the gathered keys carry their own logical RoPE phase, bit for bit.
        gathered = table[len(selected)]
        for slot, t in enumerate(tokens):
            self.assertTrue(torch.equal(bits(self.cache[gathered, 0, slot]), bits(self.k[t])), t)
            self.assertTrue(torch.equal(bits(self.cache[gathered, 1, slot]), bits(self.v[t])), t)
        self.assertEqual(n, 10)

    def test_prefix_key_order_does_not_matter(self):
        selected, picks = [1], [3, 40, 77, 120, 155]
        tokens = sorted(picks + TG.pad((-len(picks)) % self.P, self.L, self.P, set(selected), picks))
        table, dropped = self.row(selected, tokens)
        x = self.total - 1
        want = self.flash(table, dropped, x)
        # Swap the gathered block with the selected Page's block in the prefix: same output.
        swapped = [table[1], table[0]] + table[2:]
        self.assertTrue(torch.allclose(self.flash(swapped, dropped, x), want, rtol=0, atol=1e-12))

    def test_a_zero_or_empty_pad_slot_changes_the_output(self):
        selected, picks = [4], [10, 30, 90]
        padding = TG.pad((-len(picks)) % self.P, self.L, self.P, set(selected), picks)
        tokens = sorted(picks + padding)
        table, dropped = self.row(selected, tokens)
        x = self.total - 1
        exact = self.dense(self.attended(selected, tokens, x), x)
        self.assertTrue(torch.allclose(self.flash(table, dropped, x), exact, rtol=0, atol=1e-12))
        without = self.dense(self.attended(selected, picks, x), x)
        gathered = table[len(selected)]
        for slot, t in enumerate(tokens):
            if t in padding:
                self.cache[gathered, :, slot] = 0.0                 # a zero key: score 0, not masked
        zeroed = self.flash(table, dropped, x)
        self.assertFalse(torch.allclose(zeroed, without, rtol=0, atol=1e-6))
        self.assertFalse(torch.allclose(zeroed, exact, rtol=0, atol=1e-6))

    def test_physical_positions_for_keys_would_be_wrong(self):
        # Rotating a gathered key at its physical slot instead of its logical position changes the output:
        # the runtime keeps logical positions (K is rotated once, at its logical position, when computed).
        selected, picks = [0], list(range(100, 116))
        table, dropped = self.row(selected, picks)
        x = self.total - 1
        good = self.flash(table, dropped, x)
        gathered = table[1]
        for slot, t in enumerate(picks):
            phys = len(selected) * self.P + slot
            for h in range(self.HEADS):
                raw = rope(self.k[t:t + 1, h], torch.tensor([-t]))          # undo the logical rotation
                self.cache[gathered, 0, slot, h] = rope(raw, torch.tensor([phys]))[0]
        self.assertFalse(torch.allclose(self.flash(table, dropped, x), good, rtol=0, atol=1e-6))


class GatheredAttentionNHDTests(GatheredAttentionTests):
    ORDER = 'NHD'


# ---- picks, fill, pad and the record --------------------------------------------------------------------------

def view(stream, end, eligible, pin, state=None, encode=None, separator=None, P=16):
    return types.SimpleNamespace(stream=tuple(stream), end=end, page_tokens=P, eligible=frozenset(eligible),
                                 state={} if state is None else state, encode=encode, separator=separator, pin=pin)


VOCAB = {'</think>': (9,), '<|im_end|>': (7,), '<|im_start|>user': (5, 6), 'user': (6,)}


class PickerTests(unittest.TestCase):
    def test_markers_newest_first_in_eligible_pages_first_token_of_a_sequence(self):
        P = 4
        stream = [1, 5, 6, 7,   9, 2, 5, 3,   7, 5, 6, 2,   2, 9, 7, 5]
        #         page 0         page 1         page 2         page 3
        pin = TG.pin('markers_v1')
        got = TG.PICKERS['markers_v1'].picks(view(stream, 16, {0, 1, 2, 3}, pin, encode=VOCAB.__getitem__, P=P))
        # 5 at 6 is not followed by 6: not a user start; 5 at 15 has no next token yet: not a match.
        self.assertEqual(got, [14, 13, 9, 8, 4, 3, 1])
        got = TG.PICKERS['markers_v1'].picks(view(stream, 12, {1, 2}, pin, encode=VOCAB.__getitem__, P=P))
        self.assertEqual(got, [9, 8, 4])
        quota = TG.pin('markers_v1', quota=2)
        got = TG.PICKERS['markers_v1'].picks(view(stream, 16, {0, 1, 2, 3}, quota, encode=VOCAB.__getitem__, P=P))
        self.assertEqual(got, [14, 13])
        only = TG.pin('markers_v1', markers=['<|im_end|>'])
        got = TG.PICKERS['markers_v1'].picks(view(stream, 16, {0, 1, 2, 3}, only, encode=VOCAB.__getitem__, P=P))
        self.assertEqual(got, [14, 8, 3])

    def test_incremental_index_equals_a_full_scan(self):
        rng = random.Random(3)
        pin = TG.pin('markers_v1')
        for trial in range(30):
            stream = [rng.choice((1, 2, 5, 6, 7, 9)) for _ in range(rng.randint(1, 300))]
            state, picker = {}, TG.PICKERS['markers_v1']
            for cut in sorted(rng.sample(range(1, len(stream) + 1), min(len(stream), 6))) + [len(stream)]:
                got = picker.picks(view(stream[:cut], cut, range(cut), pin, state, VOCAB.__getitem__, P=1))
                fresh = picker.picks(view(stream[:cut], cut, range(cut), pin, {}, VOCAB.__getitem__, P=1))
                full = [t for t in range(cut) if stream[t] in (7, 9) or
                        (stream[t] == 5 and t + 1 < cut and stream[t + 1] == 6)][::-1]
                self.assertEqual(got, full, trial)
                self.assertEqual(fresh, full, trial)
        with self.assertRaisesRegex(LiveV2Error, 'only grows'):
            picker.picks(view(stream[:1], 1, range(1), pin, state, VOCAB.__getitem__, P=1))

    def test_separators_newest_quota_in_eligible_pages(self):
        P = 4
        text = {1: ' ', 2: 'ab', 3: '.\n', 4: '', 8: '?!'}
        stream = [1, 2, 3, 4,   8, 1, 2, 2,   3, 3, 1, 2,   1, 1, 1, 1]
        pin = TG.pin('separators_v1', quota=3)
        separator = lambda t: TG.separator_text(text[t])
        got = TG.PICKERS['separators_v1'].picks(view(stream, 12, {0, 1, 2}, pin, separator=separator, P=P))
        self.assertEqual(got, [10, 9, 8])
        got = TG.PICKERS['separators_v1'].picks(view(stream, 12, {0, 1}, pin, separator=separator, P=P))
        self.assertEqual(got, [5, 4, 2])
        self.assertEqual(TG.pin('separators_v1')['quota'], 1024)
        self.assertFalse(TG.separator_text(''))
        self.assertTrue(TG.separator_text(' \t\n.,?!;:'))
        self.assertFalse(TG.separator_text('a.'))

    def test_marker_that_encodes_to_nothing_is_refused(self):
        pin = TG.pin('markers_v1', markers=['nothing'])
        with self.assertRaisesRegex(LiveV2Error, 'encodes to no tokens'):
            TG.PICKERS['markers_v1'].picks(view([1, 2], 2, {0}, pin, encode=lambda text: ()))


class FillTests(unittest.TestCase):
    def test_without_picks_it_is_selector_v2s_fill(self):
        order = [9, 3, 7, 1, 5]
        for room in range(0, 7):
            taken, gathered, kept = TG.fill(order, room, [], 16)
            self.assertEqual(taken, order[:room])
            self.assertEqual(gathered, [])

    def test_a_pick_inside_a_taken_page_costs_nothing(self):
        P = 4
        order = [3, 0, 1, 2]
        # 12, 13 in Page 3 (taken first); Page 0 does not fit beside the gather of 5; Page 1 holds 5: taken.
        taken, gathered, _ = TG.fill(order, 2, [12, 13, 5], P)
        self.assertEqual((taken, gathered), ([3, 1], []))
        taken, gathered, _ = TG.fill(order, 3, [12, 13, 5], P)
        self.assertEqual((taken, gathered), ([3, 0, 1], []))
        # One pick per Page: taking one of them frees no block; the gather stays.
        taken, gathered, _ = TG.fill([0, 1, 2, 3], 2, [5, 9, 13], P)
        self.assertEqual((taken, gathered), ([0], [5, 9, 13]))
        # A Page lower in the order holding enough picks to free a block is taken.
        taken, gathered, _ = TG.fill([0, 1, 2, 3], 2, [8, 9, 10, 11, 1], P)
        self.assertEqual(taken, [0, 2])
        self.assertEqual(gathered, [])

    def test_never_over_the_room_and_every_kept_pick_attended(self):
        rng = random.Random(11)
        for trial in range(300):
            P = rng.choice((1, 4, 16))
            n = rng.randint(0, 40)
            order = rng.sample(range(n), n)
            room = rng.randint(0, n + 2)
            tokens = [t for p in order for t in range(p * P, (p + 1) * P)]
            picks = rng.sample(tokens, rng.randint(0, len(tokens)))
            taken, gathered, kept = TG.fill(order, room, picks, P)
            self.assertLessEqual(len(taken) + TG.blocks(len(gathered), P), room, trial)
            self.assertEqual(kept, picks[:room * P])
            self.assertEqual(len(set(taken)), len(taken))
            self.assertTrue(set(taken) <= set(order))
            self.assertEqual(set(gathered), {t for t in kept if t // P not in set(taken)})
            # In order: no skipped Page would have fitted once the later Pages were decided.
            self.assertEqual([p for p in order if p in set(taken)], taken)

    def test_pad_newest_unattended_not_gathered(self):
        P = 4
        attended = {5, 3}
        gathered = [22, 23, 10]
        got = TG.pad(5, 24, P, attended, gathered)
        self.assertEqual(got, [19, 18, 17, 16, 11])
        got = TG.pad(3, 24, P, {5, 4, 3, 2}, [7])
        self.assertEqual(got, [6, 5, 4])
        with self.assertRaisesRegex(LiveV2Error, 'no unattended token'):
            TG.pad(2, 8, P, {0}, [4, 5, 6])

    def test_plan_whole_blocks_and_record(self):
        P = 4
        order = [4, 1, 2, 3]
        # Page 4 holds 17: taken; Page 1 fits beside one block; Pages 2 and 3 do not (two blocks).
        taken, gather, record = TG.plan(order, 3, [17, 13, 9, 8], 20, P, attended={0})
        self.assertEqual(taken, [4, 1])
        self.assertEqual(gather, dict(tokens=[8, 9, 13, 15], pad=[15]))
        self.assertEqual(record, dict(picks=4, truncated=0, covered=1, gathered=3, pad=1, blocks=1,
                                      plain_retrieved=3, retrieved=2, displaced=1, added=0))
        self.assertTrue(all(t // P not in {0, 4, 1} for t in gather['tokens']))

    def test_room_caps_the_picks(self):
        taken, gather, record = TG.plan([1, 2], 1, list(range(20, 32)) + [0], 48, 4, attended=set())
        self.assertEqual(record['picks'], 4)
        self.assertEqual(record['truncated'], 9)
        self.assertEqual(taken, [])
        self.assertEqual(gather['tokens'], [20, 21, 22, 23])


class PinAndPlanEntryTests(unittest.TestCase):
    def test_pins(self):
        self.assertEqual(TG.pin('markers_v1'), dict(picker='markers_v1',
                                                   markers=['</think>', '<|im_end|>', '<|im_start|>user'],
                                                   quota=0, pad=TG.PAD_RULE, budget=TG.BUDGET_RULE))
        self.assertEqual(TG.pin('separators_v1', quota=512)['quota'], 512)
        for bad in (dict(TG.pin('markers_v1'), markers=[]), dict(TG.pin('markers_v1'), markers=['a', 'a']),
                    dict(TG.pin('markers_v1'), quota=-1), dict(TG.pin('markers_v1'), pad='other'),
                    dict(TG.pin('separators_v1'), quota=0), dict(TG.pin('separators_v1'), markers=['a']),
                    dict(TG.pin('markers_v1'), extra=1), dict(TG.pin('markers_v1'), picker='other')):
            self.assertRaises(LiveV2Error, TG.validate_pin, bad)
        self.assertRaises(LiveV2Error, TG.pin, 'separators_v1', markers=['.'])
        self.assertRaises(LiveV2Error, TG.pin, 'tokens_v9')

    def test_contract_key_only_when_given(self):
        plain = selection_contract.default_contract(16, vector_format='compact_int8_v1', active_ratio=None,
                                                    active_token_budget=8192)
        self.assertNotIn(TG.KEY, plain)
        on = selection_contract.default_contract(16, vector_format='compact_int8_v1', active_ratio=None,
                                                 active_token_budget=8192, token_gather=TG.pin('markers_v1'))
        self.assertEqual(on[TG.KEY], TG.pin('markers_v1'))
        self.assertEqual({k: v for k, v in on.items() if k != TG.KEY}, plain)
        self.assertNotEqual(selection_contract.digest(on), selection_contract.digest(plain))
        for policy in ('goal_query', 'observation_query'):
            selection_contract.for_policy(dict(on), policy)
        for policy in ('random_fill', 'full', 'ledger_query'):
            self.assertRaises(LiveV2Error, selection_contract.for_policy, dict(on), policy)
        legacy = selection_contract.default_contract(784, token_gather=None)
        self.assertNotIn(TG.KEY, legacy)
        with self.assertRaisesRegex(LiveV2Error, 'P = 16'):
            selection_contract.for_policy(dict(legacy, token_gather=TG.pin('markers_v1')), 'goal_query')
        with self.assertRaises(LiveV2Error):
            selection_contract.validate(dict(on, token_gather=dict(TG.pin('markers_v1'), quota='x')))

    def test_plan_entry_and_dropped(self):
        P = 4
        gather = dict(tokens=[4, 5, 6, 9], pad=[5, 6])
        self.assertEqual(TG.plan_entry(gather, (0, 3), 20, P), dict(tokens=[4, 5, 6, 9], blocks=1))
        self.assertIsNone(TG.plan_entry(None, (0,), 20, P))
        self.assertEqual(TG.dropped(20, (0, 3), P, dict(tokens=[4, 5, 6, 9])), 20 - 8 - 4)
        self.assertEqual(TG.dropped(20, (0, 3), P, None), 12)
        for bad in (dict(tokens=[4, 5, 6], pad=[]), dict(tokens=[0, 5, 6, 9], pad=[]),
                    dict(tokens=[4, 6, 5, 9], pad=[]), dict(tokens=[4, 5, 6, 20], pad=[]),
                    dict(tokens=[4, 5, 6, 9], pad=[7]), dict(tokens=[4, 5, 6, 9]), dict(tokens=[], pad=[])):
            self.assertRaises(LiveV2Error, TG.plan_entry, bad, (0, 3), 20, P)

    def test_sources_device_tail_host(self):
        resident = {1: {0: (11,)}}
        tail = (3, {0: (13,)})
        catalog = {0: 'p0', 1: 'p1', 2: 'p2'}
        got = TG.sources([1, 5, 9, 14], 4, resident, tail, catalog)
        self.assertEqual(got, [('host', 'p0', 1), ('device', {0: (11,)}, 1), ('host', 'p2', 1),
                               ('device', {0: (13,)}, 2)])
        self.assertRaises(LiveV2Error, TG.sources, [17], 4, resident, tail, catalog)

    def test_cost_model(self):
        self.assertEqual(TG.cost(1), 64 * 1024)
        self.assertEqual(TG.cost(16), 1 << 20)



class LauncherTests(unittest.TestCase):
    """serve.py --token-gather and the startup config: the pin, and every refusal (launcher and config)."""
    maxDiff = None
    A = dict(instance_id='a', problem_statement='fix a')

    def setUp(self):
        import tempfile
        from contextlib import redirect_stdout
        import io
        sys.path.insert(0, str(HERE))
        sys.path.insert(0, str(HERE / 'tests'))
        import install
        import test_installer as INSTALL
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        with redirect_stdout(io.StringIO()):
            package, site, model, self.release = INSTALL.fixture(self.root)
            install.install(site, model, self.release, package_root=package, apply=True)
        self.n = 0

    @staticmethod
    def outcome(fn):
        from contextlib import redirect_stdout
        import io
        try:
            with redirect_stdout(io.StringIO()):
                return 'ok', fn()
        except Exception as error:
            return type(error).__name__, str(error)

    def prepare(self, policy, selection=None, task=True, **kwargs):
        import serve
        self.n += 1
        return self.outcome(lambda: serve.prepare(self.release, dict(self.A) if task else None, policy,
                                                  self.root / ('out%d' % self.n), 8231, selection, **kwargs))

    def load(self, config):
        import os
        from unittest.mock import patch
        import serve
        from remote_ra_http import config as http_config
        path = self.root / 'startup.json'
        path.write_bytes(serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
            return self.outcome(http_config.load)

    def main(self, *extra):
        import io
        import json
        from contextlib import redirect_stdout
        from unittest.mock import patch
        import serve
        task = self.root / 'task.json'
        task.write_text(json.dumps(self.A))
        self.n += 1
        argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--output',
                str(self.root / ('dry%d' % self.n)), '--port', '8231', *extra]
        out, err = io.StringIO(), io.StringIO()
        try:
            with patch.object(sys, 'argv', argv), redirect_stdout(out), patch.object(sys, 'stderr', err):
                serve.main()
        except SystemExit as error:
            return 'exit', error.code, err.getvalue()
        return 'ok', out.getvalue()

    def test_the_pin_in_the_contract_and_the_config(self):
        for policy in ('observation_query', 'goal_query'):
            selection = dict(token_gather='markers')
            if policy == 'goal_query':
                selection['no_goal_rewriter'] = True
            kind, value = self.prepare(policy, selection, page_tokens=16, enable_thinking=True)
            self.assertEqual(kind, 'ok', value)
            config = value[1]
            self.assertEqual(config['selection_contract'][TG.KEY], TG.pin('markers_v1'))
            self.assertEqual(self.load(config)[0], 'ok')
        kind, value = self.prepare('observation_query', dict(token_gather='markers', token_gather_quota=64,
                                                             token_gather_markers=['<|im_end|>', '</think>']),
                                   page_tokens=16)
        self.assertEqual(kind, 'ok', value)
        self.assertEqual(value[1]['selection_contract'][TG.KEY],
                         TG.pin('markers_v1', markers=['<|im_end|>', '</think>'], quota=64))
        kind, value = self.prepare('observation_query', dict(token_gather='separators'), page_tokens=16)
        self.assertEqual(value[1]['selection_contract'][TG.KEY], TG.pin('separators_v1'))
        kind, plain = self.prepare('observation_query', None, page_tokens=16)
        self.assertNotIn(TG.KEY, plain[1]['selection_contract'])

    def test_launcher_refusals(self):
        gather = dict(token_gather='markers')
        cases = [
            ('random_fill', gather, dict(page_tokens=16), '--token-gather serves goal_query and observation_query'),
            ('ledger_query', gather, dict(page_tokens=16), '--token-gather serves goal_query and observation_query'),
            ('full', gather, dict(page_tokens=16), 'selection options apply'),
            ('observation_query', gather, {}, '--token-gather needs the decoupled KV layout at --page-tokens 16'),
            ('observation_query', gather, dict(page_tokens=64), '--token-gather needs the decoupled KV layout'),
            ('observation_query', gather, dict(page_tokens=16, release_idle_pages=True),
             '--token-gather excludes --release-idle-pages'),
            ('observation_query', gather, dict(page_tokens=16, triton_attn=True), '--triton-attn is refused'),
            ('observation_query', dict(token_gather='separators', token_gather_markers=['.']),
             dict(page_tokens=16), '--token-gather-marker applies to --token-gather markers only'),
            ('observation_query', dict(token_gather_quota=5), dict(page_tokens=16),
             '--token-gather-marker and --token-gather-quota need --token-gather'),
            ('observation_query', dict(token_gather='separators', token_gather_quota=0), dict(page_tokens=16),
             '--token-gather: separators_v1 pin'),
            ('observation_query', gather, dict(page_tokens=16, protocol='kv-session', task=False, max_running=2),
             ''),
        ]
        for policy, selection, kwargs, message in cases:
            with self.subTest(policy=policy, selection=selection, kwargs=kwargs):
                kind, value = self.prepare(policy, selection, **kwargs)
                self.assertEqual(kind, 'ValueError', value)
                self.assertIn(message, value)

    def test_startup_config_refusals(self):
        kind, value = self.prepare('observation_query', dict(token_gather='markers'), page_tokens=16)
        config = value[1]
        self.assertEqual(self.load(config)[0], 'ok')
        refused = [
            (dict(config, release_idle_pages=True), 'token gather excludes --release-idle-pages'),
            (dict(config, attention_backend='TRITON_ATTN'), ''),
            (dict(config, protocol='kv_session_v1'), ''),
            (dict(config, model_profile='gemma4-31b-it'), ''),
        ]
        for value, message in refused:
            kind, text = self.load(value)
            self.assertEqual(kind, 'RuntimeError', text)
            self.assertIn(message, text)
        self.assertEqual(TG.check_config(dict(config, selection_contract=None)), None)
        with self.assertRaisesRegex(LiveV2Error, 'protocol mswea_xml_v1'):
            TG.check_config(dict(config, protocol='chat_text_v1'))
        with self.assertRaisesRegex(LiveV2Error, 'Gemma 4 is refused'):
            TG.check_config(dict(config, model_profile='gemma4-31b-it'))

    def test_dry_run_and_cli_refusal(self):
        result = self.main('--policy', 'observation_query', '--page-tokens', '16', '--token-gather', 'markers',
                           '--token-gather-marker', '</think>', '--token-gather-quota', '64')
        self.assertIn('SERVICE_DRY_RUN', result[1], result)
        self.assertEqual(self.outcome(lambda: self.main('--policy', 'random_fill', '--page-tokens', '16',
                                                        '--token-gather', 'separators'))[0], 'ValueError')

    def test_marker_plugins_refused(self):
        """integrate-1114: option B does not combine with option A. --token-gather serves the built-in goal_query and
        observation_query only, so with the marker plug-ins (goal_query_markers, observation_query_markers) the
        launcher refuses it (with or without their own options), the CLI too, and so does the selection contract."""
        import policy_registry
        message = '--token-gather serves goal_query and observation_query only'
        for policy in ('goal_query_markers', 'observation_query_markers'):
            base = dict(no_goal_rewriter=True) if policy == 'goal_query_markers' else {}
            kind, value = self.prepare(policy, dict(base), page_tokens=16, enable_thinking=True)
            self.assertEqual(kind, 'ok', value)
            self.assertNotIn(TG.KEY, value[1]['selection_contract'])
            for selection in (dict(base, token_gather='markers'), dict(base, token_gather='separators'),
                              dict(base, token_gather='markers', token_gather_quota=64),
                              dict(base, token_gather='markers', marker_set='ends', marker_cap_percent=25)):
                with self.subTest(policy=policy, selection=selection):
                    kind, value = self.prepare(policy, selection, page_tokens=16, enable_thinking=True)
                    self.assertEqual(kind, 'ValueError', value)
                    self.assertIn(message, value)
            contract = selection_contract.default_contract(
                16, vector_format='compact_int8_v1', active_ratio=None, active_token_budget=8192,
                token_gather=TG.pin('markers_v1'), **{policy: policy_registry.get(policy).pin()})
            with self.assertRaisesRegex(LiveV2Error, 'token gather serves goal_query and observation_query only'):
                selection_contract.for_policy(contract, policy)
            kind, value = self.outcome(lambda: self.main('--policy', policy, '--page-tokens', '16',
                                                         '--no-goal-rewriter', '--token-gather', 'markers'))
            self.assertEqual(kind, 'ValueError', value)
            self.assertIn(message, value)



class CapacityTests(unittest.TestCase):
    """capacity._need_by_pool: the gathered blocks are fresh blocks of phase 2 inside the shared budget."""

    def need(self, gather, resident=40, sealed_pages=600):
        from remote_ra_live_v2 import capacity
        contract = selection_contract.default_contract(16, vector_format='compact_int8_v1', active_ratio=None,
                                                       active_token_budget=8192, token_gather=gather)
        session = types.SimpleNamespace(local_turn=3, resident=tuple(range(resident)), tail=None,
                                        sealed_end=sealed_pages * 16, policy='observation_query', evicted=())
        transcript = types.SimpleNamespace(turn=3, extends=lambda messages: True, initial_length=100)
        phase1 = types.SimpleNamespace(groups=lambda: ((0,), (1, 2, 3), 4), handoff=None, max_tokens=8192)
        core = types.SimpleNamespace(
            _remote_ra_growing_adapter=types.SimpleNamespace(sessions={'s': session}, page_tokens=16),
            _remote_ra_private_v2_endpoint=types.SimpleNamespace(
                phase1=phase1, transcripts={'s': transcript},
                selection=types.SimpleNamespace(contract=contract)))
        tokens = [0] * (sealed_pages * 16 + 900)
        return capacity._need_by_pool(core, 's', tokens, [], 1536)

    def test_gathered_blocks_counted_within_the_budget(self):
        (fa1, _), (plain, _) = self.need(None)
        (_, _), (markers, _) = self.need(TG.pin('markers_v1'))
        (_, _), (quota, _) = self.need(TG.pin('separators_v1', quota=64))
        # The budget (512 Pages) bounds restores; the gather may take blocks only when some resident Page is
        # retained: here 560 sealed Pages are not resident, so every bound is the budget.
        self.assertEqual(plain, markers)
        self.assertEqual(plain, quota)
        (_, _), (plain, _) = self.need(None, resident=300, sealed_pages=600)
        (_, _), (markers, _) = self.need(TG.pin('markers_v1'), resident=300, sealed_pages=600)
        (_, _), (quota, _) = self.need(TG.pin('separators_v1', quota=64), resident=300, sealed_pages=600)
        self.assertEqual(markers - plain, 512 - 300)        # up to the budget: every gathered block fresh
        self.assertEqual(quota - plain, 4)                  # quota 64 tokens: at most 4 blocks
        self.assertEqual(fa1, self.need(TG.pin('markers_v1'))[0][0])


if __name__ == '__main__':
    unittest.main()
