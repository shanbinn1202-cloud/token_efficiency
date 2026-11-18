"""Token gather throughput at the scale of a per-token fill, on one GPU (no model, no server): today's gather path, a
device-direct read of the pinned arena, and the Page restore of the same bytes.

Prototype B of per-token scoring (deliverables/per-token-score-1115 OPTIONS.md section 2.2 and its audit): the
throughput probe only, the part that does not depend on the scoring rule. (The reuse rate of an incremental gather
needs real per-turn selections, prototype A's shadow records; it is not measured here.) A per-token fill gathers up
to about 5,051 tokens per turn (fill room mean 315.7 Pages x 16, job 3603443) where the markers gather 110.6. G1
(job 3603240) measured the gather at 144 and 1,024 tokens only (53.4 / 69.1 ms from DRAM, 49.8 / 59.3 ms of it on
the host): its two-point line gives ~141 ms at 5,051 tokens, about 5x beyond its largest set, with a 50.8-ms intercept
nobody has explained; its UVA_ADVISED was a 10-ms threshold rule, not a measured path.

What runs. The server's KV layout as G1 builds it since 2b2a752 (vLLM 0.24.0's FlashAttentionBackend
get_kv_cache_shape and get_kv_cache_stride_order, raw int8 storage viewed in the backend's physical order and permuted
back as GPUModelRunner._reshape_attention_kv_cache does, the runtime's views_from_runner, token_gather.check_views and
TensorIO.attach_token_gather): Qwen3.6-27B's 16 FullAttention layers of [blocks, 2, 16, 4, 256] bf16, and a pinned
PageArena of the runtime's 256-MiB chunks filled by TensorIO.capture_pages (2,048 DRAM Pages over 8 chunks, 1,024
resident Pages). Per token count (--sizes; the gathered slots are whole 16-token blocks, the pad included: 5,051 tokens
are 316 blocks, 5,056 slots), per spread (scattered: spread evenly over every DRAM Page of the arena, at most 15 per
Page, random offsets; clustered: runs of 12 consecutive tokens of consecutive Pages at a random place) and per source
(dram: arena Pages; resident: device blocks), with CUDA events and the wall time around every call:

(a) today: the runtime's TensorIO.gather_tokens with token_gather.sources; today_parts: the same operations in phase
    order with a device join before each mark (validate, index, alloc, group, host_copy, h2d, device, drain), whose
    result is the runtime's bit for bit (tested). The replica follows the source of gather_tokens whose SHA-256 is
    GATHER_TOKENS_SHA256; the record says whether the loaded source still is that one;
(b) direct: DirectGather (in this file only: a prototype, not runtime code). Every arena chunk is mapped once, at
    setup, as a CUDA tensor over the same pinned host memory (zero-copy, the buffer's __cuda_array_interface__: with
    CUDA's unified addressing, page-locked memory from cudaHostAlloc, which torch's pinned allocator uses, is
    device-accessible at its host address). Per call one Python pass over the sources and one H2D of the int64 index;
    per FullAttention layer one index_select per arena chunk, in which the GPU reads the token rows from host memory
    into a device buffer, and one index_put_ into the fresh blocks. No host copy, no bounce buffer, no H2D of KV bytes;
(c) restore: TensorIO.restore_pages of slots / 16 whole arena Pages, the same bytes (scattered: Pages spread over the
    arena; clustered: the Pages of the run): the reference throughput.

Checks. Every variant's fresh blocks are bit-identical to the truth (bf16 compared as int16; device rows read from the
cache, arena rows from the arena chunk, captured rows from the capture payload, all independently of the variants),
no block outside the targets changed (the whole cache against its snapshot), and today's and restore's H2D byte
counters equal the host rows x 4 KiB x layers. The copy sets run in both stride orders (HND, the server's, and NHD): a
mixed set (resident Pages, arena Pages across chunks, one-Page captures) and the --target sets. An attention check as
G1's, for today and for direct: vLLM's flash_attn_varlen_func over (a) whole restored Pages, (b) the same keys with 12
of the Pages gathered shuffled after the others, (c) the gathered blocks first; pass: max |a - b| and |a - c| <= 2e-3,
and 15 zero slots in a gathered block change the output by more than 1e-3 and 10 x that. --one-thread-sizes: today and
today_parts again with torch.set_num_threads(1) (whether G1's fixed host part is the OpenMP pool).

Record (--out, a new file; schema token-gather-bench-v1, validate_record): per row the best, median, p90 (nearest
rank) and mean of the CUDA-event times and the wall medians, the bytes moved and GB/s, the parts, exactness; the G1
estimates beside the measurements; the decision at --target from DRAM (decide): TODAY_PATH_OK, DEVICE_DIRECT_ADVISED,
NO_CLEAR_WINNER or NO_DECISION. Verdict TOKEN_GATHER_BENCH_PASS (every check passed), _FAIL (a check failed or the setup
was refused), _ERROR (no check failed, but a part could not run: the attention kernel or the device-direct mapping).
The last two stdout lines: TOKEN_GATHER_DECISION=<label> ... and TOKEN_GATHER_BENCH=<verdict> <out>. Exit 0 (PASS,
ERROR), 1 (FAIL), 2 (refused before any work: no CUDA without --cpu-test, the record exists, bad arguments). After
every phase the record so far is <out>.partial (removed when <out> is written), for a process that dies later.

GPU only, never run by the CPU tests: the CUDA alias of pinned memory (map_pinned), vLLM's backend and FlashAttention
kernel, CUDA events, pinned allocations. --cpu-test (tests/test_token_gather_bench.py): CPU tensors
(CpuTensorTestIO, an unpinned arena of 8-Page chunks), 2 layers of 2 heads x 8, the replica of vLLM 0.24.0's layout,
the fp32 reference attention, and the direct path's index math over the host buffers themselves (no mapping).

usage: python -I -B tools/token_gather_bench.py --source <release>/source --out <json> [--site <site-packages>]
       [--sizes 144,512,1024,2048,3072,4096,5051,6144,8192] [--spreads scattered,clustered] [--repeat 20]
       [--warmup 3] [--target 5051] [--one-thread-sizes 144,1024,5051] [--cpu-test]
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import os
import random
import statistics
import sys
import time
import traceback
import types

SCHEMA = 'token-gather-bench-v1'
TOOL = 'tools/token_gather_bench.py'
P = 16                                   # tokens per Page and per FullAttention block (the decoupled layout)
KV_AXIS = 1                              # K / V in vLLM 0.24.0 FlashAttention's logical [blocks, 2, P, heads, size]
GQA = 6                                  # Qwen3.6-27B: 24 query heads over 4 KV heads
ROOM_MEAN_PAGES = 315.7                  # job 3603443 (gath32ti): the fill room's mean, Pages (derived.csv:41)
TARGET_TOKENS = round(ROOM_MEAN_PAGES * P)   # 5,051
CLUSTER_RUN = 12                         # clustered: tokens per Page (a run of consecutive offsets)
SPREADS = ('scattered', 'clustered')
SOURCES = ('dram', 'resident')
VARIANTS = ('today', 'today_parts', 'direct', 'restore')
SERVER_ORDER, OTHER_ORDER = 'HND', 'NHD'  # HND: the server's (the runtime's OffloadingConnector asks for it)
TODAY_PARTS = ('validate', 'index', 'alloc', 'group', 'host_copy', 'h2d', 'device', 'drain')
DIRECT_PARTS = ('index', 'copy')
ATTENTION_TOLERANCE, PAD_MIN_EFFECT = 2e-3, 1e-3
THRESHOLD_MS = 10.0                      # G1's threshold (gather_probe.py UVA_THRESHOLD_MS)
TODAY_OK_FACTOR, DIRECT_FACTOR = 1.25, 0.5
LABELS = ('TODAY_PATH_OK', 'DEVICE_DIRECT_ADVISED', 'NO_CLEAR_WINNER', 'NO_DECISION')
VERDICTS = ('TOKEN_GATHER_BENCH_PASS', 'TOKEN_GATHER_BENCH_FAIL', 'TOKEN_GATHER_BENCH_ERROR')
RULE = ('at --target tokens from DRAM, medians of the CUDA-event times: TODAY_PATH_OK when today <= max(%.2f x '
        'restore, %g ms) (already at the Page restore throughput, or under G1\'s threshold); else '
        'DEVICE_DIRECT_ADVISED when direct <= %.2f x today; else NO_CLEAR_WINNER; NO_DECISION when a row is missing '
        'or a check failed'
        % (TODAY_OK_FACTOR, THRESHOLD_MS, DIRECT_FACTOR))
# G1 (job 3603240, verification\swe32-job3603240-20261114T150531\gather-done-*\gather.json): the DRAM gather medians
# at 144 and 1,024 tokens, and the restore throughput of its two restores (bytes / median), the optimistic end first.
G1 = dict(job=3603240, points=((144, 53.4005), (1024, 69.0626)), restore_gbps=(13.432, 9.583))
# The source of TensorIO.gather_tokens (inspect.getsource) the today_parts replica follows: b388033.
GATHER_TOKENS_SHA256 = '20c3dffb05133a196c0959cd3885d4430f56fecb1a4d5145bc8279a5aedda629'
# Geometry: blocks of the cache (block 0 is vLLM's null block), resident Pages (blocks 1..R), DRAM Pages (arena),
# scratch blocks (the arena is captured from them, a batch at a time), one-Page captures, arena chunk Pages (None:
# the runtime's 256 MiB), attention (Pages restored, Pages gathered), the mixed copy set (slots, Pages); `other`: what
# the NHD world (copy checks only) changes.
GPU = dict(layers=16, heads=4, size=256, blocks=2432, resident=1024, dram=2048, scratch=256, captures=4,
           chunk_slots=None, attention=(48, 12), mixed=(1024, 600), repeat=20, warmup=3,
           sizes=(144, 512, 1024, 2048, 3072, 4096, TARGET_TOKENS, 6144, 8192), target=TARGET_TOKENS,
           one_thread=(144, 1024, TARGET_TOKENS), other=dict(resident=512, dram=1024))
CPU = dict(layers=2, heads=2, size=8, blocks=80, resident=24, dram=48, scratch=8, captures=2, chunk_slots=8,
           attention=(8, 3), mixed=(48, 24), repeat=2, warmup=1, sizes=(16, 48, 144), target=144, one_thread=(48,),
           other={})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def blocks_of(tokens):
    return -(-tokens // P)


def slots_of(tokens):
    """Gathered slots: whole blocks (the runtime pads the last block with real tokens)."""
    return blocks_of(tokens) * P


def limit(geometry):
    """The most tokens a geometry serves: fresh blocks for the targets, at most P - 1 per Page scattered (a wholly
    picked Page is restored, not gathered) and CLUSTER_RUN per Page clustered, over the resident and the DRAM Pages."""
    g = geometry
    pages = min(g['resident'], g['dram'])
    return min(P * (g['blocks'] - 1 - g['resident'] - g['scratch']), (P - 1) * pages, CLUSTER_RUN * pages)


def ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', required=True, help='the release source (holds remote_ra_growing_worker)')
    parser.add_argument('--site', default=None, help='a site-packages appended to sys.path before --source')
    parser.add_argument('--out', required=True, help='the new JSON record')
    parser.add_argument('--sizes', default=None, help='token counts (default: the ladder 144 .. 8,192 with 5,051)')
    parser.add_argument('--spreads', default=','.join(SPREADS))
    parser.add_argument('--repeat', type=int, default=None, help='timed calls per row (GPU 20)')
    parser.add_argument('--warmup', type=int, default=None, help='untimed calls first (GPU 3)')
    parser.add_argument('--target', type=int, default=None, help='the decision\'s token count (5,051)')
    parser.add_argument('--one-thread-sizes', default=None, help='today and today_parts again at one torch thread')
    parser.add_argument('--cpu-test', action='store_true', help='the CPU tests: CPU tensors, small geometry')
    args = parser.parse_args(argv)
    g = CPU if args.cpu_test else GPU
    try:
        args.sizes = sorted(set(ints(args.sizes))) if args.sizes is not None else list(g['sizes'])
        args.one_thread_sizes = (sorted(set(ints(args.one_thread_sizes))) if args.one_thread_sizes is not None
                                 else list(g['one_thread']))
    except ValueError:
        parser.error('--sizes and --one-thread-sizes are comma-separated integers')
    args.spreads = [s for s in args.spreads.split(',') if s]
    args.repeat = g['repeat'] if args.repeat is None else args.repeat
    args.warmup = g['warmup'] if args.warmup is None else args.warmup
    args.target = g['target'] if args.target is None else args.target
    top = limit(g)
    if not args.sizes or any(n < 1 or slots_of(n) > top for n in args.sizes):
        parser.error('--sizes: 1 to %d tokens (whole blocks) in this geometry' % top)
    if not args.spreads or len(set(args.spreads)) != len(args.spreads) or not set(args.spreads) <= set(SPREADS):
        parser.error('--spreads: distinct names of %s' % ', '.join(SPREADS))
    if args.target not in args.sizes or slots_of(args.target) > limit(dict(g, **g['other'])):
        parser.error('--target must be one of --sizes (at most %d tokens: the NHD world)'
                     % limit(dict(g, **g['other'])))
    if not set(args.one_thread_sizes) <= set(args.sizes):
        parser.error('--one-thread-sizes must be among --sizes')
    if args.repeat < (1 if args.cpu_test else 3) or args.warmup < 0:
        parser.error('--repeat >= 3 (1 with --cpu-test), --warmup >= 0')
    return args


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(source, site=None):
    """torch and the runtime's modules from the release source (appended to sys.path, after --site: vLLM stays the
    environment's, as in G1); refused when a runtime module was imported from anywhere else."""
    sys.dont_write_bytecode = True
    source = str(Path(source).resolve(strict=True))
    for path in ([str(Path(site).resolve(strict=True))] if site else []) + [source]:
        if path not in sys.path:
            sys.path.append(path)
    import torch
    from remote_ra_growing_worker import tensors
    from remote_ra_live_v2 import token_gather
    for module in (tensors, token_gather):
        require(str(Path(module.__file__).resolve()).startswith(source),
                '%s imported from %s, not the release source %s' % (module.__name__, module.__file__, source))
    return torch, tensors, token_gather


def gather_tokens_source(tensors):
    """SHA-256 of TensorIO.gather_tokens' source as loaded, and whether the today_parts replica follows it."""
    digest = hashlib.sha256(inspect.getsource(tensors.TensorIO.gather_tokens).encode('utf-8')).hexdigest()
    return digest, digest == GATHER_TOKENS_SHA256


# ---- the server's KV views -----------------------------------------------------------------------------------------

def replica(layout):
    """vLLM 0.24.0's FlashAttentionBackend.get_kv_cache_shape and get_kv_cache_stride_order (flash_attn.py, a stock
    file of the runtime's pinned source closure) with get_kv_cache_layout() answering `layout`: the --cpu-test
    stand-in, and on the GPU the cross-check of the loaded backend (the tests compare it with the pinned file)."""
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
        if block_size % 16 != 0:
            raise ValueError('Block size must be a multiple of 16.')
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return {'NHD': (1, 0, 2, 3, 4, 5), 'HND': (1, 4, 0, 2, 3, 5)}[layout]
        return {'NHD': (0, 1, 2, 3, 4), 'HND': (0, 1, 3, 2, 4)}[layout]
    return type('FlashAttention024Replica', (), dict(get_kv_cache_shape=staticmethod(get_kv_cache_shape),
                                                     get_kv_cache_stride_order=staticmethod(get_kv_cache_stride_order)))


def backend(layout, cpu_test, heads, size):
    """(backend class, its name, vLLM's answers equal the replica's (None: not compared), error or None): GPU only:
    vLLM's FlashAttentionBackend with set_kv_cache_layout(layout); else (--cpu-test, or vLLM not importable: a
    finding) the replica."""
    stand_in = replica(layout)
    if cpu_test:
        return stand_in, 'replica (cpu test)', None, None
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        from vllm.v1.attention.backends.utils import set_kv_cache_layout
        set_kv_cache_layout(layout)
        equal = all(tuple(FlashAttentionBackend.get_kv_cache_shape(n, b, heads, size)) ==
                    stand_in.get_kv_cache_shape(n, b, heads, size) for n in (17, 19, 3072) for b in (16, 32)) \
            and tuple(FlashAttentionBackend.get_kv_cache_stride_order()) == stand_in.get_kv_cache_stride_order()
        return FlashAttentionBackend, 'vllm.v1.attention.backends.flash_attn.FlashAttentionBackend', equal, None
    except Exception as exc:
        return stand_in, 'replica (vLLM not importable)', None, '%s: %s' % (type(exc).__name__, exc)


class FullAttentionSpec(types.SimpleNamespace):
    """The fields of vLLM 0.24.0's FullAttentionSpec that views_from_runner reads (it checks the class name)."""


def bits(tensor):
    """bf16 as int16 (bit-exact comparison)."""
    import torch
    return tensor.contiguous().view(torch.int16)


class World:
    """The cache, the io and a synthetic session of one stride order.

    Pages 0..R-1 resident (blocks 1..R); Pages R..R+D-1 in the arena, captured by TensorIO.capture_pages from the
    scratch blocks a batch at a time (slot = Page - R); Pages R+D.. one-Page ``capture`` payloads; then the scratch
    blocks overwritten (a host row read from the device would differ). Fresh blocks after the scratch blocks.

    ``fa``: the server's views (vLLM's layout in ``layout``'s stride order, what the io gathers in); ``kv``: the same
    storage as [2, blocks, P, heads, size] views, for the bench's own reads and writes."""

    def __init__(self, torch, tensors, TG, device, geometry, layout, cpu_test, seed):
        g = self.g = dict(geometry)
        self.torch, self.tensors, self.TG, self.device = torch, tensors, TG, device
        self.layers, self.heads, self.size, self.blocks = g['layers'], g['heads'], g['size'], g['blocks']
        self.row_bytes = 2 * self.heads * self.size * 2          # K and V of one token in one layer, bf16
        self.token_bytes = self.row_bytes * self.layers          # Qwen3.6-27B: 64 KiB (token_gather.cost)
        flash, name, equal, error = backend(layout, cpu_test, self.heads, self.size)
        shape = tuple(int(n) for n in flash.get_kv_cache_shape(self.blocks, P, self.heads, self.size,
                                                                cache_dtype_str='auto'))
        order = tuple(int(i) for i in flash.get_kv_cache_stride_order())
        physical = tuple(shape[i] for i in order)
        inverse = [order.index(i) for i in range(len(order))]
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.names = ['fa%d' % i for i in range(self.layers)]
        caches = {}
        for layer in self.names:
            # GPUModelRunner._reshape_attention_kv_cache, unpadded: raw int8 bytes -> bf16 -> the backend's physical
            # shape (its stride order) -> permuted back to the logical shape
            raw = torch.zeros(math.prod(shape) * 2, dtype=torch.int8, device=device)
            cache = raw.view(torch.bfloat16).view(physical).permute(*inverse)
            cache.copy_(self.random(shape))
            caches[layer] = cache
        spec = FullAttentionSpec(block_size=P, storage_block_size=P, num_kv_heads=self.heads, head_size=self.size,
                                 page_size_bytes=P * self.row_bytes)
        groups = [types.SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=list(self.names),
                                        backend=flash)]
        runner = types.SimpleNamespace(
            kv_cache_config=types.SimpleNamespace(kv_cache_tensors=[types.SimpleNamespace(
                shared_by=[n], block_stride=0, offset=0) for n in self.names]),
            _kv_cache_spec_attn_group_iterator=lambda: iter(groups), runner_only_attn_layers=set(),
            cache_config=types.SimpleNamespace(cache_dtype='auto'))
        self.cuda = device.type == 'cuda'
        self.io = (tensors.TensorIO if self.cuda else tensors.CpuTensorTestIO)(
            tensors.views_from_runner(runner, caches, [P]), decoupled=True)
        # The worker's startup path (worker_composition.attach_loaded_worker): the check, then the axes attached.
        axes = TG.check_views(self.io.views, P, TG.runner_layouts(runner, P))
        self.io.attach_token_gather(axes)
        self.fa = [caches[n] for n in self.names]
        block_axis, token_axis = axes[(self.names[0], 0)]
        self.kv = [t.movedim((block_axis, KV_AXIS, token_axis), (1, 0, 2)) for t in self.fa]
        first = self.fa[0]
        self.layout = dict(backend=name, vllm_equals_replica=equal, backend_error=error, layout=layout,
                           shape=list(shape), stride_order=list(order),
                           stride_order_name=TG.STRIDE_ORDERS.get(TG.stride_order(first, block_axis)),
                           strides=list(first.stride()), view_block_axis=self.io.views[0].axes[0],
                           axes_attached=[block_axis, token_axis], check='token_gather.check_views PASS')
        R, D, S, C = g['resident'], g['dram'], g['scratch'], g['captures']
        chunk_bytes = None if g['chunk_slots'] is None else g['chunk_slots'] * P * self.token_bytes
        self.io.attach_arena(tensors.PageArena(self.io.views, max_pages=D + 8, pin=self.cuda, chunk_bytes=chunk_bytes))
        self.arena = self.io.arena
        self.chunk_slots = self.arena.slots.chunk_slots
        self.resident = {p: {0: (p + 1,)} for p in range(R)}
        self.scratch = list(range(R + 1, R + 1 + S))
        self.catalog, mismatched, page = {}, 0, R
        while page < R + D:
            batch = self.scratch[:min(S, R + D - page)]
            self.randomize(batch)
            payloads = self.io.capture_pages([{0: (b,)} for b in batch])
            mismatched += self.capture_mismatches(batch, payloads)
            self.catalog.update(zip(range(page, page + len(batch)), payloads))
            page += len(batch)
        for j in range(C):  # one-Page captures (as the phase-1 tail capture): host rows outside the arena
            self.randomize(self.scratch[:1])
            self.catalog[R + D + j] = self.io.capture({0: (self.scratch[0],)}, 'attention')
        self.randomize(self.scratch)
        self.capture_check = dict(pages=D, mismatched_pages=mismatched,
                                  status='PASS' if mismatched == 0 else 'FAIL')
        self.resident_pages = list(range(R))
        self.dram_pages = list(range(R, R + D))
        self.capture_pages = list(range(R + D, R + D + C))
        self.free = R + S + 1  # the next fresh block

    def random(self, shape):
        torch = self.torch
        return torch.randn(shape, generator=self.generator, device=self.device, dtype=torch.float32).to(torch.bfloat16)

    def randomize(self, blocks):
        index = self.torch.tensor(blocks, dtype=self.torch.long, device=self.device)
        for t in self.kv:
            t[:, index] = self.random((2, len(blocks)) + tuple(t.shape[2:]))

    def capture_mismatches(self, blocks, payloads):
        """Arena Pages whose bytes differ from the scratch blocks they were captured from (the capture is exact)."""
        torch, bad = self.torch, set()
        index = torch.tensor(blocks, dtype=torch.long, device=self.device)
        by_chunk = {}
        for k, payload in enumerate(payloads):
            chunk, slot = divmod(payload.slot, self.chunk_slots)
            by_chunk.setdefault(chunk, []).append((k, slot))
        for layer, name in enumerate(self.names):
            device_rows = bits(self.kv[layer][:, index].movedim(1, 0)).cpu()          # [n, 2, P, heads, size]
            for chunk, items in by_chunk.items():
                ks = torch.tensor([k for k, _ in items])
                host = bits(self.arena.host(chunk, name, 0)[torch.tensor([s for _, s in items])])
                differ = (host != device_rows[ks]).reshape(len(items), -1).any(dim=1)
                bad |= {items[int(i)][0] for i in torch.nonzero(differ).flatten().tolist()}
        return len(bad)

    def fresh(self, n):
        out = list(range(self.free, self.free + n))
        require(not out or out[-1] < self.blocks, 'bench geometry: %d blocks, %d needed' % (
            self.blocks, (out[-1] + 1) if out else 0))
        self.free += n
        return out

    def rewind(self, mark):
        self.free = mark

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize(self.device)


# ---- token sets ------------------------------------------------------------------------------------------------------

def scattered(count, pages, rng, cap=P - 1):
    """`count` distinct positions spread evenly over `pages`: every Page gets floor or ceil of count / len(pages) of
    them (count <= len(pages): `count` Pages one each), never more than `cap` (a wholly picked Page is restored by the
    fill, never gathered), at random offsets."""
    pages = list(pages)
    n = len(pages)
    require(1 <= count <= n * cap, 'scattered: %d tokens over %d Pages, at most %d per Page' % (count, n, cap))
    if count <= n:
        per = dict.fromkeys(rng.sample(pages, count), 1)
    else:
        base, extra = divmod(count, n)
        lucky = set(rng.sample(pages, extra))
        per = {p: base + (p in lucky) for p in pages}
    out = []
    for page in sorted(per):
        out += [page * P + o for o in rng.sample(range(P), per[page])]
    return sorted(out)


def clustered(count, pages, rng, run=CLUSTER_RUN):
    """`count` positions in runs of `run` consecutive offsets on consecutive Pages of `pages` (ascending, consecutive),
    from a random place; the last Page holds the rest."""
    pages = list(pages)
    k = -(-count // run)
    require(1 <= count and 1 <= run < P and k <= len(pages) and pages == list(range(pages[0], pages[0] + len(pages))),
            'clustered: %d tokens in runs of %d over %d consecutive Pages' % (count, run, len(pages)))
    start = rng.randrange(len(pages) - k + 1)
    out = []
    for i, page in enumerate(pages[start:start + k]):
        length = run if i < k - 1 else count - run * (k - 1)
        first = rng.randrange(P - length + 1)
        out += [page * P + first + j for j in range(length)]
    return sorted(out)


def positions(world, count, spread, source, rng):
    """The gathered slots' token positions: slots_of(count) of them (the pad is more tokens of the same spread)."""
    pages = world.dram_pages if source == 'dram' else world.resident_pages
    return (scattered if spread == 'scattered' else clustered)(slots_of(count), pages, rng)


def mixed(world, count, pages, device_share, rng):
    """G1's mixed set: `count` distinct positions over exactly `pages` distinct Pages, about `device_share` of the
    Pages resident, the others in DRAM: arena Pages across chunks and every one-Page capture."""
    n_dev = int(round(pages * device_share))
    captures = list(world.capture_pages)
    chosen = rng.sample(world.resident_pages, n_dev) + captures + rng.sample(world.dram_pages,
                                                                              pages - n_dev - len(captures))
    picked = [(p, rng.randrange(P)) for p in chosen]
    first = set(picked)
    rest = [(p, o) for p in chosen for o in range(P) if (p, o) not in first]
    picked += rng.sample(rest, count - pages)
    return sorted(p * P + o for p, o in picked)


def restore_pages_for(world, tokens, spread):
    """slots / 16 arena Pages (the gather's bytes): scattered, spread evenly over the arena; clustered, the Pages of
    the run (consecutive slots)."""
    n = len(tokens) // P
    if spread == 'clustered':
        held = sorted({t // P for t in tokens})
        return held[:n]
    pages = world.dram_pages
    step = len(pages) / n
    return [pages[int(i * step)] for i in range(n)]


def describe(world, tokens, found):
    host = [s for s in found if s[0] == 'host']
    PagePayload = world.tensors.PagePayload
    return dict(slots=len(tokens), blocks=len(tokens) // P, pages=len({t // P for t in tokens}),
                host_rows=len(host), device_rows=len(found) - len(host),
                capture_rows=sum(1 for s in host if type(s[1]) is not PagePayload),
                chunks=sorted({s[1].slot // world.chunk_slots for s in host if type(s[1]) is PagePayload}))


# ---- the truth and the comparison --------------------------------------------------------------------------------

def reference(world, sources):
    """Per FA layer, on the CPU, [slots, 2, heads, size]: the truth of every slot, read independently of the
    variants: a device row from the cache's [2, blocks, P, ...] view, an arena row from its chunk's host buffer
    ([slots, 2, P, heads, size]), a captured row from its payload."""
    torch, PagePayload = world.torch, world.tensors.PagePayload
    n = len(sources)
    device = [k for k, s in enumerate(sources) if s[0] == 'device']
    arena, captures = {}, []
    for k, s in enumerate(sources):
        if s[0] != 'host':
            continue
        if type(s[1]) is PagePayload:
            chunk, slot = divmod(s[1].slot, world.chunk_slots)
            arena.setdefault(chunk, []).append((k, slot, s[2]))
        else:
            captures.append(k)
    out = []
    for layer, name in enumerate(world.names):
        ref = torch.empty((n, 2, world.heads, world.size), dtype=torch.bfloat16)
        if device:
            b = torch.tensor([sources[k][1][0][0] for k in device], dtype=torch.long, device=world.device)
            o = torch.tensor([sources[k][2] for k in device], dtype=torch.long, device=world.device)
            ref[torch.tensor(device)] = world.kv[layer][:, b, o].movedim(1, 0).cpu()
        for chunk, rows in arena.items():
            host = world.arena.host(chunk, name, 0)
            at = torch.tensor([k for k, _, _ in rows])
            slot = torch.tensor([s for _, s, _ in rows])
            offset = torch.tensor([o for _, _, o in rows])
            ref[at] = host[slot, :, offset]          # advanced indices around a slice: [rows, 2, heads, size]
        for k in captures:
            ref[k] = sources[k][1][name][0][0][:, sources[k][2]]
        out.append(ref)
    return out


def snapshot(world):
    return [t.clone() for t in world.kv]


def compare(world, targets, truth, before):
    """(mismatched slots, blocks changed outside the targets): the target blocks' slots against the truth (int16 bit
    patterns), the whole cache against its snapshot."""
    torch = world.torch
    n = len(targets) * P
    b = torch.tensor([targets[k // P] for k in range(n)], dtype=torch.long, device=world.device)
    o = torch.tensor([k % P for k in range(n)], dtype=torch.long, device=world.device)
    mismatched, changed = set(), set()
    for layer, t in enumerate(world.kv):
        got = bits(t[:, b, o].movedim(1, 0)).cpu()
        bad = (got != bits(truth[layer])).reshape(n, -1).any(dim=1)
        mismatched |= {int(k) for k in torch.nonzero(bad).flatten().tolist()}
        diff = (t.view(torch.int16) != before[layer].view(torch.int16)).reshape(2, world.blocks, -1).any(dim=2).any(
            dim=0)
        changed |= {int(x) for x in torch.nonzero(diff).flatten().tolist()}
    return sorted(mismatched), sorted(changed - set(targets))


def checked(world, targets, truth, work):
    """Snapshot the cache, run work(), compare: (work's result, H2D bytes counted during it, mismatched slots,
    blocks changed outside the targets)."""
    before = snapshot(world)
    issued = getattr(world.io, '_http_h2d_issued', 0)
    result = work()
    world.sync()
    moved = getattr(world.io, '_http_h2d_issued', 0) - issued
    mismatched, outside = compare(world, targets, truth, before)
    del before
    return result, moved, mismatched, outside


# ---- timing ------------------------------------------------------------------------------------------------------

def summary(values):
    """n, best, median, p90 (nearest rank: the ceil(0.9 n)-th smallest) and mean of milliseconds, 4 decimals."""
    ordered = sorted(values)
    require(ordered, 'no timing samples')
    return dict(n=len(ordered), best_ms=round(ordered[0], 4), median_ms=round(statistics.median(ordered), 4),
                p90_ms=round(ordered[math.ceil(0.9 * len(ordered)) - 1], 4),
                mean_ms=round(statistics.fmean(ordered), 4))


def timed(world, call, repeat, warmup):
    """`warmup` untimed calls, then `repeat` timed ones: CUDA events around each call on the GPU (the call ends with
    its own device join, as gather_tokens' and restore_pages' drain), the wall time always."""
    torch = world.torch
    for _ in range(warmup):
        call()
    events, walls = [], []
    for _ in range(repeat):
        if world.cuda:  # GPU only
            torch.cuda.synchronize(world.device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        began = time.perf_counter()
        call()
        if world.cuda:
            end.record()
            torch.cuda.synchronize(world.device)
            events.append(start.elapsed_time(end))
        walls.append((time.perf_counter() - began) * 1000)
    out = summary(events or walls)
    wall = summary(walls)
    out.update(wall_median_ms=wall['median_ms'], wall_p90_ms=wall['p90_ms'],
               timer='cuda_events' if events else 'perf_counter')
    return out


class Clock:
    """Parts of one call: mark(name) joins the device, then adds the wall time since the previous mark."""

    def __init__(self, world):
        self.world = world
        self.parts, self.last = {}, None

    def start(self):
        self.world.sync()
        self.parts, self.last = {}, time.perf_counter()

    def mark(self, name):
        self.world.sync()
        now = time.perf_counter()
        self.parts[name] = self.parts.get(name, 0.0) + (now - self.last) * 1000
        self.last = now


def timed_parts(world, call, names, repeat, warmup):
    """call(clock) `warmup` + `repeat` times: per part the summary of its timed calls, and of their sums."""
    clock = Clock(world)
    for _ in range(warmup):
        call(clock)
    samples = {name: [] for name in names}
    totals = []
    for _ in range(repeat):
        call(clock)
        require(set(clock.parts) == set(names), 'parts %s, expected %s' % (sorted(clock.parts), sorted(names)))
        for name in names:
            samples[name].append(clock.parts[name])
        totals.append(sum(clock.parts.values()))
    out = summary(totals)
    out.update(timer='perf_counter, a device join at every mark', parts={n: summary(v) for n, v in samples.items()})
    return out


# ---- (a) today's path in parts ---------------------------------------------------------------------------------------

def today_parts(io, blocks, sources, clock):
    """TensorIO.gather_tokens (the source pinned by GATHER_TOKENS_SHA256: b388033 tensors.py:731-824) in phase order,
    a device join before every mark: validate (745-777), index (781-790: the device index tensors of every FA view),
    alloc (798, 802-803: the device output and the pinned bounce of every view), group (804-817: the arena rows by
    chunk, captured rows copied into the bounce, the CPU index tensors), host_copy (818), h2d (819: the bounce to the
    device), device (800, 819, 821: the device rows, the host rows, the scatter into the fresh blocks), drain (822-824).
    The same operations on the same data, only every view's phase before the next phase: the result is the runtime's
    bit for bit (tests/test_token_gather_bench.py)."""
    from remote_ra_growing_worker.contract import require as need
    from remote_ra_growing_worker.tensors import PagePayload
    clock.start()
    blocks, sources = list(blocks), list(sources)
    need(blocks, "token gather needs at least one gathered block")
    need(io.gather_axes is not None,
         "token gather needs the FullAttention block and token axes (attach_token_gather at worker startup)")
    groups, relevant = io._page_groups(blocks)
    tokens = {t.shape[io.gather_axes[(v.name, ti)][1]] for v in relevant for ti, t in enumerate(v.tensors)}
    need(all(v.ratio == 1 for v in relevant) and len(tokens) == 1,
         "token gather needs one kernel block per Page and one token count in every FullAttention view")
    Pg = tokens.pop()
    need(len(sources) == len(blocks) * Pg and
         all(type(s) is tuple and len(s) == 3 and s[0] in ("device", "host") and
             type(s[2]) is int and 0 <= s[2] < Pg for s in sources),
         "token gather needs one (device|host, source, offset) per gathered slot")
    device = [k for k, s in enumerate(sources) if s[0] == "device"]
    host = [k for k, s in enumerate(sources) if s[0] == "host"]
    targets = {b for ids in groups.values() for b in ids}
    if device:
        seen = []
        for k in device:
            if sources[k][1] not in seen:
                seen.append(sources[k][1])
        dev_groups, _ = io._page_groups(seen)
        need(not {b for ids in dev_groups.values() for b in ids} & targets,
             "token gather source block aliases a gathered block")
    for k in host:
        payload = sources[k][1]
        if type(payload) is PagePayload:
            need(io.arena is not None and payload.arena is io.arena and
                 io.arena.slots.holds(payload.slot, payload.token), "foreign or released Page arena payload")
        else:
            need(type(payload) is dict and set(payload) == {v.name for v in relevant} and
                 all(len(payload[v.name]) == 1 and len(payload[v.name][0]) == len(v.tensors)
                     for v in relevant), "token gather host payload must be one captured Page")
    torch = io.torch
    clock.mark('validate')
    long = dict(dtype=torch.long, device=io.device)
    indices = []
    for v in relevant:
        g = v.group
        entry = dict(dst_b=torch.tensor([blocks[k // Pg][g][0] for k in range(len(sources))], **long),
                     dst_o=torch.tensor([k % Pg for k in range(len(sources))], **long))
        if device:
            entry.update(dev_index=torch.tensor(device, **long),
                         src_b=torch.tensor([sources[k][1][g][0] for k in device], **long),
                         src_o=torch.tensor([sources[k][2] for k in device], **long))
        if host:
            entry['host_index'] = torch.tensor(host, **long)
        indices.append(entry)
    clock.mark('index')
    work = []
    for v, entry in zip(relevant, indices):
        for ti, tensor in enumerate(v.tensors):
            axis, token_axis = io.gather_axes[(v.name, ti)]
            cache = tensor.movedim((axis, token_axis), (0, 1))
            rest = tuple(cache.shape[2:])
            out = torch.empty((len(sources),) + rest, dtype=tensor.dtype, device=io.device)
            bounce = (torch.empty((len(host),) + rest, dtype=tensor.dtype, device="cpu", pin_memory=io._host_pin())
                      if host else None)
            work.append(dict(name=v.name, ti=ti, entry=entry, cache=cache, out=out, bounce=bounce,
                             page_axis=token_axis - (token_axis > axis), chunks=[]))
    clock.mark('alloc')
    for w in work:
        if not host:
            continue
        arena_rows = {}
        for i, k in enumerate(host):
            payload, offset = sources[k][1], sources[k][2]
            if type(payload) is PagePayload:
                chunk, slot = divmod(payload.slot, io.arena.slots.chunk_slots)
                arena_rows.setdefault(chunk, []).append((i, slot, offset))
            else:
                w['bounce'][i].copy_(payload[w['name']][0][w['ti']].select(w['page_axis'], offset))
        for chunk, rows in arena_rows.items():
            w['chunks'].append((chunk, torch.tensor([i for i, _, _ in rows], dtype=torch.long),
                                torch.tensor([s for _, s, _ in rows], dtype=torch.long),
                                torch.tensor([o for _, _, o in rows], dtype=torch.long)))
    clock.mark('group')
    for w in work:
        for chunk, at, slots, offsets in w['chunks']:
            buffer = io.arena.host(chunk, w['name'], w['ti']).movedim(1 + w['page_axis'], 1)
            w['bounce'].index_copy_(0, at, buffer[slots, offsets])
    clock.mark('host_copy')
    issued = 0
    for w in work:
        if host:
            w['staged'] = w['bounce'].to(io.device, non_blocking=True)
            issued += w['bounce'].numel() * w['bounce'].element_size()
    clock.mark('h2d')
    for w in work:
        entry, cache, out = w['entry'], w['cache'], w['out']
        if device:
            out.index_copy_(0, entry['dev_index'], cache[entry['src_b'], entry['src_o']])
        if host:
            out.index_copy_(0, entry['host_index'], w['staged'])
        cache[entry['dst_b'], entry['dst_o']] = out
    clock.mark('device')
    io._http_h2d_issued = getattr(io, "_http_h2d_issued", 0) + issued
    io.drain()
    io._http_h2d_joined = getattr(io, "_http_h2d_issued", 0)
    clock.mark('drain')


# ---- (b) the device-direct read: a prototype -------------------------------------------------------------------------

def interface(host):
    """The __cuda_array_interface__ (version 2) of a contiguous 2-byte CPU tensor's own memory, as int16."""
    require(host.device.type == 'cpu' and host.is_contiguous() and host.element_size() == 2,
            'a device-direct alias needs a contiguous 2-byte CPU tensor')
    return dict(shape=tuple(int(n) for n in host.shape), typestr='<i2', data=(int(host.data_ptr()), False),
                strides=None, version=2)


class CudaAlias:
    """A pinned host tensor's memory as a CUDA array (no copy); it keeps the host tensor alive with every alias."""

    def __init__(self, host):
        self.host = host
        self.__cuda_array_interface__ = interface(host)


def map_pinned(torch, host, device):
    """GPU only: a CUDA tensor over the same page-locked host memory (torch.as_tensor reads the interface; for a
    pointer it resolves the device with cudaPointerGetAttributes and refuses unregistered host memory). Refused when
    the memory is not pinned or the alias is not zero-copy."""
    require(host.is_pinned(), 'device-direct read needs page-locked (pinned) host memory')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    alias = torch.as_tensor(CudaAlias(host), device=device).view(host.dtype)
    require(alias.is_cuda and alias.data_ptr() == host.data_ptr() and tuple(alias.shape) == tuple(host.shape),
            'the CUDA alias of pinned memory is not zero-copy')
    return alias


class DirectGather:
    """Prototype (b), this bench only: the token gather as a device-side read of the pinned arena.

    Setup (once, as a worker would at startup): every arena chunk's host buffer of every FA view mapped by `mapper`
    (GPU: map_pinned; --cpu-test: the host buffer itself) and viewed as [chunk slots x 2 x P, heads x size], so the K
    (or V) row of token o of slot s is row (2 s + kv) P + o. A captured Page is mapped when it is first used.

    A call: the same arguments and refusals as TensorIO.gather_tokens; one Python pass over the sources builds the
    int64 index (the arena rows by chunk, the captured rows, the device rows; the destination of each in that order),
    one H2D copy of it; then per FA view: one index_select per arena chunk (and captured Page) into a device buffer,
    the device rows, one index_put_ into the fresh blocks; one device join. clock: marks index and copy."""

    def __init__(self, world, mapper):
        io = world.io
        self.io, self.torch, self.mapper = io, world.torch, mapper
        self.views = [v for v in io.views if v.kind == 'attention']
        for v in self.views:
            for ti, t in enumerate(v.tensors):
                require(io.gather_axes[(v.name, ti)] == (0, 2) and t.shape[1] == 2 and t.shape[2] == P,
                        'device-direct gather: FlashAttention views [blocks, 2, %d, heads, size] only' % P)
        self.maps, self.captured = {}, {}
        for chunk in range(len(io.arena.chunks)):
            for v in self.views:
                for ti in range(len(v.tensors)):
                    host = io.arena.host(chunk, v.name, ti)
                    self.maps[(chunk, v.name, ti)] = mapper(host).view(host.shape[0] * 2 * P, -1)
        first = self.maps[(0, self.views[0].name, 0)]
        self.info = dict(chunks_mapped=len(self.maps), alias_device=str(first.device),
                         zero_copy=first.data_ptr() == io.arena.host(0, self.views[0].name, 0).data_ptr())

    def alias(self, piece):
        """A captured Page's host piece [2, P, heads, size], mapped once: [2 P, heads x size]."""
        key = id(piece)
        if key not in self.captured:
            self.captured[key] = (piece, self.mapper(piece).view(2 * P, -1))
        return self.captured[key][1]

    def plan(self, blocks, sources):
        io, torch = self.io, self.torch
        from remote_ra_growing_worker.tensors import PagePayload
        blocks, sources = list(blocks), list(sources)
        require(blocks, 'token gather needs at least one gathered block')
        groups, relevant = io._page_groups(blocks)
        require(len(sources) == len(blocks) * P and
                all(type(s) is tuple and len(s) == 3 and s[0] in ('device', 'host') and
                    type(s[2]) is int and 0 <= s[2] < P for s in sources),
                'token gather needs one (device|host, source, offset) per gathered slot')
        targets = {b for ids in groups.values() for b in ids}
        chunks, captures, device = {}, {}, []
        for k, (kind, payload, offset) in enumerate(sources):
            if kind == 'device':
                device.append(k)
            elif type(payload) is PagePayload:
                require(io.arena is not None and payload.arena is io.arena and
                        io.arena.slots.holds(payload.slot, payload.token), 'foreign or released Page arena payload')
                chunk, slot = divmod(payload.slot, io.arena.slots.chunk_slots)
                chunks.setdefault(chunk, []).append((k, slot, offset))
            else:
                require(type(payload) is dict and set(payload) == {v.name for v in relevant} and
                        all(len(payload[v.name]) == 1 and len(payload[v.name][0]) == len(v.tensors)
                            for v in relevant), 'token gather host payload must be one captured Page')
                captures.setdefault(id(payload), (payload, []))[1].append((k, offset))
        if device:  # the distinct source Pages in one pass (gather_tokens compares with a list: quadratic)
            unique = {}
            for k in device:
                src = sources[k][1]
                require(type(src) is dict, 'token gather device source must be {FA group: (block,)}')
                unique.setdefault(tuple(sorted((g, tuple(b)) for g, b in src.items())), src)
            dev_groups, _ = io._page_groups(list(unique.values()))
            require(not {b for ids in dev_groups.values() for b in ids} & targets,
                    'token gather source block aliases a gathered block')
        order, flat, pieces = [], [], []          # out position -> slot k; the flat source rows; (kind, key, rows)
        for chunk in sorted(chunks):
            rows = chunks[chunk]
            pieces.append(('chunk', chunk, 2 * len(rows)))
            for k, slot, offset in rows:
                order.append(k)
                flat += [(2 * slot) * P + offset, (2 * slot + 1) * P + offset]
        for payload, rows in captures.values():
            pieces.append(('capture', payload, 2 * len(rows)))
            for k, offset in rows:
                order.append(k)
                flat += [offset, P + offset]
        order += device
        gids = sorted(groups)
        numbers = list(flat)
        for g in gids:
            numbers += [blocks[k // P][g][0] for k in order]
        numbers += [k % P for k in order]
        for g in gids:
            numbers += [sources[k][1][g][0] for k in device]
        numbers += [sources[k][2] for k in device]
        index = torch.tensor(numbers, dtype=torch.long).to(io.device)
        at, plan = 0, dict(relevant=relevant, pieces=pieces, n=len(order), device=len(device))

        def take(n):
            nonlocal at
            at += n
            return index.narrow(0, at - n, n)
        plan['flat'] = take(len(flat))
        plan['dst_b'] = {g: take(len(order)) for g in gids}
        plan['dst_o'] = take(len(order))
        plan['src_b'] = {g: take(len(device)) for g in gids}
        plan['src_o'] = take(len(device))
        return plan

    def copy(self, plan):
        torch = self.torch
        n = plan['n']
        for v in plan['relevant']:
            g = v.group
            for ti, tensor in enumerate(v.tensors):
                cache = tensor.movedim((0, 2), (0, 1))                    # [blocks, P, 2, heads, size]
                out = torch.empty((n,) + tuple(cache.shape[2:]), dtype=tensor.dtype, device=tensor.device)
                rows = out.view(2 * n, -1)
                at = 0
                for kind, key, count in plan['pieces']:
                    source = self.maps[(key, v.name, ti)] if kind == 'chunk' else self.alias(key[v.name][0][ti])
                    torch.index_select(source, 0, plan['flat'].narrow(0, at, count), out=rows.narrow(0, at, count))
                    at += count
                if plan['device']:
                    out.narrow(0, n - plan['device'], plan['device']).copy_(cache[plan['src_b'][g], plan['src_o']])
                cache.index_put_((plan['dst_b'][g], plan['dst_o']), out)

    def __call__(self, blocks, sources, clock=None):
        if clock is not None:
            clock.start()
        plan = self.plan(blocks, sources)
        if clock is not None:
            clock.mark('index')
        self.copy(plan)
        self.io.drain()
        if clock is not None:
            clock.mark('copy')


# ---- the attention check -------------------------------------------------------------------------------------------

def reference_attention(world, layer, table, length, q):
    """fp32 softmax attention of q ([rows, heads x GQA, size]) over the first `length` slots of the block table (the
    rows at the sequence's end, causal)."""
    torch = world.torch
    k = world.kv[layer][0][table].reshape(-1, world.heads, world.size)[:length].float()
    v = world.kv[layer][1][table].reshape(-1, world.heads, world.size)[:length].float()
    k, v = k.repeat_interleave(GQA, dim=1), v.repeat_interleave(GQA, dim=1)
    scores = torch.einsum('qhd,khd->hqk', q.float(), k) / world.size ** 0.5
    rows = q.shape[0]
    key_at = torch.arange(length, device=q.device)
    last = torch.arange(length - rows, length, device=q.device)
    scores = scores.masked_fill((key_at[None, :] > last[:, None])[None], float('-inf'))
    return torch.einsum('hqk,khd->qhd', scores.softmax(dim=-1), v)


def kernel():
    """GPU only: vLLM's flash_attn_varlen_func and the FA version vLLM picks for head size 256 (fa_utils), else the
    vllm_flash_attn module's function without a version; (function, version, source) or raises."""
    try:
        from vllm.v1.attention.backends import fa_utils
        version = fa_utils.get_flash_attn_version(requires_alibi=False, head_size=256)
        return fa_utils.flash_attn_varlen_func, version, 'vllm.v1.attention.backends.fa_utils'
    except Exception:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func, None, 'vllm.vllm_flash_attn'


def run_attention(world, use_kernel, layer, table, length, q):
    if not use_kernel:
        return reference_attention(world, layer, table, length, q)
    torch = world.torch
    function, version, _ = use_kernel
    k_cache, v_cache = world.fa[layer].unbind(KV_AXIS)  # as FlashAttentionImpl: kv_cache.unbind(1), the views' strides
    kwargs = dict(q=q, k=k_cache, v=v_cache, cu_seqlens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32,
                                                                       device=q.device),
                  max_seqlen_q=q.shape[0], seqused_k=torch.tensor([length], dtype=torch.int32, device=q.device),
                  max_seqlen_k=length, softmax_scale=world.size ** -0.5, causal=True,
                  block_table=torch.tensor([table], dtype=torch.int32, device=q.device))
    if version is not None:
        kwargs['fa_version'] = version
    out = function(**kwargs)
    return out[0] if isinstance(out, tuple) else out


def check_attention(world, gather, use_kernel, rng):
    """(a) `pages` arena Pages restored into fresh blocks, (b) the same keys with `moved` of those Pages' tokens
    gathered by `gather` from DRAM in a shuffled order into fresh blocks read after the other Pages, (c) the gathered
    blocks first; a decode query and an 8-token causal chunk; then 15 zero slots in a gathered block."""
    torch, TG = world.torch, world.TG
    n_pages, n_moved = world.g['attention']
    layer = 0
    pages = rng.sample(world.dram_pages, n_pages)
    moved, kept = pages[:n_moved], pages[n_moved:]
    mark = world.free
    restored = world.fresh(n_pages)
    world.io.restore_pages([({0: (b,)}, world.catalog[p]) for b, p in zip(restored, pages)])
    where = dict(zip(pages, restored))
    run = world.fresh(2)
    world.randomize(run)                       # the run's own keys (20 tokens valid) in its two blocks
    tokens = sorted(p * P + o for p in moved for o in range(P))
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    gathered = world.fresh(n_moved)
    found = TG.sources(tokens, P, world.resident, None, world.catalog)
    order = {t: i for i, t in enumerate(tokens)}
    gather([{0: (b,)} for b in gathered], [found[order[t]] for t in shuffled])
    length = (n_pages + 1) * P + 4
    blocks_a = [where[p] for p in pages] + run
    blocks_b = [where[p] for p in kept] + gathered + run
    blocks_c = gathered + [where[p] for p in kept] + run
    generator = torch.Generator(device=world.device).manual_seed(7)
    out = {}
    for rows in (1, 8):
        q = torch.randn(rows, world.heads * GQA, world.size, generator=generator, device=world.device,
                        dtype=torch.float32).to(torch.bfloat16)
        a = run_attention(world, use_kernel, layer, blocks_a, length, q).float()
        b = run_attention(world, use_kernel, layer, blocks_b, length, q).float()
        c = run_attention(world, use_kernel, layer, blocks_c, length, q).float()
        ref = reference_attention(world, layer, blocks_a, length, q)
        saved = [t[:, gathered[-1]].clone() for t in world.kv]
        for t in world.kv:  # a pad of zeros: 15 slots of the last gathered block
            t[:, gathered[-1], 1:].zero_()
        z = run_attention(world, use_kernel, layer, blocks_b, length, q).float()
        for t, s in zip(world.kv, saved):
            t[:, gathered[-1]].copy_(s)
        ab, ac = float((a - b).abs().max()), float((a - c).abs().max())
        effect = float((a - z).abs().max())
        out['decode' if rows == 1 else 'chunk8'] = dict(
            rows=rows, keys=length, max_abs_a_b=ab, max_abs_a_c=ac, max_abs_zero_pad=effect,
            max_abs_a_ref=float((a - ref).abs().max()), max_abs_b_ref=float((b - ref).abs().max()),
            max_abs_c_ref=float((c - ref).abs().max()),
            status='PASS' if (ab <= ATTENTION_TOLERANCE and ac <= ATTENTION_TOLERANCE
                              and effect > max(PAD_MIN_EFFECT, 10 * max(ab, ac))) else 'FAIL')
    world.rewind(mark)
    out['status'] = 'PASS' if all(r['status'] == 'PASS' for r in out.values()) else 'FAIL'
    return out


# ---- rows ----------------------------------------------------------------------------------------------------------

def gather_row(world, variant, source, spread, tokens, count, threads, call, repeat, warmup, parts=None):
    """One timed row of a gather variant: `call(blocks, sources[, clock])` on fresh blocks, checked after its
    calls (every call writes the same blocks: the last one's bytes are compared)."""
    TG = world.TG
    found = TG.sources(tokens, P, world.resident, None, world.catalog)
    truth = reference(world, found)
    mark = world.free
    targets = world.fresh(len(tokens) // P)
    blocks = [{0: (b,)} for b in targets]
    if parts is None:
        work = lambda: timed(world, lambda: call(blocks, found), repeat, warmup)  # noqa: E731
    else:
        work = lambda: timed_parts(world, lambda clock: call(blocks, found, clock), parts, repeat, warmup)  # noqa
    timing, moved, mismatched, outside = checked(world, targets, truth, work)
    world.rewind(mark)
    started = time.perf_counter()
    for _ in range(max(1, repeat)):
        TG.sources(tokens, P, world.resident, None, world.catalog)
    sources_ms = (time.perf_counter() - started) * 1000 / max(1, repeat)
    row = dict(variant=variant, source=source, spread=spread, tokens=count, pad=len(tokens) - count,
               threads=threads, **describe(world, tokens, found), bytes=len(tokens) * world.token_bytes,
               timing=timing, sources_mean_ms=round(sources_ms, 4), h2d_bytes_per_call=moved / (repeat + warmup))
    # today's path counts its H2D bytes (the host rows); the device-direct read copies none: the GPU reads them
    row['h2d_bytes_expected'] = 0 if variant == 'direct' else row['host_rows'] * world.token_bytes
    if variant == 'direct':
        row['device_read_bytes'] = row['host_rows'] * world.token_bytes
    finish(row, mismatched, outside, row['h2d_bytes_per_call'] == row['h2d_bytes_expected'])
    return row


def restore_row(world, spread, tokens, count, repeat, warmup):
    """TensorIO.restore_pages of slots / 16 arena Pages (restore_pages_for), checked like a gather."""
    pages = restore_pages_for(world, tokens, spread)
    found = [('host', world.catalog[p], o) for p in pages for o in range(P)]
    truth = reference(world, found)
    mark = world.free
    targets = world.fresh(len(pages))
    items = [({0: (b,)}, world.catalog[p]) for b, p in zip(targets, pages)]
    timing, moved, mismatched, outside = checked(
        world, targets, truth, lambda: timed(world, lambda: world.io.restore_pages(items), repeat, warmup))
    world.rewind(mark)
    slots = sorted(world.catalog[p].slot for p in pages)
    row = dict(variant='restore', source='dram', spread=spread, tokens=count, pad=len(tokens) - count,
               threads='default', slots=len(tokens), blocks=len(pages), pages=len(pages), host_rows=len(found),
               device_rows=0, capture_rows=0, chunks=sorted({s // world.chunk_slots for s in slots}),
               consecutive_slots=slots == list(range(slots[0], slots[0] + len(slots))),
               bytes=len(pages) * P * world.token_bytes, timing=timing, h2d_bytes_per_call=moved / (repeat + warmup),
               h2d_bytes_expected=len(pages) * P * world.token_bytes)
    finish(row, mismatched, outside, row['h2d_bytes_per_call'] == row['h2d_bytes_expected'])
    return row


def finish(row, mismatched, outside, bytes_ok):
    median = row['timing']['median_ms']
    row.update(gbps=round(row['bytes'] / (median * 1e6), 3) if median > 0 else None,
               mismatched_slots=len(mismatched), changed_outside_targets=len(outside),
               changed_outside_sample=outside[:8], exact=not mismatched and not outside, bytes_ok=bytes_ok)
    row['status'] = 'PASS' if row['exact'] and bytes_ok else 'FAIL'


def copy_check(world, order, name, variant, tokens, call):
    """One untimed call of a variant on a copy set: exactness, nothing else changed, the H2D bytes."""
    TG = world.TG
    if variant == 'restore':
        pages = sorted({t // P for t in tokens})[:len(tokens) // P]
        found = [('host', world.catalog[p], o) for p in pages for o in range(P)]
        mark = world.free
        targets = world.fresh(len(pages))
        work = lambda: world.io.restore_pages([({0: (b,)}, world.catalog[p]) for b, p in zip(targets, pages)])  # noqa
        expected = len(found) * world.token_bytes
    else:
        found = TG.sources(tokens, P, world.resident, None, world.catalog)
        mark = world.free
        targets = world.fresh(len(tokens) // P)
        work = lambda: call([{0: (b,)} for b in targets], found)  # noqa: E731
        expected = 0 if variant == 'direct' else sum(1 for s in found if s[0] == 'host') * world.token_bytes
    truth = reference(world, found)
    started = time.perf_counter()
    _, moved, mismatched, outside = checked(world, targets, truth, work)
    seconds = time.perf_counter() - started
    world.rewind(mark)
    passed = not mismatched and not outside and moved == expected
    return dict(order=order, name=name, variant=variant, **describe(world, tokens if variant != 'restore' else
                                                                    [p * P + o for p in pages for o in range(P)],
                                                                    found),
                mismatched_slots=len(mismatched), changed_outside_targets=len(outside),
                changed_outside_sample=outside[:8], h2d_bytes=moved, h2d_bytes_expected=expected,
                seconds=round(seconds, 6), status='PASS' if passed else 'FAIL')


def copy_sets(world, args, rng):
    """[(name, tokens)]: G1's mixed set and the --target sets of every spread (DRAM)."""
    count, pages = world.g['mixed']
    sets = [('mixed%d' % count, mixed(world, count, pages, 0.4, rng))]
    for spread in args.spreads:
        sets.append(('target%d-%s' % (args.target, spread), positions(world, args.target, spread, 'dram', rng)))
    return sets


# ---- the decision, the record ----------------------------------------------------------------------------------------

def g1_estimates(tokens):
    """G1's two-point line at `tokens` and the restore throughput's time for the same bytes (OPTIONS.md 2.2)."""
    (x0, y0), (x1, y1) = G1['points']
    slope = (y1 - y0) / (x1 - x0)
    nbytes = tokens * 16 * 2 * 4 * 256 * 2
    return dict(job=G1['job'], points=[list(p) for p in G1['points']], slope_ms_per_token=round(slope, 6),
                intercept_ms=round(y0 - slope * x0, 4), line_ms=round(y1 + (tokens - x1) * slope, 1),
                restore_throughput_ms=[round(nbytes / (g * 1e6), 1) for g in G1['restore_gbps']],
                restore_gbps=list(G1['restore_gbps']), tokens=tokens, bytes=nbytes,
                note='a two-point linear extrapolation beyond 1,024 tokens, and bytes / G1\'s restore GB/s')


def find_row(rows, variant, source, spread, tokens, threads='default'):
    return next((r for r in rows if (r['variant'], r['source'], r['spread'], r['tokens'], r['threads'])
                 == (variant, source, spread, tokens, threads)), None)


def decide(rows, target, spread):
    """The decision at `target` tokens of `spread` from DRAM (RULE)."""
    today, direct, restore = (find_row(rows, v, 'dram', spread, target) for v in ('today', 'direct', 'restore'))
    parts = find_row(rows, 'today_parts', 'dram', spread, target)
    out = dict(tokens=target, slots=slots_of(target), spread=spread, rule=RULE, today_ms=None, direct_ms=None,
               restore_ms=None, today_over_direct=None, restore_over_direct=None, today_over_restore=None,
               dominant_part=None, reason=None)
    if parts is not None:
        part_medians = {n: p['median_ms'] for n, p in parts['timing']['parts'].items()}
        out['dominant_part'] = max(part_medians, key=part_medians.get)
        out['today_parts_ms'] = part_medians
    missing = [n for n, r in (('today', today), ('direct', direct), ('restore', restore)) if r is None]
    if missing:
        out.update(label='NO_DECISION', reason='no %s row' % ', '.join(missing))
    elif not all(r['status'] == 'PASS' for r in (today, direct, restore)):
        out.update(label='NO_DECISION', reason='a check failed')
    else:
        t, d, r = (x['timing']['median_ms'] for x in (today, direct, restore))
        out.update(today_ms=t, direct_ms=d, restore_ms=r, today_over_direct=round(t / d, 2) if d > 0 else None,
                   restore_over_direct=round(r / d, 2) if d > 0 else None,
                   today_over_restore=round(t / r, 2) if r > 0 else None)
        if t <= max(TODAY_OK_FACTOR * r, THRESHOLD_MS):
            out['label'] = 'TODAY_PATH_OK'
        elif d <= DIRECT_FACTOR * t:
            out['label'] = 'DEVICE_DIRECT_ADVISED'
        else:
            out['label'] = 'NO_CLEAR_WINNER'
    out['line'] = decision_line(out)
    return out


def decision_line(d):
    def ms(x):
        return 'n/a' if x is None else '%.2f' % x
    return ('TOKEN_GATHER_DECISION=%s tokens=%d slots=%d spread=%s today_ms=%s direct_ms=%s restore_ms=%s '
            'today_over_direct=%s dominant_part=%s%s' % (
                d['label'], d['tokens'], d['slots'], d['spread'], ms(d['today_ms']), ms(d['direct_ms']),
                ms(d['restore_ms']), 'n/a' if d['today_over_direct'] is None else '%.2fx' % d['today_over_direct'],
                d['dominant_part'] or 'n/a', '' if not d['reason'] else ' reason=%s' % d['reason'].replace(' ', '_')))


def verdict_of(record):
    """FAIL: the setup failed or a check failed; ERROR: no check failed but a part could not run; else PASS."""
    errors = record.get('errors') or {}
    failed = (errors.get('bench') is not None
              or (record.get('capture_check') or {}).get('status') == 'FAIL'
              or (record.get('copy') or {}).get('status') == 'FAIL'
              or any(r.get('status') == 'FAIL' for r in record.get('rows') or [])
              or any((a or {}).get('status') == 'FAIL' for a in (record.get('attention') or {}).values()))
    if failed:
        return 'TOKEN_GATHER_BENCH_FAIL'
    if errors or any((a or {}).get('status') != 'PASS' for a in (record.get('attention') or {}).values()) \
            or (record.get('direct') or {}).get('status') != 'MAPPED':
        return 'TOKEN_GATHER_BENCH_ERROR'
    return 'TOKEN_GATHER_BENCH_PASS'


ROW_KEYS = ('variant', 'source', 'spread', 'tokens', 'pad', 'threads', 'slots', 'blocks', 'pages', 'host_rows',
            'device_rows', 'bytes', 'timing', 'gbps', 'exact', 'mismatched_slots', 'changed_outside_targets',
            'h2d_bytes_per_call', 'h2d_bytes_expected', 'status')
TIMING_KEYS = ('n', 'best_ms', 'median_ms', 'p90_ms', 'mean_ms', 'timer')
RECORD_KEYS = ('schema', 'tool', 'source', 'cpu_test', 'python', 'torch', 'device', 'threads', 'geometry', 'layout',
               'phases', 'arena', 'capture_check', 'copy', 'attention', 'direct', 'rows', 'estimates', 'decision',
               'decisions', 'gather_tokens_source', 'verdict', 'errors', 'seconds')


def validate_record(record):
    """The problems of a record against schema token-gather-bench-v1 (an empty list: none)."""
    problems = []
    if type(record) is not dict:
        return ['the record is not an object']
    problems += ['missing key %s' % k for k in RECORD_KEYS if k not in record]
    if record.get('schema') != SCHEMA:
        problems.append('schema %r' % (record.get('schema'),))
    if record.get('verdict') not in VERDICTS:
        problems.append('verdict %r' % (record.get('verdict'),))
    elif problems == [] and record['verdict'] != verdict_of(record):
        problems.append('verdict %s, the record says %s' % (record['verdict'], verdict_of(record)))
    for i, row in enumerate(record.get('rows') or []):
        missing = [k for k in ROW_KEYS if k not in row]
        if missing:
            problems.append('row %d: missing %s' % (i, ', '.join(missing)))
            continue
        if row['variant'] not in VARIANTS or row['source'] not in SOURCES or row['spread'] not in SPREADS:
            problems.append('row %d: variant / source / spread %r' % (i, (row['variant'], row['source'],
                                                                        row['spread'])))
        if [k for k in TIMING_KEYS if k not in row['timing']]:
            problems.append('row %d: timing keys %s' % (i, sorted(row['timing'])))
        if row['slots'] != slots_of(row['tokens']) or row['bytes'] <= 0:
            problems.append('row %d: slots %d for %d tokens' % (i, row['slots'], row['tokens']))
        if row['status'] not in ('PASS', 'FAIL') or (row['status'] == 'PASS') != (row['exact'] and row['bytes_ok']):
            problems.append('row %d: status %r' % (i, row['status']))
    decision = record.get('decision')
    if type(decision) is not dict or decision.get('label') not in LABELS or \
            not str(decision.get('line', '')).startswith('TOKEN_GATHER_DECISION=%s ' % decision.get('label')):
        problems.append('decision %r' % (decision,))
    return problems


# ---- the run -------------------------------------------------------------------------------------------------------

def threads_info(torch):
    out = dict(torch=torch.get_num_threads(), omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
               cpu_count=os.cpu_count())
    if hasattr(os, 'sched_getaffinity'):
        out['affinity'] = len(os.sched_getaffinity(0))
    return out


def cost_rows(world, args, plans, rows, rng):
    """Today's rows (today, today_parts, restore from DRAM; today from resident blocks; today and today_parts at one
    torch thread) for every spread and size; the token sets kept in `plans` for the direct rows."""
    torch, io = world.torch, world.io
    for spread in args.spreads:
        for count in args.sizes:
            for source in SOURCES:
                plans[(spread, count, source)] = positions(world, count, spread, source, rng)
            tokens = plans[(spread, count, 'dram')]
            rows.append(gather_row(world, 'today', 'dram', spread, tokens, count, 'default', io.gather_tokens,
                                   args.repeat, args.warmup))
            rows.append(gather_row(world, 'today_parts', 'dram', spread, tokens, count, 'default',
                                   lambda b, s, clock: today_parts(io, b, s, clock), args.repeat, args.warmup,
                                   parts=TODAY_PARTS))
            rows.append(restore_row(world, spread, tokens, count, args.repeat, args.warmup))
            rows.append(gather_row(world, 'today', 'resident', spread, plans[(spread, count, 'resident')], count,
                                   'default', io.gather_tokens, args.repeat, args.warmup))
    if 'scattered' in args.spreads and args.one_thread_sizes:
        before = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for count in args.one_thread_sizes:
                tokens = plans[('scattered', count, 'dram')]
                rows.append(gather_row(world, 'today', 'dram', 'scattered', tokens, count, 1, io.gather_tokens,
                                       args.repeat, args.warmup))
                rows.append(gather_row(world, 'today_parts', 'dram', 'scattered', tokens, count, 1,
                                       lambda b, s, clock: today_parts(io, b, s, clock), args.repeat, args.warmup,
                                       parts=TODAY_PARTS))
        finally:
            torch.set_num_threads(before)


def direct_rows(world, args, plans, rows, direct):
    for spread in args.spreads:
        for count in args.sizes:
            for source in SOURCES:
                rows.append(gather_row(world, 'direct', source, spread, plans[(spread, count, source)], count,
                                       'default', direct, args.repeat, args.warmup))
                rows[-1]['parts'] = gather_row(world, 'direct', source, spread, plans[(spread, count, source)],
                                               count, 'default', direct, args.repeat, args.warmup,
                                               parts=DIRECT_PARTS)['timing']


def mapper_for(world):
    """GPU: map_pinned; --cpu-test: the host buffer itself (the index math, not the mapping)."""
    if world.cuda:  # GPU only
        return lambda host: map_pinned(world.torch, host, world.device)
    return lambda host: host


def checkpoint(record, out):
    """The record so far as <out>.partial (replaced atomically after every phase; removed when <out> is written), so
    a process that dies in a later phase still leaves what it measured."""
    partial = Path(str(out) + '.partial')
    scratch = Path(str(out) + '.partial.tmp')
    with scratch.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, sort_keys=True, indent=1, default=str)
        stream.write('\n')
    os.replace(scratch, partial)


def bench(args, torch, tensors, TG, record, errors, save=lambda: None):
    device = torch.device('cpu' if args.cpu_test else 'cuda')
    geometry = dict(CPU if args.cpu_test else GPU)
    other_geometry = dict(geometry, **geometry['other'])
    if device.type == 'cuda':  # GPU only
        record['device'] = torch.cuda.get_device_name(device)
    record['geometry'] = dict(geometry, other=other_geometry, page_tokens=P, dtype='bfloat16', gqa=GQA,
                              server_order=SERVER_ORDER, sizes=args.sizes, spreads=args.spreads, repeat=args.repeat,
                              warmup=args.warmup, target=args.target, one_thread_sizes=args.one_thread_sizes,
                              cluster_run=CLUSTER_RUN, room_mean_pages=ROOM_MEAN_PAGES)
    rng = random.Random(1115)
    phases, clock = record['phases'], [time.perf_counter()]

    def phase(name):
        now = time.perf_counter()
        phases[name] = round(now - clock[0], 2)
        clock[0] = now
        save()
    world = World(torch, tensors, TG, device, geometry, SERVER_ORDER, args.cpu_test, seed=1115)
    phase('world_hnd')
    record['geometry'].update(row_bytes=world.row_bytes, token_bytes=world.token_bytes)
    record['layout'][SERVER_ORDER] = world.layout
    if world.layout['backend_error']:
        errors['backend'] = world.layout['backend_error']
    record['arena'] = world.arena.stats()
    record['capture_check'] = {SERVER_ORDER: world.capture_check}
    require(world.capture_check['status'] == 'PASS', 'TensorIO.capture_pages was not exact (%d Pages)'
            % world.capture_check['mismatched_pages'])
    copies = record['copy']['sets']
    sets = copy_sets(world, args, rng)
    for name, tokens in sets:
        copies.append(copy_check(world, SERVER_ORDER, name, 'today', tokens, world.io.gather_tokens))
        if name.startswith('target'):
            copies.append(copy_check(world, SERVER_ORDER, name, 'restore', tokens, None))
    use_kernel = None
    try:
        use_kernel = False if args.cpu_test else kernel()  # GPU: vLLM's FlashAttention; --cpu-test: the reference
        if use_kernel:
            record['kernel'] = dict(source=use_kernel[2], fa_version=use_kernel[1])
        record['attention']['today'] = check_attention(world, world.io.gather_tokens, use_kernel, rng)
    except Exception as exc:
        use_kernel = None
        errors['attention'] = '%s: %s' % (type(exc).__name__, exc)
        record['attention']['today'] = dict(status='ATTENTION_UNAVAILABLE', error=errors['attention'],
                                            traceback=traceback.format_exc()[-3000:])
    phase('checks_today_hnd')
    plans, rows = {}, record['rows']
    cost_rows(world, args, plans, rows, rng)
    phase('rows_today')
    # The other stride order (NHD): the gather indexes by axes and strides, both hold (copy checks only).
    other = World(torch, tensors, TG, device, other_geometry, OTHER_ORDER, args.cpu_test, seed=1116)
    phase('world_nhd')
    record['layout'][OTHER_ORDER] = other.layout
    record['capture_check'][OTHER_ORDER] = other.capture_check
    require(other.capture_check['status'] == 'PASS', 'TensorIO.capture_pages was not exact in %s' % OTHER_ORDER)
    other_sets = copy_sets(other, args, rng)
    for name, tokens in other_sets:
        copies.append(copy_check(other, OTHER_ORDER, name, 'today', tokens, other.io.gather_tokens))
        if name.startswith('target'):
            copies.append(copy_check(other, OTHER_ORDER, name, 'restore', tokens, None))
    phase('checks_today_nhd')
    # (b) last: a fault of the device-direct read cannot cost today's results.
    record['direct'] = dict(status='NOT_RUN', mapping='cuda_array_interface' if world.cuda else 'cpu_stand_in')
    try:
        direct = DirectGather(world, mapper_for(world))
        record['direct'].update(direct.info, status='MAPPED')
        for name, tokens in sets:
            copies.append(copy_check(world, SERVER_ORDER, name, 'direct', tokens, direct))
        if use_kernel is not None:
            record['attention']['direct'] = check_attention(world, direct, use_kernel, rng)
        phase('checks_direct_hnd')
        direct_rows(world, args, plans, rows, direct)
        phase('rows_direct')
        other_direct = DirectGather(other, mapper_for(other))
        for name, tokens in other_sets:
            copies.append(copy_check(other, OTHER_ORDER, name, 'direct', tokens, other_direct))
        record['direct']['captures_mapped'] = len(direct.captured) + len(other_direct.captured)
        phase('checks_direct_nhd')
    except Exception as exc:
        errors['direct'] = '%s: %s' % (type(exc).__name__, exc)
        record['direct'].update(status='DIRECT_UNAVAILABLE', error=errors['direct'],
                                traceback=traceback.format_exc()[-3000:])


def main(argv=None):
    args = parse(argv)
    out = Path(args.out)
    if out.exists():
        print('token_gather_bench: the record %s exists' % out, file=sys.stderr)
        return 2
    started = time.time()
    record = dict(schema=SCHEMA, tool=TOOL, source=str(args.source), site=args.site, cpu_test=args.cpu_test,
                  python=sys.version.split()[0], host=platform.node(), torch=None, device=None, threads=None,
                  geometry=None, layout={}, arena=None, capture_check=None, copy=dict(status=None, sets=[]),
                  attention={}, direct=dict(status='NOT_RUN'), rows=[], phases={}, estimates=g1_estimates(args.target),
                  decision=None, decisions={}, gather_tokens_source=None, verdict=None, errors={}, seconds=None)
    errors = record['errors']
    try:
        torch, tensors, TG = load(args.source, args.site)
    except Exception as exc:
        torch = None
        errors['bench'] = 'load: %s: %s' % (type(exc).__name__, exc)
    if torch is not None and not args.cpu_test and not torch.cuda.is_available():
        print('token_gather_bench: CUDA required (--cpu-test runs the CPU tests)', file=sys.stderr)
        return 2
    if torch is not None:
        digest, follows = gather_tokens_source(tensors)
        record.update(torch=torch.__version__, threads=threads_info(torch), tensors_sha256=sha(tensors.__file__),
                      token_gather_sha256=sha(TG.__file__),
                      gather_tokens_source=dict(sha256=digest, replica_sha256=GATHER_TOKENS_SHA256,
                                                replica_follows=follows))
        if not follows:
            errors['replica'] = 'TensorIO.gather_tokens is not the source today_parts follows (%s)' % digest
        try:
            bench(args, torch, tensors, TG, record, errors, save=lambda: checkpoint(record, out))
        except Exception as exc:  # the views were refused, or the setup or a check could not run: fail closed
            errors['bench'] = '%s: %s' % (type(exc).__name__, exc)
            record['traceback'] = traceback.format_exc()[-4000:]
    sets = record['copy']['sets']
    record['copy']['status'] = 'PASS' if sets and all(c['status'] == 'PASS' for c in sets) else 'FAIL'
    record['decisions'] = {spread: decide(record['rows'], args.target, spread) for spread in args.spreads}
    record['decision'] = record['decisions']['scattered' if 'scattered' in args.spreads else args.spreads[0]]
    record['verdict'] = verdict_of(record)
    record['seconds'] = round(time.time() - started, 1)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, sort_keys=True, indent=1)
        stream.write('\n')
    Path(str(out) + '.partial').unlink(missing_ok=True)
    for spread in args.spreads:
        if record['decisions'][spread] is not record['decision']:
            print(record['decisions'][spread]['line'], flush=True)
    print(record['decision']['line'], flush=True)
    print('TOKEN_GATHER_BENCH=%s %s' % (record['verdict'], out), flush=True)
    return 1 if record['verdict'] == 'TOKEN_GATHER_BENCH_FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
