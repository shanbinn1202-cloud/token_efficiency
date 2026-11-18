"""--gdn-inplace-crossing (gdn-inplace-1116): one Gated-DeltaNet slot per running request. CPU only.

vLLM 0.24's align mode allocates a second Mamba block when a request crosses a multiple of M and frees the
old one a step later, so the decoupled layout reserves K = 2S + 1 GDN slots. With the opt-in flag the
MambaManager moves the running block to the new row index instead (no allocation, no copy of new bytes),
K = S + 1, and the capacity gate counts one running slot per request. Off, nothing changes.

- layout: ``kv_layout`` K = S + 1 and the ``inplace_crossing`` record (absent when off), its refusals, the
  startup parsers (engine, worker, HTTP config), the Page arena's session count, the pools at 25 GiB
  (+2,349 / +3,523 / +4,698 attention Pages at S = 16 / 24 / 32) by the overlay's own planner;
- the real MambaManager (test_kv_pools' loader): a crossing moves the block (nothing allocated or freed,
  the old index null), multi-block chunks, S + 1 slots holding S requests that all cross in one step (off:
  vLLM finds no block, a preemption), an async speculative crossing of a stopped request and its free or
  deferred free, the hand-off's held probe next to the running slot, pool incarnations, session close, the
  refusals (speculative decoding, outside align mode, prefix caching, a cached or shared running block);
  off, every allocation equals a6d3ef7's MambaManager on a random workload;
- the real Scheduler (test_deferred_free_pools' loader) turns it on from the layout record only;
- the capacity gate at S = 16 / 24 / 32, flag on and off, with and without the hand-off, and random turn
  sequences whose actual pool peaks the in-place need bounds;
- end to end over test_decouple_e2e's fakes with an in-place align-mode KV manager and vLLM's append-only
  worker rows: one session (every per-turn check of test_decouple_e2e, no slot move, no relocation, no
  M-aligned capture; the decisions equal the flag-off run's and the evidence differs only in the
  documented fields), three sessions in one batch with K = S + 1 (no CAPACITY_WAIT, no preemption),
  --async-scheduling (per turn equal to the synchronous in-place server and to the flag-off server),
  --gdn-device-handoff (one device copy per continuing turn: the hand-off, no crossing copy), and the third
  decode fast path against the second with --phase1-in-step (and, in the slow pass, without it and with the
  hand-off);
- serve.py: the switch, its dry-run and launch.json record, startup.json and the HTTP config check, and
  the refusals (the shared pool, the Gemma 4 profile).
"""
from contextlib import ExitStack
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
import test_kv_pools as KP  # noqa: E402
import test_stageb_engine as E  # noqa: E402

from remote_ra_live_v2 import capacity  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2.capacity import CapacityWait  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error, startup_layout  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

GiB = 1024 ** 3
BASE = 'a6d3ef7'            # integrate-1114, the branch point: the flag-off reference
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'


def cdiv(a, b):
    return -(-a // b)


# ---- the layout --------------------------------------------------------------------------------------

class LayoutTests(unittest.TestCase):
    def test_slots_and_record(self):
        for S in (1, 2, 16, 24, 32):
            off, on = I0.for_server(16, S, True), I0.for_server(16, S, True, True)
            self.assertEqual((off.mamba_slots, off.inplace_crossing), (2 * S + 1, False))
            self.assertEqual((on.mamba_slots, on.inplace_crossing), (S + 1, True))
            self.assertEqual(off, I0.KvLayout(16, 784, 2 * S + 1, True))
            self.assertEqual(I0.to_additional_config(off),
                             {I0.LAYOUT_KEY: dict(page_tokens=16, mamba_tokens=784, mamba_slots=2 * S + 1)})
            self.assertEqual(I0.to_additional_config(on),
                             {I0.LAYOUT_KEY: dict(page_tokens=16, mamba_tokens=784, mamba_slots=S + 1,
                                                  inplace_crossing=True)})
            for layout in (off, on):
                self.assertEqual(I0.from_vllm_config(NS(additional_config=I0.to_additional_config(layout))), layout)
                self.assertEqual(I0.mamba_pool_blocks(layout, 3), 1 + 3 * layout.mamba_slots)
        # The pool-split validation mode at 784 takes it too; the shared pool and Gemma do not.
        self.assertEqual(I0.for_server(784, 4, True, True), I0.KvLayout(784, 784, 5, True, True))
        for args in ((784, 4, None, True), (784, 4, False, True), (16, 4, True, 1), (16, 4, True, 'on')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                I0.for_server(*args)

    def test_validation_and_refusals(self):
        I0.KvLayout(16, 784, 2, True, True)
        for args in ((16, 784, 1, True, True), (784, 784, 0, False, True), (16, 784, 3, True, 1),
                     (16, 784, 2, True), (16, 784, 2, True, False)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                I0.KvLayout(*args)
        good = dict(page_tokens=16, mamba_tokens=784, mamba_slots=17)
        for value in (False, 1, 'true', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'inplace_crossing must be true'):
                I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: dict(good, inplace_crossing=value)}))
        with self.assertRaises(ValueError):
            I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: dict(good, inplace=True)}))
        window = dict(page_tokens=16, state='sliding', window_tokens=1024, window_slots=4, forward_tokens=8192)
        with self.assertRaises(ValueError):
            I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: dict(window, inplace_crossing=True)}))
        self.assertFalse(hasattr(I0.for_window_server(16, 4, 1024, 8192), 'inplace_crossing'))

    def test_startup_parsers(self):
        on = I0.for_server(16, 16, True, True)
        record = I0.to_additional_config(on)[I0.LAYOUT_KEY]
        self.assertEqual(startup_layout(dict(page_tokens=16, kv_layout=record)), on)
        for bad in (dict(record, inplace_crossing=False), dict(record, extra=1)):
            with self.subTest(bad=bad), self.assertRaises(LiveV2Error):
                startup_layout(dict(page_tokens=16, kv_layout=bad))
        off = I0.for_server(16, 16, True)
        self.assertEqual(startup_layout(dict(page_tokens=16, kv_layout=I0.to_additional_config(off)[I0.LAYOUT_KEY])),
                         off)
        # The worker reads vLLM's additional_config and the startup record; both must agree.
        from remote_ra_live_v2.worker_composition import arena_pages, worker_layout
        runner = NS(vllm_config=NS(additional_config=I0.to_additional_config(on)),
                    kv_cache_config=NS(group_pool=(0, 1, 1, 1)), max_model_len=65536)
        self.assertEqual(worker_layout(runner, dict(page_tokens=16, kv_layout=record)), on)
        with self.assertRaises(LiveV2Error):
            worker_layout(runner, dict(page_tokens=16, kv_layout=I0.to_additional_config(off)[I0.LAYOUT_KEY]))
        # The Page arena holds every session's context: S from K = S + 1 here, from 2S + 1 off.
        for layout in (on, off):
            self.assertEqual(arena_pages(runner, {}, layout), 16 * cdiv(65536, 16) + 16)

    def test_attention_pool_at_25_gib(self):
        """The pools the overlay's planner gives at 25 GiB, P = 16 (and serve.py's own figure)."""
        w = KP.world()
        gained = {}
        for S, before in ((16, 20706), (24, 18357), (32, 16008)):
            pools = {}
            for inplace in (False, True):
                layout = I0.for_server(16, S, True, inplace)
                config = KP.plan(w, layout, 25 * GiB)[1]
                pools[inplace] = tuple(config.num_blocks_by_pool)
                self.assertEqual(pools[inplace], (serve.attention_blocks(25 * GiB, layout),
                                                  1 + 3 * layout.mamba_slots))
            self.assertEqual(pools[False][0], before)
            gained[S] = pools[True][0] - pools[False][0]
            # One GDN slot is 3 x 16 x 3,207,168 B = 146.8 attention Pages of 1 MiB.
            self.assertEqual(gained[S], (S * 153944064 + (25 * GiB - 153944064 * (2 * S + 1) - 51314688)
                                         % (16 * 65536)) // (16 * 65536))
        self.assertEqual(gained, {16: 2349, 24: 3523, 32: 4698})
        self.assertEqual({S: round(n / 1024, 2) for S, n in gained.items()}, {16: 2.29, 24: 3.44, 32: 4.59})


# ---- the real MambaManager -----------------------------------------------------------------------------

# The three MambaManager methods --gdn-inplace-crossing touches, as a6d3ef7 (= vLLM 0.24.0) has them:
# a frozen copy, so the flag-off comparison also runs in a Git archive (checked against the history
# when it is there). sha256 of the text: 352add6e4341a51640841682cf225e706c53f382ebfec8a6aed3d25554fdb26c
BASE_MAMBA_METHODS = '''\
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if (
            len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            # Mamba can't rely on blocks generated by other requests in the current step
            # To put it in the next step, we return num_gpu_blocks + 1 so
            # that kv_cache_manager will think there is no enough blocks to allocate now
            # and don't schedule it in the current step.
            return self.block_pool.num_gpu_blocks + 1
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += (
                    self.kv_cache_spec.block_size * self.num_speculative_blocks
                )
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model

            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            num_new_blocks = (
                num_required_blocks
                - len(new_computed_blocks)
                - len(self.req_to_blocks[request_id])
            )
            if num_new_blocks > 0:
                if request_id in self._allocated_block_reqs:
                    # Old request. Needs at most 1 more blocks as we can reuse the
                    # speculative blocks in previous step.
                    num_new_blocks = 1
                else:
                    # First prefill. Allocate 1 block for running state and the
                    # speculative blocks.
                    num_new_blocks = 1 + self.num_speculative_blocks

            num_evictable_computed_blocks = self._get_num_evictable_blocks(
                new_computed_blocks
            )
            return num_new_blocks + num_evictable_computed_blocks
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += self.block_size * self.num_speculative_blocks
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]
            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            # `num_required_blocks` might be less than `len(req_blocks)` if blocks are
            # over-allocated at last round.
            if num_required_blocks <= len(req_blocks):
                return []
            else:
                prev_block_len = len(req_blocks)
                blocks_allocated = request_id in self._allocated_block_reqs
                # Record the last state block
                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    self.last_state_block_idx[request_id] = prev_block_len - 1

                num_skipped_blocks = (
                    num_required_blocks - self.num_speculative_blocks - 1
                )
                # null blocks
                if prev_block_len < num_skipped_blocks:
                    req_blocks.extend(
                        [
                            self._null_block
                            for _ in range(prev_block_len, num_skipped_blocks)
                        ]
                    )

                if blocks_allocated:
                    # reuse previous speculative blocks in this step
                    for block_idx in range(
                        prev_block_len - self.num_speculative_blocks, prev_block_len
                    ):
                        if block_idx < num_skipped_blocks:
                            req_blocks.append(req_blocks[block_idx])
                            req_blocks[block_idx] = self._null_block
                        else:
                            break
                num_new_blocks = num_required_blocks - len(req_blocks)
                if blocks_allocated:
                    assert num_new_blocks <= 1
                else:
                    assert num_new_blocks <= self.num_speculative_blocks + 1
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)
                self._allocated_block_reqs.add(request_id)
                return req_blocks[prev_block_len:]
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
        if num_cached_blocks_after > num_cached_blocks_before:
            for block in self.req_to_blocks[request.request_id][
                num_cached_blocks_before:num_cached_blocks_after
            ]:
                # Skip null blocks (align-mode skipped states) and blocks that
                # were not cached this step — with sparse retention
                # (reachable_block_mask) the intermediate state snapshots carry
                # no hash and must not be recorded as cached-this-step.
                if block.is_null or block.block_hash is None:
                    continue
                self.cached_blocks_this_step.add(block.block_hash)
'''
BASE_MAMBA_METHODS_SHA256 = '352add6e4341a51640841682cf225e706c53f382ebfec8a6aed3d25554fdb26c'


def world_at(commit):
    """test_kv_pools' world of the vLLM KV modules at ``commit`` (None without the Git history)."""
    with patch.object(KP, 'BASE', commit):
        return KP.world(commit)


def base_mamba_methods(text):
    """The source of the three MambaManager methods of one single_type_kv_cache_manager.py text."""
    import ast
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MambaManager')
    lines = text.splitlines(keepends=True)
    return ''.join(''.join(lines[(n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1:n.end_lineno])
                   for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('get_num_blocks_to_allocate', 'allocate_new_blocks', 'cache_blocks'))


def base_mamba_manager(w):
    """The overlay MambaManager with a6d3ef7's three methods (BASE_MAMBA_METHODS), in the overlay module's
    globals: vLLM 0.24.0's align-mode allocator, without the flag."""
    namespace = dict(vars(sys.modules[w.single.__name__]))
    exec('class BaseMambaManager(MambaManager):\n' + BASE_MAMBA_METHODS, namespace)
    return namespace['BaseMambaManager']


def manager(w, layout, n_fa=400, n_m=None, enable=None):
    config = KP.small_config(w, layout, n_fa, n_m or I0.mamba_pool_blocks(layout, 3))
    kv = KP.kv_manager(w, config)
    if layout.inplace_crossing if enable is None else enable:
        self_check = w.single.enable_inplace_crossing(kv.coordinator.single_type_managers, speculative=False)
        assert self_check == 3
    return kv


def mamba_rows(kv, rid):
    return [list(m.req_to_blocks[rid]) for m in kv.coordinator.single_type_managers[1:]]


def live(rows):
    return [[(i, b.block_id) for i, b in enumerate(row) if not b.is_null] for row in rows]


def alloc(kv, req, new):
    blocks = kv.allocate_slots(req, new, skip_cache=True, has_scheduled_reqs=True)
    if blocks is not None:
        req.num_computed_tokens += new
    return blocks


ON, OFF = I0.for_server(16, 1, True, True), I0.for_server(16, 1, True)


class AllocatorTests(unittest.TestCase):
    def test_enable_refusals(self):
        w = KP.world()
        kv = manager(w, OFF)
        managers = kv.coordinator.single_type_managers
        self.assertFalse(any(getattr(m, 'remote_ra_inplace_crossing', False) for m in managers))
        self.assertIs(w.single.MambaManager.remote_ra_inplace_crossing, False)
        with self.assertRaisesRegex(ValueError, 'without speculative decoding'):
            w.single.enable_inplace_crossing(managers, speculative=True)
        with self.assertRaises(ValueError):
            w.single.enable_inplace_crossing(managers[:1], speculative=False)      # no Mamba group
        for field, value in (('num_speculative_blocks', 1), ('mamba_cache_mode', 'all')):
            kv = manager(w, OFF)
            setattr(kv.coordinator.single_type_managers[2], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                w.single.enable_inplace_crossing(kv.coordinator.single_type_managers, speculative=False)
        self.assertFalse(any(getattr(m, 'remote_ra_inplace_crossing', False)
                             for m in kv.coordinator.single_type_managers))

    def test_a_crossing_moves_the_running_block(self):
        w = KP.world()
        traces = {}
        for layout in (ON, OFF):
            kv = manager(w, layout, n_m=13)
            req = KP.request('a', 3000)
            trace = []
            for new in (784, 784, 432, 1, 351, 1, 1):     # chunk, chunk (a crossing), rest, decode over 2352
                need = kv.coordinator.get_num_blocks_to_allocate_by_pool(
                    request_id='a', num_tokens=req.num_computed_tokens + new,
                    new_computed_blocks=kv.empty_kv_cache_blocks.blocks, num_encoder_tokens=0,
                    total_computed_tokens=req.num_computed_tokens,
                    num_tokens_main_model=req.num_computed_tokens + new)
                self.assertIsNotNone(alloc(kv, req, new))
                trace.append((need[1], live(mamba_rows(kv, 'a')), len(mamba_rows(kv, 'a')[0]),
                              kv.get_num_free_blocks_by_pool()[1]))
            traces[layout.inplace_crossing] = trace
            kv.free(req)
            self.assertEqual(kv.get_num_free_blocks_by_pool()[1], 12)
            self.assertTrue(all(b.ref_cnt == 0 for b in kv.block_pools[1].blocks[1:]))
        on, off = traces[True], traces[False]
        first = on[0][1]
        self.assertEqual([len(row) for row in first], [1, 1, 1])
        for need, rows, length, free in on:
            # The same three blocks, last in their rows, the pool never touched after the first chunk.
            self.assertEqual([[b for _, b in row] for row in rows], [[b for _, b in row] for row in first])
            self.assertEqual([[i for i, _ in row] for row in rows], [[length - 1]] * 3)
            self.assertEqual(free, 12 - 3)
        self.assertEqual([t[0] for t in on], [3, 0, 0, 0, 0, 0, 0])
        self.assertEqual([t[2] for t in on], [1, 2, 3, 3, 3, 4, 4])
        # Off (vLLM's align mode): a new block per crossing, the old one freed at the next allocation.
        self.assertEqual([t[0] for t in off], [3, 3, 3, 0, 0, 3, 0])
        self.assertEqual([t[3] for t in off], [9, 6, 6, 9, 9, 6, 9])
        self.assertEqual([t[2] for t in off], [t[2] for t in on])

    def test_one_chunk_over_several_blocks(self):
        w = KP.world()
        kv = manager(w, ON)
        req = KP.request('a', 6000)
        alloc(kv, req, 100)
        x = live(mamba_rows(kv, 'a'))
        self.assertIsNotNone(alloc(kv, req, 3000))                 # [100, 3100): rows of 4, three nulls
        rows = mamba_rows(kv, 'a')
        self.assertEqual(live(rows), [[(3, b)] for (_, b), in x])
        self.assertTrue(all(b.is_null for row in rows for b in row[:3]))
        self.assertEqual(kv.get_num_free_blocks_by_pool()[1], 6 - 3)

    def test_s_plus_one_slots_hold_s_requests_crossing_together(self):
        """K = S + 1: S running requests (one slot each) and phase 1's probe (the + 1). Every request
        crosses an M block in the same step: on, nothing is allocated; off, vLLM finds no block for
        the second one (a preemption on the GPU: fatal here)."""
        w = KP.world()
        for S in (3, 8):
            layout = I0.for_server(16, S, True, True)
            for inplace in (True, False):
                with self.subTest(S=S, inplace=inplace):
                    kv = manager(w, layout, n_fa=4000, enable=inplace)
                    reqs = [KP.request('r%d' % i, 5000) for i in range(S)]
                    for r in reqs:
                        self.assertIsNotNone(alloc(kv, r, 784))
                    probe = kv.block_pools[1].get_new_blocks(3)       # phase 1's probe slot (+ 1)
                    self.assertEqual(kv.get_num_free_blocks_by_pool()[1], 0)
                    results = [alloc(kv, r, 784) for r in reqs]
                    if inplace:
                        self.assertTrue(all(r is not None for r in results))
                        self.assertEqual(kv.get_num_free_blocks_by_pool()[1], 0)
                    else:
                        self.assertIsNone(results[0])
                    kv.block_pools[1].free_blocks(probe)
                    for r in reqs:
                        kv.free(r)
                    self.assertEqual(kv.get_num_free_blocks_by_pool()[1], 3 * (S + 1))

    def test_async_speculative_crossing_of_a_stopped_request(self):
        """--async-scheduling: schedule(N+1) allocates for a request whose token of N is its last (an
        EOS: scheduled once more, its row voided). Its computed count holds the placeholder. On, the
        crossing moves the running block (which keeps the state after C', the carry) and nothing
        else; the stopped request's free (at once, or deferred one fence: pop_blocks_for_free_by_pool)
        returns that one block once."""
        w = KP.world()
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                kv = manager(w, ON)
                req = KP.request('a', 1600)
                alloc(kv, req, 784)
                alloc(kv, req, 784)                             # computed 1568 = 2M: the next token crosses
                before = live(mamba_rows(kv, 'a'))
                alloc(kv, req, 1)                               # N+1, speculative (the placeholder)
                after = mamba_rows(kv, 'a')
                self.assertEqual([[b for _, b in row] for row in live(after)], [[b for _, b in row] for row in before])
                self.assertEqual([[i for i, _ in row] for row in live(after)], [[2]] * 3)
                # A later allocation sees the placeholder in computed: nothing live is skipped.
                kv.remove_skipped_blocks('a', req.num_computed_tokens + 1)
                self.assertEqual(live(mamba_rows(kv, 'a')), live(after))
                blocks = [row[2] for row in after]
                if deferred:
                    pairs = kv.pop_blocks_for_free_by_pool(req)
                    mamba = [b for pool, b in pairs if pool is kv.block_pools[1] and not b.is_null]
                    self.assertEqual(sorted(b.block_id for b in mamba), sorted(b.block_id for b in blocks))
                    for pool in kv.block_pools:
                        pool.free_blocks(reversed([b for p, b in pairs if p is pool]))
                else:
                    kv.free(req)
                self.assertTrue(all(b.ref_cnt == 0 for b in blocks))
                self.assertEqual(kv.get_num_free_blocks_by_pool()[1], 6)

    def test_hand_off_holds_the_probe_next_to_one_running_slot(self):
        """--gdn-device-handoff: the probe slot is held outside the request's row until its first phase-2
        step ran; the request itself holds one live Mamba block per group (capacity.held_slots), before
        and after every crossing."""
        w = KP.world()
        for layout in (ON, OFF):
            kv = manager(w, layout, n_m=13)
            held = capacity.held_slots(NS(kv_manager=kv), (1, 2, 3))
            probe = kv.block_pools[1].get_new_blocks(3)
            req = KP.request('a', 4000)
            counts = []
            for new in (784, 784, 784, 1, 1):
                alloc(kv, req, new)
                counts.append(held('a'))
                if len(counts) == 1:
                    kv.block_pools[1].free_blocks(probe)       # released at the second allocation
            # Off: two blocks while a crossing's old block waits for the next allocation's free
            # (784, 1568 and 2352 are crossings; the last allocation frees the old block).
            self.assertEqual(counts, [1, 1, 1, 1, 1] if layout.inplace_crossing else [1, 2, 2, 2, 1])
            kv.free(req)
            self.assertEqual(kv.get_num_free_blocks_by_pool()[1], 12)

    def test_incarnations_and_session_close(self):
        from remote_ra_live_v2.pool_incarnations import install, rows
        w = KP.world()
        kv = manager(w, ON)
        pools = [kv.pool_of_group(g) for g in range(4)]
        for pool in kv.block_pools:
            install(pool)
        req = KP.request('a', 4000)
        alloc(kv, req, 784)
        first = rows(kv, 'a', pools)
        for new in (784, 784, 1):
            alloc(kv, req, new)
            now = rows(kv, 'a', pools)
            # The moved block keeps its incarnation (no reallocation): the worker's lease keys stay.
            self.assertEqual([[v for v in row if v[0]] for row in now[1:]], [[v for v in row if v[0]] for row in first[1:]])
            self.assertTrue(all(v == (0, 0) for row in now[1:] for v in row[:-1]))
        kv.free(req)                                           # session close / request end: one free
        self.assertEqual(kv.get_num_free_blocks_by_pool(), (399, 6))
        again = KP.request('b', 100)
        alloc(kv, again, 100)
        state = kv.block_pools[1]._remote_ra_incarnations
        self.assertTrue(all(state[b.block_id] == 2 for row in mamba_rows(kv, 'b') for b in row))

    def test_mamba_prefix_caching_is_skipped(self):
        """A request that allocates with caching (one the runtime does not own: the HTTP runtime refuses
        those, owned ones pass skip_cache) on a server with the flag: its full-attention blocks are hashed
        as usual, no Mamba block ever is (so no Mamba prefix hit exists), and nothing stops the engine.
        Off, vLLM hashes the Mamba state blocks too (the path is real)."""
        w = KP.world()
        hashed = {}
        for layout in (ON, OFF):
            kv = manager(w, layout, n_m=13)
            req = KP.request('a', 3000)
            req.block_hashes = [bytes([i % 256, i // 256]) * 16 for i in range(3000 // 16)]
            req.num_prompt_tokens = 3000
            for new in (784, 784, 784):
                self.assertIsNotNone(kv.allocate_slots(req, new, has_scheduled_reqs=True))
                req.num_computed_tokens += new
            fa, *mamba = kv.coordinator.single_type_managers
            hashed[layout.inplace_crossing] = (
                sum(b.block_hash is not None for b in fa.req_to_blocks['a']),
                sum(b.block_hash is not None for m in mamba for b in m.req_to_blocks['a']),
                sum(len(m.cached_blocks_this_step) for m in mamba))
            kv.free(req)
        self.assertEqual(hashed[True][1:], (0, 0))
        self.assertGreater(hashed[True][0], 0)
        self.assertEqual(hashed[True][0], hashed[False][0])
        self.assertGreater(hashed[False][1], 0)

    def test_a_cached_or_shared_running_block_is_never_moved(self):
        w = KP.world()
        kv = manager(w, ON)
        req = KP.request('a', 2000)
        alloc(kv, req, 784)
        block = kv.coordinator.single_type_managers[2].req_to_blocks['a'][-1]
        block.ref_cnt += 1
        with self.assertRaisesRegex(RuntimeError, 'live, unshared and uncached'):
            alloc(kv, req, 784)
        block.ref_cnt -= 1
        block._block_hash = b'h'
        with self.assertRaisesRegex(RuntimeError, 'live, unshared and uncached'):
            alloc(kv, req, 784)
        # Off, vLLM's own path (the block is neither moved nor checked).
        kv = manager(w, OFF)
        req = KP.request('a', 2000)
        alloc(kv, req, 784)
        kv.coordinator.single_type_managers[2].req_to_blocks['a'][-1].ref_cnt += 1
        self.assertIsNotNone(alloc(kv, req, 784))

    def test_flag_off_allocations_equal_the_branch_point(self):
        """Off, the overlay's KV manager allocates exactly as a6d3ef7's (block ids per group, free counts,
        refusals) on random multi-request workloads: M-aligned prefill chunks, decode crossings, frees. The
        reference is a frozen copy of a6d3ef7's three changed methods on the overlay manager, so this runs in
        a Git archive too; with the history, the copy is checked against a6d3ef7 and the whole a6d3ef7 KV
        modules run as a third reference."""
        import hashlib
        self.assertEqual(hashlib.sha256(BASE_MAMBA_METHODS.encode()).hexdigest(), BASE_MAMBA_METHODS_SHA256)
        w = KP.world()
        registry = w.stubs['vllm.v1.kv_cache_spec_registry'].KVCacheSpecRegistry.managers
        frozen = base_mamba_manager(w)
        runs = [(w, None), (w, frozen)]
        with patch.object(KP, 'BASE', BASE):
            text = KP.base_source('vllm/v1/core/single_type_kv_cache_manager.py')
        if text is not None:
            self.assertEqual(base_mamba_methods(text), BASE_MAMBA_METHODS)
            runs.append((world_at(BASE), None))
        traces = []
        for w, cls in runs:
            rng = random.Random(1116)
            with patch.dict(registry, {w.iface.MambaSpec: cls} if cls is not None else {}):
                kv = manager(w, I0.for_server(16, 3, True), n_fa=300)
            if cls is not None:
                self.assertTrue(all(type(m) is cls for m in kv.coordinator.single_type_managers[1:]))
            trace, reqs = [], {}
            for step in range(400):
                op = rng.random()
                if op < 0.15 and len(reqs) < 6:
                    rid = 'r%d' % step
                    reqs[rid] = KP.request(rid, rng.randint(200, 4000) + 400)
                if not reqs:
                    continue
                rid = rng.choice(sorted(reqs))
                req = reqs[rid]
                if op > 0.93 or req.num_computed_tokens >= req.num_tokens:
                    kv.free(req)
                    del reqs[rid]
                    trace.append(('free', rid, kv.get_num_free_blocks_by_pool()))
                    continue
                start = req.num_computed_tokens
                new = min(req.num_tokens - start, rng.choice((1, 1, 1, 784 - start % 784, 3000)))
                blocks = alloc(kv, req, new)
                trace.append((rid, start, new, None if blocks is None else blocks.get_block_ids(),
                              kv.get_num_free_blocks_by_pool(),
                              [[b.block_id for b in row] for row in mamba_rows(kv, rid)]))
            traces.append(trace)
        for trace in traces[1:]:
            self.assertEqual(trace, traces[0])
        self.assertGreater(sum(1 for t in traces[0] if t[0] != 'free' and t[3] is None), 0)    # refusals too


class _AnyMeta(type):
    def __getattr__(cls, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *args, **kwargs: cls(*args, **kwargs)


class _Any(metaclass=_AnyMeta):
    """A stubbed vLLM class that takes any arguments and keeps them (SchedulerOutput, NewRequestData, ...)."""
    has_structured_output_requests = False      # SchedulerOutput's default

    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)


class _AnyStub:
    def __init__(self, **names):
        self.__dict__.update(names)

    def __getattr__(self, name):
        value = type(name, (_Any,), {})
        setattr(self, name, value)
        return value


def schedule_world(inplace, prompt=1500, slots=None):
    """The real overlay Scheduler (test_deferred_free_pools' loader; output classes that keep their
    arguments) over a decoupled (16, 784) config of one session, with one running request that the
    runtime owns: its allocation goes through allocate_slots with skip_cache (as the engine boundary
    calls it) and its prefix lookups and caching are the adapter's (none). Returns (scheduler, request,
    KV manager)."""
    import test_deferred_free_pools as DF
    w = KP.world()
    with patch.object(DF, 'Stub', _AnyStub), patch.object(DF, '_SCHEDULERS', {}):
        module = DF.scheduler_module(w)
    import contextlib
    module.record_function_or_nullcontext = lambda name: contextlib.nullcontext()
    layout = I0.for_server(16, 1, True, inplace)
    config = KP.small_config(w, layout, 400, slots or I0.mamba_pool_blocks(layout, 3))
    vc = DF.vllm_config(asynchronous=False, block_size=16, connector=False)
    vc.additional_config = I0.to_additional_config(layout)
    with patch.dict(sys.modules, {'vllm.v1.core.single_type_kv_cache_manager': w.single}):
        sched = module.Scheduler(vc, config, None, block_size=784, hash_block_size=16)
    sched.encoder_cache_manager = NS(get_freed_mm_hashes=lambda: [], free=lambda request: None)
    kv = sched.kv_cache_manager
    native = kv._growing_original_allocate_slots

    class Adapter:
        def owns(self, rid):
            return True

        def allocate_slots(self, manager, request, *args, **kwargs):
            return native(request, *args, **dict(kwargs, skip_cache=True))
    kv._remote_ra_growing_adapter = Adapter()
    req = _Request(prompt)
    sched.requests['a'] = req
    sched.running.append(req)
    return sched, req, kv


class _Request:
    """The vLLM Request fields Scheduler.schedule() reads for a running, text-only, greedy request."""
    request_id, client_index, priority, arrival_time = 'a', 0, 0, 0.0
    lora_request = kv_transfer_params = structured_output_request = pooling_params = prefill_stats = None
    has_encoder_inputs = use_structured_output = is_prefill_chunk = resumable = False
    num_preemptions = num_output_placeholders = next_decode_eligible_step = num_external_computed_tokens = 0
    num_cached_tokens = -1
    max_tokens = 4000

    def __init__(self, prompt):
        self.status = KP.Status.RUNNING
        self.prompt_token_ids, self.output_token_ids, self.spec_token_ids = list(range(prompt)), [], []
        self.num_prompt_tokens, self.num_computed_tokens = prompt, 0
        self.block_hashes, self.mm_features, self.events = [], [], []
        self.sampling_params = NS()

    @property
    def all_token_ids(self):
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self):
        return self.num_prompt_tokens + len(self.output_token_ids)

    num_tokens_with_spec = num_tokens

    @property
    def num_output_tokens(self):
        return len(self.output_token_ids)

    def is_finished(self):
        return False

    def get_num_encoder_embeds(self, i):
        return 0

    def record_event(self, *args, **kwargs):
        pass


def advance(sched, req, output):
    """The step's result as update_from_output would leave the request: computed tokens advanced
    (vLLM's _update_after_schedule already did), one sampled token appended past the prompt."""
    count = output.num_scheduled_tokens.get(req.request_id, 0)
    if req.num_computed_tokens >= req.num_prompt_tokens:
        req.output_token_ids.append(7)
    return count


class SchedulerTests(unittest.TestCase):
    def test_schedule_over_crossings(self):
        """The real Scheduler.schedule() (vLLM's _mamba_block_aligned_split, allocate_slots through the
        runtime's owned-request hook with skip_cache, _update_after_schedule) over a 1,500-token prefill
        and decode steps past 1,568 and 2,352. On, the Mamba pool may hold exactly one slot: every step is
        scheduled and the request keeps one live block per group; off, a crossing holds two."""
        traces = {}
        for inplace, slots in ((True, 4), (False, None)):
            sched, req, kv = schedule_world(inplace, slots=slots)
            pool = kv.block_pools[1]
            trace = []
            while req.num_computed_tokens < 2400:
                output = sched.schedule()
                self.assertEqual(set(output.num_scheduled_tokens), {'a'})       # never preempted
                rows = [m.req_to_blocks['a'] for m in kv.coordinator.single_type_managers[1:]]
                trace.append((output.num_scheduled_tokens['a'], len(rows[0]),
                              [sum(not b.is_null for b in row) for row in rows], pool.get_num_free_blocks()))
                advance(sched, req, output)
            traces[inplace] = trace
            self.assertTrue(all(m.remote_ra_inplace_crossing is inplace
                                for m in kv.coordinator.single_type_managers[1:]))
        on, off = traces[True], traces[False]
        self.assertEqual([t[:2] for t in on], [t[:2] for t in off])          # the same schedule
        self.assertEqual(on[:2], [(784, 1, [1, 1, 1], 0), (716, 2, [1, 1, 1], 0)])
        self.assertTrue(all(t[2] == [1, 1, 1] and t[3] == 0 for t in on))
        self.assertEqual({t[1] for t in on}, {1, 2, 3, 4})                    # crossed at 784, 1568, 2352
        self.assertIn([2, 2, 2], [t[2] for t in off])

    def test_the_layout_record_turns_it_on(self):
        import test_deferred_free_pools as DF
        w = KP.world()
        module = DF.scheduler_module(w)
        for inplace in (False, True):
            layout = I0.for_server(16, 4, True, inplace)
            config = KP.plan(w, layout, 25 * GiB)[1]
            vc = DF.vllm_config(asynchronous=False, block_size=16)
            vc.additional_config = I0.to_additional_config(layout)
            with patch.dict(sys.modules, {'vllm.v1.core.single_type_kv_cache_manager': w.single}):
                sched = module.Scheduler(vc, config, None, block_size=784, hash_block_size=16)
                flags = [getattr(m, 'remote_ra_inplace_crossing', None)
                         for m in sched.kv_cache_manager.coordinator.single_type_managers]
                self.assertEqual(sorted(map(bool, flags)), [False] + [inplace] * 3)
                if inplace:
                    # The hook itself with a speculative config: refused (enable_inplace_crossing).
                    fresh = KP.kv_manager(w, config)
                    with self.assertRaisesRegex(ValueError, 'without speculative decoding'):
                        module.Scheduler._remote_ra_inplace_crossing(
                            NS(kv_cache_manager=fresh), NS(additional_config=vc.additional_config,
                                                           speculative_config=object()))
        # Without the layout key (the shared pool's configs): nothing is read or set.
        sched = DF.construct(DF.qwen_config(w, 4), asynchronous=True)
        self.assertFalse(any(getattr(m, 'remote_ra_inplace_crossing', False)
                             for m in sched.kv_cache_manager.coordinator.single_type_managers))


# ---- the worker: vLLM's preprocess_mamba and the row guard ---------------------------------------------

def worker_mamba_utils():
    """The overlay's vllm/v1/worker/mamba_utils.py (stock 0.24.0) on CPU torch: vLLM imports stubbed, the
    real copy-spec functions of model_executor/layers/mamba/mamba_utils.py, the Triton kernels defined but
    never launched; batch_memcpy replaced by a byte memmove per (src, dst, size), the kernel's own
    semantics (one load then one store per program over its range)."""
    import ast
    import ctypes
    import dataclasses
    source = ROOT / 'source-overlay' / 'vllm'
    model = ast.parse((source / 'model_executor/layers/mamba/mamba_utils.py').read_text('utf-8'))
    keep = [n for n in model.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in ('MambaCopySpec', 'get_conv_copy_spec', 'get_temporal_copy_spec')]
    spec_ns = dict(torch=torch, dataclass=dataclasses.dataclass, is_conv_state_dim_first=lambda: False)
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'mamba_utils(model_executor)', 'exec'), spec_ns)

    class Language:
        def __getattr__(self, name):
            return object

    class MambaSpec:
        def __init__(self, block_size):
            self.block_size, self.num_speculative_blocks = block_size, 0
    stubs = {
        'vllm.config': NS(CacheConfig=object),
        'vllm.model_executor.layers.mamba.mamba_utils': NS(
            MambaStateCopyFunc=object, get_conv_copy_spec=spec_ns['get_conv_copy_spec'],
            get_temporal_copy_spec=spec_ns['get_temporal_copy_spec'], is_conv_state_dim_first=lambda: False),
        'vllm.triton_utils': NS(tl=Language(), triton=NS(jit=lambda f: f)),
        'vllm.utils.math_utils': NS(cdiv=cdiv),
        'vllm.v1.core.sched.output': NS(SchedulerOutput=object),
        'vllm.v1.kv_cache_interface': NS(KVCacheConfig=object, MambaSpec=MambaSpec),
        'vllm.v1.utils': NS(CpuGpuBuffer=object),
        'vllm.v1.worker.gpu_input_batch': NS(CachedRequestState=object),
        'vllm.v1.worker.lora_model_runner_mixin': NS(GPUInputBatch=object),
    }
    tree = ast.parse((source / 'v1/worker/mamba_utils.py').read_text('utf-8'))
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith('vllm'):
            for alias in node.names:
                body += ast.parse('%s = __stub__(%r).%s' % (alias.asname or alias.name, node.module,
                                                            alias.name)).body
        else:
            body.append(node)
    namespace = dict(__name__='_gdn_inplace_worker_mamba_utils', __stub__=stubs.__getitem__)
    exec(compile(ast.Module(body=body, type_ignores=[]), 'overlay:vllm/v1/worker/mamba_utils.py', 'exec'),
         namespace)

    def memmove(src_ptrs, dst_ptrs, sizes):
        for s, d, n in zip(src_ptrs.tolist(), dst_ptrs.tolist(), sizes.tolist()):
            ctypes.memmove(int(d), int(s), int(n))
    namespace['batch_memcpy'] = memmove
    module = NS(**{k: v for k, v in namespace.items() if not k.startswith('__')})
    module.MambaSpec = MambaSpec
    return module


class Buffer:
    """CpuGpuBuffer's host half: ``np`` and copy_to_gpu(n) (here the first n entries, on the CPU)."""

    def __init__(self, n, dtype):
        import numpy as np
        self.np = np.zeros(n, dtype=dtype)

    def copy_to_gpu(self, n):
        return torch.from_numpy(self.np[:n].copy())


@unittest.skipUnless(torch is not None, 'CPU torch required for the GDN state tensors')
class WorkerCopyTests(unittest.TestCase):
    """vLLM's preprocess_mamba / collect_mamba_copy_meta (stock 0.24.0) over CPU state tensors, one request
    crossing from Mamba block index 0 to 1 (M = 4: 4 computed tokens, 1 scheduled), and the flag's row
    guard (worker_rows.check_inplace_rows) run first, as the runner hook does."""
    M, BLOCKS = 4, 8

    def run_copy(self, row):
        import numpy as np
        from remote_ra_live_v2.worker_rows import check_inplace_rows
        mu = worker_mamba_utils()
        torch.manual_seed(0)
        conv = torch.randn(self.BLOCKS, 3, 8)          # SD conv state: (blocks, state_len, dim)
        ssm = torch.randn(self.BLOCKS, 4, 4)
        before = (conv.clone(), ssm.clone())
        spec = mu.MambaSpec(self.M)
        config = NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=['gdn.0'])])
        bufs = mu.MambaCopyBuffers(src_ptrs=Buffer(8, np.uint64), dst_ptrs=Buffer(8, np.uint64),
                                   sizes=Buffer(8, np.int32), mamba_group_ids=[0], mamba_spec=spec)
        output = NS(num_scheduled_tokens={'a': 1}, finished_req_ids=set(), preempted_req_ids=set(),
                    scheduled_cached_reqs=NS(resumed_req_ids=set()))
        batch = NS(req_ids=['a'], num_accepted_tokens_cpu=np.ones(1, dtype=np.int32))
        requests = {'a': NS(num_computed_tokens=4, block_ids=[list(row)])}
        state_idx = {'a': 0}
        guard = None
        try:
            check_inplace_rows(NS(mamba_state_idx=state_idx, requests=requests), output, (0,))
        except LiveV2Error as error:
            guard = error
        mu.preprocess_mamba(output, config, NS(enable_prefix_caching=True), state_idx, batch, requests,
                            {'gdn.0': NS(kv_cache=[conv, ssm])},
                            (mu.get_conv_copy_spec, mu.get_temporal_copy_spec), bufs)
        copied = [(int(bufs.src_ptrs.np[i]), int(bufs.dst_ptrs.np[i])) for i in range(bufs.offset)]
        return before, (conv, ssm), copied, state_idx['a'], guard

    def test_a_duplicated_row_copies_the_block_onto_itself(self):
        before, after, copied, idx, guard = self.run_copy([3, 3])
        self.assertIsNone(guard)
        self.assertEqual(idx, 1)
        self.assertEqual(len(copied), 2)                                  # conv and SSM state
        self.assertTrue(all(s == d for s, d in copied))                   # the same addresses
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(a, b))                            # every byte unchanged

    def test_the_flag_off_row_copies_into_the_new_block(self):
        before, after, copied, idx, guard = self.run_copy([3, 5])
        self.assertTrue(all(s != d for s, d in copied))
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(a[5], b[3]) and torch.equal(a[3], b[3]))
        self.assertIsNotNone(guard)        # the guard never runs off the flag (its rows differ by design)

    def test_a_replaced_row_would_copy_the_null_block_and_is_refused(self):
        """The case the guard exists for: a worker row replaced by the scheduler's (null at the old index)
        makes preprocess_mamba copy block 0 over the running state; every later check would still pass."""
        before, after, copied, idx, guard = self.run_copy([0, 3])
        self.assertIsInstance(guard, LiveV2Error)
        self.assertIn('no longer names its running block', str(guard))
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(a[3], b[0]) and not torch.equal(a[3], b[3]))

    def test_the_guard_skips_new_and_stale_requests(self):
        from remote_ra_live_v2.worker_rows import check_inplace_rows

        def output(**stale):
            return NS(num_scheduled_tokens={'a': 1, 'b': 1}, finished_req_ids=stale.get('finished', set()),
                      preempted_req_ids=stale.get('preempted', set()),
                      scheduled_cached_reqs=NS(resumed_req_ids=stale.get('resumed', set())))
        good = {'a': NS(block_ids=[[0], [0, 3]]), 'b': NS(block_ids=[[0], [4, 4]])}
        bad = {'a': NS(block_ids=[[0], [0, 3]]), 'b': NS(block_ids=[[0], [0, 4]])}
        # 'a' has no previous index (a new request): never checked; 'b' moved from index 0 to 1.
        check_inplace_rows(NS(mamba_state_idx={'b': 0}, requests=good), output(), (1,))
        with self.assertRaises(LiveV2Error):
            check_inplace_rows(NS(mamba_state_idx={'b': 0}, requests=bad), output(), (1,))
        for stale in ('finished', 'preempted', 'resumed'):        # preprocess_mamba drops their index
            check_inplace_rows(NS(mamba_state_idx={'b': 0}, requests=bad), output(**{stale: {'b'}}), (1,))
        with self.assertRaises(LiveV2Error):                     # an index past the row
            check_inplace_rows(NS(mamba_state_idx={'b': 2}, requests=good), output(), (1,))


# ---- the capacity gate -----------------------------------------------------------------------------------

def inplace_allocate(kv, request, num_new_tokens, cursors):
    """test_stageb_engine's align-mode KV fake with --gdn-inplace-crossing: a Mamba row's one live block
    moves to the new last index (MambaManager._remote_ra_move_running_block); the first allocation takes a
    fresh block; the skipped slots before the state at logical - 1 are freed as before (null here)."""
    rid = request.request_id
    P, M, fa_pool, mamba_pool = kv.P, kv.M, kv.block_pool, kv.mamba_pool
    need_fa = cdiv(cursors[E.ATT[0]] + num_new_tokens, P)
    need = [cdiv(cursors[g] + num_new_tokens, M) for g in E.MAMBA]
    logical = cursors[E.MAMBA[0]]
    keep = (logical - 1) // M if logical else 0
    for m in kv.mamba:
        row = m.req_to_blocks.setdefault(rid, [])
        for i in range(min(keep, len(row))):
            if not row[i].is_null:
                mamba_pool.free_blocks([row[i]])
                row[i] = mamba_pool.null_block
    fa = kv.fa.req_to_blocks.setdefault(rid, [])
    fresh = [k for k, m in enumerate(kv.mamba) if all(b.is_null for b in m.req_to_blocks[rid])]
    if (max(0, need_fa - len(fa)) > fa_pool.get_num_free_blocks() or
            len(fresh) > mamba_pool.get_num_free_blocks()):
        return None
    new = fa_pool.get_new_blocks(max(0, need_fa - len(fa)))
    fa.extend(new)
    kv.fa.new_block_ids.extend(b.block_id for b in new)
    for k, m in enumerate(kv.mamba):
        row = m.req_to_blocks[rid]
        alive = [i for i, b in enumerate(row) if not b.is_null]
        if alive and len(row) >= need[k]:
            continue
        if alive:
            assert alive == [len(row) - 1], 'one running block, last in its row'
            block = row[-1]
            row[-1] = mamba_pool.null_block
        else:
            block = mamba_pool.get_new_blocks(1)[0]
        row.extend([mamba_pool.null_block] * (need[k] - 1 - len(row)))
        row.append(block)
    return NS(blocks=())


def inplace_kv(base):
    class InplaceKV(base):
        def __init__(self, *args):
            super().__init__(*args)
            for m in self.mamba:
                m.remote_ra_inplace_crossing = True

        def _growing_original_allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
                                             new_computed_blocks=None, num_lookahead_tokens=0,
                                             num_external_computed_tokens=0, delay_cache_blocks=False,
                                             num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
                                             has_scheduled_reqs=True, external_computed_tokens_by_group=None,
                                             computed_tokens_by_group=None, skip_cache=False):
            cursors = tuple(computed_tokens_by_group)
            record = dict(request=request.request_id, start=request.num_computed_tokens, count=num_new_tokens,
                          cursors=cursors)
            if 'skip_cache' in getattr(self, 'record_keys', ()):
                record.update(skip_cache=skip_cache, floor=hasattr(self, '_remote_ra_chunk_capacity'))
            self.calls.append(record)
            return inplace_allocate(self, request, num_new_tokens, cursors)
    return InplaceKV


class GateTests(unittest.TestCase):
    """capacity at S = 16 / 24 / 32 over test_decouple_wp2's engine harness: S - 1 sessions running past
    their first step (one GDN slot each, actually held in the Mamba pool) and the S-th session's
    continuing goal_query turn (phase 1: the probe and the Goal slot)."""

    def harness(self, layout):
        import test_decouple_wp2 as WP2
        stack = ExitStack()
        stack.enter_context(WP2.matrix_case(layout))
        if layout.inplace_crossing:
            stack.enter_context(patch.object(WP2, 'KVManager', inplace_kv(WP2.KVManager)))
        h = WP2.Harness(layout, 'goal_query', goal=True, fa_blocks=30000)
        WP2.run_script(h, WP2.script_for(16)[:3])
        return h, stack

    def admission(self, h, S, *, handoff=False):
        """(lifecycle, command, the S-th session's request) with S - 1 running requests that hold their
        one running slot each (taken from the Mamba pool) and may all still cross M blocks."""
        request, binding = h.render(E.obs(5))
        request.sampling_params = NS(max_tokens=1536)
        h.core._remote_ra_assembly = NS(dispatcher=NS(_decode_core=lambda raw: request))
        entries = {}
        for k in range(S - 1):
            prompt = 3000 + 97 * k
            running = NS(prompt_token_ids=list(range(prompt)), num_computed_tokens=prompt + 1,
                         sampling_params=NS(max_tokens=1536), output_token_ids=[])
            h.endpoint.claims['run%d' % k] = dict(geometry=NS(phase2_start=prompt))
            entries['run%d' % k] = dict(state='accepted', request=running)
        held = h.mamba_pool.get_new_blocks(3 * (S - 1))
        if handoff:
            h.endpoint.phase1.handoff = NS(slots={}, holds=lambda session: False)
            for k in range(S - 1):
                rid = 'run%d' % k
                for m in h.kv.mamba:
                    m.req_to_blocks[rid] = [held[3 * k + h.kv.mamba.index(m)]]
        lifecycle = NS(entries=entries, endpoint=NS(core=h.core), goals=None)
        command = NS(session_id='s', metadata=dict(turn=h.session.turn), core_b64='', source=None)
        return lifecycle, command, held

    def test_need_and_headroom_on_and_off(self):
        for S in (16, 24, 32):
            for inplace, handoff in itertools.product((False, True), (False, True)):
                with self.subTest(S=S, inplace=inplace, handoff=handoff):
                    layout = I0.for_server(16, S, True, inplace)
                    h, stack = self.harness(layout)
                    with stack:
                        lifecycle, command, held = self.admission(h, S, handoff=handoff)
                        with patch('base64.b64decode', lambda raw, validate: b''):
                            need = capacity.assess(lifecycle, command)
                        gdn = 1 if inplace else 2
                        self.assertEqual(need['phase1'][1], 3 * 2)                     # probe + Goal
                        self.assertEqual(need['phase2'][1], 3 * (gdn + handoff))
                        self.assertEqual(need['headroom'][1], 0 if inplace else 3 * (S - 1))
                        self.assertEqual(need['free'][1], 3 * (layout.mamba_slots - (S - 1)))
                        # The gate admits the S-th turn: K = S + 1 on, 2S + 1 off (phase 2 plus the
                        # headroom then fill the Mamba pool exactly: 3(S + 2) blocks with the hand-off).
                        self.assertEqual(capacity._two_pool_gate(need, False), need)
                        self.assertEqual(need['phase2'][1] + need['headroom'][1] <= need['free'][1], True)
                        # A turn 0 / rebuild needs the running (and crossing) slot only.
                        phase1, phase2 = capacity.turn_need_by_pool(h.core, 'absent', [0] * 100, None, 1536)
                        self.assertEqual((phase1[1], phase2[1]), (0, 3 * gdn))
                        h.mamba_pool.free_blocks(held)

    def test_a_request_before_its_first_step(self):
        """Headroom of an accepted request whose first phase-2 step has not run: its running slot (on), its
        running and crossing slot (off); with the hand-off (its running slot not yet allocated) the same."""
        for inplace, handoff in itertools.product((False, True), (False, True)):
            with self.subTest(inplace=inplace, handoff=handoff):
                h, stack = self.harness(I0.for_server(16, 4, True, inplace))
                with stack:
                    if handoff:
                        h.endpoint.phase1.handoff = NS(slots={}, holds=lambda session: False)
                    prompt = 3000
                    waiting = NS(prompt_token_ids=list(range(prompt)), num_computed_tokens=prompt,
                                 sampling_params=NS(max_tokens=1536), output_token_ids=[])
                    h.endpoint.claims['first'] = dict(geometry=NS(phase2_start=prompt))
                    lifecycle = NS(entries={'first': dict(state='accepted', request=waiting)},
                                   endpoint=NS(core=h.core), goals=None)
                    self.assertEqual(capacity.headroom_by_pool(lifecycle)[1], 3 * (1 if inplace else 2))

    def test_overlapped_admissions_can_wait_on_the_mamba_pool(self):
        """--admission-overlap with the hand-off (the review's case, S = 4): no request decoding, two
        admissions in flight each past its phase-1 drain and holding its probe slot, a third arrives. On
        (K = 5) the ledger refuses it on the Mamba pool (its phase 2 needs the running slot and its own held
        probe, each in-flight admission still reserves one more slot); off (K = 9) it is admitted. Nothing is
        overcommitted either way: the in-place flag only makes overlapped admissions wait earlier."""
        outcomes = {}
        for inplace in (True, False):
            h, stack = self.harness(I0.for_server(16, 4, True, inplace))
            with stack:
                h.endpoint.phase1.handoff = NS(slots={}, holds=lambda session: False)
                request, binding = h.render(E.obs(5))
                request.sampling_params = NS(max_tokens=1536)
                h.core._remote_ra_assembly = NS(dispatcher=NS(_decode_core=lambda raw: request))
                command = NS(session_id='s', request_id='x', metadata=dict(turn=h.session.turn), core_b64='',
                             source=None)
                probes = h.mamba_pool.get_new_blocks(6)                 # Y's and Z's held probe slots
                with patch('base64.b64decode', lambda raw, validate: b''):
                    need = capacity.assess(NS(entries={}, endpoint=NS(core=h.core), goals=None), command)
                r = capacity.reservation(need, [])
                lifecycle = NS(entries={'y': dict(state='reserving', request=None),
                                        'z': dict(state='reserving', request=None)},
                               endpoint=NS(core=h.core), goals=None, reservations={'y': r, 'z': r})
                with patch.object(h.endpoint.phase1, 'held', lambda rid: (0, 3)), \
                        patch('base64.b64decode', lambda raw, validate: b''):
                    try:
                        capacity.ledger_gate(lifecycle, command)
                        outcomes[inplace] = 'admitted'
                    except CapacityWait as wait:
                        outcomes[inplace] = str(wait)
                h.mamba_pool.free_blocks(probes)
        self.assertEqual(outcomes[False], 'admitted')
        self.assertRegex(outcomes[True], r'^CAPACITY_WAIT: Mamba: turn needs 6 \(phase 1\) / 6 \(phase 2\) blocks, '
                                         r'2 admissions in flight 6 more, running requests 0 more, watermark 0; '
                                         r'9 free$')

    def test_s_plus_one_slots_need_the_in_place_gate(self):
        """The smaller pool with the old accounting halves the concurrency: the S-th turn waits."""
        S = 16
        layout = I0.for_server(16, S, True, True)
        h, stack = self.harness(layout)
        with stack:
            lifecycle, command, held = self.admission(h, S)
            with patch('base64.b64decode', lambda raw, validate: b''):
                need = capacity.assess(lifecycle, command)
                self.assertEqual(capacity._two_pool_gate(need, False), need)
                with patch.object(capacity, 'inplace', lambda adapter: False):
                    old = capacity.assess(lifecycle, command)
            with self.assertRaisesRegex(CapacityWait, 'Mamba pool'):
                capacity._two_pool_gate(old, False)
            # On, the gate refuses one more running request's worth too: the pool is exactly S + 1.
            more = h.mamba_pool.get_new_blocks(3)
            with patch('base64.b64decode', lambda raw, validate: b''):
                tight = capacity.assess(lifecycle, command)
            with self.assertRaisesRegex(CapacityWait, 'Mamba pool'):
                capacity._two_pool_gate(tight, False)
            h.mamba_pool.free_blocks(more + held)

    def test_random_turns_peak_within_the_in_place_need(self):
        """test_decouple_wp2's random turn sequences on the in-place fake: every turn's actual pool peak
        is within the in-place need (one Mamba slot in phase 2), both pools back at their baselines."""
        import test_decouple_wp2 as WP2
        import zlib
        checks = WP2.PropertyTests('test_random_turn_sequences')
        for policy in ('full', 'goal_query', 'observation_query'):
            for seed in range(2 if SLOW else 1):
                groups = WP2.GROUP_ORDERS[seed % 2]
                with self.subTest(policy=policy, seed=seed):
                    rng = random.Random(zlib.crc32(('inplace/%s/%d' % (policy, seed)).encode()))
                    with patch.object(WP2, 'KVManager', inplace_kv(WP2.KVManager)):
                        checks.drive(I0.for_server(16, 1, True, True), policy, rng, groups)


# ---- end to end -------------------------------------------------------------------------------------------

def append_runner(base):
    """test_decouple_e2e's Runner with vLLM's worker rows: a running request's Mamba row only grows
    (scheduler_output new_block_ids are appended; a null set at an old index on the scheduler side is
    never sent), so after an in-place crossing the old index still names the running block and
    preprocess_mamba's copy reads and writes the same block."""
    import test_stageb_worker as W

    class AppendRunner(base):
        # attach_loaded_worker's flag: worker_rows.check_inplace_rows runs before every step's copy.
        _remote_ra_inplace_crossing = tuple(W.MAMBA)

        def set_rows(self, rid, block_ids):
            old = self.requests[rid].block_ids
            rows = []
            for g, ids in enumerate(block_ids):
                if g in W.MAMBA:
                    current = list(old[g])
                    assert all(b in (current[i], 0) for i, b in enumerate(ids[:len(current)])), \
                        'a Mamba row entry changed in place'
                    ids = current + list(ids[len(current):])
                rows.append(list(ids))
            super().set_rows(rid, rows)
    return AppendRunner


def inplace_case(case):
    import test_decouple_e2e as D
    return D.Case(**dict(vars(case), name=case.name + '-inplace', inplace=True))


def enter_inplace(stack):
    """test_decouple_e2e's fakes with --gdn-inplace-crossing: the layout of an in-place case, the in-place
    align-mode KV manager, append-only worker Mamba rows and the worker lifecycle's flag."""
    import test_decouple_e2e as D
    from remote_ra_live_v2.qwen import QwenAlignLifecycle
    layout_of = D.layout_of

    def inplace_layout_of(case):
        if not getattr(case, 'inplace', False):
            return layout_of(case)
        layout = I0.for_server(case.P, getattr(case, 'sessions', 1), True, True)
        assert (layout.page_tokens, layout.mamba_tokens) == (case.P, case.M)
        return layout
    stack.enter_context(patch.object(D, 'layout_of', inplace_layout_of))
    kv = inplace_kv(D.KVManager)
    kv.record_keys = ('skip_cache', 'floor')
    stack.enter_context(patch.object(D, 'KVManager', kv))
    stack.enter_context(patch.object(D, 'Runner', append_runner(D.Runner)))
    stack.enter_context(patch.object(QwenAlignLifecycle, 'inplace_crossing', True))


# Evidence fields an in-place run may differ in from the same deterministic run with the flag off: the
# layout record and free counts (kv_pools), the restore H2D (no relocation copy), the Mamba block ids the
# pool hands out (releases) and the digests and MACs over them.
INPLACE_FIELDS = [
    r'(\[\])?/kv_pools(/.*)?',
    r'\[\]/restore/h2d_bytes',
    r'\[\]/native_evidence_sha256',
    r'(/drain|/worker_frames\[\])/http_measurement/(before|after)/(h2d_issued|h2d_joined)',
    r'(/drain|/worker_frames\[\])/http_measurement/active_owned_kv_bytes',
    # The released allocations: one Mamba slot per group in place of one per crossing (fewer entries).
    r'(/drain|/worker_frames\[\])/release(\[\]\[\])?',
    r'/worker_frames\[\](/worker_receipt/body)?/(state_digest|lease_digest|request_digest)',
    r'/worker_frames\[\]/worker_receipt/mac',
    r'(/drain|/worker_frames\[\])/(plan|lease_digest)',
    r'/plan/phase1/(receipt_digest|lease_digest)',
    r'/query/body/phase1/lease_digest',
    r'/query/mac',
]


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    maxDiff = None

    def check_turn(self, h, info):
        """test_decouple_e2e's per-turn checks (their H2D rule with no slot move), plus: the running
        slot never moved, no Mamba slot was zeroed after the first step, no GDN state was captured
        between the phase-1 state and the final carry, nothing relocated."""
        import test_decouple_e2e as D
        rid = info['rid']
        self.assertEqual(len(set(h.slots[rid])), 1, 'the running Mamba slot moved')
        log = h.io.log[info['mark'][1]:]
        captures = [e for e in log if e[0] in ('capture', 'capture_into') and e[1] == 'hybrid']
        self.assertEqual(len(captures), 2 if info['geo'].local_turn and not getattr(h, 'handoff', False) else 1)
        # Restores of GDN state: the carry into phase 1's probe and (off the hand-off) the phase-1 state
        # into the phase-2 slot; never a relocation.
        restores = [e for e in log if e[0] == 'restore' and e[1] == 'hybrid']
        self.assertEqual(len(restores), bool(info['geo'].local_turn) * (1 if getattr(h, 'handoff', False) else 2))
        D.MatrixChecks.check_turn(self, h, info)

    def run_inplace(self, policy, case, deterministic=False):
        import test_decouple_e2e as D
        with ExitStack() as stack:
            enter_inplace(stack)

            def check(h, info):
                if not hasattr(h, 'geometries'):
                    h.geometries, h.active_turns, h.restores = [], [], 0
                self.check_turn(h, info)
            return D.run(policy, inplace_case(case), check=check, deterministic=deterministic)

    def test_one_session_equals_the_flag_off_run(self):
        import test_decouple_e2e as D
        import test_host_trim as HT
        case = D.CASES[2]                                       # (16, 784)
        policies = ('full', 'random50', 'goal_query', 'observation_query') if SLOW else ('goal_query',)
        for policy in policies:
            with self.subTest(policy=policy):
                h1, infos1, cancel1, files1 = self.run_inplace(policy, case, deterministic=True)
                h0, infos0, cancel0, files0 = D.run(policy, case, deterministic=True)
                self.assertEqual(len(h1.mamba_pool.blocks), 1 + 3 * 2)          # K = S + 1 = 2
                self.assertEqual(len(h0.mamba_pool.blocks), 1 + 3 * 3)          # K = 2S + 1 = 3
                self.assertEqual(D.semantics(h1, infos1, cancel1), D.semantics(h0, infos0, cancel0))
                self.assertEqual(HT.plain(h1.kv.calls), HT.plain(h0.kv.calls))
                self.assertEqual([t['operation'] for t in h1.trace], [t['operation'] for t in h0.trace])
                for a, b in zip(infos0, infos1):
                    for key in ('carry', 'page_vectors', 'new_pages'):
                        self.assertEqual(a['receipt'][key], b['receipt'][key])
                moved = sum(1 for i in infos0 for a, c in zip(h0.slots[i['rid']], h0.slots[i['rid']][1:]) if a != c)
                self.assertGreater(moved, 0)                    # the flag-off run did cross and relocate
                self.assertEqual(sorted(files1), sorted(files0))
                seen = set()
                for name, raw in files0.items():
                    load = lambda b: ([json.loads(line) for line in b.splitlines()] if name.endswith('.jsonl')
                                      else json.loads(b))
                    for path, kind in D.json_differences(load(raw), load(files1[name]), ''):
                        path = re.sub(r'/\d+(?=/|\[|$)', '/<n>', path)
                        self.assertTrue(any(re.fullmatch(p, path) for p in INPLACE_FIELDS), (name, path, kind))
                        seen.add(path)
                self.assertIn(r'[]/restore/h2d_bytes', seen)
                # The cleanup record's pool sizes show the mode (the Mamba pool of K = S + 1 slots); the
                # startup config (and READY's worker record) carry the layout record itself.
                pools = lambda files: json.loads(files['engine-cleanup.json'])['kv_pools']['num_blocks_by_pool']
                self.assertEqual((pools(files1)[1], pools(files0)[1]), (1 + 3 * 2, 1 + 3 * 3))

    def test_three_sessions_in_one_batch_with_s_plus_one_slots(self):
        """Three sessions decoding in one batch over K = S + 1 = 4 GDN slots (goal_query: phase 1 takes the
        probe and the Goal slot while the other sessions decode), without and with --gdn-device-handoff: the
        per-turn checks of test_decouple_e2e, no running request without a block (a preemption fails the
        harness), the Mamba pool's free count recorded at every allocation."""
        import test_decouple_e2e as D
        import test_gdn_handoff as GH
        variants = [('goal_query', False), ('goal_query', True)] + ([('random50', False)] if SLOW else [])
        lows_by = {}
        for policy, handoff in variants:
            with self.subTest(policy=policy, handoff=handoff):
                multi = D.Case(**dict(vars(inplace_case(D.CASES[2])), sessions=len(D.TASKS)))
                with ExitStack() as stack:
                    enter_inplace(stack)
                    D.enter_case(stack, multi)
                    output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    value = D.config(policy, output, multi, D.TASKS)
                    if handoff:
                        value = dict(value, admission=dict(GH.ON))
                    self.assertTrue(value['kv_layout']['inplace_crossing'])
                    stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
                    stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
                    harness = GH.handoff_harness(D.MultiHarness) if handoff else D.MultiHarness
                    h = harness(policy, output, multi)
                    h.active = h.restores = 0
                    self.assertEqual(len(h.mamba_pool.blocks), 1 + 3 * (len(D.TASKS) + 1))
                    lows = [h.mamba_pool.get_num_free_blocks()]
                    take = h.mamba_pool.get_new_blocks

                    def tracked(n):
                        blocks = take(n)
                        lows.append(h.mamba_pool.get_num_free_blocks())
                        return blocks
                    h.mamba_pool.get_new_blocks = tracked
                    checks = D.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')

                    def check(h, info):
                        checks.check(h, info)
                        self.assertEqual(len(set(h.slots[info['rid']])), 1)
                    h.serve(check)                       # a running request without a block fails here
                    self.assertEqual(len(h.done), 5 * len(D.TASKS))
                    self.assertEqual(max(h.widths), len(D.TASKS))
                    self.assertGreater(h.probes_while_running, 0)
                    if not handoff:
                        self.assertEqual(h.waits, 0)     # K = S + 1 with the in-place gate
                    lows_by[policy, handoff] = (min(lows), h.waits)
                    h.close()
                    D.engine_closed(h.core)
                    self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                                     (h.baseline, h.mamba_baseline))
        # The tight case that sizes the + 1: the pool ran empty (a phase 1 with its Goal slot next to two
        # decoding sessions) with no failure; with the hand-off the gate may make admissions wait instead.
        self.assertEqual(lows_by['goal_query', False][0], 0)
        self.assertGreaterEqual(lows_by['goal_query', True][0], 0)

    def test_async_scheduling_equals_sync_and_flag_off(self):
        import test_async_sched as AS
        import test_decouple_e2e as D
        runner = AS.AsyncEndToEndTests('test_one_session')
        case = D.CASES[2]
        with ExitStack() as stack:
            enter_inplace(stack)
            sync, asyn = runner.check(inplace_case(case), 'observation_query')
        self.assertGreater(asyn['voids'], 0)                      # EOS stops voided (a speculative step)
        off = runner.serve(case, 'observation_query', False)
        self.assertEqual(sorted(off['project']), sorted(sync['project']))
        for key in sorted(off['project']):
            self.assertEqual(sync['project'][key], off['project'][key], key)

    def test_decode_fast_paths_and_phase1_in_step(self):
        """test_decode_fastpath3's three-session runs in place, on its vLLM-shaped runner (numpy block
        tables; MRoPE rows on one run) with vLLM's append-only Mamba rows: the third fast path (its steady leases,
        observations and rowless updates across in-place crossings) equals the second, also with
        --phase1-in-step (the probe allocated next to the running requests' one slot each)."""
        import test_decode_fastpath3 as T3
        import test_decouple_e2e as D
        runner = T3.EndToEndTests('test_legacy_784')
        shaped = T3.vllm_runner
        variants = ((dict(vllm='mrope'), dict(vllm='np', in_step=True), dict(vllm='np', handoff=True)) if SLOW else
                    (dict(vllm='mrope'), dict(vllm='np', in_step=True)))
        for kw in variants:
            with self.subTest(**kw), ExitStack() as stack:
                enter_inplace(stack)
                stack.enter_context(patch.object(T3, 'vllm_runner', lambda mrope: append_runner(shaped(mrope))))
                counts = runner.check(inplace_case(D.CASES[2]), 'full', **kw)
                # Neither the steady materialization nor rowless updates run with the hand-off (by design).
                for name in ('observed', 'lease') if kw.get('handoff') else ('steady', 'observed', 'lease', 'rowless'):
                    self.assertGreater(counts[name], 20, (name, counts))

    def test_with_the_hand_off(self):
        import test_decouple_e2e as D
        import test_gdn_handoff as GH
        case = inplace_case(D.CASES[2])
        with ExitStack() as stack:
            enter_inplace(stack)
            stack.enter_context(patch.object(D, 'Harness', GH.handoff_harness(D.Harness)))
            stack.enter_context(patch.object(D, 'config', GH.handoff_config(D.config)))

            def check(h, info):
                if not hasattr(h, 'geometries'):
                    h.geometries, h.active_turns, h.restores = [], [], 0
                h.handoff = True
                geo, row = info['geo'], info['row']
                restored = D.BYTES['attention'] * len(info['plan']['restore'])
                # One device copy per continuing turn (the hand-off); no crossing copy.
                self.assertEqual(row['restore']['h2d_bytes'], restored)
                self.assertEqual(row['restore']['d2d_bytes'], D.HYBRID * bool(geo.local_turn))
                self.check_turn(h, dict(info, row=dict(row, restore=dict(
                    row['restore'], h2d_bytes=restored + D.HYBRID * bool(geo.local_turn)))))
            h, infos, cancel, files = D.run('goal_query', case, check=check)
            copies = [e for e in h.io.log if e[0] == 'copy']
            self.assertEqual(len(copies), sum(1 for i in infos + [cancel] if i['geo'].local_turn))
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))


# ---- the launcher ----------------------------------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    """serve.py --gdn-inplace-crossing over test_speed_defaults' installed Qwen and Gemma 4 fixtures."""

    @classmethod
    def setUpClass(cls):
        import test_speed_defaults as SD
        cls.directory = tempfile.TemporaryDirectory()
        cls.launches = SD.Launches(Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def argv(self, *extra, release=None, sessions='16', pages='16'):
        L = self.launches
        return (['--release-dir', str(release or L.qwen), '--task', str(L.tasks), '--policy', 'full',
                 '--port', '8231', '--kv-cache-gib', '25', '--page-tokens', pages,
                 '--max-running-sessions', sessions] + list(extra))

    def test_dry_run_and_startup_config(self):
        import test_decouple_wp2 as WP2
        L = self.launches
        off_calls, off_printed, _ = L.run(self.argv())
        on_calls, on_printed, _ = L.run(self.argv('--gdn-inplace-crossing'))
        self.assertNotIn('gdn_inplace_crossing', off_calls[0][1])
        self.assertIs(on_calls[0][1]['gdn_inplace_crossing'], True)
        off, on = json.loads(off_printed), json.loads(on_printed)
        self.assertNotIn('gdn_inplace_crossing', off)
        self.assertEqual(on.pop('gdn_inplace_crossing'), dict(mamba_slots=17, validated='CPU only, not GPU-validated'))
        k = off['command'].index('--additional-config') + 1
        self.assertEqual(json.loads(off['command'][k]), {I0.LAYOUT_KEY: dict(page_tokens=16, mamba_tokens=784,
                                                                                 mamba_slots=33)})
        self.assertEqual(json.loads(on['command'][k]), {I0.LAYOUT_KEY: dict(page_tokens=16, mamba_tokens=784,
                                                                                mamba_slots=17, inplace_crossing=True)})
        # Only the layout record differs (the command, and so the dry run).
        self.assertEqual(on['command'][:k] + on['command'][k + 1:], off['command'][:k] + off['command'][k + 1:])
        self.assertEqual(dict(on, command=None), dict(off, command=None))
        # Started: startup.json pins the record; the HTTP config check accepts it and refuses variants.
        _, _, out = L.run(self.argv('--gdn-inplace-crossing'), start=True)
        config = json.loads((out / 'startup.json').read_bytes())
        self.assertEqual(config['kv_layout'], dict(page_tokens=16, mamba_tokens=784, mamba_slots=17,
                                                   inplace_crossing=True))
        self.assertEqual(json.loads((out / 'launch.json').read_bytes())['gdn_inplace_crossing']['mamba_slots'], 17)
        checks = WP2.LaunchConfigTests('test_command_line_choices')
        self.assertEqual(checks.load(config)['kv_layout'], config['kv_layout'])
        bad = {
            'KV layout differs from the server layout': dict(config['kv_layout'], mamba_slots=33),
            'pinned KV layout required': dict(config['kv_layout'], inplace_crossing=False),
        }
        for message, layout in bad.items():
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                checks.load(dict(config, kv_layout=layout))
        # The flag-off record with the in-place vLLM arguments (or the reverse) is refused too.
        off_layout = dict(page_tokens=16, mamba_tokens=784, mamba_slots=33)
        with self.assertRaisesRegex(RuntimeError, 'KV layout differs from the pinned vLLM arguments'):
            checks.load(dict(config, kv_layout=off_layout))

    def test_refusals(self):
        L = self.launches
        with self.assertRaisesRegex(ValueError, 'needs the decoupled KV layout'):
            L.run(self.argv('--gdn-inplace-crossing', pages='784'))
        calls, printed, _ = L.run(self.argv('--gdn-inplace-crossing', '--kv-layout', 'decoupled', pages='784'))
        self.assertEqual(json.loads(printed)['gdn_inplace_crossing']['mamba_slots'], 17)
        gemma = ['--release-dir', str(L.gemma), '--task', str(L.tasks), '--policy', 'full', '--port', '8231',
                 '--kv-cache-gib', '28', '--page-tokens', '16', '--max-running-sessions', '4', '--gdn-inplace-crossing']
        with self.assertRaisesRegex(ValueError, 'does not apply to Gemma 4'):
            L.run(gemma)
        with self.assertRaises(ValueError):
            serve.kv_layout(ROOT, 16, 4, None, 1)


if __name__ == '__main__':
    unittest.main()
