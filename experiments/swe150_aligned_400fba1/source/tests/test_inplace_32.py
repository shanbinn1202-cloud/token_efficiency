"""--gdn-inplace-crossing with up to 32 sessions and the format-error rollback (integrate-1116). CPU only.

integrate-1116 merges three branches that were tested apart: gdn-inplace-1116 (opt-in: one Gated-DeltaNet
slot per running request, K = S + 1, and one running slot per request in the capacity gate), sessions-32-1116
(MAX_SESSIONS and MAX_PERMITS 32) and keep-rebuild-1116 (thinking on, XML: a format-error retry, the previous
request without its reply plus one user message, rolls back to the exact history; for the engine it is a
rebuild: local turn 0, a full re-prefill). Covered here together; nothing here is CUDA evidence:

- the launcher at K = 20, 24, 32 with the flag (serve.prepare on test_installer's release fixture): the layout
  record and vLLM's --additional-config with K + 1 GDN slots, the HTTP config check, the attention pool
  (vLLM's planner, test_kv_pools) and the worker's Page arena; the attention-pool refusal at 262,144 tokens
  that the smaller Mamba pool lifts (25 GiB: K = 26 and 32 are refused off, accepted on);
- serve.main's argument parsing with --max-running-sessions 32 --gdn-inplace-crossing --admission-overlap 32
  (test_speed_defaults' fixtures): the dry run's gdn_inplace_crossing record, max_num_seqs, the layout and
  admission records, and the started launch's startup.json and launch.json;
- the engine's startup check of a 32-session in-place layout (the Mamba managers' flag must match);
- end to end on test_decouple_e2e's batched engine and worker with test_gdn_inplace's in-place KV manager and
  vLLM's append-only worker rows, goal_query at 16-token Pages: 32 sessions decoding in one batch with a
  rollback-shaped rebuild in every fourth session: no preemption, the running Mamba slot never moves, no
  CAPACITY_WAIT with an ample attention pool, the pools back at their baselines, the first three sessions
  (one with the rollback) token-exact against a 3-session server without the flag; with a tight attention
  pool turns wait (rollbacks among them, abandoned and retried) and the outputs stay the same;
- sessions-32-1116's Mamba-pool CAPACITY_WAIT at 20-24 sessions (the 2S + 1 accounting: a request that has
  just crossed holds two slots and the gate adds a crossing per running request): with the flag the same
  24-session script never waits (20 too in the slow pass), the outputs equal the flag-off server's, and the
  flag-off server is unchanged (its one Mamba wait).

The frontend half of the rollback (the exact ids, the keep render, abandon) is test_keep_rebuild's; the KV
layout does not enter it, and the engine sees a rollback as the rebuild exercised here.
"""
import contextlib
from contextlib import ExitStack
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

import install  # noqa: E402
import serve  # noqa: E402
import test_installer as INSTALL  # noqa: E402

from remote_ra_http import config as http_config  # noqa: E402
from remote_ra_live_v2 import admission_flags as AF  # noqa: E402
from remote_ra_live_v2 import capacity  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import sessions  # noqa: E402
from remote_ra_live_v2.capacity import CapacityWait  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

GIB = 1024 ** 3
CAP = 32
L256 = 262144
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
# 25 GiB at 16-token Pages: the attention blocks the flag adds (K fewer GDN slots; test_gdn_inplace at 24 / 32).
GAIN_25 = {20: 2937, 24: 3523, 32: 4698}


def tasks(count, start=0):
    return [dict(instance_id='repo__i%02d-%d' % (k, 500 + k),
                 problem_statement='Repair the flaky step %02d of job j%02d.' % (k, k))
            for k in range(start, start + count)]


class BoundTests(unittest.TestCase):
    def test_caps_and_slots(self):
        self.assertEqual((sessions.MAX_SESSIONS, AF.MAX_PERMITS), (CAP, CAP))
        for S in (1, 16, 17, 24, CAP):
            on, off = I0.for_server(16, S, True, True), I0.for_server(16, S, True)
            self.assertEqual((on.mamba_slots, off.mamba_slots, on.inplace_crossing, off.inplace_crossing),
                             (S + 1, 2 * S + 1, True, False))
            self.assertEqual(I0.mamba_pool_blocks(on, 3), 1 + 3 * (S + 1))


class LauncherTests(unittest.TestCase):
    """serve.prepare on test_installer's release fixture (as test_ms32)."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def prepare(self, task, policy='full', kv_cache_gib=25, page_tokens=16, layout=None, selection=None, **kw):
        type(self).count += 1
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.count), 8231,
                                 selection, 'mswea-xml', kv_cache_gib, page_tokens, layout, **kw)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                return http_config.load()

    def test_pools_of_20_24_and_32_in_place(self):
        import test_kv_pools as KP
        from remote_ra_live_v2.worker_composition import arena_pages
        w = KP.world()
        listed = tasks(64)
        for K in (20, 24, CAP):
            with self.subTest(K=K):
                _, config, argv, _ = self.prepare(listed, max_running=K, connector_pool_gib=1, dram_arena_gib=70,
                                                  admission_overlap=K, gdn_inplace_crossing=True)
                record = dict(page_tokens=16, mamba_tokens=784, mamba_slots=K + 1, inplace_crossing=True)
                self.assertEqual((config['kv_layout'], config['max_running_sessions']), (record, K))
                self.assertEqual(config['effective_server_config']['max_num_seqs'], K)
                self.assertEqual(argv[argv.index('--additional-config') + 1],
                                 json.dumps({'remote_ra_kv_layout': record}, sort_keys=True, separators=(',', ':')))
                admission, lease = AF.timeouts(K)
                self.assertEqual(config['admission'], dict(admission_overlap=dict(
                    permits=K, admission_seconds=admission, lease_seconds=lease)))
                layout = I0.for_server(16, K, True, True)
                self.assertEqual(http_config.kv_layout(config), layout)
                loaded = self.load(config)
                self.assertEqual((loaded['kv_layout'], http_config.running_sessions(loaded)), (record, K))
                # The attention pool by vLLM's planner: K more Mamba slots' worth of attention blocks.
                _, planned = KP.plan(w, layout, 25 * GIB)
                n_fa = serve.attention_blocks(25 * GIB, layout)
                self.assertEqual(planned.num_blocks_by_pool, (n_fa, 1 + 3 * (K + 1)))
                off = serve.attention_blocks(25 * GIB, I0.for_server(16, K, True))
                self.assertEqual(n_fa - off, GAIN_25[K])
                # The worker's Page arena counts S = K - 1 sessions of the in-place layout.
                self.assertEqual(arena_pages(NS(max_model_len=L256), {}, layout), K * 16384 + K)
                self.assertEqual(arena_pages(NS(max_model_len=L256), dict(dram_arena_pages=71680), layout), 71680)

    def test_the_attention_pool_refusal_at_262144_tokens(self):
        """25 GiB, 4,096-token replies: with 2K + 1 GDN slots K <= 25 (test_ms32); with K + 1 every K up to 32.
        24 GiB: K <= 20 without the flag, every K up to 32 with it."""
        listed = tasks(64)
        long = dict(max_model_len=L256, max_tokens=4096, dram_arena_gib=70, connector_pool_gib=1)
        for K in (24, 25):
            self.prepare(listed, max_running=K, **long)
        for K in range(17, CAP + 1):
            with self.subTest(K=K):
                if K in (26, CAP):
                    with self.assertRaisesRegex(ValueError, r'--kv-cache-gib 25 with %d sessions leaves' % K):
                        self.prepare(listed, max_running=K, **long)
                _, config, _, _ = self.prepare(listed, max_running=K, gdn_inplace_crossing=True, **long)
                self.assertEqual(config['kv_layout']['mamba_slots'], K + 1)
                if K in (24, CAP):
                    self.load(config)
        with self.assertRaisesRegex(ValueError, 'with 24 sessions leaves 17332 usable'):
            self.prepare(listed, 'full', 24, max_running=24, **long)
        for K in (24, CAP):
            self.prepare(listed, 'full', 24, max_running=K, gdn_inplace_crossing=True, **long)

    def test_refusals_at_32(self):
        listed = tasks(64)
        with self.assertRaisesRegex(ValueError, 'needs the decoupled KV layout'):
            self.prepare(listed, page_tokens=784, max_running=CAP, connector_pool_gib=1, dram_arena_gib=70,
                         gdn_inplace_crossing=True)
        with self.assertRaisesRegex(RuntimeError, r'max_running_sessions must be 1\.\.32'):
            self.prepare(listed, max_running=CAP + 1, connector_pool_gib=1, dram_arena_gib=70,
                         gdn_inplace_crossing=True)


class CommandLineTests(unittest.TestCase):
    """serve.main over test_speed_defaults' Qwen fixture (32 registered tasks)."""

    @classmethod
    def setUpClass(cls):
        import test_speed_defaults as SD
        cls.directory = tempfile.TemporaryDirectory()
        cls.launches = SD.Launches(Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def argv(self, *extra):
        L = self.launches
        return ['--release-dir', str(L.qwen), '--task', str(L.tasks), '--policy', 'full', '--port', '8231',
                '--kv-cache-gib', '25', '--page-tokens', '16', '--max-running-sessions', '32',
                '--dram-arena-gib', '70'] + list(extra)

    def test_both_flags(self):
        L = self.launches
        off_calls, off_printed, _ = L.run(self.argv('--admission-overlap', '32'))
        on_calls, on_printed, _ = L.run(self.argv('--admission-overlap', '32', '--gdn-inplace-crossing'))
        # prepare(release, task, policy, output, port, selection, protocol, kv_cache_gib, page_tokens, layout,
        # max_running, ...): the pool's K is positional.
        self.assertEqual((on_calls[0][0][10], on_calls[0][1].get('admission_overlap'),
                          on_calls[0][1]['gdn_inplace_crossing']), (32, 32, True))
        self.assertNotIn('gdn_inplace_crossing', off_calls[0][1])
        off, on = json.loads(off_printed), json.loads(on_printed)
        self.assertNotIn('gdn_inplace_crossing', off)
        self.assertEqual(on.pop('gdn_inplace_crossing'), dict(mamba_slots=33, validated='CPU only, not GPU-validated'))
        command = on['command']
        self.assertEqual(command[command.index('--max-num-seqs') + 1], '32')
        k = command.index('--additional-config') + 1
        self.assertEqual(json.loads(command[k]), {I0.LAYOUT_KEY: dict(page_tokens=16, mamba_tokens=784,
                                                                      mamba_slots=33, inplace_crossing=True)})
        self.assertEqual(json.loads(off['command'][k]), {I0.LAYOUT_KEY: dict(page_tokens=16, mamba_tokens=784,
                                                                             mamba_slots=65)})
        # Only the layout record differs.
        self.assertEqual(command[:k] + command[k + 1:], off['command'][:k] + off['command'][k + 1:])
        self.assertEqual(dict(on, command=None), dict(off, command=None))
        _, _, out = L.run(self.argv('--admission-overlap', '32', '--gdn-inplace-crossing'), start=True)
        config = json.loads((out / 'startup.json').read_bytes())
        self.assertEqual(config['kv_layout'], dict(page_tokens=16, mamba_tokens=784, mamba_slots=33,
                                                   inplace_crossing=True))
        self.assertEqual((config['max_running_sessions'], config['admission']['admission_overlap']['permits']),
                         (32, 32))
        launched = json.loads((out / 'launch.json').read_bytes())
        self.assertEqual(launched['gdn_inplace_crossing'], dict(mamba_slots=33, validated='CPU only, not GPU-validated'))


class EngineStartupTests(unittest.TestCase):
    def test_engine_startup_of_a_32_session_in_place_layout(self):
        import test_decouple_wp2 as WP2
        import test_gdn_inplace as GI
        import test_stageb_engine as E
        from remote_ra_native_growing.adapter import EngineCoreAdapter
        from remote_ra_native_growing.contract import Authority
        layout = I0.for_server(16, CAP, True, True)
        with WP2.matrix_case(layout):
            kv = GI.inplace_kv(WP2.KVManager)(layout, 2000)
            self.assertEqual(kv.kv_cache_config.num_blocks_by_pool, (2000, 1 + 3 * 33))
            adapter = EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), kv, page_tokens=16,
                                        full_attention_managers=(kv.fa,), layout=layout)
            for count in range(CAP):
                adapter.open('x%d' % count, 'full', 'm', 'r')
            self.assertEqual((len(adapter.sessions), adapter.max_open), (CAP, CAP))
            # Mamba managers without the flag (vLLM's default) are refused by the in-place layout.
            plain = WP2.KVManager(layout, 2000)
            with self.assertRaisesRegex(Exception, "in-place crossing differs from the KV layout"):
                EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), plain, page_tokens=16,
                                  full_attention_managers=(plain.fa,), layout=layout)


ROLLBACK = 'rollback'           # a lane script entry: the previous request without its reply + one user message


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class InPlaceSessionsWithRollbacksTests(unittest.TestCase):
    """test_decouple_e2e's batched engine loop (MultiHarness) with up to 32 sessions, goal_query at 16-token
    Pages (each continuing turn's phase 1 takes a probe and a Goal slot from the Mamba pool)."""
    maxDiff = None

    def harness_class(self):
        import test_decouple_e2e as DE

        class RollbackHarness(DE.MultiHarness):
            """A ROLLBACK entry drops the lane's last reply and adds one user message (the format-error
            retry's shape); a CAPACITY_WAIT keeps the entry for the next pass."""

            def admit(self, lane):
                if not (lane.script and lane.script[0] == ROLLBACK):
                    return super().admit(lane)
                saved = lane.messages
                lane.script[0], lane.messages = (DE.REBUILD, 0, None, False), saved[:-1]
                info = super().admit(lane)
                if info is None:
                    lane.script[0], lane.messages = ROLLBACK, saved
                    self.waited_rollbacks += 1
                else:
                    info['rollback'] = True
                return info
        return RollbackHarness

    def run_server(self, listed, inplace, blocks, rollbacks=True, strict=True, at=2):
        """(per-lane outputs, harness, CAPACITY_WAIT reasons). rollbacks: every fourth lane (1, 5, 9, ...)
        retries its third turn as a rollback. strict: no wait at all in place, every session in one step."""
        import test_decouple_e2e as DE
        import test_gdn_inplace as GI
        reasons = []
        gate = capacity.admission_gate

        def recorded(lifecycle, command, *args, **kw):
            try:
                return gate(lifecycle, command, *args, **kw)
            except CapacityWait as error:
                reasons.append(str(error))
                raise
        case = GI.inplace_case(DE.CASES[2]) if inplace else DE.CASES[2]
        multi = DE.Case(**dict(vars(case), sessions=len(listed), blocks=blocks))
        with ExitStack() as stack:
            if inplace:
                GI.enter_inplace(stack)
            DE.enter_case(stack, multi)
            stack.enter_context(patch.object(capacity, 'admission_gate', recorded))
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = DE.config('goal_query', output, multi, listed)
            self.assertEqual(value['kv_layout'].get('inplace_crossing', False), inplace)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = self.harness_class()('goal_query', output, multi, tasks=listed)
            h.active = h.restores = h.waited_rollbacks = 0
            S = len(listed)
            self.assertEqual(len(h.mamba_pool.blocks), 1 + 3 * ((S + 1) if inplace else (2 * S + 1)))
            self.assertEqual(h.lifecycle.max_sessions, CAP)
            h.mamba_lows = [h.mamba_pool.get_num_free_blocks()]
            take = h.mamba_pool.get_new_blocks

            def tracked(n):
                got = take(n)
                h.mamba_lows.append(h.mamba_pool.get_num_free_blocks())
                return got
            h.mamba_pool.get_new_blocks = tracked
            rolled = [lane.index for lane in h.lanes if rollbacks and lane.index % 4 == 1]
            for lane in h.lanes:
                if lane.index in rolled:
                    lane.script = lane.script[:at] + [ROLLBACK] + lane.script[at:]
            checks = DE.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')
            seen = []

            def check(h, info):
                checks.check(h, info)
                if inplace:
                    self.assertEqual(len(set(h.slots[info['rid']])), 1, 'the running Mamba slot moved')
                if info.get('rollback'):
                    # The engine's side of a rollback: a rebuild at local turn 0, a full re-prefill.
                    self.assertTrue(info['binding'].rebuild)
                    self.assertEqual((info['geo'].local_turn, info['geo'].phase2_start), (0, 0))
                    seen.append(info['lane'].index)
            h.serve(check)                                   # a running request without a block fails here
            self.assertEqual(sorted(seen), rolled)
            self.assertEqual(len(h.done), 5 * S + len(rolled))
            self.assertGreater(h.probes_while_running, 0)
            self.assertEqual(h.waits, len(reasons))
            self.assertGreaterEqual(min(h.mamba_lows), 0)
            if strict:
                self.assertEqual(max(h.widths), S)           # every session in one step
                if inplace:
                    self.assertEqual(reasons, [])            # K = S + 1 with the in-place gate
            h.close()
            DE.engine_closed(h.core)
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))
            self.assertTrue(all(block.ref_cnt == 0 for pool in h.kv.block_pools for block in pool.blocks[1:]))
            outputs = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'])
                                    for i in h.done if i['lane'] is lane] for lane in h.lanes}
            return outputs, h, reasons

    def test_32_sessions_in_one_batch_with_rollbacks(self):
        """32 sessions in place with a rollback after the fourth turn in every fourth session: no wait, no
        preemption, the pool down to its last GDN slot; the first three sessions (lane 1 with its rollback)
        equal a 3-session server's without the flag. With 190 attention blocks per session turns wait for the
        attention pool (rollbacks among them: abandoned, then admitted) and the outputs stay the same."""
        import test_decouple_e2e as DE
        listed = list(DE.TASKS) + tasks(CAP - len(DE.TASKS))
        many, h, _ = self.run_server(listed, True, CAP * 400, at=4)
        self.assertGreater(h.active, 0)
        self.assertLessEqual(min(h.mamba_lows), 3)           # at most one GDN slot (3 groups) left at the peak
        three, _, _ = self.run_server(list(DE.TASKS), False, 3 * 400, at=4)
        for k in range(len(DE.TASKS)):
            self.assertEqual(many[k], three[k], k)
        tight, h, reasons = self.run_server(listed, True, CAP * 190, strict=False, at=4)
        self.assertGreater(h.waited_rollbacks, 0)
        self.assertTrue(all(r.startswith('CAPACITY_WAIT: FA pool:') for r in reasons), reasons)
        self.assertEqual(tight, many)
        if SLOW:
            # The rollback after the second turn (a shorter re-prefill).
            early, h, _ = self.run_server(listed, True, CAP * 400)
            three, _, _ = self.run_server(list(DE.TASKS), False, 3 * 400)
            for k in range(len(DE.TASKS)):
                self.assertEqual(early[k], three[k], k)

    def test_the_mamba_wait_of_the_2s_plus_1_accounting_is_gone(self):
        """sessions-32-1116 (test_ms32 BatchedTwentyFourSessionsTests, slow pass): goal_query at 20-24
        sessions waits once for the Mamba pool without the flag (a running request that has just crossed
        holds two slots, and the gate adds a crossing for every running request). With the flag no
        crossing slot exists and none is counted: the same script runs without any wait, the outputs equal
        the flag-off server's."""
        import test_decouple_e2e as DE
        for S in (20, 24) if SLOW else (24,):
            with self.subTest(S=S):
                listed = list(DE.TASKS) + tasks(S - len(DE.TASKS))
                on, h_on, reasons_on = self.run_server(listed, True, S * 400, rollbacks=False)
                off, h_off, reasons_off = self.run_server(listed, False, S * 400, rollbacks=False, strict=False)
                self.assertEqual(reasons_on, [])
                # Off: unchanged by the merge (within test_ms32's bound of two): one wait, the Mamba pool's.
                self.assertEqual(len(reasons_off), 1)
                self.assertTrue(all(r.startswith('CAPACITY_WAIT: Mamba pool:') for r in reasons_off), reasons_off)
                self.assertEqual(on, off)


if __name__ == '__main__':
    unittest.main()
