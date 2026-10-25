"""serve.py --connector-pool-gib (remote_ra_live_v2.connector_guard). CPU only; not GPU evidence.

The OffloadingConnector's CPU pool is allocated (pinned and zeroed) at worker start but
never used by this runtime: the HTTP runtime isolates the connector, so no request
loads from it or stores to it. Off (the default) adds no vLLM argument change, startup
config key or check; the identity probe covers those bytes, and here an explicit None
equals the default. On (1-31 GiB):

- launcher: the vLLM command differs only in --kv-transfer-config's cpu_bytes_to_use;
  startup.json gains ``connector_pool_bytes``; the pinned-host budget counts the
  smaller pool (configurations refused at 32 GiB pass, refusals name the pool);
- startup checks: remote_ra_http.config.load refuses bad pins; compose_engine checks
  the engine's vLLM config and attach_loaded_worker the allocated pool, once each;
- vLLM's CPU spec gives at least one offloaded block at 1 GiB in every layout;
- the sources behind the documentation say what it claims.
"""
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve
import test_align_ctx as AC
import test_kv_pools as KP
import test_language_model_only as TLM

from remote_ra_http import config as http_config
from remote_ra_live_v2 import connector_guard as G
from remote_ra_live_v2.contract import LiveV2Error

GIB = 1024 ** 3
L128, L256 = 131072, 262144
tasks = AC.tasks
CASES = TLM.CASES
TRANSFER = '--kv-transfer-config'


def transfer_bytes(argv):
    value = json.loads(argv[argv.index(TRANSFER) + 1])
    return value['kv_connector_extra_config']['cpu_bytes_to_use']


# ---- 1. launcher ----------------------------------------------------------------------------------

class LauncherTests(AC.Release, unittest.TestCase):
    maxDiff = None

    def test_off_is_the_default(self):
        for task, policy, gib, P, layout, selection, extra in CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                out = 'off%d' % P
                plain = self.outcome(task, policy, gib, P, layout, selection, out=out, **extra)
                none = self.outcome(task, policy, gib, P, layout, selection, out=out, connector_pool_gib=None,
                                    **extra)
                self.assertEqual(none, plain)
                if plain[0] == 'ok':
                    self.assertNotIn(G.POOL_KEY, plain[1][1])
                    self.assertEqual(transfer_bytes(plain[1][2]), serve.CONNECTOR_BYTES)
                    self.assertEqual(serve.CONNECTOR_BYTES, G.DEFAULT_POOL_BYTES)

    def test_on_changes_only_the_pool(self):
        for n in (1, 16, 31):
            for task, policy, gib, P, layout, selection, extra in CASES:
                with self.subTest(n=n, policy=policy, P=P, extra=extra):
                    out = 'on%d' % P
                    plain = self.outcome(task, policy, gib, P, layout, selection, out=out, **extra)
                    if plain[0] != 'ok':
                        continue                      # refusals: test_budget_counts_the_pool
                    m0, c0, a0, e0 = plain[1]
                    m1, c1, a1, e1 = self.prepare(task, policy, gib, P, layout, selection, out=out,
                                                  connector_pool_gib=n, **extra)
                    at = a0.index(TRANSFER) + 1
                    self.assertEqual(m1, m0)
                    self.assertEqual(a1[:at] + a1[at + 1:], a0[:at] + a0[at + 1:])
                    self.assertEqual(json.loads(a1[at]), dict(json.loads(a0[at]),
                                                              kv_connector_extra_config={'cpu_bytes_to_use': n * GIB}))
                    self.assertEqual(c1, dict(c0, connector_pool_bytes=n * GIB))
                    self.assertEqual(c1['effective_server_config'], c0['effective_server_config'])
                    self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                    self.assertEqual(http_config.connector_pool(self.load(c1)), n * GIB)
                    self.assertIsNone(http_config.connector_pool(self.load(c0)))

    def test_other_values_are_refused(self):
        message = '--connector-pool-gib must be an integer 1-31 (leave it out for the default 32 GiB pool)'
        for value in (0, -1, 32, 33, 64, '1', 1.0, True, False):
            with self.subTest(value=value):
                self.assertEqual(self.outcome(tasks(1)[0], connector_pool_gib=value), ('ValueError', message))

    def test_budget_counts_the_pool(self):
        self.assertEqual(serve.pinned_host_bytes(16, 16, True, L128, connector=serve.CONNECTOR_BYTES),
                         serve.pinned_host_bytes(16, 16, True, L128))
        for S, P, L, arena in ((16, 16, 65536, None), (16, 784, 65536, None), (12, 16, L128, None),
                               (16, 16, L128, 88 * 1024), (6, 16, L256, None)):
            self.assertEqual(serve.pinned_host_bytes(S, P, P != 784, L, arena) -
                             serve.pinned_host_bytes(S, P, P != 784, L, arena, connector=GIB), 31 * GIB)
        # Every 65,536-token configuration: at most 68.4 GiB with a 1 GiB pool (99.4 with 32).
        worst = max(serve.pinned_host_bytes(S, P, P != 784 or decoupled, 65536, connector=GIB)
                    for P in serve.PAGE_TOKENS for S in range(1, 17) for decoupled in (False, True))
        self.assertEqual(round(worst / GIB, 2), 68.37)
        # Refused with the 32 GiB pool, accepted with 1 GiB: 12 sessions at 131,072 tokens
        # (130.0 -> 99.0 GiB) and 6 at 262,144 (129.1 -> 98.1 GiB), in both layouts.
        for count, L in ((12, L128), (6, L256)):
            for P in (16, 784):
                with self.subTest(count=count, L=L, P=P):
                    with self.assertRaisesRegex(ValueError, r'^%d sessions at the %d-token limit may pin [\d.]+ GiB '
                                                            r'of host memory \(the 32 GiB connector pool, ' % (count, L)):
                        self.prepare(tasks(count), kv_cache_gib=24, page_tokens=P, max_model_len=L)
                    config = self.prepare(tasks(count), kv_cache_gib=24, page_tokens=P, max_model_len=L,
                                          connector_pool_gib=1)[1]
                    self.assertEqual(config[G.POOL_KEY], GIB)
        # A refusal names the pool it counted.
        with self.assertRaisesRegex(ValueError, r'^16 sessions at the 131072-token limit may pin 131\.5 GiB of host '
                                                r'memory \(the 1 GiB connector pool, GDN carries .* use fewer '
                                                r'sessions or --dram-arena-gib$'):
            self.prepare(tasks(16), kv_cache_gib=24, page_tokens=16, max_model_len=L128, connector_pool_gib=1)
        # The largest arena that fits grows by the pool's 31 GiB: 93 -> 124 GiB for 16 sessions at 262,144.
        kwargs = dict(policy='observation_query', kv_cache_gib=32, page_tokens=16, max_model_len=L256,
                      max_tokens=4096, selection=dict(active_token_budget=8192))
        self.prepare(tasks(16), dram_arena_gib=124, connector_pool_gib=1, **kwargs)
        with self.assertRaisesRegex(ValueError, r'\(the 1 GiB connector pool, .* use fewer sessions or a smaller '
                                                r'--dram-arena-gib \(at most 124 GiB\)$'):
            self.prepare(tasks(16), dram_arena_gib=126, connector_pool_gib=1, **kwargs)
        with self.assertRaisesRegex(ValueError, r'\(at most 93 GiB\)$'):
            self.prepare(tasks(16), dram_arena_gib=124, **kwargs)

    def test_dry_run_main(self):
        task = self.root / 'pool-task.json'
        task.write_text(json.dumps(tasks(1)[0]))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--output',
                    str(self.root / 'pool-dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(stream.getvalue())
        plain, pooled = main(), main('--connector-pool-gib', '1')
        self.assertEqual(transfer_bytes(plain['command']), 32 * GIB)
        self.assertEqual(transfer_bytes(pooled['command']), GIB)
        at = plain['command'].index(TRANSFER) + 1
        self.assertEqual(dict(pooled, command=None), dict(plain, command=None))
        self.assertEqual([a for i, a in enumerate(pooled['command']) if i != at],
                         [a for i, a in enumerate(plain['command']) if i != at])
        with self.assertRaisesRegex(ValueError, 'must be an integer 1-31'):
            main('--connector-pool-gib', '32')
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main('--connector-pool-gib', 'one')


# ---- 2. startup checks ----------------------------------------------------------------------------

class PinTests(unittest.TestCase):
    def test_pin_values(self):
        self.assertIsNone(G.pinned_pool(None))
        self.assertIsNone(G.pinned_pool({}))
        for n in (1, 2, 16, 24, 31):
            self.assertEqual(G.pinned_pool({G.POOL_KEY: n * GIB}), n * GIB)
        for bad in (0, -GIB, 32 * GIB, 33 * GIB, GIB + 1, GIB // 2, float(GIB), str(GIB), True, None):
            with self.subTest(bad=bad), self.assertRaisesRegex(LiveV2Error, 'connector_pool_bytes must be 1-31 GiB'):
                G.pinned_pool({G.POOL_KEY: bad})

    def test_engine_check(self):
        def engine(value, name='OffloadingConnector'):
            return NS(kv_transfer_config=NS(kv_connector=name, kv_connector_extra_config={'cpu_bytes_to_use': value}))
        pin = {G.POOL_KEY: GIB}
        for anything in (None, NS(), engine(32 * GIB), engine(GIB)):
            self.assertIsNone(G.check_engine_pool(anything, {}))       # off: nothing is read
        self.assertEqual(G.check_engine_pool(engine(GIB), pin), GIB)
        for bad, message in ((engine(32 * GIB), 'differs from the pinned'), (engine(float(GIB)), 'differs'),
                             (engine(None), 'differs'), (engine(GIB, 'LMCacheConnectorV1'), 'needs the Offloading'),
                             (NS(kv_transfer_config=None), 'needs the Offloading'), (NS(), 'needs the Offloading')):
            with self.subTest(message=message, bad=bad), self.assertRaisesRegex(LiveV2Error, message):
                G.check_engine_pool(bad, pin)

    def test_compose_engine_runs_the_engine_check_once_on_the_http_runtime(self):
        from remote_ra_live_v2 import composition
        source = inspect.getsource(composition.compose_engine)
        self.assertEqual(source.count('check_engine_pool('), 1)
        block = source[source.index('if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1":'):]
        block = block[:block.index('bind.bind_loaded_worker(')]
        self.assertIn('isolate_connector(core)', block)
        self.assertIn('check_engine_pool(core.vllm_config, load())', block)


class Tensor:
    def __init__(self, rows, row_bytes):
        self.shape, self._row = (rows, row_bytes), row_bytes

    def numel(self):
        return self.shape[0] * self._row

    def element_size(self):
        return 1


def OffloadingConnector(pool, blocks, tensors, row=65536):
    cls = type('OffloadingConnector', (), {})
    connector = cls()
    handlers = NS(gpu_to_cpu_handler=NS(dst_tensors=[Tensor(blocks, row) for _ in range(tensors)]))
    connector.connector_worker = NS(spec=NS(extra_config={'cpu_bytes_to_use': pool}, num_blocks=blocks,
                                            _handlers=handlers))
    return connector


class WorkerPoolTests(unittest.TestCase):
    def check(self, connector, config, runtime='1'):
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME=runtime), \
                patch('remote_ra_http.config.load', lambda: config):
            return G.check_worker_pool(connector)

    def test_off_reads_nothing(self):
        for connector in ('connector', None, OffloadingConnector(32 * GIB, 32768, 16)):
            self.assertIsNone(self.check(connector, {}))
            self.assertIsNone(self.check(connector, {G.POOL_KEY: GIB}, runtime='0'))

    def test_on_checks_the_allocated_pool(self):
        pin = {G.POOL_KEY: GIB}
        # 16 FA layers at P = 16: 1,024 blocks of 64 KiB per layer = 1 GiB.
        self.assertEqual(self.check(OffloadingConnector(GIB, 1024, 16), pin),
                         dict(pool_bytes=GIB, blocks=1024, allocated_bytes=GIB))
        # The shared 784-token pool: 20 blocks of 16 x 3,211,264 bytes, below 1 GiB.
        self.assertEqual(self.check(OffloadingConnector(GIB, 20, 16, 3211264), pin),
                         dict(pool_bytes=GIB, blocks=20, allocated_bytes=20 * 16 * 3211264))
        bare = OffloadingConnector(GIB, 1024, 16)
        bare.connector_worker.spec._handlers = None
        no_worker = OffloadingConnector(GIB, 1024, 16)
        no_worker.connector_worker = None
        ragged = OffloadingConnector(GIB, 1024, 16)
        ragged.connector_worker.spec._handlers.gpu_to_cpu_handler.dst_tensors[3] = Tensor(1023, 65536)
        for bad, message in ((OffloadingConnector(32 * GIB, 32768, 16), 'differs from the pinned'),
                             ('connector', 'unsupported loaded connector'),
                             (no_worker, 'offloading worker missing'),
                             (bare, 'not allocated'),
                             (OffloadingConnector(GIB, 0, 16), 'at least one block'),
                             (OffloadingConnector(GIB, 1024, 0), 'at least one block'),
                             (ragged, 'at least one block'),
                             (OffloadingConnector(GIB, 1025, 16), 'more than the pinned')):
            with self.subTest(message=message), self.assertRaisesRegex(LiveV2Error, message):
                self.check(bad, pin)

    def test_attach_loaded_worker_runs_the_worker_check_once(self):
        calls = []
        with patch.object(G, 'check_worker_pool', lambda connector: calls.append(connector)), \
                patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            plain = TLM.attach(None)[0]
        self.assertEqual(calls, ['connector'])          # the loaded connector (the fixture's stand-in)
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertEqual(TLM.attach(None)[0], plain)  # the evidence is unchanged


# ---- 3. vLLM's CPU spec at 1 GiB ------------------------------------------------------------------

class SpecTests(unittest.TestCase):
    def test_one_gib_gives_blocks_in_every_layout(self):
        w = KP.world()
        module = KP.cpu_spec_module(w)
        # (layout, KV budget, blocks at 1 GiB, blocks at the default 32 GiB); 784-token
        # blocks are 51,380,224 bytes (20 and 668), P-token blocks P x 65,536 bytes.
        for layout, budget, one, default in ((KP.I0.legacy(), 8 * GIB, 20, 668),
                                             (KP.I0.for_server(784, 4, True), 24 * GIB, 20, 668),
                                             (KP.I0.for_server(16, 1), 8 * GIB, 1024, 32768),
                                             (KP.I0.for_server(64, 16), 24 * GIB, 256, 8192)):
            with self.subTest(page_tokens=layout.page_tokens, decoupled=layout.decoupled):
                vc, config = KP.plan(w, layout, budget)
                extra = vc.kv_transfer_config.kv_connector_extra_config
                self.assertEqual(module.CPUOffloadingSpec(vc, config).num_blocks, default)
                extra['cpu_bytes_to_use'] = GIB
                spec = module.CPUOffloadingSpec(vc, config)
                self.assertEqual(spec.num_blocks, one)
                self.assertLessEqual(spec.num_blocks * spec.kv_bytes_per_offloaded_block, GIB)


# ---- 4. the sources behind the documentation -------------------------------------------------------

class SourceClaimTests(unittest.TestCase):
    CLAIMS = {
        # The HTTP runtime isolates the connector: no lookup hit, no external load, no store, no jobs.
        'source-overlay/remote_ra_http/connector.py': (
            "require(owns(request),'unowned prefix lookup');return 0,False",
            "require(owns(request) and num_external_tokens==0,'external offload transfer forbidden')",
            "require(owns(request),'unowned connector completion');return False,None",
            'return OffloadingConnectorMetadata(load_jobs={},store_jobs={},jobs_to_flush=set(),selective_load_plans={})'),
        'source-overlay/remote_ra_live_v2/composition.py': (
            'if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1":\n        isolate_connector(core)',),
        'serve.py': ("REMOTE_RA_HTTP_RUNTIME='1'",),
        # cpu_bytes_to_use sizes the pool in whole offloaded blocks...
        'source-overlay/vllm/v1/kv_offload/cpu/spec.py': (
            'int(cpu_bytes_to_use) // aligned_kv_bytes_per_offloaded_block',),
        # ...which the worker pins and zeroes at registration, one tensor per KV tensor.
        'source-overlay/vllm/v1/kv_offload/cpu/gpu_worker.py': (
            'cpu_tensor = torch.zeros(\n                    (num_cpu_blocks, cpu_page_size_bytes),',
            'pin_memory=pin_memory,'),
        # Captures and restores go through the runtime's own pinned memory.
        'source-overlay/remote_ra_growing_worker/tensors.py': (
            'class PageArena:', 'device="cpu", pin_memory=True)'),
    }

    def test_claims_hold(self):
        for name, snippets in self.CLAIMS.items():
            text = (ROOT / name).read_text(encoding='utf-8')
            for snippet in snippets:
                with self.subTest(name=name, snippet=snippet[:40]):
                    self.assertIn(snippet, text)


if __name__ == '__main__':
    unittest.main()
