"""Decoupled KV layout, WP4: the additive evidence of remote_ra_http.evidence; CPU only.

One deterministic server life (worker attached, READY, one admitted turn, its
native terminal file and runtime.jsonl row, engine teardown) runs over the (P, M)
matrix with minimal fakes of the engine owners:

- legacy (no ``kv_layout`` in the startup config): every file is byte-identical
  to the one the evidence module of 8586dbd writes (pinned SHA256 below);
- decoupled: ready.json records the layout, both pool sizes and the free blocks
  at readiness, every native terminal file and its row the free blocks after the
  drain, engine-cleanup.json the free blocks after teardown; nothing else changes.

Pool numbers come from the KV manager's interface I1 (``kv_cache_config.
num_blocks_by_pool``, ``get_num_free_blocks_by_pool()``), faked here.
"""
import asyncio
from contextlib import ExitStack
from hashlib import sha256
import json
import os
import pathlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))
import test_decouple_selection as DS
from remote_ra_http import evidence as EVIDENCE
from remote_ra_live_v2 import selection_contract as sc

EPOCH, MANIFEST, BINDING = 'e' * 64, 'm' * 64, 'b' * 64
TASK = dict(instance_id='ev', problem_statement='Fix bug.')
RID = 'req-0'
# 8586dbd's evidence module on the legacy scenario below (full / observation_query):
# worker-attached.json, ready.json, the native terminal file, runtime.jsonl, engine-cleanup.json.
LEGACY_SHA256 = {
    'full': {
        'worker-attached.json': '26ff107d589818ba83a101497c784814b1ded038b6f8262fe9aa9fc47ef5df48',
        'ready.json': '908b0a37e6d46c9d02d3d8da6bd22bac11730015ad193254469c75fa62968f74',
        'native': '7803dec907cd2b535aaa6e2b40c56e2c0d863353b3396c32689e848d8014ef25',
        'runtime.jsonl': '5cc103ac92640c7e31c63d48cd0a2526f58909183e613c73218f844312eef3e9',
        'engine-cleanup.json': '78fd0f9a7a21d798bc0428a0295b9839423e27d5d75fd8722887a5671678a181'},
    'observation_query': {
        'worker-attached.json': '26ff107d589818ba83a101497c784814b1ded038b6f8262fe9aa9fc47ef5df48',
        'ready.json': '2f4b39b023d957b77269e44fd395e0dd45b04a399ae9c888038013c1073cc83f',
        'native': '3229316591fdd48ebfd4f0b8aa8880baf82f7777eee5d342c0c1758fe54b7b3c',
        'runtime.jsonl': '2cdb9a0fb6c73cfbbd1a91cda0a1ddba0a5e21eb48eda8db8332f25751e8ab2b',
        'engine-cleanup.json': '78fd0f9a7a21d798bc0428a0295b9839423e27d5d75fd8722887a5671678a181'},
}
FILES = ('worker-attached.json', 'ready.json', 'native', 'runtime.jsonl', 'engine-cleanup.json')


def config(output, policy, P, decoupled):
    contract = None if policy == 'full' else DS.contract_for(P, decoupled)
    value = dict(schema='remote-ra-http-config-v2', policy=policy, task=TASK, tasks=[TASK],
                 output_directory=str(output), page_tokens=P, selection_seed=42, selection_fraction=.5,
                 selection_contract=contract, goal_rewriter=None, protocol='mswea_xml_v1', warmup_requests=0,
                 input_pins_sha256='i' * 64, task_sha256='k' * 64, common_config_sha256='c' * 64,
                 issue_sha256=sha256(TASK['problem_statement'].encode()).hexdigest(), model_revision='rev',
                 tokenizer_revision='rev', effective_server_config=dict(max_num_seqs=1),
                 server_config_sha256='s' * 64)
    if decoupled:
        value['kv_layout'] = DS.layout_record(P, True, sessions=1)
    return value


class Manager:
    """KVCacheManager stand-in: two pools (decoupled) or one (legacy)."""
    def __init__(self, pools):
        self.kv_cache_config = NS(num_blocks_by_pool=pools, num_blocks=pools[0] if pools else 167)
        if pools is None:
            del self.kv_cache_config.num_blocks_by_pool
        self.free = [n - 1 for n in pools] if pools else None

    def get_num_free_blocks_by_pool(self):
        return tuple(self.free)


def pools_of(P, decoupled, gib=8, sessions=1):
    """D4: N_m = 1 + 3K (K = 2S + 1), N_fa = (budget - Mamba bytes) // (P x 65,536)."""
    if not decoupled:
        return None
    slots = 2 * sessions + 1
    mamba = 1 + 3 * slots
    return (gib * 1024 ** 3 - 16 * mamba * 3207168) // (P * 65536), mamba


def plan(local_turn=0, selected=()):
    return dict(local_turn=local_turn, computed_start=0, sealed_start=0, candidate_end=0, observation=None,
                phase2_start=0, prompt_end=40, tail_candidate=False, tail_live=False, retain=[], restore=[],
                release=[], run_pages=[0, 1], selected=list(selected), resident_before=[])


def frames():
    counters = lambda n: dict(h2d_issued=n, h2d_joined=n, zero_joins=0)
    return [dict(operation='prepare', http_measurement=dict(before=counters(0), after=counters(0),
                                                             elapsed_ns=5, active_owned_kv_bytes=0)),
            dict(operation='complete', carry=dict(computed_end=41), dram=['x', 'y'],
                 http_measurement=dict(before=counters(0), after=counters(0), elapsed_ns=7,
                                       active_owned_kv_bytes=4096))]


def query_receipt(policy, P):
    if policy == 'full':
        return None
    selection = dict(contract_sha256='q' * 64, reason='cold_start', candidate_pages=0, history_tokens=0,
                     capacity_pages=0, selected_pages=[], mandatory_pages=[], recent_pages=[], retrieved_pages=[],
                     page_turns={}, projection_sha256=None, selector_config_hash=None, query_hash='h' * 64,
                     initial_prompt_end=40)
    return dict(body=dict(local_turn=0, candidate_pages=0, selection=selection, goal_hash='g' * 64,
                          goal=dict(variant='full_goal')), mac='a' * 64)


def scenario(module, output, policy, P, decoupled):
    """One server life through ``module`` (an evidence module); returns the files' bytes."""
    value = config(output, policy, P, decoupled)
    manager = Manager(pools_of(P, decoupled))
    clock = iter(range(10 ** 6, 10 ** 7, 1000))
    worker, bridge = NS(http_receipts={RID: frames()}), NS()
    binding = NS(request=RID, session='s', epoch=EPOCH, turn=0, local_turn=0, rebuild=False, generation='g0',
                 identity=BINDING, tokens=tuple(range(40)), observation_span=None, stable_end=40)
    selection = NS(task=TASK, tasks=lambda: [TASK], authorized={RID: query_receipt(policy, P)})
    entry = dict(binding=binding, drain_receipt=dict(drained=True, operation='complete'), state='complete',
                 admission_engine_ms=2.5)
    life = NS(closed=True, entries={RID: entry})
    sessions = NS(ids=['s'], index=lambda sid: 0, closed=True, namespaces={'s': 'ns-0'}, epoch=EPOCH, session='s')
    core = NS(_remote_ra_growing_adapter=NS(page_tokens=P, kv_manager=manager, requests={}),
              _remote_ra_live_v2_boundary=NS(proxy=worker, prepared={RID: dict(plan=plan())}),
              _remote_ra_live_v2_lifecycle=life, scheduler=NS(_remote_ra_live_v2_bridge=bridge),
              _remote_ra_private_v2_endpoint=NS(selection=selection), _remote_ra_live_session_authority=sessions,
              _remote_ra_growing_owner=NS(identity=NS(epoch=EPOCH, manifest=MANIFEST)))
    receipt = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', copies_drained=True, loaded_sources_verified=True,
                   tensor_geometry=dict(page_tokens=P), source_closure=dict(files=1),
                   vector_space=dict(hidden_size=5120))
    read_text = pathlib.Path.read_text

    def proc(path, *args, **kwargs):
        name = str(path).replace('\\', '/')
        if name == '/proc/sys/kernel/random/boot_id':
            return 'boot\n'
        if name == '/proc/self/stat':
            return '1 (python) ' + ' '.join(['0'] * 30)
        return read_text(path, *args, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(module, 'load', lambda: value))
        stack.enter_context(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        stack.enter_context(patch.object(module.os, 'getpid', lambda: 4242))
        stack.enter_context(patch.object(module.time, 'monotonic_ns', lambda: next(clock)))
        stack.enter_context(patch.object(pathlib.Path, 'read_text', proc))
        report = module.worker_attached(core, worker, bridge, life, receipt)
        authority = NS(identity=NS(epoch=EPOCH, manifest=MANIFEST), policy=policy, session_id='s',
                       verify_admission=lambda: True, live_v2_readiness=report,
                       loaded=NS(manifest=dict(qualification_parent_manifest='p' * 64)))
        frontend = module.FrontendEvidence(authority, None, dict(max_num_seqs=1))
        frontend.write_ready(NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]), NS(started=True))
        wire = json.dumps(dict(model='m', messages=[dict(role='system', content='sys'),
                                                    dict(role='user', content=TASK['problem_statement'])])).encode()

        class Raw:
            async def body(self):
                return wire
        asyncio.run(frontend.begin(Raw(), NS(_remote_ra_live_four_policy=NS(session=NS(turn=0)))))
        if decoupled:      # the request's blocks are freed by the drain; another session holds some
            manager.free = [manager.free[0] - 3, manager.free[1] - 3]
        module.native_terminal(NS(core=core, worker=worker), entry, 'complete')
        frontend.complete(NS(internal_id=RID, engine_complete=True), NS(id='resp-0'))
        if decoupled:      # teardown closed every session
            manager.free = [n - 1 for n in pools_of(P, decoupled)]
        module.engine_closed(core)
    native = output / ('native-' + sha256(RID.encode()).hexdigest() + '.json')
    return {name: (native if name == 'native' else output / name).read_bytes() for name in FILES}


class EvidenceMatrixTests(unittest.TestCase):
    def run_case(self, policy, P, decoupled):
        with tempfile.TemporaryDirectory() as tmp:
            return scenario(EVIDENCE, Path(tmp), policy, P, decoupled)

    def test_legacy_evidence_is_byte_identical(self):
        for policy in ('full', 'observation_query'):
            files = self.run_case(policy, 784, False)
            self.assertEqual({k: sha256(v).hexdigest() for k, v in files.items()}, LEGACY_SHA256[policy], policy)
            for name in ('worker-attached.json', 'ready.json', 'native', 'engine-cleanup.json'):
                self.assertNotIn(b'kv_pools', files[name])
                self.assertNotIn(b'kv_layout', files[name])
            self.assertNotIn(b'kv_pools', files['runtime.jsonl'])

    def test_decoupled_evidence_is_additive(self):
        for name, P, M, decoupled in DS.CASES:
            for policy in ('full', 'observation_query'):
                files = self.run_case(policy, P, decoupled)
                if not decoupled:
                    continue
                total = list(pools_of(P, True))
                baseline = [n - 1 for n in total]
                layout = DS.layout_record(P, True, sessions=1)
                legacy = {k: json.loads(v) for k, v in self.run_case(policy, 784, False).items()
                          if k != 'runtime.jsonl'}
                attached = json.loads(files['worker-attached.json'])
                self.assertEqual(attached['kv_pools'], dict(kv_layout=layout, num_blocks_by_pool=total,
                                                            free_blocks_by_pool=baseline), name)
                ready = json.loads(files['ready.json'])
                self.assertEqual((ready['kv_layout'], ready['num_blocks_by_pool'], ready['free_blocks_by_pool']),
                                 (layout, total, baseline), name)
                self.assertEqual(set(ready) - set(legacy['ready.json']),
                                 {'kv_layout', 'num_blocks_by_pool', 'free_blocks_by_pool'})
                native = json.loads(files['native'])
                self.assertEqual(native['kv_pools'], dict(free_blocks_by_pool=[baseline[0] - 3, baseline[1] - 3]))
                self.assertEqual(set(native) - set(legacy['native']), {'kv_pools'})
                row = json.loads(files['runtime.jsonl'])
                self.assertEqual(row['kv_pools'], native['kv_pools'])
                cleanup = json.loads(files['engine-cleanup.json'])
                self.assertEqual(cleanup['kv_pools'], dict(num_blocks_by_pool=total, free_blocks_by_pool=baseline))
                self.assertEqual(set(cleanup) - set(legacy['engine-cleanup.json']), {'kv_pools'})
                # Both pools are back at their readiness baseline after teardown.
                self.assertEqual(cleanup['kv_pools']['free_blocks_by_pool'], ready['free_blocks_by_pool'])

    def test_design_pool_sizes(self):
        # DESIGN 4.4: 8 GiB, one session: 10 Mamba blocks, 123.2K FA tokens at P = 16.
        fa, mamba = pools_of(16, True)
        self.assertEqual(mamba, 10)
        self.assertEqual(round(fa * 16 / 1000, 1), 123.2)
        fa, mamba = pools_of(16, True, gib=24, sessions=4)
        self.assertEqual((mamba, round(fa * 16 / 1000, 1)), (28, 371.3))

    def test_startup_refuses_a_contract_that_does_not_match_the_layout(self):
        for P, decoupled, contract in ((16, True, sc.default_contract(16)),
                                       (784, False, sc.default_contract(784, vector_format=sc.COMPACT_INT8_V1))):
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(DS, 'contract_for', lambda *a, **k: contract):
                with self.assertRaisesRegex(Exception, 'vector format'):
                    scenario(EVIDENCE, Path(tmp), 'observation_query', P, decoupled)

    def test_decoupled_startup_requires_two_native_pools(self):
        # The config names a decoupled layout, the engine has one shared pool.
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sys.modules[__name__], 'pools_of', lambda *a, **k: None):
            with self.assertRaisesRegex(Exception, 'two native block pools'):
                scenario(EVIDENCE, Path(tmp), 'full', 16, True)

    def test_retry_keeps_the_first_pool_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            scenario(EVIDENCE, output, 'full', 16, True)
            value = config(output, 'full', 16, True)
            native = output / ('native-' + sha256(RID.encode()).hexdigest() + '.json')
            before = native.read_bytes()
            manager = Manager(pools_of(16, True))
            manager.free = [5, 5]
            core = NS(_remote_ra_growing_adapter=NS(page_tokens=16, kv_manager=manager),
                      _remote_ra_live_v2_boundary=NS(prepared={RID: dict(plan=plan())}),
                      _remote_ra_private_v2_endpoint=NS(selection=NS(authorized={RID: None})),
                      _remote_ra_live_session_authority=NS(index=lambda sid: 0))
            binding = NS(request=RID, session='s', epoch=EPOCH, turn=0, local_turn=0, rebuild=False, generation='g0',
                         identity=BINDING, tokens=tuple(range(40)), observation_span=None, stable_end=40)
            entry = dict(binding=binding, drain_receipt=dict(drained=True, operation='complete'), state='complete',
                         admission_engine_ms=2.5)
            endpoint = NS(core=core, worker=NS(http_receipts={RID: frames()}))
            with patch.object(EVIDENCE, 'load', lambda: value), patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'):
                EVIDENCE.native_terminal(endpoint, entry, 'complete')      # another pool sample: kept as written
                self.assertEqual(native.read_bytes(), before)
                entry['admission_engine_ms'] = 3.5
                with self.assertRaisesRegex(Exception, 'changed on retry'):
                    EVIDENCE.native_terminal(endpoint, entry, 'complete')


if __name__ == '__main__':
    unittest.main()
