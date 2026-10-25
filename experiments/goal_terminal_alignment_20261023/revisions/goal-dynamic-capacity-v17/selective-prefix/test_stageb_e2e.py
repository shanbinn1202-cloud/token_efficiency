"""Stage B CPU end to end (M2): real engine + real worker + real Selector V2.

CPU only; nothing here is CUDA evidence or READY.
- Engine: the real SplitPrivateEndpoint (render_claim, rebuild, reserve),
  Phase1Coordinator, EngineCoreAdapter, PrivateBlockOwner, EngineRpcBoundary,
  LiveEndpoint, EngineLifecycle and LiveSessionAuthority over the FakePool and
  fake KV manager of test_stageb_engine (native align-mode Mamba rows added).
- Worker: the real RunnerAllocationOwner, NativeQueryForward, CudaWorkerBackend,
  QwenAlignLifecycle, WorkerBridge and guarded_execute over the FakeRunner,
  Store and FakeIO of test_stageb_worker. Every KV slot holds (position,
  token) and every Mamba slot a hash chain, so each forward proves what it
  attends and that its GDN state is the dense state at its start.
- EngineWorkerProxy reaches worker_dispatch through a fake executor that JSON
  round-trips every payload and receipt (the RPC boundary); one run also uses
  the in-process call of the UniProc executor.
- The real QuerySelectionAuthority (no Goal rewriter) and the real
  mswea_protocol.Transcript on the frontend and in the engine, with the
  character tokenizer of test_mswea_protocol. Every prompt is compared with the
  upstream server's token bookkeeping (test_mswea_protocol.Upstream).
- remote_ra_http.evidence (native_terminal, FrontendEvidence.begin/complete/
  failed, engine_closed) writes into a temporary directory (patched load()).

P = 16. The selection contract is the default one with two geometry scalings:
the switch threshold is 20 Pages (character tokens make a turn about ten
Pages, so local turns 1 and 2 stay below it) and the recent quota keeps its
784-token Page count (2 Pages).
"""
import asyncio
import base64
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_stageb_engine as E      # engine fakes: FakePool, KV managers, Req
import test_mswea_protocol as X     # tokenizer, transcript factory, upstream bookkeeping
import test_stageb_worker as W      # worker fakes (CPU torch): FakeRunner, Store, FakeIO, chain

from contracts import canonical_prompt_hash
from remote_ra_growing.contract import select_random50
from remote_ra_http.config import digest as http_digest
from remote_ra_http.evidence import FrontendEvidence, engine_closed
from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
from remote_ra_live_v2.composition import EngineBridge, EngineWorkerProxy, LiveEndpoint
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
from remote_ra_live_v2.generation_guard import guarded_execute
from remote_ra_live_v2.lifecycle import EngineLifecycle
from remote_ra_live_v2.native_probe import NativeQueryForward
from remote_ra_live_v2.pool_incarnations import install
from remote_ra_live_v2.private_endpoint import SplitPrivateEndpoint
from remote_ra_live_v2.qwen import QwenAlignLifecycle
from remote_ra_live_v2.session_authority import LiveSessionAuthority
from remote_ra_live_v2.worker_backend import CudaWorkerBackend
from remote_ra_live_v2.worker_composition import WorkerBridge, worker_dispatch
from remote_ra_native_growing.adapter import EngineCoreAdapter
from remote_ra_native_growing.contract import Authority
from runtime_renderer import EngineSourceRenderer, PLAIN_GOAL_SCHEMA, XML_SOURCE_SCHEMA

P = E.P                               # 16: selection Pages must be 16-token aligned
EPOCH, SOURCE = 'e' * 64, 'c' * 64
TASK = dict(instance_id='e2e', problem_statement='Fix bug.')
SYSTEM = 'sys'
MAX_TOKENS = 4096                     # max_num_batched_tokens == runner.max_num_tokens
BLOCKS = 400
BYTES = dict(attention=1000, hybrid=7)  # fake H2D bytes per restored FA block / Mamba slot
HYBRID = BYTES['hybrid'] * len(E.MAMBA)
QUERY = ('goal_query', 'observation_query')


def contract():
    value = selection_contract.default_contract(P)
    value.update(switch_threshold_tokens=20 * P,
                 recent_quota_pages=selection_contract.default_recent_quota_pages(784))
    return selection_contract.validate(value)


def config(policy, output):
    return dict(schema='remote-ra-http-config-v2', policy=policy, protocol='mswea_xml_v1',
                selection_contract=contract() if policy in QUERY else None, task=dict(TASK),
                issue_sha256=sha256(TASK['problem_statement'].encode()).hexdigest(),
                page_tokens=P, output_directory=str(output))


class KVManager(E.KVManager):
    """test_stageb_engine's manager with native align-mode behaviour: new FA blocks
    join the native zero queue (take_new_block_ids); a Mamba row gets a fresh
    block per newly reached Page and frees the slots before the one holding the
    state at logical - 1 (MambaManager.remove_skipped_blocks)."""

    def take_new_block_ids(self):
        ids, self.fa.new_block_ids[:] = list(self.fa.new_block_ids), []
        return ids

    def _growing_original_allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
            delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=False,
            reserved_blocks=0, has_scheduled_reqs=True, external_computed_tokens_by_group=None,
            computed_tokens_by_group=None, skip_cache=False):
        rid, cursors = request.request_id, tuple(computed_tokens_by_group)
        self.calls.append(dict(request=rid, start=request.num_computed_tokens, count=num_new_tokens,
                               cursors=cursors, skip_cache=skip_cache,
                               floor=hasattr(self, '_remote_ra_chunk_capacity')))
        pool, null = self.block_pool, self.block_pool.null_block
        need = [-(-(c + num_new_tokens) // P) for c in cursors]
        keep = (cursors[1] - 1) // P if cursors[1] else 0
        for m in self.mamba:
            row = m.req_to_blocks.setdefault(rid, [])
            for i in range(min(keep, len(row))):
                if not row[i].is_null:
                    pool.free_blocks([row[i]])
                    row[i] = null
        fa = self.fa.req_to_blocks.setdefault(rid, [])
        mamba = [g for g, m in zip(E.MAMBA, self.mamba)
                 if len(m.req_to_blocks[rid]) < need[g] or m.req_to_blocks[rid][need[g] - 1].is_null]
        if max(0, need[0] - len(fa)) + len(mamba) > pool.get_num_free_blocks():
            return None
        new = pool.get_new_blocks(max(0, need[0] - len(fa)))
        fa.extend(new)
        self.fa.new_block_ids.extend(b.block_id for b in new)
        for g, m in zip(E.MAMBA, self.mamba):
            row = m.req_to_blocks[rid]
            row.extend([null] * (need[g] - len(row)))
            if row[need[g] - 1].is_null:
                row[need[g] - 1] = pool.get_new_blocks(1)[0]
        return NS(blocks=())


class IO(W.FakeIO):
    """FakeIO plus the view page sizes and joined H2D counters evidence reads."""

    def __init__(self, store):
        super().__init__(store)
        for view in self.views:
            view.page_bytes = BYTES[view.kind]
        self._http_h2d_issued = self._http_h2d_joined = 0

    def restore(self, groups, kind, payload):
        super().restore(groups, kind, payload)
        size = sum(len(ids) for ids in groups.values()) * BYTES[kind]
        self._http_h2d_issued += size
        self._http_h2d_joined += size


class Renderer:
    """Engine renderer seam: the real EngineSourceRenderer.render_growing over the
    real Transcript. Loaded HF tokenizer/template identity checks are out of scope."""
    render_growing = EngineSourceRenderer.render_growing

    def _check_identity(self):
        return None

    def transcript(self):
        return X.transcript(allow_rebuild=True)      # as EngineSourceRenderer.transcript

    def __call__(self, source):
        assert set(source) == {'schema', 'text'} and source['schema'] == PLAIN_GOAL_SCHEMA
        return NS(token_ids=tuple(X.TOK.encode(source['text'])))


def sampling():
    return NS(n=1, temperature=0, max_tokens=1536, min_tokens=0, stop=[], ignore_eos=False,
              stop_token_ids=[X.PAD], eos_token_id=X.EOS, repetition_penalty=1.0,
              presence_penalty=0.0, frequency_penalty=0.0, logit_bias=None, allowed_token_ids=None,
              bad_words=[], structured_outputs=None, logits_processors=None, skip_special_tokens=True)


class Req(E.Req):
    def __init__(self, rid, tokens):
        super().__init__(rid, tokens)
        self.sampling_params = sampling()


class Harness:
    """One engine, one worker, the frontend transcript and evidence of one session."""

    def __init__(self, policy, output, rpc='json'):
        self.policy, self.output, self.rpc_mode = policy, Path(output), rpc
        # ---- worker, as attach_loaded_worker wires it ----------------------
        self.store = W.Store()
        self.runner = W.FakeRunner(self.store)
        self.runner.max_num_tokens = MAX_TOKENS
        self.io = IO(self.store)
        self.owner = RunnerAllocationOwner(self.runner, self.io, dict(enabled=True, source=SOURCE,
            page_tokens=P, loaded_sources_verified=True, source_closure_verified=True))
        self.runner._remote_ra_live_v2_allocation_owner = self.owner
        self.backend = CudaWorkerBackend(self.io, self.owner, source=SOURCE)
        self.owner.bind_backend(self.backend)
        self.driver = NativeQueryForward(self.runner, self.owner, self.backend)
        self.runner._remote_ra_native_query_forward = self.driver
        self.qwen = QwenAlignLifecycle(self.backend, page_tokens=P)
        self.bridge = WorkerBridge(self.qwen, self.owner)
        self.owner.bind_bridge(self.bridge)
        self.runner._remote_ra_live_v2_bridge = self.bridge
        self.runner.execute_model = guarded_execute(self.runner, self.runner.execute_model)
        self.worker = NS(_remote_ra_live_v2_backend=self.backend, _remote_ra_live_v2_qwen=self.qwen)
        # ---- engine, as compose_engine wires it ----------------------------
        self.pool = E.FakePool(BLOCKS)
        install(self.pool)
        self.kv = KVManager(self.pool)
        self.authority = Authority(epoch=EPOCH, loaded_source=SOURCE)
        self.adapter = EngineCoreAdapter(self.authority, self.kv, page_tokens=P,
                                         full_attention_managers=(self.kv.fa,))
        self.proxy = EngineWorkerProxy(NS(collective_rpc=self.collective_rpc),
            dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True))
        self.boundary = EngineRpcBoundary(self.proxy, self.adapter)
        self.adapter.backend = self.boundary
        owner = NS(owner_check=lambda: None, verify_identity=lambda: None,
                   identity=NS(manifest=SOURCE, epoch=EPOCH), session_id='s', policy=policy)
        self.natives, self.requests, self.enqueued = {}, {}, set()
        assembly = NS(renderer=Renderer(), state='active',
                      provider=NS(provenance=NS(page_tokens=P, model='m'), renderer_identity=SOURCE),
                      dispatcher=NS(codec=NS(epoch=EPOCH, identity=SOURCE),
                                    _decode_core=lambda raw: self.natives[raw]))
        core = NS(_remote_ra_growing_owner=owner, _remote_ra_growing_adapter=self.adapter,
                  _remote_ra_assembly=assembly, _remote_ra_live_v2_boundary=self.boundary,
                  _remote_ra_trusted_issue=dict(TASK), preprocess_add_request=self.preprocess,
                  add_request=lambda request, wave: self.enqueued.add(request.request_id),
                  abort_requests=self.abort,
                  scheduler=NS(connector=NS(has_pending_push_work=lambda: False),
                               max_num_scheduled_tokens=MAX_TOKENS,
                               kv_cache_manager=NS(free=self.free)))
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, self.proxy)
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
        core._remote_ra_live_v2_bridge = EngineBridge()
        self.core, self.endpoint = core, core._remote_ra_private_v2_endpoint
        self.sessions, self.selection = core._remote_ra_live_session_authority, self.endpoint.selection
        self.live = LiveEndpoint(core, assembly, self.proxy)
        self.lifecycle = core._remote_ra_live_v2_lifecycle = EngineLifecycle(self.live)
        # ---- frontend --------------------------------------------------------
        self.frontend, self.upstream = X.transcript(allow_rebuild=True), X.Upstream()
        self.evidence = FrontendEvidence(NS(identity=NS(epoch=EPOCH), policy=policy, session_id='s'),
                                         None, {})
        self.evidence.ready = True
        self.baseline = self.pool.get_num_free_blocks()
        self.ordinal = 0
        self.messages = [dict(role='system', content=SYSTEM),
                         dict(role='user', content=TASK['problem_statement'])]
        self.rpc_log, self.marks, self.slots = [], {}, defaultdict(list)
        self.tamper, self.frees = {}, defaultdict(int)
        self.positions = defaultdict(list)       # namespace -> committed logical positions

    # ---- process boundaries ------------------------------------------------
    def collective_rpc(self, method, args, timeout):
        assert method == 'remote_ra_live_v2_dispatch' and timeout == 30
        operation, payload = args
        if self.rpc_mode == 'json':
            payload = json.loads(json.dumps(payload))
        if operation in self.tamper:
            payload = self.tamper.pop(operation)(payload)
        self.rpc_log.append((operation, payload.get('request', payload.get('namespace'))))
        self.marks[(payload.get('request'), operation)] = self.mark()
        receipt = worker_dispatch(self.worker, operation, payload)
        return [json.loads(json.dumps(receipt)) if self.rpc_mode == 'json' else receipt]

    def mark(self):
        return len(self.runner.zeroed), len(self.io.log), len(self.runner.computed)

    def zeroed(self, since, until=None):
        until = until or self.mark()
        runner = {b for batch in self.runner.zeroed[since[0]:until[0]] for b in batch}
        worker = {b for op, kind, keys in self.io.log[since[1]:until[1]] if op == 'zero' for g, b in keys}
        return runner | worker

    def preprocess(self, native):
        request = Req(native.request_id, native.prompt_token_ids)
        self.requests[request.request_id] = request
        return request, 0

    def free(self, request):
        """KVCacheManager.free with the growing hooks, then the runner's finished row."""
        from vllm.v1.core.sched.output import SchedulerOutput
        rid = request.request_id
        self.frees[rid] += 1
        self.enqueued.discard(rid)
        self.adapter.before_free(request)
        self.kv.native_free(rid)
        self.adapter.after_free(request)
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids = {rid}
        self.runner._update_states(cleanup)

    def abort(self, request_ids):
        # Scheduler finish_requests frees only the requests it holds.
        for rid in request_ids:
            if rid in self.enqueued:
                self.enqueued.discard(rid)
                self.free(self.requests[rid])

    @property
    def session(self):
        return self.adapter.sessions['s']

    # ---- one request ---------------------------------------------------------
    def submit(self, messages):
        """Frontend XML render, registered add and EngineLifecycle.admit."""
        tokens, stable, span = self.frontend.render(messages)
        rid = 'r%d-%s' % (self.ordinal, self.policy)
        self.natives[rid.encode()] = NS(request_id=rid, external_req_id='x' + rid,
            prompt_token_ids=list(tokens), sampling_params=NS(n=1), client_index=0,
            current_wave=0, data_parallel_rank=None, mm_features=None, prompt_embeds=None,
            prompt_is_token_ids=None, lora_request=None, pooling_params=None)
        body = dict(request=rid, session='s', turn=self.ordinal, generation='g%d' % self.ordinal,
                    core=base64.b64encode(rid.encode()).decode(), external_id='x' + rid,
                    prompt_hash=canonical_prompt_hash(tokens),
                    source=dict(schema=XML_SOURCE_SCHEMA, messages=[dict(m) for m in messages]))
        wire = json.dumps(dict(model='m', messages=messages)).encode()

        class Raw:
            async def body(self):
                return wire
        handler = NS(_remote_ra_live_four_policy=NS(session=self.session))
        asyncio.run(self.evidence.begin(Raw(), handler))
        s = self.session
        info = dict(rid=rid, tokens=tokens, stable=stable, span=span, body=body,
                    messages=[dict(m) for m in messages], mark=self.mark(), namespace=s.namespace,
                    pins={b.block_id for b in self.adapter.owner.pinned(s.namespace)})
        self.ordinal += 1
        info['view'] = self.lifecycle.admit(body)
        claim = self.endpoint.claims[rid]
        info.update(request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                    namespace=self.session.namespace, ident=(rid, 's', body['turn'], body['generation']))
        return info

    def step(self, request, count, first):
        from vllm.v1.core.sched.output import NewRequestData
        rid = request.request_id
        result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
            delay_cache_blocks=False, num_encoder_tokens=0,
            full_sequence_must_fit=request.status.name == 'WAITING', reserved_blocks=0,
            has_scheduled_reqs=False)
        assert result is not None
        rows = tuple([b.block_id for b in m.req_to_blocks[rid]]
                     for m in self.kv.coordinator.single_type_managers)
        new, produced = [], 0
        if first:
            new = [NewRequestData(rid, list(request.prompt_token_ids), [], sampling(), None, rows,
                                  request.num_computed_tokens, None)]
        else:
            self.runner.set_rows(rid, rows)
            produced = len(self.runner.requests[rid].output_token_ids)
        self.slots[rid].append(rows[1][-1])     # the running Mamba slot of this forward
        self.runner.execute_model(NS(num_scheduled_tokens={rid: count},
            total_num_scheduled_tokens=count, scheduled_new_reqs=new, finished_req_ids=set(),
            new_block_ids_to_zero=self.kv.take_new_block_ids(),
            kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
        request.num_computed_tokens += count
        request.status.name = 'RUNNING'
        request.output_token_ids.extend(self.runner.requests[rid].output_token_ids[produced:])

    def generate(self, info, outputs, chunk='page', stop_after=None):
        """Scheduler loop: Mamba-aligned prefill chunks (or one chunk), then decode."""
        request, N = info['request'], len(info['request'].prompt_token_ids)
        self.runner.samples[request.request_id] = iter(outputs)
        steps = 0
        while request.num_computed_tokens < N and steps != stop_after:
            start = request.num_computed_tokens
            count = min(N, (start // P + 1) * P) - start if chunk == 'page' else N - start
            self.step(request, count, steps == 0)
            steps += 1
        while len(request.output_token_ids) < len(outputs) and steps != stop_after:
            self.step(request, 1, False)
            steps += 1
        info['steps'] = steps
        return steps

    def complete(self, info):
        rid = info['rid']
        info['drain_mark'] = self.mark()
        self.lifecycle.completed(rid)           # complete drain, commit, transcript, native_terminal
        self.free(info['request'])              # native free after the authenticated drain
        generated = list(info['request'].output_token_ids)
        info['committed'] = self.frontend.commit(info['tokens'], generated)
        info['upstream'] = self.upstream.query(info['messages'], generated)
        self.evidence.complete(NS(internal_id=rid, engine_complete=True), NS(id='resp-' + rid))
        info.update(receipt=self.boundary.receipts[(rid, 'complete')],
                    plan=self.boundary.prepared[rid]['plan'], frames=self.proxy.http_receipts[rid],
                    native=self.native_file(rid),
                    row=json.loads((self.output/'runtime.jsonl').read_bytes().splitlines()[-1]))
        return info

    def native_file(self, rid):
        return json.loads((self.output/('native-%s.json' % sha256(rid.encode()).hexdigest())).read_bytes())

    def reply(self, info, target, eos):
        """Reply tokens putting C' = N + g - 1 at ``target`` mod P."""
        N = len(info['tokens'])
        if eos:     # no action marker: generation ends on EOS and the boundary is "\n"
            base = len(X.reply('Done', True))
            return X.reply('Done' + '.' * ((target - (N + base - 1)) % P), True)
        text = 'THOUGHT: %s\n<mswea_bash_command>ls</mswea_bash_command>'
        pad = (target - (N + len(X.reply(text % '')) - 1)) % P
        return X.reply(text % ('x' * pad))

    def turn(self, observation=None, *, target, eos=False, chunk='page', messages=None):
        """One client round trip: render, admit, generate, complete, evidence."""
        if messages is None:
            messages = self.messages + [dict(role='user', content=observation)] * bool(observation)
        info = self.submit(messages)
        outputs = self.reply(info, target, eos)
        self.generate(info, outputs, chunk)
        assert info['request'].output_token_ids == outputs
        self.complete(info)
        # Mini-SWE/LiteLLM may trim the stored reply; the transcript compares strip().
        self.messages = messages + [dict(role='assistant', content=' ' + info['committed']['text'] + '\n')]
        return info

    def close(self):
        self.lifecycle.disconnect()
        self.sessions.close()


# Observation, C' target (mod P), EOS reply, prefill chunking, expected geometry
# (local_turn, C, Q, o, e, N, n, L, tail, tail_candidate, tail_live, fresh Pages).
OUT = '<returncode>%d</returncode>\n<output>\n%s\n</output>'
SCRIPT = [
    (None, 5, False, 'page', (0, 0, 0, 29, 29, 59, 0, 0, False, False, False, ())),
    (OUT % (0, 'src/app.py'), 14, False, 'page',                  # tail live, below threshold
     (1, 117, 112, 120, 184, 214, 7, 112, True, False, True, (8, 9, 10, 11))),
    (OUT % (1, 'Traceback: boom in main'), 0, False, 'page',      # tail candidate, below threshold
     (2, 270, 256, 273, 350, 380, 17, 272, True, True, False, (17, 18, 19, 20, 21))),
    (OUT % (0, 'all tests pass now, good'), 13, False, 'whole',   # C == Q, first selective turn
     (3, 432, 432, 435, 513, 543, 27, 432, False, False, False, (27, 28, 29, 30, 31, 32))),
    (OUT % (0, 'ok ok'), 1, False, 'page',                        # tail candidate, selective
     (4, 605, 592, 608, 667, 697, 38, 608, True, True, False, (38, 39, 40, 41))),
    ('ok', 14, True, 'page',                                      # short observation in the tail Page
     (5, 753, 752, 756, 766, 796, 47, 752, True, False, True, ())),
    (OUT % (0, 'done'), 7, False, 'page',                         # after EOS: o = C + 2, tail candidate
     (6, 814, 800, 816, 874, 904, 51, 816, True, True, False, (51, 52, 53, 54))),
]
REBUILD = ('Format error: expected exactly one action.', 3,       # the retry drops the last reply
           (0, 0, 0, 822, 822, 852, 0, 0, False, False, False, ()))
AFTER = (OUT % (0, 'retry ok'), 9,                                # local turn 1 of the new namespace
         (1, 915, 912, 918, 980, 1010, 57, 912, True, False, True, (58, 59, 60, 61)))
CANCEL = OUT % (0, 'last')


class StageBEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        # The worker fakes read their module Page size at call time; Table's is a class value.
        self.enterContext(patch.object(W, 'P', P))
        self.enterContext(patch.object(W.Table, 'block_size', P))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def harness(self, policy, rpc='json'):
        value = config(policy, self.output)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        return Harness(policy, self.output, rpc)

    # ---- invariants at a turn boundary ---------------------------------------
    def check_turn(self, h, info, expected_geometry):
        rid, b, geo, plan = info['rid'], info['binding'], info['geo'], info['plan']
        s, ns, adapter = h.session, info['namespace'], h.adapter
        full = tuple(b.tokens) + tuple(info['request'].output_token_ids)
        C2 = len(full) - 1
        Q2 = C2 // P * P
        lease_id = plan['phase1']['lease_id'] if plan['phase1'] else None
        self.assertEqual((geo.local_turn, geo.C, geo.Q, geo.o, geo.e, geo.N, geo.n, geo.L, geo.tail,
                          geo.tail_candidate, geo.tail_live, tuple(geo.fresh_pages)), expected_geometry)
        self.assertEqual((b.local_turn, s.local_turn, s.turn, s.namespace), (geo.local_turn,
                         geo.local_turn + 1, b.turn + 1, ns))
        # Token equivalence: frontend == engine == upstream server bookkeeping.
        expected, prompt_end, text = info['upstream']
        self.assertEqual((list(info['tokens']), list(b.tokens)), (expected, expected))
        self.assertEqual(b.observation_span[1] if b.observation_span else b.stable_end, prompt_end)
        self.assertEqual(info['committed']['text'], text)
        self.assertEqual(list(h.frontend.stream), h.upstream.logical)
        self.assertEqual(list(h.endpoint.transcript.stream), h.upstream.logical)
        self.assertEqual(h.endpoint.transcript.turn, s.local_turn)
        # Worker frames: one phase-1 transaction per continuing turn, then phase 2.
        head = ['prepare', 'phase1', 'phase1_drain', 'reserve'] if geo.local_turn else ['prepare', 'reserve']
        ops = [f['operation'] for f in info['frames']]
        self.assertEqual(ops, head + ['allocation_update'] * info['steps'] + ['complete'])
        # Exactly once: phase 1 computes [C, e), phase 2 [e, N + g - 1), nothing else.
        computed = h.runner.computed[info['mark'][2]:info['drain_mark'][2]]
        goal = [x for r, x in computed if r.startswith('goal-')]
        self.assertEqual([x for r, x in computed if r == lease_id], list(range(geo.C, geo.e)) * bool(lease_id))
        self.assertEqual([x for r, x in computed if r == rid], list(range(geo.phase2_start, C2)))
        self.assertEqual(len(computed), len(goal) + C2 - geo.C)
        h.positions[ns] += [x for r, x in computed if r in (rid, lease_id)]
        self.assertEqual(sorted(h.positions[ns]), list(range(C2)))
        # What each forward read: R + tail + run in phase 1, A_t + run in phase 2.
        self.assertEqual(h.runner.plans[rid], (geo.L, geo.L - len(plan['selected']) * P))
        self.assertEqual(h.runner.visible[rid], plan['selected'])
        lease = None
        if lease_id:
            lease = h.driver.leases[lease_id]['payload']['lease']
            self.assertEqual(h.runner.plans[lease_id], (geo.Q, geo.Q - len(plan['resident_before']) * P))
            self.assertEqual(h.runner.visible[lease_id], plan['resident_before'])
            self.assertEqual(sorted(goal), list(range(len(lease['goal']['tokens']))) if lease['goal'] else [])
            # Phase 1 zeroes only its own fresh blocks, never borrowed pins.
            zero = h.zeroed(h.marks[(rid, 'phase1')], h.marks[(rid, 'reserve')])
            owned = set(lease['zero']) | set(lease['goal']['zero'] if lease['goal'] else ())
            self.assertEqual(zero, owned)
            self.assertFalse(zero & (set(lease['borrowed']) | info['pins']))
        else:
            self.assertEqual(goal, [])
        # Phase 2 zeroes restore targets and transients, never pins/retain/handover.
        prepared = h.boundary.prepared[rid]
        ids = lambda pages: {b.block_id for groups in pages.values() for group in groups for b in group}
        target = prepared['target']
        restore = ids({p: target[p] for p in plan['restore']})
        # A rebuild closed the old namespace (its pins are free) before this admission.
        pinned = set() if b.rebuild else info['pins']
        protected = pinned | ids({p: target[p] for p in plan['retain']}) | ids(prepared['run'])
        zero = h.zeroed(h.marks[(rid, 'reserve')], info['drain_mark'])
        self.assertFalse(zero & protected)
        self.assertTrue(restore <= zero)
        calls = [c for c in h.kv.calls if c['request'] == rid]
        self.assertTrue(all(c['skip_cache'] and not c['floor'] for c in calls))
        self.assertEqual(calls[0]['cursors'][0], geo.phase2_start - (geo.L - len(plan['selected']) * P))
        # GDN carry: the dense state after C' - 1, committed on both sides.
        carry = h.backend.carry['s']
        self.assertEqual((carry.computed_end, carry.digest, carry.namespace), (C2, s.boundary, ns))
        self.assertEqual(carry.payload['blocks'], [W.chain(full[:C2])] * len(E.MAMBA))
        self.assertEqual((s.computed_tokens, s.sealed_tokens, s.next_token), (full[:C2], full[:Q2], full[C2]))
        # Catalogs: engine == worker == selection == [0, Q'/P); CPU and GPU contents exact.
        page = lambda p: [(x, full[x]) for x in range(p * P, (p + 1) * P)]
        w = h.owner.sessions[ns]
        self.assertEqual(set(s.catalog), set(range(Q2 // P)))
        self.assertEqual(set(w['catalog']), set(s.catalog))
        if h.policy in QUERY:
            self.assertEqual(set(h.selection.vector_catalog['s']), set(s.catalog))
            self.assertEqual(len(h.selection.turn_ends['s']), s.local_turn)
        for p, payload in w['catalog'].items():
            self.assertEqual(payload['blocks'], [page(p)])
            self.assertEqual(s.catalog[p]['content'], h.io.payload_digest(payload))
        engine_pins = adapter.owner.sessions.get(ns, {})
        self.assertEqual(tuple(sorted(engine_pins)), s.resident)
        self.assertEqual((tuple(w['resident']), h.owner._tail_wire(w['tail'])), (s.resident, s.tail))
        self.assertEqual(adapter.tail_mirror(adapter.owner.tails.get(ns)), s.tail)
        self.assertEqual({p: [[b.block_id for b in g] for g in v] for p, v in engine_pins.items()},
                         {p: h.owner._wire(v) for p, v in w['resident_blocks'].items()})
        self.assertEqual(s.resident, tuple(sorted(set(plan['selected']) | set(range(geo.L // P, Q2 // P)))))
        for p, groups in engine_pins.items():
            self.assertEqual(h.store.kv[(0, groups[0][0].block_id)], page(p))
        self.assertEqual(s.tail is not None, C2 > Q2)
        if s.tail is not None:
            self.assertEqual(h.store.kv[(0, s.tail['block_ids'][0][0])],
                             [(x, full[x]) for x in range(Q2, C2)] + [None] * (Q2 + P - C2))
        # Selection: the plan is the policy's decision (the signed Selector V2 one).
        n = geo.n
        if h.policy == 'full':
            self.assertEqual(plan['selected'], list(range(n)))
        elif h.policy == 'random50':
            self.assertEqual(plan['selected'], list(range(n)) if n < 2 else
                             list(select_random50(n, b.native_prefix)))
            self.assertEqual(len(plan['selected']), n if n < 2 else n // 2)
        else:
            body = h.selection.authorized[rid]['body']
            self.assertIn(rid, h.selection.used)
            self.assertEqual(plan['selected'], body['selection']['selected_pages'])
            self.assertEqual(plan['selected'], sorted(body['ranking'][:body['selected_count']]))
            self.assertEqual(sorted(body['ranking']), list(range(n)))
            # A local turn 0 has no candidates whatever its stable end (spec 0).
            self.assertEqual((body['candidate_pages'], body['local_turn'], body['turn']),
                             (n, geo.local_turn, b.turn))
            active = selection_contract.selection_active(n, geo.local_turn, h.selection.contract)
            self.assertEqual(body['selection']['reason'], None if active else
                             'cold_start' if geo.local_turn == 0 else 'below_switch_threshold')
            forward = h.policy == 'goal_query' and active and not body['goal']['cache_hit']
            self.assertEqual(body['goal']['forward'], forward)
            self.assertEqual(lease is not None and lease['goal'] is not None, forward)
        # Refcounts: every pin held once, the pool at baseline minus the pins.
        pins = adapter.owner.pinned(ns)
        self.assertTrue(all(block.ref_cnt == 1 for block in pins))
        self.assertEqual(len(pins), len(s.resident) + (s.tail is not None))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        self.assertEqual(adapter.owner.block_namespaces, {block.block_id: ns for block in pins})
        self.assertEqual((s.pending, adapter.requests, adapter.owner.leases, h.endpoint.phase1.records,
                          h.enqueued), (None, {}, {}, {}, set()))
        self.assertEqual((h.owner.records, h.owner.claims, h.backend.requests, h.backend.phase1_states,
                          h.backend.latest, h.bridge.assignments, h.runner.requests,
                          h.runner.input_batch.req_ids, h.runner.mamba_state_idx,
                          h.runner._selective_load_plans, h.driver.active),
                         ({}, {}, {}, {}, {}, {}, {}, [], {}, {}, None))
        self.assertTrue(all(state.terminal for state in h.qwen.requests.values()))
        # Evidence: native terminal file and the HTTP runtime row.
        native, row = info['native'], info['row']
        self.assertEqual((native['request_id'], native['turn'], native['local_turn'], native['rebuild'],
                          native['operation'], native['terminal_state']),
                         (rid, b.turn, geo.local_turn, b.rebuild, 'complete', 'complete'))
        self.assertEqual((native['plan'], [f['operation'] for f in native['worker_frames']]),
                         (json.loads(json.dumps(plan)), ops))
        self.assertEqual(native['prompt_token_ids'], list(b.tokens))
        phase1 = [f for f in info['frames'] if f['operation'] == 'phase1']
        self.assertEqual(row['stage_b'], dict(local_turn=geo.local_turn, rebuild=b.rebuild,
            computed_start=geo.C, sealed_start=geo.Q, candidate_end=geo.L,
            observation=[geo.o, geo.e] if geo.local_turn else None, phase2_start=geo.phase2_start,
            prompt_end=geo.N, tail_candidate=geo.tail_candidate, tail_live=geo.tail_live,
            retain=plan['retain'], restore=plan['restore'], release=plan['release'],
            run_pages=list(geo.run_pages),
            phase1=None if not phase1 else dict(start=geo.C, end=geo.e, forward_tokens=geo.e - geo.C,
                goal_forward=lease['goal'] is not None, phase1_ns=phase1[0]['phase1_ns']),
            computed_end=C2, sealed_end=Q2))
        self.assertEqual((row['request_id'], row['ordinal'], row['selected_logical_pages'],
                          row['prompt_token_ids']), (rid, b.turn + 1, plan['selected'], list(b.tokens)))
        # H2D: phase 1 restores one carry; phase 2 the restored Pages, the phase-1
        # state and one latest-state relocation per Mamba slot move.
        moves = sum(1 for a, c in zip(h.slots[rid], h.slots[rid][1:]) if a != c)
        self.assertEqual(row['prepare']['h2d_bytes'], HYBRID * bool(geo.local_turn))
        self.assertEqual(row['restore']['h2d_bytes'], BYTES['attention'] * len(plan['restore']) +
                         HYBRID * (bool(geo.local_turn) + moves))
        if h.policy in QUERY:
            query = row['query']
            self.assertEqual(query['selection']['selected_pages'], plan['selected'])
            tokens = (lease['goal']['tokens'] if lease and lease['goal'] else
                      list(b.tokens[geo.o:geo.e]) if lease and h.policy == 'observation_query' else None)
            self.assertEqual(query['query_token_sha256'], None if tokens is None else http_digest(tokens))
            self.assertEqual(query['observation'] is None, geo.local_turn == 0)
        else:
            self.assertIsNone(row['query'])
        return info

    # ---- scenario ------------------------------------------------------------------
    def run_policy(self, policy, rpc='json'):
        h = self.harness(policy, rpc)
        infos = []
        for observation, target, eos, chunk, expected in SCRIPT:
            infos.append(self.check_turn(h, h.turn(observation, target=target, eos=eos, chunk=chunk),
                                         expected))
        # Tail candidates: the CPU copy of Page Q/P is the phase-1 capture; its block is
        # retained into the prefix when selected, released at commit otherwise.
        candidates = [i for i in infos if i['geo'].tail_candidate]
        self.assertEqual(len(candidates), 3)
        for i in candidates:
            page = i['geo'].Q // P
            self.assertIn(str(page), i['receipt']['new_pages'])
            self.assertIn(page, i['plan']['retain'] if page in i['plan']['selected'] else i['plan']['release'])
        if policy != 'random50':      # Full, and Selector V2's recent safety set
            self.assertTrue(all(i['geo'].Q // P in i['plan']['selected'] for i in candidates))
        # Regression (query_selection.begin_turn): a local turn 0 has n = 0 candidates
        # even with stable_end >= P; checked in check_turn via the signed receipt.
        self.assertGreaterEqual(infos[0]['binding'].stable_end, P)
        # The EOS reply ended local turn 5: its boundary is one token.
        self.assertEqual(infos[5]['committed']['boundary'], tuple(X.TOK.encode('\n')))
        # First selective turn: R' = [0, n) (dense), nothing to restore, R' - A_t released.
        first = next((i for i in infos if i['plan']['selected'] != list(range(i['geo'].n))), None)
        self.assertEqual(first is None, policy == 'full')
        if first is not None:
            plan, geo = first['plan'], first['geo']
            Rp = set(plan['resident_before']) | ({geo.Q // P} if geo.tail_candidate else set())
            self.assertEqual(Rp, set(range(geo.n)))
            self.assertEqual((plan['restore'], plan['release']), ([], sorted(Rp - set(plan['selected']))))
            self.assertTrue(plan['release'])
        # Format-error retry that drops the reply: rebuild, local turn 0, new namespace.
        old = h.session.namespace
        observation, target, expected = REBUILD
        retry = h.messages[:-1] + [dict(role='user', content=observation)]
        info = self.check_turn(h, h.turn(messages=retry, target=target), expected)
        s = h.session
        self.assertTrue(info['binding'].rebuild and info['committed']['rebuild'])
        self.assertEqual((s.ns_generation, s.turn, s.local_turn), (1, len(SCRIPT) + 1, 1))
        self.assertNotEqual(s.namespace, old)
        self.assertEqual(h.sessions.namespace, s.namespace)
        closes = [(op, key) for op, key in h.rpc_log if op == 'close_session']
        self.assertEqual(closes, [('close_session', old)])
        self.assertEqual(h.proxy.http_receipts['session'][-1]['reason'], 'rebuild')
        self.assertFalse(old in h.adapter.owner.sessions or old in h.adapter.owner.tails or
                         old in h.owner.sessions or old in h.adapter.owner.block_namespaces.values())
        self.assertEqual((h.upstream.sessions, h.frontend.sessions, h.endpoint.transcript.sessions), (2, 2, 2))
        self.assertEqual(sorted(h.positions[old]), list(range(infos[-1]['receipt']['computed_end'])))
        observation, target, expected = AFTER
        after = self.check_turn(h, h.turn(observation, target=target), expected)
        if policy in QUERY:   # selection resumes at local turn 1 of the new namespace
            self.assertIsNone(h.selection.authorized[after['rid']]['body']['selection']['reason'])
            self.assertEqual(after['plan']['restore'], [])
        # Cancel after admission: the session is poisoned, committed state unchanged.
        s = h.session
        committed = (s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s'])
        info = h.submit(h.messages + [dict(role='user', content=CANCEL)])
        h.generate(info, X.reply('abc'), stop_after=2)
        receipt = h.lifecycle.cancel(info['ident'])
        self.assertEqual(receipt['state'], 'cancelled')
        h.evidence.failed(NS(internal_id=info['rid']), RuntimeError('client disconnected'))
        self.assertTrue(s.poisoned and h.owner.sessions[s.namespace]['poisoned'])
        self.assertEqual((s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s']), committed)
        ops = [f['operation'] for f in h.proxy.http_receipts[info['rid']]]
        self.assertEqual(ops, ['prepare', 'phase1', 'phase1_drain', 'reserve',
                               'allocation_update', 'allocation_update', 'cancel'])
        native = h.native_file(info['rid'])
        self.assertEqual((native['operation'], native['terminal_state']), ('cancel', 'cancelled'))
        pins = h.adapter.owner.pinned(s.namespace)
        self.assertTrue(all(block.ref_cnt == 1 for block in pins))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        self.assertFalse(h.runner.requests or h.owner.records or h.owner.claims or h.adapter.requests
                         or h.backend.requests or h.backend.phase1_states or h.backend.latest)
        # Both transcripts stay pending after a failed turn: the frontend fails closed
        # first; past it, the engine lifecycle refuses the poisoned session.
        with self.assertRaisesRegex(LiveV2Error, 'neither committed nor abandoned'):
            h.frontend.render(h.messages + [dict(role='user', content=CANCEL)])
        h.frontend.abandon()
        with self.assertRaisesRegex(LiveV2Error, 'poisoned'):
            h.submit(h.messages + [dict(role='user', content=CANCEL)])
        h.evidence.failed(NS(internal_id=None), RuntimeError('refused'))
        # Shutdown close: every block back, both sides empty, cleanup receipt written.
        h.close()
        engine_closed(h.core)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertTrue(all(block.ref_cnt == 0 for block in h.pool.blocks[1:]))
        self.assertEqual(h.rpc_log[-1], ('close_session', s.namespace))
        self.assertEqual(h.proxy.http_receipts['session'][-1]['reason'], 'shutdown')
        self.assertFalse(h.adapter.sessions or h.adapter.owner.sessions or h.adapter.owner.tails or
                         h.adapter.owner.leases or h.adapter.owner.block_namespaces)
        self.assertFalse(h.owner.sessions or h.owner.session_namespaces or h.backend.carry or
                         h.qwen.requests or h.owner.records or h.owner.claims)
        cleanup = json.loads((self.output/'engine-cleanup.json').read_bytes())
        self.assertEqual(len(cleanup['admitted_request_ids']), len(SCRIPT) + 3)
        rows = [json.loads(line) for line in (self.output/'runtime.jsonl').read_bytes().splitlines()]
        self.assertEqual([r['ordinal'] for r in rows], list(range(1, len(SCRIPT) + 3)))
        self.assertEqual(len((self.output/'failures.jsonl').read_bytes().splitlines()), 2)
        self.assertNotIn('append', [op for op, key in h.rpc_log])
        infos += [info, after]
        return h, infos

    def test_full(self):
        h, infos = self.run_policy('full')
        self.assertTrue(all(i['plan']['sparse']['dropped_full_attention_tokens'] == 0 for i in infos
                            if 'plan' in i))

    def test_random50(self):
        h, infos = self.run_policy('random50')
        tail = [i for i in infos if 'plan' in i and i['geo'].tail_candidate]
        unselected = [i for i in tail if i['geo'].Q // P not in i['plan']['selected']]
        self.assertTrue(unselected)                        # its CPU copy came from phase 1
        self.assertTrue(any(i['plan']['restore'] for i in infos if 'plan' in i))
        for i in unselected:
            self.assertIn(i['geo'].Q // P, i['plan']['release'])
            self.assertIn(str(i['geo'].Q // P), i['receipt']['new_pages'])

    def test_goal_query(self):
        h, infos = self.run_policy('goal_query')
        goals = [h.selection.authorized[i['rid']]['body']['goal'] for i in infos if 'plan' in i]
        self.assertTrue(any(g['forward'] for g in goals))
        self.assertTrue(any(g['cache_hit'] for g in goals))
        self.assertTrue(any(i['plan']['restore'] for i in infos if 'plan' in i))
        # reset_session emptied the Goal cache: the first selective turn after the
        # rebuild runs a Goal forward again.
        self.assertEqual((goals[-1]['forward'], goals[-1]['cache_hit']), (True, False))

    def test_observation_query(self):
        h, infos = self.run_policy('observation_query')
        # The endpoint passed the engine owner's policy: no Goal rewriter, never attempted.
        self.assertEqual((h.selection.policy, h.selection.rewriter, h.selection.constructor.rewriter),
                         ('observation_query', None, None))
        goals = [h.selection.authorized[i['rid']]['body']['goal'] for i in infos if 'plan' in i]
        self.assertTrue(goals and not any(g['slm_attempted'] or g['forward'] for g in goals))

    def test_startup_policy_must_be_the_engine_owner_policy(self):
        value = config('goal_query', self.output)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        with self.assertRaisesRegex(Exception, 'startup config and engine owner policies differ'):
            Harness('observation_query', self.output)

    def fault(self, policy, turns, fault):
        """A failed admission of local turn ``turns``: the session is poisoned, every
        native reference is back and the next request is refused."""
        h = self.harness(policy)
        for observation, target, eos, chunk, expected in SCRIPT[:turns]:
            self.check_turn(h, h.turn(observation, target=target, eos=eos, chunk=chunk), expected)
        s = h.session
        committed = (s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s'])
        pins = h.adapter.owner.pinned(s.namespace)
        messages = h.messages + [dict(role='user', content=SCRIPT[turns][0])]
        with self.assertRaisesRegex(Exception, fault(h)):
            h.submit(messages)
        rid = 'r%d-%s' % (turns, policy)
        entry = h.lifecycle.entries[rid]
        self.assertEqual((entry['state'], entry['drain_receipt']['operation']), ('cancelled', 'cancel'))
        self.assertTrue(s.poisoned and h.owner.sessions[s.namespace]['poisoned'])
        self.assertEqual((s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s']), committed)
        self.assertEqual(h.adapter.owner.pinned(s.namespace), pins)
        self.assertTrue(all(block.ref_cnt == 1 for block in pins))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        self.assertFalse(h.adapter.owns(rid) or h.adapter.owner.leases or h.endpoint.phase1.records)
        self.assertEqual((h.owner.records, h.owner.claims, h.backend.requests, h.backend.phase1_states,
                          h.backend.latest, h.runner.requests), ({}, {}, {}, {}, {}, {}))
        self.assertTrue(all(lease in h.driver.drains for lease in h.driver.leases))
        self.assertEqual(h.native_file(rid)['operation'], 'cancel')
        h.evidence.failed(NS(internal_id=rid), RuntimeError('admission failed'))
        h.frontend.abandon()
        with self.assertRaisesRegex(LiveV2Error, 'poisoned'):
            h.submit(messages)
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertFalse(h.adapter.sessions or h.adapter.owner.block_namespaces or h.owner.sessions
                         or h.backend.carry)
        return h, rid

    def test_worker_reserve_refusal_frees_never_enqueued_rows_once(self):
        # Adapter reserve installed rows and staged the plan; the worker refuses the
        # reserve RPC (a retained Page block swapped). The request never reaches the
        # scheduler: LiveEndpoint.after_drain frees its rows through
        # kv_cache_manager.free exactly once (the FakePool asserts on a double free).
        def swap(h):
            def tamper(payload):
                a, b = payload['plan']['retain'][:2]
                payload['target'][str(a)] = payload['target'][str(b)]
                return payload
            h.tamper['reserve'] = tamper
            return 'retained Page block changed'
        h, rid = self.fault('observation_query', 4, swap)
        self.assertEqual(h.frees[rid], 1)
        self.assertNotIn(rid, h.enqueued)
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[rid]],
                         ['prepare', 'phase1', 'phase1_drain', 'cancel'])

    def test_selection_fault_after_phase1_and_goal_forward(self):
        # Spec 3.10 / GPU smoke 11: selection raises after phase 1 (with a Goal
        # forward); phase1c.abort frees the un-adopted run blocks.
        def fail(h):
            original = h.selection.plan_turn
            def plan_turn(*args):
                self.assertIsNotNone(args[4].goal_vector)
                original(*args)
                raise RuntimeError('selection failed')
            h.selection.plan_turn = plan_turn
            return 'selection failed'
        h, rid = self.fault('goal_query', 3, fail)
        self.assertEqual(h.frees[rid], 0)                 # nothing was installed
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[rid]],
                         ['prepare', 'phase1', 'phase1_drain', 'cancel'])

    def test_failed_rebuilt_admission_poisons_after_closing_the_old_namespace(self):
        # The transcript reset and the old namespace close happen in render_claim,
        # before admission; the worker then refuses the rebuilt prepare. The new
        # namespace is poisoned at the lifecycle and nothing leaks.
        h = self.harness('goal_query')
        for observation, target, eos, chunk, expected in SCRIPT[:3]:
            self.check_turn(h, h.turn(observation, target=target, eos=eos, chunk=chunk), expected)
        old = h.session.namespace
        h.tamper['prepare'] = lambda payload: dict(payload, ns_generation=-1)
        retry = h.messages[:-1] + [dict(role='user', content=REBUILD[0])]
        with self.assertRaisesRegex(LiveV2Error, 'prepare identity/assignment mismatch'):
            h.submit(retry)
        s, rid = h.session, 'r3-goal_query'
        self.assertEqual((s.ns_generation, s.local_turn, s.resident, s.tail), (1, 0, (), None))
        self.assertNotEqual(s.namespace, old)
        self.assertEqual(h.lifecycle.sessions['s'], (3, 'poisoned'))
        self.assertEqual(h.lifecycle.entries[rid]['state'], 'cancelled')
        self.assertTrue(h.owner.terminals[rid]['poisoned'])            # tombstone: no late prepare
        self.assertFalse(old in h.owner.sessions or h.owner.session_namespaces or h.backend.carry)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)     # the old pins are free
        self.assertEqual([op for op, key in h.rpc_log][-3:], ['close_session', 'prepare', 'cancel'])
        h.evidence.failed(NS(internal_id=rid), RuntimeError('rebuild refused'))
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertFalse(h.adapter.sessions or h.adapter.owner.block_namespaces or h.owner.sessions)

    def test_goal_query_in_process_rpc(self):
        # UniProc collective_rpc is an in-process call: no JSON copy between the halves.
        self.run_policy('goal_query', rpc='inprocess')


if __name__ == '__main__':
    unittest.main()
