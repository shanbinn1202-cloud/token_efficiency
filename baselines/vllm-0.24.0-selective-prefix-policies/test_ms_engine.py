"""Multi-session stage 1, engine side: one engine serves a session per startup task.

CPU only; nothing here is CUDA evidence or READY. The real engine and worker of
test_stageb_e2e (SplitPrivateEndpoint, LiveSessionAuthority, EngineLifecycle,
Phase1Coordinator, EngineCoreAdapter, QuerySelectionAuthority, RunnerAllocationOwner,
CudaWorkerBackend ...) serve two or three sessions, one task each, interleaved turn
by turn at batch size 1. Commands enter through the production RegisteredRouteBridge
and RegisteredAddDispatcher, so command authentication, the render envelope and the
per-session turn, busy, poison and generation checks are the shipped ones. Each
session has its own frontend transcript and upstream token bookkeeping
(test_mswea_protocol.Upstream), standing in for the frontend side.
"""
import base64
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_stageb_e2e as S        # engine/worker fakes and the single-session harness
import test_stageb_engine as E
import test_mswea_protocol as X
import test_stageb_worker as W

from contracts import canonical_prompt_hash
from registered_add_protocol import RegisteredAddDispatcher
from remote_ra_http.evidence import engine_closed
from remote_ra_live_v2 import sessions as stage1
from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
from remote_ra_live_v2.composition import EngineBridge, EngineWorkerProxy, LiveEndpoint, trusted_tasks
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
from remote_ra_live_v2.generation_guard import guarded_execute
from remote_ra_live_v2.lifecycle import EngineLifecycle
from remote_ra_live_v2.mswea_protocol import MARKER, MarkerStop
from remote_ra_live_v2.native_probe import NativeQueryForward
from remote_ra_live_v2.pool_incarnations import install
from remote_ra_live_v2.private_endpoint import SplitPrivateEndpoint
from remote_ra_live_v2.qwen import QwenAlignLifecycle
from remote_ra_live_v2.registered_route import RegisteredRouteBridge
from remote_ra_live_v2.session_authority import LiveSessionAuthority
from remote_ra_live_v2.worker_backend import CudaWorkerBackend
from remote_ra_live_v2.worker_composition import WorkerBridge
from remote_ra_native_growing.adapter import EngineCoreAdapter
from remote_ra_native_growing.contract import Authority
from runtime_renderer import XML_SOURCE_SCHEMA

P = S.P
SERVER = 's'                       # the owner's session id: session 0
A = dict(S.TASK)
B = dict(instance_id='e2e-b', problem_statement='Parse ISO week dates.')
C = dict(instance_id='e2e-c', problem_statement='Add a dry run flag.')
QUERY = S.QUERY


def config(policy, output, tasks):
    value = S.config(policy, output)
    value['tasks'] = [dict(t) for t in tasks]
    return value


class MultiHarness(S.Harness):
    """One engine and one worker; one session per task and a frontend lane per session.

    The engine is wired as compose_engine wires it, with the startup task list
    from ``trusted_tasks``; ``server`` is the owner's session id (session 0's).
    The single-session methods of test_stageb_e2e.Harness (RPC boundary, native
    free, scheduler steps, replies) are reused unchanged.
    """

    def __init__(self, policy, output, blocks=600, server=SERVER):
        self.policy, self.output, self.rpc_mode = policy, Path(output), 'json'
        # ---- worker, as attach_loaded_worker wires it ----------------------
        self.store = W.Store()
        self.runner = W.FakeRunner(self.store)
        self.runner.max_num_tokens = S.MAX_TOKENS
        self.io = S.IO(self.store)
        self.owner = RunnerAllocationOwner(self.runner, self.io, dict(enabled=True, source=S.SOURCE,
            page_tokens=P, loaded_sources_verified=True, source_closure_verified=True))
        self.runner._remote_ra_live_v2_allocation_owner = self.owner
        self.backend = CudaWorkerBackend(self.io, self.owner, source=S.SOURCE)
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
        self.pool = E.FakePool(blocks)
        install(self.pool)
        self.kv = S.KVManager(self.pool)
        self.authority = Authority(epoch=S.EPOCH, loaded_source=S.SOURCE)
        self.adapter = EngineCoreAdapter(self.authority, self.kv, page_tokens=P,
                                         full_attention_managers=(self.kv.fa,))
        self.proxy = EngineWorkerProxy(NS(collective_rpc=self.collective_rpc),
            dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True))
        self.boundary = EngineRpcBoundary(self.proxy, self.adapter)
        self.adapter.backend = self.boundary
        owner = NS(owner_check=lambda: None, verify_identity=lambda: None,
                   identity=NS(manifest=S.SOURCE, epoch=S.EPOCH), session_id=server, policy=policy)
        self.natives, self.requests, self.enqueued = {}, {}, set()
        assembly = NS(renderer=S.Renderer(), state='active',
                      provider=NS(provenance=NS(page_tokens=P, model='m'), renderer_identity=S.SOURCE),
                      dispatcher=NS(codec=NS(epoch=S.EPOCH, identity=S.SOURCE),
                                    _decode_core=lambda raw: self.natives[raw]))
        task = dict(A)
        core = NS(_remote_ra_growing_owner=owner, _remote_ra_growing_adapter=self.adapter,
                  _remote_ra_assembly=assembly, _remote_ra_live_v2_boundary=self.boundary,
                  _remote_ra_trusted_issue=task, _remote_ra_trusted_issues=trusted_tasks(task),
                  preprocess_add_request=self.preprocess,
                  add_request=lambda request, wave: self.enqueued.add(request.request_id),
                  abort_requests=self.abort,
                  scheduler=NS(connector=NS(has_pending_push_work=lambda: False),
                               max_num_scheduled_tokens=S.MAX_TOKENS,
                               kv_cache_manager=NS(free=self.free)))
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, self.proxy)
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
        core._remote_ra_live_v2_bridge = EngineBridge()
        self.core, self.endpoint = core, core._remote_ra_private_v2_endpoint
        self.sessions, self.selection = core._remote_ra_live_session_authority, self.endpoint.selection
        self.live = LiveEndpoint(core, assembly, self.proxy)
        self.lifecycle = core._remote_ra_live_v2_lifecycle = EngineLifecycle(self.live)
        self.route = RegisteredRouteBridge(self.lifecycle)
        self.dispatcher = RegisteredAddDispatcher(codec=None, boundary=None,
            decode_core=lambda raw: raw, preprocess=lambda request: request)
        # ---- frontend: one lane per session ---------------------------------
        self.lanes = [NS(index=k, sid=sid, task=t, frontend=X.transcript(allow_rebuild=True),
                         upstream=X.Upstream(), turn=0,
                         messages=[dict(role='system', content=S.SYSTEM),
                                   dict(role='user', content=t['problem_statement'])])
                      for k, (sid, t) in enumerate(zip(self.sessions.ids, core._remote_ra_trusted_issues))]
        self.baseline = self.pool.get_num_free_blocks()
        self.ordinal = 0
        self.rpc_log, self.marks, self.slots = [], {}, defaultdict(list)
        self.tamper, self.frees = {}, defaultdict(int)

    # ---- one request ---------------------------------------------------------
    def command(self, lane, tokens, messages, *, session=None, instance=None, turn=None):
        """A registered-add command as the frontend controller builds it for ``lane``."""
        rid = 'r%d-%d' % (self.ordinal, lane.index)
        self.ordinal += 1
        self.natives[rid.encode()] = NS(request_id=rid, external_req_id='x' + rid,
            prompt_token_ids=list(tokens), sampling_params=NS(n=1), client_index=0,
            current_wave=0, data_parallel_rank=None, mm_features=None, prompt_embeds=None,
            prompt_is_token_ids=None, lora_request=None, pooling_params=None)
        sid = lane.sid if session is None else session
        return NS(command_id='c' + rid, request_id=rid, session_id=sid, lease_seconds=60,
                  metadata=dict(instance_id=sid if instance is None else instance,
                                turn=lane.turn if turn is None else turn, generation='g-' + rid),
                  core_b64=base64.b64encode(rid.encode()).decode(), external_id='x' + rid,
                  prompt_hash=canonical_prompt_hash(list(tokens)),
                  source=dict(schema=XML_SOURCE_SCHEMA, messages=[dict(m) for m in messages]))

    def send(self, command):
        return self.route.add(self.dispatcher, command, 'fp-' + command.request_id)

    def submit(self, lane, messages):
        """Frontend XML render of the lane's session, then the registered add."""
        tokens, stable, span = lane.frontend.render(messages)
        command = self.command(lane, tokens, messages)
        rid = command.request_id
        info = dict(rid=rid, lane=lane, tokens=tokens, messages=[dict(m) for m in messages],
                    mark=self.mark(), turn=lane.turn)
        try:
            info['ack'] = self.send(command)
        except BaseException:
            lane.frontend.abandon()
            raise
        claim = self.endpoint.claims[rid]
        info.update(request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                    namespace=self.adapter.sessions[lane.sid].namespace,
                    ident=(rid, lane.sid, lane.turn, command.metadata['generation']))
        return info

    def complete(self, info):
        lane, rid = info['lane'], info['rid']
        info['drain_mark'] = self.mark()
        self.route.finalize(self.dispatcher, rid, True)   # drain, commit, native_terminal
        self.free(info['request'])                         # native free after the drain
        generated = list(info['request'].output_token_ids)
        info['committed'] = lane.frontend.commit(info['tokens'], generated)
        info['upstream'] = lane.upstream.query(info['messages'], generated)
        lane.turn += 1
        info.update(receipt=self.boundary.receipts[(rid, 'complete')],
                    plan=self.boundary.prepared[rid]['plan'], native=self.native_file(rid))
        return info

    def turn(self, lane, observation=None, *, target=5, eos=False, messages=None):
        if messages is None:
            messages = lane.messages + [dict(role='user', content=observation)] * bool(observation)
        info = self.submit(lane, messages)
        outputs = self.reply(info, target, eos)
        self.generate(info, outputs)
        self.complete(info)
        lane.messages = messages + [dict(role='assistant', content=' ' + info['committed']['text'] + '\n')]
        return info

    def script(self, lane, k):
        """Turn ``k`` of the single-session script, played by ``lane``."""
        observation, target, eos, _, _ = S.SCRIPT[k]
        return self.turn(lane, observation, target=target, eos=eos)

    def state(self, lane):
        """Everything a turn of another session must leave untouched."""
        s = self.adapter.sessions[lane.sid]
        ns = s.namespace
        carry = self.backend.carry.get(lane.sid)
        selection = self.selection
        return dict(namespace=ns, turn=s.turn, local_turn=s.local_turn, resident=s.resident,
            tail=s.tail, catalog=dict(s.catalog), boundary=s.boundary, poisoned=s.poisoned,
            pins=[b.block_id for b in self.adapter.owner.pinned(ns)],
            carry=None if carry is None else (carry.namespace, carry.computed_end, carry.digest),
            worker=json.dumps(self.owner.sessions.get(ns), sort_keys=True, default=repr),
            transcript=(self.endpoint.transcripts[lane.sid].stream,
                        self.endpoint.transcripts[lane.sid].turn),
            lifecycle=self.lifecycle.sessions.get(lane.sid),
            vectors=sorted(selection.vector_catalog.get(lane.sid, {})),
            ends=list(selection.turn_ends.get(lane.sid, ())),
            goal=sorted(selection.goal_cache.get(lane.sid, {})))


class MultiSessionEngineTests(unittest.TestCase):
    def setUp(self):
        stubs = W.vllm_stubs()
        request = stubs['vllm.v1.request'] = ModuleType('vllm.v1.request')
        request.RequestStatus = NS(FINISHED_STOPPED='FINISHED_STOPPED')
        self.enterContext(patch.dict(sys.modules, stubs))
        self.enterContext(patch.object(W, 'P', P))
        self.enterContext(patch.object(W.Table, 'block_size', P))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def harness(self, policy, tasks):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = config(policy, output, tasks)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        return MultiHarness(policy, output)

    # ---- invariants -------------------------------------------------------------
    def check(self, h, info):
        """One completed turn of one session: its own tokens, transcript, carry,
        pages, selection tables and evidence; every session stays isolated."""
        lane, b, rid = info['lane'], info['binding'], info['rid']
        s = h.adapter.sessions[lane.sid]
        full = tuple(b.tokens) + tuple(info['request'].output_token_ids)
        C2 = len(full) - 1
        Q2 = C2 // P * P
        # Routed to its own session; turn is that session's request ordinal.
        self.assertEqual((b.session, b.turn, s.turn, lane.turn), (lane.sid, info['turn'],
                         info['turn'] + 1, info['turn'] + 1))
        self.assertEqual(h.lifecycle.sessions[lane.sid], (lane.turn, None))
        # Tokens: frontend == engine == upstream bookkeeping of this session.
        expected, prompt_end, text = info['upstream']
        self.assertEqual((list(info['tokens']), list(b.tokens)), (expected, expected))
        self.assertEqual(b.observation_span[1] if b.observation_span else b.stable_end, prompt_end)
        self.assertEqual(info['committed']['text'], text)
        transcript = h.endpoint.transcripts[lane.sid]
        self.assertEqual(list(lane.frontend.stream), lane.upstream.logical)
        self.assertEqual(list(transcript.stream), lane.upstream.logical)
        self.assertEqual((transcript.turn, transcript.pending), (s.local_turn, None))
        # Namespace, GDN carry (a hash chain of this session's tokens) and pages.
        ns = s.namespace
        self.assertEqual((info['namespace'], h.sessions.namespaces[lane.sid]), (ns, ns))
        carry = h.backend.carry[lane.sid]
        self.assertEqual((carry.computed_end, carry.digest, carry.namespace), (C2, s.boundary, ns))
        self.assertEqual(carry.payload['blocks'], [W.chain(full[:C2])] * len(E.MAMBA))
        self.assertEqual((s.computed_tokens, s.sealed_tokens), (full[:C2], full[:Q2]))
        page = lambda p: [(x, full[x]) for x in range(p * P, (p + 1) * P)]
        w = h.owner.sessions[ns]
        self.assertEqual(set(s.catalog), set(range(Q2 // P)))
        self.assertEqual(set(w['catalog']), set(s.catalog))
        for p, payload in w['catalog'].items():
            self.assertEqual(payload['blocks'], [page(p)])
        for p, groups in h.adapter.owner.sessions.get(ns, {}).items():
            self.assertEqual(h.store.kv[(0, groups[0][0].block_id)], page(p))
        # Selection tables and the signed receipt belong to this session.
        if h.policy in QUERY:
            self.assertEqual(set(h.selection.vector_catalog[lane.sid]), set(s.catalog))
            self.assertEqual(len(h.selection.turn_ends[lane.sid]), s.local_turn)
            body = h.selection.authorized[rid]['body']
            self.assertEqual((body['session'], body['turn']), (lane.sid, b.turn))
            goal = h.selection.constructor._sessions[lane.sid]
            self.assertEqual(goal.instance_id, lane.task['instance_id'])
        # Engine evidence row: session id, index and the per-session turn.
        native = info['native']
        self.assertEqual((native['session_id'], native['session_index'], native['turn'],
                          native['terminal_state']), (lane.sid, lane.index, b.turn, 'complete'))
        self.isolated(h)
        return info

    def isolated(self, h):
        """Disjoint namespaces and pins; engine and worker agree session by session."""
        sessions = dict(h.adapter.sessions)
        names = [s.namespace for s in sessions.values()]
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual({sid: h.sessions.namespaces[sid] for sid in sessions},
                         {sid: s.namespace for sid, s in sessions.items()})
        owned = {}
        for s in sessions.values():
            pins = h.adapter.owner.pinned(s.namespace)
            self.assertTrue(all(block.ref_cnt == 1 for block in pins))
            self.assertEqual(len(pins), len(s.resident) + (s.tail is not None))
            self.assertEqual(tuple(sorted(h.adapter.owner.sessions.get(s.namespace, {}))), s.resident)
            for block in pins:
                self.assertNotIn(block.block_id, owned)
                owned[block.block_id] = s.namespace
        self.assertEqual(h.adapter.owner.block_namespaces, owned)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(owned))
        self.assertEqual((h.adapter.requests, h.adapter.owner.leases, h.endpoint.phase1.records,
                          h.enqueued), ({}, {}, {}, set()))
        self.assertTrue(all(sessions[sid].namespace == ns for sid, ns in h.owner.session_namespaces.items()))
        self.assertEqual(set(h.owner.sessions), set(h.owner.session_namespaces.values()))
        self.assertTrue(all(sessions[sid].namespace == carry.namespace for sid, carry in h.backend.carry.items()))

    # ---- scenarios ------------------------------------------------------------------
    def test_interleaved_turns_of_two_sessions(self):
        for policy in ('goal_query', 'random50'):
            with self.subTest(policy=policy):
                h = self.harness(policy, [A, B])
                a, b = h.lanes
                self.assertEqual(h.sessions.ids, (SERVER, SERVER + '-s1'))
                self.assertEqual([a.sid, b.sid], stage1.session_ids(SERVER, 2))
                infos = defaultdict(list)
                for k in range(5):
                    for lane in (a, b):
                        infos[lane.sid].append(self.check(h, h.script(lane, k)))
                    if k == 1:
                        # Turn checks are per session: B's previous or next turn is refused
                        # before any native work, and B goes on at its own turn.
                        for turn in (b.turn - 1, b.turn + 1):
                            with self.assertRaisesRegex(LiveV2Error, 'session busy/poisoned or stale turn'):
                                h.send(h.command(b, (1,), b.messages, turn=turn))
                        # A's turn under B's id is B's stale turn too.
                        with self.assertRaisesRegex(LiveV2Error, 'stale turn'):
                            h.send(h.command(b, (1,), b.messages, turn=a.turn + 1))
                self.assertEqual([h.adapter.sessions[l.sid].turn for l in (a, b)], [5, 5])
                self.assertEqual(h.lifecycle.sessions, {a.sid: (5, None), b.sid: (5, None)})
                self.assertEqual((h.endpoint.transcripts[a.sid].sessions,
                                  h.endpoint.transcripts[b.sid].sessions), (1, 1))
                if policy == 'goal_query':   # selection became active in both sessions
                    for lane in (a, b):
                        reasons = [h.selection.authorized[i['rid']]['body']['selection']['reason']
                                   for i in infos[lane.sid]]
                        self.assertEqual(reasons[0], 'cold_start')
                        self.assertIn(None, reasons)

    def test_foreign_or_unknown_session_ids_are_rejected(self):
        h = self.harness('full', [A, B])
        a, b = h.lanes
        self.check(h, h.script(a, 0))
        self.check(h, h.script(b, 0))
        before = [h.state(lane) for lane in h.lanes]
        rejected = [dict(session=SERVER + '-s2'),             # not issued by a two-task server
                    dict(session=SERVER + '-s9'), dict(session='http-other'),
                    dict(session=SERVER + '-s1', instance=SERVER)]  # instance_id must name the session
        for case in rejected:
            with self.assertRaisesRegex(LiveV2Error, 'session not initialized by loaded server'):
                h.send(h.command(a, (1,), a.messages, **case))
        # Past authentication, the render envelope names one of the server's sessions.
        command = h.command(a, (1,), a.messages, session=SERVER + '-s9')
        body = dict(request=command.request_id, session=command.session_id, turn=0,
                    generation='g', core=command.core_b64, source=command.source,
                    external_id=command.external_id, prompt_hash=command.prompt_hash)
        with self.assertRaisesRegex(LiveV2Error, 'exact server-owned render envelope required'):
            h.lifecycle.admit(body)
        self.assertEqual([h.state(lane) for lane in h.lanes], before)
        self.assertEqual(set(h.adapter.sessions), {a.sid, b.sid})
        self.assertFalse(h.endpoint.claims.keys() - {'r0-0', 'r1-1'})
        for k in (1, 2):
            for lane in (b, a):
                self.check(h, h.script(lane, k))

    def test_a_request_must_route_to_its_session(self):
        # Three tasks: C's command carrying A's task message fails in render_claim,
        # before any native work. Only C is refused afterwards.
        h = self.harness('observation_query', [A, B, C])
        a, b, c = h.lanes
        for lane in h.lanes:
            self.check(h, h.script(lane, 0))
        before = [h.state(lane) for lane in (a, b)]
        with self.assertRaisesRegex(LiveV2Error, 'request task differs from its session'):
            h.send(h.command(c, (1,), a.messages + [dict(role='user', content=S.SCRIPT[1][0])]))
        self.assertEqual(h.lifecycle.sessions[c.sid], (1, 'poisoned'))
        self.assertEqual(h.endpoint.transcripts[c.sid].pending, None)
        self.assertEqual([h.state(lane) for lane in (a, b)], before)
        with self.assertRaisesRegex(LiveV2Error, 'poisoned'):
            h.script(c, 1)
        for k in (1, 2):
            for lane in (a, b):
                self.check(h, h.script(lane, k))

    def test_disjoint_namespaces_pins_and_carries(self):
        h = self.harness('random50', [A, B, C])
        ids = h.sessions.ids
        self.assertEqual(ids, tuple(stage1.session_ids(SERVER, 3)))
        # All sessions are open from startup, one namespace generation 0 each.
        self.assertEqual(h.sessions.namespaces, {sid: h.adapter._namespace(sid, 'm', S.SOURCE, 0) for sid in ids})
        self.assertEqual(list(h.adapter.sessions), list(ids))
        self.assertEqual(len(set(h.endpoint.transcripts[sid] for sid in ids)), 3)
        self.assertEqual(h.selection.tasks(), [A, B, C])
        for k in range(3):
            for lane in h.lanes:
                self.check(h, h.script(lane, k))
        pins = {lane.sid: {b.block_id for b in h.adapter.owner.pinned(h.sessions.namespaces[lane.sid])}
                for lane in h.lanes}
        self.assertTrue(all(pins.values()))
        self.assertFalse(pins[ids[0]] & pins[ids[1]] or pins[ids[0]] & pins[ids[2]] or
                         pins[ids[1]] & pins[ids[2]])
        self.assertEqual(set(h.backend.carry), set(ids))
        self.assertEqual(len({c.digest for c in h.backend.carry.values()}), 3)

    def test_rebuild_of_one_session_leaves_the_other_intact(self):
        h = self.harness('goal_query', [A, B])
        a, b = h.lanes
        for k in range(4):
            for lane in (a, b):
                self.check(h, h.script(lane, k))
        other = h.state(b)
        old = h.sessions.namespaces[a.sid]
        h.rpc_log.clear()
        # A format-error retry that drops A's last reply: A rebuilds at local turn 0.
        retry = a.messages[:-1] + [dict(role='user', content=S.REBUILD[0])]
        info = self.check(h, h.turn(a, messages=retry, target=S.REBUILD[1]))
        s = h.adapter.sessions[a.sid]
        self.assertTrue(info['binding'].rebuild and info['committed']['rebuild'])
        self.assertEqual((s.ns_generation, s.turn, s.local_turn), (1, 5, 1))
        self.assertNotEqual(s.namespace, old)
        self.assertEqual([key for op, key in h.rpc_log if op == 'close_session'], [old])
        self.assertFalse(old in h.adapter.owner.sessions or old in h.owner.sessions or
                         old in h.adapter.owner.block_namespaces.values())
        self.assertEqual(h.endpoint.transcripts[a.sid].sessions, 2)
        # B: namespace, pins, carry, catalog, transcript and selection tables untouched.
        self.assertEqual(h.state(b), other)
        self.assertEqual(h.endpoint.transcripts[b.sid].sessions, 1)
        self.assertEqual(h.adapter.sessions[b.sid].ns_generation, 0)
        self.check(h, h.script(b, 4))
        self.check(h, h.turn(a, S.AFTER[0], target=S.AFTER[1]))
        self.check(h, h.script(b, 5))

    def test_poisoning_one_session_leaves_the_others_serviceable(self):
        h = self.harness('observation_query', [A, B, C])
        a, b, c = h.lanes
        for k in range(3):
            for lane in h.lanes:
                self.check(h, h.script(lane, k))
        # A is cancelled mid-generation: A is poisoned, its committed state unchanged.
        keys = ('namespace', 'turn', 'resident', 'tail', 'catalog', 'boundary', 'pins', 'carry', 'transcript')
        committed = lambda state: {k: state[k] for k in keys}
        before = committed(h.state(a))
        others = [h.state(b), h.state(c)]
        info = h.submit(a, a.messages + [dict(role='user', content=S.CANCEL)])
        h.generate(info, X.reply('abc'), stop_after=2)
        ack = h.route.cancel(h.dispatcher, NS(command_id='x-' + info['rid'], request_id=info['rid']))
        self.assertEqual(ack.state, 'cancelled')
        a.frontend.abandon()
        s = h.adapter.sessions[a.sid]
        self.assertTrue(s.poisoned and h.owner.sessions[s.namespace]['poisoned'])
        self.assertEqual(h.lifecycle.sessions[a.sid], (3, 'poisoned'))
        self.assertEqual(committed(h.state(a)), before)
        self.assertEqual([h.state(b), h.state(c)], others)
        self.isolated(h)
        # B's admission is refused by the worker: only B is poisoned.
        others = h.state(c)
        h.tamper['prepare'] = lambda payload: dict(payload, ns_generation=-1)
        with self.assertRaisesRegex(LiveV2Error, 'prepare identity/assignment mismatch'):
            h.script(b, 3)
        self.assertEqual(h.lifecycle.sessions[b.sid], (3, 'poisoned'))
        self.assertEqual(h.state(c), others)
        # A and B refuse their next request at once; C keeps serving.
        with self.assertRaisesRegex(LiveV2Error, 'poisoned'):
            h.script(a, 3)
        with self.assertRaisesRegex(LiveV2Error, 'poisoned'):
            h.script(b, 3)
        for k in (3, 4):
            self.check(h, h.script(c, k))

    def test_shutdown_closes_every_session(self):
        h = self.harness('random50', [A, B, C])
        a, b, c = h.lanes
        for k in range(2):
            for lane in h.lanes:
                self.check(h, h.script(lane, k))
        names = dict(h.sessions.namespaces)
        h.rpc_log.clear()
        # B is in flight: close releases A and C now and defers B to its native free.
        info = h.submit(b, b.messages + [dict(role='user', content=S.CANCEL)])
        h.generate(info, X.reply('abc'), stop_after=2)
        h.sessions.close()
        self.assertEqual([key for op, key in h.rpc_log if op == 'close_session'], [names[a.sid], names[c.sid]])
        self.assertFalse(h.sessions.closed)
        self.assertEqual(list(h.adapter.sessions), [b.sid])
        h.lifecycle.disconnect()       # cancels B; its native free closes it
        self.assertTrue(h.sessions.closed)
        self.assertEqual([key for op, key in h.rpc_log if op == 'close_session'],
                         [names[a.sid], names[c.sid], names[b.sid]])
        self.assertTrue(all(r['reason'] == 'shutdown' for r in h.proxy.http_receipts['session']))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertTrue(all(block.ref_cnt == 0 for block in h.pool.blocks[1:]))
        self.assertFalse(h.adapter.sessions or h.adapter.owner.sessions or h.adapter.owner.tails or
                         h.adapter.owner.leases or h.adapter.owner.block_namespaces)
        self.assertFalse(h.owner.sessions or h.owner.session_namespaces or h.backend.carry or
                         h.qwen.requests or h.owner.records or h.owner.claims)
        engine_closed(h.core)
        cleanup = json.loads((h.output/'engine-cleanup.json').read_bytes())
        self.assertEqual(cleanup['session_id'], SERVER)
        self.assertEqual(cleanup['sessions'], [dict(session_id=lane.sid, session_index=lane.index,
                                                    namespace=names[lane.sid]) for lane in h.lanes])
        self.assertEqual(len(cleanup['admitted_request_ids']), 7)
        with self.assertRaisesRegex(LiveV2Error, 'live session closed'):
            h.script(a, 2)

    def test_one_task_config_keeps_one_session_with_the_server_id(self):
        h = self.harness('goal_query', [A])
        self.assertEqual(h.core._remote_ra_trusted_issues, [A])
        self.assertEqual((h.sessions.ids, h.sessions.session, list(h.adapter.sessions)),
                         ((SERVER,), SERVER, [SERVER]))
        self.assertEqual(h.sessions.namespace, h.adapter.sessions[SERVER].namespace)
        self.assertIs(h.endpoint.transcript, h.endpoint.transcripts[SERVER])
        self.assertEqual((h.endpoint.tasks, h.selection.tasks()), ([A], [A]))
        (a,) = h.lanes
        for k in range(3):
            self.check(h, h.script(a, k))
        with self.assertRaisesRegex(LiveV2Error, 'session not initialized by loaded server'):
            h.send(h.command(a, (1,), a.messages, session=SERVER + '-s1'))

    def test_trusted_task_list(self):
        def tasks(value, task=A):
            with patch('remote_ra_http.config.load', lambda: value):
                return trusted_tasks(dict(task))
        self.assertEqual(tasks(dict(task=A, tasks=[A])), [A])
        self.assertEqual(tasks(dict(task=A, tasks=[A, B, C])), [A, B, C])
        with self.assertRaisesRegex(LiveV2Error, 'session 0 must serve the trusted startup issue'):
            tasks(dict(task=B, tasks=[B, A]))
        with self.assertRaisesRegex(LiveV2Error, 'no problem statement may contain another'):
            tasks(dict(task=A, tasks=[A, dict(B, problem_statement=A['problem_statement'] + ' Too.')]))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertEqual(trusted_tasks(dict(A)), [A])

    def test_session_caps_and_per_session_generations(self):
        self.assertEqual(EngineLifecycle(NS()).max_sessions, stage1.MAX_SESSIONS)
        kv = S.KVManager(E.FakePool(8))
        adapter = EngineCoreAdapter(Authority(epoch=S.EPOCH, loaded_source=S.SOURCE), kv,
                                    page_tokens=P, full_attention_managers=(kv.fa,))
        for k in range(stage1.MAX_SESSIONS):
            adapter.open('x%d' % k, 'full', 'm', 'r')
        with self.assertRaisesRegex(Exception, 'session collision/limit'):
            adapter.open('x%d' % stage1.MAX_SESSIONS, 'full', 'm', 'r')
        # Generations: replay and the 256 limit are per session.
        h = self.harness('full', [A, B])
        a, b = h.lanes
        authenticate = lambda lane, generation: h.sessions.authenticate_command(NS(
            session_id=lane.sid, metadata=dict(instance_id=lane.sid, turn=0, generation=generation)))
        for k in range(256):
            authenticate(a, 'g%d' % k)
        with self.assertRaisesRegex(LiveV2Error, 'generation replay/limit'):
            authenticate(a, 'fresh')
        authenticate(b, 'g0')
        with self.assertRaisesRegex(LiveV2Error, 'generation replay/limit'):
            authenticate(b, 'g0')
        self.check(h, h.script(b, 0))
        with self.assertRaisesRegex(LiveV2Error, 'generation replay/limit'):
            h.script(a, 0)

    def test_marker_stop_serves_every_session(self):
        h = self.harness('full', [A, B, C])
        a, b, c = h.lanes
        self.check(h, h.script(a, 0))
        self.check(h, h.script(b, 0))
        # One stop policy for the scheduler: every transcript shares its decoder.
        stop = MarkerStop(h.endpoint.transcript.decode)
        self.assertTrue(all(t.decode == h.endpoint.transcript.decode for t in h.endpoint.transcripts.values()))
        info = h.submit(c, c.messages)                 # C is mid-turn while A and B are idle
        for lane in h.lanes:
            output = X.reply('THOUGHT: %s\n<mswea_bash_command>ls</mswea_bash_' % lane.sid)
            request = NS(output_token_ids=list(output), status=None, stop_reason=None)
            self.assertFalse(stop(request))
            request.output_token_ids += X.TOK.encode('command>')
            self.assertTrue(stop(request))
            self.assertEqual((request.status, request.stop_reason), ('FINISHED_STOPPED', MARKER))
        outputs = h.reply(info, 5, False)
        self.assertTrue(stop(NS(output_token_ids=outputs, status=None, stop_reason=None)))
        h.generate(info, outputs)
        self.check(h, h.complete(info))


if __name__ == '__main__':
    unittest.main()
