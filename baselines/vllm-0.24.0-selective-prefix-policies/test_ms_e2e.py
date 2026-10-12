"""Multi-session stage 1 end to end on CPU: HTTP route, frontend and engine together.

CPU only; nothing here is CUDA evidence or READY. One API server holds one to
eight startup tasks, one session each, over one engine and one worker
(test_ms_engine.MultiHarness: the real engine and worker of test_stageb_e2e).
Every request takes the shipped path between them:

- the packaged OpenAI router (test_http.actual_router), run_owned_contract and
  invoke_contract;
- the real LiveFourPolicyController from compose_frontend, with one real
  mswea_protocol.Transcript per task, and the real FrontendEvidence;
- the real ServingController, FrontendRegistry, TrustedFrontendCapture and
  FrontendBridge, which frame each registered add with the private ProtocolCodec;
- on the engine, the production RegisteredAddDispatcher bound to the lifecycle
  (bind_registered_dispatcher), as compose_engine binds it.

OpenAIServingChat, AsyncLLM, the engine core client and the scheduler loop are
stand-ins. They call the overlay hooks in vLLM's order, and the scheduler's
success path (before_success finalizes, then the native free) and abort path.
Each session's prompts are compared with its own test_mswea_protocol.Upstream,
so interleaving cannot change a session's tokens.
"""
import asyncio
from collections import defaultdict
import contextlib
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_http as H              # actual_router
import test_ms_engine as ME        # MultiHarness and its per-session engine invariants
import test_mswea_protocol as X
import test_stageb_e2e as S
import test_stageb_worker as W

import candidate_guard
import registered_add_timing
import route_runtime as route
from entrypoint_runtime import ShutdownOwner
from frontend_registry import FrontendRegistry, TrustedFrontendCapture, frontend_request_scope
from registered_add_protocol import ProtocolCodec, RegisteredAddDispatcher
from remote_ra_http import mswea_route
from remote_ra_http.evidence import FrontendEvidence, engine_closed, worker_attached
from remote_ra_http.shutdown import accept_closed, manager_shutdown
from remote_ra_live_v2 import hooks, sessions as stage1
from remote_ra_live_v2.composition import compose_frontend
from remote_ra_live_v2.goal_async import run_inline
from remote_ra_live_v2.mswea_protocol import MarkerStop
from remote_ra_live_v2.registered_route import bind_registered_dispatcher
from runtime_renderer import EngineSourceRenderer
from serving_controller import (ServingController, generation_closed, generation_started,
                                prepare_for_serving, render_for_serving, require_shutdown_closed,
                                serving_request_scope, terminal_output)

P = S.P
SERVER = 'http-' + 'a' * 32        # 'http-' + epoch[:32], as native_authority names the server
KEY = b'k' * 32                    # the private channel key of both ends
A, B, C = ME.A, ME.B, ME.C
QUERY = S.QUERY
READY = NS(require_native=lambda: None)
TOKENIZE = NS(truncate_prompt_tokens=None, pad_prompt_tokens=None, truncation_side=None,
              do_lower_case=False, add_special_tokens=False)
FINISH_RESPONSE = mswea_route.finish_response


def observe(k, observation):
    """Session k's version of a script observation: distinct per session, so a
    leak between sessions would change the tokens."""
    if k == 0 or observation is None:
        return observation
    return observation.replace('\n</output>', ' [s%d]\n</output>' % k)


def contains(stream, part):
    part = list(part)
    return any(list(stream[i:i + len(part)]) == part for i in range(len(stream) - len(part) + 1))


async def until(predicate, spins=500):
    for _ in range(spins):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError('condition not reached')


class Engine(ME.MultiHarness):
    """test_ms_engine's engine and worker, reached only through private frames.

    compose_engine binds the assembly's RegisteredAddDispatcher to the lifecycle,
    and render_claim decodes the core request through it. ``admit`` observes
    every admission (the state before it, the claim after it) for the checks.
    """

    def __init__(self, policy, output, server):
        super().__init__(policy, output, server=server)
        self.positions, self.infos, self.events = defaultdict(list), {}, []
        self.codec = ProtocolCodec(key=KEY, epoch=S.EPOCH, identity=S.SOURCE)
        self.dispatcher = RegisteredAddDispatcher(codec=self.codec, boundary=None,
            decode_core=lambda raw: self.natives[raw], preprocess=lambda core: core)
        self.core._remote_ra_assembly.dispatcher = self.dispatcher
        self.route = bind_registered_dispatcher(self.dispatcher, self.lifecycle)
        # Both entries observed: admit (synchronous) runs admit_steps, as the
        # lifecycle's own does; a server parking goal_query admissions enters at
        # admit_steps (goal_async).
        self._admit_steps, self.lifecycle.admit_steps = self.lifecycle.admit_steps, self.admit_steps
        self.lifecycle.admit = self.admit
        # compose_engine: the engine stops every session's request on the action marker.
        self.core.scheduler._remote_ra_output_stop = MarkerStop(self.endpoint.transcript.decode)
        # Startup evidence (compose_engine -> bind_loaded_worker).
        self.core.scheduler._remote_ra_live_v2_bridge = self.core._remote_ra_live_v2_bridge
        worker_attached(self.core, self.proxy, self.core._remote_ra_live_v2_bridge, self.lifecycle,
                        dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', copies_drained=True,
                             loaded_sources_verified=True, tensor_geometry=dict(page_tokens=P),
                             source_closure=dict(source=S.SOURCE)))

    def admit(self, body):
        return run_inline(self.admit_steps(body))

    def admit_steps(self, body):
        """EngineLifecycle.admit_steps, observed: the session before, the claim after."""
        rid, sid = body['request'], body['session']
        s = self.adapter.sessions.get(sid)
        if s is not None:
            self.infos[rid] = dict(rid=rid, sid=sid, turn=body['turn'], mark=self.mark(), namespace=s.namespace,
                                   pins={b.block_id for b in self.adapter.owner.pinned(s.namespace)})
        self.events.append(('admit', sid, body['turn']))
        view = yield from self._admit_steps(body)
        claim = self.endpoint.claims[rid]
        self.infos[rid].update(request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                               namespace=self.adapter.sessions[sid].namespace,
                               ident=(rid, sid, body['turn'], body['generation']))
        return view

    async def run(self, rid, outputs, chunk='page', hold=None):
        """Scheduler loop of one request (test_stageb_e2e.Harness.generate), yielding
        to the frontend loop after every step; ``hold`` parks the request in flight
        before step ``hold.at``. The request stops as the scheduler stops it: on
        EOS/pad, else on the protocol's MarkerStop. Then the success path runs:
        before_success finalizes (drain, commit, native_terminal), then the native free."""
        info = self.infos[rid]
        request, N = info['request'], len(info['request'].prompt_token_ids)
        self.runner.samples[rid] = iter(outputs)
        steps, stopped = 0, False
        while not stopped:
            if hold is not None and steps == hold.at:
                hold.parked.set()
                await hold.gate.wait()
            if request.num_computed_tokens < N:
                start = request.num_computed_tokens
                count = min(N, (start // P + 1) * P) - start if chunk == 'page' else N - start
                self.step(request, count, steps == 0)
            else:
                self.step(request, 1, False)
            steps += 1
            if request.output_token_ids:   # Scheduler._update_request_with_output
                stopped = (request.output_token_ids[-1] in self.endpoint.transcript.stop_ids or
                           hooks.output_stop(self.core.scheduler, request))
            await asyncio.sleep(0)
        assert request.output_token_ids == outputs
        info.update(steps=steps, drain_mark=self.mark())
        self.dispatcher.finalize(rid, successful=True)   # compat EngineLifecycle.before_success
        self.free(request)                                # then the scheduler's native free
        self.events.append(('complete', info['sid'], info['turn']))
        return list(request.output_token_ids)

    def abort_request(self, rid):
        """EngineCore.abort_requests: compat before_free cancels a live request
        (its drain aborts and frees it), then the native free of the rest."""
        record = self.dispatcher._records.get(rid)
        if record is not None and record['state'] not in ('cancelled', 'complete'):
            self.dispatcher._cleanup(rid)
        self.abort([rid])
        info = self.infos.get(rid)
        self.events.append(('abort', info and info['sid'], info and info['turn']))

    def shutdown(self):
        """Native shutdown (shutdown_engine): the compat lifecycle's owned disconnect
        (protocol-v2 lifecycle, dispatcher, session authority), then engine_closed."""
        errors = []
        for action in (self.lifecycle.disconnect, self.dispatcher.disconnect, self.sessions.close):
            try:
                action()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup('engine shutdown failed', errors)
        return engine_closed(self.core)


class LLM:
    """AsyncLLM with the overlay hooks of add_request, _add_request and generate.

    Its engine core client hands each registered-add frame to the engine
    dispatcher (the engine thread) and returns the reply frame."""

    def __init__(self, server):
        self.server, self.renderer = server, server.renderer
        self._remote_ra_dynamic_mode = True
        self.engine_core = NS(client_index=0, client_count=1, _remote_ra_dynamic_mode=True,
                              resources=NS(engine_dead=False), registered_add_async=self.registered_add_async)
        self.output_processor = NS(abort_requests=lambda ids, internal=False: None)
        self.ordinal, self.cleanup_receipt = 0, None

    async def registered_add_async(self, frame, timeout):
        """One acknowledged frame; ``Server.hold_ack`` delays a session's next
        admission ack after the engine has handled the add (admission in progress)."""
        assert type(frame) is bytes and 0 < timeout <= 300    # vLLM's registered-add bound
        await asyncio.sleep(0)
        reply = self.server.engine.dispatcher.handle(frame)
        command = self.server.registry.codec.decode(frame)
        hold = self.server.ack_holds.pop(getattr(command, 'session_id', None), None)
        if hold is not None:
            hold.parked.set()
            await hold.gate.wait()
        return reply

    async def add_request(self, request_id, prompt, params):
        async with frontend_request_scope(self, prompt, request_id, params) as scope:
            # InputProcessor.process_inputs + assign_request_id (a fresh suffix).
            core = NS(request_id='%s-%08x' % (request_id, self.ordinal), external_req_id=request_id,
                      prompt_token_ids=list(prompt['prompt_token_ids']), sampling_params=params,
                      client_index=0, current_wave=0, data_parallel_rank=None, mm_features=None,
                      prompt_embeds=None, prompt_is_token_ids=None, lora_request=None, pooling_params=None)
            self.ordinal += 1
            self._remote_ra_frontend_registry.bind_after_assignment(scope, core)
            queue = NS(request_id=core.request_id)
            bridge = self._remote_ra_registered_add
            assert bridge.handles(core)
            bridge.validate_before_output(core, None)
            try:
                await bridge.submit(core, self.engine_core)
            except BaseException:
                self.output_processor.abort_requests([core.request_id], internal=True)
                raise
            return queue

    async def generate(self, prompt, params, request_id):
        exhausted, queue = False, None
        try:
            queue = await self.add_request(request_id, prompt, params)
            generation_started(self, queue, prompt, params, request_id)
            tokens = await self.server.schedule(queue.request_id, request_id, prompt)
            output = NS(request_id=request_id, finished=True, token_ids=tokens)
            terminal_output(self, queue, output)
            exhausted = True
            return output
        except BaseException:             # cancelled, disconnected or failed: abort
            if queue is not None:
                await self.abort(queue.request_id)
            raise
        finally:
            if queue is not None:
                await generation_closed(self, queue, exhausted)

    async def abort(self, request_id):
        self.output_processor.abort_requests([request_id], internal=True)
        self.server.engine.abort_request(request_id)

    def shutdown(self, timeout=None):
        require_shutdown_closed(self)
        self.cleanup_receipt = self.server.engine.shutdown()


class Serving:
    """OpenAIServingChat._create_chat_completion, nonstreaming, with the overlay hooks."""

    def __init__(self, server):
        self.server = server
        self.engine_client = server.llm
        self.renderer = server.llm.renderer
        self.openai_serving_render = NS(renderer=self.renderer)
        self.ordinal = 0

    async def create_chat_completion(self, request, raw):
        s = self.server
        async with serving_request_scope(self, request, raw) as call:
            assert call is not None
            (_conversation,), (engine_input,) = await render_for_serving(self.openai_serving_render,
                request, [dict(m) for m in request.messages], None, TOKENIZE, prompt_extras=None,
                skip_mm_cache=False)
            request_id = 'chatcmpl-%d' % self.ordinal
            self.ordinal += 1
            s.replies[request_id] = s.plans[id(request)]
            params = NS(n=1, extra_args=None)             # request.to_sampling_params
            engine_input = prepare_for_serving(self, request, engine_input, params, request_id,
                                               lora_request=None, data_parallel_rank=None)
            output = await self.engine_client.generate(engine_input, params, request_id)
            result = s.Completion(request_id)
            result.choices = [NS(message=NS(content=X.TOK.decode(output.token_ids), tool_calls=None),
                                 token_ids=list(output.token_ids), finish_reason='stop', stop_reason=None)]
            s.results[request_id] = result
            return result


def vllm_renderer():
    async def tokenize(prompts, params):
        return [dict(prompts[0])]

    async def process(prompt, arrival, skip_mm_cache=False):
        return dict(type='token', prompt=prompt['prompt'], prompt_token_ids=prompt['prompt_token_ids'])
    return NS(tokenizer=X.TOK, tokenize_prompts_async=tokenize, process_for_engine_async=process,
              model_config=NS(is_encoder_decoder=False, enable_prompt_embeds=False))


class Server:
    """One API server over one engine, wired as remote_ra_linux.live_frontend._bind
    wires it: one frontend transcript per task, shared by the capture factory and
    the live controller."""

    def __init__(self, policy, tasks, output, router, Completion, server=SERVER):
        self.policy, self.tasks, self.output = policy, tasks, Path(output)
        self.router, self.Completion = router, Completion
        self.engine = Engine(policy, output, server)
        source = EngineSourceRenderer.__new__(EngineSourceRenderer)
        source.tokenizer, source._check_identity = X.TOK, lambda: None
        self.renderer = vllm_renderer()
        self.transcripts = tuple(X.transcript(allow_rebuild=True) for _ in tasks)
        capture = TrustedFrontendCapture(renderer=self.renderer, source_renderer=source,
                                         verify_identity=lambda: True, transcripts=self.transcripts)
        self.llm = LLM(self)
        self.registry = FrontendRegistry(llm=self.llm, capture_factory=capture,
            codec=ProtocolCodec(key=KEY, epoch=S.EPOCH, identity=S.SOURCE),
            encoder=NS(encode=self.serialize), verify_identity=lambda: True,
            verify_engine_handshake=lambda: True)
        self.llm._remote_ra_frontend_registry = self.registry
        self.llm._remote_ra_registered_add = self.registry.bridge
        self.handler = Serving(self)
        self.controller = ServingController(serving=self.handler, registry=self.registry,
                                            verify_identity=lambda: True)
        for owner in (self.llm, self.handler, self.handler.openai_serving_render):
            owner._remote_ra_serving_controller = self.controller
        self.live = compose_frontend(self.handler, self.controller, policy=policy, instance_id=server,
                                     readiness=READY, tasks=tasks, transcripts=self.transcripts)
        self.llm._remote_ra_live_four_policy = self.handler._remote_ra_live_four_policy = self.live
        self.evidence = FrontendEvidence(NS(identity=NS(epoch=S.EPOCH), policy=policy, session_id=server),
                                         None, {})
        self.evidence.ready = True
        self.state = NS(engine_client=self.llm, openai_serving_chat=self.handler,
                        enable_server_load_tracking=False, _remote_ra_http_evidence=self.evidence,
                        _remote_ra_chat_route_binding=route.ContractBinding(
                            self.handler, self.controller, server, policy, lambda: True))
        # The agents: one conversation and one upstream bookkeeping per task.
        self.lanes = [NS(index=k, sid=sid, task=task, frontend=transcript, upstream=X.Upstream(), turn=0,
                         messages=[dict(role='system', content=S.SYSTEM),
                                   dict(role='user', content=task['problem_statement'])])
                      for k, (sid, task, transcript) in enumerate(zip(
                          stage1.session_ids(server, len(tasks)), tasks, self.transcripts))]
        self.plans, self.replies, self.holds, self.ack_holds, self.receive = {}, {}, {}, {}, {}
        self.results, self.committed, self.admitted = {}, {}, {}

    # ---- seams --------------------------------------------------------------
    def serialize(self, core):
        """The core-request encoder: the engine decodes a snapshot of these bytes."""
        raw = core.request_id.encode()
        self.engine.natives[raw] = NS(**vars(core))
        return [raw]

    def finish_response(self, result, call, transcript):
        """remote_ra_http.mswea_route.finish_response, recording what it committed."""
        committed = FINISH_RESPONSE(result, call, transcript)
        self.committed[call.internal_id] = committed
        return committed

    async def schedule(self, rid, external, prompt):
        """The engine scheduler serving one admitted request; returns its tokens."""
        h = self.engine
        info = self.admitted[external] = h.infos[rid]
        info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
        target, eos, chunk = self.replies.pop(external)
        outputs = h.reply(info, target, eos)
        return await h.run(rid, outputs, chunk, self.holds.pop(h.sessions.index(info['sid']), None))

    def hold(self, k, at):
        """Park session k's next admitted request in flight before scheduler step ``at``."""
        self.holds[k] = NS(at=at, parked=asyncio.Event(), gate=asyncio.Event())
        return self.holds[k]

    def hold_ack(self, k):
        """Hold the admission ack of session k's next registered add."""
        self.ack_holds[self.lanes[k].sid] = NS(parked=asyncio.Event(), gate=asyncio.Event())
        return self.ack_holds[self.lanes[k].sid]

    def disconnect(self, k):
        self.receive[k].put_nowait(dict(type='http.disconnect'))

    # ---- one HTTP request -----------------------------------------------------
    async def call(self, k, messages, *, target=5, eos=False, chunk='page'):
        """Agent k sends one chat request; on success it appends the reply it got."""
        lane = self.lanes[k]
        receive = asyncio.Queue()
        self.receive.setdefault(k, receive)      # a refused duplicate never takes the slot
        wire = json.dumps(dict(model='m', messages=messages)).encode()

        async def body():
            return wire
        raw = NS(app=NS(state=self.state), headers={}, receive=receive.get, body=body)
        request = NS(messages=[dict(m) for m in messages], tools=None, stream=False, n=1,
                     use_beam_search=False, max_tokens=None, temperature=None, return_token_ids=None,
                     stop_token_ids=[])
        self.plans[id(request)] = (target, eos, chunk)
        try:
            # Bounded: a request stuck behind an unreleased admission fails the test.
            response = await asyncio.wait_for(self.router.create_chat_completion(request, raw), 10)
        finally:
            self.plans.pop(id(request), None)
            if self.receive.get(k) is receive:
                del self.receive[k]
        external = response.content['id']
        result, info = self.results.pop(external), self.admitted.pop(external)
        rid, generated = info['rid'], list(info['request'].output_token_ids)
        text = result.choices[0].message.content
        h = self.engine
        info.update(lane=lane, messages=[dict(m) for m in messages], text=text,
                    committed=self.committed.pop(rid), upstream=lane.upstream.query(messages, generated),
                    receipt=h.boundary.receipts[(rid, 'complete')], plan=h.boundary.prepared[rid]['plan'],
                    frames=h.proxy.http_receipts[rid], native=h.native_file(rid),
                    row=next(r for r in self.rows() if r['request_id'] == rid))
        lane.turn = self.live.sessions[k].turn
        lane.messages = messages + [dict(role='assistant', content=' ' + text + '\n')]
        return info

    async def turn(self, k, observation=None, *, messages=None, **plan):
        lane = self.lanes[k]
        if messages is None:
            messages = lane.messages + [dict(role='user', content=observation)] * bool(observation)
        return await self.call(k, messages, **plan)

    async def play(self, k, i):
        """Turn ``i`` of the single-session script, played by session k."""
        observation, target, eos, chunk, _ = S.SCRIPT[i]
        return await self.turn(k, observe(k, observation), target=target, eos=eos, chunk=chunk)

    def rows(self, name='runtime.jsonl'):
        path = self.output/name
        return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []

    def frontend_state(self, k):
        live, transcript = self.live.sessions[k], self.transcripts[k]
        return dict(turn=live.turn, closed=live.closed, owner=live.owner, stream=list(transcript.stream),
                    local=transcript.turn, pending=transcript.pending, sessions=transcript.sessions,
                    evidence=self.evidence.turns[k])


class MultiSessionEndToEndTests(unittest.IsolatedAsyncioTestCase):
    # Engine invariants of one completed turn, shared with the engine-side tests.
    check_engine = ME.MultiSessionEngineTests.check
    isolated = ME.MultiSessionEngineTests.isolated
    check_turn = S.StageBEndToEndTests.check_turn

    async def asyncSetUp(self):
        stubs = W.vllm_stubs()
        request = stubs['vllm.v1.request'] = ModuleType('vllm.v1.request')
        request.RequestStatus = NS(FINISHED_STOPPED='FINISHED_STOPPED')
        self.enterContext(patch.dict(sys.modules, stubs))
        self.enterContext(patch.object(W, 'P', P))
        self.enterContext(patch.object(W.Table, 'block_size', P))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.router, self.Completion = self.enterContext(H.actual_router())
        self.enterContext(patch.object(route, 'require_installable', lambda: None))
        self.enterContext(patch.object(candidate_guard, 'require_installable', lambda: None))
        self.enterContext(patch.object(registered_add_timing, 'event', lambda *args, **kwargs: None))
        # The engine runs synchronously inside frontend callbacks: no slow-callback noise.
        asyncio.get_running_loop().slow_callback_duration = 60

    def server(self, policy, tasks, server=SERVER):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = ME.config(policy, output, tasks)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = Server(policy, tasks, output, self.router, self.Completion, server)
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        return srv

    # ---- invariants -------------------------------------------------------------
    def check(self, srv, info):
        """One turn through the route: test_ms_engine's engine invariants (tokens,
        transcripts, carry, pages, selection tables, isolation) and the frontend's
        view of the same turn (live session, reply, evidence row)."""
        lane, b = info['lane'], info['binding']
        self.check_engine(srv.engine, info)
        k = lane.index
        live = srv.live.sessions[k]
        expected, _prompt_end, text = info['upstream']
        self.assertEqual((live.turn, live.closed, live.owner, live.active), (lane.turn, False, None, None))
        self.assertEqual((info['text'], srv.evidence.turns[k]), (text, lane.turn))
        row = info['row']
        self.assertEqual((row['session_id'], row['session_index'], row['turn'], row['ordinal'],
                          row['response_id']), (lane.sid, k, b.turn, b.turn + 1, info['external']))
        self.assertEqual((row['prompt_token_ids'], info['native']['prompt_token_ids'],
                          row['selected_logical_pages']), (expected, expected, info['plan']['selected']))
        self.assertTrue(type(row['queue_wait_s']) is float and row['queue_wait_s'] >= 0)
        if srv.policy in QUERY:
            self.assertEqual(row['query']['issue_sha256'],
                             sha256(lane.task['problem_statement'].encode()).hexdigest())
            self.assertEqual(row['query']['selection']['selected_pages'], info['plan']['selected'])
        # No other task's statement ever enters this session's stream.
        for other in srv.lanes:
            if other is not lane:
                self.assertFalse(contains(lane.frontend.stream, X.TOK.encode(other.task['problem_statement'])))
        self.assertTrue(srv.live._holder is None and not srv.live._requests and not srv.evidence.pending)
        return info

    def quiescent(self, srv):
        live = srv.live
        self.assertTrue(not live.admission.locked() and live._holder is None and not live._requests
                        and all(s.owner is None and s.active is None for s in live.sessions)
                        and not srv.controller._issued and not srv.controller._active
                        and not srv.registry._submitted and not srv.evidence.pending)

    # ---- 1. interleaved turns ----------------------------------------------------
    @staticmethod
    def comparable(info):
        """A turn's evidence without its request identity (ids, binding and lease
        digests) and timing."""
        def plan(value):
            value = dict(value, binding=None)
            if value['phase1']:
                value['phase1'] = {k: v for k, v in value['phase1'].items()
                                   if k not in ('lease_id', 'lease_digest', 'receipt_digest')}
            return value
        drop = ('request_id', 'response_id', 'request_index', 'queue_wait_s', 'request_latency_ns',
                'server_run_id', 'native_evidence_sha256')
        row = {k: v for k, v in info['row'].items() if k not in drop}
        if row['stage_b']['phase1']:
            row['stage_b'] = dict(row['stage_b'], phase1=dict(row['stage_b']['phase1'], phase1_ns=None))
        native = info['native']
        return dict(row=row, plan=plan(info['plan']), operations=[f['operation'] for f in native['worker_frames']],
                    native=dict({k: native[k] for k in ('session_id', 'session_index', 'turn', 'local_turn', 'rebuild',
                                                        'prompt_token_ids', 'observation_span', 'stable_end')},
                                plan=plan(native['plan'])))

    async def interleave(self, policy):
        # Session A's script alone on a one-task server: the baseline its interleaved
        # turns must reproduce, row for row (H2D, active KV bytes, plans, selection).
        solo = self.server(policy, [A])
        alone = [self.comparable(await solo.play(0, i)) for i in range(len(S.SCRIPT))]
        srv = self.server(policy, [A, B])
        h = srv.engine
        a, b = srv.lanes
        self.assertEqual(h.sessions.ids, tuple(stage1.session_ids(SERVER, 2)))
        self.assertEqual([x.session_id for x in srv.live.sessions], [a.sid, b.sid])
        infos = defaultdict(list)
        for i in range(len(S.SCRIPT)):
            for k in (0, 1):
                infos[k].append(self.check(srv, await srv.play(k, i)))
        n = len(S.SCRIPT)
        self.assertEqual([(l.turn, h.adapter.sessions[l.sid].turn) for l in srv.lanes], [(n, n)] * 2)
        self.assertEqual([(l.upstream.sessions, l.frontend.sessions, h.endpoint.transcripts[l.sid].sessions)
                          for l in srv.lanes], [(1, 1, 1)] * 2)
        # Disjoint namespaces, pins and GDN carries; each carry is its own session's chain.
        names = [h.sessions.namespaces[l.sid] for l in srv.lanes]
        pins = [{x.block_id for x in h.adapter.owner.pinned(ns)} for ns in names]
        self.assertTrue(names[0] != names[1] and pins[0] and pins[1] and not pins[0] & pins[1])
        self.assertEqual(set(h.backend.carry), {a.sid, b.sid})
        self.assertNotEqual(h.backend.carry[a.sid].digest, h.backend.carry[b.sid].digest)
        # Evidence: one row per turn in admission order; request_index is the server-wide order.
        rows = srv.rows()
        self.assertEqual([(r['session_id'], r['session_index'], r['turn'], r['request_index']) for r in rows],
                         [(l.sid, l.index, i, 2 * i + l.index) for i in range(n) for l in srv.lanes])
        self.assertEqual([r['request_id'] for r in rows],
                         [infos[k][i]['rid'] for i in range(n) for k in (0, 1)])
        self.assertFalse(srv.rows('failures.jsonl'))
        natives = [(i['native']['session_id'], i['native']['session_index'], i['native']['turn'])
                   for k in (0, 1) for i in infos[k]]
        self.assertEqual(natives, [(l.sid, l.index, i) for l in srv.lanes for i in range(n)])
        # Interleaving B changes nothing A sees, decides, moves or is charged for.
        self.assertEqual([self.comparable(i) for i in infos[0]], alone)
        if policy in QUERY:
            # Per-session selection tables: cold start, then active selection in both.
            self.assertEqual(set(h.selection.vector_catalog), {a.sid, b.sid})
            for k, lane in enumerate(srv.lanes):
                bodies = [h.selection.authorized[i['rid']]['body'] for i in infos[k]]
                self.assertTrue(all(x['session'] == lane.sid for x in bodies))
                reasons = [x['selection']['reason'] for x in bodies]
                self.assertEqual(reasons[0], 'cold_start')
                self.assertIn(None, reasons)
                if policy == 'goal_query':
                    goals = [x['goal'] for x in bodies]
                    self.assertTrue(any(g['forward'] for g in goals) and any(g['cache_hit'] for g in goals))
        self.quiescent(srv)
        return srv, infos

    async def test_interleaved_turns_full(self):
        srv, infos = await self.interleave('full')
        self.assertTrue(all(i['plan']['selected'] == list(range(i['geo'].n)) for k in infos for i in infos[k]))

    async def test_interleaved_turns_goal_query(self):
        await self.interleave('goal_query')

    async def test_interleaved_turns_observation_query(self):
        await self.interleave('observation_query')

    # ---- 2. concurrent submission ------------------------------------------------
    async def test_concurrent_requests_queue_fifo_and_same_session_is_refused(self):
        srv = self.server('observation_query', [A, B, C])
        h = srv.engine
        a, b, c = srv.lanes
        for k in range(3):
            self.check(srv, await srv.play(k, 0))
        idle = h.state(c), srv.frontend_state(2)
        hold = srv.hold(0, 2)
        ta = asyncio.create_task(srv.play(0, 1))
        await asyncio.wait_for(hold.parked.wait(), 5)     # A is in flight in the engine
        tb = asyncio.create_task(srv.play(1, 1))
        await until(lambda: srv.live.sessions[1].owner is not None)
        queued = time.monotonic()                         # B is waiting for admission
        await asyncio.sleep(.05)
        # B waits behind A: no capture, no authorization, nothing reached the engine.
        self.assertEqual((b.frontend.pending, srv.live.sessions[1].active), (None, None))
        self.assertEqual(h.events[-1], ('admit', a.sid, 1))
        self.assertEqual(h.lifecycle.sessions[b.sid], (1, None))
        # A second request of A while A is in flight fails at once and touches nothing.
        rpc = len(h.rpc_log)
        for messages in (a.messages + [dict(role='user', content='again')], a.messages[:2]):
            with self.assertRaisesRegex(RuntimeError, 'session %s busy: one request per session' % a.sid):
                await asyncio.wait_for(srv.call(0, messages), 1)
        self.assertEqual((len(h.rpc_log), h.events[-1]), (rpc, ('admit', a.sid, 1)))
        self.assertFalse(any(x.closed for x in srv.live.sessions))
        # C queues behind B, then its client goes away: C closes, nothing else moves.
        tc = asyncio.create_task(srv.play(2, 1))
        await until(lambda: srv.live.sessions[2].owner is not None)
        srv.disconnect(2)
        with self.assertRaises(asyncio.CancelledError):
            await tc
        self.assertEqual([x.closed for x in srv.live.sessions], [False, False, True])
        self.assertEqual((h.state(c), c.frontend.pending), (idle[0], None))
        waited = time.monotonic() - queued
        hold.gate.set()
        ia, ib = await asyncio.gather(ta, tb)
        # FIFO: A completes before B is admitted; both match their own upstream.
        self.assertEqual(h.events[-4:], [('admit', a.sid, 1), ('complete', a.sid, 1),
                                         ('admit', b.sid, 1), ('complete', b.sid, 1)])
        self.check(srv, ia)
        self.check(srv, ib)
        self.assertGreaterEqual(ib['row']['queue_wait_s'], waited)
        self.assertLess(ia['row']['queue_wait_s'], waited)
        self.assertEqual((ia['row']['request_index'], ib['row']['request_index']), (3, 4))
        failures = srv.rows('failures.jsonl')
        self.assertEqual([(f['session_index'], f['turn'], f['request_index'], f['error']) for f in failures],
                         [(None, None, None, 'LiveV2Error')] * 2 + [(2, 1, None, 'CancelledError')])
        self.assertTrue(all('busy' in f['detail'] for f in failures[:2]))
        self.assertEqual(h.state(c), idle[0])
        self.quiescent(srv)
        # A and B go on; C's later requests fail at once and reach nothing.
        for i in (2, 3):
            for k in (1, 0):
                self.check(srv, await srv.play(k, i))
        rpc = len(h.rpc_log)
        with self.assertRaisesRegex(RuntimeError, 'session %s closed by an earlier failure' % c.sid):
            await asyncio.wait_for(srv.play(2, 1), 1)
        self.assertEqual((len(h.rpc_log), h.state(c)), (rpc, idle[0]))
        self.assertEqual([l.turn for l in srv.lanes], [4, 4, 1])

    # ---- 3. rebuild isolation ---------------------------------------------------------
    async def test_rebuild_of_one_session_between_turns_of_another(self):
        srv = self.server('goal_query', [A, B])
        h = srv.engine
        a, b = srv.lanes
        for i in range(4):
            for k in (0, 1):
                self.check(srv, await srv.play(k, i))
        other = h.state(b), srv.frontend_state(1)
        old = h.sessions.namespaces[a.sid]
        h.rpc_log.clear()
        # A format-error retry that drops A's last reply: A rebuilds at local turn 0.
        retry = a.messages[:-1] + [dict(role='user', content=S.REBUILD[0])]
        info = self.check(srv, await srv.turn(0, messages=retry, target=S.REBUILD[1]))
        s = h.adapter.sessions[a.sid]
        self.assertTrue(info['binding'].rebuild and info['committed']['rebuild'])
        self.assertEqual((s.ns_generation, s.turn, s.local_turn, a.turn), (1, 5, 1, 5))
        self.assertNotEqual(s.namespace, old)
        self.assertEqual([key for op, key in h.rpc_log if op == 'close_session'], [old])
        self.assertEqual(h.proxy.http_receipts['session'][-1]['reason'], 'rebuild')
        self.assertFalse(old in h.adapter.owner.sessions or old in h.owner.sessions or
                         old in h.adapter.owner.block_namespaces.values())
        self.assertEqual((a.frontend.sessions, a.upstream.sessions, h.endpoint.transcripts[a.sid].sessions),
                         (2, 2, 2))
        self.assertEqual((info['row']['stage_b']['rebuild'], info['row']['stage_b']['local_turn']), (True, 0))
        # B: transcripts, namespace, pins, carry, catalog, tables and evidence untouched.
        self.assertEqual((h.state(b), srv.frontend_state(1)), other)
        self.assertEqual(h.adapter.sessions[b.sid].ns_generation, 0)
        self.check(srv, await srv.play(1, 4))
        after = self.check(srv, await srv.turn(0, S.AFTER[0], target=S.AFTER[1]))
        self.assertEqual((after['binding'].rebuild, after['geo'].local_turn), (False, 1))
        self.check(srv, await srv.play(1, 5))
        self.assertEqual([(r['session_index'], r['turn']) for r in srv.rows()][-4:],
                         [(0, 4), (1, 4), (0, 5), (1, 5)])
        self.assertEqual((b.frontend.sessions, b.upstream.sessions), (1, 1))
        self.quiescent(srv)

    # ---- 4. failure isolation ------------------------------------------------------
    async def test_cancel_or_failure_closes_only_its_session(self):
        srv = self.server('observation_query', [A, B, C])
        h = srv.engine
        a, b, c = srv.lanes
        for i in range(2):
            for k in range(3):
                self.check(srv, await srv.play(k, i))
        keys = ('namespace', 'turn', 'resident', 'tail', 'catalog', 'boundary', 'pins', 'carry')
        committed = lambda state: {key: state[key] for key in keys}
        before, idle = committed(h.state(a)), (h.state(c), srv.frontend_state(2))
        # A's client disconnects mid-generation while B waits behind A.
        hold = srv.hold(0, 2)
        ta = asyncio.create_task(srv.play(0, 2))
        await asyncio.wait_for(hold.parked.wait(), 5)
        rid = next(r for r, i in h.infos.items() if i['sid'] == a.sid and i['turn'] == 2)
        tb = asyncio.create_task(srv.play(1, 2))
        await until(lambda: srv.live.sessions[1].owner is not None)
        srv.disconnect(0)
        with self.assertRaises(asyncio.CancelledError):
            await ta
        self.check(srv, await tb)
        # A: the engine drained and poisoned it, its committed state unchanged; the
        # frontend closed it. Every native reference of the request is back.
        s = h.adapter.sessions[a.sid]
        self.assertTrue(s.poisoned and h.owner.sessions[s.namespace]['poisoned'])
        self.assertEqual(h.lifecycle.sessions[a.sid], (2, 'poisoned'))
        self.assertEqual(committed(h.state(a)), before)
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[rid]],
                         ['prepare', 'phase1', 'phase1_drain', 'reserve',
                          'allocation_update', 'allocation_update', 'cancel'])
        native = h.native_file(rid)
        self.assertEqual((native['session_id'], native['session_index'], native['turn'], native['operation'],
                          native['terminal_state']), (a.sid, 0, 2, 'cancel', 'cancelled'))
        self.assertEqual(h.dispatcher._records[rid]['state'], 'cancelled')
        self.assertEqual(h.frees[rid], 1)
        self.assertTrue(srv.live.sessions[0].closed)
        self.assertEqual([x.closed for x in srv.live.sessions], [True, False, False])
        self.isolated(h)
        # A later request for A fails at once and reaches nothing.
        rpc, entries = len(h.rpc_log), len(h.lifecycle.entries)
        with self.assertRaisesRegex(RuntimeError, 'session %s closed by an earlier failure' % a.sid):
            await asyncio.wait_for(srv.play(0, 2), 1)
        self.assertEqual((len(h.rpc_log), len(h.lifecycle.entries)), (rpc, entries))
        # B's next admission is refused by the worker: only B is poisoned and closed.
        h.tamper['prepare'] = lambda payload: dict(payload, ns_generation=-1)
        with self.assertRaisesRegex(RuntimeError, 'registered-add rejected: .*prepare identity/assignment mismatch'):
            await srv.play(1, 3)
        self.assertEqual(h.lifecycle.sessions[b.sid], (3, 'poisoned'))
        self.assertEqual([x.closed for x in srv.live.sessions], [True, True, False])
        self.assertEqual((h.state(c), srv.frontend_state(2)), idle)
        self.quiescent(srv)
        failures = srv.rows('failures.jsonl')
        self.assertEqual([(f['session_id'], f['session_index'], f['turn'], f['error']) for f in failures],
                         [(a.sid, 0, 2, 'CancelledError'), (None, None, None, 'LiveV2Error'),
                          (b.sid, 1, 3, 'PolicyRuntimeAbort')])
        self.assertEqual([f['request_index'] for f in failures], [6, None, 8])
        # C keeps serving.
        for i in (2, 3):
            self.check(srv, await srv.play(2, i))
        self.assertEqual([r['request_index'] for r in srv.rows()], [0, 1, 2, 3, 4, 5, 7, 9, 10])

    async def test_disconnect_during_admission_closes_only_its_session(self):
        # The engine has admitted A (phase 1 ran, A is enqueued) but the ack has not
        # reached the frontend when A's client goes away: FrontendBridge's
        # compensation Cancel drains A in the engine. B, queued behind A, goes on.
        srv = self.server('goal_query', [A, B])
        h = srv.engine
        a, b = srv.lanes
        for i in range(3):
            for k in (0, 1):
                self.check(srv, await srv.play(k, i))
        keys = ('namespace', 'turn', 'resident', 'tail', 'catalog', 'boundary', 'pins', 'carry')
        committed = lambda state: {key: state[key] for key in keys}
        before = committed(h.state(a))
        ack = srv.hold_ack(0)
        ta = asyncio.create_task(srv.play(0, 3))
        await asyncio.wait_for(ack.parked.wait(), 5)
        rid = next(r for r, i in h.infos.items() if i['sid'] == a.sid and i['turn'] == 3)
        self.assertEqual((h.dispatcher._records[rid]['state'], rid in h.enqueued), ('accepted', True))
        tb = asyncio.create_task(srv.play(1, 3))
        await until(lambda: srv.live.sessions[1].owner is not None)
        srv.disconnect(0)
        with self.assertRaises(asyncio.CancelledError):
            await ta
        self.assertEqual((h.dispatcher._records[rid]['state'], h.lifecycle.sessions[a.sid]),
                         ('cancelled', (3, 'poisoned')))
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[rid]],
                         ['prepare', 'phase1', 'phase1_drain', 'reserve', 'cancel'])
        self.assertEqual((h.frees[rid], rid in h.enqueued), (1, False))
        self.assertEqual(committed(h.state(a)), before)
        self.check(srv, await tb)
        self.check(srv, await srv.play(1, 4))
        self.assertEqual([x.closed for x in srv.live.sessions], [True, False])
        self.quiescent(srv)

    async def test_engine_faults_after_phase1_leave_phase1_free_for_other_sessions(self):
        # Phase 1 is process-wide: no lease may start while any session's lease
        # lacks its drain. A's admission fails after its phase 1 (Selector V2
        # raises after a Goal forward), then the worker refuses B's reserve after
        # B's phase 1; each fault releases its lease and blocks, so C's phase 1 runs.
        srv = self.server('goal_query', [A, B, C])
        h = srv.engine
        a, b, c = srv.lanes
        for i in range(3):
            for k in range(3):
                self.check(srv, await srv.play(k, i))
        original = h.selection.plan_turn

        def plan_turn(*args):
            self.assertIsNotNone(args[4].goal_vector)
            original(*args)
            raise RuntimeError('selection failed')
        h.selection.plan_turn = plan_turn
        with self.assertRaisesRegex(RuntimeError, 'registered-add rejected: .*selection failed'):
            await srv.play(0, 3)
        h.selection.plan_turn = original

        def swap(payload):
            first, second = payload['plan']['retain'][:2]
            payload['target'][str(first)] = payload['target'][str(second)]
            return payload
        h.tamper['reserve'] = swap
        with self.assertRaisesRegex(RuntimeError, 'registered-add rejected: .*retained Page block changed'):
            await srv.play(1, 3)
        for lane, rid in ((a, 'selection'), (b, 'reserve')):
            self.assertEqual(h.lifecycle.sessions[lane.sid], (3, 'poisoned'))
            failed = next(r for r, i in h.infos.items() if i['sid'] == lane.sid and i['turn'] == 3)
            self.assertEqual([f['operation'] for f in h.proxy.http_receipts[failed]],
                             ['prepare', 'phase1', 'phase1_drain', 'cancel'])
            self.assertEqual(h.frees[failed], 0 if rid == 'selection' else 1)
        self.assertFalse(h.endpoint.phase1.records or h.adapter.owner.leases or h.driver.active)
        self.assertTrue(all(lease in h.driver.drains for lease in h.driver.leases))
        self.isolated(h)
        self.assertEqual([x.closed for x in srv.live.sessions], [True, True, False])
        for i in (3, 4):
            info = self.check(srv, await srv.play(2, i))
            self.assertEqual(info['frames'][1]['operation'], 'phase1')

    async def test_failed_rebuild_of_one_session_leaves_the_other_intact(self):
        # A's rebuild closes A's old namespace in render_claim; the worker then
        # refuses the rebuilt prepare. Only A is poisoned; B keeps its pages.
        srv = self.server('observation_query', [A, B])
        h = srv.engine
        a, b = srv.lanes
        for i in range(3):
            for k in (0, 1):
                self.check(srv, await srv.play(k, i))
        other, old = (h.state(b), srv.frontend_state(1)), h.sessions.namespaces[a.sid]
        h.tamper['prepare'] = lambda payload: dict(payload, ns_generation=-1)
        retry = a.messages[:-1] + [dict(role='user', content=S.REBUILD[0])]
        with self.assertRaisesRegex(RuntimeError, 'registered-add rejected: .*prepare identity/assignment mismatch'):
            await srv.turn(0, messages=retry, target=S.REBUILD[1])
        s = h.adapter.sessions[a.sid]
        self.assertEqual((s.ns_generation, s.local_turn, s.resident, s.tail), (1, 0, (), None))
        self.assertEqual((h.sessions.namespaces[a.sid], h.lifecycle.sessions[a.sid]), (s.namespace, (3, 'poisoned')))
        self.assertNotEqual(s.namespace, old)
        self.assertFalse(old in h.owner.sessions or old in h.adapter.owner.block_namespaces.values())
        self.assertEqual((h.state(b), srv.frontend_state(1)), other)
        self.isolated(h)
        for i in (3, 4):
            self.check(srv, await srv.play(1, i))
        names = dict(h.sessions.namespaces)
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        cleanup = json.loads((srv.output/'engine-cleanup.json').read_bytes())
        self.assertEqual(cleanup['sessions'], [dict(session_id=l.sid, session_index=l.index,
                                                    namespace=names[l.sid]) for l in srv.lanes])

    async def test_eight_sessions_through_every_layer(self):
        # MAX_SESSIONS tasks: every per-process cap on the path admits all of them.
        tasks = [A, B, C] + [dict(instance_id='e2e-%d' % k, problem_statement='Task %d: fix module m%d.' % (k, k))
                             for k in range(3, stage1.MAX_SESSIONS)]
        srv = self.server('random50', tasks)
        h = srv.engine
        self.assertEqual(h.sessions.ids, tuple(stage1.session_ids(SERVER, stage1.MAX_SESSIONS)))
        order = list(range(stage1.MAX_SESSIONS))
        for i in range(2):
            for k in (order if i == 0 else order[::-1]):
                self.check(srv, await srv.play(k, i))
        self.assertEqual(len(h.lifecycle.sessions), stage1.MAX_SESSIONS)
        rows = srv.rows()
        turns = [(0, k) for k in order] + [(1, k) for k in order[::-1]]
        self.assertEqual([(r['session_index'], r['turn'], r['request_index']) for r in rows],
                         [(k, i, n) for n, (i, k) in enumerate(turns)])
        names = dict(h.sessions.namespaces)
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        cleanup = json.loads((srv.output/'engine-cleanup.json').read_bytes())
        self.assertEqual([(x['session_id'], x['session_index'], x['namespace']) for x in cleanup['sessions']],
                         [(l.sid, l.index, names[l.sid]) for l in srv.lanes])

    # ---- 5. shutdown --------------------------------------------------------------------
    async def test_shutdown_closes_and_drains_every_session(self):
        srv = self.server('goal_query', [A, B, C])
        h = srv.engine
        a, b, c = srv.lanes
        attached = json.loads((srv.output/'worker-attached.json').read_bytes())
        self.assertEqual(attached['session_ids'], [a.sid, b.sid, c.sid])
        for i in range(3):
            for k in (0, 1):
                self.check(srv, await srv.play(k, i))
        names = dict(h.sessions.namespaces)
        h.rpc_log.clear()
        # B is in flight and C queued behind it when the frontend shuts down.
        hold = srv.hold(1, 1)
        tb = asyncio.create_task(srv.play(1, 3))
        await asyncio.wait_for(hold.parked.wait(), 5)
        tc = asyncio.create_task(srv.play(2, 0))
        await until(lambda: srv.live.sessions[2].owner is not None)
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        with self.assertRaises(asyncio.CancelledError):
            await tb
        with self.assertRaisesRegex(RuntimeError, 'serving controller closed'):
            await tc
        # Every namespace closed once for shutdown; every block back; both halves empty.
        closes = [key for op, key in h.rpc_log if op == 'close_session']
        self.assertEqual(sorted(closes), sorted(names.values()))
        self.assertEqual([r['reason'] for r in h.proxy.http_receipts['session']][-3:], ['shutdown'] * 3)
        self.assertTrue(h.sessions.closed and h.lifecycle.closed)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertTrue(all(block.ref_cnt == 0 for block in h.pool.blocks[1:]))
        self.assertFalse(h.adapter.sessions or h.adapter.owner.sessions or h.adapter.owner.tails or
                         h.adapter.owner.leases or h.adapter.owner.block_namespaces)
        self.assertFalse(h.owner.sessions or h.owner.session_namespaces or h.backend.carry or
                         h.qwen.requests or h.owner.records or h.owner.claims)
        self.assertEqual(h.lifecycle.sessions[b.sid], (3, 'poisoned'))
        self.assertNotIn(c.sid, h.lifecycle.sessions)           # C never reached the engine
        cleanup = json.loads((srv.output/'engine-cleanup.json').read_bytes())
        self.assertEqual(srv.llm.cleanup_receipt, sha256((srv.output/'engine-cleanup.json').read_bytes()).hexdigest())
        self.assertEqual(cleanup['session_id'], SERVER)
        self.assertEqual(cleanup['sessions'], [dict(session_id=l.sid, session_index=l.index,
                                                    namespace=names[l.sid]) for l in srv.lanes])
        self.assertEqual(len(cleanup['admitted_request_ids']), 7)
        self.assertTrue(srv.controller._closed and srv.registry._closed and not srv.registry._submitted)
        self.assertEqual([x.closed for x in srv.live.sessions], [False, True, True])
        failures = srv.rows('failures.jsonl')
        self.assertEqual([(f['session_index'], f['turn'], f['error']) for f in failures],
                         [(1, 3, 'CancelledError'), (2, 0, 'PolicyRuntimeAbort')])
        self.assertEqual(len(srv.rows()), 6)
        # The parent accepts the engine's CLOSED receipt; its own receipt names every session.
        manager = NS(close_lock=threading.Lock(), reader=None, watchdog=None, error=None,
                     revoked=threading.Event(), done=threading.Event(), pipe=NS(close=lambda: None),
                     child_pipe=NS(close=lambda: None), session_id=SERVER,
                     process=NS(join=lambda timeout: None, is_alive=lambda: False, exitcode=0),
                     cleanup_path=srv.output/'engine-cleanup.json',
                     identity=NS(epoch=S.EPOCH, manifest=S.SOURCE))
        accept_closed(manager, dict(cleanup_sha256=srv.llm.cleanup_receipt))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(manager_shutdown(manager), srv.llm.cleanup_receipt)
        parent = json.loads((srv.output/'parent-cleanup.json').read_bytes())
        self.assertEqual((parent['session_id'], parent['sessions']), (SERVER, [l.sid for l in srv.lanes]))
        with self.assertRaisesRegex(RuntimeError, 'controller closed'):
            await asyncio.wait_for(srv.play(0, 3), 1)

    # ---- 6. one task: the single-session end to end -------------------------------------
    async def test_one_task_server_is_the_single_session_e2e(self):
        # Server id 's', as in test_stageb_e2e: its check_turn holds unchanged.
        srv = self.server('goal_query', [A], server='s')
        h = srv.engine
        (lane,) = srv.lanes
        h.frontend, h.upstream = lane.frontend, lane.upstream     # check_turn's single-session names
        self.assertEqual((h.sessions.ids, lane.sid, srv.live.session.session_id), (('s',), 's', 's'))
        infos = [self.check_turn(h, await srv.play(0, i), S.SCRIPT[i][4]) for i in range(len(S.SCRIPT))]
        self.assertEqual(infos[5]['committed']['boundary'], tuple(X.TOK.encode('\n')))
        old = h.session.namespace
        retry = lane.messages[:-1] + [dict(role='user', content=S.REBUILD[0])]
        info = self.check_turn(h, await srv.turn(0, messages=retry, target=S.REBUILD[1]), S.REBUILD[2])
        self.assertTrue(info['binding'].rebuild and info['committed']['rebuild'])
        self.assertEqual((h.session.ns_generation, h.session.turn, h.session.local_turn), (1, len(S.SCRIPT) + 1, 1))
        self.assertNotEqual(h.session.namespace, old)
        self.assertEqual((h.upstream.sessions, h.frontend.sessions, h.endpoint.transcript.sessions), (2, 2, 2))
        after = self.check_turn(h, await srv.turn(0, S.AFTER[0], target=S.AFTER[1]), S.AFTER[2])
        self.assertIsNone(h.selection.authorized[after['rid']]['body']['selection']['reason'])
        goals = [h.selection.authorized[i['rid']]['body']['goal'] for i in infos + [info, after]]
        self.assertTrue(any(g['forward'] for g in goals) and any(g['cache_hit'] for g in goals))
        rows = srv.rows()
        self.assertEqual([(r['session_id'], r['session_index'], r['turn'], r['ordinal'], r['request_index'])
                          for r in rows], [('s', 0, t, t + 1, t) for t in range(len(S.SCRIPT) + 2)])
        # A client disconnect after admission: the session is poisoned, then closed.
        s = h.session
        committed = (s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s'])
        hold = srv.hold(0, 2)
        task = asyncio.create_task(srv.turn(0, S.CANCEL))
        await asyncio.wait_for(hold.parked.wait(), 5)
        srv.disconnect(0)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(s.poisoned and srv.live.session.closed)
        self.assertEqual((s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s']), committed)
        with self.assertRaisesRegex(RuntimeError, 'session s closed by an earlier failure'):
            await asyncio.wait_for(srv.turn(0, S.CANCEL), 1)
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        cleanup = json.loads((srv.output/'engine-cleanup.json').read_bytes())
        self.assertEqual((cleanup['session_id'], cleanup['sessions']),
                         ('s', [dict(session_id='s', session_index=0, namespace=s.namespace)]))
        self.assertEqual(len(cleanup['admitted_request_ids']), len(S.SCRIPT) + 3)
        self.assertEqual(len(srv.rows('failures.jsonl')), 2)
        self.quiescent(srv)


if __name__ == '__main__':
    unittest.main()
