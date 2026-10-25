"""Multi-session stage 2 end to end on CPU: several sessions decode in one vLLM step.

CPU only; nothing here is CUDA evidence or READY. The stage-1 end-to-end path of
test_ms_e2e (packaged OpenAI router, invoke_contract, the real
LiveFourPolicyController, ServingController, FrontendRegistry, FrontendBridge,
signed private frames, RegisteredAddDispatcher, RegisteredRouteBridge with the
capacity gate, EngineLifecycle, the real engine of test_ms_engine and the real
worker over test_stageb_worker's FakeRunner), with the stage-2 controller
(``max_num_seqs`` = number of tasks) and a stub of vLLM's engine core instead of
one scheduler loop per request:

- ``BatchEngine.serve`` is EngineCoreProc.run_busy_loop: every pass runs the
  shipped ``_process_input_queue``/``_remote_ra_hold`` (extracted from core.py;
  at most one registered-add per pass), then one step.
- One step is Scheduler.schedule + execute_model + update_from_output: every
  running request first, then waiting ones (FCFS), each through
  ``EngineCoreAdapter.allocate_slots`` (one allocation_update RPC each), then ONE
  ``execute_model`` with all of them (the FakeRunner flattens them in persistent
  batch order, drops unscheduled rows and re-adds them, condenses), then the
  stop checks; a stopped request finalizes (compat before_success) and is
  natively freed, and its runner row leaves with the next step's
  ``finished_req_ids``, as in vLLM. A running request that finds no blocks would
  be preempted: the stub raises (preemption is fatal).
- Admission (render, Goal, phase 1, selection, reserve) runs on that loop between
  steps, so a phase-1 probe drops and re-adds the running requests' rows.
- The runner reorders its persistent batch decode-first, as vLLM does for the
  GDN/Mamba backends (reorder_batch_threshold = 1), and the batched servers use
  the text-only three-row MRoPE positions of Qwen3.6.

Each session's prompts and outputs are compared with its own
test_mswea_protocol.Upstream, and each request's evidence (plans, frames, H2D and
active KV bytes) with the same session's turns run alone on the stage-1 server.

Cases: two and three sessions decoding together (full, goal_query) with
admissions mid-decode; a phase-1 probe while two others decode; a Page crossing
inside a batched step; a rebuild while others decode, with the decode-first
reorder; two completions in one step followed by a probe; CAPACITY_WAIT while
another session runs (200 blocks) and none at 600 blocks; the gate with exact
generation bounds on a small pool; admission passed on at acceptance in FIFO
order; a cancel between batched steps and a refused completion inside one;
shutdown with requests in flight and while a request waits for capacity; and
every stage-1 case of test_ms_e2e at max_num_seqs = 1 on the same engine loop.
"""
import ast
import asyncio
from collections import defaultdict
from copy import deepcopy
import functools
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_ms_e2e as E2E          # route + frontend + engine harness of stage 1
import test_ms_engine as ME        # engine harness and per-turn invariants
import test_stageb_e2e as S
import test_stageb_worker as W

from registered_add_protocol import RegisteredAdd
from remote_ra_http import mswea_route
from remote_ra_live_v2 import capacity, goal_async, hooks
from remote_ra_live_v2.composition import compose_frontend
from serving_controller import require_shutdown_closed

P = S.P
SERVER = E2E.SERVER
A, B, C = ME.A, ME.B, ME.C
QUERY = S.QUERY
KINDS = NS(**{k: k for k in ('WAKEUP', 'REGISTERED_ADD', 'ADD', 'ABORT', 'UTILITY')})


LOOP = ('_process_input_queue', '_remote_ra_hold', '_remote_ra_goal_built', '_remote_ra_resume')


def engine_input_loop():
    """EngineCoreProc._process_input_queue, _remote_ra_hold and the goal_async
    resume (_remote_ra_goal_built, _remote_ra_resume), extracted from core.py."""
    tree = ast.parse((ROOT/'source-overlay/vllm/v1/engine/core.py').read_text('utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCoreProc')
    namespace = dict(EngineCoreRequestType=KINDS, queue=queue, DEBUG=10,
                     logger=NS(isEnabledFor=lambda level: False, debug=lambda *a: None))
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in LOOP:
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'core.py', 'exec'), namespace)
    return tuple(namespace[name] for name in LOOP)


class DecodeFirstRunner(W.FakeRunner):
    """FakeRunner whose persistent batch is reordered as vLLM's GDN/Mamba
    attention backends ask (reorder_batch_threshold = 1): after _update_states
    condenses, reorder_batch_to_split_decodes_and_prefills swaps rows into
    decode, short extend, long extend, first prefill. The worker hooks then read
    rows, positions and hidden slices in this order, as on the GPU."""
    reordered = 0                                   # swaps done, for the tests

    def _update_states(self, so):
        super()._update_states(so)
        scheduled = getattr(so, 'num_scheduled_tokens', None)
        batch = self.input_batch.req_ids
        if not scheduled or not batch:
            return
        regions = []
        for rid in batch:
            req = self.requests[rid]
            start, count = req.num_computed_tokens, scheduled[rid]
            regions.append(3 if start == 0 else 2 if count > 1 else
                           1 if start < len(req.prompt_token_ids) else 0)
        target = sorted(regions)
        moves = [i for i, (r, t) in enumerate(zip(regions, target)) if r != t]
        order = sorted(moves, key=lambda i: regions[i])         # stable, as np.argsort(kind='stable')
        targets = dict(zip(order, moves))
        for src in list(targets):
            dst = targets[src]
            while src != dst:
                self.swap(src, dst)
                self.reordered += 1
                nxt = targets.get(dst, dst)
                targets[dst] = dst
                dst = nxt

    def swap(self, i, j):
        """InputBatch.swap_states: request ids and block-table rows."""
        batch = self.input_batch.req_ids
        batch[i], batch[j] = batch[j], batch[i]
        for table in self._tables():
            table.rows[i], table.rows[j] = table.rows[j], table.rows[i]


class Preempted(RuntimeError):
    """A running request found no free block: vLLM would preempt it (fatal here)."""


class BatchEngine(E2E.Engine):
    """test_ms_e2e's engine and worker behind a stub of vLLM's engine core.

    ``batched``: a finished request's runner row leaves with the next step's
    ``finished_req_ids`` (vLLM). Without it (one running sequence), it leaves at
    the native free, as in the stage-1 harness.
    """
    _process_input_queue, _remote_ra_hold, _remote_ra_goal_built, _remote_ra_resume = engine_input_loop()
    _remote_ra_goal_admissions = None

    def __init__(self, policy, output, server, *, blocks=600, batched=True, mrope=False, goals=False):
        sized = functools.partialmethod(ME.MultiHarness.__init__, blocks=blocks)
        runner = DecodeFirstRunner if batched else W.FakeRunner
        with patch.object(ME.MultiHarness, '__init__', sized), patch.object(W, 'FakeRunner', runner):
            super().__init__(policy, output, server)
        if mrope:   # Qwen3.6: text-only MRoPE, three interleaved position rows
            self.runner.uses_mrope = True
            self.runner.model_config = NS(hf_text_config=NS(rope_parameters=dict(
                mrope_interleaved=True, mrope_section=[11, 11, 10])))
        self.batched = batched
        self.core.add_request = self.add_request
        self.input_queue, self.aborts_queue = queue.Queue(), queue.Queue()
        self.process_input_queue_block, self.serving = False, 'running'
        self.waiting, self.running = [], []         # Scheduler.waiting / running, by request id
        self.plans, self.futures, self.stepped = {}, {}, defaultdict(int)
        self.finished, self.succeeded = set(), set()
        self.batches, self.admissions, self.completed = [], [], []
        self.pause = self.watch = self.dead = self.exact = None
        self.low = self.pool.get_num_free_blocks()
        get = self.pool.get_new_blocks

        def tracked(n):                            # the pool's low-water mark
            blocks = get(n)
            self.low = min(self.low, self.pool.get_num_free_blocks())
            return blocks
        self.pool.get_new_blocks = tracked
        self.wakeup = asyncio.Event()
        if goals:
            # compose_engine's goal_async attachment; a built Goal wakes the loop
            # from the Goal thread, as core.input_queue's WAKEUP does.
            loop = asyncio.get_running_loop()
            self.wakes = 0

            def wake():
                self.wakes += 1
                loop.call_soon_threadsafe(self.wakeup.set)
            attached = goal_async.attach(self.core, self.lifecycle, max_num_seqs=2 if batched else 1, wake=wake)
            assert attached is not None and attached is self.core._remote_ra_goal_admissions
            self._remote_ra_goal_admissions = attached

    # ---- EngineCoreProc ----------------------------------------------------------
    def is_running(self):
        return self.serving == 'running'

    def has_work(self):
        return bool(self.running) or any(rid in self.plans for rid in self.waiting)

    def _notify_idle_state_callbacks(self):
        pass

    def _handle_client_request(self, kind, payload):
        """EngineCore._handle_registered_add: the dispatcher's reply goes to the
        frontend's future; a goal_query admission parked at its Goal (goal_async)
        replies when it ends (``_remote_ra_ack``)."""
        assert kind == 'REGISTERED_ADD', kind
        frame, future = payload
        if self.watch:
            self.watch('handle', frame)
        goals = self._remote_ra_goal_admissions
        try:
            if goals is None:
                reply = self.dispatcher.handle(frame)
            else:
                reply = goals.handle(self.dispatcher, frame, functools.partial(self._remote_ra_ack, frame, future))
                if reply is goals.PARKED:
                    return
        except BaseException as failure:
            if not future.done():
                future.set_exception(failure)
            raise
        self._remote_ra_ack(frame, future, reply, None)

    def _remote_ra_ack(self, frame, future, reply, failure):
        if failure is not None:
            if not future.done():
                future.set_exception(RuntimeError(str(failure)))
            return
        if self.watch:
            self.watch('handled', (frame, reply))
        if not future.done():      # the frontend may have gone away; the frame was sent
            future.set_result(reply)

    async def serve(self):
        """run_busy_loop: poll the input queue, then step; ``pause`` parks it at a pass."""
        try:
            while self.is_running():
                pause = self.pause
                if pause is not None and not pause.gate.is_set() and pause.when(self):
                    pause.parked.set()
                    await pause.gate.wait()
                    if self.pause is pause:
                        self.pause = None
                    continue
                self._process_input_queue()
                if self.has_work() and self.step():
                    await asyncio.sleep(0)
                elif self.input_queue.empty() and getattr(self, '_remote_ra_held', None) is None:
                    self.wakeup.clear()
                    try:
                        await asyncio.wait_for(self.wakeup.wait(), 0.01)
                    except TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0)
        except BaseException as failure:
            self.dead = failure     # EngineDeadError: every waiting generate fails
            for future in self.futures.values():
                if not future.done():
                    future.set_exception(RuntimeError('engine core died: %r' % failure))
            raise

    def park(self, when):
        """Park the loop at the first pass boundary where ``when(engine)`` holds."""
        self.pause = NS(when=when, parked=asyncio.Event(), gate=asyncio.Event())
        return self.pause

    # ---- Scheduler -----------------------------------------------------------------
    def add_request(self, request, wave):
        """EngineCore.add_request: the request joins the waiting queue."""
        rid = request.request_id
        self.enqueued.add(rid)
        self.waiting.append(rid)
        self.wakeup.set()

    async def generate(self, rid, outputs, chunk, hold):
        """The stub's sampling plan for an enqueued request; resolves at completion.
        A waiting request is scheduled only once its plan exists."""
        assert rid in self.waiting or rid in self.running, rid
        self.runner.samples[rid] = iter(outputs)
        self.plans[rid] = NS(outputs=list(outputs), chunk=chunk, hold=hold)
        future = self.futures[rid] = asyncio.get_running_loop().create_future()
        self.wakeup.set()
        return await future

    def held(self, rid):
        hold = self.plans[rid].hold
        if hold is None or self.stepped[rid] != hold.at or hold.gate.is_set():
            return False
        hold.parked.set()
        return True

    def progress(self, rid):
        request = self.requests[rid]
        N = len(request.prompt_token_ids)
        return NS(rid=rid, sid=self.infos[rid]['sid'], start=request.num_computed_tokens, N=N,
                  outputs=len(request.output_token_ids),
                  phase='prefill' if request.num_computed_tokens < N else 'decode')

    def step(self):
        """One schedule + execute_model + update_from_output; False if nothing ran."""
        from vllm.v1.core.sched.output import NewRequestData
        order, counts, starts, produced, news = [], {}, {}, {}, []
        for rid in list(self.running) + [r for r in self.waiting if r in self.plans]:
            if self.held(rid):
                continue
            request = self.requests[rid]
            N, start = len(request.prompt_token_ids), request.num_computed_tokens
            waiting = request.status.name == 'WAITING'
            if start < N:
                chunk = self.plans[rid].chunk
                count = min(N, (start // P + 1) * P) - start if chunk == 'page' else N - start
            else:
                count = 1
            result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
                new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=waiting,
                reserved_blocks=0, has_scheduled_reqs=waiting and bool(self.running))
            if result is None:
                if waiting:
                    break                  # stays waiting (FCFS)
                raise Preempted('running request %s found no free block' % rid)
            rows = tuple([b.block_id for b in m.req_to_blocks[rid]]
                         for m in self.kv.coordinator.single_type_managers)
            if waiting:
                news.append(NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None,
                                           rows, start, None))
                self.waiting.remove(rid)
                self.running.append(rid)
                produced[rid] = 0
            else:
                self.runner.set_rows(rid, rows)
                produced[rid] = len(self.runner.requests[rid].output_token_ids)
            self.slots[rid].append(rows[1][-1])     # the running Mamba slot of this forward
            order.append(rid)
            counts[rid], starts[rid] = count, start
        if not order:
            return False
        finished, self.finished = self.finished, set()
        before = {rid: self.watch('step', rid) for rid in order} if self.watch else None
        mark = len(self.io.log)
        self.runner.execute_model(NS(num_scheduled_tokens=dict(counts),
            total_num_scheduled_tokens=sum(counts.values()), scheduled_new_reqs=news,
            finished_req_ids=finished, new_block_ids_to_zero=self.kv.take_new_block_ids(),
            kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
        self.batches.append(NS(rows=tuple((rid, starts[rid], counts[rid]) for rid in order),
                               batch=tuple(self.runner.input_batch.req_ids), io=self.io.log[mark:],
                               before=before,
                               after={rid: self.watch('step', rid) for rid in order} if self.watch else None))
        for rid in order:                          # update_from_output, in schedule order
            request = self.requests[rid]
            request.num_computed_tokens += counts[rid]
            request.status.name = 'RUNNING'
            new = self.runner.requests[rid].output_token_ids[produced[rid]:]
            request.output_token_ids.extend(new)
            self.stepped[rid] += 1
            if new and (request.output_token_ids[-1] in self.endpoint.transcript.stop_ids or
                        hooks.output_stop(self.core.scheduler, request)):
                self.finish(rid)                   # before_success, then the native free
        return True

    def finish(self, rid):
        """The success path: compat before_success finalizes (drain, commit,
        native_terminal), then the native free. A failed completion is the compat
        FINISHED_ERROR path: cancel drain and native free; this request alone fails."""
        request, info = self.requests[rid], self.infos[rid]
        assert request.output_token_ids == self.plans[rid].outputs, rid
        info.update(steps=self.stepped[rid], drain_mark=self.mark())
        try:
            self.dispatcher.finalize(rid, successful=True)
        except BaseException as failure:
            future = self.futures.pop(rid, None)
            if future is not None and not future.done():
                future.set_exception(failure)
            self.abort_request(rid)
            return
        self.succeeded.add(rid)
        self.completed.append((len(self.batches), rid))
        self.free(request)
        self.events.append(('complete', info['sid'], info['turn']))
        future = self.futures.pop(rid, None)
        if future is not None and not future.done():
            future.set_result(list(request.output_token_ids))

    def free(self, request):
        """KVCacheManager.free with the growing hooks. Batched: the runner row
        leaves with the next step's finished_req_ids."""
        rid = request.request_id
        for line in (self.running, self.waiting):
            if rid in line:
                line.remove(rid)
        if rid not in self.succeeded:
            future = self.futures.pop(rid, None)
            if future is not None and not future.done():
                future.set_exception(RuntimeError('request %s aborted in the engine' % rid))
        if not self.batched:
            return super().free(request)
        self.frees[rid] += 1
        self.enqueued.discard(rid)
        self.adapter.before_free(request)
        self.kv.native_free(rid)
        self.adapter.after_free(request)
        self.finished.add(rid)

    def preprocess(self, native):
        request, wave = super().preprocess(native)
        if self.exact is not None:      # the generation bound of the gate, per request
            self.exact[id(request.sampling_params)] = self.exact[id(native.sampling_params)]
        return request, wave

    def flush(self):
        """Finished rows leave the runner (as the next step's finished_req_ids would)."""
        if self.finished:
            finished, self.finished = self.finished, set()
            self.runner._update_states(NS(finished_req_ids=finished, new_block_ids_to_zero=[],
                                          scheduled_new_reqs=[]))

    def admit_steps(self, body):
        """EngineLifecycle.admit_steps, observed with the requests running when it starts."""
        running = [self.progress(rid) for rid in self.running]
        if self.watch:
            self.watch('admit', body)
        view = yield from super().admit_steps(body)
        self.admissions.append(NS(rid=body['request'], sid=body['session'], turn=body['turn'],
                                  running=running, local_turn=self.infos[body['request']]['geo'].local_turn))
        if self.watch:
            self.watch('admitted', body)
        return view

    def shutdown(self):
        """Native shutdown: queued input is handled, then the stage-1 owned disconnect."""
        self.serving = 'shutting_down'
        self._process_input_queue()
        self.wakeup.set()
        receipt = super().shutdown()
        self.flush()
        return receipt

    def busy(self):
        return bool(self.adapter.requests or self.endpoint.phase1.records or self.enqueued or
                    self.adapter.owner.leases)


class BatchLLM(E2E.LLM):
    """AsyncLLM whose engine core client queues each registered-add frame for the
    engine loop (EngineCoreRequestType.REGISTERED_ADD) and awaits its reply."""

    async def registered_add_async(self, frame, timeout):
        assert type(frame) is bytes and 0 < timeout <= 300    # vLLM's registered-add bound
        h, srv = self.server.engine, self.server
        command = srv.registry.codec.decode(frame)
        srv.frames.append((type(command).__name__, command.request_id, timeout))
        send = srv.send_holds.pop(command.session_id, None) if type(command) is RegisteredAdd else None
        if send is not None:
            send.parked.set()
            await send.gate.wait()
        assert h.dead is None and h.is_running(), 'engine core is not serving'
        future = asyncio.get_running_loop().create_future()
        h.input_queue.put_nowait((KINDS.REGISTERED_ADD, (frame, future)))
        h.wakeup.set()
        reply = await future
        hold = srv.ack_holds.pop(getattr(command, 'session_id', None), None)
        if hold is not None:
            hold.parked.set()
            await hold.gate.wait()
        return reply

    def shutdown(self, timeout=None):
        require_shutdown_closed(self)
        self.cleanup_receipt = self.server.engine.shutdown()


class BatchServer(E2E.Server):
    """test_ms_e2e's API server over BatchEngine; ``seqs`` is the pinned max_num_seqs."""

    def __init__(self, policy, tasks, output, router, Completion, server=SERVER, *, seqs, blocks=600,
                 mrope=False, goals=False):
        self.frames, self.send_holds = [], {}
        engine = functools.partial(BatchEngine, blocks=blocks, batched=seqs > 1, mrope=mrope, goals=goals)
        frontend = functools.partial(compose_frontend, max_num_seqs=seqs)
        with patch.object(E2E, 'Engine', engine), patch.object(E2E, 'LLM', BatchLLM), \
                patch.object(E2E, 'compose_frontend', frontend):
            super().__init__(policy, tasks, output, router, Completion, server)
        self.seqs = seqs
        self.loop = asyncio.create_task(self.engine.serve())

    async def schedule(self, rid, external, prompt):
        """The engine serving one admitted request; returns its tokens."""
        h = self.engine
        info = self.admitted[external] = h.infos[rid]
        info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
        target, eos, chunk = self.replies.pop(external)
        outputs = h.reply(info, target, eos)
        return await h.generate(rid, outputs, chunk, self.holds.pop(h.sessions.index(info['sid']), None))

    def serialize(self, core):
        raw = super().serialize(core)
        exact = self.engine.exact
        if exact is not None:           # this request's exact output length
            target, eos, _ = self.replies[core.external_req_id]
            exact[id(core.sampling_params)] = len(self.engine.reply(dict(tokens=core.prompt_token_ids),
                                                                   target, eos))
        return raw

    def hold_send(self, k):
        """Hold session k's next registered-add frame before it reaches the engine
        (its admission is held: it was captured and authorized)."""
        self.send_holds[self.lanes[k].sid] = NS(parked=asyncio.Event(), gate=asyncio.Event())
        return self.send_holds[self.lanes[k].sid]

    async def stop(self):
        self.engine.serving = 'stopped'
        self.engine.wakeup.set()
        if self.engine.pause is not None:
            self.engine.pause.gate.set()
        try:
            await asyncio.wait_for(self.loop, 5)
        except BaseException:
            pass


async def settle(predicate, timeout=5.0):
    """Wait in real time (capacity backoffs and the idle engine loop are timers)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        await asyncio.sleep(0.001)


def running(h, sid):
    """The engine request of session ``sid`` that is running, or None."""
    return next((rid for rid in h.running if h.infos[rid]['sid'] == sid), None)


def decoding(h, sid, outputs=1):
    rid = running(h, sid)
    return rid is not None and len(h.requests[rid].output_token_ids) >= outputs


def request_state(h, rid):
    """Everything of a running request that another request's admission (phase-1
    probe included) or step must leave as it was: worker record, claims, Qwen and
    backend state, runner row and cached state, its blocks' contents, engine rows."""
    rec, q, req = h.owner.records[rid], h.qwen.requests[rid], h.runner.requests[rid]
    blocks = [(g, x) for g, row in zip((W.FA,) + W.MAMBA, req.block_ids) for x in row if x]
    engine = h.requests[rid]
    return deepcopy(dict(
        record={k: rec[k] for k in ('layout', 'last_tables', 'native_rows', 'native_interval',
                                    'current_allocations', 'allocations', 'zeroed', 'attention_ready',
                                    'sparse_plan', 'generation', 'binding')},
        claims={k: v for k, v in h.owner.claims.items() if v[0] == rid},
        qwen=dict(cursor=q.cursor, current=q.current_hybrid, latest=q.latest_hybrid, step=q.latest_state_step,
                  active=q.active, vectors=q.page_vectors, receipts=len(q.receipts), execution=q.execution,
                  poisoned=q.poisoned),
        latest=h.backend.latest.get(rid, (None, None))[:2], backend=h.backend.requests[rid]['generation'],
        runner=dict(block_ids=req.block_ids, computed=req.num_computed_tokens, prompt=req.prompt_token_ids,
                    outputs=req.output_token_ids, idx=h.runner.mamba_state_idx[rid],
                    plan=h.runner._selective_load_plans[rid]),
        store={k: h.store.read('attention' if k[0] == W.FA else 'hybrid', *k) for k in blocks},
        engine=dict(computed=engine.num_computed_tokens, outputs=list(engine.output_token_ids),
                    rows=[[b.block_id for b in m.req_to_blocks[rid]] for m in h.kv.coordinator.single_type_managers],
                    phase=h.adapter.requests[rid].pending['phase'],
                    tables=h.boundary.prepared[rid].get('native_tables'),
                    cursor=h.boundary.prepared[rid].get('scheduled_cursor'))))


def step_state(h, rid):
    """A stepping request's bookkeeping that changes only at its own Page crossings."""
    if rid not in h.runner.requests or rid not in h.owner.records or 'last_tables' not in h.owner.records[rid]:
        return None
    rec, q = h.owner.records[rid], h.qwen.requests[rid]
    return deepcopy(dict(rows=h.runner.requests[rid].block_ids, idx=h.runner.mamba_state_idx.get(rid),
                         plan=h.runner._selective_load_plans.get(rid), current=q.current_hybrid,
                         latest=h.backend.latest.get(rid, (None, None))[:2],
                         claims={k: v for k, v in h.owner.claims.items() if v[0] == rid},
                         tables=rec['last_tables'], zeroed=set(rec['zeroed'])))


class BatchedEndToEndTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = E2E.MultiSessionEndToEndTests.asyncSetUp
    check_engine = ME.MultiSessionEngineTests.check

    @staticmethod
    def comparable(info):
        """test_ms_e2e's comparable turn evidence, without the capacity-wait count."""
        value = E2E.MultiSessionEndToEndTests.comparable(info)
        value['row'].pop('capacity_waits', None)
        return value
    quiescent = E2E.MultiSessionEndToEndTests.quiescent

    def server(self, policy, tasks, *, seqs=None, blocks=600, server=SERVER, goals=False):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = ME.config(policy, output, tasks)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = BatchServer(policy, tasks, output, self.router, self.Completion, server,
                          seqs=len(tasks) if seqs is None else seqs, blocks=blocks, mrope=True, goals=goals)
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        self.addAsyncCleanup(srv.stop)
        return srv

    async def alone(self, policy, tasks, turns):
        """Each session's turns run alone, one request at a time, on the stage-1 server."""
        srv = E2E.MultiSessionEndToEndTests.server(self, policy, tasks)
        return {k: [self.comparable(await srv.play(k, i)) for i in turns[k]] for k in turns}

    # ---- invariants ---------------------------------------------------------------
    def isolated(self, h):
        """test_ms_engine's isolation when nothing is in flight; with requests in
        flight, every idle session's pins are its own and no block has two owners."""
        if not h.busy():
            return ME.MultiSessionEngineTests.isolated(self, h)
        sessions = dict(h.adapter.sessions)
        names = [s.namespace for s in sessions.values()]
        self.assertEqual(len(set(names)), len(names))
        busy = {s.namespace for s in h.adapter.requests.values()}
        owned = {}
        for s in sessions.values():
            pins = h.adapter.owner.pinned(s.namespace)
            if s.namespace not in busy:
                self.assertTrue(all(block.ref_cnt == 1 for block in pins))
                self.assertEqual(len(pins), len(s.resident) + (s.tail is not None))
                self.assertEqual(tuple(sorted(h.adapter.owner.sessions.get(s.namespace, {}))), s.resident)
            for block in pins:
                self.assertNotIn(block.block_id, owned)
                owned[block.block_id] = s.namespace
                self.assertEqual(h.adapter.owner.block_namespaces[block.block_id], s.namespace)

    def check(self, srv, info):
        """One completed turn of one session, valid while other sessions decode:
        test_ms_engine's per-session engine invariants and the frontend's view."""
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
        self.assertTrue(type(row['capacity_waits']) is int and row['capacity_waits'] >= 0)
        stall = info['native']['admission_engine_ms']
        self.assertTrue(type(stall) is float and stall > 0)
        if srv.policy in QUERY:
            from hashlib import sha256
            self.assertEqual(row['query']['issue_sha256'],
                             sha256(lane.task['problem_statement'].encode()).hexdigest())
            self.assertEqual(row['query']['selection']['selected_pages'], info['plan']['selected'])
        for other in srv.lanes:
            if other is not lane:
                self.assertFalse(E2E.contains(lane.frontend.stream,
                                              E2E.X.TOK.encode(other.task['problem_statement'])))
        return info

    def attributed(self, h):
        """Every H2D byte belongs to exactly one request (tensors.attributed)."""
        own = h.io._http_requests
        self.assertTrue(set(own) <= set(h.infos))
        for name, index in (('_http_h2d_issued', 0), ('_http_h2d_joined', 1)):
            self.assertEqual(sum(v[index] for v in own.values()), getattr(h.io, name))
        self.assertEqual(h.io._http_scopes, [])

    def settled(self, srv):
        """Nothing in flight: global isolation, frontend quiescence, the runner empty."""
        h = srv.engine
        self.assertFalse(h.running or h.waiting or h.futures or h.busy())
        self.isolated(h)
        self.quiescent(srv)
        self.attributed(h)
        h.flush()
        self.assertEqual((h.runner.requests, h.runner.input_batch.req_ids, h.owner.records, h.owner.claims,
                          h.backend.requests, h.backend.latest, h.bridge.assignments),
                         ({}, [], {}, {}, {}, {}, {}))

    # ---- 1. several sessions decode together ----------------------------------------
    async def together(self, policy, tasks):
        n, turns = len(tasks), range(len(S.SCRIPT))
        alone = await self.alone(policy, tasks, {k: turns for k in range(n)})
        srv = self.server(policy, tasks)
        h = srv.engine
        self.assertTrue(srv.live.batched)

        async def agent(k):
            return [self.check(srv, await srv.play(k, i)) for i in turns]
        infos = await asyncio.gather(*(agent(k) for k in range(n)))
        # Each session saw, decided, moved and was charged exactly what it saw alone:
        # tokens, plans, frames, H2D (prepare and restore) and active KV bytes.
        for k in range(n):
            self.assertEqual([self.comparable(i) for i in infos[k]], alone[k], k)
        # The sessions really decoded together, in mixed steps, with admissions
        # (phase 1 included) arriving while others were mid-decode.
        widths = [len(b.rows) for b in h.batches]
        self.assertEqual(max(widths), n)
        self.assertTrue(any(len({count > 1 for _, _, count in b.rows}) == 2 for b in h.batches))
        mid = [a for a in h.admissions if any(r.phase == 'decode' for r in a.running)]
        self.assertTrue(mid and any(a.local_turn > 0 for a in mid))
        # Disjoint namespaces, pins and GDN carries; each carry its own chain.
        names = [h.sessions.namespaces[l.sid] for l in srv.lanes]
        pins = [{x.block_id for x in h.adapter.owner.pinned(ns)} for ns in names]
        self.assertEqual(len(set(names)), n)
        self.assertTrue(all(pins) and all(not pins[i] & pins[j] for i in range(n) for j in range(i)))
        self.assertEqual(set(h.backend.carry), {l.sid for l in srv.lanes})
        self.assertEqual(len({h.backend.carry[l.sid].digest for l in srv.lanes}), n)
        rows = srv.rows()
        self.assertEqual(sorted((r['session_index'], r['turn']) for r in rows),
                         sorted((k, i) for k in range(n) for i in turns))
        self.assertEqual(sorted(r['request_index'] for r in rows), list(range(n * len(turns))))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertTrue(all(r['capacity_waits'] == 0 for r in rows))
        self.settled(srv)
        return srv

    # The slow pass (README section 2.3): full, and goal_query's two-session run; the
    # default suite keeps goal_query's three sessions decoding together.
    @unittest.skipUnless(os.environ.get('REMOTE_RA_SLOW_TESTS') == '1',
                         'about 20 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_two_and_three_sessions_decode_together_full(self):
        for tasks in ([A, B], [A, B, C]):
            with self.subTest(sessions=len(tasks)):
                await self.together('full', tasks)

    async def test_two_and_three_sessions_decode_together_goal_query(self):
        for tasks in ([A, B], [A, B, C]) if os.environ.get('REMOTE_RA_SLOW_TESTS') == '1' else ([A, B, C],):
            with self.subTest(sessions=len(tasks)):
                srv = await self.together('goal_query', tasks)
                bodies = [srv.engine.selection.authorized[r]['body'] for r in srv.engine.infos]
                self.assertTrue(any(x['goal']['forward'] for x in bodies))

    async def warm(self, srv, turns=2):
        """Turns [0, turns) of every session, one request at a time."""
        infos = defaultdict(list)
        for i in range(turns):
            for k in range(len(srv.lanes)):
                infos[k].append(self.check(srv, await srv.play(k, i)))
        return infos

    # ---- 2. phase 1 of one session while two others decode -----------------------------
    async def test_phase1_of_one_session_while_two_others_decode(self):
        policy, tasks = 'observation_query', [A, B, C]
        alone = await self.alone(policy, tasks, {k: range(3) for k in range(3)})
        srv = self.server(policy, tasks)
        h = srv.engine
        a, b, c = srv.lanes
        infos = await self.warm(srv)
        pause = h.park(lambda e: decoding(e, a.sid, 2) and decoding(e, b.sid, 2))
        ta, tb = asyncio.create_task(srv.play(0, 2)), asyncio.create_task(srv.play(1, 2))
        await asyncio.wait_for(pause.parked.wait(), 5)
        ra, rb = running(h, a.sid), running(h, b.sid)
        seen = {}

        def watch(event, arg):
            if event in ('admit', 'admitted') and arg['session'] == c.sid:
                seen[event] = ({rid: request_state(h, rid) for rid in (ra, rb)}, len(h.runner.dropped),
                               list(h.runner.input_batch.req_ids))
        h.watch = watch
        tc = asyncio.create_task(srv.play(2, 2))
        await settle(lambda: not h.input_queue.empty())     # C's registered add, queued while A, B decode
        pause.gate.set()
        ia, ib, ic = await asyncio.gather(ta, tb, tc)
        h.watch = None
        # C's admission ran its phase-1 probe while A and B were mid-decode in the batch.
        admission = next(x for x in h.admissions if x.rid == ic['rid'])
        self.assertEqual(admission.local_turn, 2)
        self.assertEqual(sorted((r.sid, r.phase) for r in admission.running), [(a.sid, 'decode'), (b.sid, 'decode')])
        self.assertEqual([f['operation'] for f in ic['frames']][:4], ['prepare', 'phase1', 'phase1_drain', 'reserve'])
        (before, drops, batch), (after, dropped, emptied) = seen['admit'], seen['admitted']
        self.assertEqual((sorted(batch), emptied), (sorted([ra, rb]), []))
        # vLLM dropped both rows for the probe and re-added them from
        # requests[rid].block_ids at their next step; nothing of either moved.
        self.assertEqual(sorted(h.runner.dropped[drops:dropped]), sorted([ra, rb]))
        self.assertTrue({ra, rb} <= set(h.runner.readded))
        self.assertEqual(after, before)
        # A and B continued exactly as if C had not been admitted; C is C alone.
        for k, info in enumerate((ia, ib, ic)):
            self.check(srv, info)
            self.assertEqual([self.comparable(i) for i in infos[k] + [info]], alone[k], k)
        self.settled(srv)

    # ---- 3. a Page crossing inside a batched step --------------------------------------
    async def test_page_crossing_inside_a_batched_step_leaves_the_other_request_untouched(self):
        policy, tasks, turns = 'random50', [A, B], range(4)
        alone = await self.alone(policy, tasks, {0: turns, 1: turns})
        srv = self.server(policy, tasks)
        h = srv.engine
        h.watch = lambda event, rid: step_state(h, rid) if event == 'step' else None

        async def agent(k):
            return [self.check(srv, await srv.play(k, i)) for i in turns]
        infos = await asyncio.gather(agent(0), agent(1))
        h.watch = None
        for k in (0, 1):
            self.assertEqual([self.comparable(i) for i in infos[k]], alone[k], k)
        moves = captures = 0
        for batch in h.batches:
            for x, xs, xc in batch.rows:
                for y, ys, yc in batch.rows:
                    if x == y or xc != 1 or yc != 1 or (xs % P and (xs + 1) % P) or ys % P in (0, P - 1):
                        continue
                    # x decodes across a Page boundary (a new FA block and GDN slot with
                    # the latest-state relocation, or the latest capture at a Page end)
                    # while y decodes inside a Page in the same execute_model.
                    mine = {blk for row in batch.after[x]['rows'] for blk in row if blk}
                    self.assertTrue(batch.io)
                    self.assertTrue(all(blk in mine for _, _, keys in batch.io for _, blk in keys))
                    self.assertEqual(batch.after[y], batch.before[y])
                    if xs % P == 0:
                        moves += 1
                        self.assertNotEqual(batch.after[x]['current'], batch.before[x]['current'])
                    else:
                        captures += 1
                        self.assertNotEqual(batch.after[x]['latest'], batch.before[x]['latest'])
        self.assertTrue(moves >= 2 and captures >= 2, (moves, captures))
        self.settled(srv)

    async def test_rebuild_while_others_decode_and_the_decode_first_reorder(self):
        # A's format-error retry rebuilds A (render closes A's old namespace) while
        # B decodes; C is admitted (phase 1) while A's long rebuilt prompt is still
        # in prefill, and C's decode then sits ahead of A's prefill in the batch.
        policy, tasks, retry = 'observation_query', [A, B, C], S.REBUILD
        solo = E2E.MultiSessionEndToEndTests.server(self, policy, tasks)
        alone = defaultdict(list)
        for i in range(3):
            for k in range(3):
                alone[k].append(self.comparable(await solo.play(k, i)))
        messages = solo.lanes[0].messages[:-1] + [dict(role='user', content=retry[0])]
        alone[0].append(self.comparable(await solo.turn(0, messages=messages, target=retry[1])))
        for k in (1, 2):
            alone[k].append(self.comparable(await solo.play(k, 3)))
        alone[0].append(self.comparable(await solo.turn(0, S.AFTER[0], target=S.AFTER[1])))
        srv = self.server(policy, tasks)
        h = srv.engine
        a, b, c = srv.lanes
        infos = await self.warm(srv, 3)
        names = dict(h.sessions.namespaces)
        h.rpc_log.clear()
        pause = h.park(lambda e: decoding(e, b.sid, 1))
        tb = asyncio.create_task(srv.play(1, 3))
        await asyncio.wait_for(pause.parked.wait(), 5)
        messages = a.messages[:-1] + [dict(role='user', content=retry[0])]
        ta = asyncio.create_task(srv.turn(0, messages=messages, target=retry[1]))
        await settle(lambda: not h.input_queue.empty())
        pause.gate.set()
        prefill = h.park(lambda e: running(e, a.sid) is not None and e.stepped[running(e, a.sid)] >= 2)
        await asyncio.wait_for(prefill.parked.wait(), 5)
        ra = running(h, a.sid)
        self.assertEqual(h.progress(ra).phase, 'prefill')
        tc = asyncio.create_task(srv.play(2, 3))
        await settle(lambda: not h.input_queue.empty())
        prefill.gate.set()
        ia, ib, ic = await asyncio.gather(ta, tb, tc)
        for info in (ia, ib, ic):
            self.check(srv, info)
        self.assertTrue(ia['binding'].rebuild and ia['committed']['rebuild'])
        self.assertEqual([key for op, key in h.rpc_log if op == 'close_session'], [names[a.sid]])
        self.assertEqual((h.sessions.namespaces[b.sid], h.sessions.namespaces[c.sid]), (names[b.sid], names[c.sid]))
        admitted = {x.rid: x for x in h.admissions}
        self.assertEqual([(r.sid, r.phase) for r in admitted[ra].running], [(b.sid, 'decode')])
        self.assertEqual(sorted((r.sid, r.phase) for r in admitted[ic['rid']].running),
                         [(a.sid, 'prefill'), (b.sid, 'decode')])
        # vLLM's GDN backend reordered the persistent batch decode-first: C's decode
        # ran ahead of A's prefill, which the scheduler had placed first.
        swapped = [x for x in h.batches if ra in x.batch and ic['rid'] in x.batch and
                   x.batch.index(ic['rid']) < x.batch.index(ra)]
        self.assertTrue(h.runner.reordered and swapped)
        self.assertTrue(all([r for r, _, _ in x.rows].index(ra) < [r for r, _, _ in x.rows].index(ic['rid'])
                            for x in swapped))
        after = self.check(srv, await srv.turn(0, S.AFTER[0], target=S.AFTER[1]))
        self.assertEqual([self.comparable(i) for i in infos[0] + [ia, after]], alone[0])
        for k, info in ((1, ib), (2, ic)):
            self.assertEqual([self.comparable(i) for i in infos[k] + [info]], alone[k], k)
        self.settled(srv)

    async def test_two_completions_in_one_step_then_a_probe_before_the_next(self):
        # A and B stop in the same execute_model: two completions (seal, commit) in
        # one update_from_output, both rows still in the runner's batch until the
        # next step's finished_req_ids. A's next turn is admitted first: its
        # phase-1 probe runs on a batch that still holds both finished rows.
        policy, tasks = 'random50', [A, B]
        alone = await self.alone(policy, tasks, {0: range(4), 1: range(3)})
        srv = self.server(policy, tasks)
        h = srv.engine
        a, b = srv.lanes
        infos = await self.warm(srv)
        left = lambda rid: len(h.plans[rid].outputs) - len(h.requests[rid].output_token_ids)
        pause = h.park(lambda e: decoding(e, a.sid, 1))
        ta = asyncio.create_task(srv.play(0, 2))
        await asyncio.wait_for(pause.parked.wait(), 5)
        tb = asyncio.create_task(srv.play(1, 2))
        await settle(lambda: not h.input_queue.empty())
        both = h.park(lambda e: decoding(e, a.sid) and decoding(e, b.sid))
        pause.gate.set()
        await asyncio.wait_for(both.parked.wait(), 5)
        ra, rb = running(h, a.sid), running(h, b.sid)
        self.assertGreater(left(rb), left(ra))
        # A waits unscheduled (vLLM drops its row and re-adds it) until B has caught up.
        hold = h.plans[ra].hold = NS(at=h.stepped[ra], parked=asyncio.Event(), gate=asyncio.Event())
        even = h.park(lambda e: left(rb) == left(ra))
        both.gate.set()
        await asyncio.wait_for(even.parked.wait(), 5)
        hold.gate.set()
        send = srv.hold_send(0)
        even.gate.set()
        ia, ib = self.check(srv, await ta), self.check(srv, await tb)
        steps = dict((rid, n) for n, rid in h.completed)
        self.assertEqual(steps[ra], steps[rb])
        self.assertTrue(hold.parked.is_set() and ra in h.runner.readded)
        self.assertEqual(h.finished, {ra, rb})
        self.assertEqual(sorted(h.runner.input_batch.req_ids), sorted([ra, rb]))
        # A's next turn (phase 1) reaches the engine before any further step.
        drops = len(h.runner.dropped)
        ta = asyncio.create_task(srv.play(0, 3))
        await asyncio.wait_for(send.parked.wait(), 5)
        self.assertEqual(len(h.batches), steps[ra])
        send.gate.set()
        ia3 = self.check(srv, await ta)
        self.assertEqual(sorted(h.runner.dropped[drops:drops + 2]), sorted([ra, rb]))
        self.assertFalse(h.finished & {ra, rb})
        self.assertEqual([self.comparable(i) for i in infos[0] + [ia, ia3]], alone[0])
        self.assertEqual([self.comparable(i) for i in infos[1] + [ib]], alone[1])
        self.settled(srv)

    # ---- 4. capacity wait ---------------------------------------------------------------
    async def test_capacity_wait_while_another_session_runs(self):
        import test_ms2_admission as M2
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 0.02))
        policy, tasks = 'observation_query', [A, B]
        alone = await self.alone(policy, tasks, {0: range(3), 1: range(3)})
        # 200 blocks: B's turn bound plus A's headroom (about 214) exceeds what is
        # free while A runs. 600 blocks: B fits. The gate reads the pool, no constant.
        for blocks, waits in ((200, True), (600, False)):
            with self.subTest(blocks=blocks):
                srv = self.server(policy, tasks, blocks=blocks)
                h, live = srv.engine, srv.live
                a, b = srv.lanes
                infos = await self.warm(srv)
                refusals, views = [], []

                def watch(event, arg):
                    if event == 'handle':
                        command = h.codec.decode(arg)
                        if type(command) is RegisteredAdd and command.session_id == b.sid:
                            views.append((M2.engine_view(h, b), h.pool.get_num_free_blocks(),
                                          [h.progress(r) for r in h.running]))
                    elif event == 'handled':
                        ack = h.codec.decode(arg[1])
                        if ack.state == 'rejected':
                            self.assertTrue(capacity.refused(ack.error), ack.error)
                            refusals.append((views[-1], M2.engine_view(h, b), ack.error))
                h.watch = watch
                pause = h.park(lambda e: decoding(e, a.sid, 1))
                ta = asyncio.create_task(srv.play(0, 2))
                await asyncio.wait_for(pause.parked.wait(), 5)
                frontend = M2.frontend_view(srv, 1)
                started = time.monotonic()
                tb = asyncio.create_task(srv.play(1, 2))
                await settle(lambda: not h.input_queue.empty())
                pause.gate.set()
                if waits:
                    # B is between retries: admission free, slot held, its transcript
                    # restored and no authority left in the frontend.
                    await settle(lambda: refusals and M2.Stage2AdmissionTests.waiting(self, srv, 1))
                    self.assertEqual(M2.frontend_view(srv, 1), frontend)
                    self.assertFalse(live.admission.locked())
                ia, ib = await asyncio.gather(ta, tb)
                waited = time.monotonic() - started
                h.watch = None
                for k, info in enumerate((ia, ib)):
                    self.check(srv, info)
                    self.assertEqual([self.comparable(i) for i in infos[k] + [info]], alone[k], k)
                order = [e for e in h.events if e[2] == 2]
                admission = next(x for x in h.admissions if x.rid == ib['rid'])
                if waits:
                    self.assertTrue(refusals)
                    for (engine, free, progress), after, error in refusals:
                        # Nothing in the engine changed; A was running; the gate read the pool.
                        self.assertEqual(after, engine)
                        self.assertEqual([r.sid for r in progress], [a.sid])
                        self.assertTrue(error.endswith('; %d free' % free), error)
                    # B was admitted after A's completion drain, then finished correctly.
                    self.assertEqual(order, [('admit', a.sid, 2), ('complete', a.sid, 2),
                                             ('admit', b.sid, 2), ('complete', b.sid, 2)])
                    self.assertEqual(admission.running, [])
                    refused = {r for kind, r, _ in srv.frames
                               if kind == 'RegisteredAdd' and r not in h.dispatcher._records}
                    self.assertEqual(len(refused), len(refusals))
                    self.assertFalse(refused & {r for kind, r, _ in srv.frames if kind == 'Cancel'})
                else:
                    self.assertEqual(refusals, [])
                    self.assertEqual([r.sid for r in admission.running], [a.sid])
                    self.assertEqual(order[:2], [('admit', a.sid, 2), ('admit', b.sid, 2)])
                # Evidence: capacity_waits counts the retries, queue_wait_s includes them.
                row = ib['row']
                self.assertEqual((ia['row']['capacity_waits'], row['capacity_waits']), (0, len(refusals)))
                self.assertTrue(0 <= row['queue_wait_s'] <= waited)
                if waits:
                    self.assertGreater(row['queue_wait_s'], 0.01)
                self.assertFalse(srv.rows('failures.jsonl'))
                self.settled(srv)
                self.check(srv, await srv.play(1, 3))

    async def test_exact_generation_bounds_still_never_preempt(self):
        # The gate with each request's exact output length as its generation bound
        # (the tightest bound there is; the runtime uses max_tokens = 1536): five
        # sessions on a small pool wait often and run several at a time, and no
        # running request ever finds no block. Below about 170 blocks the idle
        # sessions' pins leave too little for one turn even alone; nothing then
        # runs, and the gate refuses it while they pin (every turn waits): that
        # pool is too small for five sessions. Before tb16 job 3398129's fix the
        # gate admitted it as a one-task server would and the turn preempted (fatal).
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 0.005))
        exact = {}
        self.enterContext(patch.object(capacity, 'max_tokens', lambda params: exact[id(params)]))
        tasks = [A, B, C] + [dict(instance_id='e2e-%d' % k, problem_statement='Task %d: fix module m%d.' % (k, k))
                             for k in (3, 4)]
        for policy in ('random50', 'observation_query'):
            with self.subTest(policy=policy):
                srv = self.server(policy, tasks, blocks=192)
                h = srv.engine
                h.exact = exact

                async def agent(k):
                    return [self.check(srv, await srv.play(k, i)) for i in range(5)]
                await asyncio.gather(*(agent(k) for k in range(len(tasks))))
                rows = srv.rows()
                self.assertEqual(len(rows), 25)
                self.assertFalse(srv.rows('failures.jsonl'))
                self.assertGreater(sum(r['capacity_waits'] for r in rows), 0)
                self.assertGreaterEqual(max(len(x.rows) for x in h.batches), 3)
                self.assertLess(h.low, 30)
                self.settled(srv)

    async def test_shutdown_while_a_request_waits_for_capacity(self):
        from entrypoint_runtime import ShutdownOwner
        import test_ms2_admission as M2
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 0.02))
        srv = self.server('observation_query', [A, B], blocks=200)
        h = srv.engine
        a, b = srv.lanes
        await self.warm(srv)
        names = dict(h.sessions.namespaces)
        pause = h.park(lambda e: decoding(e, a.sid, 1))
        ta = asyncio.create_task(srv.play(0, 2))
        await asyncio.wait_for(pause.parked.wait(), 5)
        tb = asyncio.create_task(srv.play(1, 2))
        await settle(lambda: not h.input_queue.empty())
        pause.gate.set()
        await settle(lambda: M2.Stage2AdmissionTests.waiting(self, srv, 1) and any(
            kind == 'RegisteredAdd' and r not in h.dispatcher._records for kind, r, _ in srv.frames))
        # B holds its slot between CAPACITY_WAIT retries, outside any serving scope:
        # shutdown_close cancels A's call only.
        self.assertEqual(len(srv.controller._active), 1)
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        with self.assertRaises(asyncio.CancelledError):
            await ta
        # Woken by A's release, B queues again and finds the controller closed.
        with self.assertRaisesRegex(RuntimeError, 'serving controller closed'):
            await asyncio.wait_for(tb, 5)
        failures = sorted(srv.rows('failures.jsonl'), key=lambda f: f['session_index'])
        self.assertEqual([(f['session_index'], f['turn'], f['error']) for f in failures],
                         [(0, 2, 'CancelledError'), (1, 2, 'PolicyRuntimeAbort')])
        self.assertGreaterEqual(failures[1]['capacity_waits'], 1)
        self.assertNotIn(b.sid, [e['identity'][1] for e in h.lifecycle.entries.values() if e['identity'][2] == 2])
        self.assertEqual(sorted(key for op, key in h.rpc_log if op == 'close_session'), sorted(names.values()))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertFalse(h.adapter.sessions or h.owner.sessions or h.owner.claims or h.runner.requests)
        self.assertTrue(srv.controller._closed and not srv.live.admission.locked())

    async def test_disconnect_landing_with_a_capacity_wakeup_cancels_the_request(self):
        # B waits for capacity, outside any serving scope. Its client goes away and
        # a release on this server wakes B in the same loop iteration: the route's
        # watcher marks the disconnect first, then B resumes, takes admission and
        # would retry from capture. It must end as a client cancellation, not be
        # refused by the serving scope's route-owner check.
        import test_ms2_admission as M2
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 30))   # only a release wakes B
        srv = self.server('observation_query', [A, B], blocks=200)
        h, live = srv.engine, srv.live
        a, b = srv.lanes
        await self.warm(srv)
        pause = h.park(lambda e: decoding(e, a.sid, 1))
        ta = asyncio.create_task(srv.play(0, 2))
        await asyncio.wait_for(pause.parked.wait(), 5)
        tb = asyncio.create_task(srv.play(1, 2))
        await settle(lambda: not h.input_queue.empty())
        pause.gate.set()
        await settle(lambda: M2.Stage2AdmissionTests.waiting(self, srv, 1) and any(
            kind == 'RegisteredAdd' and r not in h.dispatcher._records for kind, r, _ in srv.frames))
        hold = h.park(lambda e: True)            # A stays mid-decode meanwhile
        await asyncio.wait_for(hold.parked.wait(), 5)
        srv.disconnect(1)
        live._released.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(tb, 5)
        failure = srv.rows('failures.jsonl')[-1]
        self.assertEqual((failure['session_index'], failure['turn'], failure['error']), (1, 2, 'CancelledError'))
        self.assertGreaterEqual(failure['capacity_waits'], 1)
        self.assertEqual([x.closed for x in live.sessions], [False, True])
        self.assertEqual(h.lifecycle.sessions[b.sid], (2, None))     # the engine never saw B's turn 2
        self.assertTrue(live._holder is None and not live.admission.locked())
        hold.gate.set()
        self.check(srv, await ta)
        self.settled(srv)

    # ---- 5. admission passes on at engine acceptance, FIFO ---------------------------------
    async def test_admission_passes_on_at_acceptance_in_fifo_order_while_others_decode(self):
        policy, tasks = 'goal_query', [A, B, C]
        alone = await self.alone(policy, tasks, {k: range(3) for k in range(3)})
        srv = self.server(policy, tasks)
        h, live = srv.engine, srv.live
        a, b, c = srv.lanes
        infos = await self.warm(srv)
        # A's accepted ack is held: A holds admission; B, then C, queue behind it.
        ack, send_b, send_c = srv.hold_ack(0), srv.hold_send(1), srv.hold_send(2)
        ta = asyncio.create_task(srv.play(0, 2))
        await asyncio.wait_for(ack.parked.wait(), 5)
        tb = asyncio.create_task(srv.play(1, 2))
        await settle(lambda: live.sessions[1].owner is not None)
        tc = asyncio.create_task(srv.play(2, 2))
        await settle(lambda: live.sessions[2].owner is not None)
        await asyncio.sleep(.1)
        self.assertEqual((b.frontend.pending, c.frontend.pending, live.sessions[1].active,
                          live.sessions[2].active), (None, None, None, None))
        self.assertTrue(live.admission.locked())
        self.assertEqual([e for e in h.events if e[0] == 'admit'][-1], ('admit', a.sid, 2))
        # The accepted ack reaches A's frontend: admission passes on to B (FIFO),
        # whose frame reaches the engine while A decodes.
        ack.gate.set()
        await asyncio.wait_for(send_b.parked.wait(), 5)
        self.assertTrue(live.admission.locked() and live.sessions[1].active is not None)
        self.assertEqual((c.frontend.pending, live.sessions[2].active), (None, None))
        await settle(lambda: decoding(h, a.sid, 1))
        send_b.gate.set()
        # B's acceptance passes admission on to C; C's frame arrives while A and B decode.
        await asyncio.wait_for(send_c.parked.wait(), 5)
        await settle(lambda: decoding(h, b.sid, 1) and decoding(h, a.sid, 2))
        send_c.gate.set()
        ia, ib, ic = await asyncio.gather(ta, tb, tc)
        admitted = {x.rid: x for x in h.admissions}
        self.assertEqual([e for e in h.events if e[0] == 'admit'][-3:],
                         [('admit', a.sid, 2), ('admit', b.sid, 2), ('admit', c.sid, 2)])
        self.assertEqual([(r.sid, r.phase) for r in admitted[ib['rid']].running], [(a.sid, 'decode')])
        self.assertEqual(sorted((r.sid, r.phase) for r in admitted[ic['rid']].running),
                         [(a.sid, 'decode'), (b.sid, 'decode')])
        self.assertTrue(all(admitted[i['rid']].local_turn == 2 for i in (ia, ib, ic)))
        self.assertEqual((live.admission.locked(), live._holder), (False, None))
        rows = [i['row'] for i in (ia, ib, ic)]
        self.assertEqual([r['request_index'] for r in rows], sorted(r['request_index'] for r in rows))
        self.assertTrue(all(r['queue_wait_s'] >= .05 for r in rows[1:]))   # Windows clock: ~16 ms steps
        for k, info in enumerate((ia, ib, ic)):
            self.check(srv, info)
            self.assertEqual([self.comparable(i) for i in infos[k] + [info]], alone[k], k)
        cancels = [t for kind, _, t in srv.frames if kind == 'Cancel']
        self.assertEqual((len(cancels), set(cancels)), (len(srv.rows()), {60}))
        self.settled(srv)

    # ---- 6. failure isolation under batching ---------------------------------------------
    async def test_cancel_of_one_session_between_batched_steps_leaves_the_other_decoding(self):
        policy, tasks = 'observation_query', [A, B]
        alone = await self.alone(policy, tasks, {1: range(4)})
        srv = self.server(policy, tasks)
        h = srv.engine
        a, b = srv.lanes
        infos = await self.warm(srv)
        keys = ('namespace', 'turn', 'resident', 'tail', 'catalog', 'boundary', 'pins', 'carry')
        committed = lambda state: {key: state[key] for key in keys}
        before = committed(h.state(a))
        pause = h.park(lambda e: decoding(e, a.sid, 2) and decoding(e, b.sid, 2))
        ta, tb = asyncio.create_task(srv.play(0, 2)), asyncio.create_task(srv.play(1, 2))
        await asyncio.wait_for(pause.parked.wait(), 5)
        ra, rb = running(h, a.sid), running(h, b.sid)
        self.assertEqual(sorted(h.runner.input_batch.req_ids), sorted([ra, rb]))
        state = request_state(h, rb)
        # A's client goes away between two batched steps: the engine aborts A
        # (cancel drain, native free) while B is mid-decode in the same batch.
        srv.disconnect(0)
        await settle(lambda: ('abort', a.sid, 2) in h.events)
        self.assertEqual((h.running, request_state(h, rb)), ([rb], state))
        self.assertEqual(sorted(h.runner.input_batch.req_ids), sorted([ra, rb]))
        pause.gate.set()            # the frontend's cleanup Cancel needs the engine loop
        with self.assertRaises(asyncio.CancelledError):
            await ta
        ib = self.check(srv, await tb)
        # A: drained and poisoned, its committed state unchanged, every reference back.
        s = h.adapter.sessions[a.sid]
        self.assertTrue(s.poisoned and h.owner.sessions[s.namespace]['poisoned'])
        self.assertEqual(h.lifecycle.sessions[a.sid], (2, 'poisoned'))
        self.assertEqual(committed(h.state(a)), before)
        operations = [f['operation'] for f in h.proxy.http_receipts[ra]]
        self.assertEqual(operations[:4] + operations[-1:], ['prepare', 'phase1', 'phase1_drain', 'reserve', 'cancel'])
        self.assertEqual(set(operations[4:-1]), {'allocation_update'})
        self.assertEqual((h.frees[ra], h.native_file(ra)['terminal_state']), (1, 'cancelled'))
        self.assertEqual([x.closed for x in srv.live.sessions], [True, False])
        # B went on in the same batch, then alone; exactly as B alone.
        self.assertTrue(any({ra, rb} <= {r for r, _, _ in x.rows} for x in h.batches))
        ib3 = self.check(srv, await srv.play(1, 3))
        self.assertEqual([self.comparable(i) for i in infos[1] + [ib, ib3]], alone[1])
        with self.assertRaisesRegex(RuntimeError, 'session %s closed by an earlier failure' % a.sid):
            await asyncio.wait_for(srv.play(0, 2), 1)
        failures = srv.rows('failures.jsonl')
        self.assertEqual([(f['session_index'], f['turn'], f['error']) for f in failures],
                         [(0, 2, 'CancelledError'), (None, None, 'LiveV2Error')])
        self.settled(srv)

    async def test_failed_completion_in_a_batched_step_fails_only_its_session(self):
        policy, tasks = 'goal_query', [A, B]
        alone = await self.alone(policy, tasks, {1: range(4)})
        srv = self.server(policy, tasks)
        h = srv.engine
        a, b = srv.lanes
        infos = await self.warm(srv)
        keys = ('namespace', 'turn', 'resident', 'tail', 'catalog', 'boundary', 'pins', 'carry')
        committed = lambda state: {key: state[key] for key in keys}
        before = committed(h.state(a))

        def refuse(payload):
            # The worker refuses A's completion (a computed end it never reached).
            if payload['session'] != a.sid:
                h.tamper['complete'] = refuse
                return payload
            return dict(payload, computed_end=payload['computed_end'] + 1)
        h.tamper['complete'] = refuse
        pause = h.park(lambda e: decoding(e, a.sid, 1))
        ta = asyncio.create_task(srv.play(0, 2))
        await asyncio.wait_for(pause.parked.wait(), 5)
        tb = asyncio.create_task(srv.play(1, 2))
        await settle(lambda: not h.input_queue.empty())
        pause.gate.set()
        with self.assertRaisesRegex(Exception, 'complete computed/sealed end mismatch'):
            await ta
        ib = self.check(srv, await tb)
        ra = next(r for r, i in h.infos.items() if i['sid'] == a.sid and i['turn'] == 2)
        # A stopped in a step that B shared; its completion failed there.
        last = max(i for i, x in enumerate(h.batches) if ra in {r for r, _, _ in x.rows})
        self.assertIn(ib['rid'], {r for r, _, _ in h.batches[last].rows})
        self.assertEqual(h.lifecycle.sessions[a.sid], (2, 'poisoned'))
        self.assertEqual(committed(h.state(a)), before)
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]][-1], 'cancel')
        self.assertEqual((h.frees[ra], h.native_file(ra)['terminal_state']), (1, 'cancelled'))
        self.assertEqual([x.closed for x in srv.live.sessions], [True, False])
        ib3 = self.check(srv, await srv.play(1, 3))
        self.assertEqual([self.comparable(i) for i in infos[1] + [ib, ib3]], alone[1])
        self.assertEqual([(f['session_index'], f['turn']) for f in srv.rows('failures.jsonl')], [(0, 2)])
        self.settled(srv)

    # ---- 7. shutdown with several requests in flight ------------------------------------------
    async def test_shutdown_with_requests_in_flight_drains_and_closes_everything(self):
        import io
        import contextlib
        import threading
        from entrypoint_runtime import ShutdownOwner
        from remote_ra_http.shutdown import accept_closed, manager_shutdown
        from hashlib import sha256
        srv = self.server('goal_query', [A, B, C])
        h = srv.engine
        a, b, c = srv.lanes
        await self.warm(srv)
        names = dict(h.sessions.namespaces)
        h.rpc_log.clear()
        # A and B decode together; C's admission is accepted in the engine, its ack
        # not yet at the frontend (C holds admission, its request waits in vLLM).
        pause = h.park(lambda e: decoding(e, a.sid, 2) and decoding(e, b.sid, 2))
        ta, tb = asyncio.create_task(srv.play(0, 2)), asyncio.create_task(srv.play(1, 2))
        await asyncio.wait_for(pause.parked.wait(), 5)
        ack = srv.hold_ack(2)
        tc = asyncio.create_task(srv.play(2, 2))
        await settle(lambda: not h.input_queue.empty())
        pause.gate.set()
        await asyncio.wait_for(ack.parked.wait(), 5)
        rc = next(r for r, i in h.infos.items() if i['sid'] == c.sid and i['turn'] == 2)
        self.assertTrue(decoding(h, a.sid) and decoding(h, b.sid) and rc in h.waiting)
        self.assertTrue(srv.live.admission.locked())
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        for task in (ta, tb, tc):
            with self.assertRaises(asyncio.CancelledError):
                await task
        # The engine loop stopped; every namespace closed once; every block back.
        await asyncio.wait_for(srv.loop, 1)
        self.assertFalse(h.running or h.waiting or h.futures or h.enqueued)
        closes = [key for op, key in h.rpc_log if op == 'close_session']
        self.assertEqual(sorted(closes), sorted(names.values()))
        self.assertEqual([r['reason'] for r in h.proxy.http_receipts['session']][-3:], ['shutdown'] * 3)
        self.assertTrue(h.sessions.closed and h.lifecycle.closed)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertTrue(all(block.ref_cnt == 0 for block in h.pool.blocks[1:]))
        self.assertFalse(h.adapter.sessions or h.adapter.owner.sessions or h.adapter.owner.tails or
                         h.adapter.owner.leases or h.adapter.owner.block_namespaces or h.adapter.requests)
        self.assertFalse(h.owner.sessions or h.owner.session_namespaces or h.backend.carry or
                         h.qwen.requests or h.owner.records or h.owner.claims or h.backend.requests)
        self.assertEqual((h.runner.requests, h.runner.input_batch.req_ids), ({}, []))
        self.assertEqual([h.lifecycle.sessions[l.sid] for l in srv.lanes], [(2, 'poisoned')] * 3)
        last = [r for r, i in h.infos.items() if i['turn'] == 2]
        self.assertEqual([h.dispatcher._records[r]['state'] for r in last], ['cancelled'] * 3)
        self.assertEqual([h.frees[r] for r in last], [1, 1, 1])
        cleanup = json.loads((srv.output/'engine-cleanup.json').read_bytes())
        self.assertEqual(srv.llm.cleanup_receipt, sha256((srv.output/'engine-cleanup.json').read_bytes()).hexdigest())
        self.assertEqual(cleanup['sessions'], [dict(session_id=l.sid, session_index=l.index,
                                                    namespace=names[l.sid]) for l in srv.lanes])
        self.assertEqual(len(cleanup['admitted_request_ids']), 9)
        self.assertTrue(srv.controller._closed and srv.registry._closed and not srv.registry._submitted)
        self.assertEqual([x.closed for x in srv.live.sessions], [True] * 3)
        failures = srv.rows('failures.jsonl')
        self.assertEqual(sorted((f['session_index'], f['turn'], f['error']) for f in failures),
                         [(k, 2, 'CancelledError') for k in range(3)])
        self.assertEqual(len(srv.rows()), 6)
        manager = NS(close_lock=threading.Lock(), reader=None, watchdog=None, error=None,
                     revoked=threading.Event(), done=threading.Event(), pipe=NS(close=lambda: None),
                     child_pipe=NS(close=lambda: None), session_id=SERVER,
                     process=NS(join=lambda timeout: None, is_alive=lambda: False, exitcode=0),
                     cleanup_path=srv.output/'engine-cleanup.json',
                     identity=NS(epoch=S.EPOCH, manifest=S.SOURCE))
        accept_closed(manager, dict(cleanup_sha256=srv.llm.cleanup_receipt))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(manager_shutdown(manager), srv.llm.cleanup_receipt)
        with self.assertRaisesRegex(RuntimeError, 'session %s closed' % a.sid):
            await asyncio.wait_for(srv.play(0, 3), 1)


class SingleSequenceEngineLoopTests(E2E.MultiSessionEndToEndTests):
    """``max_num_seqs = 1`` with N tasks, and a one-task server: every stage-1 case
    of test_ms_e2e, assertions unchanged, on the stage-2 engine loop (one
    registered-add per pass, capacity gate in place, admission held to completion)."""

    def server(self, policy, tasks, server=SERVER):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = ME.config(policy, output, tasks)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = BatchServer(policy, tasks, output, self.router, self.Completion, server, seqs=1)
        self.assertFalse(srv.live.batched)
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        self.addAsyncCleanup(srv.stop)
        return srv


if __name__ == '__main__':
    unittest.main()
