"""--async-scheduling: vLLM's asynchronous scheduling, option C (docs/async-sched/DESIGN.md 4.3).

Opt-in (serve.py --async-scheduling, default off; both model profiles, the Gemma 4 profile
since async-gemma (docs/async-sched/GEMMA.md); needs --decode-fastpath2; refused with
--gdn-device-handoff and --phase1-in-step). Without the option none of this runs and every gate
keeps requiring the synchronous scheduler.

The step (one engine thread, UniProc worker):
1. vLLM runs ``schedule(N+1)``, ``execute_model(N+1)`` and its native preparation (update
   states, inputs, attention metadata) while forward N runs on the GPU.
2. The engine does not send step N+1's allocation updates when ``schedule`` returns: they are
   held (``EngineLink.hold``) until N's stop decisions are known.
3. Our ``before_forward(N+1)`` hook resolves N first (``resolve``):
   1. the deferred after-forward work of N (``WorkerBridge.finish_step``) closes N's leases;
   2. N's sampled ids are read once (``sampled_tokens``: vLLM's ``prev_sampled_token_ids``,
      one device read that joins N's sampling), for the requests of N that N+1 schedules. vLLM's
      ``schedule(N+1)`` skips a request whose token of N is its max_tokens-th or reaches
      max_model_len (and may leave out another one: the token budget, a preemption); the runner then
      takes it out of the batch and of ``prev_req_id_to_index``. Its stop is decided by
      ``update_from_output(N)``, as on the synchronous server, and it has no row of N+1 to void
      (GPU job 3568311 failed closed on the first such stop next to another request);
   3. the stops are decided with the predicate ``update_from_output`` applies next
      (``EngineLink.stops``: vLLM's ``check_stop`` on a shadow of the request with the token
      appended, then the output stop, ``MarkerStop``; nothing is mutated), for the tokens the
      engine has not applied yet: a step a pass boundary already drained (no worker RPC settled
      it since) had its stops decided by that update (``EngineLink.applied``, async-gemma);
   4. every row of N+1 whose request stopped at N is voided on the device (``void_rows``: its
      Gated-DeltaNet state index is NULL_BLOCK_ID, so the kernels skip it and its running slot
      keeps the state after C'; its FA slot is -1, so no KV is written at C'). On the window
      layout (Gemma 4) there is no GDN state: the token's slot is -1 in the global group and in
      every sliding-window group, so neither the global tail nor the window writes C';
   5. the held allocation updates are sent without the stopped requests' entries, and their
      engine records are restored (H2), so neither the worker nor our records advance;
   6. the continuing requests enter the step and open their leases as before.
4. ``update_from_output(N)`` then runs during forward N+1: a completion sees an untouched slot
   and tail, and the carry is the slot after C' exactly as on the synchronous server. The
   engine requires every predicted stop to be one (``EngineLink.check``); a stop that was not
   predicted ran in N+1 with an open lease and its completion fails closed (``WorkerBridge``).

The positioned carry (--thinking-history strip, thinking_cut): its sources are 'phase1' (the
phase-1 state at e, held apart from the running slot) and 'split' (captured by the local turn 0
prefill forward that ends at X). Both are taken before the request's first decode step and
never from the running slot, so the extra row of a stopped request cannot reach them; the void
still applies (it keeps the KV at C' and the slot untouched, and no Page past X is made). The
lineage rule of a third source, 'snapshot' (a copy of the running slot taken before a
speculative step, option S; not built here and refused), is written in ``SNAPSHOT_RULE``.

Pass boundaries (every client request: admission, aborts, session close, evict; Goal resume;
shutdown) drain the in-flight batch first (vllm/v1/engine/core.py ``_remote_ra_async_drain``):
its outputs are processed without scheduling another step. The lifecycle sweep
(``lifecycle.tick``) does not drain. A worker RPC other than the steady
allocation updates and a completion of a request outside the issued step settles the issued
step's after-forward work first (``settle``).

The window layout (Gemma 4, docs/async-sched/GEMMA.md): a stopped request's speculative
allocation in ``schedule(N+1)`` (a stop N+1 still schedules: an EOS, a stop id, the output stop;
a max_tokens or max_model_len stop allocates nothing there) may free its window's front block one step early (when
(C' - W + 1) % P == 0) and take a fresh block in the global pool and in each sliding group (when
C' % P == 0). Neither is in ``sliding_window.kept_blocks(C')``, the window the commit pins
(``WindowPins.pin_after``) and the worker's carry names (``final_window``, from the rows of step
N: the held update of N+1 is never sent), so the committed window is the synchronous server's;
both stay inside the request's window slot and reply reservation. The void writes no KV at C'
in any group.

Evidence contract (relaxed, the maintainer's choice): tokens, selections, Page vectors and carries
are exact; block ids and incarnations equal the synchronous server's up to a consistent renaming
(the stopped requests' speculative blocks; on the window layout also the window carry's blocks,
incarnations and digest); free counts are read after the next step's schedule took its blocks
(H6, not implemented: a verifier sets them aside). Nothing is hashed per request.
"""
from __future__ import annotations

import os

from .contract import require

CONFIG_KEY = 'async_scheduling'         # startup config: True (off: absent)
WORKER_KEY = 'async_scheduling'         # worker startup record and its echo (True; off: absent)
NULL_BLOCK_ID = 0                       # vLLM's null block: the GDN kernels skip a row at this index
NULL_SLOT = -1                          # an FA slot the KV-update kernels skip
# Worker RPCs that never wait for the issued step's after-forward work.
STEADY = frozenset({'allocation_batch', 'allocation_update'})
# Terminals: they settle only when they name a request of the issued step.
TERMINALS = frozenset({'complete', 'cancel'})
# The lineage rule of a 'snapshot' positioned carry (option S; refused in v1).
SNAPSHOT_RULE = ('snapshot carry admissible iff: position == the commit end; captured from the request\'s own '
                 'running slot by the forward whose lease ended at that position (its sequence), before any later '
                 'forward of the request was launched; the slot was not voided, relocated or released in between; '
                 'same allocation generation and identity token; otherwise the session rebuilds')


# ---- the pin (startup only) -----------------------------------------------------------------------

def check_config(config):
    """True when a startup config pins --async-scheduling, else None. Refused without
    --decode-fastpath2 and with --gdn-device-handoff or --phase1-in-step (v1). Both model profiles
    (the Gemma 4 profile since async-gemma, whose config refuses those two admission flags anyway)."""
    value = None if config is None else config.get(CONFIG_KEY)
    if value is None:
        return None
    require(value is True, 'async_scheduling must be true or absent')
    require(config.get('decode_fastpath2') is True, '--async-scheduling needs --decode-fastpath2')
    from .admission_flags import on, pinned as admission_pinned
    flags = admission_pinned(config)
    require(not on(flags, 'gdn_device_handoff') and not on(flags, 'phase1_in_step'),
            '--async-scheduling refuses --gdn-device-handoff and --phase1-in-step (v1)')
    return True


def pinned(config):
    """True when the startup config pins async scheduling, else None."""
    return check_config(config)


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def allowed():
    """The synchronous-only gates: True when async scheduling is pinned (then vLLM must run it),
    False otherwise (then vLLM must run the synchronous scheduler, as before)."""
    return loaded() is True


def batches():
    """vLLM's max_concurrent_batches the gates require: 2 with async scheduling, else 1."""
    return 2 if allowed() else 1


def worker_flag(startup):
    """Worker startup: the record the engine sent (True or absent), checked once against the pinned
    config. Returns True or None."""
    sent = startup.get(WORKER_KEY)
    require(sent is None or sent is True, 'async_scheduling worker record must be true or absent')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker async_scheduling differs from the pinned startup config')
    return sent


# ---- the engine link (UniProc: one process) ---------------------------------------------------------

_LINK = None


def link():
    """The engine's link (attach_engine), else None."""
    return _LINK


class _Tail:
    """A request's output ids with one more token appended (read-only)."""
    __slots__ = ('ids', 'token')

    def __init__(self, ids, token):
        self.ids, self.token = ids, token

    def __len__(self):
        return len(self.ids) + 1

    def __getitem__(self, index):
        n = len(self.ids) + 1
        if type(index) is int:
            if index < 0:
                index += n
            require(0 <= index < n, 'output id index out of range')
            return self.token if index == n - 1 else self.ids[index]
        return (list(self.ids) + [self.token])[index]

    def __iter__(self):
        yield from self.ids
        yield self.token


class _Shadow:
    """The request as ``update_from_output`` sees it after appending ``token``: the counts and the
    output ids advanced by one; writes (status, stop_reason) stay on the shadow; any other
    attribute reads the request."""

    def __init__(self, request, token):
        object.__setattr__(self, '_request', request)
        object.__setattr__(self, '_token', token)
        object.__setattr__(self, '_written', {})

    @property
    def num_tokens(self):
        return self._request.num_tokens + 1

    @property
    def num_output_tokens(self):
        return self._request.num_output_tokens + 1

    @property
    def output_token_ids(self):
        return _Tail(self._request.output_token_ids, self._token)

    def __getattr__(self, name):
        written = object.__getattribute__(self, '_written')
        if name in written:
            return written[name]
        return getattr(object.__getattribute__(self, '_request'), name)

    def __setattr__(self, name, value):
        self._written[name] = value


def marker_stops(policy, request, token):
    """``MarkerStop(request)`` after appending ``token``, without touching its per-request table:
    when the output without the token is known free of the marker and the token decoded alone holds
    no '>', the answer is False (MarkerStop's own argument); otherwise the whole output is decoded."""
    tokens = list(request.output_token_ids)
    if getattr(policy, 'incremental', False):
        key = (request.request_id, id(request))
        # A channel id (MarkerStop.marker_ids, the Gemma 4 profile with thinking on) decodes the whole output.
        if (policy._checked.get(key) == len(tokens) and token not in getattr(policy, 'marker_ids', ())
                and '>' not in policy.decode([token])):
            return False
    return policy._found(tokens + [token])


class EngineLink:
    """The engine half of option C, read by the worker's resolve on the same thread (UniProc).

    ``stops(rid, token)``: the stop predicate of ``update_from_output`` (check_stop, then the
    output stop), pure. ``flush(exclude)``: the held allocation updates without the entries of
    ``exclude``, whose engine records are restored. ``check(scheduler_output, finished)``: every
    stop predicted for the step just updated is one (else the step is lost: fail closed).
    ``unbatched(rids)``: an out-of-band runner pass took ``rids`` out of the runner's persistent
    batch; the scheduler's next step sends their token ids (``core``: the EngineCore, its batch
    queue)."""

    def __init__(self, scheduler, boundary, check_stop=None, core=None):
        self.scheduler, self.boundary, self.core = scheduler, boundary, core
        if check_stop is None:
            from vllm.v1.core.sched.utils import check_stop
        self.check_stop = check_stop
        self.predicted = None           # the stops decided by the last resolve (None: none decided)
        self.voids = 0                  # rows voided (counters for the runtime row and G3)
        self.resolves = 0
        self.drains = 0
        self.unbatched_requests = 0     # previous-step requests an out-of-band runner pass took out
        self.taken = {}                 # pool -> blocks taken by the schedule in flight (H6)

    def stops(self, rid, token):
        scheduler = self.scheduler
        request = scheduler.requests.get(rid)
        require(request is not None, 'async resolve: sampled request unknown to the scheduler')
        if self.check_stop(_Shadow(request, token), scheduler.max_model_len):
            return True
        policy = getattr(scheduler, '_remote_ra_output_stop', None)
        return policy is not None and marker_stops(policy, request, token)

    def applied(self, rid, cursor):
        """Whether the engine already applied the issued step's token of ``rid``: a pass boundary drained
        that step (``core._remote_ra_async_drain``) and no worker RPC settled it before this resolve (a
        CAPACITY_WAIT refusal, a utility request, a wakeup, a cancel of another request). vLLM's request
        then holds the token sampled at ``cursor``, where the worker's lease of that step ended
        (``num_tokens == cursor + 1``); before its update it does not (``num_tokens == cursor``). Its stops
        were then decided by that update, and a request that stopped there finished before this step was
        scheduled; deciding them again would count the token twice. A request no longer in the scheduler
        was updated (and finished). Any other count fails closed."""
        request = self.scheduler.requests.get(rid)
        if request is None:
            return True
        count = request.num_tokens
        require(count == cursor or count == cursor + 1,
                "async resolve: the request's tokens and the worker's cursor disagree")
        return count == cursor + 1

    def unbatched(self, rids):
        """An out-of-band runner pass (``native_probe``: phase 1's forward, the Goal forward, their cleanups)
        took ``rids`` out of the runner's persistent batch: vLLM's ``_update_states`` removes every request a
        scheduler output does not schedule. vLLM's scheduler still counts them as scheduled in the previous
        step (``prev_step_scheduled_req_ids``), so its next step would send no ``all_token_ids`` for them;
        the asynchronous runner adds a request with output tokens back from those ids alone (its own output
        ids are placeholders), and without them raises KeyError (GPU job 3519531). They leave that set, as
        vLLM's ``reset_prefix_cache`` clears it when it flushes the batch, so the next step sends their ids.
        Only at a pass boundary, where the drain updated every scheduled step (no output placeholder is left
        on a running request), so the ids sent hold every token; with a batch in flight they would miss its
        token: fail closed."""
        rids = frozenset(rids)
        if not rids:
            return
        core, scheduler = self.core, self.scheduler
        require(core is not None and not core.batch_queue,
                'async: an out-of-band runner pass with a batch in flight')
        previous = scheduler.prev_step_scheduled_req_ids
        self.unbatched_requests += len(previous & rids)
        previous.difference_update(rids)

    def flush(self, exclude):
        return self.boundary.flush_updates(exclude=exclude)

    def take_predicted(self):
        predicted, self.predicted = self.predicted, None
        return predicted

    def check(self, finished):
        """After ``update_from_output`` of the step the last resolve decided: every predicted stop
        finished (``finished``: the request ids that left the scheduler or finished in it)."""
        predicted = self.take_predicted()
        if predicted is None:
            return
        missed = set(predicted) - set(finished)
        require(not missed, 'async scheduling: a predicted stop did not stop (step lost; %d request(s))'
                % len(missed))


def attach_engine(core, boundary):
    """compose_engine (after --decode-fastpath2): the allocation updates are held when
    ``schedule`` returns (sent by the worker's resolve, or before any other worker RPC), and the
    worker's resolve reads this link. Startup only."""
    global _LINK
    require(getattr(boundary, 'batched', None) is not None, '--async-scheduling needs --decode-fastpath2')
    require(not getattr(boundary, 'async_hold', False) and _LINK is None, 'async scheduling engine attachment collision')
    require(getattr(core, 'batch_queue', None) is not None and getattr(core, 'batch_queue_size', 1) == 2,
            '--async-scheduling needs vLLM\'s batch queue of two batches')
    scheduler = core.scheduler
    flushing = scheduler.schedule
    native = getattr(flushing, '__wrapped__', None)
    require(native is not None, 'decode fast path 2 schedule wrapper missing')
    require(isinstance(getattr(scheduler, 'prev_step_scheduled_req_ids', None), set),
            'async scheduling: the scheduler\'s previous-step request set is missing')
    scheduler.schedule = native         # schedule no longer sends the updates: they are held
    boundary.async_hold = True
    _LINK = EngineLink(scheduler, boundary, core=core)
    core._remote_ra_async_link = _LINK
    return _LINK


def detach_engine():
    """Tests and shutdown: forget the link."""
    global _LINK
    _LINK = None


# ---- the worker half ---------------------------------------------------------------------------------

def attach_worker(owner, bridge, qwen, runner):
    """attach_loaded_worker / attach_window_worker (after --decode-fastpath2): the guard defers the
    step's after-forward work to the next resolve or settle; the runner stashes the step's attention
    metadata for the void. On the window layout (Gemma 4) the void writes the sliding-window groups'
    slot mappings too (``owner.async_sliding``: those groups; None on the Qwen layouts)."""
    require(getattr(bridge, 'decode_fastpath2', False), '--async-scheduling needs --decode-fastpath2')
    require(not getattr(owner, 'handoff', False) and not getattr(owner, 'phase1_in_step', False),
            '--async-scheduling refuses --gdn-device-handoff and --phase1-in-step (v1)')
    require(not getattr(bridge, 'async_scheduling', False), 'async scheduling attached twice')
    require(getattr(runner, 'use_async_scheduling', False) is True, 'the loaded runner does not schedule asynchronously')
    window = getattr(getattr(owner, 'layout', None), 'state', None) == 'sliding'
    require(window == (getattr(qwen, 'state_kind', None) == 'sliding') and
            window == (getattr(getattr(qwen, 'backend', None), 'window', False) is True),
            'async scheduling: the window layout needs the window owner, lifecycle and backend together')
    owner.async_sliding = tuple(owner.hybrid_groups) if window else None
    require(not window or (owner.async_sliding and not set(owner.async_sliding) & set(owner.attention_groups)),
            'async scheduling: the window layout\'s sliding groups are required')
    owner.async_scheduling = bridge.async_scheduling = qwen.async_scheduling = True
    runner._remote_ra_live_v2_voided = frozenset()


def guarded(runner, owner, bridge, execute, output, owned, args, kwargs):
    """The generation guard under async scheduling: no device drain before the step (a connector
    check), no entry here (the resolve in before_forward enters the continuing requests once the
    previous step's stops are known), and the step's after-forward work issued, not joined."""
    require(bridge.entered is None, 'async: an earlier step was entered and not issued')
    try:
        if owner.startup.get('source_closure_verified'):
            from .connector_guard import empty_scheduler_metadata
            empty_scheduler_metadata(output)
            check = getattr(owner.io, 'connector_check', None)
            if check is not None:
                check()
        result = execute(output, *args, **kwargs)
        entered = bridge.entered
        require(entered is not None, 'async: the step ran without its resolve (before_forward)')
        bridge.entered = None
        if entered:
            bridge.qwen.issue_execute(*entered)
        bridge.issued = entered or None
    except BaseException as original:
        bridge.discard_step()
        try:
            fail(bridge)
        except BaseException as drain_error:
            raise ExceptionGroup('generation failed; device completion unknown; leases retained',
                                 [original, drain_error]) from original
        raise
    return result


def fail(bridge):
    """A failed async step: every entered and issued request is poisoned and joined, as a failed
    synchronous execute is."""
    qwen = bridge.qwen
    identities = tuple(bridge.entered or ()) + tuple(bridge.issued or ())
    bridge.entered = bridge.issued = None
    if not identities:
        return
    live = tuple(i for i in identities if i.request in qwen.requests and not qwen.requests[i.request].terminal)
    if live:
        qwen.leave_failed_execute(*live)
        qwen.recover_failed_execute(*live)


def settle(bridge):
    """The issued step's after-forward work (its leases close on its rows), joined now: before the
    next step's leases, a pass-boundary RPC, or a terminal of one of its requests."""
    issued = bridge.issued
    if issued is None:
        return None
    bridge.issued = None
    receipts = bridge.finish_step()
    bridge.qwen.leave_issued(*issued)
    return receipts


def settles(bridge, operation, payload):
    """worker_dispatch: whether ``operation`` must settle the issued step first."""
    issued = getattr(bridge, 'issued', None)
    if issued is None or operation in STEADY:
        return False
    if operation in TERMINALS:
        request = payload.get('request') if type(payload) is dict else None
        return any(i.request == request for i in issued)
    return True


def sampled_tokens(runner, rids):
    """{request: the token sampled for it by the previous step}, for the requests of ``rids`` that
    sampled (vLLM's ``prev_req_id_to_index`` leaves out discarded partial prefills): one device read.
    ``rids`` are requests of this step too: a request this step does not schedule left the persistent
    batch in ``_update_states``, and ``InputBatch.remove_request`` took it out of ``prev_req_id_to_index``."""
    batch = runner.input_batch
    ids, index = getattr(batch, 'prev_sampled_token_ids', None), getattr(batch, 'prev_req_id_to_index', None)
    wanted = [rid for rid in rids if index and rid in index]
    if not wanted:
        return {}
    require(ids is not None and ids.ndim == 2 and ids.shape[1] == 1,
            'async resolve: the previous step\'s sampled ids are unavailable')
    host = ids[:, 0].tolist()
    return {rid: int(host[index[rid]]) for rid in wanted}


def void_positions(runner, scheduler_output, voided):
    """(batch rows, token offsets) of the voided requests in this step (one token each: decode)."""
    scheduled = scheduler_output.num_scheduled_tokens
    rows, tokens, offset = [], [], 0
    for row, rid in enumerate(runner.input_batch.req_ids):
        count = int(scheduled[rid])
        if rid in voided:
            require(count == 1, 'async void: a stopped request scheduled more than one token')
            rows.append(row)
            tokens.append(offset)
        offset += count
    require(len(rows) == len(voided), 'async void: a stopped request is missing from the batch')
    return rows, tokens


def void_rows(runner, scheduler_output, voided, attention_groups, sliding_groups=None):
    """The voided rows on the device, after vLLM's builders filled this step's attention metadata
    (and, under FULL graphs, their persistent buffers) and before the forward: every Gated-DeltaNet
    group's state index of the row is NULL_BLOCK_ID; the FA slot of its token is -1 in every FA
    layer's metadata and per-layer slot mapping and in each FA group's slot mapping
    (``attention_groups``). Returns (state index tensors, slot mappings) written.

    ``sliding_groups`` (the window layout, Gemma 4; None on the Qwen layouts): there is no
    Gated-DeltaNet state (a GDN state index is refused), and every attention layer is a global
    or a sliding-window layer with KV: the token's slot is -1 in every layer's metadata and
    per-layer slot mapping and in the slot mapping of each global group (``attention_groups``)
    and each sliding-window group (``sliding_groups``), every one of which must be there. The
    global tail and the window then never hold KV at C', and a fresh block allocated for it
    (C' % P == 0) is never written."""
    stash = getattr(runner, '_remote_ra_step_metadata', None)
    require(stash is not None, 'async void: the step\'s attention metadata was not stashed')
    attn, layers, groups = stash
    require(isinstance(attn, dict) and attn, 'async void: one attention metadata dict per step (no micro-batches)')
    rows, tokens = void_positions(runner, scheduler_output, voided)
    window = sliding_groups is not None
    seen, states, slots = set(), 0, 0

    def write(mapping):
        nonlocal slots
        if mapping is not None and id(mapping) not in seen:
            seen.add(id(mapping))
            mapping[tokens] = NULL_SLOT
            slots += 1
    for name, md in attn.items():
        index = getattr(md, 'non_spec_state_indices_tensor', None)
        if index is not None:
            require(not window, 'async void: a Gated-DeltaNet state index on the window layout')
            if id(md) not in seen:
                seen.add(id(md))
                require(index.shape[0] >= len(runner.input_batch.req_ids), 'async void: short GDN state index')
                index[rows] = NULL_BLOCK_ID
                states += 1
            continue
        mapping = getattr(md, 'slot_mapping', None)
        if mapping is None:
            continue
        write(mapping)                                  # an FA layer (window layout: global or sliding)
        if isinstance(layers, dict):
            write(layers.get(name))
    if window:
        # Every KV group of the window layout holds the token's KV: each group's mapping is required.
        every = tuple(attention_groups) + tuple(sliding_groups)
        require(isinstance(groups, dict) and all(groups.get(gid) is not None for gid in every),
                'async void: a global or sliding-window group\'s slot mapping is missing')
        for gid in every:
            write(groups[gid])
        require(slots > 0, 'async void found no slot mapping on the window layout')
        return states, slots
    for gid in attention_groups:
        write(groups.get(gid) if isinstance(groups, dict) else None)
    require(states > 0 and slots > 0, 'async void found no GDN state index or no FA slot mapping')
    return states, slots


def resolve(runner, bridge, scheduler_output):
    """before_forward(N+1): settle N, decide its stops, void their rows of N+1, send the held
    allocation updates of the rest, and enter the continuing requests."""
    engine = link()
    require(engine is not None, 'async resolve without the engine link')
    issued = bridge.issued
    scheduled = scheduler_output.num_scheduled_tokens
    stops = None
    if issued is not None:
        settle(bridge)
        qwen = bridge.qwen
        # Only the requests of N that N+1 schedules: vLLM's schedule(N+1) skips a request whose token of N is
        # its max_tokens-th or reaches max_model_len (and may leave out another one), and its runner then takes
        # it out of the batch and of prev_req_id_to_index (GPU job 3568311). Its stop is decided by
        # update_from_output(N), as on the synchronous server; it has no row here to void.
        sampled = tuple(i.request for i in issued
                        if i.request in scheduled and i.request in qwen.requests
                        and qwen.requests[i.request].prompt_complete)
        tokens = sampled_tokens(runner, sampled)
        require(set(tokens) == set(sampled), 'async resolve: a request sampled without its token')
        # A pass boundary may have drained the issued step already (EngineLink.applied): its stops were
        # decided by that update and its stopped requests are not in this step.
        stops = frozenset(rid for rid, token in tokens.items()
                          if not engine.applied(rid, qwen.requests[rid].cursor) and engine.stops(rid, token))
    voided = frozenset(rid for rid in scheduled if stops and rid in stops)
    runner._remote_ra_live_v2_voided = voided
    engine.predicted = stops
    engine.resolves += 1
    owner = bridge.owner
    if voided:
        void_rows(runner, scheduler_output, voided, owner.attention_groups, getattr(owner, 'async_sliding', None))
        engine.voids += len(voided)
    engine.flush(voided)
    entered = tuple(owner.records[rid]['identity'] for rid in runner.input_batch.req_ids
                    if rid in owner.records and rid not in voided)
    if entered:
        bridge.qwen.enter_execute(*entered)
    bridge.entered = entered
    return voided
