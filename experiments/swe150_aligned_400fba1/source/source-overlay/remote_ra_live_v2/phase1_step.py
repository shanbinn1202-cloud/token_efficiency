"""Phase 1 inside a decode step (admission design (b), ``serve.py --phase1-in-step``).

Today an admission runs its phase 1 as a separate ``execute_model`` that holds
only the probe, so every running decode waits for it (and the probe-only pass
drops every decode from the persistent batch, which the next step re-adds).
With the flag on, on a server whose sessions decode together:

- ``Phase1Coordinator.run_steps`` leases exactly as ``run`` and yields one
  ``Phase1Work``; the admission parks (goal_async, kind phase 1) when the
  injector accepts it (another request is running and the lease carries no
  Goal forward), else the work runs in place: today's probe-only pass;
- ``Phase1Steps.before`` takes the parked work into the next step after
  ``scheduler.schedule()`` when its tokens fit what the step left of the token
  budget and a batch row is free: ``phase1_arm`` (the worker's phase-1 checks
  and lease registration, as today), then an augmented copy of the scheduler
  output carries the probe's NewRequestData, token count and zero list; the
  scheduler keeps and updates from the original;
- ``Phase1Steps.after`` collects the probe (``phase1_collect``: today's receipt,
  its frame measured from the arm, and the probe row removed from the batch)
  and strips the probe's sampled token from the model output; the admission
  resumes at the next engine pass (selection, adopt, reserve, ack as today);
- fallbacks keep the separate pass: nothing else scheduled ('probe_only'),
  the tokens do not fit ('fallback'; it drops the decodes for that pass, as
  today), or no row is free for two steps ('fallback'); ``settle`` runs them
  at the start of the next pass, where admissions run today. Phase 1 is never
  chunked: PHASE1_TOO_LONG stays at one native forward (16,384 tokens);
- a server with one session (max_num_seqs 1) has no park table, so S = 1
  always takes the probe-only path, bit for bit.

CUDA graphs: a step that carries a probe holds a prefill, so it never replays
a FULL decode graph: piecewise/full run it with the piecewise graphs and
full-decode runs it eagerly (vLLM's own dispatch; the runtime forces eager
only for a probe-only pass). The probe's hooks run outside ``_model_forward``
and allocate nothing inside a captured region.

Engine thread only. Nothing here hashes per request or per step.
"""
from __future__ import annotations

from concurrent.futures import Future
import copy
import time

from .contract import require

FLAG = 'phase1_in_step'
ROW_WAIT_STEPS = 1        # a step without a free batch row; the next one falls back


class Phase1Work:
    """One admission's phase 1 (no Goal forward when injected), as run_steps yields it."""
    kind = 'phase1'

    def __init__(self, coordinator, request, payload, query_tokens):
        lease = payload['lease']
        self.coordinator, self.request, self.payload = coordinator, request, payload
        self.query_tokens = query_tokens
        self.lease_id, self.count = lease['lease_id'], lease['end'] - lease['start']
        self.goal = lease['goal'] is not None
        self.mode = self.step_ns = self.step_rows = None
        self.fallback = None
        self.future = None
        self.started = None

    def run(self):
        """The separate probe-only pass: today's phase-1 RPC."""
        if self.mode is None:
            self.mode = self.fallback or 'probe_only'
        return self.coordinator.proxy.call('phase1', self.payload, query_tokens=self.query_tokens)


def augmented(out, work):
    """A copy of the step's scheduler output that also schedules the probe row."""
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.sampling_params import SamplingParams
    lease, tokens = work.payload['lease'], work.payload['tokens']
    rid, start = lease['lease_id'], lease['start']
    count = len(tokens) - start
    require(count == work.count and rid not in out.num_scheduled_tokens, 'in-step probe row invalid')
    new = NewRequestData(rid, list(tokens), [], SamplingParams(temperature=0, max_tokens=1), None,
                         tuple(list(r) for r in lease['rows']), start, None)
    aug = copy.copy(out)
    aug.scheduled_new_reqs = list(out.scheduled_new_reqs) + [new]
    aug.num_scheduled_tokens = dict(out.num_scheduled_tokens)
    aug.num_scheduled_tokens[rid] = count
    aug.total_num_scheduled_tokens = out.total_num_scheduled_tokens + count
    blocks = getattr(out, 'num_common_prefix_blocks', None)
    if blocks is not None:
        aug.num_common_prefix_blocks = [0] * len(blocks)     # no cascade prefix with the probe
    # The lease's own fresh blocks only: borrowed R/tail KV is never zeroed.
    aug.new_block_ids_to_zero = list(out.new_block_ids_to_zero or ()) + list(lease['zero'])
    return aug


def stripped(model_output, rid):
    """The model output without the probe's row (its sampled token). Only plain
    per-row outputs are rewritten; any other output keeps the extra row, which the
    scheduler never reads (it looks rows up by its own scheduled request ids)."""
    ids = getattr(model_output, 'req_ids', None)
    index = getattr(model_output, 'req_id_to_index', None)
    if (type(ids) is not list or type(index) is not dict or rid not in index or
            getattr(model_output, 'logprobs', None) is not None or
            getattr(model_output, 'pooler_output', None) is not None):
        return model_output
    drop = index[rid]
    out = copy.copy(model_output)
    out.req_ids = [r for k, r in enumerate(ids) if k != drop]
    out.req_id_to_index = {r: k for k, r in enumerate(out.req_ids)}
    sampled = getattr(model_output, 'sampled_token_ids', None)
    if type(sampled) is list and len(sampled) == len(ids):
        out.sampled_token_ids = [v for k, v in enumerate(sampled) if k != drop]
    for name in ('prompt_logprobs_dict', 'num_nans_in_logits'):
        value = getattr(model_output, name, None)
        if type(value) is dict and rid in value:
            setattr(out, name, {k: v for k, v in value.items() if k != rid})
    return out


class Phase1Steps:
    """The engine's step injector: parked phase-1 works, at most one per step."""

    def __init__(self, coordinator, proxy, *, max_rows, has_requests):
        require(type(max_rows) is int and max_rows > 1 and callable(has_requests),
                'in-step phase 1 needs several batch rows and the scheduler')
        self.coordinator, self.proxy = coordinator, proxy
        self.max_rows, self.has_requests = max_rows, has_requests
        self.max_tokens = coordinator.max_tokens
        self.waiting, self.inflight, self.row_waits = [], None, {}
        self.counts = dict(mixed=0, probe_only=0, fallback=0)

    # ---- park table side (goal_async) ----

    def accepts(self, work):
        """Park a work only while something else runs and it carries no Goal forward
        (a Goal forward runs in place; while decodes run that is a 'fallback')."""
        running = bool(self.has_requests())
        if work.goal and running:
            work.fallback = 'fallback'
        return running and not work.goal

    def queue(self, work):
        work.future = Future()
        self.waiting.append(work)
        return work.future

    def cancel(self, work):
        """An aborted admission: a work not taken into a step never runs."""
        if work in self.waiting:
            self.waiting.remove(work)
        if work.future is not None and not work.future.done():
            work.future.cancel()

    def pending(self):
        return bool(self.waiting)

    # ---- the separate pass (fallbacks), at a pass start or an idle engine ----

    def settle(self):
        """Run in place every waiting work that cannot go into a step."""
        idle = not self.has_requests()
        for work in [w for w in self.waiting if w.fallback is not None or idle]:
            self.waiting.remove(work)
            work.mode = work.fallback or 'probe_only'
            self.counts[work.mode] += 1
            try:
                work.future.set_result(work.run())
            except BaseException as exc:
                work.future.set_exception(exc)

    def idle(self):
        """The engine has no request to step: waiting works take the separate pass."""
        self.settle()

    # ---- the step (EngineCore.step) ----

    def before(self, out):
        """The scheduler output to execute: ``out``, or a copy that carries the probe."""
        if self.inflight is not None or not self.waiting:
            return out
        work = next((w for w in self.waiting if w.fallback is None), None)
        if work is None:
            return out
        others = len(out.num_scheduled_tokens)
        if not others:
            work.fallback = 'probe_only'          # nothing else runs: its own pass
            return out
        if work.count > self.max_tokens - out.total_num_scheduled_tokens:
            work.fallback = 'fallback'            # does not fit: the separate pass
            return out
        if others + 1 > self.max_rows:
            waited = self.row_waits[work.request] = self.row_waits.get(work.request, 0) + 1
            if waited > ROW_WAIT_STEPS:
                work.fallback = 'fallback'
            return out
        self.waiting.remove(work)
        self.row_waits.pop(work.request, None)
        try:
            receipt = self.proxy.call('phase1_arm', work.payload)
            require(type(receipt) is dict and receipt.get('operation') == 'phase1_arm' and
                    receipt.get('binding') == work.payload['binding'] and
                    receipt.get('lease_id') == work.lease_id and
                    receipt.get('lease_digest') == work.payload['lease_digest'],
                    'in-step phase-1 arm receipt mismatch')
            aug = augmented(out, work)
        except BaseException as exc:
            work.future.set_exception(exc)        # the admission fails through its cleanup
            return out
        work.mode, work.step_rows, work.started = 'mixed', others + 1, time.perf_counter_ns()
        self.inflight = work
        self.counts['mixed'] += 1
        return aug

    def after(self, executed, model_output):
        """Collect the step's probe; the model output without its row."""
        work, self.inflight = self.inflight, None
        if work is None:
            return model_output
        work.step_ns = time.perf_counter_ns() - work.started
        try:
            result = self.proxy.call('phase1_collect', work.payload, query_tokens=work.query_tokens)
            work.future.set_result(result)
        except BaseException as exc:
            work.future.set_exception(exc)
        return stripped(model_output, work.lease_id)


def attach_engine(core, admission):
    """Engine startup (compose_engine), after goal_async.attach: the injector, when
    the flag is on and the sessions decode together (a park table exists)."""
    from .admission_flags import on
    if not on(admission, FLAG):
        return None
    goals = getattr(core, '_remote_ra_goal_admissions', None)
    if goals is None:
        return None             # one session: every phase 1 is the probe-only pass
    endpoint = core._remote_ra_private_v2_endpoint
    require(goals.phase1 is None and getattr(core, '_remote_ra_phase1_steps', None) is None and
            not endpoint.phase1.in_step, 'in-step phase 1 attached twice')
    steps = Phase1Steps(endpoint.phase1, core._remote_ra_live_v2_boundary.proxy,
                        max_rows=core.vllm_config.scheduler_config.max_num_seqs,
                        has_requests=core.scheduler.has_requests)
    goals.phase1 = core._remote_ra_phase1_steps = steps
    endpoint.phase1.in_step = True
    return steps


# ---- worker side -----------------------------------------------------------------------------

def step_slice(runner, scheduler_output, rid):
    """(offset, count) of ``rid``'s tokens in the step's flattened batch-row order."""
    scheduled = scheduler_output.num_scheduled_tokens
    offset = 0
    for other in runner.input_batch.req_ids:
        if other == rid:
            return offset, int(scheduled[rid])
        offset += int(scheduled[other])
    require(False, 'in-step probe row missing from the batch')


def probe_before_forward(runner, probe, scheduler_output, positions):
    """The armed probe row of a mixed step: its positions and carry restore, its IO its own."""
    from remote_ra_growing_worker.tensors import attributed
    rid = probe.active['request']
    offset, count = step_slice(runner, scheduler_output, rid)
    with attributed(probe.backend.io, probe.active['identity'].request):
        return probe.before_forward(positions, offset)


def probe_after_forward(runner, probe, scheduler_output, hidden):
    from remote_ra_growing_worker.tensors import attributed
    rid = probe.active['request']
    offset, count = step_slice(runner, scheduler_output, rid)
    with attributed(probe.backend.io, probe.active['identity'].request):
        return probe.after_forward(hidden[offset:offset + count])


def retire_row(runner, rid):
    """Remove one probe row from the runner exactly as _update_states removes a
    finished request (the other rows stay: nothing is dropped and re-added)."""
    batch = runner.input_batch
    if callable(getattr(batch, 'remove_request', None)) and callable(getattr(batch, 'condense', None)):
        runner.requests.pop(rid, None)
        getattr(runner, 'num_prompt_logprobs', {}).pop(rid, None)
        late = getattr(runner, 'late_interaction_runner', None)
        if late is not None:
            late.on_requests_finished({rid})
        batch.remove_request(rid)
        batch.condense()
        batch.refresh_metadata()
    else:
        # CPU fakes: their own finished-request path, which keeps unscheduled rows.
        from types import SimpleNamespace
        runner._update_states(SimpleNamespace(finished_req_ids={rid}, new_block_ids_to_zero=[],
                                              scheduled_new_reqs=[], num_scheduled_tokens=None))
    runner.mamba_state_idx.pop(rid, None)
    runner._selective_load_plans.pop(rid, None)
