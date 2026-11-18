"""--async-scheduling: a running request taken out of the runner's batch by an out-of-band pass (async-fix2).
CPU only, no torch.

GPU job 3519531 (perf-async-g1 g1 r2, runtime 4a498dc): the async servers fea, fga, tea and tga died when the
b4 cell's second turn began (41 of 97 requests), xea when x4's did (8 of 12); the synchronous servers passed.
EngineCore's step raised KeyError in vLLM's GPU runner (_update_states, gpu_model_runner.py): a cached request
that was not in the persistent batch, with one output token, read ``all_token_ids[req_id]``, and the scheduler
had sent no token ids for it. The mechanism:

1. step N schedules request A (the first of the cell's turns admitted) and samples its first token; the next
   admission is a pass boundary, so the engine drains (updates) N first;
2. that admission's phase 1 runs as a separate pass (native_probe.NativeQueryForward): the runner executes a
   SchedulerOutput that schedules only the probe, and vLLM's _update_states takes every request that output does
   not schedule out of the persistent batch (A included); the probe's cleanup (an empty output) does the same;
3. step N+1 schedules A as a cached request: vLLM's _make_cached_request_data sends ``all_token_ids`` only for a
   request not in ``prev_step_scheduled_req_ids``, and A was scheduled in N, so it sends none;
4. the runner finds A outside its batch; under async scheduling it rebuilds a request with output tokens from
   ``all_token_ids`` alone (its own output ids are -1 placeholders): KeyError. Synchronously it re-adds A from
   its own cached ids, which hold the real tokens, so the synchronous servers ran.

The fix (async_sched.EngineLink.unbatched, called by NativeQueryForward._runner_pass): the probe driver reports
the requests a pass took out of the batch, and they leave ``prev_step_scheduled_req_ids`` (as vLLM's
reset_prefix_cache clears that set when it flushes the batch), so the next step sends their token ids. It is
refused unless no batch is in flight and none of them holds an output placeholder (the ids would miss a token).

What runs here: the REAL overlay sources of the three parts involved, executed from their files:
- vLLM's GPU runner ``_update_states`` (and the real CachedRequestState), on a stand-in of the persistent batch
  (InputBatch's add / remove / condense: rows written from the prompt and the cached output ids);
- vLLM's scheduler ``_make_cached_request_data`` on the real overlay Scheduler (test_deferred_free_pools'
  construction with two batches on the Qwen decoupled layout), with schedule()'s previous-step bookkeeping;
- the real NativeQueryForward ``_execute`` / ``_cleanup_request`` (and, from Git, 4a498dc's) and EngineLink.
The step order is EngineCore.step_with_batch_queue's at two batches (schedule and execute N+1, then update N; a
pass boundary drains first); every input id the runner would feed is checked against the tokens actually
sampled. On 4a498dc (the base module from Git, or the fix removed) the second step raises the GPU's KeyError.
"""
import ast
import copy
from collections import deque
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_deferred_free_pools as DFP  # noqa: E402
import test_kv_pools as KP  # noqa: E402
from remote_ra_live_v2 import async_sched as A  # noqa: E402
from remote_ra_live_v2 import native_probe  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402

BASE = '4a498dcb95568c0cc2bc7d54ce83d05038a197e6'      # the runtime of GPU job 3519531
OVERLAY = ROOT / 'source-overlay'
# The request ids of the GPU failure's dump (server-fea.log): A was running, B the new turn of step N+1.
RID_A, RID_B = 'chatcmpl-a8279f7326b5fedd-8fc7b5c1', 'chatcmpl-8193d072643aa2eb-b58c501e'


# ---- the real sources -----------------------------------------------------------------------------------

def _definition(rel, cls, name):
    """The AST of ``cls.name`` (a method) or of the class ``name`` (cls None) in an overlay file."""
    tree = ast.parse((OVERLAY / rel).read_text('utf-8'))
    for node in tree.body:
        if cls is None and isinstance(node, ast.ClassDef) and node.name == name:
            return node
        if cls is not None and isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError('%s: %s.%s not found' % (rel, cls, name))


def _execute(rel, node, namespace):
    namespace = dict(namespace, __name__='_async_probe_batch_' + node.name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'overlay:' + rel, 'exec'), namespace)
    return namespace[node.name]


def _method_source(rel, cls, name):
    return ast.get_source_segment((OVERLAY / rel).read_text('utf-8'), _definition(rel, cls, name))


RUNNER = 'vllm/v1/worker/gpu_model_runner.py'
BATCH = 'vllm/v1/worker/gpu_input_batch.py'
SCHEDULER = 'vllm/v1/core/sched/scheduler.py'


class SamplingType:
    GREEDY, RANDOM, RANDOM_SEED = 'greedy', 'random', 'random_seed'


def sampling(**kw):
    """vLLM's SamplingParams as the runner's _update_states reads them (greedy, no prompt logprobs)."""
    return NS(sampling_type=SamplingType.GREEDY, prompt_logprobs=None, **kw)


# vLLM's CachedRequestState (gpu_input_batch.py), its dataclass executed from the overlay.
CachedRequestState = _execute(BATCH, _definition(BATCH, None, 'CachedRequestState'), dict(
    dataclass=__import__('dataclasses').dataclass, MultiModalFeatureSpec=object, SamplingParams=object,
    torch=NS(Generator=object, Tensor=object), LoRARequest=object, LogprobsTensors=object, PoolingParams=object,
    PoolingStates=object, length_from_prompt_token_ids_or_embeds=lambda ids, embeds: len(ids)))
# vLLM's GPUModelRunner._update_states, executed from the overlay.
update_states = _execute(RUNNER, _definition(RUNNER, 'GPUModelRunner', '_update_states'), dict(
    Callable=__import__('collections.abc').abc.Callable, CachedRequestState=CachedRequestState,
    SamplingType=SamplingType, torch=NS(Generator=None), cast=lambda kind, value: value,
    VllmModelForPooling=object, get_pp_group=lambda: NS(is_last_rank=True),
    update_scheduler_for_invalid_drafts=None, update_ngram_gpu_tensors_incremental=None))
# vLLM's SchedulerOutput / NewRequestData / CachedRequestData (dataclasses only).
OUTPUT = KP.load((OVERLAY / 'vllm/v1/core/sched/output.py').read_text('utf-8'), '_async_probe_batch_output', {},
                 'overlay:vllm/v1/core/sched/output.py')


def vllm_modules():
    """The vLLM modules native_probe imports at call time: the real SchedulerOutput dataclasses."""
    names = ('vllm', 'vllm.v1', 'vllm.v1.core', 'vllm.v1.core.sched', 'vllm.sampling_params', 'vllm.distributed',
             'vllm.distributed.kv_transfer', 'vllm.distributed.kv_transfer.kv_connector',
             'vllm.distributed.kv_transfer.kv_connector.v1', 'vllm.distributed.kv_transfer.kv_connector.v1.offloading',
             'vllm.distributed.kv_transfer.kv_connector.v1.offloading.common')
    mods = {n: types.ModuleType(n) for n in names}
    mods['vllm.v1.core.sched.output'] = OUTPUT
    mods['vllm.sampling_params'].SamplingParams = sampling
    mods['vllm.distributed.kv_transfer.kv_connector.v1.offloading.common'].OffloadingConnectorMetadata = \
        lambda **kw: NS(**kw)
    return mods


def base_probe_module():
    """4a498dc's native_probe.py (from Git; None in a Git archive), as a module of this package."""
    try:
        text = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(ROOT), 'show',
                               BASE + ':source-overlay/remote_ra_live_v2/native_probe.py'],
                              capture_output=True, check=True, timeout=60).stdout.decode('utf-8')
    except (OSError, subprocess.SubprocessError):
        return None
    module = types.ModuleType('remote_ra_live_v2._native_probe_4a498dc')
    module.__package__ = 'remote_ra_live_v2'
    exec(compile(text, '4a498dc:remote_ra_live_v2/native_probe.py', 'exec'), module.__dict__)
    return module


# ---- the runner: vLLM's _update_states on a stand-in of its persistent batch ----------------------------

class Table:
    def __init__(self):
        self.rows = {}

    def add_row(self, ids, row):
        self.rows[row] = [list(g) for g in ids]

    def append_row(self, ids, row):
        for g, new in zip(self.rows[row], ids):
            g.extend(new)

    def clear_row(self, row):
        self.rows.pop(row, None)

    def move_row(self, src, dst):
        self.rows[dst] = self.rows.pop(src)


class InputBatch:
    """vLLM's InputBatch as _update_states uses it: add_request fills the smallest removed row (else appends) and
    writes the row's token ids from the prompt and the cached output ids; remove_request leaves a hole; condense
    moves the last rows into the holes. ``prev_*``: what the previous sampling step left (async scheduling)."""

    def __init__(self):
        self._req_ids, self.req_id_to_index, self.removed = [], {}, []
        self.token_ids_cpu, self.num_tokens_no_spec, self.num_prompt_tokens = {}, {}, {}
        self.num_computed_tokens_cpu, self.block_table = {}, Table()
        self.prev_sampled_token_ids, self.prev_req_id_to_index = None, None

    @property
    def req_ids(self):
        return self._req_ids

    @property
    def num_reqs(self):
        return len(self.req_id_to_index)

    def add_request(self, request):
        if self.removed:
            row = min(self.removed)
            self.removed.remove(row)
            self._req_ids[row] = request.req_id
        else:
            row = len(self._req_ids)
            self._req_ids.append(request.req_id)
        self.req_id_to_index[request.req_id] = row
        self.num_prompt_tokens[row] = len(request.prompt_token_ids)
        self.token_ids_cpu[row] = list(request.prompt_token_ids) + list(request.output_token_ids)
        self.num_tokens_no_spec[row] = request.num_tokens
        self.num_computed_tokens_cpu[row] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, row)
        return row

    def remove_request(self, req_id):
        row = self.req_id_to_index.pop(req_id, None)
        if row is None:
            return None
        self.removed.append(row)
        self._req_ids[row] = None
        self.block_table.clear_row(row)
        return row

    def condense(self):
        holes, self.removed = sorted(self.removed), []
        for hole in holes:
            while self._req_ids and self._req_ids[-1] is None:
                self._req_ids.pop()
            if hole >= len(self._req_ids):
                break
            last = len(self._req_ids) - 1
            rid = self._req_ids.pop()
            self._req_ids[hole] = rid
            self.req_id_to_index[rid] = hole
            for table in (self.token_ids_cpu, self.num_tokens_no_spec, self.num_prompt_tokens,
                          self.num_computed_tokens_cpu):
                table[hole] = table.pop(last)
            self.block_table.move_row(last, hole)
        while self._req_ids and self._req_ids[-1] is None:
            self._req_ids.pop()

    def refresh_metadata(self):
        pass

    def update_req_spec_token_ids(self, request, scheduled_spec_tokens):
        assert not scheduled_spec_tokens.get(request.req_id)


class Runner:
    """The GPU runner around the real _update_states: ``forward`` feeds each row's input ids as vLLM's
    _prepare_input_ids does (the row's token ids, the last one replaced by the previous step's sampled id when
    the request sampled there) and checks them against the tokens actually sampled; ``sample`` keeps the
    sampled ids as _bookkeeping_sync does (async: a -1 placeholder in the cached output ids and the row, the
    ids on the device; sync: the ids). A pass that schedules only the armed probe returns its rows (the
    no-sample seam)."""

    _update_states = update_states

    def __init__(self, asynchronous, truth):
        self.use_async_scheduling, self.truth = asynchronous, truth
        self.requests, self.num_prompt_logprobs, self.encoder_cache = {}, {}, {}
        self.late_interaction_runner = NS(on_requests_finished=lambda ids: None,
                                          register_request=lambda rid, params: None)
        self.input_batch = InputBatch()
        self.speculative_config, self.use_async_spec_decode, self.is_pooling_model = None, False, False
        self.uses_mrope, self.uses_xdrope_dim, self.device = False, 0, 'cpu'
        self.execute_model_state, self.max_num_tokens = None, 16384
        self.mamba_state_idx, self._selective_load_plans = {}, {}
        self.fed, self._remote_ra_native_query_forward = [], None

    def _zero_block_ids(self, ids):
        pass

    def _may_reorder_batch(self, scheduler_output):
        pass

    def execute_model(self, so):
        self._update_states(so)
        batch = self.input_batch
        assert sorted(batch.req_ids) == sorted(so.num_scheduled_tokens), (batch.req_ids, so.num_scheduled_tokens)
        driver = self._remote_ra_native_query_forward
        if driver is not None and driver.active is not None and batch.req_ids == [driver.active['request']]:
            driver.active['rows'] = rows = NS(probe=driver.active['request'])
            return rows                                   # the no-sample seam: no sampling state
        for row, rid in enumerate(batch.req_ids):
            start, count = batch.num_computed_tokens_cpu[row], so.num_scheduled_tokens[rid]
            ids = batch.token_ids_cpu[row][start:start + count]
            prev = batch.prev_req_id_to_index
            if batch.prev_sampled_token_ids is not None and prev and rid in prev:
                ids[-1] = batch.prev_sampled_token_ids[prev[rid]]
            assert ids == self.truth[rid][start:start + count], ('stale input ids', rid, start, ids)
            self.fed.append((rid, start, tuple(ids)))
        return None

    def sample(self, so):
        """The step's sampled ids: {rid: id} for the rows that computed their last known token."""
        batch, sampled = self.input_batch, {}
        for row, rid in enumerate(batch.req_ids):
            start, count = batch.num_computed_tokens_cpu[row], so.num_scheduled_tokens[rid]
            if start + count != len(self.truth[rid]):
                continue                                  # a prefill chunk: nothing sampled
            sampled[rid] = token = 1000 + (len(self.truth[rid]) * 7 + len(rid)) % 97
            self.truth[rid].append(token)
            kept = -1 if self.use_async_scheduling else token
            self.requests[rid].output_token_ids.append(kept)
            batch.token_ids_cpu[row].append(kept)
            batch.num_tokens_no_spec[row] += 1
        if self.use_async_scheduling:
            batch.prev_sampled_token_ids = {batch.req_id_to_index[rid]: t for rid, t in sampled.items()}
            batch.prev_req_id_to_index = {rid: batch.req_id_to_index[rid] for rid in sampled}
        return sampled


# ---- the engine: the real Scheduler's cached request data, in step_with_batch_queue's order --------------

class Request:
    """vLLM's Request as _make_cached_request_data and the link read it."""

    def __init__(self, rid, prompt, computed):
        self.request_id, self.prompt_token_ids, self.output = rid, list(prompt), []
        self.num_computed_tokens, self.num_output_placeholders = computed, 0

    @property
    def all_token_ids(self):
        return self.prompt_token_ids + self.output

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output)

    @property
    def num_output_tokens(self):
        return len(self.output)


def real_scheduler():
    """The real overlay Scheduler (two batches, the Qwen decoupled layout), its dataclasses the real ones."""
    w = copy.copy(KP.world())               # a world of its own: its own scheduler module instance
    module = DFP.scheduler_module(w)
    module.CachedRequestData = OUTPUT.CachedRequestData
    return DFP.construct(DFP.qwen_config(w, 4), asynchronous=True, w=w)


class Engine:
    """One EngineCore with its UniProc runner and the probe driver. ``step`` is schedule + execute (and the
    async sample) of one step; with two batches the step is queued and updated by the next ``step`` or a
    ``drain``; synchronously it is updated at once."""

    def __init__(self, asynchronous=True, probe_module=native_probe):
        self.asynchronous, self.truth, self.batch_queue = asynchronous, {}, deque()
        self.sched = real_scheduler()
        self.runner = Runner(asynchronous, self.truth)
        self.owner = NS(async_scheduling=True) if asynchronous else NS()
        driver = probe_module.NativeQueryForward.__new__(probe_module.NativeQueryForward)
        driver.runner, driver.owner, driver.active = self.runner, self.owner, None
        self.runner._remote_ra_native_query_forward = self.driver = driver
        A.detach_engine()
        if asynchronous:
            # attach_engine's link (core: this engine's batch queue; set after construction so that this file
            # also runs on 4a498dc, whose link has no core).
            A._LINK = A.EngineLink(self.sched, None, check_stop=lambda request, limit: False)
            A._LINK.core = self
        self.outputs, self.leases = [], 0

    def close(self):
        A.detach_engine()

    def admit(self, rid, prompt, computed):
        """The scheduler's request after the admission: [0, computed) is computed (phase 1 and the carry)."""
        self.sched.requests[rid] = Request(rid, prompt, computed)
        self.truth[rid] = list(prompt)

    def step(self, new=None, cached=None):
        sched, new, cached = self.sched, dict(new or {}), dict(cached or {})
        counts = dict(new, **cached)
        news = [OUTPUT.NewRequestData(rid, list(sched.requests[rid].prompt_token_ids), [], sampling(), None,
                                      ([1, 2],), sched.requests[rid].num_computed_tokens, None) for rid in new]
        none = NS(get_block_ids=lambda allow_none=False: None)
        data = sched._make_cached_request_data([sched.requests[rid] for rid in cached], [], counts, {},
                                               {rid: none for rid in cached})
        # Scheduler.schedule after _make_cached_request_data (checked in ScheduleSourceTests).
        if not sched.use_v2_model_runner:
            sched.prev_step_scheduled_req_ids.clear()
            sched.prev_step_scheduled_req_ids.update(counts.keys())
        so = OUTPUT.SchedulerOutput(scheduled_new_reqs=news, scheduled_cached_reqs=data,
                                    num_scheduled_tokens=counts, total_num_scheduled_tokens=sum(counts.values()),
                                    scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                                    num_common_prefix_blocks=[0], finished_req_ids=set(), free_encoder_mm_hashes=[])
        for rid, count in counts.items():             # _update_after_schedule (+ AsyncScheduler's placeholders)
            request = sched.requests[rid]
            request.num_computed_tokens += count
            if self.asynchronous and request.num_computed_tokens >= request.num_tokens + request.num_output_placeholders:
                request.num_output_placeholders += 1
        self.outputs.append(so)
        self.runner.execute_model(so)
        sampled = self.runner.sample(so)
        if not self.asynchronous:
            return self.update(sampled)
        previous = self.batch_queue.pop() if self.batch_queue else None
        self.batch_queue.appendleft(sampled)
        if previous is not None:
            self.update(previous)

    def update(self, sampled):
        """update_from_output: the sampled ids appended (async: the placeholders resolved)."""
        for rid, token in sampled.items():
            request = self.sched.requests[rid]
            request.output.append(token)
            if self.asynchronous:
                request.num_output_placeholders -= 1

    def drain(self):
        """core._remote_ra_async_drain: a pass boundary updates the batch in flight."""
        while self.batch_queue:
            self.update(self.batch_queue.pop())

    def phase1(self, prompt, start):
        """An admission's phase 1 as its separate pass: the real driver's forward and its cleanup."""
        self.leases += 1
        lease = 'phase1-%032x' % self.leases
        self.driver.active = dict(request=lease, rows=None)
        rows = self.driver._execute(lease, list(prompt), start, ([3, 4],), ())
        self.driver._cleanup_request(lease)
        self.driver.active = None
        return rows


def prompt(n, seed):
    return [(seed * 31 + 7 * k) % 500 + 1 for k in range(n)]


def gpu_failure(engine):
    """GPU job 3519531's sequence (server-fea.log): A, admitted first (its phase 1 left its last 8 prompt tokens),
    is a new request of step N and samples; B's admission drains N and runs its phase 1 (to 38 of its 45
    tokens); step N+1 schedules B as new (7 tokens) and A as cached (1 token, num_output_tokens 1); then both
    decode, a step of each in flight, and the cell's turns end with a drain."""
    with patch.dict(sys.modules, vllm_modules()):
        engine.admit(RID_A, prompt(40, 1), 32)
        engine.step(new={RID_A: 8})                  # step N
        engine.drain()                               # B's registered add: a pass boundary
        engine.phase1(prompt(45, 2), 38)
        engine.admit(RID_B, prompt(45, 2), 38)
        engine.step(new={RID_B: 7}, cached={RID_A: 1})        # step N+1
        for _ in range(3):
            engine.step(cached={RID_A: 1, RID_B: 1})
        engine.drain()


# ---- tests ---------------------------------------------------------------------------------------------

class ProbePassTests(unittest.TestCase):
    def tearDown(self):
        A.detach_engine()

    def test_the_next_step_sends_the_token_ids_of_the_requests_a_probe_pass_took_out(self):
        engine = Engine()
        gpu_failure(engine)
        step = engine.outputs[1]                     # step N+1, as dumped at the GPU failure
        data = step.scheduled_cached_reqs
        self.assertEqual([n.req_id for n in step.scheduled_new_reqs], [RID_B])
        self.assertEqual((data.req_ids, data.resumed_req_ids, data.num_output_tokens), ([RID_A], set(), [1]))
        self.assertEqual(step.num_scheduled_tokens, {RID_B: 7, RID_A: 1})
        # The fix: A's token ids (the prompt and its one sampled id), the scheduler's own.
        self.assertEqual(data.all_token_ids, {RID_A: prompt(40, 1) + engine.truth[RID_A][40:41]})
        # Later steps: both requests stayed in the batch, no token ids sent.
        self.assertTrue(all(not so.scheduled_cached_reqs.all_token_ids for so in engine.outputs[2:]))
        # Every input id fed equals the sampled tokens (Runner.forward), A's decode after the probe included,
        # and A's row of the persistent batch holds its real ids, no placeholder.
        self.assertIn((RID_A, 40, (engine.truth[RID_A][40],)), engine.runner.fed)
        self.assertEqual(len(engine.runner.fed), 1 + 2 + 2 * 3)
        batch = engine.runner.input_batch
        row = batch.req_id_to_index[RID_A]
        self.assertEqual(batch.token_ids_cpu[row][:41], engine.truth[RID_A][:41])
        self.assertEqual(A.link().unbatched_requests, 1)
        self.assertEqual(engine.sched.requests[RID_A].all_token_ids, engine.truth[RID_A])
        self.assertEqual(engine.sched.requests[RID_B].all_token_ids, engine.truth[RID_B])

    def test_4a498dc_raises_the_gpu_keyerror(self):
        """4a498dc's native_probe (from Git): its probe pass reports nothing, and step N+1 raises the KeyError of
        GPU job 3519531 in vLLM's _update_states, at all_token_ids[req_id] of request A."""
        module = base_probe_module()
        if module is None:
            self.skipTest('4a498dc is not in this checkout (a Git archive)')
        self.assertFalse(hasattr(module.NativeQueryForward, '_runner_pass'))
        engine = Engine(probe_module=module)
        with self.assertRaises(KeyError) as caught:
            gpu_failure(engine)
        self.assertEqual(caught.exception.args, (RID_A,))
        self.assertEqual(len(engine.outputs), 2)
        self.assertEqual(engine.outputs[1].scheduled_cached_reqs.all_token_ids, {})

    def test_without_the_report_the_keyerror_comes_back(self):
        """Mutation (no Git needed): EngineLink.unbatched does nothing; step N+1 raises the GPU's KeyError."""
        engine = Engine()
        with patch.object(A.EngineLink, 'unbatched', lambda self, rids: None):
            with self.assertRaises(KeyError) as caught:
                gpu_failure(engine)
        self.assertEqual(caught.exception.args, (RID_A,))

    def test_a_stale_fallback_feeds_a_placeholder(self):
        """Why the token ids must come from the scheduler: the runner's own cached output ids of A are -1
        placeholders under async scheduling. Re-added from them (the fallback the fix does not take), A's row
        holds -1 where its sampled id belongs."""
        engine = Engine()
        with patch.dict(sys.modules, vllm_modules()):
            engine.admit(RID_A, prompt(40, 1), 32)
            engine.step(new={RID_A: 8})
            engine.drain()
        self.assertEqual(engine.runner.requests[RID_A].output_token_ids, [-1])
        self.assertEqual(engine.sched.requests[RID_A].output, engine.truth[RID_A][40:])

    def test_synchronous_scheduling_is_unchanged(self):
        """Synchronously the runner re-adds A from its own cached ids (the real tokens): no token ids are sent,
        the link is not involved and every input id is right, as on the synchronous GPU servers."""
        engine = Engine(asynchronous=False)
        gpu_failure(engine)
        self.assertIsNone(A.link())
        self.assertTrue(all(not so.scheduled_cached_reqs.all_token_ids for so in engine.outputs))
        self.assertEqual(len(engine.runner.fed), 1 + 2 + 2 * 3)
        self.assertEqual(engine.runner.requests[RID_A].output_token_ids, engine.truth[RID_A][40:])

    def test_refused_with_a_batch_in_flight(self):
        """The ids sent must hold every token: a probe pass while a step is in flight (no drain before it: A's
        sampled id is still a placeholder on the host) is refused, and the set is left as it was; after the
        drain the same report is taken. Without the EngineCore (no batch queue to read) it is refused."""
        engine = Engine()
        with patch.dict(sys.modules, vllm_modules()):
            engine.admit(RID_A, prompt(40, 1), 32)
            engine.step(new={RID_A: 8})                      # in flight, not drained
            self.assertEqual(engine.sched.requests[RID_A].num_output_placeholders, 1)
            with self.assertRaisesRegex(LiveV2Error, 'batch in flight'):
                engine.phase1(prompt(45, 2), 38)
        link = A.link()
        self.assertEqual(engine.sched.prev_step_scheduled_req_ids, {RID_A})
        engine.drain()
        link.unbatched({RID_A, 'phase1-gone'})
        self.assertEqual((engine.sched.prev_step_scheduled_req_ids, link.unbatched_requests), (set(), 1))
        link.unbatched(())                                   # nothing taken out: nothing to check
        link.core = None
        with self.assertRaisesRegex(LiveV2Error, 'batch in flight'):
            link.unbatched({RID_A})

    def test_the_cleanup_alone_takes_every_request_out(self):
        """The probe's cleanup is an empty scheduler output: vLLM's _update_states takes every request out of the
        batch, and the driver reports them all."""
        engine = Engine()
        with patch.dict(sys.modules, vllm_modules()):
            engine.admit(RID_A, prompt(40, 1), 32)
            engine.admit(RID_B, prompt(20, 2), 12)
            engine.step(new={RID_A: 8, RID_B: 8})
            engine.drain()
            self.assertEqual(sorted(engine.runner.input_batch.req_ids), sorted([RID_A, RID_B]))
            engine.driver._cleanup_request('phase1-none')
        self.assertEqual(engine.runner.input_batch.req_ids, [])
        self.assertEqual(engine.sched.prev_step_scheduled_req_ids, set())
        self.assertEqual(A.link().unbatched_requests, 2)


class ScheduleSourceTests(unittest.TestCase):
    """The stand-ins above follow the overlay's own code where they do not execute it."""

    def test_schedule_records_the_previous_step_after_the_cached_request_data(self):
        text = _method_source(SCHEDULER, 'Scheduler', 'schedule')
        made = text.index('cached_reqs_data = self._make_cached_request_data(')
        cleared = text.index('self.prev_step_scheduled_req_ids.clear()')
        updated = text.index('self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())')
        self.assertLess(made, cleared)
        self.assertLess(cleared, updated)
        self.assertIn('if req_id not in self.prev_step_scheduled_req_ids:\n'
                      '                    all_token_ids[req_id] = req.all_token_ids.copy()',
                      _method_source(SCHEDULER, 'Scheduler', '_make_cached_request_data'))

    def test_the_async_runner_caches_placeholders_and_needs_the_token_ids(self):
        keep = _method_source(RUNNER, 'GPUModelRunner', '_bookkeeping_sync')
        self.assertIn('sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None', keep)
        self.assertIn('req_state.output_token_ids.extend(sampled_ids)', keep)
        update = _method_source(RUNNER, 'GPUModelRunner', '_update_states')
        self.assertIn('unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)', update)
        self.assertIn('if self.use_async_scheduling and num_output_tokens > 0:\n'
                      '                    # We must recover the output token ids for resumed requests in the\n'
                      '                    # async scheduling case, so that correct input_ids are obtained.\n'
                      '                    resumed_token_ids = req_data.all_token_ids[req_id]', update)

    def test_the_probe_passes_report_through_the_link(self):
        for name in ('_execute', '_cleanup_request'):
            text = _method_source('remote_ra_live_v2/native_probe.py', 'NativeQueryForward', name)
            self.assertIn('self._runner_pass(', text, name)
            self.assertNotIn('self.runner.execute_model(', text, name)
            self.assertNotIn('self.runner._update_states(', text, name)


if __name__ == '__main__':
    unittest.main()
