"""Gemma 4, sixteen sessions (gemma-s1 step 5): --release-idle-pages on the window layout. CPU only.

The window layout keeps every session's sliding window pinned in pool 1 (W-GPU). Under
GPU pressure the capacity gate releases idle sessions' global Pages (pool 0) exactly
as for Qwen (remote_ra_live_v2.idle_release) and the session's next turn readmits
them before its phase 1, which borrows the window as before. With a fixed active
budget the launcher then sizes pool 0 by the selective rule
(window_capacity.selective_pages: one selective turn and the other sessions' tails)
instead of one session at the context limit; a first turn or a rebuild that can never
fit is refused (CAPACITY_EXCEEDED). Nothing here is CUDA evidence. Covered:

- the window worker's attachment: the release record checked once, evict/readmit
  accepted, the evidence echo; nothing without the record;
- end to end (test_gemma_e2e's engine and worker: vLLM's KV manager with the sliding
  managers, WindowPins, WindowPhase1Coordinator, WindowAllocationOwner, a runner that
  checks every full-attention and sliding read): sixteen sessions decoding together,
  admitted between batched steps, with pool 0 at the selective rule's minimum for the
  test geometry. Every turn completes without preemption and reads exactly what it
  reads with a pool that never releases (tokens, selections, attended Pages, replies
  per session); victims are idle sessions only, least recently used first; a
  readmission restores exactly the released Pages with their KV; evictions and
  readmissions never touch pool 1 or a window (engine pins and worker carry);
  capacity.jsonl, the native records and engine-cleanup.json agree; both pools are
  back at their baselines at shutdown. Without the switch the same pool cannot serve
  the sixteen sessions;
- CAPACITY_EXCEEDED: a rebuild longer than pool 0 can ever free is refused before
  anything changes, and the session's next ordinary turn still runs;
- a rebuild with every window slot taken: its sliding need credits the window its render
  frees, so idle sessions are released for its global shortfall (the fake KV manager takes
  the server's max_num_batched_tokens there, vLLM's sliding admission cap).

The default suite runs six sessions (four turns of the pressure runs, five of the rebuild);
the sixteen sessions, compared with a pool that never releases, run with
REMOTE_RA_SLOW_TESTS=1 (gemma-goal: the archive suite stays short).
"""
from contextlib import ExitStack
import copy
import importlib
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

import test_gemma_e2e as GE
import test_stageb_e2e as S
import test_stageb_worker as W

from remote_ra_live_v2 import idle_release as R
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2 import window_capacity as WC
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.sliding_window import kept_blocks

try:
    import torch
except ImportError:
    torch = None

P, WINDOW, N, FA, SLIDING = GE.P, GE.WINDOW, GE.N, GE.FA, GE.SLIDING
SESSIONS = 16
SHORT = 6                       # the default suite's sessions
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
BUDGET = 10 * P                 # --active-token-budget of the test (the switch threshold too)
REPLY_BOUND = 1536              # the capacity gate's reply bound (no max_tokens in the fake params)
REFERENCE_BLOCKS = 3000         # never releases anything


def tasks(count):
    return [dict(instance_id='repo__g%02d-%d' % (k, 400 + k),
                 problem_statement='Resolve the defect %02d of module g%02d.' % (k, k)) for k in range(count)]


def contract(policy):
    record = dict(page_tokens=P, state='sliding', window_tokens=WINDOW, window_slots=1, forward_tokens=N)
    fill = selection_contract.random_fill_pin() if policy == 'random_fill' else None
    value = selection_contract.default_contract(P, active_ratio=None, active_token_budget=BUDGET,
                                                vector_format=selection_contract.vector_format_for(record),
                                                random_fill=fill)
    value.update(switch_threshold_tokens=BUDGET,
                 recent_quota_pages=selection_contract.default_recent_quota_pages(784))
    return selection_contract.validate(value)


def config(policy, output, task_list, layout, release):
    value = dict(GE.config(policy, output, task_list, layout), selection_contract=contract(policy))
    if release:
        value.update(release_idle_pages=True)
    return value


def server_kv_manager(w, config, caching=True, watermark=0.0, hash_block_size=16, scheduler_block_size=784):
    """test_kv_pools.kv_manager with the server's max_num_batched_tokens, one native forward of the
    window layout (N): serve.py pins --max-num-batched-tokens to the layout's forward, so vLLM's
    sliding managers cap a waiting request's full-sequence check at the window slot plus one
    forward's transient (SlidingWindowSpec.max_admission_blocks_per_request), the cold peak the
    capacity gate counts. test_kv_pools' 16,384 leaves that check uncapped here."""
    return w.manager.KVCacheManager(
        kv_cache_config=config, max_model_len=65536, scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size, max_num_batched_tokens=N, enable_caching=caching, watermark=watermark)


def selective_blocks(layout):
    """The smallest pool 0 the selective rule accepts for the test geometry."""
    blocks = 1
    while blocks - 1 < WC.selective_pages(layout, BUDGET, REPLY_BOUND, blocks):
        blocks += 1
    return blocks


def script(index, turns):
    """One lane's turns: (observation, reply target, EOS). Lane 0 reads long outputs
    (its history passes 4K tokens), the others short and medium ones."""
    out = []
    for k in range(turns):
        if k == 0:
            observation = None
        elif index == 0:
            observation = S.OUT % (0, 'z' * 900 + ' step %d' % k)
        else:
            observation = GE.OBSERVATIONS[(k + index) % len(GE.OBSERVATIONS)] + ' (lane %d)' % index
        out.append((observation, GE.TARGETS[(k + index) % len(GE.TARGETS)], (k + index) % 4 == 3))
    return out


class Pressure(GE.Harness):
    """test_gemma_e2e.Harness serving every lane's script together: one registered add
    per pass (round robin over the lanes with work and no request in flight; the gate
    may answer CAPACITY_WAIT), then one batched step of every running request (prefill
    in Page chunks, then decode). A running request that finds no block would be
    preempted: that fails the test."""

    def __init__(self, policy, output, task_list, fa_blocks, release, turns):
        super().__init__(policy, output, task_list, fa_blocks=fa_blocks)
        if release:
            self.owner.release_idle = True
            self.core._remote_ra_idle_release = R.IdleRelease(self.core, output)
        for lane in self.lanes:
            lane.script = script(lane.index, turns)
        self.running, self.done, self.waits, self.widths = [], [], 0, []

    def admit(self, lane):
        from remote_ra_live_v2.capacity import CapacityWait
        observation, target, eos = lane.script[0]
        messages = lane.messages + [dict(role='user', content=observation)] * bool(observation)
        before = len(self.runner.computed)
        try:
            info = self.submit(lane, messages)
        except CapacityWait:
            self.waits += 1
            return None
        # Phase 1 ran inside the add (no other forward did): its probe's tokens.
        info['probe'] = [x for r, x in self.runner.computed[before:] if r.startswith('phase1-')]
        info['outputs'] = self.reply(info, target, eos)
        self.runner.samples[info['rid']] = iter(info['outputs'])
        info.update(first=True, produced=0)
        lane.script.pop(0)
        self.running.append(info)
        return info

    def batch_step(self):
        from vllm.v1.core.sched.output import NewRequestData
        counts, news = {}, []
        for info in self.running:
            request, rid = info['request'], info['rid']
            n, start = len(request.prompt_token_ids), request.num_computed_tokens
            waiting = request.status.name == 'WAITING'
            count = min(n, (start // P + 1) * P) - start if start < n else 1
            result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
                new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=waiting,
                reserved_blocks=0, has_scheduled_reqs=waiting and bool(counts))
            assert result is not None, 'running request %s found no free block (preemption)' % rid
            rows = tuple([b.block_id for b in m.req_to_blocks[rid]] for m in self.kv.coordinator.single_type_managers)
            if info['first']:
                news.append(NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None, rows,
                                           start, None))
                info['first'], info['produced'] = False, 0
            else:
                self.runner.set_rows(rid, rows)
                info['produced'] = len(self.runner.requests[rid].output_token_ids)
            counts[rid] = count
        self.runner.execute_model(NS(num_scheduled_tokens=dict(counts), total_num_scheduled_tokens=sum(counts.values()),
            scheduled_new_reqs=news, finished_req_ids=set(), new_block_ids_to_zero=self.kv.take_new_block_ids(),
            kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
        self.widths.append(len(counts))
        for info in list(self.running):
            request, rid = info['request'], info['rid']
            request.num_computed_tokens += counts[rid]
            request.status.name = 'RUNNING'
            request.output_token_ids.extend(self.runner.requests[rid].output_token_ids[info['produced']:])
            if len(request.output_token_ids) == len(info['outputs']):
                self.running.remove(info)
                self.complete(info)
                info['lane'].messages = info['messages'] + [dict(role='assistant',
                                                                 content=' ' + info['committed']['text'] + '\n')]
                self.done.append(info)

    def serve(self, check):
        k = 0
        while any(lane.script for lane in self.lanes) or self.running:
            busy = {info['lane'].index for info in self.running}
            ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
            admitted = None
            if ready:
                admitted = self.admit(ready[k % len(ready)])
                k += 1
            if self.running:
                before = len(self.done)
                self.batch_step()
                for info in self.done[before:]:
                    check(self, info)
            else:
                assert admitted is not None, 'no request runs and the gate refused every lane'


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class SixteenSessionTests(unittest.TestCase):
    maxDiff = None
    TURNS = 5

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.layout = I0.for_window_server(P, SESSIONS, WINDOW, N)
        self.small = selective_blocks(self.layout)

    # ---- one run -------------------------------------------------------------------------------
    def run_case(self, policy, blocks, release, observe=None, after=None, sessions=SESSIONS, turns=None):
        from remote_ra_http.evidence import engine_closed
        turns = turns or self.TURNS
        layout = self.layout if sessions == SESSIONS else GE.window_layout(policy, sessions)
        with ExitStack() as stack:
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = config(policy, output, tasks(sessions), layout, release)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = Pressure(policy, output, tasks(sessions), blocks, release, turns)
            if observe is not None:
                observe(h)
            h.serve(self.check)
            self.assertEqual(len(h.done), sessions * turns)
            self.boundary(h)
            result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                    h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane]
                      for lane in h.lanes}
            if after is not None:
                after(h)
            events = ([json.loads(line) for line in (output / R.EVENTS).read_bytes().splitlines()]
                      if (output / R.EVENTS).exists() else [])
            natives = [i['native'] for i in h.done]
            h.final_pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
            h.lifecycle.disconnect()
            h.sessions.close()
            engine_closed(h.core)
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual(h.adapter.window.pinned_blocks(), 0)
            self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                             (h.baseline, h.window_baseline))
            return result, h, events, natives, cleanup

    # ---- invariants ----------------------------------------------------------------------------
    def window_of(self, h, sid):
        s = h.adapter.sessions[sid]
        window = h.adapter.window.window(s.namespace)
        return None if window is None else {g: tuple(b.block_id for _, b in v) for g, v in window.items()}

    def check(self, h, info):
        """A finished turn: exactly-once computation, its committed window (engine pin and
        worker carry), the Pages it attended."""
        lane, geo, rid = info['lane'], info['geo'], info['rid']
        s = h.adapter.sessions[lane.sid]
        full = tuple(info['binding'].tokens) + tuple(info['request'].output_token_ids)
        C2 = len(full) - 1
        self.assertEqual(s.computed_end, C2)
        self.assertEqual([x for r, x in h.runner.computed if r == rid], list(range(geo.phase2_start, C2)))
        self.assertEqual(info['probe'], list(range(geo.C, geo.e)) if geo.local_turn else [])
        first, last = kept_blocks(C2, WINDOW, P)
        window = h.adapter.window.window(s.namespace)
        self.assertEqual({g: [i for i, _ in v] for g, v in window.items()}, {g: list(range(first, last)) for g in SLIDING})
        self.assertEqual(h.backend.carry[lane.sid].blocks(), self.window_of(h, lane.sid))
        self.assertEqual(h.runner.visible[rid], info['plan']['selected'])
        self.assertTrue(all(b.ref_cnt == 1 for v in window.values() for _, b in v))

    def boundary(self, h):
        """Nothing runs: pool 0 holds exactly the pins, pool 1 exactly the windows."""
        pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - pins)
        windows = [b for lane in h.lanes for row in (self.window_of(h, lane.sid) or {}).values() for b in row]
        self.assertEqual(len(windows), len(set(windows)))
        self.assertEqual(h.window_pool.get_num_free_blocks(), h.window_baseline - len(windows))

    # ---- the sixteen sessions ------------------------------------------------------------------
    def pressure(self, policy, sessions=SESSIONS, turns=None):
        """``sessions`` < SESSIONS (the default suite): no reference run, no run without the
        switch; the releases and their invariants as below."""
        full = sessions == SESSIONS
        small = self.small if full else selective_blocks(GE.window_layout(policy, sessions))
        if full:
            reference, big, events, natives, cleanup = self.run_case(policy, REFERENCE_BLOCKS, True)
            self.assertEqual(events, [])
            self.assertEqual(cleanup['idle_release'], dict(evictions=0, evicted_pages=0, readmissions=0,
                                                           readmitted_pages=0))
            self.assertGreater(max(big.widths), 8)           # the sessions decode together
            # The selective rule's pool cannot hold the sixteen sessions' idle pins: it must release.
            self.assertGreater(big.final_pins, self.small)
        plans, windows = [], []

        def observe(h):
            release = h.core._remote_ra_idle_release
            victims, evict, readmit = release.victims, release.evict, release.readmit

            def watched(lifecycle, command, short):
                chosen = victims(lifecycle, command, short)
                idle = [sid for sid, _ in release.idle(lifecycle, command.session_id)]
                plans.append(dict(chosen=chosen, idle=idle, used=[release.used.get(sid, 0) for sid in idle],
                                  admitting=command.session_id, short=short,
                                  running={info['lane'].sid for info in h.running}))
                return chosen

            def evicting(session_id, request_id, short):
                state = (self.window_of(h, session_id), h.backend.carry[session_id].digest,
                         h.window_pool.get_num_free_blocks())
                pages = evict(session_id, request_id, short)
                windows.append(('evict', session_id, state, (self.window_of(h, session_id),
                                h.backend.carry[session_id].digest, h.window_pool.get_num_free_blocks())))
                return pages

            def readmitting(binding, session):
                state = (self.window_of(h, binding.session), h.backend.carry[binding.session].digest,
                         h.window_pool.get_num_free_blocks())
                pages = readmit(binding, session)
                windows.append(('readmit', binding.session, state, (self.window_of(h, binding.session),
                                h.backend.carry[binding.session].digest, h.window_pool.get_num_free_blocks())))
                # The readmitted blocks hold exactly the session's KV of those Pages.
                lane = next(l for l in h.lanes if l.sid == binding.session)
                stream = list(lane.frontend.stream)
                for page in pages:
                    block = h.adapter.owner.sessions[session.namespace][page][0][0].block_id
                    self.assertEqual(h.store.kv[(FA, block)], [(x, stream[x]) for x in range(page * P, (page + 1) * P)])
                return pages
            release.victims, release.evict, release.readmit = watched, evicting, readmitting
        result, h, events, natives, cleanup = self.run_case(policy, small, True, observe, sessions=sessions,
                                                            turns=turns)
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        if full:
            # Every session saw, selected, attended and replied exactly as with the large pool.
            self.assertEqual(result, reference)
            self.assertGreater(h.waits, 0)
            self.assertTrue(any(p['chosen'] and p['running'] for p in plans))
        # Victims: idle sessions only, least recently used first, only as many as needed.
        for plan in plans:
            self.assertEqual(plan['used'], sorted(plan['used']))
            if plan['chosen']:
                self.assertEqual(plan['chosen'], plan['idle'][:len(plan['chosen'])])
                self.assertFalse(set(plan['chosen']) & (plan['running'] | {plan['admitting']}))
        self.assertEqual(sum(len(p['chosen'] or ()) for p in plans), len(evictions))
        # Global Pages only: pool 1, the session's window pin and its worker carry stay.
        self.assertEqual(len(windows), len(evictions) + len(readmissions))
        for _, _, before, after in windows:
            self.assertIsNotNone(before[0])
            self.assertEqual(before, after)
        for e in events:
            self.assertEqual(e['free_blocks_by_pool'][1], e['free_blocks_by_pool_before'][1])
        for e in evictions:
            self.assertGreater(e['free_blocks_by_pool'][0], e['free_blocks_by_pool_before'][0])
        # Each readmission restores exactly what that session's last eviction released.
        last = {}
        for e in events:
            if e['event'] == 'evict':
                self.assertNotIn(e['session_id'], last)
                last[e['session_id']] = e['pages']
            else:
                self.assertEqual(e['pages'], last.pop(e['session_id']))
        self.assertEqual(cleanup['idle_release'], dict(evictions=len(evictions),
            evicted_pages=sum(e['pages'] for e in evictions), readmissions=len(readmissions),
            readmitted_pages=sum(e['pages'] for e in readmissions)))
        rows = [n['idle_release'] for n in natives]
        self.assertEqual(sum(len(r['evicted_sessions']) for r in rows), len(evictions))
        self.assertEqual(sum(len(r['readmitted_pages']) for r in rows), sum(e['pages'] for e in readmissions))
        if full:
            # Without the switch the same pool cannot serve the sixteen sessions: once the idle sessions'
            # pins leave a turn no room, nothing runs and the gate refuses every lane (integrate-1018:
            # the window layout checks a lone turn under other sessions' pins). Before, such a turn was
            # admitted unchecked and found no blocks ("insufficient native blocks", "found no free block").
            with self.assertRaisesRegex(AssertionError, 'no request runs and the gate refused every lane'):
                self.run_case(policy, self.small, False)
        return h

    def test_sessions_observation_query(self):
        self.pressure('observation_query', SHORT, 4)

    def test_sessions_random_fill(self):
        self.pressure('random_fill', SHORT, 4)

    @unittest.skipUnless(SLOW, 'about 40 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions_observation_query(self):
        self.pressure('observation_query')

    @unittest.skipUnless(SLOW, 'about 40 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions_random_fill(self):
        self.pressure('random_fill')

    def test_a_rebuild_that_can_never_fit_is_refused(self):
        def after(h):
            lane = h.lanes[0]
            s = h.adapter.sessions[lane.sid]
            self.assertGreater(len(lane.frontend.stream), 4000)
            state = (s.namespace, s.resident, s.tail, s.evicted, s.computed_end, self.window_of(h, lane.sid),
                     h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks(), lane.turn)
            retry = lane.messages[:-1] + [dict(role='user', content='Format error: expected exactly one action.')]
            # The frontend's render of a rebuild resets its transcript; a client whose request
            # failed keeps the one it had (the agent instance usually ends here instead).
            frontend = copy.deepcopy(lane.frontend)
            with self.assertRaisesRegex(R.CapacityExceeded, 'CAPACITY_EXCEEDED: the turn needs') as caught:
                h.submit(lane, retry)
            self.assertNotIsInstance(caught.exception, AssertionError)
            lane.frontend = frontend
            self.assertEqual(state, (s.namespace, s.resident, s.tail, s.evicted, s.computed_end,
                                     self.window_of(h, lane.sid), h.pool.get_num_free_blocks(),
                                     h.window_pool.get_num_free_blocks(), lane.turn))
            # The session's next ordinary turn still runs (it readmits if it was released).
            lane.script = [(S.OUT % (0, 'retry ok'), 9, False)]
            h.serve(self.check)
            self.assertEqual(h.done[-1]['lane'], lane)
            self.assertEqual(h.done[-1]['geo'].local_turn, self.TURNS)
            self.boundary(h)
        small = selective_blocks(GE.window_layout('observation_query', SHORT))
        self.run_case('observation_query', small, True, after=after, sessions=SHORT)


    def test_a_rebuild_with_every_window_slot_taken_releases_idle_pages(self):
        """Every window slot of pool 1 taken, nothing running, pool 0 at the selective rule's minimum:
        a rebuild whose prompt alone needs more global Pages than pool 0 has free, and whose cold
        sliding peak is a whole slot plus the transient, needs idle sessions' Pages released. Its
        sliding need credits the window its render frees (window_capacity._gate_turn); without the
        credit it looked short of pool 1 by its own window, so nothing was released, and admitted
        once nothing ran it found no global Pages: vLLM leaves such a request waiting forever (the
        fake scheduler fails)."""
        from remote_ra_live_v2 import capacity as CAP
        events = []

        def after(h):
            layout, groups = h.layout, len(SLIDING)
            self.assertEqual(sum(1 for lane in h.lanes if self.window_of(h, lane.sid)), SHORT)   # every slot
            # A session whose Pages were released (it pins its window only), so the rebuild frees no Pages.
            lane = next(l for l in h.lanes if not h.adapter.sessions[l.sid].resident)
            free_fa, free_sliding = h.kv.get_num_free_blocks_by_pool()

            def retry(pad):
                return lane.messages[:-1] + [dict(role='user', content='Format error: one action. ' + 'x' * pad)]
            base = len(copy.deepcopy(lane.frontend).render(retry(0))[0])
            messages = retry(max(0, (free_fa + 12) * P - base))       # one token per character here
            tokens = copy.deepcopy(lane.frontend).render(messages)[0]
            (_, _), (fa, sliding) = CAP.turn_need_by_pool(h.core, lane.sid, tokens, messages, REPLY_BOUND)
            own = sum(len(v) for v in self.window_of(h, lane.sid).values())
            cold = groups * (layout.window_blocks + layout.transient_blocks)
            # Its prompt alone is short of global Pages; pool 1 is short only by its own window.
            self.assertGreater(-(-len(tokens) // P), free_fa)
            self.assertTrue(free_sliding < cold <= free_sliding + own, (free_sliding, cold, own))
            before = len((h.output / R.EVENTS).read_bytes().splitlines()) if (h.output / R.EVENTS).exists() else 0
            lane.messages = messages[:-1]
            lane.script = [(messages[-1]['content'], 9, False)]
            h.serve(self.check)
            done = h.done[-1]
            self.assertEqual((done['lane'], done['geo'].local_turn, done['binding'].rebuild), (lane, 0, True))
            self.assertEqual(sliding, cold - own)       # the gate's pool-1 figure credits the freed window
            rows = [json.loads(line) for line in (h.output / R.EVENTS).read_bytes().splitlines()][before:]
            events.extend(rows)
            self.assertTrue(rows and {e['event'] for e in rows} == {'evict'}
                            and {e['request_id'] for e in rows} == {done['rid']}, rows)
            self.boundary(h)
        small = selective_blocks(GE.window_layout('observation_query', SHORT))
        with patch.object(GE.KP, 'kv_manager', server_kv_manager):
            self.run_case('observation_query', small, True, after=after, sessions=SHORT, turns=4)
        self.assertTrue(events)

class SelectiveRuleTests(unittest.TestCase):
    def test_test_geometry(self):
        layout = I0.for_window_server(P, SESSIONS, WINDOW, N)
        # One selective turn: 2 x (10 selected + 161 run) + 1 tail; 15 other tails; the 1 % watermark.
        self.assertEqual(WC.selective_turn_pages(layout, BUDGET, REPLY_BOUND), 343)
        blocks = selective_blocks(layout)
        self.assertEqual(blocks - 1, 343 + 15 + int(.01 * blocks))
        with self.assertRaises(LiveV2Error):
            WC.selective_turn_pages(layout, P - 1, REPLY_BOUND)


class WorkerAttachTests(unittest.TestCase):
    """attach_window_worker with the loaded runner's seams faked: the release record."""

    def attach(self, startup):
        # The module sys.modules holds now (an earlier test's patch.dict may have dropped the
        # package attribute's copy), the one the patches below resolve too.
        WW = importlib.import_module('remote_ra_live_v2.window_worker')
        layout = I0.for_window_server(16, 2, 1024, 8192)
        io = NS(views=[], evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', drain=lambda: None,
                attach_arena=lambda arena: setattr(io, 'arena', arena))
        tensors = ModuleType('remote_ra_growing_worker.tensors')
        tensors.TensorIO = lambda views, decoupled: io
        tensors.views_from_runner = lambda runner, *inputs, skip_sliding: []
        tensors.PageArena = lambda views, max_pages: NS(stats=lambda: dict(max_pages=max_pages))
        modules = W.vllm_stubs()
        modules['vllm.distributed.kv_transfer'].get_kv_transfer_group = lambda: NS()
        modules['remote_ra_growing_worker.tensors'] = tensors

        class Owner:
            release_idle = False

            def __init__(self, runner, io, startup, layout):
                self.startup = startup

            def bind_backend(self, backend):
                self.backend = backend

            def bind_bridge(self, bridge):
                self.bridge = bridge

        class Backend:
            def __init__(self, io, owner, source):
                self.io, self.owner = io, owner

        runner = NS(kv_cache_config=NS(group_pool=(0, 1)), max_model_len=131072, execute_model=lambda so: None)
        worker = NS(model_runner=runner)
        startup = dict(dict(source='s' * 64, page_tokens=16, loaded_sources_verified=True), **startup)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(WW, 'check_view_geometry', lambda *a: None))
            for target, value in (('remote_ra_live_v2.cuda_graphs.attach', lambda *a: None),
                                  ('remote_ra_live_v2.context_limits.check_worker', lambda *a: None),
                                  ('remote_ra_live_v2.language_model_only.worker_check', lambda *a: None),
                                  ('remote_ra_live_v2.loaded_sources.verify_closure', lambda runner: {}),
                                  ('remote_ra_live_v2.connector_guard.check_worker_pool', lambda c: None),
                                  ('remote_ra_live_v2.vector_space.attach', lambda runner: dict(hidden_size=3)),
                                  ('remote_ra_live_v2.window_owner.WindowAllocationOwner', Owner),
                                  ('remote_ra_live_v2.window_lifecycle.WindowWorkerBackend', Backend),
                                  ('remote_ra_live_v2.window_lifecycle.WindowLifecycle', lambda b, page_tokens: NS()),
                                  ('remote_ra_live_v2.page_compaction.PageCompactor',
                                   lambda hidden, page: NS(evidence=lambda: {})),
                                  ('remote_ra_live_v2.native_probe.NativeQueryForward', lambda *a: NS()),
                                  ('remote_ra_live_v2.worker_composition.WorkerBridge', lambda q, o: NS()),
                                  ('remote_ra_live_v2.generation_guard.guarded_execute', lambda r, f: f)):
                stack.enter_context(patch(target, value))
            evidence = WW.attach_window_worker(worker, startup, layout, ((), ()))
        return evidence, runner._remote_ra_live_v2_allocation_owner

    def test_release_record(self):
        evidence, owner = self.attach({R.CONFIG_KEY: True})
        self.assertIs(owner.release_idle, True)
        self.assertIs(evidence[R.CONFIG_KEY], True)
        evidence, owner = self.attach({})
        self.assertIs(owner.release_idle, False)
        self.assertNotIn(R.CONFIG_KEY, evidence)
        with self.assertRaisesRegex(LiveV2Error, 'must be true or absent'):
            self.attach({R.CONFIG_KEY: False})
        # The HTTP runtime compares the record with the pinned startup config once.
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                patch('remote_ra_http.config.load', lambda: dict(kv_layout={})), \
                self.assertRaisesRegex(LiveV2Error, 'differs from the pinned startup config'):
            self.attach({R.CONFIG_KEY: True})


if __name__ == '__main__':
    unittest.main()
