"""Multi-session stage 2, worker side: several owned requests in one native step.

CPU only; nothing here is CUDA evidence or READY.
- The real worker composition (RunnerAllocationOwner, NativeQueryForward,
  CudaWorkerBackend, QwenAlignLifecycle, WorkerBridge, guarded_execute and
  worker_dispatch) runs behind test_stageb_worker's FakeRunner, which models
  vLLM's persistent batch: several rows per step flattened in row order,
  unscheduled requests dropped and re-added from requests[rid].block_ids,
  condense moving the last row into each hole.
- BatchHarness splits a turn into admission (prepare, phase 1 and its drain,
  reserve: one at a time, as on the engine thread), batched steps (one
  allocation_update per scheduled request, then one execute_model) and
  completion.  Its Mamba manager frees a moved slot in the same step, so a
  freed block can go to another request before the old claim is refreshed.
- Every forward still checks, per request, what it attends (whole selected
  Pages and the run), the KV it reads and the dense GDN chain at its start.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source-overlay'))
sys.path.insert(0, str(HERE))
import test_stageb_worker as W
from test_stageb_worker import EPOCH, FA, MAMBA, MODEL, P, SOURCE, words
from remote_ra_live_v2.contract import NativeIdentity, RenderAssignment, closed_pages, digest

try:
    import torch
except ImportError:
    torch = None

BYTES = dict(attention=1000, hybrid=7)      # fake H2D bytes per FA block / Mamba slot
HYBRID = BYTES['hybrid'] * len(MAMBA)
QUIET = NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())   # empty connector jobs every step


class CountingIO(W.FakeIO):
    """FakeIO with page sizes and the process H2D counters TensorIO keeps."""

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


class IncarnationPool(W.Pool):
    """W.Pool whose every allocation of an ID is a new incarnation (pool_incarnations)."""

    def __init__(self):
        super().__init__(); self.incarnation = {}

    def alloc(self, k=1):
        ids = super().alloc(k)
        for b in ids:
            self.incarnation[b] = self.incarnation.get(b, 0)+1
        return ids

    def hand(self, b):
        """A block freed earlier in this step, allocated again: a new incarnation."""
        assert b not in self.used and b not in self.free
        self.used.add(b); self.incarnation[b] += 1
        return b


def script(seed):
    """Every tail case of test_stageb_worker, token values shifted per session."""
    s = seed * 1000
    return [dict(observation=words(5, s+1), header=words(2, s+2), outputs=words(3, s+3)),
            dict(boundary=words(1, s+4), observation=words(3, s+5), header=words(2, s+6),
                 outputs=words(2, s+7), selected=[0]),
            dict(boundary=words(2, s+9), observation=words(3, s+10), header=words(1, s+11),
                 outputs=words(4, s+12), selected=[1, 3, 4], goal=words(6, s+8)),
            dict(boundary=words(1, s+13), observation=words(2, s+14), header=words(2, s+15),
                 outputs=words(4, s+16), selected=[0, 5]),
            dict(boundary=words(1, s+17), observation=words(1, s+18), header=words(1, s+19),
                 outputs=words(2, s+20))]


def record_hidden(qwen):
    """{request: [SHA-256 of the hidden rows of each forward]}, hashed here in the test:
    the worker's forward record no longer carries a digest of its rows."""
    hidden = {}
    after = qwen.after_forward

    def recorded(identity, hidden_states, **kwargs):
        rows = hidden_states.detach().to(device='cpu', dtype=torch.float32).contiguous()
        hidden.setdefault(identity.request, []).append(hashlib.sha256(rows.numpy().tobytes()).hexdigest())
        return after(identity, hidden_states, **kwargs)
    qwen.after_forward = recorded
    return hidden


class BatchHarness(W.Harness):
    """The stage-B fake engine with admission, batched steps and completion split."""

    def __init__(self, mrope=False):
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        from remote_ra_live_v2.native_probe import NativeQueryForward
        from remote_ra_live_v2.worker_composition import WorkerBridge
        from remote_ra_live_v2.generation_guard import guarded_execute
        from remote_ra_live_v2.qwen import QwenAlignLifecycle
        from remote_ra_live_v2.worker_backend import CudaWorkerBackend
        self.store = W.Store(); self.pool = IncarnationPool(); self.runner = W.FakeRunner(self.store)
        if mrope:
            self.runner.uses_mrope = True
            self.runner.model_config = NS(hf_text_config=NS(rope_parameters=dict(
                mrope_interleaved=True, mrope_section=[11, 11, 10])))
        self.io = CountingIO(self.store)
        self.owner = RunnerAllocationOwner(self.runner, self.io, dict(enabled=True, source=SOURCE,
            page_tokens=P, loaded_sources_verified=True, source_closure_verified=True))
        self.runner._remote_ra_live_v2_allocation_owner = self.owner
        self.backend = CudaWorkerBackend(self.io, self.owner, source=SOURCE)
        self.owner.bind_backend(self.backend)
        self.runner._remote_ra_native_query_forward = NativeQueryForward(self.runner, self.owner, self.backend)
        self.qwen = QwenAlignLifecycle(self.backend, page_tokens=P)
        self.hidden = record_hidden(self.qwen)
        self.bridge = WorkerBridge(self.qwen, self.owner); self.owner.bind_bridge(self.bridge)
        self.runner._remote_ra_live_v2_bridge = self.bridge
        self.runner.execute_model = guarded_execute(self.runner, self.runner.execute_model)
        self.worker = NS(_remote_ra_live_v2_backend=self.backend, _remote_ra_live_v2_qwen=self.qwen)
        self.sessions = {}; self.turn = 0; self.rid_ns = {}; self.frames = []
        self.running = {}; self.finished = set(); self.done = []
        self.receipts, self.process = {}, {}

    def rpc(self, op, payload):
        from remote_ra_live_v2.worker_composition import worker_dispatch
        key = payload.get('request', 'session')
        before = self.io._http_h2d_joined
        receipt = worker_dispatch(self.worker, op, payload)
        self.frames.append(op)
        self.receipts.setdefault(key, []).append(receipt)
        self.process.setdefault(key, []).append((before, self.io._http_h2d_joined))
        return receipt

    # ---- admission: one at a time, while other requests are mid-decode ------
    def admit(self, sid, *, observation, header, outputs, boundary=(), selected=None, goal=None):
        s = self.sessions[sid]; local = s['local']; turn = self.turn; self.turn += 1
        assert not any(i['sid'] == sid for i in self.running.values()), 'one request per session'
        rid = f'r{turn}'; binding = hashlib.sha256(rid.encode()).hexdigest()
        identity = NativeIdentity(rid, sid, turn, f'g{turn}', hashlib.sha256(EPOCH.encode()).hexdigest(),
                                  SOURCE, MODEL)
        if local == 0:
            T = tuple(observation)+tuple(header); o = e = len(observation); C = Q = L = S2 = 0
        else:
            C = s['C']; Q = C//P*P; T = s['full']+tuple(boundary)+tuple(observation)+tuple(header)
            o = len(s['full'])+len(boundary); e = o+len(observation); L = o//P*P; S2 = e
        N = len(T)
        a = RenderAssignment(identity, local_turn=local, computed_start=C, sealed_start=Q, candidate_end=L,
            observation_start=o, observation_end=e, phase2_start=S2, prompt_tokens=N, page_tokens=P,
            token_digest=hashlib.sha256(str(T).encode()).hexdigest(), carry_digest=s['carry'])
        base = dict(request=rid, session=sid, turn=turn, local_turn=local, generation=identity.generation,
                    binding=binding)
        prep = dict(base, namespace=s['ns'], ns_generation=s['ns_generation'], identity=asdict(identity),
                    assignment=asdict(a), tokens=list(T))
        info = dict(rid=rid, sid=sid, a=a, T=T, N=N, base=base, phase1=None, lease=None, reply=None,
                    outputs=tuple(outputs), slots=[], spans=[])
        info['protected'] = set(s['resident'].values())|({s['tail']['block']} if s['tail'] else set())
        self.rid_ns[rid] = s['ns']
        generation = self.rpc('prepare', prep)['allocation_generation']
        lease_fresh = {}
        if local:
            lease, freed = self.lease(s, a, rid, binding, generation, goal)
            self.rid_ns[lease['lease_id']] = s['ns']; info['lease'] = lease
            p1 = dict(base, allocation_generation=generation, epoch=EPOCH, source=SOURCE, model=MODEL,
                      policy='goal_query' if goal else 'observation_query', tokens=list(T[:e]),
                      prompt_digest=digest(list(T)), lease=lease, lease_digest=digest(lease))
            p1['request_digest'] = digest(p1)
            try:
                info['reply'] = self.rpc('phase1', p1)
            finally:
                drained = self.rpc('phase1_drain', dict(request=rid, binding=binding, lease_id=lease['lease_id'],
                                                        lease_digest=digest(lease), phase1_request=p1))
                assert drained['drained']
                self.pool.release(freed)
            lease_fresh = {int(k): v[0][0] for k, v in lease['run_fresh'].items()}
            from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator
            coordinator = Phase1Coordinator.__new__(Phase1Coordinator)
            coordinator.records = {rid: dict(lease=lease, request=p1)}
            info['phase1'] = coordinator._result(info['reply'], NS(request=rid, identity=binding),
                NS(C=C, e=e, page_tokens=P, tail_candidate=a.tail_candidate), NS(boundary=s['carry']),
                list(goal) if goal else None)
        n = L//P; sel = sorted(range(n) if selected is None else selected)
        Rp = set(s['resident'])|({Q//P} if a.tail_candidate else set())
        retain, restore, release = sorted(set(sel) & Rp), sorted(set(sel)-Rp), sorted(Rp-set(sel))
        target = {p: (s['tail']['block'] if a.tail_candidate and p == Q//P else s['resident'][p]) for p in retain}
        target.update({p: self.pool.alloc()[0] for p in restore})
        run = list(a.run_pages)
        handover = {p: (s['tail']['block'] if a.tail_live and p == Q//P else lease_fresh[p]) for p in run}
        plan = dict(binding=binding, namespace=s['ns'], ns_generation=s['ns_generation'], policy='goal_query',
            local_turn=local, page_tokens=P, computed_start=C, sealed_start=Q, candidate_end=L,
            observation=[o, e] if local else None, phase2_start=S2, prompt_end=N, selected=sel,
            resident_before=sorted(s['resident']),
            tail_before=dict(page=Q//P, valid_end=C) if s['tail'] else None,
            tail_candidate=a.tail_candidate, tail_live=a.tail_live, retain=retain, restore=restore,
            release=release, run_pages=run, run_tail_page=Q//P if a.tail_live else None,
            sparse=dict(logical_matched_tokens=L, dropped_full_attention_tokens=L-len(sel)*P),
            seal_at_completion=dict(from_page=L//P, rule="floor(C'/P)",
                                    phase1_sealed=[Q//P] if a.tail_candidate else []),
            phase1=dict(lease_id=info['lease']['lease_id'], lease_digest=digest(info['lease']),
                        receipt_digest=digest(info['reply']['worker_receipt'])) if local else None,
            bootstrap=None)
        self.rpc('reserve', dict(base, plan=plan, target={str(p): [[b]] for p, b in target.items()},
            handover={str(p): [[b]] for p, b in handover.items()},
            catalog={str(p): m for p, m in s['catalog'].items()}))
        # Engine admission installs [A_t][run]; native allocation appends.
        info.update(plan=plan, sel=sel, retain=retain, target=target, handover=handover,
                    fa=[target[p] for p in sel]+[handover[p] for p in run], mrows={g: [] for g in MAMBA},
                    dropped=L-len(sel)*P, queue=[target[p] for p in restore], cursor=S2, new=True)
        self.runner.samples[rid] = iter(outputs)
        self.running[rid] = info
        return info

    # ---- one native step for several requests ------------------------------
    def step(self, order=None, *, reuse=None):
        """allocation_update per scheduled request, then one execute_model.

        ``order`` is the scheduler order (new rows are added, re-added rows fill
        holes in this order).  ``reuse`` forces {(rid, group): block} for a new
        Mamba slot: a block another request freed earlier in this step.
        """
        from vllm.v1.core.sched.output import NewRequestData
        order = list(self.running) if order is None else list(order)
        sched, news, zero = {}, [], []
        wanted, handoff = set((reuse or {}).values()), set()
        for rid in order:
            info = self.running[rid]; cursor, N = info['cursor'], info['N']
            end = min(N, (cursor//P+1)*P) if cursor < N else cursor+1
            fa, mrows, new = info['fa'], info['mrows'], []
            while len(fa) < -(-(end-info['dropped'])//P):
                new += self.pool.alloc(); fa.append(new[-1])
            curr = -(-end//P)-1
            moved = []
            for g in MAMBA:
                row = mrows[g]; row += [0]*(curr+1-len(row))
                if not row[curr]:
                    want = (reuse or {}).get((rid, g))
                    if want is None:
                        row[curr] = self.pool.alloc()[0]
                    elif want in handoff:
                        handoff.remove(want); row[curr] = self.pool.hand(want)
                    else:
                        row[curr] = want             # a live alias: the owner must refuse it
                    new.append(row[curr])
                    moved += [(row, i) for i in range(curr) if row[i]]
            # The running state moved: this manager frees the old slots now, in
            # the same step; a block another request reuses goes straight to it.
            for row, i in moved:
                self.pool.release([row[i]])
                if row[i] in wanted:
                    self.pool.free.remove(row[i]); handoff.add(row[i])
                row[i] = 0
            rows = [[[b, self.pool.incarnation[b] if b else 0] for b in r] for r in [fa]+[mrows[g] for g in MAMBA]]
            self.rpc('allocation_update', dict(request=rid, binding=info['base']['binding'], start=cursor,
                                               count=end-cursor, rows=rows))
            ids = [list(fa)]+[list(mrows[g]) for g in MAMBA]
            if info['new']:
                news.append(NewRequestData(rid, list(info['T']), [], NS(temperature=0, max_tokens=1),
                                           None, ids, cursor, None))
                info['new'] = False
            else:
                self.runner.set_rows(rid, ids)
            zero += info['queue']+new; info['queue'] = []
            sched[rid] = end-cursor; info['end'] = end
            info['slots'].append(mrows[MAMBA[0]][curr])
        finished, self.finished = self.finished, set()
        mark = len(self.io.log)
        self.runner.execute_model(NS(num_scheduled_tokens=sched, total_num_scheduled_tokens=sum(sched.values()),
            scheduled_new_reqs=news, finished_req_ids=finished, new_block_ids_to_zero=zero,
            kv_connector_metadata=QUIET))
        for rid in order:
            info = self.running[rid]; info['spans'].append((info['cursor'], info['end']))
            info['cursor'] = info['end']
        return self.io.log[mark:]

    def outputs_done(self, rid):
        info = self.running[rid]
        return len(self.runner.requests[rid].output_token_ids) == len(info['outputs'])

    # ---- completion: commit, native free; the runner row leaves next step ---
    def finish(self, rid):
        info = self.running.pop(rid); s = self.sessions[info['sid']]; a = info['a']
        T, sel, fa, mrows = info['T'], info['sel'], info['fa'], info['mrows']
        L, Q = a.candidate_end, a.sealed_start
        out = tuple(self.runner.requests[rid].output_token_ids); full = T+out; C2 = len(full)-1; Q2 = C2//P*P
        r = self.rpc('complete', dict(info['base'], output_token_ids=list(out), computed_end=C2, sealed_end=Q2))
        info['complete'] = r
        info['receipts'] = [x['body'] for x in self.qwen.requests[rid].receipts]
        info['hidden'] = list(self.hidden.get(rid, ()))
        info['visible'] = self.runner.visible[rid]
        from remote_ra_native_growing.contract import digest as native_digest
        base_page = L//P; block = lambda p: fa[len(sel)+(p-base_page)]
        new_pages = set(range(base_page, Q2//P))|({Q//P} if a.tail_candidate else set())
        assert set(map(int, r['new_pages'])) == new_pages
        assert all(m['token_sha256'] == native_digest(full[int(p)*P:(int(p)+1)*P]) for p, m in r['new_pages'].items())
        assert r['resident'] == sorted(set(sel)|set(range(base_page, Q2//P))) and r['dram'] == list(range(Q2//P))
        blocks = {p: info['target'][p] for p in sel}; blocks.update({p: block(p) for p in range(base_page, Q2//P)})
        assert r['resident_blocks'] == {str(p): [[b]] for p, b in sorted(blocks.items())}
        tail2 = dict(page=Q2//P, start=Q2, valid_end=C2, block=block(Q2//P)) if C2 > Q2 else None
        assert set(map(int, r['page_vectors'])) == set(closed_pages(a.phase2_start, C2, P))
        s['catalog'].update({int(p): m for p, m in r['new_pages'].items()})
        s['vectors'] |= set(info['phase1'].page_vectors if info['phase1'] else ())|set(map(int, r['page_vectors']))
        s.update(full=full, C=C2, resident=blocks, tail=tail2, carry=r['carry']['digest'], local=s['local']+1)
        row_blocks = set(fa)|{b for g in MAMBA for b in mrows[g] if b}
        keep = set(blocks.values())|({tail2['block']} if tail2 else set())
        self.pool.release((row_blocks|info['protected'])-keep)
        self.finished.add(rid)                   # finished_req_ids of the next step
        self.done.append(info)
        assert self.pool.used == self.pins()|self.live_blocks(), 'blocks leaked or freed while in use'
        return info

    def live_blocks(self):
        return {b for i in self.running.values()
                for b in set(i['fa'])|{b for g in MAMBA for b in i['mrows'][g] if b}}

    def flush(self):
        """The finished rows leave the runner (as a later step's finished_req_ids would)."""
        if self.finished:
            self.runner._update_states(NS(finished_req_ids=self.finished, new_block_ids_to_zero=[],
                                          scheduled_new_reqs=[]))
            self.finished = set()

    def drive(self, turns, *, gap=1, order=None):
        """Engine loop: at most one admission per pass, then one step of every
        running request (several rows per step), then the completions."""
        pending, passes = list(turns), 0
        while pending or self.running:
            if pending and passes % gap == 0:
                busy = {i['sid'] for i in self.running.values()}
                for k, (sid, kwargs) in enumerate(pending):
                    if sid not in busy:
                        self.admit(sid, **kwargs); del pending[k]; break
            if self.running:
                self.step(order(self) if order else None)
                for rid in [r for r in self.running if self.outputs_done(r)]:
                    self.finish(rid)
            passes += 1
        self.flush()


def summary(info):
    """Everything a request produced that does not name a pool block or request id."""
    r = info['complete']
    phase1 = None
    if info['reply'] is not None:
        phase1 = {k: info['reply'][k] for k in ('start', 'end', 'forward_tokens', 'page_vectors',
                                               'observation_vector', 'goal_vector', 'state_digest',
                                               'tail_capture_digest')}
        phase1['positions'] = [{k: v for k, v in e.items() if k != 'request'}
                               for e in info['reply']['position_evidence']]
    return dict(local_turn=info['a'].local_turn, prompt=info['T'], outputs=info['outputs'],
        forwards=[(b['start'], b['end'], b['phase'], rows,
                   {k: v for k, v in b['position_evidence'].items() if k != 'request'})
                  for b, rows in zip(info['receipts'], info['hidden'], strict=True)],
        visible=info['visible'], phase1=phase1,
        complete=dict(computed_end=r['computed_end'], sealed_end=r['sealed_end'], resident=r['resident'],
                      dram=r['dram'], page_vectors=r['page_vectors'], boundary=r['boundary'],
                      new_pages={p: (m['token_sha256'], m['content'], m['namespace'])
                                 for p, m in r['new_pages'].items()}))


def by_session(h):
    out = {}
    for info in h.done:
        out.setdefault(info['sid'], []).append(summary(info))
    return out


def solo(sid, turns, mrope=False):
    """The same session alone: one request per step."""
    h = BatchHarness(mrope); h.open(sid)
    h.drive([(sid, t) for t in turns])
    return by_session(h)[sid], h


def interleave(*scripts):
    """Round robin over the sessions' turns: (sid, kwargs) in admission order."""
    turns = []
    for k in range(max(len(t) for _, t in scripts)):
        turns += [(sid, t[k]) for sid, t in scripts if k < len(t)]
    return turns


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class BatchedDecodeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))

    def batched(self, sids, *, gap=1, mrope=False, order=None):
        h = BatchHarness(mrope)
        for sid in sids:
            h.open(sid)
        h.drive(interleave(*[(sid, script(k)) for k, sid in enumerate(sids)]), gap=gap, order=order)
        return h

    def check_against_solo(self, h, sids, mrope=False):
        got = by_session(h)
        for k, sid in enumerate(sids):
            alone, _ = solo(sid, script(k), mrope)
            self.assertEqual(got[sid], alone, sid)
            ns = h.sessions[sid]['ns']
            self.assertEqual(h.computed(ns), list(range(h.sessions[sid]['C'])))   # every position once
            carry = h.backend.carry[sid]
            self.assertEqual(carry.payload['blocks'], [W.chain(h.sessions[sid]['full'][:carry.computed_end])]*3)

    def test_two_and_three_requests_per_step_match_each_alone(self):
        for sids in (('sa', 'sb'), ('sa', 'sb', 'sc')):
            h = self.batched(sids)
            widths = [len(s) for s in h.runner.steps if not any(r.startswith(('phase1-', 'goal-')) for r, _, _ in s)]
            self.assertEqual(max(widths), len(sids))
            # Mixed steps: one request's prefill chunk beside another's decode token.
            mixed = [s for s in h.runner.steps if len(s) > 1 and len({c > 1 for _, _, c in s}) == 2]
            self.assertTrue(mixed)
            self.check_against_solo(h, sids)
            # Per-request guard: each forward entered and joined with its batch.
            for info in h.done:
                history = h.qwen.requests[info['rid']].execution_history
                self.assertEqual(history[1:-2], ['executing', 'join_unknown', 'joined']*len(info['spans']))
                self.assertEqual([(b['start'], b['end']) for b in info['receipts']], info['spans'])
            self.assertEqual(h.owner.claims, {})
            for sid in sids:
                h.close(sid, 'shutdown')
            self.assertEqual(h.pool.used, set())

    def test_rows_and_positions_use_each_requests_offset(self):
        h = BatchHarness()
        for sid in ('sa', 'sb', 'sc'):
            h.open(sid)
        seen = []
        original = h.bridge.before_forward
        def spy(runner, so, positions):
            leases = original(runner, so, positions)
            rows = h.owner.scheduled_rows(runner, so)
            seen.append((positions.tolist(), rows, leases))
            return leases
        h.bridge.before_forward = spy
        for k, sid in enumerate(('sa', 'sb', 'sc')):
            h.admit(sid, **script(k)[0])
        h.step(); h.step()
        flat, rows, leases = seen[-1]
        self.assertEqual(len(rows), 3); self.assertIsInstance(leases, tuple)
        for (rid, row, offset, count), lease in zip(rows, leases):
            self.assertEqual(h.runner.input_batch.req_ids[row], rid)
            self.assertEqual(flat[offset:offset+count], list(range(lease.start, lease.end)))
            self.assertEqual(lease.identity.request, rid)
        self.assertEqual([r[2] for r in rows], [0, rows[0][3], rows[0][3]+rows[1][3]])
        # One request per step: a single lease, offset 0, as before batching.
        h2 = BatchHarness(); h2.open('sa'); seen.clear()
        original2 = h2.bridge.before_forward
        h2.bridge.before_forward = lambda r, so, p: seen.append(original2(r, so, p)) or seen[-1]
        h2.admit('sa', **script(0)[0]); h2.step()
        self.assertEqual(type(seen[-1]).__name__, 'StepLease')

    def test_mrope_three_rows_per_request_offset(self):
        sids = ('sa', 'sb', 'sc')
        h = self.batched(sids, mrope=True)
        self.assertEqual(max(len(s) for s in h.runner.steps), 3)
        for info in h.done:
            for body in info['receipts']:
                evidence = body['position_evidence']
                self.assertEqual(evidence['layout'], 'text_mrope_3row_interleaved_11_11_10')
                self.assertEqual(evidence['rows'], [list(range(body['start'], body['end']))]*3)
        self.check_against_solo(h, sids, mrope=True)

    def test_page_crossing_of_one_request_leaves_the_other_untouched(self):
        h = BatchHarness()
        for sid in ('sa', 'sb'):
            h.open(sid)
        turn_a = dict(observation=words(5, 1), header=words(3, 2), outputs=words(9, 3))   # N=8
        turn_b = dict(observation=words(4, 4), header=words(2, 5), outputs=words(12, 6))  # N=6
        a, b = h.admit('sa', **turn_a), h.admit('sb', **turn_b)
        A, B = a['rid'], b['rid']

        def snapshot(rid):
            rec = h.owner.records[rid]; state = h.qwen.requests[rid]
            return deepcopy(dict(rows=h.runner.requests[rid].block_ids, idx=h.runner.mamba_state_idx[rid],
                plan=h.runner._selective_load_plans[rid], current=state.current_hybrid,
                latest=h.backend.latest.get(rid, (None, None))[:2],
                claims={k: v for k, v in h.owner.claims.items() if v[0] == rid},
                tables=rec['last_tables'], zeroed=set(rec['zeroed'])))

        def crossed(info, start):
            # A Page end (latest capture) or a Page start (new FA block, new Mamba slot).
            return info['cursor'] % P == 0 or start % P == 0

        def blocks(info):
            return set(info['fa'])|{x for g in MAMBA for x in info['mrows'][g] if x}

        h.step(); h.step()                          # both prompts: [0,4) then the rest
        crossings = 0
        while not h.outputs_done(A):
            before, starts = snapshot(B), (a['cursor'], b['cursor'])
            log = h.step()
            if not crossed(a, starts[0]) or crossed(b, starts[1]):
                continue
            crossings += 1
            # A's capture, zero or relocation touched only A's blocks; B is as it was.
            self.assertTrue(log)
            self.assertTrue(all(blk in blocks(a) for _, _, keys in log for _, blk in keys))
            self.assertEqual(snapshot(B), before)
        self.assertGreaterEqual(crossings, 3)
        # A completes (its Pages are sealed: D2H captures) while B still decodes.
        before, mark = snapshot(B), len(h.io.log)
        h.finish(A)
        self.assertTrue(h.io.log[mark:])
        self.assertFalse({blk for _, _, keys in h.io.log[mark:] for _, blk in keys} & blocks(b))
        self.assertEqual(snapshot(B), before)
        while not h.outputs_done(B):
            h.step()
        h.finish(B); h.flush()
        got = by_session(h)
        self.assertEqual((got['sa'], got['sb']), (solo('sa', [turn_a])[0], solo('sb', [turn_b])[0]))

    def test_stale_claim_freed_by_one_request_is_reused_by_another_in_the_same_step(self):
        h = BatchHarness()
        for sid in ('sb', 'sa'):
            h.open(sid)
        # B takes batch row 0, A row 1: B is observed and claimed first in the step.
        b = h.admit('sb', observation=words(5, 4), header=words(2, 5), outputs=words(3, 6))
        a = h.admit('sa', observation=words(9, 1), header=words(2, 2), outputs=words(3, 3))
        A, B = a['rid'], b['rid']
        h.step()                                  # both: [0, 4); Mamba slot index 0
        self.assertEqual(h.runner.input_batch.req_ids, [B, A])
        old = {g: a['mrows'][g][0] for g in MAMBA}
        # Next step both move to slot index 1; A frees its old slots first and
        # B's new slots are exactly those blocks, still claimed by A's last step.
        self.assertEqual({h.owner.claims[(g, old[g])][0] for g in MAMBA}, {A})
        h.step(order=[A, B], reuse={(B, g): old[g] for g in MAMBA})
        self.assertEqual({g: b['mrows'][g][1] for g in MAMBA}, old)
        self.assertEqual({h.owner.claims[(g, old[g])] for g in MAMBA}, {(B, h.owner.records[B]['generation'])})
        h.drive([])
        self.assertEqual(h.owner.claims, {})
        got = by_session(h)
        alone_a, _ = solo('sa', [dict(observation=words(9, 1), header=words(2, 2), outputs=words(3, 3))])
        alone_b, _ = solo('sb', [dict(observation=words(5, 4), header=words(2, 5), outputs=words(3, 6))])
        self.assertEqual((got['sa'], got['sb']), (alone_a, alone_b))

    def test_a_live_alias_across_requests_is_still_refused(self):
        h = BatchHarness()
        for sid in ('sb', 'sa'):
            h.open(sid)
        b = h.admit('sb', observation=words(5, 4), header=words(3, 5), outputs=words(3, 6))   # N=8
        a = h.admit('sa', observation=words(4, 1), header=words(2, 2), outputs=words(3, 3))   # N=6
        h.step(); h.step()
        # Next step A decodes [6, 7) on its slot index 1; B moves to index 2 and
        # is handed A's current slot: a live alias, not a stale claim.
        live = {g: a['mrows'][g][1] for g in MAMBA}
        with self.assertRaisesRegex(Exception, 'owned by another generation'):
            h.step(order=[a['rid'], b['rid']], reuse={(b['rid'], g): live[g] for g in MAMBA})

    def test_phase1_probe_while_others_decode_keeps_their_state_exactly(self):
        h = BatchHarness()
        for sid in ('sa', 'sb', 'sc'):
            h.open(sid)
        h.drive([('sc', script(2)[0])])                       # C's turn 0, alone
        a = h.admit('sa', **script(0)[0]); b = h.admit('sb', **script(1)[0])
        A, B = a['rid'], b['rid']
        while not all(r in h.runner.requests and h.runner.requests[r].output_token_ids for r in (A, B)):
            h.step()
        self.assertEqual(h.runner.input_batch.req_ids, [A, B])

        def state(rid):
            rec = h.owner.records[rid]; q = h.qwen.requests[rid]; req = h.runner.requests[rid]
            blocks = [(g, x) for g, row in zip((FA,)+MAMBA, req.block_ids) for x in row if x]
            return deepcopy(dict(
                record={k: rec[k] for k in ('layout', 'last_tables', 'native_rows', 'native_interval',
                                            'current_allocations', 'allocations', 'zeroed', 'attention_ready',
                                            'sparse_plan', 'generation', 'binding')},
                claims={k: v for k, v in h.owner.claims.items() if v[0] == rid},
                qwen=dict(cursor=q.cursor, current=q.current_hybrid, latest=q.latest_hybrid,
                          step=q.latest_state_step, active=q.active, vectors=q.page_vectors,
                          receipts=len(q.receipts), execution=q.execution, poisoned=q.poisoned),
                latest=h.backend.latest.get(rid, (None, None))[:2], backend=h.backend.requests[rid]['generation'],
                runner=dict(block_ids=req.block_ids, computed=req.num_computed_tokens, prompt=req.prompt_token_ids,
                            outputs=req.output_token_ids, idx=h.runner.mamba_state_idx[rid],
                            plan=h.runner._selective_load_plans[rid]),
                store={k: h.store.read('attention' if k[0] == FA else 'hybrid', *k) for k in blocks}))
        before = {rid: state(rid) for rid in (A, B)}
        drops = len(h.runner.dropped)
        c = h.admit('sc', **script(2)[1])                    # phase-1 probe of C (local turn 1)
        C = c['rid']
        self.assertEqual(sorted(h.runner.dropped[drops:]), sorted([A, B]))
        self.assertEqual(h.runner.input_batch.req_ids, [])   # the probe's row left at its drain
        self.assertEqual({rid: state(rid) for rid in (A, B)}, before)
        # vLLM re-adds them from requests[rid].block_ids in scheduler order.
        readds = len(h.runner.readded)
        h.step(order=[B, A, C])
        self.assertEqual(h.runner.readded[readds:], [B, A])
        self.assertEqual(h.runner.input_batch.req_ids, [C, B, A])
        h.drive([], order=lambda hh: [r for r in (B, A, C) if r in hh.running])
        # A and B continued exactly: equal to each alone, and the drop/re-add
        # moved no GDN state (H2D = one relocation per slot move, nothing more).
        got = by_session(h)
        self.assertEqual(got['sa'], solo('sa', script(0)[:1])[0])
        self.assertEqual(got['sb'], solo('sb', script(1)[:1])[0])
        self.assertEqual(got['sc'], solo('sc', script(2)[:2])[0])
        from remote_ra_growing_worker.tensors import request_counters
        for info in (a, b):
            moves = sum(1 for x, y in zip(info['slots'], info['slots'][1:]) if x != y)
            self.assertEqual(request_counters(h.io, info['rid'])['h2d_joined'], HYBRID*moves)

    def test_one_request_steps_unchanged(self):
        # The pre-batching driver (W.Harness.run_turn) and the batched driver at
        # batch size one produce the same receipts, state and IO sequence.
        old = W.Harness(); old.open('s'); hidden = record_hidden(old.qwen)
        new = BatchHarness(); new.open('s')
        for turn in script(0):
            info = old.run_turn('s', **turn)
            new.drive([('s', turn)])
            got = new.done[-1]
            self.assertEqual(hidden[info['rid']], got['hidden'])
            self.assertEqual([r['body']['sequence'] for r in old.qwen.requests[info['rid']].receipts],
                             [b['sequence'] for b in got['receipts']])
            self.assertEqual([r['body']['position_evidence'] for r in old.qwen.requests[info['rid']].receipts],
                             [b['position_evidence'] for b in got['receipts']])
            for key in ('computed_end', 'sealed_end', 'resident', 'dram', 'page_vectors', 'boundary'):
                self.assertEqual(info['complete'][key], got['complete'][key])
        self.assertEqual([(op, kind) for op, kind, _ in old.io.log], [(op, kind) for op, kind, _ in new.io.log])
        self.assertTrue(all(len(s) == 1 for s in new.runner.steps))

    def test_guard_refuses_unowned_rows_and_a_batch_that_differs_from_the_schedule(self):
        h = BatchHarness()
        h.open('sa')
        a = h.admit('sa', **script(0)[0]); h.step()
        so = NS(num_scheduled_tokens={a['rid']: 1, 'foreign': 1}, total_num_scheduled_tokens=2,
                scheduled_new_reqs=[], finished_req_ids=set(), new_block_ids_to_zero=[], kv_connector_metadata=QUIET)
        with self.assertRaisesRegex(RuntimeError, 'owned requests only'):
            h.runner.execute_model(so)
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        with self.assertRaisesRegex(RuntimeError, 'persistent batch differs'):
            RunnerAllocationOwner.scheduled_rows(h.runner, NS(num_scheduled_tokens={a['rid']: 1, 'x': 1}))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PerRequestIOTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    @staticmethod
    def frontend(frames):
        """remote_ra_http.evidence.FrontendEvidence.complete's IO arithmetic."""
        first = frames[0]['http_measurement']['before']; last = frames[-1]['http_measurement']['after']
        prepare = sum(f['http_measurement']['after']['h2d_joined']-f['http_measurement']['before']['h2d_joined']
                      for f in frames if f['operation'] in ('phase1', 'phase1_drain'))
        total = last['h2d_joined']-first['h2d_joined']
        assert total >= prepare and last['h2d_issued'] == last['h2d_joined']
        return prepare, total-prepare, frames[-1]['http_measurement']['active_owned_kv_bytes']

    def test_overlapping_requests_attribute_h2d_and_active_kv_per_request(self):
        h = BatchHarness()
        sids = ('sa', 'sb', 'sc')
        for sid in sids:
            h.open(sid)
        h.drive(interleave(*[(sid, script(k)) for k, sid in enumerate(sids)]))
        self.assertEqual(max(len(s) for s in h.runner.steps), 3)
        attributed, inflated = 0, 0
        for info in h.done:
            rid, local = info['rid'], info['a'].local_turn
            frames = h.receipts[rid]
            self.assertEqual(frames[-1]['operation'], 'complete')
            prepare, restore, active = self.frontend(frames)
            moves = sum(1 for x, y in zip(info['slots'], info['slots'][1:]) if x != y)
            self.assertEqual(prepare, HYBRID*bool(local))
            self.assertEqual(restore, BYTES['attention']*len(info['plan']['restore']) + HYBRID*(bool(local)+moves))
            self.assertEqual(active, BYTES['attention']*len(info['fa']) + HYBRID)
            attributed += prepare+restore
            # The process counters between this request's first and last frame
            # also hold the other requests' bytes.
            process = h.process[rid][-1][1]-h.process[rid][0][0]
            inflated += process > prepare+restore
        self.assertEqual(attributed, h.io._http_h2d_joined)        # every byte, exactly once
        self.assertGreater(inflated, 0)

    def test_attribution_scopes_nest_and_measured_command_counts_own_claims(self):
        from remote_ra_growing_worker.tensors import attributed, request_counters
        from remote_ra_http.evidence import measured_command
        io = NS(_http_h2d_issued=0, _http_h2d_joined=0, views=[NS(group=0, page_bytes=16), NS(group=1, page_bytes=3)])
        def copy(n):
            io._http_h2d_issued += n; io._http_h2d_joined += n
        with attributed(io, 'a'):
            copy(5)
            with attributed(io, 'b'):
                copy(7)
            copy(1)
        self.assertEqual(request_counters(io, 'a'), dict(h2d_issued=6, h2d_joined=6, zero_joins=0))
        self.assertEqual(request_counters(io, 'b'), dict(h2d_issued=7, h2d_joined=7, zero_joins=0))
        self.assertEqual(request_counters(io, 'c'), dict(h2d_issued=0, h2d_joined=0, zero_joins=0))
        backend = NS(io=io, owner=NS(claims={(0, 1): ('a', 1), (0, 2): ('b', 2), (1, 3): ('a', 1)}))
        r = measured_command(backend, 'complete', lambda b, q: copy(4) or dict(operation='complete'), None,
                             request='b')
        m = r['http_measurement']
        self.assertEqual((m['before']['h2d_joined'], m['after']['h2d_joined']), (7, 11))
        self.assertEqual(m['active_owned_kv_bytes'], 16)
        r = measured_command(backend, 'complete', lambda b, q: dict(operation='complete'), None, request='a')
        self.assertEqual(r['http_measurement']['active_owned_kv_bytes'], 16+3)
        # Without a request: the process counters and every claim, as before.
        r = measured_command(backend, 'close_session', lambda b, q: dict(operation='close_session'), None)
        self.assertEqual(r['http_measurement']['after']['h2d_joined'], 17)
        self.assertEqual(r['http_measurement']['active_owned_kv_bytes'], 2*16+3)


class PositionOffsetTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, 'CPU torch required')
    def test_text_positions_read_the_requests_own_columns(self):
        from remote_ra_live_v2.positions import text_positions
        runner = NS(uses_mrope=False)
        flat = torch.tensor([7, 8, 9, 30, 31, 5])
        self.assertEqual(text_positions(runner, flat, 30, 2, 'b', 3)[0], (30, 31))
        with self.assertRaisesRegex(RuntimeError, 'positions changed'):
            text_positions(runner, flat, 30, 2, 'b', 0)
        tokens = list(range(40))
        req = NS(mm_features=None, prompt_embeds=None, prompt_token_ids=tokens, mrope_position_delta=0,
                 mrope_positions=torch.arange(40).repeat(3, 1))
        runner = NS(uses_mrope=True, requests={'b': req}, positions=flat)
        logical, evidence = text_positions(runner, flat.repeat(3, 1), 30, 2, 'b', 3)
        self.assertEqual((logical, evidence['rows']), ((30, 31), [[30, 31]]*3))
        rows = flat.repeat(3, 1); rows[1, 4] = 0
        with self.assertRaisesRegex(RuntimeError, 'all three native rows'):
            text_positions(runner, rows, 30, 2, 'b', 3)
        with self.assertRaisesRegex(RuntimeError, r'\(3,tokens\)'):
            text_positions(runner, flat.repeat(3, 1), 5, 1, 'b', 6)

    @unittest.skipUnless(torch is not None, 'CPU torch required')
    def test_text_request_scans_each_prompt_list_once(self):
        # Two calls per request per step (rows and positions): the ids of a
        # running request's prompt are scanned once, a replaced list again.
        from remote_ra_live_v2.positions import text_request
        class Tokens(list):
            scans = 0
            def __iter__(self):
                Tokens.scans += 1
                return super().__iter__()
        req = NS(mm_features=None, prompt_embeds=None, prompt_token_ids=Tokens(range(8)),
                 mrope_position_delta=0, mrope_positions=torch.zeros(3, 8))
        runner = NS(requests={'r': req})
        for _ in range(4):
            text_request(runner, 'r')
        self.assertEqual(Tokens.scans, 1)
        req.prompt_token_ids = [1, -2, 3, 4, 5, 6, 7, 8]
        with self.assertRaisesRegex(RuntimeError, 'prompt_token_ids required'):
            text_request(runner, 'r')
        req.mrope_position_delta = 1; req.prompt_token_ids = Tokens(range(8))
        with self.assertRaisesRegex(RuntimeError, 'position delta'):
            text_request(runner, 'r')


if __name__ == '__main__':
    unittest.main()
