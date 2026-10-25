"""No-sample native forwards: the worker phase-1 driver (Stage B, spec 3.2).

No worker-side allocation and no client authority. Failure to prove a drain
keeps EngineCore references held. The engine half is phase1_coordinator.
"""
from __future__ import annotations
from hashlib import sha256

from .contract import closed_pages, digest, plain_copy, require, same_atoms


class ProbeRows:
    """What a decoupled probe forward keeps of its hidden rows (model_queries reads it).

    Legacy copies every row of the forward to the host: ~133 MB for a 6.5K-token
    observation (5,120 x fp32 per row). Decoupled (the backend has a Page
    compactor), only the Page-closing rows are kept, as one device copy that the
    compactor turns into int8 vectors where they are, and the last row (the
    observation or Goal query) is copied to the host as fp32; one device join
    reads the finiteness of every row of the forward.
    """
    __slots__=('start','count','pages','closing','last','finite','hidden_size')

    def __init__(self,hidden,start,count,page_tokens,closing_pages):
        import torch
        view=hidden[:count].detach()
        pages=tuple(closed_pages(start,start+count,page_tokens)) if closing_pages else ()
        self.start,self.count,self.pages,self.hidden_size=start,count,pages,int(view.shape[1])
        self.closing=None
        if pages:
            first=(pages[0]+1)*page_tokens-1-start
            self.closing=view[first:first+(len(pages)-1)*page_tokens+1:page_tokens].clone()
        self.last=view[count-1].to(device='cpu',dtype=torch.float32).clone()
        self.finite=bool(torch.isfinite(view).all())

    @classmethod
    def join(cls,parts):
        """The rows of a chunked phase 1 (window layout, gemma-goal): its forwards' rows
        as one forward's, in order (the closing rows concatenated on their device)."""
        import torch
        require(parts and all(type(p) is cls for p in parts) and
                all(b.start==a.start+a.count for a,b in zip(parts,parts[1:])),
                'chunked phase-1 rows must be contiguous')
        out=cls.__new__(cls)
        out.start,out.count=parts[0].start,sum(p.count for p in parts)
        out.pages=tuple(page for p in parts for page in p.pages)
        closing=[p.closing for p in parts if p.closing is not None]
        out.closing=torch.cat(closing) if closing else None
        out.last,out.finite=parts[-1].last,all(p.finite for p in parts)
        out.hidden_size=parts[0].hidden_size
        return out


class NativeQueryForward:
    """Worker phase-1 driver: the installed execute_model no-sample seam (spec 3.2).

    Phase 1 computes [C, e) on [R][tail][fresh] with the committed carry; the
    Goal forward, when requested, runs cold on its own lease blocks.  Every
    lease is registered before use and drained exactly once; a drain of a
    never-started lease tombstones it so delayed work is refused.
    """
    # --phase1-in-step: probes whose row left the batch at collect (phase1_step).
    retired=frozenset()
    def __init__(self,runner,owner,backend):
        self.runner,self.owner,self.backend=runner,owner,backend
        self.active=None;self.drains={};self.leases={};self.executing=False
        # Startup KV layout: None is the shared pool; decoupled Mamba rows are
        # ceil(e/M) long and their ids live in pool 1 (interface I0/I2).
        from .contract import startup_layout
        self.layout=startup_layout(getattr(owner,'startup',None))

    def _register(self,payload):
        from .contract import parse_lease
        lease=parse_lease(payload['lease'],attention_groups=self.owner.attention_groups,layout=self.layout)
        rid=lease.lease_id
        record=self.owner.records[payload['request']]
        identity=record['identity']
        require(record['binding']==payload['binding']
                and record['generation']==payload['allocation_generation']
                and lease.lease_digest==payload['lease_digest']
                and lease.request==payload['request']
                and lease.session==identity.session
                and lease.generation==record['generation']
                and lease.source==identity.source
                and lease.epoch==payload['epoch']
                and sha256(lease.epoch.encode()).hexdigest()==identity.epoch,
                'foreign native phase-1 lease')
        prior=self.leases.get(rid)
        if prior is not None:
            require(self._same(prior,payload),'phase-1 replay changed payload')
            return prior
        # Every (pool, id) the lease names: borrowed pins are read-only, owned
        # blocks are zeroed. Neither may be claimed by a generation or held by
        # another undrained lease. Ids are per pool (one pool unless decoupled).
        blocks=lease.block_keys()
        require(not any(key not in self.drains for key in self.leases) and
                not any((lease.pool_of(g),b) in blocks for g,b in self.owner.claims),
                'phase-1 physical allocation alias')
        held=getattr(self.owner,'held_claims',None)
        if held:
            # --gdn-device-handoff: another request's phase-1 slot waits for its copy.
            require(not any((lease.pool_of(g),b) in blocks for g,b in held),
                    'phase-1 physical allocation aliases a held GDN slot')
        # The payload's one digest: kept with a private copy that shares its atoms.
        prior=dict(payload=plain_copy(payload),digest=digest(payload),lease=lease,
                   identity=identity,generation=record['generation'])
        self.leases[rid]=prior
        return prior

    @staticmethod
    def _same(prior,payload):
        """payload is the registered one: digest(payload) == prior['digest'], without
        hashing it again when it holds the registered atoms themselves (the phase-1
        call's own payload, the engine's copy of it on the drain RPC); otherwise
        (e.g. a payload that went through JSON) its digest is compared as before."""
        return same_atoms(prior['payload'],payload) or prior['digest']==digest(payload)

    def recover(self,payload):
        """Join/clean or tombstone a never-started RPC. Delayed work is rejected."""
        rid=payload['lease']['lease_id']
        prior=self.leases.get(rid)
        if prior is None:prior=self._register(payload)
        require(self._same(prior,payload),'phase-1 drain payload changed')
        if rid in self.drains:return self.drains[rid],prior['identity']
        require(not self.executing and (self.active is None or self.active['lease'].lease_id==rid),
                'probe execution not joined: references must remain held')
        lease=prior['lease']
        self.backend.io.drain()
        if rid not in self.retired:
            self._cleanup_request(rid)
            if lease.goal is not None:self._cleanup_request(lease.goal.lease_id)
        self.backend.io.drain()
        require(self.runner.execute_model_state is None,'probe left sampling state')
        body=dict(operation='phase1_drain',binding=payload['binding'],lease_id=rid,
                  lease_digest=lease.lease_digest,drained=True)
        receipt=self.backend.sign_receipt(prior['identity'],'phase1_drain',body)
        self.drains[rid]=receipt;self.active=None
        return receipt,prior['identity']

    def request_drained(self,request):
        return all(rid in self.drains for rid,record in self.leases.items()
                   if record['payload']['request']==request)

    def phase1(self,payload):
        require(self.active is None,'nested private probe')
        prior=self._register(payload);lease=prior['lease'];rid=lease.lease_id
        require(rid not in self.drains,'drained/cancelled phase-1 replay')
        self.active=dict(lease=lease,identity=prior['identity'],generation=prior['generation'],
            request=rid,phase='phase1',start=lease.start,tokens=tuple(payload['tokens']),
            rows=None,restored=False,position_evidence=[],state=None,tail=None)
        self.executing=True
        try:
            from .async_support import before_probe
            before_probe(self.runner)
            from .positions import runtime_mode
            runtime_mode(self.runner,'phase-1 probe')
            require(self.runner.execute_model_state is None,'phase-1 probe: pending execute_model_state')
            # One native forward each (a chunked window phase 1 runs several, gemma-goal).
            forwards=lease.forwards()
            require(all(end-start<=self.runner.max_num_tokens for start,end,_ in forwards),'PHASE1_TOO_LONG')
            from .sparse_mapping import SparsePlan
            parts=[]
            for k,(start,end,rows) in enumerate(forwards):
                if k:
                    # The previous forward's request leaves the runner; its KV stays.
                    self._cleanup_request(rid)
                    self.active.update(start=start,tokens=tuple(payload['tokens'][:end]),rows=None,
                                       restored=False,chunk=k,chunk_rows=rows)
                elif len(forwards)>1:
                    self.active.update(tokens=tuple(payload['tokens'][:end]),chunk=0,chunk_rows=rows)
                # Direct set: install_before_prepare skips probes; the runner maps
                # FA slots to x - (Q - |R|P) and keeps RoPE positions logical.
                self.runner._selective_load_plans[rid]=SparsePlan(
                    lease.sparse['logical_matched_tokens'],
                    lease.sparse['dropped_full_attention_tokens'],lease.attention_groups)
                # Only the first forward zeroes the fresh FA blocks: later ones append to them.
                parts.append(self._execute(rid,self.active['tokens'],start,rows,lease.zero if k==0 else ()))
            rows=parts[0] if len(parts)==1 else (
                ProbeRows.join(parts) if type(parts[0]) is ProbeRows else _cat(parts))
            state,tail=self.active['state'],self.active['tail']
            goal_rows=None
            if lease.goal is not None:
                # Own blocks and zero list: the phase-1 KV is handed to phase 2.
                self._cleanup_request(rid)
                goal=lease.goal
                self.active.update(request=goal.lease_id,phase='goal',start=0,
                                   tokens=goal.tokens,rows=None,restored=False)
                goal_rows=self._execute(goal.lease_id,goal.tokens,0,goal.rows,goal.zero)
            return dict(rows=rows,goal_rows=goal_rows,state=state,tail_capture=tail,
                        position_evidence=plain_copy(self.active['position_evidence']))
        finally:
            # No pending sample state was created: execute_model returns the
            # captured rows before logits/sampling state assignment.
            self.executing=False
            self.recover(payload)

    def arm(self,payload):
        """--phase1-in-step: register the lease and arm the probe as one row of the
        engine's next step (phase1_step); hooks run it there, ``collect`` reads it."""
        require(self.active is None and not self.executing,'nested private probe')
        prior=self._register(payload);lease=prior['lease'];rid=lease.lease_id
        require(rid not in self.drains and lease.goal is None,'drained/cancelled phase-1 replay or Goal forward')
        from .positions import runtime_mode
        runtime_mode(self.runner,'phase-1 probe')
        require(self.runner.execute_model_state is None,'phase-1 probe: pending execute_model_state')
        require(lease.end-lease.start<=self.runner.max_num_tokens,'PHASE1_TOO_LONG')
        from .sparse_mapping import SparsePlan
        self.runner._selective_load_plans[rid]=SparsePlan(
            lease.sparse['logical_matched_tokens'],
            lease.sparse['dropped_full_attention_tokens'],lease.attention_groups)
        self.active=dict(lease=lease,identity=prior['identity'],generation=prior['generation'],
            request=rid,phase='phase1',start=lease.start,tokens=tuple(payload['tokens']),
            rows=None,restored=False,position_evidence=[],state=None,tail=None,mixed=True)

    def collect(self,payload):
        """--phase1-in-step: the step ran the armed row; its outputs, and only its row leaves."""
        a=self.active;rid=payload['lease']['lease_id']
        require(a is not None and a.get('mixed') and a['request']==rid and a['restored'] and
                a['rows'] is not None and not self.executing and self.runner.execute_model_state is None,
                'in-step phase-1 probe not executed')
        from .phase1_step import retire_row
        retire_row(self.runner,rid)
        self.retired=self.retired|{rid}
        self.active=None
        return dict(rows=a['rows'],goal_rows=None,state=a['state'],tail_capture=a['tail'],
                    position_evidence=plain_copy(a['position_evidence']))

    def _execute(self,rid,tokens,start,rows,zero):
        from vllm.v1.core.sched.output import SchedulerOutput,NewRequestData
        from vllm.sampling_params import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata
        a=self.active;count=len(tokens)-start
        require(0<count<=self.runner.max_num_tokens,'probe exceeds native single-batch capacity')
        output=SchedulerOutput.make_empty()
        output.scheduled_new_reqs=[NewRequestData(rid,list(tokens),[],
            SamplingParams(temperature=0,max_tokens=1),None,tuple(list(r) for r in rows),start,None)]
        output.num_scheduled_tokens={rid:count}
        output.total_num_scheduled_tokens=count
        output.num_common_prefix_blocks=[0]*len(rows)
        # Only the lease's own fresh blocks: borrowed R/tail KV is never zeroed.
        output.new_block_ids_to_zero=list(zero)
        output.kv_connector_metadata=OffloadingConnectorMetadata(load_jobs={},store_jobs={},
            jobs_to_flush=set(),selective_load_plans={})
        result=self.runner.execute_model(output)
        require(a['rows'] is not None and result is a['rows'] and self.runner.execute_model_state is None,
                'native no-sample forward seam not executed')
        return result

    def _cleanup_request(self,rid):
        from vllm.v1.core.sched.output import SchedulerOutput
        cleanup=SchedulerOutput.make_empty();cleanup.finished_req_ids={rid}
        self.runner._update_states(cleanup)
        self.runner.mamba_state_idx.pop(rid,None)
        self.runner._selective_load_plans.pop(rid,None)

    def _mamba_slot(self):
        a=self.active;lease=a['lease'];idx=self.runner.mamba_state_idx[a['request']]
        expected=-(-len(a['tokens']) // (lease.mamba_tokens or lease.page_tokens))-1
        require(idx==expected,'phase-1 Mamba slot is not the current chunk end slot')
        rows=a.get('chunk_rows') or lease.rows
        return {g:(rows[g][idx],) for g in self.owner.hybrid_groups}

    def _window_slots(self,end,indexed=False):
        """Window layout (gemma-s1): the phase-1 rows' window after ``end`` tokens, per
        sliding group: the blocks ({g: blocks}), or with ``indexed`` ((index, block), ...);
        the rows of the forward that runs now (a chunked phase 1, gemma-goal)."""
        from .sliding_window import kept_blocks
        lease=self.active['lease'];first,last=kept_blocks(end,self.owner.window_tokens,lease.page_tokens)
        rows=self.active.get('chunk_rows') or lease.rows
        return {g:(tuple((i,rows[g][i]) for i in range(first,last)) if indexed else
                   tuple(rows[g][first:last])) for g in self.owner.hybrid_groups}

    def before_forward(self,positions,offset=0):
        a=self.active;rid=a['request']
        if a.get('mixed'):
            # --phase1-in-step: one row of a step that also runs generation rows.
            require(rid in self.runner.input_batch.req_ids,'in-step probe row missing from the batch')
        else:
            require(self.runner.input_batch.req_ids==[rid],'probe mixed with generation batch')
        from .positions import text_positions
        logical,evidence=text_positions(self.runner,positions,a['start'],len(a['tokens'])-a['start'],rid,offset)
        a['position_evidence'].append(dict(evidence,phase=a['phase']))
        if a['phase']=='phase1' and a.get('chunk',0)==0:
            # Native zeroed the fresh slot and preprocess_mamba ran; the carry
            # (GDN after C-1) overwrites it. No attention Page is restored.
            # A chunked phase 1's later forwards continue the window in place.
            lease=a['lease']
            # Window layout (gemma-s1): the borrowed window, already in its blocks.
            slots=(self._window_slots(lease.start) if getattr(self.backend,'window',False) else
                   self._mamba_slot())
            self.backend.restore_carry(a['identity'],slots,lease.start,
                                       lease.carry_digest,a['generation'])
        elif a['phase']=='phase1' and not getattr(self.backend,'window',False):
            # A new native prefill row can clear/move recurrent slots. Restore
            # the exact private preceding-chunk carry after preprocessing.
            require(a.get('chunk_carry') is not None, 'missing preceding chunk GDN carry')
            self.backend.io.restore(self._mamba_slot(),'hybrid',a['chunk_carry'])
            self.backend.io.drain()
        a['restored']=True

    def after_forward(self,hidden):
        require(self.active is not None and self.active['restored'],'probe forward before private restore')
        a=self.active
        self.backend.io.drain()
        count=len(a['tokens'])-a['start']
        if getattr(self.backend,'compactor',None) is None:
            rows=hidden[:count].detach().to(device='cpu').clone()
        else:
            # Decoupled: the closing rows (phase 1 only; Goal Pages are never
            # published) stay on the device, the query row comes to the host.
            rows=ProbeRows(hidden,a['start'],count,a['lease'].page_tokens,a['phase']=='phase1')
        if a['phase']=='phase1' and a['start']+count<a['lease'].end and not getattr(self.backend,'window',False):
            a['chunk_carry']=self.backend.io.capture(self._mamba_slot(),'hybrid')
            self.backend.io.drain()
        if a['phase']=='phase1' and a['start']+count==a['lease'].end:
            # The last forward of phase 1 (of its chunks, gemma-goal) captures its state.
            lease=a['lease']
            slots=(self._window_slots(lease.end,True) if getattr(self.backend,'window',False) else
                   self._mamba_slot())
            a['state']=self.backend.capture_phase1_state(a['identity'],lease.lease_id,lease.end,
                                                         slots,a['generation'])
            if lease.tail_candidate:
                # Page Q/P closed in phase 1: its CPU copy must exist before its
                # block can leave the rows (SPEC!=DESIGN 1).
                a['tail']=self.backend.io.capture(dict(zip(lease.attention_groups,
                                                           lease.tail.block_ids)),'attention')
        a['rows']=rows
        return rows


def _cat(parts):
    """Host rows of a chunked phase 1 (no Page compactor), in order."""
    import torch
    return torch.cat(parts)


def active_probe(runner):
    """The driver of a probe-only pass (phase 1 or the Goal forward alone), else None.

    An armed in-step probe (--phase1-in-step) is one row of a generation step, so
    the runner, the guards and the CUDA-graph dispatch see a generation step
    (None); the hooks run its row through ``mixed_probe``."""
    driver=getattr(runner,'_remote_ra_native_query_forward',None)
    return (driver if type(driver) is NativeQueryForward and driver.active is not None and
            not driver.active.get('mixed') else None)


def mixed_probe(runner):
    """The driver of an armed in-step probe row (--phase1-in-step), else None."""
    driver=getattr(runner,'_remote_ra_native_query_forward',None)
    return (driver if type(driver) is NativeQueryForward and driver.active is not None and
            driver.active.get('mixed') else None)


def mixed_ids(runner):
    """The request id of an armed in-step probe row, or () (--phase1-in-step)."""
    probe=mixed_probe(runner)
    return () if probe is None else (probe.active['request'],)
