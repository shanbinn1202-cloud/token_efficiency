"""Stage B worker pieces (WP-W): Qwen, carry, phase 1, allocation owner, rows.

CPU only; nothing here is CUDA evidence or READY.
- A slot-addressed synthetic TensorIO/owner drives the real backend and Qwen
  lifecycle: the synthetic forward writes its logical end into the running
  Mamba slot, so "slot == start before every forward" proves GDN continuity.
- A fake runner and a fake engine (spec 2 RPC shapes, spec 3.2/3.4 leases and
  plans) drive the real worker composition end to end: phase 1 -> phase 2 ->
  complete -> next turn.  Every KV slot records (position, token) and every
  Mamba slot a hash chain, so each forward checks what it attends and that the
  GDN state it starts from is the dense state at that position.
- CpuTensorTestIO drives the tensor helpers over real CPU tensors.
"""
from collections import namedtuple
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source-overlay'))
from remote_ra_live_v2.contract import (Allocation, NativeIdentity, RenderAssignment, ZERO_DIGEST,
                                        closed_pages, digest)
from remote_ra_live_v2.query_selection import normalized
from remote_ra_live_v2.worker_backend import CudaWorkerBackend
from remote_ra_live_v2.qwen import QwenAlignLifecycle
try:
    import torch
except ImportError:
    torch=None

P=4
LEASE='phase1-'+'1'*32
PROBE={1:(90,),2:(91,)}

class SlotIO:
    evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS'  # Constructor shape only, never READY.
    def __init__(self):self.slots={};self.log=[]
    @staticmethod
    def keys(groups):return [(g,b) for g,ids in sorted(groups.items()) for b in ids]
    def capture(self,groups,kind):
        self.log.append(('capture',kind,self.keys(groups)))
        return {'values':[self.slots.get(k,0) for k in self.keys(groups)]}
    def restore(self,groups,kind,payload):
        keys=self.keys(groups);assert len(keys)==len(payload['values']),'payload/slot count'
        self.log.append(('restore',kind,keys));self.slots.update(zip(keys,payload['values']))
    def zero(self,groups,kind):self.slots.update((k,0) for k in self.keys(groups))
    def drain(self):pass
    def payload_digest(self,payload):return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def count(self,op):return sum(1 for e in self.log if e[:2]==(op,'hybrid'))

class Owner:
    """RunnerAllocationOwner.step stand-in: current rows; new Mamba slots zeroed."""
    def __init__(self,io):self.io=io;self.current={};self.zeroed=set()
    def verify_identity(self):pass
    def lease(self,identity,generation,slots):
        self.current[identity.request]=(generation,{v.key for v in slots})
        for v in slots:
            if v.key not in self.zeroed:self.io.zero({v.group:(v.block,)},'hybrid');self.zeroed.add(v.key)
    def verify_allocations(self,identity,slots,generation):
        current=self.current[identity.request]
        assert current[0]==generation and all(v.key in current[1] for v in slots),'foreign allocation'
    def verify_zero_receipt(self,identity,slots,generation):
        assert all(v.key in self.zeroed for v in slots),'unzeroed Mamba slot'
    def release_after_drain(self,identity,generation,operation):return ()

class Session:
    def __init__(self):
        self.io=SlotIO();self.owner=Owner(self.io)
        self.backend=CudaWorkerBackend(self.io,self.owner,source='b'*64)
        self.qwen=QwenAlignLifecycle(self.backend,page_tokens=P)
        self.turn=self.local=self.C=0;self.leases=[]
    def slots(self,generation,a,b):return (Allocation(1,a,generation,'mamba',1),Allocation(2,b,generation,'mamba',1))
    def open(self,observation,header):
        identity=NativeIdentity(f'r{self.turn}','s',self.turn,f'g{self.turn}','a'*64,'b'*64,'m')
        generation=self.turn+1
        self.backend.begin(identity,generation,namespace='ns')
        if not self.local:
            a=RenderAssignment(identity,local_turn=0,computed_start=0,sealed_start=0,candidate_end=0,
                observation_start=observation,observation_end=observation,phase2_start=0,
                prompt_tokens=observation+header,page_tokens=P,token_digest='c'*64,carry_digest=ZERO_DIGEST)
        else:
            C=self.C;o=C+2;e=o+observation
            a=RenderAssignment(identity,local_turn=self.local,computed_start=C,sealed_start=C//P*P,
                candidate_end=o//P*P,observation_start=o,observation_end=e,phase2_start=e,
                prompt_tokens=e+header,page_tokens=P,token_digest='c'*64,
                carry_digest=self.backend.carry['s'].digest)
            # Phase-1 probe: native zero, carry restore, forward [C, e), state at e.
            self.io.zero(PROBE,'hybrid')
            self.backend.restore_carry(identity,PROBE,C,a.carry_digest,generation)
            assert [self.io.slots[k] for k in SlotIO.keys(PROBE)]==[C,C],'carry is not the state at C'
            self.io.slots.update({k:e for k in SlotIO.keys(PROBE)})
            self.backend.capture_phase1_state(identity,LEASE,e,PROBE,generation)
        return identity,self.qwen.open(a,tuple(range(a.prompt_tokens)),generation)
    def step(self,identity,state,count,slots):
        start=state.cursor;end=start+count;gen=state.allocation_generation
        self.owner.lease(identity,gen,slots)
        lease=self.qwen.before_forward(identity,start=start,count=count,logical_positions=range(start,end),
            attention_allocations=(Allocation(0,1,gen,'suffix',1),),hybrid_allocations=slots,
            allocation_generation=gen)
        assert all(self.io.slots[(v.group,v.block)]==start for v in slots),'GDN state discontinuous'
        self.leases.append(lease)
        for v in slots:self.io.slots[(v.group,v.block)]=end
        return self.qwen.after_forward(identity,torch.tensor([[float(x+1),1.,.5] for x in range(start,end)]))
    def complete(self,identity):
        receipt=self.qwen.finish(identity,'complete')
        self.C=self.backend.carry['s'].computed_end;self.local+=1;self.turn+=1
        return receipt

@unittest.skipUnless(torch is not None,'CPU torch required for hidden rows')
class QwenStageBTests(unittest.TestCase):
    def cold(self,s):
        """Local turn 0, N=8: aligned prefill chunks, slot moves only at Page ends."""
        identity,state=s.open(5,3)
        self.assertEqual(state.cursor,0)
        s.step(identity,state,4,s.slots(1,10,11))
        s.step(identity,state,4,s.slots(1,12,13))  # moved after end 4: relocate latest
        s.step(identity,state,1,s.slots(1,14,15))  # moved after end 8: relocate latest
        s.step(identity,state,1,s.slots(1,14,15))
        return identity,state

    def test_cursor_starts_at_phase2_start_and_phase1_state_restored_once(self):
        s=Session();identity,state=self.cold(s);s.complete(identity)
        self.assertEqual(s.C,10)
        identity,state=s.open(3,2)  # C=10, o=12, e=15, N=17
        a=state.assignment
        self.assertEqual((a.local_turn,state.cursor,a.phase2_start,a.prompt_tokens),(1,15,15,17))
        gen=state.allocation_generation;slots=s.slots(gen,20,21);s.owner.lease(identity,gen,slots)
        with self.assertRaisesRegex(RuntimeError,'discontinuous'):
            s.qwen.before_forward(identity,start=14,count=3,logical_positions=range(14,17),
                attention_allocations=(Allocation(0,1,gen,'suffix'),),hybrid_allocations=slots,
                allocation_generation=gen)
        restores=s.io.count('restore')
        s.step(identity,state,1,slots)
        self.assertTrue(state.phase1_restored)
        self.assertEqual(s.io.count('restore'),restores+1)
        self.assertEqual((s.leases[-1].phase,s.leases[-1].hybrid.restored_checkpoint_seal),('suffix',15))
        s.step(identity,state,1,slots);s.step(identity,state,1,slots)  # end 17: prompt complete
        self.assertEqual(s.leases[-1].phase,'decode')
        self.assertEqual(s.io.count('restore'),restores+1)  # no replay, no relocation
        s.complete(identity)
        carry=s.backend.carry['s']
        self.assertEqual((carry.computed_end,carry.payload['values']),(18,[18,18]))
        self.assertEqual(carry.owner,identity)

    def test_local_turn_zero_verifies_zero_and_leases_are_prefill(self):
        s=Session();identity,state=s.open(5,3)
        s.step(identity,state,4,s.slots(1,10,11))
        self.assertFalse(state.phase1_restored)
        self.assertEqual((s.leases[-1].phase,s.leases[-1].hybrid.restored_checkpoint_seal),('prefill',0))
        self.assertNotIn(('restore','hybrid'),[e[:2] for e in s.io.log])

    def test_page_vectors_every_phase_including_decode(self):
        s=Session();identity,state=s.open(5,3)
        s.step(identity,state,4,s.slots(1,10,11))
        with self.assertRaisesRegex(RuntimeError,'publication incomplete'):
            s.qwen.publish_pages(identity)  # prompt [0, 8) not complete yet
        s.step(identity,state,4,s.slots(1,12,13))
        for _ in range(4):s.step(identity,state,1,s.slots(1,14,15))  # page 2 closes at decode row 11
        self.assertEqual(set(state.page_vectors),{0,1,2})
        for page in (0,1,2):
            self.assertEqual(state.page_vectors[page],normalized([float((page+1)*P),1.,.5]))
        body=s.backend.verify_receipt(identity,'publish',s.qwen.publish_pages(identity))
        self.assertEqual(set(body['page_vectors']),{'0','1','2'})
        self.assertEqual(body['representation'],'pre_final_norm_last_token_l2')
        s.complete(identity)
        # Local turn 1: only Pages closed in phase 2 ([e, C')); earlier ones came from phase 1.
        identity,state=s.open(6,5)  # C=12, o=14, e=20, N=25
        slots=s.slots(state.allocation_generation,20,21)
        s.step(identity,state,4,slots)                                   # [20,24): Page 5
        s.step(identity,state,1,s.slots(state.allocation_generation,22,23))  # moved at 24
        self.assertEqual(set(state.page_vectors),{5})
        self.assertEqual(set(state.page_vectors),set(closed_pages(20,state.cursor,P)))
        for _ in range(3):s.step(identity,state,1,s.slots(state.allocation_generation,22,23))
        self.assertEqual(set(state.page_vectors),{5,6})               # decode row 27
        s.qwen.publish_pages(identity)

    def test_latest_only_at_page_ends_and_unaligned_move_refused(self):
        s=Session();identity,state=s.open(5,3);slots=s.slots(1,10,11)
        s.step(identity,state,2,slots);self.assertEqual(s.io.count('capture'),0)
        s.step(identity,state,2,slots);self.assertEqual(s.io.count('capture'),1)
        self.assertEqual(s.backend.latest['r0'][0],4)
        s.step(identity,state,2,slots);self.assertEqual(s.io.count('capture'),1)
        moved=s.slots(1,12,13);s.owner.lease(identity,1,moved)
        with self.assertRaisesRegex(RuntimeError,'latest evolving Mamba state absent'):
            s.qwen.before_forward(identity,start=6,count=2,logical_positions=range(6,8),
                attention_allocations=(Allocation(0,1,1,'suffix'),),hybrid_allocations=moved,
                allocation_generation=1)

    def test_finish_captures_carry_at_cursor_only_when_needed(self):
        s=Session();identity,state=self.cold(s)          # cursor 10: unaligned
        captures=s.io.count('capture');s.complete(identity)
        self.assertEqual(s.io.count('capture'),captures+1)
        self.assertEqual(s.backend.carry['s'].payload['values'],[10,10])
        s=Session();identity,state=s.open(5,3)
        s.step(identity,state,4,s.slots(1,10,11));s.step(identity,state,4,s.slots(1,10,11))
        captures=s.io.count('capture');s.complete(identity)  # cursor 8: latest already there
        self.assertEqual(s.io.count('capture'),captures)
        self.assertEqual((s.backend.carry['s'].computed_end,s.backend.carry['s'].payload['values']),(8,[8,8]))

    def test_incomplete_complete_refused_and_cancel_keeps_carry(self):
        s=Session();identity,state=self.cold(s);s.complete(identity)
        carry=s.backend.carry['s']
        identity,state=s.open(3,2);s.step(identity,state,1,s.slots(state.allocation_generation,20,21))
        with self.assertRaisesRegex(RuntimeError,'incomplete generation cannot commit'):
            s.qwen.finish(identity,'complete')
        self.assertTrue(s.qwen.finish(identity,'cancel').poisoned)
        self.assertIs(s.backend.carry['s'],carry)
        self.assertFalse(s.backend.phase1_states or s.backend.latest or s.backend.requests)


@unittest.skipUnless(torch is not None,'CPU torch required for TensorIO')
class TensorHelperTests(unittest.TestCase):
    def io(self):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO, LayerView
        conv=torch.arange(18,dtype=torch.float32).reshape(6,3)
        recurrent=torch.arange(24,dtype=torch.float32).reshape(6,2,2)
        return CpuTensorTestIO([LayerView('gdn',1,'hybrid',(conv,recurrent),(0,0))]),conv,recurrent

    def test_copy_moves_whole_state_blocks_device_side(self):
        io,conv,recurrent=self.io()
        io.copy({1:(1,2)},{1:(4,5)},'hybrid')
        self.assertTrue(torch.equal(conv[4:6],conv[1:3]) and torch.equal(recurrent[4:6],recurrent[1:3]))
        self.assertTrue(torch.equal(conv[3],torch.tensor([9.,10.,11.])))
        for source,target,reason in (({1:(1,)},{1:(1,)},'alias'),({1:(1,)},{1:(2,3)},'coverage'),
                                     ({1:(1,)},{2:(2,)},'coverage')):
            with self.assertRaisesRegex(RuntimeError,reason):io.copy(source,target,'hybrid')

    def test_capture_counts_joined_d2h_bytes(self):
        io,conv,recurrent=self.io()
        payload=io.capture({1:(2,)},'hybrid')
        self.assertEqual(io._http_d2h_joined,(3+4)*4)
        io.capture({1:(3,4)},'hybrid')
        self.assertEqual(io._http_d2h_joined,3*(3+4)*4)
        io.restore({1:(5,)},'hybrid',payload)
        self.assertTrue(torch.equal(conv[5],conv[2]) and torch.equal(recurrent[5],recurrent[2]))

# ---------------------------------------------------------------------------
# Fake runner + fake engine: the real worker composition end to end.
# ---------------------------------------------------------------------------

FA,MAMBA=0,(1,2,3)          # 1 FullAttention group, 3 Mamba groups (as the live model)
GROUPS=(FA,)+MAMBA
EPOCH='epoch-raw';SOURCE='b'*64;MODEL='m'

def chain(tokens,state='zero',start=0):
    """Dense GDN stand-in: the state after a token depends on every earlier one."""
    for x,t in enumerate(tokens,start):state=hashlib.sha256(f'{state}|{x}|{t}'.encode()).hexdigest()[:16]
    return state

def vllm_stubs():
    """Only the dataclass shapes native_probe builds; no vLLM runtime."""
    class SchedulerOutput:
        @classmethod
        def make_empty(cls):
            return NS(scheduled_new_reqs=[],num_scheduled_tokens={},total_num_scheduled_tokens=0,
                      num_common_prefix_blocks=[],new_block_ids_to_zero=[],finished_req_ids=set(),
                      kv_connector_metadata=None)
    names=('vllm','vllm.v1','vllm.v1.core','vllm.v1.core.sched','vllm.v1.core.sched.output',
           'vllm.sampling_params','vllm.distributed','vllm.distributed.kv_transfer',
           'vllm.distributed.kv_transfer.kv_connector','vllm.distributed.kv_transfer.kv_connector.v1',
           'vllm.distributed.kv_transfer.kv_connector.v1.offloading',
           'vllm.distributed.kv_transfer.kv_connector.v1.offloading.common')
    mods={n:ModuleType(n) for n in names}
    out=mods['vllm.v1.core.sched.output'];out.SchedulerOutput=SchedulerOutput
    out.NewRequestData=namedtuple('NewRequestData','req_id prompt_token_ids mm_features sampling_params '
                                  'pooling_params block_ids num_computed_tokens lora_request')
    mods['vllm.sampling_params'].SamplingParams=lambda **kw:NS(**kw)
    mods['vllm.distributed.kv_transfer.kv_connector.v1.offloading.common'].OffloadingConnectorMetadata=lambda **kw:NS(**kw)
    return mods

class Store:
    """Physical blocks: FA slots hold (position, token); Mamba slots a chain state."""
    def __init__(self):self.kv={};self.mamba={(g,0):'null' for g in MAMBA}
    def read(self,kind,g,b):
        return list(self.kv.get((g,b),[None]*P)) if kind=='attention' else self.mamba.get((g,b),'garbage')
    def write(self,kind,g,b,value):
        if kind=='attention':self.kv[(g,b)]=list(value)
        else:self.mamba[(g,b)]=value
    def zero(self,g,b):
        if g==FA:self.kv[(g,b)]=[None]*P
        else:self.mamba[(g,b)]='zero'

class FakeIO:
    evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS'  # Constructor shape only, never READY.
    def __init__(self,store):
        self.store=store;self.log=[]
        self.views=[NS(group=g,kind='attention' if g==FA else 'hybrid',allocation_tokens=P) for g in GROUPS]
    def keys(self,groups,kind):
        want={FA} if kind=='attention' else set(MAMBA)
        assert set(groups)==want and all(type(v) is tuple and v and all(type(b) is int and b>0 for b in v)
                                         for v in groups.values()),('incomplete physical group coverage',groups,kind)
        return [(g,b) for g in sorted(groups) for b in groups[g]]
    def capture(self,groups,kind):
        keys=self.keys(groups,kind);self.log.append(('capture',kind,keys))
        return {'kind':kind,'blocks':[self.store.read(kind,g,b) for g,b in keys]}
    def restore(self,groups,kind,payload):
        keys=self.keys(groups,kind);assert payload['kind']==kind and len(payload['blocks'])==len(keys)
        self.log.append(('restore',kind,keys))
        for (g,b),value in zip(keys,payload['blocks']):self.store.write(kind,g,b,value)
    def zero(self,groups,kind):
        keys=self.keys(groups,kind);self.log.append(('zero',kind,keys))
        for g,b in keys:self.store.zero(g,b)
    def drain(self):pass
    def payload_digest(self,payload):return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

class Table:
    block_size=P;blocks_per_kv_block=1
    def __init__(self):
        self.rows={};table=self
        class Rows:
            def __getitem__(self,key):row,cut=key;return table.rows[row][cut]
        self.block_table=NS(np=Rows())
    @property
    def num_blocks_per_row(self):return {r:len(v) for r,v in self.rows.items()}
    def add_row(self,ids,row):self.rows[row]=list(ids)
    def get_row(self,row):return list(self.rows[row])

class FakeRunner:
    """GPUModelRunner seams the worker touches; the forward checks what it reads.

    The persistent batch behaves like vLLM's InputBatch (gpu_model_runner
    _update_states): finished requests leave it; requests not scheduled in a
    step are dropped from it but keep their cached state (``requests``,
    ``mamba_state_idx``, sparse plans); new and re-added requests fill the
    smallest free row from ``requests[rid].block_ids``; condense moves the
    last row into each hole.  A step's tokens are flattened in batch-row
    order, so request k reads positions and hidden rows at its own offset.
    """
    def __init__(self,store):
        self.store=store;self.requests={};self.mamba_state_idx={};self._selective_load_plans={}
        self.execute_model_state=None;self.max_num_tokens=64
        self.cache_config=NS(mamba_cache_mode='align');self.num_spec_tokens=0;self.speculative_config=None
        self.use_async_scheduling=False;self.uses_xdrope_dim=0;self.uses_mrope=False
        self.input_batch=NS(req_ids=[],block_table=NS(block_tables={g:Table() for g in GROUPS}))
        self._remote_ra_final_norm_gain=torch.ones(3);self._remote_ra_vector_space=dict(kind='fake')
        self.computed=[];self.zeroed=[];self.visible={};self.samples={};self.plans={};self.new_requests=[]
        self.dropped=[];self.readded=[];self.steps=[]
    def _tables(self):return self.input_batch.block_table.block_tables.values()
    def _insert(self,rid):
        """InputBatch.add_request: the smallest free row, written from requests[rid].block_ids."""
        batch=self.input_batch.req_ids;assert rid not in batch,('already in the persistent batch',rid)
        row=batch.index(None) if None in batch else len(batch)
        if row==len(batch):batch.append(None)
        batch[row]=rid
        for g,ids in zip(GROUPS,self.requests[rid].block_ids):
            self.input_batch.block_table.block_tables[g].add_row(list(ids),row)
    def _remove(self,rid):
        batch=self.input_batch.req_ids;row=batch.index(rid);batch[row]=None
        for t in self._tables():t.rows.pop(row,None)
    def _condense(self):
        """InputBatch.condense: the last request moves into each hole."""
        batch=self.input_batch.req_ids
        while batch and batch[-1] is None:batch.pop()
        while None in batch:
            hole=batch.index(None);last=len(batch)-1
            batch[hole]=batch[last];batch.pop()
            for t in self._tables():t.rows[hole]=t.rows.pop(last)
            while batch and batch[-1] is None:batch.pop()
    def add(self,rid,tokens,block_ids,start):
        self.requests[rid]=NS(prompt_token_ids=list(tokens),output_token_ids=[],
                              block_ids=[list(r) for r in block_ids],num_computed_tokens=start)
        if self.uses_mrope:   # native _init_mrope_positions of a text-only prompt
            self.requests[rid].__dict__.update(mm_features=None,prompt_embeds=None,mrope_position_delta=0,
                                               mrope_positions=torch.arange(len(tokens)).repeat(3,1))
        self._insert(rid)
    def set_rows(self,rid,block_ids):
        """Scheduler-side new blocks: cached block_ids, and the device row while batched."""
        self.requests[rid].block_ids=[list(r) for r in block_ids]
        if rid in self.input_batch.req_ids:
            row=self.input_batch.req_ids.index(rid)
            for g,ids in zip(GROUPS,block_ids):self.input_batch.block_table.block_tables[g].add_row(ids,row)
    def _update_states(self,so):
        for rid in so.finished_req_ids or ():
            self.mamba_state_idx.pop(rid,None);self._selective_load_plans.pop(rid,None)
            if rid in self.requests:
                del self.requests[rid]
                if rid in self.input_batch.req_ids:self._remove(rid)
        if so.new_block_ids_to_zero:
            self.zeroed.append(list(so.new_block_ids_to_zero))
            for b in so.new_block_ids_to_zero:
                for g in GROUPS:self.store.zero(g,b)
        scheduled=getattr(so,'num_scheduled_tokens',None)
        if scheduled is not None:
            # Unscheduled running requests leave the batch; their cached state stays.
            for rid in [r for r in self.input_batch.req_ids if r is not None and r not in scheduled]:
                self._remove(rid);self.dropped.append(rid)
        self.new_requests+=list(so.scheduled_new_reqs)
        for new in so.scheduled_new_reqs:self.add(new.req_id,new.prompt_token_ids,new.block_ids,new.num_computed_tokens)
        for rid in scheduled or ():
            if rid not in self.input_batch.req_ids:
                assert rid in self.requests,('scheduled request without cached state',rid)
                self._insert(rid);self.readded.append(rid)   # re-added from requests[rid].block_ids
        self._condense()
    def execute_model(self,so):
        assert self.execute_model_state is None
        self._update_states(so)
        batch=list(self.input_batch.req_ids)
        assert batch and len(batch)==len(so.num_scheduled_tokens) and set(batch)==set(so.num_scheduled_tokens)
        from remote_ra_live_v2.worker_rows import install_before_prepare
        install_before_prepare(self,so)
        spans=[]
        for rid in batch:
            req=self.requests[rid];start=req.num_computed_tokens;count=so.num_scheduled_tokens[rid]
            curr=-(-(start+count)//P)-1  # native preprocess_mamba (align): copy prev -> curr
            for g in MAMBA:
                row=req.block_ids[g]
                if start>0 and (start-1)//P!=curr:self.store.mamba[(g,row[curr])]=self.store.mamba.get((g,row[(start-1)//P]),'garbage')
            self.mamba_state_idx[rid]=curr
            spans.append((rid,req,start,count,curr))
        self.steps.append(tuple((rid,start,count) for rid,_,start,count,_ in spans))
        from remote_ra_live_v2 import hooks
        from remote_ra_live_v2.native_probe import active_probe
        # Flattened in batch-row order: request k sits at its query_start_loc.
        flat=torch.cat([torch.arange(start,start+count) for _,_,start,count,_ in spans])
        if self.uses_mrope:self.positions=flat   # logical buffer beside the three MRoPE rows
        hooks.before_forward(self,so,flat.repeat(3,1) if self.uses_mrope else flat)
        hidden=torch.cat([self.forward(rid,req,start,count,curr) for rid,req,start,count,curr in spans])
        result=hooks.after_forward(self,so,hidden)
        if active_probe(self) is not None:return result
        for rid,req,start,count,_ in spans:
            req.num_computed_tokens=start+count
            if start+count>=len(req.prompt_token_ids):req.output_token_ids.append(next(self.samples[rid]))
        return NS(req_id=batch[0],req_ids=tuple(batch))
    def forward(self,rid,req,start,count,idx):
        tokens=req.prompt_token_ids+req.output_token_ids
        plan=self._selective_load_plans.get(rid)
        Lm,dropped=(plan.logical_matched_tokens,plan.dropped_full_attention_tokens) if plan else (0,0)
        assert start>=Lm,'logical cursor precedes the matched boundary'
        self.plans[rid]=(Lm,dropped)
        state=chain(tokens[:start])
        assert all(self.store.mamba.get((g,req.block_ids[g][idx]))==state for g in MAMBA),f'GDN discontinuity: {rid} at {start}'
        row=req.block_ids[FA];out=[]
        for x in range(start,start+count):
            phys=x-dropped
            seen=[self.store.kv.get((FA,row[j//P]),[None]*P)[j%P] for j in range(phys)]
            assert all(v is not None and tokens[v[0]]==v[1] for v in seen),f'{rid}: KV hole or stale KV before {x}'
            pos=[v[0] for v in seen];prefix=[p for p in pos if p<Lm]
            pages=sorted({p//P for p in prefix})
            assert [p for p in pos if p>=Lm]==list(range(Lm,x)) and prefix==[q for g in pages for q in range(g*P,(g+1)*P)],\
                f'{rid}: attended set is not whole prefix Pages + the run from {Lm}'
            self.visible[rid]=pages
            kv=self.store.kv.setdefault((FA,row[phys//P]),[None]*P);kv[phys%P]=(x,tokens[x])
            state=chain([tokens[x]],state,x);self.computed.append((rid,x))
            out.append([x+1.,tokens[x]+1.,1.])
        for g in MAMBA:self.store.mamba[(g,req.block_ids[g][idx])]=state
        return torch.tensor(out)

class Pool:
    def __init__(self,size=500):self.free=list(range(size,0,-1));self.used=set()
    def alloc(self,k=1):
        ids=[self.free.pop() for _ in range(k)];self.used.update(ids);return ids
    def release(self,ids):
        for b in ids:
            assert b in self.used,('double free',b);self.used.remove(b);self.free.append(b)  # LIFO: reuse stale blocks

class Harness:
    """Real worker objects (as attach_loaded_worker wires them) behind a fake engine."""
    def __init__(self):
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        from remote_ra_live_v2.native_probe import NativeQueryForward
        from remote_ra_live_v2.worker_composition import WorkerBridge
        from remote_ra_live_v2.generation_guard import guarded_execute
        self.store=Store();self.pool=Pool();self.runner=FakeRunner(self.store);self.io=FakeIO(self.store)
        self.owner=RunnerAllocationOwner(self.runner,self.io,dict(enabled=True,source=SOURCE,page_tokens=P,
                                                                  loaded_sources_verified=True))
        self.runner._remote_ra_live_v2_allocation_owner=self.owner
        self.backend=CudaWorkerBackend(self.io,self.owner,source=SOURCE);self.owner.bind_backend(self.backend)
        self.runner._remote_ra_native_query_forward=NativeQueryForward(self.runner,self.owner,self.backend)
        self.qwen=QwenAlignLifecycle(self.backend,page_tokens=P)
        self.bridge=WorkerBridge(self.qwen,self.owner);self.owner.bind_bridge(self.bridge)
        self.runner._remote_ra_live_v2_bridge=self.bridge
        self.runner.execute_model=guarded_execute(self.runner,self.runner.execute_model)
        self.worker=NS(_remote_ra_live_v2_backend=self.backend,_remote_ra_live_v2_qwen=self.qwen)
        self.sessions={};self.turn=0;self.rid_ns={};self.frames=[]
    def rpc(self,op,payload):
        from remote_ra_live_v2.worker_composition import worker_dispatch
        receipt=worker_dispatch(self.worker,op,payload);self.frames.append(op);return receipt
    def open(self,sid,generation=0):
        self.sessions[sid]=dict(ns=f'sparse/{sid}/{generation}',ns_generation=generation,local=0,full=(),C=0,
            resident={},tail=None,catalog={},carry=ZERO_DIGEST,vectors=set(),poisoned=False)
    def pins(self):
        return {b for s in self.sessions.values() for b in
                list(s['resident'].values())+([s['tail']['block']] if s['tail'] else [])}
    def close(self,sid,reason):
        s=self.sessions.pop(sid)
        r=self.rpc('close_session',dict(session=sid,namespace=s['ns'],epoch=EPOCH,reason=reason))
        assert r['closed'] and r['namespace']==s['ns']
        self.pool.release(set(s['resident'].values())|({s['tail']['block']} if s['tail'] else set()))
        return r
    def lease(self,s,a,rid,binding,generation,goal):
        C,Q,L,e=a.computed_start,a.sealed_start,a.candidate_end,a.observation_end
        R=sorted(s['resident']);tail=s['tail'];M=-(-e//P)
        fresh=range(Q//P+(1 if tail else 0),M)
        fa=self.pool.alloc(len(fresh));mam={g:self.pool.alloc()[0] for g in MAMBA}
        borrowed=[s['resident'][p] for p in R]+([tail['block']] if tail else [])
        rows=[borrowed+fa]+[[0]*(M-1)+[mam[g]] for g in MAMBA]
        owned=fa+[mam[g] for g in MAMBA]
        raw=dict(lease_id='phase1-'+secrets.token_hex(16),request=rid,binding=binding,session=a.identity.session,
            namespace=s['ns'],generation=generation,epoch=EPOCH,source=SOURCE,local_turn=a.local_turn,page_tokens=P,
            start=C,end=e,sealed_start=Q,candidate_end=L,resident_pages=R,
            tail=dict(page=Q//P,valid_end=C,block_ids=[[tail['block']]]) if tail else None,
            tail_candidate=a.tail_candidate,rows=rows,borrowed=borrowed,owned=owned,zero=list(owned),
            run_fresh={str(p):[[b]] for p,b in zip(fresh,fa)},
            incarnations={str(b):1 for row in rows for b in row if b},
            sparse=dict(logical_matched_tokens=Q,dropped_full_attention_tokens=Q-len(R)*P),
            carry_digest=a.carry_digest,goal=None,mac='c'*64)
        freed=[mam[g] for g in MAMBA]
        if goal:
            width=-(-len(goal)//P);gfa=self.pool.alloc(width);gm={g:self.pool.alloc()[0] for g in MAMBA}
            grows=[gfa]+[[0]*(width-1)+[gm[g]] for g in MAMBA];gowned=gfa+[gm[g] for g in MAMBA]
            raw['goal']=dict(lease_id='goal-'+secrets.token_hex(16),tokens=list(goal),rows=grows,owned=gowned,
                zero=list(gowned),incarnations={str(b):1 for b in gowned},
                payload_sha256=hashlib.sha256(b'goal').hexdigest())
            freed+=gowned;self.rid_ns[raw['goal']['lease_id']]=None
        return raw,freed
    def run_turn(self,sid,*,observation,header,outputs,boundary=(),selected=None,goal=None,
                 cancel=None,extra_row_block=None,max_tokens=None):
        s=self.sessions[sid];local=s['local'];turn=self.turn;self.turn+=1
        rid=f'r{turn}';binding=hashlib.sha256(rid.encode()).hexdigest()
        identity=NativeIdentity(rid,sid,turn,f'g{turn}',hashlib.sha256(EPOCH.encode()).hexdigest(),SOURCE,MODEL)
        if local==0:
            T=tuple(observation)+tuple(header);o=e=len(observation);C=Q=L=S2=0
        else:
            C=s['C'];Q=C//P*P;T=s['full']+tuple(boundary)+tuple(observation)+tuple(header)
            o=len(s['full'])+len(boundary);e=o+len(observation);L=o//P*P;S2=e
        N=len(T)
        a=RenderAssignment(identity,local_turn=local,computed_start=C,sealed_start=Q,candidate_end=L,
            observation_start=o,observation_end=e,phase2_start=S2,prompt_tokens=N,page_tokens=P,
            token_digest=hashlib.sha256(str(T).encode()).hexdigest(),carry_digest=s['carry'])
        base=dict(request=rid,session=sid,turn=turn,local_turn=local,generation=identity.generation,binding=binding)
        prep=dict(base,namespace=s['ns'],ns_generation=s['ns_generation'],identity=asdict(identity),
                  assignment=asdict(a),tokens=list(T))
        info=dict(rid=rid,a=a,T=T,prep=prep,phase1=None,lease=None)
        old_pins={p:b for p,b in s['resident'].items()};old_tail=s['tail']
        protected=set(old_pins.values())|({old_tail['block']} if old_tail else set())
        self.rid_ns[rid]=s['ns']
        generation=self.rpc('prepare',prep)['allocation_generation']
        mark=(len(self.runner.zeroed),len(self.io.log))
        lease_fresh={}
        if local:
            lease,freed=self.lease(s,a,rid,binding,generation,goal)
            self.rid_ns[lease['lease_id']]=s['ns'];info['lease']=lease
            p1=dict(base,allocation_generation=generation,epoch=EPOCH,source=SOURCE,model=MODEL,
                policy='goal_query' if goal else 'observation_query',tokens=list(T[:e]),
                prompt_digest=digest(list(T)),lease=lease,lease_digest=digest(lease))
            p1['request_digest']=digest(p1)
            info['p1']=p1
            if max_tokens is not None:self.runner.max_num_tokens=max_tokens
            try:
                reply=self.rpc('phase1',p1)
            finally:
                drained=self.rpc('phase1_drain',dict(request=rid,binding=binding,lease_id=lease['lease_id'],
                                                     lease_digest=digest(lease),phase1_request=p1))
                assert drained['drained'] and drained['lease_id']==lease['lease_id']
                self.pool.release(freed);self.runner.max_num_tokens=64
                lease_fresh={int(k):v[0][0] for k,v in lease['run_fresh'].items()}
                info.update(phase1_zero=self.zero_since(mark),phase1_visible=self.runner.visible.get(lease['lease_id']),
                            phase1_io=[e[:2] for e in self.io.log[mark[1]:]],phase1_drain=drained)
            # The engine's own receipt correlation (WP-E) builds the Phase1Result.
            from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator
            coordinator=Phase1Coordinator.__new__(Phase1Coordinator)
            coordinator.records={rid:dict(lease=lease,request=p1)}
            info['phase1']=coordinator._result(reply,NS(request=rid,identity=binding),
                NS(C=C,e=e,page_tokens=P,tail_candidate=a.tail_candidate),NS(boundary=s['carry']),
                list(goal) if goal else None)
            info['reply']=reply
            if cancel=='after_phase1':
                self.pool.release(lease_fresh.values());return self.cancel(sid,prep,info)
        n=L//P;sel=sorted(range(n) if selected is None else selected)
        Rp=set(s['resident'])|({Q//P} if a.tail_candidate else set())
        retain,restore,release=sorted(set(sel)&Rp),sorted(set(sel)-Rp),sorted(Rp-set(sel))
        target={p:(s['tail']['block'] if a.tail_candidate and p==Q//P else s['resident'][p]) for p in retain}
        target.update({p:self.pool.alloc()[0] for p in restore})
        run=list(a.run_pages)
        handover={p:(s['tail']['block'] if a.tail_live and p==Q//P else lease_fresh[p]) for p in run}
        plan=dict(binding=binding,namespace=s['ns'],ns_generation=s['ns_generation'],policy='goal_query',
            local_turn=local,page_tokens=P,computed_start=C,sealed_start=Q,candidate_end=L,
            observation=[o,e] if local else None,phase2_start=S2,prompt_end=N,selected=sel,
            resident_before=sorted(s['resident']),tail_before=dict(page=Q//P,valid_end=C) if s['tail'] else None,
            tail_candidate=a.tail_candidate,tail_live=a.tail_live,retain=retain,restore=restore,release=release,
            run_pages=run,run_tail_page=Q//P if a.tail_live else None,
            sparse=dict(logical_matched_tokens=L,dropped_full_attention_tokens=L-len(sel)*P),
            seal_at_completion=dict(from_page=L//P,rule="floor(C'/P)",phase1_sealed=[Q//P] if a.tail_candidate else []),
            phase1=dict(lease_id=info['lease']['lease_id'],lease_digest=digest(info['lease']),
                        receipt_digest=digest(info['reply']['worker_receipt'])) if local else None,bootstrap=None)
        info.update(plan=plan,target=target,handover=handover)
        self.rpc('reserve',dict(base,plan=plan,target={str(p):[[b]] for p,b in target.items()},
            handover={str(p):[[b]] for p,b in handover.items()},catalog={str(p):m for p,m in s['catalog'].items()}))
        # Phase 2: engine admission installs [A_t][run]; native allocation appends.
        fa=[target[p] for p in sel]+[handover[p] for p in run]
        if extra_row_block is not None:fa.append(extra_row_block)
        mrows={g:[] for g in MAMBA};dropped=L-len(sel)*P
        queue=[target[p] for p in restore]   # spec 3.6: the engine zero queue holds restore targets only
        self.runner.add(rid,T,[fa]+[mrows[g] for g in MAMBA],S2);self.runner.samples[rid]=iter(outputs)
        cursor,mark2=S2,(len(self.runner.zeroed),len(self.io.log));steps=0
        while True:
            end=min(N,(cursor//P+1)*P) if cursor<N else cursor+1
            new=[]
            while len(fa)<-(-(end-dropped)//P):new+=self.pool.alloc();fa.append(new[-1])
            curr=-(-end//P)-1
            for g in MAMBA:
                mrows[g]+= [0]*(curr+1-len(mrows[g]))
                if not mrows[g][curr]:new+=self.pool.alloc();mrows[g][curr]=new[-1]
            rows=[[[b,1 if b else 0] for b in r] for r in [fa]+[mrows[g] for g in MAMBA]]
            self.rpc('allocation_update',dict(request=rid,binding=binding,start=cursor,count=end-cursor,rows=rows))
            self.runner.set_rows(rid,[fa]+[mrows[g] for g in MAMBA])
            self.runner.execute_model(NS(num_scheduled_tokens={rid:end-cursor},total_num_scheduled_tokens=end-cursor,
                scheduled_new_reqs=[],finished_req_ids=set(),new_block_ids_to_zero=queue+new,kv_connector_metadata=None))
            queue=[];cursor=end;steps+=1
            if steps==1:info.update(sparse_plan=self.runner._selective_load_plans[rid],visible=self.runner.visible[rid])
            if cancel=='after_admission' and steps==2:return self.cancel(sid,prep,info)
            if len(self.runner.requests[rid].output_token_ids)==len(outputs):break
        info['phase2_zero']=self.zero_since(mark2)
        info['protected']=protected|set(target[p] for p in retain)|set(handover.values())
        out=tuple(self.runner.requests[rid].output_token_ids);full=T+out;C2=len(full)-1;Q2=C2//P*P
        r=self.rpc('complete',dict(base,output_token_ids=list(out),computed_end=C2,sealed_end=Q2))
        info['complete']=r
        # Engine checks (spec 3.8, adapter.complete) and commit.
        from remote_ra_native_growing.contract import digest as native_digest
        base_page=L//P;block=lambda p:fa[len(sel)+(p-base_page)]
        new_pages=set(range(base_page,Q2//P))|({Q//P} if a.tail_candidate else set())
        assert set(map(int,r['new_pages']))==new_pages
        assert all(m['token_sha256']==native_digest(full[int(p)*P:(int(p)+1)*P]) for p,m in r['new_pages'].items())
        assert r['resident']==sorted(set(sel)|set(range(base_page,Q2//P))) and r['dram']==list(range(Q2//P))
        blocks={p:target[p] for p in sel};blocks.update({p:block(p) for p in range(base_page,Q2//P)})
        assert r['resident_blocks']=={str(p):[[b]] for p,b in sorted(blocks.items())}
        assert (r['tail'] is None)==(C2==Q2)
        tail2=None
        if C2>Q2:
            assert r['tail']==dict(page=Q2//P,start=Q2,valid_end=C2,block_ids=[[block(Q2//P)]])
            tail2=dict(page=Q2//P,start=Q2,valid_end=C2,block=block(Q2//P))
        assert set(map(int,r['page_vectors']))==set(closed_pages(S2,C2,P))
        s['catalog'].update({int(p):m for p,m in r['new_pages'].items()})
        s['vectors']|=set(info['phase1'].page_vectors if info['phase1'] else ())|set(map(int,r['page_vectors']))
        s.update(full=full,C=C2,resident=blocks,tail=tail2,carry=r['carry']['digest'],local=local+1)
        # Native free: every row block drops a ref; only the new pins survive.
        self.runner._update_states(NS(finished_req_ids={rid},new_block_ids_to_zero=[],scheduled_new_reqs=[]))
        row_blocks=set(fa)|{b for g in MAMBA for b in mrows[g] if b}
        keep=set(blocks.values())|({tail2['block']} if tail2 else set())
        self.pool.release((row_blocks|protected)-keep)
        assert self.pool.used==self.pins(),'blocks leaked or freed while pinned'
        return info
    def cancel(self,sid,prep,info):
        s=self.sessions[sid]
        r=self.rpc('cancel',prep);info['cancel']=r
        assert r['poisoned'] and r['boundary']==s['carry'] and r['resident']==sorted(s['resident']) and \
            r['dram']==sorted(s['catalog']) and r['tail']==(None if s['tail'] is None else dict(page=s['tail']['page'],
            start=s['tail']['start'],valid_end=s['tail']['valid_end'],block_ids=[[s['tail']['block']]]))
        self.runner._update_states(NS(finished_req_ids={info['rid']},new_block_ids_to_zero=[],scheduled_new_reqs=[]))
        self.pool.release(self.pool.used-self.pins())
        s['poisoned']=True
        return info
    def zero_since(self,mark):
        runner=[b for batch in self.runner.zeroed[mark[0]:] for b in batch]
        worker=[b for op,kind,keys in self.io.log[mark[1]:] if op=='zero' for g,b in keys]
        return set(runner)|set(worker)
    def computed(self,ns):
        return sorted(x for rid,x in self.runner.computed if self.rid_ns.get(rid)==ns)

def words(n,seed):return tuple((seed*37+i*11)%500+1 for i in range(n))

@unittest.skipUnless(torch is not None,'CPU torch required for hidden rows')
class StageBSequenceTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules,vllm_stubs()))
        self.h=Harness();self.h.open('s')

    def check_turn(self,info,s):
        """Exactly-once, zeroing, carry lineage and catalog invariants at a turn boundary."""
        h,C=self.h,s['C']
        self.assertEqual(h.computed(s['ns']),list(range(C)))           # every position computed once
        self.assertFalse(info['phase2_zero']&info['protected'])          # never a retained/handover/pinned block
        if info['phase1'] is not None:
            lease=info['lease'];a=info['a'];R=lease['resident_pages'];goal=lease['goal']
            # Only the lease's own blocks are zeroed; only the carry is restored (no Page restore).
            self.assertEqual(info['phase1_zero'],set(lease['zero'])|set(goal['zero'] if goal else ()))
            self.assertFalse(info['phase1_zero']&set(lease['borrowed']))
            self.assertNotIn(('restore','attention'),info['phase1_io'])
            self.assertEqual(info['phase1_io'].count(('restore','hybrid')),1)
            self.assertEqual(h.runner.plans[lease['lease_id']],(a.sealed_start,a.sealed_start-len(R)*P))
            new=[r for r in h.runner.new_requests if r.req_id==lease['lease_id']]
            self.assertEqual([(r.num_computed_tokens,[list(x) for x in r.block_ids]) for r in new],
                             [(a.computed_start,lease['rows'])])
            self.assertEqual(info['phase1_visible'],sorted(info['plan']['resident_before']))
            # phase1_drain is idempotent: the replay returns the same signed receipt.
            again=h.rpc('phase1_drain',dict(request=info['rid'],binding=lease['binding'],
                lease_id=lease['lease_id'],lease_digest=digest(lease),phase1_request=info['p1']))
            self.assertEqual(again,info['phase1_drain'])
        self.assertEqual(info['visible'],info['plan']['selected'])      # phase 2 attends A_t + the run
        a=info['a'];L=a.candidate_end
        self.assertEqual((info['sparse_plan'].logical_matched_tokens,info['sparse_plan'].dropped_full_attention_tokens),
                         (L,L-len(info['plan']['selected'])*P))
        carry=h.backend.carry['s']
        self.assertEqual((carry.computed_end,carry.digest),(C,s['carry']))
        self.assertEqual(carry.payload['blocks'],[chain(s['full'][:C])]*3)  # GDN after C'-1, dense lineage
        self.assertEqual(set(s['catalog']),set(range(C//P*P//P)))
        self.assertEqual(set(h.owner.sessions[s['ns']]['catalog']),set(s['catalog']))
        self.assertEqual(s['vectors'],set(s['catalog']))                # vector keys == CPU keys == [0, Q/P)
        w=h.owner.sessions[s['ns']]
        self.assertEqual((w['computed_end'],w['sealed_end'],tuple(w['resident'])),(C,C//P*P,tuple(sorted(s['resident']))))

    def test_every_tail_case_then_rebuild(self):
        h=self.h;s=h.sessions['s']
        # Local turn 0: cold [0, 7); C'=9, tail [8, 9).
        info=h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        self.assertIsNone(info['phase1']);self.check_turn(info,s)
        self.assertEqual((s['C'],sorted(s['resident']),s['tail']['page']),(9,[0,1],2))
        # Tail live (o=11 < Q+P=12), release page 1; e=14, N=16, C'=17.
        info=h.run_turn('s',boundary=words(1,4),observation=words(3,5),header=words(2,6),outputs=words(2,7),selected=[0])
        self.assertTrue(info['a'].tail_live);self.check_turn(info,s)
        self.assertEqual(info['handover'][2],s['resident'][2])           # the live tail block became Page 2
        # Tail candidate SELECTED (o=20=Q+P) with a Goal forward; restore page 1.
        goal=words(6,8)
        info=h.run_turn('s',boundary=words(2,9),observation=words(3,10),header=words(1,11),outputs=words(4,12),
                        selected=[1,3,4],goal=goal)
        a=info['a'];self.assertTrue(a.tail_candidate);self.check_turn(info,s)
        self.assertEqual((info['plan']['retain'],info['plan']['restore'],info['plan']['release']),([3,4],[1],[0,2]))
        self.assertIn('4',info['complete']['new_pages'])                  # from the phase-1 tail capture
        self.assertEqual(info['phase1'].goal_vector,normalized([6.,goal[-1]+1.,1.]))
        self.assertEqual(info['phase1'].goal_token_count,6)
        self.assertEqual(info['phase1'].observation_vector,normalized([float(a.observation_end),info['T'][a.observation_end-1]+1.,1.]))
        # Tail candidate NOT selected (released, freed at commit, CPU copy kept); C'=36 == Q'.
        info=h.run_turn('s',boundary=words(1,13),observation=words(2,14),header=words(2,15),outputs=words(4,16),selected=[0,5])
        self.assertTrue(info['a'].tail_candidate);self.assertIn(6,info['plan']['release']);self.check_turn(info,s)
        self.assertIn('6',info['complete']['new_pages']);self.assertIsNone(s['tail'])
        # C == Q (no tail): dense selection restores every released Page.
        info=h.run_turn('s',boundary=words(1,17),observation=words(1,18),header=words(1,19),outputs=words(2,20))
        a=info['a'];self.assertEqual(a.computed_start,a.sealed_start);self.check_turn(info,s)
        self.assertEqual(info['plan']['restore'],[1,2,3,4,6])
        # Tail live with the whole observation inside the tail page (no fresh FA block).
        info=h.run_turn('s',boundary=words(1,21),observation=words(1,22),header=words(2,23),outputs=words(2,24),selected=[2,7])
        self.assertEqual(info['lease']['run_fresh'],{});self.check_turn(info,s)
        # Rebuild: close, new namespace, local turn 0; the global ordinal keeps counting.
        old=s['ns'];h.close('s','rebuild')
        self.assertNotIn(old,h.owner.sessions);self.assertNotIn('s',h.backend.carry)
        self.assertEqual(h.pool.used,set())
        h.open('s',generation=1);s=h.sessions['s']
        info=h.run_turn('s',observation=words(6,25),header=words(3,26),outputs=words(2,27))
        self.assertEqual((info['a'].local_turn,info['a'].identity.turn),(0,6));self.check_turn(info,s)
        info=h.run_turn('s',boundary=words(1,28),observation=words(5,29),header=words(2,30),outputs=words(3,31),selected=[1])
        self.assertEqual(info['a'].local_turn,1);self.check_turn(info,s)
        h.close('s','shutdown');self.assertEqual(h.pool.used,set())
        self.assertNotIn('append',h.frames)

    def test_cancel_after_phase1_poisons_and_keeps_committed_state(self):
        h=self.h;s=h.sessions['s']
        h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        carry=h.backend.carry['s']
        info=h.run_turn('s',boundary=words(1,4),observation=words(3,5),header=words(2,6),outputs=words(2,7),
                        cancel='after_phase1')
        self.assertIs(h.backend.carry['s'],carry)
        self.assertTrue(h.owner.sessions[s['ns']]['poisoned'])
        self.assertFalse(h.backend.phase1_states or h.backend.latest or h.backend.requests)
        with self.assertRaisesRegex(RuntimeError,'committed worker session'):
            h.run_turn('s',boundary=words(1,4),observation=words(3,5),header=words(2,6),outputs=words(2,7))
        h.close('s','rebuild');self.assertEqual(h.pool.used,set())

    def test_cancel_after_admission_poisons_and_close_restores_pool(self):
        h=self.h;s=h.sessions['s']
        h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        committed=dict(h.owner.sessions[s['ns']])
        h.run_turn('s',boundary=words(1,4),observation=words(6,5),header=words(5,6),outputs=words(2,7),
                   selected=[1],cancel='after_admission')
        w=h.owner.sessions[s['ns']]
        self.assertTrue(w['poisoned'])
        self.assertEqual({k:w[k] for k in ('resident','tail','computed_end','carry_digest')},
                         {k:committed[k] for k in ('resident','tail','computed_end','carry_digest')})
        self.assertEqual(h.backend.carry['s'].computed_end,9)
        h.close('s','rebuild');self.assertEqual(h.pool.used,set());self.assertEqual(h.owner.claims,{})

    def test_phase1_too_long_fails_closed_and_drains(self):
        h=self.h
        h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        with self.assertRaisesRegex(RuntimeError,'PHASE1_TOO_LONG'):
            h.run_turn('s',boundary=words(1,4),observation=words(6,5),header=words(2,6),outputs=words(2,7),max_tokens=4)
        driver=h.runner._remote_ra_native_query_forward
        self.assertTrue(driver.request_drained('r1'))
        self.assertFalse(h.runner.requests or h.runner.mamba_state_idx or h.runner._selective_load_plans)
        r=h.rpc('cancel',dict(request='r1',session='s',turn=1,generation='g1',binding=hashlib.sha256(b'r1').hexdigest()))
        self.assertTrue(r['poisoned'])

    def test_zero_set_never_touches_a_pinned_block(self):
        h=self.h;s=h.sessions['s']
        h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        released=s['resident'][1]
        with self.assertRaisesRegex(RuntimeError,'zero set touches a resident or tail block'):
            h.run_turn('s',boundary=words(1,4),observation=words(3,5),header=words(2,6),outputs=words(2,7),
                       selected=[0],extra_row_block=released)
        self.assertEqual(h.store.kv[(FA,released)],[(x,s['full'][x]) for x in range(P,2*P)])  # Page 1 intact

    def test_refusals_and_removed_operations(self):
        h=self.h;s=h.sessions['s']
        for op in ('append','query_prepare','probe_drain'):
            with self.assertRaisesRegex(RuntimeError,'operation refused'):h.rpc(op,dict(request='x',binding='y'))
        h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        with self.assertRaisesRegex(RuntimeError,'local turn 0 requires a closed worker session'):
            h.sessions['s']=dict(s,local=0,carry=ZERO_DIGEST)
            h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        h.sessions['s']=s;h.turn=1
        with self.assertRaisesRegex(RuntimeError,'committed worker session'):
            h.sessions['s']=dict(s,carry='f'*64);h.run_turn('s',boundary=words(1,4),observation=words(3,5),
                                                          header=words(2,6),outputs=words(2,7))
        h.sessions['s']=s
        with self.assertRaisesRegex(RuntimeError,'close_session namespace/reason'):
            h.rpc('close_session',dict(session='s',namespace=s['ns'],epoch=EPOCH))
        # A cancel whose prepare never ran tombstones the request: a delayed prepare is refused.
        prep=dict(request='late',session='s',turn=9,local_turn=1,generation='g9',binding='a'*64,namespace=s['ns'])
        self.assertTrue(h.rpc('cancel',prep)['poisoned'])
        with self.assertRaisesRegex(RuntimeError,'terminal replay'):h.rpc('prepare',prep)

    def test_reserve_refuses_foreign_retain_and_handover_blocks(self):
        h=self.h;s=h.sessions['s']
        h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        original=h.rpc
        def tamper(op,payload):
            if op=='reserve':
                payload=dict(payload,target=dict(payload['target'],**{'0':[[s['resident'][1]]]}))
            return original(op,payload)
        h.rpc=tamper
        with self.assertRaisesRegex(RuntimeError,'retained Page block changed'):
            h.run_turn('s',boundary=words(1,4),observation=words(3,5),header=words(2,6),outputs=words(2,7),selected=[0])
        h=self.h=Harness();h.open('s');s=h.sessions['s']
        h.run_turn('s',observation=words(5,1),header=words(2,2),outputs=words(3,3))
        original=h.rpc
        def swap(op,payload):
            if op=='reserve':   # hand over the tail and the fresh Page in the wrong slots
                payload=dict(payload,handover={'2':payload['handover']['3'],'3':payload['handover']['2']})
            return original(op,payload)
        h.rpc=swap
        with self.assertRaisesRegex(RuntimeError,'handover block is neither'):
            h.run_turn('s',boundary=words(1,4),observation=words(3,5),header=words(2,6),outputs=words(2,7),selected=[0])

    def test_install_before_prepare_composes_row_and_sparse_plan(self):
        from remote_ra_live_v2.worker_rows import install_before_prepare
        from remote_ra_live_v2.sparse_mapping import RowLayout,SparsePlan
        h=self.h;runner=h.runner
        rec=dict(layout=RowLayout(P,8,(0,),{FA:(11,)},{FA:(12,13)}))
        h.owner.records['x']=rec
        runner.add('x',list(range(12)),[[40,12,11,13]]+[[0,0,21]]*3,9)
        install_before_prepare(runner,NS(num_scheduled_tokens={'x':1}))
        self.assertEqual(runner.requests['x'].block_ids[FA],[11,12,13,40])
        self.assertEqual(runner.input_batch.block_table.block_tables[FA].get_row(0),[11,12,13,40])
        self.assertEqual(runner._selective_load_plans['x'],SparsePlan(8,4,(FA,)))
        runner.set_rows('x',[[40,12,13]]+[[0,0,21]]*3)
        with self.assertRaisesRegex(RuntimeError,'fixed sparse block missing'):
            install_before_prepare(runner,NS(num_scheduled_tokens={'x':1}))

if __name__=='__main__':unittest.main()
