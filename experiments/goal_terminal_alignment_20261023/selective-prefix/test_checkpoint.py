"""Real carry/Qwen lifecycle with explicitly synthetic tensor operations."""
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import struct
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source-overlay'
sys.path.insert(0,str(SOURCE))
from remote_ra_live_v2.contract import NativeIdentity, RenderAssignment, Allocation, ZERO_DIGEST
from remote_ra_live_v2.worker_backend import CudaWorkerBackend
from remote_ra_live_v2.qwen import QwenAlignLifecycle

P=8
LEASE='phase1-'+'0'*32

class Tensor:
    device=NS(type='cpu')
    def __init__(self,values):
        self.values=values;self.ndim=2 if values and isinstance(values[0],list) else 1
        self.shape=(len(values),len(values[0])) if self.ndim==2 else (len(values),)
    def __iter__(self):return (Tensor(v) for v in self.values)
    def __getitem__(self,index):return Tensor(self.values[index])
    def detach(self):return self
    def to(self,**kw):return self
    def clone(self):return Tensor(self.values)
    def tolist(self):return self.values
    def numpy(self):return self
    def tobytes(self):
        flat=sum(self.values,[]) if self.ndim==2 else self.values
        return struct.pack('d'*len(flat),*flat)

def tensor_module():
    module=ModuleType('torch');module.Tensor=Tensor;module.float32='float32'
    module.isfinite=lambda value:NS(all=lambda:True)
    module.tensor=lambda values,**kw:Tensor(values)
    return module

class SyntheticIO:
    evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS'  # Constructor shape only, never READY.
    def __init__(self):self.value=0;self.captures=0;self.restores=0
    def capture(self,*args):self.captures+=1;return {'synthetic_state':self.value}
    def restore(self,groups,kind,payload):self.value=payload['synthetic_state'];self.restores+=1
    def drain(self):pass
    def payload_digest(self,payload):return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

class SyntheticOwner:
    def __init__(self):self.allowed={}
    def verify_identity(self):pass
    def verify_allocations(self,identity,slots,generation):
        assert self.allowed[identity.request]==(identity,generation,slots),'foreign allocation'
    def verify_zero_receipt(self,*args):pass
    def release_after_drain(self,identity,generation,operation):return ()

class CheckpointRun:
    """One synthetic session: the GDN value always equals the logical end."""
    def __init__(self):
        self.io=SyntheticIO();self.owner=SyntheticOwner()
        self.backend=CudaWorkerBackend(self.io,self.owner,source='b'*64)
        self.qwen=QwenAlignLifecycle(self.backend,page_tokens=P)
        self.sessions={};self.carries=[]
    def assignment(self,identity,session,observation=5,header=3):
        local,C=self.sessions.get(session,(0,0))
        if not local:
            return RenderAssignment(identity,local_turn=0,computed_start=0,sealed_start=0,
                candidate_end=0,observation_start=observation,observation_end=observation,
                phase2_start=0,prompt_tokens=observation+header,page_tokens=P,
                token_digest='c'*64,carry_digest=ZERO_DIGEST)
        o=C+2;e=o+observation  # last sampled token + one boundary token, then the observation
        return RenderAssignment(identity,local_turn=local,computed_start=C,sealed_start=C//P*P,
            candidate_end=o//P*P,observation_start=o,observation_end=e,phase2_start=e,
            prompt_tokens=e+header,page_tokens=P,token_digest='c'*64,
            carry_digest=self.backend.carry[session].digest)
    def start(self,turn,session='s',**shape):
        identity=NativeIdentity('r'+str(turn),session,turn,'g'+str(turn),'a'*64,'b'*64,'model')
        generation=turn+1
        a=self.assignment(identity,session,**shape)
        slots=(Allocation(1,2,generation,'mamba',turn+1),)
        self.owner.allowed[identity.request]=(identity,generation,slots)
        self.backend.begin(identity,generation,namespace='ns-'+session)
        if a.local_turn:self.phase1(identity,a,generation)
        state=self.qwen.open(a,tuple(range(a.prompt_tokens)),generation)
        return identity,state,slots
    def phase1(self,identity,a,generation):
        # Probe slot: carry restored after native zero, forward [C, e), state at e.
        self.io.value=0
        self.backend.restore_carry(identity,{1:(9,)},a.computed_start,a.carry_digest,generation)
        assert self.io.value==a.computed_start,'carry value is not the computed end'
        self.io.value=a.observation_end
        self.backend.capture_phase1_state(identity,LEASE,a.observation_end,{1:(9,)},generation)
    def step(self,identity,state,slots,end):
        start=state.cursor;generation=state.allocation_generation
        kwargs=dict(start=start,count=end-start,logical_positions=range(start,end),
            attention_allocations=(Allocation(0,1,generation,'suffix',1),),
            hybrid_allocations=slots,allocation_generation=generation)
        first=self.qwen.before_forward(identity,**kwargs)
        restores=self.io.restores
        assert self.qwen.before_forward(identity,**kwargs) is first
        assert self.io.restores==restores,'duplicate hook restored state again'
        self.io.value=end  # Explicit synthetic forward updates evolving state.
        with patch.dict(sys.modules,torch=tensor_module()):
            self.qwen.after_forward(identity,Tensor([[float(i+1),1.] for i in range(start,end)]))
    def forward(self,identity,state,slots):
        self.step(identity,state,slots,state.assignment.prompt_tokens)
        for _ in range(4):self.step(identity,state,slots,state.cursor+1)
        self.qwen.publish_pages(identity)
    def complete(self,identity):
        receipt=self.qwen.finish(identity,'complete')
        carry=self.backend.carry[identity.session]
        local=self.sessions.get(identity.session,(0,0))[0]
        self.sessions[identity.session]=(local+1,carry.computed_end)
        self.carries.append(carry.computed_end)
        return receipt

class CheckpointTests(unittest.TestCase):
    def test_carry_continues_every_turn_with_multiple_decode(self):
        run=CheckpointRun();committed=[]
        for turn in range(3):
            identity,state,slots=run.start(turn)
            self.assertEqual(state.cursor,state.assignment.phase2_start)
            run.forward(identity,state,slots);run.complete(identity)
            carry=run.backend.carry['s'];committed.append(carry)
            self.assertEqual((carry.computed_end,carry.payload),
                             (state.assignment.prompt_tokens+4,{'synthetic_state':carry.computed_end}))
            self.assertEqual((carry.owner,carry.owner_generation,carry.namespace),(identity,turn+1,'ns-s'))
            self.assertFalse(run.backend.phase1_states or run.backend.latest or run.backend.requests)
        self.assertEqual(run.carries,[12,26,40])
        self.assertEqual(len({id(c) for c in committed}),3)
        self.assertEqual(run.io.restores,4)  # carry + phase-1 state for each local turn > 0

    def test_local_turn_zero_verifies_zero_and_refuses_zero_over_carry(self):
        run=CheckpointRun();identity,state,slots=run.start(0)
        run.forward(identity,state,slots);run.complete(identity)
        self.assertEqual(run.io.restores,0)
        other=NativeIdentity('r1','s',1,'g1','a'*64,'b'*64,'model')
        new_slots=(Allocation(1,2,2,'mamba',2),)
        run.owner.allowed['r1']=(other,2,new_slots)
        run.backend.begin(other,2,namespace='ns-s')
        with self.assertRaisesRegex(RuntimeError,'zero GDN start with a committed carry'):
            run.backend.verify_zeroed_hybrid(other,new_slots,2)

    def test_phase1_state_needs_exact_carry_lineage_and_is_captured_once(self):
        run=CheckpointRun();identity,state,slots=run.start(0)
        run.forward(identity,state,slots);run.complete(identity)
        other=NativeIdentity('r1','s',1,'g1','a'*64,'b'*64,'model')
        run.backend.begin(other,2,namespace='ns-s');carry=run.backend.carry['s']
        with self.assertRaisesRegex(RuntimeError,'carry lineage'):
            run.backend.capture_phase1_state(other,LEASE,20,{1:(9,)},2)
        for end,digest in ((carry.computed_end-1,carry.digest),(carry.computed_end,'f'*64)):
            with self.assertRaisesRegex(RuntimeError,'committed carry absent/stale'):
                run.backend.restore_carry(other,{1:(9,)},end,digest,2)
        with self.assertRaisesRegex(RuntimeError,'one phase-1 Mamba slot'):
            run.backend.restore_carry(other,{1:(9,10)},carry.computed_end,carry.digest,2)
        run.backend.restore_carry(other,{1:(9,)},carry.computed_end,carry.digest,2)
        run.backend.restore_carry(other,{1:(9,)},carry.computed_end,carry.digest,2)  # duplicate hook
        with self.assertRaisesRegex(RuntimeError,'another slot'):
            run.backend.restore_carry(other,{1:(10,)},carry.computed_end,carry.digest,2)
        with self.assertRaisesRegex(RuntimeError,'carry lineage'):
            run.backend.capture_phase1_state(other,LEASE,20,{1:(10,)},2)
        with self.assertRaisesRegex(RuntimeError,'carry lineage'):
            run.backend.capture_phase1_state(other,LEASE,carry.computed_end,{1:(9,)},2)
        run.io.value=20;run.backend.capture_phase1_state(other,LEASE,20,{1:(9,)},2)
        with self.assertRaisesRegex(RuntimeError,'carry lineage'):
            run.backend.capture_phase1_state(other,LEASE,20,{1:(9,)},2)
        with self.assertRaisesRegex(RuntimeError,'after capture'):
            run.backend.restore_carry(other,{1:(9,)},carry.computed_end,carry.digest,2)

    def test_phase1_restore_needs_exact_end_bytes_and_happens_once(self):
        run=CheckpointRun();identity,state,slots=run.start(0)
        run.forward(identity,state,slots);run.complete(identity)
        other,next_state,new_slots=run.start(1)
        e=next_state.assignment.observation_end
        with self.assertRaisesRegex(RuntimeError,'phase-1 Mamba state absent'):
            run.backend.restore_phase1_state(other,new_slots,e+1,2)
        saved=run.backend.phase1_states['r1']
        run.backend.phase1_states['r1']=type(saved)(saved.request,saved.lease_id,saved.end,
            {'synthetic_state':99},saved.digest)
        with self.assertRaisesRegex(RuntimeError,'phase-1 Mamba state absent'):
            run.backend.restore_phase1_state(other,new_slots,e,2)
        run.backend.phase1_states['r1']=saved
        run.backend.restore_phase1_state(other,new_slots,e,2)
        self.assertEqual(run.io.value,e)
        with self.assertRaisesRegex(RuntimeError,'replayed'):
            run.backend.restore_phase1_state(other,new_slots,e,2)

    def test_cross_request_epoch_namespace_and_same_generation_reuse_rejected(self):
        run=CheckpointRun();identity,state,slots=run.start(0)
        run.forward(identity,state,slots);run.complete(identity)
        for changes in (dict(epoch='d'*64),dict(generation='g0'),dict(turn=0),dict(request='r0')):
            values=dict(vars(NativeIdentity('r1','s',1,'g1','a'*64,'b'*64,'model')));values.update(changes)
            with self.assertRaisesRegex(RuntimeError,'cross-request'):
                run.backend.begin(NativeIdentity(**values),2,namespace='ns-s')
        fresh=NativeIdentity('r1','s',1,'g1','a'*64,'b'*64,'model')
        with self.assertRaisesRegex(RuntimeError,'cross-request'):
            run.backend.begin(fresh,1,namespace='ns-s')
        with self.assertRaisesRegex(RuntimeError,'another session namespace'):
            run.backend.begin(fresh,2,namespace='ns-rebuilt')
        with self.assertRaisesRegex(RuntimeError,'namespace required'):
            run.backend.begin(fresh,2,namespace='')

    def test_complete_needs_latest_at_cursor_and_cancel_keeps_carry(self):
        run=CheckpointRun();identity,state,slots=run.start(0)
        run.forward(identity,state,slots);run.complete(identity)
        carry=run.backend.carry['s']
        other,next_state,new_slots=run.start(1);run.forward(other,next_state,new_slots)
        self.assertIn('r1',run.backend.phase1_states)
        with self.assertRaisesRegex(RuntimeError,'latest Mamba state at the computed end'):
            run.backend.drain(other,'complete',2,computed_end=next_state.cursor+1)
        receipt=run.qwen.finish(other,'cancel')
        self.assertTrue(receipt.poisoned)
        self.assertIs(run.backend.carry['s'],carry)
        self.assertFalse(run.backend.phase1_states or run.backend.latest or run.backend.requests)

    def test_carry_changed_during_transaction_refused(self):
        run=CheckpointRun();identity,state,slots=run.start(0)
        run.forward(identity,state,slots);run.complete(identity)
        other,next_state,new_slots=run.start(1);run.forward(other,next_state,new_slots)
        carry=run.backend.carry['s']
        run.backend.carry['s']=type(carry)(**dict(vars(carry),computed_end=carry.computed_end+1))
        with self.assertRaisesRegex(RuntimeError,'carry changed during transaction'):
            run.qwen.finish(other,'complete')

    def test_conflicting_active_forward_rejected(self):
        run=CheckpointRun();identity,state,slots=run.start(0)
        kwargs=dict(start=0,count=8,logical_positions=range(8),attention_allocations=(Allocation(0,1,1,'attention'),),
            hybrid_allocations=slots,allocation_generation=1)
        run.qwen.before_forward(identity,**kwargs)
        kwargs['allocation_generation']=2
        with self.assertRaisesRegex(RuntimeError,'conflicting repeated'):
            run.qwen.before_forward(identity,**kwargs)

def checkpoint_engine(pipe):
    from remote_ra_live_v2.lifecycle import EngineLifecycle
    run=CheckpointRun()
    class Endpoint:
        def verify_identity(self):pass
        def frame_identity(self,b):return tuple(b[k] for k in ('request','session','turn','generation'))
        def binding_identity(self,b):return b.ident
        def render_claim(self,b):return b,NS(ident=self.frame_identity(b),identity=b['request'])
        def reserve_worker(self,body,binding):
            identity,state,slots=run.start(body['turn'],body['session'])
            binding.native=identity;binding.state=state;binding.slots=slots
        def enqueue(self,body,binding):run.forward(binding.native,binding.state,binding.slots)
        def drain(self,binding,operation):
            receipt=(run.complete(binding.native) if operation=='complete'
                     else run.qwen.finish(binding.native,operation))
            return dict(drained=receipt.synchronized,operation=operation)
        def verify_drain(self,b,o,r):assert r['drained']
    lifecycle=EngineLifecycle(Endpoint())
    try:
        while True:
            command=pipe.recv();op=command['op']
            if op=='add':pipe.send(lifecycle.admit(command['body']))
            elif op=='complete':pipe.send(lifecycle.completed(command['request']))
            elif op=='cancel':pipe.send(lifecycle.cancel(command['identity']))
            else:
                lifecycle.disconnect();pipe.send(dict(closed=True,carries=run.carries));return
    finally:pipe.close()

http_path=HERE/'http_regression.py' if (HERE/'runtime-manifest.json').exists() else HERE/'test_http.py'
spec=importlib.util.spec_from_file_location('checkpoint_http_route_cases',http_path)
http=importlib.util.module_from_spec(spec);sys.modules[spec.name]=http;spec.loader.exec_module(http)

class CheckpointHttpTests(http.HttpRouteTests):
    async def asyncSetUp(self):
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe()
        self.child=ctx.Process(target=checkpoint_engine,args=(child,));self.child.start();child.close()
    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.assertIn(self.cleanup['carries'],([],[12,26,40]))

class DeadPeerTests(unittest.IsolatedAsyncioTestCase):
    async def check_close(self,revoked):
        from frontend_registry import FrontendRegistry
        from serving_controller import ServingController
        registry=FrontendRegistry.__new__(FrontendRegistry)
        calls=[];event=threading.Event()
        if revoked:event.set()
        async def remote(*args):calls.append('REMOTE');raise AssertionError('dead peer RPC')
        core=NS(resources=NS(engine_dead=not revoked),_remote_ra_authority=NS(revoked=event),registered_add_async=remote)
        engine=NS(engine_core=core,output_processor=NS(abort_requests=lambda *a,**kw:calls.append('local-abort')),
            shutdown=lambda **kw:calls.append('native-shutdown'))
        registry.llm=engine;registry._thread=threading.get_ident();registry._capabilities={};registry._pending={}
        registry._submitted={'r':dict(assigned=NS(internal_id='r'))};registry._active={}
        registry.capture_factory=NS(_captures={})
        controller=ServingController.__new__(ServingController)
        controller._thread=threading.get_ident();controller.registry=registry
        controller._active={};controller._issued={};controller._closing=False;controller._closed=False
        with patch('remote_ra_http.diagnostics.failure') as evidence:
            with self.assertRaises(BaseExceptionGroup):await controller.shutdown_close(timeout=0)
            self.assertEqual(evidence.call_count,1)
        self.assertEqual(calls,['local-abort','native-shutdown'])
        self.assertIn('r',registry._submitted);self.assertFalse(controller._closed)
        self.assertIn('r',registry._unconfirmed_cancels)
    async def test_known_dead_engine_skips_rpc_keeps_ledger(self):await self.check_close(False)
    async def test_authority_eof_skips_rpc_keeps_ledger(self):await self.check_close(True)

if __name__=='__main__':unittest.main()
