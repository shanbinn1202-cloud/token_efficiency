import importlib,sys
from pathlib import Path
from types import SimpleNamespace as N
R=Path(__file__).resolve().parent; S=R/'selective-prefix/source-overlay'
sys.path.insert(0,str(S)); import remote_ra_live_v2
remote_ra_live_v2.__path__.insert(0,str(R))
from remote_ra_live_v2.phase1_capacity import wait_capacity,prepared_only
from remote_ra_live_v2 import capacity
from remote_ra_live_v2.idle_release import CapacityExceeded
capacity.headroom_by_pool=lambda lifecycle:lifecycle.headroom
class Pool:
 def __init__(self,free,total=101):self.free=free;self.num_gpu_blocks=total
 def get_num_free_blocks(self):return self.free
class Release:
 def __init__(self,c):self.c=c;self.events=[]
 def groups(self):return 1
 def idle(self,lifecycle,exclude):return [(sid,s) for sid,s in self.c.adapter.sessions.items() if sid!=exclude and sid in lifecycle.idle and s.resident]
 def evict(self,sid,rid,short):
  s=self.c.adapter.sessions[sid]; self.c.pool.free+=len(s.resident);s.evicted=s.resident;s.resident=();self.events.append((sid,rid))
 def victims(self,lifecycle,command,short):
  chosen=[];n=0
  for sid,s in self.idle(lifecycle,command.session_id):
   chosen.append(sid);n+=len(s.resident)
   if n>=short:return chosen
 def release(self,chosen,command,short):
  for sid in chosen:self.evict(sid,command.request_id,short)
def setup(free=10,blocks=(50,5)):
 sessions={str(i):N(resident=tuple(range(30)),evicted=(),tail=None,poisoned=False,pending=None) for i in range(3)}
 c=N(records={},pool=Pool(free),mamba_pool=Pool(90),groups=lambda:((0,),(1,),2),pending=lambda sid:False,capacity_need=lambda *a:blocks,adapter=N(sessions=sessions,requests={}))
 c.core=N(_remote_ra_live_v2_lifecycle=N(headroom=(0,0),idle=set()))
 c.core._remote_ra_idle_release=Release(c)
 return c
c=setup();gens=[];works=[]
for i in range(2):
 g=wait_capacity(c,N(session=str(i),request='r'+str(i)),None,None);gens.append(g);works.append(next(g))
 assert not works[0].done()
# Third arrival drops its own history, then may run. No query/selection was repeated.
g=wait_capacity(c,N(session='2',request='r2'),None,None)
try:next(g)
except StopIteration:pass
else:raise AssertionError('third admission should make progress')
assert c.pool.free==70 and all(c.adapter.sessions[str(i)].evicted==tuple(range(30)) for i in range(2))
assert c.adapter.sessions['2'].resident==tuple(range(30))
assert not works[0].done()
c.core._remote_ra_live_v2_lifecycle.idle={'2'}
assert works[0].done()
# Busy generation headroom prevents granting despite raw free pages.
c.core._remote_ra_live_v2_lifecycle.headroom=(25,0);assert not works[0].done()
c.core._remote_ra_live_v2_lifecycle.headroom=(0,0)
# Lease cannot be granted until the previous phase1 is drained.
c.records={'x':1};assert not works[0].done();c.records={}
for g in gens:
 try:next(g)
 except StopIteration:pass
 else:raise AssertionError('wait did not resume')
# Readiness inspects idle victims without issuing worker RPCs.
c=setup(free=40);c.adapter.sessions['0'].evicted=c.adapter.sessions['0'].resident;c.adapter.sessions['0'].resident=()
g=wait_capacity(c,N(session='0',request='r0'),None,None);w=next(g);c.core._remote_ra_live_v2_lifecycle.idle={'1','2'};before=list(c.core._remote_ra_idle_release.events);assert w.done();assert before==c.core._remote_ra_idle_release.events
try:next(g)
except StopIteration:pass
else:raise AssertionError('idle eviction failed')
assert c.pool.free>=80
# Impossible request must not wait forever.
c=setup(blocks=(90,5))
try:next(wait_capacity(c,N(session='0',request='r0'),None,None))
except CapacityExceeded:pass
else:raise AssertionError('impossible request accepted')
assert prepared_only({'phase1':None})
for record in [{'phase1':{}},{'phase1':None,'plan':{}},{'phase1':None,'native_rows':[]},{'phase1':None,'native_interval':(0,1)}]:assert not prepared_only(record)
print('PASS: long-Goal capacity, three-way progress, exact preserved pages, decode headroom, lease fence, pure readiness, idle eviction, impossible ceiling, prepared-only guard')

# Compare refactored capacity shape to the original preallocation calculation.
from remote_ra_live_v2 import window_phase1 as wp
original=(R/'window_phase1.py.before').read_text()
a=original.index('        goal_tokens = None\n',original.index('    def run('));b=original.index('        require(fa <=',a)
body=original[a:b]
code="def old_shape(self, binding, geo, goal_payload):\n    P, W = self.adapter.page_tokens, self.layout.window_tokens\n    s = self.adapter.sessions[binding.session]\n    att, win, _ = self.groups()\n    plan = phase1_chunks(geo.C, geo.e, self.max_tokens, self.chunk_tokens)\n"+'\n'.join(line[4:] for line in body.splitlines())+"\n    return goal_tokens, goal_plan, goal_ring, fresh, width, ring, fa, sliding\n"
space=dict(wp.__dict__);exec(code,space)
owner=N(adapter=N(page_tokens=16,sessions={'x':N(policy='goal_query')}),layout=N(window_tokens=1024),max_tokens=4096,chunk_tokens=4096,goal_tokens=2048,groups=lambda:((0,1),(2,3,4),5),renderer=lambda value:N(token_ids=list(range(int(value['text'])))))
for C in [0,16,1024,16000]:
 for size in [16,4096,40000]:
  geo=N(C=C,e=C+size,fresh_pages=tuple(range(size//16)))
  for n in [0,1,2048,2049,32768,45000]:
   text=str(n) if n else None
   expected=space['old_shape'](owner,N(session='x'),geo,text)
   actual=wp.WindowPhase1Coordinator._allocation_shape(owner,N(session='x'),geo,text)
   assert actual==expected
print('PASS: 144 original-vs-refactored KV allocation shapes, including long Goals and window wraps')

from remote_ra_live_v2.goal_async import Phase1Aborted
c=setup();g=wait_capacity(c,N(session='0',request='r0'),None,None);next(g);before=c.pool.free
try:g.throw(Phase1Aborted('cancel'))
except Phase1Aborted:pass
else:raise AssertionError('cancellation swallowed')
assert c.pool.free==before and not c.adapter.sessions['0'].poisoned and c.adapter.sessions['0'].evicted==tuple(range(30))
print('PASS: cancellation propagates without native allocation, poisoning, or losing committed history')

from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
owner=RunnerAllocationOwner.__new__(RunnerAllocationOwner)
owner.release_idle=True;owner.layout=N(state='sliding');owner.runner=N();owner.session_namespaces={'s':'ns'}
owner.sessions={'ns':dict(poisoned=False,resident=(0,1),resident_blocks={0:1,1:2},catalog={0:object(),1:object()})}
rec=dict(identity=N(session='s'),phase1=None);owner.records={'r':rec}
payload=dict(session='s',namespace='ns',pages=[0,1])
fn=owner._evict(payload);out=fn(N(verify_identity=lambda:None),None)
assert out['evicted']==[0,1] and owner.sessions['ns']['resident']==() and len(owner.sessions['ns']['catalog'])==2 and owner.records['r'] is rec
for change in [{'plan':{}},{'phase1':{}},{'native_rows':()}]:
 owner.sessions['ns']['resident']=(0,1);owner.sessions['ns']['resident_blocks']={0:1,1:2};owner.records={'r':dict(identity=N(session='s'),phase1=None,**{k:v for k,v in change.items() if k!='phase1'})}
 if 'phase1' in change:owner.records['r']['phase1']=change['phase1']
 try:owner._evict(payload)
 except Exception:pass
 else:raise AssertionError('active worker record eviction allowed')
print('PASS: actual worker eviction preserves CPU catalog and prepared request; active phases refused')

# Exercise the actual overlap park/resume/cancel path, not just a fake wait loop.
from remote_ra_live_v2.goal_async import OverlapGoalAdmissions
c=setup();life=N(entries={'r0':{}, 'r1':{}});out=[]
a=OverlapGoalAdmissions(life,lambda:None,permits=16)
g=wait_capacity(c,N(session='0',request='r0'),None,None)
assert a.start(g,lambda reply,error:out.append((reply,error))) is a.PARKED
assert 'r0' in a.parked and not a.ready()
c.core._remote_ra_live_v2_lifecycle.idle={'1','2'}
assert a.ready();a.resume()
assert not a.parked and out==[(None,None)]
assert life.entries['r0']['lease_wait_ms']>=0
c=setup();g=wait_capacity(c,N(session='1',request='r1'),None,None)
assert a.start(g,lambda reply,error:out.append((reply,error))) is a.PARKED
assert a.abort('r1','cancel') and isinstance(out[-1][1],Phase1Aborted)
assert not a.parked;a.close()
print('PASS: real 16-permit admission park/resume/cancel, capacity readiness and timing evidence')
