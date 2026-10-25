"""Execute candidate methods: pool ownership, native fences and reference counts."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from collections import deque
R=Path(__file__).resolve().parent

def method(path,name):
 tree=ast.parse(path.read_text());nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name];assert len(nodes)==1
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),nodes[0]],type_ignores=[])
 ns={};exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns);return ns[name]
manager=R/'candidate/vllm/v1/core/kv_cache_manager.py';sched=R/'candidate/vllm/v1/core/sched/scheduler.py'
pop=method(manager,'pop_blocks_for_free_by_pool');free=method(sched,'_growing_original__free_request_blocks');drain=method(sched,'_drain_deferred_frees')
class Pool:
 def __init__(self,owned):self.owned=owned;self.freed=[]
 def free_blocks(self,blocks):
  for b in blocks:
   assert any(b is own for own in self.owned),'cross-pool free'
   self.freed.append(b)
class TypeManager:
 def __init__(self,blocks):self.blocks=blocks
 def pop_blocks_for_free(self,rid):assert rid=='r';b=self.blocks;self.blocks=[];return b
class Manager:
 pop_blocks_for_free_by_pool=pop
 def __init__(self,mapping,groups,pools):self.coordinator=NS(group_pool=mapping,single_type_managers=[TypeManager(g) for g in groups]);self.block_pools=pools;self.immediate=[]
 def free(self,r):self.immediate.append(r.request_id)
def scheduler(m):return NS(kv_cache_manager=m,defer_block_free=True,processed_step_seq=4,sched_step_seq=7,deferred_frees=deque())
# Same numeric page IDs in different pools must never be confused.
a,b,c,d=[NS(block_id=i) for i in [1,2,1,2]];p0,p1=Pool([a,b]),Pool([c,d]);m=Manager([0,1],[[a,b],[c,d]],[p0,p1]);s=scheduler(m);req=NS(request_id='r',last_sched_seq=6)
free(s,req);assert not p0.freed and not p1.freed and len(s.deferred_frees)==1
drain(s);assert len(s.deferred_frees)==1
s.processed_step_seq=6;drain(s);assert not p0.freed and not p1.freed
s.processed_step_seq=7;drain(s);assert p0.freed==[b,a] and p1.freed==[d,c] and not s.deferred_frees
free(s,req);assert m.immediate==['r']
# Shared-pool group references, including the same block object twice, survive.
x,y,z=object(),object(),object();pool=Pool([x,y,z]);m=Manager([0,0],[[x,y],[x,z]],[pool]);s=scheduler(m);free(s,req)
s.processed_step_seq=7;drain(s);assert pool.freed==[z,x,y,x]
# Already-consumed/normal completion takes the unchanged immediate-free branch.
m=Manager([0],[[]],[Pool([])]);s=scheduler(m);req.last_sched_seq=4;free(s,req);assert m.immediate==['r'] and not s.deferred_frees
# Empty detached bookkeeping does not enqueue phantom work.
req.last_sched_seq=6;free(s,req);assert not s.deferred_frees
# FIFO fences: completing first batch must not release the later one.
x,y=object(),object();pool=Pool([x,y]);m=Manager([0],[[]],[pool]);s=scheduler(m);s.deferred_frees=deque([(5,[(0,[x])]),(7,[(0,[y])])]);s.processed_step_seq=5;drain(s);assert pool.freed==[x] and len(s.deferred_frees)==1
s.processed_step_seq=7;drain(s);assert pool.freed==[x,y] and not s.deferred_frees
print('PASS: pool-local IDs; no early release; shared references; reverse order; immediate/empty frees; FIFO fences')
