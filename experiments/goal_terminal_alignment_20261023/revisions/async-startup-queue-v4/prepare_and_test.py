from pathlib import Path
import json,difflib,queue,threading,types
from collections import deque
R=Path(__file__).resolve().parent;D=R.parents[1];p=D/'selective-prefix/source-overlay/startup_runtime.py'
s=p.read_text();before=s
s=s.replace('import threading\n','import threading\nfrom collections import deque\n')
s=s.replace('        require(getattr(core, "batch_queue", None) is None, "batch queue unsupported")','''        batch_queue = getattr(core, "batch_queue", None)
        require(batch_queue is None or (
                    isinstance(batch_queue, deque) and not batch_queue
                    and core.scheduler.scheduler_config.async_scheduling),
                "pre-serving native async batch queue must be empty")''')
assert s!=before
(R/'startup_runtime.before.py').write_text(before);(R/'startup_runtime.candidate.py').write_text(s)
ns={'__name__':'candidate_startup'};module=types.ModuleType(ns['__name__']);import sys;sys.modules[module.__name__]=module
exec(compile(s,'candidate','exec'),module.__dict__)
NS=types.SimpleNamespace
scheduler=NS(requests={},waiting=[],running=[],scheduler_config=NS(async_scheduling=True))
core=NS(batch_queue=deque(),scheduler=scheduler,input_queue=queue.Queue(),output_queue=queue.Queue(),aborts_queue=queue.Queue())
ticket=module.StartupTicket(core,None,threading.get_ident(),phase='assembling');core._remote_ra_startup_ticket=ticket
assert ticket.quiescent()
core.batch_queue=None;assert ticket.quiescent()
for q in [deque([object()]),[]]:
 core.batch_queue=q
 try:ticket.quiescent()
 except module.StartupRejected:pass
 else:raise AssertionError('invalid batch queue accepted')
core.batch_queue=deque();scheduler.scheduler_config.async_scheduling=False
try:ticket.quiescent()
except module.StartupRejected:pass
else:raise AssertionError('unqualified sync queue accepted')
scheduler.scheduler_config.async_scheduling=True;scheduler.requests['r']=object()
try:ticket.quiescent()
except module.StartupRejected:pass
else:raise AssertionError('live request accepted')
(R/'changes.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='before/startup_runtime.py',tofile='after/startup_runtime.py')))
(R/'status.json').write_text(json.dumps({'status':'CANDIDATE_CPU_PASSED','reason':'Native async pre-serving queue exists but is empty; original check required None','safety':'Still reject any in-flight batch, pending requests, IO threads or ownership changes','algorithm_changes':False},indent=2))
print('PASS: empty native async queue admitted; pending batch, wrong type, sync queue and live requests rejected')
