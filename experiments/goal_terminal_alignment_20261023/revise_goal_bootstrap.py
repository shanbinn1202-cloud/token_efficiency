from pathlib import Path
import difflib,json
R=Path(__file__).resolve().parent;P=R/'selective-prefix';O=Path('/projects/bcjw/ssun4/selective_prefix_first32_20261006/package');REV=R/'revisions/async-bootstrap-v3';REV.mkdir(exist_ok=False)
changes={}
# Original and qualified prior copy differ only in explicit async topology bounds.
for n in ['source-overlay/bootstrap_contract.py']:
    before=(P/n).read_text();after=(O/n).read_text();changes[n]=(before,after)
n='source-overlay/native_worker_registration.py';before=(P/n).read_text();old='                and not self.scheduling.async_scheduling,'
if before.count(old)!=1:raise RuntimeError('worker gate shape changed')
after=before.replace(old,'                and (not self.scheduling.async_scheduling or\n                     getattr(self.runner, "_remote_ra_async_ready", False)),');changes[n]=(before,after)
patch=[]
for n,(before,after) in changes.items():
    original=REV/'original'/n;original.parent.mkdir(parents=True,exist_ok=True);original.write_text(before);(P/n).write_text(after)
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='v2/'+n,tofile='v3/'+n))
(REV/'changes.patch').write_text(''.join(patch));(REV/'status.json').write_text(json.dumps({'validation':'pending','source':'previous qualified async topology, no change to selection'},indent=2)+'\n')
