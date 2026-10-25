"""Auditable integration edits; original repository and frozen runs are untouched."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parent
P=ROOT/'selective-prefix'
OLD=Path('/projects/bcjw/ssun4/selective_prefix_first32_20261006')
REV=ROOT/'revisions/async-config-v1'
if REV.exists():
    raise RuntimeError('revision already applied')
changes={}
def change(name,replacements):
    before=changes.get(name,(P/name).read_text())
    after=before
    for source,target,count in replacements:
        if after.count(source)!=count:
            raise RuntimeError((name,source[:90],after.count(source),count))
        after=after.replace(source,target)
    changes[name]=after

change('serve.py',[
 ('MAX_MODEL_LEN=65536;MAX_BATCHED=16384','MAX_MODEL_LEN=262144;MAX_BATCHED=16384',1),
 ('KV_CACHE_GIB=(8,16,24,32)','KV_CACHE_GIB=(8,16,24,25,32)',1),
 ('GEMMA4_KV_CACHE_GIB=(24,28,32)','GEMMA4_KV_CACHE_GIB=(24,25,28,32)',1),
 ("choices=(8,16,24,28,32),default=8","choices=(8,16,24,25,28,32),default=25",1),
 ("choices=('off','piecewise','full','full-decode'),default='off'","choices=('off','piecewise','full','full-decode'),default='full'",1),
 ('choices=(1536,4096),default=1536','choices=(1536,4096),default=4096',1),
 ('choices=(65536,131072,262144),default=65536','choices=(65536,131072,262144),default=262144',1),
 ("'--no-async-scheduling'","'--async-scheduling'",2),
 ("enforce_eager=True,mamba_cache_mode='align'","enforce_eager=True,async_scheduling=True,attention_backend='TRITON_ATTN',mamba_cache_mode='align'",1),
 ("enforce_eager=True,attention_backend=MP.GEMMA4_ATTENTION_BACKEND","enforce_eager=True,async_scheduling=True,attention_backend=MP.GEMMA4_ATTENTION_BACKEND",1),
 ("'--gdn-prefill-backend','triton',*extra,","'--gdn-prefill-backend','triton','--attention-backend','TRITON_ATTN',*extra,",1),
])
change('source-overlay/remote_ra_live_v2/model_profile.py',[
 ('GEMMA4_FORWARD_TOKENS = 8192','GEMMA4_FORWARD_TOKENS = 4096',1),
])
change('source-overlay/remote_ra_live_v2/positions.py',[
 ("('use_async_scheduling',not runner.use_async_scheduling)","('use_async_scheduling',not runner.use_async_scheduling or getattr(runner, '_remote_ra_async_ready', False))",1),
])
change('source-overlay/native_worker_registration.py',[
 ('    require(not executor.scheduler_config.async_scheduling,\n            "initial slice requires synchronous scheduling")',
  '    if executor.scheduler_config.async_scheduling:\n        from remote_ra_live_v2.async_support import attach\n        attach(wrapper.worker.model_runner)',1),
])
change('source-overlay/remote_ra_live_v2/worker_composition.py',[
 ('runner._remote_ra_live_v2_bridge = bridge\n','runner._remote_ra_live_v2_bridge = bridge\n    from .async_support import attach as attach_async\n    attach_async(runner)\n',1),
 ('finite=None if finite is None else finite[offset:offset + count], **drained))\n',
  'finite=None if finite is None else finite[offset:offset + count], **drained))\n                from .async_support import checkpoint\n                checkpoint(self.owner, state)\n',1),
 ('finite=None if every else host[offset:offset + count], finite_ok=every, **drained))\n',
  'finite=None if every else host[offset:offset + count], finite_ok=every, **drained))\n            from .async_support import checkpoint\n            checkpoint(self.owner, state)\n',1),
])
change('source-overlay/remote_ra_live_v2/allocation_owner.py',[
 ('output = tuple(payload["output_token_ids"])\n            C2',
  'output = tuple(payload["output_token_ids"])\n            from .async_support import accepted_boundary\n            accepted_boundary(self, state, payload["computed_end"])\n            C2',1),
 ('identity = rec["identity"]\n            receipt = qwen.finish(identity, "cancel")',
  'identity = rec["identity"]\n            from .async_support import drop\n            drop(self, identity.request)\n            receipt = qwen.finish(identity, "cancel")',1),
])
change('source-overlay/remote_ra_live_v2/allocation_guard.py',[
 ("for field in ('num_preemptions', 'num_output_placeholders'):","for field in ('num_preemptions',):",1),
 ("logical, total, count = request.num_computed_tokens, request.num_tokens, values['num_new_tokens']",
  "placeholders = getattr(request, 'num_output_placeholders', 0)\n    if type(placeholders) is not int or not 0 <= placeholders <= 2:\n        reject('num_output_placeholders', placeholders, 'non-spec native async depth exceeded')\n    logical, total, count = request.num_computed_tokens, request.num_tokens + placeholders, values['num_new_tokens']",1),
])
change('source-overlay/remote_ra_live_v2/chunk_protocol.py',[
 ('if count!=1 or request.num_tokens!=start+1:',"if count!=1 or request.num_tokens+getattr(request,'num_output_placeholders',0)!=start+1:",1),
])
change('source-overlay/remote_ra_live_v2/native_probe.py',[
 ("self.executing=True\n        try:\n","self.executing=True\n        try:\n            from .async_support import before_probe\n            before_probe(self.runner)\n",1),
])
change('source-overlay/vllm/v1/engine/core.py',[
 ('        authority = getattr(self, "_remote_ra_authority", None)\n        if authority is not None:\n            authority.poll()\n\n        lifecycle = None',
  '        if getattr(self, "_remote_ra_dynamic_mode", False) and self.batch_queue:\n            from remote_ra_live_v2.async_support import drain_before_admission\n            drain_before_admission(self)\n        authority = getattr(self, "_remote_ra_authority", None)\n        if authority is not None:\n            authority.poll()\n\n        lifecycle = None',1),
 ('        self._remote_ra_registered = True\n        return False',
  '        if self.batch_queue:\n            from remote_ra_live_v2.async_support import drain_before_admission\n            drain_before_admission(self, force=True)\n        self._remote_ra_registered = True\n        return False',1),
 ('        if self.is_running() and self._remote_ra_goal_built():\n            from registered_add_timing',
  '        if self.is_running() and self._remote_ra_goal_built():\n            if self.batch_queue:\n                from remote_ra_live_v2.async_support import drain_before_admission\n                drain_before_admission(self, force=True)\n            from registered_add_timing',1),
 ('        outputs, model_executed = self.step_fn()\n        # Put EngineCoreOutputs',
  '        outputs, model_executed = self.step_fn()\n        if getattr(self, "_remote_ra_dynamic_mode", False):\n            from remote_ra_live_v2.async_support import observe_outputs\n            observe_outputs(self, outputs)\n        # Put EngineCoreOutputs',1),
])
# Bounded checkpoint implementation already qualified on Qwen in the older integration.
async_text=(OLD/'package/source-overlay/remote_ra_live_v2/async_support.py').read_text()
start=async_text.index('    from remote_ra_http.config import load\n',async_text.index('def attach('))
end=async_text.index('    if not runner.use_async_scheduling:',start)
async_text=async_text[:start]+async_text[end:]
# Gemma support needs explicit acceptance-window tests; fail closed until added.
async_text=async_text.replace('    identity = state.assignment.identity\n    groups = owner.backend._groups(state.latest_hybrid)',
 '    require(not getattr(owner.backend, "window", False), "Gemma async window validation pending")\n    identity = state.assignment.identity\n    groups = owner.backend._groups(state.latest_hybrid)',1)
changes['source-overlay/remote_ra_live_v2/async_support.py']=async_text
rewriter=(OLD/'package/source-overlay/remote_ra_live_v2/vllm_rewriter.py').read_text().replace('frequency_penalty=0, seed=0,','frequency_penalty=0, seed=42,')
changes['source-overlay/remote_ra_live_v2/vllm_rewriter.py']=rewriter
change('source-overlay/remote_ra_live_v2/query_selection.py',[
 ("    importlib.import_module('dram_kv.goals.maintained')\n    rewriter = importlib.import_module('dram_kv.goals.rewriter')",
  "    from remote_ra_http.config import load\n    runtime = load().get('goal_rewriter_runtime')\n    if runtime is not None:\n        from .vllm_rewriter import VLLMGoalRewriter\n        return VLLMGoalRewriter(pin, runtime)\n    raise RuntimeError('This integration requires the separately served vLLM Goal rewriter')\n    importlib.import_module('dram_kv.goals.maintained')\n    rewriter = importlib.import_module('dram_kv.goals.rewriter')",1),
])
# Commit only after every replacement was checked. Preserve the complete original bytes.
REV.mkdir(parents=True)
patch=[]
for name,after in changes.items():
    target=P/name
    before=target.read_text() if target.exists() else ''
    original=REV/'original'/name;original.parent.mkdir(parents=True,exist_ok=True);original.write_text(before)
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original/'+name,tofile='adapted/'+name))
    target.write_text(after)
(REV/'changes.patch').write_text(''.join(patch))
(REV/'status.json').write_text(json.dumps({'status':'CPU_AND_GPU_VALIDATION_REQUIRED','files':len(changes),'gemma_async_window':'pending; fail closed','thinking_protocols':'pending','algorithm_files_modified':False},indent=2)+'\n')
print('Applied',len(changes),'files; release is NOT qualified.')
