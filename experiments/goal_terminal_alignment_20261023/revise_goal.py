"""Async acceptance boundary and capacity changes, isolated integration only."""
from pathlib import Path
import difflib,json
R=Path(__file__).resolve().parent;P=R/'selective-prefix';REV=R/'revisions/async-window-v2'
if REV.exists():raise RuntimeError('revision already applied')
changes={}
def change(name,old,new,count=1):
    text=changes.get(name,(P/name).read_text())
    if text.count(old)!=count:raise RuntimeError((name,old[:80],text.count(old),count))
    changes[name]=text.replace(old,new)
change('serve.py',"protocol='mswea-xml',kv_cache_gib=8,","protocol='mswea-xml',kv_cache_gib=25,")
change('serve.py',"max_running=None,cuda_graphs='off',max_tokens=1536,","max_running=None,cuda_graphs='full',max_tokens=4096,")
change('source-overlay/remote_ra_live_v2/sessions.py','MAX_TASKS = 64','MAX_TASKS = 1024')
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py','MAX_SLOTS = 64','MAX_SLOTS = 1024')
for name in ['phase1_coordinator.py','window_phase1.py']:
    change('source-overlay/remote_ra_live_v2/'+name,'            self.recover(binding.request)\n','            self.recover(binding.request)\n            from .async_support import after_probe\n            after_probe(self.core)\n',2 if name=='phase1_coordinator.py' else 1)
change('source-overlay/remote_ra_live_v2/window_worker.py','    runner._remote_ra_live_v2_bridge = bridge\n','    runner._remote_ra_live_v2_bridge = bridge\n    from .async_support import attach as attach_async\n    attach_async(runner)\n')
change('source-overlay/remote_ra_live_v2/async_support.py','    require(not getattr(owner.backend, "window", False), "Gemma async window validation pending")\n','''    if getattr(owner.backend, "window", False):
        # Sliding-window KV is append-only within its retained blocks. There is
        # no recurrent tensor to restore. Native removal retains the two-token
        # async lag so the accepted cursor's window remains resident.
        owner.final_window(state.assignment.identity, state.cursor, state.allocation_generation)
        snapshots = runner._remote_ra_async_checkpoints.setdefault(state.assignment.identity.request, {})
        snapshots[state.cursor] = (state.assignment.identity, state.allocation_generation, None)
        while len(snapshots) > runner._remote_ra_async_depth:
            snapshots.pop(min(snapshots))
        return
''')
change('source-overlay/remote_ra_live_v2/async_support.py',"        groups = owner.backend._groups(state.latest_hybrid)\n        views = owner.io.validate(groups, 'hybrid')\n",'''        if getattr(owner.backend, 'window', False):
            owner.final_window(identity, end, state.allocation_generation)
            views, groups = (), {}
        else:
            groups = owner.backend._groups(state.latest_hybrid)
            views = owner.io.validate(groups, 'hybrid')
''')
change('source-overlay/remote_ra_live_v2/async_support.py',"    require(cfg.parallel_config.tensor_parallel_size == 1 and", "    require(cfg.max_concurrent_batches <= 2, 'async retention is qualified for at most two native batches')\n    require(cfg.parallel_config.tensor_parallel_size == 1 and")
change('source-overlay/remote_ra_live_v2/kv_layout.py','return _cdiv(self.window_tokens, self.page_tokens) + 1','''# One extra physical block protects the accepted EOS window while native
        # async decoding can be at most two tokens ahead. Attention W is unchanged.
        return _cdiv(self.window_tokens, self.page_tokens) + 2''')
change('source-overlay/vllm/v1/core/single_type_kv_cache_manager.py','        self._null_block = block_pool.null_block\n','''        self._null_block = block_pool.null_block
        import os
        self._selective_async_lag = (2 if os.environ.get("REMOTE_RA_HTTP_RUNTIME") == "1"
                                    and type(kv_cache_spec).__name__ == "SlidingWindowSpec" else 0)
''')
change('source-overlay/vllm/v1/core/single_type_kv_cache_manager.py','num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)','''num_skipped_tokens = self.get_num_skipped_tokens(
            max(0, total_computed_tokens - self._selective_async_lag))''',2)
# Gemma thinking on: generation headers differ between a new model turn and a tool continuation.
change('serve.py',"        need(kv_session_thinking=='off','the Gemma 4 kv-session serves --kv-session-thinking off only: thinking off, '\n             'as decided for MemoryAgentBench (the header <|turn>model\\\\n<|channel>thought\\\\n<channel|>)')\n",'')
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py',"GEMMA4_THINKING = 'off'","GEMMA4_THINKING = 'on'")
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py',"GEMMA4_HEADER = '<|turn>model\\n<|channel>thought\\n<channel|>'","GEMMA4_HEADER = '<|turn>model\\n'\nGEMMA4_TOOL_HEADER = '<|channel>thought\\n'")
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py','flag = dict(enable_thinking=False)','flag = dict(enable_thinking=True)')
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py','with_header == without and without.endswith(GEMMA4_TOOL_CLOSE)','with_header == without + GEMMA4_TOOL_HEADER and without.endswith(GEMMA4_TOOL_CLOSE)')
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py','def __init__(self, *, header, im_start, vocab, decode, stop_ids, tools=None):','def __init__(self, *, header, im_start, vocab, decode, stop_ids, tools=None, tool_header=None):')
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py','        self.header, self.im_start, self.vocab = header, im_start, vocab','        self.tool_header = tool_header\n        self.header, self.im_start, self.vocab = header, im_start, vocab')
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py','        H = len(self.header)\n        if self.tools is not None', '        H = len(self.header)\n        if self.tool_header and tuple(tokens[-len(self.tool_header):]) == self.tool_header:\n            return len(self.tool_header)\n        if self.tools is not None')
change('source-overlay/remote_ra_live_v2/kv_session_protocol.py','        if self.tools is not None and request.header_tokens == 0 and not request.first and self.turn > 0:', '''        if (self.tool_header and not request.first and self.turn > 0
                and request.header_tokens == len(self.tool_header)
                and ids[-len(self.tool_header):] == self.tool_header):
            if ids[-len(self.tool_header)-1] != self.tools[1]:
                fail('HEADER_MISMATCH', 'thinking continuation must follow the tool response close')
        elif self.tools is not None and request.header_tokens == 0 and not request.first and self.turn > 0:''')
# Renderer factory passes the pinned alternate header; Qwen path stays unchanged.
name='source-overlay/gemma4_renderer.py';text=(P/name).read_text()
old='tools=tools)'
change(name,old,"tools=tools, tool_header=tuple(self.encode_exact(kv.GEMMA4_TOOL_HEADER)))")
REV.mkdir(parents=True);patch=[]
for name,after in changes.items():
    before=(P/name).read_text();orig=REV/'original'/name;orig.parent.mkdir(parents=True,exist_ok=True);orig.write_text(before)
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='v1/'+name,tofile='v2/'+name));(P/name).write_text(after)
(REV/'changes.patch').write_text(''.join(patch))
(REV/'status.json').write_text(json.dumps({'status':'CPU_AND_GPU_VALIDATION_REQUIRED','physical_window_lag_tokens':2,'attention_window_unchanged':True,'task_slots':1024,'active_sessions':16,'thinking_transport':'kv-session; alternate Gemma tool continuation header'},indent=2)+'\n')
print('applied',len(changes))
