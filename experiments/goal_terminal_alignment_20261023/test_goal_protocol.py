"""Real pinned-tokenizer transcript tests, independent of model inference."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'selective-prefix/source-overlay'))
from remote_ra_live_v2 import kv_session_protocol as kv

class GemmaThinking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tk=AutoTokenizer.from_pretrained('/projects/bcjw/ssun4/models/gemma-4-31B-it-842da379',local_files_only=True)
    def render(self,template,messages,kwargs):
        return self.tk.apply_chat_template(messages,tokenize=False,**kwargs)
    def test_generation_and_tool_headers_match_real_template(self):
        self.assertEqual(kv.check_gemma4_header(self.render,None),'<|turn>model\n')
    def test_alternate_thinking_header_survives_turn_boundary(self):
        enc=lambda s:tuple(self.tk.encode(s,add_special_tokens=False))
        turn=self.tk.convert_tokens_to_ids('<|turn>')
        toolopen=self.tk.convert_tokens_to_ids('<|tool_response>')
        toolclose=self.tk.convert_tokens_to_ids('<tool_response|>')
        tr=kv.KvTranscript(header=enc(kv.GEMMA4_HEADER),im_start=turn,vocab=len(self.tk),decode=lambda ids:self.tk.decode(ids),stop_ids=frozenset([1,50,106]),tools=(toolopen,toolclose),tool_header=enc(kv.GEMMA4_TOOL_HEADER))
        tr.turns=[{}];tr.initial_length=10;tr.stream=(42,toolopen)
        ids=(toolopen,)+enc('tool result')+(toolclose,)+enc(kv.GEMMA4_TOOL_HEADER)
        req=NS(input_ids=ids,header_tokens=len(enc(kv.GEMMA4_TOOL_HEADER)),first=False,prefix_tokens=10,boundary_tokens=1)
        tr.check(req)
        self.assertEqual(tr.phase2_suffix(ids),req.header_tokens)
        bad=NS(**vars(req));bad.input_ids=ids[:-1]+(42,)
        with self.assertRaises(kv.KvSessionError):tr.check(bad)

class WindowRetention(unittest.TestCase):
    def test_two_ahead_keeps_every_accepted_window_block(self):
        from remote_ra_live_v2.sliding_window import kept_blocks
        from remote_ra_live_v2.kv_layout import WindowLayout
        for size in [16,32]:
            layout=WindowLayout(size,1024,16,4096)
            for end in range(1000,1100):
                first,last=kept_blocks(end,1024,size)
                for ahead in [0,1,2]:
                    retained_first,_=kept_blocks(max(0,end+ahead-2),1024,size)
                    _,retained_last=kept_blocks(end+ahead,1024,size)
                    self.assertLessEqual(retained_first,first)
                    self.assertGreaterEqual(retained_last,last)
                    self.assertLessEqual(retained_last-retained_first,layout.window_blocks)
    def test_window_rollback_never_copies_gdn(self):
        from remote_ra_live_v2.async_support import checkpoint,accepted_boundary
        from unittest.mock import patch
        runner=NS(_remote_ra_async_ready=True,_remote_ra_async_depth=3,_remote_ra_async_checkpoints={})
        checked=[]
        owner=NS(runner=runner,backend=NS(window=True,latest={}),io=NS(drain=lambda:None),final_window=lambda ident,end,gen:checked.append(end))
        state=NS(assignment=NS(identity=NS(request='a'),page_tokens=16,prompt_tokens=1),allocation_generation=1,cursor=31,latest_state_step=31,active=None,host_executing=False,page_vectors={0:'keep'},receipts=[{'body':{'end':31}}])
        checkpoint(owner,state);state.cursor=32;checkpoint(owner,state)
        state.page_vectors[1]='drop'
        with patch('remote_ra_linux.native_authority.event'):accepted_boundary(owner,state,31)
        self.assertEqual(state.cursor,31);self.assertEqual(state.page_vectors,{0:'keep'});self.assertEqual(checked,[31,32,31])
if __name__=='__main__':unittest.main()
