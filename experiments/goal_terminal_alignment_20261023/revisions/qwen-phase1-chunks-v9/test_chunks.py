import sys, os, unittest, tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
D=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(D/'selective-prefix'),str(D/'selective-prefix/source-overlay')]
import test_stageb_e2e as S
import test_stageb_worker as W
from remote_ra_live_v2.phase1_chunks import aligned_chunks
class Tests(unittest.TestCase):
 def setUp(self):
  self.enterContext(patch.dict(sys.modules,W.vllm_stubs()))
  self.enterContext(patch.object(W,'P',S.P))
  self.enterContext(patch.object(W.Table,'block_size',S.P))
  self.enterContext(patch.dict(os.environ,REMOTE_RA_HTTP_RUNTIME='1'))
 def play(self,limit):
  out=Path(self.enterContext(tempfile.TemporaryDirectory()))
  cfg=S.config('goal_query',out)
  with patch('remote_ra_http.config.load',lambda:cfg),patch('remote_ra_http.evidence.load',lambda:cfg):
   h=S.Harness('goal_query',out)
   h.core.scheduler.scheduler_config=NS(async_scheduling=False)
   h.endpoint.phase1.max_tokens=limit
   rows=[]
   for obs,target in [(None,5),(S.OUT%(0,'q'*2600),14),(S.OUT%(1,'r'*1500),0),(S.OUT%(0,'s'*1200),13)]:
    info=h.turn(obs,target=target)
    rows.append((list(info['tokens']),info['committed']['text'],info['plan']['selected']))
   h.close()
   self.assertEqual(h.pool.get_num_free_blocks(),h.baseline)
   return rows
 def test_equivalence(self):
  self.assertEqual(self.play(4096),self.play(512))
 def test_lease_rejections(self):
  import copy
  from remote_ra_live_v2 import phase1_coordinator as PC
  from remote_ra_live_v2.contract import parse_lease,LiveV2Error
  captured=[]
  def record(raw,**kw):
   if raw.get('chunks'):captured.append((copy.deepcopy(raw),kw))
   return parse_lease(raw,**kw)
  with patch.object(PC,'parse_lease',record):self.play(512)
  self.assertTrue(captured)
  raw,kw=captured[0]
  self.assertGreaterEqual(len(raw['chunks']),3)
  def refused(edit):
   value=copy.deepcopy(raw);edit(value)
   with self.assertRaises(LiveV2Error):parse_lease(value,**kw)
  refused(lambda v:v['chunks'][1].__setitem__(0,v['chunks'][1][0]+1))
  refused(lambda v:v['chunks'].pop())
  refused(lambda v:v['chunks'][1][2][0].__setitem__(-1,v['chunks'][1][2][1][-1]))
  refused(lambda v:v['chunks'][1][2][0].__setitem__(0,v['chunks'][1][2][0][-1]))
 def test_tiling(self):
  for start in (0,5,16,117):
   plan=aligned_chunks(start,start+2700,512,16)
   self.assertEqual(plan[0][0],start)
   self.assertEqual(plan[-1][1],start+2700)
   for a,b in plan:self.assertGreater(b,a);self.assertLessEqual(b-a,512)
   for (a,b),(c,d) in zip(plan,plan[1:]):self.assertEqual(b,c);self.assertEqual(b%16,0)
if __name__=='__main__':unittest.main(verbosity=2)
