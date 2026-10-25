from pathlib import Path
import sys,unittest,copy
from types import SimpleNamespace as NS
from unittest.mock import patch
D=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(D/'selective-prefix'),str(D/'selective-prefix/source-overlay')]
import test_gemma_e2e as GE
import test_gemma_chunked as C
from remote_ra_live_v2.contract import parse_lease,LiveV2Error
original=GE.Harness.__init__
def fixture(self,*a,**kw):
 original(self,*a,**kw)
 self.core.scheduler.scheduler_config=NS(async_scheduling=False)
class Tests(C.ChunkedEndToEndTests):
 def play(self,policy,chunk=None,forward=None,leases=None,turns=5):
  recorded=leases if leases is not None else []
  with patch.object(GE,'GOAL',forward or 1024):
   result=super().play(policy,chunk=chunk,forward=forward,leases=recorded,turns=turns)
  if policy=='goal_query':
   goals=[r['goal'] for r in recorded if r.get('goal')]
   self.assertTrue(goals)
   if forward==4096:self.assertTrue(all(not g.get('chunks') for g in goals))
   else:self.assertTrue(any(len(g.get('chunks',[]))>=2 for g in goals))
  return result
 def test_goal_chunk_lease_rejections(self):
  leases=[];self.play('goal_query',chunk=256,leases=leases)
  raw=next(r for r in leases if (r.get('goal') or {}).get('chunks'))
  layout=GE.window_layout('goal_query',3)
  parse_lease(raw,attention_groups=(GE.FA,),layout=layout)
  def refused(edit):
   v=copy.deepcopy(raw);edit(v['goal'])
   with self.assertRaises(LiveV2Error):parse_lease(v,attention_groups=(GE.FA,),layout=layout)
  refused(lambda g:g['chunks'][1].__setitem__(0,g['chunks'][1][0]+1))
  refused(lambda g:g['chunks'].pop())
  refused(lambda g:g['chunks'][1][2][0].__setitem__(-1,g['chunks'][1][2][1][-1]))
  refused(lambda g:g['owned'].pop())
with patch.object(GE.Harness,'__init__',fixture),patch('remote_ra_live_v2.selection_contract.goal_query_text',lambda goal,variant:'Goal full query. '+('useful goal statement. '*100)):
 suite=(unittest.defaultTestLoader.loadTestsFromName(sys.argv[1],module=sys.modules[__name__]) if len(sys.argv)>1 else unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 sys.exit(not result.wasSuccessful())
