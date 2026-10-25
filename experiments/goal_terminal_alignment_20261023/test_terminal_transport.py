import asyncio,threading,unittest,sys
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R.parent/'gemma_full_20261017/harness')]
import terminal_proxy as proxy
class Response:
 def __init__(self,x):self.x=x
 def model_dump(self,**kw):return self.x
class FakeBackend:
 def __init__(self):self.local=threading.local();self.resets=[]
 def complete(self,**kw):
  n=getattr(self.local,'n',0)+1;self.local.n=n
  return Response({'n':n,'thread':threading.get_ident()})
 def reset(self):self.resets.append((threading.get_ident(),getattr(self.local,'n',0)));self.local.n=0
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_session_isolation_and_close(self):
  fake=FakeBackend()
  with patch('tebench.backends.create',return_value=fake),patch.object(proxy,'instrument_backend',side_effect=lambda x,p:x):
   t=proxy.Transport({'model':{'name':'test'}},'vanilla',R/'unused')
  body={'messages':[{'role':'user','content':'test'}]}
  first=await asyncio.gather(*(t.complete(str(i),body) for i in range(16)))
  second=await asyncio.gather(*(t.complete(str(i),body) for i in reversed(range(16))))
  self.assertTrue(all(x['n']==1 for x in first));self.assertTrue(all(x['n']==2 for x in second))
  self.assertEqual(len({x['thread'] for x in first}),16)
  with self.assertRaises(proxy.HTTPException):await t.complete('overflow',body)
  for i in range(16):await t.close(str(i))
  self.assertEqual(len(fake.resets),16);self.assertTrue(all(x[1]==2 for x in fake.resets))
  with self.assertRaises(proxy.HTTPException):await t.complete('0',body)
  third=await t.complete('new',body);self.assertEqual(third['n'],1);await t.close('new')
if __name__=='__main__':unittest.main()
