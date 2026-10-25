import ast,asyncio
from pathlib import Path
from types import SimpleNamespace as NS
p=Path(__file__).with_name('timed_environment.py')
node=next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.AsyncFunctionDef) and n.name=='exec');node.decorator_list=[]
ns={'ExecResult':lambda **kw:NS(**kw)}
class Http:
 async def post(self,*a,**kw):
  self.last=kw
  return NS(raise_for_status=lambda:None,json=lambda:{'stdout':'ok','stderr':'','return_code':0})
ns.update(httpx=NS(TimeoutException=TimeoutError,ConnectError=ConnectionError,RemoteProtocolError=ConnectionError),asyncio=asyncio)
mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
async def test():
 c=Http();s=NS(_http_client=c,_server_port=123,_memory_limit_exceeded=False,_resolve_user=lambda user:None,_merge_env=lambda env:env)
 await ns['exec'](s,'true');assert c.last['timeout'] is None and c.last['json']['timeout_sec'] is None
 await ns['exec'](s,'true',timeout_sec=900);assert c.last['timeout']==910 and c.last['json']['timeout_sec']==900
 print('PASS unspecified HTTP timeout defers to Harbor; explicit timeout preserved with10s grace')
asyncio.run(test())
