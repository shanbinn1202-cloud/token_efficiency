"""OpenAI chat transport for upstream Terminus-2, with per-trial KV ownership.

No prompt/action/parser changes. One persistent transport thread per live trial
keeps the baseline's thread-local session isolated; vLLM still batches16 sessions.
"""
import asyncio,concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
from contextlib import asynccontextmanager
R=Path(__file__).resolve().parent;G=R.parent/'gemma_full_20261017'
sys.path[:0]=[str(R),str(G/'harness')]
from fastapi import FastAPI,HTTPException,Request
from tebench.config import load
from experiments.benchmarks32_20261004.telemetry import instrument_backend,LOCAL

class Transport:
    def __init__(self,config,method,log):
        self.config=config;self.method=method;self.sessions={};self.closed=set()
        if method=='goal':
            from goal_client import GoalSessionBackend
            release=Path(os.environ['GOAL_RELEASE']);config['goal_runtime']={'release':str(release),'manifest_sha256':hashlib.sha256((release/'runtime-manifest.json').read_bytes()).hexdigest()}
            self.backend=GoalSessionBackend(config)
        else:
            from tebench.backends import create
            self.backend=create(config,method)
        self.backend=instrument_backend(self.backend,log)
    def _request(self,sid,body):
        LOCAL.case=sid
        try:
            result=self.backend.complete(model=self.config['model']['name'],messages=body['messages'],
                max_tokens=body.get('max_tokens',4096),temperature=0)
            return result.model_dump(exclude_none=True)
        finally:LOCAL.case=''
    async def complete(self,sid,body):
        if sid in self.closed:raise HTTPException(409,'Closed terminal session')
        if sid not in self.sessions:
            if len(self.sessions)>=16:raise HTTPException(429,'All16 agent session slots in use')
            self.sessions[sid]=(concurrent.futures.ThreadPoolExecutor(max_workers=1),asyncio.Lock())
        pool,lock=self.sessions[sid]
        async with lock:
            return await asyncio.get_running_loop().run_in_executor(pool,self._request,sid,body)
    async def close(self,sid):
        if sid not in self.sessions:return
        pool,lock=self.sessions[sid]
        async with lock:
            await asyncio.get_running_loop().run_in_executor(pool,self.backend.reset)
            del self.sessions[sid];self.closed.add(sid);pool.shutdown(wait=True)

@asynccontextmanager
async def lifespan(app):
    c=load(os.environ['TERMINAL_CONFIG'])
    app.state.transport=Transport(c,os.environ['TERMINAL_METHOD'],Path(os.environ['TERMINAL_OUTPUT'])/'requests.jsonl')
    yield
    for sid in list(app.state.transport.sessions):await app.state.transport.close(sid)
    app.state.transport.backend.close()
app=FastAPI(lifespan=lifespan)
@app.get('/health')
async def health():return {'ready':True,'method':os.environ['TERMINAL_METHOD']}
@app.get('/v1/models')
async def models():return {'object':'list','data':[{'id':app.state.transport.config['model']['name'],'object':'model','created':0,'owned_by':'local'}]}
@app.post('/v1/chat/completions')
async def complete(request:Request):
    body=await request.json();sid=request.headers.get('x-session-id') or body.get('session_id')
    if not sid:raise HTTPException(400,'Terminus must send its per-trial X-Session-ID')
    if body.get('stream') or body.get('tools'):raise HTTPException(400,'Terminus JSON text transport expects non-streaming text chat')
    if body.get('max_tokens',4096)>4096:raise HTTPException(400,'Output limit is4096')
    if body.get('temperature',0)!=0:raise HTTPException(400,'Aligned greedy temperature is0')
    try:
        return await app.state.transport.complete(sid,body)
    except Exception as exc:
        # Preserve a real model context limit as an agent-budget outcome in
        # Harbor, rather than disguising it as a provider/infrastructure outage.
        from litellm.exceptions import ContextWindowExceededError
        from minisweagent.exceptions import InterruptAgentFlow
        if isinstance(exc,(ContextWindowExceededError,InterruptAgentFlow)):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400,content={'error':{'type':'invalid_request_error','code':'context_length_exceeded','message':'Maximum context length262144 reached; input history is not truncated.'}})
        raise
@app.post('/close-session')
async def close(request:Request):
    body=await request.json();await app.state.transport.close(body['session_id']);return {'closed':True}
if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('TERMINAL_PROXY_PORT','19842')))
