"""Zero-GPU tests execute the shipped OpenAI router with explicit ASGI doubles.

The process/worker substitute is labeled SYNTHETIC and never writes READY.
No torch/vLLM runtime import, listener, installation, or remote action occurs.
"""
import asyncio
from contextlib import contextmanager
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source-overlay'
sys.path.insert(0,str(SOURCE))
from remote_ra_http.config import validate_wire

def engine_process(pipe):
    from remote_ra_live_v2.lifecycle import EngineLifecycle
    class Endpoint:
        def __init__(self):self.calls=[]
        def verify_identity(self):pass
        def frame_identity(self,b):return tuple(b[k] for k in ('request','session','turn','generation'))
        def binding_identity(self,b):return b.ident
        def render_claim(self,b):
            self.calls.append(('render',b['turn'],b['messages'][-1]['role']))
            return object(),NS(ident=self.frame_identity(b),identity=b['request'])
        def reserve_worker(self,r,b):self.calls.append(('worker_prepare',b.identity))
        def enqueue(self,r,b):self.calls.append(('enqueue',b.identity))
        def drain(self,b,operation):
            self.calls.append(('worker_drain',b.identity,operation))
            return dict(drained=True,operation=operation)
        def verify_drain(self,b,o,r):
            if not r['drained']:raise RuntimeError('unknown drain')
    endpoint=Endpoint();life=EngineLifecycle(endpoint)
    try:
        while True:
            command=pipe.recv()
            if command['op']=='close':
                life.disconnect();pipe.send(dict(closed=True,calls=endpoint.calls));return
            if command['op']=='add':
                result=life.admit(command['body']);pipe.send(result)
            elif command['op']=='complete':pipe.send(life.completed(command['request']))
            elif command['op']=='cancel':pipe.send(life.cancel(command['identity']))
    finally:pipe.close()

@contextmanager
def actual_router():
    """Load all decorators and response branches of exact packaged api_router."""
    names={}
    def module(name,**attrs):
        m=ModuleType(name);m.__dict__.update(attrs);names[name]=m;return m
    class Router:
        def post(self,*args,**kwargs):return lambda f:f
        def get(self,*args,**kwargs):return lambda f:f
    class Response:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    class Completion:
        def __init__(self,id):self.id=id
        def model_dump(self):return dict(id=self.id,choices=[])
    module('fastapi',APIRouter=Router,Depends=lambda x:x,FastAPI=object,Request=object)
    module('fastapi.responses',JSONResponse=Response,StreamingResponse=Response)
    module('vllm.entrypoints.openai.chat_completion.batch_serving',OpenAIServingChatBatch=object)
    module('vllm.entrypoints.openai.chat_completion.protocol',BatchChatCompletionRequest=object,
           ChatCompletionRequest=object,ChatCompletionResponse=Completion)
    module('vllm.entrypoints.openai.chat_completion.serving',OpenAIServingChat=object)
    module('vllm.entrypoints.openai.engine.protocol',ErrorResponse=type('ErrorResponse',(),{}))
    module('vllm.entrypoints.serve.utils.api_utils',load_aware_call=lambda f:f,
        validate_json_request=lambda:None,with_cancellation=lambda f:f)
    module('vllm.entrypoints.serve.utils.orca_metrics',metrics_header=lambda x:{})
    module('vllm.logger',init_logger=lambda n:None)
    with patch.dict(sys.modules,names):
        spec=importlib.util.spec_from_file_location('v24_test_actual_api_router',
            SOURCE/'vllm/entrypoints/openai/chat_completion/api_router.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        yield mod,Completion

class WireTests(unittest.TestCase):
    def test_issue_and_observation_contract(self):
        task=dict(instance_id='test',problem_statement='real issue')
        first=dict(messages=[dict(role='user',content='real issue')])
        self.assertEqual(validate_wire(first,task,0),first['messages'])
        for policy in ('full','random50','goal_query','observation_query'):
            with self.assertRaises(RuntimeError):validate_wire(dict(first,policy=policy),task,0)
        with self.assertRaises(RuntimeError):validate_wire(dict(messages=[dict(role='user',content='other')]),task,0)
        later=dict(messages=first['messages']+[dict(role='assistant',content=''),dict(role='tool',content='actual stdout')])
        validate_wire(later,task,1)
        with self.assertRaises(RuntimeError):validate_wire(dict(messages=later['messages']+[dict(role='tool',content='parallel')]),task,1)

class HttpRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe()
        self.child=ctx.Process(target=engine_process,args=(child,));self.child.start();child.close()
    async def asyncTearDown(self):
        if self.child.is_alive():
            self.pipe.send(dict(op='close'));self.cleanup=self.pipe.recv();self.child.join(5)
        if self.child.is_alive():self.child.terminate();self.child.join(5)
        self.pipe.close();self.assertEqual(self.child.exitcode,0)

    async def execute(self,policy,disconnect=False):
        import route_runtime as route
        receive=asyncio.Queue();self.turn=0
        engine=NS(_remote_ra_dynamic_mode=True)
        controller=NS(_issued={},_active={},registry=NS(_capabilities={}),serving=None)
        def authorize(req,session_id,policy):
            controller._issued[id(req)]=NS(request=req,engine_complete=False,capability=object())
        controller.authorize=authorize
        state=NS(engine_client=engine,enable_server_load_tracking=True,server_load_metrics=0)
        with actual_router() as (router,Completion):
            class Handler:
                engine_client=engine
                async def create_chat_completion(inner,request,raw):
                    call=controller._issued[id(request)]
                    body=dict(request=f'{policy}-{self.turn}',session=policy,turn=self.turn,
                        generation=f'g{self.turn}',messages=request.messages)
                    self.pipe.send(dict(op='add',body=body));ack=await asyncio.to_thread(self.pipe.recv)
                    self.assertEqual(ack['state'],'accepted')
                    identity=tuple(body[k] for k in ('request','session','turn','generation'))
                    try:
                        if disconnect:
                            await receive.put(dict(type='http.disconnect'))
                            await asyncio.Future()
                        self.pipe.send(dict(op='complete',request=body['request']))
                        result=await asyncio.to_thread(self.pipe.recv)
                        self.assertEqual(result['state'],'complete');call.engine_complete=True
                        return Completion('chatcmpl-'+body['request'])
                    except asyncio.CancelledError:
                        self.pipe.send(dict(op='cancel',identity=identity))
                        result=await asyncio.to_thread(self.pipe.recv)
                        self.assertEqual(result['state'],'cancelled');raise
            handler=Handler();controller.serving=handler;state.openai_serving_chat=handler
            state._remote_ra_chat_route_binding=route.ContractBinding(handler,controller,policy,policy,lambda:True)
            raw=NS(app=NS(state=state),headers={},receive=receive.get)
            messages=[dict(role='user',content='real issue')]
            with patch.object(route,'require_installable',lambda:None):
                for turn in range(1 if disconnect else 3):
                    self.turn=turn
                    if turn:messages += [dict(role='assistant',content=None,tool_calls=[dict(id=f'call_{turn}',type='function',
                        function=dict(name='bash',arguments=json.dumps({'command':'cat target.py'})))]),
                        dict(role='tool',tool_call_id=f'call_{turn}',content=f'actual stdout {turn}')]
                    req=NS(messages=list(messages))
                    if disconnect:
                        with self.assertRaises(asyncio.CancelledError):await router.create_chat_completion(req,raw)
                    else:
                        result=await router.create_chat_completion(req,raw)
                        self.assertEqual(result.content['id'],f'chatcmpl-{policy}-{turn}')
                    self.assertEqual(state.server_load_metrics,0)
                    self.assertFalse(controller._issued)
    async def test_full(self):await self.execute('full')
    async def test_random50(self):await self.execute('random50')
    async def test_goal(self):await self.execute('goal_query')
    async def test_observation(self):await self.execute('observation_query')
    async def test_disconnect_joins_child_worker(self):await self.execute('full',disconnect=True)

class SourceTests(unittest.TestCase):
    def test_no_frozen_provider_in_factories(self):
        for name in ('live_factory.py','live_frontend.py'):
            raw=(SOURCE/'remote_ra_linux'/name).read_text()
            self.assertNotIn('SmokeContextProvider',raw);self.assertNotIn('smoke_messages(',raw)
    def test_ready_wait_is_background_after_lifespan(self):
        raw=(SOURCE/'remote_ra_http/evidence.py').read_text()
        self.assertIn('while not server.started:',raw)
        self.assertIn('task=asyncio.create_task(ready())',raw)
        raw=(SOURCE/'vllm/entrypoints/launcher.py').read_text()
        self.assertIn('_remote_ra_ready_task = start_ready_task(app, server)',raw)

class ConnectorTests(unittest.TestCase):
    def test_only_owned_requests_bypass_legacy_stores(self):
        from remote_ra_http.connector import isolate_connector
        def unexpected(*a,**kw):raise AssertionError('legacy transfer started')
        connector=NS(_req_status={},has_pending_push_work=lambda:False,
            **{n:unexpected for n in ('on_new_request','get_num_new_matched_tokens',
                                    'update_state_after_alloc','request_finished','build_connector_meta')})
        core=NS(_remote_ra_growing_adapter=NS(owns=lambda rid:rid=='owned'),
                scheduler=NS(connector=NS(connector_scheduler=connector)))
        isolate_connector(core)
        request=NS(request_id='owned');connector.on_new_request(request)
        self.assertEqual(connector.get_num_new_matched_tokens(request,0),(0,False))
        connector.update_state_after_alloc(request,object(),0)
        self.assertEqual(connector.request_finished(request),(False,None))
        with self.assertRaises(RuntimeError):connector.on_new_request(NS(request_id='foreign'))
        with self.assertRaises(RuntimeError):connector.update_state_after_alloc(request,object(),1)

class MeasurementTests(unittest.TestCase):
    def test_count_only_joined_actual_copy_bytes(self):
        from remote_ra_http.evidence import measured_command
        io=NS(views=[NS(group=0,page_bytes=16)])
        backend=NS(io=io,owner=NS(claims={(0,1):('r',1)}))
        def joined(b,q):
            io._http_h2d_issued=io._http_h2d_joined=32
            return dict(operation='query_prepare',drained=True)
        r=measured_command(backend,'query_prepare',joined,None)
        self.assertEqual(r['http_measurement']['after']['h2d_joined'],32)
        self.assertEqual(r['http_measurement']['active_owned_kv_bytes'],16)
        def unknown(b,q):
            io._http_h2d_issued=48
            return dict(operation='query_prepare')
        with self.assertRaises(RuntimeError):measured_command(backend,'query_prepare',unknown,None)

class DiagnosticsTests(unittest.TestCase):
    def test_failure_chain_keeps_engine_rejection(self):
        # vLLM raises an empty EngineGenerateError from the registered-add failure.
        from remote_ra_http.evidence import failure_chain
        EngineGenerateError=type('EngineGenerateError',(Exception,),{})
        rejected=RuntimeError('registered-add rejected: Abort: renderer tokens differ')
        try:
            try:raise ExceptionGroup('registered-add submission and cancellation failed',
                                     [rejected,RuntimeError('cleanup')])
            except Exception as inner:raise EngineGenerateError() from inner
        except Exception as outer:chain=failure_chain(outer)
        self.assertEqual(chain[0],dict(type='EngineGenerateError',message=''))
        self.assertIn(dict(type='RuntimeError',message=str(rejected)),chain)

    def test_engine_rejection_is_logged(self):
        import contextlib,io
        from registered_add_protocol import RegisteredAddDispatcher
        out=io.StringIO()
        try:raise RuntimeError('boom')
        except RuntimeError as exc:
            with contextlib.redirect_stdout(out):
                RegisteredAddDispatcher._log_rejection(NS(request_id='r1'),'RuntimeError: boom',exc)
        tag,raw=out.getvalue().split(' ',1)
        line=json.loads(raw)
        self.assertEqual((tag,line['phase'],line['error']),
                         ('REMOTE_RA_REGISTERED_ADD','engine_rejected','RuntimeError: boom'))
        self.assertIn("RuntimeError: boom",line['traceback'])

class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_does_not_block_lifespan(self):
        from remote_ra_http.evidence import start_ready_task
        calls=[];server=NS(started=False,should_exit=False)
        app=NS(state=NS(_remote_ra_http_evidence=NS(write_ready=lambda a,s:calls.append('ready'))))
        task=start_ready_task(app,server)
        await asyncio.sleep(.03);self.assertEqual(calls,[])
        server.started=True;await asyncio.wait_for(task,.2)
        self.assertEqual(calls,['ready'])

if __name__=='__main__':unittest.main()
