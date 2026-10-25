"""Actual emitted allocator/call-shape and failure-cleanup CPU regressions."""
import ast
import asyncio
import contextlib
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE=HERE/'source-overlay'
sys.path.insert(0, str(SOURCE))

def function(path, name, namespace, class_name=None):
    tree = ast.parse((SOURCE/path).read_text('utf-8'))
    nodes = tree.body
    if class_name: nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=[node], type_ignores=[])), namespace)
    return namespace[name]

def native_allocator(manager):
    fn = function('vllm/v1/core/kv_cache_manager.py', '_growing_original_allocate_slots',
        {'RequestStatus': NS(WAITING='WAITING', PREEMPTED='PREEMPTED')}, 'KVCacheManager')
    return MethodType(fn, manager)

def fixture(policy='full', start=0, selected=(), L=0, run=()):
    """Phase 2 admitted at plan['phase2_start'] (e, or 0 at local turn 0)."""
    from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
    calls = []
    block = lambda i: NS(block_id=i)
    attention = NS(block_size=784, req_to_blocks={'r':[block(1), block(2)]})
    spec_a, spec_m = type('FullAttentionSpec', (), {}), type('MambaSpec', (), {})
    manager = NS(kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec_a()), NS(kv_cache_spec=spec_m())]),
        empty_kv_cache_blocks=NS(blocks=((), ())), max_model_len=4096, watermark_blocks=0,
        block_pool=NS(get_num_free_blocks=lambda:0))
    def capacity(**kw): calls.append(kw); return 1
    manager.coordinator=NS(get_num_blocks_to_allocate=capacity)
    manager._growing_original_allocate_slots=native_allocator(manager)
    binding=NS(request='r', session='s', identity='binding', generation='g', tokens=tuple(range(1740)))
    plan=dict(selected=list(selected), phase2_start=start, candidate_end=L, run_pages=list(run),
              sparse=dict(logical_matched_tokens=L, dropped_full_attention_tokens=L-len(selected)*784),
              policy=policy)
    boundary=EngineRpcBoundary.__new__(EngineRpcBoundary)
    boundary.adapter=NS(kv_manager=manager, managers=(attention,), page_tokens=784,
        sessions={'s':NS(policy=policy)})
    boundary.prepared={'r':dict(submitted=True,binding=binding,plan=plan,target={},run={})}
    request=NS(request_id='r', status=NS(name='WAITING'), num_computed_tokens=start,
        num_tokens=1740,num_preemptions=0,num_output_placeholders=0,spec_token_ids=[])
    kwargs=dict(num_new_computed_tokens=0,new_computed_blocks=NS(blocks=((),())),num_lookahead_tokens=0,
        num_external_computed_tokens=0,delay_cache_blocks=False,num_encoder_tokens=0,
        full_sequence_must_fit=True,reserved_blocks=0,has_scheduled_reqs=False,
        external_computed_tokens_by_group=None,computed_tokens_by_group=None,skip_cache=False)
    return NS(boundary=boundary, manager=manager, request=request, binding=binding, plan=plan,
        kwargs=kwargs, calls=calls)

def dispatch(f, count=1568):
    return f.boundary.dispatch_mixed_request(f.binding,f.plan,f.manager,f.request,(count,),f.kwargs)

class AllocationTests(unittest.TestCase):
    def test_four_policies_normal_first_admission_reaches_native_capacity(self):
        for policy in ('full','random50','goal_query','observation_query'):
            with self.subTest(policy=policy):
                f=fixture(policy)
                # No free space: None is the native capacity result, not a refusal.
                self.assertIsNone(dispatch(f))
                self.assertEqual(len(f.calls),1)
                self.assertTrue(f.calls[0]['apply_admission_cap'])
                self.assertEqual(f.calls[0]['num_tokens'],1740)
                self.assertEqual(f.calls[0]['num_tokens_by_group'],(1740,1740))
                self.assertEqual(f.request.num_computed_tokens,0)

    def test_phase2_at_e_uses_compact_groups_and_keeps_logical_cursor(self):
        # L = 1568, A_t = [1] (page 0 dropped), phase 1 computed [C, e = 1600).
        f=fixture('random50', start=1600, selected=[1], L=1568, run=[2])
        self.assertIsNone(dispatch(f,count=140))
        self.assertEqual(f.calls[0]['num_tokens'],1740)
        # FA physical = logical - (L - |A_t|P) = 1600 - 784; Mamba stays logical.
        self.assertEqual(f.calls[0]['num_tokens_by_group'],(956,1740))
        self.assertEqual(f.calls[0]['total_computed_tokens_by_group'],(816,1600))
        self.assertEqual(f.request.num_computed_tokens,1600)

    def test_waiting_must_start_at_phase2_start_and_no_capacity_floor(self):
        for start in (0, 1568, 1601):
            with self.subTest(start=start), contextlib.redirect_stdout(io.StringIO()):
                f=fixture('full', start=1600, selected=[0,1], L=1568, run=[2])
                f.request.num_computed_tokens=start
                with self.assertRaisesRegex(RuntimeError,'num_computed_tokens|num_new_tokens'):
                    dispatch(f,count=1740-start)
                self.assertFalse(f.calls)
        f=fixture('full', start=1600, selected=[0,1], L=1568, run=[2])
        self.assertIsNone(dispatch(f,count=140))
        self.assertFalse(hasattr(f.manager,'_remote_ra_chunk_capacity'))
        self.assertEqual(f.calls[0]['total_computed_tokens_by_group'],(1600,1600))

    def test_true_foreign_prefix_async_speculative_and_reservations_rejected(self):
        cases=[('num_new_computed_tokens',1),('num_external_computed_tokens',784),
            ('num_lookahead_tokens',1),('num_encoder_tokens',1),('delay_cache_blocks',True),
            ('reserved_blocks',1),('full_sequence_must_fit',1),
            ('new_computed_blocks',NS(blocks=((object(),),())))]
        for field,value in cases:
            with self.subTest(field=field), contextlib.redirect_stdout(io.StringIO()) as log:
                f=fixture();f.kwargs[field]=value
                with self.assertRaisesRegex(RuntimeError,field): dispatch(f)
                self.assertIn('value_type',log.getvalue())
                self.assertFalse(f.calls)

    def test_preemption_drafts_placeholders_partial_and_full_sequence_misuse_rejected(self):
        for field,value in [('status',NS(name='PREEMPTED')),('num_preemptions',1),
                            ('spec_token_ids',[999]),('num_output_placeholders',1),
                            ('status',NS(name='RUNNING'))]:
            with self.subTest(field=field), contextlib.redirect_stdout(io.StringIO()):
                f=fixture();setattr(f.request,field,value)
                with self.assertRaises(RuntimeError):dispatch(f)
                self.assertFalse(f.calls)
        for count in (0,1741):
            with self.subTest(count=count), contextlib.redirect_stdout(io.StringIO()):
                f=fixture()
                with self.assertRaisesRegex(RuntimeError,'num_new_tokens'):dispatch(f,count)

    def test_prefill_chunks_split_only_at_the_prompt_end(self):
        # No seal split: any chunk inside [phase2_start, N) is a valid prefill.
        for count in (1,1569,1739,1740):
            with self.subTest(count=count):
                f=fixture()
                self.assertIsNone(dispatch(f,count));self.assertEqual(len(f.calls),1)

    def test_foreign_per_group_prefix_rejected(self):
        for key,value in [('computed_tokens_by_group',(1,0)),('external_computed_tokens_by_group',(0,1))]:
            f=fixture();f.kwargs[key]=value
            with self.assertRaises(RuntimeError):dispatch(f)
            self.assertFalse(f.calls)

class CleanupTests(unittest.IsolatedAsyncioTestCase):
    def owner(self, fail=False):
        from entrypoint_runtime import ShutdownOwner
        from serving_controller import ServingController
        calls=[]
        engine=NS(_remote_ra_dynamic_mode=True,shutdown=lambda **kw:calls.append('native-shutdown'))
        async def registry_close():
            calls.append('registry-close')
            if fail:raise RuntimeError('unknown drain retained')
        registry=NS(llm=engine,_submitted={'r':object()} if fail else {},close=registry_close)
        controller=ServingController.__new__(ServingController)
        controller._thread=threading.get_ident();controller._active={};controller._issued={}
        controller._closing=False;controller._closed=False;controller.registry=registry
        engine._remote_ra_serving_controller=controller
        owner=ShutdownOwner.__new__(ShutdownOwner)
        owner.engine=engine;owner.controller=controller;owner.loop=asyncio.get_running_loop()
        owner.thread=threading.get_ident();owner.task=owner.timeout=None
        owner.verify_identity=lambda: (_ for _ in ()).throw(RuntimeError('native authority revoked'))
        engine._remote_ra_shutdown_owner=owner
        return owner,calls

    async def test_revoked_admission_does_not_block_owned_cleanup_or_duplicate_it(self):
        owner,calls=self.owner()
        await owner.close(0);await owner.close(0)
        self.assertEqual(calls,['registry-close','native-shutdown'])
        self.assertTrue(owner.controller._closed)

    async def test_unknown_drain_retains_ledger_but_native_shutdown_still_runs_once(self):
        owner,calls=self.owner(fail=True)
        for _ in range(2):
            with self.assertRaises(BaseExceptionGroup) as caught:await owner.close(0)
            self.assertIn('unknown drain',str(caught.exception.exceptions[0]))
        self.assertEqual(calls,['registry-close','native-shutdown'])
        self.assertIn('r',owner.controller.registry._submitted)
        self.assertFalse(owner.controller._closed)

    async def test_actual_launcher_awaits_http_shutdown_even_when_engine_cleanup_fails(self):
        calls=[]
        class Server:
            def __init__(self,config):self.started=True;self.should_exit=False
            async def serve(self,**kw):raise asyncio.CancelledError()
            async def shutdown(self):calls.append('http-shutdown-awaited')
        class Config:
            def __init__(self,*a,**kw):pass
            def load(self):pass
        async def watchdog(*a):await asyncio.Future()
        async def join(*a):raise RuntimeError('original engine cleanup failure')
        logger=NS(info=lambda *a:None,info_once=lambda *a:None,warning=lambda *a:None)
        ns=dict(asyncio=asyncio,uvicorn=NS(Server=Server,Config=Config),logger=logger,
            signal=NS(SIGINT=2,SIGTERM=15),H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT=1,
            H11_MAX_HEADER_COUNT_DEFAULT=1,watchdog_loop=watchdog,
            find_process_using_port=lambda port:None,_remote_ra_join_shutdown=join)
        serve=function('vllm/entrypoints/launcher.py','serve_http',ns)
        app=NS(routes=[],state=NS(engine_client=NS()))
        with patch.object(asyncio.get_running_loop(),'add_signal_handler',lambda *a:None):
            with self.assertRaisesRegex(RuntimeError,'original engine cleanup failure'):
                await serve(app,None,port=18000)
        await asyncio.sleep(0)
        self.assertEqual(calls,['http-shutdown-awaited'])

if __name__=='__main__':unittest.main()
