import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch

sys.path.insert(0, str(Path(__file__).parent/'selective-prefix/source-overlay'))
from remote_ra_live_v2.async_support import checkpoint, accepted_boundary
from remote_ra_live_v2.chunk_protocol import validate_interval


class AsyncBoundaryTests(unittest.TestCase):
    def owner(self):
        tensor = torch.zeros(4, 3)
        view = NS(name='state', group=0, tensors=(tensor,), axes=(0,))
        io = NS(torch=torch, validate=lambda groups, kind:(view,),
                _slice=lambda view,tensor,axis,bid:tensor[bid], drain=lambda:None)
        runner = NS(_remote_ra_async_ready=True, _remote_ra_async_depth=3,
                    _remote_ra_async_checkpoints={})
        owner = NS(runner=runner, io=io,
                   backend=NS(_groups=lambda slots:{0:slots}, latest={}))
        def state(rid, bid):
            return NS(assignment=NS(identity=NS(request=rid),page_tokens=16,prompt_tokens=16),
                      allocation_generation=1,latest_hybrid=(bid,),cursor=16,active=None,
                      host_executing=False,latest_state_step=16,prompt_complete=True,
                      page_vectors={0:'committed'},receipts=[{'body':{'end':16}}])
        return owner,tensor,state

    def test_async_extra_step_restores_exact_carry_and_excludes_new_page(self):
        owner, tensor, state_factory = self.owner()
        state = state_factory('a',1)
        state.cursor=31; tensor[1].fill_(31); checkpoint(owner,state)
        state.cursor=32; tensor[1].fill_(32); checkpoint(owner,state)
        state.page_vectors[1]='must_not_commit'
        state.receipts += [{'body':{'end':31}},{'body':{'end':32}}]
        # Logging is orthogonal to the CPU physical-state test.
        from unittest.mock import patch
        with patch('remote_ra_linux.native_authority.event'):
            accepted_boundary(owner,state,31)
        self.assertTrue(torch.equal(tensor[1],torch.full((3,),31.)))
        self.assertEqual(state.cursor,31)
        self.assertEqual(set(state.page_vectors),{0})
        self.assertEqual(state.receipts[-1]['body']['end'],31)

    def test_other_request_checkpoint_and_state_survive(self):
        owner,tensor,make = self.owner()
        a,b=make('a',1),make('b',2)
        tensor[1].fill_(1);checkpoint(owner,a)
        tensor[2].fill_(2);checkpoint(owner,b)
        a.cursor=17;tensor[1].fill_(7);checkpoint(owner,a)
        from unittest.mock import patch
        with patch('remote_ra_linux.native_authority.event'):
            accepted_boundary(owner,a,16)
        self.assertTrue(torch.equal(tensor[2],torch.full((3,),2.)))
        self.assertIn('b',owner.runner._remote_ra_async_checkpoints)

    def test_missing_boundary_fails_without_mutating_state(self):
        owner,tensor,make=self.owner();state=make('a',1)
        state.cursor=20;tensor[1].fill_(20);checkpoint(owner,state)
        with self.assertRaisesRegex(Exception,'no same-generation'):
            accepted_boundary(owner,state,19)
        self.assertEqual(state.cursor,20)
        self.assertTrue(torch.equal(tensor[1],torch.full((3,),20.)))

    def test_checkpoint_ring_reuses_storage_and_keeps_last_three_exact_states(self):
        owner,tensor,make=self.owner();state=make('a',1);pointers=set()
        for cursor in range(16,46):
            state.cursor=cursor;tensor[1].fill_(cursor);checkpoint(owner,state)
            snapshots=owner.runner._remote_ra_async_checkpoints['a']
            self.assertEqual(sorted(snapshots),list(range(max(16,cursor-2),cursor+1)))
            for end,(_,_,payload) in snapshots.items():
                item=payload['state'][0][0];pointers.add(item.data_ptr())
                self.assertTrue(torch.equal(item,torch.full((3,),float(end))))
        self.assertEqual(len(pointers),3)
        from unittest.mock import patch
        with patch('remote_ra_linux.native_authority.event'):
            accepted_boundary(owner,state,43)
        self.assertTrue(torch.equal(tensor[1],torch.full((3,),43.)))

    def test_placeholder_decode_keeps_original_logical_interval(self):
        req=NS(request_id='r',num_computed_tokens=17,num_tokens=17,num_output_placeholders=1)
        binding=NS(request='r',tokens=tuple(range(16)),identity='binding',generation=1)
        result=validate_interval({'scheduled_cursor':17},req,binding,{'phase2_start':0},1)
        self.assertEqual((result['start'],result['end']),(17,18))

class TaskCloseTests(unittest.TestCase):
    def test_close_isolated_idempotent_and_busy_refusal(self):
        from remote_ra_live_v2.async_support import finish_task
        adapter=NS(sessions={'a':NS(pending=None),'b':NS(pending='active')},requests={})
        closed=[]
        authority=NS(ids=('a','b'),namespaces={'a':'namespace-a','b':'namespace-b'},
                     adapter=adapter,verify=lambda:None)
        def close(sid,namespace):
            closed.append((sid,namespace))
            adapter.sessions.pop(sid)
        authority._close=close
        core=NS(batch_queue=None,_remote_ra_live_session_authority=authority)
        self.assertTrue(finish_task(core,0)['closed'])
        self.assertTrue(finish_task(core,0)['already_closed'])
        self.assertEqual(closed,[('a','namespace-a')])
        self.assertIn('b',adapter.sessions)
        with self.assertRaises(Exception): finish_task(core,1)
        self.assertIn('b',adapter.sessions)

class ProbeResumeTests(unittest.TestCase):
    def test_probe_consumes_then_clears_previous_sample_batch(self):
        import ast
        import types
        source=Path(__file__).parent/'selective-prefix/source-overlay/vllm/v1/worker/gpu_input_batch.py'
        method=next(node for node in ast.walk(ast.parse(source.read_text()))
                    if isinstance(node,ast.FunctionDef) and node.name=='update_async_output_token_ids')
        namespace={}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        tokens=[[-1]]
        batch=NS(sampling_metadata=NS(output_token_ids=tokens),
            sampled_token_ids_cpu=torch.tensor([[7]]),prev_req_id_to_index={'live':0},
            req_ids=['live'],async_copy_ready_event=NS(synchronize=lambda:None),
            prev_sampled_token_ids=torch.tensor([[7]]))
        batch.update_async_output_token_ids=types.MethodType(namespace['update_async_output_token_ids'],batch)
        from remote_ra_live_v2.async_support import before_probe
        before_probe(NS(_remote_ra_async_ready=True,input_batch=batch))
        self.assertEqual(tokens,[[7]])
        self.assertIsNone(batch.sampled_token_ids_cpu)
        self.assertIsNone(batch.prev_req_id_to_index)
        # Native sample_tokens calls this after the probe, before any new sample.
        batch.update_async_output_token_ids()
        self.assertEqual(tokens,[[7]])

    def test_native_scheduler_resends_tokens_after_private_probe(self):
        # Exercise the actual native serializer responsible for the GPU KeyError.
        import ast
        import __future__
        import itertools
        source=Path(__file__).parent/'selective-prefix/source-overlay/vllm/v1/core/sched/scheduler.py'
        tree=ast.parse(source.read_text())
        method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
                    and node.name=='_make_cached_request_data')
        namespace={'itertools':itertools,'CachedRequestData':NS}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec',
             flags=__future__.annotations.compiler_flag),namespace)
        req=NS(request_id='live',all_token_ids=[101,102,103],num_computed_tokens=2,
               num_output_tokens=1,num_output_placeholders=0)
        scheduler=NS(use_pp=False,use_v2_model_runner=False,running=[req],
            prev_step_scheduled_req_ids={'live'},scheduler_config=NS(async_scheduling=True))
        serialize=namespace['_make_cached_request_data']
        blocks={'live':NS(get_block_ids=lambda **kwargs:None)}
        before=serialize(scheduler,[req],[],{'live':1},{},blocks)
        self.assertNotIn('live',before.all_token_ids)
        from remote_ra_live_v2.async_support import after_probe
        after_probe(NS(scheduler=scheduler,batch_queue=[]))
        after=serialize(scheduler,[req],[],{'live':1},{},blocks)
        self.assertEqual(after.all_token_ids['live'],[101,102,103])
        self.assertEqual(after.num_computed_tokens,[2])
        self.assertEqual(after.resumed_req_ids,set())

if __name__=='__main__':unittest.main()
