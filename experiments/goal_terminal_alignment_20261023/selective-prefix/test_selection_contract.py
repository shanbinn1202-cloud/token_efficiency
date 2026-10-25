"""Online Selector V2 contract on 784-token Pages against upstream reference code; CPU only."""
from hashlib import sha1, sha256
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source-overlay'))
import torch
from remote_ra_live_v2 import selection_contract as contract
from remote_ra_live_v2.query_selection import normalized,rewriter_argv
from remote_ra_live_v2.vector_space import resolve,pre_final_norm
from remote_ra_native_growing.contract import select_pages
import serve


def upstream_projection(hidden_size, dimension, seed):
    # Verbatim algorithm of serve_online_selector_v2._projection (a70701e).
    generator=torch.Generator(device='cpu');generator.manual_seed(seed)
    values=torch.randint(0,2,(dimension,hidden_size),generator=generator,dtype=torch.int8).to(torch.float32)
    values.mul_(2.0).sub_(1.0).mul_(1.0/math.sqrt(dimension))
    return values,sha256(values.contiguous().numpy().tobytes()).hexdigest()


def upstream_compact(hidden, projection):
    # Verbatim algorithm of OnlineSelectorSession._compact_last (a70701e).
    vector=torch.nn.functional.linear(hidden.float(),projection)
    vector=torch.nn.functional.normalize(vector,p=2,dim=-1)
    scale=vector.abs().max()/127.0
    values=torch.clamp(torch.round(vector/scale),-127,127).to(torch.int8)
    return tuple(int(v) for v in values.tolist()),float(scale)


class ContractArithmeticTests(unittest.TestCase):
    def test_upstream_values_and_784_conversion(self):
        c=contract.default_contract()
        self.assertEqual((c['switch_threshold_tokens'],c['active_ratio'],c['recent_turns'],
                          c['recent_quota_pages'],c['projection_dimension'],c['projection_seed'],
                          c['entity_drift_steps'],c['goal_query_variant'],c['goal_rewriter']),
                         (8192,.4,2,2,128,20260904,3,'full_goal',None))
        # At upstream 16-token Pages the conversion is the identity.
        self.assertEqual(contract.default_contract(16)['recent_quota_pages'],64)
        c16=contract.default_contract(16)
        for n in (513,700,1452,4000):
            for mandatory in (1,62,700):
                upstream=max(16,mandatory*16,math.floor(n*16*.4))
                self.assertEqual(contract.capacity_pages(n,mandatory,c16),min(n,upstream//16))

    def test_threshold_capacity_mandatory_cover_and_budget(self):
        c=contract.default_contract()
        self.assertFalse(contract.selection_active(83,0,c))
        self.assertFalse(contract.selection_active(10,3,c))   # 7,840 <= 8,192
        self.assertTrue(contract.selection_active(11,3,c))    # 8,624 >  8,192
        self.assertEqual(contract.capacity_pages(11,2,c),4)   # floor(3449.6)//784
        self.assertEqual(contract.capacity_pages(83,3,c),33)
        self.assertEqual(contract.capacity_pages(12,9,c),9)   # mandatory is never infeasible
        self.assertEqual(contract.mandatory_pages(1000,20,c),2)  # cover, not floor
        self.assertEqual(contract.mandatory_pages(784,20,c),1)
        fixed=contract.default_contract(active_token_budget=8192,active_ratio=None)
        self.assertEqual(contract.capacity_pages(83,3,fixed),10)
        with self.assertRaises(Exception):contract.default_contract(active_token_budget=8192)
        with self.assertRaises(Exception):contract.default_contract(active_ratio=1.0)

    def test_page_turns_follow_upstream_last_token_ownership(self):
        boundaries=[5000,9000,20000]
        turns=[contract.page_turn(p,boundaries,784) for p in range(25)]
        self.assertEqual(turns[:6],[1]*6)                      # last token 4703 < 5000
        self.assertEqual(turns[6:11],[2]*5)                     # 5487..8623 < 9000
        self.assertEqual(turns[11:],[3]*14)

    def test_goal_query_text_matches_upstream_renderer(self):
        retrieval=dict(description='fix',files=[f'f{i}.py' for i in range(10)],symbols=['s'],tests=[],
                       errors=[],evidence_needed=[],phase_hint='exploration',status='active',constraints=[])
        goal=NS(retrieval_goal=NS(**retrieval))
        with patch.object(contract,'asdict',side_effect=lambda value:dict(vars(value))):
            full=contract.goal_query_text(goal,'full_goal')
            cap=contract.goal_query_text(goal,'candidate_entity_cap8')
        expected='<retrieval_query>\n'+json.dumps({'retrieval_goal':retrieval},ensure_ascii=False,
            sort_keys=True,separators=(',',':'))+'\n</retrieval_query>'
        self.assertEqual(full,expected)
        self.assertEqual(json.loads(cap.split('\n')[1])['retrieval_goal']['files'],retrieval['files'][-8:])
        self.assertEqual(len(retrieval['files']),10)          # maintained Goal not mutated


class VectorSpaceTests(unittest.TestCase):
    def test_projection_is_the_upstream_cohort_vector_space(self):
        c=contract.default_contract()
        matrix,value=contract.projection(5120,c)
        self.assertEqual(value,'20c2e1cf4463ed9407a9f9af0a7794fe044c02c0da369128681270a7427c87aa')
        self.assertEqual(value,upstream_projection(5120,128,20260904)[1])
        hidden=torch.randn(5120,generator=torch.Generator().manual_seed(7))
        from dram_kv.page_vectors import CompactVector
        ours=contract.compact(normalized(hidden.tolist()),matrix,CompactVector)
        values,scale=upstream_compact(hidden,matrix)
        self.assertEqual(ours.values,values)
        self.assertAlmostEqual(ours.scale,scale,places=6)
        self.assertEqual(ours.dtype,'int8')

    def test_pre_final_norm_recovers_last_decoder_layer_direction(self):
        class GemmaRMSNorm(torch.nn.Module):
            def __init__(self,d):
                super().__init__();self.weight=torch.nn.Parameter(torch.randn(d)*.3)
        class Stack(torch.nn.Module):
            def __init__(self,d):
                super().__init__();self.embed_tokens=torch.nn.Embedding(4,d)
                self.layers=torch.nn.ModuleList([torch.nn.Linear(d,d)]);self.norm=GemmaRMSNorm(d)
        class Vision(torch.nn.Module):
            def __init__(self,d):
                super().__init__();self.blocks=torch.nn.ModuleList();self.norm=GemmaRMSNorm(d)
        model=torch.nn.Module();model.visual=Vision(32);model.language_model=torch.nn.Module()
        model.language_model.model=Stack(32)
        info,gain=resolve(model)
        self.assertEqual((info['module'],info['kind'],info['decoder_layers']),('language_model.model.norm','GemmaRMSNorm',1))
        runner=NS(_remote_ra_final_norm_gain=gain)
        x=torch.randn(5,32)
        weight=model.language_model.model.norm.weight.detach()
        post=(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+weight)).to(torch.bfloat16)
        recovered=pre_final_norm(runner,post)
        cosine=torch.nn.functional.cosine_similarity(recovered,x,dim=-1)
        self.assertTrue(bool((cosine>0.999).all()))
        naive=torch.nn.functional.cosine_similarity(post.float(),x,dim=-1)
        self.assertTrue(bool((naive<0.999).any()))
        model.second=Stack(32)
        with self.assertRaisesRegex(Exception,'exactly one decoder stack'):resolve(model)


class SelectPagesTests(unittest.TestCase):
    def test_select_pages_query_decision_validation(self):
        with self.assertRaises(Exception):select_pages('goal_query',5,0,'p',(0,1,2,3,4),2)
        with self.assertRaises(Exception):select_pages('goal_query',5,2,'p',(0,1,2,3,4),6)
        with self.assertRaises(Exception):select_pages('goal_query',5,2,'p',(0,1,2,3,3),2)
        self.assertEqual(select_pages('observation_query',5,2,'p',(4,1,0,2,3),2),((1,4),None))
        self.assertEqual(select_pages('goal_query',5,0,'p',(0,1,2,3,4),5),((0,1,2,3,4),'all_pages_active'))
        self.assertEqual(select_pages('full',3,2,'p'),((0,1,2),'full_reference'))


class RewriterAndLaunchTests(unittest.TestCase):
    def test_vendored_rewriter_is_the_upstream_blob(self):
        raw=(ROOT/'source-overlay/dram_kv_scripts/qwen_goal_rewriter.py').read_bytes()
        self.assertEqual(sha1(b'blob %d\x00'%len(raw)+raw).hexdigest(),'4c426eeca888605af1defc0a1b325c7d3b22ee90')

    def test_rewriter_pin_and_isolated_command(self):
        with TemporaryDirectory() as directory:
            model=Path(directory)/'qwen3-0.6b';model.mkdir();(model/'config.json').write_bytes(b'{"a":1}')
            pin=contract.rewriter_pin(model)
            c=contract.default_contract(goal_rewriter=pin)
            self.assertEqual(c['goal_rewriter']['config_sha256'],sha256(b'{"a":1}').hexdigest())
            argv=rewriter_argv(pin,ROOT/'source-overlay','/site','/py')
            self.assertEqual(argv[:6],('/py','-I','-S','-B','-u','-c'))
            self.assertTrue(argv[9].endswith('qwen_goal_rewriter.py'))
            self.assertEqual(argv[-4:],('--max-new-tokens','384','--local-files-only','--serve'))
            (model/'config.json').write_bytes(b'{"a":2}')
            with self.assertRaisesRegex(Exception,'differs from its startup pin'):rewriter_argv(pin,ROOT,'/site')

    def test_launcher_selection_options(self):
        release=ROOT
        with patch.object(sys,'path',list(sys.path)):
            self.assertIsNone(serve.selection_contract(release,'full',{}))
            with self.assertRaises(ValueError):serve.selection_contract(release,'random50',dict(active_ratio=.3))
            with self.assertRaisesRegex(ValueError,'goal-rewriter-model'):
                serve.selection_contract(release,'goal_query',{})
            value=serve.selection_contract(release,'observation_query',dict(no_goal_rewriter=True,active_token_budget=16384))
            self.assertEqual((value['active_ratio'],value['active_token_budget'],value['goal_rewriter']),(None,16384,None))
            value=serve.selection_contract(release,'goal_query',dict(no_goal_rewriter=True,
                goal_query_variant='candidate_entity_cap8'))
            self.assertEqual((value['active_ratio'],value['goal_query_variant']),(.4,'candidate_entity_cap8'))

    def test_only_goal_query_pins_the_goal_rewriter(self):
        release=ROOT
        with patch.object(sys,'path',list(sys.path)),TemporaryDirectory() as directory:
            model=Path(directory)/'qwen3-0.6b';model.mkdir();(model/'config.json').write_bytes(b'{}')
            pinned=contract.default_contract(goal_rewriter=contract.rewriter_pin(model))
            self.assertEqual([p for p in serve.POLICIES if contract.uses_goal_rewriter(p)],['goal_query'])
            self.assertIs(contract.for_policy(pinned,'goal_query'),pinned)
            for policy in serve.POLICIES:
                # random_fill's contract always carries its random fill pin.
                fill=contract.random_fill_pin() if policy=='random_fill' else None
                self.assertEqual(contract.for_policy(contract.default_contract(random_fill=fill),policy)['goal_rewriter'],None)
                if policy!='goal_query':
                    with self.assertRaisesRegex(Exception,policy+' does not use the Goal rewriter'):
                        contract.for_policy(pinned,policy)
            # observation_query needs no rewriter choice and never pins one, even when given a checkpoint.
            for selection in ({},dict(goal_rewriter_model=str(model)),dict(no_goal_rewriter=True)):
                value=serve.selection_contract(release,'observation_query',dict(selection))
                self.assertIsNone(value['goal_rewriter'])
                self.assertEqual(serve.goal_rewriter_use('observation_query',selection,value),
                    dict(used=False,reason=serve.GOAL_REWRITER_UNUSED,requested_model=selection.get('goal_rewriter_model')))
            with self.assertRaisesRegex(ValueError,'exclude each other'):
                serve.selection_contract(release,'observation_query',dict(goal_rewriter_model=str(model),no_goal_rewriter=True))
            # goal_query is unchanged: a checkpoint is pinned, or --no-goal-rewriter is explicit.
            value=serve.selection_contract(release,'goal_query',dict(goal_rewriter_model=str(model)))
            self.assertEqual(value['goal_rewriter'],pinned['goal_rewriter'])
            self.assertEqual(serve.goal_rewriter_use('goal_query',dict(goal_rewriter_model=str(model)),value),
                             dict(used=True,reason=None,requested_model=str(model)))
            value=serve.selection_contract(release,'goal_query',dict(no_goal_rewriter=True))
            self.assertEqual(serve.goal_rewriter_use('goal_query',dict(no_goal_rewriter=True),value),
                             dict(used=False,reason='disabled by --no-goal-rewriter',requested_model=None))
            self.assertIsNone(serve.goal_rewriter_use('full',None,None))

if __name__=='__main__':unittest.main()
