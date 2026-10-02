import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from baseline_posthoc_swap import fix_single_sequence
from baseline_models import EnergyGuidance,POCFGDiffusion
from discrete_diffusion.diffusion_transformer import DiffusionTransformer
from tools.baseline_common import select_candidate


class SwapTests(unittest.TestCase):
    def sample(self):
        s=dict(arrival_times=np.array([1.,2.,3.]),marks=np.array([5,4,5]),
               checkins=np.array([20,10,21]),gps=[[2,0],[1,0],[2,1]])
        s.update({f'condition{i}':np.array([i,i+1,i+2]) for i in range(1,7)})
        s.update({f'condition{i}_indicator':np.arange(24) for i in range(1,7)})
        return s

    def test_swap_keeps_time_context_and_event_tuple(self):
        old=self.sample()
        new=fix_single_sequence(old,[(1,2)],[2,1,2])
        self.assertEqual(new['checkins'].tolist(),[10,20,21])
        self.assertEqual(new['marks'].tolist(),[4,5,5])
        self.assertEqual(new['gps'],[[1,0],[2,0],[2,1]])
        for key in ['arrival_times']+[f'condition{i}' for i in range(1,7)]:
            np.testing.assert_array_equal(old[key],new[key])

    def test_missing_empty_satisfied_and_cycle(self):
        s=self.sample()
        self.assertIs(fix_single_sequence(s,[(1,9)],[2,1,2]),s)
        self.assertIs(fix_single_sequence(s,[],[2,1,2]),s)
        self.assertIs(fix_single_sequence(s,[(1,2)],[1,2,2]),s)
        empty=dict(checkins=[],marks=[],arrival_times=[],gps=[])
        self.assertIs(fix_single_sequence(empty,[(1,2)],[]),empty)
        with self.assertRaises(ValueError):
            fix_single_sequence(s,[(1,2),(2,1)],[2,1,2])


class EnergyTests(unittest.TestCase):
    def setup_values(self,scale=1):
        guide=EnergyGuidance(SimpleNamespace(num_classes=9,type_classes=3,num_spectial=4),3.,scale)
        values=torch.full((1,9,3),-4.)
        values[:,5,0]=0
        values[:,4,2]=0
        return guide,values,torch.ones(1,3),[[([0],[1])]]

    def test_energy_matches_order_plus_five_existence_gradient(self):
        g,values,positions,constraints=self.setup_values()
        a,b,mask=g.projector.compile_batched_constraints(constraints,'cpu')
        logits=values.clone().requires_grad_(True)
        order,exist,_=g.projector.compute_constraint_violation_optimized(logits,a,b,positions,mask)
        grad=torch.autograd.grad((order+5*exist).sum(),logits)[0]
        expected=(values-grad).clamp(-70,0)
        actual=g.apply(values,positions,constraints,0)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(g.stats['calls'],1)
        self.assertGreater(g.stats['gradient_norm_sum'],0)

    def test_zero_scale_window_and_nonfinite(self):
        g,v,p,c=self.setup_values(scale=0)
        before=torch.get_rng_state()
        self.assertIs(g.apply(v,p,c,0),v)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        g,v,p,c=self.setup_values()
        self.assertIs(g.apply(v,p,c,40),v)
        self.assertIs(g.apply(v,p,c,3),v)
        v[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):
            g.apply(v,p,c,0)

    def test_absent_positions_are_not_guided(self):
        g,v,p,c=self.setup_values()
        self.assertIs(g.apply(v,torch.zeros_like(p),c,0),v)


class ToyTransformer(nn.Module):
    def forward(self,tokens,cond,t,batch):
        return cond[:,:,:7].transpose(1,2)


class CFGTests(unittest.TestCase):
    def model(self):
        dd=object.__new__(POCFGDiffusion)
        nn.Module.__init__(dd)
        dd.type_classes=2
        dd.po_encoder=nn.Sequential(nn.Linear(4,256),nn.ReLU(),nn.Linear(256,256))
        dd.po_null=nn.Parameter(torch.zeros(256))
        dd.po_drop_probability=.1
        dd.cfg_scale=1
        dd.cfg_trained=True
        dd.transformer=ToyTransformer()
        dd.reset_cfg_stats()
        dd.eval()
        return dd

    def batch(self):
        return SimpleNamespace(po_matrix=torch.tensor([[[0.,1.],[0.,0.]],[[0.,0.],[0.,0.]]]),
                               mask=torch.ones(2,3,dtype=torch.bool))

    def test_dropout_only_removes_po_and_no_edges_use_null(self):
        dd=self.model()
        bg=torch.randn(2,3,256)
        dd.po_drop_probability=1
        train,_=dd.po_condition(bg,self.batch(),'train')
        null,_=dd.po_condition(bg,self.batch(),'null')
        torch.testing.assert_close(train,null,rtol=0,atol=0)
        torch.testing.assert_close(null,bg,rtol=0,atol=0)
        dd.po_drop_probability=0
        conditional,_=dd.po_condition(bg,self.batch(),'train')
        torch.testing.assert_close(conditional[1],null[1],rtol=0,atol=0)
        self.assertGreater((conditional[0]-null[0]).abs().max().item(),0)
        self.assertGreater(dd.cfg_stats['conditional_rows'],0)
        self.assertGreater(dd.cfg_stats['null_rows'],0)

    def test_scale_limits_combination_and_capability_guard(self):
        dd=self.model()
        bg=torch.randn(2,3,256)
        batch=self.batch()
        x=torch.zeros(2,9,3)
        t=torch.zeros(2,dtype=torch.long)
        dd.cfg_scale=0
        lu=dd.raw_cfg_logits(x,bg,t,batch)
        dd.cfg_scale=1
        lc=dd.raw_cfg_logits(x,bg,t,batch)
        dd.cfg_scale=3
        actual=dd.raw_cfg_logits(x,bg,t,batch)
        torch.testing.assert_close(actual,lu+3*(lc-lu),rtol=0,atol=0)
        self.assertGreater(dd.cfg_stats['max_branch_difference'],0)
        dd.cfg_trained=False
        with self.assertRaises(RuntimeError):
            dd.raw_cfg_logits(x,bg,t,batch)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_disabled_methods_match_original_output_and_rng(self):
        path=Path(__file__).parent/'fixtures/diffusion_reference_6e4a5dd.py'
        spec=importlib.util.spec_from_file_location('frozen_diffusion',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values=torch.log_softmax(torch.randn(2,9,4),dim=1)
        outputs=[]
        for cls in (module.DiffusionTransformer,DiffusionTransformer):
            dd=object.__new__(cls)
            nn.Module.__init__(dd)
            dd.num_classes=9
            dd.projection_last_k_steps=40
            dd.use_constraint_projection=False
            dd.p_pred=lambda *args:(values,values)
            torch.manual_seed(55)
            out=dd.p_sample(values,None,torch.zeros(2,dtype=torch.long),None,diffusion_index=0)
            outputs.append((out,torch.get_rng_state()))
        torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=0)
        self.assertTrue(torch.equal(outputs[0][1],outputs[1][1]))

    def test_old_model_cannot_silently_run_cfg(self):
        dd=object.__new__(DiffusionTransformer)
        nn.Module.__init__(dd)
        with self.assertRaises(RuntimeError):
            dd.sample_fast(None,baseline_method='cfg')


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_rejects_stalled(self):
        def row(scale,strict=.5):
            return dict(state='complete',method_executed=True,memory_fraction=.1,scale=scale,temperature=3,
                metrics=dict(strict_ovr=strict,pair_coverage=.6),all_violation_gradients_zero=False)
        a,b=row(1),row(10)
        stalled=row(100,.1)
        stalled['all_violation_gradients_zero']=True
        self.assertIs(select_candidate([b,stalled,a],'baseline3'),a)


if __name__=='__main__':
    unittest.main()
