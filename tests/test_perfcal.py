import contextlib
import io
from pathlib import Path
from unittest import mock
import unittest

import torch
from constraint_projection import ConstraintProjection
from tools.perfcal_common import reference_projector_class,temperature_choice,batch_choice

ROOT=Path(__file__).resolve().parents[1]


class ProjectionEquivalenceTests(unittest.TestCase):
    def test_diagnostics_do_not_change_outputs_or_rng(self):
        kwargs=dict(num_classes=9,type_classes=3,num_spectial=4,outer_iterations=2,
                    inner_iterations=3,gumbel_temperature=1,device='cpu',verbose=False)
        ordinary=ConstraintProjection(**kwargs)
        diagnosed=ConstraintProjection(**kwargs,collect_diagnostics=True)
        a,b,mask=ordinary.compile_batched_constraints([[([0],[1])],[]],'cpu')
        logits=torch.log_softmax(torch.randn(2,9,4),dim=1)
        positions=torch.ones(2,4)
        torch.manual_seed(12)
        expected=ordinary.project_with_matrices(logits,a,b,positions,mask)
        expected_rng=torch.get_rng_state()
        torch.manual_seed(12)
        actual=diagnosed.project_with_matrices(logits,a,b,positions,mask)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        self.assertGreater(diagnosed.last_projection_stats['unmet_probe_count'],0)

    def test_saturated_constraint_probe_stays_zero(self):
        p=ConstraintProjection(9,3,4,outer_iterations=1,inner_iterations=1,
                              gumbel_temperature=.1,device='cpu',verbose=False,collect_diagnostics=True)
        a,b,mask=p.compile_batched_constraints([[([0],[1])]],'cpu')
        logits=torch.full((1,9,4),-100.)
        logits[:,6,:]=0
        p.project_with_matrices(logits,a,b,torch.ones(1,4),mask)
        self.assertEqual(p.last_projection_stats['unmet_probe_count'],1)
        self.assertEqual(p.last_projection_stats['zero_gradient_probe_count'],1)

    def test_outputs_rng_and_stopping_match_frozen_reference(self):
        old_class=reference_projector_class(ROOT)
        for temperature in (0.1,0.7,2.0):
            for batch_size in (1,3):
                with self.subTest(temperature=temperature,batch_size=batch_size):
                    kwargs=dict(num_classes=9,type_classes=3,num_spectial=4,outer_iterations=2,
                                inner_iterations=3,gumbel_temperature=temperature,device='cpu')
                    reference=old_class(**kwargs)
                    optimized=ConstraintProjection(**kwargs,verbose=False)
                    conditions=[[([0],[1])]]*batch_size
                    if batch_size>1:
                        conditions[-1]=[]
                    a,b,mask=reference.compile_batched_constraints(conditions,'cpu')
                    torch.manual_seed(99)
                    logits=torch.log_softmax(torch.randn(batch_size,9,4),dim=1)
                    positions=torch.ones(batch_size,4)
                    torch.manual_seed(123)
                    with contextlib.redirect_stdout(io.StringIO()):
                        expected=reference.project_with_matrices(logits,a,b,positions,mask)
                    expected_rng=torch.get_rng_state()
                    torch.manual_seed(123)
                    actual=optimized.project_with_matrices(logits,a,b,positions,mask)
                    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                    self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
                    for key in ('optimizer_steps','outer_iterations','inactive_multiplier_max'):
                        self.assertEqual(optimized.last_projection_stats[key],reference.last_projection_stats[key])

    def test_invariants_cached_only_per_invocation(self):
        p=ConstraintProjection(9,3,4,outer_iterations=2,inner_iterations=3,device='cpu',verbose=False)
        a,b,mask=p.compile_batched_constraints([[([0],[1])]],'cpu')
        logits=torch.log_softmax(torch.randn(1,9,4),dim=1)
        positions=torch.ones(1,4)
        with mock.patch.object(p,'effective_constraint_mask',wraps=p.effective_constraint_mask) as masks:
            with mock.patch('torch.nn.functional.log_softmax',wraps=torch.nn.functional.log_softmax) as logs:
                p.project_with_matrices(logits,a,b,positions,mask)
                first_steps=p.last_projection_stats['optimizer_steps']
                p.project_with_matrices(logits,a,b,positions,mask)
                second_steps=p.last_projection_stats['optimizer_steps']
        self.assertEqual(masks.call_count,2)
        self.assertEqual(logs.call_count,first_steps+second_steps+2)

    def test_unknown_nonfinite_gradient_is_still_rejected(self):
        p=ConstraintProjection(9,3,4,outer_iterations=1,inner_iterations=1,device='cpu',verbose=False)
        a,b,mask=p.compile_batched_constraints([[([0],[1])]],'cpu')
        with mock.patch('torch.nn.utils.clip_grad_norm_',side_effect=RuntimeError('non-finite norm')):
            with self.assertRaises(FloatingPointError):
                p.project_with_matrices(torch.zeros(1,9,3),a,b,torch.ones(1,3),mask)


class SelectionTests(unittest.TestCase):
    def row(self,label,batch=64,speed=10,strict=.6,coverage=.5,memory=.5):
        return dict(label=label,state='complete',batch_size=batch,samples_per_second=speed,
                    sampling_seconds=100,metrics={'strict_ovr':strict,'pair_coverage':coverage},
                    memory_fraction=memory,violating_probes_all_zero=False)

    def profile(self):
        return dict(memory_fraction_limit=.8,strict_ovr_tolerance=.01,
                    pair_coverage_tolerance=.01,throughput_tie_fraction=.05)

    def test_temperature_excludes_stalled_gradient(self):
        stalled=self.row('stalled',strict=.1)
        stalled['violating_probes_all_zero']=True
        valid=self.row('valid')
        self.assertIs(temperature_choice([stalled,valid]),valid)

    def test_batch_gates_and_small_batch_tie_break(self):
        base=self.row('b64')
        close=self.row('b128',128,speed=10.4)
        bad_quality=self.row('b256',256,speed=30,strict=.7)
        oom=self.row('b512',512,speed=40,memory=.9)
        chosen,reasons=batch_choice([base,close,bad_quality,oom],self.profile())
        self.assertEqual(chosen['batch_size'],64)
        self.assertTrue(reasons['b256'])
        self.assertTrue(reasons['b512'])


if __name__=='__main__':
    unittest.main()
