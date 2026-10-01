import copy
import math
from types import SimpleNamespace
import unittest

import torch
from discrete_diffusion.conditional_attention import MultiHeadAttention
from constraint_projection import ConstraintProjection
from experiment_io import decode_preserving_empty, empty_generated_record, validate_sequences
from evaluations.ovr import dataset_ovr_with_coverage, dataset_unsat_ratio_by_test_pairs


def old_attention(layer, q, k, v, mask):
    batch = q.shape[0]
    q = layer.query_linear(q).view(batch, -1, layer.num_heads, layer.d_k).transpose(1, 2)
    k = layer.key_linear(k).view(batch, -1, layer.num_heads, layer.d_k).transpose(1, 2)
    v = layer.value_linear(v).view(batch, -1, layer.num_heads, layer.d_k).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(layer.d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
    context = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(batch, -1, layer.d_model)
    return layer.out_linear(context)


class SafeAttentionTests(unittest.TestCase):
    def test_mixed_masks_preserve_valid_outputs_and_finite_gradients(self):
        layer = MultiHeadAttention(8, 2)
        with torch.no_grad():
            layer.out_linear.bias.fill_(0.7)
        q = torch.randn(4, 3, 8, requires_grad=True)
        k = torch.randn(4, 5, 8, requires_grad=True)
        v = torch.randn(4, 5, 8, requires_grad=True)
        mask = torch.tensor([[1,1,1,1,1], [1,0,1,0,0], [0,0,1,0,0], [0,0,0,0,0]], dtype=torch.bool)
        expected = old_attention(layer, q, k, v, mask).detach()
        actual = layer(q, k, v, mask)
        torch.testing.assert_close(actual[:3], expected[:3], rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[3], torch.zeros_like(actual[3])))
        actual.square().sum().backward()
        for tensor in (q, k, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(tensor.grad[3].abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_no_mask_unchanged(self):
        layer = MultiHeadAttention(8, 2)
        q = torch.randn(2, 3, 8)
        torch.testing.assert_close(layer(q,q,q), old_attention(layer,q,q,q,None), rtol=0, atol=0)

    def test_all_empty_mask_is_zero(self):
        layer = MultiHeadAttention(8,2)
        q = torch.randn(2,3,8,requires_grad=True)
        output = layer(q,q,q,torch.zeros(2,3,dtype=torch.bool))
        self.assertEqual(output.abs().sum().item(), 0)
        output.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())


def mock_batch(lengths):
    batch = SimpleNamespace(unpadded_length=torch.tensor(lengths), batch_size=len(lengths))
    batch.to = lambda device: batch
    for i in range(1,7):
        setattr(batch, f'condition{i}_indicator', torch.arange(24).repeat(len(lengths),1))
    return batch


class EmptyTrajectoryTests(unittest.TestCase):
    def test_all_empty_bypasses_decoder(self):
        for lengths in ([0], [0,0,0]):
            batch = mock_batch(lengths)
            task = SimpleNamespace(device='cpu', discrete_diffusion=SimpleNamespace(
                sample_fast=lambda *args, **kwargs: self.fail('Decoder must not run on an all-empty batch')))
            records = decode_preserving_empty(task,batch,{})
            self.assertEqual(len(records),len(lengths))
            self.assertTrue(all(len(x['checkins']) == 0 and x['generation_status']=='empty_temporal' for x in records))
            validate_sequences(records)

    def test_mixed_batch_shape_and_order_preserved(self):
        batch = mock_batch([2,0,1])
        def sample(value, **kwargs):
            self.assertIs(value,batch)
            self.assertEqual(value.batch_size,3)
            return SimpleNamespace(to_seq_list=lambda gps: [{'marker':0},{'marker':1},{'marker':2}])
        task = SimpleNamespace(device='cpu',discrete_diffusion=SimpleNamespace(sample_fast=sample))
        records = decode_preserving_empty(task,batch,{})
        self.assertEqual(records[0]['marker'],0)
        self.assertEqual(records[2]['marker'],2)
        self.assertEqual(len(records[1]['arrival_times']),0)
        self.assertEqual(records[1]['condition1_indicator'].shape,(24,))

    def test_empty_output_is_a_strict_violation_not_zero_filled_skip_metric(self):
        reference = [{'checkins':[4,5]}]
        generated = [{'checkins':[]}]
        skip,strict,coverage = dataset_ovr_with_coverage(reference,generated,{4:4,5:5})
        self.assertTrue(math.isnan(skip))
        self.assertEqual(strict,1)
        self.assertEqual(coverage,0)
        self.assertEqual(dataset_unsat_ratio_by_test_pairs(reference,generated,{4:4,5:5}),1)


class EffectiveConstraintTests(unittest.TestCase):
    def projector(self):
        return ConstraintProjection(num_classes=9,type_classes=3,num_spectial=4,
                                    outer_iterations=3,inner_iterations=2,use_gumbel_softmax=False,
                                    gumbel_temperature=1,delta_tol=1e-6,device='cpu')

    def satisfied_logits(self):
        logits = torch.full((2,9,2),-70.)
        logits[:,4,0]=0
        logits[:,5,1]=0
        return logits

    def test_padding_does_not_prevent_early_stop(self):
        p=self.projector()
        a,b,mask=p.compile_batched_constraints([[([0],[1])],[]],'cpu')
        logits=self.satisfied_logits()
        positions=torch.ones(2,2)
        order,exist=p.compute_hard_constraint_violation_optimized(logits,a,b,positions,mask)
        self.assertEqual(order.sum().item()+exist.sum().item(),0)
        result=p.project_with_matrices(logits,a,b,positions,mask)
        self.assertEqual(p.last_projection_stats['outer_iterations'],1)
        self.assertEqual(p.last_projection_stats['inactive_multiplier_max'],0)
        self.assertTrue(torch.equal(result[1],logits[1]))

    def test_empty_row_with_real_reference_constraints_is_not_optimized(self):
        p=self.projector()
        a,b,mask=p.compile_batched_constraints([[([0],[1])],[([0],[1])]],'cpu')
        positions=torch.tensor([[1,1],[0,0]])
        logits=self.satisfied_logits()
        order,exist,_=p.compute_constraint_violation_optimized(logits,a,b,positions,mask)
        self.assertEqual(order[1,0].item()+exist[1,0].item(),0)
        result=p.project_with_matrices(logits,a,b,positions,mask)
        self.assertTrue(torch.equal(result[1],logits[1]))
        self.assertEqual(p.last_projection_stats['active_rows'],1)
        self.assertEqual(p.last_projection_stats['inactive_multiplier_max'],0)

    def test_no_effective_constraints_returns_unchanged(self):
        p=self.projector()
        a,b,mask=p.compile_batched_constraints([[([0],[1])],[([0],[1])]],'cpu')
        logits=self.satisfied_logits()
        result=p.project_with_matrices(logits,a,b,torch.zeros(2,2),mask)
        self.assertTrue(torch.equal(result,logits))
        self.assertEqual(p.last_projection_stats['optimizer_steps'],0)

    def test_nonfinite_input_is_not_silently_repaired(self):
        p=self.projector()
        a,b,mask=p.compile_batched_constraints([[([0],[1])],[([0],[1])]],'cpu')
        logits=self.satisfied_logits()
        logits[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):
            p.project_with_matrices(logits,a,b,torch.ones(2,2),mask)


class RevisionTests(unittest.TestCase):
    def test_allowlist_does_not_allow_training_changes(self):
        from tools.continue_newyork_sampling import EMPTYFIX_FILES,verify_compatible_sources
        original={name:'old' for name in EMPTYFIX_FILES}
        original['train.py']='same'
        updated={name:'new' for name in EMPTYFIX_FILES}
        updated['train.py']='same'
        verify_compatible_sources(original,updated,'emptyfix-v1')
        updated['train.py']='changed'
        with self.assertRaises(RuntimeError):
            verify_compatible_sources(original,updated,'emptyfix-v1')


if __name__=='__main__':
    unittest.main()
