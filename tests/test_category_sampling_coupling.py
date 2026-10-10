"""Regression for actual sampled categories, hard support and unchanged RNG."""
import math
import os
import unittest
import torch

from tests.test_category_consistent_decoding import _dummy, _batch
from discrete_diffusion.diffusion_transformer import DiffusionTransformer, CATEGORY_DECODING_VERSION, log_onehot_to_index

DEVICE=os.environ.get('CATEGORY_DECODING_TEST_DEVICE','cpu')


def inputs(size=64):
    b=_batch([1]*size,5)
    for key in ('category_mask','poi_mask','unpadded_length'):setattr(b,key,getattr(b,key).to(DEVICE))
    x=torch.full((size,13,5),-70.,device=DEVICE)
    x[:,0,0]=0.;x[:,1,2]=0.;x[:,2,4]=0.
    x[:,4,1]=math.log(.51);x[:,5,1]=math.log(.49)
    x[:,7,3]=math.log(.6);x[:,9,3]=math.log(.4)
    d=_dummy(13);d.sampling_generator=torch.Generator(device=DEVICE).manual_seed(135398)
    DiffusionTransformer.enable_category_consistent_decoding(d,{7:4,8:4,9:5,10:6})
    d.p_pred=lambda *args:(x,x)
    return d,x,b


def run(d,b,step=0):
    return log_onehot_to_index(DiffusionTransformer.p_sample(d,None,None,
        torch.full((len(b.unpadded_length),),step,device=DEVICE,dtype=torch.long),b,diffusion_index=step))


def reference(x,generator):
    u=torch.rand(x.shape,device=x.device,dtype=x.dtype,generator=generator)
    return (x-torch.log(-torch.log(u+1e-30)+1e-30)).argmax(1)


class CategorySamplingCouplingTests(unittest.TestCase):
    def test_actual_p_sample_couples_ambiguous_categories_without_extra_draws(self):
        d,x,b=inputs()
        g=torch.Generator(device=DEVICE).manual_seed(135398)
        expected=reference(x,g)
        self.assertGreater(int((expected[:,1]!=x[:,:,1].argmax(1)).sum()),0)
        self.assertLess(int((expected[:,1]!=x[:,:,1].argmax(1)).sum()),64)
        cpu_before=torch.get_rng_state().clone()
        gpu_before=torch.cuda.get_rng_state().clone() if DEVICE.startswith('cuda') else None
        actual=run(d,b)
        self.assertTrue(torch.equal(actual[:,1],expected[:,1]))
        self.assertTrue(torch.equal(d.poi_class_to_category.to(DEVICE)[actual[:,3]],actual[:,1]))
        self.assertTrue(torch.equal(g.get_state(),d.sampling_generator.get_state()))
        self.assertTrue(torch.equal(cpu_before,torch.get_rng_state()))
        if gpu_before is not None:self.assertTrue(torch.equal(gpu_before,torch.cuda.get_rng_state()))

    def test_default_off_and_no_batch_training_path_match_old_sampler(self):
        for enabled in (False,True):
            d,x,b=inputs();d.category_consistent_decoding=enabled
            g=torch.Generator(device=DEVICE).manual_seed(135398);expected=reference(x,g)
            actual=log_onehot_to_index(d.log_sample_categorical(x)) if enabled else run(d,b)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(g.get_state(),d.sampling_generator.get_state()))

    def test_uses_supplied_actual_category_instead_of_logit_argmax(self):
        d,x,b=inputs(1);tokens=x.argmax(1);tokens[:,1]=5
        out=d._apply_category_consistent_poi_mask(x,b,tokens,final_step=True)
        self.assertEqual(float(out[0,9,3]),float(x[0,9,3]))
        self.assertTrue(torch.isneginf(out[0,7,3]))

    def test_final_support_excludes_non_poi_and_minus_seventy_leaks(self):
        d,x,b=inputs();x[:,:,1]=-70.;x[:,4,1]=0.
        x[:,:,3]=0.;x[:,7:9,3]=-70.
        actual=run(d,b)
        self.assertTrue(bool(((actual[:,3]==7)|(actual[:,3]==8)).all()))

    def test_unresolved_category_mid_step_skips_but_final_fails(self):
        d,x,b=inputs();x[:,:,1]=-70.;x[:,11,1]=0.
        expected=reference(x,torch.Generator(device=DEVICE).manual_seed(135398))
        self.assertTrue(torch.equal(run(d,b,10),expected))
        with self.assertRaisesRegex(RuntimeError,'Final sampled category'):run(d,b)

    def test_poi_mask_is_allowed_mid_step_only(self):
        d,x,b=inputs();x[:,:,1]=-70.;x[:,4,1]=0.;x[:,:,3]=-70.;x[:,12,3]=0.
        self.assertTrue(bool((run(d,b,10)[:,3]==12).all()))
        self.assertTrue(bool((run(d,b)[:,3]<11).all()))

    def test_missing_poi_category_and_invalid_inputs_fail_explicitly(self):
        d,x,b=inputs(1)
        for mapping in ({},{4:4},{13:4},{7:3},{7:7}):
            with self.assertRaises(ValueError):DiffusionTransformer.enable_category_consistent_decoding(d,mapping)
        with self.assertRaisesRegex(ValueError,'Actual sampled'):d._apply_category_consistent_poi_mask(x,b)
        with self.assertRaises(ValueError):d._apply_category_consistent_poi_mask(x,None,x.argmax(1))
        DiffusionTransformer.enable_category_consistent_decoding(d,{7:4,8:4})
        x[:,:,1]=-70.;x[:,5,1]=0.
        with self.assertRaisesRegex(RuntimeError,'no legal POI'):run(d,b)

    def test_nonfinite_input_fails(self):
        d,x,b=inputs(1);x[0,7,3]=float('nan')
        with self.assertRaises(FloatingPointError):run(d,b)

    def test_variable_lengths_and_empty_row_preserve_other_positions(self):
        d,_,_=inputs(3);b=_batch([3,1,0],9)
        for key in ('category_mask','poi_mask','unpadded_length'):setattr(b,key,getattr(b,key).to(DEVICE))
        x=torch.full((3,13,9),-70.,device=DEVICE);x[:,3,:]=0.
        for i,n in enumerate((3,1,0)):
            for j in range(n):
                x[i,:,1+j]=-70.;x[i,4,1+j]=math.log(.51);x[i,5,1+j]=math.log(.49)
                x[i,:,n+2+j]=-70.;x[i,7,n+2+j]=math.log(.6);x[i,9,n+2+j]=math.log(.4)
        d.p_pred=lambda *args:(x,x)
        expected=reference(x,torch.Generator(device=DEVICE).manual_seed(135398))
        actual=run(d,b)
        self.assertTrue(torch.equal(actual[~b.poi_mask.bool()],expected[~b.poi_mask.bool()]))
        table=d.poi_class_to_category.to(DEVICE)
        for i,n in enumerate((3,1,0)):
            for j in range(n):self.assertEqual(int(table[actual[i,n+2+j]]),int(actual[i,1+j]))

    def test_worker_resume_rejects_old_argmax_results(self):
        from tools.ablation_worker import validate_category_decoding_result
        validate_category_decoding_result({}, {})
        job={'category_consistent_decoding':True}
        result=dict(category_consistent_decoding=True,category_consistent_decoding_version=CATEGORY_DECODING_VERSION)
        validate_category_decoding_result(job,result)
        for bad in ({},dict(result,category_consistent_decoding_version='argmax-v1')):
            with self.assertRaises(RuntimeError):validate_category_decoding_result(job,bad)
        with self.assertRaises(RuntimeError):validate_category_decoding_result({},result)


if __name__=='__main__':unittest.main()
