import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from constraint_projection import ConstraintProjection
from discrete_diffusion.diffusion_transformer import DiffusionTransformer
from tools.ablation_common import generator,stream_seed,rng_digest,evaluate,mean_sd


class ProjectionAblationTests(unittest.TestCase):
    def make(self,**extra):
        options=dict(num_classes=9,type_classes=3,num_spectial=4,outer_iterations=2,inner_iterations=3,
                     lambda_init=1,mu_init=1,gumbel_temperature=3,device='cpu',verbose=False,collect_diagnostics=True)
        options.update(extra)
        return ConstraintProjection(**options)

    def execute(self,p,missing=False):
        v=torch.full((2,9,3),-5.)
        v[:,6,:]=0
        if not missing:
            v[:,5,0]=2; v[:,4,2]=2
        a,b,m=p.compile_batched_constraints([[([0],[1])],[]],'cpu')
        return p.project_with_matrices(v,a,b,torch.ones(2,3),m)

    def test_default_output_and_rng_match_frozen_optimized_reference(self):
        path=Path(__file__).parent/'fixtures/projection_reference_d5d7d01.py'
        spec=importlib.util.spec_from_file_location('projection_d5d7d01',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        kw=dict(num_classes=9,type_classes=3,num_spectial=4,outer_iterations=2,inner_iterations=3,
                lambda_init=1,mu_init=1,gumbel_temperature=3,device='cpu',verbose=False)
        results=[]
        for cls in (module.ConstraintProjection,ConstraintProjection):
            p=cls(**kw)
            torch.manual_seed(14)
            results.append((self.execute(p),torch.get_rng_state()))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
        self.assertTrue(torch.equal(results[0][1],results[1][1]))

    def test_heads_and_multiplier_updates_are_disabled(self):
        for head in ('order','exist'):
            kw={'projection_order_weight':0} if head=='order' else {'projection_existence_weight':0}
            p=self.make(**kw,early_stop=False)
            self.execute(p,missing=True)
            self.assertEqual(p.last_projection_stats[f'lambda_{head}_max'],0)
            self.assertEqual(p.last_projection_stats[f'mu_{head}_max'],0)
        p=self.make(update_multipliers=False,early_stop=False)
        self.execute(p,missing=True)
        for name in ('lambda_order','lambda_exist','mu_order','mu_exist'):
            self.assertEqual(p.last_projection_stats[name+'_max'],1)

    def test_fixed_budget_runs_500_even_if_already_satisfied(self):
        p=self.make(outer_iterations=10,inner_iterations=50,early_stop=False,use_gumbel_softmax=False)
        a,b,m=p.compile_batched_constraints([[([0],[1])]],'cpu')
        values=torch.full((1,9,2),-70.)
        values[0,4,0]=0; values[0,5,1]=0
        p.project_with_matrices(values,a,b,torch.ones(1,2),m)
        self.assertEqual(p.last_projection_stats['optimizer_steps'],500)
        self.assertEqual(p.last_projection_stats['outer_iterations'],10)

    def test_inactive_head_is_excluded_from_early_stop(self):
        p=self.make(projection_order_weight=0,use_gumbel_softmax=False,inner_iterations=1)
        a,b,m=p.compile_batched_constraints([[([0],[1])]],'cpu')
        values=torch.full((1,9,2),-70.)
        values[0,5,0]=0; values[0,4,1]=0
        p.project_with_matrices(values,a,b,torch.ones(1,2),m)
        self.assertEqual(p.last_projection_stats['outer_iterations'],1)

    def test_empty_and_disabled_constraints_skip_without_random_draws(self):
        p=self.make(generator=torch.Generator().manual_seed(9),early_stop=False)
        a,b,m=p.compile_batched_constraints([[([0],[1])]],'cpu')
        values=torch.zeros(1,9,2)
        before=rng_digest(p.generator)
        result=p.project_with_matrices(values,a,b,torch.zeros(1,2),m)
        self.assertTrue(torch.equal(result,values))
        self.assertEqual(p.last_projection_stats['optimizer_steps'],0)
        self.assertEqual(before,rng_digest(p.generator))

    def test_no_gumbel_remains_differentiable_without_rng_consumption(self):
        g=torch.Generator().manual_seed(9)
        p=self.make(generator=g,use_gumbel_softmax=False,early_stop=False)
        before=rng_digest(g)
        self.execute(p,missing=True)
        self.assertEqual(before,rng_digest(g))
        self.assertGreater(p.last_projection_stats['gradient_norm_sum'],0)

    def test_projection_generator_does_not_touch_global_rng(self):
        p=self.make(generator=torch.Generator().manual_seed(9),early_stop=False)
        before=torch.get_rng_state()
        self.execute(p)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_zero_kl_is_not_a_direction_change(self):
        p=self.make(projection_kl_weight=0,early_stop=False)
        result=self.execute(p)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(p.last_projection_stats['kl_model_to_projected_sum'],-1e-5)
        with self.assertRaises(ValueError):
            self.make(projection_kl_weight=float('nan'))

    def test_zero_kl_weight_removes_its_optimization_gradient(self):
        original=torch.nn.functional.kl_div
        torch.manual_seed(71)
        expected=self.execute(self.make(projection_kl_weight=0,early_stop=False))
        torch.manual_seed(71)
        with mock.patch('torch.nn.functional.kl_div',side_effect=lambda *a,**k:100*original(*a,**k)):
            actual=self.execute(self.make(projection_kl_weight=0,early_stop=False))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


class StreamTests(unittest.TestCase):
    def test_streams_are_stable_and_independent(self):
        self.assertEqual(stream_seed(135398,0,'spatial'),stream_seed(135398,0,'spatial'))
        self.assertNotEqual(stream_seed(135398,0,'spatial'),stream_seed(135398,0,'projection'))
        dd=object.__new__(DiffusionTransformer); nn.Module.__init__(dd); dd.num_classes=9
        outputs=[]
        for projection_draws in (0,500):
            dd.sampling_generator=generator(135398,0,'spatial','cpu')
            extra=generator(135398,0,'projection','cpu')
            torch.rand(projection_draws,generator=extra)
            before=torch.get_rng_state()
            outputs.append(dd.log_sample_categorical(torch.zeros(2,9,3)))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(outputs[0],outputs[1]))


class MetricTests(unittest.TestCase):
    def record(self,pois,marks=None):
        mapping={10:4,20:5}
        return dict(checkins=np.array(pois),marks=np.array(marks if marks is not None else [mapping[p] for p in pois]),
                    arrival_times=np.arange(1,len(pois)+1,dtype=float),
                    **{f'condition{i}_indicator':np.arange(24) for i in range(1,7)})

    def test_decomposition_and_raw_token_mismatch(self):
        refs=[self.record([10,20]),self.record([10,20])]
        gen=[self.record([20,10],[4,4]),self.record([10])]
        with mock.patch('tools.ablation_common.Get_Statistical_Metrics',return_value={'totalJSD':.1}):
            m,rows=evaluate(refs,gen,{10:4,20:5},[0,1])
        self.assertEqual(m['missing_contribution'],.5)
        self.assertEqual(m['order_contribution'],.5)
        self.assertEqual(m['strict_ovr'],1)
        self.assertAlmostEqual(m['category_poi_mismatch_rate'],1/3)
        self.assertIsNone(rows[1]['ovr_skip'])
        self.assertEqual(gen[0]['marks'].tolist(),[4,4])

    def test_all_empty_is_strict_failure_not_zero_filled_statistics(self):
        m,_=evaluate([self.record([10,20])],[self.record([])],{10:4,20:5},[0])
        self.assertEqual(m['strict_ovr'],1)
        self.assertIsNone(m['ovr_skip'])
        self.assertGreater(m['Category'], 0)
        self.assertGreater(m['CategoryTransition'], 0)
        self.assertIsNone(m['Distance'])
        self.assertEqual(m['empty_rate'],1)

    def test_sample_standard_deviation(self):
        self.assertEqual(mean_sd([1,2,3]),dict(mean=2.,sample_sd=1.,n=3))


class CacheTests(unittest.TestCase):
    def test_adaptive_state_is_not_reset_between_batches(self):
        from tools import ablation_worker as worker
        class Time:
            def __init__(self,n):
                self.batch_size=n
                self.time=torch.tensor([[1.,2.]]).repeat(n,1)
                self.mask=torch.ones(n,2,dtype=torch.bool)
                self.unpadded_length=torch.full((n,),2)
            def to(self,device): return self
            def mask_check(self): return self
        intensity=SimpleNamespace(rejections_sample_multiple=2)
        observed=[]
        def sample(n,**kwargs):
            observed.append(intensity.rejections_sample_multiple)
            intensity.rejections_sample_multiple+=1
            return Time(n)
        task=SimpleNamespace(device='cpu',tpp_model=SimpleNamespace(sample=sample,intensity_model=intensity),
            discrete_diffusion=SimpleNamespace(condition_encoder=SimpleNamespace(max_position_embeddings=3000),
                transformer=SimpleNamespace(positional_encoding=SimpleNamespace(num_embeddings=3000))))
        def make_batch(selected):
            b=SimpleNamespace(batch_size=len(selected),tmax=24.,po_matrix=torch.ones(len(selected),2,2))
            b.to=lambda device:b
            return b
        with tempfile.TemporaryDirectory() as directory:
            job=dict(output_dir=directory,seed=135398,global_start=0,indices=list(range(128)),
                     split='train',manifest_sha256='test')
            with mock.patch.object(worker,'original_task',return_value=(task,None)), \
                 mock.patch.object(worker,'references',return_value=({},list(range(128)),[])), \
                 mock.patch.object(worker.Batch,'from_sequence_list',side_effect=make_batch), \
                 mock.patch.object(worker,'seed_sampling'),mock.patch('torch.cuda.synchronize'):
                cache,_=worker.create_cache(job)
            self.assertEqual(observed,[2,3])
            self.assertEqual(cache['indices'],list(range(128)))
            self.assertEqual(cache['adaptive_states'][-1]['after'],4)


class TrackingTests(unittest.TestCase):
    def experiment(self,directory):
        from tools.run_pcdg_ablation import Ablation
        experiment=object.__new__(Ablation)
        experiment.out=Path(directory)
        experiment.args=SimpleNamespace(run_id='test')
        experiment.entity='test'; experiment.manifest={}; experiment.status=mock.Mock()
        (experiment.out/'seed-135398/full').mkdir(parents=True)
        return experiment

    def test_eventual_readback_retries_without_sampling(self):
        from tools import run_pcdg_ablation as runner
        result=dict(seed=135398,variant='full',metrics={'strict_ovr':.5},timing={},output_sha256='abc')
        fake=SimpleNamespace(log=mock.Mock(),summary={},finish=mock.Mock(),url='https://example.test/run')
        api=mock.Mock()
        api.run.side_effect=[SimpleNamespace(summary={}),SimpleNamespace(summary={}),
                             SimpleNamespace(summary={'complete':True,'result_sha256':'abc'})]
        with tempfile.TemporaryDirectory() as directory:
            experiment=self.experiment(directory)
            with mock.patch.object(runner.wandb,'init',return_value=fake), \
                 mock.patch.object(runner.wandb,'Api',return_value=api), \
                 mock.patch.object(runner.time,'sleep'),mock.patch.object(runner.subprocess,'Popen') as launch:
                experiment.sync(result)
                experiment.sync(result)
                self.assertEqual(api.run.call_count,3)
                launch.assert_not_called()

    def test_readback_failure_is_bounded(self):
        from tools import run_pcdg_ablation as runner
        fake=SimpleNamespace(log=mock.Mock(),summary={},finish=mock.Mock(),url='https://example.test/run')
        api=mock.Mock(); api.run.return_value=SimpleNamespace(summary={})
        with tempfile.TemporaryDirectory() as directory:
            experiment=self.experiment(directory)
            with mock.patch.object(runner.wandb,'init',return_value=fake), \
                 mock.patch.object(runner.wandb,'Api',return_value=api),mock.patch.object(runner.time,'sleep'):
                with self.assertRaises(RuntimeError):
                    experiment.sync(dict(seed=135398,variant='full',metrics={},timing={},output_sha256='abc'))
                self.assertEqual(api.run.call_count,3)
                self.assertFalse((experiment.out/'seed-135398/full/wandb-sync.json').exists())


if __name__=='__main__':
    unittest.main()
