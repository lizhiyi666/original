"""Run on CPU by default; DISTANCE_TEST_DEVICE=cuda explicitly exercises CUDA."""
import inspect
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from distance_kl import (BatchedDistanceObjective, DistanceObjective, distance_generator,
    distance_metadata, distance_objective_class, distance_output_directory, haversine,
    soft_histogram, validate_distance_metadata)
from tools.ablation_common import generator
from tools.distance_benchmark_fixture import fixture, make_projector


class BatchedDistanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = os.environ.get('DISTANCE_TEST_DEVICE', 'cpu')

    def test_loss_gradient_masks_rng_and_repetition(self):
        for candidates in (1, 5, 40):
            for stochastic in (False, True):
                for seed in (1, 42, 135398):
                    with self.subTest(candidates=candidates, stochastic=stochastic, seed=seed):
                        d = fixture(self.device, batch=18, max_pois=6, candidates=candidates, seed=seed)
                        objects = [cls(d['reference'], d['y'], d['poi'], d['active'])
                                   for cls in (DistanceObjective, BatchedDistanceObjective)]
                        rngs = [distance_generator(seed, self.device) for _ in objects]
                        noises = [obj.noise(rng, stochastic) for obj, rng in zip(objects, rngs)]
                        self.assertTrue(torch.equal(rngs[0].get_state(), rngs[1].get_state()))
                        if stochastic:
                            for i, row_noise in enumerate(noises[0]):
                                self.assertTrue(torch.equal(row_noise, noises[1][i, :, :row_noise.shape[1]]))
                        values, grads = [], []
                        for obj, noise in zip(objects, noises):
                            y = d['y'].clone().requires_grad_()
                            value = obj.loss(y, noise)
                            grad, = torch.autograd.grad(value, y)
                            values.append(value.detach()); grads.append(grad)
                            self.assertTrue(torch.isfinite(grad).all())
                            eligible = d['poi'] & d['active'][:, None] & (d['poi'].sum(1) >= 2)[:, None]
                            self.assertEqual(grad[~eligible].count_nonzero().item(), 0)
                        torch.testing.assert_close(values[1], values[0], rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(grads[1], grads[0], rtol=1e-5, atol=1e-6)
                        y = d['y'].clone().requires_grad_()
                        repeated = objects[1].loss(y, noises[1])
                        repeated_grad, = torch.autograd.grad(repeated, y)
                        self.assertTrue(torch.equal(repeated, values[1]))
                        self.assertTrue(torch.equal(repeated_grad, grads[1]))

    def test_empty_singleton_inactive_and_padding(self):
        d = fixture(self.device, batch=8, max_pois=6, candidates=5)
        for mask in (torch.zeros_like(d['poi']), d['poi'] & (torch.arange(d['poi'].shape[1], device=self.device) < 2)):
            obj = BatchedDistanceObjective(d['reference'], d['y'], mask, d['active'])
            rng = distance_generator(3, self.device)
            before = rng.get_state()
            self.assertIsNone(obj.noise(rng))
            self.assertTrue(torch.equal(before, rng.get_state()))
            y = d['y'].clone().requires_grad_()
            obj.loss(y, None).backward()
            self.assertEqual(y.grad.count_nonzero().item(), 0)
        obj = BatchedDistanceObjective(d['reference'], d['y'], d['poi'], d['active'])
        noise = obj.noise(distance_generator(3, self.device))
        changed = noise.clone()
        changed.masked_fill_(~obj.valid_positions[:, None, :, None], 9999)
        torch.testing.assert_close(obj.loss(d['y'], noise), obj.loss(d['y'], changed), rtol=0, atol=0)
        # The batched inner loss must never iterate over ragged rows.
        import ast
        import textwrap
        tree = ast.parse(textwrap.dedent(inspect.getsource(BatchedDistanceObjective.loss)))
        self.assertFalse(any(isinstance(n, (ast.For, ast.ListComp, ast.GeneratorExp)) for n in ast.walk(tree)))

    def test_hard_paths_use_exact_geometry_including_repeated_pois(self):
        d = fixture(self.device, batch=8, max_pois=6, candidates=5)
        obj = BatchedDistanceObjective(d['reference'], d['y'], d['poi'], d['active'])
        noise = torch.full((len(obj.rows), obj.paths, *obj.padded_ids.shape[1:]), -10000., device=self.device)
        exact_lengths = []
        for i, (_, positions, ids, _) in enumerate(obj.rows):
            selected = [6] * (len(positions) - 1) + [7]
            for j, token in enumerate(selected):
                noise[i, :, j, torch.where(ids[j] == token)[0]] = 10000.
            coords = d['reference'].coordinates[[token-6 for token in selected]].to(d['y'])
            exact_lengths.append(haversine(coords[:-1], coords[1:]).sum().repeat(8))
        hist = soft_histogram(torch.stack(exact_lengths), obj.centers, obj.bandwidth)
        expected = (obj.target * (obj.target.log() - hist.log())).sum()
        torch.testing.assert_close(obj.loss(d['y'], noise), expected, rtol=1e-5, atol=1e-6)

    def test_full_projection_budget_diagnostics_random_streams_and_disabled(self):
        d = fixture(self.device, batch=int(os.environ.get('DISTANCE_TEST_BATCH', 8)),
                    max_pois=int(os.environ.get('DISTANCE_TEST_MAX_POIS', 4)),
                    candidates=int(os.environ.get('DISTANCE_TEST_CANDIDATES', 5)))
        for stochastic, weight in ((True, 1.), (False, 1.), (True, 0.)):
            outputs, states, stats = [], [], []
            global_cpu = torch.get_rng_state()
            global_cuda = torch.cuda.get_rng_state() if self.device == 'cuda' else None
            for backend in ('legacy', 'batched', 'batched'):
                p, args = make_projector(d, backend, stochastic, weight)
                spatial = generator(d['seed'], 0, 'spatial', self.device)
                before = spatial.get_state()
                output = p.project_with_matrices(*args, poi_mask=d['poi'])
                # Consume actual categorical draws using the independent spatial stream.
                uniform = torch.rand(output.shape, device=self.device, generator=spatial)
                self.assertFalse(torch.equal(before, spatial.get_state()))
                discrete = (output - torch.log(-torch.log(uniform.clamp_min(1e-8)))).argmax(1)
                outputs.append((output, discrete))
                states.append((spatial.get_state(), p.generator.get_state(), p.distance_generator.get_state()))
                stats.append(p.last_projection_stats)
                self.assertEqual(p.last_projection_stats['optimizer_steps'], 500)
                self.assertEqual(p.last_projection_stats['outer_iterations'], 10)
                self.assertEqual(p.last_projection_stats['inactive_multiplier_max'], 0)
                self.assertTrue(torch.equal(output[~d['active']], args[0][~d['active']]))
                self.assertTrue(torch.isfinite(output).all())
                if weight:
                    self.assertEqual(p.last_projection_stats['distance_backend'], backend)
                    self.assertTrue(torch.isfinite(torch.tensor(p.last_projection_stats['distance_kl'])))
            for i in (1, 2):
                for left, right in zip(states[0], states[i]):
                    self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.equal(outputs[1][0], outputs[2][0]))
            self.assertTrue(torch.equal(outputs[1][1], outputs[2][1]))
            self.assertTrue(torch.equal(global_cpu, torch.get_rng_state()))
            if global_cuda is not None:
                self.assertTrue(torch.equal(global_cuda, torch.cuda.get_rng_state()))
            if weight == 0:
                self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
                self.assertEqual(stats[0], stats[1])
            print('FULL_PROJECTION_COMPARISON', dict(device=self.device, shape=list(d['y'].shape), stochastic=stochastic,
                distance_weight=weight, max_abs_logits=float((outputs[0][0]-outputs[1][0]).abs().max()),
                changed_argmax=int((outputs[0][0].argmax(1) != outputs[1][0].argmax(1)).sum()),
                changed_sampled_tokens=int((outputs[0][1] != outputs[1][1]).sum())), flush=True)

    def test_nonfinite_safety_guards(self):
        d = fixture(self.device, batch=4, max_pois=3, candidates=5)
        for stage in ('loss', 'gradient', 'update'):
            p, args = make_projector(d, 'batched', outer=1, inner=1)
            if stage == 'loss':
                def bad_loss(obj, y, noise):
                    return y.sum() * float('nan')
                context = patch.object(BatchedDistanceObjective, 'loss', bad_loss)
            elif stage == 'gradient':
                context = patch('torch.nn.utils.clip_grad_norm_', side_effect=RuntimeError('nonfinite'))
            else:
                def bad_step(opt, *unused, **kwargs):
                    with torch.no_grad():
                        opt.param_groups[0]['params'][0].fill_(float('nan'))
                context = patch.object(torch.optim.SGD, 'step', bad_step)
            with context, self.assertRaises(FloatingPointError):
                p.project_with_matrices(*args, poi_mask=d['poi'])

    def test_final_diagnostic_noise_refresh_and_first_gradient_probe(self):
        d = fixture(self.device, batch=4, max_pois=3, candidates=5)
        p, args = make_projector(d, 'batched', outer=2, inner=3)
        losses, noise_calls = [], []
        original_loss, original_noise = BatchedDistanceObjective.loss, BatchedDistanceObjective.noise
        def observed_loss(obj, y, noise):
            value = original_loss(obj, y, noise)
            losses.append(value.detach())
            return value
        def observed_noise(obj, rng, stochastic=True):
            noise_calls.append(rng.get_state())
            return original_noise(obj, rng, stochastic)
        with patch.object(BatchedDistanceObjective, 'loss', observed_loss), \
             patch.object(BatchedDistanceObjective, 'noise', observed_noise), \
             patch('torch.autograd.grad', wraps=torch.autograd.grad) as grad:
            output = p.project_with_matrices(*args, poi_mask=d['poi'])
        self.assertEqual(len(losses), 7)  # Six updates and exactly one final loss.
        self.assertEqual(len(noise_calls), 2)  # Once per outer loop, not inner loop.
        self.assertEqual(grad.call_count, 2)  # One distance and one constraint probe.
        self.assertEqual(p.last_projection_stats['distance_kl'], float(losses[-1]))
        obj = BatchedDistanceObjective(d['reference'], d['y'], d['poi'], d['active'])
        rng = distance_generator(d['seed'], self.device)
        obj.noise(rng)
        expected = obj.loss(output.transpose(1, 2), obj.noise(rng))
        self.assertEqual(float(expected), p.last_projection_stats['distance_kl'])


class BackendIdentityTests(unittest.TestCase):
    def test_independent_output_merge_and_mixed_backend_rejection(self):
        from experiment_io import publish_torch, sha256_file
        from merge_results import merge_parts
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'toy'
            base.mkdir()
            dataset = base / 'toy_test.pkl'
            torch.save({}, dataset)
            output = distance_output_directory(base, 'batched', 'new-run')
            self.assertNotEqual(output, base)
            self.assertEqual(distance_output_directory(base), base)
            with self.assertRaises(ValueError):
                distance_output_directory(base, 'batched')
            metadata = dict(total_samples=2, world_size=2, run_id='run', output_tag='new-run',
                data_name='toy', dataset_sha256=sha256_file(dataset), **distance_metadata('batched'))
            for rank in range(2):
                publish_torch(output / f'toy_new-run_generated_part{rank}.pkl',
                    dict(metadata=metadata, rank=rank, test_indices=[rank], t_max=24.,
                         sequences=[dict(checkins=[], arrival_times=[])]))
            destination = merge_parts('toy', 'run', 2, 'new-run', root, distance_backend='batched')
            self.assertEqual(destination.parent, output)
            self.assertFalse((base / 'toy_new-run_generated.pkl').exists())
            bad = torch.load(output / 'toy_new-run_generated_part1.pkl', weights_only=False)
            bad['metadata'].update(distance_metadata('legacy'))
            torch.save(bad, output / 'toy_new-run_generated_part1.pkl')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                merge_parts('toy', 'run', 2, 'new-run', root, distance_backend='batched')

    def test_existing_unversioned_manifest_is_rejected_before_writes(self):
        from tools.run_two_city_distance_v2 import DistanceStudy
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'experiment_runs' / 'old' / 'manifest.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            before = path.read_bytes()
            with patch('tools.run_two_city_distance_v2.ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'version'):
                    DistanceStudy(SimpleNamespace(distance_backend='batched', run_id='old'))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_version_missing_mismatch_and_unknown_fail_closed(self):
        self.assertIs(distance_objective_class(), DistanceObjective)
        self.assertIs(distance_objective_class('batched'), BatchedDistanceObjective)
        for value in ({}, dict(distance_backend='batched'),
                      dict(distance_backend='batched', distance_implementation_version='old')):
            with self.assertRaises(RuntimeError):
                validate_distance_metadata(value)
        with self.assertRaises(ValueError):
            distance_objective_class('typo')
        with self.assertRaises(RuntimeError):
            validate_distance_metadata(distance_metadata('legacy'), expected=distance_metadata('batched'))
        self.assertEqual(validate_distance_metadata({}, allow_historical=True), distance_metadata('legacy'))

    def test_sampler_model_worker_and_manifest_wiring(self):
        import sample
        from discrete_diffusion.diffusion_transformer import DiffusionTransformer
        from tools.ablation_common import projector
        from tools.run_two_city_distance_v2 import DistanceStudy, REVISION, validate_trace
        from tests.test_distance_study import trace_payload
        self.assertEqual(sample.parser.parse_args([]).distance_backend, 'legacy')
        self.assertEqual(sample.parser.parse_args(['--distance_backend', 'batched']).distance_backend, 'batched')
        self.assertEqual(inspect.signature(DiffusionTransformer).parameters['distance_backend'].default, 'legacy')
        dd = SimpleNamespace(num_classes=11, type_classes=2, num_spectial=4)
        self.assertEqual(projector(dd, 'full', distance_backend='batched').distance_backend, 'batched')
        with self.assertRaisesRegex(ValueError, 'run-id'):
            DistanceStudy(SimpleNamespace(distance_backend='batched', run_id=REVISION))
        payload = trace_payload('full')
        payload['job'].update(distance_metadata('batched'))
        payload['result'].update(distance_metadata('batched'))
        for call in payload['batch_traces'][0]['projection_stats']:
            call.update(distance_metadata('batched'))
        validate_trace(payload, 'train-only')
        payload['result'].update(distance_metadata('legacy'))
        with self.assertRaises(RuntimeError):
            validate_trace(payload, 'train-only')


if __name__ == '__main__':
    unittest.main()
