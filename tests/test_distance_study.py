"""Protocol-level regression checks for the fresh two-city controller."""
import copy
from pathlib import Path
import tempfile
import unittest

from tools.ablation_common import DISTANCE_VARIANTS, DISTANCE_VERSION, STEPS
from tools.run_two_city_distance_v2 import ALL_VARIANTS, expected_keys, validate_trace
from tools.unified_tables import render


def trace_payload(variant):
    setting = DISTANCE_VARIANTS.get(variant)
    enabled = bool(setting and setting['distance'])
    settings = None if setting is None else dict(
        token_kl_weight=setting['kl'], distance_kl_weight=setting['distance'],
        order_weight=setting['order'], existence_weight=setting['existence'],
        gumbel=setting['gumbel'], update_multipliers=setting['update'],
        distance_paths=8, distance_topk=32, distance_bins=32,
        distance_temperature=1, outer=10, inner=50, early_stop=False)
    calls = []
    for step in STEPS if setting else []:
        call = dict(diffusion_step=step, optimizer_steps=500)
        if enabled:
            call.update(distance_reference_sha256='train-only',
                        distance_estimator='straight-through-mc' if setting['gumbel'] else 'expected-route',
                        distance_kl=0.2, distance_poi_gradient_norm=0.1, distance_active_rows=2)
        calls.append(call)
    return dict(job=dict(projection_revision=DISTANCE_VERSION, variant=variant),
                result=dict(projection_revision=DISTANCE_VERSION, projection_settings=settings,
                            optimizer_steps=500 * len(calls)),
                batch_traces=[dict(global_rng_unchanged=True, effective_constraints=True,
                                   projection_stats=calls,
                                   distance_rng_before='before' if enabled else None,
                                   distance_rng_after=('after' if setting['gumbel'] else 'before') if enabled else None)])


class DistanceStudyTests(unittest.TestCase):
    def test_exactly_66_unique_results_and_new_distance_ablation(self):
        self.assertEqual(len(ALL_VARIANTS), 11)
        self.assertEqual(len(expected_keys()), 66)
        self.assertIn('no_distance_kl', ALL_VARIANTS)

    def test_all_sampling_traces_and_estimators(self):
        for variant in ALL_VARIANTS:
            if variant == 'postswap':
                continue
            with self.subTest(variant=variant):
                values = validate_trace(trace_payload(variant), 'train-only')
                self.assertEqual(bool(values), bool(DISTANCE_VARIANTS.get(variant, {}) and
                                                      DISTANCE_VARIANTS[variant]['distance']))

    def test_bad_rng_budget_reference_and_switches_rejected(self):
        original = trace_payload('full')
        mutations = [
            lambda p: p['batch_traces'][0].update(global_rng_unchanged=False),
            lambda p: p['batch_traces'][0]['projection_stats'][0].update(optimizer_steps=499),
            lambda p: p['batch_traces'][0]['projection_stats'][0].update(distance_reference_sha256='test-data'),
            lambda p: p['result']['projection_settings'].update(distance_kl_weight=0),
            lambda p: p['batch_traces'][0].update(distance_rng_after='before'),
        ]
        for mutate in mutations:
            payload = copy.deepcopy(original)
            mutate(payload)
            with self.assertRaises(RuntimeError):
                validate_trace(payload, 'train-only')

    def test_initial_report_declares_full_66_result_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render(root, {}, dict(projection_revision=DISTANCE_VERSION,
                                  variants=DISTANCE_VARIANTS, total_unique_results=66))
            self.assertIn('0/66', (root / 'report.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
