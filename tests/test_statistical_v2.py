import math
import unittest
import numpy as np

from evaluations.statistical_metrics import (Get_Statistical_Metrics, STATISTICAL_NAMES,
    require_evaluation_version, temporal_category_metrics)


def seq(marks, times=None):
    return dict(marks=marks, arrival_times=times if times is not None else list(range(len(marks))),
                checkins=marks, gps=[[0., i * .01] for i in range(len(marks))])


class StatisticalV2Tests(unittest.TestCase):
    def test_hour_boundaries_and_singleton(self):
        detail = {}
        real = [seq([4, 5, 4], [0., 1., 23.999]), seq([5], [0.5])]
        values = Get_Statistical_Metrics(real, real, diagnostics=detail)
        self.assertEqual(values['Category'], 0)
        self.assertEqual(detail['category_hourly'][0]['real_count'], 2)
        self.assertEqual(detail['category_hourly'][1]['real_count'], 1)
        self.assertEqual(detail['category_hourly'][23]['real_count'], 1)
        self.assertNotIn('Interval', values)
        require_evaluation_version(values)

    def test_same_marginal_different_hours(self):
        result = temporal_category_metrics([seq([4, 5], [1, 2])], [seq([5, 4], [1, 2])])
        self.assertAlmostEqual(result['Category'], math.log(2) * 2 / 24)

    def test_empty_hours_and_all_empty(self):
        result = temporal_category_metrics([seq([4], [5])], [seq([])])
        self.assertAlmostEqual(result['Category'], math.log(2) / 24)
        self.assertTrue(math.isnan(result['CategoryTransition']))
        self.assertTrue(math.isnan(Get_Statistical_Metrics([], [])['Category']))

    def test_transitions_conditioned_and_self_edges(self):
        detail = {}
        result = temporal_category_metrics([seq([4, 4, 5])], [seq([4, 5, 4])], detail)
        self.assertGreater(result['CategoryTransition'], 0)
        self.assertEqual(detail['category_transition_counts']['4']['real'], 2)
        self.assertAlmostEqual(detail['category_transition_rows']['5'], math.log(2))

    def test_no_cross_sequence_edges_and_identical(self):
        result = temporal_category_metrics([seq([4]), seq([5])], [seq([4]), seq([5])])
        self.assertTrue(math.isnan(result['CategoryTransition']))
        self.assertEqual(temporal_category_metrics([seq([4, 5])], [seq([4, 5])])['CategoryTransition'], 0)

    def test_version_and_total_exclude_metadata(self):
        values = Get_Statistical_Metrics([seq([4, 5])], [seq([5, 4])])
        self.assertAlmostEqual(values['totalJSD'], sum(values[k] for k in STATISTICAL_NAMES if np.isfinite(values[k])))
        self.assertTrue(math.isnan(values['Distance']))  # Historical endpoint filter is unchanged.
        with self.assertRaisesRegex(ValueError, 'version'):
            require_evaluation_version(dict(Category=0, Interval=0))
        with self.assertRaises(ValueError):
            temporal_category_metrics([seq([4], [24])], [seq([4])])


if __name__ == '__main__':
    unittest.main()
