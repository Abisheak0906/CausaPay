import math
import numpy as np
import unittest

from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation
from evaluation.repeated_seed_study import assignment_probabilities, conditional_potential_means
from causal_engine.direct_cate_estimator import DirectMultiArmCateEstimator


class CausalDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observed, hidden = build_demo_evaluation(seed=19, num_events=240)
        cls.engine = CausaPayEvaluationEngine()
        cls.engine.fit_from_batch(observed, hidden, seed=19)

    def test_overlap_has_all_treatments_and_finite_values(self):
        report = self.engine.aipw_estimator.propensity_overlap_diagnostics()
        self.assertEqual({row['treatment'] for row in report['groups']}, {'none', 'retry', 'whatsapp'})
        for row in report['groups']:
            self.assertGreater(row['assigned_count'], 0)
            self.assertTrue(math.isfinite(row['all_rows']['mean']))
            self.assertGreater(len(row['histogram']), 0)

    def test_uncertainty_threshold_math_matches_policy(self):
        report = self.engine.aipw_estimator.uncertainty_diagnostics(self.engine.test_obs)
        self.assertEqual(set(report), {'retry', 'whatsapp'})
        for action in report:
            self.assertTrue(math.isfinite(report[action]['median']))
            self.assertGreater(len(report[action]['histogram']), 0)
        decisions = self.engine.incrementality_policy.predict(self.engine.test_obs)
        self.assertEqual(len(decisions), len(self.engine.test_obs))
        self.assertTrue((decisions.loc[~decisions['is_abstain'], 'fallback_action'].isna()).all())

    def test_policy_comparison_contains_non_causal_comparator(self):
        results = {row['policy_name']: row for row in self.engine.benchmark_results}
        naive = results['Naive likelihood targeting (matched volume)']
        self.assertTrue(math.isfinite(naive['policy_value']))
        self.assertGreaterEqual(naive['intervention_cost'], 0)
        self.assertEqual(sum(naive['action_distribution'].values()), len(self.engine.test_obs))

    def test_naive_volume_matches_actual_causal_actions(self):
        decisions = self.engine.incrementality_policy.predict(self.engine.test_obs)
        causal_volume = int(decisions['recommended_action'].isin(['retry', 'whatsapp']).sum())
        naive = next(row for row in self.engine.benchmark_results if row['policy_name'].startswith('Naive'))
        naive_volume = sum(naive['action_distribution'].get(action, 0) for action in ('retry', 'whatsapp'))
        self.assertEqual(naive_volume, causal_volume)

    def test_baseline_never_recommends_whatsapp_without_consent(self):
        rows = self.engine.test_obs.copy()
        rows['whatsapp_opted_in'] = False
        self.assertNotIn('whatsapp', set(self.engine.baseline_policy.predict(rows)))

    def test_conditional_truth_integrator_has_valid_counterfactual_ordering(self):
        observed, _ = build_demo_evaluation(seed=23, num_events=40)
        means = conditional_potential_means(observed, seed=23)
        self.assertEqual(means.shape, (40, 3))
        self.assertTrue(np.isfinite(means).all())
        self.assertTrue(((means >= 0) & (means <= 1)).all())
        # The simulator has non-negative intervention effects by construction.
        self.assertTrue((means[:, 1] >= means[:, 0]).all())
        self.assertTrue((means[:, 2] >= means[:, 0]).all())

    def test_known_assignment_probability_matches_positivity_floor(self):
        observed, _ = build_demo_evaluation(seed=29, num_events=40)
        propensity = assignment_probabilities(observed)
        self.assertTrue(np.allclose(propensity.sum(axis=1), 1.0))
        self.assertGreaterEqual(float(propensity.min()), 0.05)

    def test_v1_and_v2_train_with_finite_none_referenced_effects(self):
        observed, hidden = build_demo_evaluation(seed=31, num_events=180)
        v1 = CausaPayEvaluationEngine(estimator_version='v1')
        v2 = CausaPayEvaluationEngine(estimator_version='v2')
        v1.fit_from_batch(observed, hidden, seed=31)
        v2.fit_from_batch(observed, hidden, seed=31)
        for engine in (v1, v2):
            cf = engine.aipw_estimator.predict_counterfactuals(engine.test_obs)
            self.assertTrue(np.isfinite(cf[['prob_none', 'prob_retry', 'prob_whatsapp']].to_numpy()).all())
            self.assertEqual(len(cf), len(engine.test_obs))
            decisions = engine.incrementality_policy.predict(engine.test_obs)
            opted_out = ~engine.test_obs['whatsapp_opted_in'].astype(bool).to_numpy()
            self.assertFalse((decisions.loc[opted_out, 'recommended_action'] == 'whatsapp').any())

    def test_v2_reproducible_and_honors_clip_configuration(self):
        observed, hidden = build_demo_evaluation(seed=37, num_events=180)
        train = observed.iloc[:126]
        test = observed.iloc[126:]
        left = DirectMultiArmCateEstimator(clip_threshold=.12, random_state=37).fit(train)
        right = DirectMultiArmCateEstimator(clip_threshold=.12, random_state=37).fit(train)
        self.assertEqual(left.clip_threshold, .12)
        self.assertTrue(np.allclose(left.predict_counterfactuals(test)[['prob_none', 'prob_retry', 'prob_whatsapp']], right.predict_counterfactuals(test)[['prob_none', 'prob_retry', 'prob_whatsapp']]))


if __name__ == '__main__':
    unittest.main()
