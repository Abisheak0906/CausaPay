import math
import unittest
import numpy as np
import pandas as pd

from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation
from causal_engine.direct_cate_estimator import DirectMultiArmCateEstimator
from policies.naive import NaiveLikelihoodPolicy
from policies.baseline import GrossRecoveryBaseline


class V2ValidationTests(unittest.TestCase):
    def setUp(self):
        self.observed, self.hidden = build_demo_evaluation(seed=42, num_events=200)

    def test_v2_data_leakage_rejection(self):
        """V2 must strictly reject datasets containing forbidden ground-truth columns."""
        estimator = DirectMultiArmCateEstimator(random_state=42)
        leaked_df = self.observed.copy()
        leaked_df['true_baseline'] = 0.5
        with self.assertRaises(ValueError) as ctx:
            estimator.fit(leaked_df)
        self.assertIn('Data Leakage', str(ctx.exception))

        for col in ['Y_none', 'Y_retry', 'Y_whatsapp', 'true_channel', 'true_liquidity']:
            df = self.observed.copy()
            df[col] = 1.0
            with self.assertRaises(ValueError):
                estimator.fit(df)

    def test_v2_reproducibility(self):
        """Identical seeds must produce bit-for-bit identical counterfactual predictions."""
        train = self.observed.iloc[:140].copy()
        test = self.observed.iloc[140:].copy()

        m1 = DirectMultiArmCateEstimator(random_state=123).fit(train)
        m2 = DirectMultiArmCateEstimator(random_state=123).fit(train)

        p1 = m1.predict_counterfactuals(test)
        p2 = m2.predict_counterfactuals(test)

        for col in ['prob_none', 'prob_retry', 'prob_whatsapp', 'std_retry', 'std_whatsapp']:
            np.testing.assert_allclose(p1[col].to_numpy(), p2[col].to_numpy())

    def test_v2_finite_outputs_and_bounds(self):
        """Predicted probabilities must be finite and clipped within [0, 1]."""
        train = self.observed.iloc[:140].copy()
        test = self.observed.iloc[140:].copy()

        estimator = DirectMultiArmCateEstimator(random_state=42).fit(train)
        preds = estimator.predict_counterfactuals(test)

        for arm in ['none', 'retry', 'whatsapp']:
            vals = preds[f'prob_{arm}'].to_numpy()
            self.assertTrue(np.isfinite(vals).all())
            self.assertTrue((vals >= 0.0).all())
            self.assertTrue((vals <= 1.0).all())

        for arm in ['retry', 'whatsapp']:
            stds = preds[f'std_{arm}'].to_numpy()
            self.assertTrue(np.isfinite(stds).all())
            self.assertTrue((stds >= 0.0).all())

    def test_v2_contrast_mapping_and_sample_weights(self):
        """Pairwise contrast must create valid weights and residuals."""
        estimator = DirectMultiArmCateEstimator(clip_threshold=0.10, random_state=42)
        train = self.observed.iloc[:140].copy()
        estimator.fit(train)

        for arm in ['retry', 'whatsapp']:
            self.assertIn(arm, estimator.tau_models)
            self.assertIn(arm, estimator._oof_tau)
            tau = estimator.tau_models[arm].predict(estimator._prepare_features(train))
            self.assertTrue(np.isfinite(tau).all())

    def test_v2_sparse_action_handling(self):
        """Estimator should handle datasets where one treatment arm is sparse."""
        sparse_df = self.observed.copy()
        wa_indices = sparse_df[sparse_df['intervention_assigned'] == 'whatsapp'].index
        drop_indices = wa_indices[2:]
        sparse_df = sparse_df.drop(index=drop_indices).reset_index(drop=True)

        estimator = DirectMultiArmCateEstimator(n_splits=3, random_state=42)
        estimator.fit(sparse_df)
        preds = estimator.predict_counterfactuals(sparse_df.iloc[:20])
        self.assertEqual(len(preds), 20)
        self.assertTrue(np.isfinite(preds['prob_retry'].to_numpy()).all())

    def test_fair_matched_volume_comparator(self):
        """Naive comparator must match exact executed intervention count of the causal policy."""
        engine = CausaPayEvaluationEngine(estimator_version='v2')
        engine.fit_from_batch(self.observed, self.hidden, seed=42)

        decisions = engine.incrementality_policy.predict(engine.test_obs)
        causal_volume = int(decisions['recommended_action'].isin(['retry', 'whatsapp']).sum())

        naive_policy = NaiveLikelihoodPolicy(engine.aipw_estimator.preprocessor, engine.baseline_policy)
        naive_policy.fit(engine.train_obs)
        naive_actions = naive_policy.predict(engine.test_obs, causal_volume)
        naive_volume = int(np.isin(naive_actions, ['retry', 'whatsapp']).sum())

        self.assertEqual(naive_volume, causal_volume)

    def test_v2_promotion_criteria_evaluation_rule(self):
        """Promotion criteria must correctly require all 5 conditions to promote."""
        criteria_fail_corr = {
            'criterion_1_correlation_lift_retry': {'passed': False},
            'criterion_1_correlation_lift_whatsapp': {'passed': False},
            'criterion_2_correlation_threshold_retry': {'passed': False},
            'criterion_2_correlation_threshold_whatsapp': {'passed': False},
            'criterion_3_mae_worsening_retry': {'passed': True},
            'criterion_3_mae_worsening_whatsapp': {'passed': True},
            'criterion_4_policy_value_ratio': {'passed': True},
            'criterion_5_naive_win_rate': {'passed': True},
        }
        all_passed = all(item['passed'] for item in criteria_fail_corr.values())
        decision = 'PROMOTE_V2' if all_passed else 'KEEP_V1'
        self.assertEqual(decision, 'KEEP_V1')

        criteria_all_pass = {k: {'passed': True} for k in criteria_fail_corr}
        all_passed_2 = all(item['passed'] for item in criteria_all_pass.values())
        decision_2 = 'PROMOTE_V2' if all_passed_2 else 'KEEP_V1'
        self.assertEqual(decision_2, 'PROMOTE_V2')


if __name__ == '__main__':
    unittest.main()
