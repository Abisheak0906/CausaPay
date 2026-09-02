from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from causal_engine.aipw_estimator import AIPWEstimator
from evaluation.evaluator import evaluate_policy
from policies.baseline import GrossRecoveryBaseline
from policies.incrementality import IncrementalityAwarePolicy
from policies.oracle import OraclePolicy
from simulator.bias import assign_historical_treatments
from simulator.generator import SimulatorConfig, generate_events


DEFAULT_COSTS = {"none": 0.0, "retry": 2.0, "whatsapp": 15.0}
EXTERNAL_REQUIRED_FIELDS = [
    'event_id', 'customer_id', 'amount', 'plan_tier', 'payment_method',
    'failure_context', 'decline_signal_bucket', 'engagement_score',
    'historical_failure_count', 'whatsapp_opted_in',
]
VALID_ACTIONS = ['none', 'retry', 'whatsapp']


def generate_reproducible_batch(
    *,
    seed: int = 42,
    num_events: int = 15000,
    train_fraction: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic held-out evaluation batch and split it train/test."""
    config = SimulatorConfig(seed=seed, num_events=num_events)
    observables, hidden = generate_events(config)
    biased = assign_historical_treatments(observables, hidden, seed=seed)
    train_size = int(train_fraction * len(biased))
    train_obs = biased.iloc[:train_size].copy()
    test_obs = biased.iloc[train_size:].copy()
    test_hid = hidden.iloc[train_size:].copy()
    return train_obs, test_obs, test_hid, biased


@dataclass
class CausaPayEvaluationEngine:
    """Clean boundary between the synthetic data source and the causal evaluation logic."""

    uncertainty_threshold: float = 0.20
    costs: Dict[str, float] = field(default_factory=lambda: DEFAULT_COSTS.copy())
    aipw_estimator: Optional[AIPWEstimator] = None
    baseline_policy: Optional[GrossRecoveryBaseline] = None
    incrementality_policy: Optional[IncrementalityAwarePolicy] = None
    oracle_policy: Optional[OraclePolicy] = None
    train_obs: Optional[pd.DataFrame] = None
    test_obs: Optional[pd.DataFrame] = None
    test_hid: Optional[pd.DataFrame] = None
    benchmark_results: list = field(default_factory=list)
    events_df: Optional[pd.DataFrame] = None
    full_events: Optional[pd.DataFrame] = None

    @staticmethod
    def _strip_hidden(df: pd.DataFrame) -> pd.DataFrame:
        hidden = [c for c in df.columns if c.startswith("true_") or c.startswith("Y_")]
        return df.drop(columns=hidden, errors="ignore")

    def evaluate_observable_batch(self, observables: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a plain observable batch using the fitted demo model when no hidden ground truth is supplied."""
        if self.aipw_estimator is None or self.baseline_policy is None or self.incrementality_policy is None:
            raise ValueError('A fitted demo model is required to evaluate observable-only batches.')

        observables = observables.copy()
        if 'event_id' not in observables.columns:
            observables['event_id'] = [f'evt_{i}' for i in range(len(observables))]

        baseline_actions = self.baseline_policy.predict(observables)
        inc_results = self.incrementality_policy.predict(observables)
        events = observables.reset_index(drop=True)
        events['baseline_action'] = baseline_actions
        events['recommended_action'] = inc_results['recommended_action'].values
        events['is_abstain'] = inc_results['is_abstain'].values
        events['prob_none'] = inc_results['prob_none'].values
        events['prob_retry'] = inc_results['prob_retry'].values
        events['prob_whatsapp'] = inc_results['prob_wa'].values
        events['inc_prob_retry'] = inc_results['inc_prob_retry'].values
        events['inc_prob_whatsapp'] = inc_results['inc_prob_wa'].values
        events['eniv_none'] = 0.0
        events['eniv_retry'] = inc_results['eniv_retry'].values
        events['eniv_whatsapp'] = inc_results['eniv_wa'].values
        events['eniv_wa'] = events['eniv_whatsapp']

        self.full_events = events
        self.events_df = self._strip_hidden(events)

        return {'events_df': self.events_df, 'full_events': self.full_events}

    def fit_from_batch(
        self,
        observables: pd.DataFrame,
        hidden: Optional[pd.DataFrame] = None,
        *,
        train_fraction: float = 0.7,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Train the policy stack on a batch of observable events and hidden outcomes."""
        if hidden is not None and 'intervention_assigned' not in observables.columns:
            biased = assign_historical_treatments(observables, hidden, seed=seed)
        else:
            biased = observables.copy()

        if len(biased) < 2:
            raise ValueError('Evaluation batch must contain at least two events.')

        train_size = max(1, int(train_fraction * len(biased)))
        if train_size >= len(biased):
            train_size = len(biased) - 1

        self.train_obs = biased.iloc[:train_size].copy()
        self.test_obs = biased.iloc[train_size:].copy()
        self.test_hid = hidden.iloc[train_size:].copy() if hidden is not None else None

        self.aipw_estimator = AIPWEstimator()
        self.aipw_estimator.fit(self.train_obs)

        self.baseline_policy = GrossRecoveryBaseline()
        self.baseline_policy.fit(self.train_obs)

        self.incrementality_policy = IncrementalityAwarePolicy(
            self.aipw_estimator,
            self.baseline_policy,
        )
        self.incrementality_policy.uncertainty_threshold = self.uncertainty_threshold

        self.oracle_policy = OraclePolicy()

        baseline_actions = self.baseline_policy.predict(self.test_obs)
        inc_results = self.incrementality_policy.predict(self.test_obs)
        inc_actions = inc_results['recommended_action'].values

        if self.test_hid is None:
            raise ValueError('Hidden ground-truth outcomes are required to evaluate policy performance.')

        oracle_results = self.oracle_policy.predict(self.test_obs, self.test_hid)
        oracle_actions = oracle_results['recommended_action'].values

        res_baseline = evaluate_policy(
            'Baseline (Gross Recovery)',
            baseline_actions,
            self.test_obs,
            self.test_hid,
        )
        res_inc = evaluate_policy(
            'Incrementality-Aware',
            inc_actions,
            self.test_obs,
            self.test_hid,
        )
        res_oracle = evaluate_policy(
            'Oracle',
            oracle_actions,
            self.test_obs,
            self.test_hid,
        )
        self.benchmark_results = [res_baseline, res_inc, res_oracle]

        cf = self.aipw_estimator.predict_counterfactuals(self.test_obs)
        events = self.test_obs.reset_index(drop=True)
        cf = cf.reset_index(drop=True)
        events = pd.concat([events, cf.drop(columns=['event_id'])], axis=1)
        events['baseline_action'] = baseline_actions
        events['recommended_action'] = inc_results['recommended_action'].values
        events['is_abstain'] = inc_results['is_abstain'].values
        amount = events['amount'].values
        inc_prob_retry = events['prob_retry'] - events['prob_none']
        inc_prob_wa = events['prob_whatsapp'] - events['prob_none']
        events['inc_prob_retry'] = inc_prob_retry
        events['inc_prob_whatsapp'] = inc_prob_wa
        events['eniv_none'] = 0.0
        events['eniv_retry'] = inc_prob_retry * amount - self.costs['retry']
        events['eniv_whatsapp'] = inc_prob_wa * amount - self.costs['whatsapp']
        events['eniv_wa'] = events['eniv_whatsapp']

        self.full_events = events
        self.events_df = self._strip_hidden(events)

        return {
            'train_obs': self.train_obs,
            'test_obs': self.test_obs,
            'test_hid': self.test_hid,
            'benchmark_results': self.benchmark_results,
            'events_df': self.events_df,
            'full_events': self.full_events,
        }


def build_demo_evaluation(seed: int = 42, num_events: int = 15000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility helper for the API/demo startup path. Returns observable and hidden data."""
    config = SimulatorConfig(seed=seed, num_events=num_events)
    observables, hidden = generate_events(config)
    biased = assign_historical_treatments(observables, hidden, seed=seed)
    return biased, hidden
