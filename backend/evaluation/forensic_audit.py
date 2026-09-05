"""Reproducible forensic snapshot of the synthetic held-out policy comparison."""
import json
import warnings

import numpy as np
import pandas as pd

from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation
from evaluation.evaluator import evaluate_policy


def run(seed: int = 42, num_events: int = 15000) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    observed, hidden = build_demo_evaluation(seed=seed, num_events=num_events)
    engine = CausaPayEvaluationEngine()
    engine.fit_from_batch(observed, hidden, seed=seed)
    test, truth = engine.test_obs.reset_index(drop=True), engine.test_hid.reset_index(drop=True)
    decisions = engine.incrementality_policy.predict(test).reset_index(drop=True)
    baseline = engine.baseline_policy.predict(test)
    actual_volume = int(decisions['recommended_action'].isin(['retry', 'whatsapp']).sum())
    naive = engine.naive_policy.predict(test, actual_volume)
    oracle = engine.oracle_policy.predict(test, truth)['recommended_action'].to_numpy()
    policies = {'baseline': baseline, 'naive': naive, 'causapay': decisions['recommended_action'].to_numpy(), 'oracle': oracle}
    result = {'seed': seed, 'holdout_rows': len(test), 'causapay_actual_interventions': actual_volume, 'policies': {}}
    for name, actions in policies.items():
        metrics = evaluate_policy(name, actions, test, truth)
        result['policies'][name] = {
            'policy_value': metrics['policy_value'],
            'cost': metrics['intervention_cost'],
            'actions': {key: int(value) for key, value in metrics['action_distribution'].items()},
            'oracle_action_agreement': float(np.mean(actions == oracle)),
        }
    result['abstentions'] = int(decisions['is_abstain'].sum())
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
