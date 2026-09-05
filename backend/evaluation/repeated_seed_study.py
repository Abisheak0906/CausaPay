"""Pre-specified synthetic study: unchanged CausaPay stack across seeds 0--9.

Run from backend:
    .venv\\Scripts\\python.exe -m evaluation.repeated_seed_study

It writes deterministic CSV/JSON artifacts under evaluation/results/.  The
study does not alter the simulator, models, threshold, policies, or evaluator.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation
from evaluation.evaluator import evaluate_policy

SEEDS = tuple(range(10))
NUM_EVENTS = 3_000
MONTE_CARLO_DRAWS = 512
TREATMENTS = ('none', 'retry', 'whatsapp')


def assignment_probabilities(frame: pd.DataFrame) -> np.ndarray:
    """Exact known assignment mechanism from simulator.bias; diagnostic only."""
    n = len(frame)
    none = np.zeros(n)
    retry = (frame['failure_context'].eq('insufficient_funds').to_numpy(dtype=float) * 1.2
             + frame['historical_failure_count'].to_numpy(dtype=float) * .1)
    whatsapp = (np.full(n, -2.0)
                + frame['plan_tier'].eq('Enterprise').to_numpy(dtype=float) * 1.5
                + frame['engagement_score'].to_numpy(dtype=float) * 2.0
                - (~frame['whatsapp_opted_in'].astype(bool)).to_numpy(dtype=float) * 10.0)
    none -= frame['plan_tier'].eq('Enterprise').to_numpy(dtype=float)
    logits = np.column_stack([none, retry, whatsapp])
    raw = np.exp(logits - logits.max(axis=1, keepdims=True))
    raw /= raw.sum(axis=1, keepdims=True)
    return raw * .85 + .05


def conditional_potential_means(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Monte-Carlo integration of the *unchanged* DGP over hidden variables.

    This is the conditional mean E[P(Y(a)=1) | observed X], not each row's
    inaccessible individual hidden potential probability.
    """
    rng = np.random.default_rng(100_000 + seed)
    n, draws = len(frame), MONTE_CARLO_DRAWS
    z = rng.normal(size=draws)
    retry_affinity = rng.beta(2, 5, size=draws)
    wa_beta = rng.beta(2, 8, size=draws)
    tech_base_noise = rng.normal(.7, .1, size=draws)
    tech_retry_scale = rng.normal(.05, .02, size=draws)
    tech_wa_scale = rng.normal(.01, .01, size=draws)
    output = np.empty((n, 3), dtype=float)
    for start in range(0, n, 500):
        part = frame.iloc[start:start + 500]
        e = part['engagement_score'].to_numpy()[:, None]
        enterprise = part['plan_tier'].eq('Enterprise').to_numpy()[:, None] * .1
        pro = part['plan_tier'].eq('Pro').to_numpy()[:, None] * .05
        plan = enterprise + pro
        tech = part['failure_context'].eq('technical_timeout').to_numpy()[:, None]
        opted = part['whatsapp_opted_in'].astype(bool).to_numpy()[:, None]
        base_tech = np.clip(tech_base_noise[None, :] + plan, .01, .95)
        base_funds = np.clip(.2 + .1 * e + .05 * z[None, :] + plan, .01, .95)
        base = np.where(tech, base_tech, base_funds)
        retry_tech = tech_retry_scale[None, :] * retry_affinity[None, :]
        retry_funds = (.1 + .1 * (z[None, :] > 0)) * retry_affinity[None, :]
        retry = np.clip(base + np.where(tech, retry_tech, retry_funds), 0, .99)
        wa_affinity = (wa_beta[None, :] + .3 * e) * opted
        wa_tech = tech_wa_scale[None, :] * wa_affinity
        wa_funds = (.15 + .2 * e) * wa_affinity
        whatsapp = np.clip(base + np.where(tech, wa_tech, wa_funds), 0, .99)
        output[start:start + len(part)] = np.column_stack([base.mean(1), retry.mean(1), whatsapp.mean(1)])
    return output


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict:
    estimate, truth = np.asarray(estimate, dtype=float), np.asarray(truth, dtype=float)
    correlation = float(np.corrcoef(estimate, truth)[0, 1]) if estimate.std() and truth.std() else None
    slope = float(np.polyfit(estimate, truth, 1)[0]) if estimate.std() else None
    return {'mae': float(np.mean(np.abs(estimate - truth))), 'bias': float(np.mean(estimate - truth)), 'correlation': correlation, 'calibration_slope': slope}


def run() -> tuple[pd.DataFrame, dict]:
    warnings.filterwarnings('ignore', category=FutureWarning)
    policy_rows, diagnostic_rows = [], []
    for seed in SEEDS:
        observed, hidden = build_demo_evaluation(seed=seed, num_events=NUM_EVENTS)
        engine = CausaPayEvaluationEngine()
        engine.fit_from_batch(observed, hidden, seed=seed)
        train = engine.train_obs.reset_index(drop=True)
        test, truth = engine.test_obs.reset_index(drop=True), engine.test_hid.reset_index(drop=True)
        decision = engine.incrementality_policy.predict(test).reset_index(drop=True)
        actions = {
            'consent_safe_baseline': engine.baseline_policy.predict(test),
            'causapay': decision['recommended_action'].to_numpy(),
            'oracle': engine.oracle_policy.predict(test, truth)['recommended_action'].to_numpy(),
        }
        actual_volume = int(np.isin(actions['causapay'], ['retry', 'whatsapp']).sum())
        actions['naive_likelihood_matched_volume'] = engine.naive_policy.predict(test, actual_volume)
        for name, action in actions.items():
            result = evaluate_policy(name, action, test, truth)
            policy_rows.append({'seed': seed, 'policy': name, 'policy_value': result['policy_value'], 'gross_recovered': result['gross_recovered'], 'true_incremental_recovered': result['true_incremental_recovered'], 'intervention_cost': result['intervention_cost'], 'interventions': int(np.isin(action, ['retry', 'whatsapp']).sum()), 'retry': int((action == 'retry').sum()), 'whatsapp': int((action == 'whatsapp').sum()), 'none': int((action == 'none').sum())})

        true_prop = assignment_probabilities(train)
        oof_prop = engine.aipw_estimator._oof_propensity
        oof_outcome = engine.aipw_estimator._oof_outcome_predictions
        true_train = conditional_potential_means(train, seed)
        true_test = conditional_potential_means(test, seed)
        pred_test = engine.aipw_estimator.predict_counterfactuals(test).reset_index(drop=True)
        for index, action in enumerate(TREATMENTS):
            diagnostic_rows.append({'seed': seed, 'stage': 'propensity_oof', 'treatment': action, **metrics(oof_prop[:, index], true_prop[:, index])})
            diagnostic_rows.append({'seed': seed, 'stage': 'outcome_nuisance_oof', 'treatment': action, **metrics(oof_outcome[:, index], true_train[:, index])})
        for action in ('retry', 'whatsapp'):
            estimate = pred_test[f'prob_{action}'].to_numpy() - pred_test['prob_none'].to_numpy()
            true_effect = true_test[:, TREATMENTS.index(action)] - true_test[:, 0]
            diagnostic_rows.append({'seed': seed, 'stage': 'observable_cate_holdout', 'treatment': action, **metrics(estimate, true_effect)})
        diagnostic_rows.append({'seed': seed, 'stage': 'abstention', 'treatment': 'combined', 'rate': float(decision['is_abstain'].mean()), 'count': int(decision['is_abstain'].sum())})

    policies = pd.DataFrame(policy_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = policies.groupby('policy')['policy_value'].agg(['mean', 'median', 'std', 'min', 'max']).to_dict('index')
    pivot = policies.pivot(index='seed', columns='policy', values='policy_value')
    summary['win_rate_vs_naive'] = {policy: float((pivot[policy] > pivot['naive_likelihood_matched_volume']).mean()) for policy in pivot.columns if policy != 'naive_likelihood_matched_volume'}
    diagnostic_summary = {
        f'{stage}:{treatment}': values
        for (stage, treatment), values in diagnostics.groupby(['stage', 'treatment']).mean(numeric_only=True).to_dict('index').items()
    }
    return policies, {'study': {'seeds': list(SEEDS), 'num_events_per_seed': NUM_EVENTS, 'train_fraction': .7, 'monte_carlo_draws_for_conditional_truth': MONTE_CARLO_DRAWS}, 'policy_summary': summary, 'diagnostic_summary': diagnostic_summary}, diagnostics


def write_artifacts(policy: pd.DataFrame, summary: dict, diagnostics: pd.DataFrame) -> Path:
    """Write study output; also usable to repair/re-render existing raw CSVs."""
    destination = Path(__file__).parent / 'results'
    destination.mkdir(exist_ok=True)
    policy.to_csv(destination / 'repeated_seed_policy_results.csv', index=False)
    diagnostics.to_csv(destination / 'repeated_seed_estimator_diagnostics.csv', index=False)
    (destination / 'repeated_seed_study_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    policy_summary = pd.DataFrame({key: value for key, value in summary['policy_summary'].items() if key != 'win_rate_vs_naive'}).T.reset_index(names='policy')
    win_rates = pd.DataFrame(summary['policy_summary'].get('win_rate_vs_naive', {}).items(), columns=['policy', 'win_rate_vs_naive'])
    diagnostic_summary = pd.DataFrame(summary['diagnostic_summary']).T.drop(columns=['seed'], errors='ignore').reset_index(names='diagnostic')
    def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
        headers = [str(column) for column in frame.columns]
        rows = []
        for row in frame.itertuples(index=False, name=None):
            rows.append(['' if pd.isna(value) else f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value) for value in row])
        return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)

    markdown = '# Pre-specified repeated-seed study\n\n'
    markdown += 'Seeds 0–9; 3,000 events/seed; 70/30 train/held-out split; unchanged simulator, policy, costs, threshold, and model architecture. Conditional truth is Monte-Carlo integration over the simulator\'s hidden variables (512 draws).\n\n'
    markdown += '## Policy value (realized synthetic holdout)\n\n' + markdown_table(policy_summary, digits=2)
    markdown += '\n\n## Win rate versus matched-volume naive likelihood targeting\n\n' + markdown_table(win_rates, digits=2)
    markdown += '\n\n## Estimator diagnostics (mean across seeds)\n\n' + markdown_table(diagnostic_summary, digits=4) + '\n'
    (destination / 'repeated_seed_study_report.md').write_text(markdown, encoding='utf-8')
    return destination


if __name__ == '__main__':
    policy, summary, diagnostics = run()
    write_artifacts(policy, summary, diagnostics)
    print(json.dumps(summary, indent=2, sort_keys=True))
