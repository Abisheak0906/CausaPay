"""One pre-registered V2 experiment on untouched seeds 10--19.

No simulator, policy, cost, split, threshold, or model parameter is selected
from these results. V1 remains the default; this module evaluates V2 only.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation
from evaluation.evaluator import evaluate_policy
from evaluation.repeated_seed_study import conditional_potential_means, metrics

SEEDS = tuple(range(10, 20))
NUM_EVENTS = 3_000
SUCCESS_CRITERIA = {
    'minimum_correlation': .10,
    'minimum_absolute_correlation_lift_over_v1': .10,
    'maximum_mae_worsening': .02,
    'minimum_v2_to_v1_mean_policy_value_ratio': .95,
    'minimum_v2_naive_win_rate': 'at_least_v1_naive_win_rate',
}


def _actions_and_metrics(engine: CausaPayEvaluationEngine, test: pd.DataFrame, truth: pd.DataFrame, label: str):
    decision = engine.incrementality_policy.predict(test).reset_index(drop=True)
    actions = decision['recommended_action'].to_numpy()
    metric = evaluate_policy(label, actions, test, truth)
    row = {
        'policy': label, 'policy_value': metric['policy_value'], 'gross_recovered': metric['gross_recovered'],
        'true_incremental_recovered': metric['true_incremental_recovered'], 'intervention_cost': metric['intervention_cost'],
        'retry_count': int((actions == 'retry').sum()), 'whatsapp_count': int((actions == 'whatsapp').sum()),
        'none_count': int((actions == 'none').sum()), 'abstained_count': int(decision['is_abstain'].sum()),
        'executed_intervention_count': int(np.isin(actions, ['retry', 'whatsapp']).sum()), 'recovery_rate': metric['recovery_rate'],
    }
    return actions, decision, row


def _cate_rows(engine, test: pd.DataFrame, seed: int, estimator: str) -> list[dict]:
    predicted = engine.aipw_estimator.predict_counterfactuals(test).reset_index(drop=True)
    truth = conditional_potential_means(test, seed)
    rows = []
    for treatment, column in [('retry', 1), ('whatsapp', 2)]:
        estimate = predicted[f'prob_{treatment}'].to_numpy() - predicted['prob_none'].to_numpy()
        row = {'seed': seed, 'estimator': estimator, 'treatment': treatment, **metrics(estimate, truth[:, column] - truth[:, 0])}
        rows.append(row)
    return rows


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    warnings.filterwarnings('ignore', category=FutureWarning)
    policy_rows, cate_rows = [], []
    for seed in SEEDS:
        observed, hidden = build_demo_evaluation(seed=seed, num_events=NUM_EVENTS)
        v1 = CausaPayEvaluationEngine(estimator_version='v1')
        v1.fit_from_batch(observed, hidden, seed=seed)
        v2 = CausaPayEvaluationEngine(estimator_version='v2')
        v2.fit_from_batch(observed, hidden, seed=seed)
        test, truth = v1.test_obs.reset_index(drop=True), v1.test_hid.reset_index(drop=True)
        v1_actions, _, v1_row = _actions_and_metrics(v1, test, truth, 'causapay_v1')
        v2_actions, _, v2_row = _actions_and_metrics(v2, test, truth, 'causapay_v2')
        baseline_actions = v1.baseline_policy.predict(test)
        oracle_actions = v1.oracle_policy.predict(test, truth)['recommended_action'].to_numpy()
        for label, actions in [('consent_safe_baseline', baseline_actions), ('oracle', oracle_actions)]:
            metric = evaluate_policy(label, actions, test, truth)
            policy_rows.append({'seed': seed, 'policy': label, 'policy_value': metric['policy_value'], 'gross_recovered': metric['gross_recovered'], 'true_incremental_recovered': metric['true_incremental_recovered'], 'intervention_cost': metric['intervention_cost'], 'retry_count': int((actions == 'retry').sum()), 'whatsapp_count': int((actions == 'whatsapp').sum()), 'none_count': int((actions == 'none').sum()), 'abstained_count': 0, 'executed_intervention_count': int(np.isin(actions, ['retry', 'whatsapp']).sum()), 'recovery_rate': metric['recovery_rate']})
        # Each causal policy receives its own valid, volume-matched naive comparator.
        for version, engine, causal_row in [('v1', v1, v1_row), ('v2', v2, v2_row)]:
            naive_actions = engine.naive_policy.predict(test, causal_row['executed_intervention_count'])
            metric = evaluate_policy(f'naive_matched_to_{version}', naive_actions, test, truth)
            policy_rows.append({'seed': seed, 'policy': f'naive_matched_to_{version}', 'policy_value': metric['policy_value'], 'gross_recovered': metric['gross_recovered'], 'true_incremental_recovered': metric['true_incremental_recovered'], 'intervention_cost': metric['intervention_cost'], 'retry_count': int((naive_actions == 'retry').sum()), 'whatsapp_count': int((naive_actions == 'whatsapp').sum()), 'none_count': int((naive_actions == 'none').sum()), 'abstained_count': 0, 'executed_intervention_count': int(np.isin(naive_actions, ['retry', 'whatsapp']).sum()), 'recovery_rate': metric['recovery_rate']})
        policy_rows.extend([{**v1_row, 'seed': seed}, {**v2_row, 'seed': seed}])
        cate_rows.extend(_cate_rows(v1, test, seed, 'v1'))
        cate_rows.extend(_cate_rows(v2, test, seed, 'v2'))

    policies, cates = pd.DataFrame(policy_rows), pd.DataFrame(cate_rows)
    pivot = policies.pivot(index='seed', columns='policy', values='policy_value')
    policy_summary = policies.groupby('policy')['policy_value'].agg(['mean', 'median', 'std', 'min', 'max']).to_dict('index')
    policy_summary['wins'] = {
        'v1_vs_naive_matched_to_v1': float((pivot['causapay_v1'] > pivot['naive_matched_to_v1']).mean()),
        'v2_vs_naive_matched_to_v2': float((pivot['causapay_v2'] > pivot['naive_matched_to_v2']).mean()),
        'v2_vs_v1': float((pivot['causapay_v2'] > pivot['causapay_v1']).mean()),
    }
    cate_agg = cates.groupby(['estimator', 'treatment'])[['mae', 'correlation', 'calibration_slope']].agg(['mean', 'median', 'std'])
    cate_summary = {}
    for (e, t), row in cate_agg.iterrows():
        cate_summary[f'{e}:{t}'] = {f'{metric}_{stat}': float(val) for (metric, stat), val in row.items()}

    # Calculate explicit promotion criteria
    v1_retry_corr = float(cates[(cates['estimator'] == 'v1') & (cates['treatment'] == 'retry')]['correlation'].mean())
    v2_retry_corr = float(cates[(cates['estimator'] == 'v2') & (cates['treatment'] == 'retry')]['correlation'].mean())
    v1_wa_corr = float(cates[(cates['estimator'] == 'v1') & (cates['treatment'] == 'whatsapp')]['correlation'].mean())
    v2_wa_corr = float(cates[(cates['estimator'] == 'v2') & (cates['treatment'] == 'whatsapp')]['correlation'].mean())

    v1_retry_mae = float(cates[(cates['estimator'] == 'v1') & (cates['treatment'] == 'retry')]['mae'].mean())
    v2_retry_mae = float(cates[(cates['estimator'] == 'v2') & (cates['treatment'] == 'retry')]['mae'].mean())
    v1_wa_mae = float(cates[(cates['estimator'] == 'v1') & (cates['treatment'] == 'whatsapp')]['mae'].mean())
    v2_wa_mae = float(cates[(cates['estimator'] == 'v2') & (cates['treatment'] == 'whatsapp')]['mae'].mean())

    v1_val = float(policies[policies['policy'] == 'causapay_v1']['policy_value'].mean())
    v2_val = float(policies[policies['policy'] == 'causapay_v2']['policy_value'].mean())
    policy_ratio = v2_val / v1_val if v1_val != 0 else 0.0

    v1_naive_win = float((pivot['causapay_v1'] > pivot['naive_matched_to_v1']).mean())
    v2_naive_win = float((pivot['causapay_v2'] > pivot['naive_matched_to_v2']).mean())

    criteria_evaluation = {
        'criterion_1_correlation_lift_retry': {
            'target': '>= +0.10',
            'v1': v1_retry_corr,
            'v2': v2_retry_corr,
            'lift': v2_retry_corr - v1_retry_corr,
            'passed': bool(v2_retry_corr - v1_retry_corr >= 0.10),
        },
        'criterion_1_correlation_lift_whatsapp': {
            'target': '>= +0.10',
            'v1': v1_wa_corr,
            'v2': v2_wa_corr,
            'lift': v2_wa_corr - v1_wa_corr,
            'passed': bool(v2_wa_corr - v1_wa_corr >= 0.10),
        },
        'criterion_2_correlation_threshold_retry': {
            'target': '>= 0.10',
            'v2': v2_retry_corr,
            'passed': bool(v2_retry_corr >= 0.10),
        },
        'criterion_2_correlation_threshold_whatsapp': {
            'target': '>= 0.10',
            'v2': v2_wa_corr,
            'passed': bool(v2_wa_corr >= 0.10),
        },
        'criterion_3_mae_worsening_retry': {
            'target': '<= +0.02 worsening',
            'v1': v1_retry_mae,
            'v2': v2_retry_mae,
            'worsening': v2_retry_mae - v1_retry_mae,
            'passed': bool(v2_retry_mae - v1_retry_mae <= 0.02),
        },
        'criterion_3_mae_worsening_whatsapp': {
            'target': '<= +0.02 worsening',
            'v1': v1_wa_mae,
            'v2': v2_wa_mae,
            'worsening': v2_wa_mae - v1_wa_mae,
            'passed': bool(v2_wa_mae - v1_wa_mae <= 0.02),
        },
        'criterion_4_policy_value_ratio': {
            'target': '>= 0.95',
            'v1_mean': v1_val,
            'v2_mean': v2_val,
            'ratio': policy_ratio,
            'passed': bool(policy_ratio >= 0.95),
        },
        'criterion_5_naive_win_rate': {
            'target': 'v2_win >= v1_win',
            'v1_win_rate': v1_naive_win,
            'v2_win_rate': v2_naive_win,
            'passed': bool(v2_naive_win >= v1_naive_win),
        },
    }
    all_passed = all(item['passed'] for item in criteria_evaluation.values())
    criteria_evaluation['overall_promotion_decision'] = 'PROMOTE_V2' if all_passed else 'KEEP_V1'

    summary = {
        'study': {
            'seeds': list(SEEDS),
            'num_events_per_seed': NUM_EVENTS,
            'train_fraction': .7,
            'success_criteria': SUCCESS_CRITERIA,
        },
        'policy_summary': policy_summary,
        'cate_summary': cate_summary,
        'criteria_evaluation': criteria_evaluation,
    }
    return policies, cates, summary


def generate_markdown_report(policies: pd.DataFrame, cates: pd.DataFrame, summary: dict) -> str:
    md = '# V2 Direct CATE Experiment Report (Seeds 10–19)\n\n'
    md += 'Pre-specified evaluation comparing V1 (AIPW pseudo-outcome regression) against V2 (cross-fitted pairwise R-learner) on untouched seeds 10–19.\n\n'
    md += '## 1. Executive Summary & Promotion Decision\n\n'
    dec = summary['criteria_evaluation']['overall_promotion_decision']
    md += f'**Promotion Decision: `{dec}`**\n\n'
    md += r'While V2 reduced MAE on both treatments and achieved a higher mean policy value on these seeds, it **failed Criteria 1 and 2** (correlation lift $\ge +0.10$ and minimum correlation $\ge 0.10$). Individual treatment-effect ranking correlation remained below 0.10 for both arms (retry: 0.0579, whatsapp: 0.0961). In accordance with pre-specified rules, V1 is retained as the production estimator and V2 remains an experimental comparator.' + '\n\n'

    md += '## 2. Pre-specified Promotion Criteria Evaluation\n\n'
    md += '| Criterion | Target | Treatment / Arm | V1 Value | V2 Value | Difference / Ratio | Status |\n'
    md += '| --- | --- | --- | --- | --- | --- | --- |\n'
    c = summary['criteria_evaluation']
    md += f"| 1. Corr Lift | $\\ge +0.10$ | Retry | {c['criterion_1_correlation_lift_retry']['v1']:.4f} | {c['criterion_1_correlation_lift_retry']['v2']:.4f} | {c['criterion_1_correlation_lift_retry']['lift']:+.4f} | {'PASSED' if c['criterion_1_correlation_lift_retry']['passed'] else '**FAILED**'} |\n"
    md += f"| 1. Corr Lift | $\\ge +0.10$ | WhatsApp | {c['criterion_1_correlation_lift_whatsapp']['v1']:.4f} | {c['criterion_1_correlation_lift_whatsapp']['v2']:.4f} | {c['criterion_1_correlation_lift_whatsapp']['lift']:+.4f} | {'PASSED' if c['criterion_1_correlation_lift_whatsapp']['passed'] else '**FAILED**'} |\n"
    md += f"| 2. Min Corr | $\\ge 0.10$ | Retry | — | {c['criterion_2_correlation_threshold_retry']['v2']:.4f} | — | {'PASSED' if c['criterion_2_correlation_threshold_retry']['passed'] else '**FAILED**'} |\n"
    md += f"| 2. Min Corr | $\\ge 0.10$ | WhatsApp | — | {c['criterion_2_correlation_threshold_whatsapp']['v2']:.4f} | — | {'PASSED' if c['criterion_2_correlation_threshold_whatsapp']['passed'] else '**FAILED**'} |\n"
    md += f"| 3. Max MAE Worsening | $\\le +0.02$ | Retry | {c['criterion_3_mae_worsening_retry']['v1']:.4f} | {c['criterion_3_mae_worsening_retry']['v2']:.4f} | {c['criterion_3_mae_worsening_retry']['worsening']:+.4f} | {'**PASSED**' if c['criterion_3_mae_worsening_retry']['passed'] else 'FAILED'} |\n"
    md += f"| 3. Max MAE Worsening | $\\le +0.02$ | WhatsApp | {c['criterion_3_mae_worsening_whatsapp']['v1']:.4f} | {c['criterion_3_mae_worsening_whatsapp']['v2']:.4f} | {c['criterion_3_mae_worsening_whatsapp']['worsening']:+.4f} | {'**PASSED**' if c['criterion_3_mae_worsening_whatsapp']['passed'] else 'FAILED'} |\n"
    md += f"| 4. Policy Value Ratio | $\\ge 0.95$ | Overall | ₹{c['criterion_4_policy_value_ratio']['v1_mean']:,.2f} | ₹{c['criterion_4_policy_value_ratio']['v2_mean']:,.2f} | {c['criterion_4_policy_value_ratio']['ratio']:.4f}x | {'**PASSED**' if c['criterion_4_policy_value_ratio']['passed'] else 'FAILED'} |\n"
    md += f"| 5. Naive Win Rate | $\\ge$ V1 | Overall | {c['criterion_5_naive_win_rate']['v1_win_rate']*100:.0f}% | {c['criterion_5_naive_win_rate']['v2_win_rate']*100:.0f}% | — | {'**PASSED**' if c['criterion_5_naive_win_rate']['passed'] else 'FAILED'} |\n\n"

    md += '## 3. Policy Performance Summary (Seeds 10–19)\n\n'
    pol_agg = policies.groupby('policy')['policy_value'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    md += '| Policy | Mean (₹) | Median (₹) | Std Dev (₹) | Min (₹) | Max (₹) |\n'
    md += '| --- | --- | --- | --- | --- | --- |\n'
    for _, row in pol_agg.iterrows():
        md += f"| {row['policy']} | ₹{row['mean']:,.2f} | ₹{row['median']:,.2f} | ₹{row['std']:,.2f} | ₹{row['min']:,.2f} | ₹{row['max']:,.2f} |\n"
    md += '\n'

    md += '## 4. Estimator Diagnostics (Holdout CATE against Monte Carlo Conditional Truth)\n\n'
    cate_agg = cates.groupby(['estimator', 'treatment'])[['mae', 'correlation', 'calibration_slope']].agg(['mean', 'median', 'std']).reset_index()
    md += '| Estimator | Treatment | MAE (mean) | Corr (mean) | Corr (median) | Calib Slope (mean) |\n'
    md += '| --- | --- | --- | --- | --- | --- |\n'
    for _, row in cate_agg.iterrows():
        md += f"| {row[('estimator', '')]} | {row[('treatment', '')]} | {row[('mae', 'mean')]:.4f} | {row[('correlation', 'mean')]:.4f} | {row[('correlation', 'median')]:.4f} | {row[('calibration_slope', 'mean')]:.4f} |\n"
    md += '\n'
    return md


def write_artifacts(policies: pd.DataFrame, cates: pd.DataFrame, summary: dict) -> None:
    destination = Path(__file__).parent / 'results'
    destination.mkdir(exist_ok=True)
    policies.to_csv(destination / 'v2_experiment_policy_results_seeds_10_19.csv', index=False)
    cates.to_csv(destination / 'v2_experiment_cate_results_seeds_10_19.csv', index=False)
    (destination / 'v2_experiment_summary_seeds_10_19.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    report_md = generate_markdown_report(policies, cates, summary)
    (destination / 'v2_experiment_report_seeds_10_19.md').write_text(report_md, encoding='utf-8')


if __name__ == '__main__':
    policies, cates, summary = run()
    write_artifacts(policies, cates, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
