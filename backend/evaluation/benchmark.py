import sys
import os
import json
import csv
from datetime import datetime
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from simulator.generator import SimulatorConfig, generate_events
from simulator.bias import assign_historical_treatments
from causal_engine.estimator import TLearner
from causal_engine.aipw_estimator import AIPWEstimator
from policies.baseline import GrossRecoveryBaseline
from policies.incrementality import IncrementalityAwarePolicy
from policies.oracle import OraclePolicy
from evaluation.evaluator import evaluate_policy
import warnings

warnings.filterwarnings('ignore')

def calculate_causal_metrics(model_preds, hidden_truth):
    """
    Compares estimated effects vs true simulator effects.
    True CATE = prob_a - prob_none
    Estimated CATE = est_prob_a - est_prob_none
    """
    metrics = {}
    
    for t in ['retry', 'whatsapp']:
        true_cate = hidden_truth[f'prob_{t}'].values - hidden_truth['true_baseline_self_cure_prob'].values
        est_cate = model_preds[f'prob_{t}'].values - model_preds['prob_none'].values
        
        true_ate = true_cate.mean()
        est_ate = est_cate.mean()
        
        metrics[f'ate_bias_{t}'] = est_ate - true_ate
        metrics[f'ate_mae_{t}'] = np.abs(est_ate - true_ate)
        metrics[f'cate_mae_{t}'] = np.abs(est_cate - true_cate).mean()
        metrics[f'cate_rmse_{t}'] = np.sqrt(np.mean((est_cate - true_cate)**2))
        
    # Aggregate over treatments
    metrics['ate_bias_mean'] = np.mean([np.abs(metrics['ate_bias_retry']), np.abs(metrics['ate_bias_whatsapp'])])
    metrics['ate_mae_mean'] = np.mean([metrics['ate_mae_retry'], metrics['ate_mae_whatsapp']])
    metrics['cate_mae_mean'] = np.mean([metrics['cate_mae_retry'], metrics['cate_mae_whatsapp']])
    metrics['cate_rmse_mean'] = np.mean([metrics['cate_rmse_retry'], metrics['cate_rmse_whatsapp']])
    
    return metrics

def run_benchmark():
    num_seeds = 20
    results = []
    
    # Track overall ATE metrics
    t_learner_metrics = []
    aipw_metrics = []
    
    for seed in range(42, 42 + num_seeds):
        print(f"Running seed {seed}...")
        config = SimulatorConfig(seed=seed, num_events=10000)
        obs, hid = generate_events(config)
        obs_biased = assign_historical_treatments(obs, hid, seed=seed)
        
        train_size = int(0.7 * len(obs_biased))
        train_obs = obs_biased.iloc[:train_size]
        test_obs = obs_biased.iloc[train_size:]
        test_hid = hid.iloc[train_size:]
        
        # 1. Baseline
        baseline = GrossRecoveryBaseline()
        baseline.fit(train_obs)
        a_base = baseline.predict(test_obs)
        val_base = evaluate_policy('Baseline', a_base, test_obs, test_hid)['policy_value']
        
        # 2. T-Learner
        tlearner = TLearner()
        tlearner.fit(train_obs)
        inc_t = IncrementalityAwarePolicy(tlearner, baseline)
        preds_t = tlearner.predict_counterfactuals(test_obs)
        a_t = inc_t.predict(test_obs)['recommended_action'].values
        val_t = evaluate_policy('T-Learner', a_t, test_obs, test_hid)['policy_value']
        t_learner_metrics.append(calculate_causal_metrics(preds_t, test_hid))
        
        # 3. AIPW Estimator
        aipw = AIPWEstimator(clip_threshold=0.05, n_splits=5)
        aipw.fit(train_obs)
        inc_aipw = IncrementalityAwarePolicy(aipw, baseline)
        preds_aipw = aipw.predict_counterfactuals(test_obs)
        a_aipw = inc_aipw.predict(test_obs)['recommended_action'].values
        val_aipw = evaluate_policy('AIPW', a_aipw, test_obs, test_hid)['policy_value']
        aipw_metrics.append(calculate_causal_metrics(preds_aipw, test_hid))
        
        # 4. Oracle
        oracle = OraclePolicy()
        a_or = oracle.predict(test_obs, test_hid)['recommended_action'].values
        val_or = evaluate_policy('Oracle', a_or, test_obs, test_hid)['policy_value']
        
        results.append({
            'seed': seed,
            'Baseline': val_base,
            'T-Learner': val_t,
            'AIPW': val_aipw,
            'Oracle': val_or
        })
        
    df = pd.DataFrame(results)
    
    # Calculate Summary Stats
    summary = pd.DataFrame({
        'Mean': df.mean()[1:],
        'Median': df.median()[1:],
        'Std': df.std()[1:],
    })
    
    # 95% CI (1.96 * std / sqrt(n))
    summary['95% CI (±)'] = 1.96 * summary['Std'] / np.sqrt(num_seeds)
    
    # Regret vs Oracle
    summary['Avg Regret'] = df['Oracle'].mean() - summary['Mean']

    print("\n" + "="*50)
    print("POLICY VALUE BENCHMARK (20 Seeds)")
    print("="*50)
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    # Compute causal metrics before persisting results
    df_t_met = pd.DataFrame(t_learner_metrics)
    df_aipw_met = pd.DataFrame(aipw_metrics)
    causal_summary = pd.DataFrame({
        "T-Learner": df_t_met.mean(),
        "AIPW": df_aipw_met.mean()
    })
    print("\n" + "="*50)
    print("CAUSAL ESTIMATION QUALITY")
    print("="*50)
    print(causal_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # Prepare per-seed data
    per_seed = []
    for _, row in df.iterrows():
        seed = int(row['seed'])
        per_seed.append({
            "seed": seed,
            "Baseline": float(row['Baseline']),
            "T-Learner": float(row['T-Learner']),
            "AIPW": float(row['AIPW']),
            "Oracle": float(row['Oracle']),
            "AIPW_beats_Baseline": bool(row['AIPW'] > row['Baseline']),
            "AIPW_beats_T-Learner": bool(row['AIPW'] > row['T-Learner'])
        })

    # Build result_json with causal metrics included
    result_json = {
        "configuration": {
            "num_seeds": 20,
            "seed_start": 42,
            "seed_end": 61,
            "timestamp": datetime.now().isoformat()
        },
        "per_seed": per_seed,
        "aggregate_statistics": {
            "Mean": {k: float(v) for k, v in summary['Mean'].items()},
            "Median": {k: float(v) for k, v in summary['Median'].items()},
            "Std": {k: float(v) for k, v in summary['Std'].items()},
            "95% CI (±)": {k: float(v) for k, v in summary['95% CI (±)'].items()},
            "Avg Regret": {k: float(v) for k, v in summary['Avg Regret'].items()}
        },
        "causal_estimation_metrics": {
            "T-Learner": {k: float(v) for k, v in df_t_met.mean().items()},
            "AIPW": {k: float(v) for k, v in df_aipw_met.mean().items()}
        },
        "win_rates": {
            "AIPW_beats_Baseline": (df['AIPW'] > df['Baseline']).mean() * 100,
            "AIPW_beats_T-Learner": (df['AIPW'] > df['T-Learner']).mean() * 100
        }
    }

    print("\nWin Rates:")
    print(f"AIPW beats Baseline: {(df['AIPW'] > df['Baseline']).mean() * 100:.1f}%")
    print(f"AIPW beats T-Learner: {(df['AIPW'] > df['T-Learner']).mean() * 100:.1f}%")

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(results_dir, f"benchmark_results_{timestamp_str}.json")
    with open(json_path, "w") as jf:
        json.dump(result_json, jf, indent=2)
    csv_path = os.path.join(results_dir, f"benchmark_results_{timestamp_str}.csv")
    with open(csv_path, "w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=["seed", "Baseline", "T-Learner", "AIPW", "Oracle", "AIPW_beats_Baseline", "AIPW_beats_T-Learner"])
        writer.writeheader()
        for row in per_seed:
            writer.writerow(row)
    print(f"\nResults saved to JSON: {json_path}\nResults saved to CSV: {csv_path}")
    summary['Avg Regret'] = df['Oracle'].mean() - summary['Mean']
    
    print("\n" + "="*50)
    print("POLICY VALUE BENCHMARK (20 Seeds)")
    print("="*50)
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    # --------------------------------------------------
    # Persist results
    # --------------------------------------------------
    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Prepare per-seed data
    per_seed = []
    for _, row in df.iterrows():
        seed = int(row['seed'])
        per_seed.append({
            "seed": seed,
            "Baseline": float(row['Baseline']),
            "T-Learner": float(row['T-Learner']),
            "AIPW": float(row['AIPW']),
            "Oracle": float(row['Oracle']),
            "AIPW_beats_Baseline": bool(row['AIPW'] > row['Baseline']),
            "AIPW_beats_T-Learner": bool(row['AIPW'] > row['T-Learner'])
        })

    # Aggregate causal metrics
    # Placeholder: causal_agg will be computed later after DataFrames are created

    # Full JSON structure
    # Build result_json after causal_summary is computed (placeholder for now)
    result_json = {
        "configuration": {
            "num_seeds": 20,
            "seed_start": 42,
            "seed_end": 61,
            "timestamp": datetime.now().isoformat()
        },
        "per_seed": per_seed,
        "aggregate_statistics": {
            "Mean": {k: float(v) for k, v in summary['Mean'].items()},
            "Median": {k: float(v) for k, v in summary['Median'].items()},
            "Std": {k: float(v) for k, v in summary['Std'].items()},
            "95% CI (±)": {k: float(v) for k, v in summary['95% CI (±)'].items()},
            "Avg Regret": {k: float(v) for k, v in summary['Avg Regret'].items()}
        },
        # causal_estimation_metrics will be filled later
        "win_rates": {
            "AIPW_beats_Baseline": (df['AIPW'] > df['Baseline']).mean() * 100,
            "AIPW_beats_T-Learner": (df['AIPW'] > df['T-Learner']).mean() * 100
        }
    }

    print("\nWin Rates:")
    print(f"AIPW beats Baseline: {(df['AIPW'] > df['Baseline']).mean() * 100:.1f}%")
    print(f"AIPW beats T-Learner: {(df['AIPW'] > df['T-Learner']).mean() * 100:.1f}%")
    
    # Causal Metrics Summary
    df_t_met = pd.DataFrame(t_learner_metrics)
    df_aipw_met = pd.DataFrame(aipw_metrics)
    
    causal_summary = pd.DataFrame({
        'T-Learner': df_t_met.mean(),
        'AIPW': df_aipw_met.mean()
    })
    
    print("\n" + "="*50)
    print("CAUSAL ESTIMATION QUALITY")
    print("="*50)
    print(causal_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Now that we have df_t_met and df_aipw_met, compute causal_agg and insert into result_json
    causal_agg = {
        "T-Learner": {k: float(v) for k, v in df_t_met.mean().items()},
        "AIPW": {k: float(v) for k, v in df_aipw_met.mean().items()}
    }
    result_json["causal_estimation_metrics"] = causal_agg

    # Save JSON
    json_path = os.path.join(results_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, "w") as jf:
        json.dump(result_json, jf, indent=2)

    # Save CSV (per‑seed values)
    csv_path = os.path.join(results_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=["seed", "Baseline", "T-Learner", "AIPW", "Oracle", "AIPW_beats_Baseline", "AIPW_beats_T-Learner"])
        writer.writeheader()
        for row in per_seed:
            writer.writerow(row)

    print(f"\nResults saved to JSON: {json_path}\nResults saved to CSV: {csv_path}")

if __name__ == "__main__":
    run_benchmark()
