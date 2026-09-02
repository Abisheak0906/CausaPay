# backend/evaluation/ablation_study.py
"""Ablation study for propensity-weight handling in AIPW.

Three strategies are evaluated:
A) Current clipping (0.05, 0.95)
B) Conservative clipping (0.1, 0.9)
C) IPS truncation with cap=10 and stabilization by mean propensity.

The script:
- Uses seeds 42-61.
- Keeps the same simulator, treatment assignment, train/val/test splits.
- Fits outcome models (RandomForest) and propensity model (LogisticRegression).
- Computes pseudo-outcomes manually for each strategy.
- Evaluates on the validation split, selects the strategy with lowest CATE MAE,
  then evaluates that selected strategy once on the untouched test set.
- Persists results as JSON and a human‑readable Markdown report under
  backend/evaluation/results/.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

# Adjust PYTHONPATH to include project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from simulator.generator import SimulatorConfig, generate_events
from simulator.bias import assign_historical_treatments
from causal_engine.aipw_estimator import AIPWEstimator
from policies.incrementality import IncrementalityAwarePolicy
from policies.baseline import GrossRecoveryBaseline
from policies.oracle import OraclePolicy
from evaluation.evaluator import evaluate_policy

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def effective_sample_size(weights):
    w = np.asarray(weights)
    return float((w.sum() ** 2) / np.square(w).sum())

def pseudo_outcome_distribution(gamma):
    return {
        "min": float(np.min(gamma)),
        "max": float(np.max(gamma)),
        "mean": float(np.mean(gamma)),
        "std": float(np.std(gamma)),
        "var": float(np.var(gamma)),
        "p1": float(np.percentile(gamma, 1)),
        "p5": float(np.percentile(gamma, 5)),
        "p25": float(np.percentile(gamma, 25)),
        "p50": float(np.percentile(gamma, 50)),
        "p75": float(np.percentile(gamma, 75)),
        "p95": float(np.percentile(gamma, 95)),
        "p99": float(np.percentile(gamma, 99)),
    }

def true_cate(obs_hidden, treatment):
    return obs_hidden[f"prob_{treatment}"].values - obs_hidden["prob_none"].values

def evaluate_cate(true_vals, est_vals):
    mae = mean_absolute_error(true_vals, est_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, est_vals))
    corr = np.corrcoef(true_vals, est_vals)[0, 1]
    return {"mae": mae, "rmse": rmse, "corr": corr}

# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------

def run_ablation():
    seeds = list(range(42, 62))  # 42‑61 inclusive
    treatments = ["none", "retry", "whatsapp"]
    results_per_seed = {}

    for seed in seeds:
        # ----- generate training data (20k) -----
        train_cfg = SimulatorConfig(seed=seed, num_events=20000)
        train_obs, train_hid = generate_events(train_cfg)
        train_obs = assign_historical_treatments(train_obs, train_hid, seed=seed)

        # ----- split into train / validation (first fold of 5‑fold) -----
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, val_idx = next(kf.split(train_obs))
        y = train_obs["outcome_recovered"].values
        A = train_obs["intervention_assigned"].values
        X_raw = train_obs.drop(columns=["outcome_recovered", "intervention_assigned"]).copy()
        # Use AIPWEstimator's preprocessing for consistency
        dummy_est = AIPWEstimator(n_splits=5)
        X = dummy_est.preprocessor.fit_transform(X_raw)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        A_train, A_val = A[train_idx], A[val_idx]
        hid_val = train_hid.iloc[val_idx]

        # ----- outcome models per treatment -----
        outcome_models = {}
        for t in treatments:
            mask = (A_train == t)
            if mask.sum() == 0:
                continue
            mdl = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1)
            mdl.fit(X_train[mask], y_train[mask])
            outcome_models[t] = mdl

        # ----- propensity model (multiclass logistic regression) -----
        prop = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        prop.fit(X_train, A_train)
        prop_probs_train = prop.predict_proba(X_train)
        treat_to_idx = {t: i for i, t in enumerate(prop.classes_)}

        # ----- evaluate three strategies -----
        strategy_metrics = {}
        for strat in ["A", "B", "C"]:
            # Prepare propensity matrix according to strategy
            e_matrix = np.zeros_like(prop_probs_train)
            for t in treatments:
                raw = prop_probs_train[:, treat_to_idx[t]]
                if strat == "A":
                    p = np.clip(raw, 0.05, 0.95)
                elif strat == "B":
                    p = np.clip(raw, 0.10, 0.90)
                else:  # C
                    p = raw
                e_matrix[:, treat_to_idx[t]] = p
            if strat != "C":
                e_matrix = e_matrix / e_matrix.sum(axis=1, keepdims=True)

            # ----- compute pseudo‑outcomes on training data -----
            # Build vector of outcome model predictions for each observation using its actual treatment
            m_vec = np.empty_like(y_train, dtype=float)
            for tt in treatments:
                mask_tt = (A_train == tt)
                if mask_tt.sum() > 0:
                    m_vec[mask_tt] = outcome_models[tt].predict(X_train[mask_tt])
            gamma = {}
            for t in treatments:
                t_idx = treat_to_idx[t]
                p_t = e_matrix[:, t_idx]
                if strat == "C":
                    w = (A_train == t).astype(float) / p_t
                    w = np.clip(w, 0, 10)
                    w = w * np.mean(p_t)  # stabilization
                else:
                    w = (A_train == t).astype(float) / p_t
                gamma[t] = m_vec + w * (y_train - m_vec)

            # ----- validation diagnostics -----
            # Propensity weight distribution on validation set
            prop_probs_val = prop.predict_proba(X_val)
            weight_stats = {}
            for t in treatments:
                t_idx = treat_to_idx[t]
                raw = prop_probs_val[:, t_idx]
                if strat == "A":
                    p = np.clip(raw, 0.05, 0.95)
                elif strat == "B":
                    p = np.clip(raw, 0.10, 0.90)
                else:
                    p = raw
                if strat != "C":
                    # renormalise rows for consistency with training handling
                    # (not strictly needed for weight calculation but mirrors training)
                    pass
                w = (A_val == t).astype(float) / p
                if strat == "C":
                    w = np.clip(w, 0, 10)
                    w = w * np.mean(p)
                pct_below = (w < 0.05).mean() * 100
                pct_above = (w > 0.95).mean() * 100
                ess = effective_sample_size(w)
                weight_stats[t] = {"pct_below_0.05": pct_below, "pct_above_0.95": pct_above, "ESS": ess}

            # Pseudo‑outcome stats on validation
            # Compute gamma on validation using same outcome models
            m_val = np.empty_like(y_val, dtype=float)
            for tt in treatments:
                mask_tt = (A_val == tt)
                if mask_tt.sum() > 0:
                    m_val[mask_tt] = outcome_models[tt].predict(X_val[mask_tt])
            gamma_val = {}
            for t in treatments:
                t_idx = treat_to_idx[t]
                raw = prop_probs_val[:, t_idx]
                if strat == "A":
                    p = np.clip(raw, 0.05, 0.95)
                elif strat == "B":
                    p = np.clip(raw, 0.10, 0.90)
                else:
                    p = raw
                if strat == "C":
                    w = (A_val == t).astype(float) / p
                    w = np.clip(w, 0, 10)
                    w = w * np.mean(p)
                else:
                    w = (A_val == t).astype(float) / p
                gamma_val[t] = m_val + w * (y_val - m_val)

            pseudo_stats_val = {t: pseudo_outcome_distribution(gamma_val[t]) for t in treatments}

            # True vs estimated CATE on validation
            true_retry = true_cate(hid_val, "retry")
            true_wp = true_cate(hid_val, "whatsapp")
            est_retry = gamma_val["retry"] - gamma_val["none"]
            est_wp = gamma_val["whatsapp"] - gamma_val["none"]
            cate_metrics = {
                "retry": evaluate_cate(true_retry, est_retry),
                "whatsapp": evaluate_cate(true_wp, est_wp)
            }

            # Policy values (mean pseudo‑outcome) and ATE MAE (probability‑scale)
            policy_vals = {t: np.mean(gamma_val[t]) for t in treatments}
            oracle_vals = {t: np.mean(train_hid[f"prob_{t}"]) for t in treatments}
            baseline_vals = {"none": np.mean(train_hid["prob_none"])}
            ate_mae = np.mean([abs(policy_vals[t] - oracle_vals[t]) for t in treatments])
            win_rate = np.mean([1 if policy_vals[t] > baseline_vals.get(t, 0) else 0 for t in treatments])

            strategy_metrics[strat] = {
                "propensity_weights": weight_stats,
                "pseudo_outcome_stats": pseudo_stats_val,
                "cate_metrics": cate_metrics,
                "policy_values": policy_vals,
                "oracle_values": oracle_vals,
                "baseline_values": baseline_vals,
                "ATE_MAE": ate_mae,
                "win_rate_vs_baseline": win_rate,
            }

        # ----- select best strategy based on validation CATE MAE (average over two treatments) -----
        best_strat = min(strategy_metrics.items(), key=lambda kv: np.mean([kv[1]["cate_metrics"][tr]["mae"] for tr in ["retry", "whatsapp"]]))[0]

        # ----- evaluate selected strategy on untouched test set -----
        test_cfg = SimulatorConfig(seed=999, num_events=5000)
        test_obs, test_hid = generate_events(test_cfg)
        test_obs = assign_historical_treatments(test_obs, test_hid, seed=999)
        X_test_raw = test_obs.drop(columns=["outcome_recovered", "intervention_assigned"]).copy()
        X_test = dummy_est.preprocessor.transform(X_test_raw)
        A_test = test_obs["intervention_assigned"].values
        y_test = test_obs["outcome_recovered"].values
        prop_probs_test = prop.predict_proba(X_test)

        # compute gamma on test using best strategy
        gamma_test = {}
        for t in treatments:
            t_idx = treat_to_idx[t]
            raw = prop_probs_test[:, t_idx]
            if best_strat == "A":
                p = np.clip(raw, 0.05, 0.95)
            elif best_strat == "B":
                p = np.clip(raw, 0.10, 0.90)
            else:
                p = raw
            # outcome predictions for each observation using its actual treatment
            m_test = np.empty_like(y_test, dtype=float)
            for tt in treatments:
                mask_tt = (A_test == tt)
                if mask_tt.sum() > 0:
                    m_test[mask_tt] = outcome_models[tt].predict(X_test[mask_tt])
            if best_strat == "C":
                w = (A_test == t).astype(float) / p
                w = np.clip(w, 0, 10)
                w = w * np.mean(p)
            else:
                w = (A_test == t).astype(float) / p
            gamma_test[t] = m_test + w * (y_test - m_test)

        pseudo_stats_test = {t: pseudo_outcome_distribution(gamma_test[t]) for t in treatments}
        true_retry_test = true_cate(test_hid, "retry")
        true_wp_test = true_cate(test_hid, "whatsapp")
        est_retry_test = gamma_test["retry"] - gamma_test["none"]
        est_wp_test = gamma_test["whatsapp"] - gamma_test["none"]
        cate_metrics_test = {
            "retry": evaluate_cate(true_retry_test, est_retry_test),
            "whatsapp": evaluate_cate(true_wp_test, est_wp_test),
        }
        policy_vals_test = {t: np.mean(gamma_test[t]) for t in treatments}
        oracle_vals_test = {t: np.mean(test_hid[f"prob_{t}"]) for t in treatments}
        baseline_vals_test = {"none": np.mean(test_hid["prob_none"])}
        ate_mae_test = np.mean([abs(policy_vals_test[t] - oracle_vals_test[t]) for t in treatments])
        win_rate_test = np.mean([1 if policy_vals_test[t] > baseline_vals_test.get(t, 0) else 0 for t in treatments])

        results_per_seed[seed] = {
            "selected_strategy": best_strat,
            "validation": strategy_metrics,
            "test": {
                "propensity_weights": {t: {"pct_below_0.05": ( ( (A_test == t).astype(float) / prop_probs_test[:, treat_to_idx[t]] ) < 0.05).mean() * 100,
                                              "pct_above_0.95": ( ( (A_test == t).astype(float) / prop_probs_test[:, treat_to_idx[t]] ) > 0.95).mean() * 100,
                                              "ESS": effective_sample_size((A_test == t).astype(float) / prop_probs_test[:, treat_to_idx[t]])},
                                    for t in treatments},
                "pseudo_outcome_stats": pseudo_stats_test,
                "cate_metrics": cate_metrics_test,
                "policy_values": policy_vals_test,
                "oracle_values": oracle_vals_test,
                "baseline_values": baseline_vals_test,
                "ATE_MAE": ate_mae_test,
                "win_rate_vs_baseline": win_rate_test,
            }
        }

    # -----------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"ablation_summary_{timestamp}.json")
    md_path = os.path.join(out_dir, f"ablation_report_{timestamp}.md")
    with open(json_path, "w") as jf:
        json.dump({"timestamp": timestamp, "seeds": seeds, "per_seed": results_per_seed}, jf, indent=2)
    # Simple Markdown report
    with open(md_path, "w") as mf:
        mf.write(f"# AIPW Propensity‑Weight Ablation Study ({timestamp})\n\n")
        mf.write("## Validation averages across seeds\n\n")
        agg = {"A": {"cate_mae": [], "ATE_MAE": []}, "B": {"cate_mae": [], "ATE_MAE": []}, "C": {"cate_mae": [], "ATE_MAE": []}}
        for sdata in results_per_seed.values():
            for strat, metr in sdata["validation"].items():
                avg_cate = np.mean([metr["cate_metrics"][tr]["mae"] for tr in ["retry", "whatsapp"]])
                agg[strat]["cate_mae"].append(avg_cate)
                agg[strat]["ATE_MAE"].append(metr["ATE_MAE"])
        mf.write("| Strategy | Avg CATE MAE (val) | Avg ATE MAE (val) |\n|---|---|---|\n")
        for strat in ["A", "B", "C"]:
            mf.write(f"| {strat} | {np.mean(agg[strat]["cate_mae"]):.5f} | {np.mean(agg[strat]["ATE_MAE"]):.5f} |\n")
        mf.write("\n## Selected strategy per seed (validation)\n\n")
        mf.write("| Seed | Strategy | CATE MAE retry | CATE MAE whatsapp |\n|---|---|---|---|\n")
        for seed, sdata in results_per_seed.items():
            strat = sdata["selected_strategy"]
            cmr = sdata["validation"][strat]["cate_metrics"]["retry"]["mae"]
            cmw = sdata["validation"][strat]["cate_metrics"]["whatsapp"]["mae"]
            mf.write(f"| {seed} | {strat} | {cmr:.5f} | {cmw:.5f} |\n")
        mf.write("\n## Test‑set results for the selected strategies\n\n")
        mf.write("| Seed | Strategy | ATE MAE (test) | Policy retry | Policy whatsapp | Win rate vs baseline |\n|---|---|---|---|---|---|\n")
        for seed, sdata in results_per_seed.items():
            strat = sdata["selected_strategy"]
            tdata = sdata["test"]
            mf.write(f"| {seed} | {strat} | {tdata['ATE_MAE']:.5f} | {tdata['policy_values']['retry']:.2f} | {tdata['policy_values']['whatsapp']:.2f} | {tdata['win_rate_vs_baseline']:.2f} |\n")
    return json_path, md_path

if __name__ == "__main__":
    run_ablation()
