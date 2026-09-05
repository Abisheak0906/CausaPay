import numpy as np
import pandas as pd
from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation

def run_audit(seed=42, num_events=10000):
    print(f"Running ENIV Robustness Audit (Seed={seed}, N={num_events})...")
    
    obs, hid = build_demo_evaluation(seed=seed, num_events=num_events)
    engine = CausaPayEvaluationEngine()
    engine.fit_from_batch(obs, hid, seed=seed)
    
    # Get counterfactuals for held-out
    cf = engine.aipw_estimator.predict_counterfactuals(engine.test_obs).reset_index(drop=True)
    amounts = engine.test_obs['amount'].values
    
    # Costs
    costs = {'retry': 2.0, 'whatsapp': 15.0}
    
    results = []
    for action in ['retry', 'whatsapp']:
        # Estimated Lift
        lift = cf[f'prob_{action}'].to_numpy() - cf['prob_none'].to_numpy()
        # ENIV
        eniv = lift * amounts - costs[action]
        
        # Analysis
        pos_eniv_mask = eniv > 0
        pos_eniv_values = eniv[pos_eniv_mask]
        
        results.append({
            'action': action,
            'mean_lift': lift.mean(),
            'median_lift': np.median(lift),
            'std_lift': lift.std(),
            'pos_eniv_count': int(pos_eniv_mask.sum()),
            'pos_eniv_mean': pos_eniv_values.mean() if len(pos_eniv_values) > 0 else np.nan,
            'pos_eniv_median': np.median(pos_eniv_values) if len(pos_eniv_values) > 0 else np.nan,
            'pos_eniv_p10': np.quantile(pos_eniv_values, 0.1) if len(pos_eniv_values) > 0 else np.nan,
            'pos_eniv_p50': np.median(pos_eniv_values) if len(pos_eniv_values) > 0 else np.nan,
            'pos_eniv_p90': np.quantile(pos_eniv_values, 0.9) if len(pos_eniv_values) > 0 else np.nan,
        })
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    res = run_audit()
    print("\n=== ENIV Robustness Audit Results ===")
    print(res.to_string(index=False))
