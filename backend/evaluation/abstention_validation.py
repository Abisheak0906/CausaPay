import numpy as np
import pandas as pd
from scipy import stats
from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation
from evaluation.repeated_seed_study import conditional_potential_means

def run_validation(seed=42, num_events=5000):
    print(f"Running Abstention Validation (Seed={seed}, N={num_events})...")
    
    # 1. Generate synthetic data
    obs, hid = build_demo_evaluation(seed=seed, num_events=num_events)
    
    # 2. Fit V1 CausaPay
    engine = CausaPayEvaluationEngine(estimator_version='v1')
    engine.fit_from_batch(obs, hid, seed=seed)
    
    # Use held-out set for validation
    test_obs = engine.test_obs.reset_index(drop=True)
    
    # 3. Get Causal Predictions and Uncertainty
    preds = engine.aipw_estimator.predict_counterfactuals(test_obs).reset_index(drop=True)
    
    # 4. Get CATE Truth (via Monte-Carlo integration)
    truth = conditional_potential_means(test_obs, seed)
    
    results = []
    for action, idx in [('retry', 1), ('whatsapp', 2)]:
        # Estimated Lift: Prob(Action) - Prob(None)
        est_lift = preds[f'prob_{action}'].to_numpy() - preds['prob_none'].to_numpy()
        # True Lift: Truth(Action) - Truth(None)
        true_lift = truth[:, idx] - truth[:, 0]
        # Uncertainty score
        uncertainty = preds[f'std_{action}'].to_numpy()
        # Absolute Error
        abs_error = np.abs(est_lift - true_lift)
        
        correlation = np.corrcoef(uncertainty, abs_error)[0, 1]
        
        # Analyze abstention (using the policy threshold)
        threshold = engine.incrementality_policy.uncertainty_threshold
        abstained = uncertainty > threshold
        error_abstained = abs_error[abstained].mean() if abstained.any() else np.nan
        error_not_abstained = abs_error[~abstained].mean() if (~abstained).any() else np.nan
        
        results.append({
            'action': action,
            'correlation': correlation,
            'mean_abs_error': abs_error.mean(),
            'error_abstained': error_abstained,
            'error_not_abstained': error_not_abstained,
            'abstention_rate': abstained.mean()
        })
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    res = run_validation()
    print("\n=== Abstention Validation Results ===")
    print(res.to_string(index=False))
