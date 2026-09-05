import pandas as pd
import numpy as np
from typing import Dict, Any

def evaluate_policy(policy_name: str, actions: np.ndarray, observables: pd.DataFrame, hidden: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates a policy's decisions against the hidden ground truth (simulator binary outcomes).

    All monetary metrics here are REALIZED from simulator potential outcomes — they are NOT
    model estimates.  They require access to the hidden DataFrame (only available in synthetic
    held-out evaluation, never for real uploaded CSVs).

    Key formula:
        true_incremental_recovered = Σ (Y(action_i) − Y(none_i)) * amount_i

    where Y(*) comes from the simulator's binary potential-outcome columns (Y_none, Y_retry,
    Y_whatsapp).  This is genuine counterfactual attribution, not an AIPW model estimate.
    """
    costs = {
        'none': 0.0,
        'retry': 2.0,
        'whatsapp': 15.0
    }
    
    amounts = observables['amount'].values
    
    # Calculate total costs
    total_cost = sum(costs.get(a, 0.0) for a in actions)
    
    # Calculate True Outcomes for the chosen actions (from simulator hidden outcomes)
    true_y_action = np.zeros(len(actions))
    true_y_none = hidden['Y_none'].values
    
    for i, a in enumerate(actions):
        true_y_action[i] = hidden[f'Y_{a}'].iloc[i]
        
    # Gross recovery = Y(action) * amount  [simulator ground truth]
    gross_recovered = (true_y_action * amounts).sum()
    
    # Incremental recovery = (Y(action) - Y(none)) * amount  [simulator ground truth]
    # This is the revenue attributable to the intervention beyond what would have self-cured.
    true_incremental_recovered = ((true_y_action - true_y_none) * amounts).sum()
    
    # Policy Value = True Incremental Revenue - Intervention Costs
    policy_value = true_incremental_recovered - total_cost
    
    recovery_rate = true_y_action.mean()
    action_counts = pd.Series(actions).value_counts().to_dict()
    
    return {
        'policy_name': policy_name,
        'gross_recovered': float(gross_recovered),
        # 'true_incremental_recovered' is simulator ground truth — valid only in synthetic evaluation.
        'true_incremental_recovered': float(true_incremental_recovered),
        'intervention_cost': float(total_cost),
        'policy_value': float(policy_value),
        'recovery_rate': float(recovery_rate),
        'action_distribution': action_counts
    }
