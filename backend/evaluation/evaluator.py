import pandas as pd
import numpy as np
from typing import Dict, Any

def evaluate_policy(policy_name: str, actions: np.ndarray, observables: pd.DataFrame, hidden: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates a policy's decisions against the hidden ground truth.
    Returns metrics including true incremental revenue.
    """
    costs = {
        'none': 0.0,
        'retry': 2.0,
        'whatsapp': 15.0
    }
    
    amounts = observables['amount'].values
    
    # Calculate total costs
    total_cost = sum(costs.get(a, 0.0) for a in actions)
    
    # Calculate True Outcomes for the chosen actions
    true_y_action = np.zeros(len(actions))
    true_y_none = hidden['Y_none'].values
    
    for i, a in enumerate(actions):
        true_y_action[i] = hidden[f'Y_{a}'].iloc[i]
        
    # Gross recovery = Y(action) * amount
    gross_recovered = (true_y_action * amounts).sum()
    
    # Incremental recovery = (Y(action) - Y(none)) * amount
    true_incremental_recovered = ((true_y_action - true_y_none) * amounts).sum()
    
    # Policy Value = True Incremental Revenue - Intervention Costs
    policy_value = true_incremental_recovered - total_cost
    
    recovery_rate = true_y_action.mean()
    action_counts = pd.Series(actions).value_counts().to_dict()
    
    return {
        'policy_name': policy_name,
        'gross_recovered': float(gross_recovered),
        'true_incremental_recovered': float(true_incremental_recovered),
        'intervention_cost': float(total_cost),
        'policy_value': float(policy_value),
        'recovery_rate': float(recovery_rate),
        'action_distribution': action_counts
    }
