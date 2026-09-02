import pandas as pd
import numpy as np

class OraclePolicy:
    """
    Policy C: Oracle Policy
    Has access to true simulator treatment effects (hidden ground truth).
    Chooses the true expected-value-maximizing action.
    Represents the theoretical upper bound.
    """
    def __init__(self):
        self.costs = {
            'none': 0.0,
            'retry': 2.0,
            'whatsapp': 15.0
        }
        
    def predict(self, observables: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=observables.index)
        results['event_id'] = observables['event_id']
        
        amounts = observables['amount'].values
        
        p_none = hidden['true_baseline_self_cure_prob'].values
        p_retry = hidden['prob_retry'].values
        p_wa = hidden['prob_whatsapp'].values
        
        inc_prob_retry = p_retry - p_none
        inc_prob_wa = p_wa - p_none
        
        eniv_none = np.zeros(len(amounts))
        eniv_retry = (inc_prob_retry * amounts) - self.costs['retry']
        eniv_wa = (inc_prob_wa * amounts) - self.costs['whatsapp']
        
        actions = []
        for i in range(len(amounts)):
            best_eniv = max(eniv_none[i], eniv_retry[i], eniv_wa[i])
            if best_eniv <= 0:
                actions.append('none')
            elif best_eniv == eniv_retry[i]:
                actions.append('retry')
            else:
                actions.append('whatsapp')
                
        results['recommended_action'] = actions
        return results
