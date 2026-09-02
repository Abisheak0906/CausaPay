import pandas as pd
import numpy as np

class GrossRecoveryBaseline:
    """
    Policy A: Gross Recovery Baseline
    Uses observational data to calculate the gross recovery rate of each 
    intervention by segments, and chooses the one with the highest observed rate.
    """
    def __init__(self):
        self.best_actions = {}
        
    def fit(self, observables: pd.DataFrame):
        # Segment by plan_tier, failure_context and decline_signal_bucket
        group = observables.groupby(['plan_tier', 'failure_context', 'decline_signal_bucket', 'intervention_assigned'])['outcome_recovered'].mean().reset_index()
        
        # Find argmax intervention per segment
        idx = group.groupby(['plan_tier', 'failure_context', 'decline_signal_bucket'])['outcome_recovered'].idxmax()
        best = group.loc[idx]
        
        for _, row in best.iterrows():
            key = (row['plan_tier'], row['failure_context'], row['decline_signal_bucket'])
            self.best_actions[key] = row['intervention_assigned']
            
    def predict(self, observables: pd.DataFrame) -> np.ndarray:
        actions = []
        for _, row in observables.iterrows():
            key = (row['plan_tier'], row['failure_context'], row['decline_signal_bucket'])
            # fallback to 'none' if unseen segment
            actions.append(self.best_actions.get(key, 'none'))
        return np.array(actions)
