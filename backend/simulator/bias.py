import pandas as pd
import numpy as np

def assign_historical_treatments(observables: pd.DataFrame, hidden: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Creates historical treatment assignment that is biased (confounded) 
    based on observable features, preserving positivity.
    """
    np.random.seed(seed)
    n = len(observables)
    
    # Base logits
    logit_none = np.zeros(n)
    logit_retry = np.zeros(n)
    logit_whatsapp = np.full(n, -2.0) # Penalty for whatsapp by default
    
    # 1. Plan tier effects
    is_enterprise = (observables['plan_tier'] == 'Enterprise').astype(int)
    logit_whatsapp += is_enterprise * 1.5
    logit_none -= is_enterprise * 1.0
    
    # 2. Failure context effects
    is_insufficient = (observables['failure_context'] == 'insufficient_funds').astype(int)
    logit_retry += is_insufficient * 1.2
    
    # 3. Engagement score
    logit_whatsapp += observables['engagement_score'] * 2.0
    
    # 4. WhatsApp opt in
    logit_whatsapp -= (~observables['whatsapp_opted_in']).astype(int) * 10.0 
    
    # 5. Historical failures
    logit_retry += observables['historical_failure_count'] * 0.1
    
    # Softmax to get probabilities
    logits = np.column_stack([logit_none, logit_retry, logit_whatsapp])
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Ensure positivity with a floor
    epsilon = 0.05
    probs = probs * (1 - 3*epsilon) + epsilon
    
    # Assign treatment based on probs
    treatments = ['none', 'retry', 'whatsapp']
    cum_probs = probs.cumsum(axis=1)
    u = np.random.rand(n, 1)
    treatment_idx = (u > cum_probs).sum(axis=1)
    treatment_idx = np.clip(treatment_idx, 0, 2)
    assigned_treatment = np.array(treatments)[treatment_idx]
    
    # Map observed outcome
    outcome = np.zeros(n, dtype=int)
    for i, t in enumerate(treatments):
        mask = assigned_treatment == t
        outcome[mask] = hidden[f'Y_{t}'][mask]
        
    obs = observables.copy()
    obs['intervention_assigned'] = assigned_treatment
    obs['outcome_recovered'] = outcome
    obs['amount_recovered'] = outcome * obs['amount']
    # Time to recovery (static dummy for now, if recovered)
    obs['time_to_recovery_hours'] = np.where(outcome == 1, np.random.randint(1, 72, size=n), np.nan)
    
    return obs

if __name__ == "__main__":
    from generator import SimulatorConfig, generate_events
    obs, hid = generate_events(SimulatorConfig())
    obs_biased = assign_historical_treatments(obs, hid)
    print("Treatment distribution:")
    print(obs_biased['intervention_assigned'].value_counts(normalize=True))
    
    print("\nGross recovery by intervention:")
    print(obs_biased.groupby('intervention_assigned')['outcome_recovered'].mean())
