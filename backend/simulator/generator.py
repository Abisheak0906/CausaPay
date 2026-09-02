import pandas as pd
import numpy as np
from typing import Tuple

class SimulatorConfig:
    def __init__(self, seed: int = 42, num_events: int = 10000):
        self.seed = seed
        self.num_events = num_events

def generate_events(config: SimulatorConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic payment failure events.
    Returns:
        observables (pd.DataFrame): Data visible to the model/policy
        hidden (pd.DataFrame): Ground truth hidden variables and potential outcomes
    """
    np.random.seed(config.seed)
    n = config.num_events
    
    # --- 1. Generate Observable Features ---
    customer_ids = np.random.randint(1000, 9999 + n, size=n)
    tenure_days = np.random.randint(0, 2000, size=n)
    plan_tier = np.random.choice(['Basic', 'Pro', 'Enterprise'], size=n, p=[0.6, 0.3, 0.1])
    avg_order_value = np.random.lognormal(mean=7.0, sigma=1.0, size=n)
    avg_order_value = np.clip(avg_order_value, 99, 50000).round(2)
    historical_payment_count = np.random.poisson(lam=10, size=n)
    historical_failure_count = np.random.poisson(lam=2, size=n)
    days_since_last_failure = np.where(
        historical_failure_count > 0, 
        np.random.randint(1, 365, size=n), 
        -1 # -1 denotes null/never
    )
    payment_method = np.random.choice(['Card', 'UPI Autopay', 'Netbanking'], size=n, p=[0.4, 0.5, 0.1])
    email_verified = np.random.binomial(1, 0.8, size=n).astype(bool)
    whatsapp_opted_in = np.random.binomial(1, 0.6, size=n).astype(bool)
    engagement_score = np.random.beta(a=2, b=5, size=n)
    
    failure_context = np.random.choice(['technical_timeout', 'insufficient_funds'], size=n, p=[0.3, 0.7])
    day_of_month = np.random.randint(1, 29, size=n)
    decline_signal_bucket = np.where(
        failure_context == 'technical_timeout', 'technical', 
        np.random.choice(['technical', 'soft'], size=n, p=[0.1, 0.9])
    )
    
    observables = pd.DataFrame({
        'event_id': np.arange(n),
        'customer_id': customer_ids,
        'tenure_days': tenure_days,
        'plan_tier': plan_tier,
        'amount': avg_order_value, # Treat AOV as the failed amount for simplicity
        'historical_payment_count': historical_payment_count,
        'historical_failure_count': historical_failure_count,
        'days_since_last_failure': days_since_last_failure,
        'payment_method': payment_method,
        'email_verified': email_verified,
        'whatsapp_opted_in': whatsapp_opted_in,
        'engagement_score': engagement_score,
        'failure_context': failure_context,
        'day_of_month': day_of_month,
        'decline_signal_bucket': decline_signal_bucket
    })
    
    # --- 2. Generate Hidden Ground Truth ---
    true_ltv = avg_order_value * np.random.lognormal(mean=1, sigma=0.5, size=n)
    true_liquidity_offset = np.random.normal(loc=0, scale=1, size=n)
    true_frustration_threshold = np.random.beta(a=2, b=2, size=n)
    
    # Base affinity (intrinsic responsiveness)
    true_channel_affinity_retry = np.random.beta(a=2, b=5, size=n) 
    true_channel_affinity_whatsapp = np.random.beta(a=2, b=8, size=n) + (engagement_score * 0.3)
    
    # Adjust affinities based on opt-in
    true_channel_affinity_whatsapp = np.where(whatsapp_opted_in, true_channel_affinity_whatsapp, 0.0)
    
    # Compute true baseline self-cure probability (Y_none)
    # technical_timeout has high self-cure
    # insufficient_funds has lower self-cure, depends on engagement and liquidity offset
    base_prob = np.zeros(n)
    
    mask_tech = failure_context == 'technical_timeout'
    mask_funds = failure_context == 'insufficient_funds'
    
    base_prob[mask_tech] = np.random.normal(0.7, 0.1, size=n)[mask_tech]
    base_prob[mask_funds] = (0.2 + 0.1 * engagement_score[mask_funds] + 0.05 * true_liquidity_offset[mask_funds])
    
    # Enterprise might have slightly higher baseline payment ability
    base_prob += np.where(plan_tier == 'Enterprise', 0.1, 0)
    base_prob += np.where(plan_tier == 'Pro', 0.05, 0)
    
    true_baseline_self_cure_prob = np.clip(base_prob, 0.01, 0.95)
    
    # Compute uplift from interventions
    # Retry uplift
    retry_uplift = np.zeros(n)
    retry_uplift[mask_tech] = np.random.normal(0.05, 0.02, size=n)[mask_tech] * true_channel_affinity_retry[mask_tech]
    # For insufficient funds, retry effect depends on liquidity offset
    retry_uplift[mask_funds] = (0.1 + 0.1 * (true_liquidity_offset[mask_funds] > 0)) * true_channel_affinity_retry[mask_funds]
    
    # WhatsApp uplift
    wa_uplift = np.zeros(n)
    wa_uplift[mask_tech] = np.random.normal(0.01, 0.01, size=n)[mask_tech] * true_channel_affinity_whatsapp[mask_tech]
    wa_uplift[mask_funds] = (0.15 + 0.2 * engagement_score[mask_funds]) * true_channel_affinity_whatsapp[mask_funds]
    
    # Potential Probabilities
    prob_none = true_baseline_self_cure_prob
    prob_retry = np.clip(prob_none + retry_uplift, 0.0, 0.99)
    prob_wa = np.clip(prob_none + wa_uplift, 0.0, 0.99)
    
    # Generate potential outcomes (binary 0 or 1)
    # We use a correlated uniform random variable to ensure that if someone pays under 'none', 
    # they would also likely pay under 'retry' if prob_retry > prob_none.
    u = np.random.uniform(0, 1, size=n)
    
    y_none = (u < prob_none).astype(int)
    y_retry = (u < prob_retry).astype(int)
    y_whatsapp = (u < prob_wa).astype(int)
    
    hidden = pd.DataFrame({
        'event_id': np.arange(n),
        'true_ltv': true_ltv,
        'true_liquidity_offset': true_liquidity_offset,
        'true_frustration_threshold': true_frustration_threshold,
        'true_channel_affinity_retry': true_channel_affinity_retry,
        'true_channel_affinity_whatsapp': true_channel_affinity_whatsapp,
        'true_baseline_self_cure_prob': true_baseline_self_cure_prob,
        'prob_none': true_baseline_self_cure_prob,
        'prob_retry': prob_retry,
        'prob_whatsapp': prob_wa,
        'Y_none': y_none,
        'Y_retry': y_retry,
        'Y_whatsapp': y_whatsapp
    })
    
    return observables, hidden

if __name__ == "__main__":
    config = SimulatorConfig()
    obs, hid = generate_events(config)
    print(f"Generated {len(obs)} events.")
    print("Observables preview:")
    print(obs.head())
    print("\nHidden preview:")
    print(hid.head())
    print(f"\nAverage Self-cure rate: {hid['Y_none'].mean():.3f}")
    print(f"Average Retry rate: {hid['Y_retry'].mean():.3f}")
    print(f"Average WA rate: {hid['Y_whatsapp'].mean():.3f}")
