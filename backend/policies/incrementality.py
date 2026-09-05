import pandas as pd
import numpy as np
from causal_engine.aipw_estimator import AIPWEstimator
from policies.baseline import GrossRecoveryBaseline

class IncrementalityAwarePolicy:
    """
    Policy B: Incrementality-Aware Policy
    Uses the Causal Engine estimates to make decisions.
    Chooses the intervention that maximizes Expected Net Incremental Value (ENIV).
    Falls back to Baseline if uncertainty is too high.
    """
    def __init__(self, causal_engine: AIPWEstimator, baseline_policy: GrossRecoveryBaseline):
        self.causal_engine = causal_engine
        self.baseline_policy = baseline_policy
        
        # Simulated Costs for interventions (INR)
        self.costs = {
            'none': 0.0,
            'retry': 2.0,       # processing/network cost
            'whatsapp': 15.0    # API messaging template cost
        }
        
        # Max acceptable standard deviation in predictions
        self.uncertainty_threshold = 1.0
        
    def predict(self, observables: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with 'recommended_action' and supporting metrics.
        """
        counterfactuals = self.causal_engine.predict_counterfactuals(observables)
        baseline_actions = self.baseline_policy.predict(observables)
        
        results = pd.DataFrame(index=observables.index)
        results['event_id'] = observables['event_id']
        
        actions = []
        is_abstain = []
        
        amounts = observables['amount'].values
        
        # Vectorized calculations for speed
        p_none = counterfactuals['prob_none'].values
        p_retry = counterfactuals['prob_retry'].values
        p_wa = counterfactuals['prob_whatsapp'].values
        
        std_retry = counterfactuals['std_retry'].values
        std_wa = counterfactuals['std_whatsapp'].values
        
        inc_prob_retry = p_retry - p_none
        inc_prob_wa = p_wa - p_none
        
        eniv_none = np.zeros(len(amounts))
        eniv_retry = (inc_prob_retry * amounts) - self.costs['retry']
        eniv_wa = (inc_prob_wa * amounts) - self.costs['whatsapp']
        
        results['eniv_none'] = eniv_none
        results['eniv_retry'] = eniv_retry
        results['eniv_wa'] = eniv_wa
        results['inc_prob_retry'] = inc_prob_retry
        results['inc_prob_wa'] = inc_prob_wa
        results['prob_none'] = p_none
        results['prob_retry'] = p_retry
        results['prob_wa'] = p_wa
        results['std_retry'] = std_retry
        results['std_wa'] = std_wa
        
        # Determine WhatsApp eligibility from observables; fall back to allowing WA if field missing
        wa_opt_in = observables['whatsapp_opted_in'].astype(bool).values if 'whatsapp_opted_in' in observables.columns else np.ones(len(amounts), dtype=bool)

        for i in range(len(amounts)):
            if std_retry[i] > self.uncertainty_threshold or std_wa[i] > self.uncertainty_threshold:
                # Abstain: use baseline but ensure WA is not chosen if not opted-in
                base_action = baseline_actions[i]
                if (not wa_opt_in[i]) and base_action == 'whatsapp':
                    actions.append('none')
                else:
                    actions.append(base_action)
                is_abstain.append(True)
            else:
                is_abstain.append(False)
                # Exclude WhatsApp from final selection when the customer is not opted in
                candidate_wa = eniv_wa[i] if wa_opt_in[i] else -np.inf
                best_eniv = max(eniv_none[i], eniv_retry[i], candidate_wa)

                # Resolving ties/preferences: Prefer none -> retry -> whatsapp
                if best_eniv <= 0:
                    actions.append('none')
                elif best_eniv == eniv_retry[i]:
                    actions.append('retry')
                else:
                    actions.append('whatsapp')

        results['recommended_action'] = actions
        results['is_abstain'] = is_abstain
        results['causal_preferred_action'] = np.where(
            (eniv_retry > eniv_none) & ((eniv_retry >= eniv_wa) | ~wa_opt_in), 'retry',
            np.where((eniv_wa > eniv_none) & wa_opt_in, 'whatsapp', 'none')
        )
        results['fallback_action'] = np.where(results['is_abstain'], baseline_actions, None)
        return results
