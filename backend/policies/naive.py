import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class NaiveLikelihoodPolicy:
    """Non-causal comparator: target likely recoverers, without using treatment lift."""
    def __init__(self, preprocessor, baseline_policy):
        self.preprocessor = preprocessor
        self.baseline_policy = baseline_policy
        self.model = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=1))])

    def fit(self, observables: pd.DataFrame):
        X = observables.drop(columns=['outcome_recovered', 'intervention_assigned', 'amount_recovered', 'time_to_recovery_hours'], errors='ignore')
        self.model.fit(X, observables['outcome_recovered'].astype(int))
        return self

    def predict(self, observables: pd.DataFrame, intervention_budget: int) -> np.ndarray:
        """Treat the most likely recoverers at a matched intervention volume.

        Action is the historical gross-recovery baseline; ranking deliberately ignores
        incremental effect, which makes this a useful (not causal) comparison.
        """
        X = observables.drop(columns=['outcome_recovered', 'intervention_assigned', 'amount_recovered', 'time_to_recovery_hours'], errors='ignore')
        likelihood = self.model.predict_proba(X)[:, list(self.model.named_steps['clf'].classes_).index(1)]
        actions = np.full(len(observables), 'none', dtype=object)
        if intervention_budget <= 0:
            return actions
        for idx in np.argsort(-likelihood)[:min(intervention_budget, len(actions))]:
            candidate = self.baseline_policy.predict(observables.iloc[[idx]])[0]
            if candidate == 'whatsapp' and not bool(observables.iloc[idx].get('whatsapp_opted_in', False)):
                candidate = 'retry'
            actions[idx] = candidate if candidate in {'retry', 'whatsapp'} else 'retry'
        return actions
