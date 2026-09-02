import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TLearner:
    """
    Causal Engine based on T-Learner (Two-model approach, extended to K models).
    Fits a separate ML model for each treatment arm to predict the outcome probability.
    """
    def __init__(self):
        self.models = {}
        self.treatments = ['none', 'retry', 'whatsapp']
        
        numeric_features = ['tenure_days', 'amount', 'historical_payment_count', 
                            'historical_failure_count', 'days_since_last_failure', 
                            'engagement_score', 'day_of_month']
        categorical_features = ['plan_tier', 'payment_method', 'failure_context', 'decline_signal_bucket']
        boolean_features = ['email_verified', 'whatsapp_opted_in']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('bool', 'passthrough', boolean_features)
            ])
            
    def _prepare_features(self, df: pd.DataFrame):
        df = df.copy()
        df['days_since_last_failure'] = df['days_since_last_failure'].replace(-1, 365)
        X = df[['tenure_days', 'amount', 'historical_payment_count', 
               'historical_failure_count', 'days_since_last_failure', 
               'engagement_score', 'day_of_month', 'plan_tier', 
               'payment_method', 'failure_context', 'decline_signal_bucket',
               'email_verified', 'whatsapp_opted_in']].copy()
        
        # Convert bools to int
        boolean_features = ['email_verified', 'whatsapp_opted_in']
        for col in boolean_features:
            X[col] = X[col].astype(int)
            
        return X

    def fit(self, observables: pd.DataFrame):
        X = self._prepare_features(observables)
        y = observables['outcome_recovered']
        t = observables['intervention_assigned']
        
        for treatment in self.treatments:
            mask = t == treatment
            X_t = X[mask]
            y_t = y[mask]
            
            # Simple RandomForest to capture non-linear effects
            model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42))
            ])
            
            model.fit(X_t, y_t)
            self.models[treatment] = model
            
    def predict_counterfactuals(self, observables: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe with estimated probabilities for each treatment.
        Also returns standard deviation as a proxy for uncertainty.
        """
        X = self._prepare_features(observables)
        
        results = pd.DataFrame(index=observables.index)
        results['event_id'] = observables['event_id']
        
        for treatment in self.treatments:
            model = self.models[treatment]
            probs = model.predict_proba(X)[:, 1]
            results[f'prob_{treatment}'] = probs
            
            # Extract estimators to get variance (uncertainty)
            rf = model.named_steps['classifier']
            X_transformed = model.named_steps['preprocessor'].transform(X)
            
            # Predict from all trees
            tree_preds = np.array([tree.predict_proba(X_transformed)[:, 1] for tree in rf.estimators_])
            results[f'std_{treatment}'] = np.std(tree_preds, axis=0)
            
        return results
