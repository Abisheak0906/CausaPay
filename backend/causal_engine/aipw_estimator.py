import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import warnings

class AIPWEstimator:
    """
    Augmented Inverse Propensity Weighting (AIPW) / Doubly Robust Estimator
    for a multi-arm treatment setting with Cross-Fitting (Double Machine Learning style).

    Treatments: ['none', 'retry', 'whatsapp']

    Formula for pseudo-outcome of observation i and treatment a (doubly robust):
    Gamma_{i,a} = m_a(X_i) + (I(A_i = a) / e_a(X_i)) * (Y_i - m_a(X_i))

    NOTE: Γ_{i,a} is a *pseudo-outcome* for the potential outcome under treatment a, not an individual treatment effect.

    Where:
    m_a(X_i) = E[Y | A=a, X_i] (Outcome Model)
    e_a(X_i) = P(A=a | X_i) (Propensity Model) (estimated using regularized LogisticRegression)
    """
    def __init__(self, clip_threshold=0.10, n_splits=5):
        self.clip_threshold = clip_threshold
        self.n_splits = n_splits
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

        self.final_models = {}
        self.diagnostics = {}
        # Stored after fit() for live diagnostic endpoints
        self._oof_propensity: np.ndarray | None = None   # shape (n_train, 3) — raw OOF propensities
        self._oof_outcome_predictions: np.ndarray | None = None
        self._train_actions: np.ndarray | None = None
        self._train_size: int = 0

    def _prepare_features(self, df: pd.DataFrame):
        df = df.copy()

        # Ensure all numeric features exist with sensible defaults
        numeric_defaults = {
            'tenure_days': 0,
            'amount': 0.0,
            'historical_payment_count': 0,
            'historical_failure_count': 0,
            'days_since_last_failure': -1,  # sentinel; replaced with 365 below
            'engagement_score': 0.0,
            'day_of_month': 1,
        }
        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default

        # Replace the sentinel for 'days_since_last_failure' with a large value
        # indicating a long gap when unknown.
        df['days_since_last_failure'] = df['days_since_last_failure'].fillna(-1).replace(-1, 365)

        # Ensure categorical features exist
        categorical_defaults = ['plan_tier', 'payment_method', 'failure_context', 'decline_signal_bucket']
        for col in categorical_defaults:
            if col not in df.columns:
                df[col] = 'missing'
            else:
                df[col] = df[col].fillna('missing')

        # Ensure boolean features exist and are normalized
        boolean_features = ['email_verified', 'whatsapp_opted_in']
        for col in boolean_features:
            if col not in df.columns:
                df[col] = False
            df[col] = df[col].fillna(False)

        # Now assemble feature DataFrame in a stable order
        X = df[[
            'tenure_days', 'amount', 'historical_payment_count',
            'historical_failure_count', 'days_since_last_failure',
            'engagement_score', 'day_of_month', 'plan_tier',
            'payment_method', 'failure_context', 'decline_signal_bucket',
            'email_verified', 'whatsapp_opted_in'
        ]].copy()

        # Convert booleans to ints for the preprocessor
        for col in boolean_features:
            X[col] = X[col].astype(int)

        return X

    def fit(self, observables: pd.DataFrame):
        # 1. Strict Data Leakage Check
        forbidden = ['true_baseline', 'true_channel', 'true_liquidity', 'true_frustration', 'true_ltv', 'Y_none', 'Y_retry', 'Y_whatsapp']
        for col in observables.columns:
            if col in forbidden:
                raise ValueError(f"Data Leakage detected! Hidden ground truth column '{col}' is in the training data.")

        X = self._prepare_features(observables)
        y = observables['outcome_recovered'].values
        A = observables['intervention_assigned'].values

        n = len(X)
        if n < 2:
            raise ValueError('AIPW fitting requires at least 2 training samples.')

        num_treatments = len(self.treatments)
        treatment_to_idx = {t: i for i, t in enumerate(self.treatments)}

        n_splits = min(self.n_splits, n)
        if n_splits < 2:
            raise ValueError('AIPW fitting requires at least 2 training samples for cross-validation.')

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_m = np.zeros((n, num_treatments))
        oof_e = np.zeros((n, num_treatments))

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            A_train, A_val = A[train_idx], A[val_idx]

            # Outcome Model (m)
            for t in self.treatments:
                mask = (A_train == t)
                if mask.sum() > 0:
                    model = Pipeline([('preprocessor', self.preprocessor), ('reg', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=1))])
                    model.fit(X_train[mask], y_train[mask])
                    oof_m[val_idx, treatment_to_idx[t]] = model.predict(X_val)

            # Propensity Model (e)
            e_model = Pipeline([('preprocessor', self.preprocessor), ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000))])
            e_model.fit(X_train, A_train)
            predicted = e_model.predict_proba(X_val)
            # Make the column/action mapping explicit instead of relying on
            # lexical class ordering from scikit-learn.
            for class_idx, treatment in enumerate(e_model.named_steps['clf'].classes_):
                oof_e[val_idx, treatment_to_idx[treatment]] = predicted[:, class_idx]

        # 2. Propensity Diagnostics & principled clipping
        min_prop = oof_e.min(axis=0)
        max_prop = oof_e.max(axis=0)

        self.diagnostics['min_propensity'] = {self.treatments[i]: min_prop[i] for i in range(num_treatments)}
        self.diagnostics['max_propensity'] = {self.treatments[i]: max_prop[i] for i in range(num_treatments)}

        # Trim extreme propensities to enforce overlap/positivity without breaking sum=1 too much
        # If we just clip, they don't sum to 1. So we clip and then renormalize.
        clipped_e = np.clip(oof_e, self.clip_threshold, 1.0 - self.clip_threshold)
        row_sums = clipped_e.sum(axis=1, keepdims=True)
        clipped_e = clipped_e / row_sums

        affected_rows = (oof_e != clipped_e).any(axis=1).sum()
        self.diagnostics['trimmed_propensities_count'] = int(affected_rows)

        # Store raw OOF propensities for live diagnostic endpoints
        self._oof_propensity = oof_e.copy()
        # Diagnostic-only: out-of-fold nuisance predictions, never used by the
        # final stage or policy. Keeping them permits honest validation against
        # synthetic conditional means.
        self._oof_outcome_predictions = oof_m.copy()
        self._train_actions = A.copy()
        self._train_size = n

        # 3. Compute AIPW Pseudo-outcomes
        gamma = np.zeros((n, num_treatments))

        for t_idx, t in enumerate(self.treatments):
            m_a = oof_m[:, t_idx]
            e_a = clipped_e[:, t_idx]
            I_a = (A == t).astype(float)

            # Formula: Gamma_{i,a} = m_a(X_i) + I(A_i=a)/e_a(X_i) * (Y_i - m_a(X_i))
            gamma[:, t_idx] = m_a + (I_a / e_a) * (y - m_a)

            # 4. Fit Final Stage CATE Models g_a(X) to Gamma_{i,a}
            # This regresses the pseudo-outcome on covariates to estimate the potential outcome E[Y(a) | X]
            final_model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('reg', RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=1))
            ])
            final_model.fit(X, gamma[:, t_idx])
            self.final_models[t] = final_model

    def predict_counterfactuals(self, observables: pd.DataFrame) -> pd.DataFrame:
        forbidden = ['true_baseline', 'true_channel', 'true_liquidity', 'true_frustration', 'true_ltv', 'Y_none', 'Y_retry', 'Y_whatsapp']
        for col in observables.columns:
            if col in forbidden:
                raise ValueError(f"Data Leakage detected! Hidden ground truth column '{col}' is in the test data.")

        X = self._prepare_features(observables)
        results = pd.DataFrame(index=observables.index)
        results['event_id'] = observables['event_id']

        for t in self.treatments:
            model = self.final_models[t]
            # Predict the estimated potential outcome g_a(X)
            preds = model.predict(X)

            # Since pseudo-outcomes can be outside [0,1], the regressor might predict outside [0,1].
            # Mathematically justified clipping to valid probability range [0,1].
            results[f'prob_{t}'] = np.clip(preds, 0.0, 1.0)

            # Uncertainty estimation: variance of trees in the final stage regressor
            reg = model.named_steps['reg']
            X_trans = model.named_steps['preprocessor'].transform(X)
            tree_preds = np.array([tree.predict(X_trans) for tree in reg.estimators_])
            results[f'std_{t}'] = np.std(tree_preds, axis=0)

        return results

    @staticmethod
    def _histogram(values: np.ndarray, bins: int = 12) -> list[dict]:
        """JSON-friendly histogram used by the UI; never exposes raw customer rows."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return []
        counts, edges = np.histogram(values, bins=bins, range=(0.0, max(1.0, float(values.max()))))
        return [
            {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ]

    def propensity_overlap_diagnostics(self) -> dict:
        """Describe observed support in OOF propensity estimates, not causal validity."""
        if self._oof_propensity is None or self._train_actions is None:
            raise ValueError("Propensity diagnostics are available after fit().")
        groups = []
        intervals = []
        for idx, treatment in enumerate(self.treatments):
            values = self._oof_propensity[:, idx]
            assigned = values[self._train_actions == treatment]
            q = np.quantile(values, [0, .01, .05, .5, .95, .99, 1]).tolist()
            intervals.append((q[2], q[4]))
            groups.append({
                "treatment": treatment,
                "assigned_count": int((self._train_actions == treatment).sum()),
                "all_rows": {"min": q[0], "p01": q[1], "p05": q[2], "median": q[3], "p95": q[4], "p99": q[5], "max": q[6], "mean": float(values.mean())},
                "assigned_rows": {"min": float(assigned.min()) if len(assigned) else None, "mean": float(assigned.mean()) if len(assigned) else None, "max": float(assigned.max()) if len(assigned) else None},
                "histogram": self._histogram(values),
            })
        lower, upper = max(x[0] for x in intervals), min(x[1] for x in intervals)
        has_support = lower < upper
        return {
            "source": "cross_fitted_training_propensities",
            "groups": groups,
            "central_90_percent_common_interval": {"lower": float(lower), "upper": float(upper), "has_common_support": bool(has_support)},
            "interpretation": (
                "The central propensity ranges have shared support across actions." if has_support
                else "The central propensity ranges do not share support across all actions; estimates in sparse regions need caution."
            ),
            "limitation": "Observed propensity overlap supports a positivity check only; it cannot rule out unmeasured confounding.",
        }

    def uncertainty_diagnostics(self, observables: pd.DataFrame) -> dict:
        cf = self.predict_counterfactuals(observables)
        def summary(values: np.ndarray) -> dict:
            return {"min": float(values.min()), "p05": float(np.quantile(values, .05)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "max": float(values.max()), "mean": float(values.mean()), "histogram": self._histogram(values)}
        return {"retry": summary(cf['std_retry'].to_numpy()), "whatsapp": summary(cf['std_whatsapp'].to_numpy())}
