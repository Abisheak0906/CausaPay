"""Experimental V2: cross-fitted, pairwise multi-arm R-learner.

For each active action a in {retry, whatsapp}, V2 restricts the R-loss to
observations assigned a or none.  With pi_a(x)=P(A=a | A in {a, none}, X),
it cross-fits m_a0(x)=E[Y | A in {a, none}, X] and learns tau_a by minimizing
the residualized loss (Y-m_a0(X) - (I[A=a]-pi_a(X))*tau_a(X))^2.

This is deliberately separate from V1 AIPW potential-outcome regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from causal_engine.aipw_estimator import AIPWEstimator


class DirectMultiArmCateEstimator(AIPWEstimator):
    """Multi-arm direct CATE learner using `none` as the fixed reference."""

    active_treatments = ('retry', 'whatsapp')

    def __init__(self, clip_threshold: float = .10, n_splits: int = 5, random_state: int = 42):
        super().__init__(clip_threshold=clip_threshold, n_splits=n_splits)
        self.random_state = random_state
        self.tau_models: dict[str, Pipeline] = {}
        self.none_model: Pipeline | None = None
        self._oof_tau: dict[str, np.ndarray] = {}
        self._oof_pair_outcome: dict[str, np.ndarray] = {}

    def _regressor(self, seed_offset: int = 0) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=self.random_state + seed_offset, n_jobs=1,
        )

    def _propensity_model(self) -> Pipeline:
        return Pipeline([
            ('preprocessor', clone(self.preprocessor)),
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)),
        ])

    def fit(self, observables: pd.DataFrame):
        forbidden = ['true_baseline', 'true_channel', 'true_liquidity', 'true_frustration', 'true_ltv', 'Y_none', 'Y_retry', 'Y_whatsapp']
        if any(column in forbidden for column in observables.columns):
            raise ValueError('Data Leakage detected: hidden ground-truth columns are forbidden.')
        X = self._prepare_features(observables)
        y = observables['outcome_recovered'].to_numpy(dtype=float)
        actions = observables['intervention_assigned'].to_numpy()
        if len(X) < 2:
            raise ValueError('V2 fitting requires at least two training samples.')
        folds = KFold(n_splits=min(self.n_splits, len(X)), shuffle=True, random_state=self.random_state)
        action_index = {action: index for index, action in enumerate(self.treatments)}
        oof_propensity = np.zeros((len(X), len(self.treatments)))
        oof_pair_outcome = {action: np.full(len(X), np.nan) for action in self.active_treatments}
        oof_tau_target = {action: np.full(len(X), np.nan) for action in self.active_treatments}
        oof_weight = {action: np.zeros(len(X)) for action in self.active_treatments}

        for fold, (train_idx, valid_idx) in enumerate(folds.split(X)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid, a_train = y[train_idx], y[valid_idx], actions[train_idx]
            propensity = self._propensity_model()
            propensity.fit(X_train, a_train)
            predicted = propensity.predict_proba(X_valid)
            for class_index, action in enumerate(propensity.named_steps['clf'].classes_):
                oof_propensity[valid_idx, action_index[action]] = predicted[:, class_index]

            for action_number, action in enumerate(self.active_treatments, start=1):
                pair_train = np.isin(a_train, ['none', action])
                pair_valid = np.isin(actions[valid_idx], ['none', action])
                if pair_train.sum() < 2 or not pair_valid.any():
                    continue
                outcome = Pipeline([('preprocessor', clone(self.preprocessor)), ('reg', self._regressor(100 * fold + action_number))])
                outcome.fit(X_train.iloc[pair_train], y_train[pair_train])
                m_valid = outcome.predict(X_valid.iloc[pair_valid])
                selected_valid = valid_idx[pair_valid]
                # Multiclass propensity converted to the explicit a-vs-none contrast.
                raw_a = oof_propensity[selected_valid, action_index[action]]
                raw_none = oof_propensity[selected_valid, action_index['none']]
                clipped_a = np.clip(raw_a, self.clip_threshold, 1 - self.clip_threshold)
                clipped_none = np.clip(raw_none, self.clip_threshold, 1 - self.clip_threshold)
                pi = clipped_a / (clipped_a + clipped_none)
                treatment_residual = (actions[selected_valid] == action).astype(float) - pi
                oof_pair_outcome[action][selected_valid] = m_valid
                oof_tau_target[action][selected_valid] = (y[selected_valid] - m_valid) / treatment_residual
                oof_weight[action][selected_valid] = treatment_residual ** 2

        self._oof_propensity = oof_propensity
        self._train_actions = actions.copy()
        self._train_size = len(X)
        self._oof_pair_outcome = oof_pair_outcome
        self._oof_tau = oof_tau_target
        self.diagnostics['trimmed_propensities_count'] = int((oof_propensity < self.clip_threshold).any(axis=1).sum())

        none_mask = actions == 'none'
        self.none_model = Pipeline([('preprocessor', clone(self.preprocessor)), ('reg', self._regressor(900))])
        self.none_model.fit(X.iloc[none_mask], y[none_mask])
        for action_number, action in enumerate(self.active_treatments, start=1):
            pair = np.isin(actions, ['none', action])
            valid = pair & np.isfinite(oof_tau_target[action]) & (oof_weight[action] > 0)
            model = Pipeline([('preprocessor', clone(self.preprocessor)), ('reg', self._regressor(1000 + action_number))])
            model.fit(X.iloc[valid], oof_tau_target[action][valid], reg__sample_weight=oof_weight[action][valid])
            self.tau_models[action] = model
        return self

    @staticmethod
    def _tree_std(model: Pipeline, X) -> np.ndarray:
        transformed = model.named_steps['preprocessor'].transform(X)
        predictions = np.asarray([tree.predict(transformed) for tree in model.named_steps['reg'].estimators_])
        return predictions.std(axis=0)

    def predict_counterfactuals(self, observables: pd.DataFrame) -> pd.DataFrame:
        if self.none_model is None:
            raise ValueError('V2 estimator must be fitted before prediction.')
        X = self._prepare_features(observables)
        p_none = np.clip(self.none_model.predict(X), 0, 1)
        std_none = self._tree_std(self.none_model, X)
        result = pd.DataFrame({'event_id': observables['event_id'], 'prob_none': p_none, 'std_none': std_none}, index=observables.index)
        for action in self.active_treatments:
            tau = self.tau_models[action].predict(X)
            std_tau = self._tree_std(self.tau_models[action], X)
            result[f'prob_{action}'] = np.clip(p_none + tau, 0, 1)
            result[f'std_{action}'] = np.sqrt(std_none ** 2 + std_tau ** 2)
        return result
