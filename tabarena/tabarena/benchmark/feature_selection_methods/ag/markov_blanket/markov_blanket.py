"""Markov blanket feature selection."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector


class MarkovBlanketFeatureSelector(AbstractFeatureSelector):
    """MarkovBlanket Feature Selection.

    Reference: Daphne Koller and Mehran Sahami. "Toward optimal feature selection." 
    ICML. Vol. 96. No. 28. 1996.
    """

    name = "MarkovBlanketFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> list[str]:
        start_time = time.monotonic()
        use_fi_in_delta = False
        cv_splits = 5
        k = 5

        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)

        current_features = list(X_pre.columns)

        # Pearson correlation matrix (p_ij)
        corr = X_pre.corr(method="pearson").abs()

        if self.problem_type == "regression":
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            clf = make_pipeline(StandardScaler(), Ridge())
            scoring = "neg_mean_squared_error"
        else:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            scoring = "neg_log_loss"

        while len(current_features) > self.max_features:
            if self._timed_out(time_limit, start_time):
                break

            worst_feat = None
            lowest_delta = np.inf

            for fi in current_features:
                if self._timed_out(time_limit, start_time):
                    break

                others = [fj for fj in current_features if fj != fi]
                if not others:
                    continue

                Mi = list(corr.loc[fi, others].sort_values(ascending=False).head(k).index)
                feat_set = Mi + ([fi] if use_fi_in_delta else [])

                # expected negative losses (sklearn default) need to be negated
                scores = cross_val_score(clf, X_pre[feat_set], y, cv=cv, scoring=scoring, error_score="raise")
                delta = float(-scores.mean())

                if delta < lowest_delta:
                    lowest_delta = delta
                    worst_feat = fi

            if worst_feat is None:
                break

            current_features.remove(worst_feat)

        return [str(feat) for feat in current_features]
