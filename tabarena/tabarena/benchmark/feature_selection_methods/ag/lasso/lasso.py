"""Lasso feature selection."""
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV, LogisticRegressionCV

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector


class LassoFeatureSelector(AbstractFeatureSelector):
    """Lasso Feature Selection.

    Reference: Tibshirani, Robert. "Regression shrinkage and selection via the lasso." 
    Journal of the Royal Statistical Society Series B:  Statistical Methodology 58.1 (1996): 267-288.
    Implementation Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    name = "LassoFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True, scale=True)
        
        if self.problem_type == "regression":
            lasso = LassoCV(random_state=self.random_state)
        else:
            lasso = LogisticRegressionCV(
                penalty="l1", 
                solver="liblinear",
                random_state=self.random_state
            )
        lasso.fit(X_pre, y)
        
        coef = lasso.coef_
        if coef.ndim == 2:
            scores = np.abs(coef).mean(axis=0) # for multiclass average over classes
        else:
            scores = np.abs(coef)
        return dict(zip(X_pre.columns, scores))
