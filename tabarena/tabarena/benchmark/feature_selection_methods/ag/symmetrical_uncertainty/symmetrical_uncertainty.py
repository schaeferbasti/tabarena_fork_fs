"""Symmetrical Uncertainty feature selection."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractITFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class SymmetricalUncertaintyFeatureSelector(AbstractITFeatureSelector):
    """Symmetrical Uncertainty Feature Selection.

    SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y))

    Reference: Press, W.H. et al. (1992). Numerical Recipes in C (2nd ed.).
    The form used here matches Hall (1999), where SU is the building block of CFS.
    """

    name = "SymmetricalUncertaintyFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,
    ) -> dict[str, float]:
        """SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y))."""
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        
        scores = {}
        for col in X_pre.columns:
            if self._timed_out(time_limit, start_time):
                break
            scores[col] = self._symmetrical_uncertainty(X_pre[col], y)
        return scores