"""Correlation-based Feature Selection (CFS)."""
from __future__ import annotations

import time
import numpy as np

from typing import TYPE_CHECKING

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractITFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class CFSFeatureSelector(AbstractITFeatureSelector):
    """Correlation-based Forward Selection (CFS).

    Reference: Hall, Mark A. Correlation-based feature selection for machine learning. Diss. The University of Waikato, 1999.
    Implementation Source:  https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/statistical_based/CFS.py#L40.
    The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    This particular implementation of the repo is based on http://featureselection.asu.edu, which for the CFS algorithm cites
    Hall, Mark A., and Lloyd A. Smith. "Feature selection for machine learning: comparing a correlation-based filter approach to the wrapper." 
    Proceedings of the twelfth international Florida artificial intelligence research society conference. 1999.
    The variation implemented here is a forward selection method using Symmetrical Uncertainty.
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Replaced merit-based early stopping with max_features constraint
                           - Pad with random fallback if timeout cuts loop short
                           - Use pandas instead of numpy and avoid conversion
    """

    name = "CFSFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(
        self, 
        *, 
        X: pd.DataFrame, 
        y: pd.Series, 
        time_limit: int | None = None
    ) -> list[str]:
        start_time = time.monotonic()
        F = []  # cfs score

        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)

        while len(F) < self.max_features:
            merit = -np.inf
            idx = -1
            for i in range(len(X_pre.columns)):
                if self._timed_out(time_limit, start_time):
                    break
                if i not in F:
                    F.append(i)
                    # calculate the merit of current selected features
                    t = self.merit_calculation(X_pre.iloc[:, F], y)
                    if t > merit:
                        merit = t
                        idx = i
                    F.pop()
            if idx == -1:
                break
            F.append(idx)
        selected_features = [self._original_features[i] for i in F]
        if len(selected_features) < self.max_features:
            selected_features += self.fallback_feature_selection(selected_features=selected_features)
        return [str(feat) for feat in selected_features]

    def merit_calculation(self, X, y):
        """This function calculates the merit of X given class labels y, where
        merits = (k * rcf) / sqrt (k + k*(k-1)*rff)
        rcf = (1/k)*sum(su(fi, y)) for all fi in X
        rff = (1/(k*(k-1)))*sum(su(fi, fj)) for all fi and fj in X.

        :param X:  {numpy array}, shape (n_samples, n_features) input data
        :param y:  {numpy array}, shape (n_samples) input class labels
        :return merits: {float}  merit of a feature subset X
        """
        rff = 0
        rcf = 0
        for i in range(len(X.columns)):
            fi = X.iloc[:, i]
            rcf += self._symmetrical_uncertainty(fi, y)  # su is the symmetrical uncertainty of fi and y
            for j in range(i + 1, len(X.columns)):
                fj = X.iloc[:, j]
                rff += self._symmetrical_uncertainty(fi, fj)
        rff *= 2
        return rcf / np.sqrt(len(X.columns) + rff)
