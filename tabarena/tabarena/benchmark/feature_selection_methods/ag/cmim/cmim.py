"""Conditional Mutual Information Maximization (CMIM) feature selection."""
from __future__ import annotations

import time
import numpy as np

from typing import TYPE_CHECKING

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractITFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class CMIMFeatureSelector(AbstractITFeatureSelector):
    """CMIM Feature Selection.

    Journal of Machine learning research 5.Nov (2004): 1531-1555.
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/CMIM.py#L4.
    The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Add max_features (number of features to be maximally selected by the method) constraint
                           - Use pandas instead of numpy and avoid conversion
    """

    name = "CMIMFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(  # noqa: C901
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> list[str]:
        start_time = time.monotonic()
        n_features = len(X.columns)

        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        cols = [X_pre.iloc[:, i] for i in range(n_features)]

        CMIM = np.array([self._mutual_information(col, y) for col in cols])
        m = -np.ones(n_features, dtype=int)
        F: list[int] = []

        for k in range(min(n_features, self.max_features)): # includes early stopping
            if self._timed_out(time_limit, start_time):
                break

            # Choose the feature with the highest MI as the next feature
            idx = np.argmax(CMIM)
            F.append(idx)
            CMIM[idx] = -np.inf
            if len(F) == self.max_features:
                break

            sstar = -np.inf # start with really low value for best partial score sstar
            for i in range(n_features):
                if self._timed_out(time_limit, start_time):
                    break
                if i in F:
                    continue
                while (CMIM[i] > sstar) and (m[i] < k):
                    if self._timed_out(time_limit, start_time):
                        break
                    m[i] += 1
                    CMIM[i] = min(CMIM[i], self._conditional_mutual_information(cols[i], y, cols[int(F[int(m[i])])]))
                if CMIM[i] > sstar:
                    sstar = CMIM[i]
        
        selected_features = [self._original_features[i] for i in F]
        if len(selected_features) < self.max_features:
            selected_features += self.fallback_feature_selection(selected_features=selected_features)
        return [str(feat) for feat in selected_features]

