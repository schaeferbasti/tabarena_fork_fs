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
    Changes to the implementation:
        - Time and max_features constraint
        - Use pandas instead of numpy and avoid conversion
    """

    name = "CMIMFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(  # noqa: C901
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> list[str]:
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        n_features = len(X_pre.columns)
        cols = [X_pre.iloc[:, i] for i in range(n_features)]

        partial_scores = np.array([self._mutual_information(col, y) for col in cols])
        m = -np.ones(n_features, dtype=int) 
        selected_idx: list[int] = [] # selected features indices
        selected_mask = np.zeros(n_features, dtype=bool)

        # Fleuret's algorithm
        for k in range(min(n_features, self.max_features)): # stop when max_features reached
            if self._timed_out(time_limit, start_time):
                break

            idx = int(np.argmax(partial_scores))  # next feature with the highest cmi
            selected_idx.append(idx)
            selected_mask[idx] = True
            partial_scores[idx] = -np.inf
            
            if len(selected_idx) == self.max_features:
                break

            sstar = -np.inf # keeps lowest mi score 
            for i in range(n_features):
                if selected_mask[i]:
                    continue
                while partial_scores[i] > sstar and m[i] < k:
                    if self._timed_out(time_limit, start_time):
                        break
                    m[i] += 1
                    partial_scores[i] = min(
                        partial_scores[i], 
                        self._conditional_mutual_information(cols[i], y, cols[selected_idx[m[i]]])
                    )
                if partial_scores[i] > sstar:
                    sstar = partial_scores[i]
        
        selected_features = [self._original_features[i] for i in selected_idx]
        return [str(feat) for feat in selected_features]

