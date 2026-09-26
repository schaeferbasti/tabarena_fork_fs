"""Joint Mutual Information (JMI) feature selection."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractITFeatureSelector


class JMIFeatureSelector(AbstractITFeatureSelector):
    """JMI Feature Selection.

    Reference: Yang, Howard, and John Moody. "Data visualization and feature selection: New algorithms for nongaussian data." 
    Advances in neural information processing systems 12 (1999).
    Implementation Inspiration: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/JMI.py#L4
    The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation:
        - Time and max_features constraint
        - Use pandas instead of numpy and avoid conversion
        - Follow the paper's algorithm directly
    """

    name = "JMIFeatureSelector"
    feature_scoring_method: bool = False

       
    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> list[str]:
        """
        Step 1: select first feature by plain MI: i1 = argmax_i I(Xi; Y)
        Step 2: select subsequent features by maximizing the sum of pairwise joint MI given selected features:
                i_k = argmax_i  sum_{j in selected} I(Xi, Xj; Y)
        """
        start_time = time.monotonic()

        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        n_features = len(X_pre.columns)
        cols = [X_pre.iloc[:, i] for i in range(n_features)]

        # step 1
        mi_scores = np.array([self._mutual_information(col, y) for col in cols])
        first = int(np.argmax(mi_scores))
        selected: list[int] = [first]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[first] = True

        # step 2
        while len(selected) < self.max_features:
            if self._timed_out(time_limit, start_time): 
                break

            best_score = -np.inf
            best_idx = None
            
            for i in range(n_features):
                if selected_mask[i]:
                        continue
                if self._timed_out(time_limit, start_time): 
                    break

                score = sum(
                    self._joint_mutual_information(cols[i], cols[j], y)
                    for j in selected
                )
                if score > best_score:
                    best_score = score
                    best_idx = i
                
            if best_idx is None:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True

        return [str(self._original_features[i]) for i in selected]