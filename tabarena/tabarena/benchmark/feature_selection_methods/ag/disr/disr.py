"""Double Input Symmetrical Relevance (DISR) feature selection."""
from __future__ import annotations

import time
import pandas as pd
import numpy as np

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractITFeatureSelector


class DISRFeatureSelector(AbstractITFeatureSelector):
    """DISR Feature Selection.

    Reference: Meyer, Patrick E., and Gianluca Bontempi. "On the use of variable complementarity for feature selection in cancer classification." 
    Workshops on applications of evolutionary computation. Berlin, Heidelberg: Springer Berlin Heidelberg, 2006.
    Implementation Inspiration: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/JMI.py#L4
    The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    
    We extend the Joint Mutual Information (jmi.py) implementation by normalizing each joint MI term by the joint entropy H(X_i, X_j, Y):
    criterion = argmax_i sum_{j in F} I((X_i, X_j); Y) / H(X_i, X_j, Y)
    """

    name = "DISRFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> list[str]:
        """
        Step 1: select first feature by plain MI — i1 = argmax_i I(X_i; Y)
        Step 2: greedy forward, select at each step:
                i_k = argmax_i sum_{j in selected} I((X_i, X_j); Y) / H(X_i, X_j, Y)
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
                # joint mi/entropy
                score = sum(
                    self._symmetrical_relevance(cols[i], cols[j], y)
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