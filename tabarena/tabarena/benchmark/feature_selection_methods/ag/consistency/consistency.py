"""Consistency-based feature selection."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector


class ConsistencyFeatureSelector(AbstractFeatureSelector):
    """(In-)Consistency Feature Selection.

    Reference: Liu, H., & Setiono, R. (1996, July). A probabilistic approach to feature selection-a filter solution.
    In ICML (Vol. 96, pp. 319-327).
    Implementation Source: Algorithm in the paper implemented by Bastian Schäfer
     Changes to the implementation:
        - Time constraint
        - Downsample to max_features when consistent feature subset is larger
    """

    name = "ConsistencyFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> list[str]:
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)

        n_features = len(X_pre.columns)
        n_trials = max(100, 10 * n_features) # random trials, will most likely timeout for high-dim datasets
        inconsistency_tolerance = 0.05 
        rng = np.random.default_rng(self._outer_random_state)
        
        best_size = n_features
        best_mask = np.ones(n_features, dtype=bool) 

        for _ in range(n_trials):
            if self._timed_out(time_limit, start_time):
                break

            candidate_mask = rng.random(n_features) < 0.5
            if not candidate_mask.any(): # fallback if no features selected
                candidate_mask[rng.integers(0, n_features)] = True

            candidate_size = int(candidate_mask.sum())
            if candidate_size >= best_size:
                continue
           
            inconsistency_rate = self._inconsistency_rate(X_pre.loc[:, X_pre.columns[candidate_mask]], y, time_limit, start_time)
            if inconsistency_rate is None:
                break

            if inconsistency_rate <= inconsistency_tolerance:
                best_mask = candidate_mask.copy()
                best_size = candidate_size

        selected_indices = np.where(best_mask)[0].tolist()
        selected_features = [str(self._original_features[i]) for i in selected_indices]

        # downsample to max_features if consistent subset is bigger
        if len(selected_features) > self.max_features:
            rng_final = np.random.default_rng(self._outer_random_state)
            selected_features = list(rng_final.choice(selected_features, size=self.max_features, replace=False))
        
        return selected_features

    def _inconsistency_rate(self, X_sub: pd.DataFrame, y: pd.Series, time_limit, start_time) -> float | None:
        """
        IR(S) = (sum over patterns) (|pattern_group| - max_class_count) / n.
        
        Returns None if timeout reached mid-computation.
        """
        if X_sub.shape[1] == 0:
            return 1.0  # empty set cannot discriminate at all

        row_keys = pd.util.hash_pandas_object(X_sub, index=False)

        total_disagreements = 0
        for _, indices in pd.Series(range(len(X_sub))).groupby(row_keys.values, observed=False):
            if self._timed_out(time_limit, start_time):
                return None
            group_y = y.iloc[indices]
            majority_count = int(group_y.value_counts(dropna=False).max())
            total_disagreements += len(indices) - majority_count

        return total_disagreements / len(X_sub)
