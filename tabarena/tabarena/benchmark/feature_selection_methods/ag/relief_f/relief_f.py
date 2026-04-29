"""ReliefF feature selection."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector


class ReliefFFeatureSelector(AbstractFeatureSelector):
    """ReliefF Feature Selection.

    References:
    1) Igor Kononenko, Edvard Simec, and Marko Robnik-Sikonja. "Overcoming the myopia of inductive learning algorithms with RELIEFF"
       Applied Intelligence 7.1 (1997): 39-55.
    2) Marko Robnik-Sikonja and Igor Kononenko. "An adaptation of Relief for attribute estimation in regression"
       International Conference on Machine Learning (1997): 296-304.

    Changes to the original paper:
        -  Nominal attributes are ordinal-encoded and treated as numeric rather than using the paper's 0/1 nominal diff.
        Scores for nominal features therefore depend on the integer assignment of the encoder. This avoids the
        dimensionality blow-up of one-hot encoding.
    """

    name = "ReliefFFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,
    ) -> dict[str, float]:
        start_time = time.monotonic()
        n_neighbors = 5  # local default; thread through to internal methods

        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)
        X_arr = self._minmax_scale(X_pre.to_numpy())
        y_arr = np.asarray(y)
        columns = list(X_pre.columns)

        if self.problem_type == "regression":
            scores = self._rrelieff(X_arr, y_arr, n_neighbors, time_limit, start_time)
        else:
            scores = self._relieff(X_arr, y_arr, n_neighbors, time_limit, start_time)

        return dict(zip(columns, scores))

    @staticmethod
    def _minmax_scale(X: np.ndarray) -> np.ndarray:
        """Min-max scale to [0,1]; columns with zero range left at 0."""
        X_min = X.min(axis=0)
        X_range = X.max(axis=0) - X_min
        X_range[X_range == 0] = 1.0
        return (X - X_min) / X_range

    def _relieff(
        self, X: np.ndarray, y: np.ndarray, n_neighbors: int,
        time_limit: int | None, start_time: float,
    ) -> np.ndarray:
        """ReliefF (for classification)."""
        n, d = X.shape
        k = min(n_neighbors, n - 1)
        scores = np.zeros(d)

        classes, counts = np.unique(y, return_counts=True)
        priors = dict(zip(classes, counts / n))

        # kNN per class: O(nk) instead of O(n²) memory for the distance matrix.
        nn_per_class = {}
        for c in classes:
            mask = y == c
            if mask.sum() <= 1:
                continue  # singleton or empty class — can't compute hits
            nn_per_class[c] = (
                NearestNeighbors(n_neighbors=min(k + 1, mask.sum()), metric="manhattan")
                .fit(X[mask]),
                np.where(mask)[0],
            )

        for i in range(n):
            if self._timed_out(time_limit, start_time):
                break

            xi = X[i:i+1]
            yi = y[i]

            # Skip samples whose class wasn't built (singleton classes).
            if yi not in nn_per_class:
                continue
            p_yi = priors[yi]

            # Hits: k nearest same-class neighbors (excluding self at index 0)
            hit_nbrs, hit_idx_map = nn_per_class[yi]
            _, hit_local = hit_nbrs.kneighbors(xi)
            hit_global = hit_idx_map[hit_local[0][1:k+1]]

            # Small-class guard: if a class has < k+1 samples, we get fewer than
            # k hits back; divide by what we actually got, not by k.
            n_hits = len(hit_global)
            if n_hits == 0:
                continue
            hit_diff = np.abs(X[hit_global] - xi).sum(axis=0) / n_hits

            # Misses: k nearest neighbors per other class, weighted by prior.
            miss_diff = np.zeros(d)
            for c in classes:
                if c == yi or c not in nn_per_class:
                    continue
                miss_nbrs, miss_idx_map = nn_per_class[c]
                _, miss_local = miss_nbrs.kneighbors(xi)
                miss_global = miss_idx_map[miss_local[0][:k]]

                # Same small-class guard for miss division.
                n_misses = len(miss_global)
                if n_misses == 0:
                    continue
                weight = priors[c] / (1 - p_yi)
                miss_diff += weight * np.abs(X[miss_global] - xi).sum(axis=0) / n_misses

            scores += miss_diff - hit_diff

        return scores / n

    def _rrelieff(
        self, X: np.ndarray, y: np.ndarray, n_neighbors: int,
        time_limit: int | None, start_time: float,
    ) -> np.ndarray:
        """RReliefF for regression.

        Tracks three running sums over k-nearest-neighbor pairs:
          N_dY        — Σ d(y)
          N_dF[f]     — Σ d(x_f)
          N_dY_dF[f]  — Σ d(y) * d(x_f)
        and combines via Bayes' rule:
          W[f] = N_dY_dF[f] / N_dY  -  (N_dF[f] - N_dY_dF[f]) / (N_total - N_dY)
        where d(.) is normalized to [0,1].
        """
        n, d = X.shape
        k = min(n_neighbors, n - 1)

        # Normalize y to [0,1] for d(y); guard against constant y.
        y_min, y_max = float(y.min()), float(y.max())
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_norm = (y - y_min) / y_range

        # kNN over the full dataset (no class structure for regression).
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="manhattan").fit(X)

        N_dY = 0.0
        N_dF = np.zeros(d)
        N_dY_dF = np.zeros(d)
        N_total = 0.0  # actual number of (i, neighbor) pairs accumulated

        for i in range(n):
            if self._timed_out(time_limit, start_time):
                break

            _, idx = nbrs.kneighbors(X[i:i+1])
            neighbors = idx[0][1:k+1]  # skip self at index 0

            # d(y) for each neighbor: |y_i - y_j| in [0,1]
            dy = np.abs(y_norm[i] - y_norm[neighbors])      # shape (k,)
            # d(x_f) for each neighbor and feature: |x_if - x_jf|
            # X is min-max scaled to [0,1], so dx values are also in [0, 1].
            dx = np.abs(X[i] - X[neighbors])                # shape (k, d)

            N_dY += dy.sum()
            N_dF += dx.sum(axis=0)
            N_dY_dF += (dy[:, None] * dx).sum(axis=0)
            # Use len(neighbors) instead of k, in case a row had fewer.
            N_total += len(neighbors)

        # Bayes' rule combination
        denom_hit = N_dY if N_dY > 0 else 1.0
        denom_miss = (N_total - N_dY) if (N_total - N_dY) > 0 else 1.0
        return N_dY_dF / denom_hit - (N_dF - N_dY_dF) / denom_miss