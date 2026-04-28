"""Laplacian score feature selection."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sklearn.neighbors import NearestNeighbors

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class LaplacianScoreFeatureSelector(AbstractFeatureSelector):
    """LaplacianScore Feature Selection.

    Reference: He, Xiaofei, Deng Cai, and Partha Niyogi. "Laplacian score for feature selection." 
    Advances in neural information processing systems 18 (2005).
    Implementation Source:
    https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/similarity_based/lap_score.py#L6
    The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation:
        - Time constraint
        - Efficiency refactoring
    """

    name = "LaplacianScoreFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        """This function implements the laplacian score feature selection, steps are as follows:
        1. Construct the affinity matrix W if it is not specified
        2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
        3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
        4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat).
        """
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True, scale=True)
        columns = X_pre.columns
        X_arr = X_pre.to_numpy()
       
        W = self._construct_W(X_arr)
        d = np.array(W.sum(axis=1)).ravel()
        d_total = d.sum()

        # f̃_r = f_r - (sum(f_r * d) / sum(d))
        weighted_means = (X_arr * d[:, None]).sum(axis=0) / d_total   # shape (n_features,)
        X_centered = X_arr - weighted_means # broadcast
       
        # numerator_r = f̃_r^T (D - W) f̃_r = f̃_r^T D f̃_r - f̃_r^T W f̃_r
        denom = ((X_centered ** 2) * d[:, None]).sum(axis=0) # f̃_r^T D f̃_r
        quad_W = ((W @ X_centered) * X_centered).sum(axis=0) # f̃_r^T W f̃_r (sparse-safe!)
        numer = denom - quad_W # f̃_r^T L f̃_r
    
        laplacian_scores = np.where(denom < 1e-12, np.inf, numer / np.where(denom < 1e-12, 1.0, denom)) # const features get -np.inf

        return dict(zip(columns, 1 - laplacian_scores)) # higher is better, can get negative
    
    @staticmethod
    def _construct_W(X, k=5, t=None):
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,
            metric="euclidean",
            n_jobs=-1,
        )
        nbrs.fit(X)
        W = nbrs.kneighbors_graph(X, mode="distance")

        W = W.maximum(W.T)

        W.setdiag(0) # remove self-loops
        W.eliminate_zeros()

        if t is None: # heat kernel as in the original paper
            t = np.median(W.data ** 2) if W.nnz > 0 else 1.0
        W.data = np.exp(-W.data**2 / t)

        return W.tocsr()