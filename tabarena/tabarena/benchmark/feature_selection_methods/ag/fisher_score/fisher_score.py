"""Fisher score feature selection."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import diags, lil_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class LaplacianScoreFeatureSelector(AbstractFeatureSelector):
    """LaplacianScore Feature Selection.

    Reference: He, Xiaofei, Deng Cai, and Partha Niyogi. "Laplacian score for feature selection." Advances in neural
    information processing systems 18 (2005).
    Implementation Source:
    https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/similarity_based/lap_score.py#L6
    The author of the code is Li, Jundong, Associate Professor at the
    University of Virginia and main-author of
    'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Remove overhead code for the construction of the weight matrix
                           - A sklearn preprocessing normalization is used instead of the code of the author, which
                           returned matrices filled with 0s for the datasets we used, which caused the laplacian score
                           to be 1 for all features.
    """

    name = "FisherScoreFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        """This function implements the laplacian score feature selection, steps are as follows:
        1. Construct the affinity matrix W in a fisher_score way (W_ij = 1/n_l if yi = yj = l).
        2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W.
        3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones).
        4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat).
        5. Fisher score = 1/laplacian_score - 1.
        """
        columns = X.columns
        X = X.to_numpy()
        data_encoder = OrdinalEncoder()
        X = data_encoder.fit_transform(X)
        numeric_imputer = SimpleImputer(strategy="mean")
        X = numeric_imputer.fit_transform(X)
        W = self.construct_W(X, y)
        D = np.array(W.sum(axis=1))
        L = W
        tmp = np.dot(np.transpose(D), X)
        D = diags(np.transpose(D), [0])
        Xt = np.transpose(X)
        t1 = np.transpose(np.dot(Xt, D.todense()))
        t2 = np.transpose(np.dot(Xt, L.todense()))
        D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp) / D.sum()
        L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp) / D.sum()
        D_prime[D_prime < 1e-12] = 10000
        lap_score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]
        # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
        score = 1.0 / lap_score - 1
        return dict(zip(columns, score))

    @staticmethod
    def construct_W(X, y):
        """Construct the affinity matrix W in a fisher_score way (W_ij = 1/n_l if yi = yj = l)."""
        n_samples, _n_features = np.shape(X)
        label = np.unique(y)
        n_classes = np.unique(y).size
        W = lil_matrix((n_samples, n_samples))
        for i in range(n_classes):
            class_idx = (y == label[i])
            class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
            W[class_idx_all] = 1.0 / np.sum(np.sum(class_idx))
        return W
