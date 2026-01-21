from __future__ import annotations

import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class Gini:
    """Gini feature selector"""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        feature_ranking = self.feature_ranking(self.gini_index(X_np, y_np))

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]


    def gini_index(self, X, y):
        """
        This function implements the gini index feature selection.

        Input
        ----------
        X: {numpy array}, shape (n_samples, n_features)
            input data
        y: {numpy array}, shape (n_samples,)
            input class labels

        Output
        ----------
        gini: {numpy array}, shape (n_features, )
            gini index value of each feature
        """

        n_samples, n_features = X.shape

        # initialize gini_index for all features to be 0.5
        gini = np.ones(n_features) * 0.5

        # For i-th feature we define fi = x[:,i] ,v include all unique values in fi
        for i in range(n_features):
            v = np.unique(X[:, i])
            for j in range(len(v)):
                # left_y contains labels of instances whose i-th feature value is less than or equal to v[j]
                left_y = y[X[:, i] <= v[j]]
                # right_y contains labels of instances whose i-th feature value is larger than v[j]
                right_y = y[X[:, i] > v[j]]

                # gini_left is sum of square of probability of occurrence of v[i] in left_y
                # gini_right is sum of square of probability of occurrence of v[i] in right_y
                gini_left = 0
                gini_right = 0

                for k in range(np.min(y), np.max(y) + 1):
                    if len(left_y) != 0:
                        # t1_left is probability of occurrence of k in left_y
                        t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                        t2_left = np.power(t1_left, 2)
                        gini_left += t2_left

                    if len(right_y) != 0:
                        # t1_right is probability of occurrence of k in left_y
                        t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                        t2_right = np.power(t1_right, 2)
                        gini_right += t2_right

                gini_left = 1 - gini_left
                gini_right = 1 - gini_right

                # weighted average of len(left_y) and len(right_y)
                t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)

                # compute the gini_index for the i-th feature
                value = np.true_divide(t1_gini, len(y))

                if value < gini[i]:
                    gini[i] = value
        return gini

    def feature_ranking(self, W):
        """
        Rank features in descending order according to their gini index values, the smaller the gini index,
        the more important the feature is
        """
        idx = np.argsort(W)
        return idx
