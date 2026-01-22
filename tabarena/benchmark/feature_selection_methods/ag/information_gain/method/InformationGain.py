from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

import warnings

from scipy.stats import entropy

import copy
import logging
import time

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class InformationGain:
    """InformationGain feature selector"""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model=None, n_max_features=None, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        igr_scores = []
        for col in X.columns:
            igr = self._information_gain_ratio(X[[col]], col, y)
            igr_scores.append(igr)

        igr_scores = np.array(igr_scores)
        top_features_idx = np.argsort(igr_scores)[-n_max_features:][::-1]
        X_selected = X.iloc[:, top_features_idx]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    @staticmethod
    def _entropy(values):
        """
        Calculate entropy from a frequency distribution.

        Parameters
        ----------
        values : array-like
            Value counts or frequencies

        Returns
        -------
        float
            Entropy value
        """
        # Normalize to probabilities and compute entropy
        probabilities = values / values.sum()
        return entropy(probabilities, base=2)

    def _information_gain(self, X: pd.DataFrame, feature: str, y: pd.Series) -> float:
        """
        Compute information gain for a single feature.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame
        feature : str
            Column name of the feature
        y : pd.Series
            Target variable

        Returns
        -------
        float
            Information gain value
        """
        # Calculate entropy of the target variable
        H_y = self._entropy(y.value_counts())

        # Calculate weighted conditional entropy
        weighted_entropy = 0.0

        for value in X[feature].unique():
            # Get mask for rows where feature equals this value
            mask = X[feature] == value

            # Get the subset of target values
            y_subset = y[mask]

            if len(y_subset) > 0:
                # Weight by proportion of this value
                weight = len(y_subset) / len(y)

                # Calculate entropy of this subset
                subset_entropy = self._entropy(y_subset.value_counts())

                # Add weighted entropy
                weighted_entropy += weight * subset_entropy

        # Information gain = H(Y) - H(Y|feature)
        return H_y - weighted_entropy

    def _intrinsic_value(self, X: pd.DataFrame, feature: str) -> float:
        """
        Compute intrinsic value (split information) for a feature.
        Used for information gain ratio normalization.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame
        feature : str
            Column name of the feature

        Returns
        -------
        float
            Intrinsic value
        """
        value_counts = X[feature].value_counts()
        proportions = value_counts / len(X)
        return entropy(proportions, base=2)

    def _information_gain_ratio(self, X: pd.DataFrame, feature: str, y: pd.Series) -> float:
        """
        Compute information gain ratio (normalized by intrinsic value).

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame
        feature : str
            Column name of the feature
        y : pd.Series
            Target variable

        Returns
        -------
        float
            Information gain ratio
        """
        ig = self._information_gain(X, feature, y)
        iv = self._intrinsic_value(X, feature)

        # Avoid division by zero
        if iv == 0:
            return 0.0
        return ig / iv
