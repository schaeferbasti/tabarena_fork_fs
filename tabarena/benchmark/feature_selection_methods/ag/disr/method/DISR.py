from __future__ import annotations

import pandas as pd

import warnings

from sklearn.model_selection import train_test_split

import copy
import logging
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class DISR:
    """ DISR feature selector """

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None


    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        init_feature_choice = [0] * len(X.columns)
        X_selected = self.disr(X, y, model, n_max_features, init_feature_choice, **kwargs)
        X_selected = pd.DataFrame(X, columns=X_selected.columns, index=X.index)
        self._selected_features = list(X_selected.columns)
        return X_selected


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]


    def disr(self, X_train, y_train, model, n_max_features, feature_indices, **kwargs):
        return X_train


    @staticmethod
    def evaluate_subset(self, X, y, model):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model_copy = copy.deepcopy(model)
        model_copy.params["fold_fitting_strategy"] = "sequential_local"
        model_copy = model_copy.fit(X=X_train, y=y_train, k_fold=8)
        self._model = model_copy
        return model_copy.score_with_oof(y=y_train)
