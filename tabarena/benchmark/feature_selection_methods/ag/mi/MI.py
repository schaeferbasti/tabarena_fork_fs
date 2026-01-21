import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from autogluon.features.generators.abstract import AbstractFeatureSelector

from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
import logging

from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


class MI(AbstractFeatureSelector):
    """ Select features from the data using MI algorithm """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mi = None
        self._y = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        type_family_groups_special = {}

        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                if n_max_features is not None and len(X.columns) > n_max_features:
                    X_out = X.sample(n=n_max_features, axis=1)
                    return X_out, type_family_groups_special
                else:
                    return X, type_family_groups_special

        features = []
        if n_max_features is None:
            return X, type_family_groups_special
        else:
            for i in range(n_max_features):
                best_MI = -np.inf
                best_feature = None
                for feature in X.columns:
                    try:
                        MI = mutual_info_classif(pd.DataFrame(X[feature], columns=[feature]), y)
                    except ValueError:
                        X_fixed = X.fillna(0)
                        MI = mutual_info_classif(pd.DataFrame(X_fixed[feature], columns=[feature]), y)
                    if MI > best_MI:
                        best_MI = MI
                        best_feature = feature
                    features.append(best_feature)
            if len(features) == n_max_features:
                self.selected_features = features
                return X[features], type_family_groups_special
            X_out = self._transform(X, is_train=True)
            self._selected_features = list(X_out.columns)
            return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self.fit_transform(X, self._y)
        else:
            X = self.transform(X)
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
