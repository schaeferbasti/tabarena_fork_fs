import logging

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series

from autogluon.features.generators.abstract import AbstractFeatureSelector

from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest

logger = logging.getLogger(__name__)


class Select_k_Best_F(AbstractFeatureSelector):
    """ Select k best features from the data using Chi^2 score """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._select_best = None
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        self._select_best_kwargs = {"score_func": f_regression, "k": n_max_features}
        self._select_best = SelectKBest(**self._select_best_kwargs).set_output(transform="pandas")
        X_out = self._transform(X, is_train=True)
        self._selected_features = list(X_out.columns)
        type_family_groups_special = {}
        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._select_best.fit_transform(X, self._y)
        else:
            X = self._select_best.transform(X)
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
