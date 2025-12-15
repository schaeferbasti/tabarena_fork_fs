import logging

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series

from autogluon.features.generators.abstract import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class LocalSearchFeatureSelector_Flip(AbstractFeatureSelector):
    """ Local Search, only allowing for flipping features """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ls_flip = None
        self._y = None
        self._selected_features = None


    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

        self._ls_flip_kwargs = {}
        from tabarena.benchmark.feature_selection_methods.ag.ls_flip.method.LS_Flip import LS_Flip
        self._ls_flip = LS_Flip(**self._ls_flip_kwargs)
        X_out = self._ls_flip.fit_transform(X, y)

        selected_features = list(X_out.columns)
        self.feature_metadata_in.keep_features(selected_features, inplace=True)
        type_family_groups_special = {}

        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._ls_flip.fit_transform(X, self._y)
        else:
            X = self._ls_flip.transform(X)
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
