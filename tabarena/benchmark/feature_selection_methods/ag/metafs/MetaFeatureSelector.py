import logging

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series

from autogluon.features.generators.abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class MetaFeatureSelector(AbstractFeatureGenerator):
    """MetaFS feature selection using metalearning approach."""

    def __init__(self, time_limit=1800, memory_limit=64000, model="LightGBM_BAG_L1", **kwargs):
        self._time_limit = time_limit
        self._memory_limit = memory_limit
        self._model = model

        super().__init__(**kwargs)

        self._metafs = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

        from tabarena.benchmark.feature_selection_methods.ag.metafs.method.MetaFS import MetaFS
        self._metafs_kwargs = {"time_limit": 1800, "memory_limit": 64000, "model": "LightGBM_BAG_L1"}
        self._metafs = MetaFS(**self._metafs_kwargs)

        X_out = self._metafs.fit_transform(X, y)
        self._selected_features = list(X_out.columns)
        self.feature_metadata_in = self.feature_metadata_in.keep_features(self._selected_features, inplace=False)
        type_family_groups_special = {}

        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X_out = self._metafs.fit_transform(X, self._y)
        else:
            X_out = X[self._selected_features].copy()
        return X_out


    @staticmethod
    def _get_dataset_metadata(X: DataFrame, y: Series) -> dict:
        """
        Extract dataset metadata needed by MetaFS.
        Adapt this based on what metadata your MetaFS code requires.
        """
        dataset_metadata = {"task_id": 2, "task_type": "Multiclass", "number_of_classes": 'N/A'}

        return dataset_metadata

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
