from pandas import DataFrame, Series
import time

from sklearn.ensemble import RandomForestClassifier

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from autogluon.features.generators.abstract import AbstractFeatureSelector

from tabarena.benchmark.feature_selection_methods.ag.boruta.method import BorutaPy
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded


class Boruta(AbstractFeatureSelector):
    """ Select features from the data using Boruta algorithm """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._boruta = None
        self._y = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        self._boruta_kwargs = {"estimator": rf, "n_estimators": "auto", "verbose": 2}
        self._boruta = BorutaPy(**self._boruta_kwargs)

        # Time limit
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            if kwargs["time_limit"] is not None:
                time_cur = time.time()
                time_left = kwargs["time_limit"] - (time_cur - kwargs["start_time"])
                if time_left <= kwargs["time_limit"] * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                    raise TimeLimitExceeded

        # Convert to numpy for BorutaPy
        X_np = X.values if isinstance(X, DataFrame) else X
        self._boruta.fit(X_np, y.ravel(), **kwargs)

        # Get selected feature indices and names
        selected_mask = self._boruta.support_
        self._selected_features = X.columns[selected_mask].tolist()

        # Transform data
        X_out = X[self._selected_features].copy()

        # Update metadata
        self.feature_metadata_in = self.feature_metadata_in.keep_features(self._selected_features, inplace=False)
        type_family_groups_special = {}

        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if self._selected_features is None:
            return X
        X_out = X[self._selected_features].copy()
        return X_out


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
