from pandas import DataFrame, Series

from mafese import Data
from mafese import UnsupervisedSelector

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from autogluon.features.generators.abstract import AbstractFeatureSelector

class MAFESE(AbstractFeatureSelector):
    """ Select features from the data using Boruta algorithm """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mafese = None
        self._y = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

        data = Data(X, y)
        data.split_train_test(test_size=0.1, inplace=True)

        data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
        data.X_test = scaler_X.transform(data.X_test)
        data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
        data.y_test = scaler_y.transform(data.y_test)

        self._mafese_kwargs = {"problem": "classification", "method": "DR", "n_features": 5}
        self._mafese = UnsupervisedSelector(**self._mafese_kwargs)
        self._mafese.fit(data.X_train, data.y_train)
        selected_features_indexes = self._mafese.selected_feature_indexes
        X_out = X.iloc[:, selected_features_indexes]
        selected_features = X_out.columns

        self.feature_metadata_in.keep_features(selected_features, inplace=True)
        type_family_groups_special = {}

        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._mafese.fit_transform(X, self._y)
        else:
            selected_features_indexes = self._mafese.selected_feature_indexes
            X = X.iloc[:, selected_features_indexes]
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
