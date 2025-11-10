from __future__ import annotations

import pandas as pd

from autogluon.tabular.models import CatBoostModel
from tabarena.benchmark.feature_selection_methods.ag.metafs.method.Add_Pandas_Metafeatures import add_pandas_metadata_selection_columns

import warnings
warnings.filterwarnings('ignore')


class MetaFS:
    """MetaFS feature selector with resource limits."""

    def __init__(self, time_limit: int = 300, memory_limit: int = 16000, model: str = 'LightGBM_BAG_L1'):
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.model = model
        self.dataset_id = 146820
        self._selected_features = None


    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self._y = y
        dataset_metadata = self._extract_metadata(y)
        result = self._run_with_limits(self._fit_transform_internal, X, y, self.model, dataset_metadata, self.dataset_id)

        if result is None:
            self._selected_features = list(X.columns)
            return X

        X_selected, y_selected = result
        self._selected_features = list(X_selected.columns)
        return X_selected


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y)
        return X[self._selected_features]


    def _fit_transform_internal(self, X_train, y_train, model, dataset_metadata, dataset_id):
        result_matrix = pd.read_parquet("../../tabarena/benchmark/feature_selection_methods/ag/metafs/method/Pandas_Matrix_Complete.parquet")
        datasets = pd.unique(result_matrix["dataset - id"]).tolist()
        if dataset_id in datasets:
            result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]

        comparison_result_matrix = self.create_empty_core_matrix_for_dataset(X_train, model, dataset_id)
        comparison_result_matrix = add_pandas_metadata_selection_columns(dataset_metadata, X_train, comparison_result_matrix)

        X_train_new, y_train_new = self.predict_improvement(result_matrix, comparison_result_matrix, X_train, y_train)

        if X_train_new.equals(X_train):
            return X_train_new, y_train_new
        else:
            return self._fit_transform_internal(X_train_new, y_train_new, model, dataset_metadata, dataset_id)


    def predict_improvement(self, result_matrix, comparison_result_matrix, X_train, y_train):
        y_result = result_matrix["improvement"]
        result_matrix = result_matrix.drop("improvement", axis=1)
        comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)

        clf = CatBoostModel()
        clf.fit(X=result_matrix, y=y_result)

        comparison_result_matrix.columns = comparison_result_matrix.columns.astype(str)
        comparison_result_matrix = comparison_result_matrix[result_matrix.columns]
        prediction = clf.predict(X=comparison_result_matrix)
        prediction_df = pd.DataFrame(prediction, columns=["predicted_improvement"])
        prediction_concat_df = pd.concat([comparison_result_matrix[["dataset - id", "feature - name", "model"]], prediction_df], axis=1)
        best_operation = prediction_concat_df.nlargest(n=1, columns="predicted_improvement", keep="first")
        if best_operation["predicted_improvement"].values[0] < 0:
            return X_train, y_train
        else:
            X_train, y_train = self.remove_feature(X_train, y_train, best_operation)
        return X_train, y_train


    @staticmethod
    def remove_feature(X_train, y_train, prediction_result):
        worst_feature_row = prediction_result.loc[prediction_result['predicted_improvement'].idxmin()]
        worst_feature_name = worst_feature_row['feature - name'].split(' - ')[-1]
        print(f"Removing worst feature: {worst_feature_name} (predicted_improvement={worst_feature_row['predicted_improvement']:.4f})")

        if worst_feature_name in X_train.columns:
            X_train = X_train.drop(columns=[worst_feature_name])
        else:
            print(f"Warning: Feature '{worst_feature_name}' not found in X.")
        return X_train, y_train


    @staticmethod
    def create_empty_core_matrix_for_dataset(X_train, model, dataset_id) -> pd.DataFrame:
        columns = ['dataset - id', 'feature - name', 'operator', 'model', 'improvement']
        comparison_result_matrix = pd.DataFrame(columns=columns)
        for feature1 in X_train.columns:
            featurename = "without - " + str(feature1)
            new_rows = pd.DataFrame(columns=columns)
            operator = "delete"
            new_rows.loc[len(new_rows)] = [dataset_id, featurename, operator, model, 0]
            comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
        return comparison_result_matrix


    @staticmethod
    def _extract_metadata(y: pd.Series) -> dict:
        n_classes = y.nunique()
        if n_classes < 200:
            task_type = "Supervised Classification"
        else:
            task_type = "Supervised Regression"
        return {
            "task_id": None,
            "task_type": task_type,
            "number_of_classes": 'N/A'
        }


    def _run_with_limits(self, target_func, *args):
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"MetaFS exceeded time limit of {self.time_limit} seconds")

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.time_limit))
            result = target_func(*args)
            signal.alarm(0)
            return result
        except TimeoutError as e:
            print(f"[MetaFS] {e}")
            return None
