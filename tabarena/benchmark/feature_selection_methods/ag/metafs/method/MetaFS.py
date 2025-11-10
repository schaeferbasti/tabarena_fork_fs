from __future__ import annotations

import multiprocessing
import time
import pandas as pd
import psutil
from multiprocessing import Value
import ctypes
import warnings

from autogluon.tabular.models import CatBoostModel

from tabarena.benchmark.feature_selection_methods.ag.metafs.method.Add_Pandas_Metafeatures import add_pandas_metadata_selection_columns

warnings.filterwarnings('ignore')


class MetaFS:
    """MetaFS feature selector with resource limits."""

    def __init__(self, time_limit: int = 500, memory_limit: int = 64000, model: str = 'LightGBM_BAG_L1'):
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.model = model
        self.dataset_id = 146820
        self.merge_keys = ["dataset - id", "feature - name", "operator", "model", "improvement"]
        self.last_reset_time = Value(ctypes.c_double, time.time())
        self._selected_features = None


    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Fit and transform with only X and y, inferring dataset metadata internally.
        Enforces time and memory limits.
        """
        # Extract dataset metadata from X and y
        dataset_metadata = self._extract_metadata(X_train, y_train)

        # Run feature selection with resource limits
        result = self._run_with_limits(self._fit_transform_internal, X_train, y_train, self.model, dataset_metadata, None, self.dataset_id)

        if result is None:
            # Timeout or resource limit exceeded - return original data
            print(f"[MetaFS] Resource limit exceeded. Returning original features.")
            self._selected_features = list(X_train.columns)
            return X_train

        X_selected, y_selected = result
        self._selected_features = list(X_selected.columns)
        return X_selected


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using previously selected features."""
        if self._selected_features is None:
            raise ValueError("MetaFS must be fitted before transform.")
        return X[self._selected_features]


    def _fit_transform_internal(self, X_train, y_train, model, dataset_metadata, category_to_drop, dataset_id):
        """Internal recursive feature selection logic."""
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
            return self._fit_transform_internal(X_train_new, y_train_new, model, dataset_metadata, category_to_drop, dataset_id)


    def predict_improvement(self, result_matrix, comparison_result_matrix, X_train, y_train):
        """Predict which feature to remove next."""
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
            print(f"Predicted improvement: {best_operation['predicted_improvement'].values[0]:.4f} - not good enough")
            return X_train, y_train
        else:
            print(f"Predicted improvement: {best_operation['predicted_improvement'].values[0]:.4f} - removing feature")
            X_train, y_train = self.remove_feature(X_train, y_train, best_operation)

        return X_train, y_train



    @staticmethod
    def remove_feature(X_train, y_train, prediction_result):
        """Remove the worst feature from X_train."""
        worst_feature_row = prediction_result.loc[prediction_result['predicted_improvement'].idxmin()]
        worst_feature_name = worst_feature_row['feature - name'].split(' - ')[-1]
        print(
            f"Removing worst feature: {worst_feature_name} (predicted_improvement={worst_feature_row['predicted_improvement']:.4f})")

        if worst_feature_name in X_train.columns:
            X_train = X_train.drop(columns=[worst_feature_name])
        else:
            print(f"Warning: Feature '{worst_feature_name}' not found in X.")
        return X_train, y_train

    @staticmethod
    def create_empty_core_matrix_for_dataset(X_train, model, dataset_id) -> pd.DataFrame:
        """Create comparison matrix for feature selection."""
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
    def _extract_metadata(X: pd.DataFrame, y: pd.Series) -> dict:
        """Extract dataset metadata from X and y."""
        n_classes = y.nunique()
        # Task type classification logic
        if n_classes < 200:  # You can adjust this threshold
            task_type = "Supervised Classification"
        else:
            task_type = "Supervised Regression"
        return {
            "task_id": None,
            "task_type": task_type,
            "number_of_classes": 'N/A'
        }

    def _run_with_limits(self, target_func, *args):
        """Run target_func with time and memory limits - SYNCHRONOUSLY (no multiprocessing)."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"MetaFS exceeded time limit of {self.time_limit} seconds")

        # Set alarm signal for time limit (Unix only, not on Windows)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.time_limit))

            # Run the function directly (no subprocess)
            result = target_func(*args)

            # Cancel the alarm
            signal.alarm(0)
            return result
        except TimeoutError as e:
            print(f"[MetaFS] {e}")
            return None


def _metafs_worker(queue, func, *args):
    """Worker function for multiprocessing - MUST be at module level to be picklable."""
    try:
        result = func(*args)
        queue.put(result)
    except Exception as e:
        print(f"[MetaFS] Error in subprocess: {e}")
        queue.put(None)
