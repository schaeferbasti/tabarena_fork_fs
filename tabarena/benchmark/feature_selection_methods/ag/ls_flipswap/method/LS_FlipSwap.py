from __future__ import annotations

import numpy as np
import pandas as pd

import warnings

from sklearn.model_selection import train_test_split

from tabarena.benchmark.feature_selection_methods.ag.metafs.method.utils.run_models import \
    get_sklearn_model_score_classification, get_sklearn_model_score_regression

warnings.filterwarnings('ignore')


class LS_FlipSwap:
    """LS feature selector (flip and swap) with resource limits."""

    def __init__(self, time_limit: int = 300, memory_limit: int = 16000, model: str = 'LightGBM_BAG_L1'):
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.model = model
        self.dataset_id = 146820
        self.task_type = "Supervised Classification"
        self.score = "log_loss"
        self.repeat = None
        self._selected_features = None
        self._y = None
        self.LOWER_BETTER = ["log_loss", "root_mean_squared_error", "max_error"]
        self.HIGHER_BETTER = ["roc_auc_score"]


    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self._y = y
        result = self._run_with_limits(self._fit_transform_internal, X, y, self.dataset_id)

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


    def _fit_transform_internal(self, X_train, y_train, dataset_id):
        feature_names = X_train.columns.tolist()
        init_feature_choice = [0] * len(X_train.columns)
        feature_indices = self.get_ls_indices(X_train, y_train, dataset_id, self.task_type, self.model, self.score, self.repeat, init_feature_choice)
        selected_feature_names = [feature_names[i] for i, flag in enumerate(feature_indices) if flag == 1]
        X_train_new = pd.DataFrame(X_train, columns=selected_feature_names, index=X_train.index)
        return X_train_new, y_train


    def get_ls_indices(self, X_train, y_train, dataset_id, task_type, model_name, score_name, repeat,
                       feature_indices):
        termination_condition = False
        direction = self.get_metric_direction(score_name)
        if direction == "higher":
            best_score = -np.inf
        else:
            best_score = np.inf
        best_indices = feature_indices
        while not termination_condition:
            list_of_feature_indices = self.get_flip_indices(feature_indices)
            print(list_of_feature_indices)
            improvement_found = False
            for feature_indices in list_of_feature_indices:
                feature_mask = [bool(i) for i in feature_indices]
                X_train_selection = X_train.iloc[:, feature_mask]
                score = self.evaluate_subset(X_train_selection, y_train, dataset_id, task_type, model_name, score_name, repeat)
                if direction == "higher":
                    if score > best_score:
                        best_score = score
                        best_indices = feature_indices
                        improvement_found = True
                else:  # lower
                    if score < best_score:
                        best_score = score
                        best_indices = feature_indices
                        improvement_found = True
            feature_indices = best_indices
            all_ones = all(i == 1 for i in feature_indices)
            if all_ones or not improvement_found:
                termination_condition = True
        return best_indices


    def get_metric_direction(self, score_name):
        if score_name in self.HIGHER_BETTER:
            return "higher"
        else:
            return "lower"


    @staticmethod
    def get_flip_indices(feature_indices):
        list_of_feature_indices = []
        print("Indices: " + str(feature_indices))
        # Flipping
        for idx, val in enumerate(feature_indices):
            new_feature_indices = feature_indices.copy()
            new_feature_indices[idx] = abs(1 - val)  # flip 0 to 1 or 1 to 0
            # Only add if at least one feature is selected (not all zeros)
            if sum(new_feature_indices) > 0:
                list_of_feature_indices.append(new_feature_indices)

        # Swapping
        for idx1 in range(len(feature_indices)):
            for idx2 in range(idx1 + 1, len(feature_indices)):
                new_feature_indices = feature_indices.copy()
                # Swap values at idx1 and idx2
                new_feature_indices[idx1], new_feature_indices[idx2] = new_feature_indices[idx2], new_feature_indices[
                    idx1]
                # Only add if at least one feature is selected (not all zeros)
                if sum(new_feature_indices) > 0:
                    list_of_feature_indices.append(new_feature_indices)
        list_of_feature_indices = [i for i in list_of_feature_indices if i != feature_indices]
        return list_of_feature_indices


    @staticmethod
    def evaluate_subset(X, y, dataset_id, task_type, model_name, score_name, repeat):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        if task_type == "Supervised Classification":
            results = get_sklearn_model_score_classification(X_train, y_train, X_val, y_val, dataset_id, "EnumerateFS", model_name, repeat, score_name)
        else:
            results = get_sklearn_model_score_regression(X_train, y_train, X_val, y_val, dataset_id, "EnumerateFS", model_name, repeat, score_name)
        return results["score_test"].iloc[0]


    def _run_with_limits(self, target_func, *args):
        import signal

        def timeout_handler():
            raise TimeoutError(f"Method exceeded time limit of {self.time_limit} seconds")

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.time_limit))
            result = target_func(*args)
            signal.alarm(0)
            return result
        except TimeoutError as e:
            print(f"[MetaFS] {e}")
            return None
