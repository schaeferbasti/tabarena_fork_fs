"""Sequential forward selection feature selection."""
from __future__ import annotations

import logging
import time
import numpy as np

from typing import TYPE_CHECKING

from autogluon.core.utils.exceptions import TimeLimitExceeded
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SequentialForwardSelectionFeatureSelector(AbstractFeatureSelector):
    """SequentialForwardSelection Feature Selection."""

    name = "SequentialForwardSelectionFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> list[str]:
        start_time = time.monotonic()
        current_features = []

        if self.max_features >= len(self._original_features): # trivial case
            return [str(f) for f in self._original_features]
        
        available_features = self._original_features.copy()

        try: # safeguard for AG model throwing a TimeExceededlimit error that would reset the already selected feature set 
            while len(current_features) < self.max_features and available_features:
                if self._timed_out(time_limit, start_time):
                    break

                best_score = -np.inf
                best_feature = None

                for feature in available_features:
                    if self._timed_out(time_limit, start_time):
                        break

                    time_to_fit = None
                    if time_limit is not None:
                        remaining = time_limit - (time.monotonic() - start_time)
                        time_to_fit = max(0.0, remaining * 0.9)

                    test_X = X[[*current_features, feature]]
                    score = self.evaluate_proxy_model(X=test_X, y=y, time_limit=time_to_fit)
                    del test_X

                    if score is None:
                        continue

                    if score > best_score:
                        best_score = score
                        best_feature = feature

                if best_feature is None:
                    break

                current_features.append(best_feature)
                available_features.remove(best_feature)

        except TimeLimitExceeded:
            self._log(30, f"TimeLimitExceeded during selection. Returning {len(current_features)} selected features.")

        return [str(feat) for feat in current_features]
