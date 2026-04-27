"""Information gain feature selection."""
from __future__ import annotations

import logging
import time
from math import log
from typing import TYPE_CHECKING

import numpy as np

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractITFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

class InformationGainFeatureSelector(AbstractITFeatureSelector):
    """InformationGain Feature Selection.

    Reference: Quinlan, J. Induction of Decision Trees. Mach Learn 1,
    81-106 (1986). https://doi.org/10.1023/A:1022643204877
    Changes to the implementation:
        - Time constraint
    """

    name = "InformationGainFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        ig_scores = {}

        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        
        for col in X_pre.columns:
            if self._timed_out(time_limit, start_time):
                break
            feature = X_pre[col]
            ig_scores[col] = self._mutual_information(feature, y)

        return ig_scores
