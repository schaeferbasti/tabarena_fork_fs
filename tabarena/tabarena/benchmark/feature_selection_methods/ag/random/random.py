"""Random feature selection."""
from __future__ import annotations

import logging

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)

class RandomFeatureSelector(AbstractFeatureSelector):
    """Random Feature Selection."""

    name = "RandomFeatureSelector"

    def _fit_feature_selection(self, **kwargs) -> list[dict[str, float | None]]:  # noqa: ARG002
        # Every pick comes from the random fallback -> all values are None.
        return self.fallback_feature_selection()