"""Registry of feature selection methods."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import (
        AbstractFeatureSelector,
    )

NAME_TO_MODULE_MAP = {
    "LOCOFeatureSelector": "loco.loco",
    "RandomFeatureSelector": "random.random",
    "FTestFeatureSelector": "f_test.f_test",
    "CARTFeatureSelector": "cart.cart",
    "ElasticNetFeatureSelector": "elastic_net.elastic_net",
    "GainRatioFeatureSelector": "gain_ratio.gain_ratio",
    "LaplacianScoreFeatureSelector": "laplacian_score.laplacian_score",
    "LassoFeatureSelector": "lasso.lasso",
    "MarkovBlanketFeatureSelector": "markov_blanket.markov_blanket",
    "MIFeatureSelector": "mi.mi",
    "mRMRFeatureSelector": "mrmr.mrmr",
    "ReliefFFeatureSelector": "relief_f.relief_f",
    "RFImportanceFeatureSelector": "rf_importance.rf_importance",
    "SequentialBackwardEliminationFeatureSelector": "sbe.sbe",
    "SequentialForwardSelectionFeatureSelector": "sfs.sfs",
}
FEATURE_SELECTION_METHODS = list(NAME_TO_MODULE_MAP.keys())
FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL = [
    "LOCOFeatureSelector",
    "SequentialBackwardEliminationFeatureSelector",
    "SequentialForwardSelectionFeatureSelector",
]


def get_feature_selector_from_name(
    *,
    name: str,
) -> type[AbstractFeatureSelector]:
    """Get the feature selector class from the method name.

    Parameters
    ----------
    name : str
        The name of the feature selection method.

    Returns:
    -------
    AbstractFeatureSelector
        An instance of the feature selector class corresponding to the method name.
    """
    import importlib  # noqa: PLC0415

    if name not in FEATURE_SELECTION_METHODS:
        raise ValueError(f"Method name '{name}' is not recognized. Options are: {FEATURE_SELECTION_METHODS}")

    base_path = "tabarena.benchmark.feature_selection_methods.ag."
    method_path = NAME_TO_MODULE_MAP[name]

    return getattr(importlib.import_module(base_path + method_path), name)
