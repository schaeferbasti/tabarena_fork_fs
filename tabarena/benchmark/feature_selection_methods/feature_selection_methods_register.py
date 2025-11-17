from __future__ import annotations

from copy import deepcopy


def get_default_encoding_pipeline():
    """Return the default pipeline for encoding non-numerical or categorical data.

    The default pipeline handles:
        - Text Features
        - Date Time Features

    Text features are used to generate semantic and statistical embeddings.
    """
    from autogluon.features.generators import (
        AsTypeFeatureGenerator,
        FillNaFeatureGenerator,
    )
    from autogluon.features.generators.selection import FeatureSelectionGenerator
    return {
        # Use default pre-generators but disabled conversion of bool features.
        "pre_generators": [
            AsTypeFeatureGenerator(convert_bool=False),
            FillNaFeatureGenerator(),
        ],
        "pre_enforce_types": False,
        "post_generators": [FeatureSelectionGenerator()]
    }


def get_dimensionality_reduction_pipeline():
    from autogluon.features.generators.identity import IdentityFeatureGenerator
    return [
        # Passthrough for all non-text-embedding features
        IdentityFeatureGenerator(
            infer_features_in_args={
                "valid_raw_types": None,
            }
        ),
    ]


def default_feature_selection(experiment):
    default_pipeline = get_default_encoding_pipeline()

    # Add pipeline to experiment
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_selector_kwargs"
    ] = default_pipeline

    return new_experiment


def default_model_specific_pca(experiment):
    default_pipeline = get_default_encoding_pipeline()
    dr_pipeline = get_dimensionality_reduction_pipeline()

    # Add default pipeline to experiment
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_generator_kwargs"
    ] = default_pipeline

    # new_experiment.method_kwargs["init_kwargs"]["verbosity"] = 4

    # Add model-specific dimensionality reduction to experiment
    hps = new_experiment.method_kwargs["model_hyperparameters"]
    # TODO: figure out if this generalizes to having experiments with multiple models?
    #   ... and figure out if experiments with multiple models even exist
    # set to a large number such that max_features filter does not trigger before PCA
    hps["ag.model_specific_feature_generator_kwargs"] = {
        "feature_generators": [dr_pipeline],
    }
    new_experiment.method_kwargs["model_hyperparameters"] = hps
    return new_experiment


DEFAULT_PIPELINE_WITH_FEATURE_SELECTION = "D-PRE-FS_DR"

PREPROCESSING_METHODS = {
    DEFAULT_PIPELINE_WITH_FEATURE_SELECTION: default_feature_selection,
}
