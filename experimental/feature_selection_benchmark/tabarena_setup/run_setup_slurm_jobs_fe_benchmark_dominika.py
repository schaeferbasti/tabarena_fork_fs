"""Run TabArena for feature selection benchmark with downstream model performance evaluation."""
from __future__ import annotations

import os
import sys

paths = [
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_matusd/tabarena/experimental/feature_selection_benchmark/tabarena_setup",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_matusd/tabarena",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_matusd/tabarena/tabflow_slurm"
]

# For current script
for path in paths:
    sys.path.insert(0, path)

# For child processes
existing = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = ':'.join(paths + ([existing] if existing else []))

from fr_cluster_setup import ALL_TASK_METADATA, FSBenchmarkConfig, TabArenaBenchmarkSetup, UniPathSetupDominika

preprocessing_pipelines = FSBenchmarkConfig().get_default_preprocessing_configs(
    fs_methods=[
        # "LOCOFeatureSelector",
        
        # "RandomFeatureSelector",
    
        "FTestFeatureSelector", 

        "CARTFeatureSelector", 
        "RFImportanceFeatureSelector",

        "GainRatioFeatureSelector",
        "MIFeatureSelector",
        "mRMRFeatureSelector",

        "LassoFeatureSelector", 
        "ElasticNetFeatureSelector", 

        # "LaplacianScoreFeatureSelector",
        
        "ReliefFFeatureSelector",

        "MarkovBlanketFeatureSelector", # not made explicitly for feature selection

        # "SequentialBackwardEliminationFeatureSelector",
        # "SequentialForwardSelectionFeatureSelector",
    
        #EXCLUDED
        # "Chi2FeatureSelector",  # classification only
        # "GiniFeatureSelector", # classification only
        # "ImpurityFeatureSelector",  # classification only
        # "tTestFeatureSelector", # classification only
        # "InformationGainFeatureSelector", # equivalent to MI but discretizes continuous vars, MI uses knn for calcuating entropy
        # "PearsonCorrelationFeatureSelector", # doesn't work for multi-class classification
        # "FisherScoreFeatureSelector" # classification only, equivalent to anova in ranking (scoring a bit differnt)
        # "OneRFeatureSelector", # classifcation only
        # "CFSFeatureSelector", # differential entropy missing
        # "DISRFeatureSelector",  # differential entropy missing
        # "SymmetricalUncertaintyFeatureSelector",
        # "CMIMFeatureSelector", # differential entropy missing
        # "INTERACTFeatureSelector",  # differential entropy missing
        # "ConsistencyFeatureSelector", # differential entropy missing
        # "JMIFeatureSelector",  # differential entropy missing
    ]
)


# encoding ablation
# ENCODING_PERMUTATION_SEEDS = [1, 2, 3]
# if ENCODING_PERMUTATION_SEEDS:
#     from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
#         with_encoding_permutation_variants,
#     )

#     preprocessing_pipelines = with_encoding_permutation_variants(
#         preprocessing_pipelines,
#         permutation_seeds=ENCODING_PERMUTATION_SEEDS,
#         include_baseline=False,
#     )


# # CPU setup encoding ablation
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r0f0", "r0f2", "r0f1"],
#     n_random_configs=0,
#     models=[
#         ("LightGBM", 0),
#         ("RandomForest", 0),
#         ("Linear", 0),
#     ],
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_2026_rebuttal",
#     parallel_benchmark_name="cpu_encoding_ablation"
# ).setup_jobs()

# # # GPU setup
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r0f0", "r0f2", "r0f1"],
#     # Run methods for 5 configs (1 default + 4 random) each for now
#     n_random_configs=0,
#     models=[
#         ("TabICLv2", "all"),
#         ("RealMLP", "all"),
#     ],
#     num_gpus=1,
#     fake_memory_for_estimates=280,  # To ensure TabICL knows it has 140GB VRAM
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_2026_rebuttal",
#     parallel_benchmark_name="gpu_encoding_ablation"
# ).setup_jobs()


# CPU setup
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r0f0", "r0f2", "r0f1"],
#     n_random_configs=0,
#     models=[
#         ("LightGBM", 0),
#         ("RandomForest", 0),
#         ("Linear", 0),
#     ],
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_2026_rebuttal",
#     parallel_benchmark_name="cpu"
# ).setup_jobs()

# # # GPU setup
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r0f0", "r0f2", "r0f1"],
#     # Run methods for 5 configs (1 default + 4 random) each for now
#     n_random_configs=0,
#     models=[
#         ("TabICLv2", "all"),
#         ("RealMLP", "all"),
#     ],
#     num_gpus=1,
#     fake_memory_for_estimates=280,  # To ensure TabICL knows it has 140GB VRAM
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_2026_rebuttal",
#     parallel_benchmark_name="gpu"
# ).setup_jobs()

# for no random fallback ablation
preprocessing_pipelines = [p + "__nofallback" for p in preprocessing_pipelines]

# # # GPU setup     no fallback
TabArenaBenchmarkSetup(
    # You could filter this to run less tasks
    task_metadata=ALL_TASK_METADATA,
    # Only run first three folds for now
    split_indices_to_run=["r0f0", "r0f2", "r0f1"],
    # Run methods for 5 configs (1 default + 4 random) each for now
    n_random_configs=0,
    models=[
        ("TabICLv2", "all"),
        ("RealMLP", "all"),
    ],
    num_gpus=1,
    fake_memory_for_estimates=280,  # To ensure TabICL knows it has 140GB VRAM
    preprocessing_pipelines=preprocessing_pipelines,
    path_setup=UniPathSetupDominika(),
    benchmark_name="feature_selection_benchmark_2026_rebuttal",
    parallel_benchmark_name="gpu_no_fallback"
).setup_jobs()

# CPU setup no fallback
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r0f0", "r0f2", "r0f1"],
#     n_random_configs=0,
#     models=[
#         ("LightGBM", 0),
#         ("RandomForest", 0),
#         ("Linear", 0),
#     ],
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_2026_rebuttal",
#     parallel_benchmark_name="cpu_no_fallback"
# ).setup_jobs()


#CPU setu[p for mldlcl
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r1f1"],
#     # Run methods for 5 configs (1 default + 4 random) each for now
#     n_random_configs=0,
#     models=[
#         ("LightGBM", "all"),
#         # ("RandomForest", "all"),
#         # ("Linear", "all"),
#     ],
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_DEBUG",
#     parallel_benchmark_name="DEBUG"
# ).setup_jobs()


# proxy model ablation studies
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r0f0"],
#     # Run methods for 5 configs (1 default + 4 random) each for now
#     n_random_configs=0,
#     models=[
#         ("LightGBM", "all"),
#     ],
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_2026_all",
#     parallel_benchmark_name="proxy"
# ).setup_jobs()


# CPU setup
# TabArenaBenchmarkSetup(
#     # You could filter this to run less tasks
#     task_metadata=ALL_TASK_METADATA,
#     # Only run first three folds for now
#     split_indices_to_run=["r1f1"],
#     # Run methods for 5 configs (1 default + 4 random) each for now
#     n_random_configs=0,
#     models=[
#         ("LightGBM", 0),
#         ("RandomForest", 0),
#         ("Linear", 0),
#     ],
#     preprocessing_pipelines=preprocessing_pipelines,
#     path_setup=UniPathSetupDominika(),
#     benchmark_name="feature_selection_benchmark_test_attributes",
#     parallel_benchmark_name="TEST"
# ).setup_jobs()


