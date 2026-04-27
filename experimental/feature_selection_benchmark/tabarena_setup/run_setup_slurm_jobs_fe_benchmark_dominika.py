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
        "AccuracyFeatureSelector",
        "RandomFeatureSelector",
        "ANOVAFeatureSelector",
        "ConsistencyFeatureSelector", 
        "CFSFeatureSelector",
        "DISRFeatureSelector",
        "GainRatioFeatureSelector",
        "SymmetricalUncertaintyFeatureSelector",
        "CMIMFeatureSelector", 
        "MIFeatureSelector",
        "JMIFeatureSelector", 
        "CARTFeatureSelector", 

        # "INTERACTFeatureSelector",
        # "MarkovBlanketFeatureSelector",
        # "mRMRFeatureSelector",
        # "PearsonCorrelationFeatureSelector",
        # "ReliefFFeatureSelector",
        # "RFImportanceFeatureSelector",
        # "SequentialBackwardEliminationFeatureSelector",
        # "SequentialForwardSelectionFeatureSelector",
        # "LassoFeatureSelector", # just for regression but with label encoder for classification?
        # "LaplacianScoreFeatureSelector", # OOM, Segmentation fault issues
        # "OneRFeatureSelector", # major OOM errors (tries to allocate one major array), wrong time limit computation,  max(accuracies, key=accuracies.get) -> max() iterable argument is empty
        # "ElasticNetFeatureSelector", # Only for classification
    
        #EXCLUDED
        # "Chi2FeatureSelector",  # classification only
        # "GiniFeatureSelector", # classification only
        # "ImpurityFeatureSelector",  # classification only
        # "tTestFeatureSelector", # classification only
        # "InformationGainFeatureSelector", # equivalent to MI but discretizes continuous vars, MI uses knn for calcuating entropy

    ]
)

# Setup for CPU Methods (need to create another one for GPU models)
TabArenaBenchmarkSetup(
    # You could filter this to run less tasks
    task_metadata=ALL_TASK_METADATA,
    # Only run first three folds for now
    split_indices_to_run=["r0f0"],
    # Run methods for 5 configs (1 default + 4 random) each for now
    n_random_configs=0,
    models=[
        ("LightGBM", "all"),
        ("RandomForest", "all"),
        ("Linear", "all"),
    ],
    preprocessing_pipelines=preprocessing_pipelines,
    path_setup=UniPathSetupDominika(),
).setup_jobs()