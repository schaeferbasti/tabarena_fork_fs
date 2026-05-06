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
        "CARTFeatureSelector", 

        "ANOVAFeatureSelector",

        "ConsistencyFeatureSelector", 

        "CFSFeatureSelector",
        "DISRFeatureSelector",
        "GainRatioFeatureSelector",
        "SymmetricalUncertaintyFeatureSelector",
        "CMIMFeatureSelector", 
        "MIFeatureSelector",
        "JMIFeatureSelector", 
        "mRMRFeatureSelector",
       
        "LassoFeatureSelector", # just for regression but with label encoder for classification?
        "ElasticNetFeatureSelector", # Only for classification

        # "INTERACTFeatureSelector",
        # "MarkovBlanketFeatureSelector",
        # "ReliefFFeatureSelector",
        # "RFImportanceFeatureSelector",
        # "SequentialBackwardEliminationFeatureSelector",
        # "SequentialForwardSelectionFeatureSelector",
        # "LaplacianScoreFeatureSelector", # OOM, Segmentation fault issues
        # "OneRFeatureSelector", # major OOM errors (tries to allocate one major array), wrong time limit computation,  max(accuracies, key=accuracies.get) -> max() iterable argument is empty
    
        #EXCLUDED
        # "Chi2FeatureSelector",  # classification only
        # "GiniFeatureSelector", # classification only
        # "ImpurityFeatureSelector",  # classification only
        # "tTestFeatureSelector", # classification only
        # "InformationGainFeatureSelector", # equivalent to MI but discretizes continuous vars, MI uses knn for calcuating entropy
        # "PearsonCorrelationFeatureSelector", # doesn't work for multi-class classification

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