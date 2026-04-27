"""Run TabArena for feature selection benchmark with downstream model performance evaluation."""
from __future__ import annotations

import os
import sys

paths = [
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/tabarena/experimental/feature_selection_benchmark/tabarena_setup",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/tabarena",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/tabarena/tabflow_slurm",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/tabarena/bencheval",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/tabarena/examples/tabrepo",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/tabarena/tabarena",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/autogluon/tabular/src/",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/autogluon/common/src/",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_test/autogluon/features/src/",
]

# For current script
for path in paths:
    sys.path.insert(0, path)

# For child processes
existing = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = ':'.join(paths + ([existing] if existing else []))

from fr_cluster_setup import ALL_TASK_METADATA, FSBenchmarkConfig, TabArenaBenchmarkSetup, UniPathSetupBastian

preprocessing_pipelines = FSBenchmarkConfig().get_default_preprocessing_configs(
    fs_methods=[
        #"AccuracyFeatureSelector",
        "RandomFeatureSelector",
        #"ANOVAFeatureSelector",
        #"CFSFeatureSelector",
        #"Chi2FeatureSelector",
        #"DISRFeatureSelector",
        #"GainRatioFeatureSelector",
        #"GiniFeatureSelector",
        #"ImpurityFeatureSelector",
        #"InformationGainFeatureSelector",
        #"INTERACTFeatureSelector",
        #"MarkovBlanketFeatureSelector",
        #"MIFeatureSelector",
        #"mRMRFeatureSelector",
        #"PearsonCorrelationFeatureSelector",
        #"ReliefFFeatureSelector",
        #"RFImportanceFeatureSelector",
        #"SequentialBackwardEliminationFeatureSelector",
        #"SequentialForwardSelectionFeatureSelector",
        #"SymmetricalUncertaintyFeatureSelector",
        # "LassoFeatureSelector", # just for regression but with label encoder for classification?
        # "LaplacianScoreFeatureSelector", # OOM, Segmentation fault issues
        # "ConsistencyFeatureSelector", # selected_indices = np.where(S)[0].tolist(), UnboundLocalError: cannot access local variable 'S' where it is not associated with a value
        # "JMIFeatureSelector", # time limit computed incorrectly, and error at remaining.remove(best_idx), ValueError: list.remove(x): x not in list
        # "OneRFeatureSelector", # major OOM errors (tries to allocate one major array), wrong time limit computation,  max(accuracies, key=accuracies.get) -> max() iterable argument is empty
        # "ElasticNetFeatureSelector", # Only for classification
        # "CMIMFeatureSelector", # problems with time limit and fallback of features
        # "tTestFeatureSelector", # Does not work for regression
        # "CARTFeatureSelector", # Only implemented for classification, OOM problems as well
    ]
)

# Setup for CPU Methods (need to create another one for GPU models)
TabArenaBenchmarkSetup(
    task_metadata=ALL_TASK_METADATA,
    split_indices_to_run=["r0f0"],
    n_random_configs=0,
    models=[
        ("TabICLv2", 0),
    ],
    num_gpus=1,
    fake_memory_for_estimates=140,  # To ensure TabICL knows it has 140GB VRAM
    custom_model_constraints={
                    "max_n_samples_train_per_fold": 50_000,
                    "max_n_features": 500,
                    "regression_support": False,
                },
    path_setup=UniPathSetupBastian(),
).setup_jobs()