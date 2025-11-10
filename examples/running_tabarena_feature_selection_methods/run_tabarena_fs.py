from autogluon.features import FeatureSelectionGenerator
from autogluon.tabular import TabularPredictor
from tabarena.benchmark.task.openml import OpenMLTaskWrapper


task_id = 146818  # anneal
task = OpenMLTaskWrapper.from_task_id(task_id=task_id)

methods = ["MetaFS", "Boruta", "Select_k_Best_Chi2"]

for method in methods:

    train_data, test_data = task.get_train_test_split_combined(fold=0)

    predictor = TabularPredictor(
        label=task.label,
        problem_type=task.problem_type,
        eval_metric=task.eval_metric,
    )

    predictor = predictor.fit(
        train_data=train_data,
        # presets="best",  # uncomment for a longer run
        _feature_generator_kwargs={
                "post_generators": [FeatureSelectionGenerator(method)],
            }
    )

    leaderboard = predictor.leaderboard(test_data, display=True)
