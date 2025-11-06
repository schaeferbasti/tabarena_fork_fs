"""Run a standalone TabArena Feature Selection Method using some model on any task."""

from __future__ import annotations

from autogluon.common import TabularDataset
from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features import FeatureSelectionGenerator
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from tabarena.models.utils import get_configs_generator_from_name


task_type = "binary"
cross_validation_bagging = True
refit_model = False
model_to_run = "CatBoost"

model_meta = get_configs_generator_from_name(model_name=model_to_run)
model_cls = model_meta.model_cls
model_config = model_meta.manual_configs[0]

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv')

X_train = train_data.drop("class", axis=1)
y_train = train_data["class"]
X_test = test_data.drop("class", axis=1)
y_test = test_data["class"]

# --- Using a TabArena Model: Preprocessing, Train, and Predict:
print(f"Running TabArena model {model_to_run} on task type {task_type}...")
feature_generator, label_cleaner = (
    AutoMLPipelineFeatureGenerator(
        post_generators=[FeatureSelectionGenerator("MAFESE")]
        # enable_feature_selection="Boruta"  # -> Validation accuracy: 0.8538376884293502
        # enable_feature_selection="Select_k_Best_Chi2"  # -> Validation accuracy: 0.8317764184987075
        # enable_feature_selection=True  # -> Validation accuracy: 0.8317764184987075
        # enable_feature_selection=False  # -> Validation accuracy: 0.8754126890691782
    ),
    LabelCleaner.construct(problem_type=task_type, y=y_train),
)
X_train, y_train = (
    feature_generator.fit_transform(X_train, y_train),
    label_cleaner.transform(y_train),
)
X_test, y_test = feature_generator.transform(X_test), label_cleaner.transform(y_test)

if cross_validation_bagging:
    model = BaggedEnsembleModel(
        model_cls(problem_type=task_type, **model_config),
        hyperparameters=dict(refit_folds=refit_model),
    )
    model.params["fold_fitting_strategy"] = "sequential_local"
    model = model.fit(X=X_train, y=y_train, k_fold=8)
    print(f"Validation {model.eval_metric.name}:", model.score_with_oof(y=y_train))
else:
    model = model_cls(problem_type=task_type, **model_config)
    model = model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)
