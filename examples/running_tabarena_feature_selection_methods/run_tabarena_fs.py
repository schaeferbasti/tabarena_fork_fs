import os

import pandas as pd
from autogluon.features import FeatureSelectionGenerator
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt

from tabarena.benchmark.task.openml import OpenMLTaskWrapper


def main():
    task_id = 146818  # anneal
    task = OpenMLTaskWrapper.from_task_id(task_id=task_id)

    results = {}
    methods = ["MetaFS", "Boruta", "Select_k_Best_Chi2", "Original", "MAFESE", "LS_Flip"]

    for method in methods:

        train_data, test_data = task.get_train_test_split_combined(fold=0)

        leaderboard_path = os.path.join("./leaderboards", f"leaderboard_{task_id}_{method}.parquet")

        # Load if exists
        if os.path.exists(leaderboard_path):
            leaderboard = pd.read_parquet(leaderboard_path)
        else:
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
            leaderboard.to_parquet(leaderboard_path)
        results[method] = leaderboard

    plot_leaderboards(methods, results, task)


def plot_leaderboards(methods, results, task):
    # Plot 1: Best score by method
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    scores = {method: results[method].iloc[0]['score_test'] for method in methods}
    ax1.bar(scores.keys(), scores.values())
    ax1.set_title('Best Model Score by Feature Selection Method')
    ax1.set_ylabel(task.eval_metric)
    ax1.set_xlabel('Method')
    plt.tight_layout()
    plt.savefig('best_score_by_method.png', dpi=150)
    plt.show()

    # Plot 2: All model scores per method (with proper global alignment)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for method in methods:
        sorted_leaderboard = results[method].sort_values('model').reset_index(drop=True)
        model_names = sorted_leaderboard['model'].values
        scores_all = sorted_leaderboard['score_test'].values
        ax2.plot(model_names, scores_all, marker='o', label=method, linewidth=2, markersize=6)
    ax2.set_title('Model Test Scores per Feature Selection Method')
    ax2.set_ylabel(task.eval_metric)
    ax2.set_xlabel('Models')
    ax2.tick_params(axis='x', rotation=90)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_scores_per_method.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
