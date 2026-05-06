"""Shared infrastructure and entry point for feature selection benchmark evaluation.

Usage:
    python feature_selection_benchmark_runner.py \
        --mode validity \
        --method_name FSBench__RandomFeatureSelector__5__0__lgbm__3600 \
        --data_foundry_task_id "UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792|/work/dlclarge1/purucker-fs_benchmark/.openml/tabarena_tasks" \

    python feature_selection_benchmark_runner.py \
        --mode stability \
        --method_name FSBench__RandomFeatureSelector__5__0__lgbm__3600 \
        --data_foundry_task_id "UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792|/work/dlclarge1/purucker-fs_benchmark/.openml/tabarena_tasks" \
"""
from __future__ import annotations

import argparse
import time
import os    ### NEW: Needed for checking file size
import ast   ### NEW: Needed for parsing the dictionary string in mode_kwargs
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import re
import numpy as np
import pandas as pd
from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
    selector_and_config_from_string,
)
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabflow_slurm.run_tabarena_experiment import _parse_task_id


@dataclass(frozen=True, kw_only=True)
class ExtraBenchmarkJob:
    mode: str
    method_name: str
    data_foundry_task_id: str
    repeat: int = 0
    noise: float | None = None
    noise_type: str | None = None
    ignore_cache: bool = False


@dataclass
class FeatureSelectionResult:
    method: str
    data_foundry_task_id: str
    mode: str
    mode_kwargs: dict[str, Any]
    repeat: int
    original_features: list[str]
    max_features: int
    selected_features: list[Any]
    num_classes: int
    num_samples: int
    min_samples_per_class: int
    elapsed_time_fs: float


def get_cache_path(job: ExtraBenchmarkJob) -> Path:
    cache_dir = Path(__file__).parent / "results"
    cache_dir.mkdir(parents=True, exist_ok=True)

    task_name = job.data_foundry_task_id
    task_name = re.sub(r'[|: /\[\]\(\)]', '_', task_name)
    task_name = re.sub(r'_+', '_', task_name)
    safe_method = job.method_name.replace("/", "_").replace(" ", "_")
    safe_mode = job.mode.replace("/", "_").replace(" ", "_")

    return cache_dir / f"{safe_mode}_{safe_method}_{task_name}.csv"


def _augment_dataset(
    *, mode: str, X: pd.DataFrame, y: pd.Series, rng: np.random.Generator, **kwargs
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    original_features = list(X.columns)

    if mode == "validity":
        from validity_fs_metric import get_dataset_for_validity  # noqa: PLC0415

        X = get_dataset_for_validity(X=X, rng=rng, **kwargs)
    elif mode == "stability":
        from stability_fs_metric import get_dataset_for_stability  # noqa: PLC0415

        X, y = get_dataset_for_stability(X=X, y=y, rng=rng, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    shuffled_cols = rng.permutation(X.columns)
    X = X[shuffled_cols]
    return X, y, original_features


def run_benchmark(job: ExtraBenchmarkJob) -> FeatureSelectionResult:
    rng = np.random.default_rng(42 + job.repeat)

    task_id = _parse_task_id(job.data_foundry_task_id)
    task = OpenMLTaskWrapper(task=task_id.load_local_openml_task())
    X, y = task.X, task.y

    feature_selector, config = selector_and_config_from_string(
        preprocessing_name=job.method_name
    )

    kwargs: dict[str, Any] = {}
    if job.mode == "validity":
        kwargs = {"noise": job.noise, "noise_type": job.noise_type}

    X, y, original_features = _augment_dataset(mode=job.mode, X=X, y=y, rng=rng, **kwargs)

    start_time = time.monotonic()
    feature_selector.fit_transform(
        X=X,
        y=y,
        time_limit=config.fs_time,
        eval_metric=task.eval_metric,
        problem_type=task.problem_type,
    )
    elapsed_time = time.monotonic() - start_time

    return FeatureSelectionResult(
        method=job.method_name,
        data_foundry_task_id=job.data_foundry_task_id,
        mode=job.mode,
        mode_kwargs=kwargs,
        repeat=job.repeat,
        original_features=original_features,
        max_features=feature_selector.max_features,
        selected_features=feature_selector._selected_features,
        num_classes=int(y.nunique()),
        num_samples=int(X.shape[0]),
        min_samples_per_class=y.value_counts().min(),
        elapsed_time_fs=elapsed_time,
    )


def save_results(job: ExtraBenchmarkJob, results: list[FeatureSelectionResult]) -> Path:
    cache_path = get_cache_path(job)
    df_new = pd.DataFrame([r.__dict__ for r in results])

    if not cache_path.exists() or os.path.getsize(cache_path) == 0:
        df_new.to_csv(cache_path, index=False, mode='w', header=True)
    else:
        df_new.to_csv(cache_path, index=False, mode='a', header=False)

    return cache_path


def run_extra_benchmark_job(base_job: ExtraBenchmarkJob) -> Path | None:
    cache_path = get_cache_path(base_job)

    ### NEW: Analyze the existing cache file to see what has already been computed
    existing_repeats = set()
    existing_noises = set()

    if cache_path.exists() and not base_job.ignore_cache:
        if os.path.getsize(cache_path) > 0:
            try:
                df_existing = pd.read_csv(cache_path)
                if base_job.mode == "stability" and "repeat" in df_existing.columns:
                    existing_repeats = set(df_existing["repeat"].dropna().unique())
                elif base_job.mode == "validity" and "mode_kwargs" in df_existing.columns:
                    # Safely parse the mode_kwargs string to extract already computed noise values
                    for val in df_existing["mode_kwargs"].dropna():
                        try:
                            d = ast.literal_eval(val)
                            if "noise" in d:
                                existing_noises.add(d["noise"])
                        except:
                            pass
            except Exception as e:
                print(f"Warning: Could not parse existing cache {cache_path}: {e}")

    all_results = []

    if base_job.mode == "stability":
        ### CHANGED: Defines the total runs (30) and filters out what is already done
        target_repeats = set(range(30))
        missing_repeats = sorted(list(target_repeats - existing_repeats))

        if not missing_repeats:
            print(f"All 30 stability repeats already exist in {cache_path}. Skipping.")
            return None

        print(f"Running stability benchmark (executing {len(missing_repeats)} missing repetitions)...")
        for rep in missing_repeats:
            current_job = replace(base_job, repeat=rep)
            print(f"  -> Executing repeat {rep}...")
            result = run_benchmark(current_job)
            all_results.append(result)

    elif base_job.mode == "validity":
        ### CHANGED: Defines total noises and filters out what is already done
        target_noises = {0.5, 0.75, 1.0}
        missing_noises = sorted(list(target_noises - existing_noises))

        if not missing_noises:
            print(f"All validity noises already exist in {cache_path}. Skipping.")
            return None

        print(f"Running validity benchmark for missing noises {missing_noises}...")
        for noise_val in missing_noises:
            current_job = replace(base_job, noise=noise_val)
            print(f"  -> Executing with noise {noise_val}...")
            result = run_benchmark(current_job)
            all_results.append(result)

    if all_results:
        save_results(base_job, all_results)
        print(f"Successfully appended {len(all_results)} results to {cache_path}")

    return cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FS Benchmark Runner")
    parser.add_argument("--mode", type=str, choices=["validity", "stability"], required=True)
    parser.add_argument("--method_name", type=str, default="FSBench__RandomFeatureSelector__5__0__lgbm__3600")
    parser.add_argument(
        "--data_foundry_task_id",
        type=str,
        default="UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792|/Users/schaefer.bastian/.openml/tabarena_tasks",
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    parser.add_argument("--ignore_cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    job = ExtraBenchmarkJob(
        mode=args.mode,
        method_name=args.method_name,
        data_foundry_task_id=args.data_foundry_task_id,
        repeat=args.repeat,
        noise=args.noise,
        noise_type=args.noise_type,
        ignore_cache=args.ignore_cache,
    )

    run_extra_benchmark_job(job)
