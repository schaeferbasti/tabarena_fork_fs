from pathlib import Path

import pandas as pd

import os
import sys
import subprocess

paths = [
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/tabarena/experimental/feature_selection_benchmark/tabarena_setup",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/tabarena",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/tabarena/tabflow_slurm",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/tabarena/bencheval",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/tabarena/examples/tabrepo",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/tabarena/tabarena",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/autogluon/tabular/src/",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/autogluon/common/src/",
    "/work/dlclarge1/purucker-fs_benchmark/code/fsbench_schaefeb/autogluon/features/src/",
]

# For current script
for path in paths:
    sys.path.insert(0, path)

# For child processes
existing = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = ':'.join(paths + ([existing] if existing else []))

from experimental.feature_selection_benchmark.extra_benchmark.feature_selection_benchmark_runner import (
    ExtraBenchmarkJob)

from fr_cluster_setup import ALL_TASK_METADATA, FSBenchmarkConfig, FS_TIME_LIMIT, UniPathSetupBastian


def build_jobs():
    method_names = FSBenchmarkConfig().get_default_preprocessing_configs(
        fs_methods=[
            "AccuracyFeatureSelector",
            "RandomFeatureSelector",
            "ANOVAFeatureSelector",
            "CFSFeatureSelector",
            "Chi2FeatureSelector",
            "DISRFeatureSelector",
            "GainRatioFeatureSelector",
            "GiniFeatureSelector",
            "ImpurityFeatureSelector",
            "InformationGainFeatureSelector",
            "INTERACTFeatureSelector",
            "MarkovBlanketFeatureSelector",
            "MIFeatureSelector",
            "mRMRFeatureSelector",
            "PearsonCorrelationFeatureSelector",
            "ReliefFFeatureSelector",
            "RFImportanceFeatureSelector",
            "SequentialBackwardEliminationFeatureSelector",
            "SequentialForwardSelectionFeatureSelector",
            "SymmetricalUncertaintyFeatureSelector",
            "LassoFeatureSelector",
            "LaplacianScoreFeatureSelector",
            "ConsistencyFeatureSelector",
            "JMIFeatureSelector",
            "OneRFeatureSelector",
            "ElasticNetFeatureSelector",
            "CMIMFeatureSelector",
            "CARTFeatureSelector",
        ]
    )
    task_ids = pd.read_csv(ALL_TASK_METADATA)["task_id_str"]
    task_ids.drop_duplicates(inplace=True)
    task_ids = task_ids.tolist()

    modes = ["stability", "validity"]
    noises = [0.5, 0.75, 1.0]
    noise_types = ["gaussian"]

    jobs = []
    for mode in modes:
        for method_name in method_names:
            for task_id in task_ids:
                if mode == "validity":
                    for noise in noises:
                        for noise_type in noise_types:
                            jobs.append(
                                ExtraBenchmarkJob(
                                    mode=mode,
                                    method_name=method_name,
                                    data_foundry_task_id=task_id,
                                    repeat=method_name.split("__")[-3],
                                    noise=noise,
                                    noise_type=noise_type,
                                )
                            )
                else:
                    jobs.append(
                        ExtraBenchmarkJob(
                            mode=mode,
                            method_name=method_name,
                            data_foundry_task_id=task_id,
                            repeat=method_name.split("__")[-3],
                            noise=None,
                            noise_type=None,
                        )
                    )
    return jobs


def generate_job_array(jobs):
    """Return command args for each array task"""
    commands = []
    for i, job in enumerate(jobs):
        args = f'"{job.mode}" "{job.method_name}" "{job.data_foundry_task_id}" "{job.repeat}" "{job.noise}" "{job.noise_type}"'
        commands.append(args)
    return commands


if __name__ == "__main__":
    jobs = build_jobs()
    commands = generate_job_array(jobs)

    base_log_folder = Path("../../extra_out")
    base_log_folder.mkdir(parents=True, exist_ok=True)

    MAX_ARRAY = 10000
    PARALLEL = 100

    print(f"Generated {len(commands)} jobs")

    job_ids = []
    chunk_size = MAX_ARRAY

    for start in range(0, len(commands), chunk_size):
        end = min(start + chunk_size, len(commands))
        chunk_id = f"{start:06d}-{end - 1:06d}"

        chunk_folder = base_log_folder / f"submit_{chunk_id}"
        chunk_folder.mkdir(parents=True, exist_ok=True)

        commands_file = chunk_folder / "job_commands.txt"
        with open(commands_file, "w") as f:
            f.write("\n".join(commands))

        script = f"""#!/bin/bash
#SBATCH --job-name=fs_bench_{chunk_id}
#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={FS_TIME_LIMIT // 3600}:00:00
#SBATCH --array={start}-{end - 1}%{PARALLEL}
#SBATCH --output="{chunk_folder.absolute()}/slurm-%A_%a.out"
#SBATCH --error="{chunk_folder.absolute()}/slurm-%A_%a.err"

source /work/dlclarge1/purucker-fs_benchmark/venvs/venv_fs_bench_schaefeb/bin/activate
export PYTHONPATH="{os.environ.get('PYTHONPATH', '')}"

set -euo pipefail

# Use absolute submission directory (passed from Python)
SUBMIT_DIR="{chunk_folder.absolute()}"
JOB_ROOT="$SUBMIT_DIR/$SLURM_ARRAY_JOB_ID"
mkdir -p "$JOB_ROOT"

# Copy artifacts (absolute paths)
cp "$SUBMIT_DIR/job_commands.txt" "$JOB_ROOT/"
cp "$0" "$JOB_ROOT/"

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
ARGS=$(sed -n "${{LINE_NO}}p" "$SUBMIT_DIR/job_commands.txt")

if [ -z "$ARGS" ]; then
  echo "ERROR: No command for line $LINE_NO" >&2
  exit 1
fi

eval "set -- $ARGS"
MODE="$1"
METHOD="$2"
TASK="$3"
REPEAT="$4"
NOISE="$5"
NOISE_TYPE="$6"

# Sanitize filename
FILENAME=$(echo "${{MODE}}_${{METHOD}}_${{TASK}}" | tr '|: /[]()' '_' | sed 's/__*/_/g' | cut -c1-80)

OUTFILE="$JOB_ROOT/${{FILENAME}}.out"
ERRFILE="$JOB_ROOT/${{FILENAME}}.err"

echo "SUBMIT_DIR=$SUBMIT_DIR JOB_ROOT=$JOB_ROOT FILENAME=$FILENAME" >&2

exec >"$OUTFILE" 2>"$ERRFILE"

cd {Path(__file__).absolute().parent}
python3 feature_selection_benchmark_runner.py \\
  --mode "$MODE" \\
  --method_name "$METHOD" \\
  --data_foundry_task_id "$TASK" \\
  --repeat "$REPEAT"
  --noise "$NOISE" \\
  --noise_type "$NOISE_TYPE"
"""

        batch_file = chunk_folder / f"fs_array_{chunk_id}.sh"
        with open(batch_file, "w") as f:
            f.write(script)
        os.chmod(batch_file, 0o755)

        result = subprocess.run(["sbatch", str(batch_file)], capture_output=True, text=True)
        print(f"Chunk {start}-{end - 1}: {result.stdout.strip()}")
        if result.stderr:
            print(f"ERROR chunk {start}-{end - 1}: {result.stderr.strip()}")
            break
        job_ids.append(result.stdout.strip().split()[-1])

    print(f"Submitted chunks: {job_ids}")
    print("Monitor: watch 'squeue -u schaefeb'")
