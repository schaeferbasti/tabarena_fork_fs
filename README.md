
## User Install

The following is installation instructions for users who intend to use but not develop TabArena.

### Clone the repository

```
git clone https://github.com/schaeferbasti/tabarena_fork_fs.git
```

### Evaluation (Leaderboard / Metrics)

If you don't intend to fit models, this is the simplest installation.

```
uv sync
```


## Developer Install with editable AutoGluon

Creating a custom virtual environment:
```
uv venv --seed --python 3.12 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate
```

With this installation, you will have the latest version of AutoGluon in editable form. You need this for running the example.
```
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh

git clone https://github.com/schaeferbasti/tabarena_fork_fs.git
```

## Example Run

```
cd examples/benchmarking
python run_quickstart_selectarena.py 
```

#### Minor Remarks:
ANOVA is called F-Test in the paper, Sequential Forward Selection is called SFS in the paper, and Sequential Backward Selection is called RFE in the paper. 
ReliefF is called (R)ReliefF in the paper, Accuracy is called LOCO.
re .csv files containing the metric results for each dataset and model. These are used for evaluation and analysis.