# Optuna + W&B Runbook

## 1) Install dependencies

```bash
cd /Users/huygens_onepiece/task/code/pysted/smlm-control/gym-sted-pfrl
source /Users/huygens_onepiece/task/code/pysted/smlm-control/venv/bin/activate
pip install optuna wandb optuna-dashboard
```

## 2) Login to W&B

```bash
wandb login
```

## 3) Single machine, unattended, parallel (tmux)

```bash
cd /Users/huygens_onepiece/task/code/pysted/smlm-control/gym-sted-pfrl
chmod +x scripts/start_optuna_tmux.sh scripts/start_baseline_tmux.sh

SESSION=sted_optuna \
WORKERS=4 \
STUDY=routeA_v1 \
STORAGE=sqlite:///optuna.db \
STEPS=5000 \
N_TRIALS=25 \
WANDB_PROJECT=sted-routeA \
WANDB_ENTITY=JeeseLuffy \
./scripts/start_optuna_tmux.sh
```

Attach:

```bash
tmux attach -t sted_optuna
```

Dashboard:

```bash
optuna-dashboard sqlite:///optuna.db
```

## 4) Formal baseline (100k, 3-5 seeds) with tmux

```bash
SESSION=sted_baseline \
GROUP=baseline \
SEEDS="0 1 2 3 4" \
STEPS=100000 \
NO_DROPOUT=1 \
LAMBDA_UNC=0.0 \
WANDB_PROJECT=sted-routeA \
WANDB_ENTITY=JeeseLuffy \
./scripts/start_baseline_tmux.sh
```

## 5) Cluster mode (Slurm)

- Use [optuna_worker.sbatch](/Users/huygens_onepiece/task/code/pysted/smlm-control/gym-sted-pfrl/slurm/optuna_worker.sbatch) for search workers.
- Use [baseline_array.sbatch](/Users/huygens_onepiece/task/code/pysted/smlm-control/gym-sted-pfrl/slurm/baseline_array.sbatch) for 100k seed arrays.
- For multi-node Optuna, use PostgreSQL storage (not sqlite).

## Notes

- `main.py` supports `--eval-seed` for fixed evaluation seed.
- `main.py` writes `train_metrics.tsv` with `action_mean` and `action_mean_window`.
- `optuna_worker.py` prunes trials when `action_mean_window` collapses near zero.
