#!/usr/bin/env bash
# Example submit:
#   sbatch --array=0-7 slurm/optuna_worker.sbatch
#
# NOTE:
# - For multi-node runs, do NOT use sqlite storage.
# - Use PostgreSQL, e.g.:
#   postgresql://user:pass@host:5432/optuna

#SBATCH --job-name=sted-optuna
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_optuna_%A_%a.out
#SBATCH --error=logs/slurm_optuna_%A_%a.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/path/to/gym-sted-pfrl}"
VENV_PY="${VENV_PY:-/path/to/venv/bin/python3}"

STUDY="${STUDY:-routeA_v1}"
STORAGE="${STORAGE:-postgresql://user:pass@host:5432/optuna}"
N_TRIALS="${N_TRIALS:-20}"
STEPS="${STEPS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_N_RUNS="${EVAL_N_RUNS:-3}"
EVAL_SEED="${EVAL_SEED:-2026}"
OUTDIR="${OUTDIR:-./data_optuna}"
LOGDIR="${LOGDIR:-./optuna_logs}"
WANDB_PROJECT="${WANDB_PROJECT:-sted-routeA}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-$STUDY}"
WANDB_TAGS="${WANDB_TAGS:-optuna,sted,slurm}"

WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/$LOGDIR" "$ROOT_DIR/$OUTDIR"
cd "$ROOT_DIR"

"$VENV_PY" optuna_worker.py \
  --study-name "$STUDY" \
  --storage "$STORAGE" \
  --worker-id "$WORKER_ID" \
  --n-trials "$N_TRIALS" \
  --env "ContextualMOSTED-easy-hslb-v0" \
  --steps "$STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-n-runs "$EVAL_N_RUNS" \
  --eval-seed "$EVAL_SEED" \
  --outdir "$OUTDIR" \
  --log-dir "$LOGDIR" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-group "$WANDB_GROUP" \
  --wandb-tags "$WANDB_TAGS" \
  --use-tensorboard
