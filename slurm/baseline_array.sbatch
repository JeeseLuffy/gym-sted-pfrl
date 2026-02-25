#!/usr/bin/env bash
# Example:
#   sbatch --array=0-4 slurm/baseline_array.sbatch

#SBATCH --job-name=sted-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_baseline_%A_%a.out
#SBATCH --error=logs/slurm_baseline_%A_%a.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/path/to/gym-sted-pfrl}"
VENV_PY="${VENV_PY:-/path/to/venv/bin/python3}"

SEEDS="${SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_N_RUNS="${EVAL_N_RUNS:-10}"
EVAL_SEED="${EVAL_SEED:-2026}"
GROUP_NAME="${GROUP_NAME:-baseline}"
NO_DROPOUT="${NO_DROPOUT:-1}"
LAMBDA_UNC="${LAMBDA_UNC:-0.0}"
WANDB_PROJECT="${WANDB_PROJECT:-sted-routeA}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

read -r -a SEED_ARRAY <<< "$SEEDS"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$IDX" -ge "${#SEED_ARRAY[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=$IDX out of range for seeds: $SEEDS"
  exit 1
fi
SEED="${SEED_ARRAY[$IDX]}"

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

DROP_ARG=""
if [[ "$NO_DROPOUT" -eq 1 ]]; then
  DROP_ARG="--no-dropout"
fi

"$VENV_PY" main.py \
  --env "ContextualMOSTED-easy-hslb-v0" \
  --steps "$STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-n-runs "$EVAL_N_RUNS" \
  --eval-seed "$EVAL_SEED" \
  --seed "$SEED" \
  --exp-id "${GROUP_NAME}_s${SEED}" \
  --lambda-unc "$LAMBDA_UNC" \
  --use-tensorboard \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-group "$GROUP_NAME" \
  --wandb-tags "baseline,slurm,sted" \
  --ignore-warnings \
  $DROP_ARG
