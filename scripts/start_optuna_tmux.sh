#!/usr/bin/env bash
set -euo pipefail

# Example:
#   SESSION=sted_optuna WORKERS=4 STUDY=routeA_v1 STORAGE=sqlite:///optuna.db \
#   STEPS=5000 N_TRIALS=30 WANDB_PROJECT=sted-routeA ./scripts/start_optuna_tmux.sh

SESSION="${SESSION:-sted_optuna}"
WORKERS="${WORKERS:-2}"
STUDY="${STUDY:-routeA_v1}"
STORAGE="${STORAGE:-sqlite:///optuna.db}"
STEPS="${STEPS:-5000}"
N_TRIALS="${N_TRIALS:-20}"
TIMEOUT="${TIMEOUT:-0}"        # 0 = no timeout
ENV_ID="${ENV_ID:-ContextualMOSTED-easy-hslb-v0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_N_RUNS="${EVAL_N_RUNS:-3}"
EVAL_SEED="${EVAL_SEED:-2026}"
OUTDIR="${OUTDIR:-./data_optuna}"
LOGDIR="${LOGDIR:-./optuna_logs}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-$STUDY}"
WANDB_TAGS="${WANDB_TAGS:-optuna,sted}"
NO_DROPOUT="${NO_DROPOUT:-0}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-/Users/huygens_onepiece/task/code/pysted/smlm-control/venv/bin/python3}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$ROOT_DIR/$LOGDIR" "$ROOT_DIR/$OUTDIR"

for ((i=0; i<WORKERS; i++)); do
  WIN_NAME="w${i}"
  if [[ "$i" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$WIN_NAME"
  else
    tmux new-window -t "$SESSION" -n "$WIN_NAME"
  fi

  TIMEOUT_ARG=""
  if [[ "$TIMEOUT" -gt 0 ]]; then
    TIMEOUT_ARG="--timeout $TIMEOUT"
  fi

  DROPOUT_ARG=""
  if [[ "$NO_DROPOUT" -eq 1 ]]; then
    DROPOUT_ARG="--no-dropout"
  fi

  CMD="cd \"$ROOT_DIR\" && \
    \"$VENV_PY\" optuna_worker.py \
      --study-name \"$STUDY\" \
      --storage \"$STORAGE\" \
      --worker-id \"$i\" \
      --n-trials \"$N_TRIALS\" \
      $TIMEOUT_ARG \
      --env \"$ENV_ID\" \
      --steps \"$STEPS\" \
      --eval-interval \"$EVAL_INTERVAL\" \
      --eval-n-runs \"$EVAL_N_RUNS\" \
      --eval-seed \"$EVAL_SEED\" \
      --outdir \"$OUTDIR\" \
      --log-dir \"$LOGDIR\" \
      --wandb-project \"$WANDB_PROJECT\" \
      --wandb-entity \"$WANDB_ENTITY\" \
      --wandb-group \"$WANDB_GROUP\" \
      --wandb-tags \"$WANDB_TAGS\" \
      $DROPOUT_ARG \
      --use-tensorboard"

  tmux send-keys -t "$SESSION:$WIN_NAME" "$CMD" C-m
done

echo "Started $WORKERS workers in tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
