#!/usr/bin/env bash
set -euo pipefail

# Example:
#   SESSION=sted_base GROUP=baseline STEPS=100000 SEEDS="0 1 2" NO_DROPOUT=1 \
#   WANDB_PROJECT=sted-routeA ./scripts/start_baseline_tmux.sh

SESSION="${SESSION:-sted_baseline}"
GROUP="${GROUP:-baseline}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_N_RUNS="${EVAL_N_RUNS:-10}"
EVAL_SEED="${EVAL_SEED:-2026}"
LAMBDA_UNC="${LAMBDA_UNC:-0.0}"
NO_DROPOUT="${NO_DROPOUT:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-baseline,tmux,sted}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-/Users/huygens_onepiece/task/code/pysted/smlm-control/venv/bin/python3}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

idx=0
for seed in $SEEDS; do
  win="s${seed}"
  if [[ "$idx" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$win"
  else
    tmux new-window -t "$SESSION" -n "$win"
  fi

  DROP_ARG=""
  if [[ "$NO_DROPOUT" -eq 1 ]]; then
    DROP_ARG="--no-dropout"
  fi

  WB_ARGS=""
  if [[ -n "$WANDB_PROJECT" ]]; then
    WB_ARGS="--wandb-project \"$WANDB_PROJECT\" --wandb-entity \"$WANDB_ENTITY\" --wandb-group \"$GROUP\" --wandb-tags \"$WANDB_TAGS\""
  fi

  CMD="cd \"$ROOT_DIR\" && \
    \"$VENV_PY\" main.py \
      --env \"ContextualMOSTED-easy-hslb-v0\" \
      --steps \"$STEPS\" \
      --eval-interval \"$EVAL_INTERVAL\" \
      --eval-n-runs \"$EVAL_N_RUNS\" \
      --eval-seed \"$EVAL_SEED\" \
      --seed \"$seed\" \
      --exp-id \"${GROUP}_s${seed}\" \
      --lambda-unc \"$LAMBDA_UNC\" \
      --use-tensorboard \
      --ignore-warnings \
      $DROP_ARG \
      $WB_ARGS"

  tmux send-keys -t "$SESSION:$win" "$CMD" C-m
  idx=$((idx + 1))
done

echo "Started seeds [$SEEDS] in tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
