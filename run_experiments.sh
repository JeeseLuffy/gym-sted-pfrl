#!/bin/bash
# Route A Experiment Matrix
# 4 groups × 5 seeds (except A: 1 seed)
# Usage: bash run_experiments.sh

set -e

VENV="/Users/huygens_onepiece/task/code/pysted/smlm-control/venv/bin/python3"
SCRIPT="main.py"
ENV="ContextualMOSTED-easy-hslb-v0"
STEPS=100000
EVAL_INTERVAL=5000
EVAL_N_RUNS=10
SEEDS="0 1 2 3 4"

echo "=== Route A Experiment Matrix ==="
echo "Env: $ENV"
echo "Steps: $STEPS"
echo "Seeds: $SEEDS"
echo ""

# --- Group A: Random Agent (1 seed, for quick lower bound) ---
echo ">>> Group A: Random Agent"
$VENV $SCRIPT \
  --env "$ENV" --steps 5000 --eval-interval 1000 --eval-n-runs 5 \
  --seed 0 --use-tensorboard --exp-id "A_random_s0" \
  --ignore-warnings

# --- Group B: PPO Baseline (no Dropout, no penalty) ---
echo ""
echo ">>> Group B: PPO Baseline"
for seed in $SEEDS; do
  echo "  Seed $seed..."
  $VENV $SCRIPT \
    --env "$ENV" --steps $STEPS \
    --eval-interval $EVAL_INTERVAL --eval-n-runs $EVAL_N_RUNS \
    --seed $seed --use-tensorboard --exp-id "B_baseline_s${seed}" \
    --lambda-unc 0.0 --no-dropout --ignore-warnings &
done
wait
echo "  Group B done."

# --- Group C: PPO + Dropout (lambda=0) ---
echo ""
echo ">>> Group C: PPO + Dropout (lambda=0)"
for seed in $SEEDS; do
  echo "  Seed $seed..."
  $VENV $SCRIPT \
    --env "$ENV" --steps $STEPS \
    --eval-interval $EVAL_INTERVAL --eval-n-runs $EVAL_N_RUNS \
    --seed $seed --use-tensorboard --exp-id "C_dropout_s${seed}" \
    --lambda-unc 0.0 --ignore-warnings &
done
wait
echo "  Group C done."

# --- Group D: PPO + Dropout + Uncertainty Penalty ---
# Set LAMBDA to the calibrated value (run lambda calibration first!)
LAMBDA=${LAMBDA_UNC:-0.1}
echo ""
echo ">>> Group D: PPO + Dropout + Unc Penalty (lambda=$LAMBDA)"
for seed in $SEEDS; do
  echo "  Seed $seed..."
  $VENV $SCRIPT \
    --env "$ENV" --steps $STEPS \
    --eval-interval $EVAL_INTERVAL --eval-n-runs $EVAL_N_RUNS \
    --seed $seed --use-tensorboard --exp-id "D_unc_penalty_s${seed}" \
    --lambda-unc $LAMBDA --ignore-warnings &
done
wait
echo "  Group D done."

echo ""
echo "=== All experiments complete! ==="
echo "View results: tensorboard --logdir data/"
