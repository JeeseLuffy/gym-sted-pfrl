#!/usr/bin/env python3
"""
Single-worker Optuna runner for gym-sted-pfrl.

This worker runs one trial at a time and is designed to be launched in
parallel by tmux/ssh or Slurm array jobs.
"""

import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from typing import Dict, List

import optuna


def _read_scores(scores_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(scores_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "steps": float(row["steps"]),
                    "mean": float(row["mean"]),
                    "median": float(row["median"]),
                    "stdev": float(row["stdev"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows in scores file: {scores_path}")
    return rows


def _read_train_metrics(metrics_path: str) -> List[Dict[str, float]]:
    if not os.path.exists(metrics_path):
        return []
    rows: List[Dict[str, float]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "step": float(row["step"]),
                    "action_mean": float(row["action_mean"]),
                    "action_mean_window": float(row["action_mean_window"]),
                }
            )
    return rows


def _is_action_collapsed(
    metrics_path: str,
    threshold: float,
    min_step: int,
    consecutive: int,
) -> bool:
    rows = _read_train_metrics(metrics_path)
    rows = [r for r in rows if r["step"] >= min_step]
    if len(rows) < consecutive:
        return False
    tail = rows[-consecutive:]
    return all(abs(r["action_mean_window"]) <= threshold for r in tail)


def _build_command(args, trial: optuna.Trial, exp_id: str) -> List[str]:
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    batchsize = trial.suggest_categorical("batchsize", [8, 16, 32])
    update_interval = trial.suggest_categorical("update_interval", [256, 512, 1024])
    gamma = trial.suggest_float("gamma", args.gamma_min, args.gamma_max)

    if args.lambda_unc_fixed is not None:
        lambda_unc = args.lambda_unc_fixed
    elif args.search_lambda_unc:
        lambda_unc = trial.suggest_float(
            "lambda_unc", args.lambda_unc_min, args.lambda_unc_max, log=True
        )
    else:
        lambda_unc = 0.0
    trial.set_user_attr("lambda_unc", float(lambda_unc))

    unc_n_samples = trial.suggest_categorical("unc_n_samples", [10, 20, 30])
    train_seed = args.seed_base + trial.number
    cmd = [
        sys.executable,
        "main.py",
        "--env",
        args.env,
        "--steps",
        str(args.steps),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-n-runs",
        str(args.eval_n_runs),
        "--log-interval",
        str(args.log_interval),
        "--eval-seed",
        str(args.eval_seed),
        "--seed",
        str(train_seed),
        "--outdir",
        args.outdir,
        "--exp-id",
        exp_id,
        "--no-exp-suffix",
        "--lr",
        str(lr),
        "--batchsize",
        str(batchsize),
        "--update-interval",
        str(update_interval),
        "--gamma",
        str(gamma),
        "--lambda-unc",
        str(lambda_unc),
        "--unc-n-samples",
        str(unc_n_samples),
        "--ignore-warnings",
    ]
    if args.use_tensorboard:
        cmd.append("--use-tensorboard")
    if args.no_dropout:
        cmd.append("--no-dropout")
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    if args.wandb_project:
        cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        cmd.extend(["--wandb-group", args.wandb_group or args.study_name])
        cmd.extend(["--wandb-tags", args.wandb_tags])
    return cmd


def _run_training(cmd: List[str], log_path: str, run_dir: str, args) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "train_metrics.tsv")

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("[CMD] {}\n".format(" ".join(shlex.quote(x) for x in cmd)))
        logf.flush()

        process = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=args.workdir,
        )
        while process.poll() is None:
            time.sleep(args.poll_seconds)
            if args.prune_on_action_collapse and _is_action_collapsed(
                metrics_path=metrics_path,
                threshold=args.action_collapse_threshold,
                min_step=args.action_collapse_min_step,
                consecutive=args.action_collapse_consecutive,
            ):
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise optuna.TrialPruned(
                    "Pruned due to action collapse: "
                    f"|action_mean_window| <= {args.action_collapse_threshold} "
                    f"for {args.action_collapse_consecutive} logs."
                )

        if process.returncode != 0:
            raise RuntimeError(f"Training failed with code {process.returncode}. See {log_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--pruner-startup-trials", type=int, default=8)
    parser.add_argument("--env", default="ContextualMOSTED-easy-hslb-v0")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-n-runs", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--outdir", default="./data_optuna")
    parser.add_argument("--log-dir", default="./optuna_logs")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--exp-prefix", default="optuna")
    parser.add_argument("--use-tensorboard", action="store_true", default=True)
    parser.add_argument("--no-dropout", action="store_true", default=False)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="optuna,sted")
    parser.add_argument("--search-lambda-unc", action="store_true", default=True)
    parser.add_argument("--lambda-unc-fixed", type=float, default=None)
    parser.add_argument("--lambda-unc-min", type=float, default=0.005)
    parser.add_argument("--lambda-unc-max", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--lr-max", type=float, default=5e-4)
    parser.add_argument("--gamma-min", type=float, default=0.97)
    parser.add_argument("--gamma-max", type=float, default=0.999)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--prune-on-action-collapse", action="store_true", default=True)
    parser.add_argument("--action-collapse-threshold", type=float, default=0.03)
    parser.add_argument("--action-collapse-min-step", type=int, default=2000)
    parser.add_argument("--action-collapse-consecutive", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.eval_interval,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        exp_id = "{}_w{}_t{}_{}".format(
            args.exp_prefix, args.worker_id, trial.number, uuid.uuid4().hex[:6]
        )
        run_dir = os.path.abspath(os.path.join(args.outdir, exp_id))
        log_path = os.path.join(args.log_dir, f"{exp_id}.log")

        cmd = _build_command(args, trial, exp_id)
        trial.set_user_attr("exp_id", exp_id)
        trial.set_user_attr("run_dir", run_dir)
        trial.set_user_attr("log_path", os.path.abspath(log_path))
        trial.set_user_attr("command", " ".join(shlex.quote(x) for x in cmd))

        _run_training(cmd, log_path, run_dir, args)

        scores_path = os.path.join(run_dir, "scores.txt")
        rows = _read_scores(scores_path)
        for row in rows:
            trial.report(row["mean"], int(row["steps"]))
            if trial.should_prune():
                raise optuna.TrialPruned("Pruned by Optuna pruner.")

        last = rows[-1]
        trial.set_user_attr("eval_mean", float(last["mean"]))
        trial.set_user_attr("eval_median", float(last["median"]))
        trial.set_user_attr("eval_stdev", float(last["stdev"]))
        trial.set_user_attr("final_step", int(last["steps"]))
        return float(last["mean"])

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    best = study.best_trial
    print("Best trial #{} value={:.6f}".format(best.number, best.value))
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
