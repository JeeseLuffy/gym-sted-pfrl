
import logging
import time
import os

from abc import ABCMeta, abstractmethod

from pfrl.experiments import EvaluationHook
from pfrl.experiments import StepHook

class ProgressStepHook(StepHook):
    """
    Hook function that will be called in training to log current progress
    """
    def __init__(self, log_interval=256, logger=None, outdir=None):
        super().__init__()
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)

        self.start_time = time.time()
        self.outdir = outdir
        self.metrics_path = None
        if outdir:
            self.metrics_path = os.path.join(outdir, "train_metrics.tsv")
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                f.write("step\telapsed_seconds\ttime_per_step\taction_mean\taction_mean_window\n")

    def _find_attr(self, env, attr_name):
        """Find attr_name on nested gym wrappers."""
        current = env
        max_depth = 32
        depth = 0
        while current is not None and depth < max_depth:
            if hasattr(current, attr_name):
                return getattr(current, attr_name)
            current = getattr(current, "env", None)
            depth += 1
        return None

    def __call__(self, env, agent, step):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            step: Current timestep.
        """
        if step % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            time_per_step = elapsed / max(1, step)
            action_mean = self._find_attr(env, "last_action_mean")
            action_mean_window = self._find_attr(env, "action_mean_window")
            self.logger.info(
                "elapsed:{:0.2f}min step:{} time-per-step:{:0.2f}s".format(  # NOQA
                    elapsed / 60,
                    step,
                    time_per_step,
                )
            )
            self.logger.info("statistics:%s", agent.get_statistics())
            if action_mean is not None and action_mean_window is not None:
                self.logger.info(
                    "action_mean:%0.6f action_mean_window:%0.6f",
                    float(action_mean),
                    float(action_mean_window),
                )

                if self.metrics_path:
                    with open(self.metrics_path, "a", encoding="utf-8") as f:
                        f.write(
                            "{}\t{:.6f}\t{:.6f}\t{:.8f}\t{:.8f}\n".format(
                                step,
                                elapsed,
                                time_per_step,
                                float(action_mean),
                                float(action_mean_window),
                            )
                        )

                # If wandb is initialized in main.py, log action metrics.
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "train/action_mean": float(action_mean),
                                "train/action_mean_window": float(action_mean_window),
                            },
                            step=step,
                        )
                except Exception:
                    pass

class EvaluationActionHook(EvaluationHook):

    support_train_agent = True
    support_train_agent_batch = True
    support_train_agent_async = False

    def __init__(self):
        super().__init__()

    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            evaluator: Evaluator.
            step: Current timestep. (Not the number of evaluations so far)
            eval_stats (dict): Last evaluation stats from
                pfrl.experiments.evaluator.eval_performance().
            agent_stats (List of pairs): Last agent stats from
                agent.get_statistics().
            env_stats: Last environment stats from
                env.get_statistics().
        """
        pass
        # print(agent_stats)
        # print(env_stats)
        # print("CALLLED")
