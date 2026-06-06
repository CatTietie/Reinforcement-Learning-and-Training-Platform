"""
Optuna超参数调优器：管理study生命周期，执行trial搜索，用最佳参数完成完整训练。
"""

import copy
import logging

import numpy as np
import optuna

from train import Trainer
from tuning.search_space import DEFAULT_SEARCH_SPACE, sample_from_space

logger = logging.getLogger("rl_tuner")


class PPOTuner:
    """
    PPO超参数调优器。
    使用Optuna搜索lr、clip_epsilon、lstm_hidden_size等关键超参，
    自动选出最佳组合并完成最终完整训练。
    """

    def __init__(self, base_config, n_trials=20, search_space=None,
                 db_connection=None, use_distributed=False, num_workers=1):
        self.base_config = base_config
        self.n_trials = n_trials
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.db_connection = db_connection
        self.use_distributed = use_distributed
        self.num_workers = num_workers

        tuning_config = base_config.get("tuning", {})
        self.episodes_per_trial = tuning_config.get("episodes_per_trial", 200)
        self.eval_last_n = tuning_config.get("eval_last_n", 20)
        self.final_training_episodes = tuning_config.get("final_training_episodes", 500)

        self.db = None
        if db_connection:
            from db.database import Database
            self.db = Database(db_connection)

    def _apply_params_to_config(self, config, params):
        """将采样的超参数应用到配置字典中。"""
        algo = config.setdefault("algorithm", {})
        network = config.setdefault("network", {})

        if "lr" in params:
            algo["lr"] = params["lr"]
        if "clip_epsilon" in params:
            algo["clip_epsilon"] = params["clip_epsilon"]
        if "lstm_hidden_size" in params:
            network["lstm_hidden_size"] = params["lstm_hidden_size"]
        if "gamma" in params:
            algo["gamma"] = params["gamma"]
        if "gae_lambda" in params:
            algo["gae_lambda"] = params["gae_lambda"]
        if "entropy_coef" in params:
            algo["entropy_coef"] = params["entropy_coef"]

        return config

    def _get_baseline_reward(self):
        """使用默认配置训练获取基准奖励。"""
        config = copy.deepcopy(self.base_config)
        config["training"]["num_episodes"] = self.episodes_per_trial
        config.setdefault("logging", {})["level"] = "WARNING"
        trainer = Trainer(config=config, experiment_id="baseline")
        result = trainer.train()
        last_n = min(self.eval_last_n, result["total_episodes"])
        return result["avg_final_10"]

    def objective(self, trial):
        """
        单次trial目标函数：采样超参 → 训练 → 返回最后N个episode平均奖励。
        """
        params = sample_from_space(trial, self.search_space)

        config = copy.deepcopy(self.base_config)
        config = self._apply_params_to_config(config, params)
        config["training"]["num_episodes"] = self.episodes_per_trial
        config.setdefault("logging", {})["level"] = "WARNING"

        experiment_id = f"tune_trial_{trial.number}"

        trial_db_id = None
        if self.db:
            study_name = self.base_config.get("experiment", {}).get("name", "ppo_tune")
            trial_db_id = self.db.create_trial(
                study_name=study_name,
                trial_number=trial.number,
                hyperparams_dict=params,
            )

        try:
            if self.use_distributed:
                from distributed.coordinator import DistributedTrainer
                config.setdefault("distributed", {})["num_workers"] = self.num_workers
                trainer = DistributedTrainer(config=config, experiment_id=experiment_id)
            else:
                trainer = Trainer(config=config, experiment_id=experiment_id)

            result = trainer.train()
            objective_value = result["avg_final_10"]

            if self.db and trial_db_id:
                self.db.update_trial(
                    trial_id=trial_db_id,
                    status="completed",
                    objective_value=objective_value,
                )

            logger.info(
                f"Trial {trial.number}: params={params}, "
                f"avg_reward={objective_value:.2f}"
            )
            return objective_value

        except Exception as e:
            if self.db and trial_db_id:
                self.db.update_trial(trial_id=trial_db_id, status="failed")
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("-inf")

    def run(self):
        """
        执行完整调优流程：
        1. 获取默认配置基准奖励
        2. 运行Optuna搜索
        3. 用最佳超参进行完整训练
        4. 验证改进幅度
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Step 1: 基准奖励
        logger.info("Computing baseline reward with default config...")
        baseline_reward = self._get_baseline_reward()
        logger.info(f"Baseline avg reward: {baseline_reward:.2f}")

        # Step 2: Optuna搜索
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Search complete. Best trial: params={best_params}, value={best_value:.2f}")

        # Step 3: 用最佳参数进行完整训练
        logger.info(f"Starting final full training with best params ({self.final_training_episodes} episodes)...")

        final_config = copy.deepcopy(self.base_config)
        final_config = self._apply_params_to_config(final_config, best_params)
        final_config["training"]["num_episodes"] = self.final_training_episodes

        experiment_id = "tune_best_final"

        if self.use_distributed:
            from distributed.coordinator import DistributedTrainer
            final_config.setdefault("distributed", {})["num_workers"] = self.num_workers
            final_trainer = DistributedTrainer(config=final_config, experiment_id=experiment_id)
        else:
            final_trainer = Trainer(config=final_config, experiment_id=experiment_id)

        final_result = final_trainer.train()

        # Step 4: 记录最佳trial到数据库
        if self.db:
            study_name = self.base_config.get("experiment", {}).get("name", "ppo_tune")
            best_trial = self.db.get_best_trial(study_name)
            if best_trial:
                exp_id = self.db.create_experiment(
                    name=f"{study_name}_best_final",
                    config_yaml=str(best_params),
                    training_mode="tune",
                )
                self.db.update_experiment(
                    experiment_id=exp_id,
                    status="finished",
                    total_episodes=final_result["total_episodes"],
                    final_reward=final_result["final_reward"],
                )

        improvement = (
            (final_result["avg_final_10"] - baseline_reward) / max(baseline_reward, 1.0) * 100
        )

        return {
            "best_params": best_params,
            "best_trial_value": best_value,
            "final_training_result": final_result,
            "baseline_reward": baseline_reward,
            "improvement_pct": improvement,
            "n_trials": self.n_trials,
            "study": study,
        }
