"""
超参数调优模块测试：覆盖搜索空间、PPOTuner完整流程和数据库集成。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import optuna

from tuning.search_space import DEFAULT_SEARCH_SPACE, sample_from_space
from tuning.optuna_tuner import PPOTuner
from db.database import Database


TEST_CONFIG = {
    "experiment": {"name": "test_tune", "seed": 42},
    "env": {"name": "CartPoleStandard-v0", "max_steps": 200},
    "network": {
        "hidden_sizes": [32, 32],
        "activation": "tanh",
        "lstm_hidden_size": 64,
        "lstm_num_layers": 1,
        "use_lstm": True,
    },
    "algorithm": {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "update_epochs": 2,
        "max_grad_norm": 0.5,
    },
    "training": {
        "num_episodes": 30,
        "batch_size": 32,
        "log_interval": 10,
        "save_interval": 100,
    },
    "tuning": {
        "episodes_per_trial": 30,
        "eval_last_n": 10,
        "final_training_episodes": 50,
    },
    "storage": {"model_dir": "models"},
    "logging": {"level": "WARNING", "log_dir": "logs"},
}


class TestSearchSpace:
    """测试搜索空间采样。"""

    def test_default_space_has_required_keys(self):
        assert "lr" in DEFAULT_SEARCH_SPACE
        assert "clip_epsilon" in DEFAULT_SEARCH_SPACE
        assert "lstm_hidden_size" in DEFAULT_SEARCH_SPACE

    def test_sample_returns_valid_params(self):
        study = optuna.create_study()
        trial = study.ask()
        params = sample_from_space(trial, DEFAULT_SEARCH_SPACE)

        assert "lr" in params
        assert "clip_epsilon" in params
        assert "lstm_hidden_size" in params

        assert 5e-4 <= params["lr"] <= 5e-3
        assert 0.1 <= params["clip_epsilon"] <= 0.3
        assert params["lstm_hidden_size"] in [64, 128, 256]

    def test_custom_space(self):
        custom_space = {
            "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "gamma": {"type": "float", "low": 0.9, "high": 0.999, "log": False},
        }
        study = optuna.create_study()
        trial = study.ask()
        params = sample_from_space(trial, custom_space)

        assert "lr" in params
        assert "gamma" in params
        assert "clip_epsilon" not in params
        assert 1e-5 <= params["lr"] <= 1e-3
        assert 0.9 <= params["gamma"] <= 0.999

    def test_categorical_type(self):
        space = {
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
        }
        study = optuna.create_study()
        trial = study.ask()
        params = sample_from_space(trial, space)
        assert params["batch_size"] in [16, 32, 64, 128]

    def test_int_type(self):
        space = {
            "update_epochs": {"type": "int", "low": 2, "high": 10, "log": False},
        }
        study = optuna.create_study()
        trial = study.ask()
        params = sample_from_space(trial, space)
        assert 2 <= params["update_epochs"] <= 10
        assert isinstance(params["update_epochs"], int)


class TestPPOTunerObjective:
    """测试PPOTuner的单次trial。"""

    def test_objective_returns_float(self):
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=1,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(tuner.objective, n_trials=1)

        assert len(study.trials) == 1
        assert study.trials[0].value is not None
        assert isinstance(study.trials[0].value, float)

    def test_objective_uses_sampled_params(self):
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=2,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(tuner.objective, n_trials=2)

        params_0 = study.trials[0].params
        assert "lr" in params_0
        assert "clip_epsilon" in params_0
        assert "lstm_hidden_size" in params_0


class TestPPOTunerRun:
    """测试PPOTuner的完整调优流程（含最终完整训练）。"""

    def test_tune_finds_best_params(self):
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=3,
        )
        result = tuner.run()

        assert "best_params" in result
        assert "best_trial_value" in result
        assert "final_training_result" in result
        assert "baseline_reward" in result
        assert "improvement_pct" in result
        assert "lr" in result["best_params"]
        assert "clip_epsilon" in result["best_params"]

    def test_final_training_actually_executes(self):
        """验证最终完整训练确实执行（非提前退出）。"""
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=2,
        )
        result = tuner.run()

        final = result["final_training_result"]
        assert "final_reward" in final
        assert "avg_final_10" in final
        assert "total_episodes" in final
        # 最终训练应使用final_training_episodes
        assert final["total_episodes"] == 50

    def test_baseline_reward_computed(self):
        """验证基准奖励被正确计算。"""
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=2,
        )
        result = tuner.run()

        assert result["baseline_reward"] > 0
        assert isinstance(result["baseline_reward"], float)

    def test_improvement_percentage_reported(self):
        """验证改进百分比被计算并返回。"""
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=2,
        )
        result = tuner.run()

        assert "improvement_pct" in result
        assert isinstance(result["improvement_pct"], float)


class TestTuningWithDatabase:
    """测试调优与数据库集成。"""

    def test_trial_records_created(self):
        db = Database("sqlite:///:memory:")
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=2,
            db_connection="sqlite:///:memory:",
        )
        tuner.db = db

        study = optuna.create_study(direction="maximize")
        study.optimize(tuner.objective, n_trials=2)

        trials = db.list_trials()
        assert len(trials) == 2
        assert all(t.status == "completed" for t in trials)
        assert all(t.objective_value is not None for t in trials)

    def test_get_best_trial(self):
        db = Database("sqlite:///:memory:")
        tuner = PPOTuner(
            base_config=TEST_CONFIG,
            n_trials=3,
            db_connection="sqlite:///:memory:",
        )
        tuner.db = db

        study = optuna.create_study(direction="maximize")
        study.optimize(tuner.objective, n_trials=3)

        best = db.get_best_trial(study_name="test_tune")
        assert best is not None
        assert best.objective_value == max(t.objective_value for t in db.list_trials())


class TestDatabaseTrialCRUD:
    """测试Trial相关的数据库操作。"""

    def test_create_and_query_trial(self):
        db = Database("sqlite:///:memory:")
        trial_id = db.create_trial(
            study_name="test_study",
            trial_number=0,
            hyperparams_dict={"lr": 0.001, "clip_epsilon": 0.2},
        )
        assert trial_id is not None

        trials = db.list_trials(study_name="test_study")
        assert len(trials) == 1
        assert trials[0].trial_number == 0
        assert trials[0].status == "running"

    def test_update_trial(self):
        db = Database("sqlite:///:memory:")
        trial_id = db.create_trial(
            study_name="test_study",
            trial_number=1,
            hyperparams_dict={"lr": 0.005},
        )

        db.update_trial(trial_id, status="completed", objective_value=250.5)

        trials = db.list_trials(study_name="test_study")
        assert trials[0].status == "completed"
        assert trials[0].objective_value == 250.5

    def test_experiment_with_trial_and_worker_count(self):
        db = Database("sqlite:///:memory:")
        trial_id = db.create_trial(
            study_name="test_study",
            trial_number=0,
            hyperparams_dict={"lr": 0.001},
        )

        exp_id = db.create_experiment(
            name="tune_test",
            config_yaml="test: true",
            worker_count=4,
            trial_id=trial_id,
            training_mode="tune",
        )

        exp = db.get_experiment(exp_id)
        assert exp.worker_count == 4
        assert exp.trial_id == trial_id
        assert exp.training_mode == "tune"
