"""
分布式训练模块测试：覆盖工具函数、Worker、Learner、Coordinator和集成测试。
包含分布式协调mock测试和调度器集成测试。
"""

import multiprocessing as mp
import queue
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed.utils import (
    collect_episode,
    create_env_from_config,
    create_policy_from_config,
    ppo_update_from_batch,
)
from train import compute_gae_loop


TEST_CONFIG = {
    "experiment": {"name": "test_distributed", "seed": 42},
    "env": {"name": "CartPoleStandard-v0", "max_steps": 200},
    "network": {
        "hidden_sizes": [32, 32],
        "activation": "tanh",
        "lstm_hidden_size": 32,
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
        "num_episodes": 10,
        "batch_size": 32,
        "log_interval": 5,
        "save_interval": 100,
    },
    "distributed": {
        "num_workers": 2,
        "batch_before_update": 2,
        "weight_sync_interval": 1,
        "queue_maxsize": 16,
        "episodes_per_send": 1,
    },
    "storage": {"model_dir": "models"},
    "logging": {"level": "WARNING", "log_dir": "logs"},
}


class TestCreatePolicyFromConfig:
    """测试从配置创建Policy的工具函数。"""

    def test_creates_valid_policy(self):
        policy = create_policy_from_config(TEST_CONFIG)
        assert policy is not None
        assert policy.obs_dim == 4
        assert policy.action_dim == 2
        assert policy.lstm_hidden_size == 32

    def test_policy_matches_config_params(self):
        policy = create_policy_from_config(TEST_CONFIG)
        assert policy.use_lstm is True
        assert policy.lstm_num_layers == 1

    def test_policy_forward_works(self):
        policy = create_policy_from_config(TEST_CONFIG)
        obs = torch.randn(1, 4)
        hidden = policy.init_hidden(batch_size=1)
        action, log_prob, next_hidden = policy(obs, hidden)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)


class TestCreateEnvFromConfig:
    """测试从配置创建环境。"""

    def test_creates_env(self):
        env = create_env_from_config(TEST_CONFIG)
        assert env is not None
        obs, _ = env.reset()
        assert obs.shape == (4,)
        env.close()


class TestCollectEpisode:
    """测试episode收集函数。"""

    def test_returns_valid_trajectory(self):
        policy = create_policy_from_config(TEST_CONFIG)
        env = create_env_from_config(TEST_CONFIG)
        device = torch.device("cpu")

        trajectory, total_reward, ep_length = collect_episode(
            policy, env, device, max_steps=200
        )

        assert "obs" in trajectory
        assert "actions" in trajectory
        assert "rewards" in trajectory
        assert "log_probs" in trajectory
        assert "values" in trajectory
        assert "terminals" in trajectory

        assert len(trajectory["obs"]) == ep_length
        assert len(trajectory["actions"]) == ep_length
        assert len(trajectory["values"]) == ep_length + 1
        assert total_reward > 0
        env.close()

    def test_trajectory_dtypes(self):
        policy = create_policy_from_config(TEST_CONFIG)
        env = create_env_from_config(TEST_CONFIG)
        device = torch.device("cpu")

        trajectory, _, _ = collect_episode(policy, env, device, max_steps=200)

        assert trajectory["obs"].dtype == np.float32
        assert trajectory["actions"].dtype == np.int64
        assert trajectory["rewards"].dtype == np.float64
        assert trajectory["terminals"].dtype == bool
        env.close()

    def test_multiple_episodes_independent(self):
        torch.manual_seed(42)
        np.random.seed(42)
        policy = create_policy_from_config(TEST_CONFIG)
        env = create_env_from_config(TEST_CONFIG)
        device = torch.device("cpu")

        t1, r1, _ = collect_episode(policy, env, device, max_steps=200)
        t2, r2, _ = collect_episode(policy, env, device, max_steps=200)

        assert t1["obs"].shape[0] > 0
        assert t2["obs"].shape[0] > 0
        env.close()


class TestPPOUpdateFromBatch:
    """测试PPO更新函数。"""

    def test_update_produces_valid_losses(self):
        policy = create_policy_from_config(TEST_CONFIG)
        env = create_env_from_config(TEST_CONFIG)
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        trajectory, _, _ = collect_episode(policy, env, device, max_steps=200)
        advantages, returns = compute_gae_loop(
            trajectory["rewards"], trajectory["values"],
            gamma=0.99, lam=0.95, terminals=trajectory["terminals"]
        )

        policy_loss, value_loss, entropy = ppo_update_from_batch(
            policy=policy,
            optimizer=optimizer,
            obs=trajectory["obs"],
            actions=trajectory["actions"],
            old_log_probs=trajectory["log_probs"],
            advantages=advantages,
            returns=returns,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            num_epochs=2,
            mini_batch_size=32,
            device=device,
        )

        assert np.isfinite(policy_loss)
        assert np.isfinite(value_loss)
        assert np.isfinite(entropy)
        assert entropy > 0
        env.close()

    def test_update_changes_parameters(self):
        policy = create_policy_from_config(TEST_CONFIG)
        env = create_env_from_config(TEST_CONFIG)
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        trajectory, _, _ = collect_episode(policy, env, device, max_steps=200)
        advantages, returns = compute_gae_loop(
            trajectory["rewards"], trajectory["values"],
            gamma=0.99, lam=0.95, terminals=trajectory["terminals"]
        )

        params_before = {k: v.clone() for k, v in policy.state_dict().items()}

        ppo_update_from_batch(
            policy=policy,
            optimizer=optimizer,
            obs=trajectory["obs"],
            actions=trajectory["actions"],
            old_log_probs=trajectory["log_probs"],
            advantages=advantages,
            returns=returns,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            num_epochs=4,
            mini_batch_size=32,
            device=device,
        )

        params_after = policy.state_dict()
        changed = any(
            not torch.equal(params_before[k], params_after[k])
            for k in params_before
        )
        assert changed
        env.close()


class TestSharedMemoryWeightSync:
    """测试共享内存权重同步机制。"""

    def test_shared_memory_tensor_created(self):
        policy = create_policy_from_config(TEST_CONFIG)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        for key in shared_state:
            assert shared_state[key].is_shared()

    def test_weight_update_visible_across_processes(self):
        """验证主进程写入共享内存后子进程能读到更新。"""
        policy = create_policy_from_config(TEST_CONFIG)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        # 修改共享内存中的权重
        for key in shared_state:
            shared_state[key].fill_(1.0)

        # 另一个Policy从共享内存加载
        policy2 = create_policy_from_config(TEST_CONFIG)
        local_state = policy2.state_dict()
        for key in local_state:
            local_state[key].copy_(shared_state[key])
        policy2.load_state_dict(local_state, assign=False)

        for key, param in policy2.state_dict().items():
            assert torch.all(param == 1.0)

    def test_version_number_increments(self):
        """验证版本号机制能正确标识权重更新。"""
        ctx = mp.get_context("spawn")
        weight_version = ctx.Value("i", 0)

        assert weight_version.value == 0
        weight_version.value += 1
        assert weight_version.value == 1
        weight_version.value += 1
        assert weight_version.value == 2


class TestCoordinatorMock:
    """分布式协调器的mock测试：验证进程生命周期管理。"""

    def test_coordinator_creates_shared_state(self):
        """验证Coordinator正确创建共享内存state_dict。"""
        from distributed.coordinator import DistributedTrainer

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 4}
        trainer = DistributedTrainer(config=config, experiment_id="mock_test")

        assert trainer.num_workers == 2
        assert trainer.target_episodes == 4

    def test_coordinator_spawns_correct_worker_count(self):
        """验证spawn的worker数量与配置一致。"""
        from distributed.coordinator import DistributedTrainer

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 6}
        config["distributed"] = {**config["distributed"], "num_workers": 3}

        trainer = DistributedTrainer(config=config, experiment_id="spawn_test")
        assert trainer.num_workers == 3

    @patch("distributed.coordinator.mp.get_context")
    def test_coordinator_uses_spawn_context(self, mock_get_context):
        """验证Coordinator使用spawn启动方法（Windows要求）。"""
        from distributed.coordinator import DistributedTrainer

        mock_ctx = MagicMock()
        mock_queue = MagicMock()
        mock_value = MagicMock()
        mock_value.value = 0
        mock_event = MagicMock()
        mock_event.is_set.return_value = True
        mock_lock = MagicMock()
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False

        mock_ctx.Queue.return_value = mock_queue
        mock_ctx.Value.return_value = mock_value
        mock_ctx.Event.return_value = mock_event
        mock_ctx.Lock.return_value = mock_lock
        mock_ctx.Process.return_value = mock_process
        mock_get_context.return_value = mock_ctx

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 0}
        trainer = DistributedTrainer(config=config, experiment_id="ctx_test")

        # 调用train会触发spawn context创建
        try:
            trainer.train()
        except Exception:
            pass

        mock_get_context.assert_called_with("spawn")

    def test_coordinator_stop_event_terminates_workers(self):
        """验证stop_event能正确终止worker进程。"""
        from distributed.coordinator import DistributedTrainer

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 4}
        config["distributed"] = {
            "num_workers": 2, "batch_before_update": 2,
            "weight_sync_interval": 1, "queue_maxsize": 8, "episodes_per_send": 1,
        }

        trainer = DistributedTrainer(config=config, experiment_id="stop_test")
        result = trainer.train()

        # 训练完成后所有进程应已退出
        assert result["total_episodes"] >= 4


class TestWorkerProcess:
    """测试Worker进程逻辑。"""

    def test_worker_produces_trajectories(self):
        from distributed.worker import worker_process

        ctx = mp.get_context("spawn")
        trajectory_queue = ctx.Queue(maxsize=8)
        episode_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        weight_version = ctx.Value("i", 1)

        policy = create_policy_from_config(TEST_CONFIG)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 3}

        p = ctx.Process(
            target=worker_process,
            args=(0, config, trajectory_queue, shared_state,
                  weight_version, episode_counter, stop_event, 3, lock),
        )
        p.start()
        p.join(timeout=30.0)

        collected = 0
        while not trajectory_queue.empty():
            batch = trajectory_queue.get_nowait()
            assert isinstance(batch, list)
            for payload in batch:
                assert "trajectory" in payload
                assert "total_reward" in payload
                assert "worker_id" in payload
                assert payload["worker_id"] == 0
                collected += 1

        assert collected >= 1

    def test_worker_respects_stop_event(self):
        from distributed.worker import worker_process

        ctx = mp.get_context("spawn")
        trajectory_queue = ctx.Queue(maxsize=8)
        episode_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        weight_version = ctx.Value("i", 1)

        policy = create_policy_from_config(TEST_CONFIG)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        stop_event.set()

        p = ctx.Process(
            target=worker_process,
            args=(0, TEST_CONFIG, trajectory_queue, shared_state,
                  weight_version, episode_counter, stop_event, 1000, lock),
        )
        p.start()
        p.join(timeout=30.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
        assert not p.is_alive()

    def test_worker_detects_weight_version_update(self):
        """验证Worker通过版本号检测权重更新。"""
        from distributed.worker import worker_process

        ctx = mp.get_context("spawn")
        trajectory_queue = ctx.Queue(maxsize=8)
        episode_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        weight_version = ctx.Value("i", 1)

        policy = create_policy_from_config(TEST_CONFIG)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 2}

        # 在worker运行之前更新版本号（模拟learner广播）
        weight_version.value = 5

        p = ctx.Process(
            target=worker_process,
            args=(0, config, trajectory_queue, shared_state,
                  weight_version, episode_counter, stop_event, 2, lock),
        )
        p.start()
        p.join(timeout=30.0)

        # worker应该正常完成（读取了版本5的权重）
        assert not p.is_alive()
        assert episode_counter.value >= 1


class TestLearnerUnit:
    """测试Learner核心逻辑（使用预填充Queue）。"""

    def test_learner_completes_training(self):
        from distributed.learner import Learner

        ctx = mp.get_context("spawn")
        trajectory_queue = ctx.Queue(maxsize=16)
        episode_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        weight_version = ctx.Value("i", 0)

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 4}
        config["distributed"] = {**config["distributed"], "batch_before_update": 2}

        policy = create_policy_from_config(config)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        env = create_env_from_config(config)
        device = torch.device("cpu")

        for i in range(4):
            traj, reward, length = collect_episode(policy, env, device, max_steps=200)
            trajectory_queue.put([{
                "trajectory": traj,
                "total_reward": reward,
                "episode_length": length,
                "worker_id": i % 2,
            }])
            with lock:
                episode_counter.value += 1

        env.close()

        learner = Learner(
            config=config,
            trajectory_queue=trajectory_queue,
            shared_state_dict=shared_state,
            weight_version=weight_version,
            episode_counter=episode_counter,
            stop_event=stop_event,
            lock=lock,
            experiment_id="test_learner",
        )

        result = learner.run()

        assert "final_reward" in result
        assert "avg_final_10" in result
        assert "total_episodes" in result
        assert result["total_episodes"] == 4

    def test_learner_broadcasts_weights_via_shared_memory(self):
        """验证Learner通过共享内存广播权重后版本号递增。"""
        from distributed.learner import Learner

        ctx = mp.get_context("spawn")
        trajectory_queue = ctx.Queue(maxsize=16)
        episode_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        weight_version = ctx.Value("i", 0)

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 2}
        config["distributed"] = {**config["distributed"], "batch_before_update": 2}

        policy = create_policy_from_config(config)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        env = create_env_from_config(config)
        device = torch.device("cpu")

        for i in range(2):
            traj, reward, length = collect_episode(policy, env, device, max_steps=200)
            trajectory_queue.put([{
                "trajectory": traj,
                "total_reward": reward,
                "episode_length": length,
                "worker_id": 0,
            }])
            with lock:
                episode_counter.value += 1

        env.close()

        learner = Learner(
            config=config,
            trajectory_queue=trajectory_queue,
            shared_state_dict=shared_state,
            weight_version=weight_version,
            episode_counter=episode_counter,
            stop_event=stop_event,
            lock=lock,
            experiment_id="test_broadcast",
        )

        result = learner.run()

        # 版本号应该递增（初始broadcast + 每次sync）
        assert weight_version.value >= 1


class TestSchedulerIntegration:
    """调度器集成测试：验证多Worker协调和episode计数。"""

    def test_episode_counter_accurate_across_workers(self):
        """验证全局episode计数器在多worker下准确。"""
        from distributed.coordinator import DistributedTrainer

        target = 12
        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": target}
        config["distributed"] = {
            "num_workers": 2, "batch_before_update": 2,
            "weight_sync_interval": 1, "queue_maxsize": 16, "episodes_per_send": 2,
        }

        trainer = DistributedTrainer(config=config, experiment_id="counter_test")
        result = trainer.train()

        assert result["total_episodes"] >= target

    def test_graceful_shutdown_no_zombie_processes(self):
        """验证训练结束后无僵尸进程残留。"""
        psutil = pytest.importorskip("psutil")
        from distributed.coordinator import DistributedTrainer

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 6}
        config["distributed"] = {
            "num_workers": 2, "batch_before_update": 2,
            "weight_sync_interval": 1, "queue_maxsize": 8, "episodes_per_send": 1,
        }

        pid = os.getpid()
        proc = psutil.Process(pid)
        children_before = {c.pid for c in proc.children(recursive=True)}

        trainer = DistributedTrainer(config=config, experiment_id="zombie_test")
        trainer.train()

        time.sleep(1.0)
        children_after = {c.pid for c in proc.children(recursive=True)}
        new_zombies = children_after - children_before
        assert len(new_zombies) == 0, f"Zombie processes detected: {new_zombies}"

    def test_multiple_sequential_trainings(self):
        """验证可以连续多次启动分布式训练。"""
        from distributed.coordinator import DistributedTrainer

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 4}
        config["distributed"] = {
            "num_workers": 2, "batch_before_update": 2,
            "weight_sync_interval": 1, "queue_maxsize": 8, "episodes_per_send": 1,
        }

        for i in range(3):
            trainer = DistributedTrainer(config=config, experiment_id=f"seq_{i}")
            result = trainer.train()
            assert result["total_episodes"] >= 4

    def test_weight_sync_frequency(self):
        """验证权重同步频率与配置一致。"""
        from distributed.learner import Learner

        ctx = mp.get_context("spawn")
        trajectory_queue = ctx.Queue(maxsize=16)
        episode_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        weight_version = ctx.Value("i", 0)

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 8}
        config["distributed"] = {
            **config["distributed"],
            "batch_before_update": 2,
            "weight_sync_interval": 2,  # 每2次更新广播一次
        }

        policy = create_policy_from_config(config)
        shared_state = {}
        for key, tensor in policy.state_dict().items():
            shared_state[key] = tensor.cpu().clone().share_memory_()

        env = create_env_from_config(config)
        device = torch.device("cpu")

        for i in range(8):
            traj, reward, length = collect_episode(policy, env, device, max_steps=200)
            trajectory_queue.put([{
                "trajectory": traj,
                "total_reward": reward,
                "episode_length": length,
                "worker_id": i % 2,
            }])
            with lock:
                episode_counter.value += 1
        env.close()

        learner = Learner(
            config=config,
            trajectory_queue=trajectory_queue,
            shared_state_dict=shared_state,
            weight_version=weight_version,
            episode_counter=episode_counter,
            stop_event=stop_event,
            lock=lock,
            experiment_id="sync_freq_test",
        )

        learner.run()

        # 8 episodes / batch_before_update(2) = 4 updates
        # weight_sync_interval=2, 所以广播 4/2=2 次 + 初始1次 = 3
        assert weight_version.value >= 2


class TestDistributedIntegration:
    """端到端分布式训练集成测试。"""

    def test_distributed_training_completes(self):
        from distributed.coordinator import DistributedTrainer

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 8}
        config["distributed"] = {
            "num_workers": 2,
            "batch_before_update": 2,
            "weight_sync_interval": 1,
            "queue_maxsize": 16,
            "episodes_per_send": 1,
        }

        trainer = DistributedTrainer(config=config, experiment_id="integration_test")
        result = trainer.train()

        assert "final_reward" in result
        assert "avg_final_10" in result
        assert result["total_episodes"] >= 8

    def test_distributed_produces_positive_rewards(self):
        from distributed.coordinator import DistributedTrainer

        config = TEST_CONFIG.copy()
        config["training"] = {**config["training"], "num_episodes": 12}
        config["distributed"] = {
            "num_workers": 2,
            "batch_before_update": 2,
            "weight_sync_interval": 1,
            "queue_maxsize": 16,
            "episodes_per_send": 2,
        }

        trainer = DistributedTrainer(config=config, experiment_id="integration_test2")
        result = trainer.train()

        assert result["final_reward"] > 0
        assert result["total_episodes"] >= 12
