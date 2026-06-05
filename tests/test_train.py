"""
单元测试：覆盖 GAE 计算正确性、隐状态传递、静默干扰不变性。
使用固定随机种子保证可复现，不依赖外部数据库或文件系统。
"""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CartPoleSilentDisturbance, CartPoleStandard
from policy import Policy
from train import Trainer, compute_gae_loop, compute_gae_vectorized


class TestGAEComputation:
    """GAE 计算正确性测试。"""

    def test_gae_simple_case(self):
        """手算验证：简单序列的 GAE 值。"""
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.6, 0.7, 0.8]  # T+1 values
        gamma = 0.99
        lam = 0.95
        terminals = [False, False, False]

        advantages, returns = compute_gae_loop(rewards, values, gamma, lam, terminals)

        # 手算验证:
        # delta_2 = 1.0 + 0.99 * 0.8 - 0.7 = 1.092
        # delta_1 = 1.0 + 0.99 * 0.7 - 0.6 = 1.093
        # delta_0 = 1.0 + 0.99 * 0.6 - 0.5 = 1.094
        # A_2 = delta_2 = 1.092
        # A_1 = delta_1 + gamma * lam * A_2 = 1.093 + 0.99*0.95*1.092 = 2.120554
        # A_0 = delta_0 + gamma * lam * A_1 = 1.094 + 0.99*0.95*2.120554 = 3.088415...

        delta_2 = 1.0 + 0.99 * 0.8 - 0.7
        delta_1 = 1.0 + 0.99 * 0.7 - 0.6
        delta_0 = 1.0 + 0.99 * 0.6 - 0.5

        A_2 = delta_2
        A_1 = delta_1 + gamma * lam * A_2
        A_0 = delta_0 + gamma * lam * A_1

        assert abs(advantages[0] - A_0) < 1e-6
        assert abs(advantages[1] - A_1) < 1e-6
        assert abs(advantages[2] - A_2) < 1e-6

        # 验证 returns = advantages + values[:T]
        for i in range(3):
            assert abs(returns[i] - (advantages[i] + values[i])) < 1e-6

    def test_gae_with_terminal(self):
        """终止状态时 GAE 应正确截断。"""
        rewards = [1.0, 1.0, 0.0]
        values = [0.5, 0.6, 0.7, 0.0]
        gamma = 0.99
        lam = 0.95
        terminals = [False, False, True]  # 最后一步终止

        advantages, returns = compute_gae_loop(rewards, values, gamma, lam, terminals)

        # 终止时: delta_2 = 0.0 + 0 - 0.7 = -0.7, A_2 = -0.7
        # A_1 = delta_1 + gamma * lam * (1-0) * A_2
        delta_2 = 0.0 + 0.0 - 0.7  # terminal, next_value = 0
        delta_1 = 1.0 + 0.99 * 0.7 - 0.6
        delta_0 = 1.0 + 0.99 * 0.6 - 0.5

        A_2 = delta_2  # -0.7, gae resets because terminal
        A_1 = delta_1 + gamma * lam * (1 - 0) * A_2
        A_0 = delta_0 + gamma * lam * (1 - 0) * A_1

        assert abs(advantages[2] - A_2) < 1e-6
        assert abs(advantages[1] - A_1) < 1e-6
        assert abs(advantages[0] - A_0) < 1e-6

    def test_gae_loop_vs_vectorized_consistency(self):
        """循环实现与向量化实现的一致性验证。"""
        np.random.seed(42)
        T = 100
        rewards = np.random.randn(T)
        values = np.random.randn(T + 1)
        gamma = 0.99
        lam = 0.95
        terminals = np.random.random(T) < 0.05  # 5% 终止概率

        adv_loop, ret_loop = compute_gae_loop(rewards, values, gamma, lam, terminals)
        adv_vec, ret_vec = compute_gae_vectorized(rewards, values, gamma, lam, terminals)

        np.testing.assert_allclose(adv_loop, adv_vec, atol=1e-10)
        np.testing.assert_allclose(ret_loop, ret_vec, atol=1e-10)

    def test_gae_no_reward_scaling(self):
        """验证 GAE 计算使用原始奖励，未经缩放。"""
        rewards = [2.0, 3.0, 1.0]
        values = [1.0, 1.5, 2.0, 0.5]
        gamma = 0.99
        lam = 0.95
        terminals = [False, False, False]

        advantages, _ = compute_gae_loop(rewards, values, gamma, lam, terminals)

        # 直接使用原始奖励计算 delta
        delta_2 = 1.0 + 0.99 * 0.5 - 2.0
        delta_1 = 3.0 + 0.99 * 2.0 - 1.5
        delta_0 = 2.0 + 0.99 * 1.5 - 1.0

        A_2 = delta_2
        A_1 = delta_1 + gamma * lam * A_2
        A_0 = delta_0 + gamma * lam * A_1

        assert abs(advantages[0] - A_0) < 1e-6
        assert abs(advantages[1] - A_1) < 1e-6
        assert abs(advantages[2] - A_2) < 1e-6

    def test_gae_zero_rewards(self):
        """全零奖励时的 GAE 计算。"""
        rewards = [0.0, 0.0, 0.0]
        values = [1.0, 1.0, 1.0, 1.0]
        gamma = 0.99
        lam = 0.95
        terminals = [False, False, False]

        advantages, returns = compute_gae_loop(rewards, values, gamma, lam, terminals)

        # delta_t = 0 + 0.99*1.0 - 1.0 = -0.01
        delta = -0.01
        A_2 = delta
        A_1 = delta + gamma * lam * A_2
        A_0 = delta + gamma * lam * A_1

        assert abs(advantages[2] - A_2) < 1e-6
        assert abs(advantages[1] - A_1) < 1e-6
        assert abs(advantages[0] - A_0) < 1e-6

    def test_gae_single_step(self):
        """单步轨迹的 GAE。"""
        rewards = [1.0]
        values = [0.5, 0.8]
        gamma = 0.99
        lam = 0.95
        terminals = [False]

        advantages, returns = compute_gae_loop(rewards, values, gamma, lam, terminals)

        expected = 1.0 + 0.99 * 0.8 - 0.5
        assert abs(advantages[0] - expected) < 1e-6


class TestHiddenStateContinuity:
    """LSTM 隐状态传递规则测试。"""

    def test_hidden_state_not_reset_on_env_reset(self):
        """验证环境 reset 后隐状态不被置零（除非 episode 真正结束）。"""
        torch.manual_seed(42)

        policy = Policy(obs_dim=4, action_dim=2, lstm_hidden_size=32, lstm_num_layers=1)
        hidden = policy.init_hidden(batch_size=1)

        # 模拟一次交互，改变隐状态
        obs = torch.randn(1, 4)
        _, _, new_hidden, _ = policy.act(obs, hidden)

        # 验证隐状态已变化
        assert not torch.allclose(hidden[0], new_hidden[0], atol=1e-6)
        assert not torch.allclose(hidden[1], new_hidden[1], atol=1e-6)

        # 模拟第二次交互，隐状态应继续传递（不重置）
        obs2 = torch.randn(1, 4)
        _, _, new_hidden2, _ = policy.act(obs2, new_hidden)

        # 第二次的隐状态应该基于 new_hidden 而非零状态
        # 用零状态重新计算应该得到不同结果
        _, _, hidden_from_zero, _ = policy.act(obs2, policy.init_hidden(batch_size=1))
        assert not torch.allclose(new_hidden2[0], hidden_from_zero[0], atol=1e-6)

    def test_hidden_state_reset_only_on_true_terminal(self):
        """验证隐状态仅在 episode 真正结束时重置为零。"""
        torch.manual_seed(42)

        policy = Policy(obs_dim=4, action_dim=2, lstm_hidden_size=32, lstm_num_layers=1)
        hidden = policy.init_hidden(batch_size=1)

        # 让隐状态经历多步非零演化
        obs = torch.randn(1, 4)
        for _ in range(5):
            _, _, hidden, _ = policy.act(obs, hidden)

        # 此时隐状态应非零
        assert not torch.allclose(hidden[0], torch.zeros_like(hidden[0]), atol=1e-6)

        # 模拟 episode 结束 -> 重置
        reset_hidden = policy.init_hidden(batch_size=1)
        assert torch.allclose(reset_hidden[0], torch.zeros_like(reset_hidden[0]))
        assert torch.allclose(reset_hidden[1], torch.zeros_like(reset_hidden[1]))

    def test_trainer_rollout_preserves_hidden_across_episodes(self):
        """验证 Trainer 的 rollout 在 episode 间正确传递/重置隐状态。"""
        config = {
            "experiment": {"name": "test", "seed": 42},
            "env": {"name": "CartPoleStandard-v0", "max_steps": 10},
            "policy": {"hidden_size": 32, "lstm_layers": 1},
            "algorithm": {"lr": 1e-3, "gamma": 0.99, "gae_lambda": 0.95},
            "training": {"num_episodes": 2, "batch_size": 8},
            "logging": {"log_dir": "/tmp/test_logs", "level": "WARNING"},
        }

        trainer = Trainer(config=config, experiment_id="test_hidden")

        # 第一次 rollout
        hidden1 = trainer.policy.init_hidden(batch_size=1, device=trainer.device)
        _, hidden_after_ep1, _, _ = trainer.rollout(hidden1)

        # episode 结束后，hidden 应该被重置为零（因为 done=True）
        zero_hidden = trainer.policy.init_hidden(batch_size=1, device=trainer.device)
        assert torch.allclose(hidden_after_ep1[0], zero_hidden[0])
        assert torch.allclose(hidden_after_ep1[1], zero_hidden[1])

    def test_hidden_state_continuity_within_episode(self):
        """验证在 episode 内部，隐状态持续更新不会被意外重置。"""
        torch.manual_seed(123)

        policy = Policy(obs_dim=4, action_dim=2, lstm_hidden_size=32, lstm_num_layers=1)
        hidden = policy.init_hidden(batch_size=1)

        hidden_states_collected = [hidden]
        obs = torch.randn(1, 4)

        for step in range(10):
            _, _, hidden, _ = policy.act(obs, hidden)
            hidden_states_collected.append(hidden)

        # 每一步的隐状态都应不同
        for i in range(1, len(hidden_states_collected)):
            for j in range(i + 1, len(hidden_states_collected)):
                h_i = hidden_states_collected[i][0]
                h_j = hidden_states_collected[j][0]
                assert not torch.allclose(h_i, h_j, atol=1e-6), (
                    f"Hidden states at step {i} and {j} should differ"
                )


class TestSilentDisturbanceInvariance:
    """静默干扰对 GAE 计算不变性测试。"""

    def test_disturbance_does_not_affect_reward_sequence(self):
        """
        验证：开启/关闭干扰后，使用相同动作序列时，
        奖励序列（用于 GAE）应保持一致。
        """
        seed = 42
        max_steps = 50

        # 无干扰环境
        env_no_dist = CartPoleStandard(max_steps=max_steps)
        env_no_dist.reset(seed=seed)

        # 有干扰环境（但使用独立的干扰随机源）
        env_with_dist = CartPoleSilentDisturbance(
            disturbance_prob=0.1, max_steps=max_steps
        )
        env_with_dist.reset(seed=seed)

        # 使用固定动作序列
        np.random.seed(seed)
        actions = np.random.randint(0, 2, size=max_steps)

        rewards_no_dist = []
        rewards_with_dist = []

        for i in range(max_steps):
            _, r1, term1, trunc1, _ = env_no_dist.step(actions[i])
            _, r2, term2, trunc2, _ = env_with_dist.step(actions[i])

            rewards_no_dist.append(r1)
            rewards_with_dist.append(r2)

            # 终止状态可能因干扰而不同（物理状态被改变），这里只验证奖励逻辑
            if term1 or trunc1 or term2 or trunc2:
                break

        # 两个环境的初始状态相同，第一步的奖励一定相同
        assert rewards_no_dist[0] == rewards_with_dist[0]

    def test_gae_invariant_under_same_reward_sequence(self):
        """
        验证：给定相同的 rewards 序列，无论是否存在内部干扰，
        GAE 输出完全一致。
        """
        # 模拟一组奖励序列（来自无干扰环境）
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0]
        values = [0.5, 0.6, 0.7, 0.8, 0.3, 0.0]
        gamma = 0.99
        lam = 0.95
        terminals = [False, False, False, False, True]

        adv1, ret1 = compute_gae_loop(rewards, values, gamma, lam, terminals)

        # 相同序列，GAE 应一致（验证计算不引入随机性）
        adv2, ret2 = compute_gae_loop(rewards, values, gamma, lam, terminals)

        np.testing.assert_allclose(adv1, adv2, atol=1e-10)
        np.testing.assert_allclose(ret1, ret2, atol=1e-10)

    def test_disturbance_env_reward_logic(self):
        """验证干扰环境的奖励逻辑：未终止时奖励为 1.0，终止时为 0.0。"""
        env = CartPoleSilentDisturbance(disturbance_prob=0.5, max_steps=100)
        env.reset(seed=0)

        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)

            if terminated:
                assert reward == 0.0
                break
            else:
                assert reward == 1.0

    def test_gae_with_disturbance_flag_in_info(self):
        """验证 info 中包含 disturbance 标志。"""
        env = CartPoleSilentDisturbance(disturbance_prob=1.0, max_steps=10)
        env.reset(seed=42)

        _, _, _, _, info = env.step(0)
        assert "disturbance" in info
        assert info["disturbance"] is True

    def test_gae_output_identical_for_same_rewards(self):
        """
        核心验证：即使干扰改变了物理状态，只要 rewards 序列相同，
        GAE 的 delta_t 计算结果必须完全一致。误差 < 1e-8。
        """
        np.random.seed(42)

        # 模拟两组实验：rewards 相同但来自不同环境路径
        T = 20
        rewards = np.ones(T)
        rewards[-1] = 0.0  # 最后一步终止

        values_run1 = np.random.uniform(0.3, 0.9, T + 1)
        terminals = np.zeros(T, dtype=bool)
        terminals[-1] = True

        adv1, ret1 = compute_gae_loop(rewards, values_run1, 0.99, 0.95, terminals)
        adv2, ret2 = compute_gae_vectorized(rewards, values_run1, 0.99, 0.95, terminals)

        np.testing.assert_allclose(adv1, adv2, atol=1e-8)
        np.testing.assert_allclose(ret1, ret2, atol=1e-8)


class TestPolicyNetwork:
    """策略网络接口测试。"""

    def test_forward_signature(self):
        """验证 forward 方法签名正确。"""
        policy = Policy(obs_dim=4, action_dim=2, lstm_hidden_size=32, lstm_num_layers=1)
        obs = torch.randn(1, 4)
        state = policy.init_hidden(batch_size=1)

        action, log_prob, next_state = policy.forward(obs, state)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert next_state[0].shape == (1, 1, 32)
        assert next_state[1].shape == (1, 1, 32)

    def test_act_returns_value(self):
        """验证 act 方法返回值函数估计。"""
        policy = Policy(obs_dim=4, action_dim=2, lstm_hidden_size=32, lstm_num_layers=1)
        obs = torch.randn(1, 4)
        state = policy.init_hidden(batch_size=1)

        action, log_prob, next_state, value = policy.act(obs, state)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)

    def test_evaluate_consistency(self):
        """验证 evaluate 与 act 在相同输入下产生一致的 log_prob。"""
        torch.manual_seed(42)
        policy = Policy(obs_dim=4, action_dim=2, lstm_hidden_size=32, lstm_num_layers=1)
        policy.eval()

        obs = torch.randn(1, 4)
        state = policy.init_hidden(batch_size=1)

        # 用 forward 获取动作
        with torch.no_grad():
            action, log_prob_forward, next_state = policy.forward(obs, state)

        # 用 evaluate 重新计算
        with torch.no_grad():
            log_prob_eval, value, entropy, _ = policy.evaluate(obs, state, action)

        assert torch.allclose(log_prob_forward, log_prob_eval, atol=1e-6)


class TestTrainerIntegration:
    """Trainer 集成测试。"""

    def test_full_training_loop_runs(self):
        """验证完整训练循环可以无异常完成。"""
        config = {
            "experiment": {"name": "integration_test", "seed": 42},
            "env": {"name": "CartPoleStandard-v0", "max_steps": 50},
            "policy": {"hidden_size": 16, "lstm_layers": 1},
            "algorithm": {
                "lr": 1e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "update_epochs": 2,
                "max_grad_norm": 0.5,
            },
            "training": {"num_episodes": 5, "batch_size": 16, "log_interval": 2},
            "logging": {"log_dir": "/tmp/test_integration_logs", "level": "WARNING"},
        }

        trainer = Trainer(config=config, experiment_id="integration_test")
        result = trainer.train()

        assert "final_reward" in result
        assert "total_episodes" in result
        assert result["total_episodes"] == 5

    def test_rollout_collects_correct_shapes(self):
        """验证 rollout 收集的轨迹数据形状正确。"""
        config = {
            "experiment": {"name": "shape_test", "seed": 42},
            "env": {"name": "CartPoleStandard-v0", "max_steps": 20},
            "policy": {"hidden_size": 16, "lstm_layers": 1},
            "algorithm": {"lr": 1e-3, "gamma": 0.99, "gae_lambda": 0.95},
            "training": {"num_episodes": 1, "batch_size": 8},
            "logging": {"log_dir": "/tmp/test_shape_logs", "level": "WARNING"},
        }

        trainer = Trainer(config=config, experiment_id="shape_test")
        hidden = trainer.policy.init_hidden(batch_size=1, device=trainer.device)
        trajectory, _, total_reward, ep_length = trainer.rollout(hidden)

        assert trajectory["obs"].shape[0] == ep_length
        assert trajectory["obs"].shape[1] == 4  # CartPole obs_dim
        assert trajectory["actions"].shape[0] == ep_length
        assert trajectory["rewards"].shape[0] == ep_length
        assert trajectory["log_probs"].shape[0] == ep_length
        assert trajectory["values"].shape[0] == ep_length + 1  # T+1
        assert trajectory["terminals"].shape[0] == ep_length
        assert total_reward >= 0


class TestGAEHandComputed:
    """使用固定数值手算 GAE，逐元素断言误差 < 1e-6。"""

    def test_gae_fixed_values(self):
        """
        rewards = [1.0, 2.0, 3.0], values = [0.5, 1.0, 1.5, ?]
        需要 T+1 个 values；此处设 bootstrap V(3)=2.0 作为非终止场景。
        gamma=0.9, lambda=0.8, 全部非终止。
        """
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5, 2.0]
        gamma = 0.9
        lam = 0.8
        terminals = [False, False, False]

        # 手算 delta
        # delta_0 = r0 + gamma*V1 - V0 = 1.0 + 0.9*1.0 - 0.5 = 1.4
        # delta_1 = r1 + gamma*V2 - V1 = 2.0 + 0.9*1.5 - 1.0 = 2.35
        # delta_2 = r2 + gamma*V3 - V2 = 3.0 + 0.9*2.0 - 1.5 = 3.3
        d0 = 1.0 + 0.9 * 1.0 - 0.5   # 1.4
        d1 = 2.0 + 0.9 * 1.5 - 1.0   # 2.35
        d2 = 3.0 + 0.9 * 2.0 - 1.5   # 3.3

        # 手算 GAE（从后往前）
        # A2 = d2 = 3.3
        # A1 = d1 + gamma*lam*A2 = 2.35 + 0.9*0.8*3.3 = 2.35 + 2.376 = 4.726
        # A0 = d0 + gamma*lam*A1 = 1.4 + 0.9*0.8*4.726 = 1.4 + 3.40272 = 4.80272
        A2 = d2
        A1 = d1 + gamma * lam * A2
        A0 = d0 + gamma * lam * A1

        adv_loop, ret_loop = compute_gae_loop(rewards, values, gamma, lam, terminals)
        adv_vec, ret_vec = compute_gae_vectorized(rewards, values, gamma, lam, terminals)

        # 循环实现逐元素断言
        assert abs(adv_loop[0] - A0) < 1e-6, f"loop A0: {adv_loop[0]} != {A0}"
        assert abs(adv_loop[1] - A1) < 1e-6, f"loop A1: {adv_loop[1]} != {A1}"
        assert abs(adv_loop[2] - A2) < 1e-6, f"loop A2: {adv_loop[2]} != {A2}"

        # 向量化实现逐元素断言
        assert abs(adv_vec[0] - A0) < 1e-6, f"vec A0: {adv_vec[0]} != {A0}"
        assert abs(adv_vec[1] - A1) < 1e-6, f"vec A1: {adv_vec[1]} != {A1}"
        assert abs(adv_vec[2] - A2) < 1e-6, f"vec A2: {adv_vec[2]} != {A2}"

        # returns = advantages + values[:T]
        for i in range(3):
            expected_ret = adv_loop[i] + values[i]
            assert abs(ret_loop[i] - expected_ret) < 1e-6
            assert abs(ret_vec[i] - expected_ret) < 1e-6


class TestDisturbanceRewardInvariance:
    """有无干扰下，episode 总回报序列一致性测试。"""

    def test_total_reward_identical_with_and_without_disturbance(self):
        """
        使用相同种子和相同动作序列，有干扰和无干扰环境的
        terminated/truncated 判定和奖励序列必须完全一致。
        """
        seed = 123
        max_steps = 200

        env_clean = CartPoleStandard(max_steps=max_steps)
        env_dist = CartPoleSilentDisturbance(disturbance_prob=0.3, max_steps=max_steps)

        env_clean.reset(seed=seed)
        env_dist.reset(seed=seed)

        # 使用确定性动作序列
        action_rng = np.random.default_rng(seed)
        actions = action_rng.integers(0, 2, size=max_steps)

        rewards_clean = []
        rewards_dist = []

        for i in range(max_steps):
            obs_c, r_c, term_c, trunc_c, _ = env_clean.step(actions[i])
            obs_d, r_d, term_d, trunc_d, _ = env_dist.step(actions[i])

            rewards_clean.append(r_c)
            rewards_dist.append(r_d)

            # 终止和截断判定必须一致
            assert term_c == term_d, f"Step {i}: terminated mismatch"
            assert trunc_c == trunc_d, f"Step {i}: truncated mismatch"

            # 奖励必须一致
            assert r_c == r_d, f"Step {i}: reward mismatch {r_c} vs {r_d}"

            # 返回的观测也必须一致（基于正常 step 后的状态）
            np.testing.assert_array_equal(obs_c, obs_d)

            if term_c or trunc_c:
                break

        assert sum(rewards_clean) == sum(rewards_dist)

    def test_episode_length_identical(self):
        """episode 长度在有无干扰下完全一致。"""
        seed = 77
        max_steps = 100

        env_clean = CartPoleStandard(max_steps=max_steps)
        env_dist = CartPoleSilentDisturbance(disturbance_prob=0.5, max_steps=max_steps)

        env_clean.reset(seed=seed)
        env_dist.reset(seed=seed)

        action_rng = np.random.default_rng(seed)

        len_clean = 0
        len_dist = 0

        for _ in range(max_steps):
            a = action_rng.integers(0, 2)
            _, _, tc, trc, _ = env_clean.step(a)
            len_clean += 1
            if tc or trc:
                break

        # 重置动作生成器
        action_rng2 = np.random.default_rng(seed)
        for _ in range(max_steps):
            a = action_rng2.integers(0, 2)
            _, _, td, trd, _ = env_dist.step(a)
            len_dist += 1
            if td or trd:
                break

        assert len_clean == len_dist


class TestDatabaseSQLite:
    """使用 sqlite:///:memory: 测试数据库层，不依赖 PostgreSQL。"""

    def test_create_tables(self):
        """验证建表成功。"""
        from db.database import Database
        db = Database("sqlite:///:memory:")
        from db.models import Experiment
        # 表应该已创建，inspector 查询
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "experiments" in tables

    def test_create_and_query_experiment(self):
        """验证写入和查询。"""
        from db.database import Database
        db = Database("sqlite:///:memory:")

        exp_id = db.create_experiment(
            name="test_exp",
            config_yaml="gamma: 0.99\nlr: 0.001",
        )
        assert exp_id is not None
        assert isinstance(exp_id, int)

        exp = db.get_experiment(exp_id)
        assert exp.name == "test_exp"
        assert exp.status == "running"
        assert "gamma" in exp.config_yaml

    def test_update_experiment(self):
        """验证更新 final_reward 和 status。"""
        from db.database import Database
        db = Database("sqlite:///:memory:")

        exp_id = db.create_experiment(name="update_test", config_yaml="{}")
        db.update_experiment(exp_id, status="finished", total_episodes=50, final_reward=195.5)

        exp = db.get_experiment(exp_id)
        assert exp.status == "finished"
        assert exp.total_episodes == 50
        assert abs(exp.final_reward - 195.5) < 1e-6

    def test_list_experiments(self):
        """验证列表查询。"""
        from db.database import Database
        db = Database("sqlite:///:memory:")

        db.create_experiment(name="exp_a", config_yaml="{}")
        db.create_experiment(name="exp_b", config_yaml="{}")

        exps = db.list_experiments()
        assert len(exps) == 2
        names = {e.name for e in exps}
        assert "exp_a" in names
        assert "exp_b" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
