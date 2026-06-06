"""
训练主循环：Trainer 类实现 rollout 收集、GAE 计算、PPO 策略更新。
严格遵循接口规范，GAE delta_t 基于原始奖励，隐状态在 episode 边界正确传递。
"""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from env import make_env
from policy import Policy


def compute_gae_loop(rewards, values, gamma, lam, terminals):
    """
    循环实现 GAE（广义优势估计）。

    Args:
        rewards: 原始奖励序列, 长度 T (list or ndarray)
        values: 值函数预测, 长度 T+1 (包含 bootstrap value)
        gamma: 折扣因子
        lam: GAE lambda
        terminals: 终止标志序列, 长度 T (bool)

    Returns:
        advantages: GAE 优势, 长度 T
        returns: 回报 (advantages + values[:T]), 长度 T

    注意：严禁对 reward_t 进行任何缩放或变换后再计算 delta_t。
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0

    for t in reversed(range(T)):
        if terminals[t]:
            next_value = 0.0
            gae = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - terminals[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


def compute_gae_vectorized(rewards, values, gamma, lam, terminals):
    """
    向量化实现 GAE。
    与循环实现逻辑一致，用于交叉验证。

    Args:
        rewards: 原始奖励序列, 长度 T
        values: 值函数预测, 长度 T+1
        gamma: 折扣因子
        lam: GAE lambda
        terminals: 终止标志序列, 长度 T

    Returns:
        advantages: GAE 优势, 长度 T
        returns: 回报, 长度 T
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    terminals = np.asarray(terminals, dtype=np.float64)

    T = len(rewards)
    masks = 1.0 - terminals
    deltas = rewards + gamma * values[1:] * masks - values[:T]

    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * masks[t] * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


class StructuredLogger:
    """结构化 JSON lines 日志记录器，支持按时间戳回查每 episode 的损失和回报。"""

    def __init__(self, log_dir, experiment_id, hyperparams=None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"experiment_{experiment_id}.jsonl")
        self.logger = logging.getLogger(f"rl_experiment_{experiment_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # 清除旧 handler 避免多次初始化累积
        for h in self.logger.handlers[:]:
            h.close()
            self.logger.removeHandler(h)

        handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

        # 写入元数据头（超参信息，便于回查时关联配置）
        if hyperparams:
            meta = {
                "type": "metadata",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": str(experiment_id),
                "hyperparams": hyperparams,
            }
            self.logger.info(json.dumps(meta, ensure_ascii=False))

    def log_episode(self, episode, total_reward, policy_loss, value_loss, entropy,
                    episode_length=0, lr=None, gamma=None, lam=None):
        record = {
            "type": "episode",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "episode_length": episode_length,
            "total_reward": float(total_reward),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
        }
        if lr is not None:
            record["lr"] = float(lr)
        if gamma is not None:
            record["gamma"] = float(gamma)
        if lam is not None:
            record["gae_lambda"] = float(lam)
        self.logger.info(json.dumps(record, ensure_ascii=False))

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class Trainer:
    """
    训练器：负责完整的 rollout -> GAE -> PPO 更新闭环。
    所有状态封装在实例内部，不引入额外全局状态。
    """

    def __init__(self, config, experiment_id=None, monitor_callback=None):
        self.config = config
        self.experiment_id = experiment_id or "local"
        self.monitor_callback = monitor_callback

        # 随机种子
        seed = config.get("experiment", {}).get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化环境
        env_config = config.get("env", {})
        env_name = env_config.get("name", "CartPoleSilentDisturbance-v0")
        env_kwargs = {}
        if "disturbance_prob" in env_config:
            env_kwargs["disturbance_prob"] = env_config["disturbance_prob"]
        if "max_steps" in env_config:
            env_kwargs["max_steps"] = env_config["max_steps"]
        self.env = make_env(env_name, **env_kwargs)

        # 初始化策略网络 —— 从 network 配置段读取，policy 段作为兼容回退
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        network_config = config.get("network", {})
        policy_config = config.get("policy", {})

        self._mlp_hidden = network_config.get(
            "hidden_sizes",
            [policy_config.get("hidden_size", 64)] * 2,
        )
        self._activation = network_config.get("activation", "tanh")
        self._lstm_hidden = network_config.get(
            "lstm_hidden_size",
            policy_config.get("lstm_hidden_size", 128),
        )
        self._lstm_layers = network_config.get(
            "lstm_num_layers",
            policy_config.get("lstm_layers", 1),
        )
        self._use_lstm = network_config.get("use_lstm", True)

        self.policy = Policy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=self._mlp_hidden,
            activation=self._activation,
            lstm_hidden_size=self._lstm_hidden,
            lstm_num_layers=self._lstm_layers,
            use_lstm=self._use_lstm,
        ).to(self.device)

        # 算法超参数
        algo_config = config.get("algorithm", {})
        self.lr = algo_config.get("lr", 3e-4)
        self.gamma = algo_config.get("gamma", 0.99)
        self.lam = algo_config.get("gae_lambda", 0.95)
        self.clip_ratio = algo_config.get("clip_epsilon", 0.2)
        self.value_loss_coef = algo_config.get("value_coef", 0.5)
        self.entropy_coef = algo_config.get("entropy_coef", 0.01)
        self.max_grad_norm = algo_config.get("max_grad_norm", 0.5)
        self.num_epochs = algo_config.get("update_epochs", 4)

        # 优化器 —— lr 直接从配置传入
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # 训练参数
        train_config = config.get("training", {})
        self.num_episodes = train_config.get("num_episodes", 100)
        self.max_steps_per_episode = train_config.get(
            "max_steps_per_episode", env_config.get("max_steps", 500)
        )
        self.mini_batch_size = train_config.get("batch_size", 64)
        self.log_interval = train_config.get("log_interval", 10)
        self.save_interval = train_config.get("save_interval", 100)

        # 存储
        storage_config = config.get("storage", {})
        self.model_dir = storage_config.get("model_dir", "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # 结构化日志 —— 记录超参用于回查
        log_config = config.get("logging", {})
        log_dir = log_config.get("log_dir", "logs")
        hyperparams = {
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.lam,
            "clip_epsilon": self.clip_ratio,
            "value_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "update_epochs": self.num_epochs,
            "hidden_sizes": self._mlp_hidden,
            "lstm_hidden_size": self._lstm_hidden,
            "lstm_num_layers": self._lstm_layers,
            "use_lstm": self._use_lstm,
            "activation": self._activation,
        }
        self.structured_logger = StructuredLogger(log_dir, self.experiment_id, hyperparams)

        # 控制台日志
        self.console_logger = logging.getLogger("rl_trainer")
        if not self.console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.console_logger.addHandler(handler)
        self.console_logger.setLevel(
            getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        )

    def rollout(self, hidden_state):
        """
        收集一条完整 episode 轨迹。

        隐状态传递规则：
        - hidden_state 在 episode 内持续传递
        - 仅当 episode 真正结束（terminated 或 truncated）时重置为零
        - 不引入任何全局状态

        Args:
            hidden_state: 当前 LSTM 隐状态 (h, c)

        Returns:
            trajectory: 字典，含 obs, actions, rewards, log_probs, values, terminals
            hidden_state: episode 结束后重置的隐状态
            total_reward: episode 总回报
            episode_length: episode 步数
        """
        obs, _ = self.env.reset()
        obs_list = []
        actions_list = []
        rewards_list = []
        log_probs_list = []
        values_list = []
        terminals_list = []

        total_reward = 0.0
        done = False
        step = 0

        while not done and step < self.max_steps_per_episode:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, next_hidden, value = self.policy.act(
                    obs_tensor, hidden_state
                )

            action_np = action.item()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)

            obs_list.append(obs)
            actions_list.append(action_np)
            rewards_list.append(reward)
            log_probs_list.append(log_prob.item())
            values_list.append(value.item())

            done = terminated or truncated
            terminals_list.append(done)

            total_reward += reward
            obs = next_obs
            hidden_state = next_hidden
            step += 1

        # Bootstrap value for GAE
        if not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                bootstrap_value, _ = self.policy.get_value(obs_tensor, hidden_state)
            values_list.append(bootstrap_value.item())
        else:
            values_list.append(0.0)

        # Episode 结束后重置隐状态（严格遵循规范）
        hidden_state = self.policy.init_hidden(batch_size=1, device=self.device)

        trajectory = {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(actions_list, dtype=np.int64),
            "rewards": np.array(rewards_list, dtype=np.float64),
            "log_probs": np.array(log_probs_list, dtype=np.float64),
            "values": np.array(values_list, dtype=np.float64),
            "terminals": np.array(terminals_list, dtype=bool),
        }

        return trajectory, hidden_state, total_reward, step

    def compute_gae(self, trajectory):
        """计算 GAE 优势和回报，使用循环实现。"""
        advantages, returns = compute_gae_loop(
            rewards=trajectory["rewards"],
            values=trajectory["values"],
            gamma=self.gamma,
            lam=self.lam,
            terminals=trajectory["terminals"],
        )
        return advantages, returns

    def ppo_update(self, trajectory, advantages, returns):
        """
        执行 PPO 策略更新。

        Returns:
            policy_loss: 策略损失均值
            value_loss: 值函数损失均值
            entropy: 策略熵均值
        """
        obs = torch.FloatTensor(trajectory["obs"]).to(self.device)
        actions = torch.LongTensor(trajectory["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(trajectory["log_probs"]).to(self.device)
        advantages_t = torch.FloatTensor(advantages.copy()).to(self.device)
        returns_t = torch.FloatTensor(returns.copy()).to(self.device)

        # 标准化优势（安全处理单步 episode 避免 NaN）
        if len(advantages_t) > 1:
            adv_std = advantages_t.std()
            if adv_std > 1e-8:
                advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        batch_size = len(obs)
        effective_mini_batch = min(self.mini_batch_size, batch_size)

        for epoch in range(self.num_epochs):
            indices = np.arange(batch_size)
            np.random.shuffle(indices)

            for start in range(0, batch_size, effective_mini_batch):
                end = min(start + effective_mini_batch, batch_size)
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                hidden = self.policy.init_hidden(
                    batch_size=len(mb_indices), device=self.device
                )
                new_log_probs, values, entropy, _ = self.policy.evaluate(
                    mb_obs, hidden, mb_actions
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        avg_policy_loss = total_policy_loss / max(num_updates, 1)
        avg_value_loss = total_value_loss / max(num_updates, 1)
        avg_entropy = total_entropy / max(num_updates, 1)

        return avg_policy_loss, avg_value_loss, avg_entropy

    def train(self):
        """执行完整训练流程，返回结果摘要。"""
        self.console_logger.info(
            f"Starting training: {self.num_episodes} episodes | "
            f"lr={self.lr} gamma={self.gamma} lam={self.lam} | "
            f"hidden={self._mlp_hidden} lstm={self._lstm_hidden} | "
            f"experiment_id={self.experiment_id}"
        )

        hidden_state = self.policy.init_hidden(batch_size=1, device=self.device)
        episode_rewards = []
        best_reward = float("-inf")

        for episode in range(1, self.num_episodes + 1):
            trajectory, hidden_state, total_reward, ep_length = self.rollout(
                hidden_state
            )
            episode_rewards.append(total_reward)

            advantages, returns = self.compute_gae(trajectory)

            policy_loss, value_loss, entropy = self.ppo_update(
                trajectory, advantages, returns
            )

            # 每个 episode 写入结构化日志
            self.structured_logger.log_episode(
                episode=episode,
                total_reward=total_reward,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                episode_length=ep_length,
                lr=self.lr,
                gamma=self.gamma,
                lam=self.lam,
            )

            if self.monitor_callback:
                should_stop = self.monitor_callback.on_episode_end(
                    episode=episode,
                    total_reward=total_reward,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                    episode_length=ep_length,
                    lr=self.lr,
                    gamma=self.gamma,
                    lam=self.lam,
                )
                if should_stop:
                    self.console_logger.info(
                        "Stop requested via monitor. Saving model and exiting."
                    )
                    self._save_model(episode)
                    self.structured_logger.close()
                    self.monitor_callback.close()
                    return {
                        "final_reward": total_reward,
                        "avg_final_10": (
                            np.mean(episode_rewards[-10:])
                            if len(episode_rewards) >= 10
                            else np.mean(episode_rewards)
                        ),
                        "total_episodes": len(episode_rewards),
                        "stopped": True,
                    }

            if episode % self.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.log_interval:])
                self.console_logger.info(
                    f"Episode {episode}/{self.num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Policy Loss: {policy_loss:.4f} | "
                    f"Value Loss: {value_loss:.4f} | "
                    f"Entropy: {entropy:.4f}"
                )

            if episode % self.save_interval == 0 or total_reward > best_reward:
                if total_reward > best_reward:
                    best_reward = total_reward
                self._save_model(episode)

        final_reward = episode_rewards[-1] if episode_rewards else 0.0
        avg_final = (
            np.mean(episode_rewards[-10:])
            if len(episode_rewards) >= 10
            else np.mean(episode_rewards)
        )

        self.console_logger.info(
            f"Training complete. Final reward: {final_reward:.2f}, "
            f"Avg last 10: {avg_final:.2f}"
        )
        self.structured_logger.close()
        if self.monitor_callback:
            self.monitor_callback.close()

        return {
            "final_reward": final_reward,
            "avg_final_10": avg_final,
            "total_episodes": len(episode_rewards),
        }

    def _save_model(self, episode):
        """保存模型 checkpoint。"""
        path = os.path.join(
            self.model_dir, f"experiment_{self.experiment_id}_ep{episode}.pt"
        )
        torch.save(
            {
                "episode": episode,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
