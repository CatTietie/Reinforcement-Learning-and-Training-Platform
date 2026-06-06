"""
中心Learner：接收Worker提交的轨迹，执行PPO更新，通过共享内存广播权重。
"""

import logging
import os
import queue

import numpy as np
import torch

from distributed.utils import create_policy_from_config, ppo_update_from_batch
from train import StructuredLogger, compute_gae_loop


class Learner:
    """
    中心化Learner，在主进程中运行。
    使用共享内存进行零拷贝权重广播，通过版本号通知Worker拉取更新。
    """

    def __init__(self, config, trajectory_queue, shared_state_dict,
                 weight_version, episode_counter, stop_event, lock,
                 experiment_id=None):
        self.config = config
        self.trajectory_queue = trajectory_queue
        self.shared_state_dict = shared_state_dict
        self.weight_version = weight_version
        self.episode_counter = episode_counter
        self.stop_event = stop_event
        self.lock = lock
        self.experiment_id = experiment_id or "distributed"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = create_policy_from_config(config).to(self.device)

        algo_config = config.get("algorithm", {})
        self.lr = algo_config.get("lr", 3e-4)
        self.gamma = algo_config.get("gamma", 0.99)
        self.lam = algo_config.get("gae_lambda", 0.95)
        self.clip_ratio = algo_config.get("clip_epsilon", 0.2)
        self.value_loss_coef = algo_config.get("value_coef", 0.5)
        self.entropy_coef = algo_config.get("entropy_coef", 0.01)
        self.max_grad_norm = algo_config.get("max_grad_norm", 0.5)
        self.num_epochs = algo_config.get("update_epochs", 4)

        train_config = config.get("training", {})
        self.target_episodes = train_config.get("num_episodes", 100)
        self.mini_batch_size = train_config.get("batch_size", 64)
        self.log_interval = train_config.get("log_interval", 10)
        self.save_interval = train_config.get("save_interval", 100)

        dist_config = config.get("distributed", {})
        self.batch_before_update = dist_config.get("batch_before_update", 4)
        self.weight_sync_interval = dist_config.get("weight_sync_interval", 1)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        storage_config = config.get("storage", {})
        self.model_dir = storage_config.get("model_dir", "models")
        os.makedirs(self.model_dir, exist_ok=True)

        log_config = config.get("logging", {})
        log_dir = log_config.get("log_dir", "logs")
        hyperparams = {
            "lr": self.lr, "gamma": self.gamma, "gae_lambda": self.lam,
            "clip_epsilon": self.clip_ratio, "mode": "distributed",
        }
        self.structured_logger = StructuredLogger(log_dir, self.experiment_id, hyperparams)

        self.console_logger = logging.getLogger("rl_distributed_learner")
        if not self.console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.console_logger.addHandler(handler)
        self.console_logger.setLevel(
            getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        )

    def broadcast_weights(self):
        """将当前策略权重写入共享内存并递增版本号。"""
        state_dict = self.policy.state_dict()
        for key in self.shared_state_dict:
            self.shared_state_dict[key].copy_(state_dict[key].cpu())
        self.weight_version.value += 1

    def run(self):
        """
        Learner主循环：收集轨迹 → GAE → PPO更新 → 共享内存广播权重。
        """
        self.console_logger.info(
            f"Learner started: target={self.target_episodes} episodes, "
            f"batch={self.batch_before_update}"
        )

        self.broadcast_weights()

        episode_rewards = []
        update_count = 0
        best_reward = float("-inf")
        processed_episodes = 0

        while processed_episodes < self.target_episodes:
            batch_trajectories = []
            batch_rewards = []

            while len(batch_trajectories) < self.batch_before_update:
                if processed_episodes + len(batch_trajectories) >= self.target_episodes:
                    break
                try:
                    payload = self.trajectory_queue.get(timeout=10.0)
                    if isinstance(payload, list):
                        for item in payload:
                            batch_trajectories.append(item["trajectory"])
                            batch_rewards.append(item["total_reward"])
                    else:
                        batch_trajectories.append(payload["trajectory"])
                        batch_rewards.append(payload["total_reward"])
                except queue.Empty:
                    with self.lock:
                        if self.episode_counter.value >= self.target_episodes:
                            break
                    continue

            if not batch_trajectories:
                break

            # 逐条计算GAE，拼接为大batch
            all_obs = []
            all_actions = []
            all_old_log_probs = []
            all_advantages = []
            all_returns = []

            for traj in batch_trajectories:
                advantages, returns = compute_gae_loop(
                    rewards=traj["rewards"],
                    values=traj["values"],
                    gamma=self.gamma,
                    lam=self.lam,
                    terminals=traj["terminals"],
                )
                all_obs.append(traj["obs"])
                all_actions.append(traj["actions"])
                all_old_log_probs.append(traj["log_probs"])
                all_advantages.append(advantages)
                all_returns.append(returns)

            combined_obs = np.concatenate(all_obs, axis=0)
            combined_actions = np.concatenate(all_actions, axis=0)
            combined_old_log_probs = np.concatenate(all_old_log_probs, axis=0)
            combined_advantages = np.concatenate(all_advantages, axis=0)
            combined_returns = np.concatenate(all_returns, axis=0)

            policy_loss, value_loss, entropy = ppo_update_from_batch(
                policy=self.policy,
                optimizer=self.optimizer,
                obs=combined_obs,
                actions=combined_actions,
                old_log_probs=combined_old_log_probs,
                advantages=combined_advantages,
                returns=combined_returns,
                clip_ratio=self.clip_ratio,
                value_loss_coef=self.value_loss_coef,
                entropy_coef=self.entropy_coef,
                max_grad_norm=self.max_grad_norm,
                num_epochs=self.num_epochs,
                mini_batch_size=self.mini_batch_size,
                device=self.device,
            )

            update_count += 1
            episode_rewards.extend(batch_rewards)
            processed_episodes += len(batch_trajectories)

            if update_count % self.weight_sync_interval == 0:
                self.broadcast_weights()

            for i, reward in enumerate(batch_rewards):
                ep_num = processed_episodes - len(batch_rewards) + i + 1
                self.structured_logger.log_episode(
                    episode=ep_num,
                    total_reward=reward,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                )

            if processed_episodes % self.log_interval < self.batch_before_update:
                avg_reward = np.mean(episode_rewards[-self.log_interval:])
                self.console_logger.info(
                    f"Episodes {processed_episodes}/{self.target_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"PL: {policy_loss:.4f} | VL: {value_loss:.4f} | "
                    f"Ent: {entropy:.4f}"
                )

            current_best = max(batch_rewards)
            if current_best > best_reward:
                best_reward = current_best
                self._save_model(processed_episodes)
            elif processed_episodes % self.save_interval < self.batch_before_update:
                self._save_model(processed_episodes)

        self.stop_event.set()
        self.structured_logger.close()

        final_reward = episode_rewards[-1] if episode_rewards else 0.0
        avg_final = (
            np.mean(episode_rewards[-10:])
            if len(episode_rewards) >= 10
            else np.mean(episode_rewards) if episode_rewards else 0.0
        )

        self.console_logger.info(
            f"Distributed training complete. Final reward: {final_reward:.2f}, "
            f"Avg last 10: {avg_final:.2f}, Updates: {update_count}"
        )

        return {
            "final_reward": final_reward,
            "avg_final_10": avg_final,
            "total_episodes": processed_episodes,
        }

    def _save_model(self, episode):
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
