"""
分布式训练共享工具函数：从Trainer逻辑中提取的独立可复用组件。
"""

import numpy as np
import torch
import torch.nn as nn

from env import make_env
from policy import Policy


def create_policy_from_config(config):
    """
    根据配置字典创建Policy实例。
    与Trainer.__init__中的策略创建逻辑保持一致。
    """
    env_config = config.get("env", {})
    env_name = env_config.get("name", "CartPoleSilentDisturbance-v0")
    env_kwargs = {}
    if "disturbance_prob" in env_config:
        env_kwargs["disturbance_prob"] = env_config["disturbance_prob"]
    if "max_steps" in env_config:
        env_kwargs["max_steps"] = env_config["max_steps"]
    env = make_env(env_name, **env_kwargs)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    network_config = config.get("network", {})
    policy_config = config.get("policy", {})

    hidden_sizes = network_config.get(
        "hidden_sizes",
        [policy_config.get("hidden_size", 64)] * 2,
    )
    activation = network_config.get("activation", "tanh")
    lstm_hidden_size = network_config.get(
        "lstm_hidden_size",
        policy_config.get("lstm_hidden_size", 128),
    )
    lstm_num_layers = network_config.get(
        "lstm_num_layers",
        policy_config.get("lstm_layers", 1),
    )
    use_lstm = network_config.get("use_lstm", True)

    policy = Policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        use_lstm=use_lstm,
    )

    env.close()
    return policy


def create_env_from_config(config):
    """根据配置字典创建环境实例。"""
    env_config = config.get("env", {})
    env_name = env_config.get("name", "CartPoleSilentDisturbance-v0")
    env_kwargs = {}
    if "disturbance_prob" in env_config:
        env_kwargs["disturbance_prob"] = env_config["disturbance_prob"]
    if "max_steps" in env_config:
        env_kwargs["max_steps"] = env_config["max_steps"]
    return make_env(env_name, **env_kwargs)


def collect_episode(policy, env, device, max_steps):
    """
    收集单个episode的完整轨迹。
    逻辑与Trainer.rollout()一致，但作为独立函数不依赖Trainer实例。

    Returns:
        trajectory: dict(obs, actions, rewards, log_probs, values, terminals)
        total_reward: float
        episode_length: int
    """
    obs, _ = env.reset()
    hidden_state = policy.init_hidden(batch_size=1, device=device)

    obs_list = []
    actions_list = []
    rewards_list = []
    log_probs_list = []
    values_list = []
    terminals_list = []

    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, next_hidden, value = policy.act(obs_tensor, hidden_state)

        action_np = action.item()
        next_obs, reward, terminated, truncated, info = env.step(action_np)

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

    if not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            bootstrap_value, _ = policy.get_value(obs_tensor, hidden_state)
        values_list.append(bootstrap_value.item())
    else:
        values_list.append(0.0)

    trajectory = {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.int64),
        "rewards": np.array(rewards_list, dtype=np.float64),
        "log_probs": np.array(log_probs_list, dtype=np.float64),
        "values": np.array(values_list, dtype=np.float64),
        "terminals": np.array(terminals_list, dtype=bool),
    }

    return trajectory, total_reward, step


def ppo_update_from_batch(policy, optimizer, obs, actions, old_log_probs,
                          advantages, returns, clip_ratio, value_loss_coef,
                          entropy_coef, max_grad_norm, num_epochs,
                          mini_batch_size, device):
    """
    在batch数据上执行PPO更新。
    逻辑与Trainer.ppo_update()一致。

    Args:
        policy: Policy网络
        optimizer: Adam优化器
        obs: (N, obs_dim) numpy array
        actions: (N,) numpy array
        old_log_probs: (N,) numpy array
        advantages: (N,) numpy array
        returns: (N,) numpy array
        clip_ratio: PPO clip参数
        value_loss_coef: 值函数损失系数
        entropy_coef: 熵奖励系数
        max_grad_norm: 梯度裁剪阈值
        num_epochs: 更新轮数
        mini_batch_size: mini-batch大小
        device: torch设备

    Returns:
        (avg_policy_loss, avg_value_loss, avg_entropy)
    """
    obs_t = torch.FloatTensor(obs).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(device)
    advantages_t = torch.FloatTensor(advantages.copy()).to(device)
    returns_t = torch.FloatTensor(returns.copy()).to(device)

    if len(advantages_t) > 1:
        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    batch_size = len(obs_t)
    effective_mini_batch = min(mini_batch_size, batch_size)

    for epoch in range(num_epochs):
        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, effective_mini_batch):
            end = min(start + effective_mini_batch, batch_size)
            mb_indices = indices[start:end]

            mb_obs = obs_t[mb_indices]
            mb_actions = actions_t[mb_indices]
            mb_old_log_probs = old_log_probs_t[mb_indices]
            mb_advantages = advantages_t[mb_indices]
            mb_returns = returns_t[mb_indices]

            hidden = policy.init_hidden(batch_size=len(mb_indices), device=device)
            new_log_probs, values, entropy, _ = policy.evaluate(
                mb_obs, hidden, mb_actions
            )

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(
                ratio, 1 - clip_ratio, 1 + clip_ratio
            ) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, mb_returns)
            entropy_loss = entropy.mean()

            loss = (
                policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()
            num_updates += 1

    avg_policy_loss = total_policy_loss / max(num_updates, 1)
    avg_value_loss = total_value_loss / max(num_updates, 1)
    avg_entropy = total_entropy / max(num_updates, 1)

    return avg_policy_loss, avg_value_loss, avg_entropy
