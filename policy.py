"""
策略网络定义：支持 LSTM 隐状态的 Actor-Critic 策略。
前向签名严格遵循接口规范。
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):
    """
    带 LSTM 的 Actor-Critic 策略网络。

    前向传播签名：forward(obs, state) -> (action, log_prob, next_state)
    - obs: (batch_size, obs_dim)
    - state: (h, c) 元组, h/c 形状 (num_layers, batch_size, hidden_dim)
    - 返回: action, log_prob, next_state

    隐状态传递规则：在整个 rollout 过程中 state 持续传递，
    仅当 episode 真正结束（terminated 或 truncated）时才可重置为零状态。
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes=None,
        activation="tanh",
        lstm_hidden_size=128,
        lstm_num_layers=1,
        use_lstm=True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.use_lstm = use_lstm

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}.get(
            activation, nn.Tanh
        )

        # 前置 MLP 特征提取器
        layers = []
        in_dim = obs_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_size))
            layers.append(activation_fn())
            in_dim = h_size
        self.feature_extractor = nn.Sequential(*layers)

        # LSTM 层
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
            )
            policy_input_dim = lstm_hidden_size
        else:
            self.lstm = None
            policy_input_dim = in_dim

        # Actor head（策略输出）
        self.actor = nn.Linear(policy_input_dim, action_dim)

        # Critic head（值函数输出）
        self.critic = nn.Linear(policy_input_dim, 1)

    def init_hidden(self, batch_size=1, device=None):
        """初始化 LSTM 隐状态为零状态。"""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h, c)

    def forward(self, obs, state):
        """
        前向传播。

        Args:
            obs: 观测张量, 形状 (batch_size, obs_dim)
            state: LSTM 隐状态 (h, c), 每个形状 (num_layers, batch_size, hidden_dim)

        Returns:
            action: 采样动作
            log_prob: 动作对数概率
            next_state: 更新后的隐状态
        """
        features = self.feature_extractor(obs)

        if self.use_lstm:
            # LSTM 期望输入 (batch, seq_len, features)
            lstm_input = features.unsqueeze(1)
            lstm_out, next_state = self.lstm(lstm_input, state)
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out = features
            next_state = state

        logits = self.actor(lstm_out)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, next_state

    def act(self, obs, state):
        """
        训练时交互用方法，内部调用 forward 进行动作采样。

        Args:
            obs: 观测张量, 形状 (batch_size, obs_dim)
            state: LSTM 隐状态

        Returns:
            action: 采样动作
            log_prob: 动作对数概率
            next_state: 更新后的隐状态
            value: 当前状态的值函数估计
        """
        features = self.feature_extractor(obs)

        if self.use_lstm:
            lstm_input = features.unsqueeze(1)
            lstm_out, next_state = self.lstm(lstm_input, state)
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out = features
            next_state = state

        logits = self.actor(lstm_out)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(lstm_out).squeeze(-1)

        return action, log_prob, next_state, value

    def evaluate(self, obs, state, actions):
        """
        评估给定观测和动作的 log_prob、值函数和熵（用于 PPO 更新）。

        Args:
            obs: (batch_size, obs_dim)
            state: LSTM 隐状态
            actions: (batch_size,) 已执行动作

        Returns:
            log_probs: 动作对数概率
            values: 值函数估计
            entropy: 策略熵
            next_state: 更新后的隐状态
        """
        features = self.feature_extractor(obs)

        if self.use_lstm:
            lstm_input = features.unsqueeze(1)
            lstm_out, next_state = self.lstm(lstm_input, state)
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out = features
            next_state = state

        logits = self.actor(lstm_out)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(lstm_out).squeeze(-1)

        return log_probs, values, entropy, next_state

    def get_value(self, obs, state):
        """仅计算值函数（用于 GAE 的最后一步 bootstrap）。"""
        features = self.feature_extractor(obs)

        if self.use_lstm:
            lstm_input = features.unsqueeze(1)
            lstm_out, next_state = self.lstm(lstm_input, state)
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out = features
            next_state = state

        value = self.critic(lstm_out).squeeze(-1)
        return value, next_state
