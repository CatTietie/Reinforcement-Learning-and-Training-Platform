"""
环境接口定义：CartPoleSilentDisturbance 及基类。
遵循 Gymnasium 0.26+ 标准 5 元组接口。
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnvWrapper(gym.Env):
    """环境基类，定义标准接口约束。子类必须实现 reset() 和 step()。"""

    pass


class CartPoleSilentDisturbance(EnvWrapper):
    """
    CartPole 变体环境，在 step 调用时有一定概率内部多执行一次物理更新，
    但不返回额外奖励，用于验证 GAE 计算的鲁棒性。

    静默干扰不影响返回给 agent 的 (obs, reward, terminated, truncated, info)，
    仅改变内部物理状态的演化路径。
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, disturbance_prob=0.1, max_steps=500, render_mode=None):
        super().__init__()
        self.disturbance_prob = disturbance_prob
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self._step_count = 0
        self._rng = None
        self._disturbance_rng = None
        self._disturbance_occurred = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        # 独立的干扰随机源，不影响物理初始化的随机状态
        self._disturbance_rng = np.random.default_rng(
            seed + 10000 if seed is not None else None
        )
        self.state = self._rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(
            np.float32
        )
        self._step_count = 0
        self._disturbance_occurred = False
        return np.array(self.state, dtype=np.float32), {}

    def _physics_step(self, action):
        """执行一次物理更新。"""
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 正常物理更新
        self._physics_step(action)
        self._step_count += 1

        # 基于正常更新后的状态判定终止和奖励
        state_after_normal = self.state.copy()
        x, x_dot, theta, theta_dot = state_after_normal
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self._step_count >= self.max_steps
        reward = 1.0 if not terminated else 0.0

        # 静默干扰：有概率额外执行一次物理更新。
        # 干扰是纯内部行为，不影响对外暴露的 obs/reward/terminated/truncated，
        # 也不影响后续 step 的物理演化（执行后恢复状态）。
        self._disturbance_occurred = False
        if self._disturbance_rng.random() < self.disturbance_prob:
            self._disturbance_occurred = True
            self._physics_step(action)
            # 恢复到正常 step 后的状态，确保干扰不累积
            self.state = state_after_normal

        return (
            np.array(state_after_normal, dtype=np.float32),
            reward,
            terminated,
            truncated,
            {"disturbance": self._disturbance_occurred},
        )


class CartPoleStandard(EnvWrapper):
    """标准 CartPole 环境（无干扰），用于对比测试。"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, max_steps=500, render_mode=None):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self._step_count = 0
        self._rng = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.state = self._rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(
            np.float32
        )
        self._step_count = 0
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_count += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self._step_count >= self.max_steps
        reward = 1.0 if not terminated else 0.0

        return (
            np.array(self.state, dtype=np.float32),
            reward,
            terminated,
            truncated,
            {},
        )


def make_env(env_name, **kwargs):
    """工厂方法：根据名称创建环境实例。"""
    registry = {
        "CartPoleSilentDisturbance-v0": CartPoleSilentDisturbance,
        "CartPoleStandard-v0": CartPoleStandard,
    }
    if env_name not in registry:
        raise ValueError(
            f"Unknown environment: {env_name}. Available: {list(registry.keys())}"
        )
    return registry[env_name](**kwargs)
