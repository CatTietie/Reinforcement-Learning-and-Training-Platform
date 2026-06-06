"""
Worker进程：独立收集episode轨迹并提交到中心Learner。
使用共享内存进行权重同步，减少序列化开销。
"""

import queue

import numpy as np
import torch

from distributed.utils import collect_episode, create_env_from_config, create_policy_from_config


def worker_process(worker_id, config, trajectory_queue, shared_state_dict,
                   weight_version, episode_counter, stop_event, target_episodes, lock):
    """
    Worker主函数（模块顶层函数以支持Windows spawn pickle）。

    使用共享内存张量(shared_state_dict)读取最新权重，
    通过weight_version原子变量检测权重更新。

    Args:
        worker_id: Worker编号
        config: 训练配置字典
        trajectory_queue: mp.Queue，用于提交轨迹给Learner
        shared_state_dict: dict，共享内存中的模型参数（torch.Tensor with share_memory_）
        weight_version: mp.Value('i')，权重版本号（Learner每次更新递增）
        episode_counter: mp.Value('i')，全局episode计数器
        stop_event: mp.Event，停止信号
        target_episodes: 全局目标episode总数
        lock: mp.Lock，保护episode_counter的原子递增
    """
    seed = config.get("experiment", {}).get("seed", 42) + worker_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    policy = create_policy_from_config(config).to(device)
    env = create_env_from_config(config)

    env_config = config.get("env", {})
    max_steps = config.get("training", {}).get(
        "max_steps_per_episode", env_config.get("max_steps", 500)
    )

    dist_config = config.get("distributed", {})
    episodes_per_send = dist_config.get("episodes_per_send", 1)

    # 从共享内存加载初始权重
    _load_shared_weights(policy, shared_state_dict)
    local_weight_version = weight_version.value

    while not stop_event.is_set():
        with lock:
            if episode_counter.value >= target_episodes:
                break

        # 通过版本号检测权重更新（零拷贝读取共享内存）
        current_version = weight_version.value
        if current_version > local_weight_version:
            _load_shared_weights(policy, shared_state_dict)
            local_weight_version = current_version

        # 批量收集轨迹
        batch = []
        for _ in range(episodes_per_send):
            with lock:
                if episode_counter.value >= target_episodes:
                    break

            trajectory, total_reward, ep_length = collect_episode(
                policy, env, device, max_steps
            )
            batch.append({
                "trajectory": trajectory,
                "total_reward": total_reward,
                "episode_length": ep_length,
                "worker_id": worker_id,
            })

            with lock:
                episode_counter.value += 1

        if not batch:
            break

        try:
            trajectory_queue.put(batch, timeout=10.0)
        except queue.Full:
            pass

    env.close()


def _load_shared_weights(policy, shared_state_dict):
    """从共享内存张量加载权重到本地Policy。"""
    local_state = policy.state_dict()
    for key in local_state:
        local_state[key].copy_(shared_state_dict[key])
    policy.load_state_dict(local_state, assign=False)
