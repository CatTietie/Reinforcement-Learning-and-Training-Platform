"""
DistributedTrainer：编排分布式异步PPO训练的完整生命周期。
使用共享内存实现参数服务器模式的零拷贝权重同步。
"""

import multiprocessing as mp

import torch

from distributed.learner import Learner
from distributed.utils import create_policy_from_config
from distributed.worker import worker_process


class DistributedTrainer:
    """
    分布式训练编排器。
    参数服务器模式：Learner持有权威权重，通过共享内存广播给Worker。
    Worker通过版本号检测更新，零拷贝读取最新参数。
    """

    def __init__(self, config, experiment_id=None):
        self.config = config
        self.experiment_id = experiment_id or "distributed"

        dist_config = config.get("distributed", {})
        self.num_workers = dist_config.get("num_workers", 4)
        self.queue_maxsize = dist_config.get("queue_maxsize", 32)

        train_config = config.get("training", {})
        self.target_episodes = train_config.get("num_episodes", 100)

    def train(self):
        """
        执行分布式训练：
        1. 创建共享内存参数张量
        2. spawn Worker进程
        3. 运行Learner（主进程）
        4. 等待Worker退出
        """
        ctx = mp.get_context("spawn")

        # 创建共享内存state_dict（参数服务器核心）
        policy_template = create_policy_from_config(self.config)
        shared_state_dict = {}
        for key, tensor in policy_template.state_dict().items():
            shared_tensor = tensor.cpu().clone().share_memory_()
            shared_state_dict[key] = shared_tensor

        trajectory_queue = ctx.Queue(maxsize=self.queue_maxsize)
        weight_version = ctx.Value("i", 0)
        episode_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()

        # spawn workers
        workers = []
        for i in range(self.num_workers):
            p = ctx.Process(
                target=worker_process,
                args=(
                    i,
                    self.config,
                    trajectory_queue,
                    shared_state_dict,
                    weight_version,
                    episode_counter,
                    stop_event,
                    self.target_episodes,
                    lock,
                ),
                name=f"worker-{i}",
            )
            p.start()
            workers.append(p)

        # 主进程运行Learner
        learner = Learner(
            config=self.config,
            trajectory_queue=trajectory_queue,
            shared_state_dict=shared_state_dict,
            weight_version=weight_version,
            episode_counter=episode_counter,
            stop_event=stop_event,
            lock=lock,
            experiment_id=self.experiment_id,
        )

        try:
            result = learner.run()
        finally:
            stop_event.set()
            for p in workers:
                p.join(timeout=15.0)
                if p.is_alive():
                    p.terminate()

        return result
