"""
MonitorCallback: 训练回调，将 episode 指标实时推送至监控后端。
所有网络异常均被静默捕获，绝不中断训练。
"""

import httpx


class MonitorCallback:

    def __init__(self, monitor_url: str, experiment_id: str, hyperparams: dict | None = None):
        self._base_url = monitor_url.rstrip("/")
        self._experiment_id = str(experiment_id)
        self._hyperparams = hyperparams
        self._stop_requested = False
        self._first_call = True
        try:
            self._client = httpx.Client(timeout=2.0)
        except Exception:
            self._client = None

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def on_episode_end(
        self,
        episode: int,
        total_reward: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        episode_length: int,
        lr: float,
        gamma: float,
        lam: float,
    ) -> bool:
        """
        推送指标到监控服务器。返回 True 表示收到停止指令。
        任何异常都不会向外抛出。
        """
        if self._client is None:
            return self._stop_requested

        try:
            payload: dict = {
                "experiment_id": self._experiment_id,
                "episode": episode,
                "total_reward": total_reward,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "episode_length": episode_length,
                "lr": lr,
                "gamma": gamma,
                "gae_lambda": lam,
            }

            if self._first_call and self._hyperparams:
                payload["hyperparams"] = self._hyperparams
                self._first_call = False

            resp = self._client.post(
                f"{self._base_url}/api/experiments/{self._experiment_id}/metrics",
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._stop_requested = data.get("stop", False)
        except Exception:
            pass

        return self._stop_requested

    def close(self):
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
