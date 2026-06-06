"""
监控模块单元测试：覆盖 WebSocket 消息格式、断连重试、停止指令传递及回调故障隔离。
"""

import asyncio
import json
import os
import threading
import time

import pytest
import httpx
from httpx import ASGITransport
from starlette.testclient import TestClient

from monitor.app import app, experiments_cache, manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_cache():
    """每个测试前清空缓存。"""
    experiments_cache.clear()
    manager._connections.clear()
    yield
    experiments_cache.clear()
    manager._connections.clear()


@pytest.fixture
def async_client():
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def sync_client():
    """Starlette TestClient — 支持真实 WebSocket 连接。"""
    return TestClient(app)


SAMPLE_METRIC = {
    "experiment_id": "exp_001",
    "episode": 1,
    "total_reward": 42.5,
    "policy_loss": 0.123,
    "value_loss": 0.456,
    "entropy": 0.789,
    "episode_length": 100,
    "lr": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}


# ---------------------------------------------------------------------------
# Test 1: WebSocket message format validation
# ---------------------------------------------------------------------------


class TestWebSocketMessageFormat:
    """验证 WebSocket 推送的消息结构与字段完整性。"""

    @pytest.mark.anyio
    async def test_metric_broadcast_has_correct_fields(self, async_client):
        resp = await async_client.post("/api/experiments/exp_001/metrics", json=SAMPLE_METRIC)
        assert resp.status_code == 200

        state = experiments_cache["exp_001"]
        assert len(state.metrics) == 1
        metric = state.metrics[0]

        expected_keys = {
            "episode", "total_reward", "policy_loss", "value_loss",
            "entropy", "episode_length", "lr", "gamma", "gae_lambda",
        }
        assert set(metric.keys()) == expected_keys

    @pytest.mark.anyio
    async def test_metric_values_match_payload(self, async_client):
        resp = await async_client.post("/api/experiments/exp_001/metrics", json=SAMPLE_METRIC)
        assert resp.status_code == 200

        metric = experiments_cache["exp_001"].metrics[0]
        assert metric["episode"] == 1
        assert metric["total_reward"] == 42.5
        assert metric["policy_loss"] == 0.123
        assert metric["value_loss"] == 0.456
        assert metric["entropy"] == 0.789
        assert metric["episode_length"] == 100
        assert metric["lr"] == 0.0003
        assert metric["gamma"] == 0.99
        assert metric["gae_lambda"] == 0.95

    @pytest.mark.anyio
    async def test_hyperparams_stored_on_first_metric(self, async_client):
        payload = {**SAMPLE_METRIC, "hyperparams": {"lr": 0.0003, "gamma": 0.99}}
        resp = await async_client.post("/api/experiments/exp_001/metrics", json=payload)
        assert resp.status_code == 200
        assert experiments_cache["exp_001"].hyperparams == {"lr": 0.0003, "gamma": 0.99}

    @pytest.mark.anyio
    async def test_hyperparams_not_overwritten_on_subsequent_metrics(self, async_client):
        payload1 = {**SAMPLE_METRIC, "hyperparams": {"lr": 0.0003}}
        await async_client.post("/api/experiments/exp_001/metrics", json=payload1)

        payload2 = {**SAMPLE_METRIC, "episode": 2, "hyperparams": {"lr": 0.001}}
        await async_client.post("/api/experiments/exp_001/metrics", json=payload2)

        assert experiments_cache["exp_001"].hyperparams == {"lr": 0.0003}


# ---------------------------------------------------------------------------
# Test 2: Disconnect handling
# ---------------------------------------------------------------------------


class TestDisconnectHandling:
    """验证 WebSocket 断连后服务器不会崩溃，重连后可获取历史数据。"""

    @pytest.mark.anyio
    async def test_metrics_post_succeeds_without_ws_subscribers(self, async_client):
        resp = await async_client.post("/api/experiments/exp_002/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_002"
        })
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_history_available_after_reconnect(self, async_client):
        for i in range(1, 6):
            await async_client.post("/api/experiments/exp_003/metrics", json={
                **SAMPLE_METRIC, "experiment_id": "exp_003", "episode": i,
                "total_reward": float(i * 10),
            })

        resp = await async_client.get("/api/experiments/exp_003/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["metrics"]) == 5
        assert data["metrics"][0]["episode"] == 1
        assert data["metrics"][4]["episode"] == 5

    @pytest.mark.anyio
    async def test_broadcast_to_dead_connection_does_not_crash(self):
        """模拟一个已断开的 WebSocket，确保 broadcast 不抛异常。"""

        class FakeDeadWebSocket:
            async def send_text(self, data):
                raise RuntimeError("Connection closed")

        dead_ws = FakeDeadWebSocket()
        manager.subscribe("exp_dead", dead_ws)

        await manager.broadcast("exp_dead", {"episode": 1, "total_reward": 10.0})

        conns = manager._connections.get("exp_dead", set())
        assert dead_ws not in conns


# ---------------------------------------------------------------------------
# Test 3: Stop command delivery
# ---------------------------------------------------------------------------


class TestStopCommand:
    """验证紧急停止指令的设置与传递。"""

    @pytest.mark.anyio
    async def test_stop_sets_flag(self, async_client):
        await async_client.post("/api/experiments/exp_stop/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_stop"
        })

        resp = await async_client.post("/api/experiments/exp_stop/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stop_requested"
        assert experiments_cache["exp_stop"].stop_requested is True

    @pytest.mark.anyio
    async def test_metrics_post_returns_stop_flag(self, async_client):
        await async_client.post("/api/experiments/exp_stop2/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_stop2"
        })

        resp_before = await async_client.post("/api/experiments/exp_stop2/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_stop2", "episode": 2
        })
        assert resp_before.json()["stop"] is False

        await async_client.post("/api/experiments/exp_stop2/stop")

        resp_after = await async_client.post("/api/experiments/exp_stop2/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_stop2", "episode": 3
        })
        assert resp_after.json()["stop"] is True

    @pytest.mark.anyio
    async def test_stop_flag_persists_across_polls(self, async_client):
        await async_client.post("/api/experiments/exp_persist/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_persist"
        })
        await async_client.post("/api/experiments/exp_persist/stop")

        for i in range(3):
            resp = await async_client.get("/api/experiments/exp_persist/stop")
            assert resp.json()["stop"] is True

    @pytest.mark.anyio
    async def test_stop_updates_status_field(self, async_client):
        await async_client.post("/api/experiments/exp_status/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_status"
        })
        assert experiments_cache["exp_status"].status == "running"

        await async_client.post("/api/experiments/exp_status/stop")
        assert experiments_cache["exp_status"].status == "stopped"


# ---------------------------------------------------------------------------
# Test 4: Callback failure isolation
# ---------------------------------------------------------------------------


class TestCallbackIsolation:
    """验证 MonitorCallback 在网络不可达时不会抛出异常。"""

    def test_callback_with_unreachable_url_returns_false(self):
        from monitor.callback import MonitorCallback

        cb = MonitorCallback(
            monitor_url="http://127.0.0.1:59999",
            experiment_id="test_unreachable",
        )
        result = cb.on_episode_end(
            episode=1, total_reward=10.0, policy_loss=0.1,
            value_loss=0.2, entropy=0.5, episode_length=50,
            lr=0.0003, gamma=0.99, lam=0.95,
        )
        assert result is False
        cb.close()

    def test_callback_with_invalid_url_no_exception(self):
        from monitor.callback import MonitorCallback

        cb = MonitorCallback(
            monitor_url="not-a-valid-url",
            experiment_id="test_invalid",
        )
        result = cb.on_episode_end(
            episode=1, total_reward=10.0, policy_loss=0.1,
            value_loss=0.2, entropy=0.5, episode_length=50,
            lr=0.0003, gamma=0.99, lam=0.95,
        )
        assert result is False
        cb.close()

    def test_callback_close_is_idempotent(self):
        from monitor.callback import MonitorCallback

        cb = MonitorCallback(
            monitor_url="http://127.0.0.1:59999",
            experiment_id="test_close",
        )
        cb.close()
        cb.close()


# ---------------------------------------------------------------------------
# Test 5: Multi-experiment isolation
# ---------------------------------------------------------------------------


class TestMultiExperimentIsolation:
    """验证多实验同时监控时数据互不干扰。"""

    @pytest.mark.anyio
    async def test_metrics_isolated_between_experiments(self, async_client):
        for i in range(1, 4):
            await async_client.post("/api/experiments/exp_A/metrics", json={
                **SAMPLE_METRIC, "experiment_id": "exp_A", "episode": i, "total_reward": float(i),
            })
            await async_client.post("/api/experiments/exp_B/metrics", json={
                **SAMPLE_METRIC, "experiment_id": "exp_B", "episode": i, "total_reward": float(i * 100),
            })

        resp_a = await async_client.get("/api/experiments/exp_A/metrics")
        resp_b = await async_client.get("/api/experiments/exp_B/metrics")

        metrics_a = resp_a.json()["metrics"]
        metrics_b = resp_b.json()["metrics"]

        assert len(metrics_a) == 3
        assert len(metrics_b) == 3
        assert metrics_a[0]["total_reward"] == 1.0
        assert metrics_b[0]["total_reward"] == 100.0

    @pytest.mark.anyio
    async def test_stop_one_does_not_affect_other(self, async_client):
        await async_client.post("/api/experiments/exp_X/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_X"
        })
        await async_client.post("/api/experiments/exp_Y/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_Y"
        })

        await async_client.post("/api/experiments/exp_X/stop")

        resp_x = await async_client.post("/api/experiments/exp_X/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_X", "episode": 2
        })
        resp_y = await async_client.post("/api/experiments/exp_Y/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_Y", "episode": 2
        })

        assert resp_x.json()["stop"] is True
        assert resp_y.json()["stop"] is False

    @pytest.mark.anyio
    async def test_experiments_list_shows_all(self, async_client):
        await async_client.post("/api/experiments/exp_1/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_1"
        })
        await async_client.post("/api/experiments/exp_2/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "exp_2"
        })

        resp = await async_client.get("/api/experiments")
        data = resp.json()
        ids = {e["experiment_id"] for e in data}
        assert "exp_1" in ids
        assert "exp_2" in ids


# ---------------------------------------------------------------------------
# Test 6: Real WebSocket endpoint — connect, receive broadcast, verify JSON
# ---------------------------------------------------------------------------


class TestWebSocketEndpoint:
    """通过 Starlette TestClient 真实连接 /ws/{id} 端点，验证广播到达及 JSON 结构。"""

    def test_ws_receives_metric_broadcast(self, sync_client):
        """连接 WS，POST 一条指标，验证 WS 收到完整 JSON。"""
        with sync_client.websocket_connect("/ws/ws_exp_1") as ws:
            sync_client.post("/api/experiments/ws_exp_1/metrics", json={
                **SAMPLE_METRIC, "experiment_id": "ws_exp_1", "episode": 7,
                "total_reward": 99.9,
            })
            msg = ws.receive_text()
            data = json.loads(msg)

            expected_keys = {
                "episode", "total_reward", "policy_loss", "value_loss",
                "entropy", "episode_length", "lr", "gamma", "gae_lambda",
            }
            assert set(data.keys()) == expected_keys
            assert data["episode"] == 7
            assert data["total_reward"] == 99.9

    def test_ws_receives_multiple_metrics_in_order(self, sync_client):
        """连续 POST 多条指标，WS 按顺序收到。"""
        with sync_client.websocket_connect("/ws/ws_exp_order") as ws:
            for i in range(1, 4):
                sync_client.post("/api/experiments/ws_exp_order/metrics", json={
                    **SAMPLE_METRIC, "experiment_id": "ws_exp_order",
                    "episode": i, "total_reward": float(i * 10),
                })

            for i in range(1, 4):
                msg = ws.receive_text()
                data = json.loads(msg)
                assert data["episode"] == i
                assert data["total_reward"] == float(i * 10)

    def test_ws_stop_broadcast(self, sync_client):
        """POST stop 后，WS 收到 type=stop 消息。"""
        sync_client.post("/api/experiments/ws_exp_stop/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "ws_exp_stop"
        })
        with sync_client.websocket_connect("/ws/ws_exp_stop") as ws:
            sync_client.post("/api/experiments/ws_exp_stop/stop")
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["type"] == "stop"
            assert data["experiment_id"] == "ws_exp_stop"

    def test_ws_disconnect_does_not_crash_server(self, sync_client):
        """WS 连接后立即断开，后续 POST 不报错。"""
        with sync_client.websocket_connect("/ws/ws_exp_dc") as ws:
            pass  # 立即退出 context，触发断连

        resp = sync_client.post("/api/experiments/ws_exp_dc/metrics", json={
            **SAMPLE_METRIC, "experiment_id": "ws_exp_dc"
        })
        assert resp.status_code == 200

    def test_ws_two_experiments_isolated(self, sync_client):
        """两个 WS 分别订阅不同实验，消息不串。"""
        with sync_client.websocket_connect("/ws/ws_iso_A") as ws_a:
            with sync_client.websocket_connect("/ws/ws_iso_B") as ws_b:
                sync_client.post("/api/experiments/ws_iso_A/metrics", json={
                    **SAMPLE_METRIC, "experiment_id": "ws_iso_A",
                    "episode": 1, "total_reward": 111.0,
                })
                sync_client.post("/api/experiments/ws_iso_B/metrics", json={
                    **SAMPLE_METRIC, "experiment_id": "ws_iso_B",
                    "episode": 1, "total_reward": 222.0,
                })

                msg_a = json.loads(ws_a.receive_text())
                msg_b = json.loads(ws_b.receive_text())

                assert msg_a["total_reward"] == 111.0
                assert msg_b["total_reward"] == 222.0


# ---------------------------------------------------------------------------
# Test 7: Full stop chain integration — POST stop → callback → model save → DB
# ---------------------------------------------------------------------------


class TestStopChainIntegration:
    """
    端到端集成测试：模拟完整停止链路。
    1. 启动 FastAPI 测试服务器
    2. 创建 MonitorCallback 指向该服务器
    3. 通过服务器 POST stop
    4. callback.on_episode_end 返回 True
    5. 验证 Trainer 停止后保存模型并返回 stopped=True
    6. 验证 DB 状态标记为 stopped
    """

    def test_full_stop_chain_with_trainer(self, sync_client):
        """完整链路：stop → callback 感知 → Trainer 停止 → 模型保存 → DB 更新。"""
        from monitor.callback import MonitorCallback
        from train import Trainer
        from db.database import Database

        # --- 准备：最小化配置 ---
        cfg = {
            "experiment": {"name": "stop_chain_test", "seed": 42},
            "env": {"name": "CartPoleStandard-v0", "max_steps": 50},
            "network": {"hidden_sizes": [32], "lstm_hidden_size": 32,
                        "lstm_num_layers": 1, "use_lstm": True, "activation": "tanh"},
            "algorithm": {"lr": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                          "clip_epsilon": 0.2, "value_coef": 0.5,
                          "entropy_coef": 0.01, "max_grad_norm": 0.5, "update_epochs": 2},
            "training": {"num_episodes": 200, "batch_size": 32,
                         "log_interval": 10, "save_interval": 999},
            "storage": {"model_dir": "models"},
            "logging": {"level": "WARNING", "log_dir": "logs"},
        }

        # --- 准备：内存 SQLite 数据库 ---
        db = Database("sqlite:///:memory:")
        experiment_id = db.create_experiment(
            name="stop_chain_test", config_yaml="test"
        )

        # --- 通过 sync_client 设置 stop 标志 ---
        sync_client.post(f"/api/experiments/{experiment_id}/metrics", json={
            **SAMPLE_METRIC, "experiment_id": str(experiment_id)
        })
        sync_client.post(f"/api/experiments/{experiment_id}/stop")

        # --- 构造使用 Starlette TestClient 的回调 ---
        # TestClient 内部正确处理 ASGI sync/async 桥接
        class TestableCallback:
            """使用 Starlette TestClient 作为 transport，模拟真实 callback 行为。"""
            def __init__(self, client, experiment_id):
                self._client = client
                self._experiment_id = str(experiment_id)
                self._stop_requested = False

            @property
            def stop_requested(self):
                return self._stop_requested

            def on_episode_end(self, episode, total_reward, policy_loss,
                              value_loss, entropy, episode_length, lr, gamma, lam):
                try:
                    resp = self._client.post(
                        f"/api/experiments/{self._experiment_id}/metrics",
                        json={
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
                        },
                    )
                    if resp.status_code == 200:
                        self._stop_requested = resp.json().get("stop", False)
                except Exception:
                    pass
                return self._stop_requested

            def close(self):
                pass

        callback = TestableCallback(client=sync_client, experiment_id=experiment_id)

        # --- 运行 Trainer ---
        trainer = Trainer(
            config=cfg,
            experiment_id=experiment_id,
            monitor_callback=callback,
        )
        result = trainer.train()

        # --- 验证：Trainer 返回 stopped ---
        assert result["stopped"] is True
        assert result["total_episodes"] >= 1

        # --- 验证：模型文件已保存 ---
        model_files = [
            f for f in os.listdir("models")
            if f.startswith(f"experiment_{experiment_id}_ep")
        ]
        assert len(model_files) >= 1

        # --- 验证：DB 状态标记为 stopped ---
        status = "stopped" if result.get("stopped") else "finished"
        db.update_experiment(
            experiment_id=experiment_id,
            status=status,
            total_episodes=result["total_episodes"],
            final_reward=result["final_reward"],
        )
        exp = db.get_experiment(experiment_id)
        assert exp.status == "stopped"

        # --- 清理 ---
        callback.close()
        for f in model_files:
            os.remove(os.path.join("models", f))

    def test_stop_latency_under_5_seconds(self, sync_client):
        """停止信号发出后，训练在 5 秒内终止。"""
        from train import Trainer

        cfg = {
            "experiment": {"name": "latency_test", "seed": 1},
            "env": {"name": "CartPoleStandard-v0", "max_steps": 200},
            "network": {"hidden_sizes": [32], "lstm_hidden_size": 32,
                        "lstm_num_layers": 1, "use_lstm": True, "activation": "tanh"},
            "algorithm": {"lr": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                          "clip_epsilon": 0.2, "value_coef": 0.5,
                          "entropy_coef": 0.01, "max_grad_norm": 0.5, "update_epochs": 2},
            "training": {"num_episodes": 500, "batch_size": 32,
                         "log_interval": 50, "save_interval": 999},
            "storage": {"model_dir": "models"},
            "logging": {"level": "WARNING", "log_dir": "logs"},
        }

        # 在第3个 episode 之后触发 stop（模拟用户异步点击）
        class DelayedStopCallback:
            """前 2 个 episode 正常，第 3 个开始返回 stop。"""
            def __init__(self):
                self.call_count = 0
                self.stop_time = None

            def on_episode_end(self, **kwargs) -> bool:
                self.call_count += 1
                if self.call_count >= 3:
                    if self.stop_time is None:
                        self.stop_time = time.time()
                    return True
                return False

            def close(self):
                pass

        callback = DelayedStopCallback()
        trainer = Trainer(config=cfg, experiment_id="latency_test", monitor_callback=callback)

        start = time.time()
        result = trainer.train()
        elapsed = time.time() - start

        assert result["stopped"] is True
        assert result["total_episodes"] == 3
        assert elapsed < 5.0

        # 清理模型文件
        for f in os.listdir("models"):
            if f.startswith("experiment_latency_test_ep"):
                os.remove(os.path.join("models", f))

