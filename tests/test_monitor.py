"""
监控模块单元测试：覆盖 WebSocket 消息格式、断连重试、停止指令传递及回调故障隔离。
"""

import asyncio

import pytest
import httpx
from httpx import ASGITransport

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
