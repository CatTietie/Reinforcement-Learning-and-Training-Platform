"""
FastAPI 监控后端：提供 REST API、WebSocket 实时推送、静态前端服务。
"""

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="RL Training Monitor")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExperimentState:
    experiment_id: str
    hyperparams: dict = field(default_factory=dict)
    metrics: deque = field(default_factory=lambda: deque(maxlen=1000))
    stop_requested: bool = False
    status: str = "running"


class MetricPayload(BaseModel):
    experiment_id: str | None = None
    episode: int
    total_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    episode_length: int
    lr: float
    gamma: float
    gae_lambda: float
    hyperparams: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

experiments_cache: dict[str, ExperimentState] = {}


def get_or_create_experiment(experiment_id: str) -> ExperimentState:
    if experiment_id not in experiments_cache:
        experiments_cache[experiment_id] = ExperimentState(experiment_id=experiment_id)
    return experiments_cache[experiment_id]


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = {}

    def subscribe(self, experiment_id: str, ws: WebSocket):
        self._connections.setdefault(experiment_id, set()).add(ws)

    def unsubscribe(self, experiment_id: str, ws: WebSocket):
        conns = self._connections.get(experiment_id)
        if conns:
            conns.discard(ws)
            if not conns:
                del self._connections[experiment_id]

    async def broadcast(self, experiment_id: str, message: dict):
        conns = self._connections.get(experiment_id)
        if not conns:
            return
        payload = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.discard(ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/experiments")
async def list_experiments():
    return [
        {
            "experiment_id": state.experiment_id,
            "status": state.status,
            "episodes": len(state.metrics),
        }
        for state in experiments_cache.values()
    ]


@app.get("/api/experiments/{experiment_id}/metrics")
async def get_metrics(experiment_id: str):
    state = experiments_cache.get(experiment_id)
    if state is None:
        return {"metrics": []}
    return {"metrics": list(state.metrics)}


@app.get("/api/experiments/{experiment_id}/hyperparams")
async def get_hyperparams(experiment_id: str):
    state = experiments_cache.get(experiment_id)
    if state is None:
        return {"hyperparams": {}}
    return {"hyperparams": state.hyperparams}


@app.post("/api/experiments/{experiment_id}/metrics")
async def post_metrics(experiment_id: str, payload: MetricPayload):
    state = get_or_create_experiment(experiment_id)

    if payload.hyperparams and not state.hyperparams:
        state.hyperparams = payload.hyperparams

    metric = {
        "episode": payload.episode,
        "total_reward": payload.total_reward,
        "policy_loss": payload.policy_loss,
        "value_loss": payload.value_loss,
        "entropy": payload.entropy,
        "episode_length": payload.episode_length,
        "lr": payload.lr,
        "gamma": payload.gamma,
        "gae_lambda": payload.gae_lambda,
    }
    state.metrics.append(metric)

    await manager.broadcast(experiment_id, metric)

    return {"stop": state.stop_requested}


@app.post("/api/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    state = get_or_create_experiment(experiment_id)
    state.stop_requested = True
    state.status = "stopped"
    await manager.broadcast(experiment_id, {"type": "stop", "experiment_id": experiment_id})
    return {"status": "stop_requested", "experiment_id": experiment_id}


@app.get("/api/experiments/{experiment_id}/stop")
async def check_stop(experiment_id: str):
    state = experiments_cache.get(experiment_id)
    if state is None:
        return {"stop": False}
    return {"stop": state.stop_requested}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/{experiment_id}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str):
    await websocket.accept()
    manager.subscribe(experiment_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(experiment_id, websocket)


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
