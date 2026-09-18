from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from humanoid_training.server import app


def test_health_and_recipes() -> None:
    client = TestClient(app)
    assert client.get("/api/health").json()["ok"] is True
    recipes = client.get("/api/recipes").json()["recipes"]
    ids = {r["id"] for r in recipes}
    assert "cartpole-balance" in ids
    assert "g1-walk" in ids
    page = client.get("/")
    assert page.status_code == 200
    assert "Humanoid Training" in page.text
    assert "Robots" in page.text
    assert "Data" in page.text


def test_inspect_dataset_fixture() -> None:
    client = TestClient(app)
    path = Path(__file__).resolve().parent / "fixtures" / "lerobot_tiny"
    body = client.post("/api/datasets/inspect", json={"uri": str(path)})
    assert body.status_code == 200
    data = body.json()
    assert data["ok"] is True
    assert data["total_episodes"] == 2


def test_api_train_cartpole(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HT_RUNS_DIR", str(tmp_path))
    # Re-importing not needed: default_runs_dir reads env each call.
    os.environ["HT_RUNS_DIR"] = str(tmp_path)
    client = TestClient(app)
    spec = {
        "spec_version": "0.1.0",
        "name": "cartpole-balance",
        "robot": {"id": "cartpole", "source": "catalog"},
        "task": {"recipe": "cartpole-balance"},
        "train": {"method": "rl", "steps": 30, "seed": 0},
        "eval": {"episodes": 1, "record_video": True},
        "backend": {"prefer": ["gymnasium"], "compute": "local"},
    }
    created = client.post("/api/runs", json={"spec": spec})
    assert created.status_code == 200, created.text
    run_id = created.json()["run_id"]
    deadline = time.time() + 60
    body = {}
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body.get("status") not in {"queued", "running"}:
            break
        time.sleep(0.2)
    assert body.get("status") in {"completed", "passed"}, body
    video = client.get(f"/api/runs/{run_id}/artifacts/eval.mp4")
    assert video.status_code == 200
    assert video.headers["content-type"].startswith("video/")
