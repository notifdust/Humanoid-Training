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


def test_record_dataset_api(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HT_CACHE", str(tmp_path / "cache"))
    client = TestClient(app)
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "imitation"},
    }
    body = client.post(
        "/api/datasets/record",
        json={"spec": spec, "episodes": 3, "include_failure": True},
    )
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["ok"] is True
    assert data["total_episodes"] == 3
    assert (Path(data["path"]) / "meta" / "info.json").is_file()


def test_record_canvas_trajectories_api(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HT_CACHE", str(tmp_path / "cache"))
    client = TestClient(app)
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "imitation"},
        "scene": {
            "template": "kitchen-counter-v1",
            "objects": [
                {"id": "mustard", "asset": "ycb-mustard", "x": -0.18, "y": 0.04},
                {"id": "bowl", "asset": "bowl-white", "x": 0.16, "y": -0.02},
            ],
        },
    }
    mustard = spec["scene"]["objects"][0]
    bowl = spec["scene"]["objects"][1]
    traj = [
        {
            "x": mustard["x"] + (t / 11) * (bowl["x"] - mustard["x"]),
            "y": mustard["y"] + (t / 11) * (bowl["y"] - mustard["y"]),
        }
        for t in range(12)
    ]
    body = client.post(
        "/api/datasets/record",
        json={"spec": spec, "trajectories": [traj]},
    )
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["ok"] is True
    assert data["total_episodes"] == 1
    assert data["episodes"][0]["success"] is True
    assert "canvas" in data["path"]


def test_record_without_objects_is_400(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HT_CACHE", str(tmp_path / "cache"))
    client = TestClient(app)
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "imitation"},
        "scene": {"template": "kitchen-counter-v1", "objects": []},
    }
    body = client.post("/api/datasets/record", json={"spec": spec, "episodes": 2})
    assert body.status_code == 400
    assert "scene.objects" in body.text


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
    assert body.get("notes")
    video = client.get(f"/api/runs/{run_id}/artifacts/eval.mp4")
    assert video.status_code == 200
    assert video.headers["content-type"].startswith("video/")
