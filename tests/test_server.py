from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from humanoid_training.server import app


def test_health_and_recipes() -> None:
    client = TestClient(app)
    assert client.get("/api/health").json()["ok"] is True
    catalog = client.get("/api/recipes").json()
    recipes = catalog["recipes"]
    ids = {r["id"] for r in recipes}
    assert "cartpole-balance" in ids
    assert "g1-walk" in ids
    assert "cartpole-balance" in catalog["ready"]
    assert "g1-walk" in catalog["later"]
    page = client.get("/")
    assert page.status_code == 200
    assert "Humanoid Training" in page.text
    assert "Robots" in page.text
    assert "Data" in page.text
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "Start here" in js.text
    assert "Train again" in js.text


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


def test_api_g1_walk_blocks_with_next_step(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HT_RUNS_DIR", str(tmp_path))
    os.environ["HT_RUNS_DIR"] = str(tmp_path)
    client = TestClient(app)
    from humanoid_training.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    created = client.post("/api/runs", json={"spec": spec})
    assert created.status_code == 200, created.text
    run_id = created.json()["run_id"]
    deadline = time.time() + 30
    body = {}
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body.get("status") not in {"queued", "running"}:
            break
        time.sleep(0.1)
    assert body.get("status") == "blocked", body
    err = body.get("error") or ""
    assert "Playground" in err
    assert "CPU studio" in err or "g1-stand" in err


def test_studio_js_projects_catalog_not_recipe_ids() -> None:
    """The browser must group and fall back from GET /api/recipes, not baked-in ids."""
    js = (Path(__file__).resolve().parents[1] / "studio" / "app.js").read_text(encoding="utf-8")
    assert "function worksHere" in js
    assert "function firstReadyRecipe" in js
    assert "function firstImitateRecipe" in js
    assert "/api/recipes/pick-and-place" not in js
    assert '"cartpole-balance"' not in js
    assert "run.recipe === \"pick-and-place\"" not in js
    assert "function keepTrainSummary" in js
    assert "function runFacts" in js
    assert "id=\"open-imitate\"" in js


def test_studio_js_bc_freshness_contract() -> None:
    """Runs UI must not stale-label honest linear-BC / keep_episodes notes."""
    js = (Path(__file__).resolve().parents[1] / "studio" / "app.js").read_text(encoding="utf-8")
    assert "function hasBcEvidence" in js
    assert "function runFacts" in js
    assert 'notes.includes("linear BC")' in js
    assert "keep_episodes=" in js
    assert "arm_ik=on" in js
    assert "Keep at least one episode" in js
    assert "keepEpisodes: null" in js
    assert "function rebindDatasetOntoStarter" in js
    assert "function clearDemoSession" in js
    assert "delete state.starter.data.keep_episodes" in js


def test_studio_css_disabled_cursor() -> None:
    css = (Path(__file__).resolve().parents[1] / "studio" / "styles.css").read_text(
        encoding="utf-8"
    )
    assert "cursor: not-allowed" in css
    assert "button.busy:disabled" in css
