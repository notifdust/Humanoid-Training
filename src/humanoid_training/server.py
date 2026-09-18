from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from humanoid_training.catalog import load_robot_catalog
from humanoid_training.errors import RecipeError, SpecError, repo_root
from humanoid_training.recipes import default_user_spec, expand_spec, list_recipes, load_recipe
from humanoid_training.runner import default_runs_dir, load_manifest, new_run_id, run_job
from humanoid_training.spec import validate_spec

app = FastAPI(title="Humanoid Training Studio", version="0.1.0")


class SpecBody(BaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in spec.items() if not str(k).startswith("_")}


def _runs_root() -> Path:
    return default_runs_dir()


def _find_run(run_id: str) -> Path:
    path = _runs_root() / run_id
    if not path.is_dir() or not (path / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail=f"Unknown run {run_id}")
    return path


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "name": "humanoid-training"}


@app.get("/api/robots")
def robots() -> dict[str, Any]:
    return {"robots": load_robot_catalog()}


@app.get("/api/recipes")
def recipes() -> dict[str, Any]:
    return {"recipes": [r.as_public_dict() for r in list_recipes()]}


@app.get("/api/recipes/{recipe_id}")
def recipe_detail(recipe_id: str) -> dict[str, Any]:
    try:
        recipe = load_recipe(recipe_id)
    except RecipeError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return {
        "recipe": recipe.as_public_dict(),
        "defaults": recipe.data,
        "starter_spec": default_user_spec(recipe_id),
    }


@app.post("/api/specs/validate")
def api_validate(body: SpecBody) -> dict[str, Any]:
    errors = validate_spec(body.spec)
    return {"ok": not errors, "errors": errors}


@app.post("/api/specs/expand")
def api_expand(body: SpecBody) -> dict[str, Any]:
    try:
        expanded = expand_spec(body.spec)
    except (SpecError, RecipeError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return {"spec": _public_spec(expanded), "meta": expanded.get("_expanded")}


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    root = _runs_root()
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for child in sorted(root.iterdir(), reverse=True):
        manifest_path = child / "manifest.json"
        if manifest_path.is_file():
            items.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    return {"runs": items}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    path = _find_run(run_id)
    manifest = load_manifest(path)
    log_path = path / "run.log"
    manifest["log"] = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    return manifest


@app.get("/api/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str):
    if name not in {"eval.mp4", "manifest.json", "spec.json", "run.log", "checkpoint.npz"}:
        raise HTTPException(status_code=400, detail="Unknown artifact")
    path = _find_run(run_id) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not produced for this run")
    media = "video/mp4" if name.endswith(".mp4") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=name)


@app.post("/api/runs")
def start_run(body: SpecBody) -> dict[str, Any]:
    try:
        expanded = expand_spec(body.spec)
    except (SpecError, RecipeError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    public = _public_spec(expanded)
    run_id = new_run_id(str(public.get("name") or "job"))
    runs_dir = _runs_root()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    queued = {
        "run_id": run_id,
        "status": "queued",
        "recipe": (public.get("task") or {}).get("recipe"),
        "run_dir": str(run_dir),
        "error": None,
        "metrics": {},
        "artifacts": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(queued, indent=2), encoding="utf-8")

    def _work() -> None:
        run_job(body.spec, runs_dir=runs_dir, log=None, run_id=run_id)

    threading.Thread(target=_work, daemon=True).start()
    return queued


studio_dir = repo_root() / "studio"
if studio_dir.is_dir():
    app.mount("/", StaticFiles(directory=studio_dir, html=True), name="studio")
