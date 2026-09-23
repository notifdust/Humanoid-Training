from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from humanoid_training import __version__
from humanoid_training.catalog import load_robot_catalog
from humanoid_training.datasets import inspect_lerobot_dataset
from humanoid_training.demos import (
    dataset_cache_dir,
    record_object_trajectories,
    record_scripted_pick_place,
)
from humanoid_training.errors import AdapterUnavailable, RecipeError, SpecError, repo_root
from humanoid_training.artifacts import SERVED_ARTIFACTS
from humanoid_training.deploy import assess_deploy, deploy_run
from humanoid_training.recipes import default_user_spec, expand_spec, load_recipe, public_catalog
from humanoid_training.runner import default_runs_dir, load_manifest, new_run_id, run_job, write_manifest
from humanoid_training.spec import validate_spec

app = FastAPI(title="Humanoid Training Studio", version="0.1.0")


@app.middleware("http")
async def studio_no_store(request, call_next):
    """Studio JS/CSS must not stick after a pull. Eval videos stay cacheable under /api."""
    response = await call_next(request)
    if request.url.path in {"/", "/index.html", "/app.js", "/styles.css"}:
        response.headers["Cache-Control"] = "no-store"
    return response


class SpecBody(BaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)


class DatasetBody(BaseModel):
    uri: str = ""


class RecordBody(BaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)
    dest: str = ""
    episodes: int = 4
    include_failure: bool = True
    trajectories: list[Any] = Field(default_factory=list)


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
    from humanoid_training.hardware import engine_status

    return {
        "ok": True,
        "name": "humanoid-training",
        "version": __version__,
        "engines": engine_status(),
    }


@app.get("/api/robots")
def robots() -> dict[str, Any]:
    return {"robots": load_robot_catalog()}


@app.post("/api/datasets/inspect")
def api_inspect_dataset(body: DatasetBody) -> dict[str, Any]:
    return inspect_lerobot_dataset(body.uri)


@app.post("/api/datasets/record")
def api_record_dataset(body: RecordBody) -> dict[str, Any]:
    try:
        expanded = expand_spec(body.spec)
        dest = Path(body.dest).expanduser() if body.dest else dataset_cache_dir(
            str(expanded.get("name") or "dataset") + ("-canvas" if body.trajectories else "")
        )
        if not dest.is_absolute():
            dest = Path.cwd() / dest
        if body.trajectories:
            result = record_object_trajectories(expanded, dest, body.trajectories)
        else:
            result = record_scripted_pick_place(
                expanded,
                dest,
                episodes=max(1, min(int(body.episodes), 12)),
                include_failure=bool(body.include_failure),
                seed=int((expanded.get("train") or {}).get("seed") or 1),
            )
    except (SpecError, RecipeError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    result["dest"] = str(dest)
    return result


@app.get("/api/recipes")
def recipes() -> dict[str, Any]:
    return public_catalog()


@app.get("/api/recipes/{recipe_id}/gold/{name}")
def recipe_gold(recipe_id: str, name: str):
    from humanoid_training.gold import GOLD_FILES, gold_dir

    if name not in GOLD_FILES:
        raise HTTPException(status_code=400, detail="Unknown gold file")
    try:
        recipe = load_recipe(recipe_id)
    except RecipeError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    path = gold_dir(recipe) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{recipe_id} has no gold/{name}")
    if name.endswith(".mp4"):
        media = "video/mp4"
    else:
        media = "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media, filename=name)


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
            try:
                items.append(load_manifest(child))
            except Exception:
                continue
    return {"runs": items}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    path = _find_run(run_id)
    manifest = load_manifest(path)
    log_path = path / "run.log"
    manifest["log"] = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    return manifest


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str):
    path = _find_run(run_id)
    log_path = path / "run.log"

    def generate():
        import time

        pos = 0
        while True:
            chunk = ""
            if log_path.is_file():
                text = log_path.read_text(encoding="utf-8")
                if len(text) > pos:
                    chunk = text[pos:]
                    pos = len(text)
            try:
                man = load_manifest(path)
            except Exception:
                man = {"status": "running"}
            payload = {
                "status": man.get("status"),
                "chunk": chunk,
                "metrics": man.get("metrics") or {},
                "error": man.get("error"),
                "artifacts": man.get("artifacts") or {},
                "notes": man.get("notes") or [],
                "facts": man.get("facts") or {},
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if man.get("status") not in {None, "queued", "running"}:
                break
            time.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str):
    if name not in SERVED_ARTIFACTS:
        raise HTTPException(status_code=400, detail="Unknown artifact")
    path = _find_run(run_id) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not produced for this run")
    if name.endswith(".mp4"):
        media = "video/mp4"
    elif name.endswith(".xml"):
        media = "application/xml"
    elif name.endswith(".json"):
        media = "application/json"
    else:
        media = "application/octet-stream"
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
        "facts": {},
        "artifacts": {},
    }
    write_manifest(run_dir, queued)

    def _work() -> None:
        run_job(body.spec, runs_dir=runs_dir, log=None, run_id=run_id)

    threading.Thread(target=_work, daemon=True).start()
    return queued


@app.get("/api/runs/{run_id}/deploy")
def assess_deploy_api(run_id: str) -> dict[str, Any]:
    """Phase 4 preflight: assess only. Does not attempt deploy."""
    path = _find_run(run_id)
    report = assess_deploy(load_manifest(path))
    return {**report, "deployed": False}


@app.post("/api/runs/{run_id}/deploy")
def deploy_run_api(run_id: str) -> dict[str, Any]:
    """Phase 4 gate: assess + fail closed. Never starts robot torque."""
    path = _find_run(run_id)
    manifest = load_manifest(path)
    report = assess_deploy(manifest)
    try:
        deploy_run(run_id, runs_dir=_runs_root())
    except AdapterUnavailable as err:
        return {
            **report,
            "ok": False,
            "deployed": False,
            "error": str(err),
        }
    return {**report, "deployed": True, "error": None}


studio_dir = repo_root() / "studio"
if studio_dir.is_dir():
    app.mount("/", StaticFiles(directory=studio_dir, html=True), name="studio")
