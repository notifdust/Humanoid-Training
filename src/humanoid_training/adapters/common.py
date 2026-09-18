from __future__ import annotations

from pathlib import Path
from typing import Any

from humanoid_training.errors import RecipeError
from humanoid_training.recipes import load_recipe


def recipe_adapter_config(spec: dict[str, Any], adapter_name: str) -> dict[str, Any]:
    recipe_id = (spec.get("task") or {}).get("recipe")
    if not recipe_id:
        raise RecipeError("task.recipe is required")
    recipe = load_recipe(str(recipe_id))
    adapters = recipe.data.get("adapters") or {}
    return dict(adapters.get(adapter_name) or {})


def ignored_scene_fields(spec: dict[str, Any], honors_scene: bool) -> list[str]:
    if honors_scene:
        return []
    scene = spec.get("scene") or {}
    ignored = []
    if scene.get("template"):
        ignored.append("scene.template")
    if scene.get("objects"):
        ignored.append("scene.objects")
    return ignored


def write_payload_files(run_dir: Path, payload: Any) -> None:
    run_dir = Path(run_dir)
    for rel, content in payload.files.items():
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
