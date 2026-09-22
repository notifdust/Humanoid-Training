from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from humanoid_training.errors import RecipeError, repo_root
from humanoid_training.spec import deep_merge, require_valid, spec_hash

_STUDIO_AVAILABILITY = {"cpu", "gpu"}


@dataclass(frozen=True)
class Recipe:
    id: str
    path: Path
    data: dict[str, Any]

    @property
    def title(self) -> str:
        return str(self.data.get("title") or self.id)

    @property
    def summary(self) -> str:
        return str(self.data.get("summary") or "")

    @property
    def phase(self) -> int:
        return int(self.data.get("phase", 0))

    def as_public_dict(self) -> dict[str, Any]:
        """What the studio is allowed to know. No hardcoded recipe ids in the UI."""
        studio = dict(self.data.get("studio") or {})
        runnable = bool(self.data.get("runnable", False))
        availability = str(studio.get("availability") or "").strip().lower()
        if availability not in _STUDIO_AVAILABILITY:
            availability = "cpu" if runnable else "gpu"
        start_here = studio.get("start_here")
        if start_here is None:
            start_here = availability == "cpu"
        launch_here = _launch_here(self, availability)
        if launch_here:
            promise = str(studio.get("ready_promise") or studio.get("promise") or self.summary)
            train_hint = str(studio.get("ready_hint") or studio.get("train_hint") or "")
        else:
            promise = str(studio.get("promise") or self.summary)
            train_hint = str(studio.get("train_hint") or "")
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "phase": self.phase,
            "language": self.data.get("language"),
            "robot": self.data.get("robot"),
            "runnable": runnable,
            "availability": availability,
            "launch_here": launch_here,
            "start_here": bool(start_here),
            "promise": promise,
            "train_hint": train_hint,
            "blocked_hint": str(studio.get("blocked_hint") or ""),
            "scene_hint": str(studio.get("scene_hint") or ""),
            "record_hint": str(studio.get("record_hint") or ""),
            "has_gold": (self.path / "gold" / "eval.mp4").is_file(),
            "has_scene": bool(self.data.get("scene")),
            "scene_preview": bool(
                ((self.data.get("adapters") or {}).get("mujoco") or {}).get("scene_preview")
            ),
            "imitate": (self.data.get("train") or {}).get("method") == "imitation",
            "method": str((self.data.get("train") or {}).get("method") or ""),
            "adapters": sorted((self.data.get("adapters") or {}).keys()),
        }


def recipes_dir() -> Path:
    return repo_root() / "recipes"


def load_recipe(recipe_id: str) -> Recipe:
    path = recipes_dir() / recipe_id / "recipe.yaml"
    if not path.is_file():
        known = ", ".join(r.id for r in list_recipes()) or "(none)"
        raise RecipeError(f"Unknown recipe '{recipe_id}'. Known: {known}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RecipeError(f"{path} must be a mapping")
    data.setdefault("id", recipe_id)
    _validate_studio(recipe_id, data)
    return Recipe(id=recipe_id, path=path.parent, data=data)


def _validate_studio(recipe_id: str, data: dict[str, Any]) -> None:
    studio = data.get("studio")
    if studio is None:
        return
    if not isinstance(studio, dict):
        raise RecipeError(f"{recipe_id}: studio must be a mapping")
    avail = studio.get("availability")
    if avail is not None and str(avail).strip().lower() not in _STUDIO_AVAILABILITY:
        raise RecipeError(f"{recipe_id}: studio.availability must be cpu or gpu")


def _gpu_engine_configured(data: dict[str, Any], name: str) -> bool:
    adapters = data.get("adapters") or {}
    cfg = adapters.get(name) or {}
    if not isinstance(cfg, dict) or cfg.get("unsupported"):
        return False
    if name == "playground":
        return bool(cfg.get("env_name") or cfg.get("train_command"))
    if name in {"mjlab", "isaaclab"}:
        return bool(cfg.get("task"))
    return False


def _launch_here(recipe: Recipe, availability: str) -> bool:
    """True when Train on this machine will launch, not merely compile-and-block.

    CPU recipes always launch. GPU recipes launch when any configured engine
    (playground / mjlab / isaaclab) is launch-ready on this host.
    """
    if availability == "cpu":
        return True
    from humanoid_training.hardware import adapter_launch_ready

    for name in ("playground", "mjlab", "isaaclab"):
        if _gpu_engine_configured(recipe.data, name) and adapter_launch_ready(name):
            return True
    return False


def public_catalog() -> dict[str, Any]:
    """Grouped recipe list. The studio projects this; it must not invent groups."""
    recipes = [r.as_public_dict() for r in list_recipes()]
    ready = [r for r in recipes if r["launch_here"]]
    later = [r for r in recipes if not r["launch_here"]]
    return {
        "recipes": recipes,
        "ready": [r["id"] for r in ready],
        "later": [r["id"] for r in later],
        "start_here": [r["id"] for r in recipes if r["start_here"]],
    }


def english_list(items: list[str]) -> str:
    names = [str(item) for item in items if item]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def list_recipes() -> list[Recipe]:
    root = recipes_dir()
    if not root.is_dir():
        return []
    recipes = []
    for child in sorted(root.iterdir()):
        if (child / "recipe.yaml").is_file():
            recipes.append(load_recipe(child.name))
    return recipes


def _recipe_defaults(recipe: Recipe) -> dict[str, Any]:
    data = recipe.data
    spec: dict[str, Any] = {
        "spec_version": "0.1.0",
        "name": recipe.id,
        "robot": {
            "id": data.get("robot"),
            "source": "catalog",
        },
        "task": {
            "recipe": recipe.id,
            "language": data.get("language"),
        },
        "train": dict(data.get("train") or {}),
        "eval": dict(data.get("eval") or {}),
        "backend": dict(data.get("backend") or {}),
    }
    if data.get("scene"):
        spec["scene"] = dict(data["scene"])
    if data.get("success"):
        spec["task"]["success"] = dict(data["success"])
    if data.get("data"):
        spec["data"] = dict(data["data"])
    return spec


def expand_spec(user_spec: dict[str, Any]) -> dict[str, Any]:
    """Fill recipe defaults, then overlay the user document. User wins."""
    recipe_id = (user_spec.get("task") or {}).get("recipe")
    if not recipe_id:
        raise RecipeError("task.recipe is required")
    recipe = load_recipe(str(recipe_id))
    expanded = deep_merge(_recipe_defaults(recipe), user_spec)
    expanded["task"]["recipe"] = recipe.id
    require_valid(expanded)
    expanded["_expanded"] = {
        "recipe_version": recipe.data.get("version"),
        "recipe_id": recipe.id,
        "spec_hash": spec_hash({k: v for k, v in expanded.items() if not k.startswith("_")}),
    }
    return expanded


def default_user_spec(recipe_id: str) -> dict[str, Any]:
    """The short document the studio would save for a beginner."""
    recipe = load_recipe(recipe_id)
    spec = {
        "spec_version": "0.1.0",
        "name": recipe.id,
        "robot": {"id": recipe.data.get("robot"), "source": "catalog"},
        "task": {"recipe": recipe.id},
        "train": {"method": (recipe.data.get("train") or {}).get("method", "rl")},
    }
    if recipe.data.get("scene"):
        spec["scene"] = dict(recipe.data["scene"])
    if recipe.data.get("backend"):
        spec["backend"] = dict(recipe.data["backend"])
    if recipe.data.get("data"):
        spec["data"] = dict(recipe.data["data"])
    return spec
