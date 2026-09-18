from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from humanoid_training.errors import RecipeError, repo_root
from humanoid_training.spec import deep_merge, require_valid, spec_hash


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
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "phase": self.phase,
            "language": self.data.get("language"),
            "robot": self.data.get("robot"),
            "runnable": bool(self.data.get("runnable", False)),
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
    return Recipe(id=recipe_id, path=path.parent, data=data)


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
    return spec
