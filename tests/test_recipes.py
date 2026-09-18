from __future__ import annotations

from pathlib import Path

from humanoid_training.recipes import expand_spec, list_recipes
from humanoid_training.spec import load_spec


def test_recipe_catalog_contains_phase0() -> None:
    ids = {r.id for r in list_recipes()}
    assert {"cartpole-balance", "g1-walk", "pick-and-place"} <= ids


def test_expand_fills_gymnasium_defaults() -> None:
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    )
    expanded = expand_spec(spec)
    assert expanded["backend"]["prefer"] == ["gymnasium"]
    assert expanded["train"]["method"] == "rl"
    assert expanded["task"]["success"]["min_return"] == 400
    assert expanded["_expanded"]["recipe_id"] == "cartpole-balance"


def test_mustard_example_expands() -> None:
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-mustard-in-bowl.json"
    )
    expanded = expand_spec(spec)
    assert expanded["task"]["recipe"] == "pick-and-place"
    assert expanded["train"]["method"] == "imitation"
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    )
    spec["train"]["steps"] = 12
    expanded = expand_spec(spec)
    assert expanded["train"]["steps"] == 12
