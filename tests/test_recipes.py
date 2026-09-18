from __future__ import annotations

from pathlib import Path

from humanoid_training.recipes import expand_spec, list_recipes
from humanoid_training.spec import load_spec


def test_recipe_catalog_contains_core() -> None:
    ids = {r.id for r in list_recipes()}
    assert {"cartpole-balance", "g1-walk", "g1-stand", "g1-reach", "pick-and-place"} <= ids


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


def test_user_overlay_wins() -> None:
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    )
    spec["train"]["steps"] = 12
    expanded = expand_spec(spec)
    assert expanded["train"]["steps"] == 12


def test_pick_and_place_has_scene_objects() -> None:
    spec = {"spec_version": "0.1.0", "name": "pick-and-place", "robot": {"id": "unitree-g1-29dof", "source": "catalog"}, "task": {"recipe": "pick-and-place"}, "train": {"method": "imitation"}}
    expanded = expand_spec(spec)
    ids = {obj["id"] for obj in expanded["scene"]["objects"]}
    assert ids == {"mustard", "bowl"}


def test_start_here_recipes() -> None:
    flags = {r.id: r.as_public_dict()["start_here"] for r in list_recipes()}
    assert flags["cartpole-balance"] is True
    assert flags["g1-stand"] is True
    assert flags["pick-and-place"] is True
    assert flags["g1-walk"] is False
