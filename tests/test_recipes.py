from __future__ import annotations

from pathlib import Path

import pytest

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


def test_recipe_catalog_is_the_studio_contract() -> None:
    """Studio must not hardcode which tasks work on CPU — recipes declare it."""
    from humanoid_training.recipes import public_catalog

    catalog = public_catalog()
    by_id = {r["id"]: r for r in catalog["recipes"]}
    assert by_id["cartpole-balance"]["availability"] == "cpu"
    assert by_id["g1-stand"]["availability"] == "cpu"
    assert by_id["pick-and-place"]["availability"] == "cpu"
    assert by_id["g1-walk"]["availability"] == "gpu"
    assert by_id["g1-reach"]["availability"] == "gpu"
    assert by_id["cartpole-balance"]["promise"]
    assert by_id["g1-walk"]["blocked_hint"]
    assert by_id["pick-and-place"]["imitate"] is True
    assert by_id["pick-and-place"]["method"] == "imitation"
    assert by_id["g1-stand"]["method"] == "hold"
    assert by_id["pick-and-place"]["scene_hint"]
    assert by_id["pick-and-place"]["record_hint"]
    assert "WASD" in by_id["pick-and-place"]["record_hint"]
    assert set(catalog["ready"]) == {"cartpole-balance", "g1-stand", "pick-and-place"}
    assert set(catalog["later"]) == {"g1-walk", "g1-reach"}
    assert set(catalog["start_here"]) == {"cartpole-balance", "g1-stand", "pick-and-place"}
    assert by_id["cartpole-balance"]["has_gold"] is True
    assert by_id["g1-stand"]["has_gold"] is True
    assert by_id["pick-and-place"]["has_gold"] is True
    assert by_id["g1-walk"]["has_gold"] is False
    assert by_id["g1-walk"]["launch_here"] is False
    assert by_id["g1-reach"]["launch_here"] is False
    assert by_id["cartpole-balance"]["launch_here"] is True


def test_launch_here_walk_when_playground_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from humanoid_training.recipes import public_catalog

    monkeypatch.setattr("humanoid_training.hardware.playground_ready", lambda: True)
    catalog = public_catalog()
    by_id = {r["id"]: r for r in catalog["recipes"]}
    assert by_id["g1-walk"]["launch_here"] is True
    assert by_id["g1-reach"]["launch_here"] is False
    assert "g1-walk" in catalog["ready"]
    assert "g1-reach" in catalog["later"]
    assert "Playground" in by_id["g1-walk"]["promise"] or "walking" in by_id["g1-walk"]["promise"].lower()
    assert "stand clip" in by_id["g1-walk"]["train_hint"]


def test_launch_here_walk_when_mjlab_or_isaac_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from humanoid_training.recipes import public_catalog

    monkeypatch.setattr("humanoid_training.hardware.playground_ready", lambda: False)
    monkeypatch.setattr("humanoid_training.hardware.mjlab_ready", lambda: True)
    monkeypatch.setattr("humanoid_training.hardware.isaac_launch_ready", lambda: False)
    catalog = public_catalog()
    by_id = {r["id"]: r for r in catalog["recipes"]}
    assert by_id["g1-walk"]["launch_here"] is True
    assert by_id["g1-reach"]["launch_here"] is False
    assert "g1-walk" in catalog["ready"]
    assert "g1-reach" in catalog["later"]


def test_recipes_pin_real_upstream_task_ids() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    walk = yaml.safe_load((root / "recipes" / "g1-walk" / "recipe.yaml").read_text(encoding="utf-8"))
    reach = yaml.safe_load((root / "recipes" / "g1-reach" / "recipe.yaml").read_text(encoding="utf-8"))
    walk_ad = walk.get("adapters") or {}
    reach_ad = reach.get("adapters") or {}
    assert (walk_ad.get("mjlab") or {}).get("task") == "Mjlab-Velocity-Flat-Unitree-G1"
    assert (walk_ad.get("isaaclab") or {}).get("task") == "Isaac-Velocity-Flat-G1-v0"
    assert (walk_ad.get("playground") or {}).get("env_name") == "G1JoystickFlatTerrain"
    assert not (reach_ad.get("playground") or {}).get("env_name")
    assert (reach_ad.get("playground") or {}).get("unsupported")
    assert (reach_ad.get("mjlab") or {}).get("unsupported")
    assert (reach_ad.get("isaaclab") or {}).get("unsupported")
    assert not (reach_ad.get("mjlab") or {}).get("task")
    assert not (reach_ad.get("isaaclab") or {}).get("task")


def test_recipes_module_does_not_hardcode_start_here_ids() -> None:
    from humanoid_training import recipes as recipes_mod

    src = Path(recipes_mod.__file__).read_text(encoding="utf-8")
    assert "cartpole-balance" not in src
    assert "g1-stand" not in src
    assert "pick-and-place" not in src


def test_studio_availability_must_be_cpu_or_gpu() -> None:
    from humanoid_training.errors import RecipeError
    from humanoid_training.recipes import _validate_studio

    _validate_studio("ok", {"studio": {"availability": "cpu"}})
    try:
        _validate_studio("bad", {"studio": {"availability": "tpu"}})
    except RecipeError as err:
        assert "cpu or gpu" in str(err)
    else:
        raise AssertionError("expected RecipeError")


def test_english_list() -> None:
    from humanoid_training.recipes import english_list

    assert english_list([]) == ""
    assert english_list(["Cartpole"]) == "Cartpole"
    assert english_list(["A", "B"]) == "A and B"
    assert english_list(["A", "B", "C"]) == "A, B, and C"
