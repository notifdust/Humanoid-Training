from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_training.adapters import select_adapter
from humanoid_training.errors import NoAdapter
from humanoid_training.recipes import expand_spec
from humanoid_training.spec import load_spec


def _expand(name: str, **overlay):
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / name)
    for key, value in overlay.items():
        spec[key] = value
    return expand_spec(spec)


def test_cartpole_selects_gymnasium() -> None:
    spec = _expand("cartpole-balance.json")
    assert select_adapter(spec).name == "gymnasium"


def test_g1_walk_compiles_playground() -> None:
    spec = _expand("g1-walk.json")
    adapter = select_adapter(spec)
    assert adapter.name == "playground"
    payload = adapter.compile(spec)
    assert payload.env_name == "G1JoystickFlatTerrain"
    assert "train.sh" in payload.files
    assert "G1JoystickFlatTerrain" in payload.files["train.sh"]
    assert "--num_timesteps" in payload.files["train.sh"]
    assert "--logdir" in payload.files["train.sh"]
    assert "--seed" in payload.command
    assert "Phase 1/3" not in payload.files["train.sh"]


def test_g1_walk_compiles_mjlab_real_task() -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    spec["backend"] = {"prefer": ["mjlab"], "compute": "local-docker"}
    expanded = expand_spec(spec)
    adapter = select_adapter(expanded)
    assert adapter.name == "mjlab"
    payload = adapter.compile(expanded)
    assert payload.env_name == "Mjlab-Velocity-Flat-Unitree-G1"
    assert "python -m mjlab.scripts.train" in payload.files["train_mjlab.sh"]
    assert "Mjlab-Velocity-Flat-Unitree-G1" in payload.files["train_mjlab.sh"]
    assert "--video True" in payload.files["train_mjlab.sh"]
    assert "G1Reach-v0" not in payload.files["train_mjlab.sh"]
    assert "Velocity-G1-Flat-v0" not in payload.files["train_mjlab.sh"]


def test_g1_walk_compiles_isaac_when_preferred() -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    spec["backend"] = {"prefer": ["isaaclab"], "compute": "osmo"}
    expanded = expand_spec(spec)
    adapter = select_adapter(expanded)
    assert adapter.name == "isaaclab"
    payload = adapter.compile(expanded)
    assert payload.env_name == "Isaac-Velocity-Flat-G1-v0"
    assert "Isaac-Velocity-Flat-G1-v0" in payload.files["osmo_workflow.yaml"]
    assert "Isaac-Reach-G1-v0" not in payload.files["osmo_workflow.yaml"]
    assert "--video" in payload.files["train_isaac.sh"]


def test_g1_walk_rejects_gymnasium() -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    spec["backend"] = {"prefer": ["gymnasium"], "compute": "local"}
    expanded = expand_spec(spec)
    with pytest.raises(NoAdapter, match="G1 locomotion"):
        select_adapter(expanded)


def test_g1_stand_selects_mujoco() -> None:
    spec = _expand("g1-stand.json")
    adapter = select_adapter(spec)
    assert adapter.name == "mujoco"
    payload = adapter.compile(spec)
    assert "unitree_g1" in payload.env_name


def test_deleted_g1_reach_is_unknown_recipe() -> None:
    from humanoid_training.errors import RecipeError

    with pytest.raises(RecipeError, match="Unknown recipe 'g1-reach'"):
        expand_spec(
            {
                "spec_version": "0.1.0",
                "name": "missing-reach",
                "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
                "task": {"recipe": "g1-reach"},
                "train": {"method": "rl"},
            }
        )


def test_pick_and_place_selects_mujoco_preview() -> None:
    spec = _expand("g1-mustard-in-bowl.json")
    adapter = select_adapter(spec)
    assert adapter.name == "mujoco"
    payload = adapter.compile(spec)
    assert payload.ignored_fields == []
    assert payload.extra.get("honors_scene") is True
    assert payload.extra.get("scene_preview") is True


def test_lerobot_support_ok_for_imitation() -> None:
    from humanoid_training.adapters.lerobot import LeRobotAdapter

    spec = _expand("g1-mustard-in-bowl.json")
    support = LeRobotAdapter().support(spec)
    assert support.ok is True
    payload = LeRobotAdapter().compile(spec)
    assert payload.extra.get("policy") == "act"
    assert "train_lerobot.sh" in payload.files
    assert "--policy.type=act" in payload.files["train_lerobot.sh"]


def test_pick_and_place_cpu_still_selects_mujoco() -> None:
    """Prefer lists lerobot first; without GPU it is not launch-ready."""
    spec = _expand("g1-mustard-in-bowl.json")
    prefer = list((spec.get("backend") or {}).get("prefer") or [])
    assert prefer and prefer[0] == "lerobot"
    assert select_adapter(spec).name == "mujoco"
