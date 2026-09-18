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


def test_g1_walk_compiles_isaac_when_preferred() -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    spec["backend"] = {"prefer": ["isaaclab"], "compute": "osmo"}
    expanded = expand_spec(spec)
    adapter = select_adapter(expanded)
    assert adapter.name == "isaaclab"
    payload = adapter.compile(expanded)
    assert payload.env_name == "Isaac-Velocity-Flat-G1-v0"
    assert "Isaac-Velocity-Flat-G1-v0" in payload.files["osmo_workflow.yaml"]


def test_g1_walk_rejects_gymnasium() -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    spec["backend"] = {"prefer": ["gymnasium"], "compute": "local"}
    expanded = expand_spec(spec)
    with pytest.raises(NoAdapter, match="G1 locomotion"):
        select_adapter(expanded)
