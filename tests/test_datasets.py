from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_training.datasets import inspect_lerobot_dataset, load_lerobot_arrays
from humanoid_training.demos import record_object_trajectories, record_scripted_pick_place
from humanoid_training.errors import RecipeError
from humanoid_training.recipes import expand_spec
from humanoid_training.train_bc import fit_linear_bc, predict_linear_bc


def test_inspect_local_lerobot_fixture() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "lerobot_tiny"
    result = inspect_lerobot_dataset(str(path))
    assert result["ok"] is True
    assert result["format"] == "lerobot"
    assert result["total_episodes"] == 2
    assert result["episodes"][0]["tasks"] == ["pick mustard"]


def test_inspect_hf_is_closed() -> None:
    result = inspect_lerobot_dataset("hf:example/g1-mustard-demos")
    assert result["ok"] is False
    assert "Hugging Face" in result["error"]


def test_inspect_missing_info() -> None:
    result = inspect_lerobot_dataset("/tmp/not-a-dataset")
    assert result["ok"] is False
    assert "meta/info.json" in result["error"]


def _pick_spec() -> dict:
    return expand_spec(
        {
            "spec_version": "0.1.0",
            "name": "pick-and-place",
            "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
            "task": {"recipe": "pick-and-place"},
            "train": {"method": "imitation"},
        }
    )


def test_record_and_filter_scripted_demos(tmp_path: Path) -> None:
    spec = _pick_spec()
    dest = tmp_path / "demos"
    meta = record_scripted_pick_place(spec, dest, episodes=4, include_failure=True, seed=1)
    assert meta["ok"] is True
    assert meta["total_episodes"] == 4
    assert any(ep.get("success") is False for ep in meta["episodes"])
    keep = [ep["episode_index"] for ep in meta["episodes"] if ep.get("success") is not False]
    obs_all, _act_all = load_lerobot_arrays(dest)
    obs_keep, act_keep = load_lerobot_arrays(dest, keep_episodes=keep)
    assert len(obs_keep) < len(obs_all)
    weights = fit_linear_bc(obs_keep, act_keep)
    mustard = spec["scene"]["objects"][0]
    bowl = spec["scene"]["objects"][1]
    from humanoid_training.compose import object_world_pos, table_layout

    layout = table_layout(spec["scene"])
    m = object_world_pos(mustard, layout)
    b = object_world_pos(bowl, layout)
    delta = predict_linear_bc(weights, [m[0], m[1], b[0], b[1]])
    assert delta.shape == (2,)
    toward = (b[0] - m[0]) * float(delta[0]) + (b[1] - m[1]) * float(delta[1])
    assert toward > 0


def test_record_requires_scene_objects(tmp_path: Path) -> None:
    spec = _pick_spec()
    spec["scene"]["objects"] = []
    with pytest.raises(RecipeError, match="scene.objects"):
        record_scripted_pick_place(spec, tmp_path / "empty")


def test_record_canvas_trajectory_reaches_bowl(tmp_path: Path) -> None:
    spec = _pick_spec()
    mustard = spec["scene"]["objects"][0]
    bowl = spec["scene"]["objects"][1]
    traj = []
    for t in range(16):
        a = t / 15
        traj.append(
            {
                "x": mustard["x"] + a * (bowl["x"] - mustard["x"]),
                "y": mustard["y"] + a * (bowl["y"] - mustard["y"]),
            }
        )
    meta = record_object_trajectories(spec, tmp_path / "canvas", [traj])
    assert meta["ok"] is True
    assert meta["total_episodes"] == 1
    assert meta["episodes"][0]["success"] is True


def test_record_empty_trajectories_raises(tmp_path: Path) -> None:
    spec = _pick_spec()
    with pytest.raises(RecipeError, match="No canvas trajectories"):
        record_object_trajectories(spec, tmp_path / "none", [])
