from __future__ import annotations

from pathlib import Path

import numpy as np

from humanoid_training.recipes import expand_spec
from humanoid_training.runner import run_job
from humanoid_training.spec import load_spec


def test_cartpole_train_writes_video(tmp_path: Path) -> None:
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    )
    spec["train"] = {"method": "rl", "steps": 40, "seed": 1}
    spec["eval"] = {"episodes": 2, "record_video": True}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] in {"completed", "passed"}
    assert manifest["adapter"]["adapter"] == "gymnasium"
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "eval.mp4").is_file()
    assert (run_dir / "checkpoint.npz").is_file()
    assert manifest["metrics"]["eval_episodes"] == 2


def test_g1_walk_is_blocked_without_playground(tmp_path: Path) -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] == "blocked"
    assert "Playground" in (manifest["error"] or "")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "train.sh").is_file()
    assert (run_dir / "engine_payload.json").is_file()
    assert (run_dir / "engines" / "isaaclab" / "osmo_workflow.yaml").is_file()


def test_g1_walk_compile_only(tmp_path: Path) -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    manifest = run_job(spec, runs_dir=tmp_path, compile_only=True)
    assert manifest["status"] == "compiled"
    assert expand_spec(spec)["task"]["recipe"] == "g1-walk"
    assert (Path(manifest["run_dir"]) / "engines" / "mjlab" / "train_mjlab.sh").is_file()


def test_g1_stand_hold_mini_humanoid(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "mini_humanoid.xml"
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-stand.json")
    spec["adapters"] = {
        "mujoco": {
            "mjcf": str(fixture),
            "horizon": 60,
            "render_every": 5,
        }
    }
    spec["eval"] = {"episodes": 1, "record_video": True}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] in {"completed", "passed"}, manifest.get("error")
    assert manifest["adapter"]["adapter"] == "mujoco"
    assert manifest["metrics"]["passed"] is True


def test_pick_and_place_preview_composes_scene(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "mini_humanoid.xml"
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "hold"},
        "scene": {
            "template": "kitchen-counter-v1",
            "objects": [
                {"id": "mustard", "asset": "ycb-mustard", "x": -0.18, "y": 0.04},
                {"id": "bowl", "asset": "bowl-white", "x": 0.16, "y": -0.02},
            ],
        },
        "backend": {"prefer": ["mujoco"], "compute": "local"},
        "adapters": {
            "mujoco": {
                "mjcf": str(fixture),
                "horizon": 40,
                "render_every": 5,
            }
        },
        "eval": {"episodes": 1, "record_video": True},
    }
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] in {"completed", "passed"}, manifest.get("error")
    assert manifest["adapter"]["adapter"] == "mujoco"
    assert "scene.objects" not in (manifest.get("ignored_fields") or [])
    xml = Path(manifest["run_dir"]) / "composed_scene.xml"
    assert xml.is_file()
    text = xml.read_text(encoding="utf-8")
    assert "mustard" in text
    assert "ht_table" in text
    assert manifest["artifacts"].get("composed_scene.xml")


def test_pick_and_place_imitation_puts_mustard_in_bowl(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "mini_humanoid.xml"
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "imitation", "seed": 1},
        "backend": {"prefer": ["mujoco"], "compute": "local"},
        "adapters": {
            "mujoco": {
                "mjcf": str(fixture),
                "horizon": 80,
                "render_every": 5,
            }
        },
        "eval": {"episodes": 1, "record_video": True},
        "data": {"min_episodes": 3},
    }
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "checkpoint.npz").is_file()
    assert (run_dir / "lerobot_dataset" / "meta" / "info.json").is_file()
    notes = " ".join(manifest.get("notes") or [])
    assert "linear BC" in notes
    assert "mustard_bowl_dist" in notes


def test_idle_stand_ctrl_raises_both_arms() -> None:
    from humanoid_training.adapters.mujoco_adapter import _idle_stand_ctrl

    hold = np.zeros(7)
    names = {
        "right_shoulder_pitch_joint": 0,
        "right_elbow_joint": 1,
        "waist_yaw_joint": 2,
        "left_shoulder_pitch_joint": 3,
        "left_elbow_joint": 4,
        "right_shoulder_roll_joint": 5,
        "right_shoulder_yaw_joint": 6,
    }
    start = _idle_stand_ctrl(hold, names, 0, 100)
    mid = _idle_stand_ctrl(hold, names, 50, 100)
    end = _idle_stand_ctrl(hold, names, 99, 100)
    assert abs(start[0]) < 1e-9
    assert mid[0] < -0.8
    assert end[0] < -0.8
    assert mid[3] < -0.5


def test_named_pose_ctrl_blends_to_reach() -> None:
    from humanoid_training.adapters.mujoco_adapter import _PICK_POSE, _named_pose_ctrl

    hold = np.array([0.2, -0.2, 0.0, 1.28, 0.0, 0.0])
    names = {
        "right_shoulder_pitch_joint": 0,
        "right_shoulder_roll_joint": 1,
        "right_shoulder_yaw_joint": 2,
        "right_elbow_joint": 3,
        "waist_yaw_joint": 4,
        "waist_pitch_joint": 5,
    }
    half = _named_pose_ctrl(hold, names, _PICK_POSE, 0.5)
    assert abs(half[0] - 0.5 * (0.2 + _PICK_POSE["right_shoulder_pitch_joint"])) < 1e-9
    full = _named_pose_ctrl(hold, names, _PICK_POSE, 1.0)
    assert abs(full[3] - _PICK_POSE["right_elbow_joint"]) < 1e-9


def test_mini_arm_has_no_freejoint_to_pin() -> None:
    import mujoco

    from humanoid_training.adapters.mujoco_adapter import _snapshot_freejoint

    mjcf = Path(__file__).resolve().parent / "fixtures" / "mini_arm.xml"
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert _snapshot_freejoint(model, data) is None


def test_mini_arm_ik_moves_hand() -> None:
    import mujoco

    from humanoid_training.adapters.mujoco_adapter import (
        _arm_actuator_ids,
        _find_hand,
        _hold_ctrl,
        _ik_toward,
    )

    mjcf = Path(__file__).resolve().parent / "fixtures" / "mini_arm.xml"
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hand = _find_hand(mujoco, model)
    arms = _arm_actuator_ids(mujoco, model)
    assert hand >= 0
    assert arms
    hold = _hold_ctrl(model, data)
    start = np.array(data.xpos[hand], dtype=float)
    target = start + np.array([-0.08, 0.0, -0.10])
    before = float(np.linalg.norm(data.xpos[hand] - target))
    for _ in range(80):
        data.ctrl[:] = _ik_toward(mujoco, model, data, hand, target, arms, hold, gain=1.0)
        mujoco.mj_step(model, data)
    after = float(np.linalg.norm(data.xpos[hand] - target))
    assert after < before * 0.6


def test_pick_and_place_arm_fixture_moves_joints(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "mini_arm.xml"
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "imitation", "seed": 1},
        "backend": {"prefer": ["mujoco"], "compute": "local"},
        "adapters": {
            "mujoco": {
                "mjcf": str(fixture),
                "horizon": 80,
                "render_every": 5,
            }
        },
        "eval": {"episodes": 1, "record_video": True},
        "data": {"min_episodes": 3},
    }
    manifest = run_job(spec, runs_dir=tmp_path)
    notes = " ".join(manifest.get("notes") or [])
    assert "arm_ik=on" in notes, notes
    log = (Path(manifest["run_dir"]) / "run.log").read_text(encoding="utf-8")
    assert "right-arm IK" in log


def test_imitation_physics_only_without_display(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("HT_NO_RENDER", "1")
    monkeypatch.setenv("MUJOCO_GL", "glfw")

    class Boom:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Renderer must not be constructed without a display")

    import mujoco

    monkeypatch.setattr(mujoco, "Renderer", Boom)
    fixture = Path(__file__).resolve().parent / "fixtures" / "mini_humanoid.xml"
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "imitation", "seed": 1},
        "backend": {"prefer": ["mujoco"], "compute": "local"},
        "adapters": {
            "mujoco": {
                "mjcf": str(fixture),
                "horizon": 80,
                "render_every": 5,
            }
        },
        "eval": {"episodes": 1, "record_video": True},
        "data": {"min_episodes": 3},
    }
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert not (run_dir / "eval.mp4").is_file()
    notes = " ".join(manifest.get("notes") or [])
    assert "no eval.mp4" in notes


def test_make_renderer_skips_glfw_without_display(monkeypatch) -> None:
    from humanoid_training.adapters.mujoco_adapter import _make_renderer

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("HT_NO_RENDER", raising=False)
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    logs: list[str] = []

    class Fake:
        @staticmethod
        def Renderer(*_args, **_kwargs):
            raise AssertionError("should not construct GLFW renderer")

    assert _make_renderer(Fake, object(), logs.append) is None
    assert any("DISPLAY" in line for line in logs)


def test_make_renderer_respects_ht_no_render(monkeypatch) -> None:
    from humanoid_training.adapters.mujoco_adapter import _make_renderer

    monkeypatch.setenv("HT_NO_RENDER", "1")
    monkeypatch.setenv("DISPLAY", ":99")
    logs: list[str] = []

    class Fake:
        @staticmethod
        def Renderer(*_args, **_kwargs):
            raise AssertionError("HT_NO_RENDER must skip Renderer")

    assert _make_renderer(Fake, object(), logs.append) is None
    assert any("HT_NO_RENDER" in line for line in logs)
