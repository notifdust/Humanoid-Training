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
    notes = " ".join(manifest.get("notes") or [])
    assert "greedy_train_eval=" in notes
    ckpt = np.load(run_dir / "checkpoint.npz")
    assert "W" in ckpt.files and "weights" in ckpt.files
    assert "eval.mp4" in manifest["artifacts"]
    assert "checkpoint.npz" in manifest["artifacts"]
    assert "train_returns.json" in manifest["artifacts"]
    assert manifest.get("facts", {}).get("kind") == "rl"
    assert "greedy_train_eval" in (manifest.get("facts") or {})


def test_g1_walk_is_blocked_without_playground(tmp_path: Path) -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] == "blocked"
    err = manifest["error"] or ""
    assert "Playground" in err
    assert "blocked" in err.lower() or "CPU studio" in err or "g1-stand" in err
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "train.sh").is_file()
    assert (run_dir / "engine_payload.json").is_file()
    assert (run_dir / "engines" / "isaaclab" / "osmo_workflow.yaml").is_file()
    mjlab_sh = (run_dir / "engines" / "mjlab" / "train_mjlab.sh").read_text(encoding="utf-8")
    assert "Mjlab-Velocity-Flat-Unitree-G1" in mjlab_sh
    assert "Velocity-G1-Flat-v0" not in mjlab_sh


def test_g1_reach_is_blocked_with_cpu_next_step(tmp_path: Path) -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-reach.json")
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] == "blocked"
    err = manifest["error"] or ""
    assert "reach" in err.lower()
    assert "pick-and-place" in err.lower()
    assert "G1Reach-v0" not in err
    assert "Isaac-Reach-G1-v0" not in err
    run_dir = Path(manifest["run_dir"])
    assert not (run_dir / "train_mjlab.sh").is_file()
    assert not (run_dir / "eval.mp4").is_file()


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
    notes = " ".join(manifest.get("notes") or [])
    assert "not balance policy" in notes or "not a balance policy" in notes
    # mini_humanoid has no actuators — must not claim arm wave.
    assert "no actuators" in notes or "arm actuators not mapped" in notes
    assert "raise and wave" not in notes
    assert (manifest.get("facts") or {}).get("kind") == "hold"
    assert (manifest.get("facts") or {}).get("nu") == 0


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
    facts = manifest.get("facts") or {}
    assert facts.get("kind") == "imitation"
    assert facts.get("arm_mode")
    assert facts.get("object") == "mustard"
    assert facts.get("container") == "bowl"
    assert "bc_steps" in facts


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
    assert "arm_mode=IK+BC" in notes, notes
    assert (manifest.get("facts") or {}).get("arm_mode") == "IK+BC"
    assert (manifest.get("facts") or {}).get("kind") == "imitation"
    log = (Path(manifest["run_dir"]) / "run.log").read_text(encoding="utf-8")
    assert "right-arm IK+BC" in log or "IK+BC" in log
    # Mini arm cannot reach mustard — must not silently pass as if BC ran.
    assert "arm never attached" in notes or manifest.get("metrics", {}).get("passed") is False
    assert "bc_steps=0" in notes or "attach_step=-1" in notes


def test_puppet_freejoint_uses_bc_for_mustard(tmp_path: Path, monkeypatch) -> None:
    """Pinned freejoint path must call linear BC so demos change the mustard path."""
    import humanoid_training.adapters.mujoco_adapter as adapter

    calls = {"n": 0}
    real_predict = adapter.predict_linear_bc

    def counting_predict(weights, obs):
        calls["n"] += 1
        return real_predict(weights, obs)

    monkeypatch.setattr(adapter, "predict_linear_bc", counting_predict)
    # Keep the arm near the mustard so grasp succeeds on this tiny fixture.
    monkeypatch.setattr(adapter, "_PICK_POSE", {})
    monkeypatch.setattr(adapter, "_LIFT_POSE", {})
    monkeypatch.setattr(adapter, "_PLACE_POSE", {})

    fixture = Path(__file__).resolve().parent / "fixtures" / "mini_puppet.xml"
    spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {"recipe": "pick-and-place"},
        "train": {"method": "imitation", "seed": 1},
        "scene": {
            "template": "kitchen-counter-v1",
            "table_pos": [0.36, -0.12, 0.90],
            "table_size": [0.10, 0.10, 0.02],
            "objects": [
                {"id": "mustard", "asset": "ycb-mustard", "x": 0.0, "y": 0.0},
                {"id": "bowl", "asset": "bowl-white", "x": 0.08, "y": 0.04},
            ],
        },
        "backend": {"prefer": ["mujoco"], "compute": "local"},
        "adapters": {
            "mujoco": {
                "mjcf": str(fixture),
                "horizon": 400,
                "render_every": 10,
                "keyframe": 0,
            }
        },
        "eval": {"episodes": 1, "record_video": False},
        "data": {"min_episodes": 3},
    }
    manifest = run_job(spec, runs_dir=tmp_path)
    notes = " ".join(manifest.get("notes") or [])
    log = (Path(manifest["run_dir"]) / "run.log").read_text(encoding="utf-8")
    assert "playback+BC" in log or "arm_mode=playback+BC" in notes, (log, notes)
    assert calls["n"] > 0, "linear BC never ran on the puppet path"
    assert "bc_steps=" in notes
    assert "mustard_bowl_dist" in notes


def test_keep_only_failures_fails_mustard_in_bowl(tmp_path: Path, monkeypatch) -> None:
    """Full train→eval: keep_episodes of only misses must not place mustard in bowl."""
    import humanoid_training.adapters.mujoco_adapter as adapter
    from humanoid_training.demos import record_scripted_pick_place

    monkeypatch.setattr(adapter, "_PICK_POSE", {})
    monkeypatch.setattr(adapter, "_LIFT_POSE", {})
    monkeypatch.setattr(adapter, "_PLACE_POSE", {})

    # Compact counter matching mini_puppet.xml reach — same as BC wiring test.
    scene = {
        "template": "kitchen-counter-v1",
        "table_pos": [0.36, -0.12, 0.90],
        "table_size": [0.10, 0.10, 0.02],
        "objects": [
            {"id": "mustard", "asset": "ycb-mustard", "x": 0.0, "y": 0.0},
            {"id": "bowl", "asset": "bowl-white", "x": 0.08, "y": 0.04},
        ],
    }
    record_spec = {
        "spec_version": "0.1.0",
        "name": "pick-and-place",
        "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
        "task": {
            "recipe": "pick-and-place",
            "success": {"object": "mustard", "container": "bowl"},
        },
        "scene": scene,
        "train": {"method": "imitation", "seed": 3},
    }
    demos = tmp_path / "demos"
    meta = record_scripted_pick_place(record_spec, demos, episodes=4, include_failure=True, seed=3)
    fails = [ep["episode_index"] for ep in meta["episodes"] if ep.get("success") is False]
    oks = [ep["episode_index"] for ep in meta["episodes"] if ep.get("success") is not False]
    assert fails and oks

    fixture = Path(__file__).resolve().parent / "fixtures" / "mini_puppet.xml"

    def _spec(keep: list[int]) -> dict:
        return {
            "spec_version": "0.1.0",
            "name": "pick-and-place",
            "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
            "task": {
                "recipe": "pick-and-place",
                "success": {"object": "mustard", "container": "bowl", "min_z": 0.5, "hold_s": 0.1},
            },
            "train": {"method": "imitation", "seed": 3},
            "scene": scene,
            "backend": {"prefer": ["mujoco"], "compute": "local"},
            "adapters": {
                "mujoco": {
                    "mjcf": str(fixture),
                    "horizon": 600,
                    "render_every": 20,
                    "keyframe": 0,
                }
            },
            "eval": {"episodes": 1, "record_video": False},
            "data": {
                "datasets": [str(demos)],
                "keep_episodes": keep,
                "min_episodes": 1,
            },
        }

    bad = run_job(_spec(fails), runs_dir=tmp_path / "bad")
    assert bad.get("metrics", {}).get("passed") is False, bad.get("notes")
    bad_notes = " ".join(bad.get("notes") or [])
    assert f"keep_episodes={fails}" in bad_notes

    good = run_job(_spec(oks), runs_dir=tmp_path / "good")
    assert good.get("metrics", {}).get("passed") is True, good.get("notes")
    good_notes = " ".join(good.get("notes") or [])
    assert f"keep_episodes={oks}" in good_notes


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
