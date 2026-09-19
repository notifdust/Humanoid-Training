"""Shared MuJoCo simulate / render helpers used by hold and imitation launches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.adapters.base import LogFn
from humanoid_training.compose import STAND_CAMERA, camera_xyaxes, object_world_pos


def reset(mujoco: Any, model: Any, data: Any, keyframe: int, log: LogFn) -> None:
    if model.nkey > 0:
        idx = min(keyframe, model.nkey - 1)
        mujoco.mj_resetDataKeyframe(model, data, idx)
        log(f"reset to keyframe {idx}")
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def simulate(
    mujoco: Any,
    model: Any,
    data: Any,
    spec: dict[str, Any],
    *,
    honors: bool,
    ctrl: np.ndarray | None,
    horizon: int,
    render_every: int,
    log: LogFn,
    step_fn: Any,
    pin_base: tuple[int, int, np.ndarray] | None = None,
    stop_box: dict[str, bool] | None = None,
) -> tuple[list[np.ndarray], list[float], list[bool], dict[str, int]]:
    eval_cfg = spec.get("eval") or {}
    record_video = bool(eval_cfg.get("record_video", True))
    success_cfg = (spec.get("task") or {}).get("success") or {}
    min_z = float(success_cfg.get("min_z", 0.5))
    renderer = None
    camera: Any = -1
    if model.ncam > 0:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ht_eval")
        if cam_id >= 0:
            camera = cam_id
    if record_video:
        renderer = make_renderer(mujoco, model, log)

    pelvis = body_id(mujoco, model, "pelvis")
    if pelvis < 0:
        pelvis = 1 if model.nbody > 1 else 0
    objects = list((spec.get("scene") or {}).get("objects") or []) if honors else []
    object_ids = {
        str(obj["id"]): body_id(mujoco, model, str(obj["id"]))
        for obj in objects
        if obj.get("id")
    }
    frames: list[np.ndarray] = []
    upright: list[bool] = []
    zs: list[float] = []
    for step in range(horizon):
        if pin_base is not None:
            qadr, dadr, q0 = pin_base
            data.qpos[qadr : qadr + 7] = q0
            data.qvel[dadr : dadr + 6] = 0.0
            mujoco.mj_forward(model, data)
        if step_fn is not None:
            step_fn(model, data, step)
        if ctrl is not None:
            data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        z = float(data.xpos[pelvis, 2])
        zs.append(z)
        upright.append(z >= min_z)
        if renderer is not None and step % render_every == 0:
            if camera != -1:
                renderer.update_scene(data, camera=camera)
            else:
                renderer.update_scene(data)
            frames.append(np.asarray(renderer.render()).copy())
        if step in {0, horizon // 2, horizon - 1}:
            log(f"step {step}/{horizon} pelvis_z={z:.3f}")
        if stop_box is not None and stop_box.get("stop"):
            log(f"early stop at step {step}/{horizon}")
            break
    return frames, zs, upright, object_ids


def make_renderer(mujoco: Any, model: Any, log: LogFn) -> Any:
    """GLFW aborts the process on a headless runner. Never construct it without a display."""
    if os.environ.get("HT_NO_RENDER") == "1":
        log("HT_NO_RENDER=1; physics-only eval")
        return None
    gl = os.environ.get("MUJOCO_GL", "glfw").lower()
    if gl in {"glfw", ""} and not os.environ.get("DISPLAY"):
        log("no DISPLAY for MuJoCo GLFW; physics-only eval")
        return None
    try:
        return mujoco.Renderer(model, 360, 640)
    except Exception as err:
        log(f"renderer unavailable ({err}); physics-only eval")
        return None


def video_notes(frames: list[np.ndarray]) -> list[str]:
    if frames:
        return []
    return ["no eval.mp4 (headless / HT_NO_RENDER); physics success still counted"]


def body_id(mujoco: Any, model: Any, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def placement_report(
    data: Any,
    object_ids: dict[str, int],
    objects: list[dict[str, Any]],
    layout: dict[str, Any],
    tol: float = 0.08,
) -> list[dict[str, Any]]:
    rows = []
    for obj in objects:
        oid = str(obj.get("id") or "")
        bid = object_ids.get(oid, -1)
        if not oid or bid is None or bid < 0:
            rows.append({"id": oid or "?", "ok": False, "xy_err": 999.0})
            continue
        want = object_world_pos(obj, layout)
        got = data.xpos[bid]
        err = float(np.hypot(float(got[0]) - want[0], float(got[1]) - want[1]))
        rows.append({"id": oid, "ok": err <= tol, "xy_err": err})
    return rows


def snapshot_freejoint(model: Any, data: Any) -> tuple[int, int, np.ndarray] | None:
    """Stand-keyframe pose of a freejoint, if the model has one."""
    for i in range(int(model.njnt)):
        if int(model.jnt_type[i]) != 0:  # mjJNT_FREE
            continue
        qadr = int(model.jnt_qposadr[i])
        dadr = int(model.jnt_dofadr[i])
        return qadr, dadr, np.array(data.qpos[qadr : qadr + 7], dtype=np.float64, copy=True)
    return None


def hold_ctrl(model: Any, data: Any) -> np.ndarray | None:
    if model.nu == 0:
        return None
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for i in range(model.nu):
        jnt = int(model.actuator_trnid[i, 0])
        if jnt < 0:
            continue
        qadr = int(model.jnt_qposadr[jnt])
        ctrl[i] = float(data.qpos[qadr])
    return ctrl


def model_with_stand_camera(mujoco: Any, mjcf: Path) -> Any:
    """Frame the full G1 so stand idle is visible; Menagerie default is a wide shot."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    pos = STAND_CAMERA["camera_pos"]
    target = STAND_CAMERA["camera_target"]
    spec.worldbody.add_camera(
        name="ht_eval",
        pos=list(pos),
        xyaxes=camera_xyaxes(pos, target),
    )
    spec.worldbody.add_light(pos=[0.4, -0.5, 2.2], dir=[0.1, 0.2, -1.0])
    return spec.compile()


HAND_BODIES = (
    "right_wrist_yaw_link",
    "right_rubber_hand",
    "right_palm_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
)

ARM_ACTUATOR_BITS = (
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "waist_yaw",
)


def find_hand(mujoco: Any, model: Any) -> int:
    for name in HAND_BODIES:
        bid = body_id(mujoco, model, name)
        if bid >= 0:
            return bid
    return -1


def actuator_name_map(mujoco: Any, model: Any) -> dict[str, int]:
    names: dict[str, int] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            names[str(name)] = i
    return names


def arm_actuator_ids(mujoco: Any, model: Any) -> list[int]:
    ids = []
    for name, idx in actuator_name_map(mujoco, model).items():
        if any(bit in name for bit in ARM_ACTUATOR_BITS):
            ids.append(idx)
    return sorted(ids)
