"""Open-loop G1 arm poses, qpos drive, and Jacobian IK. No launch policy here."""

from __future__ import annotations

from typing import Any

import numpy as np

# Open-loop poses that put the standing G1 wrist on the kitchen mustard / bowl.
PICK_POSE = {
    "right_shoulder_pitch_joint": -0.30,
    "right_shoulder_roll_joint": 0.04,
    "right_shoulder_yaw_joint": 0.42,
    "right_elbow_joint": 1.00,
    "right_wrist_pitch_joint": 0.23,
    "waist_yaw_joint": 0.21,
}
LIFT_POSE = {
    "right_shoulder_pitch_joint": -0.18,
    "right_shoulder_roll_joint": 0.20,
    "right_shoulder_yaw_joint": 0.25,
    "right_elbow_joint": 0.72,
    "right_wrist_pitch_joint": 0.10,
    "waist_yaw_joint": 0.18,
}
PLACE_POSE = {
    "right_shoulder_pitch_joint": -0.84,
    "right_shoulder_roll_joint": 0.04,
    "right_shoulder_yaw_joint": 0.07,
    "right_elbow_joint": 1.60,
    "right_wrist_pitch_joint": 0.05,
    "waist_yaw_joint": 0.34,
}
REACH_POSE = PICK_POSE


def phase_alpha(step: int, start: int, end: int) -> float:
    if end <= start:
        return 1.0
    return float(np.clip((step - start) / float(end - start), 0.0, 1.0))


def blend_poses(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    t = float(np.clip(t, 0.0, 1.0))
    keys = set(a) | set(b)
    out: dict[str, float] = {}
    for key in keys:
        av = float(a[key]) if key in a else float(b[key])
        bv = float(b[key]) if key in b else av
        out[key] = (1.0 - t) * av + t * bv
    return out


def drive_named_pose(model: Any, data: Any, names: dict[str, int], pose: dict[str, float]) -> None:
    """Write joint qpos+ctrl so eval motion is not waiting on weak wrist actuators."""
    import mujoco

    for name, val in pose.items():
        idx = names.get(name)
        if idx is None:
            continue
        jnt = int(model.actuator_trnid[idx, 0])
        if jnt < 0:
            continue
        data.qpos[int(model.jnt_qposadr[jnt])] = float(val)
        data.qvel[int(model.jnt_dofadr[jnt])] = 0.0
        data.ctrl[idx] = float(val)
    mujoco.mj_forward(model, data)


def drive_ctrl_subset(model: Any, data: Any, names: dict[str, int], ctrl: np.ndarray) -> None:
    import mujoco

    for name, idx in names.items():
        if not any(bit in name for bit in ("shoulder", "elbow", "wrist", "waist_")):
            continue
        jnt = int(model.actuator_trnid[idx, 0])
        if jnt < 0:
            continue
        data.qpos[int(model.jnt_qposadr[jnt])] = float(ctrl[idx])
        data.qvel[int(model.jnt_dofadr[jnt])] = 0.0
    mujoco.mj_forward(model, data)


def named_pose_ctrl(
    hold: np.ndarray,
    names: dict[str, int],
    pose: dict[str, float],
    alpha: float,
) -> np.ndarray:
    ctrl = np.array(hold, copy=True)
    a = float(np.clip(alpha, 0.0, 1.0))
    for name, val in pose.items():
        idx = names.get(name)
        if idx is None:
            continue
        ctrl[idx] = (1.0 - a) * float(hold[idx]) + a * float(val)
    return ctrl


def idle_stand_ctrl(
    hold: np.ndarray,
    names: dict[str, int],
    step: int,
    horizon: int,
) -> np.ndarray:
    """Raise both arms and wave so a stand eval is not a still photo."""
    ctrl = np.array(hold, copy=True)
    t = float(step) / float(max(horizon - 1, 1))
    up = min(1.0, t / 0.18)
    wave = float(np.sin(2.0 * np.pi * 3.0 * t))

    def nudge(name: str, delta: float) -> None:
        idx = names.get(name)
        if idx is None:
            return
        ctrl[idx] = float(hold[idx] + delta)

    nudge("right_shoulder_pitch_joint", -1.25 * up)
    nudge("right_shoulder_roll_joint", 0.45 * up)
    nudge("right_elbow_joint", 0.25 * up)
    nudge("right_shoulder_yaw_joint", 0.55 * wave * up)
    nudge("waist_yaw_joint", 0.55 * wave)
    nudge("left_shoulder_pitch_joint", -1.05 * up)
    nudge("left_shoulder_roll_joint", -0.40 * up)
    nudge("left_elbow_joint", 0.25 * up)
    nudge("left_shoulder_yaw_joint", -0.45 * wave * up)
    return ctrl


def ik_toward(
    mujoco: Any,
    model: Any,
    data: Any,
    hand_id: int,
    target: np.ndarray,
    arm_actuators: list[int],
    hold: np.ndarray,
    gain: float = 0.55,
    damping: float = 1e-3,
) -> np.ndarray:
    """Damped-least-squares Jacobian IK on arm/waist position actuators."""
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, hand_id)
    cols = []
    for idx in arm_actuators:
        jnt = int(model.actuator_trnid[idx, 0])
        if jnt < 0:
            continue
        cols.append((idx, int(model.jnt_dofadr[jnt]), int(model.jnt_qposadr[jnt]), jnt))
    if not cols:
        return np.array(hold, copy=True)
    J = jacp[:, [c[1] for c in cols]]
    err = np.asarray(target, dtype=np.float64) - np.asarray(data.xpos[hand_id], dtype=np.float64)
    jj = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(jj, err * gain)
    ctrl = np.array(hold, copy=True)
    for (idx, _dof, qadr, jnt), delta in zip(cols, dq):
        lo, hi = -2.0, 2.0
        if int(model.jnt_limited[jnt]):
            lo, hi = float(model.jnt_range[jnt, 0]), float(model.jnt_range[jnt, 1])
        ctrl[idx] = float(np.clip(float(data.qpos[qadr]) + float(delta), lo, hi))
    return ctrl
