from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.adapters.base import EnginePayload, EvalResult, LogFn, Support
from humanoid_training.adapters.common import ignored_scene_fields, recipe_adapter_config
from humanoid_training.assets import resolve_mjcf
from humanoid_training.compose import (
    STAND_CAMERA,
    camera_xyaxes,
    compose_mjcf,
    object_world_pos,
    primitive_for,
    scene_wants_compose,
    table_layout,
)
from humanoid_training.datasets import load_lerobot_arrays, resolve_local_dataset
from humanoid_training.demos import record_scripted_pick_place, require_pick_objects
from humanoid_training.errors import AdapterUnavailable, RecipeError
from humanoid_training.train_bc import fit_linear_bc, predict_linear_bc, save_bc
from humanoid_training.video import write_eval_video


class MujocoAdapter:
    """CPU MuJoCo adapter: keyframe hold, scene preview, or linear-BC imitation."""

    name = "mujoco"

    def support(self, spec: dict[str, Any]) -> Support:
        cfg = recipe_adapter_config(spec, self.name)
        if cfg.get("unsupported"):
            return Support(False, str(cfg["unsupported"]))
        if not cfg.get("mjcf"):
            return Support(False, "recipe does not define adapters.mujoco.mjcf")
        method = (spec.get("train") or {}).get("method")
        if method in {"hold", "rl", "imitation"}:
            return Support(True)
        if cfg.get("scene_preview"):
            return Support(True)
        return Support(False, f"mujoco adapter runs hold, rl, or imitation (got {method})")

    def compile(self, spec: dict[str, Any]) -> EnginePayload:
        cfg = recipe_adapter_config(spec, self.name)
        mjcf = str(cfg["mjcf"])
        honors = bool(cfg.get("honors_scene")) or scene_wants_compose(spec, cfg)
        method = (spec.get("train") or {}).get("method")
        notes = [
            "Native MuJoCo on CPU.",
            f"MJCF {mjcf}",
        ]
        if honors:
            notes.append("scene.objects are compiled into the MJCF (table + primitives).")
        if method == "imitation":
            notes.append(
                "Imitation: linear BC on object-space demos, G1 right arm reaches "
                "with a seed pose + Jacobian IK, then carries mocap mustard. "
                "Not finger grasping, not ACT."
            )
        else:
            notes.append(
                "Stand holds the keyframe while both arms raise and the waist "
                "yaw-waves — not walking RL."
            )
        payload = EnginePayload(
            adapter=self.name,
            env_name=mjcf,
            command=["ht", "train", "--adapter", "mujoco"],
            ignored_fields=ignored_scene_fields(spec, honors_scene=honors),
            notes=notes,
            extra={
                "mjcf": mjcf,
                "keyframe": cfg.get("keyframe", 0),
                "honors_scene": honors,
                "scene_preview": bool(cfg.get("scene_preview")),
                "method": method,
            },
        )
        payload.files["engine_payload.json"] = json.dumps(payload.as_dict(), indent=2)
        return payload

    def launch(
        self,
        spec: dict[str, Any],
        payload: EnginePayload,
        run_dir: Path,
        log: LogFn,
    ) -> EvalResult:
        try:
            import mujoco
        except ImportError as exc:
            raise AdapterUnavailable(
                "MuJoCo is not installed. Run: pip install 'humanoid-training[mujoco]'"
            ) from exc

        method = (spec.get("train") or {}).get("method")
        if method == "imitation":
            return _launch_imitation(spec, run_dir, log, mujoco)
        return _launch_hold(spec, run_dir, log, mujoco)


def _launch_hold(spec: dict[str, Any], run_dir: Path, log: LogFn, mujoco: Any) -> EvalResult:
    cfg = recipe_adapter_config(spec, "mujoco")
    mjcf = resolve_mjcf(str(cfg.get("mjcf")), log=log)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    honors = bool(cfg.get("honors_scene")) or scene_wants_compose(spec, cfg)
    scene = spec.get("scene") or {}
    xml_path = run_dir / "composed_scene.xml"
    if honors:
        log(f"composing scene into {mjcf}")
        model, _xml = compose_mjcf(mjcf, scene, dest_xml=xml_path)
        log(f"wrote {xml_path.name}")
    else:
        log(f"loading {mjcf} with stand eval camera")
        model = _model_with_stand_camera(mujoco, mjcf)
    data = mujoco.MjData(model)
    _reset(mujoco, model, data, int(cfg.get("keyframe", 0)), log)
    hold = _hold_ctrl(model, data)
    ctrl = None if hold is None else np.array(hold, copy=True)
    names = _actuator_name_map(mujoco, model)
    horizon = int(cfg.get("horizon", 180))

    def stand_idle(_model: Any, _data: Any, step: int) -> None:
        if ctrl is None:
            return
        ctrl[:] = _idle_stand_ctrl(hold, names, step, horizon)

    frames, zs, upright, object_ids = _simulate(
        mujoco,
        model,
        data,
        spec,
        honors=honors,
        ctrl=ctrl,
        horizon=horizon,
        render_every=int(cfg.get("render_every", 3)),
        log=log,
        step_fn=stand_idle if ctrl is not None and names else None,
    )
    video_path = None
    if frames:
        video_path = write_eval_video(frames, run_dir / "eval.mp4", fps=30, log=log)

    success_cfg = (spec.get("task") or {}).get("success") or {}
    min_z = float(success_cfg.get("min_z", 0.5))
    hold_s = float(success_cfg.get("hold_s", 1.0))
    dt = float(model.opt.timestep)
    need_steps = max(1, int(hold_s / max(dt, 1e-4)))
    last_hold = upright[-need_steps:] if upright else []
    stand_rate = float(np.mean(last_hold)) if last_hold else 0.0
    mean_z = float(np.mean(zs)) if zs else 0.0
    layout = table_layout(scene)
    objects = list(scene.get("objects") or []) if honors else []
    placement = _placement_report(data, object_ids, objects, layout)
    preview = bool(cfg.get("scene_preview"))
    success_type = str(success_cfg.get("type") or "upright-duration")

    if preview and success_type == "object-in-container":
        placed = all(row["ok"] for row in placement) if placement else True
        passed = stand_rate >= 0.9 and mean_z >= min_z and placed
        success_rate = 1.0 if passed else (0.5 if placed else 0.0)
        notes = [
            "scene preview — not pick-and-place skill; G1 idles with a visible arm raise",
            f"mean_pelvis_z={mean_z:.3f}",
            f"objects_placed={sum(r['ok'] for r in placement)}/{len(placement)}",
        ]
        notes.extend(f"{row['id']} xy_err={row['xy_err']:.3f}m" for row in placement)
    else:
        success_rate = stand_rate
        passed = stand_rate >= 0.9 and mean_z >= min_z
        notes = [
            f"mean_pelvis_z={mean_z:.3f}",
            f"min_z={min_z}",
            f"hold_s={hold_s}",
            "both arms raise and waist yaw-waves on the stand keyframe — not walking",
        ]
        if placement:
            notes.append(f"objects_placed={sum(r['ok'] for r in placement)}/{len(placement)}")
    notes.extend(_video_notes(frames))

    return EvalResult(
        success_rate=success_rate,
        mean_return=mean_z,
        episodes=1,
        video_path=video_path,
        passed=passed,
        notes=notes,
    )


def _launch_imitation(spec: dict[str, Any], run_dir: Path, log: LogFn, mujoco: Any) -> EvalResult:
    cfg = recipe_adapter_config(spec, "mujoco")
    mjcf = resolve_mjcf(str(cfg.get("mjcf")), log=log)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    local = resolve_local_dataset(spec)
    if local:
        data_dir = Path(local["path"])
        log(f"using LeRobot dataset {data_dir}")
    else:
        data_dir = run_dir / "lerobot_dataset"
        n_ep = int((spec.get("data") or {}).get("min_episodes") or 4)
        n_ep = max(2, min(n_ep, 8))
        log("no local LeRobot dataset; recording scripted object-space demos")
        try:
            meta = record_scripted_pick_place(
                spec,
                data_dir,
                episodes=n_ep,
                horizon=48,
                include_failure=False,
                seed=int((spec.get("train") or {}).get("seed") or 1),
            )
        except RecipeError as err:
            raise AdapterUnavailable(str(err)) from err
        log(f"wrote {meta.get('total_episodes')} episodes / {meta.get('total_frames')} frames")

    keep = (spec.get("data") or {}).get("keep_episodes")
    obs, act = load_lerobot_arrays(data_dir, keep_episodes=keep)
    log(f"BC fitting on {len(obs)} frames")
    weights = fit_linear_bc(obs, act)
    save_bc(run_dir / "checkpoint.npz", weights)
    log("wrote checkpoint.npz")

    success_cfg = (spec.get("task") or {}).get("success") or {}
    obj_id = str(success_cfg.get("object") or "mustard")
    try:
        mustard, bowl = require_pick_objects(spec)
    except RecipeError as err:
        raise AdapterUnavailable(str(err)) from err
    scene = spec.get("scene") or {}
    xml_path = run_dir / "composed_scene.xml"
    log(f"composing mocap object '{obj_id}' into {mjcf}")
    model, _xml = compose_mjcf(mjcf, scene, dest_xml=xml_path, movable=[obj_id])
    data = mujoco.MjData(model)
    _reset(mujoco, model, data, int(cfg.get("keyframe", 0)), log)

    layout = table_layout(scene)
    start = np.array(object_world_pos(mustard, layout), dtype=np.float64)
    bowl_xy = np.array(object_world_pos(bowl, layout)[:2], dtype=np.float64)
    mustard_id = _body_id(mujoco, model, obj_id)
    mocap = int(model.body_mocapid[mustard_id]) if mustard_id >= 0 else -1
    if mocap < 0:
        raise AdapterUnavailable(f"object '{obj_id}' is not a mocap body; cannot imitate")
    data.mocap_pos[mocap] = start
    mujoco.mj_forward(model, data)

    hold = _hold_ctrl(model, data)
    ctrl = None if hold is None else np.array(hold, copy=True)
    pos = start.copy()
    prim = primitive_for(mustard)
    table_z = float(layout["table_pos"][2] + layout["table_size"][2] + prim["half_height"] + 0.002)
    radius = float(primitive_for(bowl)["size"][0])
    hand_id = _find_hand(mujoco, model)
    arm_acts = _arm_actuator_ids(mujoco, model)
    names = _actuator_name_map(mujoco, model)
    use_arm = hand_id >= 0 and bool(arm_acts) and ctrl is not None
    horizon = int(cfg.get("horizon", 180))
    seed_steps = min(280, max(12, horizon // 5))
    grasp_close = 0.13
    grasp_late = 0.18
    carry_clip = 0.012 if horizon < 200 else 0.0035
    if use_arm:
        log(
            f"right-arm IK on body id={hand_id} actuators={len(arm_acts)} "
            f"seed_steps={seed_steps} carry_clip={carry_clip:.4f}"
        )
    grasped = False
    released = False
    waypoint = start.copy()
    grasp_step = -1
    place_step = -1

    def step_fn(_model: Any, _data: Any, _step: int = 0) -> None:
        nonlocal pos, grasped, released, waypoint, grasp_step, place_step
        if use_arm:
            hand = np.asarray(_data.xpos[hand_id], dtype=np.float64)
            if not grasped and not released:
                if _step < seed_steps:
                    alpha = float(_step + 1) / float(max(seed_steps, 1))
                    ctrl[:] = _named_pose_ctrl(hold, names, _REACH_POSE, alpha)
                else:
                    hover = start.copy()
                    hover[2] = table_z + (0.06 if _step < seed_steps + 80 else 0.0)
                    ctrl[:] = _ik_toward(
                        mujoco, _model, _data, hand_id, hover, arm_acts, hold, gain=0.75
                    )
                _data.mocap_pos[mocap] = start
                pos = start.copy()
                dist_h = float(np.linalg.norm(hand - start))
                close_enough = dist_h < grasp_close or (
                    _step >= seed_steps + 40 and dist_h < grasp_late
                )
                if close_enough:
                    grasped = True
                    grasp_step = _step
                    log(f"grasp at step {_step} hand_mustard_dist={dist_h:.3f}m")
                return
            if grasped and not released:
                obs_t = np.array(
                    [waypoint[0], waypoint[1], bowl_xy[0], bowl_xy[1]], dtype=np.float64
                )
                delta = predict_linear_bc(weights, obs_t)
                waypoint[0] += float(np.clip(delta[0], -carry_clip, carry_clip))
                waypoint[1] += float(np.clip(delta[1], -carry_clip, carry_clip))
                dist = float(np.hypot(waypoint[0] - bowl_xy[0], waypoint[1] - bowl_xy[1]))
                waypoint[2] = table_z + 0.05 * min(1.0, dist / 0.2)
                ctrl[:] = _ik_toward(
                    mujoco, _model, _data, hand_id, waypoint, arm_acts, hold, gain=0.75
                )
                _data.mocap_pos[mocap] = hand
                pos = hand.copy()
                hand_dist = float(np.hypot(hand[0] - bowl_xy[0], hand[1] - bowl_xy[1]))
                if dist <= radius and hand_dist <= radius + 0.10:
                    released = True
                    place_step = _step
                    pos = np.array([bowl_xy[0], bowl_xy[1], table_z], dtype=np.float64)
                    _data.mocap_pos[mocap] = pos
                    log(f"place at step {_step} hand_bowl_dist={hand_dist:.3f}m")
                return
            _data.mocap_pos[mocap] = pos
            retract = pos.copy()
            retract[2] += 0.10
            ctrl[:] = _ik_toward(
                mujoco, _model, _data, hand_id, retract, arm_acts, hold, gain=0.75
            )
            return
        obs_t = np.array([pos[0], pos[1], bowl_xy[0], bowl_xy[1]], dtype=np.float64)
        delta = predict_linear_bc(weights, obs_t)
        pos[0] += float(delta[0])
        pos[1] += float(delta[1])
        dist = float(np.hypot(pos[0] - bowl_xy[0], pos[1] - bowl_xy[1]))
        lift = 0.05 * min(1.0, dist / 0.2)
        pos[2] = table_z + lift
        _data.mocap_pos[mocap] = pos

    frames, zs, upright, object_ids = _simulate(
        mujoco,
        model,
        data,
        spec,
        honors=True,
        ctrl=ctrl,
        horizon=horizon,
        render_every=int(cfg.get("render_every", 3)),
        log=log,
        step_fn=step_fn,
    )
    video_path = None
    if frames:
        video_path = write_eval_video(frames, run_dir / "eval.mp4", fps=30, log=log)

    min_z = float(success_cfg.get("min_z", 0.5))
    hold_s = float(success_cfg.get("hold_s", 0.5))
    dt = float(model.opt.timestep)
    need_steps = max(1, int(hold_s / max(dt, 1e-4)))
    radius = float(primitive_for(bowl)["size"][0])
    dist = float(np.hypot(pos[0] - bowl_xy[0], pos[1] - bowl_xy[1]))
    in_bowl = dist <= radius
    stand_rate = float(np.mean(upright[-need_steps:] if upright else [0.0]))
    mean_z = float(np.mean(zs)) if zs else 0.0
    passed = bool(in_bowl and stand_rate >= 0.9 and mean_z >= min_z)
    log(f"mustard-to-bowl dist={dist:.3f}m radius={radius:.3f}m in_bowl={in_bowl}")
    return EvalResult(
        success_rate=1.0 if passed else (0.6 if in_bowl else 0.0),
        mean_return=mean_z,
        episodes=1,
        video_path=video_path,
        passed=passed,
        notes=[
            "linear BC + G1 right-arm seed pose + IK — mocap mustard, not finger grasping, not ACT",
            f"frames={len(obs)} keep_episodes={keep if keep is not None else 'all'}",
            f"mustard_bowl_dist={dist:.3f}",
            f"mean_pelvis_z={mean_z:.3f}",
            "arm_ik=on" if use_arm else "arm_ik=off (no hand body)",
            f"grasped_step={grasp_step} placed_step={place_step}",
            *_video_notes(frames),
        ],
    )


def _reset(mujoco: Any, model: Any, data: Any, keyframe: int, log: LogFn) -> None:
    if model.nkey > 0:
        idx = min(keyframe, model.nkey - 1)
        mujoco.mj_resetDataKeyframe(model, data, idx)
        log(f"reset to keyframe {idx}")
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def _simulate(
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
        renderer = _make_renderer(mujoco, model, log)

    pelvis = _body_id(mujoco, model, "pelvis")
    if pelvis < 0:
        pelvis = 1 if model.nbody > 1 else 0
    objects = list((spec.get("scene") or {}).get("objects") or []) if honors else []
    object_ids = {
        str(obj["id"]): _body_id(mujoco, model, str(obj["id"]))
        for obj in objects
        if obj.get("id")
    }
    frames: list[np.ndarray] = []
    upright: list[bool] = []
    zs: list[float] = []
    for step in range(horizon):
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
    return frames, zs, upright, object_ids


def _make_renderer(mujoco: Any, model: Any, log: LogFn) -> Any:
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


def _video_notes(frames: list[np.ndarray]) -> list[str]:
    if frames:
        return []
    return ["no eval.mp4 (headless / HT_NO_RENDER); physics success still counted"]


def _body_id(mujoco: Any, model: Any, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _placement_report(
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


def _hold_ctrl(model: Any, data: Any) -> np.ndarray | None:
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


def _model_with_stand_camera(mujoco: Any, mjcf: Path) -> Any:
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


_HAND_BODIES = (
    "right_wrist_yaw_link",
    "right_rubber_hand",
    "right_palm_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
)

_ARM_ACTUATOR_BITS = (
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "waist_yaw",
    "waist_pitch",
    "waist_roll",
)


def _find_hand(mujoco: Any, model: Any) -> int:
    for name in _HAND_BODIES:
        bid = _body_id(mujoco, model, name)
        if bid >= 0:
            return bid
    return -1


def _actuator_name_map(mujoco: Any, model: Any) -> dict[str, int]:
    names: dict[str, int] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            names[str(name)] = i
    return names


def _arm_actuator_ids(mujoco: Any, model: Any) -> list[int]:
    ids = []
    for name, idx in _actuator_name_map(mujoco, model).items():
        if any(bit in name for bit in _ARM_ACTUATOR_BITS):
            ids.append(idx)
    return sorted(ids)


# Open-loop pose that puts the standing G1 right wrist near the kitchen mustard.
_REACH_POSE = {
    "right_shoulder_pitch_joint": -0.86,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": -0.12,
    "right_elbow_joint": 1.70,
    "waist_yaw_joint": 0.40,
    "waist_pitch_joint": 0.30,
}


def _named_pose_ctrl(
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


def _idle_stand_ctrl(
    hold: np.ndarray,
    names: dict[str, int],
    step: int,
    horizon: int,
) -> np.ndarray:
    """Raise both arms and yaw-wave so a stand eval is not a still photo."""
    ctrl = np.array(hold, copy=True)
    t = float(step) / float(max(horizon - 1, 1))
    lift = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
    wave = float(np.sin(2.0 * np.pi * 2.0 * t))

    def nudge(name: str, delta: float) -> None:
        idx = names.get(name)
        if idx is None:
            return
        ctrl[idx] = float(hold[idx] + delta)

    nudge("right_shoulder_pitch_joint", -1.15 * lift)
    nudge("right_shoulder_roll_joint", 0.40 * lift)
    nudge("right_elbow_joint", 0.40 * lift + 0.18 * wave * lift)
    nudge("right_shoulder_yaw_joint", 0.25 * wave * lift)
    nudge("waist_yaw_joint", 0.50 * wave)
    nudge("left_shoulder_pitch_joint", -0.85 * lift)
    nudge("left_elbow_joint", 0.30 * lift)
    return ctrl


def _ik_toward(
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
