from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.adapters.base import EnginePayload, EvalResult, LogFn, Support
from humanoid_training.adapters.common import ignored_scene_fields, recipe_adapter_config
from humanoid_training.adapters import mujoco_control as control
from humanoid_training.adapters import mujoco_runtime as runtime
from humanoid_training.assets import resolve_mjcf
from humanoid_training.compose import (
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

# Tests monkeypatch these names on this module. Launch looks them up here
# at runtime so a patch is visible to hold / imitation without importing
# mujoco_control's constants directly inside step closures.
_PICK_POSE = control.PICK_POSE
_LIFT_POSE = control.LIFT_POSE
_PLACE_POSE = control.PLACE_POSE
_REACH_POSE = control.REACH_POSE
_idle_stand_ctrl = control.idle_stand_ctrl
_named_pose_ctrl = control.named_pose_ctrl
_drive_named_pose = control.drive_named_pose
_drive_ctrl_subset = control.drive_ctrl_subset
_blend_poses = control.blend_poses
_phase_alpha = control.phase_alpha
_ik_toward = control.ik_toward
_make_renderer = runtime.make_renderer
_snapshot_freejoint = runtime.snapshot_freejoint
_reset = runtime.reset
_simulate = runtime.simulate
_hold_ctrl = runtime.hold_ctrl
_body_id = runtime.body_id
_find_hand = runtime.find_hand
_actuator_name_map = runtime.actuator_name_map
_arm_actuator_ids = runtime.arm_actuator_ids
_placement_report = runtime.placement_report
_video_notes = runtime.video_notes
_model_with_stand_camera = runtime.model_with_stand_camera


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
                "Imitation: linear BC steers mocap mustard from demos; G1 right arm "
                "plays pick/lift/place poses to follow. Not finger grasping, not ACT."
            )
        else:
            notes.append(
                "Stand holds the keyframe while both arms raise and wave — not walking RL."
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
    pin_base = _snapshot_freejoint(model, data)
    if pin_base is not None:
        log("pinning floating base so arm idle cannot tip the G1")
    arm_driven = False

    def stand_idle(_model: Any, _data: Any, step: int) -> None:
        nonlocal arm_driven
        if ctrl is None:
            return
        ctrl[:] = _idle_stand_ctrl(hold, names, step, horizon)
        if names:
            arm_driven = True
        _drive_ctrl_subset(_model, _data, names, ctrl)

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
        pin_base=pin_base,
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
        if arm_driven:
            motion_note = (
                "open-loop arm raise+wave on the stand keyframe — not a balance policy, not walking"
            )
        elif model.nu == 0:
            motion_note = (
                "static hold (fixture has no actuators) — not arm wave, not balance policy"
            )
        else:
            motion_note = (
                "stand keyframe hold — arm actuators not mapped; not a balance policy"
            )
        notes = [
            f"mean_pelvis_z={mean_z:.3f}",
            f"min_z={min_z}",
            f"hold_s={hold_s}",
            motion_note,
            *(["pelvis pinned (no balance policy)"] if pin_base is not None else []),
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
        facts={
            "kind": "scene_preview"
            if preview and success_type == "object-in-container"
            else "hold",
            "mean_pelvis_z": mean_z,
            "arm_driven": bool(arm_driven),
            "pinned": pin_base is not None,
            "nu": int(model.nu),
        },
    )


def _launch_imitation(spec: dict[str, Any], run_dir: Path, log: LogFn, mujoco: Any) -> EvalResult:
    cfg = recipe_adapter_config(spec, "mujoco")
    mjcf = resolve_mjcf(str(cfg.get("mjcf")), log=log)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    local = resolve_local_dataset(spec)
    dataset_source = "local"
    if local:
        data_dir = Path(local["path"])
        log(f"using LeRobot dataset {data_dir}")
    else:
        data_dir = run_dir / "lerobot_dataset"
        dataset_source = "auto-scripted"
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
    try:
        obs, act = load_lerobot_arrays(data_dir, keep_episodes=keep)
    except ValueError as err:
        raise AdapterUnavailable(
            f"{err} Uncheck fewer takes in Data, or clear data.keep_episodes."
        ) from err
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
    pin_base = _snapshot_freejoint(model, data) if use_arm else None
    seed_steps = min(280, max(12, horizon // 5))
    puppet = pin_base is not None
    if use_arm:
        log(
            f"right-arm {'playback+BC' if puppet else 'IK+BC'} on body id={hand_id} "
            f"actuators={len(arm_acts)}"
        )
    if puppet:
        log("pinning floating base — arm qpos playback; mustard path is linear BC")
    grasped = False
    released = False
    done = False
    waypoint = start.copy()
    grasp_step = -1
    place_step = -1
    place_dist = -1.0
    carry_span0 = 0.0
    bc_steps = 0
    hold_pose = {name: float(hold[idx]) for name, idx in names.items()} if hold is not None else {}
    reach_end = max(80, int(horizon * 0.22))
    settle_end = reach_end + max(40, int(horizon * 0.04))
    recover_steps = 180
    carry_clip = 0.0022
    stop_box = {"stop": False}

    def step_fn(_model: Any, _data: Any, _step: int = 0) -> None:
        nonlocal pos, grasped, released, done, waypoint, grasp_step, place_step
        nonlocal carry_span0, bc_steps, place_dist
        if use_arm and puppet:
            if not grasped:
                alpha = _phase_alpha(_step, 0, reach_end)
                pose = _blend_poses(hold_pose, _PICK_POSE, alpha)
                if _step >= reach_end:
                    pose = dict(_PICK_POSE)
                _drive_named_pose(_model, _data, names, pose)
                ctrl[:] = _named_pose_ctrl(hold, names, pose, 1.0)
                _data.mocap_pos[mocap] = start
                pos = start.copy()
                hand = np.asarray(_data.xpos[hand_id], dtype=np.float64)
                dist_h = float(np.linalg.norm(hand - start))
                if _step >= settle_end - 1 and dist_h < 0.12:
                    grasped = True
                    grasp_step = _step
                    carry_span0 = float(
                        np.hypot(start[0] - bowl_xy[0], start[1] - bowl_xy[1])
                    )
                    waypoint = start.copy()
                    log(f"attach at step {_step} hand_mustard_dist={dist_h:.3f}m")
                return
            if grasped and not released:
                obs_t = np.array(
                    [waypoint[0], waypoint[1], bowl_xy[0], bowl_xy[1]], dtype=np.float64
                )
                delta = predict_linear_bc(weights, obs_t)
                bc_steps += 1
                waypoint[0] += float(np.clip(delta[0], -carry_clip, carry_clip))
                waypoint[1] += float(np.clip(delta[1], -carry_clip, carry_clip))
                dist = float(np.hypot(waypoint[0] - bowl_xy[0], waypoint[1] - bowl_xy[1]))
                progress = 1.0 - dist / max(carry_span0, 1e-3)
                progress = float(np.clip(progress, 0.0, 1.0))
                if progress < 0.3:
                    pose = _blend_poses(_PICK_POSE, _LIFT_POSE, progress / 0.3)
                else:
                    pose = _blend_poses(_LIFT_POSE, _PLACE_POSE, (progress - 0.3) / 0.7)
                _drive_named_pose(_model, _data, names, pose)
                ctrl[:] = _named_pose_ctrl(hold, names, pose, 1.0)
                waypoint[2] = table_z + 0.05 * min(1.0, dist / 0.2)
                _data.mocap_pos[mocap] = waypoint
                pos = waypoint.copy()
                if dist <= radius:
                    released = True
                    place_step = _step
                    place_dist = dist
                    pos = np.array([bowl_xy[0], bowl_xy[1], table_z], dtype=np.float64)
                    _data.mocap_pos[mocap] = pos
                    log(f"place at step {_step} mustard_bowl_dist={dist:.3f}m (BC path)")
                elif bc_steps > max(400, horizon // 2):
                    released = True
                    place_step = _step
                    place_dist = dist
                    log(f"place timeout at step {_step} mustard_bowl_dist={dist:.3f}m")
                return
            if not done:
                recover_t = _phase_alpha(_step, max(place_step, 0), max(place_step, 0) + recover_steps)
                pose = _blend_poses(_PLACE_POSE, hold_pose, recover_t)
                _drive_named_pose(_model, _data, names, pose)
                ctrl[:] = _named_pose_ctrl(hold, names, pose, 1.0)
                _data.mocap_pos[mocap] = pos
                if recover_t >= 1.0:
                    done = True
                    stop_box["stop"] = True
                    log(f"eval done at step {_step}")
                return
            _data.mocap_pos[mocap] = pos
            return
        if use_arm:
            hand = np.asarray(_data.xpos[hand_id], dtype=np.float64)
            if not grasped and not released:
                if _step < seed_steps:
                    alpha = float(_step + 1) / float(max(seed_steps, 1))
                    ctrl[:] = _named_pose_ctrl(hold, names, _PICK_POSE, alpha)
                else:
                    hover = start.copy()
                    hover[2] = table_z + (0.06 if _step < seed_steps + 40 else 0.0)
                    ctrl[:] = _ik_toward(
                        mujoco, _model, _data, hand_id, hover, arm_acts, hold, gain=0.75
                    )
                _data.mocap_pos[mocap] = start
                pos = start.copy()
                dist_h = float(np.linalg.norm(hand - start))
                if _step >= seed_steps and dist_h < 0.08:
                    grasped = True
                    grasp_step = _step
                    waypoint = start.copy()
                    log(f"attach at step {_step} hand_mustard_dist={dist_h:.3f}m")
                return
            if grasped and not released:
                obs_t = np.array(
                    [waypoint[0], waypoint[1], bowl_xy[0], bowl_xy[1]], dtype=np.float64
                )
                delta = predict_linear_bc(weights, obs_t)
                bc_steps += 1
                clip = 0.012
                waypoint[0] += float(np.clip(delta[0], -clip, clip))
                waypoint[1] += float(np.clip(delta[1], -clip, clip))
                dist = float(np.hypot(waypoint[0] - bowl_xy[0], waypoint[1] - bowl_xy[1]))
                waypoint[2] = table_z + 0.05 * min(1.0, dist / 0.2)
                ctrl[:] = _ik_toward(
                    mujoco, _model, _data, hand_id, waypoint, arm_acts, hold, gain=0.75
                )
                _data.mocap_pos[mocap] = hand
                pos = hand.copy()
                hand_dist = float(np.hypot(hand[0] - bowl_xy[0], hand[1] - bowl_xy[1]))
                if dist <= radius and hand_dist <= radius + 0.08:
                    released = True
                    place_step = _step
                    place_dist = dist
                    pos = np.array([bowl_xy[0], bowl_xy[1], table_z], dtype=np.float64)
                    _data.mocap_pos[mocap] = pos
                    log(f"place at step {_step} hand_bowl_dist={hand_dist:.3f}m")
                    stop_box["stop"] = True
                return
            _data.mocap_pos[mocap] = pos
            ctrl[:] = hold
            return
        obs_t = np.array([pos[0], pos[1], bowl_xy[0], bowl_xy[1]], dtype=np.float64)
        delta = predict_linear_bc(weights, obs_t)
        bc_steps += 1
        pos[0] += float(delta[0])
        pos[1] += float(delta[1])
        dist = float(np.hypot(pos[0] - bowl_xy[0], pos[1] - bowl_xy[1]))
        lift = 0.05 * min(1.0, dist / 0.2)
        pos[2] = table_z + lift
        _data.mocap_pos[mocap] = pos
        if dist <= radius:
            place_dist = dist
            stop_box["stop"] = True

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
        pin_base=pin_base,
        stop_box=stop_box,
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
    # Prefer pre-snap place distance so notes are not cosmetically 0.000.
    report_dist = float(place_dist) if place_dist >= 0 else dist
    in_bowl = report_dist <= radius or dist <= radius
    stand_rate = float(np.mean(upright[-need_steps:] if upright else [0.0]))
    mean_z = float(np.mean(zs)) if zs else 0.0
    arm_mode = "playback+BC" if puppet else ("IK+BC" if use_arm else "mocap-BC")
    never_attached = bool(use_arm and grasp_step < 0)
    container_id = str(success_cfg.get("container") or bowl.get("id") or "bowl")
    if never_attached:
        in_bowl = False
        log(
            f"arm never attached — BC carry did not run; marking {obj_id}-in-{container_id} failed"
        )
    passed = bool(in_bowl and stand_rate >= 0.9 and mean_z >= min_z and not never_attached)
    if arm_mode == "mocap-BC":
        lead = (
            f"{obj_id} carry: linear BC on demos (mocap only) — no arm actuators; "
            "not finger grasping, not ACT"
        )
    elif arm_mode == "IK+BC":
        lead = (
            f"G1 arm: IK reach + linear BC {obj_id} — mocap attach, not finger grasping, not ACT"
        )
    else:
        lead = (
            f"G1 arm: pick pose playback; {obj_id} carry: linear BC on demos — "
            "mocap, not finger grasping, not ACT"
        )
    log(
        f"{obj_id}-to-{container_id} dist={report_dist:.3f}m radius={radius:.3f}m in_bowl={in_bowl}"
    )
    dist_note = f"{obj_id}_{container_id}_dist={report_dist:.3f}"
    notes = [
        lead,
        f"dataset={dataset_source} frames={len(obs)} keep_episodes={keep if keep is not None else 'all'} bc_steps={bc_steps}",
        dist_note,
        f"mean_pelvis_z={mean_z:.3f}",
        f"arm_mode={arm_mode}",
        *(["pelvis pinned (no balance policy)"] if pin_base is not None else []),
        f"attach_step={grasp_step} placed_step={place_step}",
        *_video_notes(frames),
    ]
    if never_attached:
        notes.append("arm never attached — BC did not run (IK/reach miss)")
    return EvalResult(
        success_rate=1.0 if passed else (0.6 if in_bowl else 0.0),
        mean_return=mean_z,
        episodes=1,
        video_path=video_path,
        passed=passed,
        notes=notes,
        facts={
            "kind": "imitation",
            "arm_mode": arm_mode,
            "bc_steps": int(bc_steps),
            "frames": int(len(obs)),
            "keep_episodes": keep if keep is not None else "all",
            "dataset": dataset_source,
            "object": obj_id,
            "container": container_id,
            "object_container_dist": report_dist,
            "attach_step": int(grasp_step),
            "placed_step": int(place_step),
            "never_attached": never_attached,
            "pinned": pin_base is not None,
        },
    )
