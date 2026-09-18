from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.adapters.base import EnginePayload, EvalResult, LogFn, Support
from humanoid_training.adapters.common import ignored_scene_fields, recipe_adapter_config
from humanoid_training.assets import resolve_mjcf
from humanoid_training.compose import compose_mjcf, object_world_pos, scene_wants_compose, table_layout
from humanoid_training.errors import AdapterUnavailable
from humanoid_training.video import write_eval_video


class MujocoAdapter:
    """CPU MuJoCo adapter. Hold a keyframe (stand) and record an eval video."""

    name = "mujoco"

    def support(self, spec: dict[str, Any]) -> Support:
        cfg = recipe_adapter_config(spec, self.name)
        if cfg.get("unsupported"):
            return Support(False, str(cfg["unsupported"]))
        if not cfg.get("mjcf"):
            return Support(False, "recipe does not define adapters.mujoco.mjcf")
        method = (spec.get("train") or {}).get("method")
        if method in {"hold", "rl"}:
            return Support(True)
        if cfg.get("scene_preview"):
            return Support(True)
        return Support(False, f"mujoco adapter runs hold, rl, or a scene preview (got {method})")

    def compile(self, spec: dict[str, Any]) -> EnginePayload:
        cfg = recipe_adapter_config(spec, self.name)
        mjcf = str(cfg["mjcf"])
        honors = bool(cfg.get("honors_scene")) or scene_wants_compose(spec, cfg)
        notes = [
            "Native MuJoCo on CPU. Hold-keyframe is a preview, not locomotion RL.",
            f"MJCF {mjcf}",
        ]
        if honors:
            notes.append("scene.objects are compiled into the MJCF (table + primitives).")
        if cfg.get("scene_preview") and (spec.get("train") or {}).get("method") == "imitation":
            notes.append(
                "Scene preview only — imitation from demos is Phase 2. "
                "This run does not train ACT."
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

        cfg = recipe_adapter_config(spec, self.name)
        mjcf = resolve_mjcf(str(cfg.get("mjcf") or payload.env_name), log=log)
        os.environ.setdefault("MUJOCO_GL", "glfw")
        honors = bool(cfg.get("honors_scene")) or scene_wants_compose(spec, cfg)
        scene = spec.get("scene") or {}
        xml_path = run_dir / "composed_scene.xml"
        if honors:
            log(f"composing scene into {mjcf}")
            model, _xml = compose_mjcf(mjcf, scene, dest_xml=xml_path)
            log(f"wrote {xml_path.name}")
        else:
            log(f"loading {mjcf}")
            model = mujoco.MjModel.from_xml_path(str(mjcf))
        data = mujoco.MjData(model)
        keyframe = int(cfg.get("keyframe", 0))
        if model.nkey > 0:
            idx = min(keyframe, model.nkey - 1)
            mujoco.mj_resetDataKeyframe(model, data, idx)
            log(f"reset to keyframe {idx}")
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        ctrl = _hold_ctrl(model, data)
        horizon = int(cfg.get("horizon", 180))
        render_every = int(cfg.get("render_every", 3))
        eval_cfg = spec.get("eval") or {}
        record_video = bool(eval_cfg.get("record_video", True))
        success_cfg = (spec.get("task") or {}).get("success") or {}
        min_z = float(success_cfg.get("min_z", 0.5))
        hold_s = float(success_cfg.get("hold_s", 1.0))
        dt = float(model.opt.timestep)
        need_steps = max(1, int(hold_s / max(dt, 1e-4)))

        renderer = None
        frames: list[np.ndarray] = []
        camera: Any = -1
        if model.ncam > 0:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ht_eval")
            if cam_id >= 0:
                camera = cam_id
        if record_video:
            try:
                renderer = mujoco.Renderer(model, 360, 640)
            except Exception as err:
                log(f"renderer unavailable ({err}); physics-only eval")

        pelvis = _body_id(mujoco, model, "pelvis")
        if pelvis < 0:
            pelvis = 1 if model.nbody > 1 else 0
        layout = table_layout(scene)
        objects = list(scene.get("objects") or []) if honors else []
        object_ids = {
            str(obj["id"]): _body_id(mujoco, model, str(obj["id"]))
            for obj in objects
            if obj.get("id")
        }

        upright = []
        zs = []
        for step in range(horizon):
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

        video_path = None
        if frames:
            video_path = write_eval_video(frames, run_dir / "eval.mp4", fps=30, log=log)

        last_hold = upright[-need_steps:] if upright else []
        stand_rate = float(np.mean(last_hold)) if last_hold else 0.0
        mean_z = float(np.mean(zs)) if zs else 0.0
        placement = _placement_report(data, object_ids, objects, layout)
        preview = bool(cfg.get("scene_preview"))
        success_type = str(success_cfg.get("type") or "upright-duration")

        if preview and success_type == "object-in-container":
            placed = all(row["ok"] for row in placement) if placement else True
            passed = stand_rate >= 0.9 and mean_z >= min_z and placed
            success_rate = 1.0 if passed else (0.5 if placed else 0.0)
            notes = [
                "scene preview — not pick-and-place skill, not imitation training",
                f"mean_pelvis_z={mean_z:.3f}",
                f"objects_placed={sum(r['ok'] for r in placement)}/{len(placement)}",
            ]
            notes.extend(
                f"{row['id']} xy_err={row['xy_err']:.3f}m" for row in placement
            )
        else:
            success_rate = stand_rate
            passed = stand_rate >= 0.9 and mean_z >= min_z
            notes = [
                f"mean_pelvis_z={mean_z:.3f}",
                f"min_z={min_z}",
                f"hold_s={hold_s}",
            ]
            if placement:
                notes.append(
                    f"objects_placed={sum(r['ok'] for r in placement)}/{len(placement)}"
                )

        return EvalResult(
            success_rate=success_rate,
            mean_return=mean_z,
            episodes=1,
            video_path=video_path,
            passed=passed,
            notes=notes,
        )


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
