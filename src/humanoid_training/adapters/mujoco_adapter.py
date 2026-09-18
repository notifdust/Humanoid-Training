from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.adapters.base import EnginePayload, EvalResult, LogFn, Support
from humanoid_training.adapters.common import ignored_scene_fields, recipe_adapter_config
from humanoid_training.assets import resolve_mjcf
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
        if method not in {"hold", "rl"}:
            return Support(False, f"mujoco adapter runs hold or rl (got {method})")
        return Support(True)

    def compile(self, spec: dict[str, Any]) -> EnginePayload:
        cfg = recipe_adapter_config(spec, self.name)
        mjcf = str(cfg["mjcf"])
        payload = EnginePayload(
            adapter=self.name,
            env_name=mjcf,
            command=["ht", "train", "--adapter", "mujoco"],
            ignored_fields=ignored_scene_fields(
                spec, honors_scene=bool(cfg.get("honors_scene"))
            ),
            notes=[
                "Native MuJoCo on CPU. Hold-keyframe is a preview, not locomotion RL.",
                f"MJCF {mjcf}",
            ],
            extra={"mjcf": mjcf, "keyframe": cfg.get("keyframe", 0)},
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
        log(f"loading {mjcf}")
        os.environ.setdefault("MUJOCO_GL", "glfw")
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
        if record_video:
            try:
                renderer = mujoco.Renderer(model, 360, 640)
            except Exception as err:
                log(f"renderer unavailable ({err}); physics-only eval")

        pelvis = 1 if model.nbody > 1 else 0
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
                renderer.update_scene(data)
                frames.append(np.asarray(renderer.render()).copy())
            if step in {0, horizon // 2, horizon - 1}:
                log(f"step {step}/{horizon} pelvis_z={z:.3f}")

        video_path = None
        if frames:
            video_path = write_eval_video(frames, run_dir / "eval.mp4", fps=30, log=log)

        last_hold = upright[-need_steps:] if upright else []
        success_rate = float(np.mean(last_hold)) if last_hold else 0.0
        mean_z = float(np.mean(zs)) if zs else 0.0
        passed = success_rate >= 0.9 and mean_z >= min_z
        return EvalResult(
            success_rate=success_rate,
            mean_return=mean_z,
            episodes=1,
            video_path=video_path,
            passed=passed,
            notes=[
                f"mean_pelvis_z={mean_z:.3f}",
                f"min_z={min_z}",
                f"hold_s={hold_s}",
            ],
        )


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
