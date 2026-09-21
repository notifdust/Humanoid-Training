from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.adapters.base import EnginePayload, EvalResult, LogFn, Support
from humanoid_training.adapters.common import ignored_scene_fields, recipe_adapter_config
from humanoid_training.train_pg import train_linear_policy
from humanoid_training.video import write_eval_video


class GymnasiumAdapter:
    name = "gymnasium"

    def support(self, spec: dict[str, Any]) -> Support:
        cfg = recipe_adapter_config(spec, self.name)
        if cfg.get("unsupported"):
            return Support(False, str(cfg["unsupported"]))
        if not cfg.get("env_id"):
            return Support(False, "recipe does not define adapters.gymnasium.env_id")
        method = (spec.get("train") or {}).get("method")
        if method != "rl":
            return Support(False, f"gymnasium adapter only runs train.method=rl (got {method})")
        return Support(True)

    def compile(self, spec: dict[str, Any]) -> EnginePayload:
        cfg = recipe_adapter_config(spec, self.name)
        env_id = str(cfg["env_id"])
        steps = int((spec.get("train") or {}).get("steps") or 400)
        seed = int((spec.get("train") or {}).get("seed") or 1)
        payload = EnginePayload(
            adapter=self.name,
            env_name=env_id,
            command=[
                "ht",
                "train",
                "--adapter",
                "gymnasium",
                f"--env={env_id}",
            ],
            ignored_fields=ignored_scene_fields(spec, honors_scene=False),
            notes=[
                "CPU smoke-test backend. Not a humanoid engine.",
                f"Linear policy gradient for {env_id}, {steps} episodes, seed {seed}.",
            ],
            extra={"env_id": env_id, "steps": steps, "seed": seed},
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
            import gymnasium as gym
        except ImportError as exc:
            raise RuntimeError(
                "gymnasium is not installed. Run: pip install -e ."
            ) from exc

        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        cfg = recipe_adapter_config(spec, self.name)
        env_id = str(cfg.get("env_id") or payload.env_name)
        train_cfg = spec.get("train") or {}
        eval_cfg = spec.get("eval") or {}
        steps = int(train_cfg.get("steps") or 400)
        seed = int(train_cfg.get("seed") or 1)
        eval_episodes = int(eval_cfg.get("episodes") or 5)
        record_video = bool(eval_cfg.get("record_video", True))

        log(f"training {env_id} for {steps} episodes (seed={seed})")
        weights, history, greedy_eval = train_linear_policy(
            env_id=env_id,
            episodes=steps,
            seed=seed,
            log=log,
        )
        # Store both keys so checkpoint.npz matches BC layout (weights) and legacy gym (W).
        np.savez(run_dir / "checkpoint.npz", W=weights, weights=weights)
        (run_dir / "train_returns.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        env = gym.make(env_id, render_mode="rgb_array" if record_video else None)
        returns: list[float] = []
        frames: list[np.ndarray] = []
        max_eval_frames = 1800
        for ep in range(eval_episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)
            done = False
            ep_return = 0.0
            while not done:
                if record_video and len(frames) < max_eval_frames:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
                action = int(np.argmax(obs @ weights))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                done = bool(terminated or truncated)
            returns.append(ep_return)
            log(f"eval episode {ep + 1}/{eval_episodes} return={ep_return:.0f}")
        env.close()

        video_path = None
        if record_video:
            video_path = write_eval_video(frames, run_dir / "eval.mp4", log=log)

        mean_return = float(np.mean(returns)) if returns else 0.0
        success = (spec.get("task") or {}).get("success") or {}
        threshold = float(success.get("min_return", 400.0))
        passed_flags = [r >= threshold for r in returns]
        success_rate = float(np.mean(passed_flags)) if passed_flags else 0.0
        passed = success_rate >= 0.8 and mean_return >= threshold
        notes = [
            f"mean_return={mean_return:.1f}",
            f"success_threshold={threshold:.0f}",
            f"greedy_train_eval={greedy_eval:.1f}",
            "checkpoint.npz is the linear policy weights (eval used in-memory copy)",
        ]
        notes = [n for n in notes if n]
        return EvalResult(
            success_rate=success_rate,
            mean_return=mean_return,
            episodes=len(returns),
            video_path=video_path,
            passed=passed,
            notes=notes,
            facts={
                "kind": "rl",
                "greedy_train_eval": float(greedy_eval),
                "success_threshold": threshold,
            },
        )
