from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.assets import cache_dir
from humanoid_training.compose import object_world_pos, table_layout
from humanoid_training.datasets import write_lerobot_dataset


def dataset_cache_dir(name: str) -> Path:
    return cache_dir() / "datasets" / name


def scene_object(spec: dict[str, Any], object_id: str) -> dict[str, Any]:
    for obj in (spec.get("scene") or {}).get("objects") or []:
        if obj.get("id") == object_id:
            return dict(obj)
    return {"id": object_id, "x": 0.0, "y": 0.0}


def record_scripted_pick_place(
    spec: dict[str, Any],
    dest: Path,
    *,
    episodes: int = 4,
    horizon: int = 48,
    include_failure: bool = False,
    seed: int = 1,
) -> dict[str, Any]:
    """Write a LeRobot dataset of object-space pick-and-place demos.

    The expert moves the mustard primitive into the bowl. That is *not* G1
    grasping — it is a scripted demonstration of the success criterion so
    CPU imitation can close the recipe → data → train → eval loop.
    """
    success = (spec.get("task") or {}).get("success") or {}
    obj_id = str(success.get("object") or "mustard")
    container_id = str(success.get("container") or "bowl")
    layout = table_layout(spec.get("scene") or {})
    mustard = scene_object(spec, obj_id)
    bowl = scene_object(spec, container_id)
    start = np.array(object_world_pos(mustard, layout)[:2], dtype=np.float64)
    goal = np.array(object_world_pos(bowl, layout)[:2], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n = max(1, int(episodes))
    recorded = []
    for i in range(n):
        fail = bool(include_failure) and i == n - 1
        offset = rng.uniform(-0.05, 0.05, size=2) if i else np.zeros(2)
        origin = start + offset
        recorded.append(
            _scripted_episode(origin, goal, horizon=horizon, fail=fail)
        )
    task = str((spec.get("task") or {}).get("language") or "Pick up the object and put it in the container.")
    robot = str((spec.get("robot") or {}).get("id") or "unitree-g1-29dof")
    return write_lerobot_dataset(
        dest,
        recorded,
        robot_type=robot,
        task=task,
        fps=30,
    )


def _scripted_episode(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    fail: bool,
) -> dict[str, Any]:
    target = np.array(goal, dtype=np.float64)
    if fail:
        target = start + np.array([-0.22, 0.18])
    pos = np.array(start, dtype=np.float64)
    gain = 0.35
    frames = []
    for t in range(horizon):
        action = gain * (target - pos)
        obs = [float(pos[0]), float(pos[1]), float(goal[0]), float(goal[1])]
        frames.append({"observation": obs, "action": [float(action[0]), float(action[1])]})
        pos = pos + action
    inside = float(np.hypot(pos[0] - goal[0], pos[1] - goal[1])) < 0.09
    return {"frames": frames, "success": bool(inside and not fail)}
