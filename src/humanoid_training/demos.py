from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from humanoid_training.assets import cache_dir
from humanoid_training.compose import object_world_pos, primitive_for, table_layout
from humanoid_training.datasets import write_lerobot_dataset
from humanoid_training.errors import RecipeError


def dataset_cache_dir(name: str) -> Path:
    return cache_dir() / "datasets" / name


def require_pick_objects(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mustard and bowl must be on the spec. Silent 0,0 fallbacks make BC a no-op."""
    success = (spec.get("task") or {}).get("success") or {}
    obj_id = str(success.get("object") or "mustard")
    container_id = str(success.get("container") or "bowl")
    objects = list((spec.get("scene") or {}).get("objects") or [])
    if not objects:
        raise RecipeError(
            "Recording pick-and-place demos needs scene.objects (mustard and bowl). "
            "Open the pick-and-place recipe so the kitchen counter is on the spec."
        )
    ids = {str(obj.get("id")) for obj in objects}
    missing = [name for name in (obj_id, container_id) if name not in ids]
    if missing:
        raise RecipeError(
            "scene.objects must include "
            + " and ".join(f"'{name}'" for name in (obj_id, container_id))
            + f" (missing {missing})."
        )
    return scene_object(spec, obj_id), scene_object(spec, container_id)


def scene_object(spec: dict[str, Any], object_id: str) -> dict[str, Any]:
    for obj in (spec.get("scene") or {}).get("objects") or []:
        if obj.get("id") == object_id:
            return dict(obj)
    raise RecipeError(f"scene.objects has no '{object_id}'")


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
    mustard, bowl = require_pick_objects(spec)
    layout = table_layout(spec.get("scene") or {})
    start = np.array(object_world_pos(mustard, layout)[:2], dtype=np.float64)
    goal = np.array(object_world_pos(bowl, layout)[:2], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n = max(1, int(episodes))
    recorded = []
    for i in range(n):
        fail = bool(include_failure) and i == n - 1
        offset = rng.uniform(-0.05, 0.05, size=2) if i else np.zeros(2)
        origin = start + offset
        recorded.append(_scripted_episode(origin, goal, horizon=horizon, fail=fail))
    return _write_pick_dataset(spec, dest, recorded)


def record_object_trajectories(
    spec: dict[str, Any],
    dest: Path,
    trajectories: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Write a LeRobot dataset from canvas-drag paths in table-frame x/y.

    Same observation/action schema as the scripted expert: world-xy mustard
    and bowl, action = delta. Not G1 grasping.
    """
    _mustard, bowl = require_pick_objects(spec)
    layout = table_layout(spec.get("scene") or {})
    bowl_xy = np.array(object_world_pos(bowl, layout)[:2], dtype=np.float64)
    radius = float(primitive_for(bowl)["size"][0])
    recorded = []
    for raw in trajectories:
        points = [(float(p.get("x", 0.0)), float(p.get("y", 0.0))) for p in raw if isinstance(p, dict)]
        if not points:
            continue
        resampled = _resample_points(points)
        world = np.array([_table_to_world_xy(x, y, layout) for x, y in resampled], dtype=np.float64)
        frames = []
        for i, pos in enumerate(world):
            nxt = world[i + 1] if i + 1 < len(world) else pos
            action = nxt - pos
            frames.append(
                {
                    "observation": [float(pos[0]), float(pos[1]), float(bowl_xy[0]), float(bowl_xy[1])],
                    "action": [float(action[0]), float(action[1])],
                }
            )
        dist = float(np.hypot(world[-1, 0] - bowl_xy[0], world[-1, 1] - bowl_xy[1]))
        recorded.append({"frames": frames, "success": dist <= radius})
    if not recorded:
        raise RecipeError("No canvas trajectories to save. Drag the mustard into the bowl first.")
    return _write_pick_dataset(spec, dest, recorded)


def _write_pick_dataset(
    spec: dict[str, Any],
    dest: Path,
    recorded: list[dict[str, Any]],
) -> dict[str, Any]:
    task = str(
        (spec.get("task") or {}).get("language")
        or "Pick up the object and put it in the container."
    )
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
    for _t in range(horizon):
        action = gain * (target - pos)
        obs = [float(pos[0]), float(pos[1]), float(goal[0]), float(goal[1])]
        frames.append({"observation": obs, "action": [float(action[0]), float(action[1])]})
        pos = pos + action
    inside = float(np.hypot(pos[0] - goal[0], pos[1] - goal[1])) < 0.09
    return {"frames": frames, "success": bool(inside and not fail)}


def _table_to_world_xy(x: float, y: float, layout: dict[str, Any]) -> tuple[float, float]:
    tx, ty, _tz = layout["table_pos"]
    return (float(tx) + float(x), float(ty) + float(y))


def _resample_points(
    points: Sequence[tuple[float, float]],
    n_min: int = 8,
    n_max: int = 80,
) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RecipeError("Each demo sample needs x and y.")
    if arr.shape[0] == 1:
        arr = np.vstack([arr, arr])
    n = int(arr.shape[0])
    target = min(n_max, max(n_min, n))
    if n == target:
        return arr
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])
    if total < 1e-8:
        return np.repeat(arr[:1], target, axis=0)
    samples = np.linspace(0.0, total, target)
    out = np.empty((target, 2), dtype=np.float64)
    out[:, 0] = np.interp(samples, dist, arr[:, 0])
    out[:, 1] = np.interp(samples, dist, arr[:, 1])
    return out
