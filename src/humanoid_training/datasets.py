from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OBS_DIM = 4
ACT_DIM = 2
FPS = 30


def inspect_lerobot_dataset(uri: str) -> dict[str, Any]:
    """Read a local LeRobot dataset (v2 meta/info.json). Does not invent a format."""
    raw = (uri or "").strip()
    if not raw:
        return {
            "ok": False,
            "error": "Pass a file: path to a LeRobot dataset directory.",
        }
    if raw.startswith("hf:") or raw.startswith("https://huggingface.co/"):
        return {
            "ok": False,
            "uri": raw,
            "error": (
                "Hugging Face Hub import is not wired yet. Download the dataset "
                "locally and inspect a directory that contains meta/info.json."
            ),
        }
    path = _as_path(raw)
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        return {
            "ok": False,
            "uri": raw,
            "path": str(path),
            "error": (
                f"Not a LeRobot dataset: missing {info_path}. "
                "Expected the Hugging Face LeRobot layout (meta/info.json)."
            ),
        }
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        return {"ok": False, "uri": raw, "path": str(path), "error": f"Invalid info.json: {err}"}

    episodes = _load_episodes(path / "meta" / "episodes.jsonl", info)
    features = info.get("features") or {}
    return {
        "ok": True,
        "uri": raw,
        "path": str(path),
        "format": "lerobot",
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes", len(episodes)),
        "total_frames": info.get("total_frames"),
        "features": sorted(features.keys()) if isinstance(features, dict) else [],
        "episodes": episodes,
        "info": {
            k: info.get(k)
            for k in (
                "codebase_version",
                "robot_type",
                "fps",
                "total_episodes",
                "total_frames",
                "total_tasks",
            )
        },
    }


def resolve_local_dataset(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first inspectable local dataset on the spec, or None."""
    uris = list((spec.get("data") or {}).get("datasets") or [])
    for uri in uris:
        result = inspect_lerobot_dataset(str(uri))
        if result.get("ok"):
            return result
    return None


def write_lerobot_dataset(
    dest: Path,
    episodes: Sequence[dict[str, Any]],
    *,
    robot_type: str = "unitree_g1",
    task: str = "pick and place",
    fps: int = FPS,
    obs_dim: int = OBS_DIM,
    action_dim: int = ACT_DIM,
) -> dict[str, Any]:
    """Write LeRobot v2 layout: meta/info.json + episodes.jsonl + per-episode JSONL frames.

    Frames are JSONL rather than parquet so the studio does not depend on pyarrow.
    The directory shape and feature schema match LeRobot v2.1.
    """
    dest = Path(dest)
    meta = dest / "meta"
    chunk = dest / "data" / "chunk-000"
    meta.mkdir(parents=True, exist_ok=True)
    chunk.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    ep_rows = []
    global_index = 0
    for i, episode in enumerate(episodes):
        frames = list(episode.get("frames") or [])
        path = chunk / f"episode_{i:06d}.jsonl"
        lines = []
        for f_i, frame in enumerate(frames):
            obs = _as_list(frame.get("observation") or frame.get("observation.state"), obs_dim)
            act = _as_list(frame.get("action"), action_dim)
            row = {
                "observation.state": obs,
                "action": act,
                "timestamp": float(f_i) / float(fps),
                "frame_index": f_i,
                "episode_index": i,
                "index": global_index,
                "task_index": 0,
            }
            lines.append(json.dumps(row))
            global_index += 1
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_frames += len(frames)
        ep_rows.append(
            {
                "episode_index": i,
                "length": len(frames),
                "tasks": [task],
                "success": bool(episode.get("success", True)),
            }
        )

    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ep_rows),
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n",
        encoding="utf-8",
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.jsonl",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [obs_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return inspect_lerobot_dataset(str(dest))


def load_lerobot_arrays(
    path: str | Path,
    keep_episodes: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load observation.state / action arrays, honoring keep_episodes."""
    dest = Path(path)
    info = inspect_lerobot_dataset(str(dest))
    if not info.get("ok"):
        raise FileNotFoundError(info.get("error") or f"Not a LeRobot dataset: {dest}")
    keep: set[int] | None = None
    if keep_episodes is not None:
        keep = {int(i) for i in keep_episodes}
    obs_rows: list[list[float]] = []
    act_rows: list[list[float]] = []
    for episode in info["episodes"]:
        idx = int(episode["episode_index"])
        if keep is not None and idx not in keep:
            continue
        frame_path = dest / "data" / "chunk-000" / f"episode_{idx:06d}.jsonl"
        if not frame_path.is_file():
            continue
        for line in frame_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            obs_rows.append(_as_list(row.get("observation.state"), OBS_DIM))
            act_rows.append(_as_list(row.get("action"), ACT_DIM))
    if not obs_rows:
        raise ValueError("Dataset has no frames after keep/drop filtering.")
    return (
        np.asarray(obs_rows, dtype=np.float64),
        np.asarray(act_rows, dtype=np.float64),
    )


def _as_path(raw: str) -> Path:
    path = Path(raw.removeprefix("file:")).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _as_list(value: Any, dim: int) -> list[float]:
    if value is None:
        return [0.0] * dim
    if isinstance(value, (int, float)):
        out = [float(value)]
    else:
        out = [float(v) for v in list(value)]
    if len(out) < dim:
        out = out + [0.0] * (dim - len(out))
    return out[:dim]


def _load_episodes(path: Path, info: dict[str, Any]) -> list[dict[str, Any]]:
    if path.is_file():
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                item = {
                    "episode_index": row.get("episode_index", len(rows)),
                    "length": row.get("length"),
                    "tasks": row.get("tasks") or [],
                }
                if "success" in row:
                    item["success"] = row["success"]
                rows.append(item)
        return rows
    total = int(info.get("total_episodes") or 0)
    return [{"episode_index": i, "length": None, "tasks": []} for i in range(total)]
