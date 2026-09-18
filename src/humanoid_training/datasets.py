from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    path = Path(raw.removeprefix("file:")).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
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
                rows.append(
                    {
                        "episode_index": row.get("episode_index", len(rows)),
                        "length": row.get("length"),
                        "tasks": row.get("tasks") or [],
                    }
                )
        return rows
    total = int(info.get("total_episodes") or 0)
    return [{"episode_index": i, "length": None, "tasks": []} for i in range(total)]
