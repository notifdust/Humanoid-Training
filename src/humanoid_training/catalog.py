from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from humanoid_training.errors import repo_root


def load_robot_catalog() -> list[dict[str, Any]]:
    path = repo_root() / "robots" / "catalog.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    robots = data.get("robots", [])
    if not isinstance(robots, list):
        raise ValueError("robots/catalog.yaml must have a 'robots' list")
    return robots


def get_robot(robot_id: str) -> dict[str, Any] | None:
    for robot in load_robot_catalog():
        if robot.get("id") == robot_id:
            return robot
    return None
