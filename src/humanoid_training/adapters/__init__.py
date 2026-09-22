from __future__ import annotations

from humanoid_training.adapters.base import Adapter
from humanoid_training.adapters.common import (
    ignored_scene_fields,
    recipe_adapter_config,
    write_payload_files,
)
from humanoid_training.adapters.gym_adapter import GymnasiumAdapter
from humanoid_training.adapters.isaaclab import IsaacLabAdapter
from humanoid_training.adapters.lerobot import LeRobotAdapter
from humanoid_training.adapters.mjlab import MJLabAdapter
from humanoid_training.adapters.mujoco_adapter import MujocoAdapter
from humanoid_training.adapters.playground import PlaygroundAdapter
from humanoid_training.errors import NoAdapter


def registry() -> dict[str, Adapter]:
    adapters: list[Adapter] = [
        GymnasiumAdapter(),
        MujocoAdapter(),
        PlaygroundAdapter(),
        MJLabAdapter(),
        IsaacLabAdapter(),
        LeRobotAdapter(),
    ]
    return {a.name: a for a in adapters}


def select_adapter(spec: dict) -> Adapter:
    """First launch-ready preferred engine, else first compile-ok (block later)."""
    from humanoid_training.hardware import adapter_launch_ready

    preferred = list((spec.get("backend") or {}).get("prefer") or [])
    adapters = registry()
    if not preferred:
        preferred = list(adapters.keys())

    failures: list[str] = []
    compile_ok: list[Adapter] = []
    for name in preferred:
        adapter = adapters.get(name)
        if adapter is None:
            failures.append(f"{name}: unknown adapter")
            continue
        cfg = recipe_adapter_config(spec, name)
        if cfg.get("unsupported"):
            failures.append(f"{name}: {cfg['unsupported']}")
            continue
        support = adapter.support(spec)
        if not support.ok:
            failures.append(f"{name}: {support.reason}")
            continue
        if adapter_launch_ready(name):
            return adapter
        compile_ok.append(adapter)
        failures.append(f"{name}: compiles here but is not launch-ready")
    if compile_ok:
        return compile_ok[0]
    raise NoAdapter(
        "No adapter could run this spec. Tried:\n"
        + "\n".join(f"  - {line}" for line in failures)
        + "\nSee docs/ROADMAP.md for which engines are live."
    )


__all__ = [
    "ignored_scene_fields",
    "recipe_adapter_config",
    "registry",
    "select_adapter",
    "write_payload_files",
]
