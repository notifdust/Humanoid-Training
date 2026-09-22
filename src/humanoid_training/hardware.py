"""Host probes the studio catalog and GPU adapters may call.

Keep this module free of recipe/adapter imports so `recipes.public_catalog`
can project `launch_here` without a cycle.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in _FALSE or key == "":
        return False
    return True


def _cli_from_env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped.lower() in _FALSE or stripped == "":
        return None
    path = Path(stripped).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(stripped)


def gpu_available() -> bool:
    """NVIDIA GPU present, or `HT_GPU` / `HT_PLAYGROUND_GPU` override.

    `=0` forces no. Any other non-empty value forces yes. Unset → nvidia-smi.
    `HT_GPU` wins over the Playground-era alias.
    """
    override = _env_bool("HT_GPU")
    if override is not None:
        return override
    override = _env_bool("HT_PLAYGROUND_GPU")
    if override is not None:
        return override
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        proc = subprocess.run(
            [smi, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def playground_cli() -> str | None:
    """Path to `train-jax-ppo`, or None.

    `HT_PLAYGROUND_CLI` overrides PATH: a filesystem path, a command name,
    or `0`/`false` to pretend the CLI is missing.
    """
    override = _cli_from_env("HT_PLAYGROUND_CLI")
    if os.environ.get("HT_PLAYGROUND_CLI") is not None:
        return override
    return shutil.which("train-jax-ppo")


def playground_installed() -> bool:
    if playground_cli():
        return True
    try:
        import mujoco_playground  # noqa: F401
    except ImportError:
        return False
    return True


def playground_ready() -> bool:
    """Can Phase 3a launch G1 walk here? Needs the CLI and a GPU."""
    return playground_cli() is not None and gpu_available()


def mjlab_cli() -> str | None:
    """Entry for `python -m mjlab.scripts.train`, or a `HT_MJLAB_CLI` override."""
    override = _cli_from_env("HT_MJLAB_CLI")
    if os.environ.get("HT_MJLAB_CLI") is not None:
        return override
    try:
        import mjlab  # noqa: F401
    except ImportError:
        return None
    return sys.executable


def mjlab_train_argv(task: str, extra: list[str] | None = None) -> list[str]:
    cli = mjlab_cli()
    if not cli:
        return ["python", "-m", "mjlab.scripts.train", task, *(extra or [])]
    path = Path(cli)
    if path.is_file() and path.name != Path(sys.executable).name:
        return [cli, task, *(extra or [])]
    return [cli, "-m", "mjlab.scripts.train", task, *(extra or [])]


def mjlab_ready() -> bool:
    return mjlab_cli() is not None and gpu_available()


def lerobot_cli() -> str | None:
    """`lerobot-train` / fake CLI via `HT_LEROBOT_CLI`, else import probe."""
    override = _cli_from_env("HT_LEROBOT_CLI")
    if os.environ.get("HT_LEROBOT_CLI") is not None:
        return override
    found = shutil.which("lerobot-train")
    if found:
        return found
    try:
        import lerobot  # noqa: F401
    except ImportError:
        return None
    # Prefer the console script when import works but PATH is thin.
    return shutil.which("lerobot-train") or sys.executable


def lerobot_train_argv(extra: list[str] | None = None) -> list[str]:
    cli = lerobot_cli()
    extra = list(extra or [])
    if not cli:
        return ["lerobot-train", *extra]
    path = Path(cli)
    if path.is_file() and path.name not in {Path(sys.executable).name, "python", "python3"}:
        return [cli, *extra]
    # Fall back to module form for older installs.
    return [cli, "-m", "lerobot.scripts.train", *extra]


def lerobot_ready() -> bool:
    return lerobot_cli() is not None and gpu_available()


def isaac_cli() -> str | None:
    """`isaaclab.sh` (or a fake) via `HT_ISAAC_CLI` or PATH."""
    override = _cli_from_env("HT_ISAAC_CLI")
    if os.environ.get("HT_ISAAC_CLI") is not None:
        return override
    found = shutil.which("isaaclab.sh")
    if found:
        return found
    home = Path.home() / "IsaacLab" / "isaaclab.sh"
    if home.is_file():
        return str(home)
    return None


def osmo_cli() -> str | None:
    override = _cli_from_env("HT_OSMO_CLI")
    if os.environ.get("HT_OSMO_CLI") is not None:
        return override
    return shutil.which("osmo")


def osmo_ready() -> bool:
    """Phase 3d: OSMO CLI present so we can submit + poll + harvest."""
    if _env_bool("HT_OSMO_HARVEST") is False:
        return False
    return osmo_cli() is not None


def docker_bin() -> str | None:
    return shutil.which("docker")


def docker_gpu_requested() -> bool:
    return _env_bool("HT_DOCKER_GPU") is True


def isaac_launch_ready() -> bool:
    """Isaac can launch here: local GPU CLI/Docker, or OSMO remote harvest."""
    if isaac_cli() and gpu_available():
        return True
    if docker_bin() is not None and docker_gpu_requested() and gpu_available():
        return True
    return osmo_ready()


def adapter_launch_ready(name: str) -> bool:
    """True when this adapter can launch here, not merely compile-and-block."""
    if name == "playground":
        return playground_ready()
    if name == "mjlab":
        return mjlab_ready()
    if name == "isaaclab":
        return isaac_launch_ready()
    if name == "lerobot":
        return lerobot_ready()
    return True


def engine_status() -> dict[str, Any]:
    pg = playground_cli()
    mj = mjlab_cli()
    isa = isaac_cli()
    osmo = osmo_cli()
    lr = lerobot_cli()
    return {
        "gpu": gpu_available(),
        "playground": playground_installed(),
        "playground_cli": pg,
        "playground_ready": bool(pg) and gpu_available(),
        "mjlab_cli": mj,
        "mjlab_ready": bool(mj) and gpu_available(),
        "isaac_cli": isa,
        "osmo_cli": osmo,
        "osmo_ready": osmo_ready(),
        "isaac_launch_ready": isaac_launch_ready(),
        "lerobot_cli": lr,
        "lerobot_ready": bool(lr) and gpu_available(),
    }
