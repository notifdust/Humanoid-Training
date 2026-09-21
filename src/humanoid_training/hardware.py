"""Host probes the studio catalog and GPU adapters may call.

Keep this module free of recipe/adapter imports so `recipes.public_catalog`
can project `launch_here` without a cycle.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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


def gpu_available() -> bool:
    """NVIDIA GPU present, or `HT_PLAYGROUND_GPU` override.

    `HT_PLAYGROUND_GPU=0` forces no. Any other non-empty value forces yes
    (tests, or a box where nvidia-smi is hidden). Unset → probe nvidia-smi.
    """
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
    raw = os.environ.get("HT_PLAYGROUND_CLI")
    if raw is not None:
        stripped = raw.strip()
        if stripped.lower() in _FALSE or stripped == "":
            return None
        path = Path(stripped).expanduser()
        if path.is_file():
            return str(path.resolve())
        return shutil.which(stripped)
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


def engine_status() -> dict[str, Any]:
    cli = playground_cli()
    return {
        "gpu": gpu_available(),
        "playground": playground_installed(),
        "playground_cli": cli,
        "playground_ready": bool(cli) and gpu_available(),
    }
