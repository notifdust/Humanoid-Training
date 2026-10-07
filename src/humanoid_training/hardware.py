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


def playground_jax_stack() -> dict[str, Any]:
    """Detect jax/brax pins that make train-jax-ppo exit 1.

    Playground (brax 0.14) still calls ``jax.device_put_replicated``, removed
    in jax 0.10+. A bare ``pip install playground`` can pull jax 0.11 and
    a CPU-only jaxlib even when nvidia-smi sees a GPU.
    """
    if os.environ.get("HT_PLAYGROUND_CLI") is not None:
        # Test / operator override — do not require a real jax stack.
        return {"ok": True, "skipped": True, "reasons": [], "backend": None, "jax_version": None}
    try:
        import jax
    except ImportError:
        return {
            "ok": False,
            "skipped": False,
            "reasons": [
                "jax is not installed. "
                "pip install -e '.[playground]' && pip install 'jax[cuda12]==0.9.2'"
            ],
            "backend": None,
            "jax_version": None,
        }
    reasons: list[str] = []
    version = getattr(jax, "__version__", "unknown")
    if not hasattr(jax, "device_put_replicated"):
        reasons.append(
            f"jax {version} is too new for Playground/brax "
            "(missing jax.device_put_replicated). "
            "Pin: pip install 'jax[cuda12]==0.9.2' 'jaxlib==0.9.2'"
        )
    backend = None
    try:
        backend = str(jax.default_backend())
    except Exception:  # noqa: BLE001 — surface as unknown
        backend = "unknown"
    if gpu_available() and backend == "cpu":
        reasons.append(
            "nvidia-smi sees a GPU but jax default_backend() is cpu "
            "(CUDA jaxlib missing). "
            "Install: pip install 'jax[cuda12]==0.9.2'"
        )
    return {
        "ok": not reasons,
        "skipped": False,
        "reasons": reasons,
        "backend": backend,
        "jax_version": version,
    }


def playground_ready() -> bool:
    """Can Phase 3a launch G1 walk here? Needs CLI, GPU, and a compatible jax."""
    if playground_cli() is None or not gpu_available():
        return False
    return bool(playground_jax_stack().get("ok"))


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


def hf_cli() -> str | None:
    """Hugging Face CLI (`hf` or `huggingface-cli`)."""
    override = _cli_from_env("HT_HF_CLI")
    if os.environ.get("HT_HF_CLI") is not None:
        return override
    return shutil.which("hf") or shutil.which("huggingface-cli")


def hf_token_present() -> bool:
    """True when HF_TOKEN / HUGGING_FACE_HUB_TOKEN is set (non-empty)."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip() and raw.strip().lower() not in _FALSE:
            return True
    return False


def hf_jobs_ready() -> bool:
    """Host can attempt HF Jobs submit (CLI + token). Live harvest still separate."""
    if _env_bool("HT_HF_JOBS") is False:
        return False
    return hf_cli() is not None and hf_token_present()


def mjlab_motion_ready() -> bool:
    """Motion-imitation mjlab tasks need a WandB registry / motion pin."""
    for name in ("HT_MJLAB_MOTION", "WANDB_MOTION_REGISTRY"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip() and raw.strip().lower() not in _FALSE:
            return True
    return False


def groot_stack_hint() -> dict[str, Any]:
    """Cheap probes for NVIDIA GR00T / Arena path — not a live train."""
    return {
        "gpu": gpu_available(),
        "isaac_local_ready": isaac_local_ready(),
        "isaac_cli": isaac_cli(),
        "lerobot_ready": lerobot_ready(),
        "hf_token": hf_token_present(),
    }


def docker_bin() -> str | None:
    return shutil.which("docker")


def docker_gpu_requested() -> bool:
    return _env_bool("HT_DOCKER_GPU") is True


def isaac_local_ready() -> bool:
    """Local Isaac path only (CLI or GPU Docker) — not OSMO remote harvest."""
    if isaac_cli() and gpu_available():
        return True
    if docker_bin() is not None and docker_gpu_requested() and gpu_available():
        return True
    return False


def isaac_launch_ready() -> bool:
    """Isaac can launch here: local GPU CLI/Docker, or OSMO remote harvest."""
    if isaac_local_ready():
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
    hf = hf_cli()
    jax_stack = playground_jax_stack()
    return {
        "gpu": gpu_available(),
        "playground": playground_installed(),
        "playground_cli": pg,
        "playground_ready": playground_ready(),
        "playground_jax": jax_stack,
        "mjlab_cli": mj,
        "mjlab_ready": bool(mj) and gpu_available(),
        "mjlab_motion_ready": mjlab_motion_ready(),
        "isaac_cli": isa,
        "osmo_cli": osmo,
        "osmo_ready": osmo_ready(),
        "isaac_local_ready": isaac_local_ready(),
        "isaac_launch_ready": isaac_launch_ready(),
        "lerobot_cli": lr,
        "lerobot_ready": bool(lr) and gpu_available(),
        "hf_cli": hf,
        "hf_token": hf_token_present(),
        "hf_jobs_ready": hf_jobs_ready(),
    }
