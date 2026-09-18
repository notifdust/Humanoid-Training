from __future__ import annotations

import os
import subprocess
from pathlib import Path

from humanoid_training.errors import AdapterUnavailable, repo_root

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"


def cache_dir() -> Path:
    env = os.environ.get("HT_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / ".cache"


def menagerie_robot_dir(name: str) -> Path:
    return cache_dir() / "mujoco_menagerie" / name


def ensure_menagerie_robot(name: str, log=None) -> Path:
    """Sparse-clone one robot from MuJoCo Menagerie into the local cache."""
    dest = menagerie_robot_dir(name)
    marker = dest / ".ht-ready"
    if marker.is_file() and any(dest.glob("*.xml")):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"fetching MuJoCo Menagerie robot '{name}' (one-time)")
    tmp = dest.parent / f".clone-{name}"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                MENAGERIE_REPO,
                str(tmp),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", name],
            check=True,
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        src = tmp / name
        if not src.is_dir():
            raise AdapterUnavailable(f"Menagerie has no robot folder '{name}'")
        import shutil

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        shutil.rmtree(tmp, ignore_errors=True)
        marker.write_text("ok\n", encoding="utf-8")
    except subprocess.CalledProcessError as err:
        raise AdapterUnavailable(
            "Could not download MuJoCo Menagerie assets. Need git and network.\n"
            f"{err.stderr or err.stdout}"
        ) from err
    return dest


def resolve_mjcf(spec_path: str, log=None) -> Path:
    """Resolve 'menagerie:unitree_g1/scene.xml' or a filesystem path."""
    if spec_path.startswith("menagerie:"):
        rest = spec_path.removeprefix("menagerie:")
        robot, _, rel = rest.partition("/")
        root = ensure_menagerie_robot(robot, log=log)
        path = root / (rel or "scene.xml")
        if not path.is_file():
            raise AdapterUnavailable(f"MJCF not found: {path}")
        return path
    path = Path(spec_path)
    if not path.is_file():
        candidate = repo_root() / spec_path
        if candidate.is_file():
            return candidate
        raise AdapterUnavailable(f"MJCF not found: {spec_path}")
    return path
