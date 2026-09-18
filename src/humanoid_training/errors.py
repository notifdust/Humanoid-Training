from __future__ import annotations

from pathlib import Path


class SpecError(ValueError):
    """Job spec failed schema or semantic checks."""


class RecipeError(ValueError):
    """Recipe package is missing or invalid."""


class AdapterError(RuntimeError):
    """Adapter could not compile or run a spec."""


class AdapterUnavailable(AdapterError):
    """The engine is not installed or not in this phase yet."""


class NoAdapter(AdapterError):
    """No adapter in backend.prefer can run this recipe."""


def repo_root() -> Path:
    """Directory that contains recipes/ and spec/."""
    import os

    env = os.environ.get("HT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "recipes").is_dir() and (
            candidate / "spec" / "job_spec.schema.json"
        ).is_file():
            return candidate
    cwd = Path.cwd()
    if (cwd / "recipes").is_dir():
        return cwd
    raise FileNotFoundError(
        "Cannot find the Humanoid Training repo root. Set HT_ROOT or run from the clone."
    )
