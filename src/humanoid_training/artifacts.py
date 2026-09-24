from __future__ import annotations

from pathlib import Path

from humanoid_training.errors import repo_root

# Files a launch may write into the run directory. The studio only plays
# what collect_artifacts lists; keep this list as the single inventory.
RUN_ARTIFACTS = (
    "eval.mp4",
    "checkpoint.npz",
    "composed_scene.xml",
    "train_returns.json",
)

# Job metadata the HTTP API may stream. Not produced by every adapter.
META_ARTIFACTS = (
    "manifest.json",
    "spec.json",
    "run.log",
)

SERVED_ARTIFACTS = RUN_ARTIFACTS + META_ARTIFACTS

_STAND_GOLD_BYTES: bytes | None = None


def g1_stand_gold_path() -> Path:
    return repo_root() / "recipes" / "g1-stand" / "gold" / "eval.mp4"


def looks_like_g1_stand_gold(path: Path) -> bool:
    """True when path is byte-identical to the checked-in G1 stand gold clip.

    Used by walk proof + deploy gates so a stand video cannot pass as walk.
    """
    global _STAND_GOLD_BYTES
    candidate = Path(path)
    if not candidate.is_file():
        return False
    gold = g1_stand_gold_path()
    if not gold.is_file():
        return False
    try:
        size = candidate.stat().st_size
        if size != gold.stat().st_size or size < 8:
            return False
        if _STAND_GOLD_BYTES is None:
            _STAND_GOLD_BYTES = gold.read_bytes()
        return candidate.read_bytes() == _STAND_GOLD_BYTES
    except OSError:
        return False


def resolve_eval_mp4(manifest: dict) -> Path | None:
    """Prefer run_dir/eval.mp4, else an artifacts path that exists on disk."""
    run_dir = Path(manifest.get("run_dir") or "")
    if run_dir.is_dir():
        local = run_dir / "eval.mp4"
        if local.is_file():
            return local
    art = (manifest.get("artifacts") or {}).get("eval.mp4")
    if art:
        path = Path(str(art))
        if path.is_file():
            return path
    return None
