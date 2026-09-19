from __future__ import annotations

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
