from __future__ import annotations

from humanoid_training.artifacts import RUN_ARTIFACTS, SERVED_ARTIFACTS
from humanoid_training.runner import collect_artifacts


def test_served_artifacts_include_run_files_and_metadata() -> None:
    assert "eval.mp4" in SERVED_ARTIFACTS
    assert "train_returns.json" in SERVED_ARTIFACTS
    assert "manifest.json" in SERVED_ARTIFACTS
    assert "spec.json" in SERVED_ARTIFACTS
    assert set(RUN_ARTIFACTS) <= set(SERVED_ARTIFACTS)


def test_collect_artifacts_names_existing_files(tmp_path) -> None:
    video = tmp_path / "eval.mp4"
    video.write_bytes(b"fake")
    (tmp_path / "checkpoint.npz").write_bytes(b"ckpt")
    (tmp_path / "train_returns.json").write_text("[]", encoding="utf-8")
    arts = collect_artifacts(tmp_path, video)
    assert arts["eval.mp4"] == str(video)
    assert "checkpoint.npz" in arts
    assert "train_returns.json" in arts
    assert "composed_scene.xml" not in arts
