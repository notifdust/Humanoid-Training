from __future__ import annotations

from pathlib import Path

from humanoid_training.datasets import inspect_lerobot_dataset


def test_inspect_local_lerobot_fixture() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "lerobot_tiny"
    result = inspect_lerobot_dataset(str(path))
    assert result["ok"] is True
    assert result["format"] == "lerobot"
    assert result["total_episodes"] == 2
    assert result["episodes"][0]["tasks"] == ["pick mustard"]


def test_inspect_hf_is_closed() -> None:
    result = inspect_lerobot_dataset("hf:example/g1-mustard-demos")
    assert result["ok"] is False
    assert "Hugging Face" in result["error"]


def test_inspect_missing_info() -> None:
    result = inspect_lerobot_dataset("/tmp/not-a-dataset")
    assert result["ok"] is False
    assert "meta/info.json" in result["error"]
