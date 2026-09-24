from __future__ import annotations

import os
from pathlib import Path

import pytest

from humanoid_training.gold import (
    GOLD_NOTES,
    compare_eval_videos,
    cpu_recipes,
    gold_clip_path,
    gold_dir,
    gpu_recipes,
    probe_video,
)
from humanoid_training.recipes import list_recipes


def test_cpu_recipes_ship_gold_eval_and_notes() -> None:
    recipes = cpu_recipes()
    assert recipes, "catalog has no availability:cpu recipes"
    ids = {r.id for r in recipes}
    assert ids == {r.id for r in list_recipes() if r.as_public_dict()["availability"] == "cpu"}
    for recipe in recipes:
        public = recipe.as_public_dict()
        clip = gold_clip_path(recipe)
        notes = gold_dir(recipe) / GOLD_NOTES
        assert clip.is_file(), f"{recipe.id} is missing gold/eval.mp4"
        assert notes.is_file(), f"{recipe.id} is missing {GOLD_NOTES}"
        assert notes.read_text(encoding="utf-8").strip()
        assert public["has_gold"] is True
        probe = probe_video(clip)
        assert probe["width"] >= 32 and probe["height"] >= 32
        assert probe["duration"] > 0.2
        assert probe["size"] > 1000


def test_gpu_recipes_do_not_ship_a_fake_success_clip() -> None:
    for recipe in gpu_recipes():
        public = recipe.as_public_dict()
        assert public["has_gold"] is False
        assert not gold_clip_path(recipe).is_file(), (
            f"{recipe.id} has gold/eval.mp4 — GPU recipes must not check in a success clip"
        )


def test_gold_clip_resembles_itself() -> None:
    for recipe in cpu_recipes():
        clip = gold_clip_path(recipe)
        result = compare_eval_videos(clip, clip)
        assert result.ok, f"{recipe.id}: {result.reason}"
        assert result.mean_abs is not None
        assert result.mean_abs < 1.0


def test_gold_clips_from_different_recipes_do_not_match() -> None:
    recipes = cpu_recipes()
    assert len(recipes) >= 2
    a = gold_clip_path(recipes[0])
    b = gold_clip_path(recipes[1])
    result = compare_eval_videos(a, b)
    assert result.ok is False


def test_gold_module_does_not_hardcode_recipe_ids() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "humanoid_training" / "gold.py"
    text = src.read_text(encoding="utf-8")
    for banned in ("cartpole-balance", "g1-stand", "pick-and-place", "g1-walk"):
        assert banned not in text


@pytest.mark.gold
def test_cpu_train_eval_resembles_gold(tmp_path: Path) -> None:
    """Retrain each CPU recipe and compare eval.mp4 to the checked-in gold clip."""
    if os.environ.get("HT_GOLD") != "1":
        pytest.skip("set HT_GOLD=1 (CI xvfb job) to retrain against gold clips")
    if os.environ.get("HT_NO_RENDER") == "1":
        pytest.skip("HT_NO_RENDER=1 skips gold retrain")

    from humanoid_training.gold import record_gold

    for recipe in cpu_recipes():
        gold = gold_clip_path(recipe)
        recorded = record_gold(recipe, dest_root=tmp_path, runs_dir=tmp_path / "runs")
        assert recorded["ok"] is True, recorded
        assert recorded["passed"] is True, recorded
        facts = recorded.get("facts") or {}
        method = recipe.as_public_dict()["method"]
        if method == "rl":
            assert facts.get("kind") == "rl"
            assert facts.get("engine") == "gymnasium"
        elif method == "hold":
            assert facts.get("kind") == "hold"
            assert facts.get("engine") == "mujoco"
        elif method == "imitation":
            assert facts.get("kind") == "imitation"
            assert facts.get("policy") == "linear-bc"
        assert facts.get("sim_only") is True
        produced = Path(recorded["path"])
        result = compare_eval_videos(produced, gold)
        assert result.ok, f"{recipe.id}: {result.reason} {result}"
