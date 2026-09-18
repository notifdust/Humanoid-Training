from __future__ import annotations

from pathlib import Path

from humanoid_training.recipes import expand_spec
from humanoid_training.runner import run_job
from humanoid_training.spec import load_spec


def test_cartpole_train_writes_video(tmp_path: Path) -> None:
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    )
    spec["train"] = {"method": "rl", "steps": 40, "seed": 1}
    spec["eval"] = {"episodes": 2, "record_video": True}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] in {"completed", "passed"}
    assert manifest["adapter"]["adapter"] == "gymnasium"
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "eval.mp4").is_file()
    assert (run_dir / "checkpoint.npz").is_file()
    assert manifest["metrics"]["eval_episodes"] == 2


def test_g1_walk_is_blocked_without_playground(tmp_path: Path) -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] == "blocked"
    assert "Playground" in (manifest["error"] or "")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "train.sh").is_file()
    assert (run_dir / "engine_payload.json").is_file()
    assert (run_dir / "engines" / "isaaclab" / "osmo_workflow.yaml").is_file()


def test_g1_walk_compile_only(tmp_path: Path) -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    manifest = run_job(spec, runs_dir=tmp_path, compile_only=True)
    assert manifest["status"] == "compiled"
    assert expand_spec(spec)["task"]["recipe"] == "g1-walk"
