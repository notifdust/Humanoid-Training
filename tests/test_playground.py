from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_training.adapters.playground import (
    apply_flag,
    harvest_rollout_mp4,
)
from humanoid_training.errors import RecipeError
from humanoid_training.runner import run_job
from humanoid_training.spec import load_spec


FAKE_TRAINER = """#!/usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--env_name", default="env")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_timesteps", type=int, default=1)
parser.add_argument("--num_videos", type=int, default=1)
parser.add_argument("--logdir", default="logs")
args = parser.parse_args()
print(
    f"train-jax-ppo env={args.env_name} seed={args.seed} "
    f"steps={args.num_timesteps} videos={args.num_videos} logdir={args.logdir}",
    flush=True,
)
code = int(os.environ.get("HT_FAKE_PPO_EXIT", "0"))
if code != 0:
    print("fake trainer failing", flush=True)
    raise SystemExit(code)
if os.environ.get("HT_FAKE_PPO_NO_VIDEO") == "1" or args.num_videos <= 0:
    raise SystemExit(0)
dest = Path(args.logdir) / args.env_name / "rollout0.mp4"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(b"fake-mp4")
raise SystemExit(0)
"""


def _walk_spec(**overlay):
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    spec["train"] = {"method": "rl", "steps": 8, "seed": 1, **(overlay.get("train") or {})}
    if "eval" in overlay:
        spec["eval"] = overlay["eval"]
    return spec


def _fake_cli(tmp_path: Path) -> Path:
    path = tmp_path / "train-jax-ppo"
    path.write_text(FAKE_TRAINER, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_apply_flag_replaces_existing() -> None:
    argv = apply_flag(["train-jax-ppo", "--env_name", "old"], "--env_name", "new")
    assert argv.count("--env_name") == 1
    assert argv[argv.index("--env_name") + 1] == "new"


def test_harvest_prefers_rollout0(tmp_path: Path) -> None:
    logdir = tmp_path / "logs" / "exp"
    logdir.mkdir(parents=True)
    (logdir / "rollout1.mp4").write_bytes(b"one")
    (logdir / "rollout0.mp4").write_bytes(b"zero")
    dest = tmp_path / "eval.mp4"
    got = harvest_rollout_mp4(tmp_path / "logs", dest)
    assert got == dest
    assert dest.read_bytes() == b"zero"


def test_g1_walk_blocked_when_cli_present_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "0")
    manifest = run_job(_walk_spec(), runs_dir=tmp_path / "runs")
    assert manifest["status"] == "blocked"
    err = manifest["error"] or ""
    assert "NVIDIA GPU" in err or "no NVIDIA GPU" in err
    assert "g1-stand" in err
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()


def test_g1_walk_launch_harvests_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "1")
    monkeypatch.setenv("HT_GPU", "1")
    manifest = run_job(_walk_spec(), runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "eval.mp4").is_file()
    assert (run_dir / "eval.mp4").read_bytes() == b"fake-mp4"
    facts = manifest.get("facts") or {}
    assert facts.get("kind") == "rl"
    assert facts.get("engine") == "playground"
    assert facts.get("device") == "gpu"
    assert facts.get("env") == "G1JoystickFlatTerrain"
    assert facts.get("runner") == "inprocess"
    job = (run_dir / "job.json").read_text(encoding="utf-8")
    assert "train-jax-ppo" in job or str(cli) in job
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "G1JoystickFlatTerrain" in log
    notes = " ".join(manifest.get("notes") or [])
    assert "stand clip" not in notes.lower() or "not" in notes.lower()


def test_g1_walk_exit0_without_video_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "1")
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_FAKE_PPO_NO_VIDEO", "1")
    manifest = run_job(_walk_spec(), runs_dir=tmp_path / "runs")
    assert manifest["status"] == "blocked"
    err = manifest["error"] or ""
    assert "rollout" in err.lower()
    assert "stand" in err.lower()
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()
    stand_gold = (
        Path(__file__).resolve().parents[1] / "recipes" / "g1-stand" / "gold" / "eval.mp4"
    )
    if stand_gold.is_file():
        # Must not copy the CPU stand clip in as a walk eval.
        assert (Path(manifest["run_dir"]) / "eval.mp4").is_file() is False


def test_g1_walk_nonzero_exit_is_not_a_walk_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "1")
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_FAKE_PPO_EXIT", "1")
    manifest = run_job(_walk_spec(), runs_dir=tmp_path / "runs")
    assert manifest["status"] == "completed"
    assert manifest.get("metrics", {}).get("passed") is False
    facts = manifest.get("facts") or {}
    assert facts.get("engine") == "playground"
    assert facts.get("kind") == "rl"
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()
    notes = " ".join(manifest.get("notes") or [])
    assert "exited 1" in notes


def test_g1_walk_record_video_false_skips_harvest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "1")
    monkeypatch.setenv("HT_GPU", "1")
    spec = _walk_spec()
    spec["eval"] = {"episodes": 1, "record_video": False}
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error")
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()
    assert (manifest.get("facts") or {}).get("engine") == "playground"


def test_unknown_reach_recipe_does_not_launch_playground_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with Playground ready, a deleted g1-reach id must not train walk."""
    from humanoid_training.errors import RecipeError

    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "1")
    monkeypatch.setenv("HT_GPU", "1")
    with pytest.raises(RecipeError, match="Unknown recipe 'g1-reach'"):
        run_job(
            {
                "spec_version": "0.1.0",
                "name": "missing-reach",
                "robot": {"id": "unitree-g1-29dof", "source": "catalog"},
                "task": {"recipe": "g1-reach"},
                "train": {"method": "rl", "steps": 8},
            },
            runs_dir=tmp_path / "runs",
        )


def test_hardware_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from humanoid_training.hardware import gpu_available, playground_cli, playground_ready

    monkeypatch.delenv("HT_GPU", raising=False)
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "1")
    monkeypatch.setenv("HT_PLAYGROUND_CLI", "0")
    assert gpu_available() is True
    assert playground_cli() is None
    assert playground_ready() is False

    monkeypatch.setenv("HT_GPU", "0")
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "1")
    assert gpu_available() is False

    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "0")
    monkeypatch.setenv("HT_GPU", "0")
    assert gpu_available() is False
    assert playground_cli() == str(cli.resolve())
    assert playground_ready() is False
