from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_training.adapters import select_adapter
from humanoid_training.recipes import expand_spec
from humanoid_training.runner import run_job
from humanoid_training.spec import load_spec


FAKE_LEROBOT = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

joined = " ".join(sys.argv)
print(f"lerobot-train {joined}", flush=True)
code = int(os.environ.get("HT_FAKE_LEROBOT_EXIT", "0"))
if code != 0:
    print("fake lerobot failing", flush=True)
    raise SystemExit(code)

out = None
for arg in sys.argv[1:]:
    if arg.startswith("--output_dir="):
        out = Path(arg.split("=", 1)[1])
        break
if out is None:
    out = Path("lerobot_logs")
if os.environ.get("HT_FAKE_LEROBOT_NO_VIDEO") == "1":
    raise SystemExit(0)
dest = out / "videos" / "eval.mp4"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(b"fake-lerobot-act-mp4")
raise SystemExit(0)
"""


def _mustard_spec(**overlay):
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-mustard-in-bowl.json"
    )
    for key, value in overlay.items():
        spec[key] = value
    return expand_spec(spec)


def _write_cli(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_pick_and_place_selects_lerobot_when_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "lerobot-train", FAKE_LEROBOT)
    monkeypatch.setenv("HT_LEROBOT_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    spec = _mustard_spec()
    assert select_adapter(spec).name == "lerobot"


def test_pick_and_place_act_harvests_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "lerobot-train", FAKE_LEROBOT)
    monkeypatch.setenv("HT_LEROBOT_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    # Prefer only lerobot so we do not fall through to mujoco.
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-mustard-in-bowl.json"
    )
    spec["backend"] = {"prefer": ["lerobot"], "compute": "local"}
    spec["train"] = {**(spec.get("train") or {}), "steps": 8, "seed": 1}
    spec["data"] = {"min_episodes": 2}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "eval.mp4").is_file()
    assert (run_dir / "eval.mp4").read_bytes() == b"fake-lerobot-act-mp4"
    facts = manifest.get("facts") or {}
    assert facts.get("kind") == "imitation"
    assert facts.get("policy") == "act"
    assert facts.get("engine") == "lerobot"
    assert facts.get("device") == "gpu"
    notes = " ".join(manifest.get("notes") or [])
    assert "ACT" in notes
    assert "finger grasping" in notes.lower() or "Not finger grasping" in notes


def test_pick_and_place_act_exit0_without_video_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "lerobot-train", FAKE_LEROBOT)
    monkeypatch.setenv("HT_LEROBOT_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_FAKE_LEROBOT_NO_VIDEO", "1")
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-mustard-in-bowl.json"
    )
    spec["backend"] = {"prefer": ["lerobot"], "compute": "local"}
    spec["train"] = {**(spec.get("train") or {}), "steps": 4}
    spec["data"] = {"min_episodes": 2}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] != "passed"
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()
    err = (manifest.get("error") or "") + " ".join(manifest.get("notes") or [])
    assert "linear-BC" in err or "mp4" in err.lower() or "substitut" in err.lower()


def test_pick_and_place_act_nonzero_exit_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "lerobot-train", FAKE_LEROBOT)
    monkeypatch.setenv("HT_LEROBOT_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_FAKE_LEROBOT_EXIT", "1")
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-mustard-in-bowl.json"
    )
    spec["backend"] = {"prefer": ["lerobot"], "compute": "local"}
    spec["train"] = {**(spec.get("train") or {}), "steps": 4}
    spec["data"] = {"min_episodes": 2}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["status"] != "passed"
    facts = manifest.get("facts") or {}
    assert facts.get("policy") == "act"
    assert facts.get("engine") == "lerobot"
    assert facts.get("returncode") == 1
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()
