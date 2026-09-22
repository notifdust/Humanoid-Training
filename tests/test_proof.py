from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_training.cli import main
from humanoid_training.errors import AdapterUnavailable
from humanoid_training.proof import (
    first_ready_engine,
    proof_walk_spec,
    run_walk_proof,
    summarize_walk_proof,
    walk_engines_ready,
)


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
print(f"fake proof env={args.env_name} steps={args.num_timesteps}", flush=True)
if args.num_videos > 0:
    dest = Path(args.logdir) / args.env_name / "rollout0.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"fake-walk-proof-mp4")
raise SystemExit(0)
"""


def _fake_cli(tmp_path: Path) -> Path:
    path = tmp_path / "train-jax-ppo"
    path.write_text(FAKE_TRAINER, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_walk_engines_ready_default_cpu() -> None:
    ready = walk_engines_ready()
    assert ready["playground"] is False
    assert ready["mjlab"] is False
    assert ready["isaaclab"] is False
    assert first_ready_engine() is None


def test_proof_walk_blocked_without_gpu(tmp_path: Path) -> None:
    with pytest.raises(AdapterUnavailable, match="Phase 3c walk proof is blocked"):
        run_walk_proof(runs_dir=tmp_path)


def test_cli_proof_walk_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["proof", "walk", "--out", str(tmp_path)]) == 12
    err = capsys.readouterr().err
    assert "Phase 3c" in err
    assert "pip install playground" in err


def test_cli_help_lists_proof(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as err:
        main(["-h"])
    assert err.value.code == 0
    assert "proof" in capsys.readouterr().out


def test_proof_walk_passes_with_fake_playground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _fake_cli(tmp_path)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    report = run_walk_proof(runs_dir=tmp_path / "runs", steps=16)
    assert report["ok"] is True
    assert report["engine"] == "playground"
    assert report["video"]
    assert Path(report["video"]).read_bytes() == b"fake-walk-proof-mp4"
    assert report["facts"].get("kind") == "rl"


def test_proof_walk_prefer_mjlab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "mjlab-train"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "log='logs'\n"
        "for i,a in enumerate(args):\n"
        "  if a=='--log-root' and i+1 < len(args): log=args[i+1]\n"
        "dest=Path(log)/'videos'/'train'/'x.mp4'\n"
        "dest.parent.mkdir(parents=True, exist_ok=True)\n"
        "dest.write_bytes(b'fake-mjlab-proof')\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("HT_MJLAB_CLI", str(fake))
    monkeypatch.setenv("HT_GPU", "1")
    # Playground also "ready" would win without --prefer.
    monkeypatch.setenv("HT_PLAYGROUND_CLI", "0")
    report = run_walk_proof(runs_dir=tmp_path / "runs", prefer=["mjlab"], steps=8)
    assert report["ok"] is True
    assert report["engine"] == "mjlab"


def test_summarize_rejects_missing_video(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = summarize_walk_proof(
        {
            "status": "passed",
            "run_dir": str(run_dir),
            "facts": {"kind": "rl", "engine": "playground"},
        }
    )
    assert report["ok"] is False
    assert "eval.mp4" in (report["error"] or "")


def test_proof_walk_spec_targets_ready_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "humanoid_training.proof.walk_engines_ready",
        lambda: {"playground": False, "mjlab": True, "isaaclab": False},
    )
    spec = proof_walk_spec(steps=32)
    assert spec["backend"]["prefer"][0] == "mjlab"
    assert spec["train"]["steps"] == 32
    assert spec["eval"]["record_video"] is True
