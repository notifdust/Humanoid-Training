from __future__ import annotations

from pathlib import Path

from humanoid_training.cli import main
from humanoid_training.english import english_from_facts
from humanoid_training.export_job import export_spec_path


def test_english_from_facts_hold_and_imitation() -> None:
    assert "not walking" in english_from_facts({"kind": "hold", "engine": "mujoco", "sim_only": True})
    assert "ACT" in english_from_facts({"kind": "imitation", "policy": "act", "engine": "lerobot"})
    assert "linear-BC" in english_from_facts({"kind": "imitation", "policy": "linear-bc", "engine": "mujoco"})
    assert "RL" in english_from_facts({"kind": "rl", "engine": "playground", "policy": "ppo", "device": "gpu"})


def test_english_blocked_uses_error_line() -> None:
    text = english_from_facts({}, status="blocked", error="Need a GPU\nmore detail")
    assert text.startswith("Need a GPU")


def test_export_spec_writes_readme(tmp_path: Path) -> None:
    spec = Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json"
    dest = tmp_path / "out"
    report = export_spec_path(spec, dest)
    assert report["ok"] is True
    assert (dest / "README.md").is_file()
    assert (dest / "engine_payload.json").is_file() or (dest / "train_isaac.sh").is_file() or list(dest.iterdir())


def test_cli_export_spec(tmp_path: Path, capsys) -> None:
    spec = Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    dest = tmp_path / "export-cart"
    assert main(["export", str(spec), "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out.replace("True", "true")
    assert (dest / "README.md").is_file()
