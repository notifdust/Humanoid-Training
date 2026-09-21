from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_training.cli import main


def test_cli_help_lists_fetch_assets_and_record(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as err:
        main(["-h"])
    assert err.value.code == 0
    out = capsys.readouterr().out
    assert "fetch-assets" in out
    assert "record" in out
    assert "serve" in out


def test_cli_train_help_lists_docker(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as err:
        main(["train", "-h"])
    assert err.value.code == 0
    out = capsys.readouterr().out
    assert "--docker" in out


def test_cli_docker_without_daemon_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("humanoid_training.docker_runner.docker_bin", lambda: None)
    spec = Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    assert main(["train", str(spec), "--docker", "--out", str(tmp_path)]) == 12
    err = capsys.readouterr().err
    assert "Docker is not on PATH" in err


def test_cli_fetch_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "unitree_g1"

    def fake_ensure(name: str, log=None) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        if log:
            log(f"fake fetch {name}")
        return dest

    monkeypatch.setattr("humanoid_training.assets.ensure_menagerie_robot", fake_ensure)
    assert main(["fetch-assets", "unitree_g1"]) == 0
    printed = capsys.readouterr().out
    assert str(dest) in printed


def test_cli_serve_prints_open_url(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    assert main(["serve", "--host", "0.0.0.0", "--port", "8000"]) == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000" in out
    assert "Cartpole" in out
