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
