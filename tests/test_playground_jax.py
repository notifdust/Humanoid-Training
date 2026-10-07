from __future__ import annotations

from humanoid_training.hardware import playground_jax_stack, playground_ready


def test_playground_jax_stack_skips_under_cli_override(monkeypatch) -> None:
    monkeypatch.setenv("HT_PLAYGROUND_CLI", "0")
    monkeypatch.setenv("HT_GPU", "1")
    stack = playground_jax_stack()
    assert stack["skipped"] is True
    assert stack["ok"] is True
    # Override CLI to missing → not ready even with GPU.
    assert playground_ready() is False


def test_playground_ready_with_fake_cli(tmp_path, monkeypatch) -> None:
    cli = tmp_path / "train-jax-ppo"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    assert playground_ready() is True
    assert playground_jax_stack()["skipped"] is True
