from __future__ import annotations

import os

from humanoid_training.proof import assess_walk_proof_host
from humanoid_training.recipes import public_catalog


def test_osmo_only_does_not_ready_robomimic_pickplace(monkeypatch) -> None:
    """OSMO harvest is real for rsl_rl walk — not for robomimic PickPlace."""
    monkeypatch.setenv("HT_OSMO_CLI", "/bin/true")
    monkeypatch.setenv("HT_GPU", "0")
    monkeypatch.delenv("HT_ISAAC_CLI", raising=False)
    monkeypatch.delenv("HT_DOCKER_GPU", raising=False)
    catalog = public_catalog()
    by = {r["id"]: r for r in catalog["recipes"]}
    assert by["g1-pickplace"]["launch_here"] is False
    assert "g1-pickplace" in catalog["later"]
    # Flat / rough walk may still be launch_here via OSMO (Phase 3d path).
    assert by["g1-walk"]["launch_here"] is True
    assert "g1-walk" in catalog["ready"]


def test_phase_3c_assess_not_ok_on_osmo_only(monkeypatch) -> None:
    monkeypatch.setenv("HT_OSMO_CLI", "/bin/true")
    monkeypatch.setenv("HT_GPU", "0")
    monkeypatch.delenv("HT_ISAAC_CLI", raising=False)
    report = assess_walk_proof_host()
    assert report["ok"] is False
    assert report["phase"] == "3c"
    assert report["engine"] is None
    assert any("GPU" in r or "OSMO" in r for r in report["reasons"])
    assert "Phase 3d" in (report["note"] or "") or "OSMO" in (report["note"] or "")


def test_phase_3c_assess_ok_needs_local_gpu_engine(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "train-jax-ppo"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("HT_PLAYGROUND_CLI", str(fake))
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_OSMO_CLI", "0")
    report = assess_walk_proof_host()
    assert report["ok"] is True
    assert report["engine"] == "playground"
    assert report["reasons"] == []
