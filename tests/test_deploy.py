from __future__ import annotations

import json
from pathlib import Path

import pytest

from humanoid_training.deploy import assess_deploy, deploy_run, hardware_profile_present
from humanoid_training.errors import AdapterUnavailable
from humanoid_training.runner import run_job
from humanoid_training.spec import load_spec


def _walk_manifest(**overlay) -> dict:
    base = {
        "run_id": "walk-1",
        "recipe": "g1-walk",
        "status": "passed",
        "metrics": {"passed": True, "success_rate": 1.0, "eval_episodes": 1},
        "artifacts": {"eval.mp4": "/tmp/eval.mp4"},
        "facts": {"kind": "rl", "engine": "playground", "device": "gpu", "sim_only": True},
    }
    base.update(overlay)
    if "facts" in overlay:
        facts = dict(base.get("facts") or {})
        facts.update(overlay["facts"])
        base["facts"] = facts
    return base


def test_hardware_profile_absent_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HT_HARDWARE_PROFILE", raising=False)
    assert hardware_profile_present() is False


def test_hardware_profile_requires_passed_hardware_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"passed": True, "kind": "sim"}), encoding="utf-8")
    monkeypatch.setenv("HT_HARDWARE_PROFILE", str(bad))
    assert hardware_profile_present() is False

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"passed": True, "kind": "hardware"}), encoding="utf-8")
    monkeypatch.setenv("HT_HARDWARE_PROFILE", str(good))
    assert hardware_profile_present() is True


def test_assess_deploy_blocks_cartpole() -> None:
    report = assess_deploy(
        {
            "run_id": "c1",
            "recipe": "cartpole-balance",
            "status": "passed",
            "metrics": {"passed": True},
            "artifacts": {"eval.mp4": "x"},
            "facts": {"kind": "rl", "engine": "gymnasium"},
        }
    )
    assert report["ok"] is False
    assert report["sim_only"] is True
    assert any("g1-walk" in r for r in report["reasons"])


def test_assess_deploy_blocks_walk_without_hardware_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HT_HARDWARE_PROFILE", raising=False)
    report = assess_deploy(_walk_manifest())
    assert report["ok"] is False
    assert "HT_HARDWARE_PROFILE" in (report["error"] or "")


def test_assess_deploy_ok_shape_with_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "hw.json"
    profile.write_text(json.dumps({"passed": True, "kind": "hardware"}), encoding="utf-8")
    monkeypatch.setenv("HT_HARDWARE_PROFILE", str(profile))
    report = assess_deploy(_walk_manifest())
    assert report["ok"] is True
    assert report["sim_only"] is True  # still sim-only until live torque ships


def test_deploy_run_fails_closed_even_with_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "hw.json"
    profile.write_text(json.dumps({"passed": True, "kind": "hardware"}), encoding="utf-8")
    monkeypatch.setenv("HT_HARDWARE_PROFILE", str(profile))
    runs = tmp_path / "runs"
    run_dir = runs / "walk-1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(_walk_manifest()), encoding="utf-8"
    )
    (run_dir / "eval.mp4").write_bytes(b"fake")
    with pytest.raises(AdapterUnavailable, match="not wired"):
        deploy_run("walk-1", runs_dir=runs)


def test_runner_stamps_sim_only(tmp_path: Path) -> None:
    spec = load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    )
    spec["train"] = {"method": "rl", "steps": 40, "seed": 1}
    spec["eval"] = {"episodes": 1, "record_video": True}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert (manifest.get("facts") or {}).get("sim_only") is True
    assert (manifest.get("facts") or {}).get("runner") == "inprocess"


def test_assess_deploy_blocks_stand_and_mustard() -> None:
    for recipe in ("g1-stand", "pick-and-place"):
        report = assess_deploy(
            {
                "run_id": "r1",
                "recipe": recipe,
                "status": "passed",
                "metrics": {"passed": True},
                "artifacts": {"eval.mp4": "x"},
                "facts": {"kind": "hold" if recipe == "g1-stand" else "imitation"},
            }
        )
        assert report["ok"] is False
        assert any("g1-walk" in r for r in report["reasons"])
        assert len(report["reasons"]) == 1


def test_assess_deploy_blocks_queued_and_failed_metrics() -> None:
    queued = assess_deploy(_walk_manifest(status="queued"))
    assert queued["ok"] is False
    assert any("queued" in r for r in queued["reasons"])

    failed = assess_deploy(
        _walk_manifest(status="completed", metrics={"passed": False, "success_rate": 0.0})
    )
    assert failed["ok"] is False
    assert any("metrics.passed" in r for r in failed["reasons"])


def test_assess_deploy_blocks_missing_eval_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "hw.json"
    profile.write_text(json.dumps({"passed": True, "kind": "hardware"}), encoding="utf-8")
    monkeypatch.setenv("HT_HARDWARE_PROFILE", str(profile))
    report = assess_deploy(_walk_manifest(artifacts={}))
    assert report["ok"] is False
    assert any("eval.mp4" in r for r in report["reasons"])


def test_api_deploy_get_is_preflight_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HT_RUNS_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from humanoid_training.server import app

    run_dir = tmp_path / "walk-1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(_walk_manifest()), encoding="utf-8")
    client = TestClient(app)
    resp = client.get("/api/runs/walk-1/deploy")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is False
    assert body.get("deployed") is False
    assert body.get("reasons")
    # GET must not attempt the unwired driver path.
    assert "not wired" not in (body.get("error") or "").lower()


def test_api_deploy_unknown_run_is_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HT_RUNS_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from humanoid_training.server import app

    client = TestClient(app)
    resp = client.post("/api/runs/does-not-exist/deploy")
    assert resp.status_code == 404


def test_api_deploy_walk_with_profile_still_not_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HT_RUNS_DIR", str(tmp_path))
    profile = tmp_path / "hw.json"
    profile.write_text(json.dumps({"passed": True, "kind": "hardware"}), encoding="utf-8")
    monkeypatch.setenv("HT_HARDWARE_PROFILE", str(profile))
    run_dir = tmp_path / "walk-1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(_walk_manifest(artifacts={"eval.mp4": str(run_dir / "eval.mp4")})),
        encoding="utf-8",
    )
    (run_dir / "eval.mp4").write_bytes(b"fake")
    from fastapi.testclient import TestClient
    from humanoid_training.server import app

    client = TestClient(app)
    resp = client.post("/api/runs/walk-1/deploy")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("deployed") is False
    assert "not wired" in (body.get("error") or "").lower()


def test_cli_deploy_exit_12(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from humanoid_training.cli import main

    runs = tmp_path / "runs"
    run_dir = runs / "walk-1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(_walk_manifest()), encoding="utf-8"
    )
    code = main(["deploy", "walk-1", "--out", str(runs)])
    assert code == 12
    err = capsys.readouterr().err
    assert "sim-only" in err.lower() or "Hardware deploy is blocked" in err
