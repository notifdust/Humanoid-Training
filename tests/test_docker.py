from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from humanoid_training.docker_runner import docker_run_command, run_job_via_docker
from humanoid_training.errors import AdapterUnavailable
from humanoid_training.runner import run_job, write_manifest
from humanoid_training.spec import load_spec


def _cartpole():
    return load_spec(
        Path(__file__).resolve().parents[1] / "spec" / "examples" / "cartpole-balance.json"
    )


def _walk():
    return load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")


def test_docker_run_command_mounts_runs_and_passes_run_id() -> None:
    cmd = docker_run_command(
        spec_in_container="/runs/job.spec.json",
        image="humanoid-training:cpu",
        runs_host=Path("/tmp/runs"),
        run_id="abc123",
        compile_only=True,
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "/tmp/runs:/runs" in cmd
    assert "HT_RUN_ID=abc123" in cmd
    assert "HT_IN_CONTAINER=1" in cmd
    assert cmd[-1] == "--compile-only"
    assert "train" in cmd
    assert "/runs/job.spec.json" in cmd
    # Host repo must not overlay the image install; --docker must not recurse.
    joined = " ".join(cmd)
    assert ":/workspace" not in joined
    assert "--docker" not in cmd


def test_docker_missing_fails_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("humanoid_training.docker_runner.docker_bin", lambda: None)
    with pytest.raises(AdapterUnavailable, match="Docker is not on PATH"):
        run_job_via_docker(_cartpole(), runs_dir=tmp_path)


def test_run_job_honors_ht_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HT_RUN_ID", "fixed-run-id")
    spec = _cartpole()
    spec["train"] = {"method": "rl", "steps": 20, "seed": 1}
    spec["eval"] = {"episodes": 1, "record_video": False}
    manifest = run_job(spec, runs_dir=tmp_path)
    assert manifest["run_id"] == "fixed-run-id"
    assert (tmp_path / "fixed-run-id" / "manifest.json").is_file()
    assert (manifest.get("facts") or {}).get("runner") == "inprocess"


def test_run_job_does_not_auto_docker_local_docker_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """g1-walk's local-docker compute is a GPU/Isaac target, not this CPU image."""

    def boom(*_args, **_kwargs):
        raise AssertionError("run_job must not auto-route walk onto the CPU Docker runner")

    monkeypatch.setattr("humanoid_training.docker_runner.run_job_via_docker", boom)
    manifest = run_job(_walk(), runs_dir=tmp_path, compile_only=True)
    assert manifest["status"] == "compiled"
    assert (manifest.get("facts") or {}).get("runner") == "inprocess"


def test_docker_success_stamps_runner_and_does_not_recurse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr("humanoid_training.docker_runner.docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr("humanoid_training.docker_runner.ensure_image", lambda *a, **k: None)

    def fake_run(cmd, check=False, capture_output=True, text=True):
        captured["cmd"] = list(cmd)
        run_id = next(item.split("=", 1)[1] for item in cmd if str(item).startswith("HT_RUN_ID="))
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            run_dir,
            {
                "run_id": run_id,
                "status": "passed",
                "facts": {"kind": "rl", "runner": "inprocess"},
                "metrics": {"passed": True},
            },
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("humanoid_training.docker_runner.subprocess.run", fake_run)
    manifest = run_job_via_docker(_cartpole(), runs_dir=tmp_path, run_id="dock-1")
    assert manifest["facts"]["runner"] == "docker"
    assert "host_spec_hash" in manifest["facts"]
    assert "--docker" not in captured["cmd"]
    assert "HT_IN_CONTAINER=1" in captured["cmd"]
    assert not (tmp_path / "dock-1.spec.json").is_file()


def test_docker_preserves_blocked_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("humanoid_training.docker_runner.docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr("humanoid_training.docker_runner.ensure_image", lambda *a, **k: None)

    def fake_run(cmd, check=False, capture_output=True, text=True):
        run_id = next(item.split("=", 1)[1] for item in cmd if str(item).startswith("HT_RUN_ID="))
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            run_dir,
            {
                "run_id": run_id,
                "status": "blocked",
                "error": "Playground is not installed. Use g1-stand on the CPU studio.",
                "facts": {"kind": "compile"},
            },
        )
        return SimpleNamespace(returncode=12, stdout="", stderr="blocked")

    monkeypatch.setattr("humanoid_training.docker_runner.subprocess.run", fake_run)
    manifest = run_job_via_docker(_cartpole(), runs_dir=tmp_path, run_id="walk-dock")
    assert manifest["status"] == "blocked"
    assert "Playground" in (manifest.get("error") or "")
    assert manifest["facts"]["runner"] == "docker"


def test_docker_refuses_gpu_recipe_on_cpu_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("GPU recipes must not invoke docker on the CPU image")

    monkeypatch.setattr("humanoid_training.docker_runner.docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr("humanoid_training.docker_runner.ensure_image", boom)
    monkeypatch.setattr("humanoid_training.docker_runner.subprocess.run", boom)
    with pytest.raises(AdapterUnavailable, match="CPU Docker"):
        run_job_via_docker(_walk(), runs_dir=tmp_path, run_id="walk-dock")
    assert called["n"] == 0


def test_docker_gpu_flag_allows_mocked_walk_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HT_DOCKER_GPU", "1")
    monkeypatch.setattr("humanoid_training.docker_runner.docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr("humanoid_training.docker_runner.ensure_image", lambda *a, **k: None)

    def fake_run(cmd, check=False, capture_output=True, text=True):
        run_id = next(item.split("=", 1)[1] for item in cmd if str(item).startswith("HT_RUN_ID="))
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            run_dir,
            {
                "run_id": run_id,
                "status": "blocked",
                "error": "Playground is not installed. Use g1-stand on the CPU studio.",
                "facts": {"kind": "compile"},
            },
        )
        return SimpleNamespace(returncode=12, stdout="", stderr="blocked")

    monkeypatch.setattr("humanoid_training.docker_runner.subprocess.run", fake_run)
    manifest = run_job_via_docker(_walk(), runs_dir=tmp_path, run_id="walk-gpu-dock")
    assert manifest["status"] == "blocked"
    assert manifest["facts"]["runner"] == "docker"


def test_docker_missing_manifest_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("humanoid_training.docker_runner.docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr("humanoid_training.docker_runner.ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(
        "humanoid_training.docker_runner.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="no such image"),
    )
    with pytest.raises(AdapterUnavailable, match="Docker train failed"):
        run_job_via_docker(_cartpole(), runs_dir=tmp_path, run_id="missing")
