from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_training.adapters import select_adapter
from humanoid_training.recipes import expand_spec
from humanoid_training.runner import run_job
from humanoid_training.spec import load_spec


FAKE_MJLAB = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
task = args[0] if args else "unknown"
log_root = "logs"
video = False
i = 0
while i < len(args):
    if args[i] == "--log-root" and i + 1 < len(args):
        log_root = args[i + 1]
        i += 2
        continue
    if args[i] == "--video" and i + 1 < len(args) and args[i + 1] == "True":
        video = True
        i += 2
        continue
    i += 1
print(f"mjlab task={task} log_root={log_root} video={video}", flush=True)
code = int(os.environ.get("HT_FAKE_MJLAB_EXIT", "0"))
if code != 0:
    print("fake mjlab failing", flush=True)
    raise SystemExit(code)
if os.environ.get("HT_FAKE_MJLAB_NO_VIDEO") == "1" or not video:
    raise SystemExit(0)
dest = Path(log_root) / "videos" / "train" / "rl-video-step-0.mp4"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(b"fake-mjlab-mp4")
raise SystemExit(0)
"""

FAKE_ISAAC = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

joined = " ".join(sys.argv)
print(f"isaaclab {joined}", flush=True)
code = int(os.environ.get("HT_FAKE_ISAAC_EXIT", "0"))
if code != 0:
    print("fake isaac failing", flush=True)
    raise SystemExit(code)
want_video = "--video" in sys.argv
if os.environ.get("HT_FAKE_ISAAC_NO_VIDEO") == "1" or not want_video:
    raise SystemExit(0)
dest = Path("logs") / "rsl_rl" / "g1_flat" / "videos" / "play" / "rl-video.mp4"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(b"fake-isaac-mp4")
raise SystemExit(0)
"""

FAKE_OSMO = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

args = sys.argv[1:]
print("osmo", " ".join(args), flush=True)

def die(code=1):
    raise SystemExit(code)

if len(args) >= 2 and args[0] == "workflow" and args[1] == "submit":
    # stdout looks like real CLI text; also support --format-type json
    if "--format-type" in args and "json" in args:
        print(json.dumps({"workflow_id": "ht-walk-osmo-1"}), flush=True)
    else:
        print("Workflow submit successful.", flush=True)
        print("Workflow ID        - ht-walk-osmo-1", flush=True)
    raise SystemExit(0)

if len(args) >= 3 and args[0] == "workflow" and args[1] == "query":
    wid = args[-1]
    if "--format-type" in args and "json" in args:
        print(json.dumps({"workflow_id": wid, "status": "COMPLETED"}), flush=True)
    else:
        print(f"Workflow ID : {wid}", flush=True)
        print("Status      : COMPLETED", flush=True)
    raise SystemExit(0)

if len(args) >= 3 and args[0] == "workflow" and args[1] == "rsync" and args[2] == "download":
    # ... download <id> [task] <remote>:<local>
    mapping = args[-1]
    if ":" not in mapping:
        die(2)
    remote, local = mapping.split(":", 1)
    dest = Path(local)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "rollout0.mp4"
    out.write_bytes(b"fake-osmo-harvest-mp4")
    print(f"downloaded {out}", flush=True)
    raise SystemExit(0)

die(2)
"""

FAKE_DOCKER = """#!/usr/bin/env python3
import sys
from pathlib import Path

args = sys.argv[1:]
host = None
for i, item in enumerate(args):
    if item == "-v" and i + 1 < len(args) and "/ht_run" in args[i + 1]:
        host = args[i + 1].split(":", 1)[0]
        break
print("docker", " ".join(args), flush=True)
if not host:
    raise SystemExit(2)
dest = Path(host) / "isaac_logs" / "play.mp4"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(b"fake-docker-isaac-mp4")
raise SystemExit(0)
"""


def _walk_spec(**overlay):
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    spec["train"] = {"method": "rl", "steps": 8, "seed": 1, **(overlay.get("train") or {})}
    if "eval" in overlay:
        spec["eval"] = overlay["eval"]
    if "backend" in overlay:
        spec["backend"] = overlay["backend"]
    return spec


def _write_cli(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_walk_selects_mjlab_when_playground_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "mjlab-train", FAKE_MJLAB)
    monkeypatch.setenv("HT_MJLAB_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    spec = expand_spec(_walk_spec())
    assert select_adapter(spec).name == "mjlab"


def test_walk_prefer_isaaclab_selects_isaac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "isaaclab.sh", FAKE_ISAAC)
    monkeypatch.setenv("HT_ISAAC_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    spec = expand_spec(_walk_spec(backend={"prefer": ["isaaclab"], "compute": "local-docker"}))
    assert select_adapter(spec).name == "isaaclab"


def test_g1_walk_mjlab_harvests_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "mjlab-train", FAKE_MJLAB)
    monkeypatch.setenv("HT_MJLAB_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    spec = _walk_spec(backend={"prefer": ["mjlab"], "compute": "local-docker"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "eval.mp4").read_bytes() == b"fake-mjlab-mp4"
    facts = manifest.get("facts") or {}
    assert facts.get("kind") == "rl"
    assert facts.get("engine") == "mjlab"
    assert facts.get("device") == "gpu"
    assert facts.get("env") == "Mjlab-Velocity-Flat-Unitree-G1"
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Mjlab-Velocity-Flat-Unitree-G1" in log


def test_g1_walk_mjlab_exit0_without_video_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "mjlab-train", FAKE_MJLAB)
    monkeypatch.setenv("HT_MJLAB_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_FAKE_MJLAB_NO_VIDEO", "1")
    spec = _walk_spec(backend={"prefer": ["mjlab"], "compute": "local-docker"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "blocked"
    err = (manifest.get("error") or "").lower()
    assert "mp4" in err or "video" in err
    assert "stand" in err
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()


def test_g1_walk_mjlab_nonzero_exit_is_not_a_walk_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "mjlab-train", FAKE_MJLAB)
    monkeypatch.setenv("HT_MJLAB_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_FAKE_MJLAB_EXIT", "1")
    spec = _walk_spec(backend={"prefer": ["mjlab"], "compute": "local-docker"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "completed"
    assert manifest.get("metrics", {}).get("passed") is False
    assert (manifest.get("facts") or {}).get("engine") == "mjlab"
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()


def test_g1_walk_isaac_harvests_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _write_cli(tmp_path, "isaaclab.sh", FAKE_ISAAC)
    monkeypatch.setenv("HT_ISAAC_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    spec = _walk_spec(backend={"prefer": ["isaaclab"], "compute": "osmo"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "eval.mp4").read_bytes() == b"fake-isaac-mp4"
    facts = manifest.get("facts") or {}
    assert facts.get("kind") == "rl"
    assert facts.get("engine") == "isaaclab"
    assert facts.get("device") == "gpu"
    assert facts.get("env") == "Isaac-Velocity-Flat-G1-v0"
    assert facts.get("launch") == "isaaclab.sh"
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Isaac-Velocity-Flat-G1-v0" in log
    assert (run_dir / "osmo_workflow.yaml").is_file()


def test_g1_walk_isaac_blocked_without_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HT_GPU", "1")
    spec = _walk_spec(backend={"prefer": ["isaaclab"], "compute": "osmo"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "blocked"
    err = manifest.get("error") or ""
    assert "Isaac" in err
    assert "g1-stand" in err or "pick-and-place" in err
    assert not (Path(manifest["run_dir"]) / "eval.mp4").is_file()
    assert (Path(manifest["run_dir"]) / "osmo_workflow.yaml").is_file()


def test_g1_walk_isaac_docker_harvests_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = _write_cli(tmp_path, "docker", FAKE_DOCKER)
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_DOCKER_GPU", "1")
    monkeypatch.setattr("humanoid_training.hardware.docker_bin", lambda: str(docker))
    spec = _walk_spec(backend={"prefer": ["isaaclab"], "compute": "local-docker"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "eval.mp4").read_bytes() == b"fake-docker-isaac-mp4"
    facts = manifest.get("facts") or {}
    assert facts.get("engine") == "isaaclab"
    assert facts.get("launch") == "docker"


def test_g1_walk_osmo_harvests_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    osmo = _write_cli(tmp_path, "osmo", FAKE_OSMO)
    monkeypatch.setenv("HT_OSMO_CLI", str(osmo))
    monkeypatch.setenv("HT_OSMO_POLL_SECONDS", "0.01")
    monkeypatch.setenv("HT_OSMO_TIMEOUT_SECONDS", "2")
    # CI runners often have `docker` on PATH; OSMO must still win when
    # HT_DOCKER_GPU is off.
    monkeypatch.setenv("HT_DOCKER_GPU", "0")
    monkeypatch.setenv("HT_GPU", "0")
    spec = _walk_spec(backend={"prefer": ["isaaclab"], "compute": "osmo"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error") or manifest.get("notes")
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "eval.mp4").read_bytes() == b"fake-osmo-harvest-mp4"
    facts = manifest.get("facts") or {}
    assert facts.get("engine") == "isaaclab"
    assert facts.get("launch") == "osmo"
    assert facts.get("device") == "remote"
    assert facts.get("workflow_id") == "ht-walk-osmo-1"
    assert (run_dir / "osmo_workflow.yaml").is_file()
    yaml_text = (run_dir / "osmo_workflow.yaml").read_text(encoding="utf-8")
    assert "ht_eval" in yaml_text
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "workflow submit" in log
    assert "osmo status=COMPLETED" in log or "COMPLETED" in log


def test_osmo_not_stolen_by_docker_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: CI has docker but no GPU Docker opt-in; OSMO must harvest."""
    osmo = _write_cli(tmp_path, "osmo", FAKE_OSMO)
    fake_docker = _write_cli(
        tmp_path,
        "docker",
        "#!/usr/bin/env python3\nimport sys\nprint('should-not-run')\nsys.exit(125)\n",
    )
    monkeypatch.setenv("HT_OSMO_CLI", str(osmo))
    monkeypatch.setenv("HT_OSMO_POLL_SECONDS", "0.01")
    monkeypatch.setenv("HT_OSMO_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("HT_DOCKER_GPU", "0")
    monkeypatch.setenv("HT_GPU", "1")  # would have triggered old buggy branch
    monkeypatch.setattr("humanoid_training.hardware.docker_bin", lambda: str(fake_docker))
    spec = _walk_spec(backend={"prefer": ["isaaclab"], "compute": "osmo"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error")
    assert (manifest.get("facts") or {}).get("launch") == "osmo"
    assert (Path(manifest["run_dir"]) / "eval.mp4").is_file()


def test_osmo_ready_makes_walk_launch_here(monkeypatch: pytest.MonkeyPatch) -> None:
    from humanoid_training.recipes import public_catalog

    monkeypatch.setattr("humanoid_training.hardware.playground_ready", lambda: False)
    monkeypatch.setattr("humanoid_training.hardware.mjlab_ready", lambda: False)
    monkeypatch.setattr("humanoid_training.hardware.osmo_ready", lambda: True)
    monkeypatch.setattr("humanoid_training.hardware.isaac_launch_ready", lambda: True)
    catalog = public_catalog()
    by_id = {r["id"]: r for r in catalog["recipes"]}
    assert by_id["g1-walk"]["launch_here"] is True
    assert "g1-reach" not in by_id


def test_parse_workflow_id_and_status() -> None:
    from humanoid_training.adapters.osmo_remote import parse_workflow_id, parse_workflow_status

    assert parse_workflow_id("Workflow ID        - abc-123\n") == "abc-123"
    assert parse_workflow_id('{"workflow_id": "w1"}') == "w1"
    assert parse_workflow_status("Status      : COMPLETED\n") == "COMPLETED"
    assert parse_workflow_status('{"status": "FAILED"}') == "FAILED"


def test_walk_does_not_claim_invented_reach_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 3f: no g1-reach recipe. Walk payloads must not invent reach env ids."""
    cli = _write_cli(tmp_path, "mjlab-train", FAKE_MJLAB)
    monkeypatch.setenv("HT_MJLAB_CLI", str(cli))
    monkeypatch.setenv("HT_GPU", "1")
    spec = _walk_spec(backend={"prefer": ["mjlab"], "compute": "local"})
    manifest = run_job(spec, runs_dir=tmp_path / "runs")
    assert manifest["status"] == "passed", manifest.get("error")
    payload = (Path(manifest["run_dir"]) / "train_mjlab.sh").read_text(encoding="utf-8")
    assert "Mjlab-Velocity-Flat-Unitree-G1" in payload
    assert "G1Reach-v0" not in payload
    assert "Isaac-Reach-G1-v0" not in payload
    from humanoid_training.recipes import public_catalog

    catalog = public_catalog()
    assert "g1-reach" not in {r["id"] for r in catalog["recipes"]}
