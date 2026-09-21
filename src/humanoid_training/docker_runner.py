"""Phase 1 local Docker runner. In-process launch stays the default."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from humanoid_training.errors import AdapterUnavailable, repo_root
from humanoid_training.recipes import expand_spec
from humanoid_training.runner import default_runs_dir, load_manifest, new_run_id, write_manifest
from humanoid_training.spec import spec_hash

LogFn = Callable[[str], None]

DEFAULT_IMAGE = "humanoid-training:cpu"


def docker_bin() -> str | None:
    return shutil.which("docker")


def docker_image() -> str:
    return os.environ.get("HT_DOCKER_IMAGE") or DEFAULT_IMAGE


def docker_run_command(
    *,
    spec_in_container: str,
    image: str,
    runs_host: Path,
    run_id: str,
    compile_only: bool,
) -> list[str]:
    """CPU image already has the package. Mount only the runs directory.

    Do not mount the host repo over /workspace: that hides the image's
    editable install. Do not pass --docker: the container must run
    in-process train or it would recurse.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{runs_host}:/runs",
        "-e",
        "HT_ROOT=/workspace",
        "-e",
        "HT_RUNS_DIR=/runs",
        "-e",
        f"HT_RUN_ID={run_id}",
        "-e",
        "HT_IN_CONTAINER=1",
        "-e",
        "SDL_VIDEODRIVER=dummy",
        "-e",
        "SDL_AUDIODRIVER=dummy",
        "-w",
        "/workspace",
        image,
        "train",
        spec_in_container,
        "--out",
        "/runs",
    ]
    if compile_only:
        cmd.append("--compile-only")
    return cmd


def ensure_image(image: str, log: LogFn | None = None) -> None:
    docker = docker_bin()
    if not docker:
        raise AdapterUnavailable(
            "Docker is not on PATH. CPU recipes still train in-process: "
            "ht train spec/examples/cartpole-balance.json"
        )
    inspect = subprocess.run(
        [docker, "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        return
    if log:
        log(f"building Docker image {image}")
    build = subprocess.run(
        [docker, "build", "-t", image, str(repo_root())],
        check=False,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise AdapterUnavailable(
            "Could not build the CPU Docker image. "
            f"{(build.stderr or build.stdout or '').strip()[:400]}"
        )


def run_job_via_docker(
    user_spec: dict[str, Any],
    runs_dir: Path | None = None,
    log: LogFn | None = None,
    compile_only: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Launch the same expand→compile→launch loop inside the CPU container."""
    docker = docker_bin()
    if not docker:
        raise AdapterUnavailable(
            "Docker is not on PATH. CPU recipes still train in-process: "
            "ht train spec/examples/cartpole-balance.json"
        )
    expanded = expand_spec(user_spec)
    public = {k: v for k, v in expanded.items() if not str(k).startswith("_")}
    run_id = run_id or new_run_id(str(public.get("name") or "job"))
    host_runs = Path(runs_dir or default_runs_dir()).resolve()
    host_runs.mkdir(parents=True, exist_ok=True)
    incoming = host_runs / f"{run_id}.spec.json"
    incoming.write_text(json.dumps(user_spec, indent=2), encoding="utf-8")
    image = docker_image()
    ensure_image(image, log=log)
    cmd = docker_run_command(
        spec_in_container=f"/runs/{incoming.name}",
        image=image,
        runs_host=host_runs,
        run_id=run_id,
        compile_only=compile_only,
    )
    cmd[0] = docker
    if log:
        log(" ".join(cmd))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.stdout and log:
        for line in proc.stdout.splitlines():
            log(line)
    run_dir = host_runs / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        err = (proc.stderr or proc.stdout or f"docker exit {proc.returncode}").strip()
        raise AdapterUnavailable(f"Docker train failed: {err[:500]}")
    manifest = load_manifest(run_dir)
    facts = dict(manifest.get("facts") or {})
    facts["runner"] = "docker"
    facts["host_spec_hash"] = spec_hash(public)
    manifest["facts"] = facts
    write_manifest(run_dir, manifest)
    return manifest
