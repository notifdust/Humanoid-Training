"""Shared subprocess + video harvest for GPU engine adapters."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from humanoid_training.errors import AdapterUnavailable

LogFn = Callable[[str], None]


def write_job(run_dir: Path, payload: dict[str, Any]) -> None:
    path = Path(run_dir) / "job.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stream_process(
    command: list[str],
    run_dir: Path,
    log: LogFn,
    *,
    missing: str,
) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
            env=env,
            text=True,
            bufsize=1,
        )
    except OSError as err:
        raise AdapterUnavailable(f"{missing} ({err}). Run dir: {run_dir}") from err
    write_job(
        run_dir,
        {"command": command, "pid": proc.pid, "cwd": str(run_dir), "status": "running"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip("\n"))
    return int(proc.wait())


def harvest_mp4(
    logdir: Path,
    dest: Path,
    *,
    prefer: tuple[str, ...] = (),
) -> Path | None:
    """Copy an engine mp4 to dest. Prefer filenames in `prefer` (e.g. rollout0.mp4)."""
    logdir = Path(logdir)
    if not logdir.is_dir():
        return None
    videos = sorted(p for p in logdir.rglob("*.mp4") if p.is_file())
    if not videos:
        return None
    src = None
    for name in prefer:
        hits = [p for p in videos if p.name == name]
        if hits:
            src = hits[0]
            break
    if src is None:
        src = videos[0]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest
