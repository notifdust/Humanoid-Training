"""OSMO submit → poll → rsync download for Phase 3d remote harvest.

Emits the existing OSMO CLI only. Browser never holds credentials.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from humanoid_training.errors import AdapterUnavailable

LogFn = Callable[[str], None]

WORKSPACE_VIDEO_DIR = "/osmo/run/workspace/ht_eval"
_ID_RE = re.compile(r"Workflow ID\s*[-:]\s*(\S+)", re.IGNORECASE)
_TERMINAL_OK = {"COMPLETED"}
_TERMINAL_BAD = {
    "FAILED",
    "FAILED_EXEC_TIMEOUT",
    "FAILED_SERVER_ERROR",
    "FAILED_QUEUE_TIMEOUT",
    "FAILED_SUBMISSION",
    "FAILED_CANCELED",
    "FAILED_BACKEND_ERROR",
    "FAILED_IMAGE_PULL",
    "FAILED_EVICTED",
    "FAILED_START_ERROR",
    "FAILED_START_TIMEOUT",
    "FAILED_PREEMPTED",
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_workflow_id(text: str) -> str | None:
    """Extract a workflow id from OSMO submit text or JSON stdout."""
    text = (text or "").strip()
    if not text:
        return None
    # Prefer a JSON object line (submit --format-type json).
    for chunk in (text, *text.splitlines()):
        chunk = chunk.strip()
        if not chunk.startswith("{"):
            continue
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            for key in ("workflow_id", "workflowId", "id", "Workflow ID"):
                val = data.get(key)
                if val:
                    return str(val).strip()
            nested = data.get("workflow")
            if isinstance(nested, dict):
                for key in ("id", "workflow_id", "workflowId"):
                    val = nested.get(key)
                    if val:
                        return str(val).strip()
    match = _ID_RE.search(text)
    if match:
        return match.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("workflow id"):
            parts = re.split(r"[-:]", line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip().split()[0]
    return None


def parse_workflow_status(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    for chunk in (text, *text.splitlines()):
        chunk = chunk.strip()
        if not chunk.startswith("{"):
            continue
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            for key in ("status", "Status", "state", "State"):
                val = data.get(key)
                if val:
                    return str(val).strip().upper()
            nested = data.get("workflow")
            if isinstance(nested, dict):
                for key in ("status", "Status", "state"):
                    val = nested.get(key)
                    if val:
                        return str(val).strip().upper()
    match = re.search(r"Status\s*[-:]\s*(\S+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    return None


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


def submit_workflow(osmo: str, yaml_path: Path, log: LogFn) -> str:
    cmd = [osmo, "workflow", "submit", "--format-type", "json", str(yaml_path)]
    log(" ".join(cmd))
    proc = _run(cmd, cwd=yaml_path.parent)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in (proc.stdout or "").splitlines():
        log(line)
    if proc.returncode != 0:
        # Retry without json for older CLIs.
        cmd = [osmo, "workflow", "submit", str(yaml_path)]
        log(" ".join(cmd))
        proc = _run(cmd, cwd=yaml_path.parent)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for line in (proc.stdout or "").splitlines():
            log(line)
        if proc.returncode != 0:
            raise AdapterUnavailable(
                f"osmo workflow submit exited {proc.returncode}. "
                f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
            )
    workflow_id = parse_workflow_id(out)
    if not workflow_id:
        raise AdapterUnavailable(
            "osmo workflow submit succeeded but printed no Workflow ID. "
            "Cannot poll or harvest. Not a silent Isaac success."
        )
    log(f"osmo workflow_id={workflow_id}")
    return workflow_id


def wait_for_workflow(
    osmo: str,
    workflow_id: str,
    log: LogFn,
    *,
    poll_s: float | None = None,
    timeout_s: float | None = None,
) -> str:
    poll_s = _env_float("HT_OSMO_POLL_SECONDS", 5.0) if poll_s is None else poll_s
    timeout_s = _env_float("HT_OSMO_TIMEOUT_SECONDS", 3600.0) if timeout_s is None else timeout_s
    deadline = time.monotonic() + max(timeout_s, 1.0)
    last = ""
    while time.monotonic() < deadline:
        cmd = [osmo, "workflow", "query", "--format-type", "json", workflow_id]
        proc = _run(cmd)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            cmd = [osmo, "workflow", "query", workflow_id]
            proc = _run(cmd)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        status = parse_workflow_status(text) or ""
        if status and status != last:
            log(f"osmo status={status}")
            last = status
        if status in _TERMINAL_OK:
            return status
        if status in _TERMINAL_BAD or status.startswith("FAILED"):
            raise AdapterUnavailable(
                f"osmo workflow {workflow_id} ended with status={status}. "
                "Not harvesting a clip. Check OSMO logs, or train on a local GPU."
            )
        time.sleep(max(poll_s, 0.01))
    raise AdapterUnavailable(
        f"osmo workflow {workflow_id} timed out after {timeout_s}s "
        f"(last status={last or 'unknown'}). Not a silent Isaac success."
    )


def download_eval_mp4(
    osmo: str,
    workflow_id: str,
    dest_dir: Path,
    log: LogFn,
    *,
    task: str = "train",
    remote_dir: str = WORKSPACE_VIDEO_DIR,
) -> Path | None:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # OSMO treats local path as a directory; files land inside it.
    mapping = f"{remote_dir}:{dest_dir}"
    cmd = [osmo, "workflow", "rsync", "download", workflow_id, task, mapping]
    log(" ".join(cmd))
    proc = _run(cmd)
    for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines()[:40]:
        log(line)
    if proc.returncode != 0:
        # Lead-task form without explicit task name.
        cmd = [osmo, "workflow", "rsync", "download", workflow_id, mapping]
        log(" ".join(cmd))
        proc = _run(cmd)
        for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines()[:40]:
            log(line)
        if proc.returncode != 0:
            raise AdapterUnavailable(
                f"osmo rsync download exited {proc.returncode}. "
                f"Expected mp4 under {remote_dir} on task {task}. "
                "Not substituting a stand clip."
            )
    videos = sorted(p for p in dest_dir.rglob("*.mp4") if p.is_file())
    return videos[0] if videos else None


def harvest_osmo_walk(
    osmo: str,
    yaml_path: Path,
    run_dir: Path,
    log: LogFn,
) -> dict[str, Any]:
    """Submit → wait COMPLETED → rsync ht_eval → path to first mp4."""
    run_dir = Path(run_dir)
    workflow_id = submit_workflow(osmo, yaml_path, log)
    status = wait_for_workflow(osmo, workflow_id, log)
    staging = run_dir / "osmo_download"
    video = download_eval_mp4(osmo, workflow_id, staging, log)
    return {
        "workflow_id": workflow_id,
        "status": status,
        "video": video,
        "staging": staging,
    }
