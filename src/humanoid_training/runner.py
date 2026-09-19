from __future__ import annotations

import json
import platform
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from humanoid_training import __version__
from humanoid_training.adapters import registry, select_adapter, write_payload_files
from humanoid_training.artifacts import RUN_ARTIFACTS
from humanoid_training.errors import AdapterUnavailable, NoAdapter, repo_root
from humanoid_training.recipes import expand_spec
from humanoid_training.spec import spec_hash

LogFn = Callable[[str], None]


def default_runs_dir() -> Path:
    import os

    env = os.environ.get("HT_RUNS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "runs"


def new_run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    slug = slug[:32] or "run"
    return f"{stamp}-{slug}-{uuid4().hex[:6]}"


def _log_file(run_dir: Path, log: LogFn | None) -> LogFn:
    path = run_dir / "run.log"

    def emit(message: str) -> None:
        line = message.rstrip()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if log:
            log(line)

    return emit


def run_job(
    user_spec: dict[str, Any],
    runs_dir: Path | None = None,
    log: LogFn | None = None,
    compile_only: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Expand, compile, optionally train. Always writes a manifest."""
    expanded = expand_spec(user_spec)
    public_spec = {k: v for k, v in expanded.items() if not str(k).startswith("_")}
    hashed = spec_hash(public_spec)
    run_id = run_id or new_run_id(str(public_spec.get("name") or "job"))
    run_dir = Path(runs_dir or default_runs_dir()) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    emit = _log_file(run_dir, log)

    (run_dir / "spec.json").write_text(
        json.dumps(public_spec, indent=2),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "studio_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "spec_hash": hashed,
        "recipe": (public_spec.get("task") or {}).get("recipe"),
        "run_dir": str(run_dir),
        "artifacts": {},
        "metrics": {},
        "adapter": None,
        "error": None,
    }
    write_manifest(run_dir, manifest)
    emit(f"run {run_id} spec_hash={hashed}")

    try:
        adapter = select_adapter(expanded)
        payload = adapter.compile(expanded)
        write_payload_files(run_dir, payload)
        compiled_engines = [payload.adapter]
        for name, candidate in registry().items():
            if name == payload.adapter:
                continue
            if not candidate.support(expanded).ok:
                continue
            extra = candidate.compile(expanded)
            write_payload_files(run_dir / "engines" / name, extra)
            compiled_engines.append(name)
            emit(f"also compiled {name} -> engines/{name}/")
        manifest["adapter"] = payload.as_dict()
        manifest["compiled_engines"] = compiled_engines
        manifest["ignored_fields"] = payload.ignored_fields
        write_manifest(run_dir, manifest)
        emit(f"compiled adapter={payload.adapter} env={payload.env_name}")
        for note in payload.notes:
            emit(note)

        if compile_only:
            manifest["status"] = "compiled"
            emit("compile-only; skipping launch")
            write_manifest(run_dir, manifest)
            return manifest

        result = adapter.launch(expanded, payload, run_dir, emit)
        manifest["status"] = "passed" if result.passed else "completed"
        manifest["metrics"] = {
            "success_rate": result.success_rate,
            "mean_return": result.mean_return,
            "eval_episodes": result.episodes,
            "passed": result.passed,
        }
        manifest["notes"] = result.notes
        manifest["artifacts"] = collect_artifacts(run_dir, result.video_path)
        emit(
            f"eval success_rate={result.success_rate:.2f} "
            f"mean_return={result.mean_return:.1f} passed={result.passed}"
        )
    except AdapterUnavailable as err:
        manifest["status"] = "blocked"
        manifest["error"] = str(err)
        emit(f"blocked: {err}")
    except NoAdapter as err:
        manifest["status"] = "blocked"
        manifest["error"] = str(err)
        emit(f"blocked: {err}")
    except Exception as err:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(err).__name__}: {err}"
        manifest["traceback"] = traceback.format_exc()
        emit(f"failed: {err}")
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)

    return manifest


KNOWN_ARTIFACTS = RUN_ARTIFACTS


def collect_artifacts(run_dir: Path, video_path: Path | None = None) -> dict[str, str]:
    """Name the files a run produced. The studio only plays what is listed here."""
    artifacts: dict[str, str] = {}
    if video_path and Path(video_path).is_file():
        artifacts["eval.mp4"] = str(video_path)
    for name in RUN_ARTIFACTS:
        path = Path(run_dir) / name
        if path.is_file():
            artifacts.setdefault(name, str(path))
    return artifacts


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    last_error: Exception | None = None
    for _ in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as err:
            last_error = err
            import time

            time.sleep(0.02)
    raise last_error  # type: ignore[misc]


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = run_dir / "manifest.json"
    tmp = run_dir / ".manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


# Back-compat for older imports.
_write_manifest = write_manifest
