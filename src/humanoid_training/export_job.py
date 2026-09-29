"""Export a compiled job for power users (N7) — not a new orchestrator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from humanoid_training.adapters import select_adapter, write_payload_files
from humanoid_training.errors import AdapterUnavailable
from humanoid_training.recipes import expand_spec
from humanoid_training.runner import default_runs_dir, load_manifest
from humanoid_training.spec import load_spec


def export_spec_to(spec: dict[str, Any], dest: Path) -> dict[str, Any]:
    """Compile the expanded spec into `dest` and write a short README."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    expanded = expand_spec(spec)
    adapter = select_adapter(expanded)
    payload = adapter.compile(expanded)
    write_payload_files(dest, payload)
    (dest / "expanded_spec.json").write_text(
        json.dumps({k: v for k, v in expanded.items() if not str(k).startswith("_")}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    readme = _readme(expanded, adapter.name, payload)
    (dest / "README.md").write_text(readme, encoding="utf-8")
    return {
        "ok": True,
        "dest": str(dest),
        "adapter": adapter.name,
        "files": sorted(p.name for p in dest.iterdir() if p.is_file()),
        "env_name": payload.env_name,
    }


def export_spec_path(path: str | Path, dest: Path) -> dict[str, Any]:
    return export_spec_to(load_spec(path), dest)


def export_run(run_id: str, dest: Path, *, runs_dir: Path | None = None) -> dict[str, Any]:
    """Copy compile artifacts from an existing run into `dest`."""
    root = Path(runs_dir or default_runs_dir())
    run_dir = root / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id} under {root}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(run_dir)
    copied: list[str] = []
    for name in (
        "engine_payload.json",
        "osmo_workflow.yaml",
        "train_isaac.sh",
        "train_mjlab.sh",
        "job.json",
        "spec.json",
        "manifest.json",
        "proof_3c.json",
    ):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
    if not copied:
        # Re-compile from the stored spec if the run never wrote payloads.
        spec_path = run_dir / "spec.json"
        if spec_path.is_file():
            return export_spec_to(json.loads(spec_path.read_text(encoding="utf-8")), dest)
        raise AdapterUnavailable(
            f"Run {run_id} has no exportable payload files and no spec.json."
        )
    recipe = manifest.get("recipe") or ""
    adapter = (manifest.get("facts") or {}).get("engine") or manifest.get("adapter") or ""
    (dest / "README.md").write_text(
        "\n".join(
            [
                f"# Exported run `{run_id}`",
                "",
                f"- recipe: `{recipe}`",
                f"- adapter/engine: `{adapter}`",
                f"- status: `{manifest.get('status')}`",
                "",
                "Payload files were copied from the run directory.",
                "This is an escape hatch for researchers — not a new cluster product.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "dest": str(dest),
        "run_id": run_id,
        "adapter": adapter,
        "files": sorted(p.name for p in dest.iterdir() if p.is_file()),
    }


def _readme(expanded: dict[str, Any], adapter: str, payload: Any) -> str:
    recipe = (expanded.get("task") or {}).get("recipe") or expanded.get("name") or ""
    notes = "\n".join(f"- {n}" for n in (payload.notes or [])[:8])
    cmd = " ".join(str(c) for c in (payload.command or [])[:12])
    return "\n".join(
        [
            f"# Exported job: `{recipe}`",
            "",
            f"Adapter: `{adapter}`",
            f"Env / task: `{payload.env_name}`",
            "",
            "## Command (compiled)",
            "",
            "```",
            cmd or "(see payload files)",
            "```",
            "",
            "## Notes",
            notes or "- (none)",
            "",
            "Re-run from the studio with Train, or use these files inside the engine.",
            "Humanoid Training does not replace Isaac Lab / OSMO / Docker.",
            "",
        ]
    )
