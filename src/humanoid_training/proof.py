"""Phase 3c GPU-box walk proof.

Fake-CLI unit tests are not the exit test. This module is the operator
command to run on a machine with an NVIDIA GPU and a walk engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from humanoid_training import hardware
from humanoid_training.errors import AdapterUnavailable, repo_root
from humanoid_training.runner import default_runs_dir, run_job
from humanoid_training.spec import load_spec

LogFn = Callable[[str], None]

PROOF_STEPS = 2_048
WALK_ENGINES = ("playground", "mjlab", "isaaclab")


def walk_engines_ready() -> dict[str, bool]:
    return {
        "playground": hardware.playground_ready(),
        "mjlab": hardware.mjlab_ready(),
        "isaaclab": hardware.isaac_launch_ready(),
    }


def first_ready_engine(prefer: list[str] | None = None) -> str | None:
    order = list(prefer or WALK_ENGINES)
    ready = walk_engines_ready()
    for name in order:
        if ready.get(name):
            return name
    return None


def _blocked_message(ready: dict[str, bool]) -> str:
    status = hardware.engine_status()
    lines = [
        "Phase 3c walk proof is blocked on this machine — not a silent failure.",
        "Need an NVIDIA GPU and one walk engine:",
        f"  gpu={status.get('gpu')} playground_ready={ready['playground']} "
        f"mjlab_ready={ready['mjlab']} isaac_launch_ready={ready['isaaclab']}",
        "",
        "On a GPU box, pick one:",
        "  pip install playground && ht proof walk",
        "  # mjlab installed → ht proof walk --prefer mjlab",
        "  # HT_ISAAC_CLI=/path/to/isaaclab.sh → ht proof walk --prefer isaaclab",
        "",
        "Do not check a stand clip in as walk gold.",
        "CPU studio: use g1-stand / pick-and-place, or ht train --compile-only.",
    ]
    return "\n".join(lines)


def proof_walk_spec(
    *,
    prefer: list[str] | None = None,
    steps: int = PROOF_STEPS,
    engine: str | None = None,
) -> dict[str, Any]:
    """Short beginner walk spec aimed at the chosen GPU engine."""
    path = repo_root() / "spec" / "examples" / "g1-walk.json"
    spec = load_spec(path)
    chosen = engine or first_ready_engine(prefer)
    if not chosen:
        raise AdapterUnavailable(_blocked_message(walk_engines_ready()))
    prefer_list = [chosen] + [e for e in (prefer or list(WALK_ENGINES)) if e != chosen]
    spec["backend"] = {"prefer": prefer_list, "compute": "local"}
    train = dict(spec.get("train") or {})
    train["method"] = "rl"
    train["steps"] = int(steps)
    train.setdefault("seed", 1)
    spec["train"] = train
    spec["eval"] = {"episodes": 1, "record_video": True}
    return spec


def run_walk_proof(
    *,
    runs_dir: Path | None = None,
    prefer: list[str] | None = None,
    steps: int = PROOF_STEPS,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """Launch a short G1 walk and require a real eval.mp4 + facts.engine."""
    ready = walk_engines_ready()
    engine = first_ready_engine(prefer)
    if not engine:
        raise AdapterUnavailable(_blocked_message(ready))
    if log:
        log(f"proof walk via {engine} steps={steps}")
    spec = proof_walk_spec(prefer=prefer, steps=steps, engine=engine)
    manifest = run_job(spec, runs_dir=runs_dir or default_runs_dir(), log=log)
    report = summarize_walk_proof(manifest, expected_engine=engine)
    if not report["ok"]:
        raise AdapterUnavailable(report["error"] or "Walk proof failed.")
    return report


def summarize_walk_proof(manifest: dict[str, Any], *, expected_engine: str | None = None) -> dict[str, Any]:
    """Judge a completed walk run for the Phase 3c exit test."""
    run_dir = Path(manifest.get("run_dir") or "")
    facts = dict(manifest.get("facts") or {})
    status = str(manifest.get("status") or "")
    video = run_dir / "eval.mp4" if run_dir else Path()
    engine = str(facts.get("engine") or "")
    errors: list[str] = []
    if status != "passed":
        errors.append(f"status={status!r} (want passed). {manifest.get('error') or ''}".strip())
    if not video.is_file():
        errors.append("missing eval.mp4 — not a walk proof. Do not substitute a stand clip.")
    elif video.stat().st_size < 8:
        errors.append("eval.mp4 is empty or tiny — not a walk proof.")
    if engine not in WALK_ENGINES:
        errors.append(f"facts.engine={engine!r} (want playground|mjlab|isaaclab)")
    if expected_engine and engine and engine != expected_engine:
        errors.append(f"facts.engine={engine!r} but proof targeted {expected_engine!r}")
    if facts.get("kind") and facts.get("kind") != "rl":
        errors.append(f"facts.kind={facts.get('kind')!r} (want rl)")
    ok = not errors
    return {
        "ok": ok,
        "phase": "3c",
        "run_id": manifest.get("run_id"),
        "run_dir": str(run_dir) if run_dir else None,
        "status": status,
        "engine": engine or None,
        "video": str(video) if video.is_file() else None,
        "facts": facts,
        "error": "; ".join(errors) if errors else None,
    }
