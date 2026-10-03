"""Phase 3c GPU-box walk proof.

Fake-CLI unit tests are not the exit test. This module is the operator
command to run on a machine with an NVIDIA GPU and a walk engine.

Host readiness (`assess_walk_proof_host`) and a durable `proof_3c.json`
are part of the operator harness. A live walking clip on a real GPU is
still the phase exit test — do not claim Phase 3c done from CPU CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from humanoid_training import hardware
from humanoid_training.errors import AdapterUnavailable, repo_root
from humanoid_training.runner import default_runs_dir, run_job
from humanoid_training.spec import load_spec

LogFn = Callable[[str], None]

PROOF_STEPS = 2_048
WALK_ENGINES = ("playground", "mjlab", "isaaclab")
PROOF_REPORT_NAME = "proof_3c.json"


def n1b_operator_handoff() -> dict[str, Any]:
    """Locked A/B/C paths to close Phase 3c. Not a live train.

    CPU Cloud Agents surface this so operators know what to do next —
    permissions do not create a GPU.
    """
    return {
        "locked": True,
        "phase": "3c",
        "goal": "Live walking eval.mp4 + proof_3c.json from a real GPU engine",
        "paths": [
            {
                "id": "A",
                "title": "Local NVIDIA GPU box",
                "commands": [
                    "pip install -e .",
                    "pip install playground",
                    "ht proof walk --check",
                    "ht proof walk",
                ],
                "done_when": "proof_3c.json with live_clip:true and a walking eval.mp4",
            },
            {
                "id": "B",
                "title": "Self-hosted Cursor worker on a GPU machine",
                "commands": [
                    "cursor worker start",
                    "# start Cloud Agent on that worker → ht proof walk",
                ],
                "done_when": "Same as path A, harvested by the agent on the worker",
            },
            {
                "id": "C",
                "title": "No local GPU — Phase 3d remote harvest",
                "commands": [
                    "# add OSMO credentials, or HF_TOKEN + hf CLI",
                    "ht proof osmo   # or: ht proof hf-jobs",
                ],
                "done_when": "Remote eval.mp4 with facts.launch=osmo (or HF Jobs harvest)",
            },
        ],
        "refuse": [
            "Do not check in a stand clip as walk gold",
            "Do not mark Phase 3c done from CPU CI or fake-CLI unit tests",
            "Do not open B6 / more Hub pins until N1b lands",
        ],
    }


def walk_engines_ready() -> dict[str, bool]:
    """Engines that can run a *local* Phase 3c GPU walk proof.

    OSMO harvest is Phase 3d — it must not make Phase 3c assess ok:true.
    """
    return {
        "playground": hardware.playground_ready(),
        "mjlab": hardware.mjlab_ready(),
        "isaaclab": hardware.isaac_local_ready(),
    }


def first_ready_engine(prefer: list[str] | None = None) -> str | None:
    order = list(prefer or WALK_ENGINES)
    ready = walk_engines_ready()
    for name in order:
        if ready.get(name):
            return name
    return None


def proof_command(*, prefer: list[str] | None = None, engine: str | None = None) -> str:
    """CLI a beginner/operator can paste on a GPU box."""
    chosen = engine or (prefer[0] if prefer else None)
    if chosen and chosen in WALK_ENGINES and chosen != "playground":
        return f"ht proof walk --prefer {chosen}"
    if prefer and prefer != list(WALK_ENGINES):
        return "ht proof walk --prefer " + " ".join(prefer)
    return "ht proof walk"


def _blocked_message(ready: dict[str, bool]) -> str:
    status = hardware.engine_status()
    lines = [
        "Phase 3c walk proof is blocked on this machine — not a silent failure.",
        "Need an NVIDIA GPU and one walk engine:",
        f"  gpu={status.get('gpu')} playground_ready={ready['playground']} "
        f"mjlab_ready={ready['mjlab']} isaac_local_ready={ready['isaaclab']} "
        f"osmo_ready={status.get('osmo_ready')} (OSMO is Phase 3d, not 3c)",
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


def assess_walk_proof_host(*, prefer: list[str] | None = None) -> dict[str, Any]:
    """Phase 3c preflight: can this host run `ht proof walk`? Does not train.

    Requires a local GPU walk engine. OSMO-only readiness is Phase 3d
    (`ht proof osmo`) — it must not report Phase 3c ok:true.
    """
    ready = walk_engines_ready()
    status = hardware.engine_status()
    engine = first_ready_engine(prefer)
    reasons: list[str] = []
    if not status.get("gpu"):
        reasons.append("No NVIDIA GPU detected on this machine.")
    if not any(ready.values()):
        reasons.append(
            "No local walk engine ready (Playground, mjlab, or local Isaac Lab)."
        )
        if status.get("osmo_ready"):
            reasons.append(
                "OSMO is ready for Phase 3d harvest (ht proof osmo) — not Phase 3c local proof."
            )
    elif engine is None and prefer:
        reasons.append(
            "Preferred engine(s) not ready: " + ", ".join(prefer) + "."
        )
    # Phase 3c is a local GPU proof — never ok without GPU + a local engine.
    ok = bool(status.get("gpu") and engine is not None)
    cmd = proof_command(prefer=prefer, engine=engine)
    handoff = None if ok else n1b_operator_handoff()
    return {
        "ok": ok,
        "phase": "3c",
        "engine": engine if ok else None,
        "ready": ready,
        "engines": status,
        "command": cmd,
        "reasons": reasons,
        "next_step": None if ok else _blocked_message(ready),
        "error": None if ok else _blocked_message(ready),
        "handoff": handoff,
        "live_clip": False,
        "note": (
            "Host can launch a short local walk proof."
            if ok
            else "Host cannot complete Phase 3c here — use handoff path A/B (GPU) or C (OSMO/HF)."
        ),
    }


def write_proof_report(report: dict[str, Any], run_dir: Path | None) -> Path | None:
    """Persist the Phase 3c judge report next to the run artifacts."""
    if run_dir is None:
        return None
    path = Path(run_dir)
    if not path.is_dir():
        return None
    dest = path / PROOF_REPORT_NAME
    payload = {**report, "report_file": PROOF_REPORT_NAME}
    dest.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return dest


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
    host = assess_walk_proof_host(prefer=prefer)
    engine = host.get("engine")
    if not host.get("ok") or not engine:
        raise AdapterUnavailable(host.get("error") or _blocked_message(walk_engines_ready()))
    if log:
        log(f"proof walk via {engine} steps={steps}")
    spec = proof_walk_spec(prefer=prefer, steps=steps, engine=str(engine))
    manifest = run_job(spec, runs_dir=runs_dir or default_runs_dir(), log=log)
    report = summarize_walk_proof(manifest, expected_engine=str(engine))
    run_dir = Path(manifest.get("run_dir") or "")
    written = write_proof_report(report, run_dir if run_dir.is_dir() else None)
    if written is not None:
        report["report_path"] = str(written)
        if log:
            log(f"wrote {written.name}")
    if not report["ok"]:
        raise AdapterUnavailable(report["error"] or "Walk proof failed.")
    return report


def summarize_walk_proof(manifest: dict[str, Any], *, expected_engine: str | None = None) -> dict[str, Any]:
    """Judge a completed walk run for the Phase 3c exit test."""
    from humanoid_training.artifacts import looks_like_g1_stand_gold, resolve_eval_mp4

    run_dir = Path(manifest.get("run_dir") or "")
    facts = dict(manifest.get("facts") or {})
    status = str(manifest.get("status") or "")
    recipe = str(manifest.get("recipe") or "")
    video = resolve_eval_mp4(manifest)
    engine = str(facts.get("engine") or "")
    errors: list[str] = []
    if recipe and recipe != "g1-walk":
        errors.append(f"recipe={recipe!r} (want g1-walk). Do not proof a stand or mustard run.")
    elif not recipe:
        errors.append("recipe missing (want g1-walk)")
    if status != "passed":
        errors.append(f"status={status!r} (want passed). {manifest.get('error') or ''}".strip())
    if video is None:
        errors.append("missing eval.mp4 — not a walk proof. Do not substitute a stand clip.")
    elif video.stat().st_size < 8:
        errors.append("eval.mp4 is empty or tiny — not a walk proof.")
    elif looks_like_g1_stand_gold(video):
        errors.append(
            "eval.mp4 matches recipes/g1-stand/gold — refuse stand clip as walk proof."
        )
    if engine not in WALK_ENGINES:
        errors.append(f"facts.engine={engine!r} (want playground|mjlab|isaaclab)")
    if expected_engine and engine and engine != expected_engine:
        errors.append(f"facts.engine={engine!r} but proof targeted {expected_engine!r}")
    if facts.get("kind") != "rl":
        errors.append(f"facts.kind={facts.get('kind')!r} (want rl)")
    ok = not errors
    return {
        "ok": ok,
        "phase": "3c",
        "run_id": manifest.get("run_id"),
        "run_dir": str(run_dir) if run_dir else None,
        "status": status,
        "engine": engine or None,
        "video": str(video) if video is not None else None,
        "facts": facts,
        "error": "; ".join(errors) if errors else None,
        "live_clip": bool(ok and video is not None),
    }
