"""Phase 4 hardware deploy gate.

Policies stay sim-only until a hardware eval profile passes. This module
fails closed on a CPU laptop (and on any host without a live walk proof +
hardware profile). It does not drive a robot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from humanoid_training.errors import AdapterUnavailable
from humanoid_training.runner import default_runs_dir, load_manifest


def hardware_profile_present() -> bool:
    """True only when an operator-supplied hardware eval profile exists.

    Set HT_HARDWARE_PROFILE to a JSON file that marks a passed hardware
    eval. Absent / invalid → sim-only. Never invent a pass.
    """
    import os

    raw = os.environ.get("HT_HARDWARE_PROFILE")
    if raw is None or not str(raw).strip() or str(raw).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    path = Path(str(raw).strip()).expanduser()
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    return bool(data.get("passed")) and str(data.get("kind") or "") == "hardware"


def _blocked_message(run_id: str, *, reason: str) -> str:
    return "\n".join(
        [
            "Hardware deploy is blocked — not a silent robot start.",
            f"  run={run_id}",
            f"  reason={reason}",
            "",
            "Phase 4 exit test needs:",
            "  1. A live G1 walk clip (ht proof walk on a GPU box)",
            "  2. A hardware eval profile that passed (HT_HARDWARE_PROFILE)",
            "  3. Reduced-speed control with NaN / pose-limit kill switches",
            "",
            "Until then every run stays facts.sim_only=true.",
            "Do not deploy a stand clip or mustard clip to a humanoid.",
        ]
    )


def assess_deploy(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return ok/blocked assessment. Never launches actuators."""
    from humanoid_training.artifacts import looks_like_g1_stand_gold, resolve_eval_mp4

    run_id = str(manifest.get("run_id") or "")
    facts = dict(manifest.get("facts") or {})
    recipe = str(manifest.get("recipe") or "")
    status = str(manifest.get("status") or "")
    reasons: list[str] = []

    if status not in {"passed", "completed"}:
        reasons.append(f"run status is {status!r}, not a passed sim eval")
    elif status == "completed" and not (manifest.get("metrics") or {}).get("passed"):
        reasons.append("sim metrics.passed is false")

    if recipe != "g1-walk":
        reasons.append(
            f"recipe={recipe!r} — only a passed g1-walk is eligible for hardware "
            "(stand / mustard / cartpole stay sim-only)"
        )
    else:
        if facts.get("kind") != "rl":
            reasons.append(f"facts.kind={facts.get('kind')!r} — need a walk RL eval")
        if not facts.get("engine"):
            reasons.append("facts.engine missing — walk must stamp the engine that ran")
        video = resolve_eval_mp4(manifest)
        if video is None:
            reasons.append("no eval.mp4 on disk — refuse to deploy without a sim video")
        elif video.stat().st_size < 8:
            reasons.append("eval.mp4 is empty or tiny — refuse to deploy")
        elif looks_like_g1_stand_gold(video):
            reasons.append(
                "eval.mp4 matches g1-stand gold — refuse stand clip as walk deploy"
            )
        if not hardware_profile_present():
            reasons.append("no passed HT_HARDWARE_PROFILE (policies stay sim-only)")

    ok = not reasons
    return {
        "ok": ok,
        "run_id": run_id,
        "recipe": recipe,
        # Cleared only after a real hardware eval succeeds — not wired yet.
        "sim_only": True,
        "reasons": reasons,
        "error": None if ok else _blocked_message(run_id, reason="; ".join(reasons)),
    }


def deploy_run(run_id: str, *, runs_dir: Path | None = None) -> dict[str, Any]:
    """Fail-closed deploy entrypoint. Raises AdapterUnavailable until Phase 4 is real."""
    root = Path(runs_dir or default_runs_dir())
    run_dir = root / run_id
    if not run_dir.is_dir():
        raise AdapterUnavailable(
            f"Unknown run {run_id!r} under {root}. Train a g1-walk first, then "
            "ht proof walk on a GPU box before any hardware attempt."
        )
    manifest = load_manifest(run_dir)
    manifest.setdefault("run_dir", str(run_dir))
    report = assess_deploy(manifest)
    if not report["ok"]:
        raise AdapterUnavailable(report["error"] or "Deploy blocked.")
    # Even with a profile present, this repo does not yet drive Unitree hardware.
    raise AdapterUnavailable(
        "Hardware profile is present, but Unitree G1 deploy is not wired in this "
        "repo yet. NaN / pose-limit kill switches and reduced-speed control must "
        "ship before any live torque. Run stays sim-only."
    )
