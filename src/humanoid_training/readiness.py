"""Host readiness preflights for Phase 3c/3d/3e (CPU-safe).

Live GPU / OSMO / ACT proofs remain the phase exit tests. These helpers
only answer: can this machine attempt that proof?
"""

from __future__ import annotations

from typing import Any

from humanoid_training import hardware
from humanoid_training.proof import assess_walk_proof_host


def assess_act_host() -> dict[str, Any]:
    """Phase 3e preflight: can this host launch LeRobot ACT? Does not train."""
    status = hardware.engine_status()
    ready = hardware.lerobot_ready()
    reasons: list[str] = []
    if not status.get("gpu"):
        reasons.append("No NVIDIA GPU detected on this machine.")
    if not status.get("lerobot_cli"):
        reasons.append("LeRobot CLI missing (pip install 'lerobot[training]').")
    elif not status.get("lerobot_ready"):
        reasons.append("LeRobot present but not ready (needs GPU).")
    cmd = "ht proof act"
    return {
        "ok": ready,
        "phase": "3e",
        "ready": {"lerobot": ready, "gpu": bool(status.get("gpu"))},
        "engines": status,
        "command": cmd if ready else "pip install 'lerobot[training]' && ht proof act",
        "reasons": reasons,
        "next_step": None
        if ready
        else "\n".join(
            [
                "Phase 3e ACT proof is blocked on this machine — not a silent failure.",
                "Need an NVIDIA GPU and LeRobot:",
                f"  gpu={status.get('gpu')} lerobot_ready={status.get('lerobot_ready')}",
                "",
                "On a GPU box:",
                "  pip install 'lerobot[training]'",
                "  ht train spec/examples/g1-mustard-in-bowl.json",
                "  # or: ht proof act",
                "",
                "CPU studio keeps linear-BC on pick-and-place.",
            ]
        ),
        "error": None if ready else "ACT proof blocked — need LeRobot + GPU.",
        "live_clip": False,
        "note": (
            "Host can launch LeRobot ACT on local demos."
            if ready
            else "Host cannot complete Phase 3e ACT here — use a GPU box."
        ),
    }


def assess_osmo_host() -> dict[str, Any]:
    """Phase 3d preflight: can this host submit→poll→rsync an OSMO walk?"""
    status = hardware.engine_status()
    ready = hardware.osmo_ready()
    reasons: list[str] = []
    if not status.get("osmo_cli"):
        reasons.append("OSMO CLI missing (install + log in to an OSMO pool).")
    elif not ready:
        reasons.append("OSMO CLI present but harvest not ready (check HT_OSMO_HARVEST).")
    return {
        "ok": ready,
        "phase": "3d",
        "ready": {"osmo": ready},
        "engines": status,
        "command": "ht train spec/examples/g1-walk.json"
        if ready
        else "osmo login  # then: ht train spec/examples/g1-walk.json",
        "reasons": reasons,
        "next_step": None
        if ready
        else "\n".join(
            [
                "Phase 3d OSMO harvest is blocked on this machine — not a silent failure.",
                f"  osmo_ready={status.get('osmo_ready')} osmo_cli={bool(status.get('osmo_cli'))}",
                "",
                "From a laptop without CUDA (with OSMO logged in):",
                "  ht train spec/examples/g1-walk.json",
                "Isaac prefer path submits osmo_workflow.yaml → poll → rsync ht_eval/*.mp4.",
                "",
                "Opt out: HT_OSMO_HARVEST=0",
            ]
        ),
        "error": None if ready else "OSMO harvest blocked — need osmo CLI + pool.",
        "live_clip": False,
        "note": (
            "Host can attempt OSMO submit→poll→rsync harvest for g1-walk."
            if ready
            else "Host cannot complete Phase 3d harvest here — need OSMO credentials + pool."
        ),
    }


def assess_all() -> dict[str, Any]:
    """Bundle walk / OSMO / ACT readiness for operators and health UIs."""
    walk = assess_walk_proof_host()
    osmo = assess_osmo_host()
    act = assess_act_host()
    return {
        "walk": walk,
        "osmo": osmo,
        "act": act,
        "any_ok": bool(walk.get("ok") or osmo.get("ok") or act.get("ok")),
    }
