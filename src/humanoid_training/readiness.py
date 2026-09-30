"""Host readiness preflights for Phase 3c/3d/3e (+ HF Jobs / GR00T) — CPU-safe.

Live GPU / OSMO / ACT / HF Jobs proofs remain the phase exit tests. These
helpers only answer: can this machine attempt that proof?
"""

from __future__ import annotations

from typing import Any

from humanoid_training import hardware
from humanoid_training.proof import assess_walk_proof_host


def assess_act_host() -> dict[str, Any]:
    """Phase 3e preflight: can this host launch LeRobot ACT? Does not train.

    `ht proof act` is check-only — live ACT is `ht train` on pick-and-place.
    """
    status = hardware.engine_status()
    ready = hardware.lerobot_ready()
    reasons: list[str] = []
    if not status.get("gpu"):
        reasons.append("No NVIDIA GPU detected on this machine.")
    if not status.get("lerobot_cli"):
        reasons.append("LeRobot CLI missing (pip install 'lerobot[training]').")
    elif not status.get("lerobot_ready"):
        reasons.append("LeRobot present but not ready (needs GPU).")
    train_cmd = "ht train spec/examples/g1-mustard-in-bowl.json"
    return {
        "ok": ready,
        "phase": "3e",
        "ready": {"lerobot": ready, "gpu": bool(status.get("gpu"))},
        "engines": status,
        "command": train_cmd if ready else "pip install 'lerobot[training]' && " + train_cmd,
        "check_command": "ht proof act",
        "reasons": reasons,
        "next_step": None
        if ready
        else "\n".join(
            [
                "Phase 3e ACT readiness check failed — not a silent failure.",
                "Need an NVIDIA GPU and LeRobot:",
                f"  gpu={status.get('gpu')} lerobot_ready={status.get('lerobot_ready')}",
                "",
                "On a GPU box:",
                "  pip install 'lerobot[training]'",
                f"  {train_cmd}",
                "",
                "`ht proof act` only checks readiness — it does not train.",
                "CPU studio keeps linear-BC on pick-and-place.",
            ]
        ),
        "error": None if ready else "ACT readiness blocked — need LeRobot + GPU.",
        "live_clip": False,
        "note": (
            "Host can launch LeRobot ACT — run ht train on pick-and-place demos "
            "(ht proof act only checks readiness)."
            if ready
            else "Host cannot launch ACT here — use a GPU box. ht proof act is check-only."
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


def assess_hf_jobs_host() -> dict[str, Any]:
    """Phase 3d alternate: can this host talk to HF Jobs? Check-only.

    Live harvest (submit → poll → eval.mp4) is not wired yet — same honesty
    bar as `ht proof osmo` before a live pool proof.
    """
    status = hardware.engine_status()
    ready = hardware.hf_jobs_ready()
    reasons: list[str] = []
    if not status.get("hf_cli"):
        reasons.append("Hugging Face CLI missing (pip install huggingface_hub[cli]).")
    if not status.get("hf_token"):
        reasons.append("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) not set — run `hf auth login`.")
    if status.get("hf_cli") and status.get("hf_token") and not ready:
        reasons.append("HF Jobs opt-out via HT_HF_JOBS=0.")
    return {
        "ok": ready,
        "phase": "3d-hf",
        "ready": {
            "hf_cli": bool(status.get("hf_cli")),
            "hf_token": bool(status.get("hf_token")),
            "hf_jobs": ready,
        },
        "engines": status,
        "command": "hf jobs --help  # then wire harvest; OSMO remains the live Phase 3d path",
        "check_command": "ht proof hf-jobs",
        "reasons": reasons,
        "next_step": None
        if ready
        else "\n".join(
            [
                "HF Jobs readiness check failed — not a silent failure.",
                f"  hf_cli={bool(status.get('hf_cli'))} hf_token={bool(status.get('hf_token'))}",
                "",
                "On a laptop without CUDA:",
                "  pip install 'huggingface_hub[cli]'",
                "  hf auth login",
                "  # Live harvest (submit→poll→eval.mp4) is not wired yet.",
                "  # Use OSMO for Phase 3d today: ht proof osmo / ht train g1-walk",
                "",
                "Opt out: HT_HF_JOBS=0",
            ]
        ),
        "error": None if ready else "HF Jobs blocked — need hf CLI + HF_TOKEN.",
        "live_clip": False,
        "live_harvest": False,
        "note": (
            "Host has HF CLI + token — HF Jobs harvest adapter is still TODO "
            "(OSMO remains the live Phase 3d path)."
            if ready
            else "Host cannot use HF Jobs here — need huggingface CLI + token. Check-only."
        ),
    }


def assess_groot_host() -> dict[str, Any]:
    """GR00T / Isaac Lab-Arena preflight — check-only, does not fine-tune.

    NVIDIA's G1 apple→plate course owns the VLA stack. We only report whether
    this host looks ready to attempt that path.
    """
    hints = hardware.groot_stack_hint()
    status = hardware.engine_status()
    reasons: list[str] = []
    if not hints.get("gpu"):
        reasons.append("No NVIDIA GPU detected.")
    if not hints.get("isaac_local_ready"):
        reasons.append("Isaac Lab local path not ready (HT_ISAAC_CLI + GPU, or HT_DOCKER_GPU).")
    if not hints.get("lerobot_ready"):
        reasons.append("LeRobot not ready (GR00T datasets convert to LeRobot format).")
    # Intentionally never ok:true on CPU — and never claim a live fine-tune.
    ok = False
    if hints.get("gpu") and hints.get("isaac_local_ready") and hints.get("lerobot_ready"):
        # Stack looks present; still check-only (we do not own GR00T train).
        ok = True
    return {
        "ok": ok,
        "phase": "groot",
        "ready": hints,
        "engines": status,
        "command": (
            "Follow NVIDIA GR00T G1 course (Isaac Lab-Arena teleop → LeRobot → fine-tune). "
            "Humanoid Training does not train GR00T itself."
        ),
        "check_command": "ht proof groot",
        "wrap": "lerobot",
        "reasons": reasons,
        "next_step": None
        if ok
        else "\n".join(
            [
                "GR00T / Arena readiness check failed — not a silent failure.",
                f"  gpu={hints.get('gpu')} isaac_local_ready={hints.get('isaac_local_ready')} "
                f"lerobot_ready={hints.get('lerobot_ready')}",
                "",
                "Honest wrap (paper-aligned): Arena teleop → LeRobot dataset → NVIDIA GR00T "
                "post-training. We already inspect LeRobot v2; we do not invent a VLA policy.",
                "See ht datasets pins (groot-lerobot-path) and GET /api/datasets/pins.",
                "",
                "On a GPU box with Isaac Lab + LeRobot:",
                "  ht proof groot   # should report ok:true when stack is local-ready",
                "  # then follow NVIDIA docs for Arena teleop + GR00T 1.7 post-training",
            ]
        ),
        "error": None if ok else "GR00T path blocked — need GPU + local Isaac + LeRobot.",
        "live_clip": False,
        "note": (
            "Host looks ready for NVIDIA's GR00T/Arena G1 path via LeRobot wrap — "
            "ht proof groot only checks readiness; fine-tune stays upstream."
            if ok
            else (
                "Host cannot run GR00T/Arena here. CPU keeps pick-and-place (linear-BC); "
                "ACT when LeRobot+GPU; GR00T fine-tune stays on NVIDIA's course."
            )
        ),
    }


def assess_all() -> dict[str, Any]:
    """Bundle walk / OSMO / ACT / HF Jobs / GR00T readiness for operators."""
    walk = assess_walk_proof_host()
    osmo = assess_osmo_host()
    act = assess_act_host()
    hf_jobs = assess_hf_jobs_host()
    groot = assess_groot_host()
    return {
        "walk": walk,
        "osmo": osmo,
        "act": act,
        "hf_jobs": hf_jobs,
        "groot": groot,
        "any_ok": bool(
            walk.get("ok")
            or osmo.get("ok")
            or act.get("ok")
            or hf_jobs.get("ok")
            or groot.get("ok")
        ),
    }
