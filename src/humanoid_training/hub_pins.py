"""Curated Hugging Face Hub pins — not a new dataset format.

These are operator shortcuts so the studio can name real upstream artifacts
(LAFAN1 G1 motions, example LeRobot demos) without claiming we own conversion
or that a Hub id is train-ready on this laptop.
"""

from __future__ import annotations

from typing import Any


# Honest format tags. Studio must not treat motion pins as LeRobot ACT inputs.
FORMAT_LEROBOT = "lerobot_v2"
FORMAT_MOTION_CSV = "unitree_retarget_csv"
FORMAT_MOTION_NPZ = "beyondmimic_npz"
FORMAT_UPSTREAM_COURSE = "upstream_course"


def curated_hub_pins() -> list[dict[str, Any]]:
    """Stable list for API / CLI / Data room. Keep ids short and unique."""
    return [
        {
            "id": "lafan1-g1-csv",
            "title": "LAFAN1 → Unitree G1 (CSV retarget)",
            "uri": "hf:lvhaidong/LAFAN1_Retargeting_Dataset",
            "repo_id": "lvhaidong/LAFAN1_Retargeting_Dataset",
            "include": "g1/**",
            "format": FORMAT_MOTION_CSV,
            "kind": "motion",
            "recipe_hint": "g1-track",
            "paper": "BeyondMimic / Unitree LAFAN1 retarget",
            "trainable_here": False,
            "summary": (
                "G1-retargeted LAFAN1 CSVs (30 FPS). Not LeRobot. Convert with "
                "mjlab.scripts.csv_to_npz, upload to WandB registry, then set "
                "HT_MJLAB_MOTION for recipe g1-track."
            ),
            "next_step": (
                "1) hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset "
                "--include 'g1/**' --local-dir data/LAFAN1_g1\n"
                "2) MUJOCO_GL=egl python -m mjlab.scripts.csv_to_npz "
                "--input-file <csv> --output-name <name> --input-fps 30 --output-fps 50\n"
                "3) Set HT_MJLAB_MOTION=<org>/motions/<name> and Train g1-track on a GPU."
            ),
        },
        {
            "id": "lafan1-g1-npz",
            "title": "LAFAN1 G1 BeyondMimic NPZ (Isaac / unitree_rl_lab)",
            "uri": "hf:wty-yy/LAFAN1_g1",
            "repo_id": "wty-yy/LAFAN1_g1",
            "include": None,
            "format": FORMAT_MOTION_NPZ,
            "kind": "motion",
            "recipe_hint": "g1-track",
            "paper": "BeyondMimic (Isaac Mimic path)",
            "trainable_here": False,
            "summary": (
                "Preconverted NPZ for Isaac / unitree_rl_lab Mimic. Body order may "
                "differ from mjlab — use mjlab's own csv_to_npz for g1-track, "
                "not these NPZs, unless you stay on the Isaac BeyondMimic stack."
            ),
            "next_step": (
                "For mjlab Tracking (g1-track): prefer the CSV pin + mjlab.scripts.csv_to_npz.\n"
                "For Isaac BeyondMimic / unitree_rl_lab: follow that repo's train.py with these NPZs."
            ),
        },
        {
            "id": "lerobot-pusht",
            "title": "LeRobot PushT (format example)",
            "uri": "hf:lerobot/pusht",
            "repo_id": "lerobot/pusht",
            "include": None,
            "format": FORMAT_LEROBOT,
            "kind": "imitation",
            "recipe_hint": "pick-and-place",
            "paper": "ACT / LeRobot",
            "trainable_here": False,
            "summary": (
                "Public LeRobot v2 layout used to verify Hub inspect/cache. Not a G1 "
                "mustard demo — record local demos or point at your own hf:user/dataset."
            ),
            "next_step": (
                "Inspect with hf:lerobot/pusht to confirm Hub cache works. "
                "For mustard→bowl Train, record local demos or use your Hub LeRobot id."
            ),
        },
        {
            "id": "groot-lerobot-path",
            "title": "GR00T N1 via LeRobot (NVIDIA course)",
            "uri": None,
            "repo_id": None,
            "include": None,
            "format": FORMAT_UPSTREAM_COURSE,
            "kind": "vla",
            "recipe_hint": "pick-and-place",
            "paper": "NVIDIA GR00T N1 / Isaac Lab-Arena",
            "trainable_here": False,
            "summary": (
                "Humanoid Training does not train GR00T. The honest wrap is: Arena "
                "teleop → LeRobot dataset → NVIDIA GR00T post-training. Same LeRobot "
                "layout we already inspect; fine-tune stays on NVIDIA's course."
            ),
            "next_step": (
                "On a GPU box with Isaac Lab + LeRobot: ht proof groot\n"
                "Then follow NVIDIA GR00T G1 apple→plate / Arena docs. "
                "CPU studio keeps pick-and-place linear-BC; ACT when lerobot[training]+GPU."
            ),
        },
    ]


def pins_by_kind(kind: str | None = None) -> list[dict[str, Any]]:
    pins = curated_hub_pins()
    if not kind:
        return pins
    want = str(kind).strip().lower()
    return [p for p in pins if str(p.get("kind") or "").lower() == want]


def public_hub_pins() -> dict[str, Any]:
    """Studio / API projection."""
    pins = curated_hub_pins()
    return {
        "pins": pins,
        "motion": [p["id"] for p in pins if p.get("kind") == "motion"],
        "imitation": [p["id"] for p in pins if p.get("kind") == "imitation"],
        "vla": [p["id"] for p in pins if p.get("kind") == "vla"],
        "note": (
            "Curated Hub shortcuts only. Motion pins are not LeRobot ACT inputs. "
            "Conversion and WandB registry stay upstream (mjlab / BeyondMimic / NVIDIA)."
        ),
    }
