"""English summaries from run facts — beginner-facing, not TensorBoard.

Studio projects `run.english` from the API (this module). Keep JS as a
fallback only when the field is missing.
"""

from __future__ import annotations

from typing import Any


def english_from_facts(
    facts: dict[str, Any] | None,
    *,
    status: str | None = None,
    error: str | None = None,
) -> str:
    """One short sentence a beginner can act on."""
    facts = dict(facts or {})
    kind = str(facts.get("kind") or "")
    engine = str(facts.get("engine") or "")
    policy = str(facts.get("policy") or "")
    device = str(facts.get("device") or "")
    st = str(status or "").lower()

    if st in {"blocked", "queued"} and error:
        first = str(error).strip().splitlines()[0]
        return first[:200] if first else "Can't train here — see the next step."

    if st == "failed" or st == "error":
        if error:
            first = str(error).strip().splitlines()[0]
            return f"Run broke: {first[:180]}"
        return "Run finished but did not pass."

    if facts.get("video") == "missing":
        return (
            "Engine finished but wrote no eval.mp4 — not a headless skip, "
            "not a stand substitute."
        )

    sim = ""
    if facts.get("sim_only") is True:
        sim = " Still sim-only — not cleared for hardware."

    if kind == "hold":
        return f"Held a pinned pose{f' via {engine}' if engine else ''} — not walking.{sim}".strip()

    if kind == "scene_preview":
        return f"Scene preview{f' via {engine}' if engine else ''} — not a trained policy.{sim}".strip()

    if kind == "rl":
        bit = f"{engine or 'engine'} RL"
        if policy:
            bit += f" ({policy})"
        if device == "remote":
            bit += " on remote GPU"
        elif device == "gpu":
            bit += " on GPU"
        return f"Locomotion / RL rollout from {bit}.{sim}".strip()

    if kind == "imitation":
        if policy in {"act", "linear-bc", "bc"}:
            label = {"act": "ACT", "linear-bc": "linear-BC", "bc": "BC"}.get(policy, policy)
            where = engine or "local"
            return f"Imitation ({label}) via {where} — demos, not finger grasping.{sim}".strip()
        return f"Imitation run{f' via {engine}' if engine else ''}.{sim}".strip()

    if kind == "gym" or engine == "gymnasium":
        return f"Gymnasium rollout{f' ({policy})' if policy else ''}.{sim}".strip()

    if st == "passed" or st == "completed":
        return f"Finished{f' on {engine}' if engine else ''}.{sim}".strip()

    if error:
        return str(error).strip().splitlines()[0][:200]

    return "No English summary for these facts yet."


def english_for_manifest(manifest: dict[str, Any] | None) -> str:
    """Project English from a run manifest (API / CLI)."""
    m = dict(manifest or {})
    return english_from_facts(
        m.get("facts") if isinstance(m.get("facts"), dict) else {},
        status=str(m.get("status") or ""),
        error=str(m.get("error") or "") or None,
    )
