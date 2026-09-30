from __future__ import annotations

import json
from pathlib import Path

from humanoid_training.adapters import select_adapter
from humanoid_training.cli import main
from humanoid_training.datasets import resolve_hub_motion
from humanoid_training.hub_pins import curated_hub_pins, public_hub_pins
from humanoid_training.recipes import expand_spec
from humanoid_training.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def test_curated_hub_pins_honest() -> None:
    pins = curated_hub_pins()
    ids = {p["id"] for p in pins}
    assert "lafan1-g1-csv" in ids
    assert "lafan1-g1-npz" in ids
    assert "groot-lerobot-path" in ids
    assert "lerobot-pusht" in ids
    for pin in pins:
        assert pin.get("trainable_here") is False
    pub = public_hub_pins()
    assert "lafan1-g1-csv" in pub["motion"]
    assert "groot-lerobot-path" in pub["vla"]
    assert "LeRobot" in pub["note"] or "motion" in pub["note"].lower()


def test_cli_datasets_pins(capsys) -> None:
    assert main(["datasets", "pins"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert any(p["id"] == "lafan1-g1-csv" for p in out["pins"])


def test_g1_track_notes_mention_lafan1() -> None:
    spec = expand_spec(load_spec(ROOT / "spec" / "examples" / "g1-track.json"))
    assert (spec.get("data") or {}).get("hub_motion")
    adapter = select_adapter(spec)
    payload = adapter.compile(spec)
    blob = " ".join(payload.notes)
    assert "LAFAN1" in blob
    assert "csv_to_npz" in blob or "HT_MJLAB_MOTION" in blob


def test_resolve_hub_motion_rejects_non_hub() -> None:
    out = resolve_hub_motion("/tmp/not-a-hub")
    assert out["ok"] is False


def test_groot_wrap_is_lerobot() -> None:
    from humanoid_training.readiness import assess_groot_host

    report = assess_groot_host()
    assert report["wrap"] == "lerobot"
    assert report["ok"] is False
    assert "LeRobot" in (report.get("next_step") or "") or "LeRobot" in (report.get("note") or "")
