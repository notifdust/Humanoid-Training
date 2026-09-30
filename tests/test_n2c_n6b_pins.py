from __future__ import annotations

import json
from pathlib import Path

import yaml

from humanoid_training.adapters import select_adapter
from humanoid_training.cli import main
from humanoid_training.datasets import parse_hub_dataset_uri
from humanoid_training.recipes import expand_spec, list_recipes, public_catalog
from humanoid_training.readiness import (
    assess_all,
    assess_groot_host,
    assess_hf_jobs_host,
)
from humanoid_training.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def test_assess_hf_jobs_and_groot_blocked_on_cpu() -> None:
    hf = assess_hf_jobs_host()
    assert hf["ok"] is False
    assert hf["phase"] == "3d-hf"
    assert hf["live_harvest"] is False
    assert hf["reasons"]
    groot = assess_groot_host()
    assert groot["ok"] is False
    assert groot["phase"] == "groot"
    assert groot["live_clip"] is False
    bundle = assess_all()
    assert "hf_jobs" in bundle and "groot" in bundle
    assert bundle["any_ok"] is False


def test_cli_proof_hf_jobs_groot(capsys) -> None:
    assert main(["proof", "hf-jobs"]) == 12
    hf = json.loads(capsys.readouterr().out)
    assert hf["ok"] is False
    assert hf["phase"] == "3d-hf"
    assert main(["proof", "groot"]) == 12
    groot = json.loads(capsys.readouterr().out)
    assert groot["ok"] is False
    assert groot["phase"] == "groot"


def test_g1_track_and_fixed_pickplace_pins() -> None:
    track = yaml.safe_load((ROOT / "recipes" / "g1-track" / "recipe.yaml").read_text())
    assert (track.get("adapters") or {}).get("mjlab", {}).get("task") == (
        "Mjlab-Tracking-Flat-Unitree-G1"
    )
    assert (track.get("adapters") or {}).get("mjlab", {}).get("requires_motion") is True
    fixed = yaml.safe_load(
        (ROOT / "recipes" / "g1-pickplace-fixed" / "recipe.yaml").read_text()
    )
    assert (fixed.get("adapters") or {}).get("isaaclab", {}).get("task") == (
        "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
    )
    assert (fixed.get("adapters") or {}).get("isaaclab", {}).get("workflow") == "robomimic"
    ids = {r.id for r in list_recipes()}
    assert "g1-track" in ids
    assert "g1-pickplace-fixed" in ids
    catalog = public_catalog()
    assert "g1-track" in catalog["later"]
    assert "g1-pickplace-fixed" in catalog["later"]


def test_g1_track_compiles_mjlab_notes_motion() -> None:
    spec = expand_spec(load_spec(ROOT / "spec" / "examples" / "g1-track.json"))
    adapter = select_adapter(spec)
    assert adapter.name == "mjlab"
    payload = adapter.compile(spec)
    assert payload.env_name == "Mjlab-Tracking-Flat-Unitree-G1"
    assert any("HT_MJLAB_MOTION" in n or "Motion" in n for n in payload.notes)


def test_g1_track_blocked_without_motion(monkeypatch) -> None:
    monkeypatch.delenv("HT_MJLAB_MOTION", raising=False)
    monkeypatch.delenv("WANDB_MOTION_REGISTRY", raising=False)
    monkeypatch.setenv("HT_GPU", "1")
    monkeypatch.setenv("HT_MJLAB_CLI", "python")
    pub = next(r.as_public_dict() for r in list_recipes() if r.id == "g1-track")
    assert pub["launch_here"] is False
    assert pub["launch_path"] == "blocked"


def test_parse_hub_dataset_uri() -> None:
    assert parse_hub_dataset_uri("hf:lerobot/pusht") == ("lerobot/pusht", None)
    assert parse_hub_dataset_uri("https://huggingface.co/datasets/org/name") == (
        "org/name",
        None,
    )
    assert parse_hub_dataset_uri("/tmp/local") is None


def test_inspect_hub_uri_does_not_claim_local_layout() -> None:
    from humanoid_training.datasets import inspect_lerobot_dataset

    # Without a successful download, Hub URIs must fail closed with an actionable error.
    out = inspect_lerobot_dataset("hf:humanoid-training/does-not-exist-xyz")
    assert out["ok"] is False
    err = (out.get("error") or "").lower()
    assert "hub" in err or "huggingface" in err or "download" in err
