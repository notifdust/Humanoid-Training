from __future__ import annotations

from pathlib import Path

from humanoid_training.adapters import select_adapter
from humanoid_training.adapters.isaaclab import IsaacLabAdapter, WORKFLOW_ROBOMIMIC
from humanoid_training.errors import AdapterUnavailable
from humanoid_training.recipes import expand_spec, list_recipes
from humanoid_training.spec import load_spec


def test_g1_pickplace_in_catalog() -> None:
    ids = {r.id for r in list_recipes()}
    assert "g1-pickplace" in ids
    assert "g1-reach" not in ids
    pub = next(r.as_public_dict() for r in list_recipes() if r.id == "g1-pickplace")
    assert pub["availability"] == "gpu"
    assert pub["imitate"] is False
    assert pub["has_gold"] is False
    assert pub["method"] == "imitation"


def test_g1_pickplace_pins_real_isaac_task() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "recipes" / "g1-pickplace" / "recipe.yaml").read_text(encoding="utf-8"))
    isaac = (data.get("adapters") or {}).get("isaaclab") or {}
    assert isaac.get("task") == "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
    assert isaac.get("workflow") == "robomimic"
    assert not (root / "recipes" / "g1-pickplace" / "gold" / "eval.mp4").is_file()


def test_g1_pickplace_compiles_robomimic_not_rsl_rl() -> None:
    spec = expand_spec(load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-pickplace.json"))
    adapter = select_adapter(spec)
    assert adapter.name == "isaaclab"
    payload = adapter.compile(spec)
    assert payload.env_name == "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
    assert payload.extra.get("workflow") == WORKFLOW_ROBOMIMIC
    assert "robomimic/train.py" in " ".join(str(c) for c in payload.command)
    assert "rsl_rl" not in " ".join(str(c) for c in payload.command)
    assert "train_isaac.sh" in payload.files
    assert "osmo_workflow.yaml" not in payload.files
    assert "Robomimic" in " ".join(payload.notes)
    sh = payload.files["train_isaac.sh"]
    assert "HT_ISAAC_DATASET" in sh
    assert "robomimic/train.py" in sh


def test_g1_pickplace_launch_blocks_without_dataset(tmp_path: Path) -> None:
    spec = expand_spec(load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-pickplace.json"))
    adapter = IsaacLabAdapter()
    payload = adapter.compile(spec)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name, body in payload.files.items():
        (run_dir / name).write_text(body, encoding="utf-8")
    try:
        adapter.launch(spec, payload, run_dir, log=lambda _m: None)
    except AdapterUnavailable as err:
        assert "Robomimic" in str(err) or "HT_ISAAC_DATASET" in str(err)
        assert "PickPlace" in str(err) or "pickplace" in str(err).lower() or "dataset" in str(err).lower()
    else:
        raise AssertionError("expected AdapterUnavailable without dataset")
