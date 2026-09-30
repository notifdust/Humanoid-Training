from __future__ import annotations

from pathlib import Path

from humanoid_training.adapters import select_adapter
from humanoid_training.cli import main
from humanoid_training.recipes import expand_spec, list_recipes, public_catalog
from humanoid_training.readiness import assess_act_host, assess_all, assess_osmo_host
from humanoid_training.spec import load_spec


def test_assess_act_and_osmo_blocked_on_cpu() -> None:
    act = assess_act_host()
    assert act["ok"] is False
    assert act["phase"] == "3e"
    assert act["reasons"]
    osmo = assess_osmo_host()
    assert osmo["ok"] is False
    assert osmo["phase"] == "3d"
    bundle = assess_all()
    assert bundle["walk"]["phase"] == "3c"
    assert bundle["any_ok"] is False


def test_cli_proof_act_osmo_check(capsys) -> None:
    assert main(["proof", "act"]) == 12
    act = __import__("json").loads(capsys.readouterr().out)
    assert act["ok"] is False
    assert act["phase"] == "3e"
    assert main(["proof", "osmo"]) == 12
    osmo = __import__("json").loads(capsys.readouterr().out)
    assert osmo["ok"] is False
    assert osmo["phase"] == "3d"


def test_g1_walk_rough_pins_upstream() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "recipes" / "g1-walk-rough" / "recipe.yaml").read_text(encoding="utf-8"))
    ad = data.get("adapters") or {}
    assert (ad.get("mjlab") or {}).get("task") == "Mjlab-Velocity-Rough-Unitree-G1"
    assert (ad.get("isaaclab") or {}).get("task") == "Isaac-Velocity-Rough-G1-v0"
    assert not (root / "recipes" / "g1-walk-rough" / "gold" / "eval.mp4").is_file()
    ids = {r.id for r in list_recipes()}
    assert "g1-walk-rough" in ids
    pub = next(r.as_public_dict() for r in list_recipes() if r.id == "g1-walk-rough")
    assert pub["availability"] == "gpu"
    assert pub["has_gold"] is False


def test_g1_walk_rough_compiles_mjlab_or_isaac() -> None:
    spec = expand_spec(
        load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk-rough.json")
    )
    adapter = select_adapter(spec)
    assert adapter.name in {"mjlab", "isaaclab"}
    payload = adapter.compile(spec)
    assert payload.env_name in {
        "Mjlab-Velocity-Rough-Unitree-G1",
        "Isaac-Velocity-Rough-G1-v0",
    }


def test_catalog_later_includes_rough() -> None:
    catalog = public_catalog()
    assert "g1-walk-rough" in catalog["later"]
    assert "g1-walk" in catalog["later"]
