from __future__ import annotations

from pathlib import Path

from humanoid_training.spec import load_spec, spec_hash, validate_spec


def test_examples_are_valid() -> None:
    root = Path(__file__).resolve().parents[1] / "spec" / "examples"
    for path in sorted(root.glob("*.json")):
        errors = validate_spec(load_spec(path))
        assert errors == [], path.name + ": " + "; ".join(errors)


def test_hash_is_stable() -> None:
    spec = load_spec(Path(__file__).resolve().parents[1] / "spec" / "examples" / "g1-walk.json")
    assert spec_hash(spec) == spec_hash(spec)
    assert len(spec_hash(spec)) == 16
