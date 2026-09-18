from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from humanoid_training.errors import SpecError, repo_root

SPEC_VERSION = "0.1.0"


def schema_path() -> Path:
    return repo_root() / "spec" / "job_spec.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def load_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SpecError(f"{path} must contain a JSON/YAML object")
    return data


def validate_spec(spec: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors (empty means structurally valid)."""
    validator = Draft202012Validator(load_schema())
    errors = sorted(validator.iter_errors(spec), key=lambda e: list(e.path))
    messages = []
    for err in errors:
        loc = ".".join(str(p) for p in err.path) or "<root>"
        messages.append(f"{loc}: {err.message}")
    return messages


def require_valid(spec: dict[str, Any]) -> dict[str, Any]:
    messages = validate_spec(spec)
    if messages:
        raise SpecError("Invalid job spec:\n" + "\n".join(f"  - {m}" for m in messages))
    return spec


def canonical_dumps(spec: dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_dumps(spec).encode("utf-8")).hexdigest()[:16]


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge; overlay wins. Lists are replaced, not concatenated."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
