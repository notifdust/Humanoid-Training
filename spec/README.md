# Job spec

The studio’s source of truth. UI and CLI both expand a short document
against a recipe, then adapters compile it. See
[ARCHITECTURE.md](../docs/ARCHITECTURE.md).

| Path | Role |
|---|---|
| `job_spec.schema.json` | Draft JSON Schema (`spec_version` 0.1.x) |
| `examples/` | Specs a beginner UI / CLI would write |

Unknown fields are allowed. Adapters must ignore what they cannot honor
and record that in the run manifest (`facts` + notes).

This schema is not frozen. Breaking changes need a `spec_version` bump
and an adapter note.
