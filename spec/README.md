# Job spec

The studio's source of truth. See [ARCHITECTURE.md](../docs/ARCHITECTURE.md).

- `job_spec.schema.json` — draft JSON Schema (`spec_version` 0.1.x)
- `examples/` — documents a UI would write

Unknown fields are allowed. Adapters must ignore what they cannot honor
and record that in the run manifest.

This schema is not frozen. Change it in breaking ways only with a
`spec_version` bump and an adapter note.
