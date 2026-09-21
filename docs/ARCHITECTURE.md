# Architecture: spec, recipes, adapters, studio

This repo is a **compiler**, not a simulator. The missing product is a
studio that compiles a job into engines that already exist (MuJoCo,
Playground, mjlab, Isaac Lab, LeRobot).

```
Studio (browser)     projection of GET /api/recipes (public_catalog)
        │
        ▼
Job spec (JSON)      source of truth for one run
        │
        ├─ expand    recipe.yaml defaults, user overlay wins
        ├─ compile   adapter → engine payload files
        └─ launch    in-process by default; `ht train --docker` uses the CPU image
                     → eval.mp4 + boolean + manifest
```

If a change cannot be expressed as a spec field or a recipe field, it
does not belong in the UI.

---

## 1. Job spec

A run is a document. Schema: `spec/job_spec.schema.json`. Examples: `spec/examples/`.

```json
{
  "spec_version": "0.1.0",
  "name": "g1-mustard-in-bowl",
  "robot": { "id": "unitree-g1-29dof", "source": "catalog" },
  "scene": { "template": "kitchen-counter-v1", "objects": [] },
  "task": { "recipe": "pick-and-place", "success": { "type": "object-in-container" } },
  "data": { "datasets": [], "keep_episodes": [0, 1, 2] },
  "train": { "method": "imitation", "seed": 1 },
  "backend": { "prefer": ["mujoco"], "compute": "local" },
  "eval": { "episodes": 1, "record_video": true }
}
```

Rules:

- Recipes supply defaults. The beginner document is short. The runner
  executes the **expanded** spec and hashes it.
- User overlay wins (`expand_spec`).
- Unknown fields are preserved. Adapters that cannot honor a field must
  record it (`ignored_fields`) or fail closed.
- Success is a boolean plus a video when the machine can render.

---

## 2. Recipes are the studio contract

Each `recipes/<id>/recipe.yaml` is a versioned task package. It owns
**what the UI may say**, not just train hyperparameters.

```yaml
id: pick-and-place
runnable: true
studio:
  availability: cpu    # cpu | gpu  — home screen grouping
  start_here: true
  promise: Mustard slides into the bowl. The arm follows. Not finger grasping.
  train_hint: You will see the mustard move into the bowl…
  blocked_hint: ""     # GPU recipes only
  scene_hint: Top-down kitchen counter. Drag mustard if you want…
  record_hint: Record mustard-into-bowl takes, uncheck the bad ones, then Train.
```

`GET /api/recipes` returns `public_catalog()`:

```json
{
  "recipes": [/* as_public_dict() */],
  "ready": ["cartpole-balance", "g1-stand", "pick-and-place"],
  "later": ["g1-walk", "g1-reach"],
  "start_here": ["cartpole-balance", "g1-stand", "pick-and-place"]
}
```

The browser **projects** `availability`, `promise`, `train_hint`,
`blocked_hint`, `scene_hint`, and `record_hint`. It must not hardcode
recipe ids to decide what works on a laptop, which task to Train as a
fallback, or which imitation recipe to record.

`runnable` means an adapter can launch something. `availability: gpu`
means launch will **block** here with a next step (compile payloads,
no fake walk clip). `start_here` defaults to `availability == cpu` when
omitted.

Gold clips live in `recipes/<id>/gold/eval.mp4` for `availability: cpu`
recipes. CI’s xvfb job retrains and compares frames. GPU recipes must
not check in a success clip. The studio projects `has_gold` from that
file; see [ROADMAP.md](./ROADMAP.md) Phase 2.5.

---

## 3. Adapters

Implemented protocol (`src/humanoid_training/adapters/base.py`):

```
support(spec) -> Support          # can this adapter claim the spec?
compile(spec) -> EnginePayload    # files + command; never hide them
launch(spec, payload, run_dir) -> EvalResult   # boolean + optional video
```

`poll` / `eval` as separate RPCs are not built. Launch is in-process and
returns the eval. That is enough for CPU recipes. Phase 3a (Playground
walk) is the first job that needs the split: submit → stream `run.log`
→ harvest `eval.mp4`.

| Adapter | Role today |
|---|---|
| `gymnasium` | Cartpole RL + eval video |
| `mujoco` | G1 stand hold; pick-and-place linear-BC + arm poses |
| `playground` | Compile G1 walk; launch blocked until Phase 3a (even if Playground is installed) |
| `mjlab` | Compile reach/walk; launch blocked on CPU |
| `isaaclab` | Compile OSMO YAML; launch not wired |
| `lerobot` | Compile future ACT script; CPU imitation stays on mujoco |

The MuJoCo adapter is three modules, not one god file:

- `mujoco_adapter.py` — `support` / `compile` / `launch` (hold vs imitation)
- `mujoco_runtime.py` — simulate, renderer, pin, body ids
- `mujoco_control.py` — open-loop poses, qpos drive, Jacobian IK

Selection: `backend.prefer` order, skip `unsupported`, first `support.ok`.
The runner also compiles **other** supporting adapters into
`engines/<name>/` so a researcher can take the payload to a GPU box.

Adapters must never invent a dataset format, hide generated files, or
claim success without the spec's eval.

---

## 4. Runner

`run_job` is the only orchestrator today: expand → compile → launch →
manifest. It is in-process by default. `ht train --docker` runs that
same loop inside the CPU image (`Dockerfile`). Missing Docker fails
closed; Cartpole still trains without it.

`src/humanoid_training/artifacts.py` is the inventory of files a run
may write (`RUN_ARTIFACTS`) and the names the studio HTTP API may
stream (`SERVED_ARTIFACTS`). `collect_artifacts` and
`GET /api/runs/.../artifacts/{name}` share that list.

`EvalResult.facts` is the machine-readable eval. The runner copies it
onto `manifest.json` as `facts`. Notes stay English for humans. The
studio **projects** `facts` (arm_mode, bc_steps, keep_episodes, kind)
and only scrapes notes for older runs that have none.

Every run directory contains:

| File | Meaning |
|---|---|
| `spec.json` | expanded public spec |
| `manifest.json` | status, metrics, facts, notes, artifact paths |
| `run.log` | line log (studio SSE tails this) |
| `eval.mp4` | when GLFW/display can render |
| `checkpoint.npz` | policy weights when the adapter trains |
| `composed_scene.xml` | MjSpec export when a scene was compiled |
| `train_returns.json` | gymnasium training curve when present |

Status:

- `passed` — eval boolean true
- `completed` — ran, boolean false
- `blocked` — adapter unavailable (expected on CPU for walk/reach)
- `compiled` — `--compile-only`
- `failed` — unexpected exception

The CPU Docker image is Phase 1 (`ht train --docker`). It remounts only
`/runs`, sets `HT_RUN_ID`, and runs in-process train inside the container
(never `--docker`, or it would recurse). Walk/reach `backend.compute:
local-docker` still means a future GPU/Isaac container — the studio does
not auto-route those onto this CPU image. HF Jobs / OSMO are Phase 3.
The studio-server stays in-process: it does not SSH and does not put
cloud credentials in the browser.

---

## 5. Studio

`studio/` is a static SPA. `ht serve` is FastAPI: recipes, expand,
datasets, runs, SSE, artifacts.

| Room | Owns | Must not own |
|---|---|---|
| Tasks | Recipe catalog projection | Physics, hardcoded GPU lists |
| Scene canvas | `scene.objects` x/y | Training loop |
| Data | LeRobot keep/drop → `data.keep_episodes` | Model code |
| Runs | Video + English status | Kubernetes |

The job spec editor is **Advanced**, not the home screen. Home is
`availability: cpu` recipes, then GPU skip. Copy for blocked runs,
empty runs, and Data-room record hints is generated from the catalog.

---

## 6. Data

Canonical demos: LeRobot v2 layout on disk (`meta/info.json` + episode
JSONL). Canvas drag, WASD, and gamepad sticks all write that same
table-frame trajectory schema. A second Save **appends** takes. Hugging Face Hub import is closed until we wrap their API.
`keep_episodes` is a spec field. Empty keep is refused. Failure-only
keep must miss mustard-in-bowl.

---

## 7. Layout

```
studio/                    # browser projection
src/humanoid_training/
  spec.py                  # schema
  recipes.py               # expand + public_catalog
  artifacts.py             # run file inventory
  adapters/                # compile / launch
    mujoco_adapter.py      # hold + imitation launches
    mujoco_runtime.py      # simulate / render
    mujoco_control.py      # poses / IK
  runner.py                # in-process job
  docker_runner.py         # Phase 1 CPU `ht train --docker`
  gold.py                  # gold/eval.mp4 record + coarse compare
  server.py                # studio API
spec/                      # schema + examples
recipes/<id>/recipe.yaml   # defaults + studio contract
recipes/<id>/gold/         # eval.mp4 + notes.md (CPU recipes)
robots/catalog.yaml
```

Nothing here requires inventing physics, a policy family, or a dataset
format.

See [VISION.md](./VISION.md) for the product, [ROADMAP.md](./ROADMAP.md)
for phase gates.
