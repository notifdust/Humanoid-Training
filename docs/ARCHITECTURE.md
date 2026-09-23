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
  "later": ["g1-walk"],
  "start_here": ["cartpole-balance", "g1-stand", "pick-and-place"]
}
```

On a GPU box with Playground, mjlab, or Isaac Lab ready, `g1-walk`
moves from `later` to `ready` because `launch_here` is true.
There is no `g1-reach` recipe: no upstream G1 reach env exists to pin.

The browser **projects** `availability`, `launch_here`, `promise`, `train_hint`,
`blocked_hint`, `scene_hint`, and `record_hint`. It must not hardcode
recipe ids to decide what works on a laptop, which task to Train as a
fallback, or which imitation recipe to record.

`runnable` means an adapter can launch something when the engine is
present. `availability: gpu` means the laptop grouping is "needs GPU".
`launch_here` is the Train button contract: true → launch; false →
compile and block. `start_here` defaults to `availability == cpu` when
omitted. GPU recipes may set `ready_promise` / `ready_hint` for the
GPU-ready copy.

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
returns the eval. CPU recipes train inside that call. Phase 3 GPU walk still returns from `launch`, but the adapter
**subprocesses** the engine CLI (`train-jax-ppo`, `python -m mjlab.scripts.train`,
or `isaaclab.sh`), streams stdout into `run.log` (studio SSE tails it),
then harvests `*.mp4` → `eval.mp4`. Missing CLI / GPU / video
fails closed. Do not substitute the G1 stand clip. OSMO submit does
not harvest a remote clip.

| Adapter | Role today |
|---|---|
| `gymnasium` | Cartpole RL + eval video |
| `mujoco` | G1 stand hold; pick-and-place linear-BC + arm poses |
| `playground` | Compile + launch G1 walk via `train-jax-ppo` when GPU + CLI are present; otherwise compile and block |
| `mjlab` | Compile + launch G1 walk `Mjlab-Velocity-Flat-Unitree-G1` when mjlab + GPU are present; otherwise compile and block |
| `isaaclab` | Compile OSMO YAML; launch via `isaaclab.sh`, GPU Docker, or OSMO submit→poll→rsync harvest |
| `lerobot` | Compile + launch ACT (`lerobot-train`) when LeRobot + GPU are present; CPU imitation stays on mujoco linear-BC |

The MuJoCo adapter is three modules, not one god file:

- `mujoco_adapter.py` — `support` / `compile` / `launch` (hold vs imitation)
- `mujoco_runtime.py` — simulate, renderer, pin, body ids
- `mujoco_control.py` — open-loop poses, qpos drive, Jacobian IK

Selection: `backend.prefer` order, skip `unsupported`, first
**launch-ready** engine, else first `support.ok` (compile-and-block).
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
studio **projects** `facts` (arm_mode, bc_steps, keep_episodes, kind,
engine, device, runner) and only scrapes notes for older runs that have none.

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
(never `--docker`, or it would recurse). GPU recipes (`availability: gpu`)
are **refused** on that image unless `HT_DOCKER_GPU=1` (future GPU
container). Walk/reach `backend.compute: local-docker` still means that
future image — the studio does not auto-route those onto the CPU image.
HF Jobs / OSMO are later Phase 3. The studio-server stays in-process: it
does not SSH and does not put cloud credentials in the browser.

`src/humanoid_training/hardware.py` probes GPU (`nvidia-smi` /
`HT_GPU` / `HT_PLAYGROUND_GPU`) and engine CLIs (`HT_PLAYGROUND_CLI`,
`HT_MJLAB_CLI`, `HT_ISAAC_CLI`, `HT_OSMO_CLI`, `HT_DOCKER_GPU`) so the
catalog can set `launch_here` without importing adapters.

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

The job spec editor is **Advanced**, not the home screen. Home groups
`launch_here` recipes first, then GPU skip. Copy for blocked runs,
empty runs, and Data-room record hints is generated from the catalog.
Runs projects `facts.engine` / `facts.device` / `facts.runner` as a
backend badge (`playground · gpu`).

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
  hardware.py              # GPU / engine CLI probes
  proof.py                 # Phase 3c GPU walk proof
  deploy.py                # Phase 4 fail-closed hardware deploy gate
  adapters/                # compile / launch
    process.py             # shared subprocess + mp4 harvest
    playground.py
    mjlab.py
    isaaclab.py
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
