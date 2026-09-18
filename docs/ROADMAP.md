# Roadmap

Build a studio that **compiles a job spec into existing engines**. Do not
build a simulator. Each phase has an exit test: if that test fails, the
phase is not done, even if the UI looks finished.

```
Phase 0  contract + one real train loop     ← done
Phase 1  studio shell (pick recipe → video) ← rooms + G1 stand + composed pick-and-place scene
Phase 2  demonstration data (LeRobot)       ← inspect + keep/drop started; teleop/train later
Phase 3  Isaac Lab as a second engine
Phase 4  real G1/H1 deploy with safety gates
```

---

## Phase 0 — Contract (done in this PR)

**Goal.** A versioned job spec, recipe expansion, adapter interface, and a
local runner that can train *something* and write an eval video plus a
manifest. If this loop is not reproducible, later UI is theater.

**Exit test.**

```bash
pip install -e ".[dev]"
ht train spec/examples/cartpole-balance.json --out runs/demo
# writes runs/demo/<id>/eval.mp4 and manifest.json
pytest -q
```

| Deliverable | Status |
|---|---|
| JSON Schema for the job spec | done |
| Recipe packages + expansion (user spec overlays defaults) | done |
| Adapter protocol: compile / launch / eval, fail closed | done |
| `gymnasium` adapter: CartPole RL + eval video on CPU | done |
| `playground` adapter: compile G1 walk to `G1JoystickFlatTerrain` | done (compile; launch needs GPU + Playground) |
| `isaaclab` adapter: compile G1 walk to `Isaac-Velocity-Flat-G1-v0` | done (compile only) |
| Local in-process runner + CLI `ht` | done |
| Run manifest (spec hash, adapter, seed, metrics, artifacts) | done |

**Not in Phase 0:** Docker GPUs, hosted compute, a 3D scene editor.

---

## Phase 1 — Studio shell

**Goal.** A browser app that authors the spec: pick a robot and a recipe,
click Train, watch eval clips. The UI is a projection of the spec.

**Exit test.** A person with no CLI knowledge can open the studio, train
`cartpole-balance`, and play the eval video. A G1 walk run shows a clear
"needs Playground + GPU" state instead of a crash.

| Deliverable | Status |
|---|---|
| `ht serve` API: recipes, validate, expand, runs, artifacts | done |
| Studio UI: Recipes, Spec, Train, Runs + video | done (Spec is Advanced; rooms are Robots / Tasks / Data / Runs) |
| Live log streaming (SSE) | done |
| Scene canvas (object placement writes the spec) | done (pick-and-place) |
| Dragged objects compiled into MuJoCo | done (MjSpec: table + primitives + eval camera) |
| Spec editor in the recipe view | done (collapsed behind Advanced) |
| Local Docker runner | later |

---

## Phase 2 — Data

**Goal.** Teleoperate, review episodes, train from demos using the
LeRobot dataset format. Non-experts *show* the task instead of designing
rewards.

**Exit test.** Record or import a LeRobot dataset, drop bad episodes,
train an imitation recipe, get an eval video.

| Deliverable | Status |
|---|---|
| LeRobot adapter (dataset I/O + ACT / similar) | inspect local `meta/info.json`; train launch still blocked |
| Episode review (keep / drop) | keep_episodes on the spec; no video editor yet |
| `pick-and-place` recipe backed by real data | CPU **scene preview** in MuJoCo; imitation later |
| Teleop session into the studio | not started |

---

## Phase 3 — Second engine

**Goal.** The same `g1-walk` spec runs on Isaac Lab (container or OSMO)
without rewriting the task. Playground/mjlab remains the fast path.

**Exit test.** `backend.prefer: [isaaclab]` submits a job and the studio
shows the same eval-video UI as CartPole, with a backend badge.

| Deliverable | Status |
|---|---|
| Isaac Lab compile (OSMO YAML / train command) | done (payload only) |
| mjlab adapter | done (compile only) |
| Launch via Docker / OSMO / cloud | not started |
| Backend downgrade messaging in the UI | partial |

---

## Phase 4 — Real robot

**Goal.** Deploy a checkpoint to a Unitree G1/H1 with clamps, watchdog,
and an e-stop. Policies stay `sim-only` until a hardware eval profile
passes.

**Exit test.** A sim-successful G1 walk policy runs on hardware at reduced
speed; a NaN or pose-limit kills the policy; no silent checkpoint
fallback.

Not started. Do not pretend a checkbox is sim-to-real.

---

## Recipe library (the actual product)

Ship boring tasks that *work*, with gold notes. Empty canvases are not
a v1 feature.

| Recipe | Phase | Runnable today |
|---|---|---|
| `cartpole-balance` | 0 smoke test | yes (CPU) |
| `g1-stand` | 1 | yes (CPU MuJoCo, Menagerie G1) |
| `g1-walk` | 0 compile / 3 train | compile only until Playground, mjlab, or Isaac Lab is present |
| `g1-reach` | 1 compile | compile only |
| `pick-and-place` | 2 | yes — CPU scene preview (G1 + table + dragged objects). Imitation from demos not trained. |
| imitation-from-demos | 2 | not started |

---

## Non-goals until the loop is real

- A new physics engine or policy architecture
- A new dataset format (use LeRobot)
- A new cluster orchestrator (emit OSMO / HF Jobs)
- Competing with LeLab on SO-ARM101
- Fleet operations (Foxglove / Formant)

---

## How to work this repo

1. Change the **spec or a recipe** first, then the adapter, then the UI.
2. Every runnable recipe must write `eval.mp4` + `manifest.json`.
3. Adapters fail closed with a sentence a beginner can act on.
4. Do not add a studio control that cannot be expressed in the job spec.
