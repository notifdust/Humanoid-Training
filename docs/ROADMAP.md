# Roadmap

Build a studio that **compiles a job spec into existing engines**. Do not
build a simulator. Each phase has an exit test: if that test fails, the
phase is not done, even if the UI looks finished.

```
Phase 0  contract + one real train loop     ← done
Phase 1  studio shell (pick recipe → video) ← v1 done; scene canvas later
Phase 2  demonstration data (LeRobot)
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
| Studio UI: Recipes, Spec, Train, Runs + video | done (v1) |
| Live log streaming (SSE) | later |
| Local Docker runner | later |
| Scene canvas (object placement) | later |

---

## Phase 2 — Data

**Goal.** Teleoperate, review episodes, train from demos using the
LeRobot dataset format. Non-experts *show* the task instead of designing
rewards.

**Exit test.** Record or import a LeRobot dataset, drop bad episodes,
train an imitation recipe, get an eval video.

| Deliverable | Status |
|---|---|
| LeRobot adapter (dataset I/O + ACT / similar) | not started |
| Episode review (keep / drop) | not started |
| `pick-and-place` recipe backed by real data | spec example only |
| Teleop session into the studio | not started |

---

## Phase 3 — Second engine

**Goal.** The same `g1-walk` spec runs on Isaac Lab (container or OSMO)
without rewriting the task. Playground/mjlab remains the fast path.

**Exit test.** `backend.prefer: [isaaclab]` submits a job and the studio
shows the same eval-video UI as CartPole, with a backend badge.

| Deliverable | Status |
|---|---|
| Isaac Lab compile (OSMO YAML / train command) | this PR (payload only) |
| Launch via Docker / OSMO / cloud | not started |
| mjlab adapter | not started |
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
| `g1-walk` | 0 compile / 1–3 train | compile only until Playground or Isaac Lab is present |
| `g1-stand` | 1 | not started |
| `g1-reach` | 1–2 | not started |
| `pick-and-place` | 2 | spec only |
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
