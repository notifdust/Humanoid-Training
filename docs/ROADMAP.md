# Roadmap

Build a studio that **compiles a job spec into existing engines**. Do not
build a simulator. Each phase has an exit test: if that test fails, the
phase is not done, even if the UI looks finished.

```
Phase 0   contract + one real train loop     ← done (Cartpole on CPU)
Phase 1   studio shell (pick recipe → video) ← done (rooms + G1 stand + CPU Docker)
Phase 2   demonstration data (LeRobot)       ← done with substitutions (linear-BC, not ACT)
Phase 2.5 recipe gold + CI videos            ← done
Phase 3a  first real G1 walk (Playground)    ← launch path done
Phase 3b  mjlab / Isaac G1 walk + OSMO       ← launch path done (no G1 reach env upstream)
Phase 3c  GPU-box walk proof                 ← next (`ht proof walk`; live clip still needs a GPU)
Phase 3d  remote harvest (OSMO / GPU queue)  ← launch path done (needs OSMO pool for live proof)
Phase 3e  ACT on the same demos                 ← launch path done (needs GPU for live ACT)
Phase 3f  honest G1 manipulation recipe     ← done (deleted dishonest g1-reach)
Phase 3g  compare two runs in Runs           ← done (side-by-side in Runs room)
Phase 4   real G1/H1 deploy with safety gates ← gate done (live torque **not** wired)
```

This file is the **continuation plan from what is actually running**, not
from the original vision’s wish list. Vision still says where the product
is going ([VISION.md](./VISION.md)). Architecture still says how
([ARCHITECTURE.md](./ARCHITECTURE.md)).

---

## Where the loop is today

A beginner on a laptop can:

1. Open the studio (`./run-studio.sh` → http://127.0.0.1:8000).
2. Train **Cartpole**, **G1 stand**, or **Pick and place** and play `eval.mp4`.
3. Click **G1 walk** and get a blocked next step plus compiled
   payloads — unless this machine has an NVIDIA GPU and a walk
   engine (Playground `train-jax-ppo`, mjlab, or Isaac Lab
   `isaaclab.sh` / GPU Docker), in which case Train launches that
   engine and plays its eval clip.

There is no **G1 reach** recipe. No upstream G1 reach env exists; the
CPU arm preview is **Pick and place**.

The compiler contract holds: job spec → recipe expansion → adapter
compile/launch → `manifest.json` with `facts` → studio projects those
facts. In-process `run_job` is the default. `ht train --docker` is the
Phase 1 CPU runner and **fail-closes** when Docker is missing.

Honest substitutions that already shipped (do not relitigate by faking
the original names):

| Vision asked for | What ships | What it is not |
|---|---|---|
| G1 walk train | Playground / mjlab / Isaac Lab on a GPU box; blocked next step on CPU | Not a gold walk clip |
| ACT / diffusion | ACT on LeRobot demos when GPU + `lerobot` ready; else linear-BC on mujoco | Not finger grasping; not live ACT on a CPU laptop |
| Gamepad teleop | Canvas drag, WASD, and a gamepad stick write the same table-frame takes | Not a Unitree XR / leader-arm stack |
| Balance / locomotion RL | G1 stand holds a pinned pelvis and waves | Not walking, not a balance policy |
| Isaac / OSMO job | Local isaaclab.sh / GPU Docker, or OSMO submit→poll→rsync harvest | Not a silent Isaac success without a clip |
| Hosted GPU | OSMO harvest path when `osmo` CLI is present; CPU Docker still Phase 1 | Live OSMO pool proof still needed |

---

## Phase 0 — Contract (done)

**Exit test (met).** `ht train spec/examples/cartpole-balance.json` writes
`eval.mp4` + `manifest.json`. `pytest -q` is green.

| Deliverable | Status |
|---|---|
| JSON Schema for the job spec | done |
| Recipe packages + expansion (user spec overlays defaults) | done |
| Adapter protocol: compile / launch / eval, fail closed | done |
| `gymnasium` adapter: CartPole RL + eval video on CPU | done |
| `playground` adapter: compile G1 walk to `G1JoystickFlatTerrain` | done (Phase 3a launches `train-jax-ppo` when GPU + CLI are present) |
| `isaaclab` adapter: compile G1 walk to `Isaac-Velocity-Flat-G1-v0` | done (Phase 3b launches isaaclab.sh / GPU Docker) |
| `mjlab` adapter: compile G1 walk to `Mjlab-Velocity-Flat-Unitree-G1` | done (Phase 3b launches `python -m mjlab.scripts.train`) |
| Local in-process runner + CLI `ht` | done |
| Run manifest (spec hash, adapter, seed, metrics, `facts`) | done |

---

## Phase 1 — Studio shell (done)

**Exit test (met).** A person with no CLI knowledge can open the studio,
train `cartpole-balance`, and play the eval video. A G1 walk run shows
"needs Playground + GPU" instead of a crash.

| Deliverable | Status |
|---|---|
| `ht serve` API: recipes, validate, expand, runs, artifacts | done |
| Studio UI: Tasks / Train / Runs + video (Spec is Advanced) | done |
| Live log streaming (SSE) | done |
| Scene canvas (object placement writes the spec) | done (pick-and-place) |
| Dragged objects compiled into MuJoCo | done |
| Local Docker runner | done (`ht train --docker`; studio stays in-process) |

---

## Phase 2 — Data (done with substitutions)

**Exit test (met, substituted).** Record or import a *local* LeRobot
dataset, drop bad episodes, train an imitation recipe, get an eval
video. Keep-only-failure demos fail mustard-in-bowl; keep-success demos
pass. ACT is **not** that exit test.

| Deliverable | Status |
|---|---|
| LeRobot adapter (dataset I/O + ACT) | write/inspect local v2 layout; ACT launches when LeRobot + GPU (Phase 3e) |
| Episode review (keep / drop) | `keep_episodes` filters BC frames; empty keep is refused |
| `pick-and-place` recipe | scripted or canvas demos → ACT when LeRobot+GPU; else linear-BC mustard + G1 arm poses |
| Teleop into the studio | canvas, WASD, gamepad stick; Save **appends**; Space does not immediately record a second take |
| Hugging Face Hub import | fail-closed (`hf:` / huggingface.co URLs tell you to download locally) |

---

## Recipe library (the actual product)

| Recipe | Runnable today | Engine | Honest result |
|---|---|---|---|
| `cartpole-balance` | yes (CPU) | gymnasium | Pole stays up. Gold `eval.mp4` in-tree. |
| `g1-stand` | yes (CPU) | mujoco + Menagerie G1 | Stand + both-arm wave, pelvis pinned. Gold clip. |
| `pick-and-place` | yes (CPU linear-BC; ACT on GPU+LeRobot) | lerobot / mujoco | Same demos. `facts.policy=act` or `linear-bc`. Gold clip is CPU linear-BC. |
| `g1-walk` | yes on GPU + Playground, mjlab, or Isaac Lab; blocked on CPU | playground / mjlab / isaaclab | Walking eval from the engine that launched. No gold clip. |
| Unitree H1 | catalog only | — | No recipe. Do not add one until G1 walk trains for real. |

`g1-reach` was deleted in Phase 3f. There is no upstream G1 reach env
to pin; do not invent `G1Reach-v0` / `Isaac-Reach-G1-v0`. CPU arm motion
is `pick-and-place`.

Gold notes live in each `recipe.yaml`. CPU recipes also ship
`gold/eval.mp4` + `gold/notes.md`. CI’s `gold` job retrains those recipes
under xvfb and compares decoded frames to the checked-in clip. GPU
recipes must not check in a success video.

---

## What this round re-checked (2026-09-21)

Live, not just “the tests used to pass”:

- `pytest -q` — green.
- CLI: `ht recipes` groups cpu/gpu; Cartpole compile; G1 walk **exit 12**
  with Playground / mjlab next steps; payloads include
  `engines/mjlab/train_mjlab.sh` and `engines/isaaclab/osmo_workflow.yaml`.
- `ht train --docker` — **exit 12**, "Docker is not on PATH"; in-process
  train still works.
- CLI train with a display: Cartpole, G1 stand, pick-and-place all
  **passed** with `eval.mp4` and `facts.runner=inprocess`. Stand facts:
  `kind=hold`, `arm_driven`, `pinned`. Pick facts: `kind=imitation`,
  `arm_mode=playback+BC`, `dataset=auto-scripted`.
- Studio browser: Tasks home groups ready vs GPU-skip; selecting G1
  **keeps** Cartpole (`start_here`); Cartpole Train → video "worked";
  G1 walk Compile → "can't train here" / no video; Pick-and-place copy
  says demo-following, not grasping; Data room matches `record_hint`;
  Runs lists the blocked walk next to the Cartpole clip.
- Studio API pick-and-place (short overlay) **passed** with `eval.mp4`.

Do not treat a green pytest as a substitute for "the humanoid walked."
It did not.

---

## Phase 2.5 — Gold clips + CI video (done)

**Exit test (met).** Every `availability: cpu` recipe has `gold/eval.mp4`
and `gold/notes.md`. `pytest` probes those files. The GitHub `gold` job
runs `xvfb-run pytest -m gold`, which retrains each CPU recipe and
fails if the new `eval.mp4` does not coarsely match gold (duration +
downscaled frames — not a byte hash, not a learned judge). GPU recipes
have no success clip.

| Deliverable | Status |
|---|---|
| `recipes/<id>/gold/eval.mp4` + `notes.md` for CPU recipes | done |
| `ht gold` to regenerate clips from the beginner spec | done |
| Catalog `has_gold`; studio plays the clip on the task page | done |
| CI xvfb job retrains and compares frames | done |
| GPU recipes must not ship a fake walk clip | done (test) |

Regenerate clips after a CPU recipe’s eval changes:

```bash
unset HT_NO_RENDER
ht gold
```

---

## Phase 3a — first real G1 walk (Playground) (launch path done)

**Exit test.** On a machine with an NVIDIA GPU and
`pip install playground`, `ht train spec/examples/g1-walk.json` launches
(not merely compiles), writes `eval.mp4` of **walking**, stamps
`facts.kind=rl` and `facts.engine=playground`, and the studio Plays
that video with a backend badge. Without GPU / Playground, behavior
stays the blocked next step (compile payloads, exit 12).

This repo's CI and the CPU studio **do not** have a GPU. The launch
path is tested with a fake `train-jax-ppo` that writes `rollout0.mp4`.
A live walking video is still the GPU-box proof. Do not check in a
walk gold clip.

| Deliverable | Status |
|---|---|
| `PlaygroundAdapter.launch` runs `train-jax-ppo --env_name G1JoystickFlatTerrain` | done |
| Stream stdout into `run.log`; write `job.json` | done |
| Harvest `**/rollout*.mp4` → `eval.mp4`; fail closed if exit 0 and no clip | done |
| `facts.kind=rl`, `facts.engine=playground`, `facts.device=gpu` | done |
| Catalog `launch_here`; studio Train vs Compile and backend badge | done |
| CPU Docker refuses GPU recipes (`HT_DOCKER_GPU=1` escape hatch) | done |
| Walk recipe must not invent reach env ids (`G1Reach-v0`) | done |
| No fake walk gold / stand-clip substitute | done |

```bash
# GPU box
pip install playground
ht train spec/examples/g1-walk.json
# CPU laptop — still exit 12, payloads on disk
ht train spec/examples/g1-walk.json
# Do not:
ht train spec/examples/g1-walk.json --docker   # refused (Phase 1 CPU image)
```

Host probes (no recipe ids): `HT_GPU` / `HT_PLAYGROUND_GPU`,
`HT_PLAYGROUND_CLI`, `HT_MJLAB_CLI`, `HT_ISAAC_CLI`, `HT_OSMO_CLI`,
`HT_DOCKER_GPU`.

---

## Phase 3b — mjlab / Isaac G1 walk + OSMO (launch path done)

**Exit test.** `backend.prefer: [mjlab]` or `[isaaclab]` on a GPU
box launches the real G1 *walk* task and the studio shows the same
eval-video UI as Cartpole, with a backend badge
(`mjlab · gpu` / `isaaclab · gpu`). Without that engine, behavior
stays the blocked next step.

**No G1 reach recipe.** There is no G1 reach environment upstream:

- mjlab G1: `Mjlab-Velocity-Flat-Unitree-G1`, Rough, Tracking. Cube
  lift is YAM (`Mjlab-Lift-Cube-Yam`), not G1.
- Isaac Lab G1: `Isaac-Velocity-Flat-G1-v0` / Rough. Reach is
  Franka / UR10 / OpenArm. G1 manipulation is PickPlace, not Reach
  (PickPlace is not pinned here until a real task id is confirmed).
- Playground `G1JoystickFlatTerrain` is walking. It is not a reach env.

Do not invent `G1Reach-v0` or `Isaac-Reach-G1-v0`. Phase 3f deletes the
blocked `g1-reach` stub rather than keep a fake catalog card.

| Deliverable | Status |
|---|---|
| Pin `g1-walk` mjlab to `Mjlab-Velocity-Flat-Unitree-G1` | done |
| Pin `g1-walk` Isaac to `Isaac-Velocity-Flat-G1-v0` | done |
| Delete dishonest `g1-reach` recipe (Phase 3f) | done |
| `MJLabAdapter.launch` runs `python -m mjlab.scripts.train TASK --video True` | done |
| Harvest mjlab `*.mp4` under `--log-root`; fail closed if exit 0 and no clip | done |
| `IsaacLabAdapter.launch` via `isaaclab.sh` train+play, or GPU Docker | done |
| Harvest Isaac `*.mp4`; fail closed if exit 0 and no clip | done |
| `osmo workflow submit` → poll → rsync harvest `ht_eval/*.mp4` | done (Phase 3d harness) |
| Catalog `launch_here` for playground **or** mjlab **or** isaac ready | done |
| `select_adapter` first launch-ready GPU engine, else first compile-ok | done |
| No fake walk/reach gold / stand-clip substitute | done |

```bash
# GPU box — mjlab
# pip/uv install mjlab
ht train spec/examples/g1-walk.json   # or overlay backend.prefer: [mjlab]
# GPU box — Isaac Lab
# HT_ISAAC_CLI=/path/to/isaaclab.sh
ht train spec/examples/g1-walk.json   # overlay backend.prefer: [isaaclab]
# GPU Docker (not the Phase 1 CPU image)
HT_DOCKER_GPU=1 ht train spec/examples/g1-walk.json  # prefer isaaclab
# CPU laptop — still exit 12
ht train spec/examples/g1-walk.json
```

This repo's CI and the CPU studio **do not** have a GPU. Launch is
tested with fake `HT_MJLAB_CLI` / `HT_ISAAC_CLI` scripts that write
an mp4. A live walking video is still the GPU-box proof.

---

## Continuation roadmap (follow in order)

Do not skip ahead to ACT, grasping, or hardware while the walk path
is still unproven on a real GPU. Each phase has an exit test.

### Phase 3c — GPU-box walk proof (next)

**Exit test.** On a machine with an NVIDIA GPU:

```bash
pip install playground   # or mjlab / Isaac Lab
ht proof walk
# optional: ht proof walk --prefer mjlab
# optional: HT_ISAAC_CLI=/path/to/isaaclab.sh ht proof walk --prefer isaaclab
```

That must write a **walking** `eval.mp4`, stamp `facts.engine` to the
engine that ran, and the studio must Play it with a backend badge.
`ht proof walk` fails closed on a CPU laptop (exit 12) with the next
step — that is expected.

| Deliverable | Status |
|---|---|
| `ht proof walk` operator command | done (harness) |
| Short proof train + require `eval.mp4` + `facts.engine` | done (harness) |
| Live walk clip from a real GPU box | **not done** (needs GPU) |
| Still no gold walk clip checked into the repo | keep |

Do **not** declare Phase 3c complete from fake-CLI unit tests alone.
Do **not** check in a stand clip as walk gold.

### Phase 3d — Remote harvest (OSMO / hosted GPU)

**Exit test.** From a laptop without CUDA, with the OSMO CLI logged in:

```bash
ht train spec/examples/g1-walk.json
# backend.prefer includes isaaclab (default after playground/mjlab miss)
```

That submits `osmo_workflow.yaml`, polls until `COMPLETED`, rsyncs
`/osmo/run/workspace/ht_eval/*.mp4` into the run dir as `eval.mp4`, and
the studio Plays it with `facts.launch=osmo` / `facts.device=remote`.
Browser holds no cloud credentials.

| Deliverable | Status |
|---|---|
| OSMO submit → poll → rsync harvest | done (harness) |
| Workflow copies clips to `ht_eval` for a stable download path | done |
| Catalog `launch_here` when `osmo_ready` (no local GPU required) | done |
| Live OSMO cluster proof | **not done** (needs OSMO credentials + pool) |
| HF Jobs path | not started (same harvest contract later) |

Opt out of harvest attempts: `HT_OSMO_HARVEST=0`.
Poll knobs: `HT_OSMO_POLL_SECONDS`, `HT_OSMO_TIMEOUT_SECONDS`.

### Phase 3e — ACT on the same demos

**Exit test.** GPU + LeRobot runs ACT (or another policy) on the same
local dataset; `facts.policy=act` vs `linear-bc`. CPU linear-BC stays.
Finger grasping stays out of scope.

| Deliverable | Status |
|---|---|
| Prefer `lerobot` before `mujoco` on pick-and-place | done |
| `lerobot_ready` probe (`HT_LEROBOT_CLI` + GPU) | done |
| Launch `lerobot-train --policy.type=act` on local demos | done (harness) |
| `facts.policy=act` vs `linear-bc` on mujoco fallback | done |
| Fail closed on missing video / nonzero exit (no mustard sub) | done |
| Live ACT train on a GPU box | **not done** (needs `pip install 'lerobot[training]'` + NVIDIA GPU) |

Fake CLI for CI: `HT_LEROBOT_CLI=/path/to/fake` with `HT_GPU=1`.

### Phase 3f — Honest G1 manipulation recipe

**Exit test (met).** Delete `g1-reach`, or replace it with a real upstream
task id under an honest name. No invented env ids.

| Deliverable | Status |
|---|---|
| Delete `recipes/g1-reach` + `spec/examples/g1-reach.json` | done |
| Catalog `later` is only `g1-walk` on CPU | done |
| Unknown `g1-reach` recipe id fails closed (`RecipeError`) | done |
| Do not invent `G1Reach-v0` / `Isaac-Reach-G1-v0` | done |
| Pin Isaac G1 PickPlace under a new honest recipe | **not done** (no confirmed upstream task id in-repo) |

CPU arm motion stays `pick-and-place`. Add a PickPlace recipe only after
confirming a real Isaac/mjlab task string on a GPU box.

### Phase 3g — Compare two runs (Evaluate in Runs)

**Exit test (met).** Side-by-side videos/facts for two run ids of the same
recipe. No fifth studio room.

| Deliverable | Status |
|---|---|
| Checkbox select two runs in the Runs list | done |
| Compare enabled only when both share `run.recipe` | done |
| Side-by-side videos + `facts` under the Runs room | done |
| No fifth rail button / Evaluate room | done |

Pick two runs of the same task → **Compare**. Still four rooms.

### Phase 4 — Real robot (after 3c)

**Exit test.** Sim-successful G1 walk runs on hardware at reduced
speed; NaN or pose-limit kills the policy; no silent checkpoint
fallback. Policies stay `sim-only` until a hardware eval profile
passes.

| Deliverable | Status |
|---|---|
| Stamp `facts.sim_only=true` on every in-process run | done |
| `ht deploy <run_id>` fail-closed gate (cartpole / stand / mustard refused) | done |
| Require passed `g1-walk` + `eval.mp4` + `HT_HARDWARE_PROFILE` | done (harness) |
| Runs UI shows `sim-only` (and `policy=` when present) | done |
| Studio **Deploy to robot** + `POST /api/runs/{id}/deploy` fail closed | done |
| Unitree reduced-speed driver + NaN / pose-limit kills | **not done** |
| Live hardware eval that clears `sim_only` | **not done** (needs robot + live walk proof) |

```bash
ht deploy <run_id>          # exit 12 until a hardware profile + driver exist
# Optional operator file once a real hardware eval passes:
# HT_HARDWARE_PROFILE=/path/to/hw.json  # {"passed": true, "kind": "hardware"}
```

Do not treat a green `assess_deploy` checklist as live torque. The driver
is still unwired on purpose.

---

## Explicitly not next

- A new physics engine, policy family, or dataset format
- A new cluster orchestrator (emit OSMO / HF Jobs / Docker)
- Competing with LeLab on SO-ARM101
- Fleet operations (Foxglove / Formant)
- A fifth studio room before Phase 4 (Evaluate stays in Runs)
- Fake walk / ACT / Isaac / OSMO success on a CPU laptop
- Finger grasping before ACT ships
- Hardware deploy before a live walk clip exists

---

## How to work this repo

1. Change the **spec or a recipe** first, then the adapter, then the UI.
2. Every runnable recipe must write `eval.mp4` + `manifest.json`.
3. Adapters fail closed with a sentence a beginner can act on.
4. Do not add a studio control that cannot be expressed in the job spec.
5. `EvalResult.facts` is the Runs contract. Notes are English for humans.
6. If a GPU engine is missing, compile the payload and block. Never
   substitute a hold-pose clip.
