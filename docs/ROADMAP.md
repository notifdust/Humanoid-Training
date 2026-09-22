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
Phase 3c  GPU-box walk proof                 ← next
Phase 3d  remote harvest (OSMO / GPU queue)  ← after 3c
Phase 3e  ACT on the same demos              ← after 3d (or parallel once 3c is green)
Phase 3f  honest G1 manipulation recipe      ← replace or delete g1-reach
Phase 3g  compare two runs in Runs           ← Evaluate without a fifth room
Phase 4   real G1/H1 deploy with safety gates ← only after a real walk clip exists
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
4. Click **G1 reach** and still get a blocked next step: there is
   no G1 reach environment in Playground, mjlab, or Isaac Lab.

The compiler contract holds: job spec → recipe expansion → adapter
compile/launch → `manifest.json` with `facts` → studio projects those
facts. In-process `run_job` is the default. `ht train --docker` is the
Phase 1 CPU runner and **fail-closes** when Docker is missing.

Honest substitutions that already shipped (do not relitigate by faking
the original names):

| Vision asked for | What ships | What it is not |
|---|---|---|
| G1 walk train | Playground / mjlab / Isaac Lab on a GPU box; blocked next step on CPU | Not a gold walk clip |
| ACT / diffusion | Linear-BC on LeRobot v2 JSONL; G1 arm pose playback | Not finger grasping, not ACT |
| Gamepad teleop | Canvas drag, WASD, and a gamepad stick write the same table-frame takes | Not a Unitree XR / leader-arm stack |
| Balance / locomotion RL | G1 stand holds a pinned pelvis and waves | Not walking, not a balance policy |
| Isaac / OSMO job | Local `isaaclab.sh` or GPU Docker harvests `eval.mp4`; `osmo workflow submit` has no remote harvest | Not a silent Isaac success on CPU |
| Hosted GPU | CPU Docker image + in-process studio; OSMO submit is fail-closed | No eval-video harvest from the cloud |

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
| LeRobot adapter (dataset I/O + ACT) | write/inspect local v2 layout; `train_lerobot.sh` compiled; ACT launch blocked |
| Episode review (keep / drop) | `keep_episodes` filters BC frames; empty keep is refused |
| `pick-and-place` recipe | scripted or canvas/WASD/gamepad demos → linear-BC mustard + G1 pick/lift/place poses |
| Teleop into the studio | canvas, WASD, gamepad stick; Save **appends**; Space does not immediately record a second take |
| Hugging Face Hub import | fail-closed (`hf:` / huggingface.co URLs tell you to download locally) |

---

## Recipe library (the actual product)

| Recipe | Runnable today | Engine | Honest result |
|---|---|---|---|
| `cartpole-balance` | yes (CPU) | gymnasium | Pole stays up. Gold `eval.mp4` in-tree. |
| `g1-stand` | yes (CPU) | mujoco + Menagerie G1 | Stand + both-arm wave, pelvis pinned. Gold clip. |
| `pick-and-place` | yes (CPU) | mujoco + LeRobot demos | Mustard into bowl via linear-BC. Gold clip. |
| `g1-walk` | yes on GPU + Playground, mjlab, or Isaac Lab; blocked on CPU | playground / mjlab / isaaclab | Walking eval from the engine that launched. No gold clip. |
| `g1-reach` | no | — | Blocked. No G1 reach env in Playground, mjlab, or Isaac Lab. Do not map locomotion or Franka Reach as G1 reach. |
| Unitree H1 | catalog only | — | No recipe. Do not add one until G1 walk trains for real. |

Gold notes live in each `recipe.yaml`. CPU recipes also ship
`gold/eval.mp4` + `gold/notes.md`. CI’s `gold` job retrains those recipes
under xvfb and compares decoded frames to the checked-in clip. GPU
recipes must not check in a success video.

---

## What this round re-checked (2026-09-21)

Live, not just “the tests used to pass”:

- `pytest -q` — green.
- CLI: `ht recipes` groups cpu/gpu; Cartpole compile; G1 walk/reach **exit 12**
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
| `g1-reach` must not launch walk (Playground mapping stays `unsupported`) | done |
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

**Reach is pinned, not launched.** There is no G1 reach environment
upstream:

- mjlab G1: `Mjlab-Velocity-Flat-Unitree-G1`, Rough, Tracking. Cube
  lift is YAM (`Mjlab-Lift-Cube-Yam`), not G1.
- Isaac Lab G1: `Isaac-Velocity-Flat-G1-v0` / Rough. Reach is
  Franka / UR10 / OpenArm. G1 manipulation is PickPlace, not Reach.
- Playground `G1JoystickFlatTerrain` is walking. It is not a reach env.

`g1-reach` marks every adapter `unsupported`. Train blocks. Do not
invent `G1Reach-v0` or `Isaac-Reach-G1-v0`.

| Deliverable | Status |
|---|---|
| Pin `g1-walk` mjlab to `Mjlab-Velocity-Flat-Unitree-G1` | done |
| Pin `g1-walk` Isaac to `Isaac-Velocity-Flat-G1-v0` | done |
| Delete Playground walk placeholder from `g1-reach` | done |
| `MJLabAdapter.launch` runs `python -m mjlab.scripts.train TASK --video True` | done |
| Harvest mjlab `*.mp4` under `--log-root`; fail closed if exit 0 and no clip | done |
| `IsaacLabAdapter.launch` via `isaaclab.sh` train+play, or GPU Docker | done |
| Harvest Isaac `*.mp4`; fail closed if exit 0 and no clip | done |
| `osmo workflow submit osmo_workflow.yaml` then block (no remote harvest) | done |
| Catalog `launch_here` for playground **or** mjlab **or** isaac ready | done |
| `select_adapter` first launch-ready GPU engine, else first compile-ok | done |
| `g1-reach` stays `launch_here=false` even if mjlab is ready | done |
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
ht train spec/examples/g1-reach.json
```

This repo's CI and the CPU studio **do not** have a GPU. Launch is
tested with fake `HT_MJLAB_CLI` / `HT_ISAAC_CLI` scripts that write
an mp4. A live walking video is still the GPU-box proof.

---

## Continuation roadmap (follow in order)

Do not skip ahead to ACT, grasping, or hardware while the walk path
is still unproven on a real GPU. Each phase has an exit test. If that
test fails, the phase is not done.

### Phase 3c — GPU-box walk proof (next)

**Exit test.** On a machine with an NVIDIA GPU, one of these succeeds
end-to-end and the studio Plays a **walking** `eval.mp4` with a backend
badge:

```bash
pip install playground
ht train spec/examples/g1-walk.json
# or
# mjlab installed → backend.prefer: [mjlab]
# HT_ISAAC_CLI=…/isaaclab.sh → backend.prefer: [isaaclab]
```

| Deliverable | Status |
|---|---|
| Live walk clip from Playground **or** mjlab **or** Isaac Lab | not started |
| Studio Plays that clip; `facts.engine` matches the engine that ran | not started |
| Still no gold walk clip checked into the repo | keep |

Do **not** check in a stand clip as walk gold. Do **not** declare 3c
done from fake-CLI unit tests alone.

### Phase 3d — Remote harvest (OSMO / hosted GPU)

**Exit test.** From a laptop without CUDA, Train on `g1-walk` submits
a job (OSMO, HF Jobs, or a GPU Docker queue), harvests the remote
`eval.mp4` into the same run dir, and the studio Plays it with
`facts.runner` / `facts.engine` set. Browser still holds no cloud
credentials.

| Deliverable | Status |
|---|---|
| Submit + poll + pull video (not submit-and-block) | not started |
| Same Runs UI as local launch (badge, video, facts) | not started |
| Emit existing orchestrators only (OSMO / HF Jobs / Docker) | not started |

This is vision requirement #3 (non-experts skip CUDA). It is not a
new cluster product.

### Phase 3e — ACT on the same demos

**Exit test.** On a GPU box with `pip install lerobot`, Train on
`pick-and-place` can run ACT (or another LeRobot policy) on the same
local dataset linear-BC already uses. Manifest stamps
`facts.policy=act` (vs `linear-bc`). CPU studio still runs linear-BC.
Studio copy labels which policy ran.

| Deliverable | Status |
|---|---|
| Launch compiled `train_lerobot.sh` when LeRobot + GPU are present | not started |
| `facts.policy` distinguishes ACT vs linear-BC | not started |
| Finger grasping still explicitly out of scope | keep |

Do not tighten IK and call it a grasp. Do not remove the CPU linear-BC
path.

### Phase 3f — Honest G1 manipulation recipe

**Exit test.** Either delete `g1-reach`, or replace it with a recipe
pinned to a **real** upstream task id (e.g. Isaac PickPlace-*-G1, or
a future verified G1 reach), with `launch_here` / promise / train_hint
that match what the engine actually does.

| Deliverable | Status |
|---|---|
| No dead "reach" card that can never launch | not started |
| No invented env ids (`G1Reach-v0`, `Isaac-Reach-G1-v0`) | keep |
| Hub import into `HT_CACHE` (optional, boring) | not started |

### Phase 3g — Compare two runs (Evaluate in Runs)

**Exit test.** From Runs, a beginner can open two passed checkpoints
for the same recipe and see both videos / facts side by side (seed,
keep_episodes, engine, policy). No fifth studio room.

| Deliverable | Status |
|---|---|
| Side-by-side compare for two run ids | not started |
| Driven by `facts` + artifacts, not note scraping | not started |

### Phase 4 — Real robot (after 3c)

**Exit test.** A sim-successful G1 walk policy runs on hardware at
reduced speed; a NaN or pose-limit kills the policy; no silent
checkpoint fallback. Policies stay `sim-only` until a hardware eval
profile passes.

Do not start this before Phase 3c produces a real walk video.
Clamps, watchdog, and e-stop are the product — not a "deploy"
checkbox.

---

## Explicitly not next

- A new physics engine, policy family, or dataset format
- A new cluster orchestrator (emit OSMO / HF Jobs / Docker)
- Competing with LeLab on SO-ARM101
- Fleet operations (Foxglove / Formant)
- A fifth studio room before Phase 3g
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
