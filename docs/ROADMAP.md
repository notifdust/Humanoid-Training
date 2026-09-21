# Roadmap

Build a studio that **compiles a job spec into existing engines**. Do not
build a simulator. Each phase has an exit test: if that test fails, the
phase is not done, even if the UI looks finished.

```
Phase 0  contract + one real train loop     ← done (Cartpole on CPU)
Phase 1  studio shell (pick recipe → video) ← done (rooms + G1 stand + CPU Docker)
Phase 2  demonstration data (LeRobot)       ← done with substitutions (linear-BC, not ACT)
Phase 2.5 recipe gold + CI videos           ← done
Phase 3a first real G1 walk (Playground)    ← launch path done (needs a GPU box for the walking clip)
Phase 3b mjlab reach, then Isaac / OSMO
Phase 4  real G1/H1 deploy with safety gates
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
   payloads — unless this machine has an NVIDIA GPU and
   `train-jax-ppo` on PATH, in which case Train launches Playground PPO
   and plays the engine's `rollout0.mp4` as `eval.mp4`.
4. Click **G1 reach** and still get a blocked next step (mjlab / Isaac,
   not Playground locomotion).

The compiler contract holds: job spec → recipe expansion → adapter
compile/launch → `manifest.json` with `facts` → studio projects those
facts. In-process `run_job` is the default. `ht train --docker` is the
Phase 1 CPU runner and **fail-closes** when Docker is missing.

Honest substitutions that already shipped (do not relitigate by faking
the original names):

| Vision asked for | What ships | What it is not |
|---|---|---|
| G1 walk train | Playground `train-jax-ppo` on a GPU box; blocked next step on CPU | Not a gold walk clip; not mjlab/Isaac yet |
| ACT / diffusion | Linear-BC on LeRobot v2 JSONL; G1 arm pose playback | Not finger grasping, not ACT |
| Gamepad teleop | Canvas drag, WASD, and a gamepad stick write the same table-frame takes | Not a Unitree XR / leader-arm stack |
| Balance / locomotion RL | G1 stand holds a pinned pelvis and waves | Not walking, not a balance policy |
| Isaac / OSMO job | `osmo_workflow.yaml` + `train_mjlab.sh` on disk | Not submitted, not evaluated |
| Hosted GPU | CPU Docker image + in-process studio | No cloud queue, no GPU image |

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
| `isaaclab` adapter: compile G1 walk to `Isaac-Velocity-Flat-G1-v0` | done (payload only) |
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
| `g1-walk` | yes on GPU + Playground; blocked on CPU | playground / mjlab / isaaclab | Walking eval from Playground rollout. No gold clip. |
| `g1-reach` | compile only | mjlab / isaaclab | Blocked. Playground mapping is a **locomotion placeholder** — do not call that a reach env. |
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

Host probes (no recipe ids): `HT_PLAYGROUND_GPU`, `HT_PLAYGROUND_CLI`.

---

## Next — Phase 3b: mjlab reach, then Isaac / OSMO

**Exit test.** `backend.prefer: [isaaclab]` (or mjlab) submits a job
and the studio shows the same eval-video UI as Cartpole, with a backend
badge.

1. **Pin reach.** `g1-reach`’s Playground `G1JoystickFlatTerrain` map is
   a placeholder. Delete it or replace it with a verified reach env id
   *before* claiming reach trains. mjlab `G1Reach-v0` / Isaac
   `Isaac-Reach-G1-v0` must be confirmed on a GPU box, then launched
   from the generated `train_mjlab.sh`.
2. **Submit `osmo_workflow.yaml`.** The file is already an artifact.
   Phase 3 is the submitter (OSMO, a GPU Docker image, or a ssh-less
   local `docker run` of `nvcr.io/nvidia/isaac-lab:2.2.0`) plus log/video
   harvest. The studio still does not hold cloud credentials in the
   browser.
3. **Hosted GPU queue** is how non-experts skip CUDA install (vision
   requirement #3). That is a product behind the same spec, not a new
   orchestrator — emit OSMO / HF Jobs / plain Docker.

---

## Next — ACT, then grasping (still Phase 2 leftovers)

Linear-BC on mocap mustard is the CPU preview. It is not the
manipulation policy.

1. **ACT (or another LeRobot policy) on the same dataset.**
   `engines` already get `train_lerobot.sh`. Needs `pip install lerobot`
   and a GPU. CPU linear-BC stays as the laptop path. The studio must
   label which policy ran (`facts.policy=act` vs `linear-bc`).
2. **Finger grasping is a new policy**, not more pose playback. Do not
   tighten IK and call it a grasp. When it exists, the recipe
   `promise` / `train_hint` have to change; the current copy is
   deliberately narrow.
3. **Hub import** after local inspect is boring: download into
   `HT_CACHE`, then the Data room already works.

---

## Phase 4 — Real robot (not started)

**Exit test.** A sim-successful G1 walk policy runs on hardware at
reduced speed; a NaN or pose-limit kills the policy; no silent
checkpoint fallback. Policies stay `sim-only` until a hardware eval
profile passes.

Do not start this before Phase 3a produces a real walk video. Clamps,
watchdog, e-stop are the product here — not a "deploy" checkbox.

---

## Explicitly not next

- A new physics engine, policy family, or dataset format
- A new cluster orchestrator (emit OSMO / HF Jobs / Docker)
- Competing with LeLab on SO-ARM101
- Fleet operations (Foxglove / Formant)
- A fifth studio room until two checkpoints exist to compare
  (Evaluate is folded into Runs on purpose)
- Fake walk / ACT / Isaac success on a CPU laptop

---

## How to work this repo

1. Change the **spec or a recipe** first, then the adapter, then the UI.
2. Every runnable recipe must write `eval.mp4` + `manifest.json`.
3. Adapters fail closed with a sentence a beginner can act on.
4. Do not add a studio control that cannot be expressed in the job spec.
5. `EvalResult.facts` is the Runs contract. Notes are English for humans.
6. If a GPU engine is missing, compile the payload and block. Never
   substitute a hold-pose clip.
