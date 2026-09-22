# Product vision: a studio for robot training, not another simulator

This repo should not try to replace MuJoCo, Isaac Lab, or Isaac Sim.
Those are physics and learning *engines*. The missing product is a
**studio** that sits above them: a visual workflow for choosing a robot,
defining a task, collecting data, launching training, judging success from
video, and (later) deploying to hardware.

The honest version of "drag and drop robot training" is **Canva, not
Photoshop**. Non-technical users compose working templates. Researchers
and engineers still drop into the engine when they need to.

---

## Has this already been done?

Pieces of it exist. The complete thing does not.

If someone says "open a tab, drop a robot in a kitchen, click Train, and a
humanoid can fold laundry," that product is not shipping today. What exists
is a stack of excellent but incomplete layers.

### Simulators and learning frameworks (the engines)

| Product | What it is | What it is not |
|---|---|---|
| [Isaac Lab](https://developer.nvidia.com/isaac/lab) | NVIDIA's GPU-scale framework for RL and imitation learning. The default for serious humanoid / loco-manipulation research. | A no-code UI. Tasks are Python configs and CLI. Isaac Sim is a heavy local/cloud install. |
| [Isaac Lab Arena](https://isaac-sim.github.io/IsaacLab-Arena/) | Closest NVIDIA GUI: natural-language prompt → environment YAML, Streamlit editor, preview, policy evaluation. | Still YAML-centric. Experimental. Evaluation-oriented more than "click train a policy from scratch." |
| [MuJoCo Playground](https://playground.mujoco.org/) | `pip install playground`, train policies in minutes on one GPU, Colab notebooks, sim-to-real examples. RSS 2025 outstanding demo. | Notebooks and CLI, not a studio. |
| [mjlab](https://github.com/mujocolab/mjlab) | Isaac Lab-style API on MuJoCo Warp. Lighter stack, same mental model. | Research framework. |
| [RoboCasa](https://github.com/robocasa/robocasa) / RoboSuite | Huge kitchen scenes, objects, and tasks on MuJoCo. | Environment library, not a product UI. |
| Genesis, Newton, PyBullet | Other physics backends people will keep wanting. | Same problem: engines, not studios. |

### Workflow products that look like this idea

| Product | Closest to | Gap |
|---|---|---|
| [LeLab](https://huggingface.co/docs/lerobot/en/lelab) (Hugging Face) | The real-robot "unbox to trained policy" GUI. Calibrate, teleop, record, train locally or on HF Jobs, deploy. | **SO-ARM101 only.** Not humanoids, not Isaac/MuJoCo training at scale. |
| [LeRobot](https://github.com/huggingface/lerobot) + EnvHub + [LeIsaac](https://huggingface.co/docs/lerobot/en/envhub_leisaac) | Shared dataset format, policies, teleop, Isaac Lab tasks, cloud via NVIDIA Brev. | Still a Python/CLI ecosystem. The "one loop" is for developers. |
| [SimArena](https://simarena.ai/) (CodecFlow) | Marketing is almost this vision: browser, drag-and-drop scene, teleop, LeRobot export, export to MuJoCo XML / mjlab / Isaac Sim, "no GPU." | Early. Training on their Fabric cloud is still a roadmap item. Public GitHub surface is small. Treat as a competitor to watch, not as a solved market. |
| [Unitree G1-D](https://www.unitree.com/mobile/G1-D/) | Vendor "one-click" data, train, sim, deploy for Unitree humanoids. | Locked to one manufacturer. Fine as a backend target, bad as the whole product. |
| [NVIDIA OSMO](https://developer.nvidia.com/osmo) | YAML orchestrator for sim → train → eval → hardware-in-the-loop across cloud and on-prem GPUs. | Infra for people who already write workflows. No studio. |
| [Foxglove](https://foxglove.dev/) / Formant | Observability, teleop, fleet data. | They help you *see* robots, not *train* policies. Integrate later; do not clone. |
| RoboRenForce and similar "unified frameworks" | One config system across Isaac Lab, mjlab, Gymnasium, VLA training. | Code-first. Useful as a compiler target, not a UI. |

**Net:** the industry is converging on "make robot learning less painful," but every player owns one layer. Hugging Face owns real cheap arms and datasets. NVIDIA owns simulation and cluster orchestration. Unitree owns their humanoid. CodecFlow is trying to own the browser scene editor. Nobody owns a **humanoid-capable, engine-agnostic studio whose success metric is a robot doing the task on video**.

That is the opening.

---

## The trap

"Drag and drop" is the right *feeling* and the wrong *mechanism* if it means:

- dragging reward terms onto a humanoid and expecting PPO to work
- exposing learning-rate sliders to a teacher or lab tech
- building a new physics engine "but simpler"

From-scratch reinforcement learning on a humanoid is still a research
activity. Contact, timing, observation design, domain randomization, and
sim-to-real are where projects die. A pretty UI over a broken reward does
not help a non-technical user. It just fails more beautifully.

The products that actually work for non-experts (LeLab, Unitree's one-click
path) do a narrower thing: **fine-tune a known robot on a known class of
task from demonstrations or a canned recipe**, then show you the result.

That is the correct first product.

---

## What "good" looks like

A good platform is Figma-like in the UI and compiler-like underneath.

### For someone with no robotics background

They never hear the words PPO, USD, or domain randomization.

1. Pick a robot from a catalog (Unitree G1, H1, a tabletop arm, later a custom URDF).
2. Pick a recipe: "walk to a point," "pick this object and put it there," "follow me."
3. Optionally rearrange a 3D scene, or describe it in a sentence.
4. Optionally *show* the robot what good looks like (gamepad, leader arm, VR, or a phone video).
5. Click **Train**. The studio picks the engine, compute, and hyperparameters.
6. They watch evaluation videos, not loss curves. Pass / fail is "did the bottle go in the bin."
7. If it worked in sim, a gated **Try on robot** flow with speed limits and an e-stop.

If step 5 requires installing CUDA, Isaac Sim, and a conda env, we have already lost this user.

### For someone who *does* know this stack

The studio is not a toy. Every visual action writes a versioned **job spec**.
They can:

- open the spec as YAML/JSON
- pin Isaac Lab 2.x vs mjlab vs Playground
- override rewards, observations, domain randomization
- export a runnable script / container
- diff two runs
- attach the same spec to a cluster job (OSMO, HF Jobs, or a local GPU)

If power users cannot escape the GUI, they will not use this. If beginners
are forced into the escape hatch, they will not use this either.

### The product is the loop, not the scene editor

The valuable loop is:

**recipe → data → train → eval video → promote or iterate**

Scene editing is one input to that loop. So is a Hugging Face dataset, a
Unitree XR recording, or a sentence like "G1 picks the mustard bottle off
the counter."

---

## Recommended wedge

Do not start as "the operating system for all robots." Start as:

> **A visual studio that can take a Unitree-class humanoid, a canned
> loco-manipulation recipe, and either demos or a preset RL config, then
> produce an eval video — talking to MuJoCo Playground / mjlab first,
> Isaac Lab second.**

Why that wedge:

- This repo is named Humanoid-Training. Tabletop SO-101 is already LeLab's
  home turf. Competing with Hugging Face on cheap arms is a bad first move.
- Humanoid training is where people currently suffer most (Isaac install,
  GPU hours, opaque configs, "my G1 fell over at 40k steps").
- MuJoCo Playground / mjlab are lighter to wrap than Isaac Sim. Isaac Lab
  is the backend you add when the user actually needs photorealism or
  NVIDIA's ecosystem — not the thing you boot a first-time user into.
- Eval-video-first UX is the one thing every existing tool does poorly.
  Researchers stare at TensorBoard. Non-experts need to see the robot.

What we refuse to own on day one:

- A new physics engine
- A new policy architecture
- A new dataset format (use [LeRobot dataset format](https://github.com/huggingface/lerobot))
- A new cluster orchestrator (emit OSMO / HF Jobs / plain Docker)
- A fleet operations product (Foxglove / Formant)

---

## Studio surface (what the UI actually contains)

Five rooms. Not a dozen.

1. **Robots** — catalog + import URDF/MJCF. Calibration and camera attach for real hardware later.
2. **Tasks** — recipes with a plain-language description, success criteria, and a scene. Natural language compiles to a spec; drag-and-drop is for placing objects and goals, not for wiring neural nets.
3. **Data** — record demos, import LeRobot datasets, preview episodes like a video editor, drop the bad takes.
4. **Train** — one primary button, advanced panel collapsed. Backend badge: "running on mjlab · 1× RTX." Live eval clips every N minutes.
5. **Evaluate & deploy** — side-by-side videos, success rate, "this checkpoint is worse than last Tuesday." Deploy to sim by default. Real robot is a separate, scary-looking door.

A Python pane exists. It is an *inspector* on the generated spec and hooks,
not the home screen.

---

## How we should talk to Isaac / MuJoCo

We do not embed Isaac Sim in the browser. We **compile**.

```
Studio UI  →  Job spec (JSON)  →  Adapter  →  Engine
                              ↘  Orchestrator  →  GPU job
```

- **mjlab / MuJoCo Playground adapter:** first training path. Fast iteration, fewer install landmines, enough for locomotion and many manipulation recipes.
- **Isaac Lab adapter:** when the recipe needs their robots, sensors, or scale. The studio submits a containerized job (local Docker, cloud, or OSMO YAML). The UI streams logs and eval video back.
- **LeRobot adapter:** imitation learning and real-robot record/train/deploy, including humanoid support that LeLab does not cover yet.
- **Scene export:** MJCF / USD as *artifacts*, so a researcher can open the same scene in native MuJoCo or Isaac Sim.

The job spec is the product contract. The UI is a projection of the spec.
The adapters are replaceable. That is how we stay a platform instead of
becoming "a thin Isaac Lab tutorial website."

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the spec shape and adapter
boundaries, and [ROADMAP.md](./ROADMAP.md) for what is implemented now.

---

## What would make this actually usable by non-experts

These are product requirements, not niceties.

1. **Working recipes beat empty canvases.** Ship 5–10 tasks that train to a
   visible success on a public humanoid (G1 walk, pick-and-place)
   with pinned seeds, hardware notes, and "this took 40 minutes on an L4."
2. **Success is a video and a boolean**, not a reward plot. Show the
   failure mode in English: "fell at 1.2s," "gripper closed early."
3. **Compute is someone else's problem.** Browser users get a hosted GPU
   queue or a one-click local Docker. "Install Isaac Sim" is an advanced
   path.
4. **Defaults are opinionated and frozen per recipe.** Advanced users
   unlock them. Beginners do not wander into `entropy_coef`.
5. **Demos before rewards.** If a non-expert can show the task, imitation
   or residual RL has a chance. Asking them to design a reward is asking
   them to do research.
6. **Sim-to-real is a gated pipeline**, not a checkbox. Domain randomization
   presets, latency models, and a "this policy is sim-only" label until a
   hardware eval passes.

---

## Competitive stance

| If we try to beat… | We will lose because… |
|---|---|
| Isaac Lab at simulation fidelity | NVIDIA |
| LeRobot at dataset/policy standards | Hugging Face + the ecosystem |
| LeLab at SO-101 unboxing | They already shipped the GUI |
| SimArena at "browser CAD for robots" | That is their whole company |
| Unitree at G1 hardware bring-up | They own the robot |

| We can win if… |

- the humanoid *recipe library* is the best on the internet (reproducible, video-backed)
- one spec runs on Playground today and Isaac Lab tomorrow without rewriting the task
- a teacher, artist, or lab intern can produce a training run without a PhD
- a researcher can export the run and keep working in their engine of choice

That combination is not shipped.

---

## Phased build (technical, not calendar)

**Phase 0 — contract.** Job spec schema, one recipe (G1 velocity / walk),
one adapter (mjlab or Playground), headless train + eval video on disk.
No UI yet. If this is not reproducible, the studio is theater.

**Phase 1 — studio shell.** Browser app that authors the spec: pick robot,
pick recipe, set a few knobs, start job, watch eval clips. Local Docker
runner is enough.

**Phase 2 — data.** Teleop into LeRobot datasets, episode review, train
from demos (ACT / diffusion / a LeRobot policy) on a manipulation recipe.

**Phase 3 — second engine.** Isaac Lab adapter + cloud job submission.
Same spec, different backend badge.

**Phase 4 — real robot.** G1/H1 deploy with safety limits. Only after sim
eval gates are real.

Each phase should leave behind recipes and videos, not just code. The
library of known-good trainings *is* the product.

---

## Decision we should make explicitly

Build a **studio + compiler + recipe library**.

Do not build a simulator.

Do not pretend from-scratch humanoid RL is a consumer feature.

Do watch SimArena and LeLab monthly; steal UX, do not duplicate their
exact wedge.

The rest of this repo should be in service of that sentence.
