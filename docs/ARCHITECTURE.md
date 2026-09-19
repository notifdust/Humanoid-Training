# Architecture: studio, spec, adapters

The platform is three layers. Keep them separate so a UI rewrite or a new
simulator does not contaminate the rest.

```
┌──────────────────────────────────────────────────────────┐
│  Studio (browser)                                        │
│  Robots · Tasks · Data · Train · Evaluate                │
└────────────────────────────┬─────────────────────────────┘
                             │ reads/writes
┌────────────────────────────▼─────────────────────────────┐
│  Job spec  (versioned JSON / YAML)                       │
│  robot · scene · task · data · train · eval · deploy     │
└───────────────┬─────────────────────────────┬────────────┘
                │ compile                     │ schedule
┌───────────────▼──────────────┐   ┌──────────▼────────────┐
│  Adapters                    │   │  Runners              │
│  playground · mjlab          │   │  local docker         │
│  isaaclab · lerobot          │   │  hf jobs · osmo       │
└──────────────────────────────┘   └───────────────────────┘
```

The spec is the source of truth. The studio is a projection. Adapters are
lossy compilers into someone else's Python. Runners are how GPUs get
involved.

---

## 1. Job spec

A training run is a document, not a pile of CLI flags. Example (illustrative,
not a frozen schema):

```json
{
  "spec_version": "0.1.0",
  "name": "g1-mustard-in-bowl",
  "robot": {
    "id": "unitree-g1-29dof",
    "source": "catalog"
  },
  "scene": {
    "template": "kitchen-counter-v1",
    "objects": [
      { "id": "mustard", "asset": "ycb-mustard", "pose": "on:counter" },
      { "id": "bowl", "asset": "bowl-white", "pose": "on:counter" }
    ]
  },
  "task": {
    "recipe": "pick-and-place",
    "language": "Pick up the mustard bottle and put it in the bowl.",
    "success": {
      "type": "object-in-container",
      "object": "mustard",
      "container": "bowl",
      "hold_s": 0.5
    }
  },
  "data": {
    "datasets": ["hf:myorg/g1-mustard-demos"],
    "min_episodes": 30
  },
  "train": {
    "method": "imitation",
    "policy": "lerobot.act",
    "steps": 80000,
    "seed": 1
  },
  "backend": {
    "prefer": ["mjlab", "isaaclab"],
    "compute": "local-docker"
  },
  "eval": {
    "episodes": 20,
    "record_video": true
  }
}
```

Rules:

- Unknown fields are preserved, never silently dropped. Adapters ignore
  what they cannot honor and must record that in the run manifest.
- Recipes supply defaults. The document a beginner saves is short. The
  document a runner executes is fully expanded and hashed.
- Success criteria are first-class. If we cannot evaluate a boolean, we
  do not call the recipe done.
- `backend.prefer` is a hint. The compiler may downgrade (Isaac Lab →
  mjlab) when the recipe allows it, and must say so in the UI.

Store specs and run manifests in git-friendly YAML plus an object store
for videos, checkpoints, and datasets.

---

## 2. Studio

A single-page app. Suggested split:

| Surface | Owns | Must not own |
|---|---|---|
| Scene canvas | Layout, object placement, camera frustums, goal markers | Physics step, training loop |
| Recipe picker | Catalog of known-good tasks | Arbitrary reward algebra for v1 |
| Data bay | Episode timeline, keep/drop, teleop session | Model code |
| Run dashboard | Queue, logs, eval grid, promote checkpoint | Kubernetes details |
| Spec inspector | Diff, export, "open in VS Code" | Being the default path |

Implementation sketch (changeable):

- TypeScript + a real-time 3D viewport (Three.js / react-three-fiber for
  layout; native engine stream for high-fidelity playback)
- A small API (`studio-server`) that validates specs, expands recipes,
  and talks to runners
- WebSocket/SSE for log lines and "new eval mp4"

Do not stream Isaac Sim's full UI into the browser as the product. Optional
pixel streaming is an advanced view, not the authoring surface.

---

## 3. Adapters

Each adapter implements a narrow interface:

```
compile(expanded_spec) -> engine_payload
launch(payload, runner) -> run_id
poll(run_id) -> status, metrics, artifacts
eval(checkpoint, spec) -> videos, success_rate
export_scene(spec) -> mjcf | usd
```

### Playground / mjlab (v1)

Best first target: Python in a container, one GPU, minutes-to-hours, no
Omniverse. Use this for locomotion recipes and simpler manipulation.

Compile strategy: map `robot.id` to a known env name (`G1JoystickFlatTerrain`,
etc.), map success/eval to their eval scripts, pass seed and steps through.

### Isaac Lab (v2)

Wrap official containers. The adapter writes the Python task config or
Arena env spec, then a command that `isaaclab.sh` already understands.
Prefer submitting via [OSMO](https://developer.nvidia.com/osmo) YAML when
the user has a cluster; otherwise Docker.

### LeRobot (parallel track)

Imitation learning, dataset I/O, and real-robot control. The studio should
read and write LeRobot datasets natively so we stay compatible with LeLab,
HF Hub, and LeIsaac.

### What an adapter must never do

- Invent a parallel dataset format
- Hide the generated engine files from the user
- Claim success without running the spec's eval

If an adapter cannot express a spec field, it fails closed with a readable
error: "This recipe needs deformable cloth; only Isaac Lab can run it."

---

## 4. Runners

Runners execute compiled payloads. They are dumb.

| Runner | For |
|---|---|
| `local-docker` | Contributors and anyone with an NVIDIA GPU |
| `hf-jobs` | People already in the LeRobot world |
| `osmo` | Labs with heterogeneous GPU / HIL |
| `hosted` (later) | True no-install beginners |

The studio-server never SSHs into a researcher's machine and never bakes
cloud credentials into the frontend.

Every run writes a **manifest**: expanded spec hash, adapter version, image
digest, seed, hardware, git commit, artifact URLs. Reproducibility is a
feature, not a paper appendix.

---

## 5. Recipes

A recipe is a versioned package:

```
recipes/pick-and-place/
  recipe.yaml          # defaults, required spec fields, success type
  adapters/
    mjlab.md           # what this adapter can and cannot do
    isaaclab.py.tmpl
    lerobot.yaml.tmpl
  gold/
    eval.mp4           # what "good" looks like
    notes.md           # GPU, wall clock, known failure modes
```

Gold videos are part of CI in spirit: a change that cannot match the
qualitative success of `gold/eval.mp4` on the pinned seed is a regression
even if the code runs.

v1 recipe list should be boring and reliable:

1. G1 stand (hold + arm wave preview — not balance RL)
2. G1 walk to a pose
3. G1 reach a target
4. Fixed-base pick-and-place (can be a cheaper arm if G1 is too hard)
5. One "from demos" imitation task (CPU linear-BC today; ACT later)

---

## 6. Data

- Canonical demonstration format: LeRobot dataset on disk / Hub
- Canonical log format for live robots later: MCAP (Foxglove can view it;
  we do not need to rebuild that viewer)
- Videos are first-class artifacts, compressed and seekable
- Episode review in the studio is a cut tool, not a labeling startup

---

## 7. Safety and deploy

Real-robot deploy is a different state machine from sim eval.

- Policy is tagged `sim-only` until a hardware eval profile passes
- Default velocity/torque clamps come from the robot catalog, not the user
- E-stop and a watchdog that kills the policy on NaNs / pose limits
- No silent fallback to a previous checkpoint on the robot

Until Phase 4, the deploy button only targets simulation.

---

## 8. What we implement in *this* repository

Suggested layout when code exists:

```
studio/                      # browser shell (Phase 1)
src/humanoid_training/       # spec, recipes, adapters, runner, API
  adapters/                  # gymnasium · playground · isaaclab
spec/                        # JSON schema + example job specs
recipes/                     # versioned task packages
robots/catalog.yaml
docs/ROADMAP.md
```

Phase 0 of [VISION.md](./VISION.md) is live as `ht train`. The studio
shell is `ht serve` with Robots / Tasks / Data / Runs rooms. Dragged
`scene.objects` compile into MuJoCo via MjSpec (`composed_scene.xml`).
The Data room records and inspects local LeRobot datasets. Keep/drop writes
`data.keep_episodes` into the job spec — failure-only keeps fail mustard-in-bowl;
empty keep is refused. Pick-and-place imitation is linear BC on demos (mustard→bowl).
Notes say `dataset=local|auto-scripted` and `attach_step` (mocap, not fingers).
On the Menagerie G1 the arm plays pick/lift/place poses while BC steers mocap mustard.
Pelvis pinned. Not ACT, not finger grasping. `g1-walk` compiles and blocks on CPU
(GPU Playground / Isaac Lab). `gold/eval.mp4` folders are not required yet —
recipe gold notes document honesty instead.

Nothing in this architecture requires inventing physics.
