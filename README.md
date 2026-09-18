# Humanoid Training

A studio for training robots without making people become Isaac Lab
experts first.

This is not another simulator. MuJoCo, Isaac Lab, mjlab, and LeRobot
already exist. The goal of this repo is the layer above them: a visual
workflow that talks to those engines, plus a library of recipes that
actually train.

Think Canva sitting on top of the print shop, not a new printing press.

## Run the studio

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ht serve --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. You need a display for eval video (MuJoCo uses GLFW).

### What to click

1. **Start here → Cartpole balance → Train this recipe.** About 30s. Play the eval video. That is the whole product loop.
2. **G1 stand → Train this recipe.** First time downloads the Unitree G1 from MuJoCo Menagerie into `.cache/`.
3. **Pick and place → Record a demo**, drag mustard into the bowl, **Save**, **Train from demos**. Or skip Record — Train writes scripted LeRobot takes. That is object-space BC, not G1 grasping.

Rooms: Robots · Tasks · Data · Runs. The job spec is under **Advanced**.

### Same jobs from the CLI

```bash
ht recipes
ht train spec/examples/cartpole-balance.json
# CPU RL → runs/<id>/eval.mp4

ht fetch-assets unitree_g1
ht train spec/examples/g1-stand.json

ht train spec/examples/g1-walk.json --compile-only
# Playground + mjlab + Isaac Lab payloads; GPU launch later

ht record spec/examples/g1-mustard-in-bowl.json --out .cache/datasets/g1-mustard
ht train spec/examples/g1-mustard-in-bowl.json
```

Headless (CI / SSH): `HT_NO_RENDER=1 ht train spec/examples/g1-stand.json` still computes success; it will not write `eval.mp4`.

## Read this first

- **[Roadmap](docs/ROADMAP.md)** — phases, exit tests, what is live now
- **[Product vision](docs/VISION.md)** — landscape and why we compile instead of replacing engines
- **[Architecture](docs/ARCHITECTURE.md)** — job spec, adapters, runners

## What works today

| Recipe | What happens |
|---|---|
| `cartpole-balance` | Gymnasium RL on CPU, eval video |
| `g1-stand` | MuJoCo G1 from Menagerie, hold stand pose, eval video |
| `g1-walk` | Compile to Playground / mjlab / Isaac Lab (GPU to launch) |
| `g1-reach` | Compile to mjlab / Isaac Lab |
| `pick-and-place` | Drag scene; record scripted or canvas-drag LeRobot demos; CPU BC puts mustard in the bowl. Not G1 grasping, not ACT. |

## Non-goals (for now)

- A new physics engine
- A new policy architecture or dataset format
- Competing with LeLab on SO-ARM101 unboxing
- Fleet operations (use Foxglove / Formant later)

## Tests

```bash
pytest
```

## License

TBD.
