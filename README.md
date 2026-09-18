# Humanoid Training

A studio for training robots without making people become Isaac Lab
experts first.

This is not another simulator. MuJoCo, Isaac Lab, mjlab, and LeRobot
already exist. The goal of this repo is the layer above them: a visual
workflow that talks to those engines, plus a library of recipes that
actually train.

Think Canva sitting on top of the print shop, not a new printing press.

## Read this first

- **[Roadmap](docs/ROADMAP.md)** — phases, exit tests, what is live now
- **[Product vision](docs/VISION.md)** — landscape and why we compile instead of replacing engines
- **[Architecture](docs/ARCHITECTURE.md)** — job spec, adapters, runners

## What works today

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

ht recipes
ht train spec/examples/cartpole-balance.json
# CPU RL smoke test → eval.mp4

ht fetch-assets unitree_g1
ht train spec/examples/g1-stand.json
# real Unitree G1 in MuJoCo, hold the stand keyframe → eval.mp4

ht train spec/examples/g1-walk.json --compile-only
# Playground + mjlab + Isaac Lab payloads; GPU launch later

ht serve --host 0.0.0.0 --port 8000
# pick a recipe, drag scene objects, edit the spec, click Train
```

| Recipe | What happens |
|---|---|
| `cartpole-balance` | Gymnasium RL on CPU, eval video |
| `g1-stand` | MuJoCo G1 from Menagerie, hold stand pose, eval video |
| `g1-walk` | Compile to Playground / mjlab / Isaac Lab (GPU to launch) |
| `g1-reach` | Compile to mjlab / Isaac Lab |
| `pick-and-place` | Scene you can drag in the studio; train is Phase 2 |

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
