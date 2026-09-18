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

## What works today (Phase 0 + studio shell)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

ht recipes
ht train spec/examples/cartpole-balance.json
# writes runs/<id>/eval.mp4 and manifest.json

ht train spec/examples/g1-walk.json --compile-only
# writes Playground train.sh + Isaac OSMO YAML; does not need a GPU

ht serve --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000 — pick a recipe, click Train, watch the eval video
```

`cartpole-balance` is the CPU smoke test that proves the loop:
**spec → adapter → train → eval video**. `g1-walk` compiles to
`G1JoystickFlatTerrain` (MuJoCo Playground) and
`Isaac-Velocity-Flat-G1-v0` (Isaac Lab). Live G1 training waits on a GPU
box with those engines installed.

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
