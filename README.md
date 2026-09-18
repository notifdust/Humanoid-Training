# Humanoid Training

A studio for training robots without making people become Isaac Lab
experts first.

This is not another simulator. MuJoCo, Isaac Lab, mjlab, and LeRobot
already exist. The goal of this repo is the layer above them: a visual
workflow that talks to those engines, plus a library of recipes that
actually train.

Think Canva sitting on top of the print shop, not a new printing press.

## Read this first

- **[Product vision](docs/VISION.md)** — what already exists, why the
  "drag-and-drop RL" version of this idea fails, and the wedge we should
  take (humanoid recipes, eval videos, compile-to-engine).
- **[Architecture](docs/ARCHITECTURE.md)** — job spec as source of truth,
  adapters for Playground / mjlab / Isaac Lab / LeRobot, runners, safety.

## Current status

Vision and architecture only. No studio app yet. The first code that
should land is a frozen job-spec schema, one humanoid recipe, and one
adapter that can train and emit an eval video.

## Non-goals (for now)

- A new physics engine
- A new policy architecture or dataset format
- Competing with LeLab on SO-ARM101 unboxing
- Fleet operations (use Foxglove / Formant later)

## License

TBD.
