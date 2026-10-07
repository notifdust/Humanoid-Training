# Recipes

A recipe is a versioned, opinionated task. The studio never starts from a
blank reward function. See [ROADMAP.md](../docs/ROADMAP.md).

| id | Runnable now | Engine |
|---|---|---|
| `cartpole-balance` | yes (CPU) | gymnasium |
| `g1-stand` | yes (CPU) | mujoco + Menagerie G1 — stand + wave, not walking |
| `pick-and-place` | yes (CPU BC; ACT on GPU+LeRobot) | mujoco + LeRobot demos |
| `g1-walk` | GPU + Playground / mjlab / Isaac; blocked on CPU | first ready engine launches |
| `g1-walk-rough` | GPU + mjlab / Isaac | rough terrain velocity |
| `g1-track` | GPU + mjlab + `HT_MJLAB_MOTION` | motion tracking (LAFAN1 Hub pin) |
| `g1-pickplace` | GPU + Isaac + `HT_ISAAC_DATASET` | Robomimic locomanipulation |
| `g1-pickplace-fixed` | GPU + Isaac + dataset | Robomimic fixed-base PickPlace |

Each folder is:

```
recipe.yaml     # defaults, success, adapter maps, studio: UI contract
gold/           # eval.mp4 + notes.md for CPU recipes; CI retrains and compares
```

Do not add a gold clip on a GPU recipe (that would look like a fake walk).
Do not add recipes that cannot train or honestly block.
