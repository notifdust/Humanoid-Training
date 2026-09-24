# Recipes

A recipe is a versioned, opinionated task. The studio never starts from a
blank reward function. See [ROADMAP.md](../docs/ROADMAP.md).

| id | Runnable now | Engine |
|---|---|---|
| `cartpole-balance` | yes (CPU) | gymnasium |
| `g1-stand` | yes (CPU) | mujoco + Menagerie G1 — stand + both-arm wave, pelvis pinned, not walking |
| `g1-walk` | yes on GPU + Playground, mjlab, or Isaac Lab; blocked on CPU | playground / mjlab / isaaclab — first ready engine launches |
| `pick-and-place` | yes (CPU BC) | mujoco + LeRobot demos; BC steers mustard, arm plays poses |

Each folder is:

```
recipe.yaml     # defaults, success, adapter maps, studio: UI contract
gold/           # eval.mp4 + notes.md for CPU recipes; CI retrains and compares
```

See [ROADMAP.md](../docs/ROADMAP.md). Do not add a gold clip on a GPU
recipe (that would look like a fake walk). Do not add recipes that
cannot train or honestly block.
