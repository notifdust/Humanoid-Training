# Recipes

A recipe is a versioned, opinionated task. The studio never starts from a
blank reward function. See [ROADMAP.md](../docs/ROADMAP.md).

| id | Runnable now | Engine |
|---|---|---|
| `cartpole-balance` | yes (CPU) | gymnasium |
| `g1-stand` | yes (CPU) | mujoco + Menagerie G1 — stand + both-arm wave, pelvis pinned, not walking |
| `g1-walk` | blocked on CPU (compile only) | playground / mjlab / isaaclab — needs GPU |
| `g1-reach` | compile only | mjlab / isaaclab |
| `pick-and-place` | yes (CPU BC) | mujoco + LeRobot demos; BC steers mustard, arm plays poses |

Each folder is:

```
recipe.yaml     # defaults, success, adapter maps
gold/           # later: eval.mp4 + notes that CI should resemble
```
