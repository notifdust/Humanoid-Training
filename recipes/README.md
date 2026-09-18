# Recipes

A recipe is a versioned, opinionated task. The studio never starts from a
blank reward function. See [ROADMAP.md](../docs/ROADMAP.md).

| id | Runnable now | Engine |
|---|---|---|
| `cartpole-balance` | yes (CPU) | gymnasium |
| `g1-stand` | yes (CPU) | mujoco + Menagerie G1 |
| `g1-walk` | compile only | playground / mjlab / isaaclab |
| `g1-reach` | compile only | mjlab / isaaclab |
| `pick-and-place` | scene preview (CPU MuJoCo) | mujoco now; lerobot later |

Each folder is:

```
recipe.yaml     # defaults, success, adapter maps
gold/           # later: eval.mp4 + notes that CI should resemble
```
