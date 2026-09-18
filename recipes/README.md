# Recipes

A recipe is a versioned, opinionated task. The studio never starts from a
blank reward function. See [ROADMAP.md](../docs/ROADMAP.md).

| id | Runnable now | Engine |
|---|---|---|
| `cartpole-balance` | yes (CPU) | gymnasium |
| `g1-walk` | compile only | playground / isaaclab |
| `pick-and-place` | spec only | Phase 2 |

Each folder is:

```
recipe.yaml     # defaults, success, adapter maps
gold/           # later: eval.mp4 + notes that CI should resemble
```
