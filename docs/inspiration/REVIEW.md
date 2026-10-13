# Work review (post–N1b)

Date: 2026-10-13. Against [VISION.md](../VISION.md) and this library.

## What we proved

| Claim | Evidence |
|---|---|
| Studio + compiler, not a simulator | Adapters for gymnasium / mujoco / Playground / mjlab / Isaac / LeRobot / OSMO |
| Laptop Canva loop | Cartpole, G1 stand, mustard→bowl + gold videos + CI |
| Humanoid wedge (live walk) | `ht proof walk` → Playground run `20261007T193550Z-g1-walk-9e796b`, `live_clip=true` |
| Fail closed | No stand-clip substitution; CPU walk exits 12; deploy stays `sim_only` |
| Jax pin for Playground | `jax==0.9.2` + `jax[cuda12]==0.9.2` documented and operator-verified |

## Gaps vs inspiration

| Gap | Who owns the reference | Our status |
|---|---|---|
| Beginner train **without** local CUDA | LeLab (HF Jobs), SimArena (Fabric), OSMO | Harness for OSMO/HF Jobs; **no live remote harvest yet** |
| Longer / nicer walk clip | Playground paper (~30 min on 2×4090), LIFT pretrain | Proof uses 2048 timesteps — smoke, not production gait |
| Motion tracking as recipe | BeyondMimic, mjlab Tracking, Unitree RL mjlab | `g1-track` + LAFAN1 Hub pins; **no live train** |
| Locomanipulation IL | Isaac Mimic PickPlace G1, Arena GR00T path | Recipe pins; needs Isaac + dataset |
| Real-robot deploy | Unitree RL mjlab ONNX path, G1-Playground DDS | Gate only; no torque driver |
| Scene editor | SimArena (WASM drag-drop) | Out of scope — we compile recipes, not own a browser physics editor |

## Verdict

Architecture and honesty match the vision. The wedge video exists.
The **beginner bar** (“no CUDA install”) is the open product hole — that is
Phase 3d / N2. Do not reopen theme polish or invent engines until remote
harvest or ACT live lands.
